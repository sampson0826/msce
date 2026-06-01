"""MSCE Calibration Benchmark — Confidence vs Accuracy curves
验证: MSCE的置信度是否比单模型更准确反映实际正确率

Key metric: ECE (Expected Calibration Error) — 越低越好
Key story: GPT-5.5 high confidence + wrong ≠ MSCE low confidence + uncertain
"""
import json, os, time, sys
from collections import defaultdict
from openai import OpenAI
import httpx

# Load .env if available
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
except ImportError:
    pass

sys.path.insert(0, os.path.dirname(__file__))

# ═══════════════════════════════════════════════════════════════════════════
# API Config
# ═══════════════════════════════════════════════════════════════════════════
MKEAI_KEY = os.environ.get("MKEAI_API_KEY", "")
MKEAI_BASE = os.environ.get("MKEAI_BASE_URL", "https://api.mkeai.com/v1")
DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY", "")

_mkeai_http = httpx.Client(verify=False, timeout=httpx.Timeout(120))
_mkeai_client = OpenAI(api_key=MKEAI_KEY, base_url=MKEAI_BASE, http_client=_mkeai_http) if MKEAI_KEY else None
_deepseek_client = OpenAI(api_key=DEEPSEEK_KEY, base_url="https://api.deepseek.com/v1") if DEEPSEEK_KEY else None

# ═══════════════════════════════════════════════════════════════════════════
# Confidence estimation from model text (same method as frontier_benchmark)
# ═══════════════════════════════════════════════════════════════════════════

def estimate_confidence(text):
    """Estimate a model's self-reported confidence from language markers."""
    if not text:
        return 0.5
    text_lower = text.lower()
    markers = [
        ("100%", 1.0), ("绝对正确", 1.0), ("definitely", 1.0), ("certainly", 0.95),
        ("确定", 0.95), ("毫无疑问", 0.95), ("毫无疑问", 0.95),
        ("应该", 0.7), ("likely", 0.7), ("probably", 0.7), ("很可能", 0.7),
        ("可能", 0.5), ("perhaps", 0.5), ("maybe", 0.5), ("大概", 0.5),
        ("不确定", 0.3), ("unsure", 0.3), ("不太确定", 0.3), ("not sure", 0.3),
        ("不知道", 0.1), ("无法确定", 0.2),
    ]
    conf = 0.9  # default assumption: models are overconfident
    for marker, c in markers:
        if marker in text_lower:
            conf = min(conf, c)
    return conf


def extract_final_answer(text):
    """Extract the final answer from model output for comparison."""
    import re
    patterns = [
        r'最终答案[：:]\s*(.+?)(?:\n|$)',
        r'答案是[：:]\s*(.+?)(?:\n|$)',
        r'答案[：:]\s*(.+?)(?:\n|$)',
        r'Answer[：:]\s*(.+?)(?:\n|$)',
        r'结论[：:]\s*(.+?)(?:\n|$)',
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if lines:
        return lines[-1][:200]
    return text[-200:] if text else ""


def check_correct(model_answer, correct_answer_full):
    """Check if model answer is correct by comparing key numerical/text result."""
    import re
    key_result = correct_answer_full.split("。")[0].split("；")[0].split("，")[0].strip()
    key_number = re.search(r'[\d.]+', key_result)
    answer_lower = model_answer.lower()
    if key_number:
        return key_number.group() in answer_lower or key_result.lower() in answer_lower
    return key_result.lower() in answer_lower


# ═══════════════════════════════════════════════════════════════════════════
# Single-model run (for GPT-5.5 baseline)
# ═══════════════════════════════════════════════════════════════════════════

def run_gpt55(question):
    """Run GPT-5.5 on a single question. Returns (answer, confidence, elapsed)."""
    t0 = time.time()
    try:
        resp = _mkeai_client.chat.completions.create(
            model="gpt-5.5",
            messages=[
                {"role": "system", "content": "你是一个精确的推理助手。先推理，再给出最终答案。如果你不确定，请明确说'不确定'。"},
                {"role": "user", "content": question},
            ],
            temperature=0.7,
            max_tokens=2048,
            timeout=120,
        )
        elapsed = time.time() - t0
        answer = resp.choices[0].message.content or ""
        conf = estimate_confidence(answer)
        return answer, conf, elapsed
    except Exception as e:
        elapsed = time.time() - t0
        return f"[ERROR: {str(e)[:200]}]", 0.5, elapsed


# ═══════════════════════════════════════════════════════════════════════════
# MSCE run (via product_engine)
# ═══════════════════════════════════════════════════════════════════════════

def run_msce_calibrated(question):
    """Run MSCE on a single question. Returns (answer, confidence, disagreement, elapsed)."""
    from product_engine import run_msce
    t0 = time.time()
    try:
        result = run_msce(question, domain="math", skip_appeal=True)
        elapsed = time.time() - t0

        # Extract top answer from surviving candidates
        top_answer = ""
        for c in result.get("candidates", []):
            for s in result.get("verdict", {}).get("surviving", []):
                if c["strategy"] == s["id"] and c["success"]:
                    top_answer = c["answer"]
                    break
            if top_answer:
                break

        # Fallback: first successful candidate
        if not top_answer:
            for c in result.get("candidates", []):
                if c["success"] and c["answer"]:
                    top_answer = c["answer"]
                    break

        return (
            top_answer,
            result.get("confidence", 0.5),
            result.get("disagreement", 0.0),
            result.get("low_confidence", False),
            elapsed,
        )
    except Exception as e:
        elapsed = time.time() - t0
        return f"[ERROR: {str(e)[:200]}]", 0.5, 0.0, True, elapsed


# ═══════════════════════════════════════════════════════════════════════════
# Calibration Metrics
# ═══════════════════════════════════════════════════════════════════════════

def compute_ece(predictions, n_buckets=10):
    """Expected Calibration Error — lower is better.

    predictions: list of (confidence, is_correct) tuples.
    """
    bucket_size = 1.0 / n_buckets
    ece = 0.0
    bucket_details = []

    for b in range(n_buckets):
        low = b * bucket_size
        high = (b + 1) * bucket_size
        in_bucket = [(c, a) for c, a in predictions if low <= c < high]
        # Include high=1.0 in last bucket
        if b == n_buckets - 1:
            in_bucket = [(c, a) for c, a in predictions if low <= c <= high]

        if not in_bucket:
            bucket_details.append({
                "bucket": f"{low:.1f}-{high:.1f}",
                "count": 0,
                "avg_confidence": 0,
                "accuracy": None,
                "gap": None,
            })
            continue

        avg_conf = sum(c for c, _ in in_bucket) / len(in_bucket)
        accuracy = sum(1 for _, a in in_bucket if a) / len(in_bucket)
        gap = abs(avg_conf - accuracy)
        ece += (len(in_bucket) / len(predictions)) * gap

        bucket_details.append({
            "bucket": f"{low:.1f}-{high:.1f}",
            "count": len(in_bucket),
            "avg_confidence": round(avg_conf, 4),
            "accuracy": round(accuracy, 4),
            "gap": round(gap, 4),
        })

    return round(ece, 4), bucket_details


def compute_metrics(predictions):
    """Compute comprehensive calibration metrics."""
    n = len(predictions)
    correct = sum(1 for _, a in predictions if a)
    accuracy = correct / n if n else 0
    avg_conf = sum(c for c, _ in predictions) / n if n else 0

    # Confident-wrong: confidence > 0.8 but answer wrong
    confident_wrong = sum(1 for c, a in predictions if c > 0.8 and not a)
    confident_total = sum(1 for c, _ in predictions if c > 0.8)

    # Uncertain-correct: confidence < 0.5 but answer right
    uncertain_correct = sum(1 for c, a in predictions if c < 0.5 and a)

    ece, buckets = compute_ece(predictions)

    return {
        "n": n,
        "accuracy": round(accuracy, 4),
        "avg_confidence": round(avg_conf, 4),
        "confidence_accuracy_gap": round(avg_conf - accuracy, 4),
        "ece": ece,
        "confident_wrong": {"count": confident_wrong, "of_total_high_conf": confident_total},
        "uncertain_correct": uncertain_correct,
        "buckets": buckets,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main Benchmark
# ═══════════════════════════════════════════════════════════════════════════

def main():
    from benchmark_questions import BENCHMARK

    print("=" * 70)
    print("MSCE Calibration Benchmark — Confidence vs Accuracy")
    print("=" * 70)
    print(f"\nQuestions: {len(BENCHMARK)}")
    print(f"Metrics: ECE (Expected Calibration Error), Confident-Wrong Rate")
    print()

    results = []
    gpt55_preds = []  # (confidence, is_correct)
    msce_preds = []   # (confidence, is_correct)

    for i, q in enumerate(BENCHMARK):
        question = q["q"]
        correct_answer = q.get("answer", "")
        domain = q.get("domain", "math")

        print(f"[{i+1}/{len(BENCHMARK)}] {question[:80]}...")

        # GPT-5.5
        gpt_answer, gpt_conf, gpt_time = run_gpt55(question)
        gpt_correct = check_correct(gpt_answer, correct_answer)
        gpt55_preds.append((gpt_conf, gpt_correct))
        print(f"  GPT-5.5: {'✓' if gpt_correct else '✗'} conf={gpt_conf:.2f} ({gpt_time:.0f}s)")

        # MSCE
        msce_answer, msce_conf, msce_disag, msce_low, msce_time = run_msce_calibrated(question)
        msce_correct = check_correct(msce_answer, correct_answer)
        msce_preds.append((msce_conf, msce_correct))
        print(f"  MSCE:    {'✓' if msce_correct else '✗'} conf={msce_conf:.2f} disag={msce_disag:.2f} low={msce_low} ({msce_time:.0f}s)")

        results.append({
            "q_idx": i + 1,
            "question": question[:150],
            "domain": domain,
            "gpt55_correct": gpt_correct,
            "gpt55_confidence": round(gpt_conf, 4),
            "gpt55_time": round(gpt_time, 2),
            "msce_correct": msce_correct,
            "msce_confidence": round(msce_conf, 4),
            "msce_disagreement": round(msce_disag, 4),
            "msce_low_confidence": msce_low,
            "msce_time": round(msce_time, 2),
        })

        # Running stats
        gpt_acc = sum(1 for _, a in gpt55_preds if a) / len(gpt55_preds)
        msce_acc = sum(1 for _, a in msce_preds if a) / len(msce_preds)
        print(f"  [Running] GPT-5.5 acc={gpt_acc:.1%} | MSCE acc={msce_acc:.1%}")

    # ═══════════════════════════════════════════════════════════════════════
    # Final Report
    # ═══════════════════════════════════════════════════════════════════════
    gpt_metrics = compute_metrics(gpt55_preds)
    msce_metrics = compute_metrics(msce_preds)

    print("\n" + "=" * 70)
    print("CALIBRATION RESULTS")
    print("=" * 70)

    print(f"\n--- GPT-5.5 ---")
    print(f"  Accuracy:           {gpt_metrics['accuracy']:.1%}")
    print(f"  Avg Confidence:     {gpt_metrics['avg_confidence']:.2f}")
    print(f"  Confidence Gap:     {gpt_metrics['confidence_accuracy_gap']:+.2f}  ← overconfidence")
    print(f"  ECE:                {gpt_metrics['ece']:.4f}  ← lower is better")
    print(f"  Confident-Wrong:    {gpt_metrics['confident_wrong']['count']}/{gpt_metrics['confident_wrong']['of_total_high_conf']} high-conf predictions")

    print(f"\n--- MSCE ---")
    print(f"  Accuracy:           {msce_metrics['accuracy']:.1%}")
    print(f"  Avg Confidence:     {msce_metrics['avg_confidence']:.2f}")
    print(f"  Confidence Gap:     {msce_metrics['confidence_accuracy_gap']:+.2f}")
    print(f"  ECE:                {msce_metrics['ece']:.4f}")
    print(f"  Confident-Wrong:    {msce_metrics['confident_wrong']['count']}/{msce_metrics['confident_wrong']['of_total_high_conf']} high-conf predictions")

    print(f"\n--- Key Comparison ---")
    ece_improvement = gpt_metrics['ece'] - msce_metrics['ece']
    print(f"  ECE Improvement:    {ece_improvement:+.4f} ({'better' if ece_improvement > 0 else 'worse'})")
    print(f"  Conf-Wrong Reduction: GPT-5.5={gpt_metrics['confident_wrong']['count']} → MSCE={msce_metrics['confident_wrong']['count']}")

    # Per-bucket comparison
    print(f"\n--- Reliability Diagram ---")
    print(f"{'Bucket':<12} {'GPT-5.5 Acc':<14} {'MSCE Acc':<14} {'Better':<10}")
    print(f"{'':-<50}")
    for b in range(10):
        gb = gpt_metrics['buckets'][b]
        mb = msce_metrics['buckets'][b]
        g_acc = f"{gb['accuracy']:.3f}" if gb['accuracy'] is not None else "N/A"
        m_acc = f"{mb['accuracy']:.3f}" if mb['accuracy'] is not None else "N/A"
        better = ""
        if gb['accuracy'] is not None and mb['accuracy'] is not None:
            gap_g = abs(gb['avg_confidence'] - gb['accuracy'])
            gap_m = abs(mb['avg_confidence'] - mb['accuracy'])
            better = "MSCE" if gap_m < gap_g else ("GPT-5.5" if gap_g < gap_m else "tie")
        print(f"{gb['bucket']:<12} {g_acc:<14} {m_acc:<14} {better:<10}")

    # Save
    output = {
        "timestamp": time.time(),
        "results": results,
        "gpt55_metrics": gpt_metrics,
        "msce_metrics": msce_metrics,
        "ece_improvement": round(ece_improvement, 4),
        "n_questions": len(BENCHMARK),
    }

    outpath = os.path.join(os.path.dirname(__file__), "results/calibration_results.json")
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to {outpath}")
    return output


if __name__ == "__main__":
    main()
