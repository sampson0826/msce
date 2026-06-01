"""MSCE Frontier Benchmark v3.0 — 6-model weighted integration vs GPT-5.5 baseline
Validates: MSCE's 3-layer filter + weighted integration vs strongest single model.
"""
import json, os, time, re, sys

sys.path.insert(0, os.path.dirname(__file__))
from product_engine import (
    run_msce, run_single_model, get_client, _repair_json,
    PRODUCT_CONFIG, STRATEGY_PROMPTS, JUDGE_PROMPT,
    GENERATOR_MAX_TOKENS, GENERATOR_TIMEOUT,
)
from product_engine import _parse_structured_answer
from openai import OpenAI
import httpx

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
except ImportError:
    pass

# ═══════════════════════════════════════════════════════════════════════════
# API Config
# ═══════════════════════════════════════════════════════════════════════════
MKEAI_KEY = os.environ.get("MKEAI_API_KEY", "")
MKEAI_BASE = os.environ.get("MKEAI_BASE_URL", "https://api.mkeai.com/v1")

_mkeai_http = httpx.Client(verify=False, timeout=httpx.Timeout(120.0, connect=10.0))
_mkeai_client = OpenAI(api_key=MKEAI_KEY, base_url=MKEAI_BASE, http_client=_mkeai_http) if MKEAI_KEY else None

# ═══════════════════════════════════════════════════════════════════════════
# Binary correctness judge — separate from scoring judge
# ═══════════════════════════════════════════════════════════════════════════

CORRECTNESS_JUDGE_PROMPT = """你是一个精确的答案验证器。判断每个候选答案是否正确。

## 判断标准：
- correct=true：答案的核心结论正确。数值在合理误差范围内、逻辑推理自洽。
- correct=false：答案的核心结论错误。数值错误、推理矛盾、或遗漏关键约束。

注意：
- 如果推理过程有瑕疵但最终答案正确，仍然算correct=true
- 如果推理过程看起来很认真但最终答案错误，必须correct=false
- 不要被答案长度影响判断

## 输出格式（严格JSON）：
{"correctness":{"候选名":{"correct":true/false,"reason":"一句话理由"}},"verdict":"总结"}

只输出JSON。"""


def judge_correctness(question, candidates_dict, domain="math"):
    """Evaluate binary correctness for a set of answers.

    Judge cascade (per Sam): simple domains (math/logic/science) → gpt-5.5 fast,
    complex domains (constraint_propagation/verbal/cross_domain) → grok-4.1-thinking.

    Args:
        question: the question text
        candidates_dict: {"name": "answer text", ...}
        domain: question domain for judge cascade

    Returns:
        {"name": {"correct": bool, "reason": str}, ...}
    """
    SIMPLE_DOMAINS = {"math", "logic", "science"}
    if domain in SIMPLE_DOMAINS:
        judge_model = "gpt-5.5"
        judge_timeout = 15
        max_attempts = 1
    else:
        judge_model = PRODUCT_CONFIG["judge"]["model"]
        judge_timeout = 60
        max_attempts = 2

    # Build candidate text with core conclusions (length normalization)
    candidate_text = ""
    for name, answer in candidates_dict.items():
        core = answer
        # Try structured parse
        parsed, _, _ = _parse_structured_answer(answer)
        if parsed and len(parsed) < 500:
            core = parsed
        elif len(core) > 500:
            core = core[:300] + "\n...\n" + core[-200:]
        candidate_text += f"\n### {name}\n{core}\n"

    # Best-of-N for reliability (N=1 for simple domains, 2 for complex)
    best_result = None
    for attempt in range(max_attempts):
        try:
            resp = _mkeai_client.chat.completions.create(
                model=judge_model,
                messages=[
                    {"role": "system", "content": CORRECTNESS_JUDGE_PROMPT},
                    {"role": "user", "content": f"## 问题：\n{question}\n\n## 候选答案：\n{candidate_text}"},
                ],
                temperature=0.2,
                max_tokens=800,
                timeout=judge_timeout,
            )
            text = resp.choices[0].message.content
            result = _repair_json(text) if text else {}
            if "correctness" in result and "error" not in result:
                best_result = result
                break
            if best_result is None:
                best_result = result
        except Exception as e:
            print(f"  Correctness judge attempt {attempt+1} error: {e}")
            continue

    if best_result is None or "error" in best_result:
        return {}
    return best_result.get("correctness", {})


# ═══════════════════════════════════════════════════════════════════════════
# Self-confidence estimation (for GPT-5.5 baseline which may not self-report)
# ═══════════════════════════════════════════════════════════════════════════

def _estimate_confidence(text):
    """Estimate a model's self-reported confidence from language / structured output."""
    if not text:
        return 0.5

    # v3.0 structured format
    _, conf, _ = _parse_structured_answer(text)
    if conf != 0.5:  # parsed a real confidence value
        return conf

    # Fallback: language markers
    text_lower = text.lower()
    markers = [
        ("100%", 1.0), ("绝对正确", 1.0), ("definitely", 1.0), ("certainly", 0.95),
        ("确定", 0.95), ("毫无疑问", 0.95),
        ("应该", 0.7), ("likely", 0.7), ("probably", 0.7), ("很可能", 0.7),
        ("可能", 0.5), ("perhaps", 0.5), ("maybe", 0.5), ("大概", 0.5),
        ("不确定", 0.3), ("unsure", 0.3), ("不太确定", 0.3),
        ("不知道", 0.1), ("无法确定", 0.2),
    ]
    conf = 0.9
    for marker, c in markers:
        if marker in text_lower:
            conf = min(conf, c)
    return conf


# ═══════════════════════════════════════════════════════════════════════════
# Load Questions
# ═══════════════════════════════════════════════════════════════════════════

def load_questions(extended=True):
    if extended:
        from benchmark_extended import BENCHMARK
    else:
        from benchmark_questions import BENCHMARK
    return BENCHMARK


# ═══════════════════════════════════════════════════════════════════════════
# Main Benchmark
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("MSCE Frontier Benchmark v3.0 — Weighted Integration vs GPT-5.5")
    print("=" * 70)

    questions = load_questions()
    print(f"\nLoaded {len(questions)} questions")

    results = []
    confident_wrong_cases = []

    for i, q in enumerate(questions):
        question_text = q["q"]
        domain = q.get("domain", "math")

        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(questions)}] {question_text[:80]}...")
        print(f"  Domain: {domain}")

        # ── GPT-5.5 baseline ──
        print(f"\n  --- GPT-5.5 ---")
        try:
            gpt_answer = run_single_model(question_text, "mkeai", "gpt-5.5")
        except Exception as e:
            gpt_answer = f"[ERROR: {e}]"
        gpt_conf = _estimate_confidence(gpt_answer)
        gpt_time = 0  # approximate
        print(f"  Self-confidence: {gpt_conf:.2f}")

        # ── MSCE v3.0 ──
        print(f"\n  --- MSCE v3.0 (6-model weighted integration) ---")
        try:
            msce_result = run_msce(question_text, domain=domain)
            msce_conf = msce_result.get("confidence", 0.0)
            msce_disag = msce_result.get("disagreement", 0.0)
            msce_time = msce_result.get("elapsed_time", 0)
            top_strategy = msce_result.get("top_strategy", "?")
            top_answer = msce_result.get("top_answer", "")
            outliers = msce_result.get("outliers", [])
            low_conf_ids = msce_result.get("low_confidence_ids", [])
            judge_scores = msce_result.get("judge_scores", {})

            print(f"  Top: {top_strategy}, conf={msce_conf:.2f}, disag={msce_disag:.2f}")
            print(f"  Outliers: {outliers}, Low-conf: {low_conf_ids}")
            print(f"  Judge scores: {json.dumps(judge_scores, ensure_ascii=False)}")
            print(f"  Time: {msce_time:.1f}s")
        except Exception as e:
            msce_result = {"error": str(e)}
            msce_conf = 0.0
            msce_disag = 0.0
            msce_time = 0
            top_answer = ""
            print(f"  MSCE FAILED: {e}")

        # ── Binary correctness judge ──
        candidates_for_judge = {"gpt5.5": gpt_answer}
        if top_answer:
            candidates_for_judge["msce"] = top_answer

        correctness = judge_correctness(question_text, candidates_for_judge)

        gpt_correct = correctness.get("gpt5.5", {}).get("correct", False) if correctness else False
        msce_correct = correctness.get("msce", {}).get("correct", False) if correctness else False

        gpt_reason = correctness.get("gpt5.5", {}).get("reason", "") if correctness else ""
        msce_reason = correctness.get("msce", {}).get("reason", "") if correctness else ""

        print(f"  GPT-5.5: {'✓' if gpt_correct else '✗'} ({gpt_reason[:60]})")
        print(f"  MSCE:    {'✓' if msce_correct else '✗'} ({msce_reason[:60]})")

        # ── Confident wrongness check ──
        if not gpt_correct and gpt_conf > 0.8:
            case = {
                "q_idx": i + 1,
                "question": question_text[:100],
                "gpt_answer_snippet": gpt_answer[:300],
                "gpt_confidence": gpt_conf,
                "msce_correct": msce_correct,
                "msce_confidence": msce_conf,
                "msce_disagreement": msce_disag,
            }
            confident_wrong_cases.append(case)
            print(f"  ⚠️  GPT-5.5 CONFIDENTLY WRONG! (self-conf={gpt_conf:.2f})")
            if msce_correct:
                print(f"  ✅ MSCE got it RIGHT!")

        # Both correct, MSCE more cautious
        if gpt_correct and msce_correct and msce_conf < 0.7:
            print(f"  💡 Both correct, MSCE more cautious (conf={msce_conf:.2f})")

        results.append({
            "q_idx": i + 1,
            "question": question_text[:200],
            "domain": domain,
            "gpt55_correct": gpt_correct,
            "gpt55_self_conf": round(gpt_conf, 2),
            "gpt55_time": round(gpt_time, 2),
            "msce_correct": msce_correct,
            "msce_confidence": round(msce_conf, 4),
            "msce_disagreement": round(msce_disag, 4),
            "msce_time": round(msce_time, 2),
            "gpt_reason": gpt_reason,
            "msce_reason": msce_reason,
        })

        # Running stats
        gpt_acc = sum(1 for r in results if r["gpt55_correct"]) / len(results)
        msce_acc = sum(1 for r in results if r["msce_correct"]) / len(results)
        print(f"\n  [Running] GPT-5.5: {gpt_acc:.1%} | MSCE: {msce_acc:.1%}")

    # ═══════════════════════════════════════════════════════════════════════
    # Final Report
    # ═══════════════════════════════════════════════════════════════════════
    n = len(results)
    gpt_acc = sum(1 for r in results if r["gpt55_correct"]) / n
    msce_acc = sum(1 for r in results if r["msce_correct"]) / n
    gpt_avg_conf = sum(r["gpt55_self_conf"] for r in results) / n
    msce_avg_conf = sum(r["msce_confidence"] for r in results) / n
    msce_avg_disagree = sum(r["msce_disagreement"] for r in results) / n

    print("\n" + "=" * 70)
    print("FINAL RESULTS — MSCE v3.0 Weighted Integration")
    print("=" * 70)
    print(f"\nQuestions: {n}")
    print(f"\n| Metric | GPT-5.5 | MSCE v3.0 |")
    print(f"|--------|---------|------------|")
    print(f"| Accuracy | {gpt_acc:.1%} | {msce_acc:.1%} |")
    print(f"| Avg Confidence | {gpt_avg_conf:.2f} | {msce_avg_conf:.2f} |")
    print(f"| Avg Disagreement | — | {msce_avg_disagree:.2f} |")

    print(f"\n--- Confident Wrongness ---")
    print(f"GPT-5.5 confidently wrong (self-conf > 0.8 but answer wrong): {len(confident_wrong_cases)} cases")
    for case in confident_wrong_cases:
        print(f"\n  Q{case['q_idx']}: {case['question']}")
        print(f"  GPT-5.5: CONF={case['gpt_confidence']:.2f} but WRONG")
        print(f"  MSCE:    {'RIGHT' if case['msce_correct'] else 'WRONG'}, conf={case['msce_confidence']:.2f}")
        print(f"  GPT snippet: {case['gpt_answer_snippet'][:200]}")

    # Save
    output = {
        "version": "3.0",
        "timestamp": time.time(),
        "config": {k: v for k, v in PRODUCT_CONFIG.items() if k != "appeal"},
        "results": results,
        "summary": {
            "n_questions": n,
            "gpt55_accuracy": round(gpt_acc, 4),
            "msce_accuracy": round(msce_acc, 4),
            "gpt55_avg_confidence": round(gpt_avg_conf, 4),
            "msce_avg_confidence": round(msce_avg_conf, 4),
            "msce_avg_disagreement": round(msce_avg_disagree, 4),
            "confident_wrong_cases": len(confident_wrong_cases),
            "confident_wrong_details": confident_wrong_cases,
        }
    }

    outpath = os.path.join(os.path.dirname(__file__), "results/frontier_benchmark_results.json")
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to {outpath}")
    return output


if __name__ == "__main__":
    main()
