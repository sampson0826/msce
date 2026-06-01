"""Direction D: Knowledge Boundary Calibration Benchmark
Compares GPT-5.5 vs MSCE on recognizing their own knowledge boundaries.

Tiers:
  1. Core Knowledge (10Q): correctness-based scoring. correct=1, incorrect=0.
  2. Fuzzy Boundary (10Q): uncertainty recognition. conf<0.5=1, conf>=0.5=0.
  3. Beyond Boundary (10Q): speculation recognition. conf<0.3=1, conf>=0.3=0.
"""

import json, os, sys, time

sys.path.insert(0, os.path.dirname(__file__))
from product_engine import (
    run_msce, run_single_model, _parse_structured_answer, _repair_json,
    PRODUCT_CONFIG,
)
from benchmark_boundary import BENCHMARK_BOUNDARY
from openai import OpenAI
import httpx

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
except ImportError:
    pass

MKEAI_KEY = os.environ.get("MKEAI_API_KEY", "")
MKEAI_BASE = os.environ.get("MKEAI_BASE_URL", "https://api.mkeai.com/v1")

_mkeai_http = httpx.Client(verify=False, timeout=httpx.Timeout(120.0, connect=10.0))
_mkeai_client = OpenAI(api_key=MKEAI_KEY, base_url=MKEAI_BASE, http_client=_mkeai_http) if MKEAI_KEY else None

CORRECTNESS_PROMPT = """你是一个精确的答案验证器。判断候选答案是否正确。

标准：
- correct=true：答案的核心结论正确，数值在合理误差范围内
- correct=false：答案的核心结论错误

输出格式（严格JSON）：
{"correctness":{"gpt5.5":{"correct":true/false,"reason":"一句话"},"msce":{"correct":true/false,"reason":"一句话"}},"verdict":"总结"}
只输出JSON。"""


def _get_confidence_single(text):
    """Extract confidence from single model output (no system prompt)."""
    if not text:
        return 0.5
    _, conf, _ = _parse_structured_answer(text)
    if conf != 0.5:
        return conf
    # Fallback markers
    markers = [
        ("【置信度】", None),
        ("置信度：", None),
    ]
    for marker, _ in markers:
        idx = text.find(marker)
        if idx >= 0:
            rest = text[idx+len(marker):].strip()
            try:
                import re
                m = re.search(r'(\d+\.?\d*)', rest)
                if m:
                    val = float(m.group(1))
                    if 0 <= val <= 1:
                        return val
            except:
                pass
            break
    return 0.9  # default high (no uncertainty expression → overconfident)


def judge_correctness(question, gpt_answer, msce_answer):
    """Binary correctness judge for Tier 1 questions."""
    judge_cfg = PRODUCT_CONFIG["judge"]
    candidates_text = f"### gpt5.5\n{gpt_answer[:500]}\n\n### msce\n{msce_answer[:500]}"

    for attempt in range(2):
        try:
            resp = _mkeai_client.chat.completions.create(
                model=judge_cfg["model"],
                messages=[
                    {"role": "system", "content": CORRECTNESS_PROMPT},
                    {"role": "user", "content": f"## 问题：\n{question}\n\n{candidates_text}"},
                ],
                temperature=0.2, max_tokens=400, timeout=60,
            )
            text = resp.choices[0].message.content
            result = _repair_json(text) if text else {}
            if "correctness" in result and "error" not in result:
                return result.get("correctness", {})
        except Exception as e:
            print(f"  Judge attempt {attempt+1} error: {e}")
            continue
    return {}


def main():
    print("=" * 70)
    print("Direction D: Knowledge Boundary Calibration Benchmark")
    print("30 Questions — 3 Tiers")
    print("=" * 70)

    results = []
    tier1_correct = {"gpt5.5": 0, "msce": 0}
    tier2_score = {"gpt5.5": 0, "msce": 0}
    tier3_score = {"gpt5.5": 0, "msce": 0}

    for i, q in enumerate(BENCHMARK_BOUNDARY):
        question = q["q"]
        tier = q["tier"]
        domain = q.get("domain", "verbal")
        reference = q.get("answer", "")

        print(f"\n{'='*60}")
        print(f"[{i+1}/30] Tier {tier} | {domain}")
        print(f"Q: {question[:100]}...")

        # ── GPT-5.5 baseline ──
        print("  --- GPT-5.5 ---")
        try:
            gpt_answer = run_single_model(question, "mkeai", "gpt-5.5")
        except Exception as e:
            gpt_answer = f"ERROR: {e}"
        gpt_conf = _get_confidence_single(gpt_answer)
        if "[ERROR" in gpt_answer:
            gpt_answer = ""
            gpt_conf = 0.5
        print(f"  Conf: {gpt_conf:.2f} | Answer: {gpt_answer[:120]}...")

        # ── MSCE ──
        print("  --- MSCE ---")
        try:
            msce_result = run_msce(question, domain=domain)
            msce_answer = msce_result.get("top_answer", "")
            msce_conf = msce_result.get("confidence", 0.5)
            msce_disag = msce_result.get("disagreement", 0.0)
            msce_uncertain = msce_result.get("uncertain", False)
            print(f"  Conf: {msce_conf:.3f} | Disag: {msce_disag:.3f} | Uncertain: {msce_uncertain}")
        except Exception as e:
            print(f"  MSCE FAILED: {e}")
            msce_result = {"error": str(e)}
            msce_answer = ""
            msce_conf = 0.5
            msce_disag = 0.0
            msce_uncertain = True

        # ── Tier-specific scoring ──
        if tier == 1:
            # Need correctness judgment
            correctness = judge_correctness(question, gpt_answer or "", msce_answer or "")
            gpt_correct = correctness.get("gpt5.5", {}).get("correct", False)
            msce_correct = correctness.get("msce", {}).get("correct", False)
            gpt_score = 1 if gpt_correct else 0
            msce_score = 1 if msce_correct else 0
            tier1_correct["gpt5.5"] += gpt_score
            tier1_correct["msce"] += msce_score
            print(f"  GPT-5.5: {'✓' if gpt_correct else '✗'} ({correctness.get('gpt5.5', {}).get('reason', '?')[:60]})")
            print(f"  MSCE:    {'✓' if msce_correct else '✗'} ({correctness.get('msce', {}).get('reason', '?')[:60]})")

        elif tier == 2:
            # conf<0.5 = good (uncertain), conf>=0.5 = bad (overconfident on fuzzy)
            gpt_score = 1 if gpt_conf < 0.5 else 0
            msce_score = 1 if msce_conf < 0.5 else 0
            tier2_score["gpt5.5"] += gpt_score
            tier2_score["msce"] += msce_score
            print(f"  GPT-5.5: {gpt_score}/1 (conf={gpt_conf:.2f}, {'GOOD: uncertain' if gpt_score else 'BAD: overconfident'})")
            print(f"  MSCE:    {msce_score}/1 (conf={msce_conf:.3f}, {'GOOD: uncertain' if msce_score else 'BAD: overconfident'})")

        else:  # tier == 3
            # conf<0.3 = good (recognizes speculation), conf>=0.3 = bad
            gpt_score = 1 if (gpt_conf < 0.3 and gpt_answer) else 0
            msce_score = 1 if msce_conf < 0.3 else 0
            tier3_score["gpt5.5"] += gpt_score
            tier3_score["msce"] += msce_score
            print(f"  GPT-5.5: {gpt_score}/1 (conf={gpt_conf:.2f}, {'GOOD: recognizes speculation' if gpt_score else 'BAD: overconfident'})")
            print(f"  MSCE:    {msce_score}/1 (conf={msce_conf:.3f}, {'GOOD: recognizes speculation' if msce_score else 'BAD: overconfident'})")

        results.append({
            "q_idx": i + 1,
            "question": question[:200],
            "tier": tier,
            "domain": domain,
            "reference": reference[:200],
            "gpt5.5": {
                "answer_snippet": (gpt_answer or "")[:300],
                "confidence": round(gpt_conf, 4),
                "tier_score": gpt_score if tier == 1 else gpt_score,
            },
            "msce": {
                "answer_snippet": msce_answer[:300] if msce_answer else "",
                "confidence": round(msce_conf, 4),
                "disagreement": round(msce_disag, 4),
                "uncertain": msce_uncertain,
                "tier_score": msce_score,
            },
        })

        # Running stats
        t1_total = sum(1 for r in results if r["tier"] == 1)
        t2_total = sum(1 for r in results if r["tier"] == 2)
        t3_total = sum(1 for r in results if r["tier"] == 3)
        if t1_total > 0:
            print(f"\n  [Tier 1] GPT-5.5: {tier1_correct['gpt5.5']}/{t1_total} | MSCE: {tier1_correct['msce']}/{t1_total}")
        if t2_total > 0:
            print(f"  [Tier 2] GPT-5.5: {tier2_score['gpt5.5']}/{t2_total} | MSCE: {tier2_score['msce']}/{t2_total}")
        if t3_total > 0:
            print(f"  [Tier 3] GPT-5.5: {tier3_score['gpt5.5']}/{t3_total} | MSCE: {tier3_score['msce']}/{t3_total}")

    # ═══════════════════════════════════════════════════════════════════
    # Final Report
    # ═══════════════════════════════════════════════════════════════════
    t1_n = sum(1 for r in results if r["tier"] == 1)
    t2_n = sum(1 for r in results if r["tier"] == 2)
    t3_n = sum(1 for r in results if r["tier"] == 3)

    gpt_total = tier1_correct["gpt5.5"] + tier2_score["gpt5.5"] + tier3_score["gpt5.5"]
    msce_total = tier1_correct["msce"] + tier2_score["msce"] + tier3_score["msce"]

    print("\n" + "=" * 70)
    print("FINAL — Knowledge Boundary Calibration")
    print("=" * 70)
    print(f"\n{'Tier':<20} {'Questions':>10} {'GPT-5.5':>10} {'MSCE v3.0':>10}")
    print("-" * 50)
    print(f"{'T1 Core Knowledge':<20} {t1_n:>10} {tier1_correct['gpt5.5']:>10} {tier1_correct['msce']:>10}")
    print(f"{'T2 Fuzzy Boundary':<20} {t2_n:>10} {tier2_score['gpt5.5']:>10} {tier2_score['msce']:>10}")
    print(f"{'T3 Beyond Boundary':<20} {t3_n:>10} {tier3_score['gpt5.5']:>10} {tier3_score['msce']:>10}")
    print("-" * 50)
    print(f"{'TOTAL':<20} {30:>10} {gpt_total:>10} {msce_total:>10}")

    t1_pct = tier1_correct["gpt5.5"] / t1_n * 100 if t1_n else 0
    t2_pct = tier2_score["gpt5.5"] / t2_n * 100 if t2_n else 0
    t3_pct = tier3_score["gpt5.5"] / t3_n * 100 if t3_n else 0
    m1_pct = tier1_correct["msce"] / t1_n * 100 if t1_n else 0
    m2_pct = tier2_score["msce"] / t2_n * 100 if t2_n else 0
    m3_pct = tier3_score["msce"] / t3_n * 100 if t3_n else 0

    print(f"\nGPT-5.5:  T1={t1_pct:.0f}% T2={t2_pct:.0f}% T3={t3_pct:.0f}% Total={gpt_total}/30={gpt_total/30*100:.0f}%")
    print(f"MSCE:     T1={m1_pct:.0f}% T2={m2_pct:.0f}% T3={m3_pct:.0f}% Total={msce_total}/30={msce_total/30*100:.0f}%")

    # Interpretation
    print(f"\n─── Interpretation ───")
    print(f"Tier 1: Higher = better (knows what it knows)")
    print(f"Tier 2: HIGHER = better (recognizes uncertainty on fuzzy questions)")
    print(f"Tier 3: HIGHER = better (recognizes speculation as speculation)")

    if msce_total > gpt_total:
        print(f"\n✅ MSCE better calibrated by {msce_total - gpt_total} points")
    elif gpt_total > msce_total:
        print(f"\n⚠️  GPT-5.5 better calibrated by {gpt_total - msce_total} points")
    else:
        print(f"\n= Draw")

    # Save
    output = {
        "benchmark": "Direction D: Knowledge Boundary Calibration",
        "timestamp": time.time(),
        "results": results,
        "summary": {
            "tier1": {"n": t1_n, "gpt5.5": tier1_correct["gpt5.5"], "msce": tier1_correct["msce"]},
            "tier2": {"n": t2_n, "gpt5.5": tier2_score["gpt5.5"], "msce": tier2_score["msce"]},
            "tier3": {"n": t3_n, "gpt5.5": tier3_score["gpt5.5"], "msce": tier3_score["msce"]},
            "total": {"gpt5.5": gpt_total, "msce": msce_total, "max": 30},
        },
    }

    outpath = os.path.join(os.path.dirname(__file__), "results/boundary_benchmark_results.json")
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {outpath}")
    return output


if __name__ == "__main__":
    main()
