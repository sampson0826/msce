"""MSCE Expanded Benchmark Runner — 120 questions with parallel execution + Judge cascade.
Run: python3 run_expanded_benchmark.py
"""
import json, os, time, sys, threading
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(__file__))

from benchmark_extended import BENCHMARK as BENCHMARK_60
from benchmark_expanded_120 import BENCHMARK_EXPANDED
from frontier_benchmark import (
    judge_correctness, _estimate_confidence,
    run_msce, run_single_model, get_client
)

BENCHMARK = BENCHMARK_60 + BENCHMARK_EXPANDED

_print_lock = threading.Lock()


def _p(*args, **kwargs):
    """Thread-safe print."""
    with _print_lock:
        print(*args, **kwargs)


def run_one_question(q, idx, total):
    """Run a single benchmark question. Returns result dict. (Thread-safe)"""
    question_text = q["q"]
    domain = q.get("domain", "math")

    _p(f"\n{'='*60}")
    _p(f"[{idx+1}/{total}] {domain}: {question_text[:80]}...")

    # GPT-5.5 baseline
    try:
        gpt_answer = run_single_model(question_text, "mkeai", "gpt-5.5")
    except Exception as e:
        gpt_answer = f"[ERROR: {e}]"
    gpt_conf = _estimate_confidence(gpt_answer)

    # MSCE v3.0
    try:
        msce_result = run_msce(question_text, domain=domain)
        msce_conf = msce_result.get("confidence", 0.0)
        msce_disag = msce_result.get("disagreement", 0.0)
        msce_time = msce_result.get("elapsed_time", 0)
        top_answer = msce_result.get("top_answer", "")
    except Exception as e:
        msce_result = {"error": str(e)}
        msce_conf = 0.0
        msce_disag = 0.0
        msce_time = 0
        top_answer = ""

    # Binary correctness
    candidates_for_judge = {"gpt5.5": gpt_answer}
    if top_answer:
        candidates_for_judge["msce"] = top_answer

    correctness = judge_correctness(question_text, candidates_for_judge, domain=domain)
    gpt_correct = correctness.get("gpt5.5", {}).get("correct", False) if correctness else False
    msce_correct = correctness.get("msce", {}).get("correct", False) if correctness else False
    gpt_reason = correctness.get("gpt5.5", {}).get("reason", "") if correctness else ""
    msce_reason = correctness.get("msce", {}).get("reason", "") if correctness else ""

    _p(f"  GPT-5.5: {'✓' if gpt_correct else '✗'} (conf={gpt_conf:.2f})")
    _p(f"  MSCE:    {'✓' if msce_correct else '✗'} (conf={msce_conf:.2f}, disag={msce_disag:.2f})")

    return {
        "q_idx": idx + 1,
        "question": question_text[:200],
        "domain": domain,
        "gpt55_correct": gpt_correct,
        "gpt55_self_conf": round(gpt_conf, 2),
        "msce_correct": msce_correct,
        "msce_confidence": round(msce_conf, 4),
        "msce_disagreement": round(msce_disag, 4),
        "msce_time": round(msce_time, 2),
        "gpt_reason": gpt_reason,
        "msce_reason": msce_reason,
    }


def main():
    print("=" * 70)
    print("MSCE Expanded Benchmark — 120 Questions (Parallel + Judge Cascade)")
    print("60 original + 60 search tests (per Sam's requirements)")
    print("=" * 70)

    questions = BENCHMARK
    total = len(questions)
    print(f"\nTotal questions: {total}")
    print(f"Domains: math={sum(1 for q in questions if q['domain']=='math')}, "
          f"logic={sum(1 for q in questions if q['domain']=='logic')}, "
          f"science={sum(1 for q in questions if q['domain']=='science')}, "
          f"verbal={sum(1 for q in questions if q['domain']=='verbal')}, "
          f"constraint_propagation={sum(1 for q in questions if q['domain']=='constraint_propagation')}, "
          f"cross_domain={sum(1 for q in questions if q['domain']=='cross_domain')}")
    print(f"\nOptimizations: 3-way parallel + Judge cascade + MAX_RETRIES=1")
    t_start = time.time()

    results = {}
    confident_wrong_cases = []
    completed = 0

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(run_one_question, q, i, total): i
            for i, q in enumerate(questions)
        }

        for future in as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
                results[idx] = result
                completed += 1

                # Running stats
                done = [results[k] for k in sorted(results.keys())]
                gpt_acc = sum(1 for r in done if r["gpt55_correct"]) / len(done)
                msce_acc = sum(1 for r in done if r["msce_correct"]) / len(done)
                _p(f"  [Progress] {completed}/{total} | GPT-5.5: {gpt_acc:.1%} | MSCE: {msce_acc:.1%} | Elapsed: {time.time()-t_start:.0f}s")
            except Exception as e:
                _p(f"  [ERROR Q{idx+1}]: {e}")

    # Reorder results by q_idx
    ordered_results = [results[k] for k in sorted(results.keys())]

    # Final Report
    n = len(ordered_results)
    gpt_acc = sum(1 for r in ordered_results if r["gpt55_correct"]) / n
    msce_acc = sum(1 for r in ordered_results if r["msce_correct"]) / n
    gpt_avg_conf = sum(r["gpt55_self_conf"] for r in ordered_results) / n
    msce_avg_conf = sum(r["msce_confidence"] for r in ordered_results) / n
    msce_avg_disagree = sum(r["msce_disagreement"] for r in ordered_results) / n

    # Per-domain breakdown
    domains = {}
    for r in ordered_results:
        d = r["domain"]
        if d not in domains:
            domains[d] = {"n": 0, "gpt": 0, "msce": 0}
        domains[d]["n"] += 1
        if r["gpt55_correct"]:
            domains[d]["gpt"] += 1
        if r["msce_correct"]:
            domains[d]["msce"] += 1

    # Confident wrong cases
    for r in ordered_results:
        if not r["gpt55_correct"] and r["gpt55_self_conf"] > 0.8:
            confident_wrong_cases.append({
                "q_idx": r["q_idx"],
                "question": r["question"][:100],
                "gpt_confidence": r["gpt55_self_conf"],
                "msce_correct": r["msce_correct"],
                "msce_confidence": r["msce_confidence"],
                "msce_disagreement": r["msce_disagreement"],
            })

    total_time = time.time() - t_start

    print("\n" + "=" * 70)
    print("FINAL RESULTS — MSCE v3.0 Expanded (120 Questions)")
    print("=" * 70)
    print(f"\nTotal Questions: {n}")
    print(f"Total Time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"\n| Metric | GPT-5.5 | MSCE v3.0 |")
    print(f"|--------|---------|------------|")
    print(f"| Accuracy | {gpt_acc:.1%} | {msce_acc:.1%} |")
    print(f"| Avg Confidence | {gpt_avg_conf:.2f} | {msce_avg_conf:.2f} |")
    print(f"| Avg Disagreement | — | {msce_avg_disagree:.2f} |")

    print(f"\n--- Per-Domain Breakdown ---")
    print(f"| Domain | n | GPT-5.5 | MSCE |")
    print(f"|--------|---|---------|------|")
    for d in sorted(domains.keys()):
        dd = domains[d]
        print(f"| {d} | {dd['n']} | {dd['gpt']/dd['n']:.1%} | {dd['msce']/dd['n']:.1%} |")

    print(f"\n--- Confident Wrongness (GPT-5.5) ---")
    print(f"GPT-5.5 confidently wrong (self-conf > 0.8 but answer wrong): {len(confident_wrong_cases)} cases")

    output = {
        "version": "3.0-expanded-optimized",
        "n_questions": n,
        "total_time_s": round(total_time, 1),
        "optimizations": ["3-way parallel", "judge cascade", "max_retries=1"],
        "timestamp": time.time(),
        "results": ordered_results,
        "summary": {
            "n_questions": n,
            "gpt55_accuracy": round(gpt_acc, 4),
            "msce_accuracy": round(msce_acc, 4),
            "gpt55_avg_confidence": round(gpt_avg_conf, 4),
            "msce_avg_confidence": round(msce_avg_conf, 4),
            "msce_avg_disagreement": round(msce_avg_disagree, 4),
            "confident_wrong_cases": len(confident_wrong_cases),
            "total_time_s": round(total_time, 1),
            "domain_breakdown": {d: {"n": dd["n"], "gpt55_acc": round(dd["gpt"]/dd["n"], 4), "msce_acc": round(dd["msce"]/dd["n"], 4)} for d, dd in domains.items()},
        }
    }

    outpath = os.path.join(os.path.dirname(__file__), "results/expanded_benchmark_results.json")
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to {outpath}")
    return output


if __name__ == "__main__":
    main()
