"""MSCE Audit: Google Co-Scientist (Nature, May 2026)
Cross-constraint verification of core claims.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from product_engine import run_msce

QUESTIONS = [
    {
        "id": "Q1",
        "claim": "Elo tournament selects for correctness",
        "question": (
            "Google Co-Scientist uses an Elo tournament where 6 AI agents debate hypotheses "
            "and an AI judge scores pairwise debates. The paper claims this tournament selects "
            "the best scientific hypotheses. "
            "Critical question: Does an Elo tournament with AI judges select for scientific "
            "correctness, or does it select for debate persuasiveness? "
            "Consider: (1) Elo measures win/loss, not truth; (2) AI judges are known to exhibit "
            "sycophancy bias (preferring arguments that match their priors); "
            "(3) the tournament rewards hypotheses that sound convincing, not necessarily correct. "
            "Answer: TRUE (selects correctness) or FALSE (selects persuasiveness) with reasoning."
        ),
    },
    {
        "id": "Q2",
        "claim": "Multi-agent structure self-corrects errors",
        "question": (
            "The Google Co-Scientist paper claims its 6-agent tournament structure 'self-corrects' "
            "errors through iterative debate and refinement. "
            "However, an independent arXiv evaluation of similar scientific AI agents found that "
            "agents ignore evidence in 68% of reasoning traces and revise their beliefs only 26% "
            "of the time when presented with contrary findings. "
            "Are these two claims logically consistent? Can a system where agents ignore evidence "
            "68% of the time and rarely revise beliefs (26%) genuinely 'self-correct'? "
            "Answer CONSISTENT or INCONSISTENT with reasoning."
        ),
    },
    {
        "id": "Q3",
        "claim": "Tournament scaffold drives performance gains",
        "question": (
            "The Co-Scientist paper attributes its hypothesis-generation performance to the "
            "multi-agent tournament architecture. But independent analysis found that the base "
            "language model (Gemini) accounts for 41.4% of performance variance — more than the "
            "agent scaffold. "
            "Does the evidence support the claim that the tournament scaffold is the primary "
            "driver of Co-Scientist's performance, or does the base model deserve more credit? "
            "Answer: SCAFFOLD (scaffold is primary driver) or BASE_MODEL (base model is primary driver)."
        ),
    },
    {
        "id": "Q4",
        "claim": "Quality scales with test-time compute without saturation",
        "question": (
            "The Co-Scientist paper claims that hypothesis quality 'scales with test-time compute' "
            "without observed saturation. "
            "However, if 68% of agent reasoning ignores evidence and belief revision is only 26%, "
            "then more compute means more rounds of agents ignoring evidence and not revising beliefs. "
            "Is the 'quality scales with compute' claim logically compatible with the evidence-ignoring "
            "behavior? Or does more compute simply amplify a flawed process? "
            "Answer COMPATIBLE or INCOMPATIBLE with reasoning."
        ),
    },
    {
        "id": "Q5",
        "claim": "3 successful validations prove general capability",
        "question": (
            "The Co-Scientist paper presents 3 successful wet-lab validations (AML drug repurposing, "
            "liver fibrosis, antimicrobial resistance) as evidence of the system's general capability. "
            "But the paper tested 11 biomedical problems and achieved notable success in only 3. "
            "Furthermore, all validations are preclinical (no human trials). "
            "Do 3 successes out of 11 attempts (27% success rate), all in preclinical settings, "
            "constitute sufficient evidence for the claim of 'accelerating scientific discovery'? "
            "Answer SUFFICIENT or INSUFFICIENT with reasoning."
        ),
    },
]

def main():
    results = []
    print("=" * 70)
    print("MSCE Audit: Google Co-Scientist (Nature, May 2026)")
    print("Cross-Constraint Verification of 5 Core Claims")
    print("=" * 70)

    for q in QUESTIONS:
        print(f"\n{'─' * 60}")
        print(f"[{q['id']}] Testing claim: {q['claim']}")
        print(f"{'─' * 60}")

        t0 = time.time()
        result = run_msce(q["question"], domain="science")
        elapsed = time.time() - t0

        result["claim"] = q["claim"]
        result["question_id"] = q["id"]
        results.append(result)

        # Print summary
        print(f"  Answer:       {result.get('top_answer', 'N/A')[:120]}")
        print(f"  Confidence:   {result.get('confidence', 0):.3f}")
        print(f"  Disagreement: {result.get('disagreement', 0):.3f}")
        print(f"  Uncertain:    {result.get('uncertain', 'N/A')}")
        print(f"  Top Strategy: {result.get('top_strategy', 'N/A')}")
        print(f"  Time:         {elapsed:.1f}s")

        # Per-model details
        for entry in result.get("reasoning_trail", []):
            status = entry.get("status", "?")
            icon = {"selected": "+", "contributing": "~", "outlier": "!", "low_confidence": "-"}
            print(f"    [{icon.get(status, '?')}] {entry['strategy']:25s} | "
                  f"conf={entry.get('self_confidence', 0):.2f} | "
                  f"score={entry.get('judge_score', 0):.1f} | "
                  f"wt={entry.get('weight', 0):.3f} | "
                  f"{status}")

    # Save full results
    outpath = os.path.join(os.path.dirname(__file__), "results/coscientist_audit_results.json")
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nFull results saved: {outpath}")

    # Summary
    print("\n" + "=" * 70)
    print("VERDICT SUMMARY")
    print("=" * 70)
    for r in results:
        print(f"  [{r['question_id']}] {r['claim']}:")
        print(f"      confidence={r.get('confidence',0):.3f}  disagreement={r.get('disagreement',0):.3f}  uncertain={r.get('uncertain','?')}")

if __name__ == "__main__":
    main()
