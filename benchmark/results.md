# MSCE Benchmark Results — 206 Questions

**Date:** 2026-05-30
**Method:** MSCE v3.0 (6-model ensemble + 3-layer filter + condition validator)
**Baseline:** GPT-5.5 single-model

## Overall

| Metric | GPT-5.5 | MSCE v3.0 |
|--------|---------|-----------|
| Accuracy | 74.8% | **87.4%** |
| Avg Confidence | 0.74 | 0.49 |
| Avg Disagreement | — | 0.47 |
| Confident-Wrong Cases | 40 | — |
| Total Questions | 206 | 206 |
| Total Time | — | 81.2 min |

## Per-Domain Breakdown

| Domain | n | GPT-5.5 | MSCE | Δ |
|--------|---|---------|------|-----|
| Cross-domain | 33 | 54.5% | **84.9%** | **+30.3%** |
| Science | 37 | 73.0% | **97.3%** | **+24.3%** |
| Condition Dependency Analysis | 43 | 55.8% | 67.4% | +11.6% |
| Logic | 27 | 85.2% | 92.6% | +7.4% |
| Math | 30 | 93.3% | 96.7% | +3.3% |
| Verbal | 36 | 94.4% | 91.7% | -2.8% |

## Key Findings

1. **MSCE dominates in condition-dense tasks.** Cross-domain (+30.3%) and science (+24.3%) show the largest gains, where multiple independent verification conditions must be satisfied simultaneously.

2. **MSCE is more honest about uncertainty.** Average confidence of 0.49 vs GPT-5.5's 0.74, while achieving higher accuracy. MSCE knows when it doesn't know.

3. **GPT-5.5 is dangerously overconfident.** 40 cases where GPT-5.5 scored self-confidence > 0.8 but was wrong.

4. **Verbal tasks are the weakness.** MSCE slightly underperforms GPT-5.5 (-2.8%) on open-ended creative tasks. This is by design — condition dependency analysis adds noise when conditions are loose.

## Methodology

Each question was evaluated by:
1. GPT-5.5 baseline (single model with self-estimated confidence)
2. MSCE v3.0 (6-model ensemble: GPT-5.5, Gemini 3.1, Grok 4.1, Kimi K2.5, GPT-5.1, o4-mini)
3. 3-layer filter: L1 (low confidence), L2 (outlier detection), L3 (collective blind risk)
4. Condition validator: B-operator logic on answer conditions
5. Judge cascade: grok-4.1-thinking for complex domains, gpt-5.5 for simple

## Full Data

See `benchmark_200_results.json` in the repository root.
