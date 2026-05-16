# StabilityBench Dataset — Zenodo Record

**DOI:** 10.5281/zenodo.20041757
**Version:** v2
**Paper:** The Recursive Stability Index: A Benchmark for Multi-Generation LLM Degradation
**Author:** Deng Xinhang

## Dataset Contents

### 1. Primary Benchmark Data (16 models)
- `all_models_summary.json` — Global β + per-capability β for all 16 models
- `n100/` — Full lineage data (JSONL) for models evaluated at n=100 seeds
- `latest_models/` — Lineage data for recently released models

### 2. Baseline Comparisons
- `perplexity_baseline_full.json` — Continuation perplexity baseline (9 models, n=10 seeds)
- `pure_judge/` — Multi-judge pure constraint extraction results (3 judges × 78 texts)
- Cross-seed validation data in `cross_seed/`

### 3. K=5 Depth Extension
- `k5/` — K=5 experiment data (5 models: 3 DeepSeek + GPT-4o-mini + Claude Opus 4.7)

### 4. Downstream Task Validation
- `downstream_code_v3_results.json` — Code generation β vs. quality correlation (5 models × 35 problems)

### 5. Neural Validation (P3)
- `p3_results.json` — Per-token gradient decomposition on Qwen2.5-1.5B-Instruct

### 6. Analysis Scripts
- `run_benchmark.py` — Main benchmark runner
- `run_k5_experiment.py` — K=5 depth extension
- `run_perplexity_full.py` — Perplexity baseline computation
- `run_downstream_code_v3.py` — Downstream code validation
- `pure_judge_extractor.py` — Multi-judge pure constraint extraction
- `analyze_k5.py` — K=5 exponential vs. linear analysis
- `integrate_results-2026-05-17.py` — Result integration script

### 7. Core Library
- `synthetic_decay_monitor/` — Full Python library:
  - `constraint_extractor.py` — HybridConstraintExtractor
  - `decay_engine.py` — β computation engine
  - `provider_adapter.py` — Multi-provider API interface
  - `data_lineage.py` — Lineage data structures
  - `executor_classifier.py` — Executor decay diagnosis
  - `report.py` — JSON report generation

### 8. Paper
- `paper/main.md` — English paper (Markdown)
- `paper/main_cn.md` — Chinese paper (Markdown)
- `paper/main.pdf` — English paper (PDF)
- `paper/main_cn.pdf` — Chinese paper (PDF)

## File Format Notes

**JSONL lineage files:** One JSON object per line. Each entry:
```json
{
  "id": "G1_gpt-4o-mini_s100_0027",
  "generation": 1,
  "text": "...",
  "capability_tags": ["creative_writing"],
  "source_model": "gpt-4o-mini"
}
```

**Report files:** Standard JSON with `decay_analysis` key containing β, per-capability trajectories, and executor diagnoses.

## Reproducibility

1. Install: `pip install -r requirements.txt`
2. Set API keys in `.env` (QUICKROUTER_API_KEY, DEEPSEEK_API_KEY)
3. Run: `python3 run_benchmark.py --model gpt-4o-mini --provider quickrouter`

All β values are reproducible at reported precision (test-retest confirmed on GPT-4o-mini).

## License

CC-BY-4.0
