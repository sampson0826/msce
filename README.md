# stabilitybench

**Measure LLM recursive stability via constraint residual beta.**

[![PyPI - Version](https://img.shields.io/pypi/v/stabilitybench)](https://pypi.org/project/stabilitybench/)
[![arXiv](https://img.shields.io/badge/arXiv-coming_soon-b31b1b)](https://arxiv.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![GPU free](https://img.shields.io/badge/GPU%20free-yes-success.svg)]()

When AI generates data to train the next AI, quality degrades. stabilitybench tells you **what** is breaking -- logic, style, or facts -- and recommends the fix. No GPU needed.

## Quick Install

```bash
pip install stabilitybench
```

## Quick Start

```python
from synthetic_decay_monitor import (
    DecayEngine, HybridConstraintExtractor,
    generate_synthetic_lineage, generate_json_report,
)

# 1. Create or load data lineage
lineage = generate_synthetic_lineage(
    ["Prove sqrt(2) is irrational.", "Explain photosynthesis."],
    n_generations=4,
)

# 2. Run decay analysis (CPU-only, <1 second)
engine = DecayEngine(lineage, HybridConstraintExtractor())
engine.run_all_capabilities()

# 3. Get results
for t in engine.get_all_trajectories():
    print(f"{t['capability']}: beta={t['beta']:.3f}, collapse gen {t['predicted_collapse_gen']}")
```

Or via CLI:

```bash
stabilitybench --preset qwen --generations 4
```

## What it measures

Traditional metrics (perplexity, BERTScore) only tell you **that** quality dropped. stabilitybench tells you **why**:

| Executor | What breaks | Alpha | Signature |
|----------|-------------|-------|-----------|
| **E-I** Axiom | Logic chains, reasoning | 0.40 | "therefore", "because" vanish |
| **E-II** Scale | Vocabulary, style | 0.20 | Bigram repetition, filler words multiply |
| **E-III** Boundary | Facts, names, numbers | 0.08 | Proper nouns erode, numbers randomize |

## Citation

```bibtex
@software{decay_eval_2026,
  author = {Deng, Xinhang},
  title = {stabilitybench: Measure LLM recursive stability via constraint residual beta},
  year = {2026},
  url = {https://github.com/constraint-residual/decay-monitor}
}
```

## License

MIT -- Deng Xinhang, 2026.
