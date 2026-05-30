# MSCE — Multi-Source Constraint Engine

**Detect hidden conflicts between scientific theories and the data they cite.**

[![PyPI](https://img.shields.io/badge/pip%20install-msce-blue)](https://pypi.org)
[![Python](https://img.shields.io/badge/python-3.10%2B-green)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
[![Stars](https://img.shields.io/github/stars/msce-ai/msce)](https://github.com/msce-ai/msce/stargazers)

<p align="center">
  <img src="assets/demo.gif" width="700" alt="MSCE demo">
</p>

## What is this?

When a physicist proposes a theory to solve the Hubble tension, they check 1-2 constraints. But there are **8 independent observational constraints** that must ALL be satisfied simultaneously. MSCE checks them all at once — and reveals conflicts no single reviewer can catch.

## 30-Second Demo

```bash
pip install msce
msce check hubble --quick
```

**Output:** A heatmap showing 6 mainstream H₀ solutions × 8 constraints. **All red.**

<p align="center">
  <img src="assets/heatmap.png" width="600" alt="6 solutions × 8 constraints heatmap">
</p>

## The Hubble Tension Result

| Proposal | Passes | Violations | MSCE Confidence |
|----------|--------|------------|-----------------|
| Early Dark Energy (EDE) | 3 | 3 | **0.076** |
| Modified Gravity (f(R)) | 3 | 4 | 0.253 |
| Extra Neutrinos (ΔN_eff) | 3 | 2 | 0.287 |
| Decaying Dark Matter (DDM) | 5 | 2 | 0.358 |
| Local Void | 6 | 2 | 0.171 |
| Systematic Error | 6 | 0 | 0.108 |

**Even 2-factor combinations perform worse than single proposals** — because the mechanisms interact nonlinearly, creating new conflicts rather than resolving existing ones.

→ [Full analysis notebook](notebooks/hubble_tension.ipynb)

## Run It Yourself

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/msce-ai/msce/blob/main/notebooks/hubble_tension.ipynb)

```python
import msce

# Check a paper against known cosmological constraints
result = msce.analyze("hubble_tension", quick=True)
print(f"Confidence: {result['confidence']:.3f}")
print(f"All proposals fail cross-constraint check: {result['all_fail']}")

# Check your own theory
result = msce.check(
    theory="My modified gravity model",
    constraints=["cmb_spectrum", "bao_scale", "sn_hubble", "bbn", "s8", "age", "gravity", "cross"],
    domain="cosmology"
)
```

## What MSCE Does

MSCE is not a "smarter AI" — it's a **constraint conflict detector**. It doesn't tell you what's right. It tells you what can't ALL be right at the same time.

```
Your theory ──→  [6-model ensemble]  ──→ Cross-constraint consistency check
                        │                            │
                  6 independent              8 observational
                  LLMs vote on                constraints verified
                  each constraint             simultaneously
```

## Benchmark: 206 Questions

MSCE achieves **87.4% accuracy** across 206 constraint-heavy questions, compared to GPT-5.5's 74.8% — a **+12.6% improvement**.

| Domain | GPT-5.5 | MSCE | Δ |
|--------|---------|------|-----|
| **Cross-domain** | 54.5% | **84.9%** | **+30.3%** |
| **Science** | 73.0% | **97.3%** | **+24.3%** |
| Constraint Propagation | 55.8% | 67.4% | +11.6% |
| Logic | 85.2% | 92.6% | +7.4% |
| Math | 93.3% | 96.7% | +3.3% |
| Verbal | 94.4% | 91.7% | -2.8% |

MSCE excels in constraint-dense domains. It falls slightly behind in open-ended creative tasks — **and that's by design.**

## Installation

```bash
pip install msce
```

Requirements: Python 3.10+, no GPU needed.

## Documentation

- [Quickstart](examples/quickstart.py)
- [Hubble Tension Analysis](notebooks/hubble_tension.ipynb)
- [Benchmark Results](benchmark/results.md)
- [API Reference](https://github.com/msce-ai/msce/wiki)

## FAQ

**Is this an AGI?** No. It's a specialized constraint-checking system that uses 6 LLMs as voters.

**Can it check my paper?** Yes. `msce check paper.pdf --constraints constraints.json`

**What domains does it support?** Currently cosmology and physics. Engineering and biology coming later.

**Is the code fully open source?** The CLI and data formats are MIT-licensed. The 6-model voting engine is available as a hosted API.

## License

MIT — see [LICENSE](LICENSE) for details.

## Citation

If you use MSCE in your research:

```bibtex
@software{msce2026,
  title={MSCE: Multi-Source Constraint Engine},
  author={Deng, Xinhang and MSCE Collaboration},
  year={2026},
  doi={10.5281/zenodo.20041757},
  url={https://github.com/msce-ai/msce}
}
```
