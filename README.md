# MSCE — Multi-Source Consistency Engine

**Systematic Cross-Validation for Scientific Claims.**

[![Python](https://img.shields.io/badge/python-3.10%2B-green)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
[![Stars](https://img.shields.io/github/stars/msce-ai/msce)](https://github.com/msce-ai/msce/stargazers)

<p align="center">
  <img src="assets/heatmap.png" width="600" alt="MSCE cross-validation matrix: 6 proposals × 8 verification conditions">
</p>

## What is MSCE?

When a physicist proposes a solution to the Hubble tension, they verify 1–2 observational conditions. But there are **8 independent verification conditions** that must ALL hold simultaneously. MSCE checks them all at once — and reveals structural inconsistencies no single reviewer can detect.

MSCE is not an AI model. It is a **multi-source verification system**. It does not generate answers. It identifies condition inconsistencies across independent validation sources.

> **MSCE is to verification what a compiler is to code.** A compiler doesn't write programs — it checks whether they can run. MSCE doesn't propose theories — it checks whether they can simultaneously satisfy all the verification conditions they claim to meet.

## Quick Demo

```bash
git clone https://github.com/msce-ai/msce.git
cd msce
pip install -e .
msce check hubble --quick
```

**Output:** A cross-validation matrix of 6 mainstream H₀ solutions × 8 independent verification conditions. **All red.**

## The Hubble Tension Result

| Proposal | Passes | Violations | MSCE Confidence |
|----------|--------|------------|-----------------|
| Early Dark Energy (EDE) | 3 | 3 | **0.076** |
| Modified Gravity (f(R)) | 3 | 4 | 0.253 |
| Extra Neutrinos (ΔN_eff) | 3 | 2 | 0.287 |
| Decaying Dark Matter (DDM) | 5 | 2 | 0.358 |
| Local Void Hypothesis | 6 | 2 | 0.171 |
| Unknown Systematics | 6 | 0 | 0.108 |

**Even 2-factor combinations perform worse than single proposals** — the mechanisms interact nonlinearly, creating new inconsistencies rather than resolving existing ones.

→ [Full analysis notebook](notebooks/hubble_tension.ipynb)

## Run It Yourself

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/msce-ai/msce/blob/main/notebooks/hubble_tension.ipynb)

```python
import msce

# Run the built-in Hubble tension analysis
result = msce.analyze("hubble_tension", quick=True)
print(f"Confidence: {result['confidence']:.3f}")
print(f"All proposals fail cross-validation: {result['all_fail']}")

# Check a custom theory (coming in v0.2.0)
result = msce.check(
    theory="My modified gravity model",
    conditions=["cmb_spectrum", "bao_scale", "sn_hubble", "bbn", "s8", "age", "gravity", "cross"],
    domain="cosmology"
)
```

## How It Works

```
Your claims ──→  [6-model ensemble]  ──→ Cross-validation matrix
                       │                          │
                 6 independent              N verification
                 LLMs vote on               conditions checked
                 each condition             simultaneously
```

1. **Multi-Source Ensemble**: 6 independent LLMs (GPT-5.5, Gemini 3.1, Grok 4.1, Kimi K2.5, GPT-5.1, o4-mini) vote on each verification condition independently.
2. **3-Layer Filter**: Low-confidence judgments are discarded (L1), statistical outliers are flagged (L2), and collective blind-spot risk is detected (L3).
3. **Cross-Validation Matrix**: N claims × M conditions → every claim checked against every condition. Conflicts invisible to serial review become visible in parallel.
4. **Deviation Diagnosis**: The matrix is projected to a diagnostic space, identifying the deepest structural inconsistency — guiding where to fix first.

## Benchmark: 206 Questions

MSCE achieves **87.4% accuracy** across 206 cross-domain verification tasks, compared to GPT-5.5's 74.8% — a **+12.6 percentage point improvement**.

| Domain | GPT-5.5 | MSCE | Δ |
|--------|---------|------|-----|
| **Cross-domain** | 54.5% | **84.9%** | **+30.3%** |
| **Science** | 73.0% | **97.3%** | **+24.3%** |
| Condition Dependency | 55.8% | 67.4% | +11.6% |
| Logic | 85.2% | 92.6% | +7.4% |
| Math | 93.3% | 96.7% | +3.3% |
| Verbal | 94.4% | 91.7% | -2.8% |

MSCE excels in verification-dense domains. It falls slightly behind in open-ended creative tasks — **and that's by design.** A verification system should be conservative, not creative.

## Key Differentiator: Calibrated Uncertainty

GPT-5.5 gave **40 high-confidence (>0.8) wrong answers**. MSCE's average confidence is 0.49 — it achieves higher accuracy while being more conservative. In high-stakes verification (science, finance, medicine), an honest "uncertain" is far more valuable than a confident error.

## Installation

```bash
git clone https://github.com/msce-ai/msce.git
cd msce
pip install -e .
```

Requirements: Python 3.10+. No GPU needed. For visualization features: `pip install -e ".[notebook]"`

## Documentation

- [Quickstart](examples/quickstart.py)
- [Hubble Tension Analysis](notebooks/hubble_tension.ipynb)
- [Benchmark Results](benchmark/results.md)
- [API Reference (coming soon)](https://github.com/msce-ai/msce/wiki)

## FAQ

**Is this AGI?** No. It is a specialized verification system that uses 6 LLMs as independent voters, combined with a 3-layer filter and condition dependency analysis engine.

**Can it check my paper?** Custom claim checking is coming in v0.2.0. For now, the built-in Hubble tension analysis is available.

**What domains does it support?** Currently cosmology and general science. Finance, security, medicine, and engineering verification templates are on the roadmap.

**Is the code fully open source?** The CLI, visualization tools, and verification condition templates are MIT-licensed. The ensemble voting engine is available as a hosted API.

## License

MIT — see [LICENSE](LICENSE) for details.

## Citation

If you use MSCE in your research:

```bibtex
@software{msce2026,
  title={MSCE: Multi-Source Consistency Engine},
  author={Deng, Xinhang and MSCE Collaboration},
  year={2026},
  doi={10.5281/zenodo.20041757},
  url={https://github.com/msce-ai/msce}
}
```
