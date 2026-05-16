# Decay Monitor

**Your AI training data is dying. This tells you how — and what to do about it.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-65%2F65-brightgreen.svg)](tests/)
[![GPU free](https://img.shields.io/badge/GPU%20free-yes-success.svg)]()

<p align="center">
  <img src="https://img.shields.io/badge/accuracy-86%25-success?style=for-the-badge" alt="86% accuracy">
  <img src="https://img.shields.io/badge/speed-%3C1s-blue?style=for-the-badge" alt="<1 second">
</p>

---

## The problem

When AI generates data to train the next AI, quality degrades. Everyone knows this ("model collapse"). But nobody can answer the question that matters:

> **Is my data losing its logic, its style, or its facts?**

These are three completely different diseases. They need three completely different cures. If you treat logic collapse with more style data, you waste GPU hours. If you treat fact erosion with more proof data, you're digging a different hole.

## What this does

```
$ decay-monitor scan qwen-gen5-samples.jsonl

  math_reasoning           ███░░░░░░░░░░░░░░░░░ E-I_loss      💀 COLLAPSED
  code_generation          ████░░░░░░░░░░░░░░░░ E-I_loss      💀 COLLAPSED  
  factual_knowledge        ███████░░░░░░░░░░░░░ E-III_loss    ⚠  CRITICAL
  creative_writing         ██░░░░░░░░░░░░░░░░░░ E-II_loss     💀 COLLAPSED
  general                  ███░░░░░░░░░░░░░░░░░ E-II_loss     💀 COLLAPSED

  🔴 creative_writing collapses FIRST  → gen 3
  🟢 factual_knowledge is most resilient → gen 7

  ── Diagnosis ──────────────────────────────────────
  4x E-II (Style collapse — vocabulary erosion)
     → Add 200 diverse style exemplars + human preference pairs
  2x E-I  (Axiom collapse — logic chain breaking)  
     → Add 500 formal proofs + multi-step reasoning chains
  
  ── Intervention priority ──────────────────────────
  1. 🔴 Calibration data — style diversity (most urgent)
  2. 🟡 Axiom data — formal logic chains
  3. 🟢 Boundary data — edge cases + rare facts (preventive)
```

In 1 second. On CPU. No GPU. No API key.

## The three ways data dies

| Type | What breaks | How you spot it | How you fix it |
|------|------------|-----------------|----------------|
| **E-I** Axiom collapse | Logic chains, reasoning, proofs | Logic connector words vanish ("therefore", "because", "thus") | Add formal proofs, derivations, multi-step reasoning |
| **E-II** Scale collapse | Vocabulary diversity, style, creativity | Words get shorter, filler words multiply, bigrams repeat | Add diverse style exemplars, human preference data |
| **E-III** Boundary collapse | Facts, numbers, proper nouns | Capital letters disappear, numbers lose precision | Add edge cases, domain-specific boundary examples |

Traditional quality metrics (perplexity, BERTScore) only tell you **that** quality dropped — never **why**. This tool tells you the why. And the fix.

## Real experiment: Qwen2.5-7B recursive generation

We took Qwen2.5-7B-Instruct and recursively used its outputs as inputs for 5 generations — a real-world simulation of synthetic data training loops.

**Key findings:**

- **creative_writing dies first** (gen 3, β=0.313) — Style erosion is the canary in the coal mine
- **math_reasoning follows** (gen 4, β=0.266) — Logic structure collapses next  
- **factual_knowledge is the last to fall** (gen 7, β=0.173) — Facts are surprisingly durable
- **E-II (style collapse) dominates** — 4 of 6 capabilities show vocabulary erosion as the primary failure mode

This means: if you're doing recursive synthetic data training, **monitor your creative/stylistic outputs first**. They'll break before your reasoning outputs do. Most teams watch reasoning benchmarks — that's watching the wrong thing.

## Comparison: Decay Monitor vs LLM-as-Judge

We benchmarked against Qwen2.5-7B-Instruct acting as an LLM judge — the standard approach to automated text quality assessment.

| | Decay Monitor | LLM Judge (Qwen2.5-7B) |
|---|---|---|
| Accuracy (executor typing) | **86%** | 25% |
| E-I detection | ✅ 100% | ✅ 100% |
| E-II detection | ✅ 100% | ❌ 25% |
| E-III detection | ✅ 100% | ❌ 25% |
| Speed | < 1 second | ~120 seconds |
| GPU required | No | Yes (16 GB) |
| Bias | E-II slightly overestimated in mixed cases | **Severe E-I bias** (classifies everything as logic collapse) |

**Why the LLM judge fails:** A 7B model sees all quality degradation as "everything drops together." It can't distinguish between "the logic is broken" and "the style is boring" because its 5-dimension scoring is inherently co-linear for degraded text. Our 8 text features are statistically independent — each targets a specific degradation fingerprint.

## Quick Start

```bash
git clone https://github.com/your-org/decay-monitor.git
cd decay-monitor

# Install (CPU only, no dependencies beyond numpy)
pip install numpy matplotlib

# Run the demo
python -m constraint_residual.synthetic_decay_monitor.cli --demo 6 --hybrid --paper

# Scan your own data
python -m constraint_residual.synthetic_decay_monitor.cli --input your_data.jsonl --hybrid --output report.html
```

Your JSONL format:
```json
{"id": "0", "text": "Therefore, because the integral...", "generation": 0, "capability_tags": ["math_reasoning"]}
{"id": "1", "text": "The French Revolution began in 1789...", "generation": 1, "capability_tags": ["factual_knowledge"]}
```

## How it works (the 30-second version)

1. **8 text features** extracted per sample — logic connector density, bigram repetition rate, proper noun capitalization, number precision, syntax variation, filler word ratio, unique word ratio, word truncation rate
2. **Cross-generation delta** computed — how much each feature changed from gen N-1 to gen N
3. **Executor composition estimated** — the signature of which features changed maps to which executor type is failing
4. **Layer traversal** theory — E-III (boundary, α=0.08) exposes first, then E-II (scale, α=0.20), then E-I (axiom, α=0.40). The order is predictable.

No neural network. No embedding model. Just counting words and comparing deltas. That's why it's fast and works everywhere.

## Server / API

```bash
# Start FastAPI server
uvicorn server:app --host 0.0.0.0 --port 8000

# Upload JSONL for analysis
curl -X POST http://localhost:8000/analyze -F "file=@samples.jsonl"

# Get dashboard
open http://localhost:8000/dashboard

# Docker
docker build -t decay-monitor .
docker run -p 8000:8000 decay-monitor
```

Endpoints: `/analyze` (POST), `/analyze/demo` (GET), `/report/{id}`, `/history`, `/dashboard`, `/alerts`, `/health`

## Paper

See [VALIDATION_REPORT.md](VALIDATION_REPORT.md) for the full experimental validation report (v3.0, 2026-05-10).

Generate publication-quality figures:
```bash
python -m constraint_residual.synthetic_decay_monitor.cli --demo 6 --hybrid --paper --paper-dir paper_figures
```

Produces 4 figures (PNG + SVG):
- Stability trajectories (S_n per generation)
- Executor composition stacked area charts (layer traversal)
- Text feature fingerprints (gen0 vs genN comparison)
- Collapse prediction timeline

## Architecture

```
data_lineage.py       — Parse JSONL, track data provenance across generations
constraint_extractor.py — 8 text features + Hybrid extractor (no model needed)
decay_engine.py       — Estimate β, compute stability trajectories, executor composition
executor_classifier.py — Diagnose degradation type, recommend interventions
stress_propagation.py — Simulate cascade failure across capability graph
server.py             — FastAPI server + SQLite history + webhook alerting
report.py             — JSON/HTML/paper-figure generation
```

## Roadmap

- [x] Hybrid text-feature extractor (86% accuracy, no GPU)
- [x] Real Qwen2.5-7B recursive generation experiment
- [x] FastAPI server + Docker
- [x] Paper-quality figure generation
- [ ] Multi-language support (Chinese verified, others pending)
- [ ] Real-time training pipeline integration (wandb callback)
- [ ] Enterprise dashboard (multi-project, alert history, trend analysis)

## License

MIT — use it, fork it, build a business on it. Just credit the source.

## Citation

```bibtex
@software{decay_monitor_2026,
  author = {Deng, Xinhang},
  title = {Decay Monitor: Constraint-layer health diagnostic for synthetic data pipelines},
  year = {2026},
  url = {https://github.com/your-org/decay-monitor}
}
```
