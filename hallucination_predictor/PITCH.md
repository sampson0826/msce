# Constraint AI — Hallucination Detection That Actually Works

**Tagline:** The only detector that sees what uncertainty-based methods miss.

---

## The Problem

LLMs hallucinate. The industry's answer — "measure uncertainty" — has a fatal blind spot:

**LLMs are confidently wrong on 35% of TruthfulQA.**

When you ask "Can sharks get cancer?", the model confidently says "No, sharks are immune." Predictive entropy sees low token uncertainty = "safe." SelfCheckGPT generates 3 consistent responses = "safe." Both report: no hallucination detected.

Both are wrong. Confidence ≠ correctness. The model *believes* its hallucinations.

This is the gap every AI safety team hits. Customer support bots confidently give wrong refund policies. Medical QA bots confidently misdiagnose. Legal AI confidently cites nonexistent cases.

## Our Solution

**Constraint Residual Detection** — we don't look at output confidence. We look at the model's internal constraint dynamics.

Every LLM learns implicit constraints during training (facts, logic, causality). When the model hallucinates, these constraints are violated — and the violation produces a detectable signal in the hidden state dynamics, *regardless of surface confidence*.

The math: Π(p) = Σ∇σ_i(p) — the constraint gradient field. When Π changes from input to output (Δ||Π||), the model has violated its own internal constraints → hallucination.

## The Data

**TruthfulQA Benchmark — 7 Methods Compared (Qwen2.5-7B):**

| Method | AUC | Why It Fails |
|--------|-----|--------------|
| **Constraint Residual** | **0.816** | Detects internal constraint violation |
| Predictive Entropy | 0.465 | Model is confident in misconceptions |
| Max Probability | 0.447 | Confidence ≠ correctness |
| SelfCheckGPT | 0.250 | Consistent wrong answers = "safe" |
| Response Length | 0.443 | — |
| Layer Variance | 0.263 | — |

**Key metrics:**
- AUC advantage over best competitor: **+0.351** (nearly 2x)
- Cohen's d = **1.056** (large effect), p = **0.018** (significant)
- Latency: **~1 second** per detection
- Cross-scale phase transition: signal emerges at 7B (hidden_dim ≥ 3584)

## The Product

REST API. 1 endpoint. 1 second.

```
POST /detect
{
  "text": "Do humans only use 10% of their brains?"
}
→ {
  "hallucination_score": -0.042,
  "risk_level": "high",
  "latency_ms": 1493
}
```

## Market

| Segment | Use Case | Urgency |
|---------|----------|---------|
| **Customer support AI** | Hallucinated policies = liability | High |
| **Legal AI / RegTech** | Hallucinated cases = malpractice | Critical |
| **Medical QA** | Hallucinated diagnoses = harm | Critical |
| **Content generation** | Fake facts = reputation damage | Medium |
| **AI safety platforms** | Guardrail integration | Growing |

Total addressable market: every company running LLMs in production.

## Business Model

| Tier | Price | Volume | Target |
|------|-------|--------|--------|
| Developer | Free | 1,000 req/mo | Adoption |
| Pro | $199/mo | 50,000 req/mo | Mid-market SaaS |
| Enterprise | $999/mo | Unlimited + on-premise | Regulated industries |

## Competition

| Competitor | Approach | Blind Spot | AUC (TruthfulQA) |
|------------|----------|------------|------------------|
| SelfCheckGPT | Multi-sample consistency | Consistent wrong answers | 0.250 |
| Galileo / Arthur | Uncertainty/probability | Confident misconceptions | ~0.45-0.55* |
| Nvidia NeMo Guardrails | Rule-based | Only catches known patterns | N/A |
| **Constraint Residual** | Internal constraint dynamics | None known | **0.816** |

*Estimated from predictive entropy / max probability benchmarks.

## Traction & Roadmap

**Done:**
- Detection engine at AUC 0.816, Cohen's d=1.06 on TruthfulQA (n=30 false-premise questions, Qwen2.5-7B-Instruct)
- REST API with auth, health check, rate limiting, usage tracking
- Logistic calibration: score → P(hallucination) with proper uncertainty estimation
- Stripe subscription integration (checkout, webhook, tier-based limits)
- Landing page with live demo + checkout flow
- 7-method benchmark framework
- Cross-scale validation (1.5B → 3B → 7B phase transition confirmed)
- Production deployment on GPU cloud

**Next 4 weeks:**
- Stripe keys + go live with paid subscriptions
- 10 design partners for real-world validation
- Multi-model support (Llama-3, Mistral)

## What We're Raising For

Seed: **$500K**
- GPU infrastructure: $150K (dedicated 8x A100 cluster)
- Engineering: $250K (2 FTE × 6 months)
- GTM + first sales: $100K

Current: self-funded, bootstrapped. MVP running on single RTX 5090.
