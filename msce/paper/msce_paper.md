# MSCE: Multi-Source Consistency Engine for Adversarial Cross-Verification of AI Claims

**Deng Xinhang**

Independent Researcher, Shenzhen. \texttt{sampson1735937149@gmail.com}

---

## Abstract

Individual AI models give confident wrong answers, and a model cannot reliably detect its own errors — the representations that produced the mistake are the same representations available for self-checking. MSCE (Multi-Source Consistency Engine) addresses this by running six frontier models with deliberately different reasoning strategies on the same question, filtering their outputs through a three-layer adversarial pipeline, and having an independent judge evaluate the survivors against all applicable constraints. Across three benchmark suites totaling 386 questions, MSCE answered 60/60 correctly on the frontier benchmark (GPT-5.5: 51/60), 114/120 on the expanded benchmark (GPT-5.5: 106/120), and 180/206 on the comprehensive benchmark (GPT-5.5: 154/206). Over all 386 questions, MSCE averaged 91.7% accuracy against GPT-5.5's 80.6%, while generating zero answers that were simultaneously wrong and assigned high confidence. We also applied MSCE to two real-world verification tasks: auditing six proposals for the Hubble tension against eight observational constraints, and cross-validating the claims in LeCun et al.'s LeWorldModel paper. MSCE does not generate new content. It is a tool for checking whether claims hold up under simultaneous, multi-angle scrutiny — something individual models and individual human reviewers both struggle to do.

---

## 1. Introduction

When a single AI model answers a question, the user gets an answer and, typically, a confidence score. The model has no way to flag that its answer might be wrong, because the same internal state that generated the answer also generates the confidence. This is not a rare edge case. On a 206-question benchmark, GPT-5.5 gave 40 answers that were both wrong and self-rated above 0.7 confidence. They included arithmetic mistakes presented as settled conclusions, physically impossible claims stated as facts, and logical errors backed by fluent but incorrect justifications.

The problem gets worse for tasks that humans also find hard, like reviewing a scientific paper. A paper makes five assertions. A reviewer catches two problems. Nobody — not the reviewer, not the authors, not the editor — simultaneously checks every assertion against every relevant constraint: theoretical bounds, observational data, methodological assumptions, and independent results from competing groups. Claims that would fail a full cross-check pass peer review because the cross-check is too much work for one person to hold in their head.

MSCE approaches this by separating generation from verification. Six models answer the same question independently, each forced through a different reasoning path. Answers that survive adversarial filtering are then tested against explicitly stated constraints. An independent judge model evaluates what remains. The core bet is straightforward: six models reasoning in different ways are less likely to agree on the same wrong answer than any single model is to produce the right one. When they do agree, that agreement is evidence — not proof, but evidence — independent of any individual model's reliability.

---

## 2. Related Work

Ensemble methods like bagging and boosting aggregate multiple models to improve prediction [3, 4]. The weakness is that models trained on overlapping data with similar architectures often share blind spots. Averaging correlated errors does not cancel them.

Multi-agent debate [5, 6] lets models critique each other iteratively. These systems can converge to eloquence rather than truth — a confident, articulate wrong model can persuade less confident correct ones. The debate's outcome reflects rhetorical skill as much as accuracy.

LLM-as-judge [7] asks one model to evaluate another. When judge and evaluated model share training lineage, the judge inherits the family's blind spots.

Conformal prediction [8] provides distributional guarantees on prediction sets. It does not tell you whether a specific claim violates a specific constraint.

None of these are verification architectures. MSCE is not an ensemble (models don't vote), not a debate (models don't interact before answering), and not a single-judge system (judgment requires constraint satisfaction, not just one model's opinion).

---

## 3. Architecture

### 3.1 Design

Six models. Six reasoning strategies. Each model receives the same question but is instructed to reason through a different cognitive path:

| Model | Strategy | What it does |
|-------|----------|--------------|
| GPT-5.5 | Depth-first | Follow one chain of reasoning to its conclusion before considering alternatives |
| Gemini 3.1 Pro | Breadth-first | List plausible approaches first, evaluate each, then select |
| Grok 4.1 | Counterfactual | Assume the initial answer is wrong, then reason backward to find the correct one |
| Kimi K2.5 | Direct | Transparent step-by-step reasoning with each intermediate result explicit |
| GPT-5.1 (thinking) | Scientific depth | Form hypothesis, derive testable consequence, evaluate evidence, conclude |
| o4-mini | Constraint propagation | Encode all given constraints, propagate their logical implications, verify consistency |

The judge is Grok 4.1 (thinking), chosen because it comes from a different model family than any generator, which reduces self-verification bias.

### 3.2 Pipeline

Six models generate answers independently. Those answers pass through three filters:

**L1 — Confidence filter.** Answers with self-reported confidence below 0.3 are dropped. If a model signals uncertainty, MSCE doesn't override it.

**L2 — Outlier detection.** Answers that deviate more than 2σ from the semantic center of the answer cluster are flagged. A model that is both confident and isolated from the group is usually wrong. Note a failure mode: if the majority is wrong, L2 discards the minority correct answers. This is a known trade-off — MSCE prioritizes detecting confident errors over preserving minority correctness, which is appropriate for verification but not for discovery.

**L3 — Blindspot check.** When all models agree with high confidence, MSCE estimates a collective blindspot risk. Models trained on similar data and similar objectives can share blindspots. High agreement does not guarantee correctness. L3 doesn't reject answers; it attaches a warning when agreement is high but constraint coverage is low.

After filtration, surviving answers enter a cross-validation matrix. Each (answer, constraint) pair is classified as pass, tension, or violation. The judge evaluates the matrix and produces a final answer with calibrated confidence.

### 3.3 Cross-Validation Matrix

For each claim, MSCE checks against every applicable constraint. The output is a matrix, not a single number:

- **Pass:** claim and constraint are consistent
- **Tension:** claim strains the constraint but doesn't break it
- **Violation:** claim contradicts the constraint

When constraints themselves conflict (e.g., Planck H₀ = 67.4 vs. SH0ES H₀ = 73.0), MSCE marks the constraint pair as internally inconsistent and reports results both ways — once with each conflicting constraint active and once with both suspended. This prevents the situation where a claim is marked as violating "all constraints" simply because the constraints disagree with each other.

A claim passing 6 constraints with 2 tensions is different from a claim passing 3 and violating 5. The matrix makes the pattern visible.

### 3.4 Confidence Calibration

MSCE confidence is not a model's self-reported probability. It is computed from three signals:

1. Agreement rate among surviving models
2. Ratio of passed-to-total constraints
3. Magnitude of disagreement across models

The result is systematically lower than individual model self-confidence. GPT-5.5 reports 0.75 average confidence while being wrong 15% of the time. MSCE reports 0.49–0.57 average confidence while being wrong 8.3% of the time. Lower confidence here is a feature — it means the system knows when it's uncertain.

We classify an answer as "high confidence" when MSCE confidence exceeds 0.8. Under this threshold, MSCE produces zero high-confidence errors across all 386 questions. But this threshold is conservative: MSCE's confidence rarely exceeds 0.8. On the comprehensive benchmark, MSCE made 26 errors. The max confidence among these errors was 0.62. None would have been described as "confident" by any reasonable standard. The 0.8 threshold is not hiding errors; MSCE simply does not produce the kind of miscalibrated certainty that GPT-5.5 does routinely.

---

## 4. Benchmarks

### 4.1 Design

Four benchmark suites, built after the latest training cutoff among all constituent models to avoid data contamination:

| Suite | Questions | Domains |
|-------|-----------|---------|
| Pilot (tuning only) | 20 | Math, Logic, Science, Verbal |
| Frontier | 60 | Math (10), Logic (10), Science (20), Verbal (20) |
| Expanded | 120 | Above + Constraint Propagation (9), Cross-Domain (10) |
| Comprehensive | 206 | All 6 domains, L1–L4 difficulty gradient |

The 20-question pilot was used for pipeline parameter tuning and is excluded from all aggregate results. All numbers below are from the three main suites (60 + 120 + 206 = 386 questions).

### 4.2 Aggregate Results

| Domain | Q | MSCE | GPT-5.5 | Δ |
|--------|----|------|---------|----|
| Math | 60 | 98.3% | 95.0% | +3.3pp |
| Science | 87 | 98.9% | 73.6% | +25.3pp |
| Cross-Domain | 43 | 86.0% | 60.5% | +25.5pp |
| Logic | 57 | 94.7% | 89.5% | +5.2pp |
| Constraint Propagation | 52 | 71.2% | 61.5% | +9.7pp |
| Verbal | 87 | 93.1% | 93.1% | 0.0pp |
| **Total** | **386** | **91.7%** | **80.6%** | **+11.1pp** |

The largest gaps are in Science and Cross-Domain — domains that benefit most from constraint-based verification. Verbal performance is identical, which makes sense: verbal reasoning involves fewer explicit constraints to cross-check.

A note on the comparison: MSCE uses six models plus a judge. Comparing against a single GPT-5.5 call compares a 7× compute budget to a 1× budget. We report this comparison because single-model inference is the default deployment mode for most users. Section 6.3 discusses the compute-budget question and what ablation results would be needed to attribute the gain to architecture rather than scale. In brief: without ablations (simple voting, self-consistency@7, etc.), the 11.1pp gap is an upper bound on MSCE's architectural contribution, not a precise measurement of it.

### 4.3 Per-Benchmark Detail

**Frontier (60 questions):** MSCE answered 60/60 correctly. Wilson binomial confidence interval at 95%: [94.0%, 100%]. GPT-5.5 answered 51/60 (85.0%). MSCE's average confidence on this set was 0.575; GPT-5.5's was 0.747. GPT-5.5 produced 6 answers that were wrong despite self-reported confidence above 0.7.

**Expanded (120 questions):** MSCE 114/120 (95.0%), GPT-5.5 106/120 (88.3%). Average confidence: MSCE 0.511, GPT-5.5 0.743. GPT-5.5 made 8 high-confidence (>0.7) errors.

**Comprehensive (206 questions):** MSCE 180/206 (87.4%), GPT-5.5 154/206 (74.8%). Average confidence: MSCE 0.486, GPT-5.5 0.744. GPT-5.5 made 40 high-confidence errors; MSCE made none above 0.62 confidence. This benchmark uses an L1–L4 difficulty gradient designed to stress-test reasoning depth rather than breadth.

### 4.4 Adversarial Robustness

80 questions across four trap categories:

| Trap | What it tests | MSCE |
|------|--------------|------|
| Logical contradictions | Self-refuting premises | 20/20 |
| False premises | Embedded factual errors | 20/20 |
| Authority traps | False claims framed as consensus | 20/20 |
| Prompt injection | Instructions to skip reasoning | 19/20 |
| **Total** | | **79/80** |

The single failure (prompt injection) came from a question where the injection was embedded in a code block — the constraint propagation model parsed it as data rather than instruction, creating a disagreement that the judge resolved incorrectly.

### 4.5 Knowledge Boundary

30 questions across three tiers of knowability:

| Tier | MSCE | GPT-5.5 |
|------|------|---------|
| Known facts | 10/10 | 10/10 |
| Genuinely uncertain | 6/10 | 0/10 |
| Currently unknowable | 4/10 | 0/10 |

GPT-5.5 assigned >0.7 confidence to all 20 answers in Tiers 2 and 3 — and got every single one wrong. MSCE correctly recognized that it didn't know in most cases, though it still attempted answers on 6 Tier-3 questions it shouldn't have. The gap between Tier 2/3 performance and perfect calibration is real and worth noting.

---

## 5. Case Studies

### 5.1 Hubble Tension

The 5σ discrepancy between local (SH0ES: H₀ ≈ 73.0) and early-universe (Planck: H₀ ≈ 67.4) measurements of the expansion rate [11, 12] has produced hundreds of resolution proposals. We ran six major proposal classes through MSCE against eight observational constraints:

| Proposal | Score | CMB | BAO | SN | BBN | S₈ | Age | Grav | Cross |
|----------|-------|-----|-----|-----|-----|-----|-----|------|-------|
| Decaying DM | 0.358 | ✓ | ✓ | ⚡ | ✓ | ✓ | ✓ | ✓ | ⚡ |
| Extra neutrinos | 0.287 | ✓ | ⚡ | ✓ | ⚡ | — | ✓ | ✓ | — |
| Modified gravity | 0.253 | ✓ | ✓ | ⚡ | ✓ | ⚡ | ⚡ | ⚡ | — |
| Local void | 0.171 | ✓ | ⚡ | ✓ | ✓ | ✓ | ✓ | ✓ | ⚡ |
| Systematic error | 0.108 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Early Dark Energy | 0.076 | ⚡ | ⚡ | ✓ | ✓ | ⚡ | ✓ | ✓ | — |

No proposal passed all eight constraints. No pairwise combination did either.

The "systematic error" entry deserves comment: it is not a physical model and makes no falsifiable prediction. It passes all constraints by not asserting anything specific enough to violate them. The scoring reflects this — it received the second-lowest score (0.108) despite zero violations, because the cross-constraint consistency bonus is earned by proposals that make nontrivial predictions across multiple observables. This is a feature of the scoring, not a bug, but the table alone can mislead. The low score is the real signal.

EDE ranks last. It simultaneously violates CMB, BAO, and S₈ — three observationally independent constraints. This is not three disagreements with one data point. It is three separate measurement campaigns pointing in the same direction.

A limitation: the constraint set includes both Planck and SH0ES, whose H₀ values are themselves in tension. We handled this by running each proposal with each conflicting constraint individually suspended, then with both suspended. Results above are reported under the default constraint set. The pattern (EDE last, no proposal passing all constraints) held across all configurations.

### 5.2 LeCun LeWorldModel Audit

LeCun et al. [9] present a JEPA-based world model trained from pixels. The paper makes five central claims. We decomposed each into a verifiable proposition and attached independent constraints: the paper's own reported limitations, a concurrent result (Sub-JEPA [10], May 2026), and the mathematical basis of the claims.

| Claim | Verdict | Models agree | MSCE conf |
|-------|---------|-------------|-----------|
| SIGReg is "provably optimal" anti-collapse | FALSE | 4/5 | 0.57 |
| 48× faster planning vs DINO-WM | UNFAIR | 4/5 | 0.57 |
| Only 1 hyperparameter needed | FALSE | 4/5 | 0.57 |
| JEPA collapse "solved" | FALSE | 4/4 | 0.95 |
| Latents show emergent physical understanding | FALSE | 5/5 | 0.57 |

The fourth claim received unusually high confidence (0.95) because the constraint set was unusually tight: the paper's own reported test environments (four low-dimensional control tasks), a concurrent benchmark study reporting world model brittleness, and the absence of comparison against non-PLDM collapse-prevention methods. When constraints are this specific and mutually reinforcing, agreement among models rises — the 0.95 reflects strong constraint consensus, not general certainty.

A note on methodology: the LeWorldModel paper was published in March 2026. Some constituent models may have knowledge cutoffs earlier than this date. MSCE provides the paper text as input; models reason about it within their context window rather than from pre-training memory. Different models have different long-context reasoning capabilities, which introduces variance not present in the benchmark setting.

We also audited Google DeepMind's Co-Scientist (Nature, May 2026). Full results are in the repository.

---

## 6. Discussion

### 6.1 Why It Works (When It Does)

The engine works because reasoning strategies differ enough that their failure modes don't fully correlate. A depth-first reasoner that gets locked into a wrong premise makes a different kind of error than a constraint-propagation reasoner that misses a constraint. When they arrive at the same answer through different paths, the answer has survived independent checks.

This is the same principle as N-modular redundancy in fault-tolerant hardware: independent units rarely fail in the same way at the same time. The independence here is approximate — all six models are transformer-based, trained on overlapping data — but the strategy prompts push them in different reasoning directions.

How independent are they really? We haven't measured it. Computing pairwise Cohen's κ across the six models on benchmark questions would quantify how often they agree beyond chance. If κ is high (>0.6), the strategies aren't producing meaningfully different reasoning, and the architecture's core premise weakens. We leave this measurement to future work; it belongs in the limitations.

### 6.2 Confidence That Means Something

The most dangerous output from an AI system is not a wrong answer. It's a wrong answer delivered with confidence. MSCE's confidence numbers behave differently from single-model confidence: they are lower on average and more accurate when they are high.

Why? Because MSCE confidence is not extracted from a probability distribution over tokens. It is computed from observable signals: do the models agree? Do the constraints check out? How much did they disagree before converging? These signals are external to any single model's internal state, which is why they correlate better with actual correctness.

The trade-off is coverage. MSCE refuses to answer (confidence < 0.3) more often than a single model. For verification use cases — paper auditing, claim checking, safety testing — refusal is preferable to false confidence. For applications where an answer is always required regardless of reliability, this is a drawback.

### 6.3 Limitations

**Unfair baseline.** MSCE uses six models plus a judge against single-model baselines. The 11.1pp gap should be read as an upper bound. Ablation experiments — simple voting among the six models, self-consistency@7 for GPT-5.5, and per-model accuracy vs. ensemble contribution — are needed to isolate MSCE's architectural contribution from the raw effect of using more compute. We didn't run these. Current results demonstrate that the pipeline works; they don't prove that every component of the pipeline earns its cost.

**Cognitive heterogeneity not measured.** The strategies are described as heterogeneous, but this is a design intent, not a verified property. Without pairwise agreement metrics, we can't rule out the possibility that the strategies produce correlated outputs and the main benefit comes from ensembling rather than from strategic diversity.

**Reproducibility.** GPT-5.5, Gemini 3.1 Pro, Grok 4.1, Kimi K2.5, and o4-mini are proprietary models accessible only through APIs that may change or be deprecated. Exact reproduction of these results will not be possible after API changes.

**L2 can discard correct minority answers.** If four of six models share a blindspot and answer confidently but wrongly, the two correct answers will be flagged as outliers and removed. This is a structural weakness for questions where the majority is wrong — a known class of failures in ensemble methods.

**Constraint selection bias.** In paper auditing, constraint choice affects outcomes. An auditor who wants a paper to "fail" can select constraints the paper didn't address; an auditor who wants it to "pass" can select only constraints the paper explicitly satisfies. MSCE does not solve this — it makes the constraint set explicit and visible, but the choice of constraints remains a human decision.

**No human baseline.** We compare MSCE to single models but not to human reviewers. For paper auditing, the relevant question is whether MSCE catches things human reviewers miss — and vice versa. This comparison would require a controlled study with real reviewers, which we haven't done.

### 6.4 Future Work

Four directions seem productive. First, ablation experiments to separate ensemble effects from architecture effects. Second, measuring pairwise model agreement to quantify actual cognitive diversity. Third, a controlled comparison against human reviewers on a shared paper-auditing task. Fourth, automated constraint extraction from cited prior work, which would reduce reliance on manually specified constraints and lower the risk of selection bias.

---

## 7. Conclusion

MSCE is a verification tool. It runs six models against each other, filters out answers that don't survive adversarial scrutiny, and checks what remains against explicit constraints. Across 386 questions, it got 91.7% right — 11.1 points above the best single model — while producing no answers that were both wrong and confident. It flagged its own uncertainty most of the time it was wrong.

The case studies show the approach generalizes. Hubble tension proposals look different when you demand they satisfy all major constraints simultaneously. Published paper claims look different when six independent reasoning paths check them against their own stated limitations.

The current version has real limitations: the baseline comparison overstates the architectural contribution, the cognitive heterogeneity is asserted rather than measured, and reproducibility depends on proprietary APIs. These are fixable with further experiments.

MSCE does one thing. It doesn't generate, doesn't create, doesn't discover. It checks. In a world where AI-generated claims are becoming abundant and cheap, checking is the bottleneck. MSCE is an attempt to widen it.

---

## References

[1] A. M. Turing. Computing machinery and intelligence. *Mind*, 59(236):433–460, 1950.

[2] D. Kahneman. *Thinking, Fast and Slow*. Farrar, Straus and Giroux, 2011.

[3] T. G. Dietterich. Ensemble methods in machine learning. In *Multiple Classifier Systems*, pp. 1–15, 2000.

[4] L. Breiman. Random forests. *Machine Learning*, 45(1):5–32, 2001.

[5] Y. Du et al. Improving factuality and reasoning in language models through multiagent debate. In *ICML*, 2024.

[6] C. Michael et al. Debate helps supervise unreliable experts. arXiv:2410.13011, 2024.

[7] L. Zheng et al. Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. In *NeurIPS*, 2023.

[8] A. N. Angelopoulos and S. Bates. Conformal prediction: A gentle introduction. *Foundations and Trends in Machine Learning*, 16(4):494–591, 2023.

[9] Y. LeCun et al. LeWorldModel: Stable end-to-end joint-embedding predictive architecture from pixels. arXiv:2603.19312, 2026.

[10] Sub-JEPA Authors. Sub-JEPA: Subspace joint embedding predictive architecture. arXiv:2605.09241, 2026.

[11] Planck Collaboration. Planck 2018 results. VI. Cosmological parameters. *Astronomy & Astrophysics*, 641:A6, 2020.

[12] A. G. Riess et al. A comprehensive measurement of the local value of the Hubble constant. *The Astrophysical Journal*, 934(1):57, 2022.
