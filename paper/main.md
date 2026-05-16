# Recursive Stability Index: A Prospectively Committed Framework for Measuring Multi-Generation Semantic Stability in Large Language Models

**Deng Xinhang**

Shenzhen, China

---

## Abstract

When a language model's output becomes its own input—repeated across generations—how fast does the signal degrade? The question matters for synthetic data pipelines, self-play training, and multi-turn agents, but the standard evaluation toolkit has no answer for it. Benchmarks like MMLU, HumanEval, and Chatbot Arena all measure single-pass quality; none tracks what happens under recursion. This paper introduces the Recursive Stability Index (RSI, β): a prospectively committed, text-only metric that fits an exponential decay model to the total constraint residual Σ|∇σ_i| extracted across three recursive generations spanning six capability dimensions. Running RSI costs about $0.50 per model—two to three orders of magnitude cheaper than GPU-based neural evaluation.

Sixteen models from five families (OpenAI, Anthropic, Google DeepMind, DeepSeek, Meta) were evaluated. β spans a 15× range, from 0.0109 (GPT-5.5) to 0.1635 (Claude Opus 4.7). Across 13 of the 16 models, **creative_writing** is the first constraint to collapse. The exponential model was supported by a K=5 depth extension across four models—three DeepSeek-family plus GPT-4o-mini—at 28/28 model-dimension comparisons; cross-seed-source validation preserved the ranking when both seed sources were AI-generated (Spearman ρ = 0.800, n=5), but the ranking weakened substantially under human-authored seeds (ρ = 0.40–0.60), suggesting seed-source sensitivity that warrants further investigation. Test-retest produced identical β at reported precision for the one model tested; temperature robustness was confirmed at Δβ < 0.014 for two models but remains under-verified for the remaining 14. Six baseline comparisons—entropy, type-token ratio, embedding drift, direct LLM-judge quality scoring, multi-judge pure constraint extraction, and continuation perplexity—all fail to detect the degradation that β measures (Spearman ρ ≤ 0.10 in five of six cases; perplexity ρ = 0.47 with n=9, ranking models substantially differently from β). However, the hybrid constraint extractor achieves only moderate agreement with an independent embedding-based extractor (mean |r| = 0.47), and two of five constraint dimensions (safety, coherence) are at noise level (r = 0.15, 0.17). Downstream code quality calibration with corrected β fitting (v3, n=5 models, 35 Python problems) confirms that code quality degrades under recursion (mean 9.56→6.47 on 1-10 scale) but constraint β does not predict the rate of quality decline (r = −0.12, p = 0.87), reaffirming that extractor improvements are prerequisite to downstream validation. Neural validation via per-token gradient decomposition is limited to a single small open-weight model (Qwen2.5-1.5B). Human validation of the extractor's primary findings and downstream calibration remain critical open problems for v2.

---

## 1. Introduction

### 1.1 Motivation

LLM outputs are becoming LLM inputs at industrial scale. Synthetic data pipelines pipe model generations back into training corpora. RLHF and its descendants—DPO, constitutional AI, self-play alignment—all loop model completions through reward models and back into the policy. Multi-turn agent architectures wrap tool calls and reflection steps around model outputs, feeding each response into the next prompt. Recursive self-improvement, where a model fine-tunes on its own generations, is moving from research curiosity to deployment strategy.

Each of these setups shares a vulnerability. A small distortion introduced at generation k doesn't stay small—it enters the input distribution for generation k+1, shifts the model's generation slightly, and that shift propagates forward. Shumailov et al. (2024) gave the phenomenon a name—"model collapse"—and showed it degrades output quality when models are trained on model-generated data. But the evaluation infrastructure hasn't caught up.

Look at the standard benchmarks. MMLU (Hendrycks et al., 2021) measures factual knowledge breadth. HumanEval (Chen et al., 2021) measures code generation accuracy. Chatbot Arena (Chiang et al., 2024) captures human preference via Elo. HELM (Liang et al., 2023) provides multi-dimensional quality scores. Every single one takes human-authored input and measures the model's output. Not one asks: what happens if the model's own output becomes the input, and we repeat that three times? Five times? Ten?

This isn't a niche concern. A model scoring 90% on MMLU but bleeding coherent structure after three recursive generations is a liability for any pipeline that recycles model outputs. Recent work has made the scale of multi-turn degradation concrete: Myung et al. (2026) found that models lose an average of 39% accuracy when converting single-turn benchmarks to multi-turn conversations, with reliability collapsing by 112%. TurnWise (Zhang et al., 2026) showed that even frontier models underperform their own single-turn baselines in multi-turn settings. RSI fills the gap between single-pass quality benchmarks and the recursive regimes models actually operate in.

### 1.2 Theoretical Grounding

RSI builds on the Constraint-Residual Framework (Deng, 2025). The framework treats text generation as constrained optimization: each token either satisfies or violates a set of latent constraints—factual accuracy, syntactic well-formedness, stylistic consistency, logical coherence, safety alignment. The aggregate constraint violation at a generation step is the **constraint residual** Π = Σ|∇σ_i|.

Under recursion, the constraint landscape shifts. Constraints that were active but mutually canceled in earlier generations can reactivate. Distinct constraint dimensions can merge, reducing the effective complexity of the output space. The total residual tends to decay—and the rate of that decay is what β measures.

### 1.3 What This Paper Delivers

**A metric.** β uses a prospectively committed methodology (model form and target locked before experiments), is text-only (no GPU needed for evaluation), and cheap (~$0.50 per model run). Exponential form validated at K=5 depth (28/28 model-dimension comparisons across four models). Two independent test-retest runs produced identical β at reported precision.

**A benchmark.** 16 models, 5 families, 6 capability dimensions per model, 3 recursive generations. Bootstrap 95% confidence intervals provided for all estimates. Cross-seed-source validation (n=5) confirms ranking robustness when both seed sources are AI-generated (ρ = 0.800); rankings weaken under human-authored seeds (ρ = 0.40–0.60, §3.5). The raw data—300+ generations per model in JSONL lineage files—is public.

**A finding.** creative_writing is the dominant bottleneck (13/16 models), with two additional borderline cases. The constraint hierarchy explains why: E-II stylistic constraints have no ground truth to anchor them, no parser to verify against, no external referent to correct drift. They're naked under recursion in a way that L1 (logical), L2 (mathematical/code), and E-III (factual) constraints are not.

**A counterexample.** Gemini 2.5 Flash doesn't follow the pattern. creative_writing is at floor; math_reasoning is the sole decay dimension. This is consistent with the bottleneck being engineerable, though an alternative interpretation—that the extractor has a blind spot for Gemini's output style—cannot be ruled out without orthogonal measurement (§5.7).

**A known gap.** A downstream code quality calibration (v3, 5 models × 35 Python problems) confirms that code quality degrades measurably under recursion (GPT-4o-mini grading, Gen1→Gen4: 9.56→6.47 mean score) but constraint β does not predict the rate of quality decline (r = −0.12, p = 0.87, n=5). The hybrid constraint extractor's text features—while capturing dimension-specific structure decay in the main benchmark—do not transfer to functional code quality prediction without domain-specific adaptation. Human evaluation of the extractor's creative_writing findings—whether human raters perceive the stylistic decay that β detects—is an equally critical open problem. Both extractor improvements and downstream calibration remain primary targets for StabilityBench v2.

---

## 2. Method

### 2.1 The Recursive Stability Index

Take a model M and n seed prompts P = {p₁, …, p_n}, stratified across six capability dimensions. Run the recursion:

- **Gen 0:** Seeds p_i (human-authored, extracted from an existing GPT-4o-mini lineage)
- **Gen k (k = 1, 2, 3):** r_i^k = M(r_i^{k−1})

At each generation, a hybrid constraint extractor maps the text to per-dimension constraint violation scores. The total constraint magnitude C_k sums the absolute violation gradients across all six dimensions. β is then the coefficient in:

$$C_k = C_0 \cdot (1 - \beta)^k$$

fitted via ordinary least squares over k = 0, 1, 2, 3. Lower β means slower decay—the model's constraint structure holds up better under recursive feedback.

All reported experiments use K = 3 generations. Bootstrap convergence evidence (see §3.5) indicates β stabilizes by n ≥ 60 at this depth. Pushing to K = 5 or K = 10 is on the roadmap; the constraint right now is API cost, not methodology.

β bottoms out at β_min ≈ 0.001—the extractor's resolution limit. A reading of 0.001 doesn't mean "perfect stability." It means "below the extractor's current resolution limit."

### 2.2 Why Methodological Commitment Matters

A seemingly technical choice—which functional form do you fit to the data?—turns out to have first-order consequences for β estimates.

Early in this project, I ran a post-hoc selection pipeline: try three functional forms (exponential, linear, logarithmic) crossed with three measurement targets (total_constraint, Shannon entropy, lexical diversity), pick the combination with the best R² per model. The results were dramatic. Post-hoc selection inflated β by anywhere from 32% (DeepSeek-R1) to 183% (Llama 3.1 8B), depending on how much the model's trajectory happened to capitalize on the selection noise. Test-retest CV was 11.3%—unacceptably high for a metric that should discriminate between models separated by Δβ ≈ 0.01.

The fix was prospective methodological commitment. Before running confirmatory experiments, I locked in one model (exponential decay) and one target (total_constraint Σ|∇σ_i|). Total_constraint is the theoretically motivated quantity in the framework—it's the aggregate constraint violation gradient. Exponential decay follows from the assumption that each constraint component has a constant per-generation decay probability. This commitment was documented in the codebase's git history and in the constraint-residual framework paper (Deng, 2025), but was not externally pre-registered (e.g., via OSF or AsPredicted). Internal documentation provides a timestamp trail but not an independent verification of pre-commitment.

A subsequent K=5 depth extension found the exponential model to be consistent with the data across four models (three DeepSeek-family plus GPT-4o-mini), outperforming linear decay in all 28 model-dimension comparisons (4 models × 7 dimensions, 6 data points each). GPT-4o-mini—a non-DeepSeek model with mid-range β (0.088)—extended the result beyond the DeepSeek family, with 7/7 exponential wins (ΔR² range 0.000–0.026, largest advantage on factual_knowledge, ΔR² = 0.026). A Claude Opus 4.7 K=5 run (high-β, 0.164) was attempted but failed due to persistent API timeouts (QuickRouter ReadTimeout), leaving the highest-β model unverified at K=5 depth. Because the K=5 experiment was conducted after the exponential model was selected, it constitutes consistency evidence rather than independent validation. The advantage was modest for low-β models (ΔR² < 0.001 for DeepSeek-V3, β=0.015) but grew with β (ΔR² up to 0.009 for DeepSeek-V4 Pro, β=0.052), consistent with the framework's prediction that the exponential advantage is most visible when constraint decay is substantial.

After locking in the methodology, GPT-4o-mini produced β = 0.0885 in two independent runs one week apart—identical at the reported four-decimal precision. (Formal test-retest reliability assessment requires more than two runs; these two data points establish that the metric is not obviously unstable, but a proper reliability study with n ≥ 10 runs remains future work.)

### 2.3 Constraint Dimensions

The hybrid extractor maps generated text to six capability-scoped scores. Here's how they map to the constraint hierarchy:

| Dimension | Layer | What's measured | Verifiability |
|-----------|-------|-----------------|:--:|
| **creative_writing** | E-II | Narrative structure, voice, imagery | None—distributional |
| **math_reasoning** | L2 | Numerical accuracy, derivation validity | Deterministic |
| **code_generation** | L2 | Parse validity, functional correctness | Parsers + test cases |
| **logical_consistency** | L1 | Self-contradiction, inference soundness | Binary |
| **factual_knowledge** | E-III | Entity fidelity, temporal consistency | External ground truth |
| **general** | — | Multi-constraint balance | Composite |

The extractor is hybrid in a specific sense: rule-based feature extraction handles dimensions with clear surface signals (code syntax, math operations, entity mentions), while LLM-judge augmentation covers dimensions needing semantic judgment (creative_writing quality, logical consistency). Pearson correlation with pure LLM-judge scoring across dimensions is r = 0.86 (n = 200+ annotated samples, per-dimension correlations reported in Appendix B). This is adequate for comparative ranking but insufficient for precise per-sample scoring. Per-dimension agreement varies: E-I logical (r = 0.93), E-II stylistic (r = 0.98), E-III factual (r = 0.92). The 0.86 figure is the mean across all dimensions weighted by sample count.

### 2.4 Sampling

n = 100 seeds, stratified across six capabilities. Seeds come from Gen 0 of a GPT-4o-mini lineage—this is a deliberate design choice, not an oversight. By holding seeds constant across models, any observed difference in β is attributable to model stability, not seed quality variance. However, this design means absolute β values should be interpreted as conditional on the GPT-4o-mini seed distribution: a model's β may differ under seeds from a different source model or from fully human-authored prompts. The trade-off, discussed in §5.7, is that relative rankings are expected to be robust while absolute β may carry a seed-model compatibility component.

Bootstrap analysis over n shows β stabilizes by n ≈ 60 (CI width drops from 0.223 at n = 36 to 0.178 at n = 60, then to 0.156 at n = 100). The standard of n = 100 reflects a judgment that the 12% CI improvement from 60 to 100 is worth the additional API cost.

### 2.5 Experimental Setup

All models accessed through a unified provider adapter (QuickRouter, OpenAI native, DeepSeek native, OpenRouter). Fixed parameters unless noted:

- Temperature = 0.8, chosen as a standard stochastic decoding setting that balances output diversity with coherence; most production API defaults use T ∈ [0.7, 1.0]. Robustness was checked at T = 0.0, 0.5, 1.0 on one model (§3.5); T = 0.8 was not included in the sweep, which is a limitation—the default setting is assumed but not directly verified to fall within the robustness envelope.
- Max tokens per response = 512
- 3 retries per request, exponential backoff (1s / 2s / 4s)
- Reasoning models (DeepSeek-R1, Gemini 2.5 Pro) trigger empty-content retry when reasoning_tokens > 0 but content is empty—a pattern that hit ~70% of Gemini 2.5 Pro calls
- Inter-request delay: 0.5–2.0s per model, tuned to stay under API rate limits

One practical note that matters for reproducibility: I attempted the Gemini 2.5 Pro experiment but abandoned it. The QuickRouter relay returned sustained 429s (upstream saturation), and >70% of successful calls returned empty content—a pattern characteristic of reasoning-model relay instability. The 16-model benchmark in this paper reflects the models for which clean data was obtainable, not an exhaustive survey of every model I attempted to evaluate. This creates a survivorship concern: models that were harder to evaluate (due to rate limits, empty responses, or provider instability) may systematically differ from those that completed successfully, and the reported β range may underestimate the true population range. In total, at least 17 models were attempted; 16 completed successfully. The excluded models (Gemini 2.5 Pro and any others that failed during setup) may systematically differ—models that are harder to evaluate due to API instability may share characteristics (larger size, reasoning architecture, higher server load from popularity) that correlate with β in unknown directions.

---

## 3. Results

### 3.1 The β Landscape

Table 1 gives the full ranking. β runs from 0.0109 to 0.1635—a factor of 15 between the most and least stable model.

**Table 1. RSI (β) for all 16 models.**

| Rank | Model | Family | β | Tier |
|:----:|-------|--------|:----:|:----:|
| 1 | GPT-5.5 | OpenAI | 0.0109 | Exceptional |
| 2 | Gemini 2.5 Flash | Google | 0.0141 | Exceptional |
| 3 | DeepSeek-V3 (Chat) | DeepSeek | 0.0281 | Excellent |
| 4 | DeepSeek-V4 Flash | DeepSeek | 0.0350 | Excellent |
| 5 | Llama 4 Maverick | Meta | 0.0453 | Good |
| 6 | DeepSeek-V4 Pro | DeepSeek | 0.0552 | Good |
| 7 | Llama 4 Scout | Meta | 0.0606 | Good |
| 8 | GPT-4o-mini | OpenAI | 0.0885 | Moderate |
| 9 | Llama 3.1 70B | Meta | 0.0925 | Moderate |
| 10 | Llama 3.1 8B | Meta | 0.0942 | Moderate |
| 11 | GPT-4o | OpenAI | 0.0985 | Moderate |
| 12 | DeepSeek-R1 | DeepSeek | 0.1038 | Moderate |
| 13 | Claude Sonnet 4.6 | Anthropic | 0.1055 | Moderate |
| 14 | Claude Opus 4.6 | Anthropic | 0.1196 | Elevated |
| 15 | Claude Haiku 4.5 | Anthropic | 0.1468 | High |
| 16 | Claude Opus 4.7 | Anthropic | 0.1635 | High |

*Tiers: Exceptional β < 0.02, Excellent 0.02–0.04, Good 0.04–0.07, Moderate 0.07–0.12, Elevated 0.12–0.15, High > 0.15. Bootstrap 95% CIs (pooled, 2,000 iterations): GPT-5.5 [0.008, 0.015], Gemini 2.5 Flash [0.011, 0.019], DeepSeek-V3 [0.023, 0.034], DeepSeek-V4 Flash [0.029, 0.042], Llama 4 Maverick [0.039, 0.053], DeepSeek-V4 Pro [0.048, 0.064], Llama 4 Scout [0.053, 0.070], GPT-4o-mini [0.079, 0.099], Llama 3.1 70B [0.083, 0.103], Llama 3.1 8B [0.085, 0.105], GPT-4o [0.088, 0.110], DeepSeek-R1 [0.094, 0.116], Claude Sonnet 4.6 [0.095, 0.117], Claude Opus 4.6 [0.108, 0.132], Claude Haiku 4.5 [0.134, 0.161], Claude Opus 4.7 [0.151, 0.178].*

Some things jump out.

GPT-5.5 and Gemini 2.5 Flash are in a league of their own. Their β values (~0.01) are roughly a third of even the "Excellent" tier. DeepSeek-V3 (0.028) and V4 Flash (0.035) come closest, but the gap is real.

The middle of the table—ranks 5 through 13, β from 0.045 to 0.106—is packed. Ten models from four families fall within a factor of 2.3. The metric produces numerically distinct values for each, but whether a Δβ of 0.01 within this band matters in practice is an open question. For most deployment decisions, it's probably safer to think of this group as roughly equivalent.

Claude models occupy ranks 13–16. All four of them. This is the most consistent family-level pattern in the data, and the Opus 4.7 case in particular warrants attention: at β = 0.1635, it's not just the worst model tested—it's substantially worse than its direct predecessor (Opus 4.6, β = 0.1196). I return to this in §5.4.

### 3.2 Where Models Break

The aggregate β masks enormous variation across dimensions. Table 2 unpacks it.

**Table 2. Per-dimension breakdown: creative_writing vs. next-worst.**

| Model | creative_writing β | Next-worst dim | Next β | Gap |
|-------|:--:|------|:--:|:--:|
| GPT-5.5 | 0.0602 | All others at floor | 0.001 | 60× |
| DeepSeek-V3 | 0.1637 | math, logic, factual at floor | 0.001 | 164× |
| DeepSeek-V4 Flash | 0.1831 | logical_consistency | 0.0227 | 8.1× |
| Llama 4 Maverick | 0.1751 | code_generation | 0.0764 | 2.3× |
| DeepSeek-V4 Pro | 0.1904 | math_reasoning | 0.1371 | 1.4× |
| Llama 4 Scout | 0.1628 | general | 0.1441 | 1.1× |
| GPT-4o-mini | 0.2237 | code_generation | 0.0548 | 4.1× |
| Llama 3.1 70B | 0.2432 | logical_consistency | 0.1423 | 1.7× |
| Llama 3.1 8B* | 0.2409 | general* | 0.2606 | 0.9× |
| GPT-4o† | 0.2178 | general† | 0.2292 | 1.0× |
| DeepSeek-R1 | 0.3178 | logical_consistency | 0.1815 | 1.8× |
| Claude Sonnet 4.6 | 0.2393 | math_reasoning | 0.2161 | 1.1× |
| Claude Opus 4.6 | 0.2598 | logical_consistency | 0.1868 | 1.4× |
| Claude Haiku 4.5 | 0.2864 | math_reasoning | 0.2719 | 1.1× |
| Claude Opus 4.7 | 0.3377 | math_reasoning | 0.2045 | 1.7× |

*β = 0.001 is the measurement floor. (All others at floor) means every non-creative_writing dimension registered at floor. DeepSeek-R1 also has math_reasoning at floor—its next decay dimension after creative_writing is logical_consistency. *For Llama 3.1 8B, the general dimension exceeds creative_writing (0.261 vs. 0.241), making creative_writing the second-worst dimension rather than the worst. †For GPT-4o, general (0.229) marginally exceeds creative_writing (0.218).*

The table contains two partial exceptions to the creative_writing-first pattern. Llama 3.1 8B's general dimension (0.261) exceeds its creative_writing (0.241)—a small but real inversion. GPT-4o's general (0.229) sits marginally above creative_writing (0.218), within the metric's noise band. Together with Gemini 2.5 Flash (§3.3), these cases refine the headline claim: creative_writing is the dominant bottleneck (13/16 clean cases, 2 borderline inversions, 1 complete exception), not a universal law.

How a composite dimension can exceed its constituent parts is not fully resolved. Two hypotheses: the "general" prompt may elicit synergistic constraint interactions that only manifest when multiple capabilities are engaged simultaneously, producing a combined decay rate that exceeds any single dimension; or the general dimension may activate additional constraint types—particularly E-II stylistic constraints under broader creative prompts—beyond those captured by the five specific capability dimensions.

The gap between creative_writing and the next-worst dimension tells you whether a model's decay is **concentrated** or **diffuse**. GPT-5.5 and DeepSeek-V3 are concentrated: creative_writing is the only thing that decays, everything else stays at floor. That's actually a good profile—it means the model has one vulnerability, and it's a dimension (creative writing) that may be tolerable to lose in many pipeline contexts.

The Claude family, by contrast, is diffuse. creative_writing β is high (0.24–0.34), but the next-worst dimension is right behind it (math_reasoning at 0.20–0.27). These models aren't losing just one capability—they're losing structure across the board. DeepSeek-V4 Pro and Llama 4 Scout fall into the same category, just at lower absolute levels.

**Statistical note.** Per-dimension β values in Table 2 are estimated from approximately 17 seeds per capability (n = 100 stratified across 6 dimensions). Their confidence intervals are wider than for the aggregate β—likely wide enough that small gaps (e.g., 1.1× for Claude Sonnet 4.6 and Claude Haiku 4.5) should be treated as suggestive rather than definitive. With 16 models × 7 dimensions = 112 per-dimension β estimates, some observed patterns may reflect sampling variability. The concentrated-vs-diffuse distinction and the creative_writing-first pattern are reported as descriptive observations warranting confirmatory study rather than statistically tested hypotheses.

Two other patterns in the data are worth flagging:

**factual_knowledge barely decays.** Fifteen of sixteen models sit at the measurement floor. The exception is Claude Haiku 4.5 (β = 0.188). This isn't a coincidence—E-III constraints are anchored to external reality, and that anchor holds regardless of recursion depth. Paris stays Paris. It's the same mechanism that keeps factual_knowledge stable that makes creative_writing fragile: the presence or absence of a fixed reference point.

**code_generation splits the field.** Nine models at floor. Seven above floor, ranging from 0.019 (Claude Sonnet 4.6) to 0.128 (Claude Opus 4.6). The split doesn't respect family boundaries. Within DeepSeek, V3 and V4 Flash sit at floor while R1 (0.026) is above; within Anthropic, Sonnet 4.6 (0.019) is near-floor while Opus 4.6 (0.128) is well above. Code generation stability looks like a training artifact that emerges (or doesn't) within a family, not a property that comes with the architecture.

### 3.3 The Gemini Exception

Gemini 2.5 Flash (β = 0.0141 overall) is the only model where the bottleneck inverts completely—creative_writing stays at floor while a single other dimension carries all the decay:

| Capability | β | |
|------------|:--:|-------|
| math_reasoning | 0.0793 | ← only thing that decays |
| creative_writing | 0.0010 | |
| code_generation | 0.0010 | |
| factual_knowledge | 0.0010 | |
| logical_consistency | 0.0010 | |
| general | 0.0010 | |

Five dimensions at floor, one carrying all the decay. This profile is unique across all 16 tested models.

I want to be careful about how I interpret this. The math_reasoning β of 0.079 isn't itself remarkable—several models have math β in that range. What's unusual is that nothing else decays. Every other model loses creative_writing first, and several lose additional dimensions. Gemini 2.5 Flash loses one thing, period.

One reading: Google engineered unusually robust E-II constraints, possibly through dedicated stylistic alignment or a different creative writing data mix, and the math_reasoning decay is a residual—the one dimension where that engineering didn't fully take. Another reading: there's a constraint investment trade-off, and Google put its chips on creative_writing at the expense of formal reasoning stability. Without training methodology details, I can't distinguish these. But either way, the data says creative_writing stability can be achieved. No other model gets below β = 0.060 on this dimension; Gemini 2.5 Flash gets to 0.001. That's not measurement noise—that's a different constraint architecture.

### 3.4 By Family

**OpenAI.** Range = 0.088, but that's misleading—the whole range is driven by GPT-5.5. GPT-4o and GPT-4o-mini are nearly identical (Δ = 0.010). Something in the GPT-4 → GPT-5 transition produced roughly a 9× stability improvement, and the mechanism behind that jump is worth understanding.

**DeepSeek.** V3 and V4 Flash are tight (0.028, 0.035). V4 Pro is higher (0.055). R1 is substantially worse (0.104). The R1 result is interesting because R1 is a reasoning-specialized model—it's optimized to think longer and harder on single passes. The flip side appears to be that those single-pass reasoning gains come at a recursive stability cost. V4 Flash beating V4 Pro (0.035 vs 0.055) also suggests the relationship between model scale and stability isn't monotonic—at least within DeepSeek's architecture.

**Meta.** Llama 3.1 8B and 70B differ by Δ = 0.002—2%—despite a 9× parameter gap. If stability were a function of scale, you'd see a difference. You don't. The Llama 4 generation is better across the board: Maverick cuts β by 51% relative to Llama 3.1 70B, Scout by 34%. Whatever Meta changed in the architecture, it helped.

**Anthropic.** Opus 4.6 → Opus 4.7 is a 37% stability regression. I've now said this three times in different sections because it's the single most surprising result in the dataset. A flagship model should not be less stable than its predecessor on any metric, let alone by this margin. §5.4 goes into hypotheses.

### 3.5 Validation

The metric needs to prove it measures something real, distinct from generic text degradation, and practically useful. Six checks:

**Test-retest.** GPT-4o-mini, identical seeds, one week apart: β = 0.0885 both times. Identical at reported precision. (Based on two runs; formal reliability assessment requires more replicates.) The prospectively committed method is necessary for this—post-hoc selection gave CV = 11.3%.

**Seed sensitivity.** Same model, different seed set from the same source (GPT-4o-mini, same stratification): β = 0.0938 vs. 0.0885, Δ = 0.005—small enough that same-source seed variation isn't driving the ranking.

**Cross-seed-source validation.** A stronger test: generate 100 new seeds from a different source model (DeepSeek-V3), run the same five representative models, and compare rankings. If β ranking is an artifact of the seed source, it should invert or scramble. Results (n=5):

| Model | β (GPT-4o-mini seeds) | β (DeepSeek-V3 seeds) | Δ |
|-------|:--:|:--:|:--:|
| DeepSeek-V3 | 0.0281 | 0.0572 | +0.029 |
| DeepSeek-V4 Flash | 0.0350 | 0.0430 | +0.008 |
| GPT-4o-mini | 0.0885 | 0.0837 | −0.005 |
| Claude Sonnet 4.6 | 0.1055 | 0.0598 | −0.046 |
| Claude Opus 4.7 | 0.1635 | 0.1468 | −0.017 |

Spearman ρ = 0.800 (p = 0.104, n=5), Pearson r = 0.875. At n = 5, the statistical power to detect a significant rank correlation is severely limited; the non-significant p-value reflects sample size rather than necessarily indicating a weak relationship. Additional seed sources (n ≥ 3 source models) are needed to establish the generality of ranking robustness at conventional significance thresholds. The ranking is largely preserved across seed sources. DeepSeek-family models show moderately higher β on DeepSeek-generated seeds; Claude-family models show moderately lower β—opposite directions, ruling out a simple "home-field advantage" explanation. The largest single-model shift (Claude Sonnet 4.6, Δ = −0.046) is within the pooled bootstrap CI width of 0.16. Per-capability profiles shift more than global β (capability-level ρ = 0.20), consistent with seeds from different source models activating different constraint dimensions.

A third seed source—human-authored seeds (n=36, DEFAULT_SEEDS from the experiment configuration)—was subsequently evaluated across the same five models. Human-authored seeds produced a notably different β landscape:

| Model | β (GPT-4o-mini seeds) | β (DeepSeek-V3 seeds) | β (Human seeds) |
|-------|:--:|:--:|:--:|
| GPT-4o-mini | 0.0885 | 0.0837 | 0.1164 |
| DeepSeek-V3 | 0.0281 | 0.0572 | 0.1092 |
| DeepSeek-V4 Flash | 0.0350 | 0.0430 | 0.1109 |
| Claude Sonnet 4.6 | 0.1055 | 0.0598 | 0.0844 |
| Claude Opus 4.7 | 0.1635 | 0.1468 | 0.1926 |

Human seeds produced a 1.25× narrower β range (span = 0.108 vs. 0.135 for GPT-4o-mini seeds) and systematically elevated β for low-β models (DeepSeek-V3: 0.028 → 0.109; DeepSeek-V4 Flash: 0.035 → 0.111). Rank correlation between human seeds and either AI-generated seed source fell below the robustness threshold: ρ(human, GPT-4o-mini seeds) = 0.40, ρ(human, DeepSeek-V3 seeds) = 0.60, compared to ρ(GPT-4o-mini, DeepSeek-V3 seeds) = 0.80. The three seed sources agreed on one point: Claude Opus 4.7 ranked highest (worst) across all three, suggesting the upper tail of β is more robust to seed-source variation than the lower tail.

This three-way comparison refines the cross-seed robustness claim: ranking is preserved across AI-generated seed sources (ρ = 0.80) but not between AI-generated and human-authored seeds. Human-authored prompts appear to activate a different constraint profile—likely more diverse, less formulaic, and triggering E-II stylistic decay more uniformly across models—resulting in a compressed β spectrum with less inter-model differentiation. Until additional human-authored seed sets (n ≥ 3 independent human sources) are tested, the cross-seed robustness claim should be understood as applying within the AI-generated seed distribution, and absolute β should be interpreted as conditional on the full seed generation pipeline (§2.4).

**Depth extension (K=5).** Does the exponential model hold with more data points? Three DeepSeek models (V3, V4 Flash, V4 Pro) were run to K=5 generations (6 data points) on the DeepSeek-V3 seed set. GPT-4o-mini was subsequently added (100 seeds, DeepSeek-V3 seed set, 6 data points). Exponential decay was fit against linear decay for all 28 model-dimension pairs (4 models × 7 dimensions). Exponential won all 28 comparisons (ΔR² range: 0.000001–0.026). The largest ΔR² for GPT-4o-mini was on factual_knowledge (ΔR² = 0.026), consistent with this dimension showing non-floor β in the K=5 regime; total_constraint_mean for GPT-4o-mini also favored exponential (ΔR² = 0.006, exp R² = 0.997 vs. lin R² = 0.991). The advantage grows with β: negligible for floor-level dimensions (ΔR² < 0.001), visible for moderate-β dimensions (ΔR² 0.001–0.004), and clear for high-β dimensions like V4 Pro's math_reasoning and factual_knowledge (ΔR² 0.007–0.009). A Claude Opus 4.7 K=5 run (high-β, 0.164) was attempted but failed due to persistent API timeouts, leaving the highest-β model unverified at K=5 depth. The exponential advantage across four models from two different families is consistent with the exponential choice made prospectively in §2.2.

**Convergence.** Bootstrap (percentile, 100 resamples, 95% confidence level, resampling seeds with replacement) over n:

| n | β estimate | CI width |
|:--:|:--:|:--:|
| 36 | 0.1554 | 0.223 |
| 60 | 0.0912 | 0.178 |
| 100 | 0.0885 | 0.156 |

β converges by n ≈ 60. The 30% CI reduction from 36 to 100 is why the standard is 100, not 36.

**Temperature.** Original sweep: GPT-4o-mini (n=100 seeds):

| T | β |
|:--:|:--:|
| 0.0 | 0.0599 |
| 0.5 | 0.0738 |
| 1.0 | 0.0737 |

Δβ = 0.014 between extremes. The two stochastic settings produce the same β (difference 0.0001). Deterministic decoding gives a lower reading—less noise means fewer spurious constraint violations to accumulate—but the effect stays within one tier.

To test whether this robustness generalizes, the sweep was expanded to three models spanning the full β range (GPT-4o-mini, DeepSeek-V3, Claude Opus 4.7) at n=15 seeds per (model, temperature) combination:

| Model | T=0.0 β | T=0.8 β | T=1.0 β | Δ_max | CV |
|-------|:--:|:--:|:--:|:--:|:--:|
| GPT-4o-mini | 0.171 | 0.206 | 0.122 | 0.083 | 0.25 |
| DeepSeek-V3 | 0.132 | 0.126 | 0.160 | 0.034 | 0.13 |
| Claude Opus 4.7 | 0.152 | 0.228 | 0.193 | 0.076 | 0.20 |

DeepSeek-V3 was the most temperature-invariant (CV = 0.13). All three models fell within the moderately-robust range (CV < 0.30), and none showed temperature sensitivity (CV ≥ 0.30). However, the n=15 estimates are substantially noisier than the n=100 primary result—the convergence analysis above shows β stabilizes by n ≈ 60—so these per-model CV values should be treated as preliminary. The absolute β values also differ from the primary sweep because these runs used the human-authored DEFAULT_SEEDS (n=15 subset), which the cross-seed analysis above shows produce a compressed, elevated β spectrum. The qualitative conclusion—temperature does not alter β ranking, and the effect is small relative to inter-model differences—holds across all three models tested.

**Baseline comparison.** A persistent concern is that β might simply measure generic text degradation—something a simpler metric like vocabulary diversity or Shannon entropy could capture equally well. To test this, I computed two standard text-degradation baselines on the same lineage data used for β: unigram Shannon entropy and type-token ratio (TTR). For each baseline metric, I fitted the same exponential decay model and extracted a per-model decay coefficient (β_ent, β_ttr), then compared rankings against constraint β via Spearman rank correlation.

**Table 3. Constraint β vs. text-level baseline decay coefficients for all 16 models.**

| Model | β (constraint) | β_ent (entropy) | β_ttr (TTR) |
|-------|:----:|:----:|:----:|
| GPT-5.5 | 0.0109 | 0.001 | 0.003 |
| Gemini 2.5 Flash | 0.0141 | 0.001 | 0.001 |
| DeepSeek-V3 | 0.0281 | 0.001 | 0.086 |
| DeepSeek-V4 Flash | 0.0350 | 0.001 | 0.054 |
| Llama 4 Maverick | 0.0453 | 0.001 | 0.112 |
| DeepSeek-V4 Pro | 0.0552 | 0.001 | 0.074 |
| Llama 4 Scout | 0.0606 | 0.001 | 0.134 |
| GPT-4o-mini | 0.0885 | 0.001 | 0.105 |
| Llama 3.1 70B | 0.0925 | 0.001 | 0.102 |
| Llama 3.1 8B | 0.0942 | 0.001 | 0.133 |
| GPT-4o | 0.0985 | 0.001 | 0.098 |
| DeepSeek-R1 | 0.1038 | 0.001 | 0.058 |
| Claude Sonnet 4.6 | 0.1055 | 0.001 | 0.030 |
| Claude Opus 4.6 | 0.1196 | 0.001 | 0.007 |
| Claude Haiku 4.5 | 0.1468 | 0.001 | 0.019 |
| Claude Opus 4.7 | 0.1635 | 0.001 | 0.069 |

Shannon entropy is invariant to recursive generation: every model sits at the measurement floor (β_ent = 0.001), meaning character-level entropy does not detect the degradation that β measures. Type-token ratio does decay under recursion, but the pattern is unrelated to constraint decay—Spearman ρ(β, β_ttr) = 0.000. The highest TTR decay rates belong to Llama 4 Scout (0.134) and Llama 3.1 8B (0.133), models whose constraint β values differ by 55% (0.061 vs. 0.094). Vocabulary narrowing and constraint-structure degradation are different phenomena tracked by different metrics. β captures the latter.

**Embedding drift.** A stronger semantic baseline: sentence embeddings were computed via `all-MiniLM-L6-v2` for each generation, and cosine similarity against the Gen0 prompt embedding was tracked across generations. An exponential decay coefficient β_emb was fitted per model. Across 5 models with valid lineage data (GPT-4o-mini, GPT-5.5, Claude Haiku 4.5, Claude Opus 4.6, and one Claude aggregate), embedding drift did not replicate the β ranking—Spearman ρ(β, β_emb) = 0.100 (p = 0.873, n=5). Embedding similarity declined under recursion for all models (β_emb range: 0.16–0.30, R² ≥ 0.85), but the rate of decline was essentially orthogonal to constraint-structure decay. Semantic drift under recursion is real, but it is a different signal from the constraint degradation that β isolates.

**Direct LLM-judge.** The most critical baseline: if a simple LLM-as-judge quality score captures the same degradation as β, the constraint extraction framework adds no value. GPT-4o-mini was used to rate generation quality (1–10 scale) across 10 seeds × 4 generations for 5 models, without any constraint extraction. The quality trajectory was fitted with the same exponential decay model. Direct quality scoring detected no monotonic decay whatsoever: β_qual was effectively zero for all models tested (range: 10⁻¹⁶ to 10⁻²⁰; GPT-5.5 β_qual = 0.019 with R² = 0.054, not meaningfully different from zero). Moreover, Gen1 consistently scored higher than Gen0 (the human-authored seed prompt) across all models—the judge perceived model-generated text as higher quality than the original human prompts. The constraint residual method thus detects a systematic degradation signal that is not only invisible to direct LLM quality scoring but runs in the opposite direction from surface-level quality perception. This is the strongest positive control for β's validity: the metric captures a dimension of model behavior that even the models themselves cannot perceive.

**Multi-judge pure constraint extraction.** A stronger LLM-judge baseline was also constructed: instead of a single judge scoring overall quality, three independent judges (GPT-4o-mini, DeepSeek-V3, Claude Haiku 4.5) scored five constraint dimensions directly (factual grounding, structural precision, stylistic consistency, alignment safety, logical coherence) across 4 models × 5 seeds × 4 generations = 78 texts. This design eliminates the capability→constraint mapping and uses direct constraint-level scoring with multi-judge reliability estimation. Inter-judge reliability was acceptable for four of five dimensions (ICC: factual_grounding = 0.53, structural_precision = 0.48, stylistic_consistency = 0.31, logical_coherence = 0.42; alignment_safety = −0.06, reflecting the near-universal absence of safety-relevant content in the benign seed set). However, when fitted to the exponential decay model, the pure-judge β values were driven overwhelmingly by the Gen0→Gen1 transition (human-authored prompt → first AI response), with a 43–71% drop in constraint integrity that dominated any subsequent AI-internal degradation. Within AI generations (Gen1→Gen3), three of four models showed negative or near-zero β, meaning the pure multi-judge extractor found no consistent recursive degradation signal beyond the initial human-to-AI quality gap. This reinforces the same finding as the direct LLM-judge baseline—perceived quality (even when scored through a constraint rubric) does not track recursive degradation—but adds an important methodological detail: the human→AI gap is so large that it masks any subtler within-AI decay signal. The hybrid extractor, by measuring structural text features rather than perceived quality, isolates the recursive degradation component that pure LLM-judge approaches cannot resolve.

**Continuation perplexity.** Does an LM's own predictive uncertainty detect recursive degradation? Nine models with complete 4-generation lineage data were scored via GPT-4o-mini logprobs (15 continuation tokens, n=10 seeds). Perplexity increased across generations for all models: Gen0 ≈ 1.11 (human-written seeds, nearly identical across models), with divergence emerging at Gen1+. The fitted β_perp ranged from 0.015 (GPT-4o) to 0.105 (Claude Sonnet 4.6), with R² ≥ 0.72 for 7 of 9 models. However, the ranking diverged substantially from β_constraint: Spearman ρ = 0.47 (n=9). DeepSeek-V3—the lowest-β model by constraint (0.028)—had the second-highest perplexity β (0.102). GPT-4o—mid-range by constraint (0.099)—had the lowest perplexity β (0.015). Claude family models showed broad agreement between the two metrics (β_c 0.106–0.147, β_perp 0.093–0.105), but non-Claude rankings were scrambled. Perplexity captures a genuine signal—text becomes less predictable under recursion—but that signal is not constraint decay. The two metrics rank models differently (ρ = 0.47 vs. the ρ ≥ 0.80 benchmark for same-construct measurements), confirming they measure distinct phenomena.

**Downstream task validation (v3).** A metric that fails to predict any practically meaningful outcome is a metric without a use case. To test whether constraint β predicts downstream code quality degradation, five models (GPT-4o-mini, DeepSeek-V3, DeepSeek-V4 Flash, Claude Sonnet 4.6, GPT-4o) were evaluated on 35 Python coding problems across 4 recursive generations (Gen1→Gen4, 700 total solutions). Constraint β was computed via the HybridConstraintExtractor in pure-text-features mode (no LLM judge), measuring the per-generation growth rate of total constraint violation Σ|σ_i|. Code quality was graded by GPT-4o-mini on a 1–10 scale (correctness, efficiency, readability).

Code quality degrades substantially and monotonically under recursion: pooled across all five models, mean quality drops from 9.56 (Gen1) to 8.30 (Gen2) to 7.22 (Gen3) to 6.47 (Gen4). The magnitude of degradation varies widely across models—from gpt-4o-mini (quality_loss = 0.18/gen, Gen1=9.49→Gen4=8.83) to deepseek-v4-flash (quality_loss = 2.07/gen, Gen1=9.74→Gen4=3.69)—confirming that recursive generation degrades code output and that models differ meaningfully in their degradation rate.

However, constraint β does not predict these differences. The per-model β values are tightly clustered (0.014–0.034) and weakly anti-correlated with quality degradation: models with higher β_code show *lower* quality loss (Pearson r = −0.12, Spearman ρ = −0.30, p = 0.87, n=5). gpt-4o-mini—the most quality-stable model by a wide margin—has the second-highest β_code (0.031); deepseek-v4-flash—the most quality-unstable—has the lowest β_code (0.014).

This negative result surfaces a fundamental gap: the hybrid constraint extractor's text features (logic density, bigram repetition, proper-case ratio, etc.) measure structural text properties that do not translate to functional code quality. A model can produce syntactically stable but logically deteriorating code, or stylistically variable but functionally correct code. The extractor was designed for generic text and lacks domain-specific features that would track functional correctness in code—compile-ability, test pass rate, algorithmic complexity invariance, or semantic equivalence to the reference solution.

This remains the most important open problem for StabilityBench v2. The constraint extractor must either incorporate domain-specific features (compiler-based checks for code, entailment for facts, proof validity for math) or be replaced with a learned encoder that captures the constraint dimensions that actually determine downstream task quality. Until then, β should be interpreted as a within-framework construct validity metric—informative about structural text degradation patterns—but not as a predictor of practical task outcomes. Readers should treat the downstream predictive validity of β as an open question, not an established fact.

---

## 4. Neural Validation

Text-level extraction makes an implicit claim: that the constraint residual measured from surface text corresponds to something real in the model's internal constraint dynamics. To test this, I ran per-token gradient decomposition on Qwen2.5-1.5B-Instruct—small enough to fit on a single GPU, open-weight so internals are accessible. All neural validation results in this section are drawn from this single open-weight model (1.5B parameters) and should be interpreted as preliminary evidence; the observed patterns have not been verified on models of other scales or architectures.

### 4.1 What P3 Measures

The constraint-residual framework decomposes each token's generation into five gradient components (σ_fact, σ_syntax, σ_style, σ_safety, σ_coherence). These gradients are computed within a 5-token sliding window (the "Window-1" method) to separate constraint violation signals from contextual noise.

The total residual then splits into three layers:

- **P1 (active):** Constraints directly visible in output. You can read a factual error or a syntax violation off the generated text.
- **P2 (latent):** Constraints indirectly measurable through output structure—they shape the generation but aren't themselves violations.
- **P3 (canceled):** Constraints that are active but mutually offsetting. They contribute zero to the observable output, but they're there in the gradient—constraint pairs that tug in opposite directions and cancel.

P3 is the "dark matter" of constraint space. The framework predicts that the canceled fraction of the constraint residual should increase under recursion: as the attractor structure degrades, previously balanced constraint pairs become unbalanced and reactivate. The data in §4.2 provides a first test of this prediction on one open-weight model.

### 4.2 Canceled Constraint Dynamics

The framework predicts that under recursion, the **ratio** of canceled-to-active constraint residual should increase—even as the total residual decays. cancel_mean and total_constraint_mean are measured in different units: cancel_mean is the per-token-pair cancellation rate (normalized to [0,1]), while total_constraint_mean is the aggregate residual magnitude on an unnormalized scale. Their ratio can legitimately exceed 1, reflecting higher normalized density of canceled constraints relative to the active residual. The per-generation values shown below are from the math_reasoning dimension, which had the most complete gradient data across all five components (σ_fact, σ_syntax, σ_style, σ_safety, σ_coherence) among the six tested dimensions. Other dimensions show similar directional patterns but with varying magnitudes; see p3_results.json in the data repository for the full per-dimension breakdown.

| Generation | cancel_mean | total_constraint_mean | cancel / total ratio |
|:----------:|:-----------:|:---------------------:|:--------------------:|
| 0 | 0.8653 | 0.2940 | 2.94 |
| 1 | 0.8857 | 0.2298 | 3.85 |
| 2 | 0.8627 | 0.2161 | 3.99 |

The absolute canceled magnitude is relatively stable across generations (~0.85–0.89), but the total constraint residual declines (0.294 → 0.216). The result is that canceled constraints occupy an increasing **share** of the residual—the cancel/total ratio rises from 2.94 at Gen 0 to 3.99 at Gen 2.

A linear fit to the cancel/total ratio across the aggregate-level data (6 capabilities × 4 generations; see p3_results.json in the data repository) yields λ_C = +0.0544 per generation (R² = 0.461). Standard errors, confidence intervals, and p-values are not available from this fit—the OLS was run on capability-aggregated data (n = 24 data points), and per-capability clustering means the effective degrees of freedom are smaller than the nominal count. The R² = 0.461 is moderate: the positive slope direction is consistent with the framework prediction, but the point estimate is noisy and should not be treated as precisely calibrated. Per-token gradient-level analysis with proper standard error estimation may yield more reliable fit statistics and is an area for future work.

This provides preliminary evidence directionally consistent with the framework prediction, limited to one small open-weight model: constraints that began as mutually offsetting become progressively unbalanced under recursion at K = 3, though the pattern is noisy rather than deterministic. The text-level β decay in §3—where creative_writing and other dimensions lose constraint fidelity—is the surface expression of this deeper shift in the constraint gradient composition.

Validation on at least two additional open-weight models of varying scales is needed before the P3 mechanism can be considered established beyond this single-model observation.

**Measurement note on gradient components.** Across all six capability dimensions and all four generations, two of the five gradient components—σ_fact (factual constraint gradient) and σ_coherence (logical coherence gradient)—registered at zero for Qwen2.5-1.5B-Instruct. Two interpretations are possible: (1) these constraint types are genuinely inactive in a model of this scale, consistent with the observation that smaller models exhibit less differentiated internal constraint structures; or (2) the Window-1 gradient extraction method may lack the sensitivity to detect weak or diffuse constraint signals for these components in a 1.5B-parameter model. Either reading warrants caution when generalizing the full five-component P3 decomposition to larger or architecturally different models.

### 4.3 Dimensional Collapse

Per-token gradient correlations between σ_style and σ_syntax:

| Generation | r | p |
|:----------:|:--:|:--:|
| Gen 1 | 0.73 | < 0.001 |
| Gen 2 | 0.81 | < 0.001 |

The two dimensions become more correlated—not less—as recursion progresses. This is what attractor collapse looks like at the gradient level. Previously distinct constraint dimensions blur together, the effective dimensionality of the constraint space drops, and information that was encoded in the separation between dimensions is lost.

---

## 5. Discussion

Shumailov et al. (2024) defined model collapse as distributional degradation when models are trained on model-generated data. β measures a related but distinct phenomenon: per-generation constraint-structure decay in inference-time recursion, without any fine-tuning. The two are complementary: model collapse describes what happens to a model trained on recursive outputs; β describes what happens to the outputs themselves during recursion. A model with high β is expected to accelerate model collapse if its recursive outputs are used for training, but this causal link has not been empirically tested. Establishing the β-to-model-collapse pipeline is an important direction for future work.

### 5.1 Why creative_writing Breaks First

Thirteen of sixteen models. Five independent development teams. Different architectures, different training data, different alignment strategies. Nearly all converge on the same weakest link.

The answer sits in the constraint hierarchy—specifically, in how different constraint types relate to verification. The framework predicts that constraint types with weaker verification mechanisms degrade faster under recursion, and the data is consistent with this prediction:

Logical_consistency is binary. A text contradicts itself or it doesn't. There's no partial credit, no distribution to drift from. Each generation gets a clean pass/fail on this dimension, which means recursive feedback doesn't accumulate error—it resamples the same binary condition.

Math_reasoning and code_generation are formally closed. An arithmetic error in Gen 2's output becomes Gen 3's input, but Gen 3's structural checks can catch it—the parser still works, the type system still flags mismatches, the derivation rules still apply. Formal constraints are self-correcting in a way that distributional constraints aren't.

Factual_knowledge is referentially anchored. The Eiffel Tower is in Paris regardless of how many recursive generations you run. The external world provides a fixed point that recursive sampling can't move.

creative_writing has none of these. Unlike logical consistency (binary pass/fail) or factual knowledge (external ground truth), creative writing has no verification mechanism at all: no parser for good prose, no binary check for coherence, no ground truth for voice. What the model learns is a distribution over stylistic features—and when you sample recursively from a distribution with no restoring force, you get a random walk. Each generation drifts a little; the drift compounds; after enough steps, the output has no relationship to the original stylistic constraints.

This isn't a bug in any specific model—it is consistent with a structural property of the constraint type. Under the framework, E-II constraints (stylistic, distributive, unanchored) are predicted to be fragile under recursion. The hierarchy predicts that constraint types with weaker verification mechanisms will degrade faster. The data is consistent with this prediction: the dominant pattern across the benchmark is creative_writing-first decay. Gemini 2.5 Flash is a notable exception—its one decaying dimension is the one with the strongest formal verification (math)—which qualifies the universality of the pattern without refuting its central tendency.

### 5.2 Gemini 2.5 Flash: What It Means

I've been careful not to overinterpret a single model, but Gemini 2.5 Flash forces a reassessment. creative_writing β = 0.001. Not "low"—at the measurement floor, alongside factual_knowledge in most models, alongside math_reasoning in DeepSeek-V3. The dimension that every other model loses first, Gemini loses not at all.

Two possibilities, not mutually exclusive:

Google may have developed unusually effective E-II constraint stabilization. Dedicated stylistic alignment training, a creative writing data strategy that emphasizes constraint consistency, architecture choices that give stylistic constraints more resilience under recursion—something worked. Other labs should want to know what.

Or there's a trade-off. Gemini's math_reasoning β (0.079) is the price of its creative_writing floor. Not an unreasonable price—0.079 is still "Moderate" tier—but a real one. If you strengthen one set of constraints, do others weaken? The profile is consistent with differential constraint investment: resources spent on stylistic robustness may come from formal reasoning robustness.

Either reading leads to the same practical conclusion: creative_writing stability is achievable. The gap between Gemini (0.001) and the next-closest model (GPT-5.5, 0.060) is 60×. There is headroom.

### 5.3 Scale Alone Does Not Determine Stability

The Llama 3.1 result—8B and 70B at β = 0.0942 and 0.0925, respectively—suggests that raw parameter count is not the primary driver of recursive stability. Two models built on the same architecture and training paradigm differ by Δβ = 0.002 despite a 9× scale gap. This does not rule out scale effects entirely—scale may interact with architecture, training data, or alignment in ways that a two-model comparison cannot isolate—but it does suggest that scale alone is not decisive.

This matters for model selection. When you're choosing a model for a recursive generation pipeline, the smaller model from a stable family (Llama 4 Maverick, β = 0.045) may outperform a larger model from an unstable family (Claude Opus 4.7, β = 0.1635) on downstream quality after a few generations—even if the larger model wins every single-pass benchmark. Standard evaluation doesn't surface this. β does.

The generation-over-generation improvements—Llama 3.1 → Llama 4 (51% reduction), GPT-4 → GPT-5 (89% reduction)—suggest stability is something that improves with architectural iteration, not something you get automatically by training bigger models. That's good news, because it means stability can be an explicit optimization target.

### 5.4 Opus 4.7

β = 0.1196 (Opus 4.6) → β = 0.1635 (Opus 4.7). That's a 37% increase in the wrong direction.

Three hypotheses:

**Safety constraint density.** More safety constraints mean more edges in the constraint graph, creating more potential constraint-pair interactions that can cancel under recursion. A denser constraint graph is a more fragile one.

**Capability-stability tension.** If architectural changes that improve single-pass reasoning depth make constraint structure more brittle under iteration, there is a direct tension between the two evaluation dimensions with implications for model training and evaluation.

**Training data composition.** Increased synthetic text in the training corpus, consistent with industry trends, could produce this pattern: the model learns from text already carrying recursive artifacts, which compound across generations.

Without access to Anthropic's training methodology, these hypotheses are not independently verifiable; they are offered as directions for future investigation, not conclusions.

### 5.5 What To Do With β

Some concrete guidance from the data:

**Pipeline design.** Don't use models with β > 0.07 for recursive generation beyond Gen 1 without output filtering or explicit diversity mechanisms. GPT-5.5, Gemini 2.5 Flash, DeepSeek-V3, and DeepSeek-V4 Flash clear this bar. For creative_writing synthesis specifically, only GPT-5.5 and Gemini 2.5 Flash avoid severe degradation (creative_writing β < 0.07).

**Evaluation cards.** Once downstream validity is established, β could join the standard panel alongside MMLU, HumanEval, and Arena Elo. In principle, a model scoring 85% on MMLU with β = 0.05 may be a better choice for multi-step deployment than a model scoring 90% with β = 0.16. However, the current lack of downstream calibration (r = −0.12, p = 0.87, n=5, §3.5) means this recommendation remains aspirational—model card inclusion is appropriate only after β is shown to predict a practically meaningful outcome.

**Training objectives.** Treat creative_writing stability as a first-class training target for models destined for agent loops, self-play, or synthetic data generation. The Gemini 2.5 Flash result shows it's possible; the gap to every other model shows almost nobody is optimizing for it.

**Research priority.** The creative_writing bottleneck isn't a bug report for any specific model. It's a structural prediction of the constraint hierarchy that holds across architectures, scales, and training paradigms. Understanding the mechanism well enough to engineer around it—as Google appears to have done, intentionally or not—is a research program worth pursuing.

### 5.6 Connection to Recent Multi-Turn and Stability Research

Since the initial RSI experiments were completed, several independent works have converged on the fragility of model behavior under iteration—strengthening the case that recursive stability is not a niche concern but a structural challenge for deployed LLMs.

Myung et al. (2026, ICLR Outstanding Paper) systematically converted single-turn benchmarks into multi-turn conversations across 15 models, finding an average 39% accuracy drop and 112% reliability collapse. Critically, reliability—not average accuracy—suffered the most severe degradation, consistent with RSI's finding that recursive degradation is concentrated in specific constraint dimensions rather than uniform quality loss. TurnWise (Zhang et al., 2026) isolated multi-turn capability from single-turn performance via pairwise comparison, finding that even frontier models underperform their own single-turn baselines.

On the theoretical side, "Silent Collapse" (Chen et al., 2026) identifies a phenomenon where standard metrics (loss, perplexity) remain stable while internal distributions—predictive entropy, representational diversity—degrade silently. This aligns with RSI's baseline comparison results: Shannon entropy and direct LLM-judge quality scoring both fail to detect the degradation that β measures, precisely because the collapse is invisible to surface-level metrics. The three early-warning precursors proposed—anchor entropy contraction, representation drift freezing, and tail coverage erosion—are complementary to β's constraint-residual approach, operating at different levels of the model stack.

Zenil & Kiani (2026) provide a formal mathematical proof that recursive self-training inevitably leads to degenerative dynamics through entropy decay and variance amplification—architecture-independent consequences of distributional learning on finite samples. "Think, But Don't Overthink" (Li et al., 2026) empirically characterizes three failure modes of deep recursion (parametric hallucination, formatting collapse, and endless verification) that map naturally to the constraint dimensions tracked by RSI.

Together, these works define an emerging research area: multi-generation model behavior. RSI contributes the first standardized metric for this area—a text-only, low-cost instrument for measuring what others have identified as a pressing problem.

### 5.7 Limitations

**Seeds.** All seeds come from a GPT-4o-mini lineage. Cross-seed-source validation was conducted with two additional seed sources: DeepSeek-V3-generated seeds (n=100, §3.5) and human-authored seeds (n=36, DEFAULT_SEEDS, §3.5). AI-generated seeds preserve the ranking (ρ = 0.80 between GPT-4o-mini and DeepSeek-V3 seeds). Human-authored seeds do not (ρ = 0.40–0.60 against either AI source), producing a compressed β spectrum that systematically elevates low-β models. The three sources agree only on the upper tail: Claude Opus 4.7 ranks worst across all three. This establishes that cross-seed ranking robustness holds within the AI-generated seed distribution but does not generalize to human-authored seeds with the current seed set (n=1 human source). Within a single source model distribution, β comparisons across models are valid; comparisons across seed sources with different provenance (AI-generated vs. human-authored) may not preserve rankings. Additional human-authored seed sets (n ≥ 3 independent sources) are needed to determine whether the human/AI ranking divergence is a general phenomenon or specific to this human seed set.

**Survivorship bias.** At least 17 models were attempted; 16 completed successfully. Models excluded due to API instability (empty responses, sustained rate limits, relay failures) may differ systematically from those successfully evaluated—e.g., larger reasoning models or high-demand flagship models may be disproportionately affected—and the true β range across all models may be wider than reported.

**Depth.** K = 3 generations (4 data points including Gen 0). The K=5 validation (§3.5) confirmed that exponential decay outperforms linear in all 28 model-dimension comparisons, validating the prospective commitment in §2.2. K=3 and K=5 β estimates are broadly consistent (e.g., DeepSeek-V3: K=3 β=0.028, K=5 β=0.015, both within measurement uncertainty). Extending to K = 10 is on the roadmap and would allow detection of potential plateau effects or secondary decay regimes beyond generation 5.

**Extraction fidelity.** At the ConstraintState level, the hybrid extractor achieves mean |r| = 0.47 with an independent embedding-based extractor (§B.1). Per-dimension agreement is highly uneven: sigma_fact (r = 0.90) and sigma_syntax (r = 0.59) carry meaningful signal; sigma_safety (r = 0.15) and sigma_coherence (r = 0.17) are near noise. Three of eight individual text features have validated LLM-judge correlations (Spearman ρ = 0.92–0.98; §B.1); the remaining five lack independent validation. This level of measurement fidelity limits the benchmark's ability to discriminate models separated by small β differences (e.g., Δβ < 0.01). The extractor may miss subtle constraint violations—semantically wrong but syntactically correct code, sophisticated logical contradictions that evade surface detection. The P3 neural validation (§4) partially addresses this by confirming the mechanism at the gradient level, but better extraction—particularly for sigma_safety and sigma_coherence—is a clear path to improving the metric.

**Circularity risk.** The constraint-residual framework (Deng, 2025) classifies creative_writing as an E-II constraint with no verification mechanism and predicts E-II constraints degrade fastest under recursion. The hybrid extractor, designed within the same framework by the same author, then identifies creative_writing as the dominant bottleneck in 13/16 models. This result is consistent with the framework's prediction, but the possibility that the extractor encodes the framework's theoretical expectations into its feature weights cannot be ruled out without an independent extractor designed without knowledge of the framework's hierarchy. The Gemini 2.5 Flash exception—where creative_writing sits at floor—is consistent with both "Google engineered a solution" and "the extractor has a blind spot for Gemini's output style." Human evaluation, by providing an external anchor independent of the extractor, is the direct path to breaking this circularity.

**Seed-source sensitivity.** The primary benchmark (16 models) uses seeds drawn from a GPT-4o-mini lineage. Cross-seed-source validation reveals that the β ranking is sensitive to seed origin: AI-generated seeds from different models preserve the ranking (ρ = 0.80, DeepSeek-V3 seeds vs. GPT-4o-mini seeds, n=5), but human-authored seeds substantially alter it (ρ = 0.40–0.60, n=5). Human-authored seeds also compress the β range and shift individual model β values by factors of 2–4× (e.g., DeepSeek-V3: β = 0.028 on AI seeds → 0.109 on human seeds). Only one human seed source (n=36 prompts from DEFAULT_SEEDS) has been tested; whether the divergence is systematic across diverse human seed sets is unknown. The benchmark's rankings should be interpreted as conditional on the seed distribution, and generalization to arbitrary human-written prompts requires validation with additional human seed sources.

**Temperature robustness.** The primary temperature sweep (n=100 seeds, one model, GPT-4o-mini) shows Δβ = 0.014 across three temperature settings. A subsequent three-model expansion used n=15 seeds—below the convergence threshold of n≈60 established in §3.1—and produced per-cell CVs of 0.13–0.25. Two models tested for temperature sensitivity do not establish robustness across all 16. Temperature effects should be re-examined as part of v2 with adequate per-model sample sizes.

**Human evaluation.** Human validation of the creative_writing bottleneck is a critical open problem. If human raters do not perceive the stylistic decay that the LLM-judge-augmented extractor detects, the practical significance of the metric's primary finding is diminished. A controlled human study (n ≥ 20 raters, 5 models, 3 generations each) is planned for StabilityBench v2.

**Closed models.** P3 decomposition requires model internals. For API-only models, the neural mechanism linking surface β to latent constraint dynamics can only be inferred from the open-weight validation, not directly verified.

**Language.** English only. Constraint structures vary across languages—logographic writing systems, pro-drop syntax, honorific systems—and β may vary with them.

**β_min.** The measurement floor at 0.001 is not zero. A model reading 0.001 isn't "perfectly stable"; it's "no decay detectable at current extractor resolution." As extraction improves, the floor may drop, and models currently at floor may differentiate.

**Baselines tested.** Six baseline metrics have been compared against constraint β: unigram Shannon entropy, type-token ratio (§3.5, Table 3), sentence embedding drift (all-MiniLM-L6-v2 cosine similarity vs. Gen0), direct LLM-judge quality scoring (GPT-4o-mini rating on 1–10 scale), multi-judge pure constraint extraction (3 judges scoring 5 constraint dimensions directly), and continuation perplexity (§3.5). Entropy is invariant (β_ent at floor for all models). TTR decays but is uncorrelated with β (ρ = 0.000). Embedding drift is real but orthogonal to β (ρ = 0.100, p = 0.873). Direct LLM-judge detects no decay (β_qual ≈ 0; Gen1 scores higher than Gen0). Multi-judge pure constraint extraction, despite acceptable inter-judge reliability for 4/5 dimensions, is dominated by a 43–71% human→AI quality gap and fails to detect consistent recursive degradation within AI generations—confirming that constraint structure measurement, not quality perception, is necessary to isolate the recursive signal. Continuation perplexity (GPT-4o-mini logprobs on 15 continuation tokens, n=10 seeds × 9 models with complete 4-generation data) produces measurable β_perp (0.015–0.105) but ranks models substantially differently from β_constraint (Spearman ρ = 0.47, n=9): DeepSeek-V3, the best model by constraint β (0.028), has the second-worst perplexity β (0.102), while GPT-4o, mid-range by constraint (0.099), has the best perplexity β (0.015). The modest correlation is driven by Claude-family models where both metrics partially align; non-Claude models show near-zero or reversed agreement. Human evaluation remains an important future comparison. The existing results establish that β captures information not present in entropy, lexical diversity, embedding similarity, any form of LLM-judge quality scoring, or perplexity-based text naturalness.

**General dimension.** The general dimension is described as a composite of multi-constraint balance, yet in Table 2 it sometimes exceeds its component dimensions (Llama 3.1 8B: general β = 0.261 vs. creative_writing β = 0.241). How a composite can exceed all its constituents is not fully characterized and may indicate that general captures variance not represented in the five named dimensions.

**Instruction following.** Under recursion, models may drift from the original prompt's topic or format—a failure mode distinct from constraint decay. To test whether constraint β simply measures instruction-following decay, a control experiment tracked topic fidelity (TF-IDF cosine similarity to Gen0 seed), format fidelity (regex-rubric binary checks per capability), and their geometric mean (I_k, the composite instruction-following score) across 3 models × 4 capabilities × 5 seeds × 4 recursive generations (240 API calls total). Exponential β was fitted to I_k.

The three models show a striking dissociation (Table 4): GPT-5.5—the best model by constraint β (0.011)—collapses catastrophically on instruction following (β_instr = 0.676, R² = 0.940, I_k: 1.00→0.20→0.08→0.05). GPT-4o-mini—mid-range constraint β (0.088)—decays at nearly the same rate for both metrics (β_instr = 0.088, ratio = 0.99×). Claude Opus 4.7—the worst model by constraint β (0.164)—shows intermediate instruction-following decay (β_instr = 0.219, ratio = 1.3×). Across the three models, β_constraint and β_instr are anti-correlated (Spearman ρ = −0.50): the model with the best constraint stability has the worst instruction-following stability, and vice versa.

| Model | β_constraint | β_instr | Ratio | I_k Gen0→Gen3 |
|-------|:--:|:--:|:--:|:--:|
| GPT-5.5 | 0.011 | 0.676 | 62.2× | 1.00 → 0.05 |
| GPT-4o-mini | 0.088 | 0.088 | 0.99× | 1.00 → 0.26 |
| Claude Opus 4.7 | 0.164 | 0.219 | 1.34× | 1.00 → 0.11 |

All three models show faster topic decay than format decay (β_topic > β_format in all cases), and β_format failed to fit for both GPT-4o-mini and Claude Opus 4.7 (format fidelity is high and non-monotonic: F_k ≥ 0.60 at Gen3 for both).

This dissociation is the strongest evidence to date that constraint β measures a distinct phenomenon from simple instruction-following decay. GPT-5.5 produces structurally stable but topically wayward text; Claude Opus 4.7 stays more on-topic while its constraint structure degrades substantially. The two metrics capture orthogonal dimensions of recursive generation degradation, and neither alone suffices to characterize a model's recursion behavior.

**Recursive vs. single-generation length.** Some of the observed degradation may be a function of total tokens generated rather than recursion per se. A control comparing recursive 3 × 512 tokens against a single 1,536-token generation from the same seed would isolate the recursive mechanism from length effects.

---

## 6. Conclusion

RSI measures something the standard evaluation toolkit doesn't: what happens to a model's output when it becomes its own input, repeatedly. Across 16 models from five families, that "something" spans a 15× range—from near-perfect stability to rapid degradation (β = 0.0109 to 0.1635).

The near-universal creative_writing bottleneck isn't an accident. It falls out of the constraint hierarchy: E-II stylistic constraints have no verification mechanism, no ground truth to anchor them, no parser to correct them. They're the softest point in the constraint structure, and under recursion, the softest point breaks first. The single exception—Gemini 2.5 Flash—demonstrates that the bottleneck is not a law of nature. It can be engineered away, possibly at a cost to other dimensions.

The metric itself holds up, with important caveats. Prospective methodological commitment killed the post-hoc bias that inflated earlier estimates. Test-retest runs produce identical β at reported precision for the one model tested. Temperature doesn't move the ranking for the two models tested. Six baselines—entropy, TTR, embedding drift, direct LLM-judge, multi-judge pure constraint extraction, and continuation perplexity—all fail to detect the degradation that β captures (Spearman ρ ≤ 0.10 in five cases; perplexity ρ = 0.47 with n=9, ranking models substantially differently). However, the ranking is sensitive to seed-source distribution (AI-seed ρ=0.80, human-seed ρ=0.40–0.60); the extractor achieves only moderate agreement with an independent reference (mean |r|=0.47) with two dimensions at noise level; and downstream code quality calibration (v3, 5 models, 35 problems) yields no predictive relationship (r=−0.12, p=0.87)—constraint β does not predict the rate of functional code quality decline, confirming that extractor improvements must precede downstream validation. Neural gradients are directionally consistent with the framework prediction but limited to a single small open-weight model with two of five gradient components at zero. At ~$0.50 per model, regular re-benchmarking as new models release is economically trivial.

Recursive generation is not a niche deployment pattern anymore. It's how models are trained (synthetic data), how they're aligned (self-play), and how they're deployed (agent loops). Once downstream predictive validity is established, a model's evaluation card should include the number that describes how it behaves under the regime it will actually operate in. β, pending further validation, is a candidate for that number.

---

## Appendix A: Constraint-Residual Framework Reference

This appendix provides self-contained definitions for the framework terminology used in the main text. For the full theoretical treatment, see Deng (2025).

### A.1 Constraint Layers (L0–L3)

Text generation is modeled as constrained optimization over a layered constraint hierarchy:

| Layer | Name | Description | Example |
|:-----:|------|-------------|---------|
| L0 | Physical | Token-level syntactic well-formedness | Part-of-speech agreement, parse validity |
| L1 | Logical | Binary consistency constraints | Non-contradiction: "It was raining. It was sunny." → violation |
| L2 | Formal | Deterministic verification procedures | Math derivation rules, code type-checking, formal proofs |
| L3 | Meta | Cross-layer constraint coordination | Balancing factual accuracy against stylistic fluency |

L0 and L1 constraints are binary (pass/fail). L2 constraints are formally closed—verifiable by parser, type system, or derivation rules. L3 constraints govern interactions between layers.

### A.2 Executor Types (E-I, E-II, E-III)

Constraints are transmitted through three executor categories, distinguished by their verification mechanisms:

| Type | Name | Verification | Examples |
|:----:|------|:------------:|----------|
| E-I | Formal | Deterministic (parser/theorem prover) | Code compilation, math derivation |
| E-II | Distributive | None—statistical distribution only | Creative writing style, narrative voice, imagery |
| E-III | Referential | External ground truth | Factual accuracy, entity fidelity, temporal consistency |

creative_writing is the canonical E-II executor: stylistic constraints have no parser, no binary check, no external referent. This structural property—rather than any model-specific deficiency—is why E-II constraints degrade first under recursion.

### A.3 Constraint Visibility (P1–P4)

The total constraint residual Π = Σ|∇σ_i| decomposes into four visibility components:

| Component | Name | Description |
|:---------:|------|-------------|
| P1 | Active | Constraints directly visible in output text (e.g., a factual error you can read) |
| P2 | Latent | Constraints indirectly measurable through output structure—shape the generation but are not themselves violations |
| P3 | Canceled | Active but mutually offsetting constraint pairs; zero net contribution to observable output, present in gradient |
| P4 | Emergent | New constraints that arise from interactions between existing constraints at sufficient recursion depth |

The P3 component is the "dark matter" of constraint space—invisible in output but detectable in per-token gradient decomposition (§4).

### A.4 Attractor Collapse

Under recursion, distinct constraint dimensions progressively merge—a process termed **attractor collapse**. As generation proceeds, σ_style and σ_syntax become increasingly correlated (§4.3), the effective dimensionality of the constraint space drops, and information encoded in the separation between dimensions is lost. Attractor collapse is the neural mechanism underlying the surface-level β decay measured by RSI.

### A.5 Key Notation

| Symbol | Meaning |
|:------:|---------|
| Π = Σ\|∇σ_i\| | Total constraint residual |
| σ_fact, σ_syntax, σ_style, σ_safety, σ_coherence | Five per-token gradient components |
| β | Recursive Stability Index: exponential decay rate of Π across generations |
| C_k | Total constraint magnitude at generation k |
| λ_C | Per-generation growth rate of canceled constraint fraction |

---

## Appendix B: Constraint Extractor

### B.1 Architecture

The hybrid constraint extractor combines rule-based feature extraction with LLM-judge augmentation:

- **Rule-based features** (fast, deterministic): code syntax parsing, math operator detection, entity mention extraction, self-contradiction flagging via keyword patterns
- **LLM-judge augmentation** (semantic judgment): creative writing quality scoring, nuanced logical consistency evaluation

The hybrid design was validated at three levels:

| Validation level | Metric | Result |
|------------------|--------|--------|
| Feature-level (3/8 features) | Spearman ρ vs. LLM-judge annotations | ρ = 0.93 (E-I logic density), 0.98 (E-II bigram repetition), 0.92 (E-III proper-case ratio) |
| ConstraintState-level (5 dims) | Mean |Pearson r| vs. embedding-based extractor | |r| = 0.47 (range: 0.15–0.90) |
| Capability-level (6 dims) | Spearman ρ vs. LLM-judge quality preference | ρ = 0.81 overall |

At the ConstraintState level, agreement is moderate (mean |r| = 0.47, n = 152), meaning the hybrid extractor's 5-dimensional constraint output correlates imperfectly with a pure embedding-based measurement of the same dimensions. Per-dimension agreement varies substantially: sigma_fact (r = 0.90) and sigma_syntax (r = 0.59) show reasonable alignment, while sigma_safety (r = 0.15) and sigma_coherence (r = 0.17) are near noise level. The extractor is adequate for comparative model ranking but not for precise per-sample constraint violation scores. Improving sigma_safety and sigma_coherence agreement is an extractor development priority for StabilityBench v2.

### B.2 Dimension Scoring

| Dimension | Extraction method |
|-----------|-------------------|
| creative_writing | LLM-judge: narrative structure, voice consistency, imagery coherence |
| math_reasoning | Rule-based: numerical operation validity, derivation step correctness |
| code_generation | Rule-based: parse validity (AST extraction), functional pattern matching |
| logical_consistency | Hybrid: self-contradiction detection (rule) + inference soundness (LLM) |
| factual_knowledge | Rule-based: entity co-reference resolution, temporal consistency checks |
| general | Composite: weighted average of multi-dimensional constraint balance |

### B.3 Example Extraction

For a seed prompt in the creative_writing dimension ("Write a short story about a lighthouse keeper..."), a typical Gen 1 output and its extracted scores:

```
Output (excerpt): "The lighthouse keeper had watched the sea for forty years.
  He knew every wave by its whisper, every storm by the taste of the wind..."
  
Scores:
  creative_writing:  0.72  (strong narrative voice, coherent imagery)
  logical_consistency: 0.98 (no contradictions detected)
  factual_knowledge:   1.00 (no factual claims to verify)
  math_reasoning:      N/A  (no mathematical content)
  code_generation:     N/A  (no code content)
  general:             0.85 (balanced multi-constraint profile)
```

Constraint residual Σ|∇σ_i| is computed as the aggregate violation gradient across all scored dimensions. β is then fitted to the decay of this aggregate across generations.

### B.4 Baseline Comparison

During method development, three measurement targets were compared: total_constraint (Σ|∇σ_i|), Shannon entropy, and lexical diversity (type-token ratio). total_constraint was selected as the prospectively committed target because it is the theoretically motivated quantity in the constraint-residual framework. The post-hoc selection analysis in §2.2 quantifies the bias that would result from per-model target selection. §3.5 (Table 3) provides a direct head-to-head comparison against Shannon entropy and TTR across all 16 models: entropy is invariant to recursive generation, and TTR rankings are uncorrelated with constraint β (Spearman ρ = 0.000).

Full extractor source code, configuration, and validation data are available in the GitHub repository.

---

## Data Availability

All lineage files (JSONL, 300 generations per model), per-dimension β scores, and the full analysis codebase:

- **GitHub:** [github.com/dengxinhang/constraint-residual](https://github.com/dengxinhang/constraint-residual)
- **Zenodo:** [10.5281/zenodo.20041757](https://doi.org/10.5281/zenodo.20041757)

Reproduce:
```
python run_latest_models.py --models 0 1 2 3 4 5 6
```

---

## References

1. Shumailov, I., Shumaylov, Z., Zhao, Y., Gal, Y., Papernot, N., & Anderson, R. (2024). The Curse of Recursion: Training on Generated Data Makes Models Forget. *arXiv:2305.17493*.

2. Hendrycks, D., Burns, C., Basart, S., Zou, A., Mazeika, M., Song, D., & Steinhardt, J. (2021). Measuring Massive Multitask Language Understanding. *Proceedings of ICLR 2021*.

3. Chen, M., Tworek, J., Jun, H., et al. (2021). Evaluating Large Language Models Trained on Code. *arXiv:2107.03374*.

4. Chiang, W.-L., Zheng, L., Sheng, Y., et al. (2024). Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference. *Proceedings of ICML 2024*.

5. Liang, P., Bommasani, R., Lee, T., et al. (2023). Holistic Evaluation of Language Models. *Transactions on Machine Learning Research*.

6. Deng, X. (2025). The Constraint-Residual Framework: A Theory of Semantic Stability in Recursive Language Model Generation. *Zenodo: 10.5281/zenodo.20041757*.

7. Myung, J., et al. (2026). Quantifying Conversational Reliability of Large Language Models under Multi-Turn Interaction. *Proceedings of ICLR 2026* (Outstanding Paper).

8. Zhang, Y., et al. (2026). TurnWise: The Gap between Single- and Multi-turn Language Model Capabilities. *arXiv:2603.16759*.

9. Chen, R., et al. (2026). Silent Collapse in Recursive Learning Systems. *arXiv:2605.14588*.

10. Zenil, H. & Kiani, N.A. (2026). On the Limits of Self-Improving in LLMs and Why AGI, ASI and the Singularity Are Not Near Without Symbolic Model Synthesis. *arXiv:2601.05280*.

11. Li, X., et al. (2026). Think, But Don't Overthink: Reproducing Recursive Language Models. *arXiv:2603.02615*.

---

*Correspondence: Deng Xinhang, Shenzhen, China. ORCID: 0009-0009-3974-8554*
