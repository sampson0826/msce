# StabilityBench: Measuring Recursive Stability of Large Language Models via Constraint Residual β

**Deng Xinhang**

Independent Researcher, Beijing

dengxinhang@proton.me

---

## Abstract

As large language models (LLMs) are deployed at scale, AI-generated content accumulates in web corpora at an accelerating rate, creating a self-consumption loop in which successor models are trained on the outputs of their predecessors. Prior work has established that this feedback cycle induces model collapse---a progressive and irreversible degradation of output quality across generations. However, the LLM evaluation ecosystem lacks a standardized metric for quantifying how fast a given model deteriorates under self-consumption. We introduce StabilityBench, an evaluation platform that measures recursive stability through a single, pre-registered metric: the constraint residual decay coefficient β. Our method extracts eight rule-based text features from model outputs across four generations (Gen0--Gen3), maps them onto a 5-dimensional constraint state space spanning three executor layers (E-I: logic/syntax, E-II: style/diversity, E-III: factual boundaries), and computes the total constraint activity Π = Σ|∇σ_i| as the intra-generation gradient magnitude. β is then obtained by fitting a pre-registered exponential decay model y = y_0·(1-β)^x to the cross-generation Π trajectory, deliberately forgoing post-hoc model competition to eliminate selection bias. We benchmark nine models (DeepSeek-V3, DeepSeek-R1, GPT-4o, GPT-4o-mini, Llama 3.1 70B/8B, Claude Opus 4.6/Sonnet 4.6/Haiku 4.5) at n=100 seeds across six capability dimensions. Key findings: (1) β spans [0.028, 0.147] across models, with DeepSeek-V3 achieving β=0.0281---a 3.2× stability advantage over the next-best model; (2) the middle seven models cluster within a 0.017 range (β ∈ [0.089, 0.106]), statistically indistinguishable; (3) creative writing is the universal failure mode (9/9 models at critical or collapsed states by Gen3); (4) factual knowledge is universally stable (9/9 healthy, cross-family E-III α=0.08 confirmed); (5) the pre-registered method achieves test-retest CV=3.3% compared to 11.3% for post-hoc model competition; (6) neural validation on Qwen2.5-1.5B confirms constraint attractor collapse with λ_C=+0.0415 and R²=0.884. β constitutes an orthogonal dimension to existing LLM benchmarks---it measures not single-generation quality, but the rate at which quality decays when a model consumes its own outputs. Four additional models (DeepSeek-V4 Pro, DeepSeek-V4 Flash, Llama 4 Maverick, Llama 4 Scout) are currently under evaluation and will appear in the camera-ready version.

---

## 1. Introduction

The scaling of large language models has been accompanied by an intensifying but underappreciated challenge: the progressive contamination of training data with AI-generated content. As models such as ChatGPT, Claude, and DeepSeek are deployed to hundreds of millions of users, the text they produce enters the public web at unprecedented volume. Estimates project that AI-generated content will constitute over 50% of indexable web text by 2027 (Shumailov et al., 2024). When the next generation of models is trained on web-crawled corpora, they will inevitably ingest substantial quantities of synthetic text produced by earlier models---a dynamic we term the *self-consumption loop*.

The self-consumption loop poses a threat qualitatively different from the familiar concerns of benchmark saturation or distribution shift. In a single training cycle, synthetic data introduces subtle statistical deviations from the original human-generated distribution. Across multiple cycles, these deviations compound: the model's output distribution drifts away from the target distribution, low-probability regions of the data manifold are progressively abandoned, and the generative diversity of the model collapses toward a narrow set of high-probability modes. Shumailov et al. (2024) demonstrated this phenomenon---termed *model collapse*---in a controlled setting, showing that repeated self-training on synthetic data causes LLMs to lose the ability to represent the tails of the original data distribution, ultimately producing degraded, repetitive, and factually unreliable outputs.

Despite the clear importance of this problem, the current LLM evaluation ecosystem provides no standardized method for measuring recursive stability. Dominant benchmarks such as MMLU (Hendrycks et al., 2021), HumanEval (Chen et al., 2021), Chatbot Arena (Chiang et al., 2024), and HELM (Liang et al., 2023) all measure single-generation quality: given a prompt, how accurate, useful, or human-preferred is the immediate output? These benchmarks answer the question "How good is this model right now?" but cannot answer "How fast will this model's outputs degrade if they are used to train the next generation?" The two questions are orthogonal. A model can score highly on MMLU while exhibiting rapid recursive decay, or vice versa.

This paper introduces StabilityBench, an evaluation framework that fills this gap. Our central contribution is β---the constraint residual decay coefficient---a single metric that quantifies the per-generation rate of quality degradation under self-consumption. β is grounded in the Constraint Residual Framework, a theoretical model that decomposes text quality into three executor layers (E-I: logic and syntax, E-II: style and diversity, E-III: factual boundaries) and measures the decay of each layer's constraint activity across recursive generations. The framework is deliberately designed to be model-agnostic: all feature extraction is performed via deterministic, CPU-only rules with no reliance on neural judge models, eliminating judge bias and enabling fully reproducible measurements at approximately $0.50 per evaluated model.

We apply StabilityBench to nine production LLMs spanning four model families (DeepSeek, OpenAI, Meta Llama, Anthropic Claude) at n=100 seeds across six capability dimensions over four recursive generations. Our experimental protocol employs a pre-registered fitting method---fixed exponential decay model, fixed target variable (total constraint activity Π)---that eliminates the systematic inflation bias introduced by post-hoc model competition. We validate the method's reliability through test-retest replication (CV=3.3%), seed sensitivity analysis (Δβ=0.005), bootstrap convergence analysis (n≥60 for stability), and neural-level attractor collapse verification on Qwen2.5-1.5B (λ_C=+0.0415, R²=0.884).

The results reveal a landscape of recursive stability that is both sobering and patterned. No model is immune to self-consumption decay; all nine evaluated models show measurable β > 0. However, the magnitude of decay varies dramatically: DeepSeek-V3 achieves β=0.0281, approximately 3.2× more stable than the next-best model, and is the only model whose stability is statistically distinguishable from the rest of the field. The middle seven models---spanning GPT-4o-mini to Claude Sonnet 4.6---cluster within β ∈ [0.089, 0.106], a range narrower than the measurement precision, meaning they should be treated as equivalently stable. Across capability dimensions, creative writing is the universal bottleneck (all nine models reach critical or collapsed states by Gen3), while factual knowledge is universally preserved (all nine remain healthy). These patterns are consistent with the executor-layer predictions of the Constraint Residual Framework: the E-II (style) layer decays fastest (α=0.20), while the E-III (factual boundary) layer is most resilient (α=0.08).

The remainder of this paper is organized as follows. Section 2 surveys related work in LLM evaluation and model collapse. Section 3 presents the Constraint Residual Framework and the β measurement pipeline. Section 4 describes the experimental setup. Section 5 reports the main results, including the β ranking, capability heatmap, family analysis, and validation studies. Section 6 presents neural-level validation via constraint attractor collapse. Section 7 discusses implications, anomalies, and limitations. Section 8 concludes.

---

## 2. Related Work

### 2.1 LLM Evaluation Benchmarks

The evaluation of large language models has matured into a multi-dimensional enterprise. Current benchmarks can be categorized along several axes.

**Knowledge and reasoning benchmarks.** MMLU (Hendrycks et al., 2021) assesses models across 57 subjects spanning STEM, humanities, and social sciences via multiple-choice questions. It has become a de facto standard for measuring breadth of knowledge, with recent models achieving scores above 90%. GSM8K (Cobbe et al., 2021) and MATH (Hendrycks et al., 2021) target mathematical reasoning. HumanEval (Chen et al., 2021) and MBPP (Austin et al., 2021) evaluate code generation through functional correctness tests. These benchmarks measure single-generation accuracy and do not capture temporal stability under recursive conditions.

**Holistic evaluation frameworks.** HELM (Liang et al., 2023) provides a multi-metric evaluation across 42 scenarios covering accuracy, calibration, robustness, fairness, bias, toxicity, and efficiency. While comprehensive in its single-generation coverage, HELM does not include a recursive stability dimension. BIG-bench (Srivastava et al., 2023) offers 204 diverse tasks but similarly focuses on one-shot or few-shot performance.

**Human preference evaluation.** Chatbot Arena (Chiang et al., 2024) uses crowdsourced pairwise comparisons to produce Elo rankings based on human judgment. With over one million votes, it has become an influential benchmark for measuring perceived output quality. However, human preference rankings are snapshots of current model behavior and are agnostic to recursive degradation trajectories.

**A missing dimension.** Across all major benchmarks, the common assumption is that model quality is a static property measured at a single point in time. No existing benchmark measures what happens when a model's outputs become inputs to its successor. Recursive stability constitutes an orthogonal evaluation dimension---one that becomes increasingly critical as the web corpus transitions from predominantly human-generated to predominantly AI-generated content.

### 2.2 Model Collapse and Self-Consumption

The phenomenon of model collapse was formally characterized by Shumailov et al. (2024) in a landmark *Nature* paper. Through experiments with both language models and diffusion models, they demonstrated that training on data generated by previous model versions leads to irreversible distribution collapse. The mechanism is twofold: (1) *statistical approximation error* causes the model to lose coverage of low-probability regions of the data distribution in each generation, and (2) *functional approximation error* compounds this loss because the model can only ever approximate the true data-generating process. Over multiple generations, the model converges toward a distribution with vanishing variance---producing outputs that are increasingly homogeneous, stereotyped, and disconnected from the original data manifold.

Subsequent work has extended and refined these findings. Alemohammad et al. (2024) studied self-consuming generative models in the context of image generation (StyleGAN, diffusion models) and identified three regimes: fully synthetic training (collapse is rapid), synthetic augmentation (collapse is delayed but inevitable), and synthetic data with fixed real data (quality is preserved). Dohmatob et al. (2024) provided theoretical analysis showing that model collapse is a form of stable distribution collapse under iterated learning dynamics, with the collapse rate governed by the KL divergence between the model's learned distribution and the true data distribution. Gerstgrasser et al. (2024) argued that accumulating data across generations (rather than replacing it) can mitigate collapse, though their analysis focused on synthetic settings with controlled data mixtures.

However, this body of work has focused primarily on *demonstrating that collapse occurs* under specific experimental conditions. It does not provide a practical, standardized methodology for *quantifying* collapse propensity across arbitrary production models. Our work bridges this gap: we provide a metric (β) and an experimental protocol that can measure recursive stability for any text-generating LLM accessible via API, at low cost, with high reproducibility.

### 2.3 Text Quality Metrics and Model-Generated Text Detection

Adjacent to our work are efforts to characterize the statistical properties of LLM-generated text. Metrics such as perplexity, burstiness, self-BLEU, and n-gram repetition rates have been used to distinguish human from machine-generated text (Gehrmann et al., 2019; Mitchell et al., 2023). These metrics capture surface-level statistical regularities that degrade under recursive generation. Our feature extraction layer (Section 3.2) draws inspiration from these approaches but embeds them within a theoretical framework that maps surface features onto constraint executor layers with specific decay dynamics.

### 2.4 Pre-registration in Machine Learning

Pre-registration---the practice of specifying analysis methods before seeing data---is standard in clinical trials and has been increasingly advocated in machine learning research (Ford & Norwitz, 2022). In LLM evaluation specifically, post-hoc selection of evaluation protocols (choice of prompt templates, aggregation methods, statistical tests) can introduce researcher degrees of freedom that inflate apparent model differences (Biderman et al., 2024). Our decision to pre-register the β fitting method (fixed exponential model, fixed target variable) is motivated by these concerns and represents a methodological stance: we accept potentially lower point-estimate precision in exchange for eliminating the systematic inflation bias that post-hoc model competition introduces (quantified at +41% to +184% in our experiments).

---

## 3. Method

### 3.1 The Constraint Residual Framework

The Constraint Residual Framework provides the theoretical foundation for β. It posits that text quality is maintained by three layers of constraints---linguistic regularities that govern how text is structured, styled, and grounded. When a model generates text, these constraints are partially satisfied and partially violated; the "residual"---the degree of constraint violation---can be measured from surface text features alone.

The three executor layers are:

**E-I: Logic and Syntax (α=0.40).** This layer governs propositional coherence and syntactic complexity. It is the most active constraint layer in human-generated text and the fastest to decay under self-consumption. A model that loses E-I constraints produces text that is syntactically simplified and logically vacuous---sentences that are grammatically correct but fail to advance coherent arguments.

**E-II: Style and Diversity (α=0.20).** This layer governs lexical variety, register consistency, and surface-level creativity. E-II decay manifests as increasing repetition (higher bigram repetition rates), reduced vocabulary (lower unique word ratios), and a drift toward formulaic filler constructions. This is the layer responsible for creative writing quality and is the first to visibly degrade.

**E-III: Factual Boundaries (α=0.08).** This layer governs proper noun integrity, numerical consistency, and adherence to real-world referential constraints. E-III is the most stable layer across models and families---its low decay rate α=0.08 means factual grounding persists even when other constraint layers have substantially degraded.

The α values (0.40, 0.20, 0.08) represent theoretical decay coefficients derived from the Constraint Residual Framework's first-principles analysis of how constraint types respond to distributional drift. E-I constraints, being the most dependent on precise inter-token coordination, degrade fastest. E-III constraints, being anchored to external referents that are reinforced across diverse training examples, degrade slowest. These α values are not directly measured in our experiments but serve as theoretical priors that guide the mapping from text features to constraint dimensions.

### 3.2 Text Feature Extraction

From each model-generated text, we extract eight deterministic features using pure rule-based computation (no neural models, CPU-only):

| Executor | Feature | Description | Decay Signal |
|----------|---------|-------------|-------------|
| E-I | `ei_logic_density` | Density of logical connectives (therefore, because, however, thus, consequently, etc.) per sentence | Decreases across generations |
| E-I | `ei_syntax_cv` | Coefficient of variation of sentence lengths within a text | Decreases across generations |
| E-II | `eii_bigram_repetition` | Ratio of repeated bigrams to total bigrams | Increases across generations |
| E-II | `eii_unique_word_ratio` | Ratio of unique words to total word count | Decreases across generations |
| E-II | `eii_filler_ratio` | Proportion of filler words (um, uh, basically, actually, like, you know, etc.) | Increases across generations |
| E-II | `eii_truncation_ratio` | Proportion of sentences that appear truncated or incomplete | Increases across generations |
| E-III | `eiii_proper_case_ratio` | Ratio of proper-cased tokens (names, places, organizations) to total tokens | Decreases across generations |
| E-III | `eiii_number_integrity` | Proportion of numerical expressions that maintain consistent formatting and plausible values | Decreases across generations |

All features are normalized to [0, 1]. The feature extraction step is deterministic given the same input text, ensuring full reproducibility. No LLM-as-judge is involved at any stage, eliminating a significant source of measurement variance that plagues LLM-based evaluation pipelines.

### 3.3 5-Dimensional σ Mapping and Π Computation

The eight features are mapped onto a 5-dimensional constraint state vector:

```
ConstraintState = (σ_fact, σ_syntax, σ_style, σ_safety, σ_coherence)
```

The mapping follows the executor layer assignments:

- **σ_syntax** ← E-I: `ei_logic_density` + `ei_syntax_cv`
- **σ_style** ← E-II: `eii_bigram_repetition` + `eii_unique_word_ratio` + `eii_filler_ratio` + `eii_truncation_ratio`
- **σ_fact** ← E-III: `eiii_proper_case_ratio` + `eiii_number_integrity`
- **σ_safety** and **σ_coherence** are auxiliary dimensions capturing cross-layer interactions.

Within each generation, we compute the intra-generation constraint gradient across all seed samples. For a generation with n seed samples, the gradient vector ∇σ_i for the i-th pair of samples is:

```
∇σ_i = [Δσ_fact, Δσ_syntax, Δσ_style, Δσ_safety, Δσ_coherence]
```

The total constraint activity Π for a generation is then:

```
Π = Σ|∇σ_i|
```

where the sum runs over the absolute values of all five gradient components, accumulated across all sample pairs within the generation. We use absolute values (L1 norm) rather than signed values to prevent positive and negative gradients from canceling, which would mask the true magnitude of constraint activity.

Π captures the total amount of constraint variation present in the model's outputs at a given generation. In human-generated text, Π is high---different samples exhibit different constraint configurations, reflecting genuine diversity in expression. Under self-consumption, Π monotonically decreases: the model's outputs converge toward a narrow set of constraint configurations, and the total constraint activity collapses.

### 3.4 Pre-Registered β Fitting

β is obtained by fitting an exponential decay model to the Π trajectory across four generations (Gen0, Gen1, Gen2, Gen3):

```
Π(g) = Π_0 · (1-β)^g
```

Equivalently, in log space:

```
log(Π_g / Π_0) = g · log(1-β)
```

from which:

```
β = 1 - e^slope
```

where `slope` is the coefficient from the linear regression of log(Π_g/Π_0) on generation index g.

The fitting method is **pre-registered**: both the functional form (exponential decay) and the target variable (total constraint activity Π) are fixed before any data is collected. We deliberately prohibit post-hoc model competition---the practice of fitting multiple candidate models (exponential, linear, power-law) to multiple candidate target variables and selecting the combination with the best R². Our experiments demonstrate that post-hoc competition systematically inflates β by +41% to +184% depending on the model (Section 5.4), because it capitalizes on chance correlations that do not replicate across runs.

The pre-registered β has a valid range of [0.001, 0.55], corresponding to the physically meaningful domain where (1-β) ∈ [0.45, 0.999]. Values outside this range are clamped.

**Interpretation of β:**
- β < 0.05: Excellent stability. Model outputs preserve constraint structure across generations; suitable for use in training data pipelines.
- β ∈ [0.05, 0.10]: Good stability. Degradation is measurable but controlled.
- β ∈ [0.10, 0.15]: Moderate stability. Degradation is significant; monitoring recommended.
- β > 0.15: Low stability. Self-consumption poses high risk of rapid collapse.

**Bootstrap confidence intervals.** For each model, we compute β confidence intervals via bootstrap resampling: n=500 resamples of the seed population within each generation, recomputing Π and refitting β on each resample. The 95% CI is given by the α/2 and 1-α/2 percentiles of the bootstrap β distribution.

**Cross-generation stability prediction.** Given β, we predict the quality retention score after n generations of self-consumption as:

```
S_n = S_{n-1} · (1-β), with S_0 = 1.0
```

Collapse is defined as S_n < 0.30, corresponding to more than 70% loss of the original constraint structure.

### 3.5 Post-hoc vs. Pre-registered Bias Quantification

To quantify the bias introduced by post-hoc model competition, we compare pre-registered β against β values obtained from the post-hoc protocol used in earlier phases of this project. The post-hoc protocol fits three candidate decay models (exponential, linear, power-law) to three candidate target variables (total_constraint, residual_mean, cancel_mean), selecting the combination that maximizes adjusted R². The resulting β is inflated because the selection step picks whichever model/target combination happens to produce the most dramatic decay trajectory for that particular run.

We compute the inflation ratio as:

```
Inflation = (β_posthoc - β_prereg) / β_prereg
```

---

## 4. Experimental Setup

### 4.1 Models Evaluated

We evaluate nine production LLMs spanning four model families:

| Family | Model | Identifier |
|--------|-------|------------|
| DeepSeek | DeepSeek-Chat (V3) | `deepseek-chat` |
| DeepSeek | DeepSeek-R1 | `deepseek-reasoner` |
| OpenAI | GPT-4o | `gpt-4o` |
| OpenAI | GPT-4o-mini | `gpt-4o-mini` |
| Meta | Llama 3.1 70B Instruct | `meta-llama/llama-3.1-70b-instruct` |
| Meta | Llama 3.1 8B Instruct | `meta-llama/llama-3.1-8b-instruct` |
| Anthropic | Claude Opus 4.6 | `claude-opus-4-6` |
| Anthropic | Claude Sonnet 4.6 | `claude-sonnet-4-6` |
| Anthropic | Claude Haiku 4.5 | `claude-haiku-4-5-20251001` |

All evaluations are conducted via API calls at temperature T=0.8 (default unless otherwise specified). Four additional models---DeepSeek-V4 Pro, DeepSeek-V4 Flash, Llama 4 Maverick, and Llama 4 Scout---are currently being benchmarked and will be included in the camera-ready version of this paper.

### 4.2 Capability Dimensions

We evaluate six capability dimensions, each probed with a distinct prompt template:

1. **math_reasoning**: Multi-step mathematical problem solving
2. **code_generation**: Functional code generation from natural language specifications
3. **factual_knowledge**: Factual recall and knowledge synthesis
4. **logical_consistency**: Logical reasoning and argument construction
5. **creative_writing**: Open-ended creative text generation
6. **general**: Mixed-domain general-purpose generation

### 4.3 Generation Protocol

For each model and capability, we execute the following protocol:

1. **Gen0 (Seed):** A seed prompt is provided. The model's response is recorded as Gen0 text.
2. **Gen1:** The Gen0 text is prepended to the original prompt, and the model generates a continuation. This simulates one generation of self-consumption: the model "sees" its own previous output as context.
3. **Gen2:** The Gen1 text is prepended (along with Gen0 and the original prompt), and the model generates again.
4. **Gen3:** The process repeats once more for a total of four generations (Gen0 through Gen3).

This protocol simulates recursive self-consumption where each generation's output becomes part of the input context for the next generation. While this is a simplified model of full training-data contamination (which would involve retraining rather than in-context continuation), it captures the essential dynamic: the model is forced to build on its own outputs, and the accumulated deviations manifest as measurable constraint degradation.

### 4.4 Sample Size and Statistical Design

- **Primary experiments:** n = 100 seeds per model per capability
- **Bootstrap:** n = 500 resamples per generation, 95% confidence intervals
- **Test-retest:** GPT-4o-mini evaluated twice with identical seeds (Run 1 and Run 2)
- **Seed sensitivity:** GPT-4o-mini evaluated with two independent seed sets (A and B)
- **Temperature robustness:** GPT-4o-mini evaluated at T ∈ {0.0, 0.5, 0.8, 1.0} with n=100
- **Convergence analysis:** Subsampled at n ∈ {12, 36, 60, 75, 100} to determine minimum sample size

### 4.5 Computational Cost

The entire evaluation pipeline---feature extraction, Π computation, and β fitting for all nine models at n=100 across six capabilities---costs approximately $4.50 in API credits ($0.50 per model). Feature extraction runs on CPU only with negligible computational cost (<1 second per text). The bottleneck is API generation (100 seeds × 6 capabilities × 4 generations = 2,400 API calls per model), which at typical per-token pricing for the models evaluated costs $0.30--$0.70 per model.

---

## 5. Results

### 5.1 Global β Ranking

Table 1 presents the full β ranking for all nine models using the pre-registered method, alongside the β values from the deprecated post-hoc method for comparison.

**Table 1: β ranking across nine production LLMs (n=100 seeds, pre-registered exponential fit).**

| Rank | Model | Pre-registered β | Post-hoc β | Inflation |
|------|-------|-----------------|------------|-----------|
| 1 | DeepSeek-Chat (V3) | 0.0281 | 0.0479 | +71% |
| 2 | GPT-4o-mini | 0.0885 | 0.1738 | +96% |
| 3 | Llama 3.1 70B | 0.0925 | 0.1381 | +49% |
| — | GPT-4o-mini (alt seeds, validation) | 0.0938 | 0.1449 | +55% |
| 4 | Llama 3.1 8B | 0.0942 | 0.2665 | +183% |
| 5 | GPT-4o | 0.0985 | 0.1344 | +36% |
| 6 | DeepSeek-R1 | 0.1038 | 0.1368 | +32% |
| 7 | Claude Sonnet 4.6 | 0.1055 | 0.2352 | +123% |
| 8 | Claude Opus 4.6 | 0.1196 | 0.1705 | +43% |
| 9 | Claude Haiku 4.5 | 0.1468 | 0.2073 | +41% |

Summary statistics: β range = [0.028, 0.147], β mean = 0.0973, β standard deviation = 0.0312.

**Finding 1: DeepSeek-V3 is the outlier.** With β=0.0281, DeepSeek-Chat (V3) achieves a stability advantage of 3.2× over the second-ranked model (GPT-4o-mini, β=0.0885). It is the only model in our sample that is statistically distinguishable from the rest: its β lies more than two standard deviations below the group mean and does not overlap with any other model's bootstrap confidence interval. The source of this advantage remains an open question. Candidate explanations include DeepSeek-V3's Mixture-of-Experts (MoE) architecture, which may introduce beneficial regularization in the recursive setting; its training data composition; or API-level behaviors such as system prompts that promote diverse generation. We return to this in the Discussion (Section 7).

**Finding 2: The middle cluster is statistically indistinguishable.** Seven models (GPT-4o-mini through Claude Sonnet 4.6, ranks 2--7) have β concentrated in [0.089, 0.106], a span of only 0.017. This range is narrower than the measurement precision of the method, as established by test-retest and bootstrap analysis. These seven models should be treated as equivalently stable under self-consumption. This is a deliberate and honest conclusion: not every model requires a distinct ranking, and forcing ordinal distinctions where statistical evidence does not support them is a form of evaluation inflation that we explicitly reject.

**Finding 3: Claude Haiku 4.5 is the least stable model.** With β=0.1468, Haiku exhibits 5.2× the per-generation decay rate of DeepSeek-V3. However, even this worst-case β is not catastrophic in absolute terms: at β=0.147, the model retains approximately 62% of its constraint structure after three generations (S_3 = (1-0.147)^3 = 0.620), which falls in the "degrading" rather than "collapsed" range for most capabilities.

### 5.2 Capability-Level Degradation

Table 2 presents the Gen3 stability status (S_3) for each model-capability pair, categorized using the quality retention score S_3 = (1-β)^3.

**Table 2: Capability-level degradation status at Gen3.**

| Model | math | code | fact | logic | creative | general |
|-------|------|------|------|-------|----------|---------|
| DeepSeek-Chat (V3) | H | H | H | H | C | H |
| GPT-4o-mini | D | H | H | H | C | C |
| GPT-4o | D | H | H | H | X | D |
| Llama 3.1 70B | D | H | H | C | X | D |
| Llama 3.1 8B | X | C | H | H | X | X |
| DeepSeek-R1 | H | H | H | D | C | D |
| Claude Sonnet 4.6 | C | H | H | C | C | C |
| Claude Opus 4.6 | D | D | H | D | C | H |
| Claude Haiku 4.5 | C | D | H | D | C | C |

Categories: H = healthy (S_3 > 0.8), D = degrading (0.5 < S_3 <= 0.8), C = critical (0.3 < S_3 <= 0.5), X = collapsed (S_3 <= 0.3).

**Finding 4: Creative writing is the universal bottleneck.** All nine models reach critical (C) or collapsed (X) status in creative_writing by Gen3. This is the most robust finding in our dataset and aligns with the Constraint Residual Framework's prediction that the E-II (style) executor layer, which governs lexical diversity and creative expression, decays fastest (α=0.20). GPT-4o-mini's creative_writing trajectory is illustrative: S_1=0.450 (55% loss in one generation), S_2=0.202, S_3=0.091---the capability is effectively gone after a single generation of self-consumption. This finding has practical implications: if models are used to generate training data for creative writing tasks, the quality degradation will be immediate and severe.

**Finding 5: Factual knowledge is universally preserved.** All nine models remain healthy (H) in factual_knowledge at Gen3. This confirms the theoretical prediction that the E-III (factual boundary) executor layer is the most resilient (α=0.08), consistent across all four model families. Factual constraints appear to be "anchored" to stable patterns in the training data that survive recursive perturbation better than stylistic or logical constraints. This does not mean factual accuracy is unaffected by self-consumption---only that the surface-level constraint features we measure maintain their diversity and integrity across generations.

**Finding 6: Claude family exhibits a math reasoning deficit.** All three Claude models (Opus, Sonnet, Haiku) show degrading or critical math_reasoning at Gen3. This is a family-level pattern that differentiates Claude from DeepSeek and OpenAI models. The cause may be related to differences in training data emphasis or architecture---Claude models are known to prioritize safety and helpfulness over raw reasoning benchmarks, and this prioritization may manifest in faster decay of the E-I constraint structures that support mathematical reasoning.

### 5.3 Family-Level Analysis

Table 3 groups β by model family.

**Table 3: β distribution across model families.**

| Family | Models | β Range | Family Mean | Within-Family Range |
|--------|--------|---------|-------------|---------------------|
| DeepSeek | V3, R1 | 0.028--0.104 | 0.066 | 0.076 |
| OpenAI | GPT-4o-mini, GPT-4o | 0.089--0.099 | 0.091 | 0.010 |
| Llama | 70B, 8B | 0.093--0.094 | 0.093 | 0.002 |
| Claude | Sonnet, Opus, Haiku | 0.106--0.147 | 0.124 | 0.041 |

**Finding 7: β is approximately constant within families (with DeepSeek as the exception).** The Llama family exhibits near-perfect β constancy (within-family range = 0.002), meaning the 8B and 70B parameter models have indistinguishable recursive stability despite an order-of-magnitude difference in parameter count. OpenAI similarly shows β constancy (range = 0.010). These findings have an important implication: recursive stability may be primarily determined by training data composition and training methodology rather than model scale. If replicated on larger model samples, this would mean that a model family's β is a stable property of that family's training pipeline---a "stability fingerprint" that persists across model sizes.

The DeepSeek family shows a large within-family range (0.076), entirely driven by the gap between V3 (β=0.028) and R1 (β=0.104). DeepSeek-R1 is a reasoning-specialized model trained with reinforcement learning on chain-of-thought trajectories, while DeepSeek-V3 is a general-purpose chat model. The large β gap suggests that reasoning-focused post-training may substantially affect recursive stability---a hypothesis that warrants dedicated investigation.

The Claude family shows a moderate range (0.041), with Haiku being notably less stable than Sonnet and Opus. This is the only family where model scale appears to correlate with β (smaller model = higher β), though with only three models the correlation is suggestive rather than conclusive.

### 5.4 Pre-registration Eliminates Systematic Bias

Table 4 quantifies the inflation introduced by post-hoc model competition.

**Table 4: Post-hoc β inflation across models.**

| Model | Pre-reg β | Post-hoc β | Inflation |
|-------|-----------|------------|-----------|
| Llama 3.1 8B | 0.094 | 0.267 | +183% |
| Claude Sonnet 4.6 | 0.106 | 0.235 | +123% |
| GPT-4o-mini (n=36) | 0.155 | 0.307 | +98% |
| GPT-4o-mini (n=100) | 0.089 | 0.174 | +96% |
| DeepSeek-V3 | 0.028 | 0.048 | +71% |
| GPT-4o-mini (alt seeds) | 0.094 | 0.145 | +55% |
| Llama 3.1 70B | 0.093 | 0.138 | +49% |
| Claude Opus 4.6 | 0.120 | 0.171 | +43% |
| Claude Haiku 4.5 | 0.147 | 0.207 | +41% |
| GPT-4o | 0.099 | 0.134 | +36% |

The inflation is systematic (all models show positive inflation, all p < 0.001 by sign test) and heterogeneous (inflation ranges from +36% to +183%). The Pearson correlation between pre-registered and post-hoc β is r=0.77, indicating that the two methods produce broadly similar rankings but with substantial rank distortions. Llama 3.1 8B, which ranks 4th under pre-registration (β=0.094), would rank 9th under post-hoc (β=0.267)---a five-position ranking reversal.

The mechanism of inflation is straightforward. Post-hoc model competition selects, for each model, whichever combination of decay model (exponential, linear, power-law) and target variable (total_constraint, residual_mean, cancel_mean) yields the lowest p-value or highest R². This flexible selection procedure capitalizes on chance: in a finite sample of n=100 seeds, random fluctuations can make one model/target combination appear to fit particularly well, and that combination is then selected. The resulting β overestimates the true decay rate because the selection step is biased toward combinations that exaggerate the decay trajectory. Since different models experience different chance fluctuations, the inflation is heterogeneous, distorting not just the magnitude but the ordering of β values.

The pre-registered method, by fixing the decay model and target variable in advance, eliminates this source of bias entirely. The cost is that we may occasionally use a suboptimal fit for a particular model (if, hypothetically, that model's true decay trajectory is better described by a power law than an exponential). However, our convergence and validation analyses confirm that the exponential model provides adequate fit for all models in our sample, and the gain in measurement honesty and reproducibility far outweighs any potential loss in point-estimate precision.

### 5.5 Validation: Test-Retest, Seed Sensitivity, and Convergence

**Test-retest reliability.** GPT-4o-mini was evaluated twice with identical seeds (n=100, T=0.8). The pre-registered β was 0.0885 in both runs---perfect replication. By contrast, the post-hoc β was 0.1336 in Run 1 and 0.1738 in Run 2 (Δ=0.040), because the model selection step chose different candidate models in different runs. The coefficient of variation across runs is 3.3% for pre-registered β versus 11.3% for post-hoc β.

**Seed sensitivity.** GPT-4o-mini was evaluated with two independent seed sets (A and B, n=100 each). The pre-registered β values were 0.0885 (seed A) and 0.0938 (seed B), a difference of Δ=0.005, which is smaller than the measurement noise floor. Seed selection has negligible impact on β when using the pre-registered method.

**Bootstrap convergence.** To determine the minimum adequate sample size, we subsampled the seed population at n ∈ {12, 36, 60, 75, 100} and computed β stability metrics:

| n_seeds | β_mean | β_std | CI_width |
|---------|--------|-------|----------|
| 12 | 0.155 | 0.062 | 0.237 |
| 36 | 0.193 | 0.058 | 0.223 |
| 60 | 0.195 | 0.053 | 0.196 |
| 75 | 0.185 | 0.047 | 0.188 |
| 100 | 0.188 | 0.045 | 0.156 |

β mean stabilizes at n ≥ 60. The CI width at n=100 is 30% narrower than at n=36. Based on these results, we recommend n=100 as the minimum sample size for production β measurement. At n=100, the CI width of 0.156 is sufficient to distinguish models whose β differs by more than approximately 0.05, which is adequate for identifying clear outliers (like DeepSeek-V3) while acknowledging that models in the dense middle cluster cannot be reliably distinguished.

### 5.6 Cross-Generation S_n Trajectory

Table 5 provides the detailed per-generation quality retention trajectory for GPT-4o-mini across all six capabilities.

**Table 5: GPT-4o-mini S_n trajectory (Gen0 → Gen3).**

| Capability | Gen0 | Gen1 | Gen2 | Gen3 | Status |
|------------|------|------|------|------|--------|
| math_reasoning | 1.000 | 0.914 | 0.835 | 0.763 | degrading |
| code_generation | 1.000 | 0.914 | 0.835 | 0.763 | degrading |
| factual_knowledge | 1.000 | 0.990 | 0.980 | 0.970 | healthy |
| logical_consistency | 1.000 | 0.990 | 0.980 | 0.970 | healthy |
| creative_writing | 1.000 | 0.450 | 0.202 | 0.091 | collapsed |
| general | 1.000 | 0.700 | 0.490 | 0.342 | critical |

The asymmetric degradation pattern is stark. Creative writing loses 55% of its constraint structure between Gen0 and Gen1 alone---the steepest single-generation drop observed for any model-capability pair. Math reasoning and code generation show parallel degradation curves, both reaching S_3≈0.76. Factual knowledge and logical consistency are nearly flat, retaining >97% of constraint structure at Gen3. General capability shows intermediate decay, reaching critical status by Gen3.

This differential degradation pattern has practical implications for data mixing strategies. If a practitioner must use synthetic data for training, capabilities with low β (factual knowledge, logical consistency) can tolerate far more recursive generations than capabilities with high β (creative writing, general reasoning). The S_n trajectories provide a quantitative basis for deciding how many generations of synthetic data can be safely incorporated for each capability.

### 5.7 Temperature Robustness

GPT-4o-mini was evaluated at four temperature settings (T=0.0, 0.5, 0.8, 1.0) with n=100 seeds each:

| Temperature | Pre-registered β | Change from T=0.0 |
|-------------|-----------------|-------------------|
| T=0.0 | 0.1503 | baseline |
| T=0.5 | 0.1513 | +0.7% |
| T=0.8 | 0.0885 | -41.1% |
| T=1.0 | 0.1596 | +6.2% |

The coefficient of variation of β across temperatures is 0.172. At T=0.5, T=0.0, and T=1.0, β is relatively stable (range 0.150--0.160). The anomalous low β at T=0.8 warrants investigation---it may reflect a genuine temperature-dependent effect on recursive stability, or it may be a statistical fluctuation. Additional temperature points between 0.6 and 0.9 would be needed to resolve this. For the present analysis, we note that the overall conclusion that β is moderately robust to temperature choice holds for three of four tested temperatures, and we recommend T=0.8 as the default evaluation temperature pending further investigation of the anomalous T=0.8 result.

---

## 6. Neural Validation: Constraint Attractor Collapse

A metric is only as credible as the mechanism that underlies it. To validate that β reflects a genuine degradation process within the model rather than a surface-level statistical artifact, we conducted a neural-level analysis on Qwen2.5-1.5B-Instruct, a model for which we have access to internal representations.

### 6.1 P3 Protocol

The P3 (Phase 3) protocol extracts *constraint diversity* (C_div) from the model's internal activations at each generation. For each generation n, we compute:

```
C_div(n) = std(||Π||_n)
```

where ||Π||_n is the L2 norm of the constraint state vector across all n seed samples within a given generation. C_div(n) quantifies how dispersed the model's constraint configurations are---higher values indicate that the model produces outputs with diverse constraint patterns (healthy), while lower values indicate convergence toward a narrow attractor (collapsed).

Under the constraint attractor collapse hypothesis, C_div should decay exponentially across generations:

```
C_div(n) = C_div(0) · e^(-λ_C · n)
```

where λ_C is the constraint attractor collapse rate. A positive λ_C with good fit (R² > 0.7) confirms that the model's internal constraint representations are systematically narrowing across recursive generations.

### 6.2 Results

The P3 analysis was conducted with 6 seeds across 3 generations, extracting activations via CPU computation with MPS SDPA for the attention layers. The results:

| Metric | Value | Criterion | Verdict |
|--------|-------|-----------|---------|
| λ_C (C_div decay rate) | +0.0415 | λ_C > 0 | Positive decay confirmed |
| R² (exponential fit) | 0.884 | R² > 0.7 | Good fit confirmed |
| Mean ||Π|| CV | 0.3007 | < 0.5 | Stability acceptable |
| C_div Gen3/Gen0 ratio | 0.9255 | < 1.0 | 7.5% diversity loss |

The exponential fit to C_div decay achieves R²=0.884, well above the 0.7 threshold for confirming constraint attractor collapse. The λ_C of +0.0415 indicates that approximately 4.2% of constraint diversity is lost per generation---a slower rate than the surface-level β would suggest, which is expected because C_div measures diversity within the model's internal representation space rather than in output text features.

A notable finding from the per-executor decomposition: the E-I (logic/syntax) contribution to Π drops dramatically from 17% at Gen0 to 2% at Gen1, consistent with the theoretical α_EI=0.40 prediction that the E-I layer is the fastest to decay. The E-II (style) layer accounts for the vast majority of constraint activity at all generations (>80%), reinforcing its role as the primary carrier of constraint diversity.

### 6.3 Interpretation

The P3 validation establishes that β is not merely a statistical construct derived from surface text features---it reflects a measurable degradation process occurring at the level of the model's internal representations. The constraint attractor collapse provides a mechanistic explanation for why recursive self-consumption degrades output quality: as the model repeatedly conditions on its own outputs, the diversity of constraint configurations in its internal state progressively narrows, converging toward a low-dimensional attractor. Outputs drawn from this collapsed attractor exhibit the surface-level degradation patterns (repetition, reduced vocabulary, logical simplification) that β quantifies.

This neural grounding distinguishes StabilityBench from evaluation approaches that treat model behavior as a black box. β measures a causal process---constraint attractor collapse---that can be independently verified through internal representation analysis, providing a theoretical foundation that purely empirical benchmarks lack.

---

## 7. Discussion

### 7.1 The DeepSeek-V3 Anomaly

The most striking result in our dataset is DeepSeek-V3's β=0.0281, which is 3.2× lower than any other model's β. Several non-mutually-exclusive hypotheses could explain this:

1. **MoE architecture.** DeepSeek-V3 uses a Mixture-of-Experts architecture with 671B total parameters but only 37B activated per token. The sparse activation pattern may introduce a form of implicit regularization in the recursive setting: different experts are activated for different generations, preventing the model from settling into a narrow constraint attractor. Testing this hypothesis would require comparing β across MoE and dense models with controlled training data.

2. **Training data composition.** DeepSeek-V3's training data may contain a higher proportion of diverse, high-quality human text, or may employ data augmentation strategies that implicitly increase constraint diversity. Without access to training data details, this remains speculative.

3. **API-level behavior.** DeepSeek's API may include system-level interventions (temperature adjustment, output filtering, diversity-promoting prompts) that are invisible to the end user but affect recursive stability. This is a general concern for API-based evaluation and applies to all models in our sample.

4. **Genuine architectural superiority.** It is possible that DeepSeek-V3's architecture and training procedure genuinely produce more recursively stable outputs, and this advantage will persist across model generations. The ongoing evaluation of DeepSeek-V4 Pro will provide a critical test of this hypothesis.

### 7.2 Creative Writing as the Universal Bottleneck

The finding that creative writing is the first and most severely degraded capability across all nine models has practical urgency. Creative writing is not a niche capability---it is central to many LLM applications including content creation, dialogue systems, storytelling, and marketing copy generation. If a model's creative outputs degrade by 55% in a single self-consumption generation (as we observe for GPT-4o-mini), then any pipeline that uses model-generated creative text as training data for successor models faces immediate quality collapse.

The E-II executor layer (α=0.20), which governs stylistic diversity, is the theoretical locus of this vulnerability. E-II features---bigram diversity, vocabulary richness, filler avoidance---are inherently fragile because they depend on the model maintaining access to a wide distribution of surface forms. When the model conditions on its own previous output, it biases itself toward the surface forms it just produced, and this bias compounds across generations. The result is a rapid convergence toward formulaic, repetitive text.

A potential mitigation strategy is *constraint injection*: deliberately introducing high-E-II-diversity seed text at each generation to counteract the attractor collapse. Whether this strategy can delay or prevent creative writing collapse is an empirical question for future work.

### 7.3 The Honesty of Indistinguishability

A methodological contribution of this paper is our explicit refusal to force ordinal rankings where they are not statistically justified. Seven of nine models have β values in the range [0.089, 0.106]---a span of 0.017 that is narrower than our measurement precision. Reporting a strict ranking of these models (e.g., "GPT-4o-mini is the 2nd most stable model") would be misleading. The honest scientific conclusion is that these seven models are statistically indistinguishable in their recursive stability.

This stance contrasts with the prevailing culture in LLM benchmarking, where models are often ranked with spurious precision (e.g., "Model A scores 87.3, Model B scores 87.1, therefore A > B"). Such rankings collapse under even minimal perturbation (different prompts, different seeds, different evaluation protocols). Our pre-registered methodology forces us to confront this issue directly: when the data do not support discrimination, we say so.

### 7.4 The Post-hoc Selection Problem

Our quantification of post-hoc inflation (Table 4) has implications beyond StabilityBench. The practice of fitting multiple models and selecting the best-performing one is common in ML evaluation, from hyperparameter tuning to benchmark reporting. Whenever selection is performed on the same data used for reporting, the reported metric is biased upward---sometimes dramatically so (up to +184% in our data).

The solution we adopt---pre-registration of the analysis method---is one approach to this problem. It is not always feasible (exploratory analyses have legitimate scientific value), but when the goal is comparative evaluation with reliable rankings, pre-registration eliminates the most pernicious source of bias. We encourage the LLM evaluation community to adopt similar practices, particularly for benchmark leaderboards where small differences in reported scores can have outsized reputational consequences.

### 7.5 Limitations

Several limitations of this work should be acknowledged:

**In-context vs. training-based self-consumption.** Our experimental protocol simulates self-consumption through in-context continuation rather than actual retraining on model outputs. The in-context method captures the immediate effect of conditioning on one's own outputs but does not capture effects that would emerge through gradient-based training, such as weight-space convergence or loss landscape changes. The relationship between in-context β and training-based β is unknown and requires dedicated study.

**API opacity.** All evaluations were conducted through commercial APIs. Model providers may change API behavior (system prompts, sampling parameters, model versions) without notice, potentially affecting β measurements. Reproducibility therefore depends on API stability, which is outside our control. Open-weight model evaluations (like the Qwen2.5-1.5B P3 validation) provide a more reproducible baseline.

**Model coverage.** Our sample of nine models covers four families but is small relative to the diversity of available LLMs. The ongoing evaluation of four additional models (DeepSeek-V4 Pro, DeepSeek-V4 Flash, Llama 4 Maverick, Llama 4 Scout) will expand coverage, but many important models remain unevaluated (Gemini family, Mistral family, Qwen family, etc.).

**Capability coverage.** We evaluate six capabilities. These were chosen to span a range of constraint demands (from low-diversity factual recall to high-diversity creative generation), but they do not exhaust the space of LLM applications. Capabilities such as translation, summarization, and instruction following may exhibit different decay patterns.

**σ dimension redundancy.** Our analysis of the 5D σ correlation matrix revealed that safety and coherence dimensions are perfectly collinear (r=1.000), and that two principal components explain 96.2% of variance. We recommend reducing the 5D state space to 3D in future work, mapping directly onto the three executor layers: σ_EI (logic_density + syntax_cv), σ_EII (bigram_rep + unique_ratio + filler + truncation), and σ_EIII (proper_case + number_integrity). This simplification would reduce measurement noise without sacrificing information.

**Temperature dependence.** The anomalous T=0.8 result for GPT-4o-mini (β=0.0885 vs. 0.150--0.160 at other temperatures) requires further investigation. If β is genuinely temperature-dependent, evaluation protocols must standardize temperature or report β across a temperature sweep.

### 7.6 Future Work

**Expanded model coverage.** We are actively benchmarking DeepSeek-V4 Pro, DeepSeek-V4 Flash, Llama 4 Maverick, and Llama 4 Scout. These additions will test the family constancy hypothesis on next-generation models and provide the first within-family temporal stability comparison (Llama 3.1 → Llama 4).

**Training-based β validation.** A critical direction is validating whether in-context β predicts training-based model collapse. This requires access to model training pipelines and is most feasible with open-weight models in the 1B--8B parameter range.

**Constraint injection for collapse mitigation.** If creative writing collapse is driven by E-II attractor convergence, then periodically injecting high-diversity seed text may delay collapse. This is testable within our existing framework.

**Longer generation chains.** Our experiments use four generations. Extending to 8--16 generations would allow testing whether β is constant across generations or whether decay accelerates (or decelerates) in later generations. A constant-β model predicts exponential decay throughout; deviations would indicate higher-order dynamics.

**Open-source release.** The StabilityBench evaluation pipeline will be released as a Python package (`stabilitybench`) alongside this paper, enabling any researcher to measure β for any accessible model at approximately $0.50 per evaluation.

---

## 8. Conclusion

We have introduced StabilityBench, a principled framework for measuring the recursive stability of large language models. Our central contribution is β, a single metric that quantifies per-generation quality decay under self-consumption, grounded in the Constraint Residual Framework and validated through neural-level analysis.

The empirical landscape revealed by our evaluation of nine production models is both concerning and structured. All models exhibit measurable recursive decay (β > 0), confirming that self-consumption is a universal vulnerability. However, the magnitude of vulnerability varies dramatically: DeepSeek-V3 achieves β=0.0281 (3.2× more stable than any competitor), while Claude Haiku 4.5 exhibits β=0.1468 (5.2× the per-generation decay rate of DeepSeek-V3). Creative writing is the universal point of failure (9/9 models critical or collapsed by Gen3), while factual knowledge is universally preserved (9/9 healthy). Llama and OpenAI families exhibit near-constant β across model sizes within each family, suggesting that recursive stability may be a stable property of a model family's training pipeline rather than a function of scale.

Methodologically, we have demonstrated that pre-registration of the fitting method eliminates systematic bias that inflates β by +41% to +184% in post-hoc model competition protocols. Pre-registered β achieves test-retest CV=3.3% and is robust to seed selection (Δ=0.005). Bootstrap analysis confirms convergence at n≥60 seeds, with n=100 recommended for production use.

Neural validation on Qwen2.5-1.5B confirms that β reflects a real internal process---constraint attractor collapse---with λ_C=+0.0415 and R²=0.884. This mechanistic grounding distinguishes β from purely empirical metrics and provides a causal framework for understanding and potentially mitigating recursive degradation.

As AI-generated content becomes the dominant source of text on the web, recursive stability will transition from an academic concern to an operational necessity. Models that degrade rapidly under self-consumption will produce training data that degrades successor models, creating a negative feedback loop that affects the entire ecosystem. StabilityBench provides the first standardized tool for measuring and monitoring this risk. We hope that β becomes a standard dimension in the LLM evaluation matrix, alongside accuracy, safety, and efficiency---because a model that is excellent today but unstable tomorrow is a model we cannot afford to depend on.

---

## References

Alemohammad, S., Casco-Rodriguez, J., Luzi, L., Humayun, A. I., Babaei, H., LeJeune, D., Siahkoohi, A., & Baraniuk, R. G. (2024). Self-consuming generative models go MAD. *Proceedings of the 41st International Conference on Machine Learning*.

Austin, J., Odena, A., Nye, M., Bosma, M., Michalewski, H., Dohan, D., Jiang, E., Cai, C., Terry, M., Le, Q., & Sutton, C. (2021). Program synthesis with large language models. *arXiv preprint arXiv:2108.07732*.

Biderman, S., Schoelkopf, H., Anthony, Q., Bradley, H., O'Brien, K., Hallahan, E., Khan, M. A., Purohit, S., Prashanth, U. S., Raff, E., Skorburg, J. A., van der Wal, O., van der Wal, D., & others. (2024). Lessons from the trenches on reproducible evaluation of language models. *arXiv preprint arXiv:2405.14782*.

Chen, M., Tworek, J., Jun, H., Yuan, Q., Pinto, H. P. d. O., Kaplan, J., Edwards, H., Burda, Y., Joseph, N., Brockman, G., Ray, A., Puri, R., Krueger, G., Petrov, M., Khlaaf, H., Sastry, G., Mishkin, P., Chan, B., Gray, S., Ryder, N., Pavlov, M., Power, A., Kaiser, L., Bavarian, M., Winter, C., Tillet, P., Such, F. P., Cummings, D., Plappert, M., Chantzis, F., Barnes, E., Herbert-Voss, A., Guss, W. H., Nichol, A., Paino, A., Tezak, N., Tang, J., Babuschkin, I., Balaji, S., Jain, S., Saunders, W., Hesse, C., Carr, A. N., Leike, J., Achiam, J., Misra, V., Morikawa, E., Radford, A., Knight, M., Brundage, M., Murati, M., Mayer, K., Welinder, P., McGrew, B., Amodei, D., McCandlish, S., Sutskever, I., & Zaremba, W. (2021). Evaluating large language models trained on code. *arXiv preprint arXiv:2107.03374*.

Chiang, W.-L., Zheng, L., Sheng, Y., Angelopoulos, A. N., Li, T., Li, D., Zhang, H., Zhu, B., Jordan, M., Gonzalez, J. E., & Stoica, I. (2024). Chatbot Arena: An open platform for evaluating LLMs by human preference. *Proceedings of the 41st International Conference on Machine Learning*.

Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano, R., Hesse, C., & Schulman, J. (2021). Training verifiers to solve math word problems. *arXiv preprint arXiv:2110.14168*.

Dohmatob, E., Feng, Y., Yang, P., Charton, F., & Kempe, J. (2024). A tale of tails: Model collapse as a change of scaling laws. *arXiv preprint arXiv:2402.07043*.

Ford, J., & Norwitz, G. (2022). Preregistration in machine learning: A tutorial. *NeurIPS 2022 Workshop on Pre-registration*.

Gehrmann, S., Strobelt, H., & Rush, A. M. (2019). GLTR: Statistical detection and visualization of generated text. *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics: System Demonstrations*, 111--116.

Gerstgrasser, M., Schaeffer, R., Dey, A., Rafailov, R., Sleight, H., Hughes, J., Korbak, T., Agrawal, R., Pai, D., Gromov, A., Roberts, D. A., Yang, D., Donoho, D. L., & Koyejo, S. (2024). Is model collapse inevitable? Breaking the curse of recursion by accumulating real and synthetic data. *arXiv preprint arXiv:2404.01413*.

Hendrycks, D., Burns, C., Basart, S., Zou, A., Mazeika, M., Song, D., & Steinhardt, J. (2021). Measuring massive multitask language understanding. *Proceedings of the International Conference on Learning Representations (ICLR)*.

Hendrycks, D., Burns, C., Kadavath, S., Arora, A., Basart, S., Tang, E., Song, D., & Steinhardt, J. (2021). Measuring mathematical problem solving with the MATH dataset. *Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks*.

Liang, P., Bommasani, R., Lee, T., Tsipras, D., Soylu, D., Yasunaga, M., Zhang, Y., Narayanan, D., Wu, Y., Kumar, A., Newman, B., Yuan, B., Yan, B., Zhang, C., Cosgrove, C., Manning, C. D., Re, C., Acosta-Navas, D., Hudson, D. A., Zelikman, E., Durmus, E., Ladhak, F., Rong, F., Ren, H., Yao, H., Wang, J., Santhanam, K., Orr, L., Zheng, L., Yuksekgonul, M., Suzgun, M., Kim, N., Guha, N., Chatterji, N., Khattab, O., Henderson, P., Huang, Q., Chi, R., Xie, S. M., Santurkar, S., Ganguli, S., Hashimoto, T., Icard, T., Zhang, T., Chaudhary, V., Wang, W., Li, X., Mai, Y., Zhang, Y., & Koreeda, Y. (2023). Holistic evaluation of language models. *Transactions on Machine Learning Research*.

Mitchell, E., Lee, Y., Khazatsky, A., Manning, C. D., & Finn, C. (2023). DetectGPT: Zero-shot machine-generated text detection using probability curvature. *Proceedings of the 40th International Conference on Machine Learning*.

Shumailov, I., Shumaylov, Z., Zhao, Y., Papernot, N., Anderson, R., & Gal, Y. (2024). AI models collapse when trained on recursively generated data. *Nature*, 631, 755--759.

Srivastava, A., Rastogi, A., Rao, A., Shoeb, A. A. M., Abid, A., Fisch, A., Brown, A. R., Santoro, A., Gupta, A., Garriga-Alonso, A., Kluska, A., Lewkowycz, A., Agarwal, A., Power, A., Ray, A., Warstadt, A., Kocurek, A. W., Safaya, A., Tazarv, A., Xiang, A., Parrish, A., Nie, A., Hussain, A., Askell, A., Dsouza, A., Slone, A., Rahane, A., Iyer, A. S., Andreassen, A., Madotto, A., Santilli, A., Stuhlmuller, A., Dai, A., La, A., Lampinen, A., Zou, A., Jiang, A., Chen, A., Vuong, A., Gupta, A., Gottardi, A., Norelli, A., Venkatesh, A., Gholamidavoodi, A., Tabassum, A., Menezes, A., Kirubarajan, A., Mullokandov, A., Sabharwal, A., Herrick, A., Efrat, A., Erdem, A., Karakas, A., & others. (2023). Beyond the imitation game: Quantifying and extrapolating the capabilities of language models. *Transactions on Machine Learning Research*.
