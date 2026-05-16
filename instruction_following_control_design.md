# Instruction-Following Control Experiment Design

## Motivation

Section 5.7 of the StabilityBench paper identifies a confound: under recursion, models may drift from the original prompt's topic or output format. This is **instruction decay** -- a failure to maintain adherence to the original instruction -- rather than **constraint-structure degradation** as measured by beta. The current hybrid constraint extractor would register both as increased constraint violation, but the root causes are fundamentally different:

- **Constraint decay**: The model's output loses factual accuracy, logical coherence, code correctness, or stylistic quality under recursion, even when it stays on-topic.
- **Instruction decay**: The model drifts to a different topic, changes output format, or ignores structural requirements from the original prompt, even if within the new topic its outputs remain high-quality.

Disentangling these requires measuring topic and format fidelity independently of constraint quality, then comparing the decay rates.

## Design Overview

Run recursive generation (Gen0 -> Gen1 -> Gen2 -> Gen3) as in the main RSI experiment, but add two orthogonal fidelity tracks that measure **what changed** rather than **how good it is**:

1. **Topic fidelity**: Cosine similarity of Gen_k text embedding to Gen0 prompt embedding (via sentence-transformers)
2. **Format fidelity**: Binary + categorical check of whether Gen_k follows the output format specified in the Gen0 prompt

Then fit exponential decay models to both fidelity tracks and compare the resulting beta_instruction against the constraint beta from the main experiment. If beta_instruction and beta_constraint are correlated, the paper's limitation is empirically significant. If they are uncorrelated (as embedding drift already is: rho = 0.100), instruction decay is an orthogonal axis of degradation.

## Models

3 models spanning the full beta range:

| Model | Provider | Constraint beta (K=3) | Rationale |
|-------|----------|----------------------|-----------|
| GPT-5.5 | quickrouter | 0.011 | Low beta: near-perfect stability |
| GPT-4o-mini | quickrouter | 0.088 | Mid beta: reference model |
| Claude Opus 4.7 | quickrouter | 0.164 | High beta: rapid degradation |

This spread covers 15x range in constraint beta, maximizing the chance to detect whether instruction beta tracks constraint beta or diverges.

## Seeds

n = 20 seeds per model (60 total runs), sampled from the existing GPT-4o-mini Gen0 lineage (`experiment_data/n100/gpt-4o-mini_s100_lineage.jsonl`). Stratified across 4 capability dimensions (5 seeds each) to ensure coverage:

- math_reasoning
- code_generation
- factual_knowledge
- creative_writing

These 4 dimensions are chosen because they span the full constraint hierarchy (L2 formal, E-III referential, E-II distributional) and because creative_writing is the dominant bottleneck -- topic drift here would be especially revealing.

Each seed is an existing Gen0 prompt from the GPT-4o-mini lineage. These prompts already have implicit output format expectations (math: step-by-step derivation; code: function with docstring; factual: paragraph form; creative_writing: narrative prose). The format fidelity check compares each generation against the *expected format* for that capability.

## Recursive Generation Protocol

For each model M and each seed prompt p_i:

```
Gen 0: p_i (original seed)
Gen 1: r_i^1 = M(p_i)
Gen 2: r_i^2 = M(r_i^1)
Gen 3: r_i^3 = M(r_i^2)
```

Same generation parameters as the main experiment:
- Temperature: 0.8
- Max tokens: 512
- API delay: 2.0s (QuickRouter)

Generation is run via the existing `create_provider("quickrouter", model=...)` infrastructure, reusing `provider_adapter.py`.

## Fidelity Metrics

### 1. Topic Fidelity (Continuous)

**Method**: Sentence embeddings via `all-MiniLM-L6-v2` (same embedder as the existing embedding drift baseline in `compute_embedding_drift.py`).

For each seed i at generation k:
```
topic_fidelity_i_k = cosine_similarity(embed(r_i^k), embed(p_i))
```

Aggregate across seeds per generation:
```
T_k = mean_i(topic_fidelity_i_k)
```

T_0 = 1.0 by construction (prompt is identical to itself).

**Expected behavior under instruction decay**: T_k monotonically decreases as models drift to adjacent or unrelated topics. A model that stays perfectly on-topic would maintain T_k ~ 1.0. A model that veers into a completely different domain would show T_k << 1.0.

**Exponential fit**: T_k = T_0 * exp(-beta_topic * k), fitted via OLS on log(T_k) for k = 0, 1, 2, 3. This produces **beta_topic**: the rate at which semantic focus on the original topic decays.

**Implementation note**: The existing `compute_embedding_drift.py` already computes this exact metric (named `avg_cos_sim_vs_gen0` there). The difference is that this experiment intentionally re-runs generation with 20 seeds to get fresh data, rather than reusing existing lineage data, and pairs it with format fidelity tracking.

### 2. Format Fidelity (Categorical)

**Method**: For each capability dimension, define a format rubric. Score each generation against the rubric using a structured checklist (automated where possible, LLM-judge where needed).

**Format rubrics by capability**:

| Capability | Expected Format | Automated Check | LLM-Judge Check |
|------------|----------------|-----------------|-----------------|
| math_reasoning | Step-by-step derivation with final answer | Presence of numbered steps or bullet points; final answer in \\boxed{} or "Answer:" | Steps are logically ordered; intermediate results are stated |
| code_generation | Function definition with docstring | Contains `def ` or `function `; contains docstring/comment block | Code is syntactically complete (not truncated mid-expression); follows requested language |
| factual_knowledge | Paragraph-form prose with specific claims | Text is prose (not bullet points or code); >50 words | Stays on the factual topic; makes specific claims rather than generic hedging |
| creative_writing | Narrative prose in requested genre | Text is prose narrative; >100 words; consistent tense | Maintains requested genre (sci-fi, mystery, etc.); has narrative structure (beginning, middle, end) |

**Scoring**:
- Each generation gets a binary format score: 1 = follows expected format, 0 = does not.
- A composite format fidelity score F_k = fraction of seeds at generation k that maintain format.
- F_0 = 1.0 (seeds are well-formed by construction).

**Expected behavior under instruction decay**: F_k drops as models:
- Switch from step-by-step math to narrative prose
- Drop docstrings or generate incomplete code
- Produce bullet-point lists instead of paragraph prose
- Abandon narrative structure for meta-commentary ("Here's a story about...")

**Exponential fit**: F_k = F_0 * exp(-beta_format * k). This produces **beta_format**: the rate at which output format adherence decays.

### 3. Composite Instruction Fidelity

Combine topic and format into a single instruction fidelity score:

```
I_k = T_k * F_k
```

Where T_k is topic fidelity (continuous, 0-1) and F_k is format fidelity (binary fraction, 0-1). Fit exponential: I_k = I_0 * exp(-beta_instruction * k).

This produces **beta_instruction**: the unified rate of instruction decay.

## Analysis Plan

### Primary Comparison

For each of the 3 models, compute:

| Coefficient | Meaning | Source |
|-------------|---------|--------|
| beta_constraint | Constraint structure decay | Main RSI experiment (existing data) |
| beta_topic | Semantic topic drift | This experiment |
| beta_format | Output format decay | This experiment |
| beta_instruction | Combined topic+format | This experiment (I_k = T_k * F_k) |

### Hypothesis Tests

**H1: beta_instruction is distinguishable from zero.**
- Null: beta_instruction = 0 (no instruction decay under recursion)
- Test: One-sample t-test against beta=0, using per-seed beta_instruction estimates via bootstrap (n=1000 resamples, 20 seeds)

**H2: beta_instruction is not fully redundant with beta_constraint.**
- Compute Spearman rho(beta_instruction, beta_constraint) across 3 models
- With only 3 models, rho can only be -1, -0.5, 0, 0.5, or 1. This is underpowered.
- Fallback: report the per-model ratio beta_instruction / beta_constraint as a descriptive statistic. If instruction decay accounts for, say, 20% of constraint decay in GPT-5.5 but 60% in Claude Opus 4.7, that reveals model-specific instruction-following fragility.

**H3: Topic and format fidelity decay at different rates.**
- Within each model, compare beta_topic vs beta_format
- If format decays faster than topic (beta_format > beta_topic), models maintain semantic coherence but lose structural discipline -- a distinct failure mode

### Power Analysis

With n=20 seeds per model:
- Within-model beta estimates: standard error of mean beta ~ sigma_beta / sqrt(20). If per-seed beta variance is 0.02 (conservative, based on existing cross-seed data), SE ~ 0.0045. This is sufficient to detect differences of ~0.01 between beta_topic and beta_format.
- Between-model comparison: 3 models is underpowered for correlation but adequate for descriptive characterization. The primary output is per-model ratios, not a formal correlation test.

### Controls

1. **Embedding model robustness**: Compute topic fidelity with both `all-MiniLM-L6-v2` (384-dim, lightweight) and `all-mpnet-base-v2` (768-dim, higher quality). Report both; use MiniLM for primary analysis to match the existing embedding drift baseline.
2. **Prompt length normalization**: Some Gen_k outputs are substantially shorter or longer than Gen0. Cosine similarity with MiniLM is somewhat length-sensitive. Report both raw cosine similarity and length-normalized (truncate to min(len(Gen0), len(Gen_k)) tokens).
3. **Empty response handling**: If a generation is empty or identical to the prompt (failed generation), exclude it from fidelity computation and flag the seed as degraded. If >20% of seeds are degraded for a model, that model's beta_instruction is marked as unreliable.

### Expected Data Volume

| Item | Count |
|------|-------|
| Seeds | 20 |
| Models | 3 |
| Generations per seed | 4 (Gen0, Gen1, Gen2, Gen3) |
| Total API calls | 20 * 3 * 3 = 180 |
| Embedding computations | 20 * 3 * 4 = 240 |
| Format rubric checks | 20 * 3 * 3 = 180 (Gen0 is reference) |

Estimated API cost: ~$2-4 total (180 QuickRouter calls at ~$0.01-0.02 per call for mid-tier models).

## Implementation Path

The experiment can be built by extending the existing `run_k5_experiment.py` pattern:

1. **Generation**: Reuse `create_provider()` and the recursive generation loop from `run_k5_experiment.py`. Models configured as:
   ```python
   {"model": "gpt-5.5", "provider": "quickrouter", "name": "gpt-5.5_instr_ctrl", "temp": 0.8, "delay": 2.0},
   {"model": "gpt-4o-mini", "provider": "quickrouter", "name": "gpt-4o-mini_instr_ctrl", "temp": 0.8, "delay": 2.0},
   {"model": "claude-opus-4-7", "provider": "quickrouter", "name": "claude-opus-4-7_instr_ctrl", "temp": 0.8, "delay": 2.0},
   ```

2. **Topic fidelity**: Reuse `SentenceTransformer("all-MiniLM-L6-v2")` from `compute_embedding_drift.py`. Compute and store per-seed, per-generation cosine similarities.

3. **Format fidelity**: New module. Automated checks via regex/rules (presence of `def`, `\boxed{}`, paragraph length thresholds). LLM-judge checks via a lightweight prompt to GPT-4o-mini for format classification.

4. **Analysis**: Fit exponential decay to T_k and F_k trajectories per model. Compute beta_topic, beta_format, beta_instruction. Compare against existing constraint beta values.

5. **Output**: JSON report at `experiment_data/instruction_following_control/results.json` with per-model fidelity trajectories, fitted betas, and comparison statistics.

## Limitations

1. **3 models, 20 seeds**: Underpowered for formal correlation testing. This is a characterization experiment, not a hypothesis test. Results should be reported as descriptive with confidence intervals, not p-values.
2. **Format rubric subjectivity**: The LLM-judge component of format checking introduces judge-model bias. Using GPT-4o-mini as the format judge means format fidelity for GPT-4o-mini's own outputs is self-judged. Cross-model judging (e.g., DeepSeek-V3 as format judge) would be a better design but adds complexity.
3. **Topic fidelity vs. quality**: High cosine similarity doesn't guarantee topic maintenance -- a model could produce low-quality but semantically similar text. The embedder measures semantic neighborhood, not correctness. This is by design (we're measuring fidelity, not quality), but the distinction matters for interpretation.
4. **Single seed source**: All seeds from GPT-4o-mini lineage. If instruction decay is seed-source-dependent (as constraint beta is, per S5.7), results may not generalize across seed distributions.
5. **The interaction term**: topic fidelity and format fidelity are multiplied in the composite I_k. This assumes independence, but format collapse may accelerate topic drift (or vice versa). Report T_k and F_k separately alongside I_k to allow inspection of interaction effects.
