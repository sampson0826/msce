# StabilityBench Human Evaluation Experiment: Design Document

**Status:** Pre-registration draft
**Date:** 2026-05-16
**Principal Investigator:** Deng Xinhang
**Paper DOI:** 10.5281/zenodo.20041757

---

## 1. Motivation and Research Question

### 1.1 The Gap

StabilityBench measures recursive stability (beta) of LLMs via deterministic, rule-based feature extraction -- no neural judges, no human annotators. The paper's central finding is that **creative_writing is the first and most universal constraint to collapse under recursive self-consumption**: 9/9 production models reach critical or collapsed status by Gen3. However, this finding relies entirely on the constraint residual measurement pipeline. A reviewer will ask:

> "Do humans actually perceive the degradation that beta detects? If beta says creative_writing quality drops 55% by Gen1 for GPT-4o-mini, can human raters tell the difference between Gen0 and Gen3 outputs?"

This experiment is designed to answer that question rigorously.

### 1.2 Research Questions

1. **RQ1 (Detection):** Do human raters perceive a statistically significant difference in literary quality between Gen0 (fresh) and Gen3 (after recursive self-consumption) creative writing outputs?
2. **RQ2 (Gradation):** Does the magnitude of human-perceived degradation correlate with the creative_writing beta measured by StabilityBench? If beta predicts human perception, the metric is validated. If not, the practical significance of the bottleneck finding is questioned.
3. **RQ3 (Threshold):** At what beta value does degradation become perceptible to humans? Is there a "just noticeable difference" (JND) threshold?

---

## 2. Experiment Structure

### 2.1 Model Selection

Three models spanning the full creative_writing beta range, selected from the StabilityBench dataset:

| Model | creative_writing beta | Gen3 S_3 | Degradation Expectation |
|-------|----------------------|----------|------------------------|
| GPT-5.5 | 0.060 | 0.830 (healthy) | Minimal -- near human baseline |
| GPT-4o-mini | 0.224 | 0.468 (critical) | Moderate -- clearly detectable |
| Claude Opus 4.7 | 0.338 | 0.290 (collapsed) | Severe -- should be obvious |

**Rationale:** GPT-5.5 serves as a near-ceiling control (beta = 0.060 means only ~6% constraint loss per generation). If raters cannot distinguish Gen0 from Gen3 for GPT-5.5 but can for GPT-4o-mini and Claude Opus 4.7, this provides graded evidence that beta tracks perceptible degradation. Claude Opus 4.7 (beta = 0.338, collapsed by Gen3) serves as the positive control -- if raters cannot detect degradation here, the metric has no practical validity.

### 2.2 Stimuli

**Source:** Creative writing outputs from the StabilityBench n=100 lineage dataset, specifically:
- `/experiment_data/latest_models/gpt-5.5_s100_lineage.jsonl`
- `/experiment_data/n100/gpt-4o-mini_s100_lineage.jsonl`
- `/experiment_data/latest_models/claude-opus-4-7_s100_lineage.jsonl`

**Selection criteria for the 5 seeds per model:**
1. Both Gen0 (lineage generation=1) and Gen2 (lineage generation=3) outputs exist and are non-empty (filter out `[EMPTY]` placeholder entries)
2. Gen0 text length between 200 and 3000 characters (exclude trivial or truncated outputs)
3. No NSFW or harmful content (manual screening)
4. Seeds drawn from different creative writing prompt templates to maximize diversity

**Final stimulus set:**
- 3 models x 2 generations (Gen0, Gen2) x 5 seeds = **30 individual texts**
- These form **15 text pairs** per model (5 seeds, each seed producing one Gen0-Gen2 pair)
- Total: **45 text pairs** across all models

### 2.3 Task Design

The experiment has two blocks:

#### Block A: Side-by-Side Forced Choice (Primary)

**Instructions to raters:**
> You will see two creative writing passages side by side. Both were written in response to the same prompt. Please read both carefully and select which one has higher literary quality. Consider: originality, coherence, imagery, emotional resonance, and overall artistic merit. If you cannot decide, you must still make a choice (forced choice).

**Presentation:**
- Two texts displayed side by side (left panel / right panel)
- Left/right position of Gen0 vs Gen2 randomized per trial (counterbalanced)
- Prompts presented above the two texts for context
- Rater selects "Text A is better" or "Text B is better"
- No "tie" option (forced choice)

**Trials per rater:** 15 trials (5 seeds x 3 models, one pair per trial)
- Each trial compares Gen0 vs Gen2 for one seed of one model
- Order: trials presented in random order across models and seeds

#### Block B: Absolute Quality Rating (Secondary)

**Instructions:**
> For each passage below, rate its literary quality on a scale of 1 (very poor) to 10 (excellent). Consider the same criteria as before: originality, coherence, imagery, emotional resonance, and overall artistic merit.

**Presentation:**
- Single text displayed per screen
- 1-10 Likert scale below
- **Subset:** 2 models x 2 generations x 3 seeds = 12 texts total
  - Models: GPT-4o-mini and Claude Opus 4.7 (skip GPT-5.5 to reduce rater fatigue; its near-zero degradation makes absolute ratings unlikely to yield signal)
  - This yields within-rater Gen0 vs Gen2 comparisons on an absolute scale

**Total trials per rater:** 15 forced-choice + 12 absolute rating = 27 judgments
**Estimated time:** 12-18 minutes

### 2.4 Attention and Quality Checks

1. **Embedded attention check (1 trial):** One trial presents two identical texts (duplicate of a Gen0 output). Raters who claim to detect a difference fail the attention check.
2. **Minimum reading time:** Trials with < 5 seconds response time flagged for exclusion (insufficient reading time for texts averaging 500-2000 words)
3. **Post-survey self-report:** "Did you read all texts carefully? (Yes / Mostly / Skimmed / No)" -- exclude raters who answer "Skimmed" or "No"
4. **Consistency check:** For Block B, compute within-rater Gen0 - Gen2 difference. Raters whose ratings show no variance (all 10s or all 1s) across all 12 texts are flagged.

### 2.5 Pilot Testing

Before full launch, run a **pilot with n=5 raters** to:
1. Verify instructions are clear and unambiguous
2. Check that Gen0 and Gen2 texts render correctly in the survey platform
3. Measure actual completion time variance
4. Confirm attention check correctly catches inattentive responses
5. Refine text selection if any pair shows ceiling/floor effects (all raters agree, no variance)

---

## 3. Rater Requirements

### 3.1 Sample Size and Power Analysis

**Primary outcome measure:** Proportion of trials where Gen0 is preferred over Gen2.

**Effect size estimation:** Based on the paper's results:
- GPT-5.5 (beta=0.060): Expected Gen0 preference rate ~0.52-0.55 (weak effect, near chance)
- GPT-4o-mini (beta=0.224): Expected Gen0 preference rate ~0.70-0.80 (moderate-to-strong effect)
- Claude Opus 4.7 (beta=0.338): Expected Gen0 preference rate ~0.85-0.95 (strong effect)

**Power analysis for the weakest expected effect (GPT-4o-mini, p=0.75 vs null=0.50):**

Using a one-sample exact binomial test against H0: p = 0.50, with alpha = 0.05 (two-sided), power = 0.80:

For each model, we aggregate across raters and seeds: n = raters x seeds. With seeds as the unit of analysis (5 seeds per model), each rater provides 5 binary judgments per model. Aggregating across raters gives the per-model sample as n_raters x 5 trials.

Simpler approach: treat each rater as providing a "Gen0 preference score" (proportion of trials where they chose Gen0, range 0-1). This continuous measure supports a one-sample t-test against mu = 0.50.

For a one-sample t-test:
- H1: mu = 0.75, H0: mu = 0.50, d = (0.75 - 0.50) / sigma
- If sigma ~ 0.25 (conservative estimate for proportion data with 5 trials per model), d = 1.0
- Required n per model: n = (Z_alpha/2 + Z_beta)^2 / d^2 = (1.96 + 0.84)^2 / 1.0^2 = 7.84 -> n >= 8

For the weakest model (GPT-5.5, expected p=0.55, sigma~0.30, d=0.17):
- Required n = (1.96 + 0.84)^2 / 0.17^2 = 277 -- prohibitive

**Conclusion:** With n=20 raters, we have 80% power to detect d >= 0.65 effect sizes. This is sufficient for GPT-4o-mini and Claude Opus 4.7 (expected d > 1.0), but underpowered for GPT-5.5 (expected d < 0.2). GPT-5.5 serves as a control: failure to detect degradation there is expected and non-problematic.

**ICC target:** Intraclass correlation coefficient (ICC) > 0.70 for two-way mixed-effects model (ICC(3,k) for consistency across fixed raters). Below 0.70 indicates raters are not applying consistent criteria.

### 3.2 Eligibility Criteria

1. **English fluency:** Self-reported as "Fluent" or "Native" on a 5-point scale (Basic / Intermediate / Proficient / Fluent / Native). Exclude raters who select Basic or Intermediate.
2. **Age:** 18+ years
3. **No prior exposure to StabilityBench outputs:** Screen with "Have you previously seen or evaluated AI-generated creative writing from this study?" -- exclude if "Yes"
4. **Device:** Desktop or tablet (not mobile phone) for adequate text display
5. **No professional literary background required:** We want everyday readers, not literary critics, to match the general audience for LLM outputs

### 3.3 Recruitment

| Channel | Target n | Compensation | Notes |
|---------|----------|-------------|-------|
| University students (Beijing) | 10-15 | 30 RMB (~$4) / 15 min | WeChat group recruitment, bilingual instructions available |
| Prolific Academic | 10-15 | $4.50 / 15 min ($18/hr rate) | English-native pool, filter by 95%+ approval rate |

**Total target:** n >= 20 valid raters after exclusions (recruit 25-30 to allow for 20% exclusion rate)

**Prolific prescreening filters:**
- First language: English
- Approval rate: >= 95%
- Minimum submissions: >= 100
- Exclude: participants from prior AI text evaluation studies (Prolific custom allowlist)

---

## 4. Statistical Analysis Plan

### 4.1 Pre-Registered Hypotheses

These hypotheses are registered before any data collection begins.

**H1 (Detection -- GPT-4o-mini):**
- H0: The proportion of trials where raters prefer Gen0 over Gen2 for GPT-4o-mini creative_writing outputs is 0.50 (chance).
- H1: The proportion is significantly greater than 0.50.
- Test: One-sample t-test on per-rater Gen0 preference proportion for GPT-4o-mini (5 trials per rater). Alpha = 0.05, one-sided.

**H2 (Detection -- Claude Opus 4.7):**
- H0: Proportion Gen0 preference = 0.50.
- H1: Proportion > 0.50.
- Test: Same as H1. Alpha = 0.05, one-sided.

**H3 (Detection -- GPT-5.5):**
- H0: Proportion Gen0 preference = 0.50.
- H1: Proportion != 0.50.
- Test: Same as H1 but two-sided (exploratory -- we do not predict a strong effect). Alpha = 0.05.

**H4 (Gradation -- beta correlation):**
- H0: No positive correlation between model-level creative_writing beta and model-level human-perceived degradation.
- H1: There is a significant positive correlation.
- Test: Compute per-model mean Gen0 preference proportion across all raters and all seeds. Correlate the 3 model-level values (beta vs. mean Gen0 preference) using Kendall's tau (appropriate for n=3 with monotonic prediction). Alpha = 0.05, one-sided.
- **Acknowledged limitation:** n=3 models gives very low statistical power for this correlation. This is a descriptive/pattern-level test, not a definitive validation. Strong evidence requires future experiments with 6+ models.

**H5 (Absolute rating -- within-rater Gen0 > Gen2):**
- H0: Mean absolute quality rating for Gen0 equals mean rating for Gen2 (for GPT-4o-mini and Claude Opus 4.7).
- H1: Mean Gen0 rating > mean Gen2 rating.
- Test: Paired t-test (or Wilcoxon signed-rank if non-normal) on per-rater mean Gen0 and mean Gen2 ratings. Alpha = 0.05, one-sided. Separate tests for each model.

### 4.2 Secondary / Exploratory Analyses

1. **Inter-rater reliability:** ICC(3,k) for forced-choice judgments. Compute across all 15 trials. Target > 0.70.
2. **Seed-level variance:** Do some creative_writing seeds show larger perceived degradation than others? Compute per-seed Gen0 preference proportion and test for heterogeneity (Cochran's Q).
3. **Rater-level covariates:** Does English fluency level, age, or recruitment channel predict Gen0 preference rates? Exploratory linear mixed-effects model.
4. **Response time analysis:** Does longer reading time predict more accurate detection of Gen0 > Gen2 differences?
5. **Attention check failure rate:** What proportion of raters fail the duplicate-text attention check? This provides a baseline for random responding.
6. **Block A vs Block B convergence:** Do forced-choice preferences correlate with absolute rating differences at the rater level?

### 4.3 Multiple Comparison Correction

- Primary hypotheses H1-H3: Bonferroni correction across the 3 models (adjusted alpha = 0.05/3 = 0.0167)
- H4 and H5 are analyzed separately as they address distinct questions (gradation and absolute rating validation)

### 4.4 Handling of [EMPTY] and Degenerate Outputs

GPT-5.5's creative_writing lineage shows approximately 75% [EMPTY] outputs at Gen1 (12/16 empty). These represent API generation failures where the model refused to produce creative writing content. For seed selection:

1. **Filter seeds:** Only include seeds where BOTH Gen0 and Gen2 have non-empty, substantive outputs (>= 200 characters)
2. **Document attrition:** Report how many of the original 16 seeds per model survived filtering
3. **Sensitivity analysis:** If attrition is severe (> 50%), consider augmenting with additional seed runs

### 4.5 Interpretation Framework: Outcome Mapping

| Outcome Pattern | Interpretation for StabilityBench |
|-----------------|-----------------------------------|
| H1 and H2 significant (Gen0 preferred for GPT-4o-mini and Claude Opus 4.7); H3 not significant (GPT-5.5 near chance) | **Full validation.** Beta correctly predicts which models show perceptible degradation and which do not. |
| All H1-H3 significant (including GPT-5.5) | **Partial validation, high sensitivity.** Humans are more sensitive than beta suggests. The metric's threshold mapping may need recalibration but its ordinal ranking is preserved. |
| H2 significant but H1 not (Claude degrades perceptibly but GPT-4o-mini does not) | **Partial validation, compressed range.** Beta overestimates perceptible differences in the mid-range. The metric may have a nonlinear mapping to human perception. |
| None of H1-H3 significant | **Refutation.** Humans cannot reliably perceive the degradation that beta measures. The practical significance of the creative_writing bottleneck finding is questioned. The metric may be capturing real statistical changes that are below the human perceptual threshold. |
| H4 significant (positive beta-perception correlation) | **Metric validation.** Beta magnitude predicts human-perceived degradation magnitude. This is the strongest possible outcome and warrants emphasis in the paper. |
| H4 not significant | Expected given n=3 models. Does not invalidate beta if H1-H3 support ordinal detection. Future work with more models needed. |

---

## 5. Implementation Plan

### 5.1 Survey Platform

**Recommended: Custom web application** (preferred over Google Forms for counterbalancing and randomization requirements)

**Rationale against Google Forms:**
- Google Forms cannot randomize left/right text position per trial per rater
- Cannot enforce minimum reading time
- Cannot support complex counterbalancing schemes
- Limited support for long text display side-by-side

**Minimal viable implementation** using a single-page HTML/JS application:
- Backend: Static JSON file with all stimulus texts pre-baked; no server needed
- Frontend: HTML + vanilla JS or minimal framework
- Data storage: Write responses to a Google Sheet via Google Forms API, or use a simple Firebase/Flask backend to log responses
- Hosting: GitHub Pages (free) or Vercel

**Alternative (lower effort):** Qualtrics (if institutional license available) or Gorilla.sc (academic license, ~$50/month). Both support full randomization and timing.

### 5.2 Stimulus Preparation

**Step 1: Extract texts from lineage JSONL files**
```python
# For each target model, extract creative_writing Gen0 (generation=1) 
# and Gen2 (generation=3) texts for all seeds
# Filter: text length >= 200 chars, not [EMPTY]
```

**Step 2: Manual quality screening**
- Two independent screeners review all candidate texts
- Exclude: NSFW content, hate speech, personally identifiable information, texts that are clearly truncated mid-sentence
- Inter-screener agreement: Cohen's kappa > 0.80 target
- Disagreements resolved by third screener (PI)

**Step 3: Format for survey presentation**
- Clean whitespace, normalize paragraph breaks
- Font: serif (Georgia or similar) at 16px for readability
- Text panels: fixed width (600-700px), scrollable if needed
- Prompt displayed above in italic, clearly labeled "Prompt:"

**Step 4: Counterbalancing scheme**
For 3 models x 5 seeds = 15 pairs:
- Each pair has 2 possible left/right configurations (Gen0-left, Gen0-right)
- 15 pairs x 2 = 30 possible presentation orders
- Randomize pair order and left/right assignment independently per rater
- Seed: rater ID or session token ensures reproducibility

### 5.3 Rater Flow

```
1. Welcome screen + informed consent
2. Demographics + eligibility screening (1 min)
3. Instructions + example trial (2 min)
4. Block A: 15 forced-choice trials (8-10 min)
   - Trial structure: prompt display -> 2 texts -> choice -> next
   - Progress bar: "Trial X of 15"
5. Block B: 12 absolute rating trials (4-6 min)
   - Trial structure: single text -> 1-10 scale -> next
6. Attention check (embedded in Block A or B, randomized position)
7. Post-survey self-report (1 min)
8. Debriefing + compensation code
Total: ~15-20 min
```

### 5.4 Compensation

| Channel | Amount | Rationale |
|---------|--------|-----------|
| University students | 30 RMB | Above Beijing minimum hourly wage (~25 RMB/hr), appropriate for 15 min |
| Prolific | $4.50 | Prolific's recommended $18/hr rate for academic surveys |

Total budget: 25 raters x $4.50 average = **~$113** + survey platform costs (~$50 for one month of Gorilla or $0 for custom app)

### 5.5 Timeline

| Phase | Duration | Activities | Deliverable |
|-------|----------|-----------|-------------|
| **Week 1: Preparation** | 5 days | Extract + screen texts, build survey app, pilot test (n=5) | Working survey, screened stimulus set, pilot report |
| **Week 2: Recruitment** | 3 days | Launch on Prolific + WeChat, monitor completion rates | Raw data from first 15 raters |
| **Week 2-3: Data collection** | 7 days | Continue recruitment until n >= 20 valid after exclusions | Full raw dataset |
| **Week 3: Analysis** | 3 days | Run pre-registered analyses, compute effect sizes, draft results section | Analysis report, figures |
| **Week 3: Write-up** | 2 days | Integrate findings into paper Section X (Human Validation) | Revised paper draft |

**Total: ~3 weeks from launch to results integrated in paper.**

### 5.6 Data Management

- **Raw data storage:** CSV with columns: rater_id, trial_id, model, seed, generation_left, generation_right, chosen_side, response_time_ms, block (A/B), condition
- **Privacy:** No PII collected beyond Prolific ID or anonymous WeChat ID. Data stored locally, not in cloud.
- **Pre-registration:** Upload this design document + analysis script to OSF or AsPredicted before data collection begins. Timestamp establishes priority.
- **Data sharing:** De-identified raw data + analysis script included in paper's supplementary materials

---

## 6. Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| GPT-5.5 has too few non-[EMPTY] creative_writing outputs to select 5 seeds | Medium | High | Fall back to DeepSeek-V3 (beta_cw=0.164, S_3=C) as the "low degradation" model; document and justify the substitution |
| Gen0 and Gen2 texts are too similar for raters to distinguish | Medium | High | Pilot test will catch this; if true, beta's practical significance is genuinely limited -- report honestly |
| Raters consistently prefer Gen2 texts over Gen0 (reverse effect) | Low | High | This would be a fascinating finding: degradation may produce texts that some readers prefer. Report as a discovery, not a failure. |
| High exclusion rate (> 30%) reduces effective n below target | Medium | Medium | Over-recruit by 30%; if still below n=15, extend data collection by one week |
| Survey platform technical failure (randomization bug, data loss) | Low | High | Pilot test thoroughly; log all responses to multiple backends (local storage + server); store texts as static data to avoid server dependency |
| Rater fatigue degrades response quality in later trials | Medium | Low | Randomize trial order per rater; analyze response quality as function of trial position; 15 trials is below typical fatigue threshold |
| Prolific raters are non-native English speakers misrepresenting fluency | Low | Medium | Prolific's prescreening is reliable; add "explain a metaphor" open-response question as a covert English fluency check |

---

## 7. Reporting Standards

The experiment results section in the paper will include:

1. **CONSORT-style flow diagram:** Raters recruited -> screened -> excluded -> analyzed
2. **Rater demographics table:** Age (mean, SD, range), gender, English fluency distribution, recruitment channel
3. **Primary results table:**

| Model | creative_writing beta | Gen0 Pref. Proportion | 95% CI | t(df) | p (corrected) | Cohen's d |
|-------|----------------------|----------------------|--------|-------|---------------|-----------|
| GPT-5.5 | 0.060 | -- | -- | -- | -- | -- |
| GPT-4o-mini | 0.224 | -- | -- | -- | -- | -- |
| Claude Opus 4.7 | 0.338 | -- | -- | -- | -- | -- |

4. **Beta-perception scatter plot:** x = creative_writing beta, y = mean Gen0 preference proportion, with 95% CI error bars
5. **Inter-rater reliability:** ICC with interpretation
6. **Seed-level heterogeneity analysis:** Per-seed Gen0 preference proportions with error bars
7. **Absolute rating results:** Mean Gen0 and Gen2 ratings per model, paired test statistics
8. **Transparent reporting of all exclusions and deviations from pre-registered protocol**

---

## 8. Pre-Registration Checklist

- [x] Research questions stated
- [x] Model selection criteria and justification  
- [x] Seed selection criteria (objective, replicable)
- [x] Sample size justification with power analysis
- [x] All hypotheses stated in testable form with H0 and H1
- [x] Statistical tests specified (including one-sided vs two-sided)
- [x] Alpha levels and multiple comparison correction specified
- [x] Exclusion criteria for raters specified
- [x] Exclusion criteria for stimuli specified
- [x] Interpretation framework: what each outcome pattern means
- [x] Pilot testing plan
- [x] Data management and privacy plan
- [ ] Analysis script written and time-stamped (to be completed before data collection)
- [ ] OSF/AsPredicted registration (to be completed before data collection)

---

## Appendix A: Stimulus Selection Protocol (Pseudocode)

```python
import json, random

def select_stimuli(lineage_path, model_name, n_seeds=5):
    with open(lineage_path) as f:
        samples = [json.loads(line) for line in f]
    
    # Filter creative_writing only
    cw = [s for s in samples 
          if 'creative_writing' in s.get('capability_tags', [])]
    
    # Group by seed_id (extracted from id field)
    seeds = {}
    for s in cw:
        seed_id = s['id'].split('_')[-1]  # Extract numeric seed
        gen = s['generation']
        if seed_id not in seeds:
            seeds[seed_id] = {}
        seeds[seed_id][gen] = s
    
    # Find seeds with non-empty Gen0 (gen=1) and Gen2 (gen=3)
    valid = []
    for seed_id, gens in seeds.items():
        if 1 in gens and 3 in gens:
            t0 = gens[1]['text']
            t2 = gens[3]['text']
            if len(t0) >= 200 and len(t2) >= 200:
                if '[EMPTY]' not in t0 and '[EMPTY]' not in t2:
                    valid.append((seed_id, gens[0]['text'], t0, t2))
    
    # Manual screening step (human reviewer) goes here
    # ...
    
    # Randomly select n_seeds
    selected = random.sample(valid, min(n_seeds, len(valid)))
    return selected
```

## Appendix B: Statistical Power Simulation

```python
import numpy as np
from scipy import stats

def simulate_power(n_raters, true_p, n_sims=10000):
    """Simulate power for one-sample t-test on proportion data."""
    significant = 0
    for _ in range(n_sims):
        # Each rater gives a proportion of Gen0 preferences (5 trials)
        # Simulate as binomial then convert to proportion
        preferences = np.random.binomial(5, true_p, n_raters) / 5.0
        t, p = stats.ttest_1samp(preferences, 0.50)
        if p < 0.05:  # one-sided
            significant += 1
    return significant / n_sims

# Expected power at n=20
for model, true_p in [("GPT-5.5", 0.55), ("GPT-4o-mini", 0.75), ("Claude Opus 4.7", 0.88)]:
    power = simulate_power(20, true_p)
    print(f"{model}: power = {power:.3f} at n=20")
# Expected output:
# GPT-5.5: power ~ 0.12-0.20 (underpowered -- expected)
# GPT-4o-mini: power ~ 0.85-0.95 (well-powered)
# Claude Opus 4.7: power ~ 0.98-0.999 (well-powered)
```

## Appendix C: Example Creative Writing Prompts (from Seed Dataset)

The following are actual seed prompts from the StabilityBench creative_writing dataset:

1. "Write a short story about an astronomer who discovers that the stars are going out one by one."
2. "Compose a poem about the relationship between silence and understanding."
3. "Write a story from the perspective of a book that has been sitting unread on a library shelf for 200 years."
4. "Write a dialogue between a river and a mountain that have been neighbors for millions of years."
5. "Describe a color that doesn't exist, to someone who has been blind since birth but can perceive other senses acutely."
6. "Write a letter from a future version of yourself to your present self, giving one piece of advice without revealing any specific events."
7. "Imagine a world where emotions can be traded like currency. Write a narrative that explores the consequences."
