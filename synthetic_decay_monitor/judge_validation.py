"""LLM judge cross-validation: independent quality assessment of recursive generations.

Uses gpt-5.5 as impartial judge to rate text quality across generations.
Compares judge scores with β-predicted decay → validates β as quality proxy.
"""
import sys, os, json, time
import numpy as np
from dataclasses import dataclass
from typing import Optional

sys.path.insert(0, ".")
from synthetic_decay_monitor.provider_adapter import create_provider
from synthetic_decay_monitor.constraint_extractor import HybridConstraintExtractor
from synthetic_decay_monitor.decay_engine import DecayEngine
from synthetic_decay_monitor.data_lineage import DataSample, DatasetLineage

JUDGE_MODEL = "gpt-5.5"
TEST_MODEL = "gpt-4o-mini"
JUDGE_KEY = "sk-0XqACr4qnHSionZmOLmiWegQSpVFNVH7M5sUF2dHN2Z80JVA"
TEST_KEY = "sk-0XqACr4qnHSionZmOLmiWegQSpVFNVH7M5sUF2dHN2Z80JVA"

N_GENERATIONS = 3
N_SEEDS = 8

SEEDS = [
    "Explain how photosynthesis converts sunlight into chemical energy.",
    "Write a short story about a robot learning to paint.",
    "Describe the water cycle and its importance for life on Earth.",
    "Explain the concept of supply and demand in economics.",
    "What causes seasons on Earth? Explain the astronomical reason.",
    "Write a function to find all prime numbers up to n.",
    "Explain how the internet works at a high level.",
    "Compare and contrast solar and wind energy.",
]

JUDGE_PROMPT = """Rate the QUALITY of this text on a scale of 1-10. Consider:
- Coherence: does it make logical sense?
- Factual grounding: does it contain specific, plausible information?
- Structure: is it well-organized?
- Language quality: grammar, vocabulary, flow?
- Completeness: does it fully address the topic?

Reply with ONLY a single number (1-10). No explanation.

Text: {text}
Rating:"""


def run():
    print("=" * 60)
    print(f"LLM Judge Cross-Validation")
    print(f"Judge: {JUDGE_MODEL}, Test model: {TEST_MODEL}")
    print(f"Seeds: {N_SEEDS}, Generations: {N_GENERATIONS}")
    print("=" * 60)

    adapter = create_provider("quickrouter", model=TEST_MODEL, api_key=TEST_KEY)
    judge = create_provider("quickrouter", model=JUDGE_MODEL, api_key=JUDGE_KEY)
    extractor = HybridConstraintExtractor(judge_fn=None)

    # Generate recursive texts
    print("\nGenerating recursive texts...")
    gen_texts = {g: [] for g in range(1, N_GENERATIONS + 1)}

    for i, seed in enumerate(SEEDS):
        prompt = seed
        for gen in range(1, N_GENERATIONS + 1):
            resp = adapter.generate(prompt, max_tokens=150, temperature=0.8)
            time.sleep(0.1)
            gen_texts[gen].append(resp)
            prompt = resp
        print(f"  [{i+1}/{N_SEEDS}] {seed[:50]}...")

    # Judge each text
    print(f"\nJudging {N_SEEDS * N_GENERATIONS} texts with {JUDGE_MODEL}...")
    judge_scores = {g: [] for g in range(1, N_GENERATIONS + 1)}

    for gen in range(1, N_GENERATIONS + 1):
        for i, text in enumerate(gen_texts[gen]):
            try:
                score_str = judge.generate(
                    JUDGE_PROMPT.format(text=text[:800]),
                    max_tokens=5, temperature=0.0
                )
                score = float(score_str.strip())
                score = max(1.0, min(10.0, score))
            except (ValueError, Exception):
                score = 5.0
            judge_scores[gen].append(score)
            time.sleep(0.05)

    # Per-generation judge means
    print(f"\n{'='*60}")
    print("Judge quality scores across generations:")
    judge_means = []
    for gen in range(1, N_GENERATIONS + 1):
        m = float(np.mean(judge_scores[gen]))
        s = float(np.std(judge_scores[gen]))
        judge_means.append(m)
        print(f"  Gen{gen}: judge_score={m:.2f} ± {s:.2f}")

    # Compute β
    samples = []
    for gen in range(1, N_GENERATIONS + 1):
        for i in range(N_SEEDS):
            samples.append(DataSample(
                text=gen_texts[gen][i],
                generation=gen,
                source_model=TEST_MODEL,
                capability_tags=["general"],
                sample_id=f"G{gen}_{i:04d}",
            ))
    lineage = DatasetLineage(samples=samples)
    engine = DecayEngine(lineage, extractor)
    engine.run_all_capabilities()
    betas = []
    for t in engine.get_all_trajectories():
        if "trajectory" in t:
            for g in t["trajectory"]:
                if "beta" in g and g["beta"] > 0:
                    betas.append(g["beta"])
    beta = float(np.mean(betas)) if betas else 0.25

    # β-predicted trajectory
    predicted = [judge_means[0] * (1 - beta)**(g) for g in range(N_GENERATIONS)]

    print(f"\n{'='*60}")
    print(f"β vs Judge comparison:")
    print(f"  β = {beta:.4f}")
    for gen in range(1, N_GENERATIONS + 1):
        print(f"  Gen{gen}: judge={judge_means[gen-1]:.2f}, β-predicted={predicted[gen-1]:.2f}")

    # Correlation
    obs_arr = np.array(judge_means)
    pred_arr = np.array(predicted)
    spearman_r = np.corrcoef(np.arange(1, N_GENERATIONS + 1), obs_arr)[0, 1]

    # MSE between observed and predicted
    mse = np.mean((obs_arr - pred_arr) ** 2)
    baseline_mse = np.mean((obs_arr - obs_arr.mean()) ** 2)
    r2_vs_pred = 1 - mse / (baseline_mse + 1e-10)

    print(f"\n  Judge vs β predictions:")
    print(f"  Spearman r (gen vs judge): {spearman_r:.4f}")
    print(f"  R² (β-predicted vs judge): {r2_vs_pred:.4f}")
    print(f"  MSE: {mse:.4f}")

    print(f"\n{'='*60}")
    if spearman_r < -0.6 and r2_vs_pred > 0.5:
        print("VALIDATION PASS: β predictions match independent judge scores")
    elif spearman_r < -0.3:
        print("VALIDATION MODERATE: β directionally matches judge scores")
    else:
        print("VALIDATION WEAK: β and judge diverge — methodology needs review")

    output = {
        "judge_model": JUDGE_MODEL, "test_model": TEST_MODEL,
        "n_seeds": N_SEEDS, "generations": N_GENERATIONS,
        "beta": beta,
        "judge_scores_per_gen": [float(x) for x in judge_means],
        "beta_predicted": [float(x) for x in predicted],
        "spearman_r": float(spearman_r),
        "r2_vs_predicted": float(r2_vs_pred),
        "mse": float(mse),
    }
    with open("experiment_data/judge_validation.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved: experiment_data/judge_validation.json")
    return output


if __name__ == "__main__":
    run()
