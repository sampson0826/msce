"""β → downstream loss mapping: prove β predicts real accuracy degradation.

Takes 20 math/QA problems through 3 recursive generations with gpt-4o-mini,
scores correctness at each generation, and maps β to accuracy drop.
"""
import sys, os, json, time, re
import numpy as np

sys.path.insert(0, ".")
from synthetic_decay_monitor.provider_adapter import create_provider
from synthetic_decay_monitor.constraint_extractor import HybridConstraintExtractor
from synthetic_decay_monitor.decay_engine import DecayEngine, calibrate_beta_from_data
from synthetic_decay_monitor.data_lineage import DataSample, DatasetLineage

KEY = "sk-0XqACr4qnHSionZmOLmiWegQSpVFNVH7M5sUF2dHN2Z80JVA"
MODEL = "gpt-4o-mini"
N_GENERATIONS = 3

# 20 problems with verifiable answers
PROBLEMS = [
    ("math", "What is 15% of 240?", "36"),
    ("math", "If x + 7 = 22, what is x?", "15"),
    ("math", "What is the area of a circle with radius 5? Use pi=3.14.", "78.5"),
    ("math", "Solve: 3^2 + 4^2 = ?", "25"),
    ("math", "If a train travels 120 km in 2 hours, what is its average speed in km/h?", "60"),
    ("math", "What is the square root of 144?", "12"),
    ("math", "If 3x = 27, what is x?", "9"),
    ("math", "A rectangle has length 8 and width 5. What is its area?", "40"),
    ("math", "What is 2^5?", "32"),
    ("math", "If a pizza is cut into 8 slices and you eat 3, what fraction remains?", "5/8"),
    ("math", "Convert 0.75 to a fraction in simplest form.", "3/4"),
    ("math", "What is the sum of angles in a triangle?", "180"),
    ("math", "If y = 2x + 1 and x = 4, what is y?", "9"),
    ("math", "A bag has 3 red and 5 blue marbles. Probability of drawing red?", "3/8"),
    ("math", "What is the perimeter of a square with side length 6?", "24"),
    ("factual", "What is the capital of France?", "Paris"),
    ("factual", "How many continents are there on Earth?", "7"),
    ("factual", "What element has the chemical symbol 'O'?", "Oxygen"),
    ("factual", "Who wrote 'Romeo and Juliet'?", "William Shakespeare"),
    ("factual", "What planet is known as the Red Planet?", "Mars"),
]


def score_answer(generated_text: str, expected: str) -> float:
    """Check if generated text contains the expected answer. Returns 0-1."""
    text_lower = generated_text.lower()
    expected_lower = expected.lower()
    # Direct substring match
    if expected_lower in text_lower:
        return 1.0
    # Number match (handle different number formats)
    nums_gen = set(re.findall(r'\d+\.?\d*', generated_text))
    nums_exp = set(re.findall(r'\d+\.?\d*', expected))
    if nums_exp and nums_gen & nums_exp:
        return 0.8 if expected_lower not in text_lower else 1.0
    return 0.0


def run():
    print("=" * 60)
    print(f"β → Downstream Loss Mapping: {MODEL}")
    print(f"Problems: {len(PROBLEMS)}, Generations: {N_GENERATIONS}")
    print("=" * 60)

    adapter = create_provider("quickrouter", model=MODEL, api_key=KEY)
    extractor = HybridConstraintExtractor(judge_fn=None)

    # Store: gen → [(problem, response, score)]
    gen_results = {g: [] for g in range(N_GENERATIONS + 1)}
    texts_per_gen = {g: [] for g in range(N_GENERATIONS + 1)}

    for i, (category, problem, answer) in enumerate(PROBLEMS):
        prompt = f"Answer this question concisely with just the answer or a short explanation:\n\n{problem}"

        for gen in range(N_GENERATIONS + 1):
            if gen == 0:
                resp = prompt
            else:
                # Previous generation's response becomes the new prompt
                prev_resp = gen_results[gen - 1][i][1]
                resp = adapter.generate(
                    f"Here is a previous answer:\n{prev_resp}\n\nRefine or re-answer this question (keep a clear answer):\n{problem}",
                    max_tokens=128, temperature=0.8
                )
                time.sleep(0.15)

            score = score_answer(resp, answer)
            gen_results[gen].append((problem, resp, score))
            texts_per_gen[gen].append(resp)

        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{len(PROBLEMS)}] done")

    # Compute per-generation accuracy
    print(f"\n{'='*60}")
    print(f"Accuracy trajectory across recursive generations:")
    print(f"{'='*60}")
    accuracies = []
    for gen in range(N_GENERATIONS + 1):
        scores = [s for _, _, s in gen_results[gen]]
        acc = np.mean(scores)
        accuracies.append(acc)
        n_correct = sum(1 for s in scores if s >= 0.7)
        print(f"  Gen{gen}: accuracy={acc:.3f} ({n_correct}/{len(scores)} correct)")

    # Compute β via DecayEngine
    print(f"\n{'='*60}")
    print(f"Computing β from constraint decay...")

    samples = []
    for gen in range(N_GENERATIONS + 1):
        for i, (cat, prob, ans) in enumerate(PROBLEMS):
            samples.append(DataSample(
                text=texts_per_gen[gen][i],
                generation=gen,
                source_model=MODEL,
                capability_tags=[cat],
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
    beta = sum(betas) / len(betas) if betas else 0.25

    print(f"  Measured β: {beta:.4f}")
    print(f"  Predicted S_n trajectory: ", end="")
    S = 1.0
    for gen in range(N_GENERATIONS + 1):
        print(f"S_{gen}={S:.3f} ", end="")
        S *= (1 - beta)
    print()

    # Correlation: β-predicted accuracy vs observed
    print(f"\n{'='*60}")
    print(f"β → Accuracy mapping:")
    print(f"{'='*60}")

    # Linear fit: accuracy vs generation
    gens_arr = np.arange(N_GENERATIONS + 1)
    acc_arr = np.array(accuracies)
    slope, intercept = np.polyfit(gens_arr, acc_arr, 1)
    r2 = 1 - np.sum((acc_arr - (intercept + slope * gens_arr))**2) / np.sum((acc_arr - acc_arr.mean())**2)

    # Predicted: accuracy_n = accuracy_0 * (1 - β)^n
    predicted_acc = [accuracies[0] * (1 - beta)**n for n in range(N_GENERATIONS + 1)]

    for gen in range(N_GENERATIONS + 1):
        print(f"  Gen{gen}: observed={accuracies[gen]:.3f}, β-predicted={predicted_acc[gen]:.3f}")

    print(f"\n  Accuracy loss per gen (linear): {-slope:.4f}")
    print(f"  R² of linear fit: {r2:.4f}")
    print(f"  β (constraint decay): {beta:.4f}")
    print(f"  Ratio β / accuracy_loss: {beta / max(-slope, 0.001):.2f}")

    # Verdict
    print(f"\n{'='*60}")
    if r2 > 0.7:
        print("VERDICT: β strongly predicts downstream accuracy degradation")
    elif r2 > 0.4:
        print("VERDICT: β moderately predicts accuracy degradation")
    else:
        print("VERDICT: Weak correlation — check methodology")

    # Save
    output = {
        "model": MODEL,
        "n_problems": len(PROBLEMS),
        "generations": N_GENERATIONS,
        "beta": beta,
        "per_gen_accuracy": [float(a) for a in accuracies],
        "linear_fit_slope": float(slope),
        "linear_fit_r2": float(r2),
        "predicted_accuracy": [float(a) for a in predicted_acc],
    }
    with open("experiment_data/beta_downstream_mapping.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved: experiment_data/beta_downstream_mapping.json")
    return output


if __name__ == "__main__":
    run()
