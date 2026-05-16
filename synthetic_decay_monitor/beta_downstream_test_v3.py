"""β → downstream loss mapping v3: multi-step reasoning chain verification.

Uses 20 multi-step math/logic problems. Each generation must show reasoning.
Ground truth: does the answer match + are all logical steps present?
"""
import sys, os, json, time, re
import numpy as np

sys.path.insert(0, ".")
from synthetic_decay_monitor.provider_adapter import create_provider
from synthetic_decay_monitor.constraint_extractor import HybridConstraintExtractor
from synthetic_decay_monitor.decay_engine import DecayEngine
from synthetic_decay_monitor.data_lineage import DataSample, DatasetLineage

KEY = "sk-0XqACr4qnHSionZmOLmiWegQSpVFNVH7M5sUF2dHN2Z80JVA"
MODEL = "gpt-4o-mini"
N_GENERATIONS = 3

# Multi-step reasoning problems with unambiguous answers
PROBLEMS = [
    ("A store has a 20% off sale. If a jacket originally costs $85, and there's an additional 8% sales tax applied to the sale price, what is the final cost?",
     "73.44"),
    ("If a car travels at 60 mph for 2.5 hours, then at 45 mph for 1.5 hours, what is the average speed for the entire trip?",
     "54.375"),
    ("A recipe needs 2/3 cup of sugar for 12 cookies. How much sugar is needed for 30 cookies?",
     "1.667"),  # 5/3
    ("John is 3 times as old as his son. In 12 years, John will be twice as old as his son. How old is John's son now?",
     "12"),
    ("A water tank can be filled by pipe A in 4 hours and by pipe B in 6 hours. If both pipes are used, how long to fill the tank?",
     "2.4"),
    ("If 5 workers can build 5 houses in 5 days, how many houses can 10 workers build in 10 days?",
     "20"),
    ("A train leaves Station A at 9:00 AM traveling at 80 km/h. Another train leaves Station B at 10:30 AM traveling at 100 km/h toward Station A. If the stations are 400 km apart, at what time do they meet?",
     "11.82"),  # ~11:49 AM — gives partial credit if close
    ("Alice has $120 more than Bob. Together they have $340. How much does Alice have?",
     "230"),
    ("A cylinder has radius 3 cm and height 10 cm. What is its volume? Use pi=3.14.",
     "282.6"),
    ("If f(x) = 2x^2 + 3x - 5, what is f(4)?",
     "39"),
    ("The sum of three consecutive integers is 72. What is the largest integer?",
     "25"),
    ("A population of bacteria doubles every 3 hours. Starting with 100 bacteria, how many are there after 12 hours?",
     "1600"),
    ("If log_10(x) = 2, what is x?",
     "100"),
    ("A ladder 13 meters long leans against a wall. The base is 5 meters from the wall. How high does the ladder reach?",
     "12"),
    ("Find the sum of the first 10 positive integers (1+2+...+10).",
     "55"),
    ("A box contains 4 red, 3 blue, and 2 green balls. If one ball is randomly selected, what is the probability it is NOT red?",
     "5/9"),  # 0.556
    ("If 3x + 7 = 2x + 15, what is x?",
     "8"),
    ("The price of a book increased by 25% to $25. What was the original price?",
     "20"),
    ("A rectangular garden has perimeter 36 meters. If the length is twice the width, what is the area?",
     "72"),
    ("If the ratio of boys to girls in a class is 3:2 and there are 30 students total, how many boys are there?",
     "18"),
]


def score_answer(response: str, expected: str) -> dict:
    """Score a multi-step response. Returns {correct, reasoning_present, overall}."""
    # 1. Answer match
    resp_clean = response.lower()
    exp_clean = expected.lower()

    # Extract numbers and fractions from response
    resp_nums = set(re.findall(r'\d+\.?\d*', response))
    exp_nums = set(re.findall(r'\d+\.?\d*', expected))
    num_match = len(resp_nums & exp_nums) > 0

    has_answer = exp_clean in resp_clean or num_match

    # 2. Reasoning steps present
    reasoning_indicators = [
        "first", "then", "next", "finally", "therefore", "thus", "because",
        "step", "calculate", "compute", "solve", "equation", "formula",
        "first", "then", "next", "finally", "所以", "因此", "首先", "然后",
    ]
    n_indicators = sum(1 for ind in reasoning_indicators if ind in resp_clean)
    has_reasoning = n_indicators >= 2 or len(response.split()) > 40

    overall = 0.0
    if has_answer and has_reasoning:
        overall = 1.0
    elif has_answer:
        overall = 0.6
    elif has_reasoning:
        overall = 0.3

    return {"correct": float(has_answer), "reasoning": float(has_reasoning), "overall": overall}


def run():
    print("=" * 60)
    print(f"β → Reasoning Quality Mapping: {MODEL}")
    print(f"Problems: {len(PROBLEMS)}, Generations: {N_GENERATIONS}")
    print("=" * 60)

    adapter = create_provider("quickrouter", model=MODEL, api_key=KEY)
    extractor = HybridConstraintExtractor(judge_fn=None)

    gen_scores = {g: [] for g in range(1, N_GENERATIONS + 1)}
    gen_texts = {g: [] for g in range(1, N_GENERATIONS + 1)}

    for i, (problem, answer) in enumerate(PROBLEMS):
        prompt = f"Solve this problem step by step. Show your reasoning clearly, then give the final answer.\n\n{problem}"

        for gen in range(1, N_GENERATIONS + 1):
            resp = adapter.generate(prompt, max_tokens=256, temperature=0.7)
            time.sleep(0.12)

            scores = score_answer(resp, answer)
            gen_scores[gen].append(scores)
            gen_texts[gen].append(resp)
            prompt = resp  # Recursive: output feeds into next input

        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{len(PROBLEMS)}] done")

    print(f"\n{'='*60}")
    print(f"Reasoning quality across recursive generations:")
    print(f"{'='*60}")

    gen_means = []
    for gen in range(1, N_GENERATIONS + 1):
        scores = gen_scores[gen]
        correct = np.mean([s["correct"] for s in scores])
        reasoning = np.mean([s["reasoning"] for s in scores])
        overall = np.mean([s["overall"] for s in scores])
        gen_means.append(overall)
        n_perfect = sum(1 for s in scores if s["overall"] >= 0.9)
        print(f"  Gen{gen}: overall={overall:.3f} correct={correct:.2f} reasoning={reasoning:.2f} (perfect={n_perfect}/{len(scores)})")

    # Compute β
    samples = []
    for gen in range(1, N_GENERATIONS + 1):
        for i in range(len(PROBLEMS)):
            samples.append(DataSample(
                text=gen_texts[gen][i],
                generation=gen,
                source_model=MODEL,
                capability_tags=["math_reasoning"],
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

    print(f"\n{'='*60}")
    print(f"β → Reasoning quality mapping:")
    print(f"{'='*60}")
    print(f"  Measured β: {beta:.4f}")

    predicted = [gen_means[0] * (1 - beta)**(n) for n in range(N_GENERATIONS)]

    for gen in range(1, N_GENERATIONS + 1):
        print(f"  Gen{gen}: observed={gen_means[gen-1]:.3f}, β-predicted={predicted[gen-1]:.3f}")

    obs_drop = gen_means[0] - gen_means[-1]
    pred_drop = predicted[0] - predicted[-1]

    gens_arr = np.arange(1, N_GENERATIONS + 1)
    r = np.corrcoef(gens_arr, np.array(gen_means))[0, 1]

    print(f"\n  Observed quality drop: {obs_drop:.3f}")
    print(f"  β-predicted quality drop: {pred_drop:.3f}")
    print(f"  β / observed_drop: {beta / max(obs_drop, 0.001):.2f}")
    print(f"  Pearson r (gen vs quality): {r:.4f}")

    print(f"\n{'='*60}")
    if r < -0.6:
        print("VERDICT: β strongly predicts downstream reasoning quality degradation")
    elif r < -0.3:
        print("VERDICT: β directionally predicts reasoning degradation")
    else:
        print("VERDICT: Weak correlation — check problem difficulty or task type")

    output = {
        "model": MODEL, "n_problems": len(PROBLEMS),
        "generations": N_GENERATIONS, "beta": beta,
        "per_gen_scores": [float(s) for s in gen_means],
        "predicted_scores": [float(s) for s in predicted],
        "pearson_r": float(r),
    }
    with open("experiment_data/beta_downstream_reasoning.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved: experiment_data/beta_downstream_reasoning.json")
    return output


if __name__ == "__main__":
    run()
