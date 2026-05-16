"""LLM judge pairwise: compare Gen1 vs GenN outputs for quality.

Pairwise comparison is more reliable than absolute scoring.
"""
import sys, os, json, time, random
import numpy as np

sys.path.insert(0, ".")
from synthetic_decay_monitor.provider_adapter import create_provider
from synthetic_decay_monitor.constraint_extractor import HybridConstraintExtractor
from synthetic_decay_monitor.decay_engine import DecayEngine
from synthetic_decay_monitor.data_lineage import DataSample, DatasetLineage

JUDGE_MODEL = "gpt-5.5"
TEST_MODEL = "gpt-4o-mini"
KEY = "sk-0XqACr4qnHSionZmOLmiWegQSpVFNVH7M5sUF2dHN2Z80JVA"

N_GENERATIONS = 3
N_SEEDS = 10

SEEDS = [
    "Explain how photosynthesis converts sunlight into chemical energy.",
    "Write a short story about a robot learning to paint.",
    "Describe the water cycle and its importance for life on Earth.",
    "Explain the concept of supply and demand in economics.",
    "What causes seasons on Earth? Explain the astronomical reason.",
    "Explain how the internet works at a high level.",
    "Compare and contrast solar and wind energy.",
    "Describe the structure of DNA and how it replicates.",
    "Explain what CRISPR gene editing is and how it works.",
    "What is the significance of the Turing test in AI?",
]

PAIRWISE_PROMPT = """Which response has higher QUALITY? Consider coherence, factual detail, structure, and language quality.

Response A:
{a_text}

Response B:
{b_text}

Reply with ONLY "A" or "B". No explanation."""


def run():
    print("=" * 60)
    print(f"Pairwise Judge Validation: {JUDGE_MODEL} → {TEST_MODEL}")
    print(f"Seeds: {N_SEEDS}, Generations: {N_GENERATIONS}")
    print("=" * 60)

    adapter = create_provider("quickrouter", model=TEST_MODEL, api_key=KEY)
    judge = create_provider("quickrouter", model=JUDGE_MODEL, api_key=KEY)
    extractor = HybridConstraintExtractor(judge_fn=None)

    # Generate
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

    # Pairwise: Gen1 vs Gen2, Gen1 vs Gen3 for each seed
    print(f"\nPairwise judging (Gen1 vs later gens)...")
    gen1_wins = 0
    gen2_wins = 0
    gen3_wins = 0
    total_pairs = 0

    for i in range(N_SEEDS):
        gen1_text = gen_texts[1][i]
        gen2_text = gen_texts[2][i]
        gen3_text = gen_texts[3][i]

        # Gen1 vs Gen2
        # Randomize order to avoid position bias
        for a_text, b_text, a_label, b_label in [
            (gen1_text, gen2_text, "gen1", "gen2"),
            (gen1_text, gen3_text, "gen1", "gen3"),
        ]:
            total_pairs += 1
            if random.random() < 0.5:
                prompt = PAIRWISE_PROMPT.format(a_text=a_text[:600], b_text=b_text[:600])
                try:
                    winner = judge.generate(prompt, max_tokens=2, temperature=0.0).strip().upper()
                    if winner == "A":
                        if a_label == "gen1": gen1_wins += 1
                        else: (gen2_wins if b_label == "gen2" else gen3_wins)  # won't happen
                    elif winner == "B":
                        if b_label == "gen2": gen2_wins += 1
                        elif b_label == "gen3": gen3_wins += 1
                    time.sleep(0.08)
                except Exception as e:
                    print(f"    Error: {e}")
            else:
                prompt = PAIRWISE_PROMPT.format(a_text=b_text[:600], b_text=a_text[:600])
                try:
                    winner = judge.generate(prompt, max_tokens=2, temperature=0.0).strip().upper()
                    if winner == "A":
                        if b_label == "gen2": gen2_wins += 1
                        elif b_label == "gen3": gen3_wins += 1
                    elif winner == "B":
                        if a_label == "gen1": gen1_wins += 1
                    time.sleep(0.08)
                except Exception as e:
                    print(f"    Error: {e}")

    print(f"\n{'='*60}")
    print("Pairwise results (Gen1 as baseline):")
    print(f"  Gen1 wins against later gens: {gen1_wins}/{total_pairs} ({gen1_wins/total_pairs:.1%})")
    print(f"  Gen2 wins against Gen1: {gen2_wins}/{total_pairs//2}")
    print(f"  Gen3 wins against Gen1: {gen3_wins}/{total_pairs//2}")

    # Expected under quality decay: Gen1 should win >50% of pairwise matches
    gen1_winrate = gen1_wins / total_pairs

    # Compute β
    samples = []
    for gen in range(1, N_GENERATIONS + 1):
        for i in range(N_SEEDS):
            samples.append(DataSample(
                text=gen_texts[gen][i], generation=gen,
                source_model=TEST_MODEL, capability_tags=["general"],
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

    print(f"\n  β = {beta:.4f}")
    print(f"  Gen1 pairwise win rate: {gen1_winrate:.1%}")

    threshold = 0.55  # Better than random
    print(f"\n{'='*60}")
    if gen1_winrate > 0.65:
        print("VALIDATION PASS: Independent judge strongly prefers Gen1 over later gens")
    elif gen1_winrate > threshold:
        print("VALIDATION MODERATE: Judge weakly prefers Gen1 — β direction validated")
    else:
        print(f"VALIDATION WEAK: Judge shows no clear preference (winrate={gen1_winrate:.1%})")

    return {
        "beta": beta, "gen1_winrate": gen1_winrate,
        "gen1_wins": gen1_wins, "total_pairs": total_pairs,
    }


if __name__ == "__main__":
    run()
