"""Bootstrap convergence: show β stabilizes with sample size using existing n=100 data."""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from synthetic_decay_monitor.data_lineage import parse_lineage_from_jsonl
from synthetic_decay_monitor.constraint_extractor import HybridConstraintExtractor
from synthetic_decay_monitor.decay_engine import DecayEngine

LINEAGE_FILE = "experiment_data/n100/gpt-4o-mini_s100_lineage.jsonl"
N_BOOTSTRAP = 100
SAMPLE_SIZES = [12, 24, 36, 48, 60, 75, 100]

print(f"Loading lineage: {LINEAGE_FILE}")
lineage = parse_lineage_from_jsonl(LINEAGE_FILE)
all_samples = lineage.samples
gen0 = [s for s in all_samples if s.generation == 0]
gen1 = [s for s in all_samples if s.generation == 1]
gen2 = [s for s in all_samples if s.generation == 2]
gen3 = [s for s in all_samples if s.generation == 3]
n_total = len(gen0)
print(f"Total seeds: {n_total}")

# Group by capability for balanced sampling
from collections import defaultdict
by_cap = defaultdict(lambda: defaultdict(list))
for gen, samples in [(0, gen0), (1, gen1), (2, gen2), (3, gen3)]:
    for s in samples:
        cap = s.capability_tags[0] if s.capability_tags else "unknown"
        by_cap[cap][gen].append(s)

capabilities = list(by_cap.keys())
print(f"Capabilities: {capabilities}")
print(f"Samples per cap: {[(c, len(by_cap[c][0])) for c in capabilities]}")

np.random.seed(42)

results = {n: [] for n in SAMPLE_SIZES}

for n in SAMPLE_SIZES:
    print(f"\n--- n={n} ({N_BOOTSTRAP} bootstrap iterations) ---")
    per_cap_n = max(1, n // len(capabilities))

    for b in range(N_BOOTSTRAP):
        # Balanced sampling across capabilities
        sampled = []
        for cap in capabilities:
            cap_n = min(per_cap_n, len(by_cap[cap][0]))
            indices = np.random.choice(len(by_cap[cap][0]), size=cap_n, replace=False)
            for gen in range(4):
                cap_samples = by_cap[cap][gen]
                for idx in indices:
                    sampled.append(cap_samples[idx])

        # Reconstruct lineage
        from synthetic_decay_monitor.data_lineage import DatasetLineage
        boot_lineage = DatasetLineage(samples=sampled)

        # Fit beta
        extractor = HybridConstraintExtractor(judge_fn=None)
        engine = DecayEngine(boot_lineage, extractor)
        engine.run_all_capabilities()

        betas = []
        for t in engine.get_all_trajectories():
            if "trajectory" in t:
                for g in t["trajectory"]:
                    if "beta" in g and g["beta"] > 0:
                        betas.append(g["beta"])
        global_beta = sum(betas) / len(betas) if betas else 0.0
        results[n].append(global_beta)

        if (b + 1) % 20 == 0:
            mean_so_far = np.mean(results[n])
            std_so_far = np.std(results[n])
            print(f"  {b+1}/{N_BOOTSTRAP}: β_mean={mean_so_far:.4f}, σ={std_so_far:.4f}")

# Summary
print(f"\n{'='*60}")
print(f"CONVERGENCE ANALYSIS")
print(f"{'='*60}")
print(f"{'n':>6s}  {'β_mean':>8s}  {'β_std':>8s}  {'CI_95_low':>10s}  {'CI_95_high':>10s}")
print(f"{'-'*50}")
for n in SAMPLE_SIZES:
    arr = np.array(results[n])
    mean = arr.mean()
    std = arr.std()
    ci_low = np.percentile(arr, 2.5)
    ci_high = np.percentile(arr, 97.5)
    print(f"{n:6d}  {mean:8.4f}  {std:8.4f}  {ci_low:10.4f}  {ci_high:10.4f}")

# Save
out = {
    "lineage_file": LINEAGE_FILE,
    "n_bootstrap": N_BOOTSTRAP,
    "sample_sizes": SAMPLE_SIZES,
    "results": {str(n): [float(x) for x in results[n]] for n in SAMPLE_SIZES},
    "summary": {str(n): {
        "mean": float(np.mean(results[n])),
        "std": float(np.std(results[n])),
        "ci_95": [float(np.percentile(results[n], 2.5)), float(np.percentile(results[n], 97.5))],
        "ci_width": float(np.percentile(results[n], 97.5) - np.percentile(results[n], 2.5)),
    } for n in SAMPLE_SIZES},
}
with open("experiment_data/convergence_analysis.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: experiment_data/convergence_analysis.json")

# Verdict
n36 = np.array(results[36])
n100 = np.array(results[100])
print(f"\nVerdict:")
print(f"  n=36: β={n36.mean():.4f} ± {n36.std():.4f} (95% CI: [{np.percentile(n36,2.5):.4f}, {np.percentile(n36,97.5):.4f}])")
print(f"  n=100: β={n100.mean():.4f} ± {n100.std():.4f} (95% CI: [{np.percentile(n100,2.5):.4f}, {np.percentile(n100,97.5):.4f}])")
print(f"  n=36 CI width: {np.percentile(n36,97.5)-np.percentile(n36,2.5):.4f}")
print(f"  n=100 CI width: {np.percentile(n100,97.5)-np.percentile(n100,2.5):.4f}")
print(f"  CI reduction: {(1 - (np.percentile(n100,97.5)-np.percentile(n100,2.5))/(np.percentile(n36,97.5)-np.percentile(n36,2.5)))*100:.0f}%")
