"""LLM judge validation v2: per-capability mean judge preference vs β.

Design: For each capability, show judge 10 blind pairwise comparisons (Gen1 vs Gen3).
Both are LLM-generated responses — fair comparison. Hypothesis: judge prefers Gen1 > Gen3.
Compute mean judge preference per capability → correlate with per-capability β.
"""
import sys, os, json, random, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip())

import numpy as np
from synthetic_decay_monitor.data_lineage import parse_lineage_from_jsonl
from synthetic_decay_monitor.constraint_extractor import HybridConstraintExtractor
from synthetic_decay_monitor.decay_engine import DecayEngine
from synthetic_decay_monitor.provider_adapter import create_provider

LINEAGE = "experiment_data/n100/gpt-4o-mini_s100_lineage.jsonl"
JUDGE_MODEL = "gpt-4o"
PAIRS_PER_CAP = 10

print(f"Loading lineage: {LINEAGE}")
lineage = parse_lineage_from_jsonl(LINEAGE)

# Get pre-registered per-capability β from DecayEngine
print("Computing per-capability β...")
extractor = HybridConstraintExtractor(judge_fn=None)
engine = DecayEngine(lineage, extractor)
engine.run_all_capabilities()

per_cap_beta = {}
per_cap_sn = {}
for t in engine.get_all_trajectories():
    cap = t["capability"]
    traj = t["trajectory"]
    for g in traj:
        if "beta" in g and g["beta"] > 0:
            per_cap_beta[cap] = g["beta"]
            break
    per_cap_sn[cap] = traj[-1]["S_n"] if traj else 1.0

print(f"Per-capability β: {json.dumps(per_cap_beta, indent=2)}")

# Group samples by capability and generation
from collections import defaultdict
by_cap_gen = defaultdict(lambda: defaultdict(list))
for s in lineage.samples:
    cap = s.capability_tags[0] if s.capability_tags else "unknown"
    by_cap_gen[cap][s.generation].append(s)

capabilities = sorted(by_cap_gen.keys())

# Select pairs
print(f"\nSelecting {PAIRS_PER_CAP} blind pairs per capability...")
random.seed(42)
pairs = []

for cap in capabilities:
    gen1_samples = [s for s in by_cap_gen[cap][1] if s.text and len(s.text) > 50]
    gen3_samples = [s for s in by_cap_gen[cap][3] if s.text and len(s.text) > 50]
    n_pairs = min(PAIRS_PER_CAP, len(gen1_samples), len(gen3_samples))
    g1_sel = random.sample(gen1_samples, n_pairs)
    g3_sel = random.sample(gen3_samples, n_pairs)
    for g1, g3 in zip(g1_sel, g3_sel):
        if random.random() < 0.5:
            pairs.append({"cap": cap, "text_a": g1.text[:1500], "text_b": g3.text[:1500], "a_is_gen1": True})
        else:
            pairs.append({"cap": cap, "text_a": g3.text[:1500], "text_b": g1.text[:1500], "a_is_gen1": False})

print(f"Total pairs: {len(pairs)}")

# Judge via GPT-4o
adapter = create_provider("quickrouter", model=JUDGE_MODEL, temperature=0.0)
results = []

for i, pair in enumerate(pairs):
    judge_prompt = f"""Compare the quality of these two texts. Which is better?

TEXT A:
---
{pair['text_a']}
---

TEXT B:
---
{pair['text_b']}
---

Rate: 1 = A much better, 2 = A slightly better, 3 = equal, 4 = B slightly better, 5 = B much better.

Consider clarity, coherence, factual accuracy, logical structure, and writing quality.

Respond with ONLY a number (1-5)."""

    score = 3  # default
    for attempt in range(3):
        try:
            resp = adapter.generate(judge_prompt, max_tokens=5, temperature=0.0)
            for ch in resp.strip():
                if ch in "12345":
                    score = int(ch)
                    break
            break
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** (attempt + 1))
            else:
                print(f"  Pair {i+1} FAIL: {e}")

    gen1_pref = (6 - score) if pair["a_is_gen1"] else score
    results.append({"cap": pair["cap"], "gen1_pref": gen1_pref, "raw_score": score})

    if (i + 1) % 15 == 0:
        mean_pref = sum(r["gen1_pref"] for r in results) / len(results)
        print(f"  {i+1}/{len(pairs)}: mean Gen0 preference = {mean_pref:.2f} (expected >3.0)")

# Per-capability aggregation
print(f"\n{'='*60}")
print(f"JUDGE VALIDATION v2: LLM Judge vs Constraint Extractor")
print(f"{'='*60}")
print(f"Judge: {JUDGE_MODEL} | Test model: gpt-4o-mini")
print(f"N pairs: {len(results)} | Pairs per cap: {PAIRS_PER_CAP}")
print(f"\n{'Capability':<25s} {'Judge Pref':>10s} {'Gen3 S_n':>10s} {'β':>10s}")
print(f"{'-'*58}")

cap_judge = {}
for cap in capabilities:
    cap_r = [r for r in results if r["cap"] == cap]
    mean_pref = np.mean([r["gen1_pref"] for r in cap_r])
    cap_judge[cap] = mean_pref
    sn = per_cap_sn.get(cap, 0)
    beta = per_cap_beta.get(cap, 0)
    print(f"{cap:<25s} {mean_pref:10.3f} {sn:10.3f} {beta:10.4f}")

# Spearman: judge preference vs S_n (should be positive: higher S_n → judge prefers Gen0 more)
judge_vals = np.array([cap_judge[c] for c in capabilities])
sn_vals = np.array([per_cap_sn[c] for c in capabilities])
beta_vals = np.array([per_cap_beta[c] for c in capabilities])

def spearmanr(x, y):
    n = len(x)
    if n < 3: return 0.0, 1.0
    rank_x = np.argsort(np.argsort(x)).astype(float) + 1
    rank_y = np.argsort(np.argsort(y)).astype(float) + 1
    d2 = (rank_x - rank_y) ** 2
    r = 1 - 6 * d2.sum() / (n * (n**2 - 1))
    return r

r_sn = spearmanr(judge_vals, sn_vals)
r_beta = spearmanr(judge_vals, -beta_vals)  # negative: higher β → lower judge pref
print(f"\nSpearman(judge_preference, Gen3_S_n): r = {r_sn:.4f}")
print(f"Spearman(judge_preference, -β): r = {r_beta:.4f}")

mean_all = np.mean(judge_vals)
print(f"\nOverall mean Gen0 preference: {mean_all:.3f} (3=equal, >3=prefer Gen0)")
print(f"Overall pct Gen0 preferred: {100*np.mean(np.array([r['gen1_pref'] for r in results]) >= 3.5):.0f}%")
print(f"Overall pct Gen3 preferred: {100*np.mean(np.array([r['gen1_pref'] for r in results]) <= 2.5):.0f}%")

# Verdict
print(f"\nVerdict:")
if r_sn > 0.6:
    print(f"  STRONG: Judge preference aligns with constraint extractor (r={r_sn:.3f})")
elif r_sn > 0.3:
    print(f"  MODERATE: Positive correlation (r={r_sn:.3f}). Judge partially validates extractor.")
elif r_sn > 0:
    print(f"  WEAK: Directionally correct but weak (r={r_sn:.3f})")
else:
    print(f"  FAIL: No alignment (r={r_sn:.3f})")

with open("experiment_data/judge_validation_v2.json", "w") as f:
    json.dump({
        "judge_model": JUDGE_MODEL, "test_model": "gpt-4o-mini",
        "n_pairs": len(results), "pairs_per_cap": PAIRS_PER_CAP,
        "mean_gen1_preference": float(mean_all),
        "spearman_vs_Sn": float(r_sn),
        "spearman_vs_beta": float(r_beta),
        "per_cap_judge_pref": {c: float(v) for c, v in cap_judge.items()},
        "per_cap_Sn": {c: float(v) for c, v in per_cap_sn.items()},
        "per_cap_beta": {c: float(v) for c, v in per_cap_beta.items()},
        "results": results,
    }, f, indent=2)
print(f"\nSaved: experiment_data/judge_validation_v2.json")
