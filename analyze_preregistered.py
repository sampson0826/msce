"""Pre-registered analysis: fix exponential model + total_constraint target for all models.

Compares pre-registered β against original post-hoc (best-of-3 models, best-of-3 targets) β.
"""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from synthetic_decay_monitor.data_lineage import parse_lineage_from_jsonl
from synthetic_decay_monitor.constraint_extractor import HybridConstraintExtractor
from synthetic_decay_monitor.decay_engine import DecayEngine, _fit_decay_models, _ols

N100_DIR = "experiment_data/n100"

# Find all lineage files
lineage_files = {}
for f in os.listdir(N100_DIR):
    if f.endswith("_lineage.jsonl") and not f.endswith("_retest_lineage.jsonl"):
        path = os.path.join(N100_DIR, f)
        # Extract model name from filename
        name = f.replace("_lineage.jsonl", "").replace("_s100", "").replace("_openrouter", "").replace("_quickrouter", "").replace("_deepseek", "")
        # Skip duplicates (retest, alt seeds)
        if "retest" in name or "alt_seeds" in name:
            continue
        lineage_files[name] = path

print(f"Found {len(lineage_files)} lineage files:")
for name, path in sorted(lineage_files.items()):
    print(f"  {name}: {path}")

# Also include alt_seeds and retest for GPT-4o-mini
for extra in ["gpt-4o-mini_alt_seeds", "gpt-4o-mini_retest"]:
    path = os.path.join(N100_DIR, f"{extra}_lineage.jsonl")
    if os.path.exists(path):
        lineage_files[extra] = path


def pre_registered_beta(lineage_path: str) -> dict:
    """Compute β using pre-registered method: exponential model, total_constraint target."""
    lineage = parse_lineage_from_jsonl(lineage_path)
    extractor = HybridConstraintExtractor(judge_fn=None)
    engine = DecayEngine(lineage, extractor)
    engine.run_all_capabilities()

    results = {}
    for cap, snapshots in engine._snapshots.items():
        if len(snapshots) < 2:
            continue

        generations = np.array([s.generation for s in snapshots])
        total_mags = np.array([s.total_constraint for s in snapshots])

        if total_mags[0] < 1e-10:
            results[cap] = {"beta": 0.25, "r2": 0.0, "r2_adj": 0.0}
            continue

        # Pre-registered: exponential model only
        y_safe = np.maximum(total_mags, 1e-10)
        y0 = y_safe[0]
        log_ratios = np.log(y_safe / y0)
        slope, r2 = _ols(generations.astype(float), log_ratios)
        beta = 1.0 - np.exp(slope)
        beta = max(min(beta, 0.55), 0.001)
        n = len(generations)
        r2_adj = 1 - (1 - r2) * (n - 1) / max(n - 2, 1)

        results[cap] = {
            "beta": float(beta),
            "r2": float(r2),
            "r2_adj": float(r2_adj),
            "target": "total_constraint",
            "model": "exponential",
        }

    betas = [r["beta"] for r in results.values()]
    global_beta = float(np.mean(betas)) if betas else 0.0

    return {
        "per_capability": results,
        "global_beta": global_beta,
        "n_capabilities": len(results),
    }


# Load original post-hoc β values from reports
original_betas = {}
for f in os.listdir(N100_DIR):
    if f.endswith("_report.json"):
        path = os.path.join(N100_DIR, f)
        try:
            d = json.load(open(path))
            gb = d.get("decay_analysis", {}).get("global_beta", None)
            if gb is not None:
                name = f.replace("_report.json", "").replace("_s100", "").replace("_openrouter", "").replace("_quickrouter", "").replace("_deepseek", "")
                if "retest" not in name and "alt_seeds" not in name:
                    original_betas[name] = gb
        except:
            pass

# Add special cases
for extra, report_file in [
    ("gpt-4o-mini_retest", "gpt-4o-mini_retest_report.json"),
    ("gpt-4o-mini_alt_seeds", "gpt-4o-mini_alt_seeds_report.json"),
]:
    path = os.path.join(N100_DIR, report_file)
    if os.path.exists(path):
        d = json.load(open(path))
        original_betas[extra] = d.get("decay_analysis", {}).get("global_beta", 0.0)

print(f"\n{'='*70}")
print(f"PRE-REGISTERED vs POST-HOC β COMPARISON")
print(f"{'='*70}")
print(f"{'Model':<30s} {'Pre-reg β':>10s} {'Post-hoc β':>10s} {'Δ':>10s} {'R²_mean':>8s}")
print(f"{'-'*70}")

comparison = {}
for name in sorted(lineage_files.keys()):
    try:
        pr = pre_registered_beta(lineage_files[name])
    except Exception as e:
        print(f"{name:<30s} ERROR: {e}")
        continue
    ph = original_betas.get(name, float('nan'))
    delta = pr["global_beta"] - ph if not np.isnan(ph) else float('nan')
    r2_mean = np.mean([r["r2_adj"] for r in pr["per_capability"].values()])
    comparison[name] = {
        "pre_registered_beta": pr["global_beta"],
        "post_hoc_beta": ph,
        "delta": delta,
        "mean_r2_adj": float(r2_mean),
        "n_capabilities": pr["n_capabilities"],
    }
    print(f"{name:<30s} {pr['global_beta']:10.4f} {ph:10.4f} {delta:10.4f} {r2_mean:8.4f}")

# Summary stats
deltas = [c["delta"] for c in comparison.values() if not np.isnan(c["delta"])]
print(f"\nMean |Δ| = {np.mean(np.abs(deltas)):.4f}")
print(f"Max |Δ| = {np.max(np.abs(deltas)):.4f}")
print(f"Pre-reg vs post-hoc Pearson r: {np.corrcoef([c['pre_registered_beta'] for c in comparison.values() if not np.isnan(c['delta'])], [c['post_hoc_beta'] for c in comparison.values() if not np.isnan(c['delta'])])[0,1]:.4f}")

# Save
with open("experiment_data/preregistered_analysis.json", "w") as f:
    json.dump(comparison, f, indent=2)
print(f"\nSaved: experiment_data/preregistered_analysis.json")
