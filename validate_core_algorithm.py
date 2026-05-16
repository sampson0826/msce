"""Validate improved core algorithm against all existing experiment data.

Tests:
1. Multi-model fit: exponential vs linear vs power-law (R² comparison)
2. Bootstrap β CI width
3. σ dimension independence (correlation matrix)
4. Π metric comparison: total_magnitude vs pi_magnitude vs sigma_vec
"""
import sys, os, json, math
import numpy as np

sys.path.insert(0, ".")
from synthetic_decay_monitor.provider_adapter import create_provider
from synthetic_decay_monitor.constraint_extractor import HybridConstraintExtractor
from synthetic_decay_monitor.decay_engine import (
    DecayEngine, calibrate_beta_from_data, validate_decay_model,
    bootstrap_beta_ci, sigma_correlation_matrix, _fit_decay_models,
)
from synthetic_decay_monitor.data_lineage import DataSample, DatasetLineage, parse_lineage_from_jsonl

LINEAGE_FILES = [
    "experiment_data/claude_lineage.jsonl",
    "experiment_data/gpt-4o-mini_lineage.jsonl",
    "experiment_data/haiku_lineage.jsonl",
    "experiment_data/opus_lineage.jsonl",
    "experiment_data/gpt4o_lineage.jsonl",
]


def main():
    print("=" * 70)
    print("Core Algorithm Validation — Multi-Model Fit + Bootstrap + Sigma Independence")
    print("=" * 70)

    extractor = HybridConstraintExtractor(judge_fn=None)
    all_results = {}

    for path in LINEAGE_FILES:
        if not os.path.exists(path):
            print(f"\n  SKIP: {path} not found")
            continue

        model_name = os.path.basename(path).replace("_lineage.jsonl", "")
        print(f"\n{'─' * 70}")
        print(f"  {model_name}")
        print(f"{'─' * 70}")

        lineage = parse_lineage_from_jsonl(path)
        engine = DecayEngine(lineage, extractor)
        engine.run_all_capabilities()

        # Per-capability validation
        for cap, snapshots in engine._snapshots.items():
            print(f"\n  [{cap}]")

            # 1. Multi-model fit on ||Π||
            generations = np.array([s.generation for s in snapshots])
            pi_vals = np.array([s.pi_magnitude for s in snapshots])
            pi_models = _fit_decay_models(generations, pi_vals)
            best_pi = max(pi_models, key=lambda m: m["r2_adj"])

            # 2. Multi-model fit on sigma vector magnitude
            sigma_vec_mags = np.array([
                np.linalg.norm(list(s.individual_sigmas.values()))
                for s in snapshots
            ])
            sigma_models = _fit_decay_models(generations, sigma_vec_mags)
            best_sigma = max(sigma_models, key=lambda m: m["r2_adj"])

            # 3. Individual sigma trends
            sigmas_across_gen = {}
            for key in ["fact", "syntax", "style", "safety", "coherence"]:
                vals = [s.individual_sigmas.get(key, 0.5) for s in snapshots]
                if len(vals) >= 3:
                    slope, r2 = _ols_fast(generations, np.array(vals))
                    sigmas_across_gen[key] = {
                        "gen0": round(vals[0], 4), "genN": round(vals[-1], 4),
                        "trend": "down" if slope < 0 else "up",
                        "slope_per_gen": round(float(slope), 4),
                    }
            print(f"    σ trends: {json.dumps(sigmas_across_gen)}")

            # 4. Validate
            val = validate_decay_model(snapshots)
            print(f"    ||Π|| fit: best={best_pi['model']}, β={best_pi['beta']:.4f}, R²={best_pi['r2']:.4f}")
            print(f"    σ-vec fit: best={best_sigma['model']}, β={best_sigma['beta']:.4f}, R²={best_sigma['r2']:.4f}")
            print(f"    bootstrap β CI: [{val['bootstrap']['ci_lower']:.4f}, {val['bootstrap']['ci_upper']:.4f}] "
                  f"width={val['bootstrap']['ci_width']:.4f}")
            print(f"    σ independence: score={val['sigma_correlation']['dimensionality_score']:.4f} "
                  f"({val['sigma_correlation']['interpretation']})")
            print(f"    verdict: {val['verdict']}")

            # Collect
            all_results.setdefault(model_name, {})[cap] = {
                "pi_best_model": best_pi["model"],
                "pi_beta": best_pi["beta"],
                "pi_r2": best_pi["r2"],
                "sigma_best_model": best_sigma["model"],
                "sigma_beta": best_sigma["beta"],
                "sigma_r2": best_sigma["r2"],
                "bootstrap_ci_width": val["bootstrap"]["ci_width"],
                "sigma_independence_score": val["sigma_correlation"]["dimensionality_score"],
                "sigma_trends": sigmas_across_gen,
            }

        # Also validate engine-level
        print(f"\n  Engine fit diagnostics:")
        for cap, diag in engine.get_fit_diagnostics().items():
            print(f"    {cap}: model={diag.get('_fit_model','?')} "
                  f"R²={diag.get('_fit_r2',0):.4f} "
                  f"target={diag.get('_fit_target','?')}")

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")

    # Aggregate stats
    all_pi_r2 = []
    all_sigma_r2 = []
    all_ci_widths = []
    expo_wins = 0
    linear_wins = 0
    power_wins = 0
    total_caps = 0

    for model, caps in all_results.items():
        for cap, r in caps.items():
            all_pi_r2.append(r["pi_r2"])
            all_sigma_r2.append(r["sigma_r2"])
            all_ci_widths.append(r["bootstrap_ci_width"])
            total_caps += 1
            if r["sigma_best_model"] == "exponential":
                expo_wins += 1
            elif r["sigma_best_model"] == "linear":
                linear_wins += 1
            else:
                power_wins += 1

    if total_caps == 0:
        print("\n  No capabilities analyzed (all lineage files missing or empty)")
        return

    print(f"\n  {total_caps} capability × model combinations analyzed")
    print(f"\n  Decay model fit (σ-vector):")
    print(f"    Exponential wins: {expo_wins}/{total_caps} ({expo_wins/total_caps:.0%})")
    print(f"    Linear wins:      {linear_wins}/{total_caps} ({linear_wins/total_caps:.0%})")
    print(f"    Power-law wins:   {power_wins}/{total_caps} ({power_wins/total_caps:.0%})")
    print(f"    Mean σ-vec R²:    {np.mean(all_sigma_r2):.4f} ± {np.std(all_sigma_r2):.4f}")
    print(f"    Mean ||Π|| R²:     {np.mean(all_pi_r2):.4f} ± {np.std(all_pi_r2):.4f}")
    print(f"    Mean bootstrap CI width: {np.mean(all_ci_widths):.4f}")

    # σ independence summary
    all_ind_scores = []
    for model, caps in all_results.items():
        for cap, r in caps.items():
            all_ind_scores.append(r["sigma_independence_score"])
    print(f"    Mean σ independence:     {np.mean(all_ind_scores):.4f} ± {np.std(all_ind_scores):.4f}")

    print(f"\n  Key findings:")
    if np.mean(all_sigma_r2) > np.mean(all_pi_r2):
        print(f"    ✓ σ-vector magnitude is a better target than ||Π|| for β fitting")
    else:
        print(f"    ✗ ||Π|| outperforms σ-vector (unexpected)")
    if expo_wins / total_caps > 0.5:
        print(f"    ✓ Exponential decay is the dominant model")
    else:
        print(f"    ⚠ Exponential not dominant — consider model-adaptive fitting")
    if np.mean(all_ci_widths) < 0.3:
        print(f"    ✓ Bootstrap CI reasonably tight (width < 0.3)")
    else:
        print(f"    ⚠ Bootstrap CI wide — need more seeds per model")

    # Save
    with open("experiment_data/algorithm_validation.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved: experiment_data/algorithm_validation.json")


def _ols_fast(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Simple OLS (slope, r2)."""
    x_mean = x.mean()
    y_mean = y.mean()
    ss_xx = np.sum((x - x_mean) ** 2)
    if ss_xx < 1e-10:
        return 0.0, 0.0
    slope = np.sum((x - x_mean) * (y - y_mean)) / ss_xx
    y_pred = slope * x + (y_mean - slope * x_mean)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y_mean) ** 2)
    r2 = 1 - ss_res / max(ss_tot, 1e-10)
    return float(slope), float(max(r2, 0.0))


if __name__ == "__main__":
    main()
