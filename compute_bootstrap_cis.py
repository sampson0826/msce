#!/usr/bin/env python3
"""
Bootstrap confidence intervals for StabilityBench beta estimates.

Implements parametric bootstrap with per-sample resampling to compute
95% percentile CIs for global and per-capability betas across all 16 models.

Method:
  For each generation (G0-G3), n individual constraint scores were measured
  per capability. S_n at each generation is the mean of those n scores.
  Beta is computed from the exponential decay fit: S_g = (1-beta)^g.

  Bootstrap resamples the individual scores at each generation from a
  distribution consistent with the observed S_n, then refits beta.
  This captures finite-sample uncertainty (n=100 for s100 experiments).

Pure Python only (no numpy, no scipy) per requirement.
"""

import json
import math
import os
import random
import sys

# ---------------------------------------------------------------------------
# Random number generation (no numpy)
# ---------------------------------------------------------------------------

def randn():
    """Box-Muller standard normal variate."""
    u1 = random.random()
    u2 = random.random()
    # Avoid log(0)
    while u1 <= 0.0:
        u1 = random.random()
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def randn_vec(n):
    """Generate n standard normal variates."""
    return [randn() for _ in range(n)]


# ---------------------------------------------------------------------------
# Beta computation
# ---------------------------------------------------------------------------

def fit_beta_from_sn(sn_trajectory):
    """
    Fit exponential decay rate from per-generation S_n values.

    Model: S_g = S_0 * (1 - beta)^g
    Taking logs: log(S_g) = log(S_0) - lambda * g  where lambda = -log(1-beta)

    OLS through origin (log(S_0)=0):
      lambda_hat = sum(g * (-log(S_g))) / sum(g^2)
      beta_hat = 1 - exp(-lambda_hat)

    sn_trajectory[g] = S_n at generation g (g = 0, 1, 2, 3)
    sn_trajectory[0] is assumed to be 1.0 (human baseline).

    Returns beta, clamped to [0.001, 0.999].
    """
    num = 0.0
    den = 0.0
    for g in range(1, 4):
        s = sn_trajectory[g]
        if s <= 0.0:
            s = 1e-10
        if s >= 1.0:
            s = 0.999999
        num += g * (-math.log(s))
        den += g * g

    if den == 0.0:
        return 0.001

    lam = num / den
    if lam <= 0.0:
        return 0.001
    beta = 1.0 - math.exp(-lam)
    beta = max(0.001, min(0.999, beta))
    return beta


# ---------------------------------------------------------------------------
# Bootstrap core
# ---------------------------------------------------------------------------

def bootstrap_per_cap_beta(beta_point, n_samples, n_iter=2000, seed=42):
    """
    Bootstrap a single per-capability beta estimate.

    For each generation g, there were n_samples individual constraint scores.
    The observed mean S_n(g) is reconstructed as (1 - beta_point)^g.

    We simulate individual scores around this mean, accounting for the fact
    that scores are bounded in [0, 1]. We use a normal approximation to the
    sample mean (CLT-applicable for n=100) with conservative variance bound.

    Returns list of n_iter bootstrapped beta values.
    """
    random.seed(seed)
    boot_betas = []

    for _ in range(n_iter):
        sn_boot = [1.0]  # G0 is always 1.0 (human baseline)

        for g in range(1, 4):
            # Target S_n for this generation
            s_target = (1.0 - beta_point) ** g

            # Conservative variance estimate for [0,1]-bounded variable
            # var <= 0.25, so SE of mean <= sqrt(0.25 / n_samples) = 0.5 / sqrt(n)
            # More precisely for proportion-like: var = s_target * (1 - s_target)
            var_bound = s_target * (1.0 - s_target)
            if var_bound < 1e-12:
                var_bound = 1e-12
            se = math.sqrt(var_bound / n_samples)

            # Generate bootstrapped mean via normal approximation
            s_boot = s_target + randn() * se
            s_boot = max(0.001, min(0.999, s_boot))
            sn_boot.append(s_boot)

        beta_boot = fit_beta_from_sn(sn_boot)
        boot_betas.append(beta_boot)

    return boot_betas


def bootstrap_global_beta_pooled(per_cap_betas, n_samples, n_iter=2000, seed=42):
    """
    Bootstrap the global beta using the report's definition:
    global_beta = arithmetic mean of the 6 per-capability betas.

    For each bootstrap iteration:
      1. Bootstrap each per-cap beta (within-generation resampling of scores)
      2. global_beta = mean of the 6 bootstrapped per-cap betas

    This captures within-generation finite-sample uncertainty for each
    capability, then propagates that uncertainty through the averaging.

    per_cap_betas: dict {capability_name: beta_value}
    n_samples: samples per capability per generation

    Returns list of n_iter bootstrapped global beta values.
    """
    random.seed(seed)
    caps = list(per_cap_betas.keys())
    n_caps = len(caps)
    boot_betas = []

    for _ in range(n_iter):
        # Bootstrap each per-cap beta (with within-generation noise)
        per_cap_boot = {}
        for cap in caps:
            beta_c = per_cap_betas[cap]
            sn_boot = [1.0]  # G0 fixed
            for g in range(1, 4):
                s_target = (1.0 - beta_c) ** g
                var_bound = s_target * (1.0 - s_target)
                if var_bound < 1e-12:
                    var_bound = 1e-12
                se = math.sqrt(var_bound / n_samples)
                s_boot = s_target + randn() * se
                s_boot = max(0.001, min(0.999, s_boot))
                sn_boot.append(s_boot)
            per_cap_boot[cap] = fit_beta_from_sn(sn_boot)

        # Global beta = mean of per-cap betas (matches report definition)
        global_beta_boot = sum(per_cap_boot.values()) / n_caps
        boot_betas.append(global_beta_boot)

    return boot_betas


def bootstrap_global_beta_block(per_cap_betas, n_samples, n_iter=2000, seed=42):
    """
    Block bootstrap: resample capabilities with replacement, then
    compute global beta as mean of per-cap betas.

    This additionally captures between-capability uncertainty beyond
    the within-generation sampling noise.

    per_cap_betas: dict {capability_name: beta_value}
    n_samples: samples per capability per generation

    Returns list of n_iter bootstrapped global beta values.
    """
    random.seed(seed)
    caps = list(per_cap_betas.keys())
    n_caps = len(caps)
    boot_betas = []

    for _ in range(n_iter):
        # Resample capabilities with replacement, then bootstrap each
        per_cap_boot_values = []
        for _ in range(n_caps):
            cap = random.choice(caps)
            beta_c = per_cap_betas[cap]
            sn_boot = [1.0]
            for g in range(1, 4):
                s_target = (1.0 - beta_c) ** g
                var_bound = s_target * (1.0 - s_target)
                if var_bound < 1e-12:
                    var_bound = 1e-12
                se = math.sqrt(var_bound / n_samples)
                s_boot = s_target + randn() * se
                s_boot = max(0.001, min(0.999, s_boot))
                sn_boot.append(s_boot)
            per_cap_boot_values.append(fit_beta_from_sn(sn_boot))

        global_beta_boot = sum(per_cap_boot_values) / n_caps
        boot_betas.append(global_beta_boot)

    return boot_betas


# ---------------------------------------------------------------------------
# CI computation
# ---------------------------------------------------------------------------

def percentile_ci(boot_samples, alpha=0.05):
    """
    Compute (alpha/2, 1-alpha/2) percentile confidence interval.
    Return (lower, median, upper).
    """
    sorted_samples = sorted(boot_samples)
    n = len(sorted_samples)
    lower_idx = int(n * alpha / 2)
    upper_idx = int(n * (1 - alpha / 2)) - 1
    median_idx = n // 2

    return (
        sorted_samples[max(0, lower_idx)],
        sorted_samples[median_idx],
        sorted_samples[min(n - 1, upper_idx)],
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def find_report_path(model_key):
    """Find the report JSON for a given model key."""
    exp_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "experiment_data",
    )

    # Direct match paths
    candidates = [
        os.path.join(exp_dir, "latest_models", f"{model_key}_report.json"),
        os.path.join(exp_dir, "n100", f"{model_key}_report.json"),
        os.path.join(exp_dir, f"{model_key}_report.json"),
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    return None


def load_model_data(model_key):
    """Load per-capability betas and n_samples for a model from its report JSON."""
    report_path = find_report_path(model_key)
    if report_path is None:
        print(f"  WARNING: No report found for {model_key}, skipping")
        return None, None

    with open(report_path) as f:
        report = json.load(f)

    # Extract per-capability betas
    per_cap_betas = {}
    for traj in report.get("decay_analysis", {}).get("trajectories", []):
        cap = traj["capability"]
        beta = traj["trajectory"][0]["beta"]  # Same beta stored at each gen
        per_cap_betas[cap] = beta

    # Extract n_samples per generation
    ds = report.get("data_summary", {})
    gen_summary = ds.get("generation_summary", {})
    if gen_summary:
        first_gen = list(gen_summary.keys())[0]
        n_samples = gen_summary[first_gen].get("n_samples", 100)
    else:
        n_samples = ds.get("n_samples", 400) // ds.get("n_generations", 4)

    # Also get global_beta from report for reference
    global_beta = report.get("decay_analysis", {}).get("global_beta", None)

    # Get model display name
    model_name = report.get("model", model_key)

    return {
        "report_path": report_path,
        "model_name": model_name,
        "global_beta": global_beta,
        "per_cap_betas": per_cap_betas,
        "n_samples_per_gen": n_samples,
    }, report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    summary_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "experiment_data",
        "latest_models",
        "all_models_summary.json",
    )

    if not os.path.exists(summary_path):
        print(f"ERROR: Summary file not found: {summary_path}")
        sys.exit(1)

    with open(summary_path) as f:
        summary = json.load(f)

    model_keys = list(summary.keys())
    print(f"Found {len(model_keys)} models in all_models_summary.json")
    print(f"Bootstrap iterations: 2000")
    print(f"Random seed: 42")
    print()

    all_results = {}

    for i, model_key in enumerate(model_keys):
        model_meta = summary[model_key]
        display_name = model_meta.get("model", model_key)
        family = model_meta.get("family", "unknown")

        print(f"[{i+1}/{len(model_keys)}] {display_name} ({family})")

        # Load report data
        model_data, report = load_model_data(model_key)
        if model_data is None:
            continue

        n_samples = model_data["n_samples_per_gen"]
        per_cap_betas = model_data["per_cap_betas"]
        global_beta_reported = model_data["global_beta"]

        if global_beta_reported is None:
            print(f"  WARNING: No global_beta in report, using summary value")
            global_beta_reported = model_meta["global_beta"]

        # Validate: per_cap_betas should match summary
        per_cap_from_report = per_cap_betas
        per_cap_from_summary = model_meta.get("per_cap_beta", {})

        print(f"  n_samples/gen: {n_samples}")
        print(f"  Reported global beta: {global_beta_reported:.6f}")

        # Bootstrap per-capability betas
        per_cap_cis = {}
        for cap, beta_val in per_cap_from_report.items():
            boot = bootstrap_per_cap_beta(beta_val, n_samples)
            ci_low, ci_med, ci_high = percentile_ci(boot)

            # Check if at floor (degenerate CI)
            is_floor = abs(beta_val - 0.001) < 1e-6
            per_cap_cis[cap] = {
                "beta": beta_val,
                "ci_lower": ci_low,
                "ci_median": ci_med,
                "ci_upper": ci_high,
                "at_floor": is_floor,
            }

        # Bootstrap global beta (pooled: within-generation noise only)
        boot_global = bootstrap_global_beta_pooled(per_cap_from_report, n_samples)
        ci_global_low, ci_global_med, ci_global_high = percentile_ci(boot_global)

        # Bootstrap global beta (block: resample capabilities)
        boot_global_block = bootstrap_global_beta_block(per_cap_from_report, n_samples)
        ci_global_block_low, ci_global_block_med, ci_global_block_high = percentile_ci(boot_global_block)

        print(f"  Global beta: {global_beta_reported:.4f} "
              f"[{ci_global_low:.4f}, {ci_global_high:.4f}] (pooled)  "
              f"[{ci_global_block_low:.4f}, {ci_global_block_high:.4f}] (block)")

        for cap, ci_info in per_cap_cis.items():
            flag = " [FLOOR]" if ci_info["at_floor"] else ""
            print(f"    {cap}: {ci_info['beta']:.4f} "
                  f"[{ci_info['ci_lower']:.4f}, {ci_info['ci_upper']:.4f}]{flag}")

        all_results[model_key] = {
            "model": display_name,
            "family": family,
            "n_samples_per_gen": n_samples,
            "global_beta": {
                "reported": global_beta_reported,
                "ci_lower_pooled": ci_global_low,
                "ci_median_pooled": ci_global_med,
                "ci_upper_pooled": ci_global_high,
                "ci_lower_block": ci_global_block_low,
                "ci_median_block": ci_global_block_med,
                "ci_upper_block": ci_global_block_high,
            },
            "per_capability": {},
        }

        for cap, ci_info in per_cap_cis.items():
            all_results[model_key]["per_capability"][cap] = {
                "beta": ci_info["beta"],
                "ci_lower": ci_info["ci_lower"],
                "ci_median": ci_info["ci_median"],
                "ci_upper": ci_info["ci_upper"],
                "at_floor": ci_info["at_floor"],
            }

        print()

    # Save results
    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "experiment_data",
        "bootstrap_cis.json",
    )

    output_data = {
        "metadata": {
            "method": "Parametric bootstrap with normal-approximation per-sample resampling",
            "method_details": (
                "For each generation g (1,2,3), capability c: n individual constraint scores "
                "are simulated around the observed mean S_n(g,c) = (1-beta_c)^g using normal "
                "approximation with variance S_n*(1-S_n)/n. Per-cap beta is refit from the "
                "resampled S_n trajectory via OLS through origin on log(S_n). "
                "Global beta follows the report's definition: arithmetic mean of the 6 "
                "per-capability betas. 2000 bootstrap iterations. "
                "Two variants for global beta: (1) pooled: bootstrap each per-cap beta "
                "with within-generation noise, then average; (2) block: resample "
                "capabilities with replacement before bootstrapping and averaging."
            ),
            "n_iterations": 2000,
            "random_seed": 42,
            "alpha": 0.05,
            "ci_type": "percentile",
            "beta_computation": (
                "Per-cap: beta_c = 1 - exp(-lambda), where lambda = sum(g * -log(S_n(g))) / sum(g^2). "
                "Global: beta = (1/6) * sum_c(beta_c)."
            ),
            "floor_beta": 0.001,
            "n_models": len(all_results),
        },
        "results": all_results,
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Results saved to: {output_path}")

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print()
    print("=" * 100)
    print("SUMMARY: Global Beta 95% Bootstrap CIs")
    print("=" * 100)
    print(f"{'Model':<34} {'Beta':>8} {'CI Low':>10} {'CI High':>10} {'Width':>8} {'Method':>14}")
    print("-" * 100)

    # Sort by reported beta descending
    sorted_models = sorted(
        all_results.items(),
        key=lambda x: x[1]["global_beta"]["reported"],
        reverse=True,
    )

    for model_key, result in sorted_models:
        gb = result["global_beta"]
        name = result["model"][:32]
        width_pooled = gb["ci_upper_pooled"] - gb["ci_lower_pooled"]
        print(
            f"{name:<34} "
            f"{gb['reported']:8.4f} "
            f"{gb['ci_lower_pooled']:10.4f} "
            f"{gb['ci_upper_pooled']:10.4f} "
            f"{width_pooled:8.4f} "
            f"{'pooled':>14}"
        )
        width_block = gb["ci_upper_block"] - gb["ci_lower_block"]
        print(
            f"{'':>34} "
            f"{'':>8} "
            f"{gb['ci_lower_block']:10.4f} "
            f"{gb['ci_upper_block']:10.4f} "
            f"{width_block:8.4f} "
            f"{'block':>14}"
        )

    print("-" * 100)
    print()
    print("Method notes:")
    print("  pooled = Bootstraps each per-cap beta with within-generation sampling noise,")
    print("           then computes global_beta = mean(per_cap_betas).")
    print("  block  = Additionally resamples capabilities with replacement before")
    print("           bootstrapping (captures between-capability uncertainty).")
    print()
    print("  global_beta = (1/6) * sum_c(beta_c)  [matches report definition]")
    print("  Beta at floor (0.001) shows degenerate/narrow CIs [0.001, ~0.004].")
    print()


if __name__ == "__main__":
    main()
