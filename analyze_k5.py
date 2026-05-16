#!/usr/bin/env python3
"""
K=5 Depth Extension Analysis Script
====================================
Analyzes constraint decay trajectories from 3 DeepSeek models across
6 recursive generations (G0→G5).

For each model and each capability, fits both exponential and linear
decay models using ordinary least squares (manual implementation, no
scipy/numpy). Compares fit quality via R² and reports the better model.

Reads:  experiment_data/k5/{model_name}_k5_report.json
Writes: experiment_data/k5/k5_analysis.json
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
K5_DIR = os.path.join(BASE_DIR, "experiment_data", "k5")
OUTPUT_PATH = os.path.join(K5_DIR, "k5_analysis.json")

MODEL_NAMES = [
    "deepseek-v3_k5",
    "deepseek-v4-flash_k5",
    "deepseek-v4-pro_k5",
    "gpt-4o-mini_k5",
    "claude-opus-4-7_k5",
]

CAPABILITIES = [
    "math_reasoning",
    "code_generation",
    "factual_knowledge",
    "logical_consistency",
    "creative_writing",
    "general",
]


# ---------------------------------------------------------------------------
# Pure-Python statistics helpers (no numpy/scipy)
# ---------------------------------------------------------------------------

def mean(values):
    """Arithmetic mean of a sequence of numbers."""
    n = len(values)
    if n == 0:
        return 0.0
    return sum(values) / n


def pearson_r(xs, ys):
    """Pearson correlation coefficient (manual)."""
    n = len(xs)
    if n < 2:
        return 0.0
    mx = mean(xs)
    my = mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    var_x = sum((x - mx) ** 2 for x in xs)
    var_y = sum((y - my) ** 2 for y in ys)
    if var_x == 0 or var_y == 0:
        return 0.0
    return cov / math.sqrt(var_x * var_y)


def linear_regression(xs, ys):
    """
    Ordinary least squares linear regression: y = a + b*x

    Returns:
        (intercept, slope, r_squared)
    """
    n = len(xs)
    if n < 2:
        return 0.0, 0.0, 0.0

    mx = mean(xs)
    my = mean(ys)

    # slope = Σ((x - mx)(y - my)) / Σ((x - mx)^2)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)

    if den == 0:
        return my, 0.0, 0.0

    slope = num / den
    intercept = my - slope * mx

    # R-squared
    r = pearson_r(xs, ys)
    r_squared = r * r

    return intercept, slope, r_squared


# ---------------------------------------------------------------------------
# Decay model fitters
# ---------------------------------------------------------------------------

def fit_exponential_decay(generations, s_values):
    """
    Fit S_n = S_0 * exp(-beta * n)

    Linearized: log(S_n) = log(S_0) - beta * n
    Returns:
        (beta, S_0_fitted, r_squared)
    """
    n = len(generations)
    if n < 2:
        return 0.0, s_values[0] if s_values else 1.0, 0.0

    # Take log of S_n values (handle zeros / near-zeros safely)
    log_s = []
    valid_gens = []
    for g, s in zip(generations, s_values):
        if s > 1e-15:
            log_s.append(math.log(s))
            valid_gens.append(g)
        else:
            # Extremely small value -- skip to avoid math domain errors
            log_s.append(math.log(1e-15))
            valid_gens.append(g)

    if len(valid_gens) < 2:
        return 0.0, s_values[0], 0.0

    intercept, neg_beta, r2 = linear_regression(valid_gens, log_s)
    beta = -neg_beta  # slope is -beta
    S0_fitted = math.exp(intercept)

    return beta, S0_fitted, r2


def fit_linear_decay(generations, s_values):
    """
    Fit S_n = S_0 - alpha * n

    Returns:
        (alpha, S_0_fitted, r_squared)
    """
    n = len(generations)
    if n < 2:
        return 0.0, s_values[0] if s_values else 1.0, 0.0

    intercept, slope, r2 = linear_regression(generations, s_values)
    # slope is -alpha (since S_n = S_0 - alpha * n, intercept is S_0)
    alpha = -slope
    S0_fitted = intercept

    return alpha, S0_fitted, r2


# ---------------------------------------------------------------------------
# Trajectory extraction
# ---------------------------------------------------------------------------

def extract_trajectory(report_json, capability):
    """
    Extract the (generations, S_n_values) list for a given capability
    from a K5 report JSON.

    The report structure is expected to match the cross_seed report format:
      decay_analysis.trajectories[] → per-capability objects containing
      a 'trajectory' list of {generation, S_n, ...} dicts.

    If capability is "total_constraint_mean", looks for a top-level
    aggregate field or averages across all per-cap trajectories.
    """
    trajectories = report_json.get("decay_analysis", {}).get("trajectories", [])

    if capability == "total_constraint_mean":
        return extract_aggregate_trajectory(trajectories)

    # Find the matching per-capability trajectory
    for cap_obj in trajectories:
        if cap_obj.get("capability") == capability:
            points = cap_obj.get("trajectory", [])
            # Sort by generation number to ensure correct order
            points_sorted = sorted(points, key=lambda p: p.get("generation", 0))
            gens = [p["generation"] for p in points_sorted]
            s_vals = [p["S_n"] for p in points_sorted]
            return gens, s_vals

    # Fallback: not found
    return [], []


def extract_aggregate_trajectory(trajectories):
    """
    Compute the mean S_n across all capabilities at each generation.
    This produces the "total_constraint_mean" trajectory.

    Assumes all per-cap trajectories have the same generation indices.
    """
    if not trajectories:
        return [], []

    # Collect S_n values by generation index
    gen_to_values = {}
    for cap_obj in trajectories:
        for point in cap_obj.get("trajectory", []):
            g = point.get("generation", 0)
            s = point.get("S_n", 1.0)
            if g not in gen_to_values:
                gen_to_values[g] = []
            gen_to_values[g].append(s)

    if not gen_to_values:
        return [], []

    # Average across capabilities per generation
    gens_sorted = sorted(gen_to_values.keys())
    s_means = [mean(gen_to_values[g]) for g in gens_sorted]

    return gens_sorted, s_means


def load_report(model_name):
    """Load a K5 report JSON file. Returns the parsed dict or None."""
    path = os.path.join(K5_DIR, f"{model_name}_report.json")
    if not os.path.exists(path):
        print(f"  [WARN] Report not found: {path}")
        return None
    with open(path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Analysis runner
# ---------------------------------------------------------------------------

def analyze_model(model_name, report):
    """
    Analyze all capabilities + aggregate for a single model.

    Returns a dict keyed by capability name, each value being:
        {
            "trajectory": list of (gen, S_n),
            "exponential": {"beta": float, "S0_fitted": float, "r2": float},
            "linear":      {"alpha": float, "S0_fitted": float, "r2": float},
            "better_model": "exponential" | "linear",
            "delta_r2": float (exp_r2 - lin_r2: positive means exp better),
        }
    """
    results = {}

    for cap in CAPABILITIES + ["total_constraint_mean"]:
        gens, s_vals = extract_trajectory(report, cap)

        if len(gens) < 2:
            results[cap] = {
                "trajectory": list(zip(gens, s_vals)),
                "exponential": {"beta": None, "S0_fitted": None, "r2": None},
                "linear": {"alpha": None, "S0_fitted": None, "r2": None},
                "better_model": "insufficient_data",
                "delta_r2": None,
            }
            continue

        # Fit exponential
        beta, s0_exp, r2_exp = fit_exponential_decay(gens, s_vals)

        # Fit linear
        alpha, s0_lin, r2_lin = fit_linear_decay(gens, s_vals)

        # Determine better model
        if r2_exp is None or r2_lin is None:
            better = "insufficient_data"
            delta = None
        elif r2_exp >= r2_lin:
            better = "exponential"
            delta = r2_exp - r2_lin
        else:
            better = "linear"
            delta = r2_lin - r2_exp

        results[cap] = {
            "trajectory": list(zip(gens, s_vals)),
            "exponential": {"beta": beta, "S0_fitted": s0_exp, "r2": r2_exp},
            "linear": {"alpha": alpha, "S0_fitted": s0_lin, "r2": r2_lin},
            "better_model": better,
            "delta_r2": delta,
        }

    return results


# ---------------------------------------------------------------------------
# Pretty-printing
# ---------------------------------------------------------------------------

def fmt_r2(r2):
    """Format R² value for display."""
    if r2 is None:
        return "   N/A  "
    return f"{r2:+.6f}"


def fmt_param(val):
    """Format a decay parameter for display."""
    if val is None:
        return "   N/A   "
    return f"{val:+.6f}"


def print_summary_table(all_results):
    """Print a formatted summary table to stdout."""
    print()
    print("=" * 120)
    print("  K=5 DEPTH EXTENSION — DECAY MODEL COMPARISON")
    print("  Exponential: S_n = S_0 * exp(-beta*n)")
    print("  Linear:      S_n = S_0 - alpha*n")
    print("=" * 120)

    # ---- Per-model summary tables ----
    for model_name in MODEL_NAMES:
        if model_name not in all_results:
            continue
        results = all_results[model_name]

        print(f"\n{'─' * 100}")
        print(f"  Model: {model_name}")
        print(f"{'─' * 100}")
        header = (
            f"  {'Capability':<24s}"
            f" {'Exp β':>10s}  {'Exp R²':>10s}"
            f" {'Lin α':>10s}  {'Lin R²':>10s}"
            f" {'Better':>12s}  {'ΔR²':>10s}"
        )
        print(header)
        print("  " + "-" * 94)

        for cap in CAPABILITIES + ["total_constraint_mean"]:
            r = results.get(cap, {})
            exp = r.get("exponential", {})
            lin = r.get("linear", {})

            beta_str = fmt_param(exp.get("beta"))
            r2_exp_str = fmt_r2(exp.get("r2"))
            alpha_str = fmt_param(lin.get("alpha"))
            r2_lin_str = fmt_r2(lin.get("r2"))
            better = r.get("better_model", "?")
            delta = r.get("delta_r2")

            if delta is None:
                delta_str = "   N/A   "
            else:
                delta_str = f"{delta:+.6f}"

            # Visual indicator for which is better
            if better == "exponential":
                marker = "EXP ******"
            elif better == "linear":
                marker = "LIN ******"
            else:
                marker = better

            row = (
                f"  {cap:<24s}"
                f" {beta_str:>10s}  {r2_exp_str:>10s}"
                f" {alpha_str:>10s}  {r2_lin_str:>10s}"
                f" {marker:>12s}  {delta_str:>10s}"
            )
            print(row)

    # ---- Cross-model comparison: total_constraint_mean ----
    print(f"\n{'═' * 100}")
    print("  CROSS-MODEL COMPARISON — total_constraint_mean")
    print(f"{'═' * 100}")
    cross_header = (
        f"  {'Model':<28s}"
        f" {'Exp β':>10s}  {'Exp R²':>10s}"
        f" {'Lin α':>10s}  {'Lin R²':>10s}"
        f" {'Better':>12s}  {'ΔR²':>10s}"
    )
    print(cross_header)
    print("  " + "-" * 94)
    for model_name in MODEL_NAMES:
        if model_name not in all_results:
            continue
        r = all_results[model_name].get("total_constraint_mean", {})
        exp = r.get("exponential", {})
        lin = r.get("linear", {})
        better = r.get("better_model", "?")
        delta = r.get("delta_r2")
        delta_str = f"{delta:+.6f}" if delta is not None else "   N/A   "
        if better == "exponential":
            marker = "EXP ******"
        elif better == "linear":
            marker = "LIN ******"
        else:
            marker = better

        row = (
            f"  {model_name:<28s}"
            f" {fmt_param(exp.get('beta')):>10s}  {fmt_r2(exp.get('r2')):>10s}"
            f" {fmt_param(lin.get('alpha')):>10s}  {fmt_r2(lin.get('r2')):>10s}"
            f" {marker:>12s}  {delta_str:>10s}"
        )
        print(row)

    # ---- Per-capability cross-model comparison ----
    for cap in CAPABILITIES:
        print(f"\n{'─' * 100}")
        print(f"  Capability: {cap}")
        print(f"{'─' * 100}")
        cap_header = (
            f"  {'Model':<28s}"
            f" {'Exp β':>10s}  {'Exp R²':>10s}"
            f" {'Lin α':>10s}  {'Lin R²':>10s}"
            f" {'Better':>12s}  {'ΔR²':>10s}"
        )
        print(cap_header)
        print("  " + "-" * 94)
        for model_name in MODEL_NAMES:
            if model_name not in all_results:
                continue
            r = all_results[model_name].get(cap, {})
            exp = r.get("exponential", {})
            lin = r.get("linear", {})
            better = r.get("better_model", "?")
            delta = r.get("delta_r2")
            delta_str = f"{delta:+.6f}" if delta is not None else "   N/A   "
            if better == "exponential":
                marker = "EXP ******"
            elif better == "linear":
                marker = "LIN ******"
            else:
                marker = better

            row = (
                f"  {model_name:<28s}"
                f" {fmt_param(exp.get('beta')):>10s}  {fmt_r2(exp.get('r2')):>10s}"
                f" {fmt_param(lin.get('alpha')):>10s}  {fmt_r2(lin.get('r2')):>10s}"
                f" {marker:>12s}  {delta_str:>10s}"
            )
            print(row)

    # ---- Consensus summary ----
    print(f"\n{'═' * 100}")
    print("  CONSENSUS: How often does each model win?")
    print(f"{'═' * 100}")
    n_caps = len(CAPABILITIES) + 1  # +1 for total_constraint_mean
    for model_name in MODEL_NAMES:
        if model_name not in all_results:
            continue
        exp_wins = 0
        lin_wins = 0
        ties = 0
        for cap in CAPABILITIES + ["total_constraint_mean"]:
            better = all_results[model_name].get(cap, {}).get("better_model", "")
            if better == "exponential":
                exp_wins += 1
            elif better == "linear":
                lin_wins += 1
            else:
                ties += 1
        print(f"  {model_name:<28s}  Exponential: {exp_wins}/{n_caps}  "
              f"Linear: {lin_wins}/{n_caps}  Other: {ties}/{n_caps}")

    print("\n" + "=" * 120)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("K=5 Depth Extension Analysis")
    print(f"Reading reports from: {K5_DIR}")
    print(f"Models: {', '.join(MODEL_NAMES)}")

    all_results = {}

    for model_name in MODEL_NAMES:
        print(f"\nProcessing {model_name} ...")
        report = load_report(model_name)
        if report is None:
            all_results[model_name] = {"error": "report_not_found"}
            continue
        results = analyze_model(model_name, report)
        all_results[model_name] = results

        # Quick per-model summary
        for cap in CAPABILITIES + ["total_constraint_mean"]:
            r = results.get(cap, {})
            better = r.get("better_model", "?")
            delta = r.get("delta_r2")
            exp_r2 = r.get("exponential", {}).get("r2")
            lin_r2 = r.get("linear", {}).get("r2")
            if delta is not None:
                print(f"  {cap:<26s}  exp R²={exp_r2:+.6f}  "
                      f"lin R²={lin_r2:+.6f}  → {better} (ΔR²={delta:+.6f})")
            else:
                print(f"  {cap:<26s}  insufficient data")

    # Print formatted summary table
    print_summary_table(all_results)

    # ---- Save output ----
    output = {
        "analysis_metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "script": "analyze_k5.py",
            "description": "K=5 depth extension: exponential vs linear decay comparison",
            "models_analyzed": [m for m in MODEL_NAMES if m in all_results],
            "n_generations": 6,
            "n_capabilities": len(CAPABILITIES),
        },
        "results": all_results,
    }

    os.makedirs(K5_DIR, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nAnalysis saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
