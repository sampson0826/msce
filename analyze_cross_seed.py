#!/usr/bin/env python3
"""
Cross-seed-source validation comparison script.

Compares original beta values (from GPT-4o-mini seeds, stored in
all_models_summary.json) against cross-seed beta values (from DeepSeek-V3 seeds,
stored in individual *_cross_report.json files under cross_seed/).

Computes: Spearman rank correlation, Pearson r, mean absolute delta, max delta.
Prints a comparison table and saves results to cross_seed/cross_seed_comparison.json.
"""

import json
import math
import os
from datetime import datetime, timezone


# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SUMMARY_PATH = os.path.join(BASE_DIR, "experiment_data", "latest_models", "all_models_summary.json")
CROSS_SEED_DIR = os.path.join(BASE_DIR, "experiment_data", "cross_seed")
OUTPUT_PATH = os.path.join(CROSS_SEED_DIR, "cross_seed_comparison.json")

# Known models that are expected to eventually have cross-seed reports
EXPECTED_CROSS_MODELS = [
    "deepseek-chat",
    "deepseek-v4-flash",
    "gpt-4o-mini",
    "claude-sonnet-4-6",
    "claude-opus-4-7",
]

CAPABILITIES = [
    "math_reasoning",
    "code_generation",
    "factual_knowledge",
    "logical_consistency",
    "creative_writing",
    "general",
]


# --- Manual statistical implementations (no scipy) ---

def mean(values):
    """Arithmetic mean."""
    if not values:
        return float("nan")
    return sum(values) / len(values)


def pearson_r(xs, ys):
    """Pearson correlation coefficient."""
    n = len(xs)
    if n < 2:
        return float("nan")
    mx = mean(xs)
    my = mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return float("nan")
    return cov / (sx * sy)


def rankdata(values):
    """Return ranks of values (1-based, average rank for ties)."""
    n = len(values)
    indexed = sorted(enumerate(values), key=lambda t: t[1])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j + 2) / 2.0  # 1-based average
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def spearman_r(xs, ys):
    """Spearman rank correlation coefficient."""
    n = len(xs)
    if n < 2:
        return float("nan")
    rank_x = rankdata(xs)
    rank_y = rankdata(ys)
    return pearson_r(rank_x, rank_y)


# --- Data loading ---

def load_original_summary(path):
    """Load all_models_summary.json and build a lookup by canonical model name."""
    with open(path, "r") as f:
        data = json.load(f)

    lookup = {}
    for key, entry in data.items():
        # Strip "_s100" suffix from the key to get canonical model name
        canonical = key
        if canonical.endswith("_s100"):
            canonical = canonical[:-5]
        # Also use the model field itself (strip _s100 if present there too)
        model_field = entry.get("model", "")
        model_canonical = model_field
        if model_canonical.endswith("_s100"):
            model_canonical = model_canonical[:-5]
        lookup[canonical] = entry
        lookup[model_canonical] = entry

    return lookup


def load_cross_seed_reports(cross_seed_dir):
    """Load all *_cross_report.json files from the cross_seed directory."""
    reports = {}
    if not os.path.isdir(cross_seed_dir):
        return reports

    for fname in sorted(os.listdir(cross_seed_dir)):
        if not fname.endswith("_cross_report.json"):
            continue
        fpath = os.path.join(cross_seed_dir, fname)
        try:
            with open(fpath, "r") as f:
                report = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"  [WARNING] Could not read {fname}: {e}")
            continue

        model_name = report.get("model", "")
        if not model_name:
            print(f"  [WARNING] {fname} missing 'model' field, skipping")
            continue

        # Extract the seed source label from the filename
        # e.g., "deepseek-v3_cross_report.json" -> "deepseek-v3"
        seed_label = fname.replace("_cross_report.json", "")

        reports[model_name] = {
            "report": report,
            "seed_label": seed_label,
            "filename": fname,
        }

    return reports


# --- Analysis ---

def compute_stats(pairs):
    """
    Given a list of (model_label, original_beta, cross_beta) tuples,
    compute Spearman rho, Pearson r, mean abs delta, max delta.
    """
    if len(pairs) < 2:
        return {
            "n_pairs": len(pairs),
            "spearman_rho": None if len(pairs) < 2 else float("nan"),
            "pearson_r": None if len(pairs) < 2 else float("nan"),
            "mean_abs_delta": None,
            "max_delta": None,
            "max_delta_pair": None,
        }

    originals = [p[1] for p in pairs]
    crosses = [p[2] for p in pairs]
    deltas = [abs(c - o) for c, o in zip(crosses, originals)]
    max_idx = deltas.index(max(deltas))

    return {
        "n_pairs": len(pairs),
        "spearman_rho": spearman_r(originals, crosses),
        "pearson_r": pearson_r(originals, crosses),
        "mean_abs_delta": mean(deltas),
        "max_delta": max(deltas),
        "max_delta_pair": pairs[max_idx][0],
    }


def run_comparison(original_lookup, cross_reports):
    """Run the full cross-seed comparison."""

    # --- Model-level comparison (global_beta) ---
    model_pairs = []
    model_details = []

    for model_name, cr in sorted(cross_reports.items()):
        report = cr["report"]
        original_entry = original_lookup.get(model_name)
        if original_entry is None:
            print(f"  [WARNING] Model '{model_name}' not found in original summary, skipping")
            continue

        orig_beta = original_entry.get("global_beta")
        cross_beta = report.get("decay_analysis", {}).get("global_beta")

        if orig_beta is None or cross_beta is None:
            print(f"  [WARNING] Model '{model_name}' missing beta values, skipping")
            continue

        model_pairs.append((model_name, orig_beta, cross_beta))
        model_details.append({
            "model": model_name,
            "original_beta": orig_beta,
            "cross_beta": cross_beta,
            "delta": cross_beta - orig_beta,
            "abs_delta": abs(cross_beta - orig_beta),
            "seed_label": cr["seed_label"],
        })

    model_stats = compute_stats(model_pairs)

    # --- Capability-level comparison (per_cap_beta) ---
    cap_pairs = []
    cap_details = []

    for model_name, cr in sorted(cross_reports.items()):
        report = cr["report"]
        original_entry = original_lookup.get(model_name)
        if original_entry is None:
            continue

        orig_cap_betas = original_entry.get("per_cap_beta", {})
        cross_cap_betas = report.get("decay_analysis", {}).get("per_cap_beta", {})

        for cap in CAPABILITIES:
            orig_b = orig_cap_betas.get(cap)
            cross_b = cross_cap_betas.get(cap)
            if orig_b is None or cross_b is None:
                continue
            pair_label = f"{model_name}/{cap}"
            cap_pairs.append((pair_label, orig_b, cross_b))
            cap_details.append({
                "model": model_name,
                "capability": cap,
                "original_beta": orig_b,
                "cross_beta": cross_b,
                "delta": cross_b - orig_b,
                "abs_delta": abs(cross_b - orig_b),
            })

    cap_stats = compute_stats(cap_pairs)

    # --- Determine pending models ---
    available_models = set(cr["report"].get("model", "") for cr in cross_reports.values())
    # Also resolve any canonical names in EXPECTED_CROSS_MODELS
    pending = [m for m in EXPECTED_CROSS_MODELS if m not in available_models]

    return {
        "model_stats": model_stats,
        "model_details": model_details,
        "cap_stats": cap_stats,
        "cap_details": cap_details,
        "available_models": sorted(available_models),
        "pending_models": pending,
    }


# --- Output ---

def print_table(results):
    """Print a formatted comparison table."""
    print()
    print("=" * 90)
    print("  CROSS-SEED-SOURCE VALIDATION COMPARISON")
    print("  Original seeds: GPT-4o-mini  |  Cross seeds: DeepSeek-V3")
    print("=" * 90)

    # Model-level table
    model_details = results["model_details"]
    if model_details:
        print()
        print(f"  MODEL-LEVEL COMPARISON (global_beta) -- {len(model_details)} models")
        print(f"  {'Model':<28s} {'Orig beta':>10s}  {'Cross beta':>10s}  {'Delta':>10s}  {'|Delta|':>10s}")
        print(f"  {'-'*28}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
        for md in model_details:
            print(f"  {md['model']:<28s}  {md['original_beta']:10.4f}  {md['cross_beta']:10.4f}  {md['delta']:+10.4f}  {md['abs_delta']:10.4f}")

        ms = results["model_stats"]
        print()
        print(f"  --- Model-level aggregate (n={ms['n_pairs']}) ---")
        rho_str = f"{ms['spearman_rho']:.4f}" if ms['spearman_rho'] is not None else "N/A"
        r_str = f"{ms['pearson_r']:.4f}" if ms['pearson_r'] is not None else "N/A"
        mad_str = f"{ms['mean_abs_delta']:.4f}" if ms['mean_abs_delta'] is not None else "N/A"
        maxd_str = f"{ms['max_delta']:.4f}" if ms['max_delta'] is not None else "N/A"
        max_pair_str = ms.get('max_delta_pair', 'N/A') if ms.get('max_delta_pair') else 'N/A'
        print(f"  Spearman rank rho = {rho_str}")
        print(f"  Pearson r         = {r_str}")
        print(f"  Mean |Delta|      = {mad_str}")
        print(f"  Max  |Delta|      = {maxd_str}  ({max_pair_str})")

        # Qualitative interpretation
        if ms['spearman_rho'] is not None and ms['spearman_rho'] is not None:
            rho = ms['spearman_rho']
            if rho >= 0.9:
                interp = "Excellent agreement -- cross-seed ranking is highly consistent"
            elif rho >= 0.7:
                interp = "Good agreement -- cross-seed ranking is largely preserved"
            elif rho >= 0.4:
                interp = "Moderate agreement -- some reordering by seed source"
            else:
                interp = "Weak agreement -- seed source substantially changes ranking"
            print(f"  Interpretation: {interp}")

    # Capability-level table
    cap_details = results["cap_details"]
    if cap_details:
        print()
        print(f"  CAPABILITY-LEVEL COMPARISON (per_cap_beta) -- {len(cap_details)} model/cap pairs")
        print(f"  {'Pair':<40s} {'Orig':>8s}  {'Cross':>8s}  {'|Delta|':>8s}")
        print(f"  {'-'*40}  {'-'*8}  {'-'*8}  {'-'*8}")
        for cd in cap_details:
            pair_label = f"{cd['model']}/{cd['capability']}"
            print(f"  {pair_label:<40s}  {cd['original_beta']:8.4f}  {cd['cross_beta']:8.4f}  {cd['abs_delta']:8.4f}")

        cs = results["cap_stats"]
        print()
        print(f"  --- Capability-level aggregate (n={cs['n_pairs']}) ---")
        rho_str = f"{cs['spearman_rho']:.4f}" if cs['spearman_rho'] is not None else "N/A"
        r_str = f"{cs['pearson_r']:.4f}" if cs['pearson_r'] is not None else "N/A"
        mad_str = f"{cs['mean_abs_delta']:.4f}" if cs['mean_abs_delta'] is not None else "N/A"
        maxd_str = f"{cs['max_delta']:.4f}" if cs['max_delta'] is not None else "N/A"
        max_pair_str = cs.get('max_delta_pair', 'N/A') if cs.get('max_delta_pair') else 'N/A'
        print(f"  Spearman rank rho = {rho_str}")
        print(f"  Pearson r         = {r_str}")
        print(f"  Mean |Delta|      = {mad_str}")
        print(f"  Max  |Delta|      = {maxd_str}  ({max_pair_str})")

    # Pending
    pending = results.get("pending_models", [])
    if pending:
        print()
        print(f"  PENDING (reports not yet available): {', '.join(pending)}")

    print()
    print("=" * 90)


def save_results(results, output_path):
    """Save full comparison results to JSON."""
    output = {
        "comparison_metadata": {
            "original_seed_source": "GPT-4o-mini",
            "cross_seed_source": "DeepSeek-V3",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "n_models_compared": results["model_stats"]["n_pairs"],
            "models_available": results["available_models"],
            "models_pending": results["pending_models"],
        },
        "model_level": {
            "spearman_rho": results["model_stats"]["spearman_rho"],
            "pearson_r": results["model_stats"]["pearson_r"],
            "mean_abs_delta": results["model_stats"]["mean_abs_delta"],
            "max_delta": results["model_stats"]["max_delta"],
            "max_delta_pair": results["model_stats"]["max_delta_pair"],
            "pairs": results["model_details"],
        },
        "capability_level": {
            "spearman_rho": results["cap_stats"]["spearman_rho"],
            "pearson_r": results["cap_stats"]["pearson_r"],
            "mean_abs_delta": results["cap_stats"]["mean_abs_delta"],
            "max_delta": results["cap_stats"]["max_delta"],
            "max_delta_pair": results["cap_stats"]["max_delta_pair"],
            "pairs": results["cap_details"],
        },
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")


# --- Main ---

def main():
    print("Cross-Seed Validation Comparison")
    print(f"  Summary path:    {SUMMARY_PATH}")
    print(f"  Cross-seed dir:  {CROSS_SEED_DIR}")

    # Load original summary
    if not os.path.exists(SUMMARY_PATH):
        print(f"ERROR: Summary file not found at {SUMMARY_PATH}")
        return
    original_lookup = load_original_summary(SUMMARY_PATH)
    n_unique = len(set(id(v) for v in original_lookup.values()))
    print(f"  Loaded {n_unique} unique models from summary")

    # Load cross-seed reports
    cross_reports = load_cross_seed_reports(CROSS_SEED_DIR)
    print(f"  Found {len(cross_reports)} cross-seed report(s):")
    for model_name, cr in cross_reports.items():
        gb = cr["report"].get("decay_analysis", {}).get("global_beta", "?")
        print(f"    - {cr['filename']}  (model={model_name}, global_beta={gb:.4f})")

    if not cross_reports:
        print("\nNo cross-seed reports found. Nothing to compare.")
        return

    # Run comparison
    results = run_comparison(original_lookup, cross_reports)

    # Print table
    print_table(results)

    # Save results
    save_results(results, OUTPUT_PATH)


if __name__ == "__main__":
    main()
