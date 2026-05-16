#!/usr/bin/env python3
"""Perplexity baseline for the StabilityBench paper.

Computes perplexity (or naturalness proxy) for each model's lineage text
at Gen0 through Gen3, fits an exponential decay rate beta_perp, and compares
the ranking against the constraint residual beta.

Approach (two-tier, logprobs-first):
  1. Continuation perplexity via gpt-4o-mini logprobs  (primary)
     - Send text as prompt, ask model to continue
     - Perplexity = exp(-mean(token_logprob)) on continuation tokens
  2. Naturalness rating 1-10 via gpt-4o-mini  (fallback)
     - Inverse mapping: ppl_proxy = 1000 / naturalness^2

Output: experiment_data/perplexity_baseline.json
"""

import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "synthetic_decay_monitor"))

from synthetic_decay_monitor.provider_adapter import (
    ProviderConfig, OpenAICompatibleAdapter,
)

EXPERIMENT_DATA = os.path.join(BASE_DIR, "experiment_data")
OUTPUT_PATH = os.path.join(EXPERIMENT_DATA, "perplexity_baseline.json")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Pilot: 3 representative models
PILOT_MODELS = [
    {
        "name": "gpt-4o-mini",
        "file": "experiment_data/n100/gpt-4o-mini_s100_lineage.jsonl",
        "summary_key": "gpt-4o-mini_s100",
    },
    {
        "name": "deepseek-v3",
        "file": "experiment_data/n100/deepseek-chat_s100_lineage.jsonl",
        "summary_key": "deepseek-chat_s100",
    },
    {
        "name": "claude-opus-4-7",
        "file": "experiment_data/latest_models/claude-opus-4-7_s100_lineage.jsonl",
        "summary_key": "claude-opus-4-7_s100",
    },
]

N_SEEDS = 10       # seeds to sample per model (10 seeds × 4 gens = 40 texts)
N_TOKENS = 15      # continuation tokens for logprobs-based perplexity
DELAY_SEC = 1.2    # rate limiting between API calls


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_api_key() -> str:
    """Load QuickRouter API key from .env."""
    env_file = os.path.join(BASE_DIR, ".env")
    if os.path.exists(env_file):
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                v = v.strip().strip('"').strip("'")
                os.environ.setdefault(k, v)
    return os.environ.get("QUICKROUTER_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "")


def load_lineage(filepath: str, max_seeds: int = N_SEEDS) -> list[dict]:
    """Load lineage JSONL entries for the first max_seeds unique seeds.

    Lineage entries have ids like G0_0000, G1_0000, G2_0000, G3_0000.
    This collects ALL generations for the first max_seeds seeds.
    """
    # Pass 1: discover unique seed IDs in order
    seed_ids = []
    seen = set()
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            raw_id = entry.get("id", "")
            # Extract trailing numeric seed ID (handles both formats:
            # "G0_0000" → "0000", "G1_claude-opus-4-7_s100_0000" → "0000")
            m = re.match(r'G\d_(?:.*_)?(\d+)$', raw_id)
            sid = m.group(1) if m else raw_id
            if sid not in seen:
                seed_ids.append(sid)
                seen.add(sid)

    target_seeds = set(seed_ids[:max_seeds])

    # Pass 2: collect all entries for target seeds
    entries = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            raw_id = entry.get("id", "")
            m = re.match(r'G\d_(?:.*_)?(\d+)$', raw_id)
            sid = m.group(1) if m else raw_id
            if sid in target_seeds:
                entries.append(entry)

    return entries


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def fit_perplexity_beta(values_by_gen: dict[int, float]) -> dict:
    """Fit ppl_k = ppl_0 * exp(beta_perp * k) using log-linear regression.

    Perplexity INCREASES with generation (more degraded text = harder to
    continue), so beta_perp > 0.  This is the OPPOSITE sign convention
    from constraint beta where C_k = C_0 * exp(-beta * k).

    Args:
        values_by_gen: {generation: mean_perplexity} for Gen0..Gen3

    Returns:
        dict with beta, r_squared, intercept, per_gen values
    """
    gens = sorted(values_by_gen.keys())
    ys = [values_by_gen[g] for g in gens]

    if len(gens) < 3:
        return {"beta": 0.001, "r_squared": 0.0, "gens": gens, "values": ys}

    # Check coefficient of variation — if nearly flat, return floor beta
    mv = mean(ys)
    if mv < 1e-6:
        return {"beta": 0.001, "r_squared": 0.0, "gens": gens, "values": ys}
    stdv = (sum((y - mv) ** 2 for y in ys) / len(ys)) ** 0.5
    if stdv / mv < 0.005:
        return {"beta": 0.001, "r_squared": 0.0, "gens": gens, "values": ys}

    # Log-linear fit: log(ppl_k) = log(ppl_0) + beta_perp * k
    # beta_perp = slope (positive when perplexity increases with k)
    log_ys = [math.log(max(y, 1e-10)) for y in ys]
    mx = mean(gens)
    my = mean(log_ys)
    num = sum((g - mx) * (ly - my) for g, ly in zip(gens, log_ys))
    den = sum((g - mx) ** 2 for g in gens)

    if den == 0:
        return {"beta": 0.001, "r_squared": 0.0, "gens": gens, "values": ys}

    slope = num / den
    # slope > 0 means perplexity increases with generation
    # slope < 0 means perplexity decreases (unexpected, floor it)
    beta = max(0.001, slope)
    intercept = my - slope * mx

    # R-squared
    ss_res = sum((ly - (intercept + slope * g)) ** 2 for g, ly in zip(gens, log_ys))
    ss_tot = sum((ly - my) ** 2 for ly in log_ys)
    r2 = max(0.0, 1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        "beta": round(beta, 6),
        "r_squared": round(r2, 4),
        "intercept": round(intercept, 4),
        "gens": gens,
        "values": [round(y, 4) for y in ys],
    }


def spearman_rho(xs: list[float], ys: list[float]) -> float:
    """Spearman rank correlation."""
    n = len(xs)
    if n < 3:
        return 0.0

    def rank(vals):
        sp = sorted((v, i) for i, v in enumerate(vals))
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n and sp[j][0] == sp[i][0]:
                j += 1
            avg_rank = 1.0 + (i + j - 1) / 2.0
            for k in range(i, j):
                ranks[sp[k][1]] = avg_rank
            i = j
        return ranks

    rx, ry = rank(xs), rank(ys)
    d2 = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    return 1.0 - (6.0 * d2) / (n * (n * n - 1.0))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 72)
    print("  Perplexity Baseline — StabilityBench")
    print("  Logprobs-first, naturalness-fallback")
    print("=" * 72)

    # Setup adapter
    api_key = load_api_key()
    if not api_key:
        print("[FATAL] No API key found. Check .env for QUICKROUTER_API_KEY.")
        return

    config = ProviderConfig(
        base_url="https://api.quickrouter.ai/v1",
        api_key=api_key,
        model="gpt-4o-mini",
        max_tokens=64,
        temperature=0.0,
        timeout_sec=90,
    )
    adapter = OpenAICompatibleAdapter(config)
    print(f"\nScoring API: {adapter.provider_name} / {adapter.model}")

    # Load constraint beta values from summary
    summary_path = os.path.join(EXPERIMENT_DATA, "latest_models", "all_models_summary.json")
    with open(summary_path) as f:
        summary = json.load(f)
    beta_lookup = {}
    for k, v in summary.items():
        beta_lookup[k] = v["global_beta"]

    # -----------------------------------------------------------------------
    # Phase 1: Score all texts
    # -----------------------------------------------------------------------
    all_model_results = {}

    for model_info in PILOT_MODELS:
        model_name = model_info["name"]
        filepath = os.path.join(BASE_DIR, model_info["file"])
        if not os.path.exists(filepath):
            print(f"\n  [{model_name}] FILE NOT FOUND: {filepath}")
            continue

        print(f"\n{'─' * 60}")
        print(f"  Model: {model_name}")
        print(f"  File:  {model_info['file']}")
        print(f"  Sampling {N_SEEDS} seeds × 4 generations = {N_SEEDS * 4} texts")

        entries = load_lineage(filepath, max_seeds=N_SEEDS)
        print(f"  Loaded {len(entries)} entries")

        # Score each text
        text_scores = []  # list of {id, gen, seed, ppl_score, method, raw}
        logprobs_ok = 0
        logprobs_fail = 0
        n_scored = 0

        for i, entry in enumerate(entries):
            text = entry.get("text", "")
            if not text or len(text) < 20:
                continue

            text_id = entry.get("id", f"unknown_{i}")
            generation = entry.get("generation", 0)

            # Attempt 1: logprobs-based continuation perplexity
            logprobs_result = adapter.continuation_perplexity(text, n_tokens=N_TOKENS)

            if logprobs_result is not None:
                ppl = logprobs_result["perplexity"]
                method = "logprobs"
                logprobs_ok += 1
                score_record = {
                    "id": text_id,
                    "generation": generation,
                    "perplexity": ppl,
                    "method": method,
                    "mean_logprob": logprobs_result["mean_logprob"],
                    "n_tokens": logprobs_result["n_tokens"],
                    "text_preview": text[:120],
                }
            else:
                # Fallback: naturalness rating
                logprobs_fail += 1
                nat_result = adapter.naturalness_score(text)
                ppl = nat_result["perplexity_proxy"]
                method = "naturalness"
                score_record = {
                    "id": text_id,
                    "generation": generation,
                    "perplexity": ppl,
                    "method": method,
                    "naturalness": nat_result["naturalness"],
                    "rationale": nat_result.get("rationale", ""),
                    "text_preview": text[:120],
                }

            text_scores.append(score_record)
            n_scored += 1

            if (n_scored) % 10 == 0:
                method_label = f"({logprobs_ok} lp / {logprobs_fail} nat)"
                print(f"    [{n_scored}/{len(entries)}] {method_label}")

            time.sleep(DELAY_SEC)

        print(f"    Done: {logprobs_ok} logprobs, {logprobs_fail} naturalness")

        # Group by generation, compute mean perplexity
        gen_ppls: dict[int, list[float]] = defaultdict(list)
        for s in text_scores:
            gen_ppls[s["generation"]].append(s["perplexity"])

        gen_means = {g: mean(ppls) for g, ppls in gen_ppls.items()}

        print(f"    Per-gen mean perplexity: " +
              " | ".join(f"G{g}: {gen_means[g]:.2f}" for g in sorted(gen_means)))

        # Fit beta (perplexity increases with generation)
        fit = fit_perplexity_beta(gen_means)

        # Get constraint beta
        constraint_beta = beta_lookup.get(model_info["summary_key"])
        if constraint_beta is None:
            # fuzzy match
            for k, v in beta_lookup.items():
                if model_info["summary_key"].replace("_s100", "") in k:
                    constraint_beta = v
                    break

        all_model_results[model_name] = {
            "model": model_name,
            "file": model_info["file"],
            "constraint_beta": constraint_beta,
            "perplexity_beta": fit["beta"],
            "perplexity_r_squared": fit["r_squared"],
            "gen_mean_perplexity": gen_means,
            "n_texts_scored": n_scored,
            "logprobs_ok": logprobs_ok,
            "logprobs_fail": logprobs_fail,
            "per_text_scores": text_scores,
        }

        print(f"    β_perp = {fit['beta']:.6f}  (R² = {fit['r_squared']:.3f})")
        print(f"    β_c    = {constraint_beta:.6f}" if constraint_beta else "    β_c    = N/A")

    # -----------------------------------------------------------------------
    # Phase 2: Compare rankings
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("  Comparison: β_constraint vs β_perplexity")
    print(f"{'=' * 60}")

    # Build vectors for models with both betas
    models_with_both = []
    c_betas = []
    p_betas = []

    for model_name, res in all_model_results.items():
        if res["constraint_beta"] is not None:
            models_with_both.append(model_name)
            c_betas.append(res["constraint_beta"])
            p_betas.append(res["perplexity_beta"])
            print(f"  {model_name:<22s}  β_c={res['constraint_beta']:.4f}  "
                  f"β_perp={res['perplexity_beta']:.4f}  "
                  f"(R²={res['perplexity_r_squared']:.3f})")
        else:
            print(f"  {model_name:<22s}  β_c=N/A  "
                  f"β_perp={res['perplexity_beta']:.4f}")

    if len(c_betas) >= 2:
        rho = spearman_rho(c_betas, p_betas)
        print(f"\n  Spearman ρ(β_constraint, β_perplexity) = {rho:.4f}")
        print(f"  n = {len(c_betas)} models")
        if abs(rho) < 0.7:
            print("  => β_perp and β_constraint rank models DIFFERENTLY")
            print("     (perplexity does not capture the same signal)")
        else:
            print("  => β_perp and β_constraint are HIGHLY CORRELATED")
            print("     (perplexity may be capturing the same underlying signal)")
    else:
        print("\n  [WARN] Not enough models with both betas for Spearman")

    # -----------------------------------------------------------------------
    # Phase 3: Save output
    # -----------------------------------------------------------------------
    output = {
        "config": {
            "method": "perplexity_baseline",
            "scoring_model": "gpt-4o-mini",
            "n_seeds": N_SEEDS,
            "n_tokens_logprobs": N_TOKENS,
            "pilot_models": [m["name"] for m in PILOT_MODELS],
        },
        "models": {k: {
            "model": v["model"],
            "file": v["file"],
            "constraint_beta": v["constraint_beta"],
            "perplexity_beta": v["perplexity_beta"],
            "perplexity_r_squared": v["perplexity_r_squared"],
            "gen_mean_perplexity": {str(g): round(ppl, 2)
                                    for g, ppl in v["gen_mean_perplexity"].items()},
            "n_texts_scored": v["n_texts_scored"],
            "logprobs_success_rate": (f"{v['logprobs_ok']}/"
                                      f"{v['logprobs_ok'] + v['logprobs_fail']}"),
        } for k, v in all_model_results.items()},
        "comparison": {
            "spearman_rho": round(rho, 4) if len(c_betas) >= 2 else None,
            "n_models": len(c_betas),
            "model_names": models_with_both,
            "constraint_betas": [round(b, 6) for b in c_betas],
            "perplexity_betas": [round(b, 6) for b in p_betas],
        },
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to: {OUTPUT_PATH}")

    # Also save detailed per-text scores
    detailed_path = os.path.join(EXPERIMENT_DATA, "perplexity_baseline_detailed.json")
    detailed = []
    for model_name, res in all_model_results.items():
        for s in res["per_text_scores"]:
            detailed.append({
                "model": model_name,
                **s,
            })
    with open(detailed_path, "w") as f:
        json.dump(detailed, f, indent=2)
    print(f"  Detailed scores: {detailed_path}")

    print(f"\n{'=' * 60}")
    print("  Perplexity baseline complete.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
