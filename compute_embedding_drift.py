#!/usr/bin/env python3
"""Compute embedding drift baseline for recursive LLM generation lineages.

For each model's recursive generation lineage, computes cosine similarity of
sentence embeddings vs. original prompt (Gen0), fits an exponential decay model,
and compares the resulting β_emb ranking against the constraint β ranking.

Output: experiment_data/embedding_drift.json
"""

import json
import os
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import spearmanr
from sentence_transformers import SentenceTransformer

# ── Configuration ────────────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(__file__), "experiment_data")
OUTPUT_PATH = os.path.join(DATA_DIR, "embedding_drift.json")

LINEAGE_FILES = {
    "gpt-4o-mini": "gpt-4o-mini_lineage.jsonl",
    "claude": "claude_lineage.jsonl",
    "gpt4o": "gpt4o_lineage.jsonl",
    "haiku": "haiku_lineage.jsonl",
    "opus": "opus_lineage.jsonl",
    "gpt-5_5": "gpt-5_5_lineage.jsonl",
}

# β (constraint) values from baseline_comparison.json for these models.
# These are the reference β values we compare against.
BASELINE_BETAS = {
    "gpt-4o-mini": 0.08848613155619972,
    "gpt4o":       0.09854234334687854,
    "claude":      0.10554868170776172,
    "opus":        0.11963765071658077,
    "haiku":       0.1467693553941845,
    "gpt-5_5":     0.01087094723806383,
}

MAX_SEEDS = 30  # use at most 30 seeds per model (all models have 12)

# ── Data loading ─────────────────────────────────────────────────────────────

def load_lineage(filepath: str) -> list[dict]:
    entries = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def group_by_seed(entries: list[dict]) -> dict[int, dict[int, str]]:
    """Group entries into {seed_id: {generation: text}}.

    Line id format: "G{gen}_{seed:04d}", e.g. "G0_0000", "G1_0007".
    """
    seeds: dict[int, dict[int, str]] = {}
    for e in entries:
        eid = e["id"]  # e.g. "G2_0005"
        gen_str, seed_str = eid.split("_")
        gen = int(gen_str[1:])  # strip "G" prefix
        seed_id = int(seed_str)
        seeds.setdefault(seed_id, {})[gen] = e["text"]
    return seeds


# ── Embedding drift computation ──────────────────────────────────────────────

def compute_embedding_drift(model_name: str) -> dict:
    """Compute β_emb (embedding drift coefficient) for one model."""
    filepath = os.path.join(DATA_DIR, LINEAGE_FILES[model_name])
    entries = load_lineage(filepath)
    seeds = group_by_seed(entries)

    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    # for each seed, compute sim(Gen0, Gen_k) for k = 1, 2, 3
    # also compute consecutive sims
    per_gen_cos_sim = {1: [], 2: [], 3: []}  # sim(Gen0, Gen_k)
    consecutive_sims = {1: []}  # sim(Gen{k-1}, Gen{k})
    gen0_gen3_sims = []  # sim(Gen0, Gen3) for total drift

    seed_ids = sorted(seeds.keys())[:MAX_SEEDS]
    n_seeds = len(seed_ids)
    n_empty_skipped = 0

    for sid in seed_ids:
        lineage = seeds[sid]
        if 0 not in lineage:
            continue
        gen0_text = lineage[0]
        emb_gen0 = embedder.encode(gen0_text)

        # check for empty texts in later generations
        for gen in [1, 2, 3]:
            if gen in lineage and (lineage[gen] is None or len(lineage[gen].strip()) == 0):
                n_empty_skipped += 1
                del lineage[gen]
            if gen in lineage and lineage[gen].strip() == gen0_text.strip():
                # Model failed to generate — text identical to prompt
                del lineage[gen]

        for gen in [1, 2, 3]:
            if gen not in lineage:
                continue
            emb = embedder.encode(lineage[gen])
            sim = float(np.dot(emb_gen0, emb) / (np.linalg.norm(emb_gen0) * np.linalg.norm(emb) + 1e-10))
            per_gen_cos_sim[gen].append(sim)

        for gen in [1, 2, 3]:
            if gen in lineage and (gen - 1) in lineage:
                emb_prev = embedder.encode(lineage[gen - 1])
                emb_cur = embedder.encode(lineage[gen])
                sim = float(np.dot(emb_prev, emb_cur) / (np.linalg.norm(emb_prev) * np.linalg.norm(emb_cur) + 1e-10))
                consecutive_sims.setdefault(gen, []).append(sim)

        if 0 in lineage and 3 in lineage:
            gen0_gen3_sims.append(per_gen_cos_sim[3][-1] if per_gen_cos_sim[3] else None)

    # Average cosine similarity vs. Gen0 per generation gap
    avg_cos_sim = {}
    for gen, sims in per_gen_cos_sim.items():
        if sims:
            avg_cos_sim[gen] = float(np.mean(sims))

    avg_consecutive = {}
    for gen, sims in consecutive_sims.items():
        if sims:
            avg_consecutive[gen] = float(np.mean(sims))

    avg_gen0_gen3 = float(np.mean(gen0_gen3_sims)) if gen0_gen3_sims else None

    n_total_later = len([e for e in entries if e["generation"] > 0])
    n_valid_remaining = sum(len(v) for v in per_gen_cos_sim.values())
    n_removed = n_total_later - n_valid_remaining

    warnings = []
    if n_empty_skipped > 0:
        warnings.append(f"Skipped {n_empty_skipped} entries with empty/identical-to-prompt text")

    # Detect degenerate lineages: all later generations removed (identical to prompt)
    degenerate = False
    if n_total_later > 0 and n_valid_remaining == 0:
        warnings.append(f"DEGENERATE LINEAGE: all {n_total_later} post-Gen0 entries are identical to prompt (recursive generation failed)")
        degenerate = True
    elif avg_cos_sim.get(1, 0.0) > 0.9999:
        warnings.append("DEGENERATE LINEAGE: all generations are near-identical to Gen0 (recursive generation failed)")
        degenerate = True

    # Fit exponential decay to cumulative similarity trajectory
    # S_k = A * exp(-β_emb * k)  where k = generation index
    # We use S_0 = 1.0 (self-similarity) plus S_1, S_2, S_3
    gens = np.array([0, 1, 2, 3], dtype=float)
    sims = np.array([1.0] + [avg_cos_sim.get(k, np.nan) for k in [1, 2, 3]], dtype=float)

    # Remove nan entries (if any generation has no data)
    mask = ~np.isnan(sims)
    gens_clean = gens[mask]
    sims_clean = sims[mask]

    beta_emb = None
    r_squared = None

    if degenerate:
        warnings.append("β_emb set to None — data unusable for drift analysis")
    elif len(gens_clean) >= 3:
        def exp_decay(k, A, beta):
            return A * np.exp(-beta * k)

        try:
            popt, pcov = curve_fit(exp_decay, gens_clean, sims_clean,
                                   p0=[1.0, 0.05], maxfev=10000,
                                   bounds=([0.8, 0.0], [1.2, 1.0]))
            A_fit, beta_fit = popt
            beta_emb = float(beta_fit)

            # compute R^2
            residuals = sims_clean - exp_decay(gens_clean, A_fit, beta_fit)
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((sims_clean - np.mean(sims_clean)) ** 2)
            if ss_tot > 0:
                r_squared = float(1 - ss_res / ss_tot)
        except Exception as e:
            warnings.append(f"curve_fit failed: {e}")
            beta_emb = None

    return {
        "model": model_name,
        "n_seeds": n_seeds,
        "n_empty_skipped": n_empty_skipped,
        "degenerate": degenerate,
        "warnings": warnings,
        "avg_cos_sim_vs_gen0": avg_cos_sim,
        "avg_consecutive_sims": avg_consecutive,
        "avg_gen0_gen3_sim": avg_gen0_gen3,
        "beta_emb": beta_emb,
        "fit_r_squared": r_squared,
        "fit_trajectory": {str(k): v for k, v in zip(gens_clean.tolist(), sims_clean.tolist())},
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("EMBEDDING DRIFT BASELINE (sentence-transformers: all-MiniLM-L6-v2)")
    print("=" * 70)

    results = {}
    for model_name in LINEAGE_FILES:
        print(f"\nProcessing {model_name}...")
        result = compute_embedding_drift(model_name)
        results[model_name] = result
        if result.get("warnings"):
            for w in result["warnings"]:
                print(f"  [WARN] {w}")
        if result["degenerate"]:
            print(f"  β_emb = EXCLUDED (degenerate lineage)")
        elif result["beta_emb"] is not None:
            print(f"  β_emb = {result['beta_emb']:.6f}  (R² = {result['fit_r_squared']:.4f})")
            print(f"  avg sim vs Gen0: {{{', '.join(f'{k}: {v:.4f}' for k, v in sorted(result['avg_cos_sim_vs_gen0'].items()))}}}")
        else:
            print(f"  β_emb = FAILED to fit")

    # ── Compare with constraint β ──
    print("\n" + "-" * 70)
    print("COMPARISON: β (constraint) vs β_emb (embedding drift)")
    print("-" * 70)

    # Build aligned lists for Spearman
    common_models = [m for m in results if m in BASELINE_BETAS and results[m]["beta_emb"] is not None and not results[m]["degenerate"]]
    excluded_models = [m for m in results if m in BASELINE_BETAS and (results[m]["degenerate"] or results[m]["beta_emb"] is None)]

    print(f"\n{'Model':<18} {'β (constraint)':>16} {'β_emb (embed)':>16} {'Δ':>10}")
    print("-" * 60)
    for m in common_models:
        bc = BASELINE_BETAS[m]
        be = results[m]["beta_emb"]
        delta = be - bc
        print(f"{m:<18} {bc:>16.6f} {be:>16.6f} {delta:>+10.6f}")

    if excluded_models:
        print(f"\nExcluded models (degenerate lineage): {', '.join(excluded_models)}")

    betas_constraint = [BASELINE_BETAS[m] for m in common_models]
    betas_emb = [results[m]["beta_emb"] for m in common_models]

    if len(common_models) >= 4:
        rho, pval = spearmanr(betas_constraint, betas_emb)
        print(f"\nSpearman ρ(β_constraint, β_emb) = {rho:.4f}  (p = {pval:.4f})")
    else:
        rho, pval = None, None
        print(f"\nNot enough models for Spearman correlation (need >= 4, got {len(common_models)})")

    # ── Save results ──
    output = {
        "method": "embedding_drift",
        "embedder": "all-MiniLM-L6-v2",
        "description": "Cosine similarity of sentence embeddings vs. Gen0 prompt, fitted with exponential decay",
        "per_model": results,
        "excluded_models": excluded_models,
        "spearman_vs_constraint_beta": {
            "rho": rho,
            "p_value": pval,
            "n_models": len(common_models),
            "models_compared": common_models,
        },
        "constraint_betas_used": {m: BASELINE_BETAS[m] for m in common_models},
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved results to: {OUTPUT_PATH}")

    # ── Section separator for downstream summary ──
    print("\n" + "=" * 70)
    print("EMBEDDING DRIFT DONE")
    print("=" * 70)

    return output


if __name__ == "__main__":
    main()
