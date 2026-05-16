#!/usr/bin/env python3
"""Compute direct LLM-judge baseline for recursive generation quality decay.

For each model's recursive generation lineage, uses GPT-4o-mini as a direct
quality judge (no constraint extraction), scores each generation's response on
a 1-10 scale, fits exponential decay to the quality trajectory, and compares
the resulting β_qual ranking against the constraint β ranking.

Output: experiment_data/direct_judge_baseline.json
"""

import json
import os
import sys
import time
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import spearmanr

# Add parent dir for imports
sys.path.insert(0, os.path.dirname(__file__))

# Load .env file if present
env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip())

from synthetic_decay_monitor.provider_adapter import create_provider, ProviderConfig, OpenAICompatibleAdapter


# ── Configuration ────────────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(__file__), "experiment_data")
OUTPUT_PATH = os.path.join(DATA_DIR, "direct_judge_baseline.json")
CACHE_PATH = os.path.join(DATA_DIR, "direct_judge_cache.json")

LINEAGE_FILES = {
    "gpt-4o-mini": "gpt-4o-mini_lineage.jsonl",
    "claude": "claude_lineage.jsonl",
    "gpt4o": "gpt4o_lineage.jsonl",
    "haiku": "haiku_lineage.jsonl",
    "opus": "opus_lineage.jsonl",
    "gpt-5_5": "gpt-5_5_lineage.jsonl",
}

# β (constraint) values from baseline_comparison.json
BASELINE_BETAS = {
    "gpt-4o-mini": 0.08848613155619972,
    "gpt4o":       0.09854234334687854,
    "claude":      0.10554868170776172,
    "opus":        0.11963765071658077,
    "haiku":       0.1467693553941845,
    "gpt-5_5":     0.01087094723806383,
}

JUDGE_MODEL = "gpt-4o-mini"
JUDGE_PROVIDER = "quickrouter"
N_SEEDS = 10        # sample 10 seeds per model
N_GENERATIONS = 4   # Gen0 - Gen3

JUDGE_PROMPT_TEMPLATE = (
    "Rate the following AI-generated text on a scale of 1-10 for overall quality, "
    "considering coherence, creativity, factual accuracy, and readability.\n\n"
    "Text: {text}\n\n"
    "Output only a JSON object: {{\"quality_score\": <1-10>}}"
)

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
    """Group entries into {seed_id: {generation: text}}."""
    seeds: dict[int, dict[int, str]] = {}
    for e in entries:
        eid = e["id"]
        gen_str, seed_str = eid.split("_")
        gen = int(gen_str[1:])
        seed_id = int(seed_str)
        seeds.setdefault(seed_id, {})[gen] = e["text"]
    return seeds


# ── Cache management ─────────────────────────────────────────────────────────

def load_cache() -> dict:
    """Load existing cache of judge scores, keyed by (model, seed, gen)."""
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH) as f:
            return json.load(f)
    return {}


def save_cache(cache: dict):
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def cache_key(model: str, seed_id: int, gen: int) -> str:
    return f"{model}|seed{seed_id}|gen{gen}"


# ── Judge scoring ────────────────────────────────────────────────────────────

def score_text(adapter, text: str, max_retries: int = 3) -> float | None:
    """Get quality score 1-10 for a single text from the judge LLM."""
    prompt = JUDGE_PROMPT_TEMPLATE.format(text=text[:3000])  # truncate for safety

    for attempt in range(max_retries):
        try:
            resp = adapter.generate(prompt, max_tokens=64, temperature=0.0)
            # Try to extract JSON
            resp_clean = resp.strip()
            # Find JSON object
            start = resp_clean.find("{")
            end = resp_clean.rfind("}")
            if start >= 0 and end > start:
                obj = json.loads(resp_clean[start:end + 1])
                score = float(obj.get("quality_score", -1))
                if 1 <= score <= 10:
                    return score
            # Fallback: look for a number
            import re
            nums = re.findall(r'\b([1-9]|10)\b', resp_clean)
            if nums:
                return float(nums[0])
            print(f"    [WARN] Could not parse score from: {resp[:100]}")
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2.0 ** attempt
                print(f"    [RETRY {attempt+1}] {e}, waiting {wait}s")
                time.sleep(wait)
            else:
                print(f"    [FAIL] Could not score after {max_retries} attempts: {e}")
    return None


def judge_model(model_name: str, adapter) -> dict:
    """Score all sampled texts for one model, with caching."""
    filepath = os.path.join(DATA_DIR, LINEAGE_FILES[model_name])
    entries = load_lineage(filepath)
    seeds = group_by_seed(entries)
    cache = load_cache()

    # Sample N_SEEDS from available seeds
    all_seed_ids = sorted(seeds.keys())
    sampled_seed_ids = all_seed_ids[:N_SEEDS]

    per_gen_scores = {g: [] for g in range(N_GENERATIONS)}
    total_scored = 0
    cache_hits = 0

    for sid in sampled_seed_ids:
        lineage = seeds[sid]
        for gen in range(N_GENERATIONS):
            if gen not in lineage:
                continue
            text = lineage[gen]
            if not text or len(text.strip()) == 0:
                continue
            # Skip entries with identical text to Gen0 (degenerate lineage)
            if gen > 0 and 0 in lineage and text.strip() == lineage[0].strip():
                continue

            ck = cache_key(model_name, sid, gen)

            if ck in cache:
                score = cache[ck]
                cache_hits += 1
            else:
                score = score_text(adapter, text)
                if score is not None:
                    cache[ck] = score
                    save_cache(cache)  # save after each successful request
                time.sleep(0.3)  # rate limiting

            if score is not None:
                per_gen_scores[gen].append(score)
                total_scored += 1

    print(f"  Scored: {total_scored} texts ({cache_hits} from cache)")

    # Compute average score per generation
    avg_scores = {}
    for gen, scores in per_gen_scores.items():
        if scores:
            avg_scores[gen] = float(np.mean(scores))

    return {
        "model": model_name,
        "n_seeds": N_SEEDS,
        "n_scored": total_scored,
        "cache_hits": cache_hits,
        "avg_quality_by_gen": avg_scores,
        "per_gen_n": {str(g): len(per_gen_scores[g]) for g in per_gen_scores},
        "per_gen_scores": {str(g): per_gen_scores[g] for g in per_gen_scores},
    }


# ── Exponential fit ──────────────────────────────────────────────────────────

def fit_quality_decay(avg_scores: dict[int, float]) -> tuple[float | None, float | None]:
    """Fit exponential decay: quality_k = Q0 * exp(-β_qual * k)."""
    gens = sorted(avg_scores.keys())
    if len(gens) < 3:
        return None, None

    ks = np.array(gens, dtype=float)
    vs = np.array([avg_scores[g] for g in gens], dtype=float)

    def exp_decay(k, Q0, beta):
        return Q0 * np.exp(-beta * k)

    try:
        # initial guess: Q0 ~ 8.0, beta ~ 0.03
        popt, _ = curve_fit(exp_decay, ks, vs, p0=[8.0, 0.03], maxfev=10000,
                            bounds=([1.0, 0.0], [10.0, 1.0]))
        Q0_fit, beta_fit = popt
        residuals = vs - exp_decay(ks, Q0_fit, beta_fit)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((vs - np.mean(vs)) ** 2)
        r_squared = float(1 - ss_res / ss_tot) if ss_tot > 0 else None
        return float(beta_fit), r_squared
    except Exception as e:
        print(f"    [WARN] Fit failed: {e}")
        return None, None


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print(f"DIRECT LLM-JUDGE BASELINE (judge: {JUDGE_MODEL} via {JUDGE_PROVIDER})")
    print("=" * 70)

    # Create the judge adapter
    print(f"\nInitializing judge: {JUDGE_MODEL} via {JUDGE_PROVIDER}...")
    api_key = os.environ.get("QUICKROUTER_API_KEY", "")
    if not api_key:
        print("ERROR: QUICKROUTER_API_KEY not set")
        sys.exit(1)
    adapter = create_provider(JUDGE_PROVIDER, model=JUDGE_MODEL, api_key=api_key,
                              max_tokens=64, temperature=0.0)
    print(f"  Using endpoint: {adapter.config.base_url}")

    # Judge each model
    raw_results = {}
    for model_name in LINEAGE_FILES:
        print(f"\n{'─'*50}")
        print(f"Judging {model_name}...")
        result = judge_model(model_name, adapter)
        raw_results[model_name] = result

        if result["avg_quality_by_gen"]:
            print(f"  avg scores: {{{', '.join(f'G{k}: {v:.2f}' for k, v in sorted(result['avg_quality_by_gen'].items()))}}}")
        else:
            print(f"  No valid scores!")

    # Fit exponential decay and compute β_qual
    print(f"\n{'='*70}")
    print("Fitting exponential decay: quality_k = Q0 * exp(-β_qual * k)")
    print("=" * 70)

    results = {}
    for model_name, r in raw_results.items():
        avg = r["avg_quality_by_gen"]
        beta_qual, r_sq = fit_quality_decay(avg)
        results[model_name] = {
            **r,
            "beta_qual": beta_qual,
            "fit_r_squared": r_sq,
        }
        if beta_qual is not None:
            print(f"  {model_name}: β_qual = {beta_qual:.6f}  (R² = {r_sq:.4f})")
        else:
            print(f"  {model_name}: β_qual = FAILED to fit")

    # ── Compare with constraint β ──
    print(f"\n{'─'*70}")
    print("COMPARISON: β (constraint) vs β_qual (direct judge)")
    print("─" * 70)

    common_models = [m for m in results if m in BASELINE_BETAS and results[m]["beta_qual"] is not None]
    failed_models = [m for m in results if m in BASELINE_BETAS and results[m]["beta_qual"] is None]

    print(f"\n{'Model':<18} {'β (constraint)':>16} {'β_qual (judge)':>16} {'Δ':>10}")
    print("-" * 60)
    for m in common_models:
        bc = BASELINE_BETAS[m]
        bq = results[m]["beta_qual"]
        delta = bq - bc
        print(f"{m:<18} {bc:>16.6f} {bq:>16.6f} {delta:>+10.6f}")

    if failed_models:
        print(f"\nFailed models: {', '.join(failed_models)}")

    betas_constraint = [BASELINE_BETAS[m] for m in common_models]
    betas_qual = [results[m]["beta_qual"] for m in common_models]

    if len(common_models) >= 4:
        rho, pval = spearmanr(betas_constraint, betas_qual)
        print(f"\nSpearman ρ(β_constraint, β_qual) = {rho:.4f}  (p = {pval:.4f})")
    else:
        rho, pval = None, None
        print(f"\nNot enough models for Spearman correlation (need >= 4, got {len(common_models)})")

    # ── Save results ──
    output = {
        "method": "direct_llm_judge",
        "judge_model": JUDGE_MODEL,
        "judge_provider": JUDGE_PROVIDER,
        "description": "Direct quality scoring (1-10) by GPT-4o-mini without constraint extraction,"
                       " fitted with exponential decay",
        "per_model": results,
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

    print("\n" + "=" * 70)
    print("DIRECT JUDGE DONE")
    print("=" * 70)

    return output


if __name__ == "__main__":
    main()
