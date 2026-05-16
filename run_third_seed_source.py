"""Third Seed Source Validation for StabilityBench.

Adds human-authored seeds (DEFAULT_SEEDS from experiment_config.py) as a third
seed source to the cross-seed validation. This strengthens the claim that β
ranking is a model property, not a seed artifact.

Three seed sources compared:
  1. Original:  GPT-4o-mini generated seeds (n=100, run_cross_seed_validation.py)
  2. Second:    DeepSeek-V3 generated seeds  (n=100, run_cross_seed_models.py)
  3. Human:     Hand-authored DEFAULT_SEEDS  (n=36,  this script)

Pipeline per model (5 models):
  Gen0: human-authored seed prompts
  Gen1: model generates from Gen0
  Gen2: model generates from Gen1
  Gen3: model generates from Gen2

After all 5 models complete, computes β via DecayEngine and prints a
3-way comparison table.

Usage:
    python run_third_seed_source.py

    # Skip generation, only re-run comparison with cached results:
    python run_third_seed_source.py --compare-only
"""
import sys, os, json, time
from datetime import datetime
from pathlib import Path

# ── Setup project root ──────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

env_path = os.path.join(PROJECT_ROOT, ".env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip())

from synthetic_decay_monitor.experiment_config import DEFAULT_SEEDS, CAPABILITIES
from synthetic_decay_monitor.data_lineage import DataSample, DatasetLineage, parse_lineage_from_jsonl
from synthetic_decay_monitor.provider_adapter import create_provider
from synthetic_decay_monitor.constraint_extractor import HybridConstraintExtractor
from synthetic_decay_monitor.decay_engine import DecayEngine
from synthetic_decay_monitor.executor_classifier import diagnose_executor_decay
from synthetic_decay_monitor.report import generate_json_report

# ── Configuration ───────────────────────────────────────────────────

OUTPUT_DIR = "experiment_data/cross_seed"
N_GENERATIONS = 3
N_SEEDS = 36  # matches DEFAULT_SEEDS: 6 per capability x 6 capabilities

# 5 representative models spanning the full β range
MODELS = [
    {"model": "gpt-4o-mini",      "provider": "quickrouter", "family": "OpenAI",
     "name": "human_seed_gpt-4o-mini",      "label": "gpt-4o-mini",      "delay": 3.0},
    {"model": "deepseek-chat",     "provider": "deepseek",    "family": "DeepSeek",
     "name": "human_seed_deepseek-v3",     "label": "deepseek-v3",     "delay": 0.5},
    {"model": "deepseek-v4-flash", "provider": "deepseek",    "family": "DeepSeek",
     "name": "human_seed_deepseek-v4-flash", "label": "deepseek-v4-flash", "delay": 0.5},
    {"model": "claude-sonnet-4-6", "provider": "quickrouter", "family": "Anthropic",
     "name": "human_seed_claude-sonnet-4-6", "label": "claude-sonnet-4-6", "delay": 3.0},
    {"model": "claude-opus-4-7",   "provider": "quickrouter", "family": "Anthropic",
     "name": "human_seed_claude-opus-4-7",   "label": "claude-opus-4-7",   "delay": 3.0},
]

# Beta values from the two existing seed sources (for 3-way comparison).
# Sourced from cross_seed_comparison.json and cross_seed_summary.json.
EXISTING_BETAS = {
    # From GPT-4o-mini seed source (original)
    "gpt_seeds": {
        "gpt-4o-mini":      0.0885,
        "deepseek-v3":      0.0281,
        "deepseek-v4-flash": 0.0350,
        "claude-sonnet-4-6": 0.1055,
        "claude-opus-4-7":   0.1635,
    },
    # From DeepSeek-V3 seed source (cross-seed)
    "deepseek_seeds": {
        "gpt-4o-mini":      0.0837,
        "deepseek-v3":      0.0572,
        "deepseek-v4-flash": 0.0430,
        "claude-sonnet-4-6": 0.0598,
        "claude-opus-4-7":   0.1468,
    },
}


# ── Helpers ─────────────────────────────────────────────────────────

def load_human_seeds() -> list[tuple[list[str], str]]:
    """Load human-authored DEFAULT_SEEDS as the third seed source.

    Returns list of (capability_tags, prompt_text) tuples.
    """
    seeds = [([tag], text) for tag, text in DEFAULT_SEEDS[:N_SEEDS]]
    print(f"Loaded {len(seeds)} human-authored seeds "
          f"({len(set(t[0] for t, _ in seeds))} capabilities, "
          f"{len(seeds) // len(CAPABILITIES)} per capability)")
    return seeds


def generate_with_retry(adapter, prompt: str, max_tokens: int = 512,
                        max_retries: int = 3) -> str:
    """Generate a response with exponential backoff retry."""
    for attempt in range(max_retries):
        try:
            resp = adapter.generate(prompt, max_tokens=max_tokens)
            if resp:
                return resp
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"    retry {attempt + 1}/{max_retries} in {wait}s: "
                      f"{type(e).__name__}")
                time.sleep(wait)
            else:
                print(f"    FAIL after {max_retries} retries: "
                      f"{type(e).__name__}")
    return "[EMPTY]"


def run_one_model(cfg: dict, seeds: list[tuple[list[str], str]]) -> dict:
    """Run recursive generation pipeline for a single model on human-authored seeds.

    Args:
        cfg: Model configuration dict with keys model, provider, name, label, delay.
        seeds: List of (capability_tags, prompt_text) tuples.

    Returns:
        Dict with name, model, family, global_beta, per_cap_beta, lineage_path, report_path.
    """
    model = cfg["model"]
    provider_name = cfg["provider"]
    name = cfg["name"]
    label = cfg["label"]
    delay = cfg["delay"]

    print(f"\n{'=' * 60}")
    print(f"[HUMAN SEED] {label} ({provider_name} / {model})")
    print(f"{'=' * 60}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    adapter = create_provider(provider_name, model=model, temperature=0.8,
                              timeout_sec=120)

    # ── Build Gen0 samples ──────────────────────────────────────
    samples: list[DataSample] = []
    for i, (tags, text) in enumerate(seeds):
        samples.append(DataSample(
            text=text,
            generation=0,
            source_model="human",
            capability_tags=tags,
            sample_id=f"HS_G0_{i:04d}",
        ))

    # ── Recursive generation: Gen1 → Gen2 → Gen3 ────────────────
    prev_texts = [s[1] for s in seeds]
    prev_tags = [[s[0]] for s in seeds]

    for gen in range(1, N_GENERATIONS + 1):
        n_prompts = len(prev_texts)
        print(f"  [Gen {gen}/{N_GENERATIONS}] generating {n_prompts} responses ...")
        curr_texts = []
        for i, text in enumerate(prev_texts):
            resp = generate_with_retry(adapter, text)
            curr_texts.append(resp)
            samples.append(DataSample(
                text=resp,
                generation=gen,
                source_model=model,
                capability_tags=prev_tags[i],
                sample_id=f"HS_G{gen}_{label}_{i:04d}",
            ))
            # Progress report every 10 samples
            if (i + 1) % 10 == 0 or i == n_prompts - 1:
                elapsed = sum(len(t) for t in curr_texts)
                print(f"    [{i + 1}/{n_prompts}] done "
                      f"({elapsed} chars total)")
            time.sleep(delay)
        prev_texts = curr_texts
        prev_tags = [[s.capability_tags[0]] if s.capability_tags else ["general"]
                      for s in samples if s.generation == gen]

    # ── Save lineage ────────────────────────────────────────────
    lineage = DatasetLineage(samples=samples)
    lineage_path = os.path.join(OUTPUT_DIR, f"{name}_lineage.jsonl")
    lineage.save(lineage_path)
    print(f"  Saved lineage: {lineage_path} ({len(samples)} samples)")

    # ── Analysis: DecayEngine → β ──────────────────────────────
    extractor = HybridConstraintExtractor(judge_fn=None)
    engine = DecayEngine(lineage, extractor)
    engine.run_all_capabilities()
    trajectories = engine.get_all_trajectories()

    per_cap_beta: dict[str, float] = {}
    for t in trajectories:
        if "trajectory" not in t:
            continue
        for g in t["trajectory"]:
            if "beta" in g and g["beta"] > 0:
                per_cap_beta[t["capability"]] = g["beta"]
                break

    betas = list(per_cap_beta.values())
    global_beta = sum(betas) / len(betas) if betas else 0.0

    # ── Diagnoses ──────────────────────────────────────────────
    diagnoses = []
    for cap, snapshots in engine._snapshots.items():
        diag = diagnose_executor_decay(
            engine._trajectories.get(cap, []), snapshots, capability=cap
        )
        diagnoses.append(diag)

    # ── Save report ────────────────────────────────────────────
    report = generate_json_report(
        lineage=lineage, trajectories=trajectories, diagnoses=diagnoses
    )
    report["decay_analysis"]["global_beta"] = global_beta
    report["decay_analysis"]["per_cap_beta"] = per_cap_beta
    report["decay_analysis"]["_method"] = (
        "third-seed-source: human-authored DEFAULT_SEEDS "
        "+ exponential + total_constraint"
    )
    report["model"] = model
    report["provider"] = provider_name
    report["family"] = cfg.get("family", "")
    report["seed_source"] = "human-authored (DEFAULT_SEEDS)"

    report_path = os.path.join(OUTPUT_DIR, f"{name}_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  Saved report: {report_path}")

    print(f"  => {label} global_beta = {global_beta:.4f}")
    for cap, b in sorted(per_cap_beta.items()):
        print(f"     {cap}: {b:.4f}")

    return {
        "name": name,
        "model": model,
        "label": label,
        "family": cfg["family"],
        "global_beta": global_beta,
        "per_cap_beta": per_cap_beta,
        "lineage_path": lineage_path,
        "report_path": report_path,
    }


def compute_rank_correlation(betas_a: dict[str, float],
                             betas_b: dict[str, float]) -> dict:
    """Compute Spearman rho and Pearson r between two β rankings."""
    import numpy as np
    common = sorted(set(betas_a) & set(betas_b))
    if len(common) < 2:
        return {"spearman_rho": None, "pearson_r": None, "n_common": len(common)}

    a_vals = np.array([betas_a[k] for k in common])
    b_vals = np.array([betas_b[k] for k in common])

    # Spearman rank
    from scipy.stats import spearmanr as sp_rho, pearsonr as pr_r
    try:
        rho, p_rho = sp_rho(a_vals, b_vals)
    except Exception:
        rho, p_rho = None, None
    try:
        r, p_r = pr_r(a_vals, b_vals)
    except Exception:
        r, p_r = None, None

    return {
        "spearman_rho": float(rho) if rho is not None else None,
        "spearman_p": float(p_rho) if p_rho is not None else None,
        "pearson_r": float(r) if r is not None else None,
        "pearson_p": float(p_r) if p_r is not None else None,
        "n_common": len(common),
        "pairs": {k: {"a": betas_a[k], "b": betas_b[k]} for k in common},
    }


def print_3way_table(human_results: dict[str, dict]):
    """Print a 3-way seed-source comparison table."""
    gpt_seeds = EXISTING_BETAS["gpt_seeds"]
    ds_seeds = EXISTING_BETAS["deepseek_seeds"]

    # Build map from model label → human-seed beta
    human_betas = {r["label"]: r["global_beta"] for r in human_results.values()}

    models_order = ["gpt-4o-mini", "deepseek-v3", "deepseek-v4-flash",
                    "claude-sonnet-4-6", "claude-opus-4-7"]

    print("\n" + "=" * 95)
    print("  3-WAY SEED-SOURCE COMPARISON: Model-Level β")
    print("=" * 95)
    header = (f"  {'Model':<22} {'GPT-4o-mini seeds':>18} "
              f"{'DeepSeek-V3 seeds':>18} {'Human seeds':>18} "
              f"{'Δ max':>10}")
    print(header)
    print("  " + "-" * 88)

    all_deltas = []
    for m in models_order:
        b1 = gpt_seeds.get(m, None)
        b2 = ds_seeds.get(m, None)
        b3 = human_betas.get(m, None)
        if b1 is None or b2 is None or b3 is None:
            print(f"  {m:<22} {'N/A':>18} {'N/A':>18} {'N/A':>18}")
            continue
        delta = max(abs(b1 - b2), abs(b1 - b3), abs(b2 - b3))
        all_deltas.append(delta)
        print(f"  {m:<22} {b1:>18.4f} {b2:>18.4f} {b3:>18.4f} "
              f"{delta:>10.4f}")

    print("  " + "-" * 88)

    # Rank correlations
    human_for_corr = {k: human_betas[k] for k in models_order if k in human_betas}
    gpt_for_corr = {k: gpt_seeds[k] for k in models_order if k in gpt_seeds}
    ds_for_corr = {k: ds_seeds[k] for k in models_order if k in ds_seeds}

    r_gh = compute_rank_correlation(gpt_for_corr, human_for_corr)
    r_dh = compute_rank_correlation(ds_for_corr, human_for_corr)
    r_gd = compute_rank_correlation(gpt_for_corr, ds_for_corr)

    print(f"\n  Rank Correlations (Spearman ρ):")
    print(f"    GPT-4o-mini seeds ↔ Human seeds:    ρ = {r_gh.get('spearman_rho', 'N/A')}")
    print(f"    DeepSeek-V3 seeds ↔ Human seeds:    ρ = {r_dh.get('spearman_rho', 'N/A')}")
    print(f"    GPT-4o-mini seeds ↔ DeepSeek seeds: ρ = {r_gd.get('spearman_rho', 'N/A')}")

    if all_deltas:
        print(f"\n  Max pairwise Δ across all models: {max(all_deltas):.4f}")
        print(f"  Mean pairwise Δ across all models: "
              f"{sum(all_deltas) / len(all_deltas):.4f}")


# ── Main ────────────────────────────────────────────────────────────

def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="Third Seed Source Validation: Human-authored seeds"
    )
    ap.add_argument("--compare-only", action="store_true",
                    help="Skip generation, only re-run comparison with cached reports")
    ap.add_argument("--model", type=str, default="",
                    help="Run only a specific model label (e.g. 'gpt-4o-mini')")
    args = ap.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Load human-authored seeds ──────────────────────────────
    seeds = load_human_seeds()

    # ── Filter models if --model specified ─────────────────────
    models_to_run = MODELS
    if args.model:
        models_to_run = [m for m in MODELS if m["label"] == args.model]
        if not models_to_run:
            print(f"Unknown model '{args.model}'. Available: "
                  f"{[m['label'] for m in MODELS]}")
            sys.exit(1)

    # ── Run generation + analysis ──────────────────────────────
    results: dict[str, dict] = {}

    if not args.compare_only:
        for cfg in models_to_run:
            try:
                r = run_one_model(cfg, seeds)
                results[r["name"]] = r
                print(f"  [OK] {r['label']} β = {r['global_beta']:.4f}")
            except Exception as e:
                print(f"  [FAIL] {cfg['label']}: {e}")
                import traceback
                traceback.print_exc()
                results[cfg["name"]] = {
                    "name": cfg["name"],
                    "model": cfg["model"],
                    "label": cfg["label"],
                    "family": cfg.get("family", ""),
                    "global_beta": None,
                    "per_cap_beta": {},
                    "error": str(e),
                }

        # ── Save aggregate summary ─────────────────────────────
        summary_path = os.path.join(OUTPUT_DIR, "human_seed_comparison.json")
        summary = {
            "comparison_metadata": {
                "seed_source": "human-authored (DEFAULT_SEEDS)",
                "n_seeds": N_SEEDS,
                "n_capabilities": len(CAPABILITIES),
                "n_generations": N_GENERATIONS,
                "generated_at": datetime.now().isoformat(),
                "n_models_run": len(results),
            },
            "results": {k: {
                "label": v.get("label", ""),
                "model": v.get("model", ""),
                "family": v.get("family", ""),
                "global_beta": v.get("global_beta"),
                "per_cap_beta": v.get("per_cap_beta", {}),
            } for k, v in results.items()},
            "existing_seed_sources": EXISTING_BETAS,
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\nSaved aggregate summary: {summary_path}")

    else:
        # Compare-only mode: load results from saved reports
        summary_path = os.path.join(OUTPUT_DIR, "human_seed_comparison.json")
        if not os.path.exists(summary_path):
            print(f"No cached results found at {summary_path}. "
                  f"Run without --compare-only first.")
            sys.exit(1)
        with open(summary_path) as f:
            cached = json.load(f)
        results = cached.get("results", {})
        print(f"Loaded {len(results)} cached results from {summary_path}")

    # ── Print 3-way comparison ─────────────────────────────────
    print_3way_table(results)

    print("\n" + "=" * 95)
    print("  Interpretation:")
    print("  - If human-authored seeds (第3源) produce the same β ranking as")
    print("    GPT-4o-mini and DeepSeek-V3 seeds, this strongly supports β as a")
    print("    model property independent of seed source.")
    print("  - Spearman ρ > 0.7 across all three pairwise comparisons would")
    print("    constitute strong evidence of seed-source invariance.")
    print("=" * 95)

    return 0


if __name__ == "__main__":
    sys.exit(main())
