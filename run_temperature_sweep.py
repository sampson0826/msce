"""Temperature Robustness Expansion for StabilityBench.

Expands the temperature robustness analysis from 1 model (GPT-4o-mini) to 3 models
spanning the full β range (low, mid, high). Tests each model at 3 temperature
settings: T=0.0, T=0.8, T=1.0.

This directly addresses the reviewer concern: "Does temperature robustness
generalize beyond GPT-4o-mini?"

Models tested:
  - GPT-4o-mini    (low β, ~0.085)   — original baseline
  - DeepSeek-V3    (mid β, ~0.043)   — mid-range representative
  - Claude-Opus-4-7 (high β, ~0.155) — upper range representative

Pipeline per (model, temperature) combination:
  Gen0: 15 human-authored seed prompts (balanced across 6 capabilities)
  Gen1: model generates from Gen0
  Gen2: model generates from Gen1
  Gen3: model generates from Gen2
  → Compute β via DecayEngine

Usage:
    python run_temperature_sweep.py

    # Run a specific model only:
    python run_temperature_sweep.py --model gpt-4o-mini

    # Run a specific temperature only:
    python run_temperature_sweep.py --temp 0.0
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
from synthetic_decay_monitor.data_lineage import DataSample, DatasetLineage
from synthetic_decay_monitor.provider_adapter import create_provider
from synthetic_decay_monitor.constraint_extractor import HybridConstraintExtractor
from synthetic_decay_monitor.decay_engine import DecayEngine
from synthetic_decay_monitor.executor_classifier import diagnose_executor_decay
from synthetic_decay_monitor.report import generate_json_report

# ── Configuration ───────────────────────────────────────────────────

OUTPUT_DIR = "experiment_data/temperature_sweep"
N_GENERATIONS = 3
N_SEEDS = 15  # subset of DEFAULT_SEEDS, balanced across 6 capabilities
TEMPERATURES = [0.0, 0.8, 1.0]

# 3 models spanning the β range (low, mid, high)
MODELS = [
    {"model": "gpt-4o-mini",      "provider": "quickrouter", "family": "OpenAI",
     "label": "gpt-4o-mini",      "delay": 3.0, "beta_range": "low"},
    {"model": "deepseek-chat",     "provider": "deepseek",    "family": "DeepSeek",
     "label": "deepseek-v3",     "delay": 0.5, "beta_range": "mid"},
    {"model": "claude-opus-4-7",   "provider": "quickrouter", "family": "Anthropic",
     "label": "claude-opus-4-7",   "delay": 3.0, "beta_range": "high"},
]


# ── Seed selection ──────────────────────────────────────────────────

def select_balanced_seeds(n: int = N_SEEDS) -> list[tuple[str, str]]:
    """Select n seeds from DEFAULT_SEEDS balanced across all capabilities.

    Uses round-robin across capabilities for balanced distribution.
    Returns list of (capability, prompt_text) tuples.
    """
    # Group seeds by capability
    by_cap: dict[str, list[tuple[str, str]]] = {}
    for tag, text in DEFAULT_SEEDS:
        by_cap.setdefault(tag, []).append((tag, text))

    # Round-robin selection
    selected = []
    cap_order = CAPABILITIES
    per_cap_idx = {cap: 0 for cap in cap_order}
    while len(selected) < n:
        for cap in cap_order:
            if len(selected) >= n:
                break
            idx = per_cap_idx[cap]
            if idx < len(by_cap.get(cap, [])):
                selected.append(by_cap[cap][idx])
                per_cap_idx[cap] += 1

    cap_counts = {}
    for tag, _ in selected:
        cap_counts[tag] = cap_counts.get(tag, 0) + 1
    print(f"Selected {len(selected)} seeds balanced across "
          f"{len(cap_counts)} capabilities: {cap_counts}")
    return selected


# ── Helpers ─────────────────────────────────────────────────────────

def generate_with_temperature_retry(
    adapter, prompt: str, temperature: float,
    max_tokens: int = 512, max_retries: int = 4
) -> str:
    """Generate a response with retry logic, handling T=0.0 edge cases.

    Temperature=0.0 may cause empty responses for reasoning models
    (e.g., DeepSeek-R1, Gemini 2.5 Pro). We detect this and retry
    with slight perturbation.

    Returns the generated text, or "[EMPTY]" after all retries exhausted.
    """
    for attempt in range(max_retries):
        try:
            resp = adapter.generate(prompt, max_tokens=max_tokens,
                                    temperature=temperature)
            if resp and len(resp.strip()) > 1:
                return resp

            # Empty/short response on T=0.0: retry with minimal jitter
            if temperature == 0.0 and attempt < max_retries - 1:
                wait = 2.0 * (attempt + 1)
                print(f"      (T=0.0 empty response, retry {attempt + 1}/{max_retries - 1} "
                      f"in {wait:.0f}s)")
                time.sleep(wait)
                # On last retry, try with T=0.01 as fallback
                if attempt == max_retries - 2:
                    print(f"      (T=0.01 fallback on final attempt)")
                    try:
                        resp = adapter.generate(prompt, max_tokens=max_tokens,
                                                temperature=0.01)
                        if resp and len(resp.strip()) > 1:
                            return resp
                    except Exception:
                        pass
                continue

            if not resp:
                print(f"      (empty response, attempt {attempt + 1})")
                time.sleep(1.0)
                continue

        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"      retry {attempt + 1}/{max_retries} in {wait}s: "
                      f"{type(e).__name__}")
                time.sleep(wait)
            else:
                print(f"      FAIL after {max_retries} retries: "
                      f"{type(e).__name__}")

    return "[EMPTY]"


def run_one_combination(
    cfg: dict, seeds: list[tuple[str, str]], temperature: float
) -> dict:
    """Run recursive generation for one (model, temperature) combination.

    Args:
        cfg: Model configuration dict.
        seeds: List of (capability, prompt_text) tuples.
        temperature: Sampling temperature.

    Returns:
        Dict with label, temperature, global_beta, per_cap_beta, etc.
    """
    model = cfg["model"]
    provider_name = cfg["provider"]
    label = cfg["label"]
    delay = cfg["delay"]
    temp_key = f"T{temperature:.1f}"
    combo_name = f"{label}_{temp_key}"

    print(f"\n{'=' * 60}")
    print(f"[TEMP SWEEP] {label} @ T={temperature:.1f} "
          f"({provider_name}/{model}, β_range={cfg['beta_range']})")
    print(f"{'=' * 60}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    adapter = create_provider(provider_name, model=model,
                              temperature=temperature, timeout_sec=120)

    # ── Build Gen0 samples ──────────────────────────────────────
    samples: list[DataSample] = []
    for i, (tag, text) in enumerate(seeds):
        samples.append(DataSample(
            text=text,
            generation=0,
            source_model="human",
            capability_tags=[tag],
            sample_id=f"TS_G0_{label}_{i:04d}",
        ))

    # ── Recursive generation: Gen1 → Gen2 → Gen3 ────────────────
    prev_texts = [s[1] for s in seeds]
    prev_tags = [[s[0]] for s in seeds]

    for gen in range(1, N_GENERATIONS + 1):
        n_prompts = len(prev_texts)
        print(f"  [Gen {gen}/{N_GENERATIONS}] {n_prompts} responses "
              f"@ T={temperature:.1f} ...")
        curr_texts = []
        gen_start = time.time()
        for i, text in enumerate(prev_texts):
            resp = generate_with_temperature_retry(
                adapter, text, temperature
            )
            curr_texts.append(resp)
            samples.append(DataSample(
                text=resp,
                generation=gen,
                source_model=model,
                capability_tags=prev_tags[i],
                sample_id=f"TS_G{gen}_{label}_T{temperature:.1f}_{i:04d}",
            ))
            # Progress every 5 samples
            if (i + 1) % 5 == 0 or i == n_prompts - 1:
                el = time.time() - gen_start
                print(f"    [{i + 1}/{n_prompts}] {el:.0f}s elapsed, "
                      f"~{el / (i + 1):.1f}s per sample")
            time.sleep(delay)
        prev_texts = curr_texts
        prev_tags = [[s.capability_tags[0]] if s.capability_tags else ["general"]
                      for s in samples if s.generation == gen]
        gen_elapsed = time.time() - gen_start
        print(f"  Gen {gen} complete in {gen_elapsed:.0f}s")

    # ── Save lineage ────────────────────────────────────────────
    lineage = DatasetLineage(samples=samples)
    lineage_path = os.path.join(OUTPUT_DIR,
                                f"{combo_name}_lineage.jsonl")
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
        f"temperature sweep: {label} @ T={temperature:.1f} "
        "+ exponential + total_constraint"
    )
    report["model"] = model
    report["provider"] = provider_name
    report["family"] = cfg.get("family", "")
    report["temperature"] = temperature
    report["beta_range"] = cfg["beta_range"]

    report_path = os.path.join(OUTPUT_DIR, f"{combo_name}_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  Saved report: {report_path}")

    print(f"  => {label} @ T={temperature:.1f}: β = {global_beta:.4f}")
    for cap, b in sorted(per_cap_beta.items()):
        print(f"     {cap}: {b:.4f}")

    return {
        "label": label,
        "model": model,
        "family": cfg["family"],
        "beta_range": cfg["beta_range"],
        "temperature": temperature,
        "global_beta": global_beta,
        "per_cap_beta": per_cap_beta,
        "lineage_path": lineage_path,
        "report_path": report_path,
    }


def compute_robustness_score(betas: dict[float, float]) -> dict:
    """Compute temperature robustness metrics from {T: beta} dict.

    Returns:
        Dict with mean, std, max_delta, cv (coefficient of variation),
        and robustness verdict.
    """
    if len(betas) < 2:
        return {"error": "insufficient data"}

    temps = sorted(betas.keys())
    vals = [betas[t] for t in temps]

    import numpy as np
    vals_arr = np.array(vals)
    mean_val = float(np.mean(vals_arr))
    std_val = float(np.std(vals_arr, ddof=1)) if len(vals) > 1 else 0.0
    max_delta = float(np.max(vals_arr) - np.min(vals_arr))
    cv = std_val / mean_val if mean_val > 1e-8 else float("inf")

    # Robustness verdict:
    #   CV < 0.15 → robust (temperature-invariant within 15%)
    #   CV < 0.30 → moderately robust
    #   CV >= 0.30 → temperature-sensitive
    if cv < 0.15:
        verdict = "robust"
    elif cv < 0.30:
        verdict = "moderately_robust"
    else:
        verdict = "temperature_sensitive"

    return {
        "mean_beta": mean_val,
        "std_beta": std_val,
        "max_delta": max_delta,
        "cv": cv,
        "verdict": verdict,
        "temperatures": temps,
        "betas": {f"T{t:.1f}": v for t, v in betas.items()},
    }


def print_summary_table(all_results: list[dict]):
    """Print temperature sweep summary table and robustness analysis."""
    # Group results by model
    by_model: dict[str, dict[float, dict]] = {}
    for r in all_results:
        label = r["label"]
        t = r["temperature"]
        by_model.setdefault(label, {})[t] = r

    print("\n" + "=" * 85)
    print("  TEMPERATURE ROBUSTNESS: Model-Level β across T ∈ {0.0, 0.8, 1.0}")
    print("=" * 85)
    header = (f"  {'Model':<22} {'β(T=0.0)':>10} {'β(T=0.8)':>10} "
              f"{'β(T=1.0)':>10} {'Δβ_max':>10} {'Verdict':>18}")
    print(header)
    print("  " + "-" * 82)

    robustness_results = {}
    for label in sorted(by_model.keys()):
        betas_t = {}
        row = [f"  {label:<22}"]
        for t in TEMPERATURES:
            r = by_model[label].get(t)
            if r:
                b = r["global_beta"]
                betas_t[t] = b
                row.append(f"{b:>10.4f}")
            else:
                row.append(f"{'N/A':>10}")

        if len(betas_t) >= 2:
            rob = compute_robustness_score(betas_t)
            robustness_results[label] = rob
            row.append(f"{rob['max_delta']:>10.4f}")
            row.append(f"{rob['verdict']:>18}")
        else:
            row.append(f"{'N/A':>10}")
            row.append(f"{'insufficient':>18}")

        print("".join(row))

    print("  " + "-" * 82)

    # Per-model detail
    print(f"\n  Robustness scores (CV = σ/μ, lower = more temperature-invariant):")
    for label, rob in sorted(robustness_results.items()):
        cv_str = f"{rob['cv']:.3f}" if rob['cv'] != float('inf') else "inf"
        print(f"    {label:<22} mean={rob['mean_beta']:.4f} "
              f"σ={rob['std_beta']:.4f} Δmax={rob['max_delta']:.4f} "
              f"CV={cv_str} → {rob['verdict']}")

    # Cross-model observation
    print(f"\n  Cross-model observations:")
    all_cvs = [r["cv"] for r in robustness_results.values()
               if r.get("cv", float("inf")) != float("inf")]
    if all_cvs:
        n_robust = sum(1 for cv in all_cvs if cv < 0.15)
        n_moderate = sum(1 for cv in all_cvs if 0.15 <= cv < 0.30)
        n_sensitive = sum(1 for cv in all_cvs if cv >= 0.30)
        print(f"    Robust (CV<0.15):        {n_robust}/{len(all_cvs)}")
        print(f"    Moderately robust (0.15-0.30): {n_moderate}/{len(all_cvs)}")
        print(f"    Temperature-sensitive (>=0.30): {n_sensitive}/{len(all_cvs)}")
        print(f"    Mean CV across models: {sum(all_cvs)/len(all_cvs):.3f}")
        reviewer_note = (
            "strong" if sum(all_cvs)/len(all_cvs) < 0.15 else
            "moderate" if sum(all_cvs)/len(all_cvs) < 0.25 else
            "weak"
        )
        print(f"    → Overall evidence for temperature robustness: {reviewer_note}")


# ── Main ────────────────────────────────────────────────────────────

def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="Temperature Robustness Expansion: 3 models x 3 temperatures"
    )
    ap.add_argument("--model", type=str, default="",
                    help="Run only a specific model label (e.g. 'gpt-4o-mini')")
    ap.add_argument("--temp", type=float, default=-1.0,
                    help="Run only a specific temperature (e.g. 0.0, 0.8, 1.0)")
    ap.add_argument("--seeds", type=int, default=N_SEEDS,
                    help=f"Number of seed prompts (default: {N_SEEDS})")
    args = ap.parse_args()

    # ── Select balanced seed subset ──────────────────────────────
    seeds = select_balanced_seeds(args.seeds)

    # ── Filter models / temperatures ─────────────────────────────
    models_to_run = MODELS
    temps_to_run = TEMPERATURES

    if args.model:
        models_to_run = [m for m in MODELS if m["label"] == args.model]
        if not models_to_run:
            print(f"Unknown model '{args.model}'. Available: "
                  f"{[m['label'] for m in MODELS]}")
            sys.exit(1)

    if args.temp >= 0:
        temps_to_run = [args.temp]

    # ── Build run queue: models x temperatures ──────────────────
    run_queue = []
    for cfg in models_to_run:
        for t in temps_to_run:
            run_queue.append((cfg, t))

    print(f"\nTemperature sweep: {len(models_to_run)} model(s) x "
          f"{len(temps_to_run)} temperature(s) = {len(run_queue)} run(s)")
    print(f"Seeds: {len(seeds)} across {len(set(t for t, _ in seeds))} capabilities")
    print(f"Generations: {N_GENERATIONS} (Gen0 → Gen{N_GENERATIONS})")
    print(f"Estimated total samples: "
          f"{len(run_queue) * len(seeds) * (N_GENERATIONS + 1)}")
    print(f"Models run sequentially (delay varies by provider)")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Run all combinations sequentially ─────────────────────────
    all_results = []
    total_start = time.time()

    for idx, (cfg, t) in enumerate(run_queue):
        print(f"\n[{idx + 1}/{len(run_queue)}] {cfg['label']} @ T={t:.1f}")
        try:
            r = run_one_combination(cfg, seeds, t)
            all_results.append(r)
            print(f"  [OK #{idx + 1}] {r['label']} @ T={t:.1f}: "
                  f"β = {r['global_beta']:.4f}")
        except Exception as e:
            print(f"  [FAIL #{idx + 1}] {cfg['label']} @ T={t:.1f}: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({
                "label": cfg["label"],
                "model": cfg["model"],
                "temperature": t,
                "global_beta": None,
                "error": str(e),
            })

        # Inter-run cooldown to avoid rate limits
        if idx < len(run_queue) - 1:
            cooldown = 2.0
            print(f"  (cooldown {cooldown:.0f}s before next run)")
            time.sleep(cooldown)

    total_elapsed = time.time() - total_start
    print(f"\nAll {len(run_queue)} runs complete in "
          f"{total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")

    # ── Save aggregate results ─────────────────────────────────
    sweep_path = os.path.join(OUTPUT_DIR, "temperature_sweep.json")
    sweep_data = {
        "metadata": {
            "description": "Temperature robustness expansion: 3 models x 3 temps",
            "n_seeds": len(seeds),
            "n_generations": N_GENERATIONS,
            "temperatures": temps_to_run,
            "generated_at": datetime.now().isoformat(),
            "total_elapsed_sec": total_elapsed,
        },
        "results": [
            {
                "label": r["label"],
                "model": r.get("model", ""),
                "family": r.get("family", ""),
                "beta_range": r.get("beta_range", ""),
                "temperature": r["temperature"],
                "global_beta": r.get("global_beta"),
                "per_cap_beta": r.get("per_cap_beta", {}),
            }
            for r in all_results
        ],
    }
    with open(sweep_path, "w") as f:
        json.dump(sweep_data, f, indent=2, ensure_ascii=False)
    print(f"Saved aggregate results: {sweep_path}")

    # ── Print summary table ────────────────────────────────────
    print_summary_table(all_results)

    # ── Interpretation guidance ─────────────────────────────────
    print("\n" + "=" * 85)
    print("  Interpretation for Reviewer Response:")
    print("  - \"Robust\" (CV < 0.15): β is effectively temperature-invariant,")
    print("    strengthening the claim that β measures a structural property.")
    print("  - \"Moderately robust\" (0.15 <= CV < 0.30): β shows some temperature")
    print("    dependence but ranking order is preserved.")
    print("  - \"Temperature-sensitive\" (CV >= 0.30): β depends on temperature —")
    print("    report temperature as a required parameter in all experiments.")
    print("=" * 85)

    return 0


if __name__ == "__main__":
    sys.exit(main())
