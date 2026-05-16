"""Batch experiment: benchmark latest LLMs (May 2026) on recursive stability β.

Models: DeepSeek-V4 Pro, DeepSeek-V4 Flash, GPT-5.5, Llama 4 Maverick,
        Llama 4 Scout, Gemini 3.1 Pro

Reuses Gen0 seeds from existing gpt-4o-mini lineage to save cost.
"""
import sys, os, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Manual .env loading for background shell compatibility
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip())

from synthetic_decay_monitor.data_lineage import parse_lineage_from_jsonl, DataSample, DatasetLineage
from synthetic_decay_monitor.constraint_extractor import HybridConstraintExtractor
from synthetic_decay_monitor.decay_engine import DecayEngine
from synthetic_decay_monitor.provider_adapter import create_provider
from synthetic_decay_monitor.executor_classifier import diagnose_executor_decay
from synthetic_decay_monitor.report import generate_json_report

OUTPUT_DIR = "experiment_data/latest_models"
SEED_SOURCE = "experiment_data/n100/gpt-4o-mini_s100_lineage.jsonl"
N_GENERATIONS = 3  # Gen1, Gen2, Gen3

# ── Model definitions ─────────────────────────────────────────────────
MODELS = [
    {
        "model": "deepseek-v4-pro",
        "provider": "deepseek",
        "family": "DeepSeek",
        "experiment_name": "deepseek-v4-pro_s100",
        "temperature": 0.8,
        "max_tokens": 512,
        "timeout_sec": 180,
        "api_delay": 1.0,  # reasoning model: slower to avoid rate limits
    },
    {
        "model": "deepseek-v4-flash",
        "provider": "deepseek",
        "family": "DeepSeek",
        "experiment_name": "deepseek-v4-flash_s100",
        "temperature": 0.8,
        "max_tokens": 512,
        "timeout_sec": 120,
        "api_delay": 0.5,
    },
    {
        "model": "meta-llama/llama-4-maverick",
        "provider": "openrouter",
        "family": "Llama",
        "experiment_name": "llama-4-maverick_s100",
        "temperature": 0.8,
        "max_tokens": 512,
        "timeout_sec": 120,
        "api_delay": 0.5,
    },
    {
        "model": "meta-llama/llama-4-scout",
        "provider": "openrouter",
        "family": "Llama",
        "experiment_name": "llama-4-scout_s100",
        "temperature": 0.8,
        "max_tokens": 512,
        "timeout_sec": 120,
        "api_delay": 0.5,
    },
    # GPT-5.5 via QuickRouter (may need retries for upstream saturation)
    {
        "model": "gpt-5.5",
        "provider": "quickrouter",
        "family": "OpenAI",
        "experiment_name": "gpt-5.5_s100",
        "temperature": 0.8,
        "max_tokens": 512,
        "timeout_sec": 180,
        "api_delay": 2.0,  # longer delay to avoid upstream saturation
    },
    # Claude Opus 4.7 via QuickRouter
    {
        "model": "claude-opus-4-7",
        "provider": "quickrouter",
        "family": "Claude",
        "experiment_name": "claude-opus-4-7_s100",
        "temperature": 0.8,
        "max_tokens": 512,
        "timeout_sec": 120,
        "api_delay": 0.5,
    },
    # Gemini 2.5 Flash via QuickRouter (优质gemini group) — 2.5 Pro returns empty content
    {
        "model": "gemini-2.5-flash",
        "provider": "quickrouter",
        "family": "Gemini",
        "experiment_name": "gemini-2.5-flash_s100",
        "temperature": 0.8,
        "max_tokens": 512,
        "timeout_sec": 120,
        "api_delay": 1.0,
    },
    # Gemini 2.5 Pro via QuickRouter (优质gemini group) — reasoning model, empty retry handled
    {
        "model": "gemini-2.5-pro",
        "provider": "quickrouter",
        "family": "Gemini",
        "experiment_name": "gemini-2.5-pro_s100",
        "temperature": 0.8,
        "max_tokens": 512,
        "timeout_sec": 120,
        "api_delay": 2.0,
    },
]


def load_gen0_seeds(path: str) -> list[tuple[list[str], str]]:
    """Extract Gen0 (seed prompt) samples from an existing lineage file."""
    lineage = parse_lineage_from_jsonl(path)
    seeds = []
    for s in lineage.samples:
        if s.generation == 0:
            seeds.append((s.capability_tags, s.text))
    print(f"Loaded {len(seeds)} Gen0 seeds from {path}")
    return seeds


def run_experiment(model_cfg: dict, seeds: list[tuple[list[str], str]]) -> dict:
    """Run recursive generation + analysis for one model."""
    model = model_cfg["model"]
    provider = model_cfg["provider"]
    experiment_name = model_cfg["experiment_name"]

    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {experiment_name}")
    print(f"Model: {model} | Provider: {provider} | Seeds: {len(seeds)}")
    print(f"{'='*60}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Create adapter
    adapter = create_provider(
        provider,
        model=model,
        temperature=model_cfg.get("temperature", 0.8),
        timeout_sec=model_cfg.get("timeout_sec", 120),
    )
    api_delay = model_cfg.get("api_delay", 0.5)

    # ── Generation ─────────────────────────────────────────────
    samples: list[DataSample] = []

    # Gen 0: seed prompts
    for i, (tags, text) in enumerate(seeds):
        samples.append(DataSample(
            text=text, generation=0, source_model="human",
            capability_tags=tags, sample_id=f"G0_{i:04d}",
        ))

    prev_texts = [s[1] for s in seeds]

    for gen in range(1, N_GENERATIONS + 1):
        print(f"\n[Gen {gen}] {model} → {len(prev_texts)} responses...")
        curr_texts = []
        for i, text in enumerate(prev_texts):
            resp = ""
            for attempt in range(3):
                try:
                    resp = adapter.generate(text, max_tokens=model_cfg.get("max_tokens", 512))
                    break
                except Exception as e:
                    if attempt < 2:
                        wait = 2 ** (attempt + 1)
                        print(f"  [{i+1}/{len(prev_texts)}] retry {attempt+1}/3 in {wait}s: {e}")
                        time.sleep(wait)
                    else:
                        print(f"  [{i+1}/{len(prev_texts)}] FAIL: {e}")
                        resp = ""
            if not resp:
                resp = "[EMPTY]"
            curr_texts.append(resp)
            samples.append(DataSample(
                text=resp, generation=gen, source_model=model,
                capability_tags=seeds[i][0],
                sample_id=f"G{gen}_{model_cfg['experiment_name']}_{i:04d}",
            ))
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(prev_texts)} done")
            time.sleep(api_delay)
        prev_texts = curr_texts

    # Save lineage
    lineage_path = os.path.join(OUTPUT_DIR, f"{experiment_name}_lineage.jsonl")
    lineage = DatasetLineage(samples=samples)
    lineage.save(lineage_path)
    print(f"Saved lineage: {lineage_path} ({len(samples)} samples)")

    # ── Analysis ───────────────────────────────────────────────
    extractor = HybridConstraintExtractor(judge_fn=None)
    engine = DecayEngine(lineage, extractor)
    engine.run_all_capabilities()

    trajectories = engine.get_all_trajectories()
    collapse_order = engine.get_collapse_order()

    # Per-capability β
    per_cap_beta = {}
    per_cap_sn = {}
    for t in trajectories:
        cap = t["capability"]
        traj = t["trajectory"]
        for g in traj:
            if "beta" in g and g["beta"] > 0:
                per_cap_beta[cap] = g["beta"]
                break
        per_cap_sn[cap] = traj[-1]["S_n"] if traj else 1.0

    # Global β (average across capabilities)
    betas = list(per_cap_beta.values())
    global_beta = sum(betas) / len(betas) if betas else 0.0

    # Diagnoses
    diagnoses = []
    for cap, snapshots in engine._snapshots.items():
        diag = diagnose_executor_decay(
            engine._trajectories.get(cap, []), snapshots, capability=cap
        )
        diagnoses.append(diag)

    # Save report
    report = generate_json_report(
        lineage=lineage,
        trajectories=trajectories,
        diagnoses=diagnoses,
    )
    report["decay_analysis"]["global_beta"] = global_beta
    report["decay_analysis"]["_method"] = "pre-registered: exponential + total_constraint"
    report["model"] = model
    report["provider"] = provider
    report["family"] = model_cfg.get("family", "")

    report_path = os.path.join(OUTPUT_DIR, f"{experiment_name}_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Saved report: {report_path}")

    result = {
        "model": model,
        "experiment_name": experiment_name,
        "family": model_cfg.get("family", ""),
        "global_beta": global_beta,
        "per_cap_beta": per_cap_beta,
        "per_cap_sn": per_cap_sn,
        "n_samples": len(samples),
        "n_capabilities": len(engine._snapshots),
    }

    print(f"\n>>> {model}: β = {global_beta:.4f}")
    for cap in sorted(per_cap_beta.keys()):
        sn = per_cap_sn.get(cap, 0)
        beta_c = per_cap_beta.get(cap, 0)
        print(f"    {cap:<25s} β={beta_c:.4f}  S_n={sn:.3f}")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", help="Model indices to run (0-based, default: all)")
    parser.add_argument("--skip-generation", action="store_true", help="Skip generation, re-analyze existing lineage")
    args = parser.parse_args()

    # Load seeds
    seeds = load_gen0_seeds(SEED_SOURCE)
    print(f"Seed distribution: {len(set(t[0][0] for t in seeds))} capabilities")

    # Select models
    indices = [int(x) for x in args.models] if args.models else list(range(len(MODELS)))
    selected = [MODELS[i] for i in indices]

    print(f"\nRunning {len(selected)} model(s):")
    for m in selected:
        print(f"  [{MODELS.index(m)}] {m['experiment_name']} ({m['provider']})")

    results = []
    errors = []

    for m in selected:
        if args.skip_generation:
            # Re-analyze only
            lineage_path = os.path.join(OUTPUT_DIR, f"{m['experiment_name']}_lineage.jsonl")
            if not os.path.exists(lineage_path):
                print(f"SKIP {m['experiment_name']}: no lineage file")
                continue
            print(f"Re-analyzing {m['experiment_name']}...")
            # Re-use run_experiment but skip generation
        try:
            result = run_experiment(m, seeds)
            results.append(result)
        except Exception as e:
            print(f"ERROR: {m['experiment_name']}: {e}")
            import traceback
            traceback.print_exc()
            errors.append({"model": m["model"], "error": str(e)})

    # ── Summary ──────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"BATCH EXPERIMENT SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':<35s} {'Family':<12s} {'β':>8s}")
    print(f"{'-'*60}")
    for r in sorted(results, key=lambda x: x["global_beta"]):
        print(f"{r['model']:<35s} {r['family']:<12s} {r['global_beta']:8.4f}")

    if errors:
        print(f"\nErrors ({len(errors)}):")
        for e in errors:
            print(f"  {e['model']}: {e['error']}")

    # Compare with old models
    print(f"\n{'='*70}")
    print(f"COMBINED RANKING (old + new)")
    print(f"{'='*70}")

    # Load old results from pre_registered_summary
    old_summary_path = "experiment_data/n100/pre_registered_summary.json"
    all_betas = {}
    if os.path.exists(old_summary_path):
        with open(old_summary_path) as f:
            old = json.load(f).get("results", {})
        for name, info in old.items():
            label = name.replace("_s100", "").replace("_", " ")
            all_betas[label] = info["global_beta"]

    for r in results:
        label = r["experiment_name"].replace("_s100", "")
        all_betas[label] = r["global_beta"]

    for name in sorted(all_betas, key=lambda x: all_betas[x]):
        print(f"  {name:<40s} β = {all_betas[name]:.4f}")

    # Save summary
    summary_path = os.path.join(OUTPUT_DIR, "batch_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "results": {r["experiment_name"]: r for r in results},
            "errors": errors,
            "all_betas": all_betas,
            "method": "pre-registered: exponential + total_constraint",
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {summary_path}")


if __name__ == "__main__":
    main()
