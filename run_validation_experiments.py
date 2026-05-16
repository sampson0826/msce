"""Phase B validation experiments: test-retest, model expansion, seed sensitivity."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load .env for API keys in background shell
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip())

from synthetic_decay_monitor.provider_adapter import create_provider
from synthetic_decay_monitor.experiment_config import ExperimentConfig, DEFAULT_SEEDS, expand_seeds
from synthetic_decay_monitor.experiment_runner import ExperimentRunner


def run_experiment(name, model, provider, seeds, temperature=0.8, generations=3,
                   output_dir="experiment_data/n100", sleep_sec=0.5, max_retries=3,
                   family="", seed_descr=""):
    """Run a single experiment and return (name, beta, n_samples, elapsed_sec)."""
    t0 = time.time()
    config = ExperimentConfig(
        model=model, provider=provider, generations=generations,
        seeds=seeds, temperature=temperature, output_dir=output_dir,
        experiment_name=name, family=family, sleep_sec=sleep_sec,
        max_retries=max_retries,
    )
    adapter = create_provider(provider, model=model, temperature=temperature)
    runner = ExperimentRunner(config, adapter=adapter)
    result = runner.run()
    elapsed = time.time() - t0
    beta = result["global_beta"]
    n = result["n_samples"]
    print(f"\n{'='*60}")
    print(f"EXPERIMENT DONE: {name}")
    print(f"  β={beta:.4f}  n_samples={n}  elapsed={elapsed:.0f}s  {seed_descr}")
    print(f"{'='*60}\n")
    return name, beta, n, elapsed


# ── Experiments ──────────────────────────────────────────────

experiments = []

# 1. Test-retest: GPT-4o-mini with EXACT same seeds as original
print("\n" + "="*60)
print("SETUP: Loading original seeds for test-retest")
print("="*60)
orig_config = json.load(open("experiment_data/n100/gpt-4o-mini_quickrouter_config.json"))
orig_seeds = [(s[0], s[1]) for s in orig_config["seeds"]]
print(f"Loaded {len(orig_seeds)} seeds from original GPT-4o-mini config")
experiments.append({
    "name": "gpt-4o-mini_retest",
    "model": "gpt-4o-mini",
    "provider": "quickrouter",
    "seeds": orig_seeds,
    "family": "OpenAI",
    "seed_descr": f"{len(orig_seeds)} seeds (IDENTICAL to original)",
})

# 2. GPT-4o (new model for OpenAI family)
adapter_gpt4o = create_provider("quickrouter", model="gpt-4o")
gpt4o_seeds = expand_seeds(100, adapter=adapter_gpt4o)
print(f"Generated {len(gpt4o_seeds)} seeds for GPT-4o")
experiments.append({
    "name": "gpt-4o_s100",
    "model": "gpt-4o",
    "provider": "quickrouter",
    "seeds": gpt4o_seeds,
    "family": "OpenAI",
    "seed_descr": f"{len(gpt4o_seeds)} seeds (newly generated)",
})

print(f"\n{len(experiments)} experiments queued. Running sequentially...\n")

results = []
for i, exp in enumerate(experiments):
    print(f"\n[{i+1}/{len(experiments)}] {exp['name']}")
    r = run_experiment(**{k: v for k, v in exp.items() if k != 'seed_descr'},
                      seed_descr=exp.get('seed_descr', ''))
    results.append({
        "name": r[0], "beta": r[1], "n_samples": r[2],
        "elapsed_sec": r[3], "seed_descr": exp.get("seed_descr", ""),
    })

# ── Summary ──────────────────────────────────────────────────

print("\n" + "="*60)
print("VALIDATION SUMMARY")
print("="*60)
orig_beta = 0.1336
for r in results:
    print(f"  {r['name']}: β={r['beta']:.4f}  n={r['n_samples']}  "
          f"elapsed={r['elapsed_sec']:.0f}s  [{r['seed_descr']}]")

# Save results
with open("experiment_data/validation_results.json", "w") as f:
    json.dump({
        "original_gpt4o_mini_beta": orig_beta,
        "experiments": results,
    }, f, indent=2)
print(f"\nSaved: experiment_data/validation_results.json")
