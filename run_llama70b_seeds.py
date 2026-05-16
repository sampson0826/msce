"""Parallel run: Llama 3.1 70B experiment + alternative seed set generation."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load .env
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip())

from synthetic_decay_monitor.provider_adapter import create_provider
from synthetic_decay_monitor.experiment_config import ExperimentConfig, expand_seeds, DEFAULT_SEEDS
from synthetic_decay_monitor.experiment_runner import ExperimentRunner


t0 = time.time()

# ── Step 1: Generate alternative seeds for seed sensitivity ──
print("="*60)
print("Step 1: Generating alternative 100-seed set")
print("="*60)
adapter_seed = create_provider("quickrouter", model="claude-haiku-4-5-20251001")
# Clear cache so we get genuinely new seeds
import synthetic_decay_monitor.experiment_config as ec
ec._SEED_CACHE = {}
alt_seeds = expand_seeds(100, adapter=adapter_seed)
print(f"Generated {len(alt_seeds)} alternative seeds")
# Save for later
with open("experiment_data/n100/alternative_seeds_100.json", "w") as f:
    json.dump([{"capability": s[0], "prompt": s[1]} for s in alt_seeds], f, indent=2, ensure_ascii=False)
print("Saved: experiment_data/n100/alternative_seeds_100.json")

# ── Step 2: Llama 3.1 70B via OpenRouter ──
print("\n" + "="*60)
print("Step 2: Llama 3.1 70B Instruct via OpenRouter")
print("="*60)
seeds_70b = expand_seeds(100, adapter=create_provider("openrouter", model="meta-llama/llama-3.1-70b-instruct"))
print(f"Generated {len(seeds_70b)} seeds for Llama 70B")

config = ExperimentConfig(
    model="meta-llama/llama-3.1-70b-instruct",
    provider="openrouter",
    generations=3,
    seeds=seeds_70b,
    temperature=0.8,
    output_dir="experiment_data/n100",
    experiment_name="llama70b",
    family="Llama",
    sleep_sec=0.5,
    max_retries=3,
)
adapter = create_provider("openrouter", model="meta-llama/llama-3.1-70b-instruct")
runner = ExperimentRunner(config, adapter=adapter)
result = runner.run()

elapsed = time.time() - t0
print(f"\nTotal elapsed: {elapsed:.0f}s")
print(f"Llama 70B β = {result['global_beta']:.4f}")
