"""Seed sensitivity test: GPT-4o-mini with alternative 100-seed set."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip())

from synthetic_decay_monitor.provider_adapter import create_provider
from synthetic_decay_monitor.experiment_config import ExperimentConfig
from synthetic_decay_monitor.experiment_runner import ExperimentRunner

# Load alternative seeds
with open("experiment_data/n100/alternative_seeds_100.json") as f:
    alt_raw = json.load(f)
alt_seeds = [(s["capability"], s["prompt"]) for s in alt_raw]
print(f"Loaded {len(alt_seeds)} alternative seeds")

config = ExperimentConfig(
    model="gpt-4o-mini",
    provider="quickrouter",
    generations=3,
    seeds=alt_seeds,
    temperature=0.8,
    output_dir="experiment_data/n100",
    experiment_name="gpt-4o-mini_alt_seeds",
    family="OpenAI",
    sleep_sec=0.5,
    max_retries=3,
)
adapter = create_provider("quickrouter", model="gpt-4o-mini")
runner = ExperimentRunner(config, adapter=adapter)
result = runner.run()

print(f"\nSeed sensitivity β (alt seeds) = {result['global_beta']:.4f}")
print(f"Original β (same seeds, retest) = 0.1738")
print(f"Original β (first run) = 0.1336")
