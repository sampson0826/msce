"""K=5 extension: add GPT-4o-mini + Claude Opus 4.7 (QuickRouter).
Reuses existing K=5 infrastructure. Run after original 3 DeepSeek models complete.
"""
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

from synthetic_decay_monitor.data_lineage import parse_lineage_from_jsonl, DataSample, DatasetLineage
from synthetic_decay_monitor.constraint_extractor import HybridConstraintExtractor
from synthetic_decay_monitor.decay_engine import DecayEngine
from synthetic_decay_monitor.provider_adapter import create_provider
from synthetic_decay_monitor.executor_classifier import diagnose_executor_decay
from synthetic_decay_monitor.report import generate_json_report

SEED_PATH = "experiment_data/cross_seed/deepseek_seeds_lineage.jsonl"
OUTPUT_DIR = "experiment_data/k5"
N_GENERATIONS = 5
N_SEEDS = 100  # Use all 100 seeds

MODELS_EXTENSION = [
    {"model": "gpt-4o-mini",      "provider": "quickrouter", "family": "OpenAI",
     "name": "gpt-4o-mini_k5",     "temp": 0.8, "delay": 3.0},
    {"model": "claude-opus-4-7",   "provider": "quickrouter", "family": "Anthropic",
     "name": "claude-opus-4-7_k5",  "temp": 0.8, "delay": 3.0},
]


def load_seeds():
    lineage = parse_lineage_from_jsonl(SEED_PATH)
    seeds = [(s.capability_tags, s.text) for s in lineage.samples if s.generation == 0]
    print(f"Loaded {len(seeds)} seeds")
    if len(seeds) > N_SEEDS:
        seeds = seeds[:N_SEEDS]
        print(f"  Using first {N_SEEDS} seeds")
    return seeds


def run_model(cfg, seeds):
    model, provider, name = cfg["model"], cfg["provider"], cfg["name"]
    done_file = os.path.join(OUTPUT_DIR, f".done_{name}")
    if os.path.exists(done_file):
        print(f"\n  [{name}] Already done, skipping. Remove {done_file} to re-run.")
        return None

    print(f"\n{'='*60}")
    print(f"K=5: {name}  (provider={provider})")
    print(f"{'='*60}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    adapter = create_provider(provider, model=model, temperature=cfg["temp"], timeout_sec=120)

    samples = []
    for i, (tags, text) in enumerate(seeds):
        samples.append(DataSample(text=text, generation=0, source_model=model,
                                  capability_tags=tags, sample_id=f"K5_G0_{i:04d}"))

    prev_texts = [s[1] for s in seeds]
    for gen in range(1, N_GENERATIONS + 1):
        print(f"  [Gen {gen}/{N_GENERATIONS}] {name} -> {len(prev_texts)} responses...")
        curr_texts = []
        for i, text in enumerate(prev_texts):
            resp = ""
            for attempt in range(3):
                try:
                    resp = adapter.generate(text, max_tokens=512)
                    break
                except Exception as e:
                    if attempt < 2:
                        time.sleep(2 ** (attempt + 1))
                    else:
                        print(f"    [{i+1}] FAIL: {type(e).__name__}")
                        resp = ""
            if not resp:
                resp = "[EMPTY]"
            curr_texts.append(resp)
            samples.append(DataSample(text=resp, generation=gen, source_model=model,
                                      capability_tags=seeds[i][0],
                                      sample_id=f"K5_G{gen}_{name}_{i:04d}"))
            if (i + 1) % 20 == 0:
                print(f"    {i+1}/{len(prev_texts)} done")
            time.sleep(cfg["delay"])
        prev_texts = curr_texts

    lineage = DatasetLineage(samples=samples)
    lp = os.path.join(OUTPUT_DIR, f"{name}_lineage.jsonl")
    lineage.save(lp)
    print(f"  Saved: {lp}")

    extractor = HybridConstraintExtractor(judge_fn=None)
    engine = DecayEngine(lineage, extractor)
    engine.run_all_capabilities()
    trajectories = engine.get_all_trajectories()

    per_cap = {}
    for t in trajectories:
        for g in t["trajectory"]:
            if "beta" in g and g["beta"] > 0:
                per_cap[t["capability"]] = g["beta"]
                break
    betas = list(per_cap.values())
    global_beta = sum(betas) / len(betas) if betas else 0.0

    diagnoses = []
    for cap, snapshots in engine._snapshots.items():
        diag = diagnose_executor_decay(engine._trajectories.get(cap, []), snapshots, capability=cap)
        diagnoses.append(diag)

    report = generate_json_report(lineage=lineage, trajectories=trajectories, diagnoses=diagnoses)
    report["decay_analysis"]["global_beta"] = global_beta
    report["decay_analysis"]["per_cap_beta"] = per_cap
    report["decay_analysis"]["k_depth"] = N_GENERATIONS
    report["decay_analysis"]["_method"] = f"K={N_GENERATIONS} depth: exponential + total_constraint"
    report["model"] = model

    rp = os.path.join(OUTPUT_DIR, f"{name}_report.json")
    with open(rp, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Mark done
    with open(done_file, "w") as f:
        f.write(f"completed {time.strftime('%Y-%m-%dT%H:%M:%S')}")

    return {"name": name, "model": model, "global_beta": global_beta, "per_cap_beta": per_cap}


def main():
    seeds = load_seeds()
    results = {}
    for cfg in MODELS_EXTENSION:
        try:
            r = run_model(cfg, seeds)
            if r:
                results[cfg["name"]] = r
                print(f"  K=5 beta = {r['global_beta']:.4f}")
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

    sp = os.path.join(OUTPUT_DIR, "k5_extension_summary.json")
    with open(sp, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {sp}")


if __name__ == "__main__":
    main()
