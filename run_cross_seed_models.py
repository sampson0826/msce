"""Run 5 representative models on DeepSeek-V3 generated seeds for cross-seed-source validation."""
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
OUTPUT_DIR = "experiment_data/cross_seed"
N_GENERATIONS = 3

# 5 models spanning full β range. QuickRouter models need high delay (upstream saturation).
MODELS = [
    {"model": "deepseek-chat", "provider": "deepseek", "family": "DeepSeek", "name": "deepseek-v3_cross", "temp": 0.8, "delay": 0.5},
    {"model": "deepseek-v4-flash", "provider": "deepseek", "family": "DeepSeek", "name": "deepseek-v4-flash_cross", "temp": 0.8, "delay": 0.5},
    {"model": "gpt-4o-mini", "provider": "quickrouter", "family": "OpenAI", "name": "gpt-4o-mini_cross", "temp": 0.8, "delay": 3.0},
    {"model": "claude-sonnet-4-6", "provider": "quickrouter", "family": "Anthropic", "name": "claude-sonnet-4-6_cross", "temp": 0.8, "delay": 3.0},
    {"model": "claude-opus-4-7", "provider": "quickrouter", "family": "Anthropic", "name": "claude-opus-4-7_cross", "temp": 0.8, "delay": 3.0},
]

def load_seeds():
    lineage = parse_lineage_from_jsonl(SEED_PATH)
    seeds = []
    for s in lineage.samples:
        if s.generation == 0:
            seeds.append((s.capability_tags, s.text))
    print(f"Loaded {len(seeds)} Gen0 seeds from {SEED_PATH}")
    return seeds

def run_one_model(cfg, seeds):
    model, provider, name = cfg["model"], cfg["provider"], cfg["name"]
    print(f"\n{'='*60}")
    print(f"CROSS-SEED: {name} ({provider})")
    print(f"{'='*60}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    adapter = create_provider(provider, model=model, temperature=cfg["temp"], timeout_sec=120)

    samples = []
    for i, (tags, text) in enumerate(seeds):
        samples.append(DataSample(text=text, generation=0, source_model="deepseek-chat",
                                  capability_tags=tags, sample_id=f"CS_G0_{i:04d}"))

    prev_texts = [s[1] for s in seeds]
    for gen in range(1, N_GENERATIONS + 1):
        print(f"[Gen {gen}] {name} → {len(prev_texts)} responses...")
        curr_texts = []
        for i, text in enumerate(prev_texts):
            resp = ""
            for attempt in range(3):
                try:
                    resp = adapter.generate(text, max_tokens=512)
                    break
                except Exception as e:
                    if attempt < 2:
                        w = 2 ** (attempt+1)
                        print(f"  [{i+1}/{len(prev_texts)}] retry {attempt+1}/3 in {w}s: {type(e).__name__}")
                        time.sleep(w)
                    else:
                        print(f"  [{i+1}/{len(prev_texts)}] FAIL: {type(e).__name__}")
                        resp = ""
            if not resp:
                resp = "[EMPTY]"
            curr_texts.append(resp)
            samples.append(DataSample(text=resp, generation=gen, source_model=model,
                                      capability_tags=seeds[i][0],
                                      sample_id=f"CS_G{gen}_{name}_{i:04d}"))
            if (i+1) % 20 == 0:
                print(f"  {i+1}/{len(prev_texts)} done")
            time.sleep(cfg["delay"])
        prev_texts = curr_texts

    # Save lineage
    lineage = DatasetLineage(samples=samples)
    lineage_path = os.path.join(OUTPUT_DIR, f"{name}_lineage.jsonl")
    lineage.save(lineage_path)
    print(f"Saved: {lineage_path}")

    # Analysis
    extractor = HybridConstraintExtractor(judge_fn=None)
    engine = DecayEngine(lineage, extractor)
    engine.run_all_capabilities()
    trajectories = engine.get_all_trajectories()

    per_cap_beta = {}
    for t in trajectories:
        for g in t["trajectory"]:
            if "beta" in g and g["beta"] > 0:
                per_cap_beta[t["capability"]] = g["beta"]
                break

    betas = list(per_cap_beta.values())
    global_beta = sum(betas)/len(betas) if betas else 0.0

    diagnoses = []
    for cap, snapshots in engine._snapshots.items():
        diag = diagnose_executor_decay(engine._trajectories.get(cap, []), snapshots, capability=cap)
        diagnoses.append(diag)

    report = generate_json_report(lineage=lineage, trajectories=trajectories, diagnoses=diagnoses)
    report["decay_analysis"]["global_beta"] = global_beta
    report["decay_analysis"]["per_cap_beta"] = per_cap_beta
    report["decay_analysis"]["_method"] = "cross-seed validation: DeepSeek-V3 seeds + exponential + total_constraint"
    report["model"] = model
    report["provider"] = provider
    report["family"] = cfg.get("family", "")

    report_path = os.path.join(OUTPUT_DIR, f"{name}_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return {"name": name, "model": model, "family": cfg["family"], "global_beta": global_beta,
            "per_cap_beta": per_cap_beta}

def main():
    seeds = load_seeds()
    results = {}
    for cfg in MODELS:
        try:
            r = run_one_model(cfg, seeds)
            results[cfg["name"]] = r
            print(f"  ✓ β = {r['global_beta']:.4f}")
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            import traceback; traceback.print_exc()

    # Summary
    summary_path = os.path.join(OUTPUT_DIR, "cross_seed_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved summary to {summary_path}")

    print("\n" + "="*70)
    print("CROSS-SEED-SOURCE COMPARISON")
    print("="*70)
    print(f"{'Model':<25} {'β (GPT-4o-mini seeds)':>22} {'β (DeepSeek seeds)':>22} {'Δ':>8}")
    print("-"*78)

    original = {
        "deepseek-v3_cross": 0.0281,
        "deepseek-v4-flash_cross": 0.0350,
        "gpt-4o-mini_cross": 0.0885,
        "claude-sonnet-4-6_cross": 0.1055,
        "claude-opus-4-7_cross": 0.1635,
    }
    for name, orig in original.items():
        r = results.get(name, {})
        nb = r.get("global_beta")
        if nb:
            print(f"{name:<25} {orig:>22.4f} {nb:>22.4f} {abs(orig-nb):>8.4f}")
        else:
            print(f"{name:<25} {orig:>22.4f} {'FAILED':>22}")

if __name__ == "__main__":
    main()
