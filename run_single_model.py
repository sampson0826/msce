"""Run a single model on existing seeds. Usage:
  python run_single_model.py --model deepseek-chat --provider deepseek --name deepseek-v3_cross --delay 0.5 --output-dir experiment_data/cross_seed --seed-path experiment_data/cross_seed/deepseek_seeds_lineage.jsonl --n-gen 3
"""
import sys, os, json, time, argparse

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--provider", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--family", default="")
    ap.add_argument("--delay", type=float, default=1.0)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--n-gen", type=int, default=3)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seed-path", required=True)
    ap.add_argument("--sample-prefix", default="X")
    args = ap.parse_args()

    model, provider, name = args.model, args.provider, args.name
    print(f"\n{'='*60}")
    print(f"MODEL: {name} ({provider}/{model}) | gen={args.n_gen} | delay={args.delay}s")
    print(f"{'='*60}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load seeds
    lineage = parse_lineage_from_jsonl(args.seed_path)
    seeds = [(s.capability_tags, s.text) for s in lineage.samples if s.generation == 0]
    print(f"Loaded {len(seeds)} Gen0 seeds")

    adapter = create_provider(provider, model=model, temperature=args.temp, timeout_sec=120)

    samples = []
    for i, (tags, text) in enumerate(seeds):
        samples.append(DataSample(text=text, generation=0, source_model="deepseek-chat",
                                  capability_tags=tags, sample_id=f"{args.sample_prefix}_G0_{i:04d}"))

    prev_texts = [s[1] for s in seeds]
    for gen in range(1, args.n_gen + 1):
        print(f"[Gen {gen}/{args.n_gen}] {name} -> {len(prev_texts)} responses...")
        curr_texts = []
        for i, text in enumerate(prev_texts):
            resp = ""
            for attempt in range(3):
                try:
                    resp = adapter.generate(text, max_tokens=args.max_tokens)
                    break
                except Exception as e:
                    if attempt < 2:
                        w = 2 ** (attempt + 1)
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
                                      sample_id=f"{args.sample_prefix}_G{gen}_{name}_{i:04d}"))
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(prev_texts)} done")
            time.sleep(args.delay)
        prev_texts = curr_texts

    # Save lineage
    out_lineage = DatasetLineage(samples=samples)
    lp = os.path.join(args.output_dir, f"{name}_lineage.jsonl")
    out_lineage.save(lp)
    print(f"Saved: {lp}")

    # Analysis
    extractor = HybridConstraintExtractor(judge_fn=None)
    engine = DecayEngine(out_lineage, extractor)
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

    report = generate_json_report(lineage=out_lineage, trajectories=trajectories, diagnoses=diagnoses)
    report["decay_analysis"]["global_beta"] = global_beta
    report["decay_analysis"]["per_cap_beta"] = per_cap
    report["decay_analysis"]["_method"] = f"single-model run: {name}, n_gen={args.n_gen}"
    report["model"] = model
    report["provider"] = provider
    report["family"] = args.family

    rp = os.path.join(args.output_dir, f"{name}_report.json")
    with open(rp, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Saved: {rp}")

    print(f"\n  beta = {global_beta:.4f}")
    for cap, b in per_cap.items():
        print(f"    {cap}: {b:.4f}")

    # Write marker
    with open(os.path.join(args.output_dir, f".done_{name}"), "w") as f:
        json.dump({"beta": global_beta, "per_cap": per_cap}, f)

    return global_beta


if __name__ == "__main__":
    main()
