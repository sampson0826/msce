"""Regenerate all reports using pre-registered decay engine (exponential + total_constraint)."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from synthetic_decay_monitor.data_lineage import parse_lineage_from_jsonl, DatasetLineage
from synthetic_decay_monitor.constraint_extractor import HybridConstraintExtractor
from synthetic_decay_monitor.decay_engine import DecayEngine
from synthetic_decay_monitor.executor_classifier import diagnose_executor_decay
from synthetic_decay_monitor.report import generate_json_report

N100_DIR = "experiment_data/n100"

# Find all lineage files
lineage_files = []
for f in os.listdir(N100_DIR):
    if f.endswith("_lineage.jsonl"):
        path = os.path.join(N100_DIR, f)
        lineage_files.append((f, path))

print(f"Re-analyzing {len(lineage_files)} lineage files with pre-registered engine...\n")

results_summary = {}
errors = []

for fname, path in sorted(lineage_files):
    name = fname.replace("_lineage.jsonl", "")
    print(f"  {name}...", end=" ", flush=True)
    t0 = time.time()

    try:
        lineage = parse_lineage_from_jsonl(path)
    except Exception as e:
        print(f"SKIP (bad lineage: {e})")
        errors.append((name, str(e)))
        continue

    extractor = HybridConstraintExtractor(judge_fn=None)
    engine = DecayEngine(lineage, extractor)
    engine.run_all_capabilities()

    trajectories = engine.get_all_trajectories()
    collapse_order = engine.get_collapse_order()

    diagnoses = []
    for cap, snapshots in engine._snapshots.items():
        diag = diagnose_executor_decay(
            engine._trajectories.get(cap, []), snapshots, capability=cap
        )
        diagnoses.append(diag)

    betas = []
    for t in trajectories:
        if "trajectory" in t:
            for g in t["trajectory"]:
                if "beta" in g and g["beta"] > 0:
                    betas.append(g["beta"])
    global_beta = sum(betas) / len(betas) if betas else 0.0

    # Generate and save report
    report = generate_json_report(
        lineage=lineage,
        trajectories=trajectories,
        diagnoses=diagnoses,
    )
    report["decay_analysis"]["global_beta"] = global_beta
    report["decay_analysis"]["_method"] = "pre-registered: exponential + total_constraint"

    report_path = os.path.join(N100_DIR, f"{name}_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    results_summary[name] = {
        "global_beta": global_beta,
        "n_samples": len(lineage.samples),
        "n_capabilities": len(engine._snapshots),
    }
    print(f"β={global_beta:.4f} ({elapsed:.1f}s)")

# Summary
print(f"\n{'='*60}")
print(f"REGENERATED REPORTS SUMMARY")
print(f"{'='*60}")
print(f"{'Model':<45s} {'β':>8s}  {'n':>5s}")
print(f"{'-'*60}")
for name in sorted(results_summary.keys()):
    r = results_summary[name]
    print(f"{name:<45s} {r['global_beta']:8.4f}  {r['n_samples']:5d}")

if errors:
    print(f"\nErrors ({len(errors)}):")
    for name, err in errors:
        print(f"  {name}: {err}")

# Save summary
with open(os.path.join(N100_DIR, "pre_registered_summary.json"), "w") as f:
    json.dump({"results": results_summary, "errors": errors, "method": "exponential + total_constraint (pre-registered)"}, f, indent=2)
print(f"\nSaved: {N100_DIR}/pre_registered_summary.json")
