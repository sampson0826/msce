"""
CLI 入口 — 合成数据衰减监测器命令行工具。

用法:
    decay-eval --input data.jsonl --output report.html

    # or via python -m
    python -m synthetic_decay_monitor \\
        --input data.jsonl \\
        --output report.json \\
        --format json
"""

import sys
import os
import json
import argparse
from pathlib import Path

from synthetic_decay_monitor.data_lineage import (
    parse_lineage_from_jsonl, generate_synthetic_lineage, DatasetLineage,
)
from synthetic_decay_monitor.constraint_extractor import (
    EmbeddingConstraintExtractor, ConstraintFieldSnapshot, LLMJudgeConstraintExtractor,
    create_local_llm_judge, HybridConstraintExtractor,
)
from synthetic_decay_monitor.decay_engine import (
    DecayEngine, CapabilityStability, simulate_decay, S_CRITICAL,
)
from synthetic_decay_monitor.executor_classifier import (
    ExecutorClassifier, diagnose_executor_decay, DegradationDiagnosis,
)
from synthetic_decay_monitor.stress_propagation import (
    CapabilityTopology, run_cascade_analysis,
)
from synthetic_decay_monitor.report import (
    generate_json_report, generate_html_report,
)


def main():
    parser = argparse.ArgumentParser(
        description="Synthetic Data Decay Monitor — Constraint Layer Health Diagnostic",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  decay-eval --input data.jsonl
  decay-eval --input data.jsonl --output report.html
  decay-eval --demo 10 --output demo_report.html
  python -m synthetic_decay_monitor --input data.jsonl
        """,
    )
    parser.add_argument(
        "--input", "-i",
        help="Path to JSONL data file with text + generation fields",
    )
    parser.add_argument(
        "--output", "-o", default="decay_report.html",
        help="Output path (.json or .html, default: decay_report.html)",
    )
    parser.add_argument(
        "--format", "-f", choices=["json", "html"], default=None,
        help="Output format (auto-detected from extension if not specified)",
    )
    parser.add_argument(
        "--demo", type=int, default=None, metavar="N",
        help="Run with demo data (N generations of synthetic decay)",
    )
    parser.add_argument(
        "--text-field", default="text",
        help="JSON field name for text content (default: 'text')",
    )
    parser.add_argument(
        "--generation-field", default="generation",
        help="JSON field name for generation index (default: 'generation')",
    )
    parser.add_argument(
        "--capability-field", default="capability_tags",
        help="JSON field name for capability tags (default: 'capability_tags')",
    )
    parser.add_argument(
        "--embedding-model", default="all-MiniLM-L6-v2",
        help="Sentence-transformers model name (default: all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--no-cascade", action="store_true",
        help="Skip cascade analysis (faster for large datasets)",
    )
    parser.add_argument(
        "--llm-judge", action="store_true",
        help="Use LLM-as-Judge backend (higher resolution, slower, needs GPU)",
    )
    parser.add_argument(
        "--hybrid", action="store_true",
        help="Use Hybrid backend — text features for E-I/E-II + optional LLM for E-III (no GPU needed in pure text mode)",
    )
    parser.add_argument(
        "--paper", action="store_true",
        help="Generate publication-quality matplotlib figures alongside report",
    )
    parser.add_argument(
        "--paper-dir", default="paper_figures",
        help="Output directory for paper figures (default: paper_figures)",
    )
    parser.add_argument(
        "--judge-model", default="Qwen/Qwen2.5-7B-Instruct",
        help="Model for LLM judge (default: Qwen/Qwen2.5-7B-Instruct)",
    )
    parser.add_argument(
        "--device", default="cuda",
        help="Device for LLM judge: cuda, cpu, mps (default: cuda)",
    )

    args = parser.parse_args()

    # ---- 数据加载 ----
    print("=" * 60)
    print("Synthetic Data Decay Monitor v0.1.0")
    print("=" * 60)

    if args.demo:
        print(f"\n[Demo] Generating {args.demo}-generation synthetic lineage...")
        demo_texts = [
            "The derivative of x^2 is 2x. To solve the equation, we apply the chain rule.",
            "Python functions are defined using the 'def' keyword. They can accept arguments and return values.",
            "The capital of France is Paris. The Eiffel Tower was completed in 1889 and stands 330 meters tall.",
            "Therefore, based on the evidence presented, we can conclude that the hypothesis is supported.",
            "In the quiet village, the baker rose before dawn each day, kneading dough with hands that knew every curve.",
            "The water cycle involves evaporation, condensation, and precipitation. These processes are driven by solar energy.",
            "To optimize the algorithm, we can use dynamic programming to cache intermediate results and reduce complexity.",
            "The novel explores themes of identity and belonging through the eyes of a narrator who has lost both.",
            "Please provide a step-by-step solution showing all work and explaining the reasoning at each step.",
            "The study followed 10,000 participants over 20 years, tracking cardiovascular outcomes against dietary patterns.",
        ] * 3
        lineage = generate_synthetic_lineage(
            demo_texts, n_generations=args.demo, decay_pattern={"*": 0.25}
        )
        print(f"  Generated {lineage.n_samples} samples across {lineage.n_generations} generations")
    elif args.input:
        print(f"\n[Loading] {args.input}")
        lineage = parse_lineage_from_jsonl(
            args.input,
            text_field=args.text_field,
            generation_field=args.generation_field,
            capability_field=args.capability_field,
        )
        print(f"  Loaded {lineage.n_samples} samples, {lineage.n_generations} generations")
    else:
        parser.print_help()
        return 1

    # ---- 约束提取 ----
    print("\n[Extracting] Constraint vectors...")
    judge_model_name = None

    if args.hybrid:
        print("  Backend: Hybrid (text features, no GPU needed)")
        extractor = HybridConstraintExtractor(judge_fn=None)
    elif args.llm_judge:
        print(f"  Backend: LLM-as-Judge ({args.judge_model})")
        # GPU上自动启用fast_mode（省显存，不需要hidden state提取）
        device = args.device if hasattr(args, 'device') else "cuda"
        try:
            extractor = create_local_llm_judge(
                model_name=args.judge_model,
                device=device,
                fast_mode=(device == "cuda"),
            )
            judge_model_name = args.judge_model
        except Exception as e:
            print(f"  ERROR loading LLM judge: {e}")
            print("  Falling back to embedding extractor...")
            extractor = EmbeddingConstraintExtractor(embedding_model=args.embedding_model)
    else:
        print(f"  Backend: Embedding ({args.embedding_model})")
        try:
            extractor = EmbeddingConstraintExtractor(
                embedding_model=args.embedding_model
            )
            gen0_texts = [s.text for s in lineage.generations.get(0, [])[:20]]
            if gen0_texts:
                extractor.calibrate_fact_centroid(gen0_texts)
                print(f"  Calibrated fact centroid from {len(gen0_texts)} Gen-0 samples")
        except ImportError as e:
            print(f"  ERROR: {e}")
            print("  Install: pip install sentence-transformers")
            return 1

    # ---- 衰减引擎 ----
    print("\n[Analyzing] Decay trajectories...")
    engine = DecayEngine(lineage, extractor)
    engine.run_all_capabilities()

    trajectories = engine.get_all_trajectories()
    collapse_order = engine.get_collapse_order()

    for t in trajectories:
        if "trajectory" in t and t["trajectory"]:
            last = t["trajectory"][-1]
            print(f"  {t['capability']}: S_{last['generation']}={last['S_n']:.3f} "
                  f"β={last['beta']:.3f} [{last['status']}] "
                  f"→ collapse gen {t['predicted_collapse_gen']}")

    # ---- 执行者分类 ----
    print("\n[Diagnosing] Executor degradation types...")
    classifier = ExecutorClassifier()
    diagnoses = []
    for cap, snapshots in engine._snapshots.items():
        diag = diagnose_executor_decay(
            engine._trajectories.get(cap, []),
            snapshots,
            capability=cap,
        )
        diagnoses.append(diag)
        print(f"  {cap}: {diag['diagnosis'].degradation_type} "
              f"(severity={diag['diagnosis'].severity:.2f}) "
              f"→ {diag['diagnosis'].intervention_type}")

    # ---- 应力传播 ----
    cascade = None
    if not args.no_cascade and collapse_order:
        print("\n[Simulating] Stress propagation cascade...")
        stability_map = {
            c["capability"]: c["current_S_n"]
            for c in collapse_order
        }
        cascade = run_cascade_analysis(collapse_order, stability_map)
        print(f"  First to collapse: {cascade.get('first_to_collapse', '?')}")
        print(f"  Cascade steps: {cascade.get('total_cascade_steps', 0)}")
        print(f"  Weakest edge: {cascade.get('weakest_edge', {}).get('edge', 'N/A')}")

    # ---- 报告生成 ----
    print("\n[Generating] Report...")
    json_report = generate_json_report(
        lineage=lineage,
        trajectories=trajectories,
        diagnoses=diagnoses,
        cascade=cascade,
        meta={
            "input_file": args.input,
            "backend": "llm-judge" if args.llm_judge else "embedding",
            "embedding_model": args.embedding_model,
            "judge_model": judge_model_name,
        },
    )

    output_path = args.output
    fmt = args.format
    if fmt is None:
        if output_path.endswith(".html"):
            fmt = "html"
        elif output_path.endswith(".json"):
            fmt = "json"
        else:
            fmt = "html"

    if fmt == "html":
        generate_html_report(json_report, output_path)
    else:
        with open(output_path, "w") as f:
            json.dump(json_report, f, indent=2, ensure_ascii=False)

    # Always export JSON companion (for reproducibility)
    json_path = output_path.rsplit(".", 1)[0] + "_data.json"
    with open(json_path, "w") as f:
        json.dump(json_report, f, indent=2, ensure_ascii=False)

    # Paper figures
    if args.paper:
        print("\n[Generating] Paper figures...")
        from synthetic_decay_monitor.report import generate_paper_figures
        paper_outputs = generate_paper_figures(engine, json_report, output_dir=args.paper_dir)
        print(f"  {len(paper_outputs)} figures → {args.paper_dir}/")
        for p in paper_outputs:
            print(f"    {os.path.basename(p)}")

    summary = json_report.get("warnings", [])
    n_critical = len([w for w in summary if w.get("severity") == "critical"])
    print(f"\n{'=' * 60}")
    print(f"Report: {output_path}")
    print(f"Status: {n_critical} critical warnings")
    if n_critical > 0:
        print("ACTION REQUIRED — see report for intervention recommendations.")
    else:
        print("No critical issues detected. Monitor regularly.")
    print(f"{'=' * 60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
