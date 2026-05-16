"""GPT-4o cross-model recursive generation experiment.

Compares GPT-4o degradation dynamics against Qwen2.5-7B baseline.
Produces: experiment_data/gpt4o_lineage.jsonl, gpt4o_report.json
"""
import sys, os, json, time, argparse
from datetime import datetime
from collections import defaultdict

from openai import OpenAI
from synthetic_decay_monitor.data_lineage import DataSample, DatasetLineage
from synthetic_decay_monitor.constraint_extractor import HybridConstraintExtractor
from synthetic_decay_monitor.decay_engine import DecayEngine
from synthetic_decay_monitor.executor_classifier import diagnose_executor_decay
from synthetic_decay_monitor.report import generate_json_report

SEED_PROMPTS = [
    ("math_reasoning", "Prove that the square root of 2 is irrational using proof by contradiction. Show each step clearly."),
    ("math_reasoning", "Solve the integral of x*sin(x) from 0 to pi, explaining the integration by parts method."),
    ("code_generation", "Write a Python function that implements the quicksort algorithm with in-place partitioning."),
    ("code_generation", "Write a JavaScript function that debounces an async API call with cancellation support."),
    ("factual_knowledge", "Explain the causes and key events of the French Revolution, including dates and major figures."),
    ("factual_knowledge", "Describe the process of photosynthesis, including the light-dependent and light-independent reactions."),
    ("logical_consistency", "If all A are B, and some B are C, what can we conclude about A and C? Explain the logical reasoning."),
    ("logical_consistency", "Analyze the following argument for fallacies: 'Since the policy reduced crime in New York, it will reduce crime everywhere.'"),
    ("creative_writing", "Write a short story about an astronomer who discovers that the stars are going out one by one."),
    ("creative_writing", "Compose a poem about the relationship between silence and understanding."),
    ("general", "Explain the importance of biodiversity for ecosystem stability, with specific examples."),
    ("general", "Describe the water cycle and how climate change is affecting it."),
]

SYSTEM_PROMPT = "You are a helpful AI assistant. Generate a detailed, informative response. Write naturally and comprehensively."


def generate_gpt4o(client, prompt: str, model: str = "gpt-4o", max_tokens: int = 512) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=0.8,
        top_p=0.95,
    )
    return resp.choices[0].message.content.strip()


def run_experiment(client, n_generations: int = 3, output_dir: str = "experiment_data"):
    os.makedirs(output_dir, exist_ok=True)
    samples = []
    prev_texts = [p[1] for p in SEED_PROMPTS]
    prev_tags = [[p[0]] for p in SEED_PROMPTS]

    # Gen 0: human seeds
    for i, (tags, text) in enumerate(zip(prev_tags, prev_texts)):
        samples.append(DataSample(
            text=text, generation=0, source_model="human",
            capability_tags=tags, sample_id=f"G0_{i:04d}",
        ))

    # Gen 1-N: recursive GPT-4o generation
    for gen in range(1, n_generations + 1):
        print(f"\n[Gen {gen}] Generating {len(prev_texts)} responses...")
        curr_texts = []
        for i, text in enumerate(prev_texts):
            try:
                resp = generate_gpt4o(client, text)
                curr_texts.append(resp)
                print(f"  [{i+1}/{len(prev_texts)}] {len(resp)} chars")
            except Exception as e:
                print(f"  [{i+1}/{len(prev_texts)}] ERROR: {e}")
                curr_texts.append(text)
            time.sleep(0.3)

        for i, text in enumerate(curr_texts):
            samples.append(DataSample(
                text=text, generation=gen,
                source_model="gpt-4o",
                capability_tags=prev_tags[i],
                sample_id=f"G{gen}_{i:04d}",
            ))
        prev_texts = curr_texts

    lineage = DatasetLineage(samples=samples)

    # Save JSONL
    jsonl_path = os.path.join(output_dir, "gpt4o_lineage.jsonl")
    with open(jsonl_path, "w") as f:
        for s in sorted(samples, key=lambda x: (x.generation, x.sample_id)):
            f.write(json.dumps({
                "id": s.sample_id, "text": s.text,
                "generation": s.generation, "source_model": s.source_model,
                "capability_tags": s.capability_tags,
            }, ensure_ascii=False) + "\n")
    print(f"\nSaved: {jsonl_path} ({len(samples)} samples)")

    return lineage


def run_analysis(lineage, output_dir: str = "experiment_data"):
    print("\n[Analysis] Running DecayEngine...")
    extractor = HybridConstraintExtractor(judge_fn=None)
    engine = DecayEngine(lineage, extractor)
    engine.run_all_capabilities()

    trajectories = engine.get_all_trajectories()
    collapse_order = engine.get_collapse_order()

    print("\n=== GPT-4o Stability Trajectories ===")
    for t in trajectories:
        if "trajectory" in t and t["trajectory"]:
            last = t["trajectory"][-1]
            print(f"  {t['capability']}: S_{last['generation']}={last['S_n']:.3f} "
                  f"beta={last['beta']:.3f} [{last.get('status','?')}] "
                  f"-> collapse gen {t.get('predicted_collapse_gen', -1)}")

    # Executor diagnoses
    diagnoses = []
    for cap, snapshots in engine._snapshots.items():
        diag = diagnose_executor_decay(
            engine._trajectories.get(cap, []), snapshots, capability=cap
        )
        diagnoses.append(diag)
        print(f"  {cap}: {diag['diagnosis'].degradation_type} "
              f"(sev={diag['diagnosis'].severity:.2f})")

    # Global beta
    betas = []
    for t in trajectories:
        if "trajectory" in t:
            for g in t["trajectory"]:
                if "beta" in g and g["beta"] > 0:
                    betas.append(g["beta"])
    global_beta = sum(betas) / len(betas) if betas else 0.0
    print(f"\nGlobal beta: {global_beta:.4f}")

    # Save report
    json_report = generate_json_report(
        lineage=lineage, trajectories=trajectories,
        diagnoses=diagnoses,
    )
    report_path = os.path.join(output_dir, "gpt4o_report.json")
    with open(report_path, "w") as f:
        json.dump(json_report, f, indent=2, ensure_ascii=False)
    print(f"Report: {report_path}")

    return trajectories, collapse_order, global_beta


def compare_with_qwen(gpt4o_beta: float, output_dir: str = "experiment_data"):
    """Compare GPT-4o results with Qwen baseline."""
    qwen_jsonl = os.path.join(output_dir, "real_lineage.jsonl")
    if not os.path.exists(qwen_jsonl):
        qwen_jsonl = "/Users/dengxinhang/paper/experiment_data/real_lineage.jsonl"

    if os.path.exists(qwen_jsonl):
        from synthetic_decay_monitor.data_lineage import parse_lineage_from_jsonl
        qwen_lineage = parse_lineage_from_jsonl(qwen_jsonl)
        qwen_extractor = HybridConstraintExtractor(judge_fn=None)
        qwen_engine = DecayEngine(qwen_lineage, qwen_extractor)
        qwen_engine.run_all_capabilities()
        qwen_traj = qwen_engine.get_all_trajectories()

        qwen_betas = []
        for t in qwen_traj:
            if "trajectory" in t:
                for g in t["trajectory"]:
                    if "beta" in g and g["beta"] > 0:
                        qwen_betas.append(g["beta"])
        qwen_beta = sum(qwen_betas) / len(qwen_betas) if qwen_betas else 0.0
    else:
        qwen_beta = 0.178

    print(f"\n=== Cross-Model Comparison ===")
    print(f"Qwen2.5-7B beta:  {qwen_beta:.4f}")
    print(f"GPT-4o beta:       {gpt4o_beta:.4f}")
    print(f"Ratio (GPT4o/Qwen): {gpt4o_beta/qwen_beta:.2f}x" if qwen_beta > 0 else "")

    comparison = {
        "qwen_beta": qwen_beta,
        "gpt4o_beta": gpt4o_beta,
        "ratio": gpt4o_beta / qwen_beta if qwen_beta > 0 else None,
        "within_hypothesis_range": 0.05 <= gpt4o_beta <= 0.25,
    }
    comp_path = os.path.join(output_dir, "gpt4o_comparison.json")
    with open(comp_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"Comparison: {comp_path}")
    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--n-generations", type=int, default=3)
    parser.add_argument("--output-dir", default="experiment_data")
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--jsonl", help="Use existing JSONL instead of generating")
    args = parser.parse_args()

    if args.skip_generation or args.jsonl:
        jsonl_path = args.jsonl or os.path.join(args.output_dir, "gpt4o_lineage.jsonl")
        from synthetic_decay_monitor.data_lineage import parse_lineage_from_jsonl
        lineage = parse_lineage_from_jsonl(jsonl_path)
    else:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("ERROR: OPENAI_API_KEY not set")
            sys.exit(1)
        client = OpenAI(api_key=api_key)
        lineage = run_experiment(client, n_generations=args.n_generations, output_dir=args.output_dir)

    trajectories, collapse_order, gpt4o_beta = run_analysis(lineage, output_dir=args.output_dir)
    compare_with_qwen(gpt4o_beta, output_dir=args.output_dir)
