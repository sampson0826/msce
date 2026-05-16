"""Real recursive generation experiment: Qwen2.5-7B → gen0→gen1→...→gen5 → decay monitor.

Produces:
  - data/real_lineage.jsonl     — all generations with metadata
  - data/real_report.json       — full diagnostic report
  - data/real_summary.txt       — human-readable terminal output (for GIF)
"""

import sys, os, json, time, argparse
from datetime import datetime
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from synthetic_decay_monitor.data_lineage import DataSample, DatasetLineage
from synthetic_decay_monitor.constraint_extractor import HybridConstraintExtractor
from synthetic_decay_monitor.decay_engine import DecayEngine
from synthetic_decay_monitor.executor_classifier import diagnose_executor_decay
from synthetic_decay_monitor.report import generate_json_report, generate_paper_figures


# Seed prompts covering different capability dimensions
SEED_PROMPTS = [
    # math_reasoning
    "Prove that the square root of 2 is irrational using proof by contradiction. Show each step clearly.",
    "Solve the integral of x*sin(x) from 0 to pi, explaining the integration by parts method.",
    # code_generation
    "Write a Python function that implements the quicksort algorithm with in-place partitioning.",
    "Write a JavaScript function that debounces an async API call with cancellation support.",
    # factual_knowledge
    "Explain the causes and key events of the French Revolution, including dates and major figures.",
    "Describe the process of photosynthesis, including the light-dependent and light-independent reactions.",
    # logical_consistency
    "If all A are B, and some B are C, what can we conclude about A and C? Explain the logical reasoning.",
    "Analyze the following argument for fallacies: 'Since the policy reduced crime in New York, it will reduce crime everywhere.'",
    # creative_writing
    "Write a short story about an astronomer who discovers that the stars are going out one by one.",
    "Compose a poem about the relationship between silence and understanding.",
    # general
    "Explain the importance of biodiversity for ecosystem stability, with specific examples.",
    "Describe the water cycle and how climate change is affecting it.",
]

GENERATION_PROMPT = """You are a helpful AI assistant. Generate a detailed, informative response to the following prompt. Write naturally and comprehensively.

Prompt: {prompt}

Response:"""


def load_model(model_name: str = "Qwen/Qwen2.5-7B-Instruct", device: str = "cuda"):
    print(f"[Load] {model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True, local_files_only=True
    )
    model.eval()
    return model, tokenizer


def generate_responses(model, tokenizer, prompts: list[str], max_new_tokens: int = 256) -> list[str]:
    responses = []
    for prompt in prompts:
        formatted = GENERATION_PROMPT.format(prompt=prompt)
        inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=max_new_tokens, temperature=0.8,
                top_p=0.95, do_sample=True, pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        responses.append(response.strip())
    return responses


def auto_tag(text: str) -> list[str]:
    """Lightweight capability tagging."""
    text_lower = text.lower()
    tags = []
    if any(w in text_lower for w in ["math", "calculate", "equation", "solve", "proof", "integral", "theorem"]):
        tags.append("math_reasoning")
    if any(w in text_lower for w in ["code", "function", "python", "javascript", "algorithm", "def ", "class "]):
        tags.append("code_generation")
    if any(w in text_lower for w in ["fact", "history", "science", "capital", "date", "century", "process"]):
        tags.append("factual_knowledge")
    if any(w in text_lower for w in ["therefore", "because", "however", "logic", "conclusion", "argument", "fallacy"]):
        tags.append("logical_consistency")
    if any(w in text_lower for w in ["story", "poem", "creative", "narrative", "character"]):
        tags.append("creative_writing")
    if not tags:
        tags.append("general")
    return tags


def run_experiment(
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    n_generations: int = 5,
    max_new_tokens: int = 256,
    output_dir: str = "experiment_data",
    skip_generation: bool = False,
):
    os.makedirs(output_dir, exist_ok=True)

    # ---- Phase 1: Recursive Generation ----
    samples = []
    prev_prompts = SEED_PROMPTS[:]

    if not skip_generation:
        model, tokenizer = load_model(model_name)

    for gen in range(n_generations + 1):
        print(f"\n{'='*60}")
        print(f"GENERATION {gen}")
        print(f"{'='*60}")

        if gen == 0:
            texts = SEED_PROMPTS  # gen0 = original seed prompts
            source_model = "seed"
            for i, text in enumerate(texts):
                tags = auto_tag(text)
                samples.append(DataSample(
                    text=text, generation=0, source_model=source_model,
                    capability_tags=tags, sample_id=f"G0_{i:04d}",
                ))
            print(f"  {len(texts)} seed prompts → gen 0 baseline")
            # Generate first batch of responses from seeds
            prev_prompts = generate_responses(model, tokenizer, SEED_PROMPTS, max_new_tokens)
            continue

        # gen ≥ 1: use previous generation's outputs as new prompts
        texts = prev_prompts
        source_model = f"qwen-gen{gen}"
        for i, text in enumerate(texts):
            if not text or len(text.split()) < 5:
                continue
            tags = auto_tag(text)
            samples.append(DataSample(
                text=text, generation=gen, source_model=source_model,
                capability_tags=tags, sample_id=f"G{gen}_{i:04d}",
            ))

        n_valid = sum(1 for t in texts if t and len(t.split()) >= 5)
        print(f"  {n_valid}/{len(texts)} valid responses at gen {gen}")

        # Generate next generation (except at final gen)
        if gen < n_generations:
            valid_texts = [t for t in texts if t and len(t.split()) >= 10]
            if len(valid_texts) >= 3:
                # Use these as prompts for next gen: take first sentence as prompt
                next_prompts = []
                for t in valid_texts:
                    sentences = t.replace("!", ".").replace("?", ".").split(".")
                    prompt = sentences[0].strip()[:200]
                    if len(prompt.split()) >= 3:
                        next_prompts.append(prompt)
                # Pad to match seed count
                while len(next_prompts) < len(SEED_PROMPTS):
                    next_prompts.append(next_prompts[len(next_prompts) % max(1, len(next_prompts))])
                prev_prompts = generate_responses(model, tokenizer, next_prompts[:len(SEED_PROMPTS)], max_new_tokens)
            else:
                print(f"  ⚠ Too few valid texts, stopping generation")
                break

    # Save lineage
    lineage_path = os.path.join(output_dir, "real_lineage.jsonl")
    with open(lineage_path, "w") as f:
        for s in samples:
            f.write(json.dumps({
                "id": s.sample_id, "text": s.text, "generation": s.generation,
                "source_model": s.source_model, "capability_tags": s.capability_tags,
            }, ensure_ascii=False) + "\n")

    lineage = DatasetLineage(samples=samples)
    print(f"\n[Save] {len(samples)} samples → {lineage_path}")
    print(f"  Generations: {lineage.n_generations}")
    print(f"  Capabilities: {list(lineage.capability_coverage.keys())}")

    # ---- Phase 2: Decay Monitoring ----
    print(f"\n{'='*60}")
    print("DECAY MONITORING")
    print(f"{'='*60}")

    extractor = HybridConstraintExtractor(judge_fn=None)
    engine = DecayEngine(lineage, extractor)
    engine.run_all_capabilities()

    # Diagnoses
    diagnoses = []
    for cap, snapshots in engine._snapshots.items():
        if not snapshots:
            continue
        diag = diagnose_executor_decay(
            engine._trajectories.get(cap, []), snapshots, capability=cap
        )
        diagnoses.append(diag)

    # Build report
    trajectories = engine.get_all_trajectories()
    json_report = generate_json_report(
        lineage=lineage, trajectories=trajectories, diagnoses=diagnoses,
        meta={"experiment": "real_recursive_qwen", "model": model_name},
    )

    report_path = os.path.join(output_dir, "real_report.json")
    with open(report_path, "w") as f:
        json.dump(json_report, f, indent=2, ensure_ascii=False)

    # ---- Phase 3: Terminal Summary (GIF-ready) ----
    summary_lines = []
    sep = "=" * 60
    summary_lines.append(sep)
    summary_lines.append("SYNTHETIC DATA DECAY MONITOR — Real Experiment")
    summary_lines.append(f"Model: {model_name} | {n_generations} recursive generations | {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    summary_lines.append(sep)
    summary_lines.append("")

    for t in trajectories:
        if "error" in t or not t.get("trajectory"):
            continue
        cap = t["capability"]
        traj = t["trajectory"]
        sn_vals = [g["S_n"] for g in traj]
        beta = t.get("predicted_collapse_gen", 0)

        # Bar chart in terminal
        bar_len = 20
        filled = int(sn_vals[-1] * bar_len)
        bar = "█" * filled + "░" * (bar_len - filled)

        # Find diagnosis
        diag = next((d for d in diagnoses if d.get("capability") == cap), {})
        dtype = diag.get("diagnosis", {})
        if hasattr(dtype, 'degradation_type'):
            dtype_str = dtype.degradation_type
            severity = dtype.severity
        else:
            dtype_str = dtype.get("degradation_type", "?")
            severity = dtype.get("severity", 0)

        status = "💀 COLLAPSED" if sn_vals[-1] < 0.3 else "⚠ CRITICAL" if sn_vals[-1] < 0.5 else "✓ STABLE"
        summary_lines.append(f"  {cap:<24s} {bar} {dtype_str:<14s} {status}")

    summary_lines.append("")
    summary_lines.append(sep)
    summary_lines.append("DIAGNOSIS SUMMARY")
    summary_lines.append(sep)

    dtype_counts = Counter()
    for d in diagnoses:
        diag = d.get("diagnosis", {})
        dt = getattr(diag, 'degradation_type', '?') if hasattr(diag, 'degradation_type') else diag.get("degradation_type", "?")
        dtype_counts[dt] += 1

    for dt, count in dtype_counts.most_common():
        if "E-I" in str(dt):
            label = "E-I (Axiom collapse — logic chain breaking)"
            fix = "→ Add formal proofs + reasoning chains"
        elif "E-II" in str(dt):
            label = "E-II (Scale collapse — style/vocabulary erosion)"
            fix = "→ Add diverse style exemplars + human preference data"
        elif "E-III" in str(dt):
            label = "E-III (Boundary collapse — fact erosion)"
            fix = "→ Add edge cases + domain-specific boundary data"
        else:
            label = f"{dt} (Mixed degradation)"
            fix = "→ Add mixed calibration data"
        summary_lines.append(f"  {count}x {label}")
        summary_lines.append(f"       {fix}")

    summary_lines.append("")
    summary_lines.append(sep)
    summary_lines.append("COLLAPSE TIMELINE")
    summary_lines.append(sep)
    collapse_order = sorted(
        [t for t in trajectories if "error" not in t],
        key=lambda t: t["predicted_collapse_gen"],
    )
    for i, t in enumerate(collapse_order):
        flag = "🔴 FIRST TO COLLAPSE" if i == 0 else "🟡" if i <= 2 else "🟢"
        summary_lines.append(
            f"  {flag} {t['capability']:<24s} → gen {t['predicted_collapse_gen']}"
        )

    summary_lines.append("")
    summary_lines.append(sep)
    summary_lines.append("INTERVENTION PRIORITY")
    summary_lines.append(sep)
    summary_lines.append("  1. Add axiom data (math proofs, formal logic) — most urgent")
    summary_lines.append("  2. Add calibration data (human preferences, style diversity)")
    summary_lines.append("  3. Add boundary data (edge cases, rare facts) — preventive")
    summary_lines.append("")
    summary_lines.append("Run: decay-monitor watch --input your_data.jsonl")

    summary_text = "\n".join(summary_lines)

    summary_path = os.path.join(output_dir, "real_summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary_text)

    print(summary_text)
    print(f"\n[Save] Report → {report_path}")
    print(f"[Save] Summary → {summary_path}")

    return json_report, summary_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--output-dir", default="experiment_data")
    parser.add_argument("--skip-generation", action="store_true",
                        help="Skip generation, just run monitor on existing data")
    parser.add_argument("--jsonl", help="Path to existing JSONL for --skip-generation")
    args = parser.parse_args()

    if args.skip_generation and args.jsonl:
        from synthetic_decay_monitor.data_lineage import parse_lineage_from_jsonl
        lineage = parse_lineage_from_jsonl(args.jsonl)
        extractor = HybridConstraintExtractor(judge_fn=None)
        engine = DecayEngine(lineage, extractor)
        engine.run_all_capabilities()
        diagnoses = []
        for cap, snapshots in engine._snapshots.items():
            if not snapshots:
                continue
            diag = diagnose_executor_decay(
                engine._trajectories.get(cap, []), snapshots, capability=cap
            )
            diagnoses.append(diag)
        trajectories = engine.get_all_trajectories()
        json_report = generate_json_report(
            lineage=lineage, trajectories=trajectories, diagnoses=diagnoses,
            meta={"experiment": "real_recursive_qwen", "model": args.model},
        )
        report_path = os.path.join(args.output_dir, "real_report.json")
        os.makedirs(args.output_dir, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(json_report, f, indent=2, ensure_ascii=False)
        print(f"[Save] Report → {report_path}")
    else:
        run_experiment(
            model_name=args.model, n_generations=args.generations,
            max_new_tokens=args.max_tokens, output_dir=args.output_dir,
            skip_generation=args.skip_generation,
        )
