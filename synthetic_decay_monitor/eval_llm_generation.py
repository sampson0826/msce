"""Real LLM Generation Decay Experiment v2.
Uses Qwen2.5-7B with max_new_tokens=512 to generate 100+ word texts.
"""
import json, time, numpy as np

from synthetic_decay_monitor.data_lineage import (
    generate_synthetic_lineage, DataSample, DatasetLineage, _apply_decay, _auto_tag,
)
from synthetic_decay_monitor.constraint_extractor import (
    HybridConstraintExtractor, extract_text_features,
)
from synthetic_decay_monitor.decay_engine import (
    DecayEngine, S_CRITICAL, BASE_ALPHAS,
)
from synthetic_decay_monitor.executor_classifier import (
    diagnose_executor_decay, ExecutorClassifier,
)

# Rich prompts requesting long, marker-dense texts
CAPABILITY_PROMPTS = {
    "math_reasoning": [
        "Write a detailed 250-word mathematical proof showing why the square root of 2 is irrational. Use logical connectors (therefore, because, hence, thus, consequently). Include specific numbers and mathematical notation.",
    ],
    "factual_knowledge": [
        "Write a detailed 250-word description of the French Revolution (1789-1799). Include specific dates, numbers, proper nouns (Paris, Louis XVI, Robespierre, Bastille, Napoleon). Use connectors like therefore and however.",
    ],
    "code_generation": [
        "Write a detailed 250-word tutorial explaining the binary search algorithm with a complete Python implementation. Include time complexity (O(log n)), edge cases, and logical reasoning with connectors like therefore and because.",
    ],
    "logical_consistency": [
        "Write a 250-word logical analysis: Because atmospheric CO2 has risen from 280ppm to 420ppm since 1850 (a 50% increase), Earth's average temperature has risen 1.1 degrees Celsius. However, some regions show cooling trends. Resolve this apparent contradiction using logical reasoning.",
    ],
}

SYSTEM_PROMPT = "You are a helpful AI assistant. Provide detailed, accurate answers with specific facts, numbers, proper nouns, and logical connectors (therefore, because, however, thus, consequently)."


def generate_long_texts(model, tokenizer, n_per_capability=2):
    """Generate 150+ word texts from Qwen2.5-7B with longer token limit."""
    import torch
    all_samples = []
    for capability, prompts in CAPABILITY_PROMPTS.items():
        for i, prompt in enumerate(prompts[:n_per_capability]):
            try:
                full_prompt = f"{SYSTEM_PROMPT}\n\nUser: {prompt}\nAssistant:"
                inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs, max_new_tokens=512, temperature=0.7, do_sample=True,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
                sample = DataSample(
                    text=text,
                    generation=0,
                    source_model="Qwen2.5-7B-Instruct",
                    capability_tags=[capability],
                )
                all_samples.append(sample)
                wc = len(text.split())
                print(f"  [{capability}] Generated {len(text)} chars, {wc} words")
            except Exception as e:
                print(f"  [{capability}] ERROR: {e}")
    return all_samples


def run_llm_generation_experiment():
    print("=" * 60)
    print("REAL LLM GENERATION DECAY EXPERIMENT v2")
    print("=" * 60)

    # Load model for generation (not via judge)
    print("\n[1/5] Loading Qwen2.5-7B-Instruct for text generation...")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto",
        trust_remote_code=True, attn_implementation="sdpa",
    )
    model.eval()
    print(f"  Model loaded, GPU mem: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    # Generate texts (longer: max_new_tokens=512)
    print("\n[2/5] Generating texts from Qwen2.5-7B (max_new_tokens=512)...")
    gen0_samples = generate_long_texts(model, tokenizer, n_per_capability=2)
    print(f"  Generated {len(gen0_samples)} samples")

    # Run hybrid extractor on gen0 texts
    print("\n[3/5] Hybrid extractor on gen0...")
    for sample in gen0_samples:
        features = extract_text_features(sample.text)
        cap = sample.capability_tags[0]
        wc = len(sample.text.split())
        print(f"  {cap}: logic={features['ei_logic_density']:.3f} filler={features['eii_filler_ratio']:.3f} proper={features['eiii_proper_case_ratio']:.3f} len={wc}w")

    # Apply controlled decay and test executor recovery
    print("\n[4/5] Executor recovery test with LLM-generated base texts...")
    experiments = {}
    for exec_type, exec_mix in [
        ("E-I_dominant", {"E-I": 0.8, "E-II": 0.1, "E-III": 0.1}),
        ("E-II_dominant", {"E-I": 0.1, "E-II": 0.8, "E-III": 0.1}),
        ("E-III_dominant", {"E-I": 0.1, "E-II": 0.1, "E-III": 0.8}),
        ("balanced", {"E-I": 0.33, "E-II": 0.33, "E-III": 0.34}),
    ]:
        texts = [s.text for s in gen0_samples]
        tags_list = [s.capability_tags for s in gen0_samples]

        lineage_samples = list(gen0_samples)
        for gen in range(1, 4):
            for i, (text, tags) in enumerate(zip(texts, tags_list)):
                decayed = _apply_decay(text, generation=gen, beta=0.15, tags=tags, executor_mix=exec_mix)
                lineage_samples.append(DataSample(
                    text=decayed, generation=gen,
                    source_model="decay_sim", capability_tags=tags,
                ))

        exp_lineage = DatasetLineage(samples=lineage_samples)
        extractor = HybridConstraintExtractor(judge_fn=None)
        engine = DecayEngine(exp_lineage, extractor)
        engine.run_all_capabilities()

        classifier = ExecutorClassifier()
        diagnoses = []
        for cap, snapshots in engine._snapshots.items():
            diag = diagnose_executor_decay(engine._trajectories.get(cap, []), snapshots, capability=cap)
            diagnoses.append(diag)

        dtype_votes = {}
        for d in diagnoses:
            diag = d.get("diagnosis")
            dt = getattr(diag, "degradation_type", "unknown") if hasattr(diag, "degradation_type") else "unknown"
            dtype_votes[dt] = dtype_votes.get(dt, 0) + 1

        comps = []
        for cap, traj in engine._trajectories.items():
            if traj:
                last = traj[-1]
                comps.append(last.executor_composition)
        avg_comp = {}
        if comps:
            for k in ["E-I", "E-II", "E-III"]:
                avg_comp[k] = np.mean([c.get(k, 0) for c in comps])

        # Determine match
        expected = "E-I_loss" if exec_type == "E-I_dominant" else \
                   "E-II_loss" if exec_type == "E-II_dominant" else \
                   "E-III_loss" if exec_type == "E-III_dominant" else "mixed"
        majority_vote = max(dtype_votes, key=dtype_votes.get) if dtype_votes else "unknown"
        match = "MATCH" if majority_vote == expected else f"MISMATCH (got {majority_vote})"

        experiments[exec_type] = {
            "injected": exec_mix,
            "expected": expected,
            "diagnosis_votes": dtype_votes,
            "avg_composition": avg_comp,
            "result": match,
        }
        print(f"  {exec_type}: {match} | votes={dtype_votes} | E-I={avg_comp.get('E-I',0):.2f} E-II={avg_comp.get('E-II',0):.2f} E-III={avg_comp.get('E-III',0):.2f}")

    # Summary
    print("\n[5/5] Results Summary")
    print("-" * 40)
    correct = sum(1 for v in experiments.values() if "MATCH" in v["result"])
    print(f"  Overall: {correct}/{len(experiments)} = {correct/len(experiments)*100:.0f}%")
    for name, exp in experiments.items():
        print(f"    {name}: {exp['result']}")

    return {"experiments": experiments}


if __name__ == "__main__":
    results = run_llm_generation_experiment()
    output_path = os.path.join(os.path.dirname(__file__), "llm_generation_results_v2.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")
