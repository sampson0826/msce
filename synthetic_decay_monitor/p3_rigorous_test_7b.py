"""P3 rigorous test (7B): component-wise π validation via hidden state extraction.

Qwen2.5-7B-Instruct on RTX 5090 32GB. Memory-optimized:
- 3 seeds, 64 max_new_tokens, 256 max_length
- Dual-model: SDPA gen + eager extraction
"""
import sys, json, os
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hallucination_predictor"))

from hallucination_predictor.model_wrapper import ModelWrapper
from hallucination_predictor.constraint_functions import (
    ConstraintFunctionBank,
    compute_constraint_gradients,
    compute_residual,
)

P3_SEEDS = [
    ("math_reasoning", "Prove that the square root of 2 is irrational using proof by contradiction."),
    ("code_generation", "Write a Python function that implements quicksort."),
    ("factual_knowledge", "Explain how photosynthesis works in plants."),
]

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
GENERATIONS = 3
MAX_NEW_TOKENS = 64
MAX_LENGTH = 256


def _generate_text(wrapper, prompt, max_tokens, temperature=0.8):
    """Generate text using SDPA model."""
    if hasattr(wrapper.tokenizer, 'apply_chat_template') and wrapper.tokenizer.chat_template:
        try:
            formatted = wrapper.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
            )
        except Exception:
            formatted = prompt
    else:
        formatted = prompt

    inputs = wrapper.tokenizer(formatted, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(wrapper.device)
    with torch.no_grad():
        gen_ids = wrapper.model.generate(
            **inputs, max_new_tokens=max_tokens, do_sample=True, temperature=temperature,
            pad_token_id=wrapper.tokenizer.pad_token_id,
        )
    return wrapper.tokenizer.decode(gen_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()


def run_p3_test():
    print("=" * 60)
    print("P3 Rigorous Test (7B): Constraint Attractor Collapse")
    print(f"Model: {MODEL_NAME}")
    print(f"Seeds: {len(P3_SEEDS)}, Generations: {GENERATIONS}")
    print(f"Max tokens/gen: {MAX_NEW_TOKENS}, Max length: {MAX_LENGTH}")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")

    print("\n[1/4] Loading extraction model (eager)...")
    wrapper = ModelWrapper(model_name=MODEL_NAME, device=device)

    print("[2/4] Loading generation model (SDPA)...")
    import transformers
    wrapper_sdpa = ModelWrapper(model_name=MODEL_NAME, device=device)
    # Re-load with SDPA for efficient generation
    del wrapper_sdpa.model
    torch.cuda.empty_cache()
    wrapper_sdpa.model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map=device,
        trust_remote_code=True, attn_implementation="sdpa",
    )
    wrapper_sdpa.model.eval()

    print("\n[3/4] Running recursive generation + hidden state extraction...")
    bank = ConstraintFunctionBank()
    all_results = {}

    for cap, seed in P3_SEEDS:
        print(f"\n  {cap}:")
        cap_results = []
        prompt = seed

        for gen in range(GENERATIONS + 1):
            if gen == 0:
                state = wrapper.extract_output_state(prompt)
            else:
                gen_text = _generate_text(wrapper_sdpa, prompt, MAX_NEW_TOKENS, temperature=0.8)
                state = wrapper.extract_output_state(gen_text)
                state.generated_text = gen_text
                prompt = gen_text

            attn = state.attention_weights
            constr_states = bank.compute_all(
                state.hidden_states.float(),
                [h.float() for h in state.layer_hidden_states],
                attn.float() if attn is not None else None,
            )

            grads = compute_constraint_gradients(constr_states)
            res_mags, cancels, totals = compute_residual(grads) if grads else ([], [], [])

            cap_results.append({
                "generation": gen,
                "n_tokens": len(constr_states),
                "n_grads": len(grads),
                "residual_mean": float(np.mean(res_mags)) if res_mags else 0.0,
                "residual_std": float(np.std(res_mags)) if res_mags else 0.0,
                "residual_cv": float(np.std(res_mags) / (np.mean(res_mags) + 1e-10)) if res_mags else 0.0,
                "cancel_mean": float(np.mean(cancels)) if cancels else 0.0,
                "total_constraint_mean": float(np.mean(totals)) if totals else 0.0,
                "pi_contributions": _compute_pi_components(grads),
            })

            status = "healthy" if cap_results[-1]["residual_cv"] < 0.5 else \
                     "degrading" if cap_results[-1]["residual_cv"] < 1.0 else "collapsed"
            print(f"    Gen{gen}: {len(constr_states)} tokens, "
                  f"||Pi||={cap_results[-1]['residual_mean']:.4f}+/-{cap_results[-1]['residual_std']:.4f}, "
                  f"C_div={cap_results[-1]['residual_cv']:.3f} [{status}]")

        all_results[cap] = cap_results

    # Analysis
    print("\n[4/4] Computing attractor collapse metrics...")
    analysis = _analyze_collapse(all_results)

    print(f"\n  Global C_div trajectory:")
    for gen in range(GENERATIONS + 1):
        cdivs = []
        for cap in all_results:
            if gen < len(all_results[cap]):
                cdivs.append(all_results[cap][gen]["residual_cv"])
        mean_cdiv = np.mean(cdivs) if cdivs else 0
        print(f"    Gen{gen}: C_div = {mean_cdiv:.4f}")

    print(f"\n  Collapse rate lambda_C: {analysis['lambda_c']:.4f}/gen")
    print(f"  R^2 of exponential fit: {analysis['r_squared']:.4f}")
    print(f"  Mean ||Pi|| stability: {analysis['mean_pi_cv']:.4f} (CV across gens)")
    print(f"  C_div at Gen{GENERATIONS}/Gen0 ratio: {analysis['cdiv_ratio']:.4f}")

    # Executor decomposition
    print("\n  Executor pi component decomposition:")
    pi_comp = analysis["pi_components"]
    total_contrib = sum(abs(v) for v in pi_comp.values())
    for comp, val in sorted(pi_comp.items(), key=lambda x: -abs(x[1])):
        pct = abs(val) / (total_contrib + 1e-10) * 100
        print(f"    {comp}: {val:+.4f} ({pct:.1f}%)")

    print(f"\n  Predicted vs observed executor weights:")
    predicted = {"E-I (syntax+coherence)": 0.40, "E-II (style)": 0.20, "E-III (fact)": 0.08}
    observed = {
        "E-I (syntax+coherence)": abs(pi_comp.get("sigma_syntax", 0)) + abs(pi_comp.get("sigma_coherence", 0)),
        "E-II (style)": abs(pi_comp.get("sigma_style", 0)),
        "E-III (fact)": abs(pi_comp.get("sigma_fact", 0)),
    }
    obs_total = sum(observed.values())
    for k, v in predicted.items():
        obs_pct = observed[k] / (obs_total + 1e-10)
        pred_pct = v / (sum(predicted.values()) + 1e-10)
        print(f"    {k}: predicted={pred_pct:.1%}, observed={obs_pct:.1%}")

    # Save
    output = {
        "model": MODEL_NAME,
        "seeds": len(P3_SEEDS),
        "generations": GENERATIONS,
        "results": all_results,
        "analysis": analysis,
    }
    out_path = "/root/constraint_residual/p3_results_7b.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=float)
    print(f"\nSaved: {out_path}")

    # Verdict
    print("\n" + "=" * 60)
    if analysis["r_squared"] > 0.7 and analysis["lambda_c"] > 0.1:
        print("VERDICT: Constraint Attractor Collapse CONFIRMED at neural level")
    elif analysis["r_squared"] > 0.4:
        print("VERDICT: Weak evidence for attractor collapse. Increase seeds/generations.")
    else:
        print("VERDICT: Insufficient evidence. Check model/dataset fit.")
    print("=" * 60)

    return output


def _compute_pi_components(grads):
    if not grads:
        return {}
    components = ["sigma_fact", "sigma_syntax", "sigma_style", "sigma_safety", "sigma_coherence"]
    sums = {c: 0.0 for c in components}
    for g in grads:
        for i, c in enumerate(components):
            sums[c] += abs(g[i])
    n = len(grads)
    return {c: s / n for c, s in sums.items()}


def _analyze_collapse(all_results):
    gens = []
    cdivs = []
    for cap, results in all_results.items():
        for r in results:
            if r["n_grads"] > 0:
                gens.append(r["generation"])
                cdivs.append(r["residual_cv"])

    gens = np.array(gens)
    cdivs = np.array(cdivs)

    unique_gens = sorted(set(gens))
    mean_cdivs = []
    for g in unique_gens:
        mask = gens == g
        mean_cdivs.append(cdivs[mask].mean())

    unique_gens = np.array(unique_gens)
    mean_cdivs = np.array(mean_cdivs)

    valid = mean_cdivs > 1e-10
    if valid.sum() >= 2:
        x = unique_gens[valid]
        y = np.log(mean_cdivs[valid])
        slope, intercept = np.polyfit(x, y, 1)
        lambda_c = -slope
        y_pred = intercept + slope * x
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r_squared = 1 - ss_res / (ss_tot + 1e-10)
    else:
        lambda_c = 0.0
        r_squared = 0.0

    all_pi_means = []
    for cap, results in all_results.items():
        for r in results:
            if r["n_grads"] > 0:
                all_pi_means.append(r["residual_mean"])
    mean_pi_cv = float(np.std(all_pi_means) / (np.mean(all_pi_means) + 1e-10)) if all_pi_means else 0.0

    cdiv0 = mean_cdivs[0] if len(mean_cdivs) > 0 else 1.0
    cdiv_last = mean_cdivs[-1] if len(mean_cdivs) > 0 else 1.0
    cdiv_ratio = cdiv_last / (cdiv0 + 1e-10)

    all_pi = {}
    for cap, results in all_results.items():
        for r in results:
            for comp, val in r.get("pi_contributions", {}).items():
                all_pi[comp] = all_pi.get(comp, 0.0) + val

    return {
        "lambda_c": float(lambda_c),
        "r_squared": float(r_squared),
        "mean_pi_cv": mean_pi_cv,
        "cdiv_ratio": float(cdiv_ratio),
        "pi_components": all_pi,
    }


if __name__ == "__main__":
    run_p3_test()
