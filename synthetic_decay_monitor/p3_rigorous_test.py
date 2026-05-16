"""P3 rigorous test: component-wise π validation via hidden state extraction.

Validates the Constraint Attractor Collapse hypothesis at the neural level:
- C_div = std(||Π||) decays exponentially across recursive generations
- Mean ||Π|| stays stable (core constraint structure preserved)
- Individual π components match predicted executor weight distribution

Requires GPU with ~4GB VRAM (Qwen2.5-1.5B).
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
    compute_windowed_gradients,
    compute_windowed_pi_components,
)

P3_SEEDS = [
    ("math_reasoning", "Prove that the square root of 2 is irrational using proof by contradiction."),
    ("code_generation", "Write a Python function that implements quicksort."),
    ("factual_knowledge", "Explain how photosynthesis works in plants."),
    ("logical_consistency", "If all A are B, and some B are C, what can we conclude about A and C?"),
    ("creative_writing", "Write a short story about stars going out one by one."),
    ("general", "Explain the importance of biodiversity for ecosystems."),
]

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
GENERATIONS = 3
MAX_NEW_TOKENS = 128


def _generate_text(wrapper, prompt, max_tokens, temperature=0.8):
    """Generate text using the wrapper's model (SDPA mode)."""
    if hasattr(wrapper.tokenizer, 'apply_chat_template') and wrapper.tokenizer.chat_template:
        try:
            formatted = wrapper.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
            )
        except Exception:
            formatted = prompt
    else:
        formatted = prompt

    inputs = wrapper.tokenizer(formatted, return_tensors="pt", truncation=True, max_length=512).to(wrapper.device)
    with torch.no_grad():
        gen_ids = wrapper.model.generate(
            **inputs, max_new_tokens=max_tokens, do_sample=True, temperature=temperature,
            pad_token_id=wrapper.tokenizer.pad_token_id,
        )
    return wrapper.tokenizer.decode(gen_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()


def run_p3_test():
    print("=" * 60)
    print("P3 Rigorous Test: Constraint Attractor Collapse")
    print(f"Model: {MODEL_NAME}")
    print(f"Seeds: {len(P3_SEEDS)}, Generations: {GENERATIONS}")
    print("=" * 60)

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"\nDevice: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
    elif device == "mps":
        print(f"Apple MPS (Metal Performance Shaders) — unified memory")

    print("\n[1/4] Loading extraction model...")
    import transformers
    # MPS eager attention returns NaN for attention weights (PyTorch MPS bug).
    # Use CPU for extraction to get valid attention tensors. 1.5B model is small enough.
    extract_device = "cpu" if device == "mps" else device
    if device == "mps":
        print("  (MPS→CPU for extraction: MPS/eager produces NaN attention)")
    wrapper = ModelWrapper(model_name=MODEL_NAME, device=extract_device)
    wrapper.model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map=extract_device,
        trust_remote_code=True, attn_implementation="eager",
    )
    wrapper.model.eval()
    bank = ConstraintFunctionBank()

    print("\n[2/4] Running recursive generation + hidden state extraction...")
    print("  (Extraction: eager mode | Generation: SDPA mode)")

    # Load SDPA model for fast generation
    wrapper_sdpa = ModelWrapper(model_name=MODEL_NAME, device=device)
    wrapper_sdpa.model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map=device,
        trust_remote_code=True, attn_implementation="sdpa",
    )
    wrapper_sdpa.model.eval()

    all_results = {}  # capability -> generation -> {residuals, cancels, totals, constr_states}

    for cap, seed in P3_SEEDS:
        print(f"\n  {cap}:")
        cap_results = []
        prompt = seed

        for gen in range(GENERATIONS + 1):
            if gen == 0:
                # Extract hidden states from seed prompt (eager)
                state = wrapper.extract_output_state(prompt)
            else:
                # Step 1: Generate text with SDPA (fast, diverse sampling)
                gen_text = _generate_text(wrapper_sdpa, prompt, MAX_NEW_TOKENS, temperature=0.8)
                # Step 2: Extract hidden states from generated text (eager)
                state = wrapper.extract_output_state(gen_text)
                state.generated_text = gen_text
                prompt = gen_text

            # Compute constraint states per token
            attn = state.attention_weights  # [heads, seq, seq]
            constr_states = bank.compute_all(
                state.hidden_states.float(),
                [h.float() for h in state.layer_hidden_states],
                attn.float() if attn is not None else None,
            )

            # Compute gradients and residuals (token-level)
            grads = compute_constraint_gradients(constr_states)
            res_mags, cancels, totals = compute_residual(grads) if grads else ([], [], [])

            # Window-level decomposition (captures larger-scale variation)
            win_grads = compute_windowed_gradients(constr_states, window_size=8)
            win_mags, _, _ = compute_residual(win_grads) if win_grads else ([], [], [])
            win_pi = compute_windowed_pi_components(constr_states, window_size=8)

            cap_results.append({
                "generation": gen,
                "n_tokens": len(constr_states),
                "n_grads": len(grads),
                "n_win_grads": len(win_grads),
                "residual_mean": float(np.mean(res_mags)) if res_mags else 0.0,
                "residual_std": float(np.std(res_mags)) if res_mags else 0.0,
                "residual_cv": float(np.std(res_mags) / (np.mean(res_mags) + 1e-10)) if res_mags else 0.0,
                "cancel_mean": float(np.mean(cancels)) if cancels else 0.0,
                "total_constraint_mean": float(np.mean(totals)) if totals else 0.0,
                # Component-wise π contributions
                "pi_contributions": _compute_pi_components(grads),
                "pi_contributions_windowed": win_pi,
            })

            status = "healthy" if cap_results[-1]["residual_cv"] < 0.5 else \
                     "degrading" if cap_results[-1]["residual_cv"] < 1.0 else "collapsed"
            print(f"    Gen{gen}: {len(constr_states)} tokens, "
                  f"||Π||={cap_results[-1]['residual_mean']:.4f}±{cap_results[-1]['residual_std']:.4f}, "
                  f"C_div={cap_results[-1]['residual_cv']:.3f} [{status}]")

        all_results[cap] = cap_results

    # ── Analysis ──
    print("\n[3/4] Computing attractor collapse metrics...")
    analysis = _analyze_collapse(all_results)

    print(f"\n  Global C_div trajectory:")
    for gen in range(GENERATIONS + 1):
        cdivs = []
        for cap in all_results:
            if gen < len(all_results[cap]):
                cdivs.append(all_results[cap][gen]["residual_cv"])
        mean_cdiv = np.mean(cdivs) if cdivs else 0
        print(f"    Gen{gen}: C_div = {mean_cdiv:.4f}")

    print(f"\n  Collapse rate λ_C: {analysis['lambda_c']:.4f}/gen")
    print(f"  R² of exponential fit: {analysis['r_squared']:.4f}")
    print(f"  Mean ||Π|| stability: {analysis['mean_pi_cv']:.4f} (CV across gens)")
    print(f"  C_div at Gen{GENERATIONS}/Gen0 ratio: {analysis['cdiv_ratio']:.4f}")

    # ── Per-generation π component decomposition ──
    print("\n[4/4] Per-generation π component decomposition:")
    components = ["sigma_fact", "sigma_syntax", "sigma_style", "sigma_safety", "sigma_coherence"]
    exec_map = {"sigma_fact": "E-III", "sigma_syntax": "E-I", "sigma_style": "E-II",
                "sigma_safety": "E-I", "sigma_coherence": "E-I"}

    for gen in range(GENERATIONS + 1):
        gen_comps = {c: 0.0 for c in components}
        gen_comps_win = {c: 0.0 for c in components}
        n_contrib = 0
        for cap in all_results:
            if gen < len(all_results[cap]) and all_results[cap][gen]["n_grads"] > 0:
                for c in components:
                    gen_comps[c] += all_results[cap][gen]["pi_contributions"].get(c, 0)
                    gen_comps_win[c] += all_results[cap][gen].get("pi_contributions_windowed", {}).get(c, 0)
                n_contrib += 1

        if n_contrib > 0:
            total = sum(abs(v) for v in gen_comps.values())
            ei_total = abs(gen_comps["sigma_syntax"]) + abs(gen_comps["sigma_safety"]) + abs(gen_comps["sigma_coherence"])
            eii_total = abs(gen_comps["sigma_style"])
            eiii_total = abs(gen_comps["sigma_fact"])
            exec_total = ei_total + eii_total + eiii_total

            total_win = sum(abs(v) for v in gen_comps_win.values())
            ei_win = abs(gen_comps_win["sigma_syntax"]) + abs(gen_comps_win["sigma_safety"]) + abs(gen_comps_win["sigma_coherence"])
            eii_win = abs(gen_comps_win["sigma_style"])
            eiii_win = abs(gen_comps_win["sigma_fact"])
            exec_win_total = ei_win + eii_win + eiii_win

            print(f"\n  Gen{gen} (n={n_contrib} capabilities):")
            print(f"    Token-level:  E-I={ei_total/exec_total*100:.0f}%  E-II={eii_total/exec_total*100:.0f}%  E-III={eiii_total/exec_total*100:.0f}%")
            print(f"    Window-level: E-I={ei_win/exec_win_total*100:.0f}%  E-II={eii_win/exec_win_total*100:.0f}%  E-III={eiii_win/exec_win_total*100:.0f}%")
            for c in components:
                pct_tok = abs(gen_comps[c]) / (total + 1e-10) * 100
                pct_win = abs(gen_comps_win[c]) / (total_win + 1e-10) * 100
                bar_tok = '█' * int(pct_tok / 5)
                bar_win = '▓' * int(pct_win / 5)
                print(f"    {c}: tok {bar_tok} {pct_tok:.0f}% | win {bar_win} {pct_win:.0f}%")

    # ── Predicted vs observed (global) ──
    print(f"\n  Predicted vs observed executor weights (global avg):")
    pi_comp = analysis["pi_components"]
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
        match = "✓" if abs(obs_pct - pred_pct) < 0.15 else "△" if abs(obs_pct - pred_pct) < 0.30 else "✗"
        print(f"    {k}: predicted={pred_pct:.1%}, observed={obs_pct:.1%} [{match}]")

    # ── Save ──
    output = {
        "model": MODEL_NAME,
        "seeds": len(P3_SEEDS),
        "generations": GENERATIONS,
        "results": all_results,
        "analysis": analysis,
    }
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiment_data", "p3_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=float)
    print(f"\nSaved: {out_path}")

    # ── Verdict ──
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
    """Average contribution of each σ dimension to ||Π||."""
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
    """Fit exponential decay model C_div(n) = C_div(0) * exp(-λ_C * n)."""
    gens = []
    cdivs = []
    for cap, results in all_results.items():
        for r in results:
            if r["n_grads"] > 0:
                gens.append(r["generation"])
                cdivs.append(r["residual_cv"])

    gens = np.array(gens)
    cdivs = np.array(cdivs)

    # Per-generation mean C_div
    unique_gens = sorted(set(gens))
    mean_cdivs = []
    for g in unique_gens:
        mask = gens == g
        mean_cdivs.append(cdivs[mask].mean())

    unique_gens = np.array(unique_gens)
    mean_cdivs = np.array(mean_cdivs)

    # Exponential fit: C_div = A * exp(-λ * n)
    # log(C_div) = log(A) - λ * n
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

    # Mean ||Π|| stability
    all_pi_means = []
    for cap, results in all_results.items():
        for r in results:
            if r["n_grads"] > 0:
                all_pi_means.append(r["residual_mean"])
    mean_pi_cv = float(np.std(all_pi_means) / (np.mean(all_pi_means) + 1e-10)) if all_pi_means else 0.0

    # C_div ratio
    cdiv0 = mean_cdivs[0] if len(mean_cdivs) > 0 else 1.0
    cdiv_last = mean_cdivs[-1] if len(mean_cdivs) > 0 else 1.0
    cdiv_ratio = cdiv_last / (cdiv0 + 1e-10)

    # Aggregate π components across all tokens and generations
    all_pi = {}
    # Per-generation aggregate
    per_gen_pi = {g: {} for g in range(10)}  # gen → component → sum
    for cap, results in all_results.items():
        for r in results:
            gen = r.get("generation", 0)
            for comp, val in r.get("pi_contributions", {}).items():
                all_pi[comp] = all_pi.get(comp, 0.0) + val
                per_gen_pi.setdefault(gen, {})[comp] = per_gen_pi[gen].get(comp, 0.0) + val

    return {
        "lambda_c": float(lambda_c),
        "r_squared": float(r_squared),
        "mean_pi_cv": mean_pi_cv,
        "cdiv_ratio": float(cdiv_ratio),
        "pi_components": all_pi,
        "pi_components_per_gen": {str(g): comps for g, comps in per_gen_pi.items() if comps},
    }


if __name__ == "__main__":
    run_p3_test()
