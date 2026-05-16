"""
基线对比 —— 约束残差法 vs 不确定性基线 vs SelfCheckGPT

在 false-premise 问题对上做头对头对比。
用法：
    cd /Users/dengxinhang/paper
    source venv_hallu/bin/activate
    python -m constraint_residual.hallucination_predictor.compare_baselines
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass

from constraint_residual.hallucination_predictor.model_wrapper import ModelWrapper
from constraint_residual.hallucination_predictor.constraint_functions import (
    ConstraintFunctionBank,
    compute_constraint_gradients,
    compute_residual,
)
from constraint_residual.hallucination_predictor.run_poc import (
    _builtin_false_premise_pairs,
    _builtin_truthfulqa,
    self_judge,
    calibrate_truth_direction,
    calibrate_refusal_direction,
)


@dataclass
class MethodResult:
    name: str
    auc: float
    precision: float
    recall: float
    f1: float
    latency_ms: float
    extra_cost: str


def compute_constraint_residual_score(wrapper, bank, question_text, temperature=0.6):
    """约束残差法：Δ||Π||"""
    state = wrapper.generate_and_extract(
        prompt=question_text,
        max_new_tokens=64,
        temperature=temperature,
        do_sample=(temperature > 0),
    )

    cstates = bank.compute_all(
        state.hidden_states,
        state.layer_hidden_states,
        state.attention_weights,
    )
    gradients = compute_constraint_gradients(cstates)
    residuals, _, _ = compute_residual(gradients)

    # Filter special tokens
    special_prefixes = ('<|im_', 'system', 'user', 'assistant', '\n', '<')
    content_indices = [
        t for t in range(min(len(state.tokens), len(residuals)))
        if not any(state.tokens[t].startswith(p) for p in special_prefixes)
    ]
    filtered = [residuals[t] for t in content_indices if t < len(residuals)]
    input_mean = np.mean(filtered) if filtered else 0.0

    # Output hidden states
    response = state.generated_text
    output_mean = input_mean
    if response and len(response.strip()) > 5:
        try:
            out_state = wrapper.extract_output_state(response)
            out_cstates = bank.compute_all(
                out_state.hidden_states,
                out_state.layer_hidden_states,
                out_state.attention_weights,
            )
            out_grads = compute_constraint_gradients(out_cstates)
            out_res, _, _ = compute_residual(out_grads)
            out_filtered = [r for r in out_res if r > 1e-6]
            output_mean = np.mean(out_filtered) if out_filtered else input_mean
        except Exception:
            pass

    return output_mean - input_mean, state.inference_time_ms


def compute_attention_entropy_score(wrapper, question_text, temperature=0.6):
    """注意力熵基线：平均注意力熵（低熵=高置信度，高熵=不确定→可能幻觉）"""
    state = wrapper.generate_and_extract(
        prompt=question_text,
        max_new_tokens=64,
        temperature=temperature,
        do_sample=(temperature > 0),
    )

    if state.attention_weights is not None:
        attn = state.attention_weights  # [heads, seq, seq]
        avg_attn = attn.mean(dim=0)     # [seq, seq]
        entropies = []
        for i in range(avg_attn.shape[0]):
            row = avg_attn[i, :i+1]
            if row.sum() < 1e-10:
                continue
            row = row / row.sum()
            ent = -torch.sum(row * torch.log(row + 1e-10)).item()
            entropies.append(ent)
        score = np.mean(entropies) if entropies else 0.0
    else:
        score = 0.0

    return score, state.inference_time_ms


def compute_selfcheck_score(wrapper, question_text, n_samples=3):
    """SelfCheckGPT 基线：多次采样的回答不一致性"""
    responses = []
    total_time = 0.0
    for _ in range(n_samples):
        state = wrapper.generate_and_extract(
            prompt=question_text,
            max_new_tokens=64,
            temperature=0.8,
            do_sample=True,
        )
        responses.append(state.generated_text)
        total_time += state.inference_time_ms

    if len(responses) < 2:
        return 0.0, total_time

    # Jaccard distance between all pairs
    def tokenize(text):
        return set(text.lower().split())

    distances = []
    for i in range(len(responses)):
        ti = tokenize(responses[i])
        for j in range(i + 1, len(responses)):
            tj = tokenize(responses[j])
            union = len(ti | tj)
            intersection = len(ti & tj)
            jaccard = intersection / union if union > 0 else 0.0
            distances.append(1.0 - jaccard)

    inconsistency = np.mean(distances) if distances else 0.0
    return inconsistency, total_time


def compute_response_length_score(wrapper, question_text, temperature=0.6):
    """简单基线：回答越长越可能包含幻觉"""
    state = wrapper.generate_and_extract(
        prompt=question_text,
        max_new_tokens=64,
        temperature=temperature,
        do_sample=(temperature > 0),
    )
    return len(state.generated_text.split()), state.inference_time_ms


def compute_predictive_entropy(wrapper, question_text, temperature=0.6):
    """预测熵基线：生成 token 的平均预测熵（高熵=高不确定性→可能幻觉）"""
    state = wrapper.generate_and_extract(
        prompt=question_text,
        max_new_tokens=64,
        temperature=temperature,
        do_sample=(temperature > 0),
    )
    prefix_ids = state.input_ids.to(wrapper.device)
    gen_text = state.generated_text
    if not gen_text or len(gen_text.strip()) < 3:
        return 0.0, state.inference_time_ms

    full_ids = wrapper.tokenizer(gen_text, return_tensors="pt").input_ids[0].to(wrapper.device)
    full_seq = torch.cat([prefix_ids, full_ids])

    with torch.no_grad():
        outputs = wrapper.model(full_seq.unsqueeze(0))
        logits = outputs.logits[0]  # [seq, vocab]

    # Entropy over generated positions
    gen_start = len(prefix_ids)
    entropies = []
    for i in range(gen_start, min(len(logits) - 1, gen_start + len(full_ids))):
        probs = F.softmax(logits[i].float(), dim=-1)
        ent = -torch.sum(probs * torch.log(probs + 1e-12)).item()
        entropies.append(ent)

    return np.mean(entropies) if entropies else 0.0, state.inference_time_ms


def compute_max_probability(wrapper, question_text, temperature=0.6):
    """最大概率基线：生成 token 的平均最大概率（低置信度→可能幻觉）"""
    state = wrapper.generate_and_extract(
        prompt=question_text,
        max_new_tokens=64,
        temperature=temperature,
        do_sample=(temperature > 0),
    )
    prefix_ids = state.input_ids.to(wrapper.device)
    gen_text = state.generated_text
    if not gen_text or len(gen_text.strip()) < 3:
        return 1.0, state.inference_time_ms

    full_ids = wrapper.tokenizer(gen_text, return_tensors="pt").input_ids[0].to(wrapper.device)
    full_seq = torch.cat([prefix_ids, full_ids])

    with torch.no_grad():
        outputs = wrapper.model(full_seq.unsqueeze(0))
        logits = outputs.logits[0]

    gen_start = len(prefix_ids)
    max_probs = []
    for i in range(gen_start, min(len(logits) - 1, gen_start + len(full_ids))):
        probs = F.softmax(logits[i].float(), dim=-1)
        max_probs.append(probs.max().item())

    return 1.0 - np.mean(max_probs) if max_probs else 0.0, state.inference_time_ms


def compute_layer_variance(wrapper, question_text, temperature=0.6):
    """层方差基线：各层隐藏状态的方差（高方差=内部不一致→可能幻觉）"""
    state = wrapper.generate_and_extract(
        prompt=question_text,
        max_new_tokens=64,
        temperature=temperature,
        do_sample=(temperature > 0),
    )
    if not state.layer_hidden_states:
        return 0.0, state.inference_time_ms

    # Stack [n_layers, seq, dim] → variance across layers
    stacked = torch.stack([h.float() for h in state.layer_hidden_states])  # [L, S, D]
    layer_var = stacked.var(dim=0).mean().item()  # mean over seq and dim
    return layer_var, state.inference_time_ms


def evaluate_method(scores, labels, name, latency_ms, extra_cost):
    """评估一个方法的分类性能"""
    from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

    if len(set(labels)) < 2:
        return MethodResult(name=name, auc=0.5, precision=0, recall=0, f1=0,
                          latency_ms=latency_ms, extra_cost=extra_cost)

    try:
        auc = roc_auc_score(labels, scores)
    except Exception:
        auc = 0.5

    threshold = np.median(scores)
    preds = [1 if s > threshold else 0 for s in scores]

    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)

    return MethodResult(name=name, auc=auc, precision=precision, recall=recall,
                       f1=f1, latency_ms=latency_ms, extra_cost=extra_cost)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    print("=" * 70)
    print("BASELINE COMPARISON: 约束残差法 vs 不确定性 vs SelfCheckGPT")
    print("=" * 70)

    # 1. Load model
    wrapper = ModelWrapper(model_name=args.model, device=args.device)

    # 2. Calibrate
    bank = ConstraintFunctionBank()
    calibrate_truth_direction(wrapper, bank)
    calibrate_refusal_direction(wrapper, bank)

    # 3. Load questions — TruthfulQA (POC-style misconception questions)
    tqa = _builtin_truthfulqa()[:20]
    fp_pairs = _builtin_false_premise_pairs()[:5]  # FP version only
    print(f"\n[Data] {len(tqa)} TruthfulQA + {len(fp_pairs)} false-premise FP ({len(tqa) + len(fp_pairs)} total)\n")

    # Build flat question list
    all_questions = []
    for q in tqa:
        all_questions.append({
            "question": q["question"],
            "best_answer": q["best_answer"],
            "incorrect_answers": q.get("incorrect_answers", []),
        })
    for p in fp_pairs:
        all_questions.append({
            "question": p["question"],  # FP version only — the hallucination inducer
            "best_answer": p["best_answer"],
            "incorrect_answers": p.get("incorrect_answers", []),
        })

    # 4. Collect scores from all methods
    cr_scores, cr_times = [], []
    ae_scores, ae_times = [], []
    sc_scores, sc_times = [], []
    rl_scores, rl_times = [], []
    pe_scores, pe_times = [], []
    mp_scores, mp_times = [], []
    lv_scores, lv_times = [], []
    labels = []

    print(f"Evaluating {len(all_questions)} questions with 7 methods...\n")

    n_eval = min(25, len(all_questions))
    for i, q in enumerate(all_questions[:n_eval]):
        q_text = q["question"]
        reference = q["best_answer"]

        print(f"[{i+1}/{n_eval}] {q_text[:80]}...")

        # Constraint residual (Ours)
        score, t = compute_constraint_residual_score(wrapper, bank, q_text)
        cr_scores.append(score)
        cr_times.append(t)

        # Attention entropy
        score2, t2 = compute_attention_entropy_score(wrapper, q_text)
        ae_scores.append(score2)
        ae_times.append(t2)

        # Response length
        score4, t4 = compute_response_length_score(wrapper, q_text)
        rl_scores.append(score4)
        rl_times.append(t4)

        # Label: judge with self_judge
        state = wrapper.generate_and_extract(
            prompt=q_text, max_new_tokens=64, temperature=0.6, do_sample=True,
        )
        is_hallu = self_judge(wrapper, q_text, state.generated_text, reference)
        labels.append(1 if is_hallu else 0)

        # Predictive entropy
        pe_score, pe_t = compute_predictive_entropy(wrapper, q_text)
        pe_scores.append(pe_score)
        pe_times.append(pe_t)

        # Max probability (1 - confidence)
        mp_score, mp_t = compute_max_probability(wrapper, q_text)
        mp_scores.append(mp_score)
        mp_times.append(mp_t)

        # Layer variance
        lv_score, lv_t = compute_layer_variance(wrapper, q_text)
        lv_scores.append(lv_score)
        lv_times.append(lv_t)

        print(f"  CR={score:+.4f}  PredE={pe_score:.4f}  MaxP={mp_score:.4f}  "
              f"LayerV={lv_score:.4f}  Label={'HALLU' if is_hallu else 'OK'}")

    # SelfCheckGPT (expensive — only 8 questions)
    print("\n[SelfCheckGPT] 3x sampling on first 8 questions...")
    for i in range(min(8, n_eval)):
        q_text = all_questions[i]["question"]
        score, t = compute_selfcheck_score(wrapper, q_text, n_samples=3)
        sc_scores.append(score)
        sc_times.append(t)
        print(f"  [{i+1}/8] SC={score:.4f} ({t:.0f}ms)")

    # 5. Evaluate each method
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    results = {}
    results["constraint_residual"] = evaluate_method(
        cr_scores, labels[:len(cr_scores)],
        "Constraint Residual (Ours)", np.mean(cr_times), "1x inference"
    )
    results["predictive_entropy"] = evaluate_method(
        pe_scores, labels[:len(pe_scores)],
        "Predictive Entropy", np.mean(pe_times), "1x inference + logits"
    )
    results["max_probability"] = evaluate_method(
        mp_scores, labels[:len(mp_scores)],
        "Max Probability (1-conf)", np.mean(mp_times), "1x inference + logits"
    )
    results["attention_entropy"] = evaluate_method(
        ae_scores, labels[:len(ae_scores)],
        "Attention Entropy", np.mean(ae_times), "1x inference"
    )
    results["layer_variance"] = evaluate_method(
        lv_scores, labels[:len(lv_scores)],
        "Layer Variance", np.mean(lv_times), "1x inference"
    )
    results["response_length"] = evaluate_method(
        rl_scores, labels[:len(rl_scores)],
        "Response Length", np.mean(rl_times), "1x inference"
    )

    sc_labels = labels[:len(sc_scores)]
    results["selfcheck"] = evaluate_method(
        sc_scores, sc_labels,
        "SelfCheckGPT (3x)", np.mean(sc_times) if sc_times else 0, "3x inference"
    ) if len(sc_scores) >= 2 else None

    # 6. Print comparison table
    print(f"\n{'Method':<35s} {'AUC':>7s} {'Prec':>7s} {'Rec':>7s} {'F1':>7s} {'Latency':>9s} {'Cost':>12s}")
    print("-" * 90)
    for key, r in results.items():
        if r is None:
            continue
        print(f"{r.name:<35s} {r.auc:>7.3f} {r.precision:>7.3f} {r.recall:>7.3f} "
              f"{r.f1:>7.3f} {r.latency_ms:>7.0f}ms {r.extra_cost:>12s}")

    # 7. Save
    output = {
        "methods": {k: {
            "name": r.name, "auc": r.auc, "precision": r.precision,
            "recall": r.recall, "f1": r.f1, "latency_ms": r.latency_ms,
            "extra_cost": r.extra_cost,
        } for k, r in results.items() if r},
    }
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    with open(output_dir / "baseline_comparison.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to {output_dir / 'baseline_comparison.json'}")


if __name__ == "__main__":
    main()
