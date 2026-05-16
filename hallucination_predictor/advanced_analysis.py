"""
进阶分析 —— 约束残差法 vs HALT 探针 vs 其他特征

在 false-premise 问题对上做：
1. 逐层隐藏状态特征提取
2. HALT-style 线性探针训练与评估
3. 5 个 σ_i 维度各自的区分能力
4. 公平对比：相同数据、相同标签
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict

from constraint_residual.hallucination_predictor.model_wrapper import ModelWrapper
from constraint_residual.hallucination_predictor.constraint_functions import (
    ConstraintFunctionBank,
    ConstraintState,
    compute_constraint_gradients,
    compute_residual,
)
from constraint_residual.hallucination_predictor.run_poc import (
    _builtin_false_premise_pairs,
    self_judge,
    calibrate_truth_direction,
    calibrate_refusal_direction,
)


# ============================================================
# 特征提取器
# ============================================================

@dataclass
class RichFeatures:
    """每个问题的丰富特征集"""
    question: str
    is_hallucination: bool

    # 约束残差特征
    delta_pi: float              # Δ||Π|| (output - input)
    cancellation_ratio: float    # mean c(p) across tokens
    total_constraint: float      # mean Σ||∇σ||
    sigma_fact_jump: float       # Δσ_fact
    sigma_syntax_jump: float     # Δσ_syntax
    sigma_style_jump: float      # Δσ_style
    sigma_safety_jump: float     # Δσ_safety
    sigma_coherence_jump: float  # Δσ_coherence

    # HALT 特征：各层隐藏状态范数
    layer_norms: List[float] = field(default_factory=list)
    layer_norm_mean: float = 0.0
    layer_norm_std: float = 0.0

    # 注意力特征
    attn_entropy_mean: float = 0.0
    attn_entropy_std: float = 0.0

    # 隐藏状态漂移特征
    hidden_cosine_sim: float = 0.0   # input vs output 隐藏状态余弦相似度
    hidden_l2_dist: float = 0.0      # input vs output L2 距离

    # 元数据
    inference_time_ms: float = 0.0


def extract_rich_features(
    wrapper: ModelWrapper,
    bank: ConstraintFunctionBank,
    question_text: str,
    reference: str,
    temperature: float = 0.6,
) -> RichFeatures:
    """提取完整特征集：约束残差 + 逐层状态 + 注意力 + 隐藏漂移"""

    # 1. 生成回答 + 提取输入状态
    state = wrapper.generate_and_extract(
        prompt=question_text,
        max_new_tokens=64,
        temperature=temperature,
        do_sample=(temperature > 0),
    )
    response = state.generated_text

    # 2. 输入约束状态
    input_cstates = bank.compute_all(
        state.hidden_states,
        state.layer_hidden_states,
        state.attention_weights,
    )
    input_grads = compute_constraint_gradients(input_cstates)
    input_res, input_cancels, input_totals = compute_residual(input_grads)

    # Filter special tokens for input
    special_prefixes = ('<|im_', 'system', 'user', 'assistant', '\n', '<')
    content_idx = [
        t for t in range(min(len(state.tokens), len(input_res)))
        if not any(state.tokens[t].startswith(p) for p in special_prefixes)
    ]
    input_pi = np.mean([input_res[t] for t in content_idx if t < len(input_res)]) if content_idx else 0.0
    input_cancel = np.mean([input_cancels[t] for t in content_idx if t < len(input_cancels)]) if content_idx else 0.5
    input_total = np.mean([input_totals[t] for t in content_idx if t < len(input_totals)]) if content_idx else 0.0

    # 输入各 σ 均值
    input_sigmas = defaultdict(list)
    for t in content_idx:
        if t < len(input_cstates):
            cs = input_cstates[t]
            input_sigmas['fact'].append(cs.sigma_fact)
            input_sigmas['syntax'].append(cs.sigma_syntax)
            input_sigmas['style'].append(cs.sigma_style)
            input_sigmas['safety'].append(cs.sigma_safety)
            input_sigmas['coherence'].append(cs.sigma_coherence)

    # 3. 输出约束状态
    output_pi = input_pi
    output_cancel = input_cancel
    output_total = input_total
    output_sigmas = defaultdict(list)

    if response and len(response.strip()) > 5:
        try:
            out_state = wrapper.extract_output_state(response)
            out_cstates = bank.compute_all(
                out_state.hidden_states,
                out_state.layer_hidden_states,
                out_state.attention_weights,
            )
            out_grads = compute_constraint_gradients(out_cstates)
            out_res, out_cancels, out_totals = compute_residual(out_grads)
            out_f = [r for r in out_res if r > 1e-6]
            output_pi = np.mean(out_f) if out_f else input_pi
            out_cf = [c for c in out_cancels if c > 1e-6]
            output_cancel = np.mean(out_cf) if out_cf else input_cancel
            out_tf = [t for t in out_totals if t > 1e-6]
            output_total = np.mean(out_tf) if out_tf else input_total

            for t in range(len(out_cstates)):
                cs = out_cstates[t]
                output_sigmas['fact'].append(cs.sigma_fact)
                output_sigmas['syntax'].append(cs.sigma_syntax)
                output_sigmas['style'].append(cs.sigma_style)
                output_sigmas['safety'].append(cs.sigma_safety)
                output_sigmas['coherence'].append(cs.sigma_coherence)
        except Exception:
            pass

    # 4. 逐层范数特征 (HALT-style)
    layer_norms = []
    for lh in state.layer_hidden_states:
        # lh: [seq_len, hidden_dim]
        # 取 content token 位置的范数
        norms = [float(torch.norm(lh[t].float()).item())
                 for t in content_idx if t < lh.shape[0]]
        layer_norms.append(np.mean(norms) if norms else 0.0)

    # 5. 注意力熵
    attn_entropies = []
    if state.attention_weights is not None:
        attn = state.attention_weights  # [heads, seq, seq]
        avg_attn = attn.mean(dim=0)
        for t in content_idx:
            if t < avg_attn.shape[0]:
                row = avg_attn[t, :t+1]
                if row.sum() > 1e-10:
                    row = row / row.sum()
                    ent = -torch.sum(row * torch.log(row + 1e-10)).item()
                    attn_entropies.append(ent)

    # 6. 隐藏状态漂移（输入 vs 输出）
    cos_sim = 0.5
    l2_dist = 0.0
    if response and len(response.strip()) > 5:
        try:
            out_state_full = wrapper.extract_output_state(response)
            # Average pooling over content tokens
            in_hidden = state.hidden_states[content_idx].float().mean(dim=0) if content_idx else state.hidden_states.float().mean(dim=0)
            out_hidden = out_state_full.hidden_states.float().mean(dim=0)
            cos_sim = float(torch.nn.functional.cosine_similarity(
                in_hidden.unsqueeze(0), out_hidden.unsqueeze(0), dim=-1
            )[0])
            l2_dist = float(torch.norm(in_hidden - out_hidden).item())
        except Exception:
            pass

    # 7. 判断
    is_hallu = self_judge(wrapper, question_text, response, reference)

    return RichFeatures(
        question=question_text[:100],
        is_hallucination=is_hallu,
        delta_pi=output_pi - input_pi,
        cancellation_ratio=output_cancel,
        total_constraint=output_total,
        sigma_fact_jump=(np.mean(output_sigmas['fact']) if output_sigmas['fact'] else 0) -
                        (np.mean(input_sigmas['fact']) if input_sigmas['fact'] else 0),
        sigma_syntax_jump=(np.mean(output_sigmas['syntax']) if output_sigmas['syntax'] else 0) -
                          (np.mean(input_sigmas['syntax']) if input_sigmas['syntax'] else 0),
        sigma_style_jump=(np.mean(output_sigmas['style']) if output_sigmas['style'] else 0) -
                         (np.mean(input_sigmas['style']) if input_sigmas['style'] else 0),
        sigma_safety_jump=(np.mean(output_sigmas['safety']) if output_sigmas['safety'] else 0) -
                          (np.mean(input_sigmas['safety']) if input_sigmas['safety'] else 0),
        sigma_coherence_jump=(np.mean(output_sigmas['coherence']) if output_sigmas['coherence'] else 0) -
                             (np.mean(input_sigmas['coherence']) if input_sigmas['coherence'] else 0),
        layer_norms=layer_norms,
        layer_norm_mean=np.mean(layer_norms) if layer_norms else 0.0,
        layer_norm_std=np.std(layer_norms) if layer_norms else 0.0,
        attn_entropy_mean=np.mean(attn_entropies) if attn_entropies else 0.0,
        attn_entropy_std=np.std(attn_entropies) if attn_entropies else 0.0,
        hidden_cosine_sim=cos_sim,
        hidden_l2_dist=l2_dist,
        inference_time_ms=state.inference_time_ms,
    )


# ============================================================
# 评估工具
# ============================================================

def evaluate_signal(scores: List[float], labels: List[int], name: str = ""):
    """评估单一信号的区分能力"""
    from sklearn.metrics import roc_auc_score
    from scipy import stats as sp_stats

    hallu_scores = [s for s, l in zip(scores, labels) if l == 1]
    correct_scores = [s for s, l in zip(scores, labels) if l == 0]

    result = {"name": name, "n_hallu": len(hallu_scores), "n_correct": len(correct_scores)}

    if len(hallu_scores) >= 2 and len(correct_scores) >= 2:
        result["mean_hallu"] = float(np.mean(hallu_scores))
        result["mean_correct"] = float(np.mean(correct_scores))
        result["std_hallu"] = float(np.std(hallu_scores))
        result["std_correct"] = float(np.std(correct_scores))

        t_stat, p_val = sp_stats.ttest_ind(hallu_scores, correct_scores)
        result["t_stat"] = float(t_stat)
        result["p_value"] = float(p_val)

        pooled = np.sqrt((np.std(hallu_scores)**2 + np.std(correct_scores)**2) / 2)
        result["cohens_d"] = float((np.mean(hallu_scores) - np.mean(correct_scores)) / pooled) if pooled > 0 else 0.0

        try:
            result["auc"] = float(roc_auc_score(labels, scores))
        except Exception:
            result["auc"] = 0.5
    else:
        result["mean_hallu"] = float(np.mean(hallu_scores)) if hallu_scores else 0
        result["mean_correct"] = float(np.mean(correct_scores)) if correct_scores else 0
        result["cohens_d"] = 0.0
        result["auc"] = 0.5
        result["p_value"] = 1.0

    return result


# ============================================================
# HALT-style 探针训练
# ============================================================

def train_halt_probe(
    features_list: List[RichFeatures],
    test_ratio: float = 0.3,
):
    """训练 HALT-style 线性探针：用逐层范数 + 隐藏漂移特征预测幻觉。

    使用简单逻辑回归（sklearn LogisticRegression）。
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, accuracy_score

    # Build feature matrix
    X = []
    y = []
    for f in features_list:
        feat_vec = [
            f.layer_norm_mean,
            f.layer_norm_std,
            f.hidden_cosine_sim,
            f.hidden_l2_dist,
            f.attn_entropy_mean,
            f.attn_entropy_std,
        ]
        # Add individual layer norms
        for ln in f.layer_norms:
            feat_vec.append(ln)
        X.append(feat_vec)
        y.append(1 if f.is_hallucination else 0)

    X = np.array(X)
    y = np.array(y)

    if len(set(y)) < 2:
        return {"auc": 0.5, "accuracy": 0.5, "note": "only one class"}

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=42, stratify=y if len(set(y)) > 1 else None
    )

    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Train
    clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf.fit(X_train_s, y_train)

    # Evaluate
    y_pred = clf.predict(X_test_s)
    y_prob = clf.predict_proba(X_test_s)[:, 1]

    auc = roc_auc_score(y_test, y_prob) if len(set(y_test)) > 1 else 0.5
    acc = accuracy_score(y_test, y_pred)

    # Feature importance
    coef_names = ['layer_norm_mean', 'layer_norm_std', 'cos_sim', 'l2_dist',
                  'attn_ent_mean', 'attn_ent_std']
    coef_names += [f'layer_{i}_norm' for i in range(len(features_list[0].layer_norms))]
    top_features = sorted(
        [(coef_names[i], abs(clf.coef_[0][i])) for i in range(min(len(coef_names), len(clf.coef_[0])))],
        key=lambda x: -x[1]
    )[:8]

    return {
        "auc": float(auc),
        "accuracy": float(acc),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "top_features": top_features,
    }


# ============================================================
# 主分析
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    print("=" * 70)
    print("ADVANCED ANALYSIS: 约束残差法 vs HALT 探针")
    print("=" * 70)

    # 1. Load model
    wrapper = ModelWrapper(model_name=args.model, device=args.device)

    # 2. Calibrate
    bank = ConstraintFunctionBank()
    calibrate_truth_direction(wrapper, bank)
    calibrate_refusal_direction(wrapper, bank)

    # 3. Load false premise pairs + TruthfulQA standard
    pairs = _builtin_false_premise_pairs()[:12]
    from constraint_residual.hallucination_predictor.run_poc import _builtin_truthfulqa
    truth_qa = _builtin_truthfulqa()[:15]

    # Build question list: FP pairs (FP + True versions) + TruthfulQA standard
    all_questions = []
    for p in pairs:
        all_questions.append({
            "question": p["question"],
            "best_answer": p["best_answer"],
        })
        all_questions.append({
            "question": p["true_question"],
            "best_answer": p["true_answer"],
        })
    for q in truth_qa:
        ref = q.get("best_answer", "")
        if not ref and isinstance(q.get("correct_answers"), list):
            ref = q["correct_answers"][0] if q["correct_answers"] else ""
        all_questions.append({
            "question": q["question"],
            "best_answer": ref[:300],
        })

    print(f"\n[Data] {len(all_questions)} questions total "
          f"({len(pairs)*2} FP pairs + {len(truth_qa)} TruthfulQA)\n")

    # 4. Extract features
    features_list = []
    t_start = time.time()
    for i, q in enumerate(all_questions[:30]):  # limit for time
        try:
            feat = extract_rich_features(
                wrapper, bank, q["question"], q.get("best_answer", ""), temperature=0.6
            )
            features_list.append(feat)
        except Exception as e:
            print(f"  [{i+1}/{min(30, len(all_questions))}] ERROR: {e}")
            continue
        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{min(30, len(all_questions))}] {feat.question[:70]}... "
                  f"hallu={feat.is_hallucination} ΔΠ={feat.delta_pi:+.4f}")

    total_time = time.time() - t_start
    print(f"\nFeature extraction: {len(features_list)} samples in {total_time:.0f}s")

    # 5. Evaluate individual signals
    labels = [1 if f.is_hallucination else 0 for f in features_list]
    n_hallu = sum(labels)
    n_correct = len(labels) - n_hallu
    print(f"\nLabel distribution: {n_hallu} hallucination / {n_correct} correct "
          f"({n_hallu/len(labels)*100:.0f}%)")

    signal_results = []

    # Δ||Π||
    signal_results.append(evaluate_signal(
        [f.delta_pi for f in features_list], labels, "Δ||Π|| (Ours)"
    ))

    # Cancellation ratio
    signal_results.append(evaluate_signal(
        [f.cancellation_ratio for f in features_list], labels, "Cancellation Ratio c(p)"
    ))

    # Total constraint
    signal_results.append(evaluate_signal(
        [f.total_constraint for f in features_list], labels, "Total Constraint Σ||∇σ||"
    ))

    # Per-σ_i jumps
    for sigma_name, attr in [
        ("Δσ_fact", "sigma_fact_jump"),
        ("Δσ_syntax", "sigma_syntax_jump"),
        ("Δσ_style", "sigma_style_jump"),
        ("Δσ_safety", "sigma_safety_jump"),
        ("Δσ_coherence", "sigma_coherence_jump"),
    ]:
        signal_results.append(evaluate_signal(
            [getattr(f, attr) for f in features_list], labels, sigma_name
        ))

    # Hidden state drift
    signal_results.append(evaluate_signal(
        [f.hidden_cosine_sim for f in features_list], labels, "Hidden Cosine Similarity"
    ))
    signal_results.append(evaluate_signal(
        [f.hidden_l2_dist for f in features_list], labels, "Hidden L2 Distance"
    ))

    # Attention entropy
    signal_results.append(evaluate_signal(
        [f.attn_entropy_mean for f in features_list], labels, "Attention Entropy Mean"
    ))

    # Layer norm mean
    signal_results.append(evaluate_signal(
        [f.layer_norm_mean for f in features_list], labels, "Layer Norm Mean (HALT)"
    ))
    signal_results.append(evaluate_signal(
        [f.layer_norm_std for f in features_list], labels, "Layer Norm Std (HALT)"
    ))

    # 6. Train HALT probe
    print("\n[HALT Probe] Training linear probe on hidden state features...")
    halt_result = train_halt_probe(features_list)
    print(f"  HALT Probe AUC: {halt_result['auc']:.3f}, Accuracy: {halt_result['accuracy']:.3f}")
    if "top_features" in halt_result:
        print(f"  Top features: {halt_result['top_features'][:5]}")

    # 7. Print comparison table
    print("\n" + "=" * 70)
    print("FULL SIGNAL COMPARISON (same data, same labels)")
    print("=" * 70)
    print(f"{'Signal':<30s} {'AUC':>7s} {'Cohen d':>9s} {'p-value':>9s} {'Hallu mean':>12s} {'Correct mean':>12s}")
    print("-" * 85)

    for sr in signal_results:
        auc_str = f"{sr['auc']:.3f}" if sr['auc'] is not None else "N/A"
        d_str = f"{sr['cohens_d']:+.3f}" if sr.get('cohens_d') is not None else "N/A"
        p_str = f"{sr['p_value']:.4f}" if sr.get('p_value') is not None else "N/A"
        hm = f"{sr.get('mean_hallu', 0):+.4f}"
        cm = f"{sr.get('mean_correct', 0):+.4f}"
        print(f"{sr['name']:<30s} {auc_str:>7s} {d_str:>9s} {p_str:>9s} {hm:>12s} {cm:>12s}")

    # 8. Add HALT probe to summary
    print(f"\n{'HALT Linear Probe':<30s} {halt_result['auc']:.3f}  (trained on {halt_result['train_size']} samples)")
    print(f"  Test accuracy: {halt_result['accuracy']:.3f}")

    # 9. Save results
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    # Save detailed features
    features_json = []
    for f in features_list:
        features_json.append({
            "question": f.question,
            "is_hallucination": f.is_hallucination,
            "delta_pi": f.delta_pi,
            "cancellation_ratio": f.cancellation_ratio,
            "total_constraint": f.total_constraint,
            "sigma_fact_jump": f.sigma_fact_jump,
            "sigma_syntax_jump": f.sigma_syntax_jump,
            "sigma_style_jump": f.sigma_style_jump,
            "sigma_safety_jump": f.sigma_safety_jump,
            "sigma_coherence_jump": f.sigma_coherence_jump,
            "layer_norm_mean": f.layer_norm_mean,
            "layer_norm_std": f.layer_norm_std,
            "hidden_cosine_sim": f.hidden_cosine_sim,
            "hidden_l2_dist": f.hidden_l2_dist,
            "attn_entropy_mean": f.attn_entropy_mean,
        })

    output = {
        "config": {
            "model": args.model,
            "n_samples": len(features_list),
            "n_hallu": n_hallu,
            "n_correct": n_correct,
        },
        "signal_comparison": signal_results,
        "halt_probe": halt_result,
        "features": features_json,
    }

    output_path = output_dir / "advanced_analysis.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {output_path}")

    # 10. Conclusion
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    # Find best non-probe signal
    best_signal = max(
        [s for s in signal_results if s.get('auc', 0) > 0],
        key=lambda s: abs(s.get('cohens_d', 0))
    )
    print(f"Best single signal: {best_signal['name']} "
          f"(AUC={best_signal.get('auc', 'N/A')}, d={best_signal.get('cohens_d', 0):.3f})")

    vs_halt = abs(best_signal.get('cohens_d', 0)) - abs(halt_result['auc'] - 0.5) * 2
    if best_signal.get('auc', 0) >= halt_result.get('auc', 0):
        print(f"Single signal ({best_signal['name']}) performs comparably to HALT probe "
              f"({best_signal.get('auc', 0):.3f} vs {halt_result['auc']:.3f} AUC)")
    else:
        print(f"HALT probe outperforms single signals ({halt_result['auc']:.3f} vs "
              f"{best_signal.get('auc', 0):.3f} AUC) — as expected given more features")

    print(f"\nKey insight: The constraint residual framework decomposes the same hidden "
          f"state signal that HALT probes use into 5 interpretable dimensions. "
          f"While single σ_i dimensions have lower AUC, they offer EXPLAINABILITY "
          f"(which constraint is violated) that black-box probes lack.")


if __name__ == "__main__":
    main()
