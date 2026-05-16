"""
学习约束函数 —— 用线性探针替代手工 σ_i，学出加权残差 Π_w。

A1：训练 5 个线性探针，每个学一个约束维度在隐藏空间中的方向
A2：训练加权组合逻辑回归，学习最优 σ_i 权重

与手工版本做头对头对比。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
from copy import deepcopy

from constraint_residual.hallucination_predictor.model_wrapper import ModelWrapper
from constraint_residual.hallucination_predictor.constraint_functions import (
    ConstraintFunctionBank,
    ConstraintState,
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
from constraint_residual.hallucination_predictor.advanced_analysis import (
    extract_rich_features,
    RichFeatures,
)


# ============================================================
# 学习探针
# ============================================================

@dataclass
class LearnedProbes:
    """5 个从隐藏状态中学习出的约束探针"""
    probe_fact: torch.Tensor      # [hidden_dim]
    probe_syntax: torch.Tensor    # [hidden_dim]
    probe_style: torch.Tensor     # [hidden_dim]
    probe_safety: torch.Tensor    # [hidden_dim]
    probe_coherence: torch.Tensor # [hidden_dim]
    bias_fact: float = 0.0
    bias_syntax: float = 0.0
    bias_style: float = 0.0
    bias_safety: float = 0.0
    bias_coherence: float = 0.0

    @staticmethod
    def from_sklearn(models: Dict[str, 'LogisticRegression'], scaler: 'StandardScaler'):
        """从 sklearn 逻辑回归模型提取探针方向和偏置"""
        probes = {}
        biases = {}
        for name, clf in models.items():
            w = torch.from_numpy(clf.coef_[0].astype(np.float32))
            # coef_ shape is (1, hidden_dim) for binary classification
            # Normalize to unit vector for direction
            w_norm = torch.norm(w)
            if w_norm > 1e-8:
                w = w / w_norm
            probes[name] = w
            biases[name] = float(clf.intercept_[0]) if clf.intercept_ is not None else 0.0

        dim = next((p.shape[0] for p in probes.values()), 1)

        return LearnedProbes(
            probe_fact=probes.get('fact', torch.zeros(dim)),
            probe_syntax=probes.get('syntax', torch.zeros(dim)),
            probe_style=probes.get('style', torch.zeros(dim)),
            probe_safety=probes.get('safety', torch.zeros(dim)),
            probe_coherence=probes.get('coherence', torch.zeros(dim)),
            bias_fact=biases.get('fact', 0.0),
            bias_syntax=biases.get('syntax', 0.0),
            bias_style=biases.get('style', 0.0),
            bias_safety=biases.get('safety', 0.0),
            bias_coherence=biases.get('coherence', 0.0),
        )

    def compute_sigma(self, hidden_state: torch.Tensor, name: str) -> float:
        """用学习探针计算某维度的约束强度"""
        probe = getattr(self, f'probe_{name}')
        bias = getattr(self, f'bias_{name}')
        h = hidden_state.float()
        # Projection
        score = torch.dot(h, probe.to(h.device)).item() + bias
        # Sigmoid to [0,1]
        return float(1.0 / (1.0 + np.exp(-score)))


class LearnedConstraintBank:
    """学习版的约束函数库 —— 替代 ConstraintFunctionBank"""

    def __init__(self, probes: LearnedProbes):
        self.probes = probes
        self._truth_direction = None  # unused in learned mode

    def calibrate_truth_direction(self, *args, **kwargs):
        pass  # 不需要标定，已经学出来了

    def sigma_fact(self, hidden_state: torch.Tensor) -> float:
        return self.probes.compute_sigma(hidden_state, 'fact')

    def sigma_syntax(self, attention_weights: torch.Tensor = None) -> float:
        # 学习探针不依赖 attention，返回默认值
        return 0.5

    def sigma_style(self, layer_hidden_states: List[torch.Tensor]) -> float:
        return 0.5

    def sigma_safety(self, hidden_state: torch.Tensor) -> float:
        return self.probes.compute_sigma(hidden_state, 'safety')

    def sigma_coherence(self, h_prev: Optional[torch.Tensor], h_curr: torch.Tensor) -> float:
        # 学习 coherence 探针用两个隐藏状态的差值
        if h_prev is None:
            return 0.5
        diff = h_curr.float() - h_prev.float()
        return self.probes.compute_sigma(diff, 'coherence')

    def compute_all(
        self,
        hidden_states: torch.Tensor,
        layer_hidden_list: List[torch.Tensor],
        attention_weights: Optional[torch.Tensor] = None,
    ) -> List[ConstraintState]:
        """用学习探针计算所有 token 的约束状态"""
        seq_len = hidden_states.shape[0]
        results = []
        h_prev = None

        for t in range(seq_len):
            h_curr = hidden_states[t]

            sf = self.sigma_fact(h_curr)
            ss = 0.5  # syntax: 学习探针主要关注事实维度
            sst = 0.5  # style: 学习探针暂不支持逐层分析
            ssa = self.sigma_safety(h_curr)
            sc = self.sigma_coherence(h_prev, h_curr)

            results.append(ConstraintState(
                sigma_fact=sf,
                sigma_syntax=ss,
                sigma_style=sst,
                sigma_safety=ssa,
                sigma_coherence=sc,
            ))
            h_prev = h_curr

        return results


# ============================================================
# 训练
# ============================================================

def collect_training_data(
    wrapper: ModelWrapper,
    n_questions: int = 40,
) -> Tuple[List[np.ndarray], List[int], List[str]]:
    """收集标注训练数据：每个问题的隐藏状态 + 标签。

    返回：(hidden_state_vectors, labels, questions)
    hidden_state_vectors: 每个样本的平均隐藏状态 [n_samples, hidden_dim]
    labels: 1 = hallucination, 0 = correct
    """
    pairs = _builtin_false_premise_pairs()[:8]
    truth_qa = _builtin_truthfulqa()[:8]

    all_questions = []
    for p in pairs:
        all_questions.append({"question": p["question"], "best_answer": p["best_answer"]})
        all_questions.append({"question": p["true_question"], "best_answer": p["true_answer"]})
    for q in truth_qa:
        ref = q.get("best_answer", "")
        if not ref and isinstance(q.get("correct_answers"), list):
            ref = q["correct_answers"][0] if q["correct_answers"] else ""
        all_questions.append({"question": q["question"], "best_answer": ref[:300]})

    X = []
    y = []
    questions_list = []

    print(f"Collecting training data from {min(n_questions, len(all_questions))} questions...")
    for i, q in enumerate(all_questions[:n_questions]):
        try:
            state = wrapper.generate_and_extract(
                prompt=q["question"],
                max_new_tokens=64,
                temperature=0.6,
                do_sample=True,
            )
            response = state.generated_text

            # 标签
            is_hallu = self_judge(wrapper, q["question"], response, q.get("best_answer", ""))

            # 平均隐藏状态作为特征
            # Filter out special tokens for clean hidden state
            special_prefixes = ('<|im_', 'system', 'user', 'assistant', '\n', '<')
            content_idx = [
                t for t in range(state.hidden_states.shape[0])
                if not any(state.tokens[t].startswith(p) for p in special_prefixes)
            ]
            if content_idx:
                h_mean = state.hidden_states[content_idx].float().mean(dim=0).cpu().numpy()
            else:
                h_mean = state.hidden_states.float().mean(dim=0).cpu().numpy()

            X.append(h_mean)
            y.append(1 if is_hallu else 0)
            questions_list.append(q["question"][:80])

            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{min(n_questions, len(all_questions))}] "
                      f"hallu={is_hallu}, hidden_dim={h_mean.shape}")
        except Exception as e:
            print(f"  [{i+1}] ERROR: {e}")
            continue

    print(f"Collected {len(X)} samples ({sum(y)} hallu / {len(y) - sum(y)} correct)")
    return X, y, questions_list


def train_probes(
    X: List[np.ndarray],
    y: List[int],
) -> Dict:
    """训练 5 个逻辑回归探针，每个学一个约束维度。

    策略：由于我们只有"是否幻觉"的标签，没有每个 σ_i 的单独标签，
    我们使用不同的特征子空间来训练不同探针：
    - σ_fact: 全隐藏状态 → 幻觉标签（事实约束是最直接的信号）
    - σ_safety: 前半部分隐藏维度 → 幻觉标签（安全/拒绝信号通常在特定维度）
    - σ_style, σ_syntax, σ_coherence: 使用不同的随机子空间作为正则化

    然后训练一个 meta-learner 学习如何组合这 5 个探针的输出。
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score

    X_arr = np.array(X)
    y_arr = np.array(y)

    if len(set(y_arr)) < 2:
        print("ERROR: Only one class in training data!")
        return {}

    n, d = X_arr.shape
    print(f"Training probes: {n} samples, {d} features, {sum(y_arr)} positives")

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_arr)

    # 不同维度的子空间划分（让不同探针关注隐藏状态的不同方面）
    # 这样虽然是同一条数据，但每个探针看到的是不同子空间 → 学到互补的信号
    subspace_dims = {
        'fact': slice(0, d),              # 全空间 → 主要事实信号
        'syntax': slice(0, d // 3),       # 前 1/3 维度
        'style': slice(d // 3, 2 * d // 3),  # 中间 1/3 维度
        'safety': slice(2 * d // 3, d),   # 后 1/3 维度
        'coherence': slice(d // 4, 3 * d // 4),  # 中间 1/2 维度
    }

    probes = {}
    cv_scores = {}

    print("\nTraining 5 constraint probes:")
    for name, subspace in subspace_dims.items():
        X_sub = X_scaled[:, subspace]
        clf = LogisticRegression(max_iter=2000, class_weight='balanced', C=0.1)
        clf.fit(X_sub, y_arr)

        # Cross-validation
        try:
            cv_auc = cross_val_score(clf, X_sub, y_arr, cv=min(3, sum(y_arr), len(y_arr) - sum(y_arr)),
                                     scoring='roc_auc').mean()
        except Exception:
            cv_auc = 0.5

        probes[name] = clf
        cv_scores[name] = float(cv_auc)
        coef_norm = np.linalg.norm(clf.coef_[0])
        print(f"  σ_{name}: CV AUC={cv_auc:.3f}, ||w||={coef_norm:.2f}, "
              f"dim={X_sub.shape[1]}, bias={clf.intercept_[0]:.4f}")

    # Meta-learner: 组合 5 个探针的输出
    meta_X = np.zeros((n, 5))
    for i in range(n):
        for j, (name, subspace) in enumerate(subspace_dims.items()):
            proba = probes[name].predict_proba(X_scaled[i:i+1, subspace])[:, 1]
            meta_X[i, j] = proba[0]

    meta_clf = LogisticRegression(max_iter=2000, class_weight='balanced')
    meta_clf.fit(meta_X, y_arr)

    try:
        meta_cv = cross_val_score(meta_clf, meta_X, y_arr, cv=min(3, sum(y_arr), n - sum(y_arr)),
                                  scoring='roc_auc').mean()
    except Exception:
        meta_cv = 0.5

    # 学习到的权重
    weights = meta_clf.coef_[0]
    print(f"\nMeta-learner AUC: {meta_cv:.3f}")
    print(f"Learned σ weights: fact={weights[0]:.3f}, syntax={weights[1]:.3f}, "
          f"style={weights[2]:.3f}, safety={weights[3]:.3f}, coherence={weights[4]:.3f}")

    return {
        "probes": probes,
        "scaler": scaler,
        "subspace_dims": subspace_dims,
        "cv_scores": cv_scores,
        "meta_learner": meta_clf,
        "meta_cv": float(meta_cv),
        "weights": [float(w) for w in weights],
        "n_train": n,
        "d": d,
    }


def evaluate_learned_vs_handcrafted(
    wrapper: ModelWrapper,
    bank_handcrafted: ConstraintFunctionBank,
    train_result: Dict,
    n_test: int = 20,
) -> Dict:
    """在留出测试集上对比学习版 vs 手工版"""
    from sklearn.metrics import roc_auc_score

    # Build learned bank
    lprobes = LearnedProbes.from_sklearn(
        train_result["probes"],
        train_result["scaler"],
    )
    learned_bank = LearnedConstraintBank(lprobes)

    # Load test questions
    pairs = _builtin_false_premise_pairs()[8:12]
    truth_qa = _builtin_truthfulqa()[8:16]

    test_questions = []
    for p in pairs:
        test_questions.append({"question": p["question"], "best_answer": p["best_answer"]})
        test_questions.append({"question": p["true_question"], "best_answer": p["true_answer"]})
    for q in truth_qa:
        ref = q.get("best_answer", "")
        if not ref and isinstance(q.get("correct_answers"), list):
            ref = q["correct_answers"][0] if q["correct_answers"] else ""
        test_questions.append({"question": q["question"], "best_answer": ref[:300]})

    test_questions = test_questions[:n_test]

    hand_deltas = []
    learned_deltas_weighted = []
    learned_deltas_unweighted = []
    labels = []

    scaler = train_result["scaler"]
    subspace_dims = train_result["subspace_dims"]
    probes = train_result["probes"]
    meta_weights = np.array(train_result["weights"])

    print(f"\nEvaluating on {len(test_questions)} test questions...")

    for i, q in enumerate(test_questions):
        try:
            state = wrapper.generate_and_extract(
                prompt=q["question"],
                max_new_tokens=64,
                temperature=0.6,
                do_sample=True,
            )
            response = state.generated_text

            # Label
            is_hallu = self_judge(wrapper, q["question"], response, q.get("best_answer", ""))
            labels.append(1 if is_hallu else 0)

            # ---- Hand-crafted ----
            hc_cstates = bank_handcrafted.compute_all(
                state.hidden_states,
                state.layer_hidden_states,
                state.attention_weights,
            )
            hc_grads = compute_constraint_gradients(hc_cstates)
            hc_res, _, _ = compute_residual(hc_grads)
            special_prefixes = ('<|im_', 'system', 'user', 'assistant', '\n', '<')
            content_idx = [t for t in range(min(len(state.tokens), len(hc_res)))
                          if not any(state.tokens[t].startswith(p) for p in special_prefixes)]
            hc_input = np.mean([hc_res[t] for t in content_idx if t < len(hc_res)]) if content_idx else 0.0

            # Output hand-crafted
            hc_output = hc_input
            if response and len(response.strip()) > 5:
                try:
                    out_state = wrapper.extract_output_state(response)
                    out_cs = bank_handcrafted.compute_all(
                        out_state.hidden_states,
                        out_state.layer_hidden_states,
                        out_state.attention_weights,
                    )
                    out_grads = compute_constraint_gradients(out_cs)
                    out_res, _, _ = compute_residual(out_grads)
                    out_f = [r for r in out_res if r > 1e-6]
                    hc_output = np.mean(out_f) if out_f else hc_input
                except Exception:
                    pass

            hand_deltas.append(hc_output - hc_input)

            # ---- Learned ----
            # Input hidden: mean pooled over content tokens
            h_in = state.hidden_states[content_idx].float().mean(dim=0).cpu().numpy() if content_idx else state.hidden_states.float().mean(dim=0).cpu().numpy()
            h_in_scaled = scaler.transform(h_in.reshape(1, -1))

            # 5 probe scores for input
            input_sigmas = np.zeros(5)
            for j, (name, subspace) in enumerate(subspace_dims.items()):
                proba = probes[name].predict_proba(h_in_scaled[:, subspace])[:, 1]
                input_sigmas[j] = proba[0]

            # Output hidden
            output_sigmas = input_sigmas.copy()
            if response and len(response.strip()) > 5:
                try:
                    out_state_full = wrapper.extract_output_state(response)
                    h_out = out_state_full.hidden_states.float().mean(dim=0).cpu().numpy()
                    h_out_scaled = scaler.transform(h_out.reshape(1, -1))
                    for j, (name, subspace) in enumerate(subspace_dims.items()):
                        proba = probes[name].predict_proba(h_out_scaled[:, subspace])[:, 1]
                        output_sigmas[j] = proba[0]
                except Exception:
                    pass

            # Gradients
            sigma_grads = output_sigmas - input_sigmas  # [5]

            # Weighted Π_w = |Σ w_i * ∇σ_i|
            weighted_pi = abs(float(np.dot(sigma_grads, meta_weights)))
            learned_deltas_weighted.append(weighted_pi)

            # Unweighted Π = |Σ ∇σ_i / 5|
            unweighted_pi = abs(float(np.mean(sigma_grads)))
            learned_deltas_unweighted.append(unweighted_pi)

            if (i + 1) % 5 == 0:
                print(f"  [{i+1}/{len(test_questions)}] hallu={is_hallu} "
                      f"hc={hc_output - hc_input:+.4f} lw={weighted_pi:.4f} lu={unweighted_pi:.4f}")

        except Exception as e:
            print(f"  [{i+1}] ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Compare
    labels_arr = np.array(labels)
    print(f"\n{'='*60}")
    print(f"HEAD-TO-HEAD: Hand-crafted vs Learned Constraints")
    print(f"{'='*60}")

    results = {}
    for name, scores in [
        ("Hand-crafted Δ||Π||", hand_deltas),
        ("Learned unweighted Π", learned_deltas_unweighted),
        ("Learned weighted Π_w", learned_deltas_weighted),
    ]:
        if len(set(labels_arr)) >= 2 and len(scores) >= 2:
            auc = roc_auc_score(labels_arr[:len(scores)], scores)
        else:
            auc = 0.5
        hallu_s = [s for s, l in zip(scores, labels_arr) if l == 1]
        correct_s = [s for s, l in zip(scores, labels_arr) if l == 0]
        mean_h = np.mean(hallu_s) if hallu_s else 0
        mean_c = np.mean(correct_s) if correct_s else 0
        print(f"  {name:<30s} AUC={auc:.3f}  hallu={mean_h:+.4f}  correct={mean_c:+.4f}")
        results[name] = {"auc": float(auc), "mean_hallu": float(mean_h), "mean_correct": float(mean_c)}

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    print("=" * 60)
    print("LEARNED CONSTRAINTS: A1 + A2")
    print("=" * 60)

    # 1. Load model
    wrapper = ModelWrapper(model_name=args.model, device=args.device)

    # 2. Hand-crafted bank for comparison
    bank_hc = ConstraintFunctionBank()
    calibrate_truth_direction(wrapper, bank_hc)
    calibrate_refusal_direction(wrapper, bank_hc)

    # 3. Collect training data
    print("\n--- Step 1: Collect training data ---")
    X, y, questions = collect_training_data(wrapper, n_questions=32)

    # 4. Train probes
    print("\n--- Step 2: Train learned probes ---")
    train_result = train_probes(X, y)

    # 5. Evaluate
    print("\n--- Step 3: Evaluate on test set ---")
    results = evaluate_learned_vs_handcrafted(
        wrapper, bank_hc, train_result, n_test=16
    )

    # 6. Save
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    output = {
        "config": {"model": args.model, "n_train": len(X), "n_test": 16},
        "training": {
            "cv_scores": train_result.get("cv_scores", {}),
            "meta_cv": train_result.get("meta_cv", 0.5),
            "learned_weights": train_result.get("weights", []),
            "weight_labels": ["σ_fact", "σ_syntax", "σ_style", "σ_safety", "σ_coherence"],
        },
        "test_results": results,
    }

    output_path = output_dir / "learned_constraints.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {output_path}")

    # 7. Summary
    print("\n" + "=" * 60)
    print("SUMMARY: Learned vs Hand-crafted")
    print("=" * 60)
    for name, r in sorted(results.items(), key=lambda x: -x[1]['auc']):
        print(f"  {name:<30s} AUC = {r['auc']:.3f}")

    if results.get("Learned weighted Π_w", {}).get("auc", 0) > results.get("Hand-crafted Δ||Π||", {}).get("auc", 0):
        print(f"\n✓ Learned weighted Π_w outperforms hand-crafted Δ||Π||")
    else:
        print(f"\n✗ Learned method didn't beat hand-crafted — may need more training data")
        print(f"  This is expected with small sample size ({len(X)} train / 16 test)")


if __name__ == "__main__":
    main()
