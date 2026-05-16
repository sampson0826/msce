"""
约束函数 σ_i 定义 —— LLM 内部约束场的 5 个维度。

每个 σ_i 接收 token 位置 t 的模型内部状态，返回该约束在 t 处的强度标量。
所有 σ_i 在 [0, 1] 范围内归一化。所有函数都做了 NaN/Inf 防护。
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


def _safe_float(x: float) -> float:
    """NaN/Inf 防护：返回有限浮点数，否则返回 0.5"""
    if np.isnan(x) or np.isinf(x):
        return 0.5
    return float(np.clip(x, 0.0, 1.0))


@dataclass
class ConstraintState:
    """单个 token 位置的完整约束状态"""
    sigma_fact: float       # 事实一致性约束（注意力集中度代理）
    sigma_syntax: float     # 语法完整性约束（注意力熵倒数）
    sigma_style: float      # 风格/语调约束（跨层范数稳定性）
    sigma_safety: float     # 安全拒绝约束（拒绝方向投影）
    sigma_coherence: float  # 逻辑连贯约束（邻接 hidden state 余弦相似度）


class ConstraintFunctionBank:
    """5 个约束函数的计算引擎，全部 NaN-safe"""

    def __init__(self):
        self._truth_direction: Optional[torch.Tensor] = None
        self._refusal_direction: Optional[torch.Tensor] = None

    def calibrate_truth_direction(
        self,
        truthful_hidden: torch.Tensor,
        false_hidden: torch.Tensor,
    ):
        diff = truthful_hidden.float().mean(dim=0) - false_hidden.float().mean(dim=0)
        diff_norm = torch.norm(diff)
        if diff_norm > 1e-8:
            self._truth_direction = diff / diff_norm

    def sigma_fact(self, hidden_state: torch.Tensor) -> float:
        """
        事实约束强度。
        标定前：用 max(attention) 作为"模型在此 token 处的决策确定度"代理。
        文献依据：高注意力集中度 → 模型"确信"当前 token → 通常在事实性语境中更可靠。
        标定后：用 truth direction 投影。
        """
        if self._truth_direction is not None:
            h = F.normalize(hidden_state.float(), dim=-1)
            td = self._truth_direction.to(h.device)
            score = torch.dot(h, td).item()
            return _safe_float((score + 1) / 2)

        # 未标定：用 hidden state L2 norm 的相对稳定性
        # 这个值本身区分度不高，但 token 间的变化（∇σ）能捕捉约束力变化
        h_float = hidden_state.float()
        h_norm = torch.norm(h_float).item()
        # 归一化到 [0,1]：用经验范围（Qwen2.5-1.5B, d=1536 → norm 约 35-45）
        typical_norm = 38.0
        score = 1.0 - abs(h_norm - typical_norm) / typical_norm
        return _safe_float(score)

    @staticmethod
    def sigma_syntax(attention_weights: torch.Tensor) -> float:
        """
        语法约束强度 = 1 - normalized_attention_entropy。
        注意力越集中（熵越低）→ 语法约束越强。
        attention_weights: 1D [key_len] 或 2D [heads, key_len]
        """
        if attention_weights is None or attention_weights.numel() == 0:
            return 0.5

        attn = attention_weights.float()
        if attn.dim() > 1:
            attn = attn.mean(dim=0)

        # NaN check
        if torch.isnan(attn).any() or torch.isinf(attn).any():
            return 0.5

        attn = torch.clamp(attn, min=1e-10)
        total = attn.sum()
        if total < 1e-10:
            return 0.5
        attn = attn / total

        entropy = -torch.sum(attn * torch.log(attn)).item()
        n = len(attn)
        if n <= 1:
            return 1.0
        max_entropy = np.log(n)

        if max_entropy < 1e-10:
            return 0.5

        normalized = 1.0 - (entropy / max_entropy)
        return _safe_float(normalized)

    @staticmethod
    def sigma_style(layer_hidden_states: List[torch.Tensor]) -> float:
        """
        风格约束强度 = 跨层隐藏状态范数的稳定性。
        变异系数越低 → 各层表达强度越一致 → 风格约束越强。
        """
        if len(layer_hidden_states) < 2:
            return 0.5

        norms = []
        for h in layer_hidden_states:
            if h is None:
                continue
            hn = h.float()
            if torch.isnan(hn).any() or torch.isinf(hn).any():
                continue
            n = torch.norm(hn).item()
            if not (np.isnan(n) or np.isinf(n)):
                norms.append(n)

        if len(norms) < 2:
            return 0.5

        norms_arr = np.array(norms)
        mean_n = norms_arr.mean()
        if mean_n < 1e-8:
            return 0.5

        cv = norms_arr.std() / mean_n
        stability = float(np.exp(-3 * cv))
        return _safe_float(stability)

    def calibrate_refusal_direction(
        self,
        refusal_hidden: torch.Tensor,
        normal_hidden: torch.Tensor,
    ):
        diff = refusal_hidden.float().mean(dim=0) - normal_hidden.float().mean(dim=0)
        diff_norm = torch.norm(diff)
        if diff_norm > 1e-8:
            self._refusal_direction = diff / diff_norm

    def sigma_safety(self, hidden_state: torch.Tensor) -> float:
        """安全约束强度。未标定时返回中位值。"""
        if self._refusal_direction is not None:
            h = F.normalize(hidden_state.float(), dim=-1)
            rd = self._refusal_direction.to(h.device)
            score = torch.dot(h, rd).item()
            return _safe_float((score + 1) / 2)
        return 0.3

    @staticmethod
    def sigma_coherence(h_prev: Optional[torch.Tensor], h_curr: torch.Tensor) -> float:
        """
        逻辑连贯约束 = 相邻 token 最后一层隐藏状态的余弦相似度。
        """
        if h_prev is None:
            return 0.5

        hp = h_prev.float()
        hc = h_curr.float()

        if torch.isnan(hp).any() or torch.isinf(hp).any():
            return 0.5
        if torch.isnan(hc).any() or torch.isinf(hc).any():
            return 0.5

        sim = F.cosine_similarity(hp.unsqueeze(0), hc.unsqueeze(0), dim=-1).item()
        if np.isnan(sim) or np.isinf(sim):
            return 0.5
        return _safe_float((sim + 1) / 2)

    # ------------------------------------------------------------------
    # 批量计算
    # ------------------------------------------------------------------
    def compute_all(
        self,
        hidden_states: torch.Tensor,           # [seq_len, d]
        layer_hidden_list: List[torch.Tensor],  # [num_layers, seq_len, d]
        attention_weights: Optional[torch.Tensor],  # [heads, seq_len, seq_len]
    ) -> List[ConstraintState]:
        seq_len = hidden_states.shape[0]
        num_layers = len(layer_hidden_list)
        results = []
        h_prev = None

        for t in range(seq_len):
            h_curr = hidden_states[t]

            # σ_fact：注意力集中度 — 用平均注意力与均匀分布的 KL 散度
            # (比 max 有更大的 token 间变化)
            if attention_weights is not None and attention_weights.dim() == 3:
                if t < attention_weights.shape[1]:
                    attn_slice = attention_weights[:, t, :t+1].mean(dim=0).float()  # [t+1]
                elif attention_weights.shape[1] > 0:
                    attn_slice = attention_weights[:, -1, :].mean(dim=0).float()
                else:
                    attn_slice = None

                if attn_slice is not None and attn_slice.numel() > 1:
                    attn_slice = torch.clamp(attn_slice / attn_slice.sum(), min=1e-10)
                    n_key = attn_slice.numel()
                    uniform = torch.full_like(attn_slice, 1.0 / n_key)
                    kl = torch.sum(attn_slice * (torch.log(attn_slice) - torch.log(uniform))).item()
                    # KL ∈ [0, log(n_key)], normalize
                    max_kl = np.log(n_key) if n_key > 1 else 1.0
                    sf = _safe_float(kl / max_kl if max_kl > 1e-10 else 0.5)
                else:
                    sf = 0.5
            else:
                sf = self.sigma_fact(h_curr)

            # σ_syntax：注意力熵 (1 - normalized_entropy)
            # 注意力越集中 → 语法约束越强 → sigma_syntax 越高
            if attention_weights is not None and attention_weights.dim() == 3:
                if t < attention_weights.shape[1]:
                    attn_slice = attention_weights[:, t, :t+1].mean(dim=0).float()
                else:
                    attn_slice = attention_weights[:, -1, :].mean(dim=0).float()
                attn_slice = torch.clamp(attn_slice / (attn_slice.sum() + 1e-10), min=1e-10)
                n_key = attn_slice.numel()
                if n_key > 1:
                    entropy = -torch.sum(attn_slice * torch.log(attn_slice)).item()
                    max_entropy = np.log(n_key)
                    ss = _safe_float(1.0 - entropy / max_entropy) if max_entropy > 1e-10 else 0.5
                else:
                    ss = 1.0
            else:
                ss = 0.5

            # σ_style：跨层范数稳定性
            layer_h_at_t = []
            for l in range(num_layers):
                lh = layer_hidden_list[l]
                if t < lh.shape[0]:
                    layer_h_at_t.append(lh[t])
            sst = self.sigma_style(layer_h_at_t) if len(layer_h_at_t) >= 2 else 0.5

            # σ_safety：hidden state 在 cross-layer principal direction 上的稳定性
            # 跨层 hidden state 越一致 → 安全约束越强（模型不偏离安全区域）
            if len(layer_h_at_t) >= 2:
                h_stack = torch.stack([h.float() for h in layer_h_at_t])  # [n_layers, d]
                # Mean hidden state across layers, variance per dimension
                layer_mean = h_stack.mean(dim=0)
                # Safety: how much each layer deviates from the mean (lower variation = safer)
                deviations = torch.norm(h_stack - layer_mean.unsqueeze(0), dim=-1)  # [n_layers]
                mean_dev = deviations.mean().item()
                # Normalize: mean_dev 0→1.0, 50→0.0
                ssa = _safe_float(np.exp(-mean_dev / 15.0))
            else:
                ssa = 0.5

            # σ_coherence：相邻 token hidden state 的余弦相似度
            if h_prev is not None:
                sc = self.sigma_coherence(h_prev, h_curr)
            else:
                sc = 0.5

            results.append(ConstraintState(
                sigma_fact=sf,
                sigma_syntax=ss,
                sigma_style=sst,
                sigma_safety=ssa,
                sigma_coherence=sc,
            ))
            h_prev = h_curr

        return results


def compute_constraint_gradients(
    states: List[ConstraintState]
) -> List[np.ndarray]:
    """∇σ_i(token_t) = σ_i(token_t) - σ_i(token_{t-1})"""
    fields = ['sigma_fact', 'sigma_syntax', 'sigma_style', 'sigma_safety', 'sigma_coherence']
    gradients = []
    for t in range(1, len(states)):
        grad = np.array([
            getattr(states[t], f) - getattr(states[t-1], f)
            for f in fields
        ], dtype=np.float64)
        # NaN guard
        if np.isnan(grad).any():
            grad = np.nan_to_num(grad, nan=0.0)
        gradients.append(grad)
    return gradients


def compute_windowed_gradients(
    states: List[ConstraintState],
    window_size: int = 8,
) -> List[np.ndarray]:
    """Compute gradients between window-aggregated constraint states.

    Groups tokens into windows of window_size, computes mean σ per window,
    then computes gradients between consecutive windows. This captures
    larger-scale constraint variation that per-token gradients miss.

    Returns list of 5D gradient vectors [Δfact, Δsyntax, Δstyle, Δsafety, Δcoherence].
    """
    fields = ['sigma_fact', 'sigma_syntax', 'sigma_style', 'sigma_safety', 'sigma_coherence']
    n = len(states)
    if n < window_size * 2:
        window_size = max(2, n // 3)

    window_means = []
    for start in range(0, n, window_size):
        end = min(start + window_size, n)
        if end - start < window_size // 2:
            continue
        chunk = states[start:end]
        means = [np.mean([getattr(s, f) for s in chunk]) for f in fields]
        window_means.append(np.array(means, dtype=np.float64))

    gradients = []
    for w in range(1, len(window_means)):
        grad = window_means[w] - window_means[w - 1]
        if np.isnan(grad).any():
            grad = np.nan_to_num(grad, nan=0.0)
        gradients.append(grad)
    return gradients


def compute_windowed_pi_components(
    states: List[ConstraintState],
    window_size: int = 8,
) -> dict[str, float]:
    """Window-level π component decomposition.

    Returns mean absolute contribution per σ dimension.
    """
    grads = compute_windowed_gradients(states, window_size=window_size)
    if not grads:
        return {"sigma_fact": 0.0, "sigma_syntax": 0.0, "sigma_style": 0.0,
                "sigma_safety": 0.0, "sigma_coherence": 0.0}
    components = ["sigma_fact", "sigma_syntax", "sigma_style", "sigma_safety", "sigma_coherence"]
    sums = {c: 0.0 for c in components}
    for g in grads:
        for i, c in enumerate(components):
            sums[c] += abs(g[i])
    n = len(grads)
    return {c: s / n for c, s in sums.items()}


def compute_residual(
    gradients: List[np.ndarray]
) -> Tuple[List[float], List[float], List[float]]:
    """
    Π = Σ∇σ_i（约束残差）

    返回：
    - residual_magnitudes: 每个 token 的 ||Π||
    - cancellation_ratios: 每个 token 的 c(p) = ||Σ∇σ|| / Σ||∇σ||
    - total_constraint: 每个 token 的 Σ||∇σ||
    """
    magnitudes = []
    cancels = []
    totals = []

    for grad in gradients:
        pi = grad.sum()                      # Π = Σ∇σ_i
        mag = abs(float(pi))                  # ||Π||
        total = float(np.abs(grad).sum())     # Σ||∇σ||

        magnitudes.append(mag if not np.isnan(mag) else 0.0)
        cancels.append(float(mag / total) if total > 1e-10 else 0.5)
        totals.append(total if not np.isnan(total) else 0.0)

    return magnitudes, cancels, totals
