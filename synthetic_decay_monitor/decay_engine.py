"""
衰减引擎 — S_{n+1} = S_n · (1 - β) 的完整可计算实现。

核心模型：
- 稳定性 S_n 是第 n 代的约束基完整性度量
- β 不是常数，是执行者类型构成的函数
- 不同能力维度独立衰减，共享约束拓扑依赖
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from synthetic_decay_monitor.constraint_extractor import (
    ConstraintFieldSnapshot, ConstraintState, compute_residual,
)

# 执行者类型基础衰减系数
# α_EI > α_EII > α_EIII：E-I 一旦退化影响最大
BASE_ALPHAS = {
    "E-I": 0.40,    # 定理级约束最敏感
    "E-II": 0.20,   # 标度假说中等
    "E-III": 0.08,   # 边界条件最稳定
}

# 临界崩溃阈值
S_CRITICAL = 0.30


@dataclass
class CapabilityStability:
    capability: str
    generation: int
    S_n: float = 1.0
    beta: float = 0.25
    executor_composition: dict[str, float] = field(default_factory=dict)
    constraint_magnitudes: list[float] = field(default_factory=list)
    cancellation_ratio: float = 0.0
    raw_sigmas: dict[str, float] = field(default_factory=dict)

    @property
    def is_collapsed(self) -> bool:
        return self.S_n < S_CRITICAL

    @property
    def status(self) -> str:
        if self.S_n > 0.8:
            return "healthy"
        elif self.S_n > 0.5:
            return "degrading"
        elif self.S_n > S_CRITICAL:
            return "critical"
        return "collapsed"


def calibrate_beta(
    capability: str,
    executor_composition: dict[str, float],
    override_alphas: Optional[dict[str, float]] = None,
) -> float:
    """β = Σ(α_i * p_i)，其中 p_i 是执行者类型 i 的占比。

    如果提供多代观测数据，优先从数据拟合 β。
    executor_composition: {'E-I': 0.6, 'E-II': 0.3, 'E-III': 0.1}
    """
    alphas = override_alphas or BASE_ALPHAS
    beta = sum(
        alphas.get(exec_type, 0.0) * proportion
        for exec_type, proportion in executor_composition.items()
    )
    return min(beta, 0.95)


def calibrate_beta_from_data(
    snapshots: list[ConstraintFieldSnapshot],
) -> dict[str, float]:
    """从跨代约束场快照拟合每个能力维度的 β。

    预注册方法：固定使用指数衰减模型 + total_constraint 拟合目标。
    无模型选择、无目标选择——消除 post-hoc 偏差。
    """
    if len(snapshots) < 2:
        return {"*": 0.25}

    generations = np.array([s.generation for s in snapshots], dtype=float)
    n = len(generations)

    # 预注册目标: total_constraint = Σ|∇σ|（总约束活动量，不依赖模型选择）
    total_mags = np.array([s.total_constraint for s in snapshots])

    if total_mags[0] < 1e-10:
        return {"*": 0.25}

    # 预注册模型: 指数衰减 y = y0 * (1-β)^x  →  log(y/y0) = x * log(1-β)
    y_safe = np.maximum(total_mags, 1e-10)
    y0 = y_safe[0]
    log_ratios = np.log(y_safe / y0)
    slope, r2 = _ols(generations, log_ratios)
    beta = 1.0 - np.exp(slope)
    beta = max(min(float(beta), 0.55), 0.001)
    r2_adj = 1 - (1 - r2) * (n - 1) / max(n - 2, 1)

    return {
        "*": beta,
        "_fit_model": "exponential",
        "_fit_r2": float(r2),
        "_fit_r2_adj": float(r2_adj),
        "_fit_target": "total_constraint",
        "_preregistered": True,
    }


def _monotonicity(arr: np.ndarray) -> float:
    """量化序列的单调衰减程度。1.0 = 完美单调递减，-1.0 = 完美单调递增。"""
    if len(arr) < 2:
        return 0.0
    diffs = arr[1:] - arr[:-1]
    if np.all(diffs <= 0):
        return 1.0 - np.mean(np.abs(diffs)) / (np.std(arr) + 1e-10) * 0.1
    if np.all(diffs >= 0):
        return -1.0
    n_neg = np.sum(diffs < 0)
    return (n_neg / len(diffs)) * 2 - 1


def _fit_decay_models(x: np.ndarray, y: np.ndarray) -> list[dict]:
    """拟合指数、线性、幂律三种衰减模型，返回各模型的 β 和 R²。"""
    models = []
    n = len(x)
    y_safe = np.maximum(y, 1e-10)
    y0 = y_safe[0]

    # 1. 指数衰减: y = y0 * (1-beta)^x  →  log(y/y0) = x * log(1-beta)
    if y0 > 1e-10:
        log_ratios = np.log(y_safe / y0)
        slope, r2 = _ols(x, log_ratios)
        beta_exp = 1.0 - np.exp(slope)
        beta_exp = max(min(beta_exp, 0.55), 0.001)
        r2_adj = 1 - (1 - r2) * (n - 1) / max(n - 2, 1)
        models.append({"model": "exponential", "beta": beta_exp, "r2": r2, "r2_adj": r2_adj})

    # 2. 线性衰减: y = y0 * (1 - beta*x)
    y_norm = y_safe / y0
    slope_lin, r2_lin = _ols(x, y_norm)
    beta_lin = -slope_lin
    beta_lin = max(min(beta_lin, 0.55), 0.001)
    r2_adj_lin = 1 - (1 - r2_lin) * (n - 1) / max(n - 2, 1)
    models.append({"model": "linear", "beta": beta_lin, "r2": r2_lin, "r2_adj": r2_adj_lin})

    # 3. 幂律衰减: y = y0 * (1+x)^(-k)  →  log(y/y0) = -k * log(1+x)
    log_xp1 = np.log(x.astype(float) + 1.0)
    slope_pow, r2_pow = _ols(log_xp1, log_ratios)
    k = -slope_pow
    beta_pow = 1.0 - 2.0 ** (-k)  # 近似转换: k→β，使 2-gen 后残余匹配
    beta_pow = max(min(beta_pow, 0.55), 0.001)
    r2_adj_pow = 1 - (1 - r2_pow) * (n - 1) / max(n - 2, 1)
    models.append({"model": "power_law", "beta": beta_pow, "r2": r2_pow, "r2_adj": r2_adj_pow})

    return models


def _ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """普通最小二乘。返回 (slope, r_squared)。"""
    x_mean = x.mean()
    y_mean = y.mean()
    ss_xx = np.sum((x - x_mean) ** 2)
    if ss_xx < 1e-10:
        return 0.0, 0.0
    slope = np.sum((x - x_mean) * (y - y_mean)) / ss_xx
    y_pred = x_mean * slope + (y_mean - x_mean * slope) + x * 0  # 简化
    y_pred = slope * x + (y_mean - slope * x_mean)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y_mean) ** 2)
    r2 = 1 - ss_res / max(ss_tot, 1e-10)
    return float(slope), float(max(r2, 0.0))


def bootstrap_beta_ci(
    snapshots: list[ConstraintFieldSnapshot],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
) -> dict:
    """Bootstrap 重采样估计 β 的置信区间。

    对代内样本进行重采样，重新计算 Π 和 β，得到 β 的经验分布。
    """
    if len(snapshots) < 2:
        return {"beta": 0.25, "ci_lower": 0.0, "ci_upper": 1.0, "n_bootstrap": 0}

    betas = []
    rng = np.random.RandomState(42)
    n_gens = len(snapshots)

    for _ in range(n_bootstrap):
        resampled = []
        for s in snapshots:
            n_states = len(s.states)
            if n_states < 2:
                resampled.append(s)
                continue
            idx = rng.choice(n_states, size=n_states, replace=True)
            # 重构 states
            from copy import deepcopy
            new_s = deepcopy(s)
            new_s.states = [s.states[i] for i in idx]
            # 重新计算 snapshot
            new_s = _recompute_snapshot(new_s)
            resampled.append(new_s)

        beta_dict = calibrate_beta_from_data(resampled)
        betas.append(beta_dict.get("*", 0.25))

    betas = np.array(betas)
    beta_mean = float(np.mean(betas))
    alpha = (1 - ci) / 2
    ci_lower = float(np.percentile(betas, alpha * 100))
    ci_upper = float(np.percentile(betas, (1 - alpha) * 100))

    return {
        "beta": beta_mean,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci_width": ci_upper - ci_lower,
        "n_bootstrap": n_bootstrap,
    }


def _recompute_snapshot(snapshot: ConstraintFieldSnapshot) -> ConstraintFieldSnapshot:
    """从 states 重新计算快照的 Π 和 σ 均值。"""
    from synthetic_decay_monitor.constraint_extractor import _compute_snapshot
    return _compute_snapshot(
        snapshot.states,
        snapshot.generation,
        snapshot.capability,
    )


def sigma_correlation_matrix(
    snapshots: list[ConstraintFieldSnapshot],
) -> dict:
    """计算 5 个 σ 维度间的跨代相关性矩阵。

    验证各维度的独立性（正交性）。高相关 → 维度冗余。
    """
    dims = ["fact", "syntax", "style", "safety", "coherence"]
    n = len(dims)

    # 聚合所有代的 σ 均值
    all_sigmas = {d: [] for d in dims}
    for s in snapshots:
        for d in dims:
            all_sigmas[d].append(s.individual_sigmas.get(d, 0.5))

    corr = np.zeros((n, n))
    for i, d1 in enumerate(dims):
        for j, d2 in enumerate(dims):
            if i == j:
                corr[i][j] = 1.0
            elif len(all_sigmas[d1]) >= 3:
                c = np.corrcoef(all_sigmas[d1], all_sigmas[d2])[0, 1]
                corr[i][j] = 0.0 if np.isnan(c) else float(c)
            else:
                corr[i][j] = 0.0

    # 冗余度量：平均非对角相关系数
    off_diag = []
    for i in range(n):
        for j in range(i + 1, n):
            off_diag.append(abs(corr[i][j]))
    mean_off_diag = float(np.mean(off_diag)) if off_diag else 0.0

    return {
        "dimensions": dims,
        "correlation_matrix": corr.tolist(),
        "mean_off_diagonal": mean_off_diag,
        "dimensionality_score": 1.0 - mean_off_diag,
        "interpretation": (
            "well_separated" if mean_off_diag < 0.3 else
            "moderately_correlated" if mean_off_diag < 0.5 else
            "highly_redundant"
        ),
    }


def validate_decay_model(
    snapshots: list[ConstraintFieldSnapshot],
) -> dict:
    """全面验证衰减模型：多模型拟合 + bootstrap + σ 相关性。

    这是对外暴露的入口函数，一次调用返回所有诊断。
    """
    result = {}

    # 1. 多模型 β 拟合
    beta_result = calibrate_beta_from_data(snapshots)
    result.update(beta_result)

    # 2. Bootstrap 置信区间
    ci_result = bootstrap_beta_ci(snapshots, n_bootstrap=500)
    result["bootstrap"] = ci_result

    # 3. σ 维度独立性
    corr_result = sigma_correlation_matrix(snapshots)
    result["sigma_correlation"] = corr_result

    # 4. 从 per-sample σ 均值的模拟合 β（直接质量度量）
    sigma_vec_mags = np.array([
        np.linalg.norm(list(s.individual_sigmas.values()))
        for s in snapshots
    ])
    generations = np.array([s.generation for s in snapshots])
    models = _fit_decay_models(generations, sigma_vec_mags)
    best_sigma = max(models, key=lambda m: m["r2_adj"])
    result["sigma_vec_beta"] = best_sigma["beta"]
    result["sigma_vec_r2"] = best_sigma["r2"]

    # 5. 验证结论
    r2_threshold = 0.5
    ci_ok = ci_result.get("ci_width", 1.0) < 0.3
    dim_ok = corr_result.get("dimensionality_score", 0) > 0.5
    result["verdict"] = {
        "decay_fit_acceptable": best_sigma["r2"] > r2_threshold,
        "ci_acceptable": ci_ok,
        "dimensions_independent": dim_ok,
        "overall": "pass" if (best_sigma["r2"] > r2_threshold and ci_ok and dim_ok) else "needs_review",
    }

    return result


def _composition_from_beta(beta: float) -> dict[str, float]:
    """从 β 反推执行者构成。

    β = α_EI * p_EI + α_EII * p_EII + α_EIII * p_EIII
    假设默认分布 p = [0.33, 0.33, 0.34]，线性缩放至目标 β。
    """
    alphas = BASE_ALPHAS
    if beta < 0.01:
        return {"E-I": 0.33, "E-II": 0.33, "E-III": 0.34}
    baseline = sum(alphas[k] * 0.33 for k in ["E-I", "E-II", "E-III"])
    scale = beta / baseline
    raw = {k: 0.33 * scale for k in ["E-I", "E-II", "E-III"]}
    total = sum(raw.values())
    if total < 1e-8:
        return {"E-I": 0.33, "E-II": 0.33, "E-III": 0.34}
    return {k: v / total for k, v in raw.items()}


def estimate_executor_composition(
    snapshot: ConstraintFieldSnapshot,
    prev_snapshot: ConstraintFieldSnapshot = None,
) -> dict[str, float]:
    """从约束场快照估计执行者类型构成。

    使用跨代约束分量变化率 + 文本特征指纹区分执行者类型：
    - E-I: 公理级约束（logic_density 下降，syntax CV 偏离）
    - E-II: 标度级约束（bigram_repetition 上升，filler_ratio 上升）
    - E-III: 边界条件约束（proper_case 下降，number_integrity 下降）

    单代模式（prev_snapshot=None）：基于文本特征绝对水平（如可用）或 σ 值
    跨代模式：基于 σ + 文本特征的代间变化率
    """
    sigmas = snapshot.individual_sigmas
    stds = snapshot.sigma_stds
    tf = snapshot.text_features  # 文本特征聚合均值（如可用）
    has_tf = len(tf) >= 6  # 至少有 6 个特征键

    if prev_snapshot is not None:
        prev_sigmas = prev_snapshot.individual_sigmas
        prev_stds = prev_snapshot.sigma_stds
        prev_tf = prev_snapshot.text_features
        has_prev_tf = len(prev_tf) >= 6

        # 跨代退化信号：代间 σ 下降率
        def drop(prev_val, curr_val):
            if prev_val < 1e-6:
                return 0.0
            return max(0.0, 1.0 - curr_val / prev_val)

        def tf_drop(key, prev_dict, curr_dict):
            """文本特征的代间下降率。"""
            pv = prev_dict.get(key, 0.5)
            cv = curr_dict.get(key, 0.5)
            if pv < 1e-6:
                return 0.0
            return max(0.0, 1.0 - cv / pv)

        def tf_rise(key, prev_dict, curr_dict):
            """文本特征的代间上升率（越高越异常）。"""
            pv = prev_dict.get(key, 0.0)
            cv = curr_dict.get(key, 0.0)
            if cv < 1e-6:
                return 0.0
            if pv < 1e-6:
                return min(cv * 2, 1.0)
            return max(0.0, cv / pv - 1.0)

        # 跨代 σ 标准差增长率
        def std_rise(prev_std, curr_std):
            if prev_std < 1e-6:
                return min(curr_std * 5, 1.0)
            return max(0.0, (curr_std - prev_std) / max(prev_std, 0.02))

        # === E-I 信号：文本特征优先（logic_density 下降） ===
        if has_tf and has_prev_tf:
            # 文本特征驱动：E-I 核心指纹是逻辑连接词密度下降
            logic_drop = tf_drop("ei_logic_density", prev_tf, tf)
            # syntax CV 偏离度：仅当逻辑连接词确实在丢失时才计入（否则是正常文本波动）
            cv_curr = tf.get("ei_syntax_cv", 0.5)
            cv_deviation = abs(cv_curr - 0.6)
            cv_bonus = cv_deviation * 1.5 if logic_drop > 0.05 else 0.0
            ei_tf_signal = logic_drop * 4.5 + cv_bonus
        else:
            ei_tf_signal = 0.0

        # sigma 信号作为 fallback
        syntax_drop = drop(prev_sigmas.get("syntax", 0.5), sigmas.get("syntax", 0.5))
        coherence_drop = drop(prev_sigmas.get("coherence", 0.5), sigmas.get("coherence", 0.5))
        safety_drop = drop(prev_sigmas.get("safety", 0.5), sigmas.get("safety", 0.5))
        ei_sigma_signal = (syntax_drop * 0.5 + coherence_drop * 0.4 + safety_drop * 0.1) * 4

        # E-I：文本特征比 sigma 更可靠（sigma 维度共线无法区分退化类型）
        # 当文本特征 E-I 信号明确时优先使用，否则用 sigma
        if has_tf and has_prev_tf and ei_tf_signal > 0.3:
            ei_signal = ei_tf_signal  # E-I 指纹明确：只用文本特征
        elif has_tf and has_prev_tf:
            ei_signal = 0.4 * ei_tf_signal + 0.6 * ei_sigma_signal  # 信号弱时加 sigma
        else:
            ei_signal = ei_sigma_signal

        # === E-II 信号：文本特征 + sigma 混合（sigma 对 E-II 有一定区分力） ===
        if has_tf and has_prev_tf:
            bigram_rise = tf_rise("eii_bigram_repetition", prev_tf, tf)
            filler_rise = tf_rise("eii_filler_ratio", prev_tf, tf)
            unique_drop = tf_drop("eii_unique_word_ratio", prev_tf, tf)
            trunc_rise = tf_rise("eii_truncation_ratio", prev_tf, tf)
            eii_tf_signal = bigram_rise * 1.2 + filler_rise * 0.8 + unique_drop * 1.0 + trunc_rise * 0.6
        else:
            eii_tf_signal = 0.0

        style_drop = drop(prev_sigmas.get("style", 0.5), sigmas.get("style", 0.5))
        style_std_rise_val = std_rise(prev_stds.get("style", 0.05), stds.get("style", 0.05))
        syntax_std_rise_val = std_rise(prev_stds.get("syntax", 0.05), stds.get("syntax", 0.05))
        eii_sigma_signal = style_drop * 2.5 + style_std_rise_val * 0.6 + syntax_std_rise_val * 0.4

        if has_tf and has_prev_tf:
            eii_signal = 0.7 * eii_tf_signal + 0.3 * eii_sigma_signal
        else:
            eii_signal = eii_sigma_signal

        # === E-III 信号：文本特征 + sigma 混合（sigma fact 维度对 E-III 有区分力） ===
        if has_tf and has_prev_tf:
            proper_drop = tf_drop("eiii_proper_case_ratio", prev_tf, tf)
            num_drop = tf_drop("eiii_number_integrity", prev_tf, tf)
            # 若文本原本就没有专有名词（proper_case 接近 0），proper_drop 无意义
            if prev_tf.get("eiii_proper_case_ratio", 0.5) < 0.1:
                proper_drop = 0.0
            eiii_tf_signal = proper_drop * 3.0 + num_drop * 2.5
        else:
            eiii_tf_signal = 0.0

        fact_drop = drop(prev_sigmas.get("fact", 0.5), sigmas.get("fact", 0.5))
        fact_std_rise_val = std_rise(prev_stds.get("fact", 0.05), stds.get("fact", 0.05))
        eiii_sigma_signal = fact_drop * 3.0 + fact_std_rise_val * 0.7

        if has_tf and has_prev_tf:
            eiii_signal = 0.7 * eiii_tf_signal + 0.3 * eiii_sigma_signal
        else:
            eiii_signal = eiii_sigma_signal

    else:
        # 单代模式：文本特征绝对水平优先
        if has_tf:
            # E-I: logic_density 低 → 公理退化（基准 ~0.5-1.0 为正常）
            logic_d = tf.get("ei_logic_density", 0.5)
            cv = tf.get("ei_syntax_cv", 0.5)
            cv_dev = abs(cv - 0.6)
            ei_signal = max(0.0, 0.55 - logic_d) * 6.0 + cv_dev * 1.5

            # E-II: bigram_repetition 高 + filler_ratio 高 + unique_word 低
            bigram_rep = tf.get("eii_bigram_repetition", 0.0)
            filler = tf.get("eii_filler_ratio", 0.0)
            unique = tf.get("eii_unique_word_ratio", 0.5)
            trunc = tf.get("eii_truncation_ratio", 0.0)
            eii_signal = (bigram_rep * 2.0 + max(0.0, filler - 0.15) * 3.0 +
                          max(0.0, 0.7 - unique) * 2.0 + trunc * 1.0)

            # E-III: proper_case 低 + number_integrity 低 才触发
            # proper_case=0 可能意味着文本原本就没有专有名词（如数学文本），不应触发
            proper = tf.get("eiii_proper_case_ratio", 0.5)
            num_int = tf.get("eiii_number_integrity", 0.5)
            proper_signal = max(0.0, 0.3 - proper) if proper > 0.05 else 0.0
            eiii_signal = proper_signal * 5.0 + max(0.0, 0.4 - num_int) * 4.0
        else:
            # Fallback: sigma 值
            syntax_mean = sigmas.get("syntax", 0.5)
            style_mean = sigmas.get("style", 0.5)
            fact_mean = sigmas.get("fact", 0.5)
            coherence_std = stds.get("coherence", 0.1)
            style_std = stds.get("style", 0.1)
            fact_std = stds.get("fact", 0.1)

            ei_signal = syntax_mean * np.exp(-coherence_std * 2)
            eii_signal = (1 - style_mean) * 0.7 + style_std * 3
            eiii_signal = (1 - fact_mean) * 0.7 + fact_std * 3

    # 无退化守卫：所有信号都弱 → 无明确退化类型
    if max(ei_signal, eii_signal, eiii_signal) < 0.25:
        return {"E-I": 0.33, "E-II": 0.33, "E-III": 0.34}

    # 幂次归一化：放大信号间差异，使主导执行者类型凸显
    # 自适应 power：信号差异小 → 更高 power 以分离
    signals = [ei_signal, eii_signal, eiii_signal]
    signal_range = max(signals) - min(signals)
    power = 3.0 if signal_range < 1.5 else 2.0

    ei_pow = ei_signal ** power
    eii_pow = eii_signal ** power
    eiii_pow = eiii_signal ** power
    total = ei_pow + eii_pow + eiii_pow
    if total < 1e-8:
        return {"E-I": 0.33, "E-II": 0.33, "E-III": 0.34}

    return {
        "E-I": ei_pow / total,
        "E-II": eii_pow / total,
        "E-III": eiii_pow / total,
    }


class DecayEngine:
    """衰减引擎：管理多能力维度跨代衰减状态。

    用法:
        lineage = parse_lineage_from_jsonl("data.jsonl")
        extractor = EmbeddingConstraintExtractor()
        engine = DecayEngine(lineage, extractor)
        engine.run_all_capabilities()
        trajectory = engine.get_trajectory("math_reasoning")
        print(f"Collapse predicted at gen {trajectory['predicted_collapse_gen']}")
    """

    def __init__(self, lineage, extractor):
        self.lineage = lineage
        self.extractor = extractor
        self._trajectories: dict[str, list[CapabilityStability]] = {}
        self._snapshots: dict[str, list[ConstraintFieldSnapshot]] = {}

    def run_all_capabilities(self):
        for cap in self.lineage.capability_coverage:
            self.run_capability(cap)

    def run_capability(self, capability: str):
        snapshots = []
        for gen in range(self.lineage.n_generations):
            samples = self.lineage.samples_by_capability(capability, gen)
            if not samples:
                continue
            snapshot = self.extractor.compute_field(samples, capability=capability)
            snapshot.generation = gen
            snapshots.append(snapshot)

        self._snapshots[capability] = snapshots

        # β from multi-model decay fitting (exponential/linear/power-law, best R²)
        data_betas = calibrate_beta_from_data(snapshots)
        beta = data_betas.get("*", 0.25)

        # Store fit diagnostics on engine for external access
        fit_diag = {k: v for k, v in data_betas.items() if k.startswith("_")}
        if not hasattr(self, '_fit_diagnostics'):
            self._fit_diagnostics = {}
        self._fit_diagnostics[capability] = fit_diag

        trajectory = []
        S_n = 1.0
        for i, snap in enumerate(snapshots):
            gen = snap.generation
            exec_comp = estimate_executor_composition(
                snap, prev_snapshot=snapshots[i - 1] if i > 0 else None
            )

            if trajectory:
                S_n = trajectory[-1].S_n * (1 - beta)

            trajectory.append(CapabilityStability(
                capability=capability,
                generation=gen,
                S_n=S_n,
                beta=beta,
                executor_composition=exec_comp,
                constraint_magnitudes=[snap.pi_magnitude],
                cancellation_ratio=snap.cancellation_ratio,
                raw_sigmas=snap.individual_sigmas,
            ))

        self._trajectories[capability] = trajectory

    def get_trajectory(self, capability: str) -> dict:
        traj = self._trajectories.get(capability)
        if not traj:
            return {"capability": capability, "trajectory": [], "error": "not found"}
        fit_diag = self.get_fit_diagnostics().get(capability, {})
        return {
            "capability": capability,
            "trajectory": [
                {
                    "generation": t.generation,
                    "S_n": t.S_n,
                    "beta": t.beta,
                    "status": t.status,
                    "executor_composition": t.executor_composition,
                    "cancellation_ratio": t.cancellation_ratio,
                }
                for t in traj
            ],
            "predicted_collapse_gen": predict_collapse(traj),
            "current_status": traj[-1].status if traj else "unknown",
            "fit_model": fit_diag.get("_fit_model", "unknown"),
            "fit_r2": fit_diag.get("_fit_r2", 0.0),
            "fit_target": fit_diag.get("_fit_target", "unknown"),
        }

    def get_all_trajectories(self) -> list[dict]:
        return [self.get_trajectory(cap) for cap in self._trajectories]

    def get_collapse_order(self) -> list[dict]:
        """按预测崩溃代排序的能力维度列表。"""
        trajectories = self.get_all_trajectories()
        ranked = sorted(
            [t for t in trajectories if "error" not in t],
            key=lambda t: t["predicted_collapse_gen"],
        )
        return [
            {
                "capability": t["capability"],
                "predicted_collapse_gen": t["predicted_collapse_gen"],
                "current_S_n": t["trajectory"][-1]["S_n"] if t["trajectory"] else 0,
                "beta": (
                    t["trajectory"][-1]["beta"]
                    if t["trajectory"]
                    else 0
                ),
            }
            for t in ranked
        ]

    def get_fit_diagnostics(self) -> dict:
        """返回各能力维度的拟合诊断信息（拟合模型、R²、目标序列）。"""
        return getattr(self, '_fit_diagnostics', {})

    def validate_all_capabilities(self) -> dict:
        """对所有能力维度运行完整算法验证（多模型拟合 + bootstrap + σ 相关性）。"""
        result = {}
        for cap, snapshots in self._snapshots.items():
            result[cap] = validate_decay_model(snapshots)
        return result


def predict_collapse(trajectory: list[CapabilityStability]) -> int:
    """预测崩溃代数：S_n < S_CRITICAL 的第一代。

    如果当前轨迹尚未崩溃，用最后已知的 β 外推。
    """
    if not trajectory:
        return -1

    for t in trajectory:
        if t.is_collapsed:
            return t.generation

    last = trajectory[-1]
    if last.beta <= 0:
        return 999  # 不会崩溃

    S = last.S_n
    beta = last.beta
    gen = last.generation
    while S >= S_CRITICAL and gen < 100:
        S *= (1 - beta)
        gen += 1

    return gen if gen < 100 else -1


def simulate_decay(
    S0: float = 1.0,
    beta: float = 0.25,
    n_generations: int = 10,
    capability: str = "general",
) -> list[CapabilityStability]:
    """纯模拟：从 S0 开始，固定 β 衰减 n 代。"""
    trajectory = []
    S = S0
    for gen in range(n_generations):
        trajectory.append(CapabilityStability(
            capability=capability,
            generation=gen,
            S_n=S,
            beta=beta,
        ))
        S *= (1 - beta)
    return trajectory
