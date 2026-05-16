"""
执行者退化分类器 — 亥姆霍兹分解 + 退化类型诊断。

对衰减的约束场做 Helmholtz 分解 Π = -∇φ + curl(A)：
- 非零 ∇φ（标量势残差）→ E-II/E-III 退化 → 需要标度/边界数据
- 非零 curl(A)（向量势残差）→ E-I 退化 → 需要公理数据

低维实现用 SVD/PCA 作为 Helmholtz 分解的离散近似。
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from synthetic_decay_monitor.constraint_extractor import (
    ConstraintFieldSnapshot, ConstraintState,
)
from synthetic_decay_monitor.decay_engine import (
    CapabilityStability, estimate_executor_composition,
)

# 约束拓扑先验：能力维度 → 主导执行者类型（框架理论推导）
CAPABILITY_EXECUTOR_PRIOR = {
    "math_reasoning": "E-I",
    "code_generation": "E-I",
    "logical_consistency": "E-I",
    "style_diversity": "E-II",
    "creative_writing": "E-II",
    "instruction_following": "E-II",
    "translation": "E-II",
    "summarization": "E-II",
    "general": "E-II",
    "factual_knowledge": "E-III",
    "safety_alignment": "E-III",
}


@dataclass
class DegradationDiagnosis:
    capability: str
    generation: int
    degradation_type: str = ""              # 'E-I_loss', 'E-II_loss', 'E-III_loss', 'mixed', 'none'
    helmholtz_scalar_potential: float = 0.0  # ∇φ 分量大小
    helmholtz_vector_potential: float = 0.0  # curl(A) 分量大小
    intervention_type: str = ""             # 'add_axiom_data' / 'add_calibration_data' / 'add_boundary_data'
    recommended_data_sources: list[str] = field(default_factory=list)
    confidence: float = 0.0
    severity: float = 0.0                   # 0-1，退化严重程度
    description: str = ""

    @property
    def is_critical(self) -> bool:
        return self.severity > 0.7


class ExecutorClassifier:
    """对跨代约束场做 Helmholtz 分解，诊断执行者退化类型。

    用法:
        classifier = ExecutorClassifier()
        diagnosis = classifier.classify_degradation(snapshots, capability="math_reasoning")
        print(diagnosis.intervention_type)  # → 'add_axiom_data'
    """

    def __init__(self):
        pass

    def classify_degradation(
        self,
        snapshots: list[ConstraintFieldSnapshot],
        capability: str = "",
    ) -> DegradationDiagnosis:
        if len(snapshots) < 2:
            return DegradationDiagnosis(
                capability=capability,
                generation=0,
                degradation_type="none",
                description="Insufficient data — need at least 2 generations",
            )

        latest = snapshots[-1]
        generation = latest.generation

        # 1. 构建约束梯度矩阵（样本间的约束变化）
        grad_matrix = self._build_gradient_matrix(snapshots)

        if grad_matrix is None or grad_matrix.shape[0] < 2:
            return DegradationDiagnosis(
                capability=capability,
                generation=generation,
                degradation_type="none",
                description="Cannot build gradient matrix",
            )

        # 2. Helmholtz 分解：SVD → 确定旋度 (非对称) 和散度 (对称) 分量
        helmholtz = self._helmholtz_decompose(grad_matrix)

        # 3. 判断退化类型
        curl_mag = helmholtz["curl_magnitude"]
        grad_mag = helmholtz["gradient_magnitude"]
        total_mag = curl_mag + grad_mag

        if total_mag < 1e-8:
            return DegradationDiagnosis(
                capability=capability,
                generation=generation,
                degradation_type="none",
                description="No significant degradation detected",
            )

        curl_ratio = curl_mag / total_mag
        grad_ratio = grad_mag / total_mag

        if curl_ratio > 0.55:
            dtype = "E-I_loss"
            intervention = "add_axiom_data"
            data_sources = [
                "formal proofs and derivations",
                "mathematical reasoning chains",
                "first-principles explanations",
                "expert-verified logical deductions",
            ]
            desc = "E-I executor degradation: theorem-level constraints weakening. Need axiom-dense data."
        elif curl_ratio > 0.30:
            dtype = "mixed"
            intervention = "add_mixed_data"
            data_sources = [
                "human-annotated calibration pairs",
                "diverse reasoning paths with verification",
                "expert demonstrations across domains",
            ]
            desc = "Mixed degradation: both E-I and E-II/E-III affected. Need diverse, verified data."
        elif grad_ratio > 0.70:
            dtype = "E-II_loss"
            intervention = "add_calibration_data"
            data_sources = [
                "human preference comparisons",
                "calibrated scoring data",
                "boundary case annotations",
                "diverse style exemplars",
            ]
            desc = "E-II executor degradation: scale/calibration constraints weakening. Need human-annotated comparisons."
        else:
            dtype = "E-III_loss"
            intervention = "add_boundary_data"
            data_sources = [
                "edge case examples",
                "domain-specific boundary data",
                "rare scenario demonstrations",
                "diverse context variations",
            ]
            desc = "E-III executor degradation: boundary condition constraints weakening. Need edge-case data."

        severity = float(np.clip(total_mag / max(total_mag + 1, 1e-8), 0, 1))

        return DegradationDiagnosis(
            capability=capability,
            generation=generation,
            degradation_type=dtype,
            helmholtz_scalar_potential=grad_mag,
            helmholtz_vector_potential=curl_mag,
            intervention_type=intervention,
            recommended_data_sources=data_sources,
            severity=severity,
            confidence=min(0.5 + 0.3 * total_mag, 1.0),
            description=desc,
        )

    def classify_all(
        self,
        snapshots_by_capability: dict[str, list[ConstraintFieldSnapshot]],
    ) -> list[DegradationDiagnosis]:
        return [
            self.classify_degradation(snaps, cap)
            for cap, snaps in snapshots_by_capability.items()
        ]

    @staticmethod
    def _build_gradient_matrix(snapshots) -> Optional[np.ndarray]:
        """构建跨代约束梯度矩阵：每行是一个代的约束状态向量，列是约束维度。"""
        rows = []
        for snap in snapshots:
            sigmas = snap.individual_sigmas
            row = [
                sigmas.get("fact", 0.5),
                sigmas.get("syntax", 0.5),
                sigmas.get("style", 0.5),
                sigmas.get("safety", 0.5),
                sigmas.get("coherence", 0.5),
            ]
            rows.append(row)
        if len(rows) < 2:
            return None
        return np.array(rows)

    @staticmethod
    def _helmholtz_decompose(grad_matrix: np.ndarray) -> dict:
        """离散 Helmholtz 分解。

        - 梯度（保守）分量 = rank-1 SVD 重建 → 单调趋势（∇φ）
        - 旋度分量 = 原始矩阵 - rank-1 重建 → 非单调变化（curl(A)）

        对于方阵，直接用对称/反对称分解更精确。
        """
        m, n = grad_matrix.shape
        if m == n and m > 1:
            sym = (grad_matrix + grad_matrix.T) / 2
            skew = grad_matrix - grad_matrix.T
            grad_mag = float(np.linalg.norm(sym))
            curl_mag = float(np.linalg.norm(skew))
        elif m >= 2 and n >= 2:
            U, s, Vt = np.linalg.svd(grad_matrix, full_matrices=False)
            rank1 = np.outer(U[:, 0], Vt[0, :]) * s[0]
            grad_part = rank1
            curl_part = grad_matrix - rank1
            grad_mag = float(np.linalg.norm(grad_part))
            curl_mag = float(np.linalg.norm(curl_part))
        else:
            grad_mag = float(np.linalg.norm(grad_matrix))
            curl_mag = 0.0

        return {
            "gradient_magnitude": grad_mag,
            "curl_magnitude": curl_mag,
            "curl_ratio": curl_mag / (grad_mag + curl_mag + 1e-10),
            "symmetric_norm": grad_mag,
            "skew_norm": curl_mag,
        }


def diagnose_executor_decay(
    trajectory: list[CapabilityStability],
    snapshots: list[ConstraintFieldSnapshot],
    capability: str = "",
) -> dict:
    """Generate diagnosis from decay trajectory + constraint snapshots.

    Decision logic (simple priority chain):
    1. Cross-gen text feature deltas → dominant degradation type
    2. Helmholtz decomposition → fallback when text features unavailable
    """
    classifier = ExecutorClassifier()
    helmholtz_diag = classifier.classify_degradation(snapshots, capability)
    current = trajectory[-1] if trajectory else None
    comp = current.executor_composition if current else {}

    # Primary: cross-generation text feature deltas
    has_tf = all(len(getattr(s, 'text_features', {})) >= 6 for s in snapshots)
    final_dtype = _classify_from_text_deltas(snapshots) if has_tf and len(snapshots) >= 2 else None

    # Fallback: Helmholtz decomposition
    if final_dtype is None:
        final_dtype = helmholtz_diag.degradation_type
        if final_dtype == "none":
            final_dtype = "mixed"

    diagnosis = DegradationDiagnosis(
        capability=helmholtz_diag.capability,
        generation=helmholtz_diag.generation,
        degradation_type=final_dtype,
        helmholtz_scalar_potential=helmholtz_diag.helmholtz_scalar_potential,
        helmholtz_vector_potential=helmholtz_diag.helmholtz_vector_potential,
        intervention_type=_intervention_for_type(final_dtype),
        recommended_data_sources=_data_sources_for_type(final_dtype),
        severity=helmholtz_diag.severity,
        confidence=0.7 if has_tf else 0.5,
        description=_description_for_type(final_dtype),
    )

    dark_zone_detected = any(
        snap.cancellation_ratio < 0.1 and snap.total_constraint > 1.0
        for snap in snapshots
    )

    return {
        "capability": capability,
        "diagnosis": diagnosis,
        "current_stability": current.S_n if current else 1.0,
        "current_beta": current.beta if current else 0.25,
        "current_status": current.status if current else "healthy",
        "executor_composition": comp,
        "dark_zone_detected": dark_zone_detected,
        "intervention_urgency": (
            "immediate" if diagnosis.severity > 0.7 or dark_zone_detected
            else "soon" if diagnosis.severity > 0.4 else "monitor"
        ),
        "summary": _generate_summary(diagnosis, current, dark_zone_detected),
    }


def _classify_from_text_deltas(snapshots: list[ConstraintFieldSnapshot]) -> str | None:
    """Classify degradation type from cross-generation text feature changes.

    Returns degradation_type string, or None if no clear signal.
    """
    tf0 = snapshots[0].text_features
    tfN = snapshots[-1].text_features

    # E-I signal: logic_density drop + syntax_cv shift away from 0.6
    logic_drop = max(0.0, tf0.get("ei_logic_density", 0.5) - tfN.get("ei_logic_density", 0.5))
    cv_shift = abs(tfN.get("ei_syntax_cv", 0.5) - 0.6) - abs(tf0.get("ei_syntax_cv", 0.5) - 0.6)
    ei_signal = logic_drop * 1.5 + max(0, cv_shift) * 0.5

    # E-II signal: bigram_rep rise + filler rise + unique_word drop
    eii_signal = (
        max(0.0, tfN.get("eii_bigram_repetition", 0.0) - tf0.get("eii_bigram_repetition", 0.0)) * 2.0 +
        max(0.0, tfN.get("eii_filler_ratio", 0.0) - tf0.get("eii_filler_ratio", 0.0)) * 1.5 +
        max(0.0, tf0.get("eii_unique_word_ratio", 0.5) - tfN.get("eii_unique_word_ratio", 0.5)) * 1.5
    )

    # E-III signal: proper_case drop + number_integrity drop
    eiii_signal = (
        max(0.0, tf0.get("eiii_proper_case_ratio", 0.5) - tfN.get("eiii_proper_case_ratio", 0.5)) * 2.0 +
        max(0.0, tf0.get("eiii_number_integrity", 0.5) - tfN.get("eiii_number_integrity", 0.5)) * 1.5
    )

    # Require dominant signal to be at least 2x the runner-up
    signals = {"E-I_loss": ei_signal, "E-II_loss": eii_signal, "E-III_loss": eiii_signal}
    ranked = sorted(signals.items(), key=lambda x: -x[1])
    winner, winner_val = ranked[0]
    runner_up_val = ranked[1][1]

    if winner_val > 0.1 and winner_val > runner_up_val * 1.5:
        return winner
    return None


def _intervention_for_type(dtype: str) -> str:
    return {
        "E-I_loss": "add_axiom_data",
        "E-II_loss": "add_calibration_data",
        "E-III_loss": "add_boundary_data",
        "mixed": "add_mixed_data",
    }.get(dtype, "monitor")


def _data_sources_for_type(dtype: str) -> list[str]:
    return {
        "E-I_loss": [
            "formal proofs and derivations",
            "mathematical reasoning chains",
            "first-principles explanations",
        ],
        "E-II_loss": [
            "human preference comparisons",
            "calibrated scoring data",
            "diverse style exemplars",
        ],
        "E-III_loss": [
            "edge case examples",
            "domain-specific boundary data",
            "rare scenario demonstrations",
        ],
        "mixed": [
            "human-annotated calibration pairs",
            "diverse reasoning paths with verification",
        ],
    }.get(dtype, ["general monitoring"])


def _description_for_type(dtype: str) -> str:
    return {
        "E-I_loss": "E-I executor degradation: theorem-level constraints weakening — fastest collapse pattern.",
        "E-II_loss": "E-II executor degradation: scale/calibration constraints weakening.",
        "E-III_loss": "E-III executor degradation: boundary condition constraints weakening — slowest but still concerning.",
        "mixed": "Mixed degradation: both E-I and E-II/E-III affected.",
        "none": "No significant degradation detected.",
    }.get(dtype, "Unknown degradation pattern.")


def _generate_summary(
    diagnosis: DegradationDiagnosis,
    current: Optional[CapabilityStability],
    dark_zone: bool,
) -> str:
    parts = []
    if diagnosis.degradation_type == "E-I_loss":
        parts.append("Primary: E-I executor loss. Axiom-level constraints degrading—fastest collapse pattern.")
    elif diagnosis.degradation_type == "E-II_loss":
        parts.append("Primary: E-II executor loss. Scale calibration degrading.")
    elif diagnosis.degradation_type == "E-III_loss":
        parts.append("Primary: E-III executor loss. Boundary conditions degrading—slowest but still concerning.")
    elif diagnosis.degradation_type == "mixed":
        parts.append("Primary: Mixed executor degradation across types.")

    if current:
        parts.append(f"Stability S_{current.generation} = {current.S_n:.3f}, β = {current.beta:.3f}.")

    if dark_zone:
        parts.append("WARNING: Type III dark zone detected—constraint cancellation masking degradation.")

    parts.append(f"Recommended: {diagnosis.intervention_type} → {', '.join(diagnosis.recommended_data_sources[:2])}.")

    return " ".join(parts)
