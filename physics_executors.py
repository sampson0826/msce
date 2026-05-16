"""
Known physics executors (E₁-E₇) instantiated as Executor objects.

Each executor transmits constraint between specific layers:
  E₁-E₅, E₇: L0 → L1 (transmit fundamental symmetries to dynamics equations)
  E₆: L1 → L2 (transmit fundamental dynamics to effective theories)

Missing executors (M₁-M₃) are also defined here as gaps with known residual signals.
"""

import numpy as np
from constraint_residual.executor_models import Executor, ExecutorGap


# ============================================================
# E₁-E₇: Known physics executors
# ============================================================

def build_known_executors() -> list[Executor]:
    """Build the 7 known physics executors as Executor instances.

    Each transmission_fn is a simplified numerical representation of the
    full mathematical structure. For real physics calculations, these
    would be replaced with proper gauge field / Hilbert space computations.

    The key design: transmission_fn(upper_constraint, position) → lower_constraint
    where upper_constraint is a vector representing the constraint force from
    the layer above, and position is the state-space coordinate.
    """

    executors = []

    # E₁: Covariant derivative enforcement
    # Local gauge symmetry U(1)×SU(2)×SU(3) → Dμ = ∂μ + igAμ
    # This forces ALL fundamental interactions to be mediated by gauge fields.
    executors.append(Executor(
        id='E1',
        name='协变导数强制律',
        from_layer=0, to_layer=1,
        executor_type='E-I',
        certainty=1.0,
        evidence_strength='high',
        math_form='Dμ = ∂μ + igAμ, 局域对称 → 联络场 → 规范相互作用',
        transmission_fn=lambda uc, p: _transmit_gauge(uc, p)
    ))

    # E₂: Noether conservation mapping
    # Continuous symmetry generator → conserved current → conserved charge
    executors.append(Executor(
        id='E2',
        name='Noether守恒映射',
        from_layer=0, to_layer=1,
        executor_type='E-I',
        certainty=1.0,
        evidence_strength='high',
        math_form='∂μ j^μ = 0, 连续对称生成元 → 守恒流 → 守恒荷',
        transmission_fn=lambda uc, p: _transmit_noether(uc, p)
    ))

    # E₃: Stone unitary evolution enforcement
    # One-parameter strongly continuous unitary group → Hamiltonian → Schrödinger equation form
    executors.append(Executor(
        id='E3',
        name='Stone酉演化强制',
        from_layer=0, to_layer=1,
        executor_type='E-I',
        certainty=1.0,
        evidence_strength='high',
        math_form='U(t) = exp(-iHt/ħ), 单参强连续酉群 → 哈密顿量 → 薛定谔方程',
        transmission_fn=lambda uc, p: _transmit_unitary(uc, p)
    ))

    # E₄: Antisymmetrization enforcement (Pauli exclusion)
    # Half-integer spin → anticommutation → Pauli exclusion principle
    executors.append(Executor(
        id='E4',
        name='反对称化强制律',
        from_layer=0, to_layer=1,
        executor_type='E-I',
        certainty=1.0,
        evidence_strength='high',
        math_form='{ψ(x), ψ†(y)} = δ(x-y), 半整数自旋 → 反对易 → 泡利不相容',
        transmission_fn=lambda uc, p: _transmit_pauli(uc, p)
    ))

    # E₅: Microcausality enforcement
    # Finite c → spacelike-separated operators commute → Lagrangian form restricted
    executors.append(Executor(
        id='E5',
        name='微观因果性强制',
        from_layer=0, to_layer=1,
        executor_type='E-I',
        certainty=1.0,
        evidence_strength='high',
        math_form='[O(x), O(y)] = 0 for (x-y)² > 0, c有限 → 类空间隔对易',
        transmission_fn=lambda uc, p: _transmit_causality(uc, p)
    ))

    # E₆: Scale separation law
    # Characteristic scale hierarchy separation → effective theory
    # E-II because the fixed point positions depend on empirical input (Λ_QCD etc.)
    executors.append(Executor(
        id='E6',
        name='标度分离律',
        from_layer=1, to_layer=2,
        executor_type='E-II',
        certainty=0.85,
        evidence_strength='high',
        math_form='RG流 → Λ_high/Λ_low ≫ 1 → 有效理论约化',
        transmission_fn=lambda uc, p: _transmit_scale_separation(uc, p)
    ))

    # E₇: Decoherence selection law
    # Open system + environment entanglement → density matrix off-diagonal exponential decay
    executors.append(Executor(
        id='E7',
        name='退相干选择律',
        from_layer=0, to_layer=1,
        executor_type='E-I',
        certainty=0.95,
        evidence_strength='high',
        math_form='ρ_ij(t) → 0 exponentially, 开放系统 → 环境纠缠 → 经典世界涌现',
        transmission_fn=lambda uc, p: _transmit_decoherence(uc, p)
    ))

    return executors


# ============================================================
# Simplified transmission functions
# ============================================================

def _transmit_gauge(uc: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Gauge symmetry → constrained interaction form.

    In reality: the gauge group determines the connection form Dμ = ∂μ + igAμ.
    Here: the constraint is strengthened along directions that break gauge invariance.
    """
    # Gauge constraint is maximally strong (symmetry enforcement)
    # Strength modulated by the effective coupling at position p
    coupling = 1.0 + 0.1 * np.sin(np.linalg.norm(p))
    direction = -uc / (np.linalg.norm(uc) + 1e-10)  # Restorative
    return coupling * direction * np.linalg.norm(uc)


def _transmit_noether(uc: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Symmetry → conservation law.

    The conserved current j^μ is determined by the symmetry generator.
    Transmission maps symmetry constraints to conserved quantities.
    """
    # Strength decays slightly with distance from symmetry center
    decay = np.exp(-0.01 * np.linalg.norm(p))
    return decay * uc  # Direct transmission: symmetry → conservation


def _transmit_unitary(uc: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Unitary evolution → Hamiltonian dynamics.

    Stone's theorem: one-parameter unitary group ↔ self-adjoint Hamiltonian.
    """
    # Phase rigidity: constraints on quantum evolution
    # The Hamiltonian must be self-adjoint
    return 0.99 * uc  # Near-perfect transmission


def _transmit_pauli(uc: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Spin-statistics → Pauli exclusion.

    Half-integer spin → antisymmetric wavefunction.
    Transmission amplifies constraint for fermionic degrees of freedom.
    """
    # Spin-statistics connection is a topological constraint
    # Amplification factor depends on the spin structure at p
    spin_factor = 1.0
    for i, xi in enumerate(p[:3]):
        spin_factor *= (1.0 + 0.05 * np.sin(xi * np.pi))
    return spin_factor * uc


def _transmit_causality(uc: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Microcausality → Lagrangian form constraints.

    Spacelike commutativity restricts which terms can appear in the Lagrangian.
    """
    # Causality constraint is absolute within the light cone
    # Decays sharply outside physically relevant parameters
    r = np.linalg.norm(p[:3])  # spatial extent
    t = p[3] if len(p) > 3 else 0.0  # time coordinate
    if r <= abs(t) + 1e-6:
        return uc  # Inside light cone: full transmission
    else:
        return np.zeros_like(uc)  # Outside: no constraint (spacelike)


def _transmit_scale_separation(uc: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Scale separation → effective theory reduction.

    E-II: the RG flow direction is mathematically determined,
    but the fixed point positions depend on empirical scales.
    """
    # Transmission efficiency depends on scale ratio
    scale_ratio = np.exp(-np.linalg.norm(p) * 0.1)
    # E-II: partial transmission with empirical uncertainty
    transmission = 0.7 * scale_ratio + 0.15 * np.random.RandomState(
        int(np.linalg.norm(p) * 1000) % (2**31)
    ).normal()
    return transmission * uc


def _transmit_decoherence(uc: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Environment entanglement → classicality.

    Open system dynamics: off-diagonal elements decay exponentially.
    """
    # Decoherence rate depends on system-environment coupling strength
    coupling = np.linalg.norm(p[:3]) + 1e-6
    # Larger coupling → faster decoherence → stronger transmission
    rate = 1.0 - np.exp(-coupling)
    return rate * uc


# ============================================================
# M₁-M₃: Missing executor gap definitions
# ============================================================

def build_missing_executor_gaps() -> list[ExecutorGap]:
    """Build descriptions of the 3 known missing executor gaps.

    These are not executors — they are the residual signals that
    point to where unknown executors should exist.
    """

    gaps = []

    # M₁: 19 free parameters have no first-principles explanation
    # Residual signal: MSSM gauge couplings nearly converge at GUT scale
    # Π_known(parameter_space) ≠ 0 → missing selection mechanism
    gaps.append(ExecutorGap(
        from_layer=0, to_layer=1,
        region=np.zeros(19),  # 19-dimensional parameter space
        region_bounds=[(-1, 1)] * 19,
        transmission_completeness=0.0,  # No known executor for parameter selection
        residual_magnitude=0.5,
        residual_direction=np.ones(19) / np.sqrt(19),
        n_residual_points=1,
        candidate_type='E-I',
        candidate_type_confidence=0.4,
        candidate_math_form='RG联合不动点选择：所有β函数在Planck尺度同时为零',
        candidate_constraint_direction=np.ones(19) / np.sqrt(19),
        priority=0.85,
    ))

    # M₂: GR diverges at Planck scale
    # Residual signal: non-renormalizable infinities in perturbative GR
    gaps.append(ExecutorGap(
        from_layer=0, to_layer=1,
        region=np.array([1.22e19]),  # Planck energy in GeV
        region_bounds=[(1e18, 1e20)],
        transmission_completeness=0.0,
        residual_magnitude=0.8,
        residual_direction=np.array([-1.0]),
        n_residual_points=1,
        candidate_type='E-I',
        candidate_type_confidence=0.35,
        candidate_math_form='量子引力统一：弦论/圈量子引力/渐近安全',
        candidate_constraint_direction=np.array([-1.0]),
        priority=0.90,
    ))

    # M₃: Type III dark zone — no direct residual signal
    # Complete invisibility: unknown rules in perfect cross-balance
    gaps.append(ExecutorGap(
        from_layer=0, to_layer=1,
        region=np.zeros(4),  # Unknown region in state space
        region_bounds=[(-1, 1)] * 4,
        transmission_completeness=1.0,  # Appears "complete" — the danger
        residual_magnitude=0.0,  # Zero residual — the defining feature
        residual_direction=np.zeros(4),
        n_residual_points=0,
        candidate_type='E-I',
        candidate_type_confidence=0.1,  # Most speculative
        candidate_math_form='未知物理规则在完美交叉平衡中 — 需要极端条件打破暗区',
        candidate_constraint_direction=None,
        dark_zone_ids=[0],
        priority=0.95,  # Highest priority because it's completely hidden
    ))

    return gaps


def get_executor_summary_table() -> str:
    """Generate a markdown summary table of all known + missing executors."""
    executors = build_known_executors()
    gaps = build_missing_executor_gaps()

    lines = [
        "| ID | 名称 | 类型 | 层级 | 确定性 | 证据 |",
        "|----|------|------|------|--------|------|",
    ]
    for e in executors:
        lines.append(
            f"| {e.id} | {e.name} | {e.executor_type} | "
            f"L{e.from_layer}→L{e.to_layer} | {e.certainty:.2f} | {e.evidence_strength} |"
        )
    lines.append("| | **缺失执行者** | | | | |")
    for gi, g in enumerate(gaps):
        e_type = g.candidate_type or '?'
        c = g.candidate_type_confidence
        lines.append(
            f"| M{gi+1} | {g.candidate_math_form[:40]}... | "
            f"{e_type}? | L{g.from_layer}→L{g.to_layer} | {c:.2f} | speculative |"
        )

    return "\n".join(lines)
