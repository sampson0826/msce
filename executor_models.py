"""
Executor models — distinct from Rule/ConstraintField in core.py.

Executor: transmits constraint BETWEEN layers (L_n → L_{n+1})
Rule: constrains WITHIN a layer

Key distinction from the framework:
  Derivation (L-1 → L0): gives existence, needs NO executor
  Constraint (L0 → L1 → L2 → L3): restricts allowed forms, MUST have executor
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class Executor:
    """A constraint transmission mechanism between two layers.

    Unlike Rule (which constrains within a layer), an Executor maps
    constraint from an upper layer to a lower layer.

    Attributes:
        id: Unique identifier (e.g. 'E1')
        name: Descriptive name
        from_layer: Source layer index (0=L0, 1=L1, etc.)
        to_layer: Target layer index
        executor_type: 'E-I' (math theorem), 'E-II' (scale hypothesis), 'E-III' (boundary)
        transmission_fn: Maps constraint vector in upper layer to constraint in lower layer.
            Signature: (upper_constraint: ndarray, position: ndarray) → lower_constraint: ndarray
        certainty: How certain this executor is (0-1)
        evidence_strength: 'high' / 'medium' / 'low' / 'speculative'
        math_form: Mathematical description of the transmission
    """
    id: str
    name: str
    from_layer: int
    to_layer: int
    executor_type: str  # 'E-I', 'E-II', 'E-III'
    transmission_fn: Callable[[np.ndarray, np.ndarray], np.ndarray]
    certainty: float = 1.0
    evidence_strength: str = 'high'
    math_form: str = ''

    def transmit(self, upper_constraint: np.ndarray, position: np.ndarray) -> np.ndarray:
        """Apply transmission: upper-layer constraint → lower-layer constraint."""
        return self.transmission_fn(upper_constraint, position)

    @property
    def is_mathematically_necessary(self) -> bool:
        return self.executor_type == 'E-I'

    @property
    def needs_empirical_calibration(self) -> bool:
        return self.executor_type in ('E-II', 'E-III')


@dataclass
class ExecutorGap:
    """A detected gap in the executor network — a missing transmission mechanism.

    Represents where constraint from an upper layer reaches a lower layer
    incompletely, pointing to an unknown executor.
    """
    from_layer: int
    to_layer: int
    region: np.ndarray  # centroid of the gap in state space
    region_bounds: list[tuple[float, float]]  # extent of the gap

    # Transmission completeness: ratio of (received constraint) / (sent constraint)
    transmission_completeness: float

    # Residual characteristics
    residual_magnitude: float
    residual_direction: np.ndarray
    n_residual_points: int = 0

    # Predicted properties of the missing executor
    candidate_type: str = ''  # 'E-I' / 'E-II' / 'E-III'
    candidate_type_confidence: float = 0.0
    candidate_math_form: str = ''
    candidate_constraint_direction: Optional[np.ndarray] = None

    # Associated dark zones (if any)
    dark_zone_ids: list[int] = field(default_factory=list)

    # Priority: higher = more likely to be a real missing executor
    priority: float = 0.0

    def summary(self) -> str:
        lines = [
            f"ExecutorGap: L{self.from_layer} → L{self.to_layer}",
            f"  Transmission completeness: {self.transmission_completeness:.3f}",
            f"  Residual magnitude: {self.residual_magnitude:.4f}",
            f"  Residual points: {self.n_residual_points}",
            f"  Candidate type: {self.candidate_type or 'unknown'}",
        ]
        if self.candidate_math_form:
            lines.append(f"  Candidate math: {self.candidate_math_form}")
        if self.dark_zone_ids:
            lines.append(f"  Associated dark zones: {self.dark_zone_ids}")
        return "\n".join(lines)


@dataclass
class DarkZoneRegion:
    """A Type III invisibility region — perfect cross-constraint balance.

    In a dark zone, multiple known constraints cancel exactly:
        Σ∇σ = 0 while each ||∇σ|| >> 0

    This produces zero observable signal despite strong rules being active.
    Unknown rules can hide here undetected.
    """
    id: int
    centroid: np.ndarray
    bounds: list[tuple[float, float]]
    n_points: int = 0

    # Which rules are involved in the balance
    constraints_involved: list[str] = field(default_factory=list)

    # Cancellation ratio: ||Σ∇σ|| / Σ||∇σ|| — closer to 0 = more perfect balance
    cancellation_ratio: float = 0.0

    # Individual constraint magnitudes at centroid
    individual_magnitudes: dict[str, float] = field(default_factory=dict)

    # Topology classification
    balance_topology: str = ''  # 'mutual_cancellation' / 'cyclic' / 'hierarchical'
    balance_topology_confidence: float = 0.0

    # How to break this dark zone
    break_conditions: list[dict] = field(default_factory=list)

    # What the dark zone obscures
    suspected_hidden_content: str = ''

    def is_deeper_than(self, other: 'DarkZoneRegion') -> bool:
        """Compare which dark zone is more perfectly hidden."""
        return self.cancellation_ratio < other.cancellation_ratio

    def summary(self) -> str:
        lines = [
            f"DarkZone #{self.id}: {self.balance_topology or 'unclassified'}",
            f"  Constraints: {self.constraints_involved}",
            f"  Cancellation ratio: {self.cancellation_ratio:.6f}",
            f"  Points: {self.n_points}",
        ]
        for bc in self.break_conditions[:3]:
            lines.append(f"  Break: {bc.get('condition', '?')} "
                         f"→ predicted ||Π|| = {bc.get('predicted_magnitude', 0):.3f}")
        return "\n".join(lines)


@dataclass
class TransmissionResult:
    """Result of testing constraint transmission across a layer boundary.

    Takes known upper-layer constraints and measures how much reaches
    the lower layer through known executors.
    """
    from_layer: int
    to_layer: int
    known_executors: list[str]  # executor IDs active in this transition

    # Sampling statistics
    n_points_sampled: int = 0
    n_points_complete: int = 0  # where transmission >= threshold

    # Transmission completeness distribution
    completeness_mean: float = 0.0
    completeness_min: float = 0.0
    completeness_std: float = 0.0

    # Residual statistics over the region
    residual_magnitudes: list[float] = field(default_factory=list)

    @property
    def has_gap(self) -> bool:
        """True if transmission is incomplete somewhere in the region."""
        return self.completeness_min < 0.95 or self.completeness_mean < 0.98

    @property
    def gap_severity(self) -> float:
        """0 = no gap, 1 = complete transmission failure."""
        return 1.0 - self.completeness_mean


@dataclass
class ExperimentProposal:
    """A proposed experiment to expose a hidden executor."""
    id: str
    gap: ExecutorGap
    experiment_type: str  # 'extreme_energy' / 'extreme_density' / 'extreme_curvature' / 'precision' / 'interference'
    target_observable: str
    target_region: np.ndarray  # where in state space to look
    required_precision: float  # minimum measurement precision (relative to noise)
    predicted_signal_strength: float  # expected ||Π|| under experiment conditions
    feasibility: float  # 0-1 based on current/near-future technology
    discovery_potential: float  # 0-1 impact if confirmed
    rationale: str = ''
    priority_score: float = 0.0

    def summary(self) -> str:
        return (
            f"Experiment {self.id}: {self.experiment_type} → {self.target_observable}\n"
            f"  Precision needed: {self.required_precision:.1e}\n"
            f"  Predicted signal: {self.predicted_signal_strength:.4f}\n"
            f"  Feasibility: {self.feasibility:.2f} | Discovery potential: {self.discovery_potential:.2f}\n"
            f"  Priority: {self.priority_score:.3f}"
        )
