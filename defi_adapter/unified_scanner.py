"""
Unified Dark Zone Scanner — combines all three detection types into one pipeline.

Type I  (Physical Occlusion):  Oracle divergence masks constraint signals
Type II (Structural Occlusion): g^{-1} zero eigenvalues = unconstrained directions
Type III (Cancellation):       c(p) ≈ 0 with Σ||∇σ|| large = perfect cross-cancellation

Usage:
    scanner = UnifiedScanner("ProtocolName")
    for rule in rules:
        scanner.add_rule(rule)
    scanner.add_data_source(DataSource(...))  # optional, for Type I
    report = scanner.scan(bounds, n_points=60)
    print(report.summary())
    report.to_html("report.html")
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from constraint_residual.core import Rule, ConstraintField, ResidualDetector
from constraint_residual.dark_zone_detector import DarkZoneDetector, DarkZoneCluster
from constraint_residual.defi_adapter.type1_detector import (
    Type1Detector, Type1Candidate, DataSource,
)


@dataclass
class Type2Candidate:
    """A Type II structural occlusion region (unconstrained direction)."""
    position: np.ndarray
    min_eigenvalue: float
    max_eigenvalue: float
    condition_number: float
    unconstrained_direction: np.ndarray
    constrained_direction: np.ndarray


@dataclass
class UnifiedScanReport:
    """Complete dark zone analysis across all three types."""
    protocol_name: str
    bounds: list[tuple[float, float]]

    # Type III — Cancellation (c(p) based)
    type3_dark_zones: list[DarkZoneCluster] = field(default_factory=list)
    type3_count: int = 0

    # Type II — Structural occlusion (metric tensor eigenvalues)
    type2_candidates: list[Type2Candidate] = field(default_factory=list)
    type2_count: int = 0

    # Type I — Physical occlusion (oracle divergence)
    type1_candidates: list[Type1Candidate] = field(default_factory=list)
    type1_count: int = 0

    # General
    residual_points: int = 0
    constraint_names: list[str] = field(default_factory=list)
    n_state_dims: int = 0

    def summary(self) -> str:
        lines = [
            f"{'='*65}",
            f"Unified Dark Zone Scan: {self.protocol_name}",
            f"{'='*65}",
            f"State space: {self.n_state_dims}D, "
            f"Constraints: {len(self.constraint_names)}",
            f"",
            f"TYPE III (Cancellation): {self.type3_count} dark zones",
            f"TYPE II  (Structural):   {self.type2_count} unconstrained regions",
            f"TYPE I   (Occlusion):    {self.type1_count} oracle divergence zones",
            f"",
            f"Residual points: {self.residual_points}",
        ]

        if self.type3_dark_zones:
            lines.append(f"\n--- Type III Dark Zones ---")
            for dz in self.type3_dark_zones[:5]:
                lines.append(f"  [{dz.balance_topology}] c̄={dz.mean_cancellation_ratio:.4f}, "
                           f"{len(dz.points)} pts, {', '.join(dz.constraints_involved[:3])}")

        if self.type2_candidates:
            lines.append(f"\n--- Type II Unconstrained Directions ---")
            for c in self.type2_candidates[:5]:
                u_dir = c.unconstrained_direction
                dom = "leverage" if abs(u_dir[0]) > abs(u_dir[-1]) else "peg"
                lines.append(f"  pos=({c.position[0]:.2f}, {c.position[1]:.4f}), "
                           f"cond={c.condition_number:.0f}, direction={dom}")

        if self.type1_candidates:
            lines.append(f"\n--- Type I Oracle Divergence ---")
            for c in self.type1_candidates[:5]:
                lines.append(f"  occlusion={c.occlusion_score:.4f}, "
                           f"divergence={c.max_divergence_pct:.1f}%, "
                           f"sources: {', '.join(c.diverging_sources[:2])}")

        lines.append(f"\n{'='*65}")
        return "\n".join(lines)

    def overall_risk(self) -> str:
        """Qualitative risk assessment."""
        type3_risk = len(self.type3_dark_zones) > 0
        type2_risk = len(self.type2_candidates) > 5
        type1_risk = len(self.type1_candidates) > 0

        risks = sum([type3_risk, type2_risk, type1_risk])
        if risks >= 2:
            return "HIGH — Multiple dark zone types detected"
        elif risks == 1:
            return "MEDIUM — One dark zone type detected"
        else:
            return "LOW — No significant dark zones detected"


class UnifiedScanner:
    """Runs all three dark zone detectors on a constraint field."""

    def __init__(self, protocol_name: str,
                 type3_cancellation_eps: float = 0.15,
                 type3_individual_min: float = 0.05,
                 type2_condition_threshold: float = 100.0,
                 type1_divergence_pct: float = 1.0,
                 residual_epsilon: float = 0.02):
        self.protocol_name = protocol_name
        self.rules: list[Rule] = []
        self.data_sources: list[DataSource] = []

        # Thresholds
        self.type3_cancellation_eps = type3_cancellation_eps
        self.type3_individual_min = type3_individual_min
        self.type2_condition_threshold = type2_condition_threshold
        self.type1_divergence_pct = type1_divergence_pct
        self.residual_epsilon = residual_epsilon

    def add_rule(self, rule: Rule):
        self.rules.append(rule)

    def add_data_source(self, source: DataSource):
        self.data_sources.append(source)

    def _build_field(self) -> ConstraintField:
        return ConstraintField(rules=self.rules)

    def scan(self, bounds: list[tuple[float, float]],
             n_points: int = 60) -> UnifiedScanReport:
        """Run the complete three-type scan."""

        field = self._build_field()
        constraint_names = [r.name for r in self.rules]

        # --- Type III: Cancellation dark zones ---
        dark_detector = DarkZoneDetector(
            cancellation_eps=self.type3_cancellation_eps,
            individual_min=self.type3_individual_min,
        )
        type3_dark_zones = dark_detector.scan(field, bounds, n_points=n_points)

        # --- Residual detection ---
        residual_detector = ResidualDetector(field, epsilon=self.residual_epsilon)
        residuals = residual_detector.scan_grid(bounds, n_points=n_points)

        # --- Type II: Metric tensor eigenvalue analysis ---
        type2_candidates = self._scan_type2(field, bounds, n_points=min(n_points, 40))

        # --- Type I: Oracle divergence (if data sources configured) ---
        type1_candidates = []
        if self.data_sources:
            type1_candidates = self._scan_type1(field, bounds, n_points=min(n_points, 30))

        return UnifiedScanReport(
            protocol_name=self.protocol_name,
            bounds=bounds,
            type3_dark_zones=type3_dark_zones,
            type3_count=len(type3_dark_zones),
            type2_candidates=type2_candidates,
            type2_count=len(type2_candidates),
            type1_candidates=type1_candidates,
            type1_count=len(type1_candidates),
            residual_points=len(residuals),
            constraint_names=constraint_names,
            n_state_dims=len(bounds),
        )

    def _scan_type2(self, field: ConstraintField,
                    bounds: list[tuple[float, float]],
                    n_points: int = 40) -> list[Type2Candidate]:
        """Scan for Type II structural occlusion via metric tensor eigenvalues."""

        dims = len(bounds)
        axes = [np.linspace(lo, hi, n_points) for lo, hi in bounds]
        mesh = np.meshgrid(*axes, indexing='ij')

        candidates = []
        for idx in np.ndindex(mesh[0].shape):
            p = np.array([mesh[d][idx] for d in range(dims)], dtype=float)

            # Compute metric tensor g_{ij} = Σ_k (∂σ_k/∂x_i)(∂σ_k/∂x_j)
            g = np.zeros((dims, dims))
            for rule in field.rules:
                grad = rule.gradient(p)
                for i in range(dims):
                    for j in range(dims):
                        g[i, j] += grad[i] * grad[j]

            # Eigenvalue analysis
            eigenvalues, eigenvectors = np.linalg.eigh(g)
            idx_sort = np.argsort(eigenvalues)
            eigenvalues = eigenvalues[idx_sort]
            eigenvectors = eigenvectors[:, idx_sort]

            min_ev = eigenvalues[0]
            max_ev = eigenvalues[-1]

            if max_ev < 0.5:  # No significant constraint at all
                continue

            condition_number = max_ev / min_ev if min_ev > 1e-10 else float('inf')

            if condition_number > self.type2_condition_threshold:
                candidates.append(Type2Candidate(
                    position=p.copy(),
                    min_eigenvalue=float(min_ev),
                    max_eigenvalue=float(max_ev),
                    condition_number=float(condition_number),
                    unconstrained_direction=eigenvectors[:, 0].copy(),
                    constrained_direction=eigenvectors[:, -1].copy(),
                ))

        candidates.sort(key=lambda c: c.condition_number, reverse=True)
        return candidates

    def _scan_type1(self, field: ConstraintField,
                    bounds: list[tuple[float, float]],
                    n_points: int = 30) -> list[Type1Candidate]:
        """Scan for Type I physical occlusion via oracle divergence."""

        type1_detector = Type1Detector(
            divergence_threshold_pct=self.type1_divergence_pct,
        )
        for src in self.data_sources:
            type1_detector.add_source(src)

        return type1_detector.scan_grid(field, bounds, n_points=n_points)
