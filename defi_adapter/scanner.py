"""
DeFi Scanner — orchestrates constraint extraction, state mapping, and dark zone
detection for a DeFi protocol.

The scanner takes a set of constraints (from templates or manual specification),
a state mapper (defining the state space), and runs the full analysis pipeline:

  1. Build ConstraintField from rules
  2. Scan state space with ResidualDetector
  3. Scan for Type III dark zones with DarkZoneDetector
  4. Cross-reference residuals with dark zones
  5. Produce a ScanReport

Usage:
    scanner = DeFiScanner("Euler Finance")
    scanner.add_rule(collateral_health_template)
    scanner.add_rule(liquidation_template)
    ...
    report = scanner.scan(mapper, n_points=50)
    report.summary()
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from constraint_residual.core import Rule, ConstraintField, ResidualDetector, ResidualPoint
from constraint_residual.dark_zone_detector import DarkZoneDetector, DarkZoneCluster
from constraint_residual.defi_adapter.state_mapper import StateMapper
from constraint_residual.defi_adapter.constraint_templates import ConstraintTemplate, template_to_rule


@dataclass
class PathAnalysis:
    """Analysis of a specific state transition path.

    For example: pre-attack → attack state. Tracks how constraint metrics
    change along the path.
    """
    name: str
    points: list[np.ndarray]  # ordered state points along the path
    labels: list[str]          # label for each point
    metrics: list[dict] = field(default_factory=list)

    def analyze(self, field: ConstraintField, dark_detector: DarkZoneDetector):
        """Compute constraint metrics at each point along the path."""
        self.metrics = []
        for p in self.points:
            grad = field.constraint_gradient(p)
            mag = float(np.linalg.norm(grad))
            cr = dark_detector.cancellation_ratio(field, p)

            indiv_mags = {}
            for rule in field.rules:
                g = rule.gradient(p)
                indiv_mags[rule.name] = float(np.linalg.norm(g))

            sum_indiv = sum(indiv_mags.values())

            self.metrics.append({
                'position': p,
                'combined_magnitude': mag,
                'cancellation_ratio': cr,
                'individual_sum': sum_indiv,
                'individual_magnitudes': indiv_mags,
            })

    def is_dark_at(self, idx: int, threshold: float = 0.1) -> bool:
        """Check if path point idx is in a dark zone."""
        if idx >= len(self.metrics):
            return False
        m = self.metrics[idx]
        return m['cancellation_ratio'] < threshold and m['individual_sum'] > 0.2


@dataclass
class ScanReport:
    """Structured output from a DeFi protocol scan.

    Contains:
      - Dark zone clusters (Type III blind spots)
      - Residual points (areas where known constraints are weak)
      - Path analyses (specific state transitions evaluated)
      - Combined risk assessment
    """
    protocol_name: str
    dark_zones: list[DarkZoneCluster] = field(default_factory=list)
    residuals: list[ResidualPoint] = field(default_factory=list)
    path_analyses: list[PathAnalysis] = field(default_factory=list)
    constraint_names: list[str] = field(default_factory=list)
    n_state_dims: int = 0
    scan_bounds: list[tuple[float, float]] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"{'='*60}",
            f"DeFi Dark Zone Scan: {self.protocol_name}",
            f"{'='*60}",
            f"State dimensions: {self.n_state_dims}",
            f"Constraints analyzed: {len(self.constraint_names)}",
        ]
        for name in self.constraint_names:
            lines.append(f"  • {name}")

        lines.append(f"\nDark zones found: {len(self.dark_zones)}")
        for dz in self.dark_zones:
            lines.append(f"  [{dz.balance_topology}] {len(dz.points)} points, "
                        f"c̄={dz.mean_cancellation_ratio:.4f}")
            lines.append(f"    Involved: {', '.join(dz.constraints_involved)}")

        lines.append(f"\nResidual points: {len(self.residuals)}")

        for pa in self.path_analyses:
            lines.append(f"\nPath: {pa.name}")
            for i, (label, m) in enumerate(zip(pa.labels, pa.metrics)):
                dark_flag = " ⚠ DARK ZONE" if pa.is_dark_at(i) else ""
                lines.append(f"  [{label}] ||Π||={m['combined_magnitude']:.4f}, "
                           f"c={m['cancellation_ratio']:.4f}, "
                           f"Σ||∇σ||={m['individual_sum']:.4f}{dark_flag}")

        return "\n".join(lines)

    def highest_risk_zones(self, n: int = 5) -> list[DarkZoneCluster]:
        """Return the n most dangerous dark zones (lowest cancellation ratio)."""
        sorted_zones = sorted(self.dark_zones, key=lambda dz: dz.mean_cancellation_ratio)
        return sorted_zones[:n]


class DeFiScanner:
    """Orchestrates constraint residual analysis for a DeFi protocol.

    Usage:
        scanner = DeFiScanner("MyProtocol")
        scanner.add_template(reentrancy_guard(0), name="ReentrancyGuard")
        scanner.add_template(collateral_health(1), name="CollateralHealth")
        report = scanner.scan(mapper, n_points=50)
        report_with_path = scanner.scan_with_path(mapper, path, n_points=50)
    """

    def __init__(self, protocol_name: str,
                 residual_epsilon: float = 0.05,
                 dark_zone_cancellation_eps: float = 0.1,
                 dark_zone_individual_min: float = 0.1):
        self.protocol_name = protocol_name
        self.rules: list[Rule] = []
        self.rule_names: list[str] = []
        self.residual_epsilon = residual_epsilon
        self.dark_zone_cancellation_eps = dark_zone_cancellation_eps
        self.dark_zone_individual_min = dark_zone_individual_min

    def add_rule(self, rule: Rule):
        """Add a Rule object directly (for custom constraints)."""
        self.rules.append(rule)
        self.rule_names.append(rule.name)

    def add_template(self, template: ConstraintTemplate, name: str = "", layer: int = 2):
        """Add a constraint from a template."""
        rule_name = name or template.pattern
        self.rules.append(template_to_rule(template, rule_name, layer))
        self.rule_names.append(rule_name)

    def _build_field(self) -> ConstraintField:
        if not self.rules:
            raise ValueError("No rules added to scanner")
        return ConstraintField(rules=self.rules)

    def scan(self, mapper: StateMapper, n_points: int = 50) -> ScanReport:
        """Run a full scan over the state space defined by the mapper.

        Args:
            mapper: StateMapper with dimensions configured
            n_points: grid resolution per dimension

        Returns:
            ScanReport with dark zones and residuals
        """
        bounds = mapper.build_bounds()
        field = self._build_field()

        # Residual scan
        residual_detector = ResidualDetector(field, epsilon=self.residual_epsilon)
        residuals = residual_detector.scan_grid(bounds, n_points=n_points)

        # Dark zone scan
        dark_detector = DarkZoneDetector(
            cancellation_eps=self.dark_zone_cancellation_eps,
            individual_min=self.dark_zone_individual_min,
        )
        dark_zones = dark_detector.scan(field, bounds, n_points=n_points)

        return ScanReport(
            protocol_name=self.protocol_name,
            dark_zones=dark_zones,
            residuals=residuals,
            constraint_names=list(self.rule_names),
            n_state_dims=mapper.n_dims,
            scan_bounds=bounds,
        )

    def scan_with_path(self, mapper: StateMapper, path: PathAnalysis,
                       n_points: int = 50) -> ScanReport:
        """Run a scan and also analyze a specific state transition path.

        The path is typically: pre-condition → intermediate states → exploit state.
        This shows whether the path passes through dark zones.
        """
        report = self.scan(mapper, n_points=n_points)

        field = self._build_field()
        dark_detector = DarkZoneDetector(
            cancellation_eps=self.dark_zone_cancellation_eps,
            individual_min=self.dark_zone_individual_min,
        )
        path.analyze(field, dark_detector)
        report.path_analyses.append(path)

        return report

    def scan_specific_points(self, points: dict[str, np.ndarray]) -> dict[str, dict]:
        """Evaluate constraint metrics at specific named state points.

        Useful for: "what do the metrics look like at the pre-attack state
        vs the attack state?"

        Args:
            points: {"label": np.ndarray} — named state vectors

        Returns:
            {"label": {"combined_mag": ..., "cancellation_ratio": ..., ...}}
        """
        field = self._build_field()
        dark_detector = DarkZoneDetector(
            cancellation_eps=self.dark_zone_cancellation_eps,
            individual_min=self.dark_zone_individual_min,
        )

        results = {}
        for label, p in points.items():
            grad = field.constraint_gradient(p)
            mag = float(np.linalg.norm(grad))
            cr = dark_detector.cancellation_ratio(field, p)

            indiv = {}
            for rule in self.rules:
                g = rule.gradient(p)
                indiv[rule.name] = float(np.linalg.norm(g))

            results[label] = {
                'combined_magnitude': mag,
                'cancellation_ratio': cr,
                'individual_sum': sum(indiv.values()),
                'individual_magnitudes': indiv,
                'is_dark_zone': cr < self.dark_zone_cancellation_eps and sum(indiv.values()) > 0.2,
            }

        return results
