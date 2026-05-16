"""
Type I Dark Zone Detector — Physical / Information Occlusion.

Type I invisibility (from the Stability-Visibility Inverse Law):
  A constraint's signal is masked by physical or information occlusion.
  The constraint EXISTS and is ACTIVE, but its sensor (oracle, data source)
  reports the wrong value — so the constraint stays quiet when it should scream.

In DeFi, this maps to oracle/data source divergence:
  - Constraint A uses Chainlink price feed → "sees" price $2000
  - Constraint B uses Uniswap TWAP → "sees" price $1950
  - If the true price is $1900, both constraints underestimate risk
  - The "real" constraint tension (using true price) would be HIGH

Detection principle:
  Π_observed(p) = Σ ∇σ_i(p | oracle_i)  — what the protocol "sees"
  Π_true(p) = Σ ∇σ_i(p | true_price)    — what should be seen
  Type I residual = ||Π_true|| - ||Π_observed|| > 0 → occlusion in progress

Pragmatic approach (without knowing "true" price):
  Compare constraint tension across multiple oracle/data sources.
  Divergence in tension → potential occlusion → flag as Type I candidate.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional

from constraint_residual.core import Rule, ConstraintField


@dataclass
class DataSource:
    """A data source that feeds into constraint evaluation.

    Examples: Chainlink price feed, Uniswap TWAP, on-chain reserve ratio,
    off-chain API, etc.
    """
    name: str
    source_type: str  # 'oracle', 'onchain', 'derived', 'external'
    get_value: Callable[[np.ndarray], float]  # extracts this source's value from state
    latency_blocks: int = 0  # delay before value updates
    manipulation_cost_usd: float = float('inf')  # cost to move price 1%
    reliability: float = 1.0  # 0-1


@dataclass
class Type1Candidate:
    """A state-space region where oracle divergence creates a Type I blind spot."""
    position: np.ndarray
    occlusion_score: float        # ||Π_true_est|| - ||Π_observed||
    diverging_sources: list[str]   # which data sources disagree
    max_divergence_pct: float      # maximum divergence between sources
    affected_constraints: list[str] # constraints whose tension changes most
    recommendation: str = ''


class Type1Detector:
    """Detects Type I dark zones — constraint occlusion via data source divergence.

    Usage:
        detector = Type1Detector()
        detector.add_source(DataSource("chainlink_eth", "oracle", ...))
        detector.add_source(DataSource("uniswap_twap_eth", "oracle", ...))
        candidates = detector.scan(field, data_points, sources_map)
    """

    def __init__(self, divergence_threshold_pct: float = 1.0,
                 occlusion_threshold: float = 0.05):
        """
        Args:
            divergence_threshold_pct: minimum % divergence between sources to flag
            occlusion_threshold: minimum occlusion score to report
        """
        self.sources: list[DataSource] = []
        self.divergence_threshold_pct = divergence_threshold_pct
        self.occlusion_threshold = occlusion_threshold

    def add_source(self, source: DataSource):
        self.sources.append(source)

    def source_divergence(self, p: np.ndarray) -> dict:
        """Compute pairwise divergence between all data sources at point p.

        Returns dict with divergence metrics.
        """
        values = {}
        for src in self.sources:
            values[src.name] = src.get_value(p)

        divergences = {}
        names = list(values.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                v1, v2 = values[names[i]], values[names[j]]
                if abs(v1) > 1e-10:
                    div_pct = abs(v1 - v2) / abs(v1) * 100.0
                else:
                    div_pct = 0.0
                key = f"{names[i]} vs {names[j]}"
                divergences[key] = {
                    'value_1': v1,
                    'value_2': v2,
                    'divergence_pct': div_pct,
                }

        return {
            'values': values,
            'divergences': divergences,
            'max_divergence_pct': max(
                (d['divergence_pct'] for d in divergences.values()),
                default=0.0
            ),
        }

    def compute_occlusion_score(self, field: ConstraintField, p: np.ndarray,
                                 source_values: dict[str, float],
                                 perturbed_values: dict[str, float]) -> float:
        """Compute occlusion score by comparing constraint tension under
        different data source assumptions.

        occlusion_score = ||Π_alt|| - ||Π_ref||
        where Π_ref uses reference source values and Π_alt uses alternative values.
        A large positive score means the alternative data shows MORE constraint
        tension — the reference data is occluding risk.
        """
        # This is a simplified version — full implementation would need
        # constraint functions that accept data source parameters.
        # For now, compute using the field as-is and compare gradient magnitudes.

        grad_ref = field.constraint_gradient(p)
        mag_ref = float(np.linalg.norm(grad_ref))

        # For each source pair with divergence, estimate the "true" gradient
        # magnitude if the more conservative source were used
        max_div = 0.0
        for d in source_values.values():
            if abs(d) > 1e-10:
                max_div = max(max_div, abs(d))

        # Simple proxy: occlusion = how much additional tension the
        # most conservative source would add
        occlusion = max(0.0, max_div - mag_ref)
        return float(occlusion)

    def scan(self, field: ConstraintField, points: np.ndarray) -> list[Type1Candidate]:
        """Scan a set of state points for Type I occlusion.

        Args:
            field: the constraint field
            points: (N, D) array of state points to evaluate

        Returns:
            list of Type1Candidate objects sorted by occlusion score
        """
        candidates = []

        for p in points:
            div_result = self.source_divergence(p)

            if div_result['max_divergence_pct'] < self.divergence_threshold_pct:
                continue

            # Compute occlusion score
            occlusion = self.compute_occlusion_score(
                field, p, div_result['values'], div_result['values']
            )

            if occlusion < self.occlusion_threshold:
                continue

            # Identify which sources diverge
            diverging = []
            for key, d in div_result['divergences'].items():
                if d['divergence_pct'] >= self.divergence_threshold_pct:
                    diverging.append(key)

            # Check which constraints are affected
            affected = []
            for rule in field.rules:
                grad = rule.gradient(p)
                if float(np.linalg.norm(grad)) > 0.01:
                    affected.append(rule.name)

            candidates.append(Type1Candidate(
                position=p.copy(),
                occlusion_score=occlusion,
                diverging_sources=diverging,
                max_divergence_pct=div_result['max_divergence_pct'],
                affected_constraints=affected,
                recommendation=(
                    f"Oracle divergence {div_result['max_divergence_pct']:.1f}% — "
                    f"constraints may be reading inconsistent data. "
                    f"Cross-reference: {', '.join(diverging[:2])}"
                ),
            ))

        candidates.sort(key=lambda c: c.occlusion_score, reverse=True)
        return candidates

    def scan_grid(self, field: ConstraintField,
                  bounds: list[tuple[float, float]],
                  n_points: int = 50) -> list[Type1Candidate]:
        """Scan a grid region for Type I occlusion."""
        dims = len(bounds)
        axes = [np.linspace(lo, hi, n_points) for lo, hi in bounds]
        mesh = np.meshgrid(*axes, indexing='ij')

        all_points = []
        for idx in np.ndindex(mesh[0].shape):
            p = np.array([mesh[d][idx] for d in range(dims)], dtype=float)
            all_points.append(p)

        return self.scan(field, np.array(all_points))
