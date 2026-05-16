"""
Constraint Residual Method — Core Engine

Implements the operational definition from 规则认知体系_稳固版:
  σ(R|E) = |⟨O⟩_{R present} - ⟨O⟩_{R absent}| / ΔO

And the residual detection:
  Π_known(p) = Σᵢ ∇σ(Rᵢ)(p)
  if ||Π_known(p)|| > ε → residual → unknown rule candidate
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional
from collections.abc import Sequence


@dataclass
class Rule:
    """A rule with a computable constraint function.

    Attributes:
        name: Rule identifier
        layer: Layer index (lower = more fundamental). L-1=-1, L0=0, L1=1, L2=2, L3=3
        domain: Constraint domain label (e.g. 'EM', 'gravity', 'quantum')
        constraint_fn: σ(p) → constraint strength at point p in state space
        gradient_fn: ∇σ(p) → constraint gradient vector at p (None = use numerical)
    """
    name: str
    layer: int
    domain: str
    constraint_fn: Callable[[np.ndarray], float]
    gradient_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None
    certainty: float = 1.0

    def gradient(self, p: np.ndarray, h: float = 1e-6) -> np.ndarray:
        """Numerical gradient of constraint strength at point p."""
        if self.gradient_fn is not None:
            return self.gradient_fn(p)
        grad = np.zeros_like(p, dtype=float)
        for i in range(len(p)):
            p_fwd = p.copy()
            p_bwd = p.copy()
            p_fwd[i] += h
            p_bwd[i] -= h
            grad[i] = (self.constraint_fn(p_fwd) - self.constraint_fn(p_bwd)) / (2 * h)
        return grad


@dataclass
class ConstraintField:
    """The combined constraint field from a set of known rules."""
    rules: list[Rule]

    def combined_constraint(self, p: np.ndarray) -> float:
        """Sum of constraint strengths from all rules at point p."""
        return sum(r.constraint_fn(p) for r in self.rules)

    def constraint_gradient(self, p: np.ndarray) -> np.ndarray:
        """Π_known(p) = Σᵢ ∇σ(Rᵢ)(p) — the constraint gradient vector."""
        total = np.zeros_like(p, dtype=float)
        for r in self.rules:
            total += r.gradient(p)
        return total

    def constraint_magnitude(self, p: np.ndarray) -> float:
        """||Π_known(p)|| — scalar magnitude of combined constraint."""
        return float(np.linalg.norm(self.constraint_gradient(p)))


@dataclass
class ResidualPoint:
    """A point where known constraints leave a residual."""
    position: np.ndarray
    residual_vector: np.ndarray
    magnitude: float
    direction: np.ndarray


@dataclass
class ResidualCluster:
    """A cluster of residual points pointing to the same unknown rule."""
    points: list[ResidualPoint] = field(default_factory=list)
    mean_direction: Optional[np.ndarray] = None
    mean_magnitude: float = 0.0
    centroid: Optional[np.ndarray] = None

    def add(self, rp: ResidualPoint):
        self.points.append(rp)

    def compute_stats(self):
        if not self.points:
            return
        directions = np.array([rp.direction for rp in self.points])
        self.mean_direction = np.mean(directions, axis=0)
        self.mean_direction /= np.linalg.norm(self.mean_direction)
        self.mean_magnitude = float(np.mean([rp.magnitude for rp in self.points]))
        positions = np.array([rp.position for rp in self.points])
        self.centroid = np.mean(positions, axis=0)

    def candidate_rule_constraint(self) -> np.ndarray:
        """The predicted constraint direction of the unknown rule = -Π_known."""
        if self.mean_direction is None:
            self.compute_stats()
        return -self.mean_direction * self.mean_magnitude


@dataclass
class ResidualDetector:
    """Detects constraint residuals across a state space region.

    Usage:
        field = ConstraintField(rules=[...])
        detector = ResidualDetector(field, epsilon=0.05)
        residuals = detector.scan_grid(bounds=[(0,1), (0,1)], n_points=50)
        clusters = detector.cluster_residuals(residuals, angle_threshold_deg=30)
    """
    field: ConstraintField
    epsilon: float = 0.05

    def scan_grid(self, bounds: list[tuple[float, float]],
                  n_points: int = 50) -> list[ResidualPoint]:
        """Scan a grid over the state space region defined by bounds.
        bounds: [(min1, max1), (min2, max2), ...]
        Returns list of ResidualPoints where ||Π|| > epsilon.
        """
        dims = len(bounds)
        axes = [np.linspace(lo, hi, n_points) for lo, hi in bounds]
        mesh = np.meshgrid(*axes, indexing='ij')
        residuals: list[ResidualPoint] = []

        for idx in np.ndindex(mesh[0].shape):
            p = np.array([mesh[d][idx] for d in range(dims)], dtype=float)
            grad = self.field.constraint_gradient(p)
            mag = float(np.linalg.norm(grad))
            if mag > self.epsilon:
                direction = grad / mag
                residuals.append(ResidualPoint(
                    position=p.copy(),
                    residual_vector=grad.copy(),
                    magnitude=mag,
                    direction=direction
                ))
        return residuals

    def scan_points(self, points: np.ndarray) -> list[ResidualPoint]:
        """Scan a set of specific points. points: (N, D) array."""
        residuals: list[ResidualPoint] = []
        for p in points:
            grad = self.field.constraint_gradient(p)
            mag = float(np.linalg.norm(grad))
            if mag > self.epsilon:
                direction = grad / mag
                residuals.append(ResidualPoint(
                    position=p.copy(),
                    residual_vector=grad.copy(),
                    magnitude=mag,
                    direction=direction
                ))
        return residuals

    def cluster_residuals(self, residuals: list[ResidualPoint],
                          angle_threshold_deg: float = 30.0) -> list[ResidualCluster]:
        """Cluster residuals by direction. Points within angle_threshold_deg
        of each other are assigned to the same cluster (same unknown rule)."""
        if not residuals:
            return []

        cos_threshold = np.cos(np.radians(angle_threshold_deg))
        n = len(residuals)
        visited = [False] * n
        clusters: list[ResidualCluster] = []

        for i in range(n):
            if visited[i]:
                continue
            cluster = ResidualCluster()
            cluster.add(residuals[i])
            visited[i] = True

            for j in range(i + 1, n):
                if visited[j]:
                    continue
                cos_sim = float(np.dot(residuals[i].direction, residuals[j].direction))
                if cos_sim > cos_threshold:
                    cluster.add(residuals[j])
                    visited[j] = True

            cluster.compute_stats()
            clusters.append(cluster)

        clusters.sort(key=lambda c: c.mean_magnitude, reverse=True)
        return clusters

    def propose_unknown_rules(self, clusters: list[ResidualCluster]) -> list[dict]:
        """Generate candidate unknown rules from residual clusters."""
        proposals = []
        for i, c in enumerate(clusters):
            dir_str = ", ".join(f"{c.mean_direction[d]:.3f}" for d in range(len(c.mean_direction)))
            proposals.append({
                'id': f'R_unknown_{i+1}',
                'constraint_direction': c.candidate_rule_constraint().tolist(),
                'magnitude': c.mean_magnitude,
                'centroid': c.centroid.tolist() if c.centroid is not None else None,
                'n_points': len(c.points),
                'description': (
                    f"Unknown rule with constraint direction ≈ "
                    f"({dir_str}), "
                    f"magnitude {c.mean_magnitude:.3f}. "
                    f"Supported by {len(c.points)} residual points."
                )
            })
        return proposals
