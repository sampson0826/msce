"""
Dark Zone Detector — Type III invisibility through perfect cross-constraint balance.

Type III dark zones are the most dangerous form of invisibility:
multiple known rules exert strong constraints, but their constraint forces
cancel exactly (Σ∇σ ≈ 0 while each ||∇σ|| ≫ 0). This produces zero
observable signal — you cannot see the rules, and you cannot even suspect
there's a gap because there's no anomaly.

The key insight distinguishing Type III from "no constraints active":
  cancellation_ratio = ||Σ∇σ|| / Σ||∇σ||
  → 0: perfect mutual cancellation (dark zone)
  → 1: one rule dominates, no cancellation

Detection works by scanning the individual constraint gradients, not just
the combined field — you need to see what cancels what.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from constraint_residual.core import Rule, ConstraintField, ResidualPoint


@dataclass
class DarkZonePoint:
    """A single point where multiple constraints cancel."""
    position: np.ndarray
    cancellation_ratio: float  # ||Σ∇σ|| / Σ||∇σ||
    total_magnitude: float     # ||Σ∇σ||
    individual_sum: float      # Σ||∇σ||
    individual_gradients: dict[str, np.ndarray]  # rule_name → ∇σ


@dataclass
class DarkZoneCluster:
    """A spatially connected region of dark zone points."""
    id: int
    points: list[DarkZonePoint] = field(default_factory=list)
    centroid: Optional[np.ndarray] = None
    mean_cancellation_ratio: float = 0.0
    constraints_involved: list[str] = field(default_factory=list)
    balance_topology: str = ''
    break_direction: Optional[np.ndarray] = None
    break_conditions: list[dict] = field(default_factory=list)


class DarkZoneDetector:
    """Detects Type III dark zones in a constraint field.

    Usage:
        field = ConstraintField(rules=[...])
        detector = DarkZoneDetector(
            cancellation_eps=0.05,    # max cancellation_ratio for dark zone
            individual_min=0.1,        # min individual ||∇σ|| to count as "active"
        )
        dark_zones = detector.scan(field, bounds, n_points=50)
        for dz in dark_zones:
            print(dz.summary())
    """

    def __init__(self, cancellation_eps: float = 0.05,
                 individual_min: float = 0.1,
                 angle_threshold_deg: float = 30.0):
        """
        Args:
            cancellation_eps: Points with cancellation_ratio below this
                are candidates. Lower = more conservative (fewer false positives).
            individual_min: Each rule must have ||∇σ|| above this to be
                counted as actively participating in the cancellation.
            angle_threshold_deg: Max angle difference for clustering dark
                zone points into the same region.
        """
        self.cancellation_eps = cancellation_eps
        self.individual_min = individual_min
        self.angle_threshold_deg = angle_threshold_deg

    def cancellation_ratio(self, field: ConstraintField, p: np.ndarray) -> float:
        """Compute cancellation ratio at point p.

        ||Σ∇σ|| / Σ||∇σ|| — the ratio of combined to individual constraint strength.

        Returns:
            0.0 = perfect mutual cancellation (dark zone)
            1.0 = no cancellation (single rule dominates)
        """
        individual_mags = []
        for rule in field.rules:
            grad = rule.gradient(p)
            mag = float(np.linalg.norm(grad))
            individual_mags.append(mag)

        sum_individual = sum(individual_mags)
        total_grad = field.constraint_gradient(p)
        total_mag = float(np.linalg.norm(total_grad))

        if sum_individual < 1e-12:
            return 1.0  # No constraints active at all — not a dark zone

        return total_mag / sum_individual

    def individual_gradients(self, field: ConstraintField,
                              p: np.ndarray) -> dict[str, np.ndarray]:
        """Get individual constraint gradients from each rule at point p."""
        return {rule.name: rule.gradient(p) for rule in field.rules}

    def scan(self, field: ConstraintField,
             bounds: list[tuple[float, float]],
             n_points: int = 50) -> list[DarkZoneCluster]:
        """Scan a region for Type III dark zones.

        Returns list of DarkZoneCluster objects, sorted by cancellation
        ratio (most perfect cancellation first).
        """
        dims = len(bounds)
        axes = [np.linspace(lo, hi, n_points) for lo, hi in bounds]
        mesh = np.meshgrid(*axes, indexing='ij')

        dark_points: list[DarkZonePoint] = []

        for idx in np.ndindex(mesh[0].shape):
            p = np.array([mesh[d][idx] for d in range(dims)], dtype=float)
            cr = self.cancellation_ratio(field, p)
            total_mag = float(np.linalg.norm(field.constraint_gradient(p)))

            # Compute individual magnitudes
            indiv_grads = self.individual_gradients(field, p)
            indiv_sum = sum(float(np.linalg.norm(g)) for g in indiv_grads.values())

            # Dark zone condition: low cancellation ratio AND strong individuals
            if cr < self.cancellation_eps and indiv_sum > self.individual_min * len(field.rules):
                dark_points.append(DarkZonePoint(
                    position=p.copy(),
                    cancellation_ratio=cr,
                    total_magnitude=total_mag,
                    individual_sum=indiv_sum,
                    individual_gradients={k: v.copy() for k, v in indiv_grads.items()}
                ))

        return self._cluster_and_classify(dark_points, field)

    def _cluster_and_classify(self, dark_points: list[DarkZonePoint],
                               field: ConstraintField) -> list[DarkZoneCluster]:
        """Cluster dark zone points spatially and classify balance topology."""
        if not dark_points:
            return []

        n = len(dark_points)
        visited = [False] * n
        clusters: list[DarkZoneCluster] = []

        # Spatial clustering
        for i in range(n):
            if visited[i]:
                continue
            cluster = DarkZoneCluster(id=len(clusters) + 1)
            cluster.points.append(dark_points[i])
            visited[i] = True

            for j in range(i + 1, n):
                if visited[j]:
                    continue
                dist = float(np.linalg.norm(
                    dark_points[i].position - dark_points[j].position
                ))
                if dist < 1.0:  # Spatial proximity threshold
                    cluster.points.append(dark_points[j])
                    visited[j] = True

            # Compute cluster stats
            positions = np.array([dp.position for dp in cluster.points])
            cluster.centroid = np.mean(positions, axis=0)
            cluster.mean_cancellation_ratio = float(np.mean(
                [dp.cancellation_ratio for dp in cluster.points]
            ))

            # Identify involved constraints
            all_rules = set()
            for dp in cluster.points:
                for name, grad in dp.individual_gradients.items():
                    if float(np.linalg.norm(grad)) > self.individual_min:
                        all_rules.add(name)
            cluster.constraints_involved = sorted(all_rules)

            # Classify balance topology
            self._classify_topology(cluster)

            # Compute break direction first (needed by break condition design)
            cluster.break_direction = self._compute_break_direction(cluster)

            # Design break conditions
            cluster.break_conditions = self._design_break_conditions(cluster, field)

            clusters.append(cluster)

        clusters.sort(key=lambda c: c.mean_cancellation_ratio)
        return clusters

    def _classify_topology(self, cluster: DarkZoneCluster):
        """Classify the balance topology of a dark zone cluster.

        Three types:
        - 'mutual_cancellation': 2 rules with opposing gradients
        - 'cyclic': 3+ rules whose gradients sum to zero in a cycle
        - 'hierarchical': constraints from different layers oppose
        """
        n_constraints = len(cluster.constraints_involved)

        if n_constraints == 2:
            cluster.balance_topology = 'mutual_cancellation'
            return
        elif n_constraints <= 1:
            cluster.balance_topology = 'underconstrained'
            return

        if n_constraints >= 3:
            # Check if any 3 gradients form a closed triangle
            # Get average gradients for each constraint at centroid
            if cluster.centroid is not None and cluster.points:
                sample = cluster.points[0]
                grads = {}
                for name, grad in sample.individual_gradients.items():
                    if name in cluster.constraints_involved:
                        grads[name] = grad / (np.linalg.norm(grad) + 1e-12)

                # If 3+ vectors nearly sum to zero → cyclic
                if len(grads) >= 3:
                    grad_list = list(grads.values())
                    total = np.sum(grad_list, axis=0)
                    if float(np.linalg.norm(total)) < 0.3:
                        cluster.balance_topology = 'cyclic'
                        return

        cluster.balance_topology = 'hierarchical'

    def _compute_break_direction(self, cluster: DarkZoneCluster) -> Optional[np.ndarray]:
        """Find the direction along which the dark zone is most easily broken.

        For mutual cancellation: the break direction is along either rule's
        gradient — pushing in that direction amplifies one constraint while
        the opposing constraint may not scale identically, breaking the balance.

        For cyclic/hierarchical: use PCA on the constraint gradient space to
        find the direction of maximum variance.
        """
        if not cluster.points:
            return None

        if cluster.balance_topology == 'mutual_cancellation':
            # Break direction = gradient of the strongest individual constraint
            sample = cluster.points[0]
            max_mag = 0.0
            best_grad = None
            for grad in sample.individual_gradients.values():
                mag = float(np.linalg.norm(grad))
                if mag > max_mag:
                    max_mag = mag
                    best_grad = grad
            if best_grad is not None and max_mag > 1e-12:
                return best_grad / max_mag
            # Fallback: use first constraint's direction
            if sample.individual_gradients:
                g = list(sample.individual_gradients.values())[0]
                m = float(np.linalg.norm(g))
                if m > 1e-12:
                    return g / m

        # For other topologies: compute residual vectors
        residuals = []
        for dp in cluster.points:
            total = np.zeros_like(dp.position)
            for grad in dp.individual_gradients.values():
                total += grad
            if np.linalg.norm(total) > 1e-12:
                residuals.append(total / np.linalg.norm(total))

        if residuals:
            break_dir = np.mean(residuals, axis=0)
            nrm = float(np.linalg.norm(break_dir))
            if nrm > 1e-12:
                return break_dir / nrm
        return None

    def _design_break_conditions(self, cluster: DarkZoneCluster,
                                  field: ConstraintField) -> list[dict]:
        """Design conditions that would break this dark zone.

        Types of break:
        - extreme_energy: Push to Planck scale → cancellation may break
        - extreme_density: Push to high density → individual constraints may diverge
        - extreme_curvature: Push to strong curvature → geometric constraints break
        - symmetry_breaking: Introduce explicit symmetry breaking
        """
        conditions = []
        break_dir = cluster.break_direction
        if break_dir is None:
            return conditions

        centroid = cluster.centroid
        if centroid is None:
            centroid = np.zeros_like(break_dir)

        # 1. Extreme extrapolation along break direction
        extreme_point = centroid + break_dir * 10.0
        pred_mag = float(np.linalg.norm(field.constraint_gradient(extreme_point)))
        conditions.append({
            'condition': 'Extreme extrapolation along break direction',
            'target_region': extreme_point.tolist() if hasattr(extreme_point, 'tolist') else extreme_point,
            'predicted_magnitude': pred_mag,
            'break_mechanism': 'amplify_constraint_mismatch',
            'difficulty': 'high' if abs(pred_mag) < 0.1 else 'medium',
        })

        # 2. Introduce asymmetric perturbation
        perturbed_point = centroid + break_dir * 3.0
        # Add small perturbation orthogonal to break direction
        if len(break_dir) >= 2:
            orth = np.eye(len(break_dir))[1]
            perturbed_point = perturbed_point + orth * 1.0
        pred_mag2 = float(np.linalg.norm(field.constraint_gradient(perturbed_point)))
        conditions.append({
            'condition': 'Asymmetric perturbation + moderate extrapolation',
            'target_region': perturbed_point.tolist() if hasattr(perturbed_point, 'tolist') else perturbed_point,
            'predicted_magnitude': pred_mag2,
            'break_mechanism': 'break_symmetry',
            'difficulty': 'medium',
        })

        return conditions

    def scan_for_dark_zones_in_executor_network(
        self, executors: list, bounds: list[tuple[float, float]],
        n_points: int = 40) -> list[DarkZoneCluster]:
        """Scan for dark zones formed by executor constraints.

        Converts executors into equivalent rules, then runs the standard scan.
        """
        rules = []
        for executor in executors:
            def make_constraint_fn(ex=executor):
                def fn(p):
                    uc = np.ones_like(p)  # Unit upper constraint
                    lc = ex.transmit(uc, p)
                    return float(np.linalg.norm(lc))
                return fn

            rule = Rule(
                name=f"{executor.id}: {executor.name}",
                layer=executor.from_layer,
                domain='executor',
                constraint_fn=make_constraint_fn(executor),
                certainty=executor.certainty,
            )
            rules.append(rule)

        field = ConstraintField(rules=rules)
        return self.scan(field, bounds, n_points)
