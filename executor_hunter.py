"""
ExecutorHunter — Main search engine for unobservable executors.

Core pipeline:
  1. Take known executors → build ConstraintField
  2. Scan for constraint residuals (gaps where transmission is incomplete)
  3. Scan for Type III dark zones (perfect cross-balance → zero signal)
  4. Merge gaps + dark zones → candidate ExecutorGap objects
  5. Characterize each gap: predict executor type (E-I/E-II/E-III),
     mathematical form, and constraint direction
  6. Rank gaps by priority (discovery potential × likelihood)

Key equation:
  Transmission completeness = Σ||lower_constraint|| / Σ||upper_constraint||
  over a sampled region. When < 1, some constraint is not being transmitted
  → a missing executor.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from constraint_residual.core import Rule, ConstraintField, ResidualDetector
from constraint_residual.executor_models import (
    Executor, ExecutorGap, DarkZoneRegion, TransmissionResult,
)
from constraint_residual.dark_zone_detector import (
    DarkZoneDetector, DarkZonePoint, DarkZoneCluster,
)


class ExecutorHunter:
    """Systematic search for unknown executors in a rule hierarchy.

    Usage:
        from constraint_residual.physics_executors import build_known_executors
        from constraint_residual.executor_hunter import ExecutorHunter

        executors = build_known_executors()
        hunter = ExecutorHunter(executors)
        gaps = hunter.hunt_gaps(from_layer=0, to_layer=1,
                                bounds=[(0,1), (0,1)], n_points=40)

        for gap in gaps:
            print(gap.summary())

        # Get the full constraint chain with all transmission metrics
        chain = hunter.propagate_constraint_chain(
            region=np.zeros(4), bounds=[(-5,5)]*4
        )
    """

    def __init__(self, known_executors: list[Executor],
                 transmission_threshold: float = 0.90,
                 residual_epsilon: float = 0.05,
                 dark_zone_cancellation_eps: float = 0.05):
        """
        Args:
            known_executors: List of known Executor objects
            transmission_threshold: Min transmission completeness to
                consider a layer transition "covered" (0-1).
            residual_epsilon: Threshold for ResidualDetector
            dark_zone_cancellation_eps: Threshold for DarkZoneDetector
        """
        self.known_executors = known_executors
        self.transmission_threshold = transmission_threshold
        self.residual_epsilon = residual_epsilon
        self.dark_zone_cancellation_eps = dark_zone_cancellation_eps

        # Build detectors
        self.residual_detector = ResidualDetector(
            field=self._build_constraint_field(known_executors),
            epsilon=residual_epsilon,
        )
        self.dark_detector = DarkZoneDetector(
            cancellation_eps=dark_zone_cancellation_eps,
        )

    def _build_constraint_field(self, executors: list[Executor]) -> ConstraintField:
        """Convert executors into a ConstraintField of equivalent Rules.

        Each executor's transmission_fn is wrapped: upper-layer unit constraint
        is fed in, and the lower-layer output magnitude is used as the
        constraint strength at each point.
        """
        rules = []
        for ex in executors:
            def make_fn(e=ex):
                def constraint_fn(p):
                    uc = np.ones(len(p))  # Unit upper constraint
                    lc = e.transmit(uc, p)
                    return float(np.linalg.norm(lc))
                return constraint_fn

            rules.append(Rule(
                name=f"{ex.id}:{ex.name}",
                layer=ex.from_layer,
                domain=f"L{ex.from_layer}→L{ex.to_layer}",
                constraint_fn=make_fn(ex),
                certainty=ex.certainty,
            ))
        return ConstraintField(rules=rules)

    def refresh_field(self):
        """Rebuild the internal constraint field (call after adding executors)."""
        self.residual_detector = ResidualDetector(
            field=self._build_constraint_field(self.known_executors),
            epsilon=self.residual_epsilon,
        )

    # ================================================================
    # Main hunting method
    # ================================================================

    def hunt_gaps(self, from_layer: int, to_layer: int,
                  bounds: list[tuple[float, float]],
                  n_points: int = 40) -> list[ExecutorGap]:
        """Hunt for executor gaps between two layers.

        Pipeline:
        1. Compute transmission completeness
        2. Scan constraint residuals
        3. Scan dark zones
        4. Merge → ExecutorGap objects
        5. Characterize each gap
        6. Rank by priority
        """
        # Only use executors relevant to this layer transition
        relevant = [e for e in self.known_executors
                    if e.from_layer == from_layer and e.to_layer == to_layer]

        if not relevant:
            # No known executor for this transition → entire transition is a gap
            return [ExecutorGap(
                from_layer=from_layer, to_layer=to_layer,
                region=np.array([(b[0] + b[1]) / 2 for b in bounds]),
                region_bounds=bounds,
                transmission_completeness=0.0,
                residual_magnitude=1.0,
                residual_direction=np.ones(len(bounds)) / np.sqrt(len(bounds)),
                n_residual_points=n_points ** len(bounds),
                candidate_type='E-I',
                candidate_type_confidence=0.2,
                candidate_math_form='Entire layer transition has no known executor',
                priority=1.0,
            )]

        # Build field for this transition
        field = self._build_constraint_field(relevant)
        self.residual_detector.field = field
        self.residual_detector.epsilon = self.residual_epsilon

        # Step 1: Transmission completeness
        trans_result = self.compute_transmission_completeness(
            relevant, bounds, n_points
        )

        # Step 2: Residual scan
        residual_detector = ResidualDetector(field, epsilon=self.residual_epsilon)
        residuals = residual_detector.scan_grid(bounds, n_points=n_points)
        residual_clusters = residual_detector.cluster_residuals(residuals)

        # Step 3: Dark zone scan
        dark_clusters = self.dark_detector.scan(field, bounds, n_points=n_points)

        # Step 4: Merge into ExecutorGaps
        gaps = []

        # Gaps from residual clusters
        for rc in residual_clusters:
            gap = ExecutorGap(
                from_layer=from_layer, to_layer=to_layer,
                region=rc.centroid if rc.centroid is not None else np.zeros(len(bounds)),
                region_bounds=bounds,
                transmission_completeness=trans_result.completeness_mean,
                residual_magnitude=rc.mean_magnitude,
                residual_direction=rc.mean_direction if rc.mean_direction is not None
                else np.zeros(len(bounds)),
                n_residual_points=len(rc.points),
                candidate_constraint_direction=rc.candidate_rule_constraint(),
            )
            gaps.append(gap)

        # Gaps from dark zones (these may overlap with residual gaps)
        dark_zone_ids_used = []
        for dc in dark_clusters:
            dzr = DarkZoneRegion(
                id=dc.id,
                centroid=dc.centroid if dc.centroid is not None else np.zeros(len(bounds)),
                bounds=bounds,
                n_points=len(dc.points),
                constraints_involved=dc.constraints_involved,
                cancellation_ratio=dc.mean_cancellation_ratio,
                balance_topology=dc.balance_topology,
                break_conditions=dc.break_conditions,
                suspected_hidden_content=(
                    f"Unknown executor hidden by {dc.balance_topology} "
                    f"among {dc.constraints_involved}"
                ),
            )

            # Check if this dark zone is already covered by a residual gap
            is_covered = any(
                np.linalg.norm(dzr.centroid - g.region) < 0.5
                for g in gaps
            )

            if not is_covered:
                gap = ExecutorGap(
                    from_layer=from_layer, to_layer=to_layer,
                    region=dzr.centroid,
                    region_bounds=bounds,
                    transmission_completeness=1.0,  # Appears complete!
                    residual_magnitude=0.0,  # Zero residual — the danger
                    residual_direction=dc.break_direction
                    if dc.break_direction is not None
                    else np.zeros(len(bounds)),
                    n_residual_points=0,
                    dark_zone_ids=[dc.id],
                )
                gaps.append(gap)

            dark_zone_ids_used.append(dc.id)

        # Step 5: Characterize each gap
        for gap in gaps:
            self.characterize_executor_type(gap)

        # Step 6: Rank by priority
        for gap in gaps:
            gap.priority = self._compute_priority(gap, trans_result)

        gaps.sort(key=lambda g: g.priority, reverse=True)
        return gaps

    # ================================================================
    # Transmission completeness
    # ================================================================

    def compute_transmission_completeness(
        self, executors: list[Executor],
        bounds: list[tuple[float, float]],
        n_points: int = 30,
    ) -> TransmissionResult:
        """Compute how completely constraint is transmitted through executors.

        For each sampled point:
        1. Create unit upper constraint
        2. Transmit through each executor
        3. Sum the transmitted constraints → lower constraint
        4. Transmission ratio = ||lower|| / ||upper||
        """
        dims = len(bounds)
        axes = [np.linspace(lo, hi, n_points) for lo, hi in bounds]
        mesh = np.meshgrid(*axes, indexing='ij')

        ratios = []
        residual_mags = []

        for idx in np.ndindex(mesh[0].shape):
            p = np.array([mesh[d][idx] for d in range(dims)], dtype=float)
            upper = np.ones(dims)

            # Sum all transmitted constraints
            lower = np.zeros(dims)
            for ex in executors:
                lower += ex.transmit(upper.copy(), p)

            upper_norm = float(np.linalg.norm(upper))
            lower_norm = float(np.linalg.norm(lower))

            if upper_norm > 1e-12:
                ratios.append(min(lower_norm / upper_norm, 1.0))
                # Residual = what wasn't transmitted
                residual_mags.append(abs(upper_norm - lower_norm))

        if not ratios:
            return TransmissionResult(
                from_layer=executors[0].from_layer if executors else -1,
                to_layer=executors[0].to_layer if executors else -1,
                known_executors=[e.id for e in executors],
            )

        return TransmissionResult(
            from_layer=executors[0].from_layer if executors else -1,
            to_layer=executors[0].to_layer if executors else -1,
            known_executors=[e.id for e in executors],
            n_points_sampled=len(ratios),
            n_points_complete=sum(1 for r in ratios if r >= self.transmission_threshold),
            completeness_mean=float(np.mean(ratios)),
            completeness_min=float(np.min(ratios)),
            completeness_std=float(np.std(ratios)),
            residual_magnitudes=residual_mags,
        )

    # ================================================================
    # Executor type characterization
    # ================================================================

    def characterize_executor_type(self, gap: ExecutorGap):
        """Predict whether the missing executor is E-I, E-II, or E-III.

        Heuristic:
        1. Sample constraint behavior near the gap centroid
        2. Measure how much the residual direction varies with small
           parameter changes
        3. Low variance → E-I (mathematical necessity — the constraint
           direction is invariant)
        4. Predictable variance → E-II (scale hypothesis — constraint
           changes with scale)
        5. High/unpredictable variance → E-III (boundary condition —
           constraint is accidental)
        """
        if gap.residual_magnitude < 1e-10:
            # Dark zone gap
            gap.candidate_type = 'E-I'
            gap.candidate_type_confidence = 0.15  # Very speculative
            gap.candidate_math_form = (
                'Unknown rule in Type III dark zone — perfect cross-constraint balance. '
                'Likely E-I (mathematical structure that exactly cancels known constraints). '
                'Requires extreme conditions (Planck energy, high curvature) to expose.'
            )
            return

        # Sample around centroid at small perturbations
        centroid = gap.region
        dim = len(centroid)
        n_samples = min(50, 10 * dim)
        perturbations = np.random.randn(n_samples, dim) * 0.1

        sampled_directions = []
        for i in range(n_samples):
            p = centroid + perturbations[i]
            # Compute what the constraint residual would be at this point
            grad = self.residual_detector.field.constraint_gradient(p)
            mag = float(np.linalg.norm(grad))
            if mag > 1e-12:
                sampled_directions.append(grad / mag)

        if len(sampled_directions) < 3:
            gap.candidate_type = 'E-III'
            gap.candidate_type_confidence = 0.3
            gap.candidate_math_form = 'Insufficient data for characterization'
            return

        # Compute angular variance
        mean_dir = np.mean(sampled_directions, axis=0)
        mean_dir_norm = float(np.linalg.norm(mean_dir))
        if mean_dir_norm < 1e-12:
            gap.candidate_type = 'E-III'
            gap.candidate_type_confidence = 0.3
            return

        mean_dir = mean_dir / mean_dir_norm
        cos_sims = [float(np.dot(d, mean_dir)) for d in sampled_directions]
        mean_cos = float(np.mean(cos_sims))
        std_cos = float(np.std(cos_sims))

        # Classification logic
        if mean_cos > 0.95 and std_cos < 0.05:
            # Direction is invariant → mathematical necessity
            gap.candidate_type = 'E-I'
            gap.candidate_type_confidence = mean_cos * (1.0 - std_cos)
            gap.candidate_math_form = (
                f'Mathematical theorem-type executor. '
                f'Constraint direction is invariant under parameter variation '
                f'(cos stability={mean_cos:.3f}, std={std_cos:.4f}). '
                f'Likely a symmetry or conservation principle not yet identified.'
            )
        elif mean_cos > 0.7 and std_cos < 0.3:
            # Direction varies predictably → scale hypothesis
            gap.candidate_type = 'E-II'
            gap.candidate_type_confidence = mean_cos * 0.8
            gap.candidate_math_form = (
                f'Scale hypothesis-type executor. '
                f'Constraint direction is stable but parameter-dependent '
                f'(cos stability={mean_cos:.3f}, std={std_cos:.4f}). '
                f'Likely involves an RG flow with an empirical fixed point.'
            )
        else:
            # Direction varies strongly → boundary condition
            gap.candidate_type = 'E-III'
            gap.candidate_type_confidence = 1.0 - mean_cos if mean_cos < 0.5 else 0.3
            gap.candidate_math_form = (
                f'Boundary condition-type executor. '
                f'Constraint direction varies significantly with parameters '
                f'(cos stability={mean_cos:.3f}, std={std_cos:.4f}). '
                f'Likely an accidental feature of the specific parameter regime.'
            )

    def _compute_priority(self, gap: ExecutorGap,
                          trans_result: TransmissionResult) -> float:
        """Compute priority score for a gap.

        Priority = (1 - transmission_completeness) × residual_magnitude
                   × (1 + n_residual_points/100) × type_confidence_bonus
        """
        # Base: how incomplete is transmission
        incompleteness = 1.0 - trans_result.completeness_mean

        # Dark zone bonus: these are the most dangerous
        dark_zone_bonus = 1.5 if gap.dark_zone_ids else 1.0

        # Candidate type confidence penalty for speculative calls
        type_factor = 0.5 + 0.5 * gap.candidate_type_confidence

        priority = (
            incompleteness
            * (gap.residual_magnitude + 0.1)
            * (1.0 + gap.n_residual_points / 200)
            * dark_zone_bonus
            * type_factor
        )
        return min(priority, 1.0)

    # ================================================================
    # Constraint chain propagation
    # ================================================================

    def propagate_constraint_chain(
        self, region: np.ndarray,
        bounds: list[tuple[float, float]],
        n_points: int = 20,
    ) -> dict:
        """Propagate constraints from L0 down to L3 through all known executors.

        Returns a dict with transmission results for each layer transition,
        annotated with gap severities.
        """
        layers = sorted(set(e.from_layer for e in self.known_executors) |
                        set(e.to_layer for e in self.known_executors))

        chain = {
            'layers': layers,
            'transitions': [],
            'total_gap_severity': 0.0,
            'n_transitions_with_gaps': 0,
        }

        for i in range(len(layers) - 1):
            from_l = layers[i]
            to_l = layers[i + 1]
            relevant = [e for e in self.known_executors
                       if e.from_layer == from_l and e.to_layer == to_l]

            result = self.compute_transmission_completeness(
                relevant, bounds, n_points
            )
            chain['transitions'].append({
                'from_layer': from_l,
                'to_layer': to_l,
                'n_executors': len(relevant),
                'executor_ids': [e.id for e in relevant],
                'completeness_mean': result.completeness_mean,
                'completeness_min': result.completeness_min,
                'has_gap': result.has_gap,
                'gap_severity': result.gap_severity,
            })

            if result.has_gap:
                chain['n_transitions_with_gaps'] += 1
                chain['total_gap_severity'] += result.gap_severity

        return chain

    # ================================================================
    # Executor topology mapping
    # ================================================================

    def map_executor_topology(self) -> dict:
        """Map the topology of the executor network.

        Returns:
            dict with 'nodes' (layers), 'edges' (executors),
            'missing_edges' (where gaps are suspected).
        """
        nodes = sorted(set(e.from_layer for e in self.known_executors) |
                       set(e.to_layer for e in self.known_executors))

        edges = []
        for e in self.known_executors:
            edges.append({
                'id': e.id,
                'from': e.from_layer,
                'to': e.to_layer,
                'type': e.executor_type,
                'certainty': e.certainty,
            })

        # Identify missing transitions
        existing_transitions = set()
        for e in self.known_executors:
            existing_transitions.add((e.from_layer, e.to_layer))

        all_possible = []
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                all_possible.append((nodes[i], nodes[j]))

        missing = [(f, t) for f, t in all_possible
                   if (f, t) not in existing_transitions and f < t]

        return {
            'nodes': nodes,
            'edges': edges,
            'missing_transitions': [{'from': f, 'to': t} for f, t in missing],
        }
