"""
Experiment Designer — Design experiments to expose hidden executors.

Based on the stability-visibility exponential decay law:
  V(R) ∝ exp(-S(R)/S₀)

To make an invisible executor visible, you must either:
  1. Reduce S(R) — push to extreme conditions where the rule's stability weakens
  2. Increase sensitivity — improve measurement precision to detect smaller signals
  3. Break cross-constraint balance — disrupt Type III dark zones

The designer takes an ExecutorGap and generates concrete, ranked experiment
proposals with predicted signal strengths and required precisions.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from constraint_residual.core import Rule, ConstraintField
from constraint_residual.executor_models import (
    ExecutorGap, ExperimentProposal, DarkZoneRegion,
)


class ExperimentDesigner:
    """Design experiments to find hidden executors.

    Usage:
        from constraint_residual.executor_hunter import ExecutorHunter
        from constraint_residual.experiment_designer import ExperimentDesigner

        hunter = ExecutorHunter(executors)
        gaps = hunter.hunt_gaps(0, 1, [(0,1), (0,1)])

        designer = ExperimentDesigner(S0=1.0)
        proposals = designer.design_for_gaps(gaps)
        ranked = designer.rank_experiments(proposals)

        for p in ranked[:3]:
            print(p.summary())
    """

    def __init__(self, S0: float = 1.0, min_feasibility: float = 0.01):
        """
        Args:
            S0: Characteristic fluctuation scale for the system.
                Determines the stability-visibility relationship.
            min_feasibility: Minimum feasibility to even consider (0-1).
        """
        self.S0 = S0
        self.min_feasibility = min_feasibility

    def design_for_gaps(self, gaps: list[ExecutorGap]) -> list[ExperimentProposal]:
        """Design experiments for a list of executor gaps.

        Returns a flat list of ExperimentProposals across all gaps.
        """
        proposals = []
        for gap in gaps:
            proposals.extend(self.design_for_gap(gap))
        return proposals

    def design_for_gap(self, gap: ExecutorGap) -> list[ExperimentProposal]:
        """Design all viable experiments to expose a single executor gap.

        Strategies:
        1. Extreme extrapolation: follow the residual direction outward
        2. Precision enhancement: measure at higher resolution
        3. Symmetry breaking: perturb to break cross-constraint balance
        4. Interference: use known executors to amplify the residual
        """
        proposals = []
        dim = len(gap.region)

        # Strategy 1: Extreme extrapolation along residual direction
        if gap.residual_magnitude > 1e-10:
            proposals.append(self._design_extreme_extrapolation(gap, dim))

        # Strategy 2: Precision measurement at gap centroid
        proposals.append(self._design_precision_measurement(gap, dim))

        # Strategy 3: Symmetry breaking (only for dark zone gaps)
        if gap.dark_zone_ids:
            proposals.extend(self._design_symmetry_breaking(gap, dim))

        # Strategy 4: Constraint interference
        if gap.residual_magnitude > 1e-10:
            proposals.append(self._design_interference_experiment(gap, dim))

        # Compute priority for each
        for p in proposals:
            p.priority_score = (
                0.4 * p.feasibility + 0.6 * p.discovery_potential
            )

        return [p for p in proposals if p.feasibility >= self.min_feasibility]

    def _design_extreme_extrapolation(self, gap: ExecutorGap,
                                       dim: int) -> ExperimentProposal:
        """Push along the residual direction to extreme values.

        At higher energies/densities/curvatures, the balance of constraints
        may break, making the residual (and thus the hidden executor) detectable.
        """
        residual_dir = gap.residual_direction
        if residual_dir is None or np.linalg.norm(residual_dir) < 1e-10:
            residual_dir = np.ones(dim) / np.sqrt(dim)

        # Extrapolate: go 10× further in the residual direction
        extreme_point = gap.region + residual_dir * 10.0

        # Predicted signal: residual grows as we leave the balanced region
        # In stability-visibility law: V ∝ exp(-S/S0)
        # At extreme conditions, S decreases → V increases
        current_visibility = np.exp(-gap.residual_magnitude / self.S0)
        extreme_stability = gap.residual_magnitude * 0.3  # Weakened at extremes
        extreme_visibility = np.exp(-extreme_stability / self.S0)
        predicted_signal = extreme_visibility * gap.residual_magnitude * 10.0

        # Feasibility: depends on how extreme we need to go
        # Planck-scale experiments are infeasible
        extreme_magnitude = float(np.linalg.norm(extreme_point - gap.region))
        feasibility = np.exp(-extreme_magnitude / 50.0)

        # Discovery potential: high if the gap is fundamental
        discovery = 0.7 if 'Planck' in (gap.candidate_math_form or '') else 0.5

        return ExperimentProposal(
            id=f'extreme_{gap.from_layer}_{gap.to_layer}',
            gap=gap,
            experiment_type='extreme_energy',
            target_observable=f'Constraint residual at extreme region: '
                             f'L{gap.from_layer}→L{gap.to_layer}',
            target_region=extreme_point,
            required_precision=1.0 / (predicted_signal + 1e-6),
            predicted_signal_strength=predicted_signal,
            feasibility=feasibility,
            discovery_potential=discovery,
            rationale=(
                f'Extrapolate along residual direction to amplify signal. '
                f'Current visibility: {current_visibility:.2e}, '
                f'predicted at extreme: {extreme_visibility:.2e}. '
                f'Signal grows from {gap.residual_magnitude:.4f} to {predicted_signal:.4f}.'
            ),
        )

    def _design_precision_measurement(self, gap: ExecutorGap,
                                       dim: int) -> ExperimentProposal:
        """Improve measurement precision at the gap centroid.

        By the stability-visibility law, if the hidden rule has finite
        stability S, improving precision by factor f lets us see rules
        with S up to S0 * ln(f).
        """
        # Current residual sets the precision floor
        current_precision = gap.residual_magnitude + 1e-10

        # Required precision to see a rule of candidate stability
        # S_rule ≈ 1 / (gap.transmission_completeness + 1e-6)
        S_rule = 1.0 / (gap.transmission_completeness + 1e-6)
        required_visibility = np.exp(-S_rule / self.S0)
        required_precision = required_visibility * 0.1

        # Predicted signal: measuring at this precision should reveal
        # the hidden constraint direction
        predicted_signal = gap.residual_magnitude * 2.0

        # Feasibility: how achievable is this precision?
        # Current tech: can improve by ~10³ over current residual detection
        precision_ratio = current_precision / (required_precision + 1e-20)
        feasibility = min(1.0, np.log10(precision_ratio + 1e-10) / 6.0)

        return ExperimentProposal(
            id=f'precision_{gap.from_layer}_{gap.to_layer}',
            gap=gap,
            experiment_type='precision',
            target_observable=(
                f'High-precision measurement at gap centroid '
                f'(region = {gap.region[:min(3, dim)]})'
            ),
            target_region=gap.region,
            required_precision=required_precision,
            predicted_signal_strength=predicted_signal,
            feasibility=feasibility,
            discovery_potential=0.4,
            rationale=(
                f'Precision improvement by factor {1/required_precision:.1e} '
                f'over current detection floor. '
                f'Hidden rule stability S≈{S_rule:.1f} requires '
                f'visibility V≥{required_visibility:.2e} for detection.'
            ),
        )

    def _design_symmetry_breaking(self, gap: ExecutorGap,
                                   dim: int) -> list[ExperimentProposal]:
        """Break the symmetry maintaining a Type III dark zone.

        Type III dark zones exist because multiple constraints balance
        perfectly. By introducing an asymmetric perturbation, the balance
        breaks and the hidden rule becomes visible.
        """
        proposals = []

        # For each dimension, try an asymmetric perturbation
        for d in range(min(dim, 3)):
            perturbed = gap.region.copy()
            perturbed[d] += 5.0  # Significant perturbation

            # Breaking the symmetry should produce a measurable residual
            # The predicted signal magnitude depends on how "brittle" the balance is
            predicted_signal = 0.5 * (1.0 + 0.3 * d)

            proposals.append(ExperimentProposal(
                id=f'symbreak_{gap.from_layer}_{gap.to_layer}_d{d}',
                gap=gap,
                experiment_type='symmetry_breaking',
                target_observable=(
                    f'Constraint residual after asymmetric perturbation '
                    f'in dimension {d}'
                ),
                target_region=perturbed,
                required_precision=0.1 / predicted_signal,
                predicted_signal_strength=predicted_signal,
                feasibility=0.3,  # Moderately feasible — need controlled asymmetry
                discovery_potential=0.6,
                rationale=(
                    f'Break the Type III cross-constraint balance by introducing '
                    f'asymmetry in dimension {d}. The perfect cancellation among '
                    f'constraints should break, producing a measurable residual '
                    f'pointing to the hidden executor.'
                ),
            ))

        return proposals

    def _design_interference_experiment(self, gap: ExecutorGap,
                                         dim: int) -> ExperimentProposal:
        """Use known executor constraints to amplify the residual signal.

        If two known executors produce overlapping constraints near the gap,
        their interference pattern may reveal the missing executor's shadow.
        """
        # Sample two nearby points where known executors are strong
        p1 = gap.region + np.ones(dim) * 0.5
        p2 = gap.region - np.ones(dim) * 0.5

        # Interference: difference in constraint between two nearby points
        # can amplify a weak residual
        predicted_signal = gap.residual_magnitude * 3.0

        return ExperimentProposal(
            id=f'interference_{gap.from_layer}_{gap.to_layer}',
            gap=gap,
            experiment_type='interference',
            target_observable=(
                'Constraint gradient difference between two nearby regions '
                f'(Δp = {p1[:min(3,dim)]} vs {p2[:min(3,dim)]})'
            ),
            target_region=(p1 + p2) / 2,
            required_precision=1.0 / (predicted_signal + 1e-6),
            predicted_signal_strength=predicted_signal,
            feasibility=0.5,
            discovery_potential=0.3,
            rationale=(
                'Interference between known executor constraints at nearby '
                f'points amplifies the residual by constructive addition. '
                f'Signal predicted: {predicted_signal:.4f}.'
            ),
        )

    # ================================================================
    # Experiment ranking
    # ================================================================

    def rank_experiments(self, proposals: list[ExperimentProposal]
                         ) -> list[ExperimentProposal]:
        """Rank experiments by priority_score (descending), then by feasibility."""
        return sorted(
            proposals,
            key=lambda p: (p.priority_score, p.feasibility),
            reverse=True,
        )

    def top_experiments(self, proposals: list[ExperimentProposal],
                        n: int = 5) -> list[ExperimentProposal]:
        """Get the top N highest-priority experiments."""
        return self.rank_experiments(proposals)[:n]

    def generate_experiment_plan(self, gaps: list[ExecutorGap],
                                  n_top: int = 5) -> str:
        """Generate a readable experiment plan from executor gaps."""
        proposals = self.design_for_gaps(gaps)
        top = self.top_experiments(proposals, n_top)

        lines = [
            "# Experiment Plan: Exposing Hidden Executors",
            "",
            f"Total proposals generated: {len(proposals)}",
            f"Top {len(top)} experiments:",
            "",
        ]

        for i, p in enumerate(top):
            lines.append(f"## {i+1}. {p.id}")
            lines.append(f"**Type**: {p.experiment_type}")
            lines.append(f"**Target**: {p.target_observable}")
            lines.append(f"**Priority**: {p.priority_score:.3f}")
            lines.append(f"**Feasibility**: {p.feasibility:.2f} / **Discovery**: {p.discovery_potential:.2f}")
            lines.append(f"**Predicted signal**: {p.predicted_signal_strength:.4f}")
            lines.append(f"**Required precision**: {p.required_precision:.3e}")
            lines.append(f"**Rationale**: {p.rationale}")
            lines.append("")

        return "\n".join(lines)
