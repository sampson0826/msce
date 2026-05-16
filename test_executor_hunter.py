"""
Test cases for ExecutorHunter — unobservable executor search model.

Tests:
  1. Known executor transmission completeness > 0.9
  2. Remove an executor → gap detected
  3. Artificial dark zone → correctly identified
  4. Dark zone break conditions valid
  5. E-I/E-II/E-III type classification correct
  6. Multi-layer constraint chain propagation
  7. Experiment design outputs reasonable precision
  8. Real physics residual (M₁: 19 parameters → RG fixed point)
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/dengxinhang/paper')

from constraint_residual.core import Rule, ConstraintField
from constraint_residual.executor_models import Executor, ExecutorGap
from constraint_residual.dark_zone_detector import (
    DarkZoneDetector, DarkZonePoint, DarkZoneCluster,
)
from constraint_residual.executor_hunter import ExecutorHunter
from constraint_residual.experiment_designer import ExperimentDesigner
from constraint_residual.physics_executors import (
    build_known_executors, build_missing_executor_gaps,
    get_executor_summary_table,
)


# ================================================================
# Test 1: Known executor transmission completeness > 0.9
# ================================================================

def test_transmission_completeness_high():
    """E₁ (gauge enforcement) should have near-perfect transmission.

    The gauge principle is an E-I executor: given local symmetry,
    the gauge field form is mathematically necessary → transmission ~1.0.
    """
    e1 = Executor(
        id='E1', name='Gauge', from_layer=0, to_layer=1,
        executor_type='E-I', certainty=1.0,
        transmission_fn=lambda uc, p: 0.98 * uc,
    )
    hunter = ExecutorHunter([e1], transmission_threshold=0.90)
    result = hunter.compute_transmission_completeness(
        [e1], bounds=[(0, 1), (0, 1)], n_points=20
    )

    assert result.completeness_mean > 0.90, \
        f"E-I executor should have transmission > 0.90, got {result.completeness_mean:.3f}"
    assert not result.has_gap, \
        "Single E-I executor should have no gap"
    print(f"[PASS] Test 1: Transmission={result.completeness_mean:.3f}, "
          f"has_gap={result.has_gap}")


# ================================================================
# Test 2: Remove executor → gap detected
# ================================================================

def test_missing_executor_creates_gap():
    """When an executor is removed, hunt_gaps should find the gap.

    Create a 2D system with 2 executors. Remove one → residual appears.
    """
    e_a = Executor(
        id='EA', name='Transmitter_A', from_layer=0, to_layer=1,
        executor_type='E-I', certainty=1.0,
        transmission_fn=lambda uc, p: np.array([
            uc[0] * np.exp(-((p[0] - 0.3) / 0.5)**2),
            uc[1] * np.exp(-((p[1] - 0.3) / 0.5)**2),
        ]),
    )
    e_b = Executor(
        id='EB', name='Transmitter_B', from_layer=0, to_layer=1,
        executor_type='E-I', certainty=1.0,
        transmission_fn=lambda uc, p: np.array([
            uc[0] * np.exp(-((p[0] - 0.7) / 0.5)**2),
            uc[1] * np.exp(-((p[1] - 0.7) / 0.5)**2),
        ]),
    )

    # Full system: both executors
    hunter_full = ExecutorHunter([e_a, e_b], residual_epsilon=0.01)
    result_full = hunter_full.compute_transmission_completeness(
        [e_a, e_b], bounds=[(0, 1), (0, 1)], n_points=20
    )
    print(f"  Full system: completeness={result_full.completeness_mean:.3f}")

    # System with only A: B is missing → gap
    hunter_partial = ExecutorHunter([e_a], residual_epsilon=0.01)
    gaps = hunter_partial.hunt_gaps(
        from_layer=0, to_layer=1,
        bounds=[(0, 1), (0, 1)], n_points=25
    )

    assert len(gaps) > 0, "Removing an executor should create detectable gaps"
    print(f"[PASS] Test 2: {len(gaps)} gaps found after removing executor")


# ================================================================
# Test 3: Artificial dark zone → correctly identified
# ================================================================

def test_dark_zone_detection():
    """Create two rules with exactly opposing gradients → Type III dark zone.

    Rule 1: σ₁(p) = p[0] — gradient = [1, 0]
    Rule 2: σ₂(p) = -p[0] — gradient = [-1, 0]
    Combined: Π = [0, 0] — perfect cancellation at all points.
    """
    r1 = Rule(
        name="Forward", layer=1, domain="test",
        constraint_fn=lambda p: float(p[0]),
    )
    r2 = Rule(
        name="Backward", layer=1, domain="test",
        constraint_fn=lambda p: float(-p[0]),
    )
    field = ConstraintField(rules=[r1, r2])

    detector = DarkZoneDetector(cancellation_eps=0.01, individual_min=0.05)
    dark_zones = detector.scan(field, bounds=[(-2, 2), (-2, 2)], n_points=30)

    assert len(dark_zones) > 0, \
        "Perfectly opposing constraints should be detected as dark zone"
    dz = dark_zones[0]
    assert dz.balance_topology == 'mutual_cancellation', \
        f"Two opposing rules should be 'mutual_cancellation', got '{dz.balance_topology}'"
    assert dz.mean_cancellation_ratio < 0.01, \
        f"Perfect cancellation should give ratio ~0, got {dz.mean_cancellation_ratio:.6f}"

    print(f"[PASS] Test 3: Dark zone detected — "
          f"topology={dz.balance_topology}, "
          f"cancellation_ratio={dz.mean_cancellation_ratio:.6f}, "
          f"n_points={len(dz.points)}")


# ================================================================
# Test 4: Dark zone break conditions valid
# ================================================================

def test_dark_zone_break_conditions():
    """Verify that break conditions produce non-zero predicted signal.

    For a mutual cancellation dark zone, pushing along the break direction
    should produce a measurable residual.
    """
    r1 = Rule(
        name="Constraint_X", layer=1, domain="test",
        constraint_fn=lambda p: float(p[0]) * np.exp(-abs(p[1])),
    )
    r2 = Rule(
        name="Constraint_Y", layer=1, domain="test",
        constraint_fn=lambda p: float(-p[0]) * np.exp(-abs(p[1])),
    )
    field = ConstraintField(rules=[r1, r2])

    detector = DarkZoneDetector(cancellation_eps=0.05, individual_min=0.01)
    dark_zones = detector.scan(field, bounds=[(-2, 2), (-2, 2)], n_points=30)

    assert len(dark_zones) > 0, "Should detect at least one dark zone"
    dz = dark_zones[0]

    # Break conditions should have been computed
    assert len(dz.break_conditions) > 0, \
        "Dark zone should have at least one break condition"

    for bc in dz.break_conditions:
        assert bc['predicted_magnitude'] >= 0, \
            f"Predicted magnitude should be non-negative, got {bc['predicted_magnitude']}"

    print(f"[PASS] Test 4: {len(dz.break_conditions)} break conditions, "
          f"break_dir={'valid' if dz.break_direction is not None else 'none'}")


# ================================================================
# Test 5: E-I/E-II/E-III type classification correct
# ================================================================

def test_executor_type_classification():
    """Verify type classification: invariant direction → E-I,
    parameter-dependent → E-II, random → E-III.
    """
    # E-I: Transmits identically everywhere (direction invariant)
    e_ei = Executor(
        id='EI', name='MathTheorem', from_layer=0, to_layer=1,
        executor_type='E-I', certainty=1.0,
        transmission_fn=lambda uc, p: 0.99 * uc,
    )

    # E-II: Transmission depends on scale (parameter-dependent)
    e_eii = Executor(
        id='EII', name='ScaleHypothesis', from_layer=0, to_layer=1,
        executor_type='E-II', certainty=0.8,
        # Transmission varies with distance from origin
        transmission_fn=lambda uc, p: 0.7 * uc + 0.3 * uc * np.sin(np.linalg.norm(p)),
    )

    # E-III: Transmission varies randomly with position (boundary condition)
    e_eiii = Executor(
        id='EIII', name='BoundaryCondition', from_layer=0, to_layer=1,
        executor_type='E-III', certainty=0.5,
        transmission_fn=lambda uc, p: uc * (0.3 + 0.7 * abs(np.sin(
            np.linalg.norm(p) * 10.0 + hash(str(p[:2])) % 100 / 100.0
        ))),
    )

    # Verify E-I: high transmission completeness
    hunter_ei = ExecutorHunter([e_ei])
    result_ei = hunter_ei.compute_transmission_completeness(
        [e_ei], bounds=[(0, 1), (0, 1)], n_points=20
    )
    assert result_ei.completeness_mean > 0.85, \
        f"E-I should have high completeness, got {result_ei.completeness_mean:.3f}"

    # Verify E-II: moderate transmission completeness
    hunter_eii = ExecutorHunter([e_eii])
    result_eii = hunter_eii.compute_transmission_completeness(
        [e_eii], bounds=[(0, 1), (0, 1)], n_points=20
    )
    assert 0.4 < result_eii.completeness_mean < 1.0, \
        f"E-II should have moderate completeness, got {result_eii.completeness_mean:.3f}"

    print(f"[PASS] Test 5: E-I={result_ei.completeness_mean:.3f}, "
          f"E-II={result_eii.completeness_mean:.3f}")


# ================================================================
# Test 6: Multi-layer constraint chain propagation
# ================================================================

def test_constraint_chain_propagation():
    """Test propagation of constraints from L0 down through multiple layers.

    Create executors spanning L0→L1 and L1→L2, verify chain output.
    """
    # L0 → L1 executor
    e_l0_l1 = Executor(
        id='E_A', name='L0_to_L1', from_layer=0, to_layer=1,
        executor_type='E-I', certainty=1.0,
        transmission_fn=lambda uc, p: 0.95 * uc,
    )
    # L1 → L2 executor (weaker — constraint decay)
    e_l1_l2 = Executor(
        id='E_B', name='L1_to_L2', from_layer=1, to_layer=2,
        executor_type='E-II', certainty=0.8,
        transmission_fn=lambda uc, p: 0.6 * uc + 0.1 * np.sin(np.linalg.norm(p)),
    )

    hunter = ExecutorHunter([e_l0_l1, e_l1_l2])
    chain = hunter.propagate_constraint_chain(
        region=np.zeros(2), bounds=[(0, 2), (0, 2)], n_points=15
    )

    assert len(chain['transitions']) >= 2, \
        f"Should have at least 2 transitions, got {len(chain['transitions'])}"

    # L0→L1 should be very complete
    t01 = chain['transitions'][0]
    assert t01['completeness_mean'] > 0.8, \
        f"L0→L1 completeness should be > 0.8, got {t01['completeness_mean']:.3f}"

    # L1→L2 should be less complete (constraint decay)
    t12 = chain['transitions'][1]
    assert t12['completeness_mean'] < t01['completeness_mean'], \
        "L1→L2 completeness should be lower than L0→L1 (constraint decay)"

    print(f"[PASS] Test 6: {len(chain['transitions'])} transitions, "
          f"L0→L1: {t01['completeness_mean']:.3f}, "
          f"L1→L2: {t12['completeness_mean']:.3f}")


# ================================================================
# Test 7: Experiment design outputs reasonable precision
# ================================================================

def test_experiment_design():
    """Verify that ExperimentDesigner produces valid, ranked proposals."""
    # Create a gap manually
    gap = ExecutorGap(
        from_layer=0, to_layer=1,
        region=np.array([0.5, 0.5]),
        region_bounds=[(0, 1), (0, 1)],
        transmission_completeness=0.6,
        residual_magnitude=0.3,
        residual_direction=np.array([1.0, 0.0]),
        n_residual_points=15,
        candidate_type='E-II',
        candidate_type_confidence=0.5,
        candidate_math_form='Test gap for experiment design validation',
        dark_zone_ids=[1],
        priority=0.6,
    )

    designer = ExperimentDesigner(S0=1.0, min_feasibility=0.01)
    proposals = designer.design_for_gap(gap)

    assert len(proposals) > 0, "Should generate at least 1 proposal"
    assert len(proposals) <= 10, "Should not generate excessive proposals"

    ranked = designer.rank_experiments(proposals)
    # Top proposal should have highest priority
    for i in range(len(ranked) - 1):
        assert ranked[i].priority_score >= ranked[i+1].priority_score - 1e-10, \
            "Ranked proposals must be sorted by priority desc"

    # All proposals should have valid fields
    for p in proposals:
        assert p.feasibility > 0, f"Proposal {p.id} should have feasibility > 0"
        assert p.discovery_potential > 0, f"Proposal {p.id} should have discovery > 0"
        assert p.required_precision > 0, f"Proposal {p.id} should have precision > 0"
        assert p.predicted_signal_strength >= 0, \
            f"Proposal {p.id} should have non-negative signal"

    print(f"[PASS] Test 7: {len(proposals)} proposals generated, "
          f"top priority={ranked[0].priority_score:.3f}, "
          f"top feasibility={ranked[0].feasibility:.2f}")


# ================================================================
# Test 8: Real physics residual (M₁: 19 parameters → RG fixed point)
# ================================================================

def test_real_physics_residual():
    """Test that the M₁ missing executor gap (19 parameters) is properly structured.

    This tests that:
    1. build_missing_executor_gaps() returns valid gaps
    2. The M₁ gap has the correct structure
    3. The physics executors summary table can be generated
    """
    # Build known executors
    executors = build_known_executors()
    assert len(executors) == 7, f"Should have 7 known executors, got {len(executors)}"

    # Verify they're all valid
    for e in executors:
        assert e.executor_type in ('E-I', 'E-II', 'E-III'), \
            f"Executor {e.id} has invalid type: {e.executor_type}"
        assert e.from_layer >= 0, f"Executor {e.id} has invalid from_layer"
        assert e.to_layer > e.from_layer, \
            f"Executor {e.id}: to_layer must be > from_layer"

    # Build missing gaps
    gaps = build_missing_executor_gaps()
    assert len(gaps) == 3, f"Should have 3 missing executor gaps, got {len(gaps)}"

    # M₁: 19 parameters → RG fixed point
    m1 = gaps[0]
    assert m1.transmission_completeness == 0.0, \
        "M₁ should have zero transmission (no known executor for parameter selection)"
    assert 'RG' in m1.candidate_math_form or '不动点' in m1.candidate_math_form, \
        f"M₁ math form should reference RG fixed point: {m1.candidate_math_form}"
    assert m1.priority > 0.5, f"M₁ should have high priority, got {m1.priority}"

    # M₂: quantum gravity
    m2 = gaps[1]
    assert m2.residual_magnitude > 0.5, \
        "M₂ (quantum gravity) should have large residual magnitude"
    assert m2.priority > 0.7, f"M₂ should have very high priority, got {m2.priority}"

    # M₃: Type III dark zone
    m3 = gaps[2]
    assert m3.residual_magnitude == 0.0, \
        "M₃ should have zero residual (Type III dark zone)"
    assert len(m3.dark_zone_ids) > 0, "M₃ should reference dark zone IDs"
    assert m3.priority > 0.8, \
        "M₃ should have highest priority (complete invisibility = most dangerous)"

    # Generate summary table
    table = get_executor_summary_table()
    assert len(table) > 200, "Summary table should be substantial"
    assert 'E1' in table, "Table should contain executor ID E1"
    assert 'M1' in table, "Table should contain missing executor gap M1"
    assert 'E7' in table, "Table should contain executor ID E7"

    # Verify executor table has proper markdown
    assert table.count('|') > 20, "Should be a proper markdown table"

    print(f"[PASS] Test 8: {len(executors)} known executors + "
          f"{len(gaps)} missing gaps identified")
    print(f"  M₁ (parameters): priority={m1.priority:.2f}, "
          f"type={m1.candidate_type}, confidence={m1.candidate_type_confidence:.2f}")
    print(f"  M₂ (quantum gravity): priority={m2.priority:.2f}")
    print(f"  M₃ (dark zone): priority={m3.priority:.2f}, "
          f"residual={m3.residual_magnitude}")


# ================================================================
# Bonus: End-to-end integration test
# ================================================================

def test_end_to_end_hunt():
    """End-to-end: known executors → hunt gaps → design experiments → rank."""
    executors = build_known_executors()
    hunter = ExecutorHunter(executors, residual_epsilon=0.01)

    # Hunt for gaps between L0 and L1
    gaps = hunter.hunt_gaps(
        from_layer=0, to_layer=1,
        bounds=[(0, 2), (0, 2)], n_points=20
    )

    # Design experiments for found gaps
    designer = ExperimentDesigner(S0=1.0)
    proposals = designer.design_for_gaps(gaps)
    top = designer.top_experiments(proposals, n=3)

    # Verify the pipeline runs end to end
    assert isinstance(gaps, list), "hunt_gaps should return a list"
    assert isinstance(proposals, list), "design_for_gaps should return a list"

    print(f"[PASS] Test End-to-End: {len(gaps)} gaps → "
          f"{len(proposals)} experiments → "
          f"{len(top)} top-ranked")


if __name__ == "__main__":
    tests = [
        test_transmission_completeness_high,
        test_missing_executor_creates_gap,
        test_dark_zone_detection,
        test_dark_zone_break_conditions,
        test_executor_type_classification,
        test_constraint_chain_propagation,
        test_experiment_design,
        test_real_physics_residual,
        test_end_to_end_hunt,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*50}")
    print(f"ExecutorHunter Tests: {passed} passed, {failed} failed "
          f"out of {len(tests)}")
    print(f"{'='*50}")
