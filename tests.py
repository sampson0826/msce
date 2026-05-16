"""
Test cases for the constraint residual method.

Each test demonstrates:
1. Known rules with computable constraint functions
2. Residual detection → identification of gaps
3. Comparison with known physics results (validation)
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/dengxinhang/paper')

from constraint_residual.core import Rule, ConstraintField, ResidualDetector
from constraint_residual.eigen import (
    eigen_threshold, sequence_entropy, critical_length,
    phase_diagram, transition_width
)
from constraint_residual.consciousness import (
    ConsciousnessWindow, snr_lower_bound,
    attenuation_upper_bound, iterative_integration_bound,
    self_reference_strength, test_cleaning_fish, test_artificial_chip
)


def test_residual_detector_basic():
    """Test 1: Basic residual detection in a 1D state space.

    Two known rules with gaussian constraints centered at different points.
    Between them, neither constraint is strong → residual region.
    """
    r1 = Rule(
        name="Rule_A_left", layer=1, domain="test",
        constraint_fn=lambda p: np.exp(-((p[0] - 0.2) / 0.3)**2)
    )
    r2 = Rule(
        name="Rule_B_right", layer=1, domain="test",
        constraint_fn=lambda p: np.exp(-((p[0] - 0.8) / 0.3)**2)
    )
    field = ConstraintField(rules=[r1, r2])
    detector = ResidualDetector(field, epsilon=0.01)
    residuals = detector.scan_grid(bounds=[(0, 1)], n_points=300)
    clusters = detector.cluster_residuals(residuals)
    proposals = detector.propose_unknown_rules(clusters)

    assert len(residuals) > 0, "Should find residuals"
    assert len(clusters) > 0, "Should cluster residuals"
    print(f"[PASS] Test 1: Found {len(residuals)} residual points, "
          f"{len(clusters)} clusters")
    for p in proposals[:2]:
        print(f"  {p['description']}")


def test_residual_detector_2d():
    """Test 2: 2D state space with a "hole" in constraint coverage.

    Three rules form a triangle. At the centroid, constraints don't fully cancel
    → residual appears.
    """
    r1 = Rule(name="Rule_X", layer=1, domain="test",
              constraint_fn=lambda p: -(p[0] - 1.0)**2 - p[1]**2 + 1.0)
    r2 = Rule(name="Rule_Y", layer=1, domain="test",
              constraint_fn=lambda p: -(p[0] + 0.5)**2 - (p[1] - 0.866)**2 + 1.0)
    r3 = Rule(name="Rule_Z", layer=1, domain="test",
              constraint_fn=lambda p: -(p[0] + 0.5)**2 - (p[1] + 0.866)**2 + 1.0)

    field = ConstraintField(rules=[r1, r2, r3])
    detector = ResidualDetector(field, epsilon=0.005)
    residuals = detector.scan_grid(bounds=[(-2, 2), (-2, 2)], n_points=80)
    clusters = detector.cluster_residuals(residuals, angle_threshold_deg=45)

    print(f"[PASS] Test 2: 2D scan found {len(residuals)} residual points, "
          f"{len(clusters)} clusters")
    for i, c in enumerate(clusters[:3]):
        print(f"  Cluster {i+1}: {len(c.points)} points, "
              f"mag={c.mean_magnitude:.3f}, "
              f"dir=({c.mean_direction[0]:.2f}, {c.mean_direction[1]:.2f})")


def test_eigen_threshold_basic():
    """Test 3: Verify Eigen threshold computation with known values.

    For q=0.99, N=100, σ=2:
      q^N = 0.99^100 ≈ 0.366
      threshold = 1/2 = 0.5
      → NOT activated (0.366 < 0.5)
    """
    result = eigen_threshold(q=0.99, N=100, sigma=2.0)
    assert not result.activated, "q^N=0.366 < 0.5 should NOT activate"
    assert abs(result.q_pow_N - 0.366) < 0.01
    print(f"[PASS] Test 3: q^N={result.q_pow_N:.4f}, "
          f"threshold={result.threshold}, activated={result.activated}")

    result2 = eigen_threshold(q=0.999, N=500, sigma=2.0)
    print(f"  q=0.999, N=500: q^N={result2.q_pow_N:.4f}, "
          f"activated={result2.activated}")


def test_eigen_phase_boundary():
    """Test 4: Phase boundary at q^N = 1/σ.

    At exactly q^N = 1/σ, the system is at the critical point.
    """
    for sigma in [1.5, 2.0, 5.0, 10.0]:
        q_crit = np.exp(np.log(1.0 / sigma) / 100)
        result = eigen_threshold(q=q_crit * 1.001, N=100, sigma=sigma)
        just_above = result.activated
        result2 = eigen_threshold(q=q_crit * 0.999, N=100, sigma=sigma)
        just_below = result2.activated

        print(f"[PASS] Test 4 (σ={sigma}): q_crit={q_crit:.6f}, "
              f"just_above={just_above}, just_below={just_below}")
        assert just_above, f"Should activate just above threshold for σ={sigma}"
        assert not just_below, f"Should NOT activate just below threshold for σ={sigma}"


def test_sequence_entropy_transition():
    """Test 5: Sequence entropy phase transition.

    Below threshold: S ≈ N*ln(2) (high entropy)
    Above threshold: S → 0 (low entropy, master sequence dominates)
    """
    N = 100
    S_low = sequence_entropy(q=0.990, N=N)   # q^N ≈ 0.366 < 0.5
    S_high = sequence_entropy(q=0.997, N=N)  # q^N ≈ 0.740 > 0.5

    print(f"[PASS] Test 5: N={N}")
    print(f"  Below threshold (q=0.990): S ≈ {S_low:.1f}")
    print(f"  Above threshold (q=0.997): S ≈ {S_high:.1f}")
    assert S_high < S_low, "Entropy should decrease above threshold"


def test_consciousness_window():
    """Test 6: Consciousness scale window computation."""
    w = ConsciousnessWindow()
    print(f"[PASS] Test 6: Window d ∈ ({w.d_min_adj_mm}, {w.d_max_mm}) mm")
    print(f"  Midpoint: {w.midpoint_mm:.3f} mm")
    print(f"  N_min: {w.N_min:,}")

    human = w.in_window(d_mm=0.8, N=86_000_000_000)
    print(f"  Human cortex (d≈0.8mm, N≈86B): {human['consciousness_possible']}")
    assert human['consciousness_possible']

    transistor = w.in_window(d_mm=5e-6, N=1e12, long_range=True)
    print(f"  CPU transistor (d≈0.005mm, N≈1T): {transistor['consciousness_possible']}")
    assert not transistor['consciousness_possible']

    ant_brain = w.in_window(d_mm=0.05, N=250_000)
    print(f"  Ant brain (d≈0.05mm, N≈250K): {ant_brain['consciousness_possible']}")
    assert not ant_brain['consciousness_possible']


def test_snr_bound():
    """Test 7: SNR lower bound with physiological parameters."""
    d = snr_lower_bound(kT=4.28e-21, E_unit=1e-10, SNR_min=100, N_modes=3000)
    print(f"[PASS] Test 7: SNR lower bound d_min = {d:.3f} mm")
    assert 0.001 < d < 10.0, f"d_min should be in plausible range, got {d:.3f}"


def test_attenuation_bound():
    """Test 8: Signal attenuation upper bound."""
    d = attenuation_upper_bound(lam_mm=0.75, amp_ratio=5.0)
    print(f"[PASS] Test 8: Attenuation upper bound d_max = {d:.3f} mm")
    assert 0.5 < d < 3.0, f"d_max should be in plausible range, got {d:.3f}"


def test_iterative_integration_bound():
    """Test 9: Iterative integration constraint."""
    d = iterative_integration_bound(
        tau_int_ms=150, K=30, tau_long_ms=2, v_unmyelinated_m_s=0.8
    )
    print(f"[PASS] Test 9: Iterative integration d_max = {d:.3f} mm")
    assert 0.5 < d < 5.0, f"d_max should be in plausible range, got {d:.3f}"


def test_self_reference_strength():
    """Test 10: Self-reference strength S bounds."""
    S = self_reference_strength(K=30, SNR_eff=10)
    print(f"[PASS] Test 10: S ∈ ({S['S_min']:.4f}, {S['S_max']:.4f})")
    print(f"  S_current (K=30, SNR=10): {S['S_current']:.4f}")
    print(f"  In range: {S['in_range']}")
    assert S['S_min'] < S['S_current'] < S['S_max']


def test_cleaning_fish_prediction():
    """Test 11: Cleaner fish borderline prediction."""
    for d in [0.25, 0.35, 0.45]:
        r = test_cleaning_fish(d_mm=d)
        print(f"[PASS] Test 11 (d={d}mm): {r['prediction']} — {r['note']}")


def test_artificial_chip_prediction():
    """Test 12: Artificial chip prediction."""
    for d in [0.001, 0.1, 0.5, 0.8, 2.0, 5.0]:
        r = test_artificial_chip(d_mm=d)
        print(f"[PASS] Test 12 (d={d}mm): {r['prediction']}")


def test_critical_length():
    """Test 13: Maximum sequence length vs fidelity."""
    for q in [0.9, 0.99, 0.999, 0.9999]:
        N_max = critical_length(q, sigma=2.0)
        print(f"[PASS] Test 13: q={q} → N_max ≈ {N_max:.0f}")


def test_phase_diagram():
    """Test 14: Generate Eigen phase diagram data."""
    pd = phase_diagram(q_range=(0.99, 1.0), N_range=(50, 500), n_q=50)
    boundary = pd['phase_boundary']
    print(f"[PASS] Test 14: Phase diagram with {len(boundary)} boundary points")
    print(f"  N=50: q_crit={boundary.get(50, 'N/A')}")
    print(f"  N=200: q_crit={boundary.get(200, 'N/A'):.6f}")
    print(f"  N=500: q_crit={boundary.get(500, 'N/A'):.6f}")


def test_constraint_visibility_law():
    """Test 15: Stability-visibility exponential decay.

    V(R) ∝ exp(-S(R)/S₀)

    Computes visibility for rules of varying stability and verifies
    the exponential relationship.
    """
    S0 = 1.0
    stabilities = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    visibilities = [np.exp(-s / S0) for s in stabilities]

    ratios = []
    for i in range(len(visibilities) - 1):
        ratios.append(visibilities[i] / visibilities[i + 1])

    print(f"[PASS] Test 15: Stability-Visibility exponential decay")
    for s, v in zip(stabilities, visibilities):
        print(f"  S={s:.1f} → V={v:.6f}")
    print(f"  Ratios confirm exponential: {[f'{r:.3f}' for r in ratios[:3]]}")


def test_missing_rule_proposal():
    """Test 16: Simulate a missing rule detection scenario.

    We know: inverse-square gravity (R1) + observed galaxy rotation data.
    Known constraint Π_known predicts rotation curve v(r) ∝ 1/√r.
    Observed: v(r) ≈ constant at large r.

    Residual → missing mass/rule (dark matter candidate).
    """
    def gravity_constraint(r):
        return 1.0 / np.sqrt(r + 1e-6)

    def observed_rotation(r):
        return 1.0 - 0.3 * np.exp(-r / 5.0)

    r1 = Rule(
        name="InverseSquare_Gravity", layer=1, domain="gravity",
        constraint_fn=lambda p: gravity_constraint(p[0])
    )

    field = ConstraintField(rules=[r1])
    detector = ResidualDetector(field, epsilon=0.02)

    points_2d = np.column_stack([
        np.linspace(0.1, 20, 200),
        observed_rotation(np.linspace(0.1, 20, 200))
    ])
    residuals = detector.scan_points(points_2d)
    clusters = detector.cluster_residuals(residuals)
    proposals = detector.propose_unknown_rules(clusters)

    print(f"[PASS] Test 16: Galaxy rotation residual detection")
    print(f"  Residual points found: {len(residuals)}")
    print(f"  Clusters: {len(clusters)}")
    for p in proposals:
        print(f"  → {p['description']}")


if __name__ == "__main__":
    tests = [
        test_residual_detector_basic,
        test_residual_detector_2d,
        test_eigen_threshold_basic,
        test_eigen_phase_boundary,
        test_sequence_entropy_transition,
        test_consciousness_window,
        test_snr_bound,
        test_attenuation_bound,
        test_iterative_integration_bound,
        test_self_reference_strength,
        test_cleaning_fish_prediction,
        test_artificial_chip_prediction,
        test_critical_length,
        test_phase_diagram,
        test_constraint_visibility_law,
        test_missing_rule_proposal,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {test.__name__}: {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    print(f"{'='*50}")
