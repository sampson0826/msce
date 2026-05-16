"""Tests for decay_engine.py — decay law, executor composition estimation."""
import pytest
import numpy as np

from synthetic_decay_monitor.decay_engine import (
    BASE_ALPHAS, S_CRITICAL,
    calibrate_beta, calibrate_beta_from_data,
    estimate_executor_composition,
    DecayEngine, CapabilityStability,
    simulate_decay, predict_collapse,
)
from synthetic_decay_monitor.constraint_extractor import (
    ConstraintFieldSnapshot, ConstraintState,
)
from synthetic_decay_monitor.data_lineage import (
    generate_synthetic_lineage, DataSample, DatasetLineage,
)


class TestBaseAlphas:
    """Verify the framework's base assumptions."""

    def test_ei_most_sensitive(self):
        assert BASE_ALPHAS["E-I"] > BASE_ALPHAS["E-II"] > BASE_ALPHAS["E-III"]

    def test_critical_threshold(self):
        assert 0.2 < S_CRITICAL < 0.5

    def test_alphas_sum_to_beta_bound(self):
        # With uniform composition, beta should be ~0.227
        beta_uniform = calibrate_beta("general", {"E-I": 0.33, "E-II": 0.33, "E-III": 0.34})
        assert 0.20 < beta_uniform < 0.30


class TestCalibrateBeta:
    def test_pure_ei_produces_high_beta(self):
        beta = calibrate_beta("math", {"E-I": 1.0, "E-II": 0.0, "E-III": 0.0})
        assert beta == pytest.approx(0.40, abs=0.01)

    def test_pure_eiii_produces_low_beta(self):
        beta = calibrate_beta("factual", {"E-I": 0.0, "E-II": 0.0, "E-III": 1.0})
        assert beta == pytest.approx(0.08, abs=0.01)

    def test_mixed_composition(self):
        beta = calibrate_beta("general", {"E-I": 0.5, "E-II": 0.3, "E-III": 0.2})
        expected = 0.40 * 0.5 + 0.20 * 0.3 + 0.08 * 0.2
        assert beta == pytest.approx(expected, abs=0.01)

    def test_clamped_at_095(self):
        beta = calibrate_beta("math", {"E-I": 3.0, "E-II": 0.0, "E-III": 0.0})  # invalid proportions
        assert beta <= 0.95

    def test_override_alphas(self):
        custom = {"E-I": 0.5, "E-II": 0.3, "E-III": 0.1}
        beta = calibrate_beta("math", {"E-I": 1.0}, override_alphas=custom)
        assert beta == pytest.approx(0.5)


class TestSimulateDecay:
    def test_exponential_decay_shape(self):
        traj = simulate_decay(S0=1.0, beta=0.25, n_generations=5)
        assert len(traj) == 5
        assert traj[0].S_n == 1.0
        assert traj[1].S_n == pytest.approx(0.75)
        assert traj[2].S_n == pytest.approx(0.5625)

    def test_collapse_detection(self):
        traj = simulate_decay(S0=1.0, beta=0.25, n_generations=10)
        collapsed = [t for t in traj if t.is_collapsed]
        assert len(collapsed) > 0

    def test_beta_zero_no_decay(self):
        traj = simulate_decay(S0=1.0, beta=0.0, n_generations=5)
        for t in traj:
            assert t.S_n == 1.0


class TestPredictCollapse:
    def test_returns_collapsed_generation(self):
        traj = simulate_decay(S0=1.0, beta=0.25, n_generations=8)
        gen = predict_collapse(traj)
        assert 4 <= gen <= 7  # S_CRITICAL = 0.30

    def test_no_collapse_returns_negative(self):
        traj = simulate_decay(S0=1.0, beta=0.0, n_generations=3)
        gen = predict_collapse(traj)
        assert gen < 0 or gen == 999


class TestEstimateExecutorComposition:
    """Test the core executor composition estimation."""

    def _make_snap(self, sigmas, stds=None, text_features=None):
        stds = stds or {"fact": 0.05, "syntax": 0.05, "style": 0.05, "safety": 0.05, "coherence": 0.05}
        tf = text_features or {}
        return ConstraintFieldSnapshot(
            generation=0, n_samples=5, capability="test",
            individual_sigmas=sigmas,
            sigma_stds=stds,
            text_features=tf,
        )

    def test_single_gen_fallback(self):
        snap = self._make_snap(
            {"fact": 0.6, "syntax": 0.7, "style": 0.5, "safety": 0.5, "coherence": 0.6}
        )
        comp = estimate_executor_composition(snap)
        for k in ["E-I", "E-II", "E-III"]:
            assert k in comp
        assert sum(comp.values()) == pytest.approx(1.0, abs=0.01)

    def test_cross_gen_text_features_ei_dominant(self):
        """E-I text: logic density drops, others stable."""
        prev = self._make_snap(
            {"fact": 0.6, "syntax": 0.7, "style": 0.6, "safety": 0.5, "coherence": 0.7},
            text_features={"ei_logic_density": 0.9, "ei_syntax_cv": 0.6,
                          "eii_bigram_repetition": 0.0, "eii_filler_ratio": 0.0,
                          "eii_truncation_ratio": 0.1, "eii_unique_word_ratio": 0.9,
                          "eiii_proper_case_ratio": 0.5, "eiii_number_integrity": 0.5}
        )
        curr = self._make_snap(
            {"fact": 0.55, "syntax": 0.5, "style": 0.55, "safety": 0.5, "coherence": 0.5},
            text_features={"ei_logic_density": 0.3, "ei_syntax_cv": 0.4,
                          "eii_bigram_repetition": 0.0, "eii_filler_ratio": 0.0,
                          "eii_truncation_ratio": 0.1, "eii_unique_word_ratio": 0.88,
                          "eiii_proper_case_ratio": 0.5, "eiii_number_integrity": 0.5}
        )
        comp = estimate_executor_composition(curr, prev_snapshot=prev)
        assert comp["E-I"] > comp["E-II"], f"E-I should dominate: {comp}"
        assert comp["E-I"] > comp["E-III"], f"E-I should dominate: {comp}"

    def test_cross_gen_text_features_eii_dominant(self):
        """E-II text: filler rises, uniqueness drops, logic stable."""
        prev = self._make_snap(
            {"fact": 0.6, "syntax": 0.7, "style": 0.6, "safety": 0.5, "coherence": 0.7},
            text_features={"ei_logic_density": 0.9, "ei_syntax_cv": 0.6,
                          "eii_bigram_repetition": 0.0, "eii_filler_ratio": 0.0,
                          "eii_truncation_ratio": 0.1, "eii_unique_word_ratio": 0.9,
                          "eiii_proper_case_ratio": 0.5, "eiii_number_integrity": 0.5}
        )
        curr = self._make_snap(
            {"fact": 0.55, "syntax": 0.6, "style": 0.45, "safety": 0.5, "coherence": 0.6},
            text_features={"ei_logic_density": 0.85, "ei_syntax_cv": 0.6,
                          "eii_bigram_repetition": 0.3, "eii_filler_ratio": 0.4,
                          "eii_truncation_ratio": 0.25, "eii_unique_word_ratio": 0.6,
                          "eiii_proper_case_ratio": 0.5, "eiii_number_integrity": 0.5}
        )
        comp = estimate_executor_composition(curr, prev_snapshot=prev)
        assert comp["E-II"] > comp["E-I"], f"E-II should dominate: {comp}"

    def test_cross_gen_text_features_eiii_dominant(self):
        """E-III text: proper case drops, others stable."""
        prev = self._make_snap(
            {"fact": 0.7, "syntax": 0.7, "style": 0.6, "safety": 0.5, "coherence": 0.7},
            text_features={"ei_logic_density": 0.9, "ei_syntax_cv": 0.6,
                          "eii_bigram_repetition": 0.0, "eii_filler_ratio": 0.0,
                          "eii_truncation_ratio": 0.1, "eii_unique_word_ratio": 0.9,
                          "eiii_proper_case_ratio": 0.9, "eiii_number_integrity": 0.8}
        )
        curr = self._make_snap(
            {"fact": 0.5, "syntax": 0.68, "style": 0.58, "safety": 0.5, "coherence": 0.69},
            text_features={"ei_logic_density": 0.9, "ei_syntax_cv": 0.6,
                          "eii_bigram_repetition": 0.0, "eii_filler_ratio": 0.0,
                          "eii_truncation_ratio": 0.1, "eii_unique_word_ratio": 0.89,
                          "eiii_proper_case_ratio": 0.3, "eiii_number_integrity": 0.5}
        )
        comp = estimate_executor_composition(curr, prev_snapshot=prev)
        assert comp["E-III"] > comp["E-I"], f"E-III should dominate: {comp}"

    def test_composition_sums_to_one(self):
        snap = self._make_snap(
            {"fact": 0.6, "syntax": 0.5, "style": 0.6, "safety": 0.5, "coherence": 0.5}
        )
        comp = estimate_executor_composition(snap)
        total = sum(comp.values())
        assert abs(total - 1.0) < 0.02, f"Composition should sum to 1, got {total}"


class TestDecayEngineEndToEnd:
    """End-to-end test with synthetic data."""

    def test_runs_on_synthetic_lineage(self):
        texts = [
            "The capital of France is Paris, population 2.1 million. Therefore, the city faces challenges.",
            "The Eiffel Tower was completed in 1889 and stands 330 meters tall, attracting 7 million visitors.",
        ]
        lineage = generate_synthetic_lineage(texts, n_generations=2, decay_pattern={"*": 0.15})
        from synthetic_decay_monitor.constraint_extractor import HybridConstraintExtractor
        extractor = HybridConstraintExtractor(judge_fn=None)
        engine = DecayEngine(lineage, extractor)
        engine.run_all_capabilities()
        trajectories = engine.get_all_trajectories()
        assert len(trajectories) > 0

    def test_trajectory_has_required_fields(self):
        texts = ["The capital of France is Paris, with 2.1 million residents."]
        lineage = generate_synthetic_lineage(texts, n_generations=2, decay_pattern={"*": 0.15})
        from synthetic_decay_monitor.constraint_extractor import HybridConstraintExtractor
        extractor = HybridConstraintExtractor(judge_fn=None)
        engine = DecayEngine(lineage, extractor)
        engine.run_all_capabilities()
        for t in engine.get_all_trajectories():
            assert "capability" in t
            assert "trajectory" in t
            assert "predicted_collapse_gen" in t
            assert "current_status" in t

    def test_collapse_order_returns_ranking(self):
        texts = [
            "The derivative of x^2 is 2x. Therefore, the chain rule applies.",
            "Python functions use the def keyword and return values.",
            "The capital of France is Paris, established centuries ago.",
        ]
        lineage = generate_synthetic_lineage(texts, n_generations=3, decay_pattern={"*": 0.25})
        from synthetic_decay_monitor.constraint_extractor import HybridConstraintExtractor
        extractor = HybridConstraintExtractor(judge_fn=None)
        engine = DecayEngine(lineage, extractor)
        engine.run_all_capabilities()
        order = engine.get_collapse_order()
        assert isinstance(order, list)


class TestCapabilityStability:
    def test_status_transitions(self):
        cs = CapabilityStability(capability="test", generation=0, S_n=0.9)
        assert cs.status == "healthy"
        cs.S_n = 0.6
        assert cs.status == "degrading"
        cs.S_n = 0.4
        assert cs.status == "critical"
        cs.S_n = 0.2
        assert cs.status == "collapsed"
        assert cs.is_collapsed
