"""Tests for constraint_extractor.py — text features, hybrid extractor."""
import pytest
import numpy as np

from synthetic_decay_monitor.constraint_extractor import (
    extract_text_features, text_features_to_constraint,
    HybridConstraintExtractor, ConstraintState, ConstraintFieldSnapshot,
    _safe_float, _split_sentences,
)


class TestTextFeatures:
    """Test extract_text_features() across degradation types."""

    def test_normal_text_has_high_logic_density(self):
        text = "Therefore, because of the evidence presented, we conclude the hypothesis is supported. However, further research is needed."
        f = extract_text_features(text)
        assert f["ei_logic_density"] > 0.5, f"Expected high logic density, got {f['ei_logic_density']}"
        assert f["eii_bigram_repetition"] < 0.3, "Normal text should have low bigram repetition"

    def test_ei_degraded_text_has_low_logic_density(self):
        # Simulate E-I: logic connectors removed
        text = "the system calibration measurement parameters initial conditions not set results differ significantly"
        f = extract_text_features(text)
        assert f["ei_logic_density"] < 0.3, f"E-I text should have low logic density, got {f['ei_logic_density']}"

    def test_eii_degraded_text_has_high_repetition(self):
        # E-II: phrase repetition + short words
        text = "the system is a system is a calibration of the the measurement and the and the parameters"
        f = extract_text_features(text)
        # E-II should show high filler ratio or low unique word ratio
        assert f["eii_unique_word_ratio"] < 0.8 or f["eii_filler_ratio"] > 0.1, \
            f"E-II text should show vocabulary degradation: uniq={f['eii_unique_word_ratio']}, fill={f['eii_filler_ratio']}"

    def test_eiii_degraded_text_low_proper_case(self):
        # E-III: proper nouns lowercased
        text = "the capital of france is paris, and the population of earth is approximately 100 million according to the latest measurements of the 2000 kilometer area."
        f = extract_text_features(text)
        assert f["eiii_proper_case_ratio"] < 0.5, f"E-III text should have low proper case ratio, got {f['eiii_proper_case_ratio']}"

    def test_short_text_returns_neutral_features(self):
        f = extract_text_features("hello world")
        assert f["ei_logic_density"] == 0.5
        assert f["ei_syntax_cv"] == 0.5

    def test_all_features_in_range(self):
        text = "The capital of France is Paris. The Eiffel Tower stands 330 meters tall."
        f = extract_text_features(text)
        for key, val in f.items():
            assert 0.0 <= val <= 1.0, f"{key} = {val} out of [0, 1]"

    def test_differentiates_ei_from_eii(self):
        """E-I vs E-II must produce different feature deltas."""
        base = "The system requires careful calibration because of the mathematical principles involved."
        # E-I: lose logic connectors
        ei_text = "The system requires careful calibration of the mathematical principles involved."
        # E-II: add repetition
        eii_text = "the system system requires careful calibration calibration of the the mathematical principles involved."

        f_base = extract_text_features(base)
        f_ei = extract_text_features(ei_text)
        f_eii = extract_text_features(eii_text)

        # E-I: logic density drops
        ei_logic_drop = f_ei["ei_logic_density"] - f_base["ei_logic_density"]
        # E-II: bigram repetition rises
        eii_rep_rise = f_eii["eii_bigram_repetition"] - f_base["eii_bigram_repetition"]

        assert ei_logic_drop < 0, f"E-I should decrease logic density, got {ei_logic_drop}"
        assert eii_rep_rise >= 0, f"E-II should not decrease bigram repetition, got {eii_rep_rise}"


class TestTextFeaturesToConstraint:
    """Test text_features → ConstraintState mapping."""

    def test_output_is_constraint_state(self):
        f = extract_text_features("The cat sat on the mat because it was tired.")
        state = text_features_to_constraint(f)
        assert isinstance(state, ConstraintState)

    def test_all_sigmas_in_range(self):
        f = extract_text_features("A sample text with proper nouns like Paris and numbers like 42.")
        state = text_features_to_constraint(f)
        for attr in ['sigma_fact', 'sigma_syntax', 'sigma_style', 'sigma_safety', 'sigma_coherence']:
            val = getattr(state, attr)
            assert 0.0 <= val <= 1.0, f"{attr} = {val} out of [0, 1]"

    def test_safety_is_neutral(self):
        f = extract_text_features("Any text here without safety keywords.")
        state = text_features_to_constraint(f)
        assert state.sigma_safety == 0.5


class TestHybridConstraintExtractor:
    """Test the hybrid extractor end-to-end."""

    def test_extract_sample_returns_constraint_state(self):
        hy = HybridConstraintExtractor(judge_fn=None)
        state = hy.extract_sample("The capital of France is Paris, which has 2.1 million residents.")
        assert isinstance(state, ConstraintState)
        assert 0.0 <= state.sigma_fact <= 1.0
        assert 0.0 <= state.sigma_syntax <= 1.0

    def test_extract_batch(self):
        hy = HybridConstraintExtractor(judge_fn=None)
        texts = ["Text one here.", "A different text."]
        states = hy.extract_batch(texts)
        assert len(states) == 2
        assert all(isinstance(s, ConstraintState) for s in states)

    def test_compute_field_produces_snapshot(self):
        from synthetic_decay_monitor.data_lineage import DataSample
        hy = HybridConstraintExtractor(judge_fn=None)
        samples = [
            DataSample(text="The capital of France is Paris.", generation=0, capability_tags=["factual"]),
            DataSample(text="France's capital Paris has 2.1M residents.", generation=0, capability_tags=["factual"]),
        ]
        snap = hy.compute_field(samples, capability="factual")
        assert isinstance(snap, ConstraintFieldSnapshot)
        assert snap.n_samples == 2
        assert snap.capability == "factual"

    def test_compute_field_populates_text_features(self):
        from synthetic_decay_monitor.data_lineage import DataSample
        hy = HybridConstraintExtractor(judge_fn=None)
        samples = [
            DataSample(text="The capital of France is Paris, population 2.1 million.", generation=0, capability_tags=["factual"]),
            DataSample(text="Therefore, the city faces challenges. However, investments continue.", generation=0, capability_tags=["factual"]),
        ]
        snap = hy.compute_field(samples, capability="factual")
        assert len(snap.text_features) >= 6, f"Expected ≥6 text features, got {len(snap.text_features)}"
        assert "ei_logic_density" in snap.text_features
        assert "eiii_proper_case_ratio" in snap.text_features

    def test_pure_text_mode_no_llm_needed(self):
        hy = HybridConstraintExtractor(judge_fn=None)
        assert hy.llm_judge is None
        state = hy.extract_sample("Any text works without LLM.")
        assert state.sigma_safety == 0.5


class TestSafeFloat:
    def test_clips_to_range(self):
        assert _safe_float(1.5) == 1.0
        assert _safe_float(-0.5) == 0.0
        assert _safe_float(0.75) == 0.75

    def test_handles_nan(self):
        assert _safe_float(float('nan')) == 0.5

    def test_handles_inf(self):
        assert _safe_float(float('inf')) == 0.5


class TestSplitSentences:
    def test_splits_on_period(self):
        sents = _split_sentences("Hello world. This is a test.")
        assert len(sents) >= 1

    def test_filters_short(self):
        sents = _split_sentences("Hi. A longer sentence here for testing purposes.")
        assert all(len(s) > 3 for s in sents)
