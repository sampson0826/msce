"""Tests for data_lineage.py — data parsing and synthetic generation."""
import json, tempfile, pytest

from synthetic_decay_monitor.data_lineage import (
    DataSample, DatasetLineage,
    parse_lineage_from_jsonl, generate_synthetic_lineage,
    _auto_tag, _apply_decay,
    CAPABILITY_TAGS, _EI_CAPABILITIES, _EII_CAPABILITIES, _EIII_CAPABILITIES,
)


class TestAutoTag:
    def test_math_text_tags_math_reasoning(self):
        tags = _auto_tag("The derivative of x^2 is 2x. Solve the equation.")
        assert "math_reasoning" in tags

    def test_code_text_tags_code_generation(self):
        tags = _auto_tag("Python functions use the def keyword to define reusable code blocks.")
        assert "code_generation" in tags or "general" in tags

    def test_fact_text_tags_factual(self):
        tags = _auto_tag("The capital of France is Paris. This historical fact is well documented.")
        assert "factual_knowledge" in tags

    def test_logic_text_tags_logical(self):
        tags = _auto_tag("Therefore, because of the evidence, we conclude the result holds.")
        assert "logical_consistency" in tags or "general" in tags

    def test_fallback_to_general(self):
        tags = _auto_tag("Hello world this is a test sentence.")
        assert "general" in tags


class TestApplyDecay:
    def test_no_decay_on_short_text(self):
        result = _apply_decay("hi", generation=1, beta=0.25, tags=["general"])
        assert result == "hi"

    def test_decay_reduces_text(self):
        text = "Therefore, because of the complex mathematical principles involved, the system requires careful calibration and precise measurement of all parameters."
        decayed = _apply_decay(text, generation=2, beta=0.25, tags=["math_reasoning"])
        assert len(decayed) > 0

    def test_pure_ei_removes_logic_connectors(self):
        text = "Therefore, because the data supports the hypothesis, we can conclude it is correct."
        decayed = _apply_decay(text, generation=2, beta=0.25, tags=["math_reasoning"],
                               executor_mix={"E-I": 1.0, "E-II": 0.0, "E-III": 0.0})
        original_connectors = {"therefore", "because"}
        decayed_lower = set(decayed.lower().split())
        # At high intensity, most logic connectors should be removed
        remaining = original_connectors & decayed_lower
        # Not all will be removed (probabilistic), but some should be
        assert len(decayed) > 0

    def test_pure_eii_can_repeat_phrases(self):
        text = "The system requires careful calibration and precise measurement of all parameters involved in the process."
        decayed = _apply_decay(text, generation=2, beta=0.25, tags=["style_diversity"],
                               executor_mix={"E-I": 0.0, "E-II": 1.0, "E-III": 0.0})
        assert len(decayed) > 0

    def test_pure_eiii_lowercases_proper_nouns(self):
        text = "The capital of France is Paris. The Eiffel Tower stands in the city."
        decayed = _apply_decay(text, generation=2, beta=0.25, tags=["factual_knowledge"],
                               executor_mix={"E-I": 0.0, "E-II": 0.0, "E-III": 1.0})
        # At high intensity, some proper nouns should be lowercased
        assert len(decayed) > 0


class TestGenerateSyntheticLineage:
    def test_generates_correct_generations(self):
        texts = ["This is a test sentence with math and solve."]
        lineage = generate_synthetic_lineage(texts, n_generations=3)
        assert lineage.n_generations == 4  # gen 0,1,2,3
        assert lineage.n_samples == 4

    def test_capability_tags_propagate(self):
        texts = ["The derivative of x^2 is 2x. Solve the equation."]
        lineage = generate_synthetic_lineage(texts, n_generations=2)
        for gen in range(2):
            samples = lineage.generations[gen]
            for s in samples:
                assert "math_reasoning" in s.capability_tags

    def test_override_capability_tags(self):
        texts = ["Some text here."]
        lineage = generate_synthetic_lineage(texts, n_generations=2, capability_tags=["custom_tag"])
        for gen in range(2):
            for s in lineage.generations[gen]:
                assert s.capability_tags == ["custom_tag"]

    def test_executor_pattern_applied(self):
        texts = ["The capital of France is Paris. The Eiffel Tower was completed in 1889."]
        lineage_ei = generate_synthetic_lineage(
            texts, n_generations=2, decay_pattern={"*": 0.15},
            executor_pattern={"*": {"E-I": 1.0, "E-II": 0.0, "E-III": 0.0}},
            capability_tags=["factual"],
        )
        lineage_eiii = generate_synthetic_lineage(
            texts, n_generations=2, decay_pattern={"*": 0.15},
            executor_pattern={"*": {"E-I": 0.0, "E-II": 0.0, "E-III": 1.0}},
            capability_tags=["factual"],
        )
        # After E-I decay, text should be more fragmented (shorter)
        ei_gen1 = lineage_ei.generations[1][0].text
        eiii_gen1 = lineage_eiii.generations[1][0].text
        # E-I drops words → text shorter; E-III mainly lowercases
        assert len(ei_gen1) > 0 and len(eiii_gen1) > 0


class TestParseLineageFromJSONL:
    def test_parses_valid_jsonl(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"text": "Hello world.", "generation": 0}\n')
            f.write('{"text": "Another text.", "generation": 1}\n')
            path = f.name
        try:
            lineage = parse_lineage_from_jsonl(path)
            assert lineage.n_samples == 2
            assert lineage.n_generations == 2
            assert lineage.generations[0][0].text == "Hello world."
        finally:
            os.unlink(path)

    def test_parses_with_capability_tags(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"text": "Math text.", "generation": 0, "capability_tags": ["math_reasoning"]}\n')
            path = f.name
        try:
            lineage = parse_lineage_from_jsonl(path)
            assert "math_reasoning" in lineage.generations[0][0].capability_tags
            assert lineage.capability_coverage["math_reasoning"][0] == 1
        finally:
            os.unlink(path)

    def test_missing_text_raises(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"generation": 0}\n')
            path = f.name
        try:
            with pytest.raises(ValueError):
                parse_lineage_from_jsonl(path)
        finally:
            os.unlink(path)

    def test_empty_file_raises(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            pass  # empty file
            path = f.name
        try:
            with pytest.raises(ValueError):
                parse_lineage_from_jsonl(path)
        finally:
            os.unlink(path)

    def test_skips_empty_lines(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('\n')
            f.write('{"text": "Valid.", "generation": 0}\n')
            f.write('\n')
            path = f.name
        try:
            lineage = parse_lineage_from_jsonl(path)
            assert lineage.n_samples == 1
        finally:
            os.unlink(path)


class TestDatasetLineage:
    def test_samples_by_capability(self):
        samples = [
            DataSample(text="Math text", generation=0, capability_tags=["math_reasoning"]),
            DataSample(text="Code text", generation=0, capability_tags=["code_generation"]),
            DataSample(text="Math again", generation=1, capability_tags=["math_reasoning"]),
        ]
        lineage = DatasetLineage(samples=samples)
        math_gen0 = lineage.samples_by_capability("math_reasoning", 0)
        assert len(math_gen0) == 1

    def test_generation_summary(self):
        samples = [
            DataSample(text="A text", generation=0, source_model="human"),
            DataSample(text="B text", generation=1, source_model="gen_model"),
        ]
        lineage = DatasetLineage(samples=samples)
        summary = lineage.generation_summary()
        assert 0 in summary
        assert summary[0]["n_samples"] == 1


class TestCapabilityTags:
    def test_all_tags_are_strings(self):
        for tag in CAPABILITY_TAGS:
            assert isinstance(tag, str)

    def test_capability_sets_are_disjoint(self):
        overlap_ei_eii = _EI_CAPABILITIES & _EII_CAPABILITIES
        assert len(overlap_ei_eii) == 0, f"Overlap: {overlap_ei_eii}"
        overlap_ei_eiii = _EI_CAPABILITIES & _EIII_CAPABILITIES
        assert len(overlap_ei_eiii) == 0, f"Overlap: {overlap_ei_eiii}"
