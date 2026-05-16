"""GPU server validation: executor recovery test for all three executor types."""
import sys, os, numpy as np
from collections import Counter

sys.path.insert(0, '/root/autodl-tmp')
from synthetic_decay_monitor.data_lineage import (
    DataSample, DatasetLineage, _apply_decay,
)
from synthetic_decay_monitor.constraint_extractor import (
    HybridConstraintExtractor, extract_text_features,
)
from synthetic_decay_monitor.decay_engine import DecayEngine
from synthetic_decay_monitor.executor_classifier import (
    diagnose_executor_decay,
)

# Test texts with different capability signatures
TEST_TEXTS = {
    "math_reasoning": [
        "Therefore, because the square root of 2 is irrational, we must conclude that no rational number exists. However, this leads to a contradiction in our proof. Thus, the assumption is false. Hence, the proof by contradiction is complete.",
        "First, let us assume that p and q are coprime integers. Because squaring yields 2 = p^2/q^2, we get p^2 = 2q^2. Therefore, p must be even. However, this implies q is also even, which contradicts coprimality. Thus, no such integers exist.",
    ],
    "factual_knowledge": [
        "The French Revolution began in 1789 with the Storming of the Bastille in Paris. King Louis XVI was executed in 1793. Napoleon Bonaparte seized power in 1799. The Congress of Vienna in 1815 restored the Bourbon monarchy under Louis XVIII.",
        "According to a 2023 Nature study, Beijing population grew from 19.61 million in 2010 to 21.89 million in 2020. Professor Li Ming from Tsinghua University noted that Shanghai and Shenzhen grew at 3.2% and 5.7% annually.",
    ],
    "general": [
        "Reading books is a wonderful hobby enjoyed by millions of people worldwide. Libraries and bookstores provide access to countless stories and knowledge. The experience of holding a physical book remains irreplaceable despite digital alternatives.",
        "The cat sat peacefully on the windowsill, watching birds flutter past the glass. Sunlight streamed through the curtains, casting warm patterns across the wooden floor. It was a perfect afternoon for quiet contemplation.",
    ],
}


def run_test():
    print("=" * 60)
    print("EXECUTOR RECOVERY VALIDATION v0.2.0")
    print("=" * 60)

    results = {}
    for exec_type, exec_mix in [
        ('pure_E-I', {'E-I': 1.0, 'E-II': 0.0, 'E-III': 0.0}),
        ('pure_E-II', {'E-I': 0.0, 'E-II': 1.0, 'E-III': 0.0}),
        ('pure_E-III', {'E-I': 0.0, 'E-II': 0.0, 'E-III': 1.0}),
        ('E-I_dominant', {'E-I': 0.8, 'E-II': 0.1, 'E-III': 0.1}),
        ('E-II_dominant', {'E-I': 0.1, 'E-II': 0.8, 'E-III': 0.1}),
        ('E-III_dominant', {'E-I': 0.1, 'E-II': 0.1, 'E-III': 0.8}),
        ('balanced', {'E-I': 0.33, 'E-II': 0.33, 'E-III': 0.34}),
    ]:
        samples = []
        for cap, texts in TEST_TEXTS.items():
            for text in texts:
                # Gen 0: human
                samples.append(DataSample(
                    text=text, generation=0, source_model="human",
                    capability_tags=[cap],
                ))
                # Gen 1-4: decayed
                for gen in range(1, 5):
                    decayed = _apply_decay(
                        text, generation=gen, beta=0.25,
                        tags=[cap], executor_mix=exec_mix,
                    )
                    samples.append(DataSample(
                        text=decayed, generation=gen,
                        source_model="decay_sim", capability_tags=[cap],
                    ))

        lineage = DatasetLineage(samples=samples)
        extractor = HybridConstraintExtractor(judge_fn=None)
        engine = DecayEngine(lineage, extractor)
        engine.run_all_capabilities()

        dtype_votes = Counter()
        comps = []
        for cap, snapshots in engine._snapshots.items():
            if not snapshots:
                continue
            diag = diagnose_executor_decay(
                engine._trajectories.get(cap, []), snapshots, capability=cap
            )
            dt = getattr(diag['diagnosis'], 'degradation_type', 'unknown')
            dtype_votes[dt] += 1
            comps.append(diag.get('executor_composition', {}))

        expected_map = {
            'pure_E-I': 'E-I_loss', 'E-I_dominant': 'E-I_loss',
            'pure_E-II': 'E-II_loss', 'E-II_dominant': 'E-II_loss',
            'pure_E-III': 'E-III_loss', 'E-III_dominant': 'E-III_loss',
            'balanced': 'mixed',
        }
        expected = expected_map.get(exec_type, 'mixed')
        majority = dtype_votes.most_common(1)[0][0] if dtype_votes else 'unknown'
        match = "MATCH" if majority == expected else f"MISMATCH (got {majority}, expected {expected})"

        avg_comp = {}
        if comps:
            for k in ['E-I', 'E-II', 'E-III']:
                vals = [c.get(k, 0) for c in comps if c]
                avg_comp[k] = np.mean(vals) if vals else 0

        print(f"\n{exec_type}: {match}")
        print(f"  injected={exec_mix}")
        print(f"  votes={dict(dtype_votes)}")
        print(f"  avg comp: E-I={avg_comp.get('E-I',0):.3f} E-II={avg_comp.get('E-II',0):.3f} E-III={avg_comp.get('E-III',0):.3f}")

        # Per-capability detail
        for cap, snapshots in engine._snapshots.items():
            if not snapshots:
                continue
            diag = diagnose_executor_decay(
                engine._trajectories.get(cap, []), snapshots, capability=cap
            )
            dt = getattr(diag['diagnosis'], 'degradation_type', '?')
            comp = diag.get('executor_composition', {})
            print(f"    {cap}: {dt} (E-I={comp.get('E-I',0):.2f} E-II={comp.get('E-II',0):.2f} E-III={comp.get('E-III',0):.2f})")

        results[exec_type] = match.startswith("MATCH")

    correct = sum(results.values())
    total = len(results)
    print(f"\n{'=' * 60}")
    print(f"OVERALL: {correct}/{total} = {correct/total*100:.0f}%")
    print(f"{'=' * 60}")
    return results


if __name__ == '__main__':
    run_test()
