"""
Decay Monitor — synthetic data decay evaluation via constraint residual framework.

Measure LLM recursive stability via constraint residual beta.
Core diagnostic: Pi(p) = Sigma nabla sigma_i(p) — constraint-layer health,
not L3 surface performance monitoring.
"""

__version__ = "0.1.0"

# ── Core data types ──
from synthetic_decay_monitor.data_lineage import (
    DataSample,
    DatasetLineage,
    parse_lineage_from_jsonl,
    generate_synthetic_lineage,
)

# ── Constraint extraction ──
from synthetic_decay_monitor.constraint_extractor import (
    ConstraintState,
    ConstraintFieldSnapshot,
    HybridConstraintExtractor,
    EmbeddingConstraintExtractor,
    extract_text_features,
    text_features_to_constraint,
)

# ── Decay engine ──
from synthetic_decay_monitor.decay_engine import (
    DecayEngine,
    CapabilityStability,
    BASE_ALPHAS,
    S_CRITICAL,
    calibrate_beta,
    estimate_executor_composition,
    simulate_decay,
    predict_collapse,
)

# ── Executor classification ──
from synthetic_decay_monitor.executor_classifier import (
    ExecutorClassifier,
    diagnose_executor_decay,
    DegradationDiagnosis,
    CAPABILITY_EXECUTOR_PRIOR,
)

# ── Stress propagation ──
from synthetic_decay_monitor.stress_propagation import (
    CapabilityTopology,
    run_cascade_analysis,
)

# ── Reporting ──
from synthetic_decay_monitor.report import (
    generate_json_report,
    generate_html_report,
    generate_paper_figures,
)

__all__ = [
    # Data
    "DataSample", "DatasetLineage",
    "parse_lineage_from_jsonl", "generate_synthetic_lineage",
    # Extraction
    "ConstraintState", "ConstraintFieldSnapshot",
    "HybridConstraintExtractor", "EmbeddingConstraintExtractor",
    "extract_text_features", "text_features_to_constraint",
    # Engine
    "DecayEngine", "CapabilityStability",
    "BASE_ALPHAS", "S_CRITICAL",
    "calibrate_beta", "estimate_executor_composition",
    "simulate_decay", "predict_collapse",
    # Classification
    "ExecutorClassifier", "diagnose_executor_decay",
    "DegradationDiagnosis", "CAPABILITY_EXECUTOR_PRIOR",
    # Stress
    "CapabilityTopology", "run_cascade_analysis",
    # Reports
    "generate_json_report", "generate_html_report",
    "generate_paper_figures",
]
