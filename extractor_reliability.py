"""Extractor Reliability Study: validating the HybridConstraintExtractor.

Computes:
  (a) Agreement metrics between HybridConstraintExtractor (text-features mode)
      and an independent EmbeddingConstraintExtractor as surrogate "LLM judge"
      -- both measure the same 5 constraint dimensions from text via different
      computational paths.
  (b) Test-retest reliability of the rule-based text-feature extractor.
  (c) Per-dimension variance decomposition across models.

Output: experiment_data/extractor_reliability.json + stdout summary table.

No external API calls -- uses existing lineage JSONL data only.
No scipy -- implements Spearman and Cohen's kappa by hand.
"""

import json
import os
import sys
import math
import numpy as np
from collections import defaultdict

# --- Path setup -----------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# --- Imports from project -----------------------------------------------
from synthetic_decay_monitor.constraint_extractor import (
    HybridConstraintExtractor,
    EmbeddingConstraintExtractor,
    extract_text_features,
    text_features_to_constraint,
    ConstraintState,
)

# ===================================================================
# 1.  Load texts from all lineage JSONL files
# ===================================================================

def load_all_lineage_texts(data_dir: str) -> list[dict]:
    """Load every text sample from every lineage JSONL file.

    Returns list of dicts with keys: text, generation, source_model, capability_tags, sample_id
    """
    records = []
    jsonl_dir = os.path.join(data_dir, "experiment_data")
    for fname in sorted(os.listdir(jsonl_dir)):
        if not fname.endswith(".jsonl"):
            continue
        fpath = os.path.join(jsonl_dir, fname)
        with open(fpath, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                records.append({
                    "text": rec.get("text", ""),
                    "generation": rec.get("generation", 0),
                    "source_model": rec.get("source_model", "unknown"),
                    "capability_tags": rec.get("capability_tags", []),
                    "sample_id": rec.get("id", ""),
                })
    return records


# ===================================================================
# 2.  Per-dimension agreement: Hybrid (text features) vs Embedding
# ===================================================================

DIM_NAMES = ["sigma_fact", "sigma_syntax", "sigma_style", "sigma_safety", "sigma_coherence"]


def spearman_r(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation -- manual implementation (no scipy)."""
    n = len(x)
    if n < 3:
        return 0.0

    def rankdata(a):
        """Return ranks (1-based, average for ties)."""
        order = np.argsort(a)
        ranks = np.empty(n, dtype=float)
        i = 0
        while i < n:
            j = i
            while j < n and a[order[j]] == a[order[i]]:
                j += 1
            avg_rank = (i + j + 2) / 2.0  # 1-based average
            for k in range(i, j):
                ranks[order[k]] = avg_rank
            i = j
        return ranks

    rx = rankdata(x)
    ry = rankdata(y)

    # Pearson r on ranks = Spearman rho
    mx = np.mean(rx)
    my = np.mean(ry)
    num = np.sum((rx - mx) * (ry - my))
    den = np.sqrt(np.sum((rx - mx)**2) * np.sum((ry - my)**2))
    if den < 1e-15:
        return 0.0
    return float(num / den)


def cohens_kappa_binary(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's kappa for binary (0/1) agreement.  No scipy.

    a, b: binary arrays (1 = "above threshold", 0 = "below").
    """
    n = len(a)
    if n == 0:
        return 0.0

    # Observed agreement
    po = np.mean(a == b)

    # Expected agreement under independence
    pa1 = np.mean(a)
    pb1 = np.mean(b)
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)

    if pe >= 1.0 - 1e-15:
        return 1.0 if po > 0.99 else 0.0
    return float((po - pe) / (1.0 - pe))


def compute_per_dimension_agreement(
    hybrid_states: list[ConstraintState],
    embed_states: list[ConstraintState],
    threshold: float = 0.8,
) -> dict:
    """Compute Pearson r, Spearman rho, and Cohen's kappa per dimension."""
    n = len(hybrid_states)
    results = {}
    for dim in DIM_NAMES:
        hv = np.array([getattr(s, dim) for s in hybrid_states])
        ev = np.array([getattr(s, dim) for s in embed_states])

        # Filter out constant arrays (no variance -> undefined correlation)
        if np.std(hv) < 1e-12 or np.std(ev) < 1e-12:
            results[dim] = {"pearson_r": 0.0, "spearman_rho": 0.0, "cohens_kappa": 0.0}
            continue

        # Pearson r
        mx_h = np.mean(hv)
        mx_e = np.mean(ev)
        num = np.sum((hv - mx_h) * (ev - mx_e))
        den = np.sqrt(np.sum((hv - mx_h)**2) * np.sum((ev - mx_e)**2))
        pearson = float(num / den) if den > 1e-15 else 0.0

        # Spearman rho
        rho = spearman_r(hv, ev)

        # Cohen's kappa (binarize at threshold)
        hb = (hv >= threshold).astype(int)
        eb = (ev >= threshold).astype(int)
        kappa = cohens_kappa_binary(hb, eb)

        results[dim] = {
            "pearson_r": round(pearson, 4),
            "spearman_rho": round(rho, 4),
            "cohens_kappa": round(kappa, 4),
        }
    return results


def overall_mean_agreement(per_dim: dict) -> float:
    """Mean of the absolute Pearson r values across dimensions."""
    vals = [abs(v["pearson_r"]) for v in per_dim.values()]
    return round(float(np.mean(vals)), 4)


# ===================================================================
# 3.  Test-retest reliability
# ===================================================================

def perturb_text(text: str, perturbation: str) -> str:
    """Apply small text perturbation."""
    if perturbation == "trailing_space":
        return text + " " if not text.endswith(" ") else text
    elif perturbation == "strip_trailing":
        return text.rstrip()
    elif perturbation == "lowercase":
        return text.lower()
    elif perturbation == "uppercase":
        return text.upper()
    elif perturbation == "add_period":
        return text + "." if not text.endswith(".") else text
    elif perturbation == "extra_newline":
        return text + "\n"
    else:
        return text


def test_retest_reliability(texts: list[str]) -> dict:
    """Run extract_text_features twice on identical and perturbed inputs."""

    # (a) Identical inputs -> must produce identical features
    features1 = [extract_text_features(t) for t in texts]
    features2 = [extract_text_features(t) for t in texts]  # second pass

    identical = True
    max_diff = 0.0
    for f1, f2 in zip(features1, features2):
        for key in f1:
            diff = abs(f1[key] - f2[key])
            max_diff = max(max_diff, diff)
            if diff > 1e-12:
                identical = False

    # (b) Perturbed inputs -> measure robustness
    perturbations = ["trailing_space", "strip_trailing", "add_period", "extra_newline"]
    perturb_results = {}

    for ptype in perturbations:
        pert_texts = [perturb_text(t, ptype) for t in texts]
        pert_features = [extract_text_features(t) for t in pert_texts]
        dim_diffs = defaultdict(list)
        for f_orig, f_pert in zip(features1, pert_features):
            for key in f_orig:
                dim_diffs[key].append(abs(f_orig[key] - f_pert[key]))
        perturb_results[ptype] = {
            key: round(float(np.mean(vals)), 6) for key, vals in dim_diffs.items()
        }

    return {
        "identical_on_identical_inputs": identical,
        "max_difference": round(max_diff, 12),
        "perturbation_mean_absolute_diff": perturb_results,
        "n_samples": len(texts),
    }


# ===================================================================
# 4.  Per-dimension variance decomposition across models
# ===================================================================

def variance_decomposition(records: list[dict], hybrid_states: list[ConstraintState]) -> dict:
    """Decompose variance per dimension into between-model and within-model.

    Uses source_model as the grouping factor.
    signal_to_noise = between_model_var / (within_model_var + 1e-12)
    """
    # Group states by source_model
    by_model = defaultdict(list)
    for rec, state in zip(records, hybrid_states):
        model = rec.get("source_model", "unknown")
        by_model[model].append(state)

    # Only keep models with >= 3 samples
    by_model = {m: sts for m, sts in by_model.items() if len(sts) >= 3}

    results = {}
    grand_means = {}
    for dim in DIM_NAMES:
        all_vals = np.array([getattr(s, dim) for s in hybrid_states])
        grand_mean = float(np.mean(all_vals))
        grand_means[dim] = grand_mean

        between_var = 0.0
        within_var = 0.0
        total_n = 0

        for model, sts in by_model.items():
            vals = np.array([getattr(s, dim) for s in sts])
            n_m = len(vals)
            total_n += n_m
            # Between-model: weighted squared deviation of model mean from grand mean
            between_var += n_m * (float(np.mean(vals)) - grand_mean) ** 2
            # Within-model: sum of squared deviations from model mean
            within_var += float(np.sum((vals - np.mean(vals)) ** 2))

        n_models = len(by_model)
        if total_n > n_models and n_models > 1:
            between_var /= (n_models - 1)
            within_var /= (total_n - n_models)
        else:
            between_var = 0.0
            within_var = float(np.var(all_vals)) if len(all_vals) > 1 else 0.0

        snr = between_var / (within_var + 1e-12)

        results[dim] = {
            "between_model_var": round(between_var, 6),
            "within_model_var": round(within_var, 6),
            "signal_to_noise": round(snr, 6),
        }

    return results


# ===================================================================
# 5.  LLM Judge comparison (using existing judge validation data)
# ===================================================================

def compare_with_existing_judge_data(
    records: list[dict],
    hybrid_states: list[ConstraintState],
    data_dir: str,
) -> dict:
    """Correlate Hybrid extractor outputs with existing LLM judge data.

    The existing judge_validation_v2.json has per-capability judge preferences
    (1-5 scale, Gen1 vs GenN pairwise wins) and per-capability β values.
    We correlate the extractor's constraint scores with judge preferences.
    """
    jv2_path = os.path.join(data_dir, "experiment_data", "judge_validation_v2.json")
    if not os.path.exists(jv2_path):
        return {"error": "judge_validation_v2.json not found", "note": "skipping LLM judge comparison"}

    with open(jv2_path) as f:
        jv2 = json.load(f)

    # Compute per-capability mean constraint scores from the lineage data
    by_cap = defaultdict(list)
    for rec, state in zip(records, hybrid_states):
        for tag in rec.get("capability_tags", []):
            by_cap[tag].append(state)

    cap_constraint_means = {}
    for cap, sts in by_cap.items():
        if len(sts) < 3:
            continue
        cap_constraint_means[cap] = {
            dim: float(np.mean([getattr(s, dim) for s in sts]))
            for dim in DIM_NAMES
        }
        # Also compute overall constraint magnitude
        cap_constraint_means[cap]["overall_mean"] = float(np.mean([
            cap_constraint_means[cap][dim] for dim in DIM_NAMES
        ]))

    # Match capabilities between judge data and constraint data
    common_caps = set(jv2.get("per_cap_judge_pref", {}).keys()) & set(cap_constraint_means.keys())

    per_dim_correlations = {}
    for dim in DIM_NAMES:
        x_judge = np.array([jv2["per_cap_judge_pref"][c] for c in common_caps])
        y_constraint = np.array([cap_constraint_means[c][dim] for c in common_caps])
        if len(x_judge) >= 4 and np.std(y_constraint) > 1e-12:
            rho = spearman_r(x_judge, y_constraint)
        else:
            rho = 0.0
        per_dim_correlations[dim] = round(float(rho), 4)

    # Overall: judge preference vs mean constraint score
    x_overall = np.array([jv2["per_cap_judge_pref"][c] for c in common_caps])
    y_overall = np.array([cap_constraint_means[c]["overall_mean"] for c in common_caps])
    if len(x_overall) >= 4 and np.std(y_overall) > 1e-12:
        overall_rho = spearman_r(x_overall, y_overall)
    else:
        overall_rho = 0.0

    return {
        "common_capabilities": sorted(common_caps),
        "n_capabilities": len(common_caps),
        "per_dim_spearman_vs_judge_pref": per_dim_correlations,
        "overall_spearman_vs_judge_pref": round(float(overall_rho), 4),
        "note": "Judge data is overall quality preference (1-5 scale), not per-constraint-dimension scoring. "
                "Correlations are at the capability level, not the sample level.",
    }


# ===================================================================
# 6.  Feature-level validation (what "86% agreement" actually means)
# ===================================================================

def analyze_86_percent_claim() -> str:
    """Explain what the '86% agreement' claim in the paper actually means.

    Based on the docstring claims:
      - ei_logic_density: Spearman rho=0.93 vs judge E-I
      - eii_bigram_repetition: rho=0.98 vs judge E-II
      - eiii_proper_case_ratio: rho=0.92 vs judge E-III

    These are per-FEATURE correlations, not per-ConstraintState-dimension.
    The 86% figure is likely the mean of absolute correlation coefficients
    across the 8 features, but only 3 features were validated against judge.
    """
    # The 3 validated features with their claimed Spearman rho:
    validated = {
        "ei_logic_density": 0.93,
        "eii_bigram_repetition": 0.98,
        "eiii_proper_case_ratio": 0.92,
    }
    # The other 5 features have no reported judge validation
    unvalidated = [
        "ei_syntax_cv",
        "eii_filler_ratio",
        "eii_unique_word_ratio",
        "eii_truncation_ratio",
        "eiii_number_integrity",
    ]

    mean_validated = np.mean(list(validated.values()))
    # If unvalidated features are assumed at 0.75-0.80 (moderate unvalidated correlation)
    # the overall mean could reach 0.86

    return (
        f"The paper's '86% agreement' claim refers to per-feature Spearman rho "
        f"between individual text features and LLM-judge annotations, not to "
        f"agreement at the ConstraintState (5-dimension) level. "
        f"Only 3 of 8 features have reported judge validation: "
        f"ei_logic_density (rho={validated['ei_logic_density']}), "
        f"eii_bigram_repetition (rho={validated['eii_bigram_repetition']}), "
        f"eiii_proper_case_ratio (rho={validated['eiii_proper_case_ratio']}) -- "
        f"mean of validated features = {mean_validated:.3f}. "
        f"The 5 unvalidated features ({', '.join(unvalidated)}) have no reported "
        f"judge correlation. The 86% figure likely averages the 3 validated rhos "
        f"({mean_validated:.2f}) with assumed correlations of ~0.80-0.83 for "
        f"unvalidated features, or it may include additional unreported measurements. "
        f"It is NOT a measure of agreement between the Hybrid extractor's 5D "
        f"ConstraintState output and any independent criterion -- it is a feature-level "
        f"validation claim covering only 3/8 features with documented evidence."
    )


# ===================================================================
# Main
# ===================================================================

def main():
    data_dir = PROJECT_ROOT
    print("=" * 80)
    print("EXTRACTOR RELIABILITY STUDY")
    print("=" * 80)

    # --- Load data ---
    print("\n[1] Loading lineage texts ...")
    records = load_all_lineage_texts(data_dir)
    texts = [r["text"] for r in records if r["text"].strip()]
    print(f"    Loaded {len(texts)} texts from lineage JSONL files.")
    print(f"    Models: {sorted(set(r['source_model'] for r in records))}")
    print(f"    Generations: {sorted(set(r['generation'] for r in records))}")

    # --- Initialize extractors ---
    print("\n[2] Initializing extractors ...")
    hybrid = HybridConstraintExtractor(judge_fn=None)  # text-features-only mode

    # Use a subset for embedding extractor (it's slow with sentence-transformers)
    # Take up to 200 samples, stratified across models
    model_texts = defaultdict(list)
    for rec in records:
        if rec["text"].strip():
            model_texts[rec["source_model"]].append(rec["text"])
    sample_texts = []
    sample_records = []
    texts_per_model = max(1, 200 // max(len(model_texts), 1))
    seen = set()
    for model in sorted(model_texts.keys()):
        for i, t in enumerate(model_texts[model]):
            if i >= texts_per_model:
                break
            if t not in seen:
                seen.add(t)
                sample_texts.append(t)
                # Find matching record
                for rec in records:
                    if rec["text"] == t:
                        sample_records.append(rec)
                        break

    print(f"    Using {len(sample_texts)} texts for embedding-based comparison.")

    print("    Loading EmbeddingConstraintExtractor (sentence-transformers) ...")
    try:
        embed_extractor = EmbeddingConstraintExtractor("all-MiniLM-L6-v2")
        embedding_available = True
    except ImportError:
        print("    WARNING: sentence-transformers not installed. Skipping embedding comparison.")
        print("    Install with: pip install sentence-transformers")
        embedding_available = False

    # --- Extract ---
    print("\n[3] Extracting constraint states ...")
    hybrid_states = hybrid.extract_batch(sample_texts)
    print(f"    Hybrid extractor: {len(hybrid_states)} states extracted.")

    if embedding_available:
        embed_states = embed_extractor.extract_batch(sample_texts)
        print(f"    Embedding extractor: {len(embed_states)} states extracted.")
    else:
        embed_states = None

    # --- (a) Per-dimension agreement ---
    print("\n[4] Computing per-dimension agreement metrics ...")
    if embedding_available and embed_states:
        per_dim = compute_per_dimension_agreement(hybrid_states, embed_states, threshold=0.8)
        overall_agreement = overall_mean_agreement(per_dim)
    else:
        # Fallback: compare Hybrid text-features against internal consistency
        # (split-half reliability of the text features themselves)
        per_dim = {}
        for dim in DIM_NAMES:
            vals = np.array([getattr(s, dim) for s in hybrid_states])
            n = len(vals)
            if n >= 10:
                half = n // 2
                r = np.corrcoef(vals[:half], vals[half:2*half])[0, 1]
                per_dim[dim] = {
                    "pearson_r": round(float(r), 4) if not np.isnan(r) else 0.0,
                    "spearman_rho": 0.0,
                    "cohens_kappa": 0.0,
                }
            else:
                per_dim[dim] = {"pearson_r": 0.0, "spearman_rho": 0.0, "cohens_kappa": 0.0}
        overall_agreement = overall_mean_agreement(per_dim)

    # --- Print agreement table ---
    print(f"\n{'─' * 80}")
    print("AGREEMENT: HybridConstraintExtractor (text features) vs independent method")
    if embedding_available:
        print("(Independent method: EmbeddingConstraintExtractor)")
    else:
        print("(Independent method: split-half internal consistency -- embedding unavailable)")
    print(f"{'─' * 80}")
    print(f"{'Dimension':<20s} {'Pearson r':>10s} {'Spearman rho':>12s} {'Cohens kappa':>13s}")
    print(f"{'─' * 80}")
    for dim in DIM_NAMES:
        d = per_dim[dim]
        print(f"{dim:<20s} {d['pearson_r']:>10.4f} {d['spearman_rho']:>12.4f} {d['cohens_kappa']:>13.4f}")
    print(f"{'─' * 80}")
    print(f"{'Overall mean |r|':<20s} {overall_agreement:>10.4f}")
    print(f"N = {len(hybrid_states)} samples")

    # --- (b) Test-retest ---
    print(f"\n{'=' * 80}")
    print("TEST-RETEST RELIABILITY")
    print(f"{'=' * 80}")

    # Use all available texts for test-retest (rule-based, fast)
    tr_results = test_retest_reliability(texts)
    print(f"  Identical on identical inputs: {tr_results['identical_on_identical_inputs']}")
    print(f"  Max difference across 8 features: {tr_results['max_difference']:.2e}")
    print(f"  (Should be True and 0.0 for deterministic rule-based extractor)")

    print(f"\n  Perturbation sensitivity (mean absolute difference):")
    print(f"  {'Perturbation':<20s} {'ei_logic':>8s} {'ei_cv':>8s} {'eii_bigram':>10s} {'eii_trunc':>10s} {'eii_filler':>10s} {'eii_unique':>10s} {'eiii_proper':>12s} {'eiii_num':>10s}")
    for ptype, diffs in tr_results["perturbation_mean_absolute_diff"].items():
        vals = [f"{diffs.get(k, 0):.4f}" for k in sorted(diffs.keys())]
        print(f"  {ptype:<20s} " + " ".join(f"{v:>8s}" for v in vals))

    # --- (c) Variance decomposition ---
    print(f"\n{'=' * 80}")
    print("PER-DIMENSION VARIANCE DECOMPOSITION (across models)")
    print(f"{'=' * 80}")

    # Use all records for variance decomposition
    all_hybrid_states = hybrid.extract_batch(texts)
    var_decomp = variance_decomposition(records, all_hybrid_states)

    print(f"  {'Dimension':<20s} {'Between-Model':>13s} {'Within-Model':>13s} {'S/N Ratio':>10s}")
    print(f"  {'─' * 60}")
    for dim in DIM_NAMES:
        v = var_decomp[dim]
        print(f"  {dim:<20s} {v['between_model_var']:>13.6f} {v['within_model_var']:>13.6f} {v['signal_to_noise']:>10.4f}")

    # Identify worst dimension
    worst_dim = min(DIM_NAMES, key=lambda d: var_decomp[d]["signal_to_noise"])
    print(f"\n  Lowest S/N dimension: {worst_dim} (S/N = {var_decomp[worst_dim]['signal_to_noise']:.4f})")

    # --- (d) LLM Judge comparison ---
    print(f"\n{'=' * 80}")
    print("LLM JUDGE COMPARISON (using existing judge_validation_v2.json)")
    print(f"{'=' * 80}")

    judge_comparison = compare_with_existing_judge_data(records, all_hybrid_states, data_dir)
    if "error" in judge_comparison:
        print(f"  {judge_comparison['error']}")
    else:
        print(f"  Common capabilities: {judge_comparison['common_capabilities']}")
        print(f"  Spearman rho (extractor mean vs judge preference) per dimension:")
        for dim, rho in judge_comparison["per_dim_spearman_vs_judge_pref"].items():
            print(f"    {dim:<20s}: {rho:+.4f}")
        print(f"  Overall constraint mean vs judge pref: rho = {judge_comparison['overall_spearman_vs_judge_pref']:+.4f}")
        print(f"  Note: {judge_comparison['note']}")

    # --- (e) 86% claim analysis ---
    print(f"\n{'=' * 80}")
    print("WHAT '86% AGREEMENT' ACTUALLY MEANS")
    print(f"{'=' * 80}")
    interpretation = analyze_86_percent_claim()
    print(f"  {interpretation}")

    # --- Build output ---
    output = {
        "hybrid_vs_llm_judge": {
            "per_dimension": per_dim,
            "overall_mean_agreement": overall_agreement,
            "n_samples": len(hybrid_states),
            "method_note": (
                "Comparison is HybridConstraintExtractor (text features only) vs "
                "EmbeddingConstraintExtractor (sentence-transformers) as independent "
                "measurement method. Both target the same 5 constraint dimensions. "
                "LLM-judge per-dimension scores are not available in existing data."
                if embedding_available
                else "Split-half reliability of Hybrid extractor (embedding extractor unavailable)."
            ),
        },
        "test_retest": tr_results,
        "per_dimension_variance": var_decomp,
        "llm_judge_validation": judge_comparison,
        "interpretation": interpretation,
    }

    out_path = os.path.join(data_dir, "experiment_data", "extractor_reliability.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n{'=' * 80}")
    print(f"Results saved to: {out_path}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
