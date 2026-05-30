"""MSCE core — constraint conflict detection engine."""

import json
from pathlib import Path

# Built-in Hubble tension analysis data
_HUBBLE_DATA = None


def load_hubble_data():
    """Load the built-in Hubble tension constraint analysis results."""
    global _HUBBLE_DATA
    if _HUBBLE_DATA is not None:
        return _HUBBLE_DATA

    # Embedded results from MSCE v3.0 multi-proposal × multi-constraint analysis
    _HUBBLE_DATA = {
        "context": {
            "tension": "H₀ = 67.4±0.5 (Planck CMB) vs 73.0±1.0 (SH0ES distance ladder)",
            "significance": "~5σ",
            "constraints_checked": 8,
            "proposals_analyzed": 6,
            "combinations_analyzed": 4,
        },
        "proposals": {
            "A_early_dark_energy": {
                "name": "Early Dark Energy (EDE)",
                "mechanism": "Inject ~5% extra dark energy at z~3000, decaying by z~500",
                "parameters": "f_EDE ~ 0.05, z_c ~ 3000, n ~ 3",
                "claimed_effects": "H₀↑ to ~71-73",
                "pass_count": 3,
                "violation_count": 3,
            },
            "B_modified_gravity": {
                "name": "Modified Gravity (f(R))",
                "mechanism": "Late-time (z<1) modification to GR affecting local H₀ measurement",
                "parameters": "f_R0 ~ 10^-6 to 10^-4",
                "claimed_effects": "Only affects z<0.01 H₀ measurement",
                "pass_count": 3,
                "violation_count": 4,
            },
            "C_extra_neutrinos": {
                "name": "Extra Neutrinos (ΔN_eff)",
                "mechanism": "Additional radiation degrees of freedom ΔN_eff≈0.3-0.5",
                "parameters": "ΔN_eff ~ 0.3-0.5, m_sterile ~ 0.1-1 eV",
                "claimed_effects": "H₀↑ to ~70-71",
                "pass_count": 3,
                "violation_count": 2,
            },
            "D_decaying_dark_matter": {
                "name": "Decaying Dark Matter (DDM)",
                "mechanism": "5-10% of DM decays to dark radiation between z~10^4 and z~10^2",
                "parameters": "Γ ~ (10-100 Gyr)^-1",
                "claimed_effects": "H₀↑ to ~70-72",
                "pass_count": 5,
                "violation_count": 2,
            },
            "E_local_void": {
                "name": "Local Void Hypothesis",
                "mechanism": "Milky Way sits in a ~200Mpc underdensity; local H₀ > global",
                "parameters": "δ ~ -0.3, radius ~ 200-300 Mpc",
                "claimed_effects": "No new physics needed; observational bias",
                "pass_count": 6,
                "violation_count": 2,
            },
            "F_systematic_error": {
                "name": "Unknown Systematics",
                "mechanism": "Unrecognized systematics in Planck or SH0ES (or both)",
                "parameters": "No free parameters",
                "claimed_effects": "Tension vanishes if one side has systematic errors",
                "pass_count": 6,
                "violation_count": 0,
            },
        },
        "results": {
            "A_early_dark_energy": {"msce_confidence": 0.076, "msce_disagreement": 0.712, "pass_count": 3, "violation_count": 3},
            "B_modified_gravity": {"msce_confidence": 0.253, "msce_disagreement": 0.093, "pass_count": 3, "violation_count": 4},
            "C_extra_neutrinos": {"msce_confidence": 0.287, "msce_disagreement": 0.590, "pass_count": 3, "violation_count": 2},
            "D_decaying_dark_matter": {"msce_confidence": 0.358, "msce_disagreement": 0.422, "pass_count": 5, "violation_count": 2},
            "E_local_void": {"msce_confidence": 0.171, "msce_disagreement": 0.425, "pass_count": 6, "violation_count": 2},
            "F_systematic_error": {"msce_confidence": 0.108, "msce_disagreement": 0.582, "pass_count": 6, "violation_count": 0},
        },
        "constraint_matrix": {
            "cmb_power_spectrum":   [1, 1, 1, 0, 0, 1],
            "bao_scale":            [2, 0, 2, 1, 0, 0],
            "supernova_hubble":     [1, 1, 0, 1, 2, 0],
            "bbn_abundances":       [0, 0, 1, 0, 0, 0],
            "s8_tension":           [2, 0, 0, 1, 0, 2],
            "universe_age":         [0, 0, 0, 1, 0, 0],
            "gravity_tests":        [0, 2, 0, 0, 0, 0],
            "cross_constraint":     [2, 2, 2, 2, 1, 2],
        },
        "constraint_labels": [
            "CMB Power\nSpectrum",
            "BAO\nScale",
            "Supernova\nHubble Diag.",
            "BBN\nAbundances",
            "S₈\nTension",
            "Universe\nAge",
            "Gravity\nTests",
            "Cross-Constraint\nConsistency",
        ],
        "proposal_labels": ["EDE", "f(R)", "ΔN_eff", "DDM", "Void", "Syst."],
        "combinations": {
            "EDE + Local Void": {"msce_confidence": 0.075},
            "ΔN_eff + Local Void": {"msce_confidence": 0.208},
            "EDE + ΔN_eff": {"msce_confidence": 0.147},
            "DDM + Local Void": {"msce_confidence": 0.317},
        },
        "residual_vector": {
            "cross_constraint_consistency": 1.83,
            "s8_tension": 1.00,
            "cmb_power_spectrum": 0.83,
            "supernova_hubble_diagram": 0.83,
            "bao_scale": 0.67,
            "gravity_tests": 0.33,
            "bbn_abundances": 0.17,
            "universe_age": 0.17,
        },
    }
    return _HUBBLE_DATA


def analyze(target="hubble_tension", quick=True, **kwargs):
    """Run a constraint conflict analysis.

    Args:
        target: 'hubble_tension' (built-in) or path to custom config.
        quick: Use cached results if available.
        **kwargs: Additional domain-specific parameters.

    Returns:
        dict with keys: confidence, all_fail, proposals, heatmap_data
    """
    if target == "hubble_tension":
        data = load_hubble_data()
        results = data["results"]
        all_conf = [r["msce_confidence"] for r in results.values()]
        return {
            "target": "hubble_tension",
            "confidence": max(all_conf),
            "all_fail": max(all_conf) < 0.36,
            "num_proposals": len(results),
            "num_constraints": 8,
            "proposals": {
                k: {"name": data["proposals"][k]["name"], "confidence": v["msce_confidence"]}
                for k, v in results.items()
            },
            "heatmap_data": data["constraint_matrix"],
            "constraint_labels": data["constraint_labels"],
            "proposal_labels": data["proposal_labels"],
            "combinations": data["combinations"],
            "residual_vector": data["residual_vector"],
        }
    else:
        raise NotImplementedError(f"Target '{target}' not yet supported. Try 'hubble_tension'.")


def check(theory, constraints=None, domain="cosmology", **kwargs):
    """Check a theory against a set of constraints.

    This is the public API entry point. In v0.1.0, supports the built-in
    Hubble tension analysis. Custom theory checking comes in v0.2.0.

    Args:
        theory: Description of the theory to check.
        constraints: List of constraint names.
        domain: Scientific domain (default: cosmology).
        **kwargs: Additional parameters.

    Returns:
        dict with constraint check results.
    """
    if theory.lower() in ("hubble", "hubble_tension", "h0 tension"):
        return analyze("hubble_tension", **kwargs)

    return {
        "status": "not_implemented",
        "message": (
            "Custom theory checking is coming in v0.2.0. "
            "For now, try: msce.analyze('hubble_tension') "
            "to see the built-in Hubble tension analysis."
        ),
        "theory": theory,
        "domain": domain,
    }
