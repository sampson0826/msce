"""MSCE Audit: Yann LeCun — LeWorldModel (arXiv 2603.19312, March 2026)
Cross-constraint verification of core claims.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from product_engine import run_msce

QUESTIONS = [
    {
        "id": "Q1",
        "claim": "SIGReg (Gaussian prior) is the universal anti-collapse solution",
        "question": (
            "The LeWorldModel paper (LeCun et al., 2026) claims that Sketched Isotropic Gaussian "
            "Regularization (SIGReg) prevents JEPA representation collapse, and that the isotropic "
            "Gaussian is 'provably optimal' as the latent distribution for minimizing downstream "
            "prediction risk. The paper presents this as a general solution that enables stable "
            "end-to-end training from pixels with just 2 loss terms.\n\n"
            "Critical question: Does the evidence support the claim that an isotropic Gaussian prior "
            "is the universal anti-collapse solution for JEPA world models? Consider these constraints:\n"
            "(1) The Cramér-Wold theorem ensures Gaussian marginals imply Gaussian joint — but this "
            "is a descriptive property, not a proof that Gaussian is optimal for world model latents. "
            "It tells you what you've regularized toward, not why that regularization target is correct.\n"
            "(2) A concurrent paper (Sub-JEPA, May 2026, arXiv:2605.09241) explicitly contradicts "
            "this claim, showing that 'latent representations inherently lie on low-dimensional "
            "manifolds within a high-dimensional ambient space, and enforcing an isotropic Gaussian "
            "prior directly in this ambient space introduces an overly strong bias.' Sub-JEPA "
            "outperforms LeWM 'with very clear margins' by relaxing the Gaussian constraint.\n"
            "(3) The paper itself acknowledges SIGReg effectiveness drops in low intrinsic-dimension "
            "environments (Two-Room). This suggests Gaussian is not universal — it's conditional on "
            "environment dimensionality.\n"
            "(4) The 'optimality proof' (from LeJEPA 2025) relies on the assumption that minimizing "
            "prediction risk in latent space is the correct objective. But if the encoder maps "
            "nonlinear dynamics to a space where Gaussian is enforced, the encoder may simply learn "
            "to discard non-Gaussian structure — which is information loss, not optimal representation.\n"
            "Answer: TRUE (Gaussian is universal anti-collapse solution) or FALSE (overclaimed, "
            "conditional, or contradicted) with reasoning."
        ),
    },
    {
        "id": "Q2",
        "claim": "LeWM achieves 48x faster planning vs DINO-WM — a meaningful performance comparison",
        "question": (
            "The LeWorldModel paper claims LeWM plans '48x faster' than DINO-WM, and presents this "
            "as a key advantage. But is this comparison meaningful?\n\n"
            "Consider these constraints:\n"
            "(1) DINO-WM was designed for representation quality using pretrained DINOv2 features, "
            "not for planning speed. Its dense feature maps are ~200x larger than LeWM's latent. "
            "Comparing planning speed without controlling for representation dimensionality is like "
            "comparing a bicycle's speed to a truck's speed without noting the truck carries more cargo.\n"
            "(2) The paper's own data shows GCBC (goal-conditioned behavioral cloning) plans in "
            "~0.01 seconds — 100x faster than LeWM — but the paper dismisses this because GCBC "
            "'performs poorly.' This reveals an implicit trade-off: speed vs. performance. The 48x "
            "figure isolates the favorable side of one comparison pair while ignoring the other.\n"
            "(3) When DINO-WM outperforms LeWM on control tasks (OGBench-Cube, Two-Room), the "
            "paper attributes it to DINO-WM's pretrained features — but doesn't apply the same "
            "contextualization to the 48x speed claim, which is entirely explained by latent "
            "dimensionality, not architectural innovation.\n"
            "(4) Planning time is measured in seconds on a single GPU. For real-world robotics, "
            "the difference between 1s and 47s is meaningful. But the paper doesn't compare against "
            "the most natural baseline: a dimension-reduced DINO-WM (e.g., PCA on latent features).\n"
            "Answer: FAIR (the 48x comparison is meaningful and well-contextualized) or UNFAIR "
            "(the comparison conflates speed with representation capacity, lacks proper controls) "
            "with reasoning."
        ),
    },
    {
        "id": "Q3",
        "claim": "LeWM needs only 1 hyperparameter (λ=0.1), making it universally simple",
        "question": (
            "The LeWorldModel paper's central narrative is simplification: from PLDM's 6 loss terms "
            "with 6 hyperparameters down to 'just 1 effective hyperparameter' (λ, the SIGReg weight). "
            "The paper uses λ=0.1 as a fixed value across all experiments.\n\n"
            "Does this 1-hyperparameter claim hold under scrutiny?\n\n"
            "Consider these constraints:\n"
            "(1) The Sub-JEPA paper (May 2026) explicitly shows that LeWM's single Gaussian prior is "
            "'overly strong' and proposes subspace regularization as an improvement — which adds "
            "hyperparameters (number of subspaces, subspace dimension). If the 1-hyperparameter "
            "solution were universally sufficient, Sub-JEPA wouldn't consistently outperform it.\n"
            "(2) The paper's own results show LeWM underperforms on Two-Room and OGBench-Cube. "
            "No ablation shows whether tuning λ per-environment closes this gap. A universal λ "
            "that underperforms on 2/4 tested environments raises questions about universality.\n"
            "(3) The architecture still has implicit hyperparameters: encoder depth (4 ViT layers), "
            "predictor depth (6 transformer layers), latent dimension, number of random projections "
            "(M=1024), learning rate schedule, batch size. Calling it '1 hyperparameter' shifts "
            "these to fixed architectural choices without justifying why they don't count.\n"
            "(4) The paper doesn't report sensitivity analysis for λ. If performance is highly "
            "sensitive to λ (e.g., λ=0.05 or λ=0.2 significantly changes results), then λ is "
            "effectively a critical tuning parameter, not a 'set and forget' value.\n"
            "Answer: TRUE (1 hyperparameter claim is valid and well-supported) or FALSE "
            "(the claim overstates simplicity; implicit hyperparameters and environment-specific "
            "tuning are still needed) with reasoning."
        ),
    },
    {
        "id": "Q4",
        "claim": "End-to-end JEPA from pixels is 'solved' — stable training without engineering tricks",
        "question": (
            "The LeWorldModel paper frames its contribution as 'solving' the JEPA collapse problem: "
            "stable end-to-end training from raw pixels without EMA, stop-gradient, or pretrained "
            "encoders. The paper's title — 'Stable End-to-End Joint-Embedding Predictive Architecture "
            "from Pixels' — implies this is achieved.\n\n"
            "Is this claim well-supported?\n\n"
            "Consider these constraints:\n"
            "(1) The paper tests on only 4 environments: Push-T (2D), Reacher (robot arm), "
            "OGBench-Cube (3D grasp), Two-Room (2D navigation). All are relatively simple, "
            "low-resolution control tasks. No natural images, no video, no high-dimensional "
            "observation spaces. 'Solved on 4 low-dim control tasks' ≠ 'solved from pixels.'\n"
            "(2) The paper acknowledges SIGReg effectiveness drops in low intrinsic-dimension "
            "environments (Two-Room). If the core mechanism breaks in edge cases, the claim of "
            "'stability' is conditional, not general.\n"
            "(3) A concurrent May 2026 benchmark study (TechTimes, May 31, 2026) found 'current "
            "world models are brittle' — directly contradicting the 'solved' narrative. If LeWM "
            "had truly solved JEPA collapse, a benchmark published 2 months later wouldn't report "
            "brittleness as the headline finding.\n"
            "(4) The paper's comparison to PLDM (which used 6 loss terms) is the primary evidence "
            "of 'solving' collapse. But PLDM is a single baseline from 2025. The paper doesn't "
            "compare against other collapse-prevention methods (e.g., VicReg, Barlow Twins, SwAV) "
            "adapted to the JEPA setting, making the 'solved' claim relative to a weak strawman.\n"
            "Answer: TRUE (JEPA collapse is solved) or FALSE (claim overreaches; evidence limited "
            "to few simple domains, contradicted by independent benchmarks) with reasoning."
        ),
    },
    {
        "id": "Q5",
        "claim": "LeWM's SIGReg-learned latents capture 'physical understanding' without supervision",
        "question": (
            "The LeWorldModel paper claims emergent physical understanding: linear probing reveals "
            "that LeWM's latent space encodes position, velocity, and angles with r=0.97-0.99 "
            "correlation, and Violation-of-Expectation (VoE) experiments show the model expresses "
            "'surprise' at physically impossible events (teleportation) but not superficial changes "
            "(color swaps).\n\n"
            "Does this evidence support the claim of 'physical understanding'?\n\n"
            "Consider these constraints:\n"
            "(1) Linear decodability of state variables is expected, not surprising. A world model "
            "trained to predict next-state latents must encode current state to make predictions. "
            "If it couldn't linearly encode position, the predictor would have no usable input. "
            "This is a necessary condition for the task, not evidence of 'understanding.'\n"
            "(2) VoE with teleportation vs. color change is a weak test. The model is trained to "
            "predict next-frame latents from position-relevant features. Teleportation breaks "
            "temporal continuity — which the MSE prediction loss directly penalizes. Color is "
            "likely ignored by the encoder as task-irrelevant (the control tasks don't depend on "
            "color). The model isn't 'choosing' to be indifferent to color — it learned to discard "
            "it during training because color doesn't predict reward or next state.\n"
            "(3) The paper doesn't test intermediate cases: partial occlusion, elastic deformation, "
            "or velocity discontinuities. Physical 'understanding' that only distinguishes between "
            "complete teleportation and complete normality hasn't demonstrated the graded sensitivity "
            "that characterizes genuine physical intuition.\n"
            "(4) The r=0.97-0.99 correlation is in-distribution. No out-of-distribution "
            "generalization test is reported. If the model encounters a novel object or dynamics, "
            "do the linear readouts maintain their accuracy? Without this, 'understanding' is "
            "indistinguishable from 'memorized the training distribution.'\n"
            "Answer: TRUE (evidence supports emergent physical understanding) or FALSE "
            "(expected behavior for the training objective, not evidence of emergent understanding) "
            "with reasoning."
        ),
    },
]

def main():
    results = []
    print("=" * 70)
    print("MSCE Audit: LeCun — LeWorldModel (arXiv 2603.19312, March 2026)")
    print("Cross-Constraint Verification of 5 Core Claims")
    print("=" * 70)

    for q in QUESTIONS:
        print(f"\n{'─' * 60}")
        print(f"[{q['id']}] Testing claim: {q['claim']}")
        print(f"{'─' * 60}")

        t0 = time.time()
        result = run_msce(q["question"], domain="science")
        elapsed = time.time() - t0

        result["claim"] = q["claim"]
        result["question_id"] = q["id"]
        results.append(result)

        print(f"  Answer:       {result.get('top_answer', 'N/A')[:120]}")
        print(f"  Confidence:   {result.get('confidence', 0):.3f}")
        print(f"  Disagreement: {result.get('disagreement', 0):.3f}")
        print(f"  Uncertain:    {result.get('uncertain', 'N/A')}")
        print(f"  Top Strategy: {result.get('top_strategy', 'N/A')}")
        print(f"  Time:         {elapsed:.1f}s")

        for entry in result.get("reasoning_trail", []):
            status = entry.get("status", "?")
            icon = {"selected": "+", "contributing": "~", "outlier": "!", "low_confidence": "-"}
            print(f"    [{icon.get(status, '?')}] {entry['strategy']:25s} | "
                  f"conf={entry.get('self_confidence', 0):.2f} | "
                  f"score={entry.get('judge_score', 0):.1f} | "
                  f"wt={entry.get('weight', 0):.3f} | "
                  f"{status}")

    outpath = os.path.join(os.path.dirname(__file__), "results/lecun_worldmodel_audit_results.json")
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nFull results saved: {outpath}")

    print("\n" + "=" * 70)
    print("VERDICT SUMMARY")
    print("=" * 70)
    for r in results:
        print(f"  [{r['question_id']}] {r['claim']}:")
        print(f"      confidence={r.get('confidence',0):.3f}  disagreement={r.get('disagreement',0):.3f}  uncertain={r.get('uncertain','?')}")

if __name__ == "__main__":
    main()
