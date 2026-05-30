"""MSCE Quickstart — run the Hubble tension analysis in 3 lines."""

import msce

# 3-line quick check
print("=== MSCE Hubble Tension Quick Check ===\n")
result = msce.analyze("hubble_tension", quick=True)

print(f"Proposals analyzed: {result['num_proposals']}")
print(f"Constraints checked: {result['num_constraints']}")
print(f"Best confidence: {result['confidence']:.3f}")
print(f"All proposals fail: {result['all_fail']}")
print()

print("Per-proposal confidence:")
for key, info in result["proposals"].items():
    bar = "█" * int(info["confidence"] * 20) + "░" * (20 - int(info["confidence"] * 20))
    print(f"  {info['name']:<40} [{bar}] {info['confidence']:.3f}")

print()

# Combination results
print("Combination search:")
for name, info in result["combinations"].items():
    print(f"  {name:<35} confidence = {info['msce_confidence']:.3f}")

print()
print("Try the full notebook: notebooks/hubble_tension.ipynb")

# Optional: generate heatmap
try:
    fig = msce.heatmap(
        result["heatmap_data"],
        constraint_labels=result["constraint_labels"],
        proposal_labels=result["proposal_labels"],
        title="Hubble Tension: 6 Solutions × 8 Constraints"
    )
    fig.savefig("hubble_heatmap.png", dpi=150, bbox_inches="tight")
    print("\nHeatmap saved to hubble_heatmap.png")
except ImportError:
    print("\n(matplotlib not installed — skipping heatmap. pip install msce[notebook] for visuals)")
