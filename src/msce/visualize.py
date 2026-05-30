"""MSCE visualization — heatmaps and confidence charts."""


def heatmap(constraint_matrix, constraint_labels=None, proposal_labels=None, title="Constraint Conflict Heatmap"):
    """Generate a matplotlib heatmap of constraint check results.

    Args:
        constraint_matrix: 2D list (n_constraints × n_proposals), values 0=pass, 1=tension, 2=violation
        constraint_labels: List of constraint names
        proposal_labels: List of proposal names
        title: Chart title

    Returns:
        matplotlib Figure
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import numpy as np
    except ImportError:
        raise ImportError("matplotlib required. Install with: pip install msce[notebook]")

    data = np.array(constraint_matrix)
    n_rows, n_cols = data.shape

    cmap = mcolors.ListedColormap(["#2ecc71", "#f39c12", "#e74c3c"])
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(max(8, n_cols * 1.2), max(6, n_rows * 0.8)))
    im = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(range(n_cols))
    ax.set_yticks(range(n_rows))
    if proposal_labels:
        ax.set_xticklabels(proposal_labels, fontsize=10)
    if constraint_labels:
        ax.set_yticklabels(constraint_labels, fontsize=10)

    # Color-coded text in cells
    for i in range(n_rows):
        for j in range(n_cols):
            val = data[i, j]
            label = ["PASS", "TENSION", "FAIL"][val]
            color = ["#1a7a3a", "#8b6914", "#8b1a1a"][val]
            ax.text(j, i, label, ha="center", va="center", fontsize=9,
                    fontweight="bold", color=color)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    return fig


def confidence_bars(proposals, scores, title="MSCE Confidence by Proposal"):
    """Generate a horizontal bar chart of confidence scores.

    Args:
        proposals: List of proposal names
        scores: List of confidence scores (0-1)
        title: Chart title

    Returns:
        matplotlib Figure
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        raise ImportError("matplotlib required. Install with: pip install msce[notebook]")

    colors = ["#e74c3c" if s < 0.2 else "#f39c12" if s < 0.5 else "#2ecc71" for s in scores]

    fig, ax = plt.subplots(figsize=(10, 5))
    y_pos = range(len(proposals))
    bars = ax.barh(y_pos, scores, color=colors, edgecolor="white", linewidth=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(proposals, fontsize=11)
    ax.set_xlabel("MSCE Confidence", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axvline(x=0.36, color="red", linestyle="--", alpha=0.5, label="Pass threshold (0.36)")
    ax.set_xlim(0, 1)
    ax.legend()

    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{score:.3f}", va="center", fontsize=10)

    plt.tight_layout()
    return fig
