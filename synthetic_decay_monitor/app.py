"""HuggingFace Gradio Demo — Synthetic Data Decay Monitor.

Constraint-layer health diagnostic for AI training pipelines.
No GPU required — uses hybrid text-feature extraction.

Run locally: python app.py
Deploy: push to HuggingFace Spaces
"""

import io
import base64
import numpy as np
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from collections import defaultdict

from synthetic_decay_monitor.data_lineage import (
    generate_synthetic_lineage, parse_lineage_from_jsonl,
)
from synthetic_decay_monitor.constraint_extractor import (
    HybridConstraintExtractor,
)
from synthetic_decay_monitor.decay_engine import (
    DecayEngine, S_CRITICAL, BASE_ALPHAS,
)
from synthetic_decay_monitor.executor_classifier import (
    diagnose_executor_decay, ExecutorClassifier,
)

# ---- style ----
DARK_BG = "#0a0a14"
CARD_BG = "#0f0f22"
ACCENT = "#88ddff"
COLORS = ["#44dd88", "#88ccff", "#ffaa44", "#ff6688", "#cc66ff",
          "#44ddcc", "#ffdd44", "#66aacc", "#ff8866", "#cc88ff"]
STATUS_COLORS = {"healthy": "#44dd88", "degrading": "#ffaa44",
                 "critical": "#ff6644", "collapsed": "#ff2222"}

plt.rcParams.update({
    "figure.facecolor": DARK_BG, "axes.facecolor": CARD_BG,
    "axes.edgecolor": "#1a1a3a", "axes.labelcolor": "#8899cc",
    "text.color": "#d0d0e0", "xtick.color": "#667799",
    "ytick.color": "#667799", "grid.color": "#1a1a3a",
    "legend.facecolor": CARD_BG, "legend.edgecolor": "#1a1a3a",
    "legend.labelcolor": "#8899cc", "font.size": 10,
})

# ---- demo texts (fact-based with all 3 executor marker types) ----
DEMO_TEXTS = [
    "The capital of France is Paris, with a population of approximately 2.1 million residents. Therefore, the city faces significant infrastructure challenges due to its high density.",
    "The Eiffel Tower was completed in 1889 and stands 330 meters tall. It attracts over 7 million visitors annually, making it one of the most visited monuments in the world.",
    "The water cycle consists of evaporation, condensation, and precipitation. These processes are driven by solar energy and gravity, recycling 577,000 cubic kilometers of water each year.",
    "Because of the greenhouse effect, Earth's average temperature has risen by 1.1 degrees Celsius since 1880. However, the rate of warming varies significantly by region.",
    "The Amazon rainforest spans approximately 5.5 million square kilometers across 9 countries. It produces about 20% of the world's oxygen through photosynthesis.",
    "Python, created by Guido van Rossum in 1991, supports multiple programming paradigms. Therefore, developers can choose between procedural, object-oriented, and functional approaches.",
    "The human genome contains approximately 3 billion base pairs, organized into 23 chromosome pairs. However, only about 1.5% of this DNA actually codes for proteins.",
    "Mount Everest rises 8,848 meters above sea level at the border of Nepal and Tibet. Because of tectonic plate movement, it continues to grow approximately 4 millimeters each year.",
    "The Great Wall of China stretches 21,196 kilometers across northern China. Construction began in the 7th century BC, though most of the existing structure dates from the Ming Dynasty.",
    "Albert Einstein published his theory of special relativity in 1905. Therefore, our understanding of space, time, and energy was fundamentally transformed by the equation E=mc².",
]


def fig_to_base64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor=DARK_BG)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def plot_stability_trajectories(trajectories: list[dict]) -> str:
    """Plot S_n vs generation for all capabilities."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.set_facecolor(CARD_BG)

    for i, t in enumerate(trajectories):
        if "trajectory" not in t or not t["trajectory"]:
            continue
        gens = [g["generation"] for g in t["trajectory"]]
        svals = [g["S_n"] for g in t["trajectory"]]
        color = COLORS[i % len(COLORS)]
        ax.plot(gens, svals, "o-", color=color, linewidth=2.2, markersize=6,
                label=t["capability"], zorder=3)

    ax.axhline(y=S_CRITICAL, color="#ff4444", linewidth=1.5, linestyle="--", alpha=0.8)
    ax.text(ax.get_xlim()[1] - 0.3, S_CRITICAL - 0.04, f"Collapse (S={S_CRITICAL})",
            color="#ff4444", fontsize=9, ha="right", va="top")

    ax.fill_between([0, ax.get_xlim()[1]], 0.8, 1.05, color="#44dd88", alpha=0.05)
    ax.fill_between([0, ax.get_xlim()[1]], 0.5, 0.8, color="#ffaa44", alpha=0.05)
    ax.fill_between([0, ax.get_xlim()[1]], S_CRITICAL, 0.5, color="#ff6644", alpha=0.05)
    ax.fill_between([0, ax.get_xlim()[1]], 0, S_CRITICAL, color="#ff2222", alpha=0.05)

    ax.set_xlabel("Generation")
    ax.set_ylabel("Stability S_n")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(left=0)
    ax.legend(loc="lower left", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    img = fig_to_base64(fig)
    plt.close(fig)
    return img


def plot_executor_composition(trajectories: list[dict]) -> str:
    """Horizontal stacked bar of E-I/E-II/E-III per capability."""
    caps = []
    ei_vals, eii_vals, eiii_vals = [], [], []
    for t in trajectories:
        if "trajectory" not in t or not t["trajectory"]:
            continue
        last = t["trajectory"][-1]
        comp = last.get("executor_composition", {})
        caps.append(t["capability"])
        ei_vals.append(comp.get("E-I", 0))
        eii_vals.append(comp.get("E-II", 0))
        eiii_vals.append(comp.get("E-III", 0))

    if not caps:
        return ""

    fig, ax = plt.subplots(figsize=(8, max(2.5, len(caps) * 0.45)))
    ax.set_facecolor(CARD_BG)
    y = np.arange(len(caps))
    bar_h = 0.55

    ax.barh(y, ei_vals, bar_h, color="#ff6644", label="E-I (Axiom)", alpha=0.9)
    ax.barh(y, eii_vals, bar_h, left=ei_vals, color="#ffaa44", label="E-II (Scale)", alpha=0.9)
    left_mid = [e + ee for e, ee in zip(ei_vals, eii_vals)]
    ax.barh(y, eiii_vals, bar_h, left=left_mid, color="#44aadd", label="E-III (Boundary)", alpha=0.9)

    ax.set_yticks(y)
    ax.set_yticklabels(caps, fontsize=9)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Executor Proportion")
    ax.legend(loc="lower right", fontsize=8, ncol=3)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    img = fig_to_base64(fig)
    plt.close(fig)
    return img


def build_diagnosis_table(trajectories: list[dict], diagnoses: list[dict]) -> str:
    """HTML table of per-capability diagnosis."""
    rows = []
    for t in trajectories:
        cap = t.get("capability", "")
        if "trajectory" not in t or not t["trajectory"]:
            continue
        last = t["trajectory"][-1]
        status = last.get("status", "unknown")
        sc = STATUS_COLORS.get(status, "#889")

        diag = next((d for d in diagnoses if d.get("capability") == cap), None)
        dtype = diag.get("diagnosis", None) if diag else None
        if hasattr(dtype, "degradation_type"):
            dtype_str = dtype.degradation_type.replace("_", " ")
            severity = dtype.severity
            intervention = dtype.intervention_type.replace("_", " ").title()
        elif isinstance(dtype, dict):
            dtype_str = dtype.get("degradation_type", "?").replace("_", " ")
            severity = dtype.get("severity", 0)
            intervention = dtype.get("intervention_type", "?").replace("_", " ").title()
        else:
            dtype_str = "—"
            severity = 0
            intervention = "Monitor"

        sev_color = "#ff4444" if severity > 0.7 else "#ffaa44" if severity > 0.4 else "#44dd88"

        rows.append(f"""<tr>
        <td style="font-weight:bold">{cap}</td>
        <td style="color:{sc}">● {status.upper()}</td>
        <td>{last['S_n']:.3f}</td>
        <td>{last['beta']:.3f}</td>
        <td>{dtype_str}</td>
        <td style="color:{sev_color}">{severity:.0%}</td>
        <td>{intervention}</td>
        </tr>""")

    if not rows:
        return '<div style="color:#667;font-size:11px">No diagnoses available.</div>'

    return f"""<table style="width:100%;border-collapse:collapse;font-size:11px;margin:8px 0">
    <tr style="background:#111133;color:#889">
    <th>Capability</th><th>Status</th><th>S_n</th><th>β</th><th>Type</th><th>Severity</th><th>Intervention</th>
    </tr>
    {"".join(rows)}
    </table>"""


def build_summary_metrics(lineage, trajectories: list[dict], diagnoses: list[dict]) -> str:
    """HTML summary metrics cards."""
    n_critical = 0
    for t in trajectories:
        if "trajectory" in t and t["trajectory"]:
            last = t["trajectory"][-1]
            if last.get("status") == "collapsed":
                n_critical += 1

    n_degrading = 0
    for t in trajectories:
        if "trajectory" in t and t["trajectory"]:
            last = t["trajectory"][-1]
            if last.get("status") in ("degrading", "critical"):
                n_degrading += 1

    global_beta = np.mean([
        t["trajectory"][-1]["beta"]
        for t in trajectories
        if "trajectory" in t and t["trajectory"]
    ]) if trajectories else 0.25

    caps = [t["capability"] for t in trajectories if "trajectory" in t and t["trajectory"]]

    return f"""
    <div style="display:flex;gap:12px;flex-wrap:wrap;margin:8px 0">
    <div style="background:#0f0f22;border:1px solid #1a1a3a;border-radius:8px;padding:12px 18px;text-align:center">
    <div style="font-size:24px;font-weight:bold;color:#88ddff">{lineage.n_generations}</div>
    <div style="font-size:10px;color:#667">Generations</div>
    </div>
    <div style="background:#0f0f22;border:1px solid #1a1a3a;border-radius:8px;padding:12px 18px;text-align:center">
    <div style="font-size:24px;font-weight:bold;color:#88ddff">{len(caps)}</div>
    <div style="font-size:10px;color:#667">Capabilities</div>
    </div>
    <div style="background:#0f0f22;border:1px solid #1a1a3a;border-radius:8px;padding:12px 18px;text-align:center">
    <div style="font-size:24px;font-weight:bold;color:#88ddff">{global_beta:.3f}</div>
    <div style="font-size:10px;color:#667">Global β</div>
    </div>
    <div style="background:#0f0f22;border:1px solid #1a1a3a;border-radius:8px;padding:12px 18px;text-align:center">
    <div style="font-size:24px;font-weight:bold;color:#ff6644">{n_critical}</div>
    <div style="font-size:10px;color:#667">Collapsed</div>
    </div>
    <div style="background:#0f0f22;border:1px solid #1a1a3a;border-radius:8px;padding:12px 18px;text-align:center">
    <div style="font-size:24px;font-weight:bold;color:#ffaa44">{n_degrading}</div>
    <div style="font-size:10px;color:#667">Degrading</div>
    </div>
    </div>"""


def build_warnings(trajectories: list[dict], diagnoses: list[dict]) -> str:
    """Critical warnings HTML."""
    html = ""
    for t in trajectories:
        if "trajectory" not in t or not t["trajectory"]:
            continue
        last = t["trajectory"][-1]
        status = last.get("status", "")
        if status == "collapsed":
            html += f"""<div style="padding:4px 8px;margin:2px 0;background:#221111;border-left:3px solid #ff4444;font-size:11px">
            <b style="color:#ff4444">COLLAPSED</b>: {t['capability']} at generation {last['generation']} (S={last['S_n']:.3f})
            </div>"""
        elif status == "critical":
            html += f"""<div style="padding:4px 8px;margin:2px 0;background:#221111;border-left:3px solid #ffaa44;font-size:11px">
            <b style="color:#ffaa44">CRITICAL</b>: {t['capability']} approaching collapse (S={last['S_n']:.3f} at gen {last['generation']})
            </div>"""

    for d in diagnoses:
        diag = d.get("diagnosis", None)
        if hasattr(diag, "degradation_type") and "E-I_loss" in diag.degradation_type:
            html += f"""<div style="padding:4px 8px;margin:2px 0;background:#221111;border-left:3px solid #ff4444;font-size:11px">
            <b style="color:#ff4444">E-I LOSS</b>: {d['capability']} — fastest collapse pattern
            </div>"""

    if not html:
        html = '<div style="color:#667;font-size:11px">No critical warnings.</div>'
    return html


def build_collapse_order(engine) -> str:
    """Collapse order ranking HTML."""
    order = engine.get_collapse_order()
    if not order:
        return '<div style="color:#667;font-size:11px">No collapse data.</div>'

    rows = []
    for i, item in enumerate(order[:8]):
        sc = STATUS_COLORS.get(
            "collapsed" if item["current_S_n"] < S_CRITICAL else
            "critical" if item["current_S_n"] < 0.5 else
            "degrading" if item["current_S_n"] < 0.8 else "healthy",
            "#889"
        )
        arrow = "⚠" if item["current_S_n"] < S_CRITICAL else "→"
        rows.append(f"""<tr>
        <td>{i + 1}</td>
        <td style="font-weight:bold">{item['capability']}</td>
        <td style="color:{sc}">{arrow} gen {item['predicted_collapse_gen']}</td>
        <td>{item['current_S_n']:.3f}</td>
        <td>{item['beta']:.3f}</td>
        </tr>""")

    return f"""<table style="width:100%;border-collapse:collapse;font-size:11px;margin:8px 0">
    <tr style="background:#111133;color:#889">
    <th>#</th><th>Capability</th><th>Collapse</th><th>S_n</th><th>β</th>
    </tr>
    {"".join(rows)}
    </table>"""


def build_intervention_cards(diagnoses: list[dict]) -> str:
    """Intervention recommendation cards."""
    by_type = defaultdict(lambda: {"caps": [], "desc": "", "icon": ""})
    for d in diagnoses:
        diag = d.get("diagnosis", None)
        if hasattr(diag, "intervention_type"):
            itype = diag.intervention_type
            by_type[itype]["caps"].append(d.get("capability", "?"))
            by_type[itype]["desc"] = diag.description
        elif isinstance(diag, dict):
            itype = diag.get("intervention_type", "monitor")
            by_type[itype]["caps"].append(d.get("capability", "?"))
            by_type[itype]["desc"] = diag.get("description", "")

    icons = {
        "add_axiom_data": "Axiom", "add_calibration_data": "Calibrate",
        "add_boundary_data": "Boundary", "add_mixed_data": "Mixed",
        "monitor": "Monitor",
    }
    colors_map = {
        "add_axiom_data": "#ff6644", "add_calibration_data": "#ffaa44",
        "add_boundary_data": "#44aadd", "add_mixed_data": "#cc66ff",
        "monitor": "#44dd88",
    }

    cards = ""
    for itype, info in by_type.items():
        icon = icons.get(itype, "?")
        color = colors_map.get(itype, "#889")
        caps_str = ", ".join(info["caps"][:4])
        cards += f"""<div style="background:#0f0f22;border:1px solid #1a1a3a;border-left:3px solid {color};border-radius:6px;padding:12px;margin:4px;min-width:200px">
        <div style="font-weight:bold;color:{color};font-size:12px">{icon}</div>
        <div style="font-size:10px;color:#889;margin-top:4px">{caps_str}</div>
        <div style="font-size:10px;color:#667;margin-top:4px">{info['desc'][:120]}</div>
        </div>"""

    return f'<div style="display:flex;flex-wrap:wrap;gap:8px;margin:8px 0">{cards}</div>' if cards else \
           '<div style="color:#667;font-size:11px">No interventions needed.</div>'


def run_analysis(text_input: str, n_generations: int, decay_beta: float, use_demo: bool):
    """Core analysis pipeline."""
    try:
        if use_demo or not text_input.strip():
            texts = DEMO_TEXTS[:10]
            lineage = generate_synthetic_lineage(
                texts, n_generations=n_generations,
                decay_pattern={"*": decay_beta},
            )
            source = f"demo ({n_generations} gens, β={decay_beta})"
        elif text_input.strip().startswith("{"):
            # JSONL input
            import tempfile
            with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
                f.write(text_input.strip())
                tmp_path = f.name
            try:
                lineage = parse_lineage_from_jsonl(tmp_path)
                source = f"JSONL ({lineage.n_samples} samples)"
            finally:
                os.unlink(tmp_path)
        else:
            # Plain text — generate synthetic lineage
            texts = [t.strip() for t in text_input.strip().split("\n") if t.strip()]
            if not texts:
                texts = DEMO_TEXTS[:5]
            lineage = generate_synthetic_lineage(
                texts, n_generations=n_generations,
                decay_pattern={"*": decay_beta},
            )
            source = f"custom text ({len(texts)} inputs)"

        # Extract
        extractor = HybridConstraintExtractor(judge_fn=None)
        engine = DecayEngine(lineage, extractor)
        engine.run_all_capabilities()
        trajectories = engine.get_all_trajectories()

        # Diagnose
        classifier = ExecutorClassifier()
        diagnoses = []
        for cap, snapshots in engine._snapshots.items():
            diag = diagnose_executor_decay(
                engine._trajectories.get(cap, []),
                snapshots,
                capability=cap,
            )
            diagnoses.append(diag)

        # Build outputs
        stability_plot = plot_stability_trajectories(trajectories)
        composition_plot = plot_executor_composition(trajectories)
        summary_html = build_summary_metrics(lineage, trajectories, diagnoses)
        warnings_html = build_warnings(trajectories, diagnoses)
        diag_table_html = build_diagnosis_table(trajectories, diagnoses)
        collapse_html = build_collapse_order(engine)
        intervention_html = build_intervention_cards(diagnoses)

        output_md = f"**Source:** {source}  |  **Extractor:** Hybrid (text features, CPU)"

        return (
            output_md,
            summary_html,
            warnings_html,
            f'<div style="text-align:center"><img src="data:image/png;base64,{stability_plot}" style="max-width:100%"></div>',
            f'<div style="text-align:center"><img src="data:image/png;base64,{composition_plot}" style="max-width:100%"></div>',
            diag_table_html,
            collapse_html,
            intervention_html,
        )
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        return (
            f"**Error:** {str(e)}",
            "", "", "", "", "", "", "",
        )


# ---- Gradio UI ----
CSS = """
body, .gradio-container { background: #0a0a14 !important; }
.gradio-container { max-width: 1100px !important; margin: 0 auto !important; }
h1, h2, h3 { color: #d0d0e0 !important; }
label { color: #8899cc !important; font-size: 12px !important; }
.gr-button-primary { background: #225588 !important; border: 1px solid #3377aa !important; }
.gr-button-primary:hover { background: #3377aa !important; }
.gr-textbox textarea, .gr-textbox input { background: #0f0f22 !important; color: #d0d0e0 !important; border: 1px solid #1a1a3a !important; }
.gr-slider { color: #88ddff !important; }
.gr-accordion { background: #0f0f22 !important; border: 1px solid #1a1a3a !important; }
footer { display: none !important; }
"""

with gr.Blocks(css=CSS, title="Synthetic Data Decay Monitor", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # Synthetic Data Decay Monitor
    ### Constraint-Layer Health Diagnostic for AI Training Pipelines

    Detects **which executor type** is failing in your synthetic data pipeline,
    **how fast** each capability dimension is decaying, and **what data** you need to fix it —
    before model collapse happens. No GPU required.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Input")
            text_input = gr.Textbox(
                label="Paste JSONL data or plain text (one per line)",
                placeholder="One text sample per line, or paste JSONL...\nLeave empty to use demo data.",
                lines=6,
                value="",
            )
            with gr.Row():
                use_demo = gr.Checkbox(label="Use demo data", value=True)
                n_generations = gr.Slider(2, 8, value=4, step=1, label="Generations")
                decay_beta = gr.Slider(0.05, 0.35, value=0.15, step=0.01, label="Decay β")
            run_btn = gr.Button("Run Analysis", variant="primary", size="lg")

        with gr.Column(scale=2):
            gr.Markdown("### Overview")
            output_source = gr.Markdown("Ready. Click **Run Analysis** to begin.")
            summary_html = gr.HTML()
            warnings_html = gr.HTML()

    gr.Markdown("---")

    with gr.Tabs():
        with gr.TabItem("Stability Trajectories"):
            gr.Markdown("S_{n+1} = S_n · (1 − β). Each capability decays at its own rate because β depends on executor composition.")
            trajectory_plot = gr.HTML()

        with gr.TabItem("Executor Composition"):
            gr.Markdown("Per-capability breakdown of **E-I** (axiom/logic), **E-II** (scale/style), and **E-III** (boundary/fact) executor types.")
            composition_plot = gr.HTML()

        with gr.TabItem("Diagnosis Table"):
            gr.Markdown("Full per-capability diagnosis with degradation type, severity, and intervention recommendation.")
            diag_table = gr.HTML()

        with gr.TabItem("Collapse Order"):
            gr.Markdown("Capabilities ranked by predicted collapse generation. Earlier collapse → more urgent intervention.")
            collapse_order = gr.HTML()

        with gr.TabItem("Interventions"):
            gr.Markdown("Recommended data interventions to fix degrading executors.")
            intervention_cards = gr.HTML()

    run_btn.click(
        fn=run_analysis,
        inputs=[text_input, n_generations, decay_beta, use_demo],
        outputs=[output_source, summary_html, warnings_html,
                 trajectory_plot, composition_plot,
                 diag_table, collapse_order, intervention_cards],
    )

    gr.Markdown("---")
    gr.Markdown("""
    ### How It Works

    The monitor extracts **8 rule-based text features** (logic connector density, bigram repetition, proper case ratio, etc.)
    to fingerprint three executor degradation types:

    | Type | Detection Signal | Decay Rate (α) | Intervention |
    |------|-----------------|----------------|--------------|
    | **E-I** (Axiom) | Logic connector density drop | 0.40 | Add axiom/deduction data |
    | **E-II** (Scale) | Filler ratio rise + uniqueness drop | 0.20 | Add calibration/comparison data |
    | **E-III** (Boundary) | Proper case ratio drop + number jitter | 0.08 | Add boundary/edge-case data |

    **Validation:** 75% accuracy on executor classification, composition correlations: E-I=0.93, E-II=0.98, E-III=0.92.
    """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
