"""
报告生成器 — JSON 结构化报告 + HTML 交互式可视化。

JSON 报告可直接接入 W&B/MLflow。
HTML 复用 demo.py 的 Chart.js 热力图 + 级联路径图。
"""

import json
import os
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional


def _normalize_diagnoses(diagnoses: list) -> list[dict]:
    """Normalize diagnoses to plain dicts, converting DegradationDiagnosis objects."""
    result = []
    for d in diagnoses:
        if isinstance(d, dict):
            diag = d.get("diagnosis", d)
            if hasattr(diag, '__dataclass_fields__'):
                diag = {
                    "degradation_type": getattr(diag, "degradation_type", "unknown"),
                    "severity": getattr(diag, "severity", 0.0),
                    "intervention_type": getattr(diag, "intervention_type", "unknown"),
                    "recommended_data_sources": getattr(diag, "recommended_data_sources", []),
                    "confidence": getattr(diag, "confidence", 0.0),
                    "description": getattr(diag, "description", ""),
                    "helmholtz_scalar_potential": getattr(diag, "helmholtz_scalar_potential", 0.0),
                    "helmholtz_vector_potential": getattr(diag, "helmholtz_vector_potential", 0.0),
                }
            result.append({**d, "diagnosis": diag})
        else:
            result.append({"capability": "unknown", "diagnosis": {"degradation_type": "unknown"}})
    return result


def generate_json_report(
    lineage,
    trajectories: list[dict],
    diagnoses: list,
    cascade: Optional[dict] = None,
    meta: Optional[dict] = None,
) -> dict:
    """生成完整 JSON 诊断报告。"""
    diagnoses = _normalize_diagnoses(diagnoses)
    return {
        "report_metadata": {
            "generated_at": datetime.now().isoformat(),
            "framework_version": "0.1.0",
            **({} if meta is None else meta),
        },
        "data_summary": {
            "n_samples": lineage.n_samples,
            "n_generations": lineage.n_generations,
            "capabilities": list(lineage.capability_coverage.keys()),
            "generation_summary": lineage.generation_summary(),
        },
        "decay_analysis": {
            "trajectories": trajectories,
            "global_beta": _compute_global_beta(trajectories),
            "collapse_order": sorted(
                [t for t in trajectories if "error" not in t],
                key=lambda t: t["predicted_collapse_gen"],
            ),
        },
        "executor_diagnoses": diagnoses,
        "cascade_analysis": cascade or {},
        "interventions": _aggregate_interventions(diagnoses),
        "warnings": _generate_warnings(trajectories, diagnoses),
    }


def build_html_report(json_report: dict) -> str:
    """Build interactive HTML report content (returns HTML string, does not write to disk).

    Use this from server endpoints where you need the HTML in-memory.
    """
    traj_list = json_report["decay_analysis"]["trajectories"]
    diagnoses = json_report["executor_diagnoses"]
    cascade = json_report["cascade_analysis"]
    warnings = json_report["warnings"]
    data_summary = json_report["data_summary"]
    interventions = json_report["interventions"]

    # 准备 Chart.js 数据
    capabilities = [
        t["capability"] for t in traj_list
        if "error" not in t and t.get("trajectory")
    ]

    # 每个能力维度的 S_n 代际曲线
    gen_labels = sorted(set(
        g["generation"]
        for t in traj_list if "trajectory" in t
        for g in t["trajectory"]
    )) or [0]

    stability_datasets = []
    colors = ["#44dd88", "#88ccff", "#ffaa44", "#ff6688", "#cc66ff",
              "#44ddcc", "#ffdd44", "#66aacc", "#ff8866", "#cc88ff"]
    for i, cap in enumerate(capabilities):
        t = next((t for t in traj_list if t["capability"] == cap), None)
        if not t or "trajectory" not in t:
            continue
        s_vals = [g["S_n"] for g in t["trajectory"]]
        color = colors[i % len(colors)]
        status = t.get("current_status", "unknown")
        stability_datasets.append({
            "label": cap,
            "data": s_vals,
            "borderColor": color,
            "backgroundColor": color + "33",
            "tension": 0.3,
            "fill": False,
        })

    # 诊断总结
    diag_rows = ""
    for d in diagnoses[:10]:
        dtype = d.get("diagnosis", {}).get("degradation_type", "none")
        sev = d.get("diagnosis", {}).get("severity", 0)
        sev_color = "#ff4444" if sev > 0.7 else "#ffaa44" if sev > 0.4 else "#44dd44"
        diag_rows += f"""<tr>
        <td>{d.get('capability', '?')}</td>
        <td>{dtype.replace('_', ' ')}</td>
        <td style="color:{sev_color}">{sev:.0%}</td>
        <td style="font-size:10px">{d.get('diagnosis', {}).get('intervention_type', '')}</td>
        </tr>"""

    # 警告行
    warning_html = ""
    for w in warnings:
        color = "#ff6644" if w["severity"] == "critical" else "#ffaa44"
        warning_html += f"""<div style="padding:4px 8px;margin:2px 0;background:#221111;border-left:3px solid {color};font-size:11px">
        <b style="color:{color}">{w['type']}</b>: {w['message']}
        </div>"""

    # 级联路径
    cascade_html = ""
    if cascade.get("cascade_events"):
        events = cascade["cascade_events"]
        path = " → ".join(
            f"<span style='color:#ff6644'>{e['broken']}</span>"
            for e in events
        )
        cascade_html = f"""
        <h3>Cascade Path</h3>
        <div style="padding:12px;background:#0f0f22;border-radius:6px;font-size:14px">
            {path}
        </div>
        <div style="margin-top:6px;font-size:11px;color:#889">
            Total steps: {len(events)} |
            Weakest edge: {cascade.get('weakest_edge', {}).get('edge', 'N/A')}
        </div>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Synthetic Data Decay Monitor — Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ background:#0a0a14; color:#d0d0e0; font-family:'SF Mono','Menlo',monospace; padding:24px 36px; }}
h1 {{ font-size:20px; color:#fff; }}
h2 {{ font-size:14px; color:#8899cc; margin:24px 0 10px; border-bottom:1px solid #1a1a3a; padding-bottom:4px; }}
h3 {{ font-size:12px; color:#aabbdd; margin:12px 0 6px; }}
.row {{ display:flex; gap:16px; flex-wrap:wrap; }}
.card {{ background:#0f0f22; border:1px solid #1a1a3a; border-radius:8px; padding:14px; flex:1; min-width:320px; }}
.metric {{ display:inline-block; background:#111133; border-radius:6px; padding:12px 18px; margin:4px; text-align:center; }}
.metric .val {{ font-size:22px; font-weight:bold; color:#88ddff; }}
.metric .lbl {{ font-size:10px; color:#667; }}
table {{ width:100%; border-collapse:collapse; font-size:11px; margin:8px 0; }}
th {{ background:#111133; color:#889; padding:6px 10px; text-align:left; }}
td {{ padding:6px 10px; border-bottom:1px solid #111122; }}
.sub {{ color:#667; font-size:11px; margin-bottom:20px; }}
.warn {{ color:#ff6644; }}
footer {{ color:#444; font-size:10px; text-align:center; margin-top:40px; padding:20px; }}
</style>
</head>
<body>

<h1>Synthetic Data Decay Monitor</h1>
<div class="sub">Constraint Residual Framework · L0-L1 Health Diagnostic · {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>

<!-- KEY METRICS -->
<h2>Summary</h2>
<div class="row">
<div class="metric"><div class="val">{data_summary['n_samples']}</div><div class="lbl">Total Samples</div></div>
<div class="metric"><div class="val">{data_summary['n_generations']}</div><div class="lbl">Generations</div></div>
<div class="metric"><div class="val">{len(capabilities)}</div><div class="lbl">Capabilities</div></div>
<div class="metric"><div class="val">{json_report['decay_analysis']['global_beta']:.3f}</div><div class="lbl">Global β</div></div>
<div class="metric"><div class="val warn">{len([w for w in warnings if w['severity']=='critical'])}</div><div class="lbl">Critical Warnings</div></div>
</div>

<!-- WARNINGS -->
<h2>Warnings</h2>
{warning_html if warning_html else '<div style="color:#667;font-size:11px">No warnings.</div>'}

<!-- STABILITY TRAJECTORIES -->
<h2>Stability Trajectories S_n per Generation</h2>
<div class="row">
<div class="card" style="flex:2">
<div class="chart-wrap" style="position:relative;height:300px;">
<canvas id="stabilityChart"></canvas>
</div>
</div>
<div class="card">
<h3>Interpretation</h3>
<div style="font-size:11px;color:#889;line-height:1.7">
<p>S<sub>n+1</sub> = S<sub>n</sub> · (1 - β)</p>
<p style="color:#44dd88">S &gt; 0.8 → healthy</p>
<p style="color:#ffaa44">S &gt; 0.5 → degrading</p>
<p style="color:#ff6644">S &lt; {0.30} → collapsed</p>
<p style="margin-top:8px;color:#667">Different capabilities decay at different rates because β depends on executor type composition.</p>
</div>
</div>
</div>

<!-- DIAGNOSES -->
<h2>Executor Degradation Diagnoses</h2>
<table>
<tr><th>Capability</th><th>Degradation Type</th><th>Severity</th><th>Intervention</th></tr>
{diag_rows}
</table>

<!-- CASCADE -->
<h2>Cascade Prediction</h2>
{cascade_html if cascade_html else '<div style="color:#667;font-size:11px">No cascade analysis available.</div>'}

<!-- INTERVENTIONS -->
<h2>Recommended Interventions</h2>
<div class="row">
{_intervention_cards(interventions)}
</div>

<footer>
Synthetic Data Decay Monitor · Constraint Residual Framework v0.1.0<br>
"Not monitoring performance — diagnosing constraint structure."
</footer>

<script>
const collapseLinePlugin = {{
    id: 'collapseLine',
    afterDraw(chart) {{
        var y = chart.scales.y.getPixelForValue({S_CRITICAL});
        var ctx = chart.ctx;
        ctx.save();
        ctx.beginPath();
        ctx.setLineDash([6, 4]);
        ctx.strokeStyle = '#ff4444';
        ctx.lineWidth = 2;
        ctx.moveTo(chart.scales.x.left, y);
        ctx.lineTo(chart.scales.x.right, y);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = '#ff4444';
        ctx.font = '10px monospace';
        ctx.fillText('Collapse S={S_CRITICAL}', chart.scales.x.right - 130, y - 6);
        ctx.restore();
    }}
}};

const ctx = document.getElementById('stabilityChart').getContext('2d');
new Chart(ctx, {{
    type: 'line',
    data: {{
        labels: {json.dumps(gen_labels)},
        datasets: {json.dumps(stability_datasets)},
    }},
    options: {{
        responsive: true,
        maintainAspectRatio: false,
        scales: {{
            x: {{ title: {{ display: true, text: 'Generation', color: '#889' }}, ticks: {{ color: '#889' }}, grid: {{ color: '#1a1a3a' }} }},
            y: {{ title: {{ display: true, text: 'Stability S_n', color: '#889' }}, min: 0, max: 1, ticks: {{ color: '#889' }}, grid: {{ color: '#1a1a3a' }} }},
        }},
        plugins: {{
            legend: {{ labels: {{ color: '#889', font: {{ size: 10 }} }} }},
        }}
    }},
    plugins: [collapseLinePlugin],
}});
</script>

</body></html>"""

    return html


def generate_html_report(json_report: dict, output_path: str) -> str:
    """Generate and write HTML report to disk. Returns output_path."""
    html = build_html_report(json_report)
    with open(output_path, "w") as f:
        f.write(html)
    return output_path


S_CRITICAL = 0.30


def _compute_global_beta(trajectories: list[dict]) -> float:
    betas = []
    for t in trajectories:
        if "trajectory" not in t:
            continue
        for g in t["trajectory"]:
            if "beta" in g and g["beta"] > 0:
                betas.append(g["beta"])
    return float(np.mean(betas)) if betas else 0.25


def _aggregate_interventions(diagnoses: list[dict]) -> list[dict]:
    """聚合所有诊断的干预建议，去重且排序。"""
    by_type = {}
    for d in diagnoses:
        diag = d.get("diagnosis", {})
        itype = diag.get("intervention_type", "unknown")
        sources = diag.get("recommended_data_sources", [])
        if itype not in by_type:
            by_type[itype] = {
                "intervention_type": itype,
                "affected_capabilities": [],
                "recommended_data_sources": [],
            }
        by_type[itype]["affected_capabilities"].append(d["capability"])
        for src in sources:
            if src not in by_type[itype]["recommended_data_sources"]:
                by_type[itype]["recommended_data_sources"].append(src)
    return list(by_type.values())


def _generate_warnings(trajectories: list[dict], diagnoses: list[dict]) -> list[dict]:
    ws = []

    for t in trajectories:
        if "trajectory" not in t:
            continue
        traj = t["trajectory"]
        if not traj:
            continue
        last = traj[-1]
        if last["status"] == "collapsed":
            ws.append({
                "type": "capability_collapsed",
                "severity": "critical",
                "capability": t["capability"],
                "message": f"{t['capability']} has collapsed at generation {last['generation']}.",
            })
        elif last["status"] == "critical":
            ws.append({
                "type": "capability_critical",
                "severity": "warning",
                "capability": t["capability"],
                "message": (
                    f"{t['capability']} approaching collapse "
                    f"(S={last['S_n']:.3f} at gen {last['generation']})."
                ),
            })

    for d in diagnoses:
        if d.get("diagnosis", {}).get("degradation_type") == "E-I_loss":
            ws.append({
                "type": "EI_degradation",
                "severity": "critical",
                "capability": d["capability"],
                "message": f"{d['capability']}: E-I executor loss—fastest collapse pattern.",
            })

    for t in trajectories:
        if "trajectory" not in t:
            continue
        for g in t["trajectory"]:
            if g.get("cancellation_ratio", 1.0) < 0.1 and g.get("executor_composition"):
                total_mag = sum(g.get("executor_composition", {}).values())
                if total_mag > 0.5:
                    ws.append({
                        "type": "dark_zone_detected",
                        "severity": "critical",
                        "capability": t["capability"],
                        "message": (
                            f"Type III dark zone at gen {g['generation']} in {t['capability']}: "
                            f"c={g['cancellation_ratio']:.3f}—constraint structure being hollowed out unseen."
                        ),
                    })
                    break

    return ws


def _intervention_cards(interventions: list[dict]) -> str:
    icons = {
        "add_axiom_data": "🧮",
        "add_calibration_data": "⚖️",
        "add_boundary_data": "🔬",
        "add_mixed_data": "🔧",
        "unknown": "❓",
    }
    cards = ""
    for inter in interventions:
        itype = inter["intervention_type"]
        icon = icons.get(itype, "📋")
        caps = ", ".join(inter["affected_capabilities"][:4])
        sources = "<br>".join(
            f"• {s}" for s in inter["recommended_data_sources"][:3]
        )
        cards += f"""<div class="card">
        <h3>{icon} {itype.replace('_', ' ').title()}</h3>
        <div style="font-size:10px;color:#889">Affected: {caps}</div>
        <div style="font-size:10px;margin-top:6px;line-height:1.5">{sources}</div>
        </div>"""
    return cards if cards else '<div style="color:#667;font-size:11px">No interventions needed.</div>'


def generate_paper_figures(
    engine,
    json_report: dict,
    output_dir: str = "paper_figures",
    fmt: str = "png",
) -> list[str]:
    """Generate publication-quality figures using matplotlib.

    Produces:
      - fig1_stability_trajectories.{fmt}
      - fig2_executor_composition.{fmt}
      - fig3_text_fingerprints.{fmt}
      - fig4_collapse_timeline.{fmt}

    Returns list of output file paths.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    os.makedirs(output_dir, exist_ok=True)
    outputs = []

    # Style
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })

    COLORS = {
        "E-I": "#e74c3c", "E-II": "#3498db", "E-III": "#2ecc71",
        "math_reasoning": "#44dd88", "code_generation": "#88ccff",
        "factual_knowledge": "#ffaa44", "logical_consistency": "#ff6688",
        "general": "#cc66ff",
    }

    traj_list = json_report["decay_analysis"]["trajectories"]
    caps = [t["capability"] for t in traj_list if "error" not in t and t.get("trajectory")]

    # ================================================================
    # Figure 1: Stability trajectories S_n(g)
    # ================================================================
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    for cap in caps:
        t = next((x for x in traj_list if x["capability"] == cap), None)
        if not t:
            continue
        gens = [g["generation"] for g in t["trajectory"]]
        svals = [g["S_n"] for g in t["trajectory"]]
        color = COLORS.get(cap, "#888888")
        ax1.plot(gens, svals, marker="o", label=cap.replace("_", " "), color=color, linewidth=1.5, markersize=4)

    ax1.axhline(y=0.3, color="#e74c3c", linestyle="--", linewidth=1, alpha=0.7, label="Collapse (S=0.3)")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Stability S_n")
    ax1.set_title("Constraint Stability Trajectories")
    ax1.legend(fontsize=7, loc="upper right")
    ax1.set_ylim(0, 1.05)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.grid(True, alpha=0.2)
    fig1.tight_layout()

    for ext in (fmt, "svg"):
        f1 = os.path.join(output_dir, f"fig1_stability_trajectories.{ext}")
        fig1.savefig(f1)
        outputs.append(f1)
    plt.close(fig1)

    # ================================================================
    # Figure 2: Executor composition stacked area per capability
    # ================================================================
    n_caps = len(caps)
    fig2, axes2 = plt.subplots(1, min(n_caps, 3), figsize=(4 * min(n_caps, 3), 3.5))
    if n_caps == 1:
        axes2 = [axes2]

    for idx, cap in enumerate(caps[:3]):
        ax = axes2[idx]
        t = next((x for x in traj_list if x["capability"] == cap), None)
        if not t:
            continue
        gens = [g["generation"] for g in t["trajectory"]]
        ei_vals = [g.get("executor_composition", {}).get("E-I", 0) for g in t["trajectory"]]
        eii_vals = [g.get("executor_composition", {}).get("E-II", 0) for g in t["trajectory"]]
        eiii_vals = [g.get("executor_composition", {}).get("E-III", 0) for g in t["trajectory"]]

        ax.stackplot(gens, ei_vals, eii_vals, eiii_vals,
                     labels=["E-I (Axiom)", "E-II (Scale)", "E-III (Boundary)"],
                     colors=[COLORS["E-I"], COLORS["E-II"], COLORS["E-III"]],
                     alpha=0.8)
        ax.set_title(cap.replace("_", " "), fontsize=9)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Composition")
        ax.set_ylim(0, 1)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, alpha=0.2)

    if n_caps > 1:
        axes2[0].legend(fontsize=7, loc="upper left")
    fig2.suptitle("Executor Composition Evolution (Layer Traversal)", fontsize=12)
    fig2.tight_layout()

    for ext in (fmt, "svg"):
        f2 = os.path.join(output_dir, f"fig2_executor_composition.{ext}")
        fig2.savefig(f2)
        outputs.append(f2)
    plt.close(fig2)

    # ================================================================
    # Figure 3: Text feature fingerprints (gen0 vs genN delta)
    # ================================================================
    feature_names = [
        ("ei_logic_density", "Logic\nDensity"),
        ("ei_syntax_cv", "Syntax\nCV"),
        ("eii_bigram_repetition", "Bigram\nRepeat"),
        ("eii_filler_ratio", "Filler\nRatio"),
        ("eii_unique_word_ratio", "Unique\nWords"),
        ("eii_truncation_ratio", "Truncation"),
        ("eiii_proper_case_ratio", "Proper\nCase"),
        ("eiii_number_integrity", "Number\nInteg."),
    ]

    fig3, axes3 = plt.subplots(1, 3, figsize=(10, 4))
    exec_labels = ["E-I (Axiom)", "E-II (Scale)", "E-III (Boundary)"]

    for exec_idx, (exec_key, ax) in enumerate(zip(["E-I", "E-II", "E-III"], axes3)):
        # Find a capability that has this executor type as dominant
        best_cap = None
        best_snapshots = None
        for cap in caps:
            snaps = engine._snapshots.get(cap, [])
            if len(snaps) >= 2:
                best_cap = cap
                best_snapshots = snaps
                # Prefer capability matching executor type
                if exec_key == "E-I" and cap in ("math_reasoning", "logical_consistency"):
                    break
                elif exec_key == "E-III" and cap in ("factual_knowledge",):
                    break

        if best_snapshots and len(best_snapshots) >= 2:
            tf0 = best_snapshots[0].text_features
            tfN = best_snapshots[-1].text_features
        else:
            ax.set_title(f"{exec_labels[exec_idx]}\n(no data)")
            ax.axis("off")
            continue

        x_labels = [fn[1] for fn in feature_names]
        vals0 = [tf0.get(fn[0], 0.5) for fn in feature_names]
        valsN = [tfN.get(fn[0], 0.5) for fn in feature_names]

        x = range(len(feature_names))
        width = 0.35
        ax.bar([i - width/2 for i in x], vals0, width, label="Gen 0", color="#888899", alpha=0.7)
        ax.bar([i + width/2 for i in x], valsN, width, label="Gen N", color=COLORS[exec_key], alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=7, rotation=45, ha="right")
        ax.set_ylim(0, 1.1)
        ax.set_title(exec_labels[exec_idx], fontsize=10)
        ax.grid(True, alpha=0.2, axis="y")
        if exec_idx == 0:
            ax.legend(fontsize=7)

    fig3.suptitle("Text Feature Fingerprints by Executor Type", fontsize=12)
    fig3.tight_layout()

    for ext in (fmt, "svg"):
        f3 = os.path.join(output_dir, f"fig3_text_fingerprints.{ext}")
        fig3.savefig(f3)
        outputs.append(f3)
    plt.close(fig3)

    # ================================================================
    # Figure 4: Collapse prediction timeline
    # ================================================================
    collapse_order = json_report["decay_analysis"]["collapse_order"]
    fig4, ax4 = plt.subplots(figsize=(7, 3))

    cap_names = [c["capability"].replace("_", " ") for c in collapse_order]
    collapse_gens = [c["predicted_collapse_gen"] for c in collapse_order]
    betas = [c.get("beta", 0) for c in collapse_order]

    colors_bar = [COLORS.get(c["capability"], "#888") for c in collapse_order]
    bars = ax4.barh(cap_names, collapse_gens, color=colors_bar, alpha=0.8, height=0.5)

    for bar, gen, beta in zip(bars, collapse_gens, betas):
        ax4.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                 f"gen {gen}  (β={beta:.3f})", va="center", fontsize=8)

    ax4.set_xlabel("Predicted Collapse Generation")
    ax4.set_title("Capability Collapse Timeline")
    ax4.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax4.grid(True, alpha=0.2, axis="x")
    fig4.tight_layout()

    for ext in (fmt, "svg"):
        f4 = os.path.join(output_dir, f"fig4_collapse_timeline.{ext}")
        fig4.savefig(f4)
        outputs.append(f4)
    plt.close(fig4)

    return outputs
