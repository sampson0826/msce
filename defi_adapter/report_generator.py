"""
HTML report generator for UnifiedScanReport.

Produces a self-contained HTML page with:
  - Dark zone summary across all three types
  - Constraint field visualization (c(p) heatmap)
  - Per-type detailed findings
  - Mitigation recommendations

Usage:
    report = scanner.scan(bounds)
    html = generate_html(report, field, mapper)
    with open("report.html", "w") as f:
        f.write(html)
"""

import json
import numpy as np
from typing import Optional

from constraint_residual.core import ConstraintField
from constraint_residual.defi_adapter.unified_scanner import UnifiedScanReport
from constraint_residual.defi_adapter.state_mapper import StateMapper


def generate_html(report: UnifiedScanReport,
                  field: ConstraintField,
                  mapper: Optional[StateMapper] = None,
                  highlight_points: Optional[dict[str, np.ndarray]] = None,
                  ) -> str:
    """Generate a self-contained HTML report.

    Args:
        report: UnifiedScanReport from scanner
        field: ConstraintField used for the scan
        mapper: Optional StateMapper for human-readable dimension labels
        highlight_points: Optional {"label": point} to mark on visualizations
    """

    # Dimension labels
    if mapper:
        dim_labels = [d.name for d in mapper.dimensions]
    else:
        dim_labels = [f"dim_{i}" for i in range(report.n_state_dims)]

    # Compute 2D data for visualization (first two dimensions)
    xs = np.linspace(report.bounds[0][0], report.bounds[0][1], 50)
    ys = np.linspace(report.bounds[1][0], report.bounds[1][1], 50)

    cancellation_map = np.zeros((50, 50))
    constraint_map = np.zeros((50, 50))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            p = np.array([x, y])
            grad = field.constraint_gradient(p)
            constraint_map[j, i] = float(np.linalg.norm(grad))

            indiv_mags = [float(np.linalg.norm(r.gradient(p))) for r in field.rules]
            total_indiv = sum(indiv_mags)
            if total_indiv > 1e-10:
                cancellation_map[j, i] = constraint_map[j, i] / total_indiv
            else:
                cancellation_map[j, i] = 1.0

    # Risk assessment
    risk_color = {"HIGH": "#ff4444", "MEDIUM": "#ffaa44", "LOW": "#44dd66"}
    risk = report.overall_risk()
    risk_level = risk.split("—")[0].strip() if "—" in risk else "MEDIUM"
    rc = risk_color.get(risk_level, "#ffaa44")

    # Build type-specific sections
    type3_html = _build_type3_section(report)
    type2_html = _build_type2_section(report, dim_labels)
    type1_html = _build_type1_section(report)

    # Highlight markers
    highlight_js = ""
    if highlight_points:
        markers = []
        for label, pt in highlight_points.items():
            if len(pt) >= 2:
                markers.append({
                    'x': float((pt[0] - report.bounds[0][0]) / (report.bounds[0][1] - report.bounds[0][0])),
                    'y': float((pt[1] - report.bounds[1][0]) / (report.bounds[1][1] - report.bounds[1][0])),
                    'label': label,
                })
        highlight_js = f"const highlightPoints = {json.dumps(markers)};"
    else:
        highlight_js = "const highlightPoints = [];"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Dark Zone Scan — {report.protocol_name}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ background:#080816; color:#c8c8dd; font-family:'SF Mono','Menlo','Courier New',monospace; padding:24px 40px; }}
h1 {{ font-size:24px; color:#fff; margin-bottom:4px; }}
h2 {{ font-size:16px; color:#8899cc; margin:32px 0 12px; border-bottom:1px solid #1a1a3a; padding-bottom:6px; }}
h3 {{ font-size:13px; color:#aabbdd; margin:16px 0 8px; }}
.subtitle {{ color:#667; font-size:12px; margin-bottom:28px; }}
.row {{ display:flex; gap:20px; flex-wrap:wrap; }}
.card {{ background:#0c0c20; border:1px solid #1a1a40; border-radius:8px; padding:18px; flex:1; min-width:320px; }}
.card h4 {{ font-size:12px; color:#99aacc; margin-bottom:10px; text-transform:uppercase; letter-spacing:1px; }}
.chart-wrap {{ position:relative; height:280px; margin:8px 0; }}
.metric {{ display:inline-block; background:#0d0d28; border:1px solid #1a1a50; border-radius:6px; padding:12px 18px; margin:6px; text-align:center; }}
.metric .val {{ font-size:28px; font-weight:bold; }}
.metric .lbl {{ font-size:10px; color:#667; margin-top:2px; }}
.risk-badge {{ display:inline-block; padding:4px 12px; border-radius:4px; font-size:11px; font-weight:bold; }}
.finding {{ background:#0d0d20; border-left:3px solid; padding:10px 14px; margin:8px 0; border-radius:0 6px 6px 0; font-size:11px; }}
.finding.type3 {{ border-color:#cc44dd; }}
.finding.type2 {{ border-color:#4488ff; }}
.finding.type1 {{ border-color:#ff8844; }}
.finding .tag {{ font-size:9px; padding:2px 6px; border-radius:3px; margin-right:8px; }}
.tag.t3 {{ background:#331144; color:#cc44dd; }}
.tag.t2 {{ background:#112244; color:#4488ff; }}
.tag.t1 {{ background:#442211; color:#ff8844; }}
table {{ width:100%; border-collapse:collapse; font-size:11px; margin:8px 0; }}
th {{ background:#0f0f2a; color:#889; padding:6px 10px; text-align:left; font-weight:normal; }}
td {{ padding:6px 10px; border-bottom:1px solid #0f0f25; }}
.verdict {{ background:#0d0d28; border:1px solid {rc}; border-radius:8px; padding:20px; margin:24px 0; }}
.verdict h3 {{ color:{rc}; margin-top:0; }}
footer {{ color:#444; font-size:10px; text-align:center; margin-top:48px; padding:20px; border-top:1px solid #1a1a30; }}
.c-map {{ display:flex; gap:4px; align-items:center; font-size:10px; margin:8px 0; }}
.c-swatch {{ width:16px; height:16px; border-radius:2px; display:inline-block; }}
</style>
</head>
<body>

<h1>Dark Zone Scan Report</h1>
<div class="subtitle">
  Protocol: <strong>{report.protocol_name}</strong> &nbsp;|&nbsp;
  State space: {report.n_state_dims}D ({', '.join(dim_labels)}) &nbsp;|&nbsp;
  Constraints: {len(report.constraint_names)}
</div>

<!-- METRICS -->
<div class="row">
  <div class="metric">
    <div class="val" style="color:#cc44dd">{report.type3_count}</div>
    <div class="lbl">Type III — Cancellation</div>
  </div>
  <div class="metric">
    <div class="val" style="color:#4488ff">{report.type2_count}</div>
    <div class="lbl">Type II — Structural</div>
  </div>
  <div class="metric">
    <div class="val" style="color:#ff8844">{report.type1_count}</div>
    <div class="lbl">Type I — Occlusion</div>
  </div>
  <div class="metric">
    <div class="val" style="color:#88ddff">{report.residual_points}</div>
    <div class="lbl">Residual Points</div>
  </div>
</div>

<div class="verdict">
  <h3>Overall Risk: {risk_level}</h3>
  <p style="font-size:12px;color:#889;margin-top:6px">{risk}</p>
</div>

<!-- HEATMAPS -->
<h2>Constraint Topology Maps</h2>
<div class="row">
  <div class="card" style="flex:1.3">
    <h4>Cancellation Ratio c(p) — {dim_labels[1]} vs {dim_labels[0]}</h4>
    <div class="c-map">
      <span>c→0 (DARK)</span>
      <span class="c-swatch" style="background:#1a1aff"></span>
      <span class="c-swatch" style="background:#4444aa"></span>
      <span class="c-swatch" style="background:#666688"></span>
      <span class="c-swatch" style="background:#ddcc44"></span>
      <span class="c-swatch" style="background:#ffaa22"></span>
      <span>c→1 (CLEAR)</span>
    </div>
    <div class="chart-wrap"><canvas id="cancelHeatmap"></canvas></div>
  </div>
  <div class="card">
    <h4>Interpretation Guide</h4>
    <div style="font-size:11px;color:#889;line-height:1.8">
      <p style="color:#1a1aff">■ Blue zones</p>
      <p style="margin-left:16px">c(p) ≈ 0, Σ||∇σ|| large<br>
      → <b style="color:#cc44dd">TYPE III DARK ZONE</b><br>
      → Multiple constraints cancel perfectly.<br>
      → Each rule individually active, together blind.</p>

      <p style="color:#ffaa22;margin-top:12px">■ Yellow zones</p>
      <p style="margin-left:16px">c(p) ≈ 1<br>
      → Single constraint dominates.<br>
      → Behavior is predictable.</p>

      <p style="color:#666688;margin-top:12px">■ Gray zones</p>
      <p style="margin-left:16px">Intermediate c(p)<br>
      → Partial constraint interaction.<br>
      → Monitor for emerging dark zones.</p>
    </div>
  </div>
</div>

<!-- PER-TYPE FINDINGS -->
<h2>Detailed Findings by Type</h2>

{type3_html}
{type2_html}
{type1_html}

<!-- CONSTRAINT LIST -->
<h2>Constraints Analyzed</h2>
<div class="card">
  <table>
    <tr><th>#</th><th>Constraint</th><th>Domain</th><th>Certainty</th></tr>
"""
    for i, rule in enumerate(field.rules):
        html += f"""    <tr><td>{i+1}</td><td>{rule.name}</td><td>{rule.domain}</td><td>{rule.certainty:.0%}</td></tr>\n"""

    html += f"""  </table>
</div>

<footer>
  Constraint Residual Framework — Dark Zone Scanner<br>
  "The more stable a rule, the more invisible it is." — Stability-Visibility Inverse Law<br>
  Analysis generated by UnifiedScanner v1.0
</footer>

<script>
{highlight_js}

const cancelData = {json.dumps(cancellation_map.tolist())};

function drawHeatmap(canvasId, data, colorMin, colorMax) {{
    const ctx = document.getElementById(canvasId).getContext('2d');
    const h = data.length;
    const w = data[0].length;
    const imageData = ctx.createImageData(w, h);

    for (let y = 0; y < h; y++) {{
        for (let x = 0; x < w; x++) {{
            const idx = (y * w + x) * 4;
            let v = data[y][x];

            // Clamp and normalize
            v = Math.max(colorMin, Math.min(colorMax, v));
            const t = (v - colorMin) / (colorMax - colorMin);

            // Color: blue (dark zone) → yellow (clear) → orange (dominated)
            let r, g, b;
            if (t < 0.33) {{
                // Blue to purple
                const s = t / 0.33;
                r = Math.floor(20 + s * 100);
                g = Math.floor(20 + s * 20);
                b = Math.floor(255 - s * 100);
            }} else if (t < 0.66) {{
                // Purple to yellow
                const s = (t - 0.33) / 0.33;
                r = Math.floor(120 + s * 135);
                g = Math.floor(40 + s * 200);
                b = Math.floor(155 - s * 130);
            }} else {{
                // Yellow to orange
                const s = (t - 0.66) / 0.34;
                r = Math.floor(255);
                g = Math.floor(240 - s * 150);
                b = Math.floor(25 - s * 20);
            }}

            imageData.data[idx] = r;
            imageData.data[idx + 1] = g;
            imageData.data[idx + 2] = b;
            imageData.data[idx + 3] = 255;
        }}
    }}

    // Draw to temp canvas and scale
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = w;
    tempCanvas.height = h;
    tempCanvas.getContext('2d').putImageData(imageData, 0, 0);

    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(tempCanvas, 0, 0, ctx.canvas.width, ctx.canvas.height);

    // Draw highlight points
    {f"highlightPoints.forEach(pt => {{" if highlight_points else "// no highlights"}
        ctx.beginPath();
        ctx.arc(pt.x * ctx.canvas.width, pt.y * ctx.canvas.height, 6, 0, Math.PI * 2);
        ctx.fillStyle = '#ffffff';
        ctx.fill();
        ctx.strokeStyle = '#000000';
        ctx.lineWidth = 2;
        ctx.stroke();

        ctx.fillStyle = '#ffffff';
        ctx.font = '11px monospace';
        ctx.fillText(pt.label, pt.x * ctx.canvas.width + 10, pt.y * ctx.canvas.height - 8);
    }};
}}
{f")" if highlight_points else ""}

drawHeatmap('cancelHeatmap', cancelData, 0.0, 1.0);
</script>

</body>
</html>"""
    return html


def _build_type3_section(report: UnifiedScanReport) -> str:
    if not report.type3_dark_zones:
        return """<div class="card">
  <h4>Type III — Cancellation Dark Zones</h4>
  <p style="color:#667;font-size:11px">No Type III dark zones detected.
  Constraints do not exhibit significant cross-cancellation.</p>
</div>"""

    html = '<div class="card"><h4>Type III — Cancellation Dark Zones</h4>'
    html += f'<p style="color:#889;font-size:11px;margin-bottom:12px">{len(report.type3_dark_zones)} clusters detected where multiple constraints cancel.</p>'

    for i, dz in enumerate(report.type3_dark_zones[:8]):
        risk = "HIGH" if dz.mean_cancellation_ratio < 0.05 else "MEDIUM"
        html += f"""<div class="finding type3">
  <span class="tag t3">T3-{i+1}</span>
  <strong>{dz.balance_topology}</strong> — {len(dz.points)} points, c̄={dz.mean_cancellation_ratio:.4f}
  <br><span style="color:#889">Constraints: {', '.join(dz.constraints_involved)}</span>
  <br><span style="color:#889">Risk: {risk}</span>
</div>"""

    html += '</div>'
    return html


def _build_type2_section(report: UnifiedScanReport, dim_labels: list[str]) -> str:
    if not report.type2_candidates:
        return """<div class="card">
  <h4>Type II — Structural Occlusion (Unconstrained Directions)</h4>
  <p style="color:#667;font-size:11px">No Type II dark zones detected.
  Constraint metric tensor is well-conditioned across the state space.</p>
</div>"""

    html = '<div class="card"><h4>Type II — Structural Occlusion (Unconstrained Directions)</h4>'
    html += f'<p style="color:#889;font-size:11px;margin-bottom:12px">{len(report.type2_candidates)} regions with highly anisotropic constraint coverage.</p>'

    for i, c in enumerate(report.type2_candidates[:8]):
        u_dir = c.unconstrained_direction
        dom = dim_labels[0] if abs(u_dir[0]) > abs(u_dir[-1]) else dim_labels[-1]
        cond_str = f"{c.condition_number:.0f}" if c.condition_number < 1e6 else f"{c.condition_number:.1e}"
        html += f"""<div class="finding type2">
  <span class="tag t2">T2-{i+1}</span>
  <strong>Unconstrained: {dom}</strong> — condition number: {cond_str}
  <br><span style="color:#889">Position: ({c.position[0]:.2f}, {c.position[1]:.4f})
  — λ_min={c.min_eigenvalue:.2e}, λ_max={c.max_eigenvalue:.2e}</span>
  <br><span style="color:#889">No constraint provides meaningful gradient in the {dom} direction at this point.</span>
</div>"""

    html += '</div>'
    return html


def _build_type1_section(report: UnifiedScanReport) -> str:
    if not report.type1_candidates:
        return """<div class="card">
  <h4>Type I — Physical Occlusion (Oracle Divergence)</h4>
  <p style="color:#667;font-size:11px">No Type I dark zones detected.
  Oracle/data source consistency is within threshold, or no data sources configured.</p>
</div>"""

    html = '<div class="card"><h4>Type I — Physical Occlusion (Oracle Divergence)</h4>'
    html += f'<p style="color:#889;font-size:11px;margin-bottom:12px">{len(report.type1_candidates)} points where data source divergence masks constraint signals.</p>'

    for i, c in enumerate(report.type1_candidates[:8]):
        html += f"""<div class="finding type1">
  <span class="tag t1">T1-{i+1}</span>
  <strong>Divergence: {c.max_divergence_pct:.1f}%</strong> — occlusion score: {c.occlusion_score:.4f}
  <br><span style="color:#889">Sources: {', '.join(c.diverging_sources[:3])}</span>
  <br><span style="color:#889">Affected: {', '.join(c.affected_constraints[:3])}</span>
  <br><span style="color:#889">{c.recommendation}</span>
</div>"""

    html += '</div>'
    return html
