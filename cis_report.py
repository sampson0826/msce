#!/usr/bin/env python3
"""
CIS Comprehensive Protocol Security Report

Runs the full A.1-A.7 Constraint Invisibility Scanner analysis on all
protocols and generates a comparative HTML report with real 2D heatmaps.

Run:
  cd /Users/dengxinhang/paper && python3 constraint_residual/cis_report.py
  open constraint_residual/cis_report.html
"""

import sys, os, json, time
sys.path.insert(0, '/Users/dengxinhang/paper')
import numpy as np
from constraint_residual.dsl.compiler import load_protocol
from constraint_residual.cis_core import CISAnalyzer
from constraint_residual.dark_zone_detector import DarkZoneDetector

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cis_report.html")
DSL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dsl", "protocols")

PROTOCOLS = [
    {
        'key': 'the_dao',
        'path': 'the_dao.yaml',
        'name': 'The DAO (2016)',
        'species': 'I — mutual_cancellation',
        'species_idx': 1,
        'exploit': '3.6M ETH (~$50M)',
    },
    {
        'key': 'parity_wallet',
        'path': 'parity_wallet.yaml',
        'name': 'Parity Wallet (2017)',
        'species': 'II — cold_start_gap',
        'species_idx': 2,
        'exploit': '$300M frozen',
    },
    {
        'key': 'poly_network',
        'path': 'poly_network.yaml',
        'name': 'Poly Network (2021)',
        'species': 'III — hierarchical',
        'species_idx': 3,
        'exploit': '$600M',
    },
    {
        'key': 'aave_v3',
        'path': 'aave_v3.yaml',
        'name': 'Aave v3 (vulnerable)',
        'species': 'IV — hostile_asymmetry',
        'species_idx': 4,
        'exploit': 'CVSS 9.8 (oracle staleness)',
    },
    {
        'key': 'euler_vulnerable',
        'path': 'euler_vulnerable.yaml',
        'name': 'Euler Finance (2023)',
        'species': 'II — cold_start_gap',
        'species_idx': 2,
        'exploit': '$200M (donateToReserves)',
    },
    {
        'key': 'euler_fixed',
        'path': 'euler_fixed.yaml',
        'name': 'Euler Finance (fixed)',
        'species': 'protected',
        'species_idx': 0,
        'exploit': 'FIXED — checkLiquidity added',
    },
]

N_GRID = 64

print("=== CIS Comprehensive Protocol Security Report ===\n")

results = {}
for proto in PROTOCOLS:
    path = os.path.join(DSL_DIR, proto['path'])
    field, spec = load_protocol(path)

    t0 = time.time()
    analyzer = CISAnalyzer(field, bounds=[(0, 1), (0, 1)], n_points=N_GRID)
    report = analyzer.full_analysis(proto['name'])
    elapsed = time.time() - t0

    detector = DarkZoneDetector(cancellation_eps=0.15, individual_min=0.3)
    dz = detector.scan(field, [(0, 1), (0, 1)], n_points=N_GRID)

    results[proto['key']] = {
        'info': proto,
        'cis': report,
        'dz_count': len(dz),
        'dz_centroids': [(float(d.centroid[0]), float(d.centroid[1])) for d in dz],
        'dz_c_ratios': [float(d.mean_cancellation_ratio) for d in dz],
        'dz_topologies': [d.balance_topology for d in dz],
        'elapsed': elapsed,
    }
    print(f"  {proto['name']:30s} DZ={len(dz)}  E-I={report.e1_fraction:.1%}  "
          f"unconstrained={report.unconstrained_fraction:.1%}  {elapsed:.1f}s")

# Build heatmap data arrays (row-major: y from 1→0 so heatmap renders bottom-up)
heatmaps = {}
for proto in PROTOCOLS:
    key = proto['key']
    r = results[key]['cis']
    # Flip rows so y=0 is at bottom of heatmap
    heatmaps[key] = {
        'cancel': r.cancellation_ratio[::-1].tolist(),
        'force': r.vector_field.magnitude[::-1].tolist(),
        'div': np.abs(r.helmholtz.divergence)[::-1].tolist(),
        'eval_min': r.riemannian.eval_min[::-1].tolist(),
    }

# ═══════════════════════════════════════════════════════════════
# HTML Report with Canvas Heatmaps
# ═══════════════════════════════════════════════════════════════

species_colors = {0: '#44dd66', 1: '#cc44dd', 2: '#ddcc44', 3: '#66aadd', 4: '#dd6644'}

# Summary table rows
summary_rows = ''
for proto in PROTOCOLS:
    key = proto['key']
    r = results[key]
    c = r['cis']
    sc = species_colors.get(proto['species_idx'], '#889')
    dz_color = '#dd6644' if r['dz_count'] > 0 else '#44dd66'
    summary_rows += f"""<tr>
<td><b>{proto['name']}</b></td>
<td style="color:{sc}">{proto['species']}</td>
<td style="color:{dz_color}">{r['dz_count']}</td>
<td>{c.e1_fraction:.1%}</td>
<td>{c.e2_fraction:.1%}</td>
<td>{c.e3_fraction:.1%}</td>
<td>{c.unconstrained_fraction:.1%}</td>
<td>{c.structural_score:.3f}</td>
<td>{r['elapsed']:.1f}s</td></tr>"""

# Per-protocol cards
cards_html = ''
for proto in PROTOCOLS:
    key = proto['key']
    r = results[key]
    c = r['cis']
    sc = species_colors.get(proto['species_idx'], '#889')

    # Dark zone details
    dz_html = ""
    if r['dz_count'] > 0:
        for i, (cz, cr_val, topo) in enumerate(zip(r['dz_centroids'], r['dz_c_ratios'], r['dz_topologies'])):
            dz_html += f"""<div style="background:#1a1122;border-left:3px solid #cc44dd;padding:6px 8px;margin:3px 0;border-radius:0 3px 3px 0;font-size:10px">
<b style="color:#cc44dd">DZ [{i+1}]</b> c(p)={cr_val:.4f} · ({cz[0]:.3f},{cz[1]:.3f}) · {topo}</div>"""
    else:
        dz_html = '<p style="font-size:10px;color:#44dd66">No dark zones detected</p>'

    # Top executor candidates
    exec_html = ""
    for i, cand in enumerate(c.top_missing_executor_locations[:3]):
        type_badge = {'E-I (structural)': 'badge-ei', 'E-II (scalar/scale)': 'badge-eii', 'E-III (boundary)': 'badge-eiii'}
        badge = type_badge.get(cand['type'], '')
        exec_html += f"""<div style="font-size:10px;color:#889;margin:2px 0">
[{i+1}] ({cand['position'][0]:.3f},{cand['position'][1]:.3f}) <span class="badge {badge}">{cand['type']}</span> div={cand['divergence_score']:.1f}</div>"""

    # Stats line
    max_cond = float(np.max(c.riemannian.condition_number))
    cond_str = f"{max_cond:.1f}" if max_cond < 1e8 else f"{max_cond:.2e}"

    cards_html += f"""
<div class="card" style="margin:12px 0;border-left:3px solid {sc}">
<div class="card-header">
  <span class="card-title">{proto['name']}</span>
  <span class="card-species" style="color:{sc}">{proto['species']}</span>
  <span style="font-size:10px;color:#556;margin-left:8px">{proto['exploit']}</span>
</div>
<div class="card-body">
  <div class="card-info">
    <div class="info-section">
      <div class="info-label">Dark Zones</div>
      {dz_html}
    </div>
    <div class="info-section" style="margin-top:8px">
      <div class="info-label">Top Executor Candidates</div>
      {exec_html}
    </div>
    <div class="info-section" style="margin-top:8px">
      <div class="info-label">Stats</div>
      <div style="font-size:10px;color:#889;line-height:1.6">
        E-I: {c.e1_fraction:.1%} · E-II: {c.e2_fraction:.1%} · E-III: {c.e3_fraction:.1%}<br>
        Unconstrained: {c.unconstrained_fraction:.1%}<br>
        Max condition: {cond_str}<br>
        Structural: {c.structural_score:.4f}
      </div>
    </div>
  </div>
  <div class="card-heatmaps">
    <div class="hm-panel">
      <canvas id="cancel_{key}" class="hm-canvas"></canvas>
      <div class="hm-label">c(p) — Cancellation Ratio</div>
      <div class="hm-legend"><span class="leg-dot" style="background:#44dd66"></span>1.0 (safe)<span class="leg-dot" style="background:#dd4466;margin-left:8px"></span>0.0 (dark zone)</div>
    </div>
    <div class="hm-panel">
      <canvas id="force_{key}" class="hm-canvas"></canvas>
      <div class="hm-label">||Π|| — Constraint Force</div>
      <div class="hm-legend"><span class="leg-dot" style="background:#111"></span>weak<span class="leg-dot" style="background:#ffcc44;margin-left:8px"></span>strong</div>
    </div>
    <div class="hm-panel">
      <canvas id="div_{key}" class="hm-canvas"></canvas>
      <div class="hm-label">|∇·Π| — Divergence</div>
      <div class="hm-legend"><span class="leg-dot" style="background:#111"></span>0<span class="leg-dot" style="background:#ff6644;margin-left:8px"></span>peak (missing executor)</div>
    </div>
    <div class="hm-panel">
      <canvas id="eval_{key}" class="hm-canvas"></canvas>
      <div class="hm-label">λ_min(g) — Unconstrained</div>
      <div class="hm-legend"><span class="leg-dot" style="background:#dd4466"></span>0 (unconstrained)<span class="leg-dot" style="background:#4466dd;margin-left:8px"></span>constrained</div>
    </div>
  </div>
</div>
</div>"""

# Cross-species comparison cards
compare_cards = ''
for proto in PROTOCOLS:
    key = proto['key']
    r = results[key]
    c = r['cis']
    sc = species_colors.get(proto['species_idx'], '#889')
    dz_color = '#dd6644' if r['dz_count'] > 0 else '#44dd66'
    compare_cards += f"""<div class="mini-card" style="border-top:2px solid {sc}">
<div style="font-size:11px;font-weight:bold;margin-bottom:4px">{proto['name']}</div>
<div style="font-size:9px;color:#889;line-height:1.6">
  Species: <b style="color:{sc}">{proto['species']}</b><br>
  Exploit: {proto['exploit']}<br>
  DZ: <b style="color:{dz_color}">{r['dz_count']}</b> ·
  E-I: {c.e1_fraction:.1%}<br>
  Unconstrained: {c.unconstrained_fraction:.1%}<br>
  Structural: {c.structural_score:.3f}
</div></div>"""

html = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8">
<title>CIS — Constraint Invisibility Scanner</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ background:#0a0a12; color:#c8c8d8; font-family:'SF Mono','Menlo','Consolas',monospace; padding:24px 32px; line-height:1.5; }}
h1 {{ font-size:18px; color:#fff; margin-bottom:4px; }}
h2 {{ font-size:14px; color:#8899cc; margin:32px 0 12px; border-bottom:1px solid #1a1a3a; padding-bottom:6px; }}
.sub {{ color:#556; font-size:10px; margin-bottom:20px; }}

/* Summary table */
table {{ width:100%; border-collapse:collapse; font-size:10px; margin:8px 0; }}
th {{ background:#111133; color:#889; padding:7px 10px; text-align:left; font-weight:normal; }}
td {{ padding:7px 10px; border-bottom:1px solid #111122; }}
tr:hover td {{ background:#0f0f24; }}

/* Cards */
.card {{ background:#0e0e20; border:1px solid #1a1a3a; border-radius:8px; overflow:hidden; }}
.card-header {{ padding:10px 14px; background:#0a0a18; border-bottom:1px solid #1a1a3a; display:flex; align-items:center; }}
.card-title {{ font-size:13px; font-weight:bold; color:#fff; }}
.card-species {{ font-size:10px; margin-left:8px; }}
.card-body {{ display:flex; gap:12px; padding:12px; }}
.card-info {{ width:220px; flex-shrink:0; }}
.card-heatmaps {{ flex:1; display:flex; gap:8px; }}

/* Heatmap panels */
.hm-panel {{ flex:1; display:flex; flex-direction:column; align-items:center; min-width:120px; }}
.hm-canvas {{ width:100%; aspect-ratio:1; border-radius:4px; cursor:crosshair; image-rendering:pixelated; }}
.hm-label {{ font-size:8px; color:#889; margin-top:4px; text-align:center; }}
.hm-legend {{ font-size:7px; color:#667; margin-top:2px; display:flex; align-items:center; gap:4px; }}
.leg-dot {{ display:inline-block; width:8px; height:8px; border-radius:2px; }}

/* Info sections */
.info-section {{ }}
.info-label {{ font-size:9px; color:#667; margin-bottom:4px; text-transform:uppercase; letter-spacing:1px; }}

/* Badges */
.badge {{ display:inline-block; padding:1px 5px; border-radius:3px; font-size:8px; }}
.badge-ei {{ background:#441122; color:#dd4466; }}
.badge-eii {{ background:#223311; color:#88cc44; }}
.badge-eiii {{ background:#112244; color:#6688cc; }}

/* Mini cards */
.mini-card {{ background:#0e0e20; border:1px solid #1a1a3a; border-radius:6px; padding:10px 12px; flex:1; min-width:180px; }}

/* Tooltip */
.tooltip {{ position:fixed; background:#1a1a33; border:1px solid #334; color:#fff; padding:8px 10px; border-radius:4px; font-size:9px; pointer-events:none; display:none; z-index:100; line-height:1.5; }}

.callout {{ background:#111122; border-left:3px solid #66aadd; padding:12px 16px; margin:12px 0; font-size:10px; border-radius:0 6px 6px 0; line-height:1.8; }}

footer {{ color:#444; font-size:9px; text-align:center; margin-top:40px; padding:20px; border-top:1px solid #1a1a3a; }}
</style></head>
<body>

<h1>CIS — Constraint Invisibility Scanner</h1>
<div class="sub">Comprehensive Protocol Security Report · A.1–A.7 Full Pipeline<br>
4 dark zone species · {len(PROTOCOLS)} protocols · Helmholtz decomposition · Riemannian metric · Continuity analysis</div>

<h2>Executive Summary</h2>
<table>
<tr><th>Protocol</th><th>Species</th><th>DZ</th><th>E-I</th><th>E-II</th><th>E-III</th><th>Unconstrained</th><th>Structural</th><th>Time</th></tr>
{summary_rows}
</table>

<h2>Per-Protocol CIS Analysis</h2>
{cards_html}

<h2>Cross-Species Comparison</h2>
<div style="display:flex; gap:8px; flex-wrap:wrap; margin:8px 0;">
{compare_cards}
</div>

<div class="callout" style="border-left-color:#cc44dd">
<b>CIS Security Assessment — How to Read the Heatmaps:</b><br>
<b>c(p) → 0</b> (red, top-left): constraint forces cancel out — dark zone. Fix by adding an orthogonal constraint.<br>
<b>λ_min(g) → 0</b> (red, bottom-right): unconstrained direction exists. Fill with additional constraint.<br>
<b>|∇·Π| peak</b> (red, bottom-left): constraint force discontinuity — missing executor location. Add constraint here.<br>
<b>||Π||</b> (top-right): total constraint force magnitude. High force + low c(p) = two constraints fighting each other.<br>
<b>Hover</b> over any heatmap to see exact values at that point.
</div>

<footer>
CIS — Constraint Invisibility Scanner v2 · Full A.1-A.7 Analysis Pipeline<br>
Π = Σ∇σ_i · c(p) = ||Π|| / Σ||∇σ_i|| · g_ij = Σ(∂σ_k/∂x_i)(∂σ_k/∂x_j) · Π = −∇φ + J∇ψ · ∂ρ/∂t + ∇·Π = 0
</footer>

<div class="tooltip" id="tooltip"></div>

<script>
// ── Heatmap data ──
const N = {N_GRID};
const data = {json.dumps(heatmaps)};
const keys = {json.dumps([p['key'] for p in PROTOCOLS])};

// ── Color functions ──
function cancelColor(t) {{
    // t=0 → red (danger), t=1 → green (safe)
    t = Math.max(0, Math.min(1, t));
    const r = Math.floor((1-t) * 220 + t * 30);
    const g = Math.floor(t * 200 + (1-t) * 40);
    const b = Math.floor((1-t) * 80 + t * 40);
    return [r, g, b];
}}

function forceColor(t) {{
    // t=0 → dark, t=1 → bright yellow
    t = Math.max(0, Math.min(1, t));
    const r = Math.floor(t * 240);
    const g = Math.floor(t * 200 + (1-t) * 20);
    const b = Math.floor((1-t) * 80 + t * 30);
    return [r, g, b];
}}

function divColor(t) {{
    // t=0 → dark, t=1 → red-orange
    t = Math.max(0, Math.min(1, t));
    const r = Math.floor(80 + t * 175);
    const g = Math.floor(20 + (1-t) * 60);
    const b = Math.floor(30 + (1-t) * 60);
    return [r, g, b];
}}

function evalColor(t) {{
    // t=0 → red (danger, unconstrained), t=1 → blue (safe)
    t = Math.max(0, Math.min(1, t));
    const r = Math.floor((1-t) * 220 + t * 50);
    const g = Math.floor((1-t) * 40 + t * 100);
    const b = Math.floor(t * 220 + (1-t) * 60);
    return [r, g, b];
}}

// ── Draw functions ──
function drawHeatmap(canvas, arr, colorFn, maxVal) {{
    const ctx = canvas.getContext('2d');
    const size = N;
    canvas.width = size;
    canvas.height = size;

    const imgData = ctx.createImageData(size, size);
    for (let y = 0; y < size; y++) {{
        for (let x = 0; x < size; x++) {{
            const val = arr[y][x];
            const t = maxVal > 0 ? val / maxVal : 0;
            const [r, g, b] = colorFn(t);
            const idx = (y * size + x) * 4;
            imgData.data[idx] = r;
            imgData.data[idx+1] = g;
            imgData.data[idx+2] = b;
            imgData.data[idx+3] = 255;
        }}
    }}
    ctx.putImageData(imgData, 0, 0);

    // Dark zone markers
    canvas._arr = arr;
}}

// ── Hover tooltip ──
const tooltip = document.getElementById('tooltip');

function attachHover(canvas, name, arr, maxVal) {{
    canvas.addEventListener('mousemove', function(e) {{
        const rect = canvas.getBoundingClientRect();
        const scaleX = N / rect.width;
        const scaleY = N / rect.height;
        const x = Math.floor((e.clientX - rect.left) * scaleX);
        const y = Math.floor((e.clientY - rect.top) * scaleY);
        if (x >= 0 && x < N && y >= 0 && y < N) {{
            const val = arr[y][x];
            // Parameter-space coords (y flips back: y=0 at bottom)
            const px = (x / (N-1)).toFixed(3);
            const py = ((N-1-y) / (N-1)).toFixed(3);
            tooltip.style.display = 'block';
            tooltip.style.left = (e.clientX + 14) + 'px';
            tooltip.style.top = (e.clientY - 10) + 'px';
            tooltip.innerHTML = `<b>${{name}}</b><br>p = (${{px}}, ${{py}})<br>value = ${{val.toFixed(4)}}`;
        }} else {{
            tooltip.style.display = 'none';
        }}
    }});
    canvas.addEventListener('mouseleave', function() {{
        tooltip.style.display = 'none';
    }});
}}

// ── Render all heatmaps ──
for (const k of keys) {{
    const d = data[k];

    // Compute max values (use 95th percentile to avoid outlier domination)
    const flatForce = d.force.flat().sort((a,b)=>a-b);
    const flatDiv = d.div.flat().sort((a,b)=>a-b);
    const flatEval = d.eval_min.flat().sort((a,b)=>a-b);
    const p95 = (arr) => arr[Math.floor(arr.length * 0.95)];

    const maxF = p95(flatForce) || 1;
    const maxDiv = p95(flatDiv) || 1;
    const maxEval = p95(flatEval) || 1;

    const cCancel = document.getElementById('cancel_' + k);
    const cForce = document.getElementById('force_' + k);
    const cDiv = document.getElementById('div_' + k);
    const cEval = document.getElementById('eval_' + k);

    if (cCancel) {{
        drawHeatmap(cCancel, d.cancel, cancelColor, 1.05);
        attachHover(cCancel, 'c(p) — Cancellation Ratio', d.cancel, 1.05);
    }}
    if (cForce) {{
        drawHeatmap(cForce, d.force, forceColor, maxF);
        attachHover(cForce, '||Π|| — Force Field', d.force, maxF);
    }}
    if (cDiv) {{
        drawHeatmap(cDiv, d.div, divColor, maxDiv);
        attachHover(cDiv, '|∇·Π| — Divergence', d.div, maxDiv);
    }}
    if (cEval) {{
        drawHeatmap(cEval, d.eval_min, evalColor, maxEval);
        attachHover(cEval, 'λ_min(g) — Unconstrained', d.eval_min, maxEval);
    }}
}}
</script>

</body></html>"""

with open(OUT, 'w') as f:
    f.write(html)

print(f"\nReport: {OUT}")
os.system(f"open {OUT}")

print(f"\n{'='*60}")
print(f"CIS Report: 4 species × {len(PROTOCOLS)} protocols")
print(f"Total dark zones detected: {sum(r['dz_count'] for r in results.values())}")
print(f"Protocols with E-I structural gaps: {sum(1 for r in results.values() if r['cis'].helmholtz.has_structural_gap)}")
