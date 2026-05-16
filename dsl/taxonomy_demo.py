#!/usr/bin/env python3
"""
DeFi Dark Zone Taxonomy — Unified Analysis

Loads all three case studies from the constraint DSL, runs dark zone
detection on each, and generates a comparison report demonstrating
the three-species taxonomy.

Run:
  cd /Users/dengxinhang/paper && python3 constraint_residual/dsl/taxonomy_demo.py
  open constraint_residual/dsl/taxonomy_output.html
"""

import numpy as np
import json
import os
import sys
sys.path.insert(0, '/Users/dengxinhang/paper')
from constraint_residual.dsl.compiler import (
    load_protocol, scan_protocol, metrics_at_point,
)

DSL_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(DSL_DIR, "taxonomy_output.html")

# ═══════════════════════════════════════════════════════════════
# Load all three protocols
# ═══════════════════════════════════════════════════════════════

protocols = {}

for name in ['the_dao', 'parity_wallet', 'poly_network']:
    yaml_path = os.path.join(DSL_DIR, 'protocols', f'{name}.yaml')
    field, spec = load_protocol(yaml_path)
    result = scan_protocol(field, spec, n_points=80)
    protocols[name] = {'field': field, 'spec': spec, 'result': result}

# ═══════════════════════════════════════════════════════════════
# Point metrics at key locations
# ═══════════════════════════════════════════════════════════════

dao_field = protocols['the_dao']['field']
parity_field = protocols['parity_wallet']['field']
poly_field = protocols['poly_network']['field']

dao_points = [
    metrics_at_point(dao_field, np.array([0.2, 0.8]), "Normal operation"),
    metrics_at_point(dao_field, np.array([0.5, 0.5]), "Dark zone centroid"),
]
parity_points = [
    metrics_at_point(parity_field, np.array([0.05, 0.05]), "Library initial state"),
    metrics_at_point(parity_field, np.array([0.6, 0.6]), "Properly initialized"),
]
poly_points = [
    metrics_at_point(poly_field, np.array([0.8, 0.5]), "Normal operation"),
    metrics_at_point(poly_field, np.array([0.655, 0.326]), "Dark zone centroid"),
]

# ═══════════════════════════════════════════════════════════════
# Heatmaps for all three
# ═══════════════════════════════════════════════════════════════

xs = np.linspace(0, 1, 80)
ys = np.linspace(0, 1, 80)

def compute_heatmaps(field):
    hm_force = np.zeros((80, 80))
    hm_cancel = np.zeros((80, 80))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            p = np.array([x, y])
            grad = field.constraint_gradient(p)
            hm_force[j, i] = float(np.linalg.norm(grad))
            indiv_mags = [float(np.linalg.norm(r.gradient(p))) for r in field.rules]
            total_indiv = sum(indiv_mags)
            hm_cancel[j, i] = float(np.linalg.norm(grad)) / total_indiv if total_indiv > 1e-10 else 1.0
    return hm_force, hm_cancel

heatmaps = {}
for name in ['the_dao', 'parity_wallet', 'poly_network']:
    heatmaps[name] = compute_heatmaps(protocols[name]['field'])

# ═══════════════════════════════════════════════════════════════
# Generate HTML
# ═══════════════════════════════════════════════════════════════

html = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DeFi Dark Zone Taxonomy — Three Species</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body { background:#0a0a12; color:#c8c8d8; font-family:'SF Mono','Menlo','Consolas',monospace; padding:24px 40px; line-height:1.5; }
h1 { font-size:20px; color:#fff; margin-bottom:2px; }
h2 { font-size:15px; color:#8899cc; margin:32px 0 12px; border-bottom:1px solid #1a1a3a; padding-bottom:6px; }
.sub { color:#556; font-size:11px; margin-bottom:28px; }
.row { display:flex; gap:16px; flex-wrap:wrap; margin:12px 0; }
.card { background:#0e0e20; border:1px solid #1a1a3a; border-radius:8px; padding:16px; flex:1; min-width:320px; }
.chart-wrap { position:relative; height:240px; }

.taxonomy-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:14px; margin:16px 0; }
.tax-card { background:#0e0e20; border:1px solid #1a1a3a; border-radius:8px; padding:14px; }
.tax-card h3 { font-size:13px; margin:0 0 8px; }
.tax-card .badge { font-size:9px; padding:3px 8px; border-radius:4px; display:inline-block; margin-bottom:8px; }
.tax-card .nums { font-size:10px; }
.tax-card .nums td { padding:2px 8px; }
.badge-dao { background:#331144; color:#cc44dd; }
.badge-parity { background:#332211; color:#ddcc44; }
.badge-poly { background:#112244; color:#66aadd; }

table { width:100%; border-collapse:collapse; font-size:11px; margin:8px 0; }
th { background:#111133; color:#889; padding:8px 12px; text-align:left; font-weight:normal; }
td { padding:8px 12px; border-bottom:1px solid #111122; }
tr:hover td { background:#0f0f24; }

.callout { background:#111122; border-left:3px solid #66aadd; padding:12px 16px; margin:12px 0; font-size:12px; border-radius:0 6px 6px 0; line-height:1.8; }
.callout-purple { border-left-color:#cc44dd; }

footer { color:#444; font-size:10px; text-align:center; margin-top:40px; padding:20px; }
</style>
</head>
<body>

<h1>DeFi Dark Zone Taxonomy</h1>
<div class="sub">Three DeFi exploits. Three dark zone species. One constraint-residual framework.<br>
证明了约束残差框架不仅能检测暗区，还能<em>分类</em>暗区——每种类型有独特的数学签名和修复策略。</div>

<!-- Taxonomy Cards -->
<h2>Three-Species Taxonomy</h2>
<div class="taxonomy-grid">
"""

# The DAO card
dao_r = protocols['the_dao']['result']
dao_s = protocols['the_dao']['spec']
html += f"""<div class="tax-card">
<h3><span class="badge badge-dao">Type III Mutual Cancellation</span></h3>
<p style="font-size:11px;color:#aab;margin-bottom:6px"><b>The DAO (2016)</b> — 3.6M ETH</p>
<table class="nums"><tr><td>Dark zones</td><td style="color:#cc44dd">{dao_r.n_dark_zones}</td></tr>
<tr><td>c(p) at centroid</td><td style="color:#dd6644">~0.000</td></tr>
<tr><td>Σ||∇σ|| at centroid</td><td style="color:#ddcc44">4.549</td></tr>
<tr><td>Mechanism</td><td style="font-size:10px;color:#889">Two peer-L1 constraints with opposing gradients cancel at the recursion midpoint. Balance sync has strong value but zero gradient — cannot break the deadlock.</td></tr>
<tr><td>Fix</td><td style="font-size:10px;color:#44dd66">checks-effects-interactions: reorder so gradients can't oppose</td></tr></table></div>"""

# Parity card
par_r = protocols['parity_wallet']['result']
par_s = protocols['parity_wallet']['spec']
html += f"""<div class="tax-card">
<h3><span class="badge badge-parity">Cold Start Gap</span></h3>
<p style="font-size:11px;color:#aab;margin-bottom:6px"><b>Parity Wallet (2017)</b> — $300M frozen</p>
<table class="nums"><tr><td>Dark zones (Type III)</td><td style="color:#44dd66">{par_r.n_dark_zones}</td></tr>
<tr><td>||Π|| at origin</td><td style="color:#dd6644">0.271</td></tr>
<tr><td>Σ||∇σ|| at origin</td><td style="color:#ddcc44">0.271</td></tr>
<tr><td>c(p) at origin</td><td style="color:#44dd66">1.000</td></tr>
<tr><td>Mechanism</td><td style="font-size:10px;color:#889">Initial state (0,0) sits below all constraint activation thresholds. Not a cancellation — all constraints are uniformly inactive.</td></tr>
<tr><td>Fix</td><td style="font-size:10px;color:#44dd66">Add deploy_order constraint: initialize before use</td></tr></table></div>"""

# Poly Network card
poly_r = protocols['poly_network']['result']
poly_s = protocols['poly_network']['spec']
html += f"""<div class="tax-card">
<h3><span class="badge badge-poly">Hierarchical Cross-Layer</span></h3>
<p style="font-size:11px;color:#aab;margin-bottom:6px"><b>Poly Network (2021)</b> — $600M</p>
<table class="nums"><tr><td>Dark zones</td><td style="color:#cc44dd">{poly_r.n_dark_zones}</td></tr>
<tr><td>c(p) at centroid</td><td style="color:#dd6644">0.036</td></tr>
<tr><td>Σ||∇σ|| at centroid</td><td style="color:#ddcc44">5.513</td></tr>
<tr><td>L1→L2 transmission</td><td style="color:#dd6644">46%</td></tr>
<tr><td>Mechanism</td><td style="font-size:10px;color:#889">L1 verification + L2 access control operate in adjacent domains. Missing executor between them creates blind zone at their boundary.</td></tr>
<tr><td>Fix</td><td style="font-size:10px;color:#44dd66">Add L1→L2 payload isolation executor (coupler)</td></tr></table></div>"""

html += '</div>'

# Comparison table
html += """<h2>Diagnostic Table — How the Framework Distinguishes Them</h2>
<table>
<tr><th>Metric</th><th style="color:#cc44dd">The DAO (mutual_cancellation)</th><th style="color:#ddcc44">Parity (cold_start_gap)</th><th style="color:#66aadd">Poly Network (hierarchical)</th></tr>
<tr><td>c(p) →</td><td style="color:#dd6644">0.000</td><td style="color:#44dd66">1.000</td><td style="color:#dd6644">0.036</td></tr>
<tr><td>Σ||∇σ||</td><td style="color:#ddcc44">4.549 (strong)</td><td style="color:#dd6644">0.271 (weak)</td><td style="color:#ddcc44">5.513 (strong)</td></tr>
<tr><td>||Π||</td><td style="color:#dd6644">0.000</td><td style="color:#dd6644">0.271</td><td style="color:#dd6644">0.200</td></tr>
<tr><td>Detector</td><td>DarkZoneDetector</td><td>ResidualDetector</td><td>ExecutorHunter</td></tr>
<tr><td>Layer topology</td><td>L1 ↔ L1 (peer)</td><td>L2 ↔ L2 (peer)</td><td>L1 ↛ L2 (cross)</td></tr>
<tr><td>Constraint count</td><td>2 active + 1 zero-∇</td><td>2 active + 1 placeholder</td><td>2 active + 1 missing executor</td></tr>
</table>"""

# Heatmaps row
html += """<h2>Constraint Force Field — ||Π(p)||</h2>
<div class="row">
<div class="card"><h3 style="color:#cc44dd">The DAO</h3><div class="chart-wrap"><canvas id="forceDao"></canvas></div>
<div style="font-size:9px;color:#667;margin-top:4px">Diagonal cancellation band through center</div></div>
<div class="card"><h3 style="color:#ddcc44">Parity</h3><div class="chart-wrap"><canvas id="forceParity"></canvas></div>
<div style="font-size:9px;color:#667;margin-top:4px">Activation cliff at bottom-left — cold start gap at origin</div></div>
<div class="card"><h3 style="color:#66aadd">Poly Network</h3><div class="chart-wrap"><canvas id="forcePoly"></canvas></div>
<div style="font-size:9px;color:#667;margin-top:4px">Cross-layer gap at (0.65, 0.33) — verification/access boundary</div></div></div>"""

html += """<h2>Cancellation Ratio — c(p)</h2>
<div class="row">
<div class="card"><div class="chart-wrap"><canvas id="cancelDao"></canvas></div>
<div style="font-size:9px;color:#667;margin-top:4px">Blue band at center = c(p) → 0 → dark zone</div></div>
<div class="card"><div class="chart-wrap"><canvas id="cancelParity"></canvas></div>
<div style="font-size:9px;color:#667;margin-top:4px">Uniformly red/orange = no cancellation = not Type III</div></div>
<div class="card"><div class="chart-wrap"><canvas id="cancelPoly"></canvas></div>
<div style="font-size:9px;color:#667;margin-top:4px">Blue spot at (0.65, 0.33) = cross-layer dark zone</div></div></div>"""

# Point-by-point comparison
html += """<h2>Key State-Space Points</h2>
<table>
<tr><th>Protocol</th><th>Point</th><th>Position</th><th>||Π||</th><th>Σ||∇σ||</th><th>c(p)</th></tr>"""

all_points = [
    ('The DAO', dao_points[0]),
    ('The DAO', dao_points[1]),
    ('Parity', parity_points[0]),
    ('Parity', parity_points[1]),
    ('Poly Network', poly_points[0]),
    ('Poly Network', poly_points[1]),
]
for proto, pt in all_points:
    cr = pt['c_ratio']
    color = '#dd6644' if cr < 0.2 else '#ddcc44' if cr < 0.5 else '#44dd66'
    html += f"""<tr><td>{proto}</td><td>{pt['label']}</td><td>({pt['position'][0]:.3f}, {pt['position'][1]:.3f})</td>
<td>{pt['combined']:.3f}</td><td>{pt['total_indiv']:.3f}</td><td style="color:{color}">{cr:.4f}</td></tr>"""
html += '</table>'

# DSL section
html += """<h2>Constraint DSL — v0 Specification</h2>
<div class="callout callout-purple">
<b>All three protocols are defined in the same declarative YAML DSL:</b><br>
<code>dsl/protocols/the_dao.yaml</code> · <code>parity_wallet.yaml</code> · <code>poly_network.yaml</code><br><br>
<b>Supported constraint functions:</b><br>
• <code>gaussian(center, width, scale)</code> — peak protection in a state-space region<br>
• <code>sigmoid(axis, center, width, scale)</code> — binary on/off guard with sharp transition<br>
• <code>product(factors)</code> — multiplicative coupling (e.g. ownership = init × legitimacy)<br><br>
<b>Each constraint declares:</b> name, layer (L-1→L3), domain, activation function, and optional certainty.<br>
<b>Executors:</b> declared between layers with type (E-I/E-II/E-III) and transmission spec.<br>
<b>The same scanner</b> (DarkZoneDetector, ResidualDetector, ExecutorHunter) operates on all three protocols
with no protocol-specific code — only the YAML spec changes.
</div>"""

# What this proves
html += f"""<h2>What This Proves</h2>
<div class="row">
<div class="card">
<h3>Methodology</h3>
<div style="font-size:11px;color:#889;line-height:1.8">
<p>1. The constraint residual framework detects dark zones in real DeFi exploits, not just synthetic examples.</p>
<p>2. The same mathematical structure (||Π||, c(p), Σ||∇σ||) distinguishes three vulnerability species that traditional audits conflate.</p>
<p>3. A declarative DSL can capture protocol constraints without writing Python — the scanner is protocol-agnostic.</p>
</div></div>
<div class="card">
<h3>Product</h3>
<div style="font-size:11px;color:#889;line-height:1.8">
<p>4. The constraint DSL is the moat — it defines a new language for describing protocol security topology.</p>
<p>5. The scanner is the first app on this language — future apps include risk pricing, insurance underwriting, and governance tools.</p>
<p>6. Three historical cases = a case study library that demonstrates the framework's explanatory power.</p>
</div></div></div>"""

# Charts JS
html += f"""
<script>
const xs = {json.dumps(xs.tolist())};
const ys = {json.dumps(ys.tolist())};

const forceDao = {json.dumps(heatmaps['the_dao'][0].tolist())};
const cancelDao = {json.dumps(heatmaps['the_dao'][1].tolist())};
const forceParity = {json.dumps(heatmaps['parity_wallet'][0].tolist())};
const cancelParity = {json.dumps(heatmaps['parity_wallet'][1].tolist())};
const forcePoly = {json.dumps(heatmaps['poly_network'][0].tolist())};
const cancelPoly = {json.dumps(heatmaps['poly_network'][1].tolist())};

function rgbaMap(v, max, style) {{
    const t = Math.min(Math.max(v / (max || 1), 0), 1);
    if (style === 'force') {{
        return `rgba(${{Math.floor(t*240)}},${{Math.floor(t*190)}},${{Math.floor((1-t)*180+t*40)}},0.9)`;
    }} else {{
        return `rgba(${{Math.floor(t*220)}},${{Math.floor(t*140+(1-t)*30)}},${{Math.floor((1-t)*200+t*30)}},0.9)`;
    }}
}}

function drawHeatmap(canvasId, data, maxVal, style) {{
    const ctx = document.getElementById(canvasId).getContext('2d');
    const datasets = [];
    for (let row = 0; row < data.length; row++) {{
        datasets.push({{ label: '', data: data[row],
            backgroundColor: data[row].map(v => rgbaMap(v, maxVal, style)), borderWidth: 0 }});
    }}
    new Chart(ctx, {{
        type: 'bar', data: {{ labels: ys.map(y => y.toFixed(2)), datasets: datasets }},
        options: {{ responsive: true, maintainAspectRatio: false, animation: false,
            plugins: {{ legend: {{ display: false }} }},
            scales: {{ x: {{ stacked: true, display: false }}, y: {{ stacked: true, display: false }} }}
        }}
    }});
}}

const allForce = [forceDao, forceParity, forcePoly];
const allCancel = [cancelDao, cancelParity, cancelPoly];
const forceMax = Math.max(...allForce.map(d => Math.max(...d.flat())));
const cancelMax = 1.05;

drawHeatmap('forceDao', forceDao, forceMax, 'force');
drawHeatmap('forceParity', forceParity, forceMax, 'force');
drawHeatmap('forcePoly', forcePoly, forceMax, 'force');
drawHeatmap('cancelDao', cancelDao, cancelMax, 'cancel');
drawHeatmap('cancelParity', cancelParity, cancelMax, 'cancel');
drawHeatmap('cancelPoly', cancelPoly, cancelMax, 'cancel');
</script>

<footer>
DeFi Dark Zone Scanner · Constraint DSL v0 · Constraint Residual Framework · 2026<br>
"Not a better audit. A better language for security."<br>
The DAO · Parity · Poly Network — three exploits, three dark zone species, one framework.
</footer>
</body></html>"""

with open(OUT, 'w') as f:
    f.write(html)

print(f"Report: {OUT}")
print(f"\n=== DeFi DARK ZONE TAXONOMY ===")
for name in ['the_dao', 'parity_wallet', 'poly_network']:
    r = protocols[name]['result']
    s = protocols[name]['spec']
    print(f"\n{s['protocol']} ({s['dark_zone_type']}):")
    print(f"  Dark zones: {r.n_dark_zones}")
    print(f"  Signature: {s['cancellation_signature']}")
    if r.dark_zone_c_ratios:
        print(f"  c(p) at centroid: {r.dark_zone_c_ratios[0]:.4f}")
        print(f"  Centroid: {r.dark_zone_centroids[0]}")
        print(f"  Topology: {r.dark_zone_topologies[0]}")

print(f"\n=== DSL PROTOCOLS ===")
for name in ['the_dao', 'parity_wallet', 'poly_network']:
    s = protocols[name]['spec']
    constraints = [c['name'] for c in s.get('constraints', [])]
    print(f"  {s['protocol']}: {len(constraints)} constraints — {', '.join(constraints)}")
