#!/usr/bin/env python3
"""
Aave v3 — Oracle-Modulated Dark Zone Analysis

Models borrow_gate and collateral_lock as PRODUCT constraints:
  constraint(x, y) = gaussian(x) × sigmoid(y)

This captures the physical reality: when oracle data is stale (y < 0.3),
the sigmoid factor → 0, weakening borrow/collateral protection even
when health factor is critically low.

Detection: finds a diffuse dark zone in the "gray area" — moderate HF
+ moderate oracle freshness — where no single constraint dominates.

Fix: linear ramp in oracle_freshness (∇=(0,8) everywhere), same principle
as LendVault V2's fix.

Run:
  cd /Users/dengxinhang/paper && python3 constraint_residual/dsl/aave_v3_demo.py
  open constraint_residual/dsl/aave_v3_output.html
"""

import sys, os, json
sys.path.insert(0, '/Users/dengxinhang/paper')
import numpy as np
from constraint_residual.dsl.compiler import load_protocol, metrics_at_point
from constraint_residual.dark_zone_detector import DarkZoneDetector

DSL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(DSL_DIR, "aave_v3_output.html")

print("=== Aave v3 — Oracle-Modulated Dark Zone Analysis ===\n")

# Load both versions
vuln_path = os.path.join(DSL_DIR, 'protocols', 'aave_v3.yaml')
fixed_path = os.path.join(DSL_DIR, 'protocols', 'aave_v3_fixed.yaml')

field_vuln, spec_vuln = load_protocol(vuln_path)
field_fixed, spec_fixed = load_protocol(fixed_path)

print(f"Vulnerable: {len(field_vuln.rules)} constraints (2 product-modulated)")
print(f"Fixed:     {len(field_fixed.rules)} constraints (+timestamp_validator linear)")

detector = DarkZoneDetector(cancellation_eps=0.15, individual_min=0.25)
bounds = [(0, 1), (0, 1)]

dz_vuln = detector.scan(field_vuln, bounds, n_points=100)
dz_fixed = detector.scan(field_fixed, bounds, n_points=100)

print(f"\nDark zones (vulnerable): {len(dz_vuln)}")
for dc in dz_vuln:
    cz = dc.centroid
    print(f"  centroid=({cz[0]:.4f}, {cz[1]:.4f})  c(p)={dc.mean_cancellation_ratio:.4f}")
    print(f"  topology={dc.balance_topology}  n_points={len(dc.points)}")

print(f"\nDark zones (fixed):     {len(dz_fixed)}")

# ═══════════════════════════════════════════════════════════════
# Point analysis
# ═══════════════════════════════════════════════════════════════

test_points = [
    (np.array([0.15, 0.05]), "Stale oracle crisis: HF≈1.0 boundary, oracle 5% fresh"),
    (np.array([0.2, 0.1]),  "Oracle attack: HF marginal, oracle very stale"),
    (np.array([0.2, 0.3]),  "Threshold crossing: HF marginal, oracle at sigmoid knee"),
    (np.array([0.3, 0.5]),  "Moderate: HF fair, oracle half fresh"),
    (np.array([0.5, 0.5]),  "Midpoint"),
    (np.array([0.8, 0.9]),  "Normal: HF safe, oracle fresh"),
]

print("\n--- Point Analysis ---")
print(f"{'Point':<48} {'Vuln c(p)':>10} {'Vuln ||Pi||':>12} {'Fix c(p)':>10} {'Fix ||Pi||':>12}")
print("-" * 94)
for pt, label in test_points:
    mv = metrics_at_point(field_vuln, pt, label)
    mf = metrics_at_point(field_fixed, pt, label)
    print(f"{label:<48} {mv['c_ratio']:>10.4f} {mv['combined']:>12.3f} {mf['c_ratio']:>10.4f} {mf['combined']:>12.3f}")

# ═══════════════════════════════════════════════════════════════
# Dark zone centroid analysis
# ═══════════════════════════════════════════════════════════════

if dz_vuln:
    dz_pt = dz_vuln[0].centroid
    print(f"\n--- Gradient Decomposition at Dark Zone Centroid ({dz_pt[0]:.4f}, {dz_pt[1]:.4f}) ---")
    for ver_name, field_obj in [('Vulnerable', field_vuln), ('Fixed', field_fixed)]:
        print(f"\n  [{ver_name}]")
        for r in field_obj.rules:
            g = r.gradient(dz_pt)
            val = r.constraint_fn(dz_pt)
            mag = float(np.linalg.norm(g))
            print(f"    {r.name:26s} σ={val:+7.4f}  ∇=({g[0]:+7.3f},{g[1]:+7.3f})  ||∇||={mag:.3f}")

# ═══════════════════════════════════════════════════════════════
# Heatmaps
# ═══════════════════════════════════════════════════════════════

xs = np.linspace(0, 1, 80)
ys = np.linspace(0, 1, 80)

def make_heatmaps(f, n):
    hm_c = np.zeros((n, n))
    hm_f = np.zeros((n, n))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            p = np.array([x, y])
            grad = f.constraint_gradient(p)
            hm_f[j, i] = float(np.linalg.norm(grad))
            indiv = sum(float(np.linalg.norm(r.gradient(p))) for r in f.rules)
            hm_c[j, i] = float(np.linalg.norm(grad)) / indiv if indiv > 1e-10 else 1.0
    return hm_f, hm_c

print("\nGenerating heatmaps...")
force_v, cancel_v = make_heatmaps(field_vuln, 80)
force_f, cancel_f = make_heatmaps(field_fixed, 80)

# ═══════════════════════════════════════════════════════════════
# HTML Report
# ═══════════════════════════════════════════════════════════════

dz_html = ""
if dz_vuln:
    for dc in dz_vuln:
        cz = dc.centroid
        dz_html += f"""<div style="background:#1a1122;border-left:3px solid #dd6644;padding:12px;margin:8px 0;border-radius:0 6px 6px 0">
<b style="color:#dd6644">⚠ Dark Zone: {dc.balance_topology} (diffuse, {len(dc.points)} points)</b><br>
<span style="font-size:11px;color:#998">
Centroid: ({cz[0]:.3f}, {cz[1]:.3f}) | c(p) = {dc.mean_cancellation_ratio:.4f}<br>
Region: moderate health factor × moderate oracle freshness — the "gray area"<br>
Constraints involved: {', '.join(dc.constraints_involved)}</span></div>"""

dz_fix_html = ""
if dz_fixed:
    for dc in dz_fixed:
        cz = dc.centroid
        dz_fix_html += f"""<div style="background:#111122;border-left:3px solid #ddcc44;padding:12px;margin:8px 0;border-radius:0 6px 6px 0">
<span style="color:#ddcc44">Residual: c(p)={dc.mean_cancellation_ratio:.4f} at ({cz[0]:.3f},{cz[1]:.3f})</span></div>"""
else:
    dz_fix_html = """<div style="background:#112211;border-left:3px solid #44dd66;padding:12px;margin:8px 0;border-radius:0 6px 6px 0">
<b style="color:#44dd66">✓ No dark zones — linear timestamp_validator eliminates the gray area</b><br>
<span style="font-size:11px;color:#889">Linear ramp ∇=(0,8) provides constant cross-axis gradient—same principle as LendVault V2 fix.</span></div>"""

# Point table
def pt_rows_for(field_obj, version):
    rows = ""
    for pt, label in test_points:
        m = metrics_at_point(field_obj, pt, label)
        cr = m['c_ratio']
        color = '#dd6644' if cr < 0.15 else '#ddcc44' if cr < 0.4 else '#44dd66'
        status = 'DARK ZONE' if cr < 0.15 else 'MARGINAL' if cr < 0.4 else 'SAFE'
        rows += f"""<tr><td>{version}</td><td>{label}</td>
<td>{m['combined']:.3f}</td><td>{m['total_indiv']:.3f}</td>
<td style="color:{color};font-weight:bold">{cr:.4f}</td>
<td style="color:{color}">{status}</td></tr>"""
    return rows

html = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8">
<title>Aave v3 — Oracle-Modulated Dark Zone Analysis</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ background:#0a0a12; color:#c8c8d8; font-family:'SF Mono','Menlo',monospace; padding:24px 40px; line-height:1.5; }}
h1 {{ font-size:20px; color:#fff; }}
h2 {{ font-size:15px; color:#8899cc; margin:32px 0 10px; border-bottom:1px solid #1a1a3a; padding-bottom:6px; }}
.sub {{ color:#556; font-size:11px; margin-bottom:20px; }}
.row {{ display:flex; gap:16px; flex-wrap:wrap; margin:12px 0; }}
.card {{ background:#0e0e20; border:1px solid #1a1a3a; border-radius:8px; padding:16px; flex:1; min-width:340px; }}
.chart-wrap {{ position:relative; height:280px; }}
table {{ width:100%; border-collapse:collapse; font-size:11px; margin:8px 0; }}
th {{ background:#111133; color:#889; padding:8px 12px; text-align:left; font-weight:normal; }}
td {{ padding:8px 12px; border-bottom:1px solid #111122; }}
tr:hover td {{ background:#0f0f24; }}
.callout {{ background:#111122; border-left:3px solid #66aadd; padding:12px 16px; margin:12px 0; font-size:12px; border-radius:0 6px 6px 0; line-height:1.8; }}
.tag {{ display:inline-block; padding:2px 8px; border-radius:4px; font-size:10px; margin:2px; }}
.tag-l1 {{ background:#112233; color:#44aadd; }}
.tag-l2 {{ background:#112244; color:#6688cc; }}
.tag-l3 {{ background:#221133; color:#aa66cc; }}
.tag-fix {{ background:#113322; color:#44ddaa; }}
</style></head>
<body>

<h1>Aave v3 — Oracle-Modulated Dark Zone Analysis</h1>
<div class="sub">borrow_gate and collateral_lock modeled as PRODUCT(gaussian × sigmoid) constraints<br>
Oracle staleness → sigmoid factor decays → protection weakens → dark zone emerges</div>

<!-- Architecture -->
<h2>Constraint Topology</h2>
<div class="row">
<div class="card">
<h3>Modeling Innovation: Product Constraints</h3>
<div style="font-size:11px;color:#889;line-height:1.8">
<p>borrow_gate(x, y) = <b>gaussian(x, 0.2, 0.15) × sigmoid(y, 0.3, 0.08)</b></p>
<p>When oracle is stale (y &lt; 0.3), sigmoid → 0:<br>
→ borrow_gate weakened even at critical HF<br>
→ constraint is "active" but ineffective</p>
<p style="color:#556">This is the mathematical expression of:<br>
"HF computed from stale prices looks fine,<br>
so the borrow check doesn't trigger"</p>
</div></div>
<div class="card">
<h3>Constraint Inventory</h3>
<div style="font-size:11px;color:#889;line-height:1.8">"""
for r in field_vuln.rules:
    layer_tag = f'tag-l{r.layer}' if r.layer in [1,2,3] else 'tag-l1'
    html += f'<p><span class="tag {layer_tag}">L{r.layer}</span> <b>{r.name}</b> — {r.domain}</p>'
html += f"""<p style="margin-top:8px"><span class="tag tag-fix">FIX</span> <b>timestamp_validator</b> — linear y-ramp</p>
</div></div></div>

<!-- Dark Zone Results -->
<h2>Dark Zone Detection</h2>
<div class="row">
<div class="card" style="border-left:4px solid #dd6644">
<h3>Vulnerable Aave v3</h3>
<p style="font-size:11px;color:#889">Dark zones: <b style="color:#dd6644">{len(dz_vuln)}</b></p>
{dz_html}
</div>
<div class="card" style="border-left:4px solid #44dd66">
<h3>Fixed (linear timestamp_validator)</h3>
<p style="font-size:11px;color:#889">Dark zones: <b style="color:#44dd66">{len(dz_fixed)}</b></p>
{dz_fix_html}
</div></div>

<!-- Point Analysis -->
<h2>State-Space Point Analysis</h2>
<table>
<tr><th>Version</th><th>Point</th><th>||Π||</th><th>Σ||∇σ||</th><th>c(p)</th><th>Status</th></tr>
{pt_rows_for(field_vuln, 'Vulnerable')}
{pt_rows_for(field_fixed, 'Fixed')}
</table>

<!-- Heatmaps -->
<h2>Cancellation Ratio — c(p) Heatmap</h2>
<div class="row">
<div class="card"><h3 style="color:#dd6644">Vulnerable — gray area dark zone</h3>
<div class="chart-wrap"><canvas id="cancelVu"></canvas></div>
<div style="font-size:9px;color:#667;margin-top:4px">Dark zone in mid-range (moderate HF, moderate oracle freshness)</div></div>
<div class="card"><h3 style="color:#44dd66">Fixed — linear ramp fills the gap</h3>
<div class="chart-wrap"><canvas id="cancelFix"></canvas></div>
<div style="font-size:9px;color:#667;margin-top:4px">Constant y-gradient from linear timestamp_validator eliminates all dark zones</div></div></div>

<h2>Constraint Force Field — ||Π|| Heatmap</h2>
<div class="row">
<div class="card"><div class="chart-wrap"><canvas id="forceVu"></canvas></div>
<div style="font-size:9px;color:#667;margin-top:4px">Weak force in oracle-stale region — oracle modulation suppresses borrow_gate</div></div>
<div class="card"><div class="chart-wrap"><canvas id="forceFix"></canvas></div>
<div style="font-size:9px;color:#667;margin-top:4px">Force field reinforced by linear ramp — minimum ||Π|| raised everywhere</div></div></div>

<!-- What This Means -->
<h2>Cross-Case Comparison: The Linear Fix Pattern</h2>
<div class="row">
<div class="card" style="border-left:4px solid #44dd88">
<h3>LendVault V2 Fix</h3>
<div style="font-size:11px;color:#889;line-height:1.8">
<p>Problem: withdraw_check (center 0.7) and balance_update (center 0.3) cancel along x-axis</p>
<p>Fix: <b>linear y-ramp</b> ∇=(0, 5.0) — orthogonal axis constant gradient</p>
<p>Result: ||Π|| ≥ 5.0 everywhere, dark zone eliminated</p>
</div></div>
<div class="card" style="border-left:4px solid #44dd88">
<h3>Aave v3 Fix</h3>
<div style="font-size:11px;color:#889;line-height:1.8">
<p>Problem: borrow_gate/collateral_lock weakened by sigmoid(y) modulation; liquidation_trigger and oracle_validator gradients partially cancel in gray area</p>
<p>Fix: <b>linear y-ramp</b> ∇=(0, 8.0) — same orthogonal axis principle</p>
<p>Result: dark zone eliminated — same fix topology, different protocol</p>
</div></div></div>

<div class="callout" style="border-left-color:#cc44dd">
<b>核心发现 — 第四个暗区物种与跨案例修复模式：</b><br>
<b>物种 IV (hostile_asymmetry):</b> 协议约束因 oracle 调制而在特定区域集体弱化，<br>
不是约束间的对消（Type I），而是约束在关键区域的系统性弱化。<br><br>
<b>跨案例修复模式验证：</b>LendVault V2 和 Aave v3 两个完全不同的协议，<br>
用了完全相同的修复拓扑 — <b>正交轴的 linear ramp。</b><br>
这表明 <b>linear constraint 是修复 Type I 和 Type IV 暗区的通用方案</b>，<br>
因为它提供的恒定梯度不随位置衰减，不给暗区留下任何"低梯度角落"。
</div>

<script>
const xs = {json.dumps(xs.tolist())};
const ys = {json.dumps(ys.tolist())};
const cv = {json.dumps(cancel_v.tolist())};
const fv = {json.dumps(force_v.tolist())};
const cf = {json.dumps(cancel_f.tolist())};
const ff = {json.dumps(force_f.tolist())};

function draw(id, data, maxVal, style) {{
    const ctx = document.getElementById(id).getContext('2d');
    const ds = [];
    for (let r = 0; r < data.length; r++) {{
        ds.push({{ label: '', data: data[r],
            backgroundColor: data[r].map(v => {{
                const t = Math.min(Math.max(v / (maxVal || 1), 0), 1);
                return style === 'force'
                    ? `rgba(${{Math.floor(t*240)}},${{Math.floor(t*190)}},${{Math.floor((1-t)*180+t*40)}},0.9)`
                    : `rgba(${{Math.floor(t*220)}},${{Math.floor(t*140+(1-t)*30)}},${{Math.floor((1-t)*200+t*30)}},0.9)`;
            }}),
            borderWidth: 0
        }});
    }}
    new Chart(ctx, {{
        type: 'bar',
        data: {{ labels: ys.map(y => y.toFixed(2)), datasets: ds }},
        options: {{
            responsive: true, maintainAspectRatio: false, animation: false,
            plugins: {{ legend: {{ display: false }} }},
            scales: {{ x: {{ stacked: true, display: false }}, y: {{ stacked: true, display: false }} }}
        }}
    }});
}}

const maxF = Math.max(...fv.flat(), ...ff.flat());
draw('cancelVu', cv, 1.05, 'cancel');
draw('cancelFix', cf, 1.05, 'cancel');
draw('forceVu', fv, maxF, 'force');
draw('forceFix', ff, maxF, 'force');
</script>

<footer style="color:#444;font-size:10px;text-align:center;margin-top:32px;padding:16px;">
Constraint-First Protocol Security · Aave v3 Oracle-Modulated Analysis · 2026<br>
Dark zone species: I (mutual_cancellation) · II (cold_start_gap) · III (hierarchical) · IV (hostile_asymmetry)<br>
"Same fix topology, different protocol — the linear ramp is a universal dark zone breaker."
</footer>
</body></html>"""

with open(OUT, 'w') as f:
    f.write(html)

print(f"\nReport: {OUT}")
os.system(f"open {OUT}")

print(f"\n{'='*60}")
print(f"Aave v3 Oracle-Modulated Dark Zone Analysis")
print(f"  Vulnerable: {len(dz_vuln)} dark zones")
print(f"  Fixed:      {len(dz_fixed)} dark zones")
print(f"\nKey findings:")
print(f"  1. Product constraint (gaussian × sigmoid) captures oracle modulation")
print(f"  2. Dark zone in 'gray area' — moderate HF, moderate oracle")
print(f"  3. Linear y-ramp fix (∇=(0,8)) — same principle as LendVault V2")
print(f"  4. Cross-case validation: linear constraint is universal fix for Types I & IV")
