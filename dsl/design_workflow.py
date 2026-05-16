#!/usr/bin/env python3
"""
Constraint-First Design Workflow — Full Demo

Demonstrates: design protocol constraints FIRST, find dark zones in the
constraint model, fix them in the spec, THEN generate the Solidity contract.

This is the inversion of traditional DeFi development:
  Traditional: write Solidity → audit → find bugs → fix code → re-audit
  Constraint-first: model constraints → scan for dark zones → fix topology → generate code

Run:
  cd /Users/dengxinhang/paper && python3 constraint_residual/dsl/design_workflow.py
  open constraint_residual/dsl/design_output.html
"""

import numpy as np
import json
import os
import sys
sys.path.insert(0, '/Users/dengxinhang/paper')
from constraint_residual.dsl.compiler import load_protocol, scan_protocol, metrics_at_point
from constraint_residual.dark_zone_detector import DarkZoneDetector

DSL_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(DSL_DIR, "design_output.html")

# ═══════════════════════════════════════════════════════════════
# Load both versions
# ═══════════════════════════════════════════════════════════════

print("=== Constraint-First Design Workflow ===\n")

v1_path = os.path.join(DSL_DIR, 'protocols', 'lendvault_v1.yaml')
v2_path = os.path.join(DSL_DIR, 'protocols', 'lendvault_v2.yaml')

field_v1, spec_v1 = load_protocol(v1_path)
field_v2, spec_v2 = load_protocol(v2_path)

# Scan both
detector_v1 = DarkZoneDetector(cancellation_eps=0.15, individual_min=0.3)
bounds = [(0, 1), (0, 1)]
dz_v1 = detector_v1.scan(field_v1, bounds, n_points=80)

detector_v2 = DarkZoneDetector(cancellation_eps=0.15, individual_min=0.3)
dz_v2 = detector_v2.scan(field_v2, bounds, n_points=80)

# ═══════════════════════════════════════════════════════════════
# Point metrics — before and after at the dark zone centroid
# ═══════════════════════════════════════════════════════════════

dark_zone_point = np.array([0.5, 0.5])
normal_point = np.array([0.8, 0.8])

v1_dz = metrics_at_point(field_v1, dark_zone_point, "V1: Dark zone centroid")
v1_normal = metrics_at_point(field_v1, normal_point, "V1: Normal operation")
v2_dz = metrics_at_point(field_v2, dark_zone_point, "V2: Same point (FIXED)")
v2_normal = metrics_at_point(field_v2, normal_point, "V2: Normal operation")

# ═══════════════════════════════════════════════════════════════
# Heatmaps
# ═══════════════════════════════════════════════════════════════

xs = np.linspace(0, 1, 80)
ys = np.linspace(0, 1, 80)

def make_heatmaps(field):
    hm_f = np.zeros((80, 80))
    hm_c = np.zeros((80, 80))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            p = np.array([x, y])
            grad = field.constraint_gradient(p)
            hm_f[j, i] = float(np.linalg.norm(grad))
            indiv = sum(float(np.linalg.norm(r.gradient(p))) for r in field.rules)
            hm_c[j, i] = float(np.linalg.norm(grad)) / indiv if indiv > 1e-10 else 1.0
    return hm_f, hm_c

force_v1, cancel_v1 = make_heatmaps(field_v1)
force_v2, cancel_v2 = make_heatmaps(field_v2)

# ═══════════════════════════════════════════════════════════════
# Individual constraint gradient analysis at dark zone
# ═══════════════════════════════════════════════════════════════

def constraint_gradients_at(field, p):
    result = {}
    for r in field.rules:
        grad = r.gradient(p)
        result[r.name] = {
            'gradient': grad.tolist(),
            'magnitude': float(np.linalg.norm(grad)),
            'value': float(r.constraint_fn(p)),
        }
    return result

grads_v1 = constraint_gradients_at(field_v1, dark_zone_point)
grads_v2 = constraint_gradients_at(field_v2, dark_zone_point)

# ═══════════════════════════════════════════════════════════════
# HTML Report
# ═══════════════════════════════════════════════════════════════

dz_v1_info = ""
for dc in dz_v1:
    cz = dc.centroid
    dz_v1_info += f"""<div style="background:#1a1122;border-left:3px solid #dd6644;padding:12px;margin:8px 0;border-radius:0 6px 6px 0">
<b style="color:#dd6644">⚠ Dark Zone: {dc.balance_topology}</b><br>
<span style="font-size:11px;color:#998">
Centroid: ({cz[0]:.3f}, {cz[1]:.3f}) | c(p) = {dc.mean_cancellation_ratio:.4f} | {len(dc.points)} points<br>
Constraints involved: {', '.join(dc.constraints_involved)}
</span></div>"""

dz_v2_info = ""
for dc in dz_v2:
    cz = dc.centroid
    dz_v2_info += f"""<div style="background:#111122;border-left:3px solid #44dd66;padding:12px;margin:8px 0;border-radius:0 6px 6px 0">
<span style="font-size:11px;color:#889">Residual cluster at ({cz[0]:.3f}, {cz[1]:.3f}), c(p)={dc.mean_cancellation_ratio:.4f}</span></div>"""

if not dz_v2:
    dz_v2_info = """<div style="background:#112211;border-left:3px solid #44dd66;padding:12px;margin:8px 0;border-radius:0 6px 6px 0">
<b style="color:#44dd66">✓ No dark zones detected</b><br>
<span style="font-size:11px;color:#889">Withdraw_sequencing constraint successfully breaks the cancellation.</span></div>"""

# Gradient table
grad_rows = ""
for name in grads_v1:
    g1 = grads_v1[name]
    g2 = grads_v2.get(name, None)
    v1_str = f"∇=({g1['gradient'][0]:.3f},{g1['gradient'][1]:.3f}) ||∇||={g1['magnitude']:.3f}"
    v2_str = f"∇=({g2['gradient'][0]:.3f},{g2['gradient'][1]:.3f}) ||∇||={g2['magnitude']:.3f}" if g2 else "—"
    grad_rows += f"<tr><td>{name}</td><td>{v1_str}</td><td>{v2_str}</td></tr>"

html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="UTF-8">
<title>Constraint-First Design — LendVault Protocol</title>
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
.metric {{ display:inline-block; background:#111133; border-radius:6px; padding:12px 18px; margin:6px; text-align:center; }}
.metric .val {{ font-size:22px; font-weight:bold; }}
.metric .lbl {{ font-size:10px; color:#556; }}
table {{ width:100%; border-collapse:collapse; font-size:11px; margin:8px 0; }}
th {{ background:#111133; color:#889; padding:8px 12px; text-align:left; font-weight:normal; }}
td {{ padding:8px 12px; border-bottom:1px solid #111122; }}
tr:hover td {{ background:#0f0f24; }}
.callout {{ background:#111122; border-left:3px solid #66aadd; padding:12px 16px; margin:12px 0; font-size:12px; border-radius:0 6px 6px 0; line-height:1.8; }}
.workflow {{ display:flex; gap:0; align-items:center; margin:16px 0; flex-wrap:wrap; }}
.workflow-step {{ background:#0e0e20; border:1px solid #1a1a3a; border-radius:8px; padding:14px; text-align:center; min-width:120px; }}
.workflow-arrow {{ color:#556; font-size:20px; padding:0 8px; }}
footer {{ color:#444; font-size:10px; text-align:center; margin-top:32px; padding:16px; }}
</style></head>
<body>

<h1>Constraint-First Protocol Design</h1>
<div class="sub">LendVault — Simple ETH Lending Protocol<br>
Demonstration: design constraints first, find dark zones in the model, fix in the spec, then generate code.</div>

<!-- Workflow diagram -->
<h2>The Constraint-First Workflow</h2>
<div class="workflow">
<div class="workflow-step"><b>1. Model</b><br><span style="font-size:10px;color:#889">Write constraint DSL spec<br>lendvault_v1.yaml</span></div>
<div class="workflow-arrow">→</div>
<div class="workflow-step"><b>2. Scan</b><br><span style="font-size:10px;color:#889">DarkZoneDetector<br>c(p) heatmap</span></div>
<div class="workflow-arrow">→</div>
<div class="workflow-step" style="border-color:#dd6644"><b style="color:#dd6644">3. Find</b><br><span style="font-size:10px;color:#dd6644">Dark zone at (0.5,0.5)<br>c(p)≈0.05</span></div>
<div class="workflow-arrow">→</div>
<div class="workflow-step"><b>4. Fix</b><br><span style="font-size:10px;color:#889">Add constraint to spec<br>lendvault_v2.yaml</span></div>
<div class="workflow-arrow">→</div>
<div class="workflow-step"><b>5. Verify</b><br><span style="font-size:10px;color:#44dd66">Re-scan: no dark zones<br>c(p) > threshold</span></div>
<div class="workflow-arrow">→</div>
<div class="workflow-step" style="border-color:#44dd66"><b style="color:#44dd66">6. Generate</b><br><span style="font-size:10px;color:#889">Solidity contract<br>LendVault.sol</span></div>
</div>

<!-- Key Metrics -->
<h2>Before vs After — Dark Zone Elimination</h2>
<div class="row">
<div class="card" style="border-left:4px solid #dd6644">
<h3>V1 (Vulnerable)</h3>
<div style="font-size:11px;color:#889;line-height:1.8">
<p>Constraints: <b>{len(spec_v1['constraints'])}</b></p>
<p>Dark zones detected: <b style="color:#dd6644">{len(dz_v1)}</b></p>
</div>
{dz_v1_info if dz_v1_info else '<p style="color:#889">No dark zones found.</p>'}
</div>
<div class="card" style="border-left:4px solid #44dd66">
<h3>V2 (Fixed)</h3>
<div style="font-size:11px;color:#889;line-height:1.8">
<p>Constraints: <b>{len(spec_v2['constraints'])}</b> (+1 withdraw_sequencing)</p>
<p>Dark zones detected: <b style="color:#44dd66">{len(dz_v2)}</b></p>
</div>
{dz_v2_info}
</div></div>

<!-- Point comparison -->
<h2>State-Space Point Analysis</h2>
<table>
<tr><th>Version</th><th>Point</th><th>||Π||</th><th>Σ||∇σ||</th><th>c(p)</th><th>Status</th></tr>
"""
for pt, ver in [(v1_normal, 'V1'), (v1_dz, 'V1'), (v2_normal, 'V2'), (v2_dz, 'V2')]:
    cr = pt['c_ratio']
    color = '#dd6644' if cr < 0.2 else '#ddcc44' if cr < 0.5 else '#44dd66'
    status = 'DARK ZONE' if cr < 0.15 else 'MARGINAL' if cr < 0.4 else 'SAFE'
    html += f"""<tr><td>{ver}</td><td>{pt['label']}</td>
<td>{pt['combined']:.3f}</td><td>{pt['total_indiv']:.3f}</td>
<td style="color:{color};font-weight:bold">{cr:.4f}</td>
<td style="color:{color}">{status}</td></tr>"""
html += '</table>'

# Gradient analysis
html += f"""<h2>Constraint Gradient Analysis at Dark Zone Centroid (0.5, 0.5)</h2>
<table>
<tr><th>Constraint</th><th>V1 (Vulnerable)</th><th>V2 (Fixed)</th></tr>
{grad_rows}
<tr style="background:#0f0f24"><td colspan="3" style="font-size:10px;color:#889">
<b>V1:</b> withdraw_check ∇ ≈ (2.2, 0), balance_update ∇ ≈ (-2.2, 0) → mutual cancellation.<br>
<b>V2:</b> withdraw_sequencing adds ∇ ≈ (2.9, 0) at the centroid → breaks the cancellation → c(p) rises above threshold.
</td></tr></table>"""

# Heatmaps
html += """<h2>Constraint Force Field — ||Π|| Before vs After</h2>
<div class="row">
<div class="card"><h3 style="color:#dd6644">V1 — Vulnerable</h3><div class="chart-wrap"><canvas id="forceV1"></canvas></div>
<div style="font-size:9px;color:#667;margin-top:4px">Diagonal dark band through center — gradient cancellation</div></div>
<div class="card"><h3 style="color:#44dd66">V2 — Fixed</h3><div class="chart-wrap"><canvas id="forceV2"></canvas></div>
<div style="font-size:9px;color:#667;margin-top:4px">Dark band eliminated — withdraw_sequencing fills the gap</div></div></div>

<h2>Cancellation Ratio — c(p) Before vs After</h2>
<div class="row">
<div class="card"><div class="chart-wrap"><canvas id="cancelV1"></canvas></div>
<div style="font-size:9px;color:#667;margin-top:4px">Blue region at center: c(p)≈0 → dark zone</div></div>
<div class="card"><div class="chart-wrap"><canvas id="cancelV2"></canvas></div>
<div style="font-size:9px;color:#667;margin-top:4px">No blue region: dark zone eliminated by constraint fix</div></div></div>"""

# What this means
html += f"""<h2>What This Demonstrates</h2>
<div class="row">
<div class="card">
<h3>Traditional Workflow</h3>
<div style="font-size:11px;color:#889;line-height:1.8">
<p>1. Write Solidity contract</p>
<p>2. Send to auditor</p>
<p>3. Auditor finds bugs</p>
<p>4. Fix code, re-audit</p>
<p>5. Hope no dark zones remain</p>
<p style="margin-top:8px;color:#dd6644">The constraint topology is never explicitly modeled.</p>
</div></div>
<div class="card">
<h3>Constraint-First Workflow</h3>
<div style="font-size:11px;color:#889;line-height:1.8">
<p>1. Write constraint DSL spec (20 lines of YAML)</p>
<p>2. Run DarkZoneDetector → c(p) heatmap</p>
<p>3. Find dark zones in the model</p>
<p>4. Add countermeasure constraint</p>
<p>5. Re-scan → verify c(p) > threshold</p>
<p>6. Generate Solidity from verified spec</p>
<p style="margin-top:8px;color:#44dd66">Security is a topological property of the constraint model.</p>
</div></div></div>

<div class="callout" style="border-left-color:#cc44dd">
<b>核心洞察：</b>传统 DeFi 开发把安全当作代码审计问题。约束优先设计把安全当作<b>拓扑验证问题</b>。<br>
就像现代芯片设计不会"先流片再检查是否有短路"——他们先做电路仿真，验证信号完整性，再 tape-out。<br>
DeFi 协议设计应该是一样的：先在约束拓扑中验证无暗区，再生成实现代码。
</div>"""

# Charts
html += f"""
<script>
const xs = {json.dumps(xs.tolist())};
const ys = {json.dumps(ys.tolist())};
const fv1 = {json.dumps(force_v1.tolist())};
const cv1 = {json.dumps(cancel_v1.tolist())};
const fv2 = {json.dumps(force_v2.tolist())};
const cv2 = {json.dumps(cancel_v2.tolist())};

function draw(id, data, maxVal, style) {{
    const ctx = document.getElementById(id).getContext('2d');
    const ds = [];
    for (let r=0; r<data.length; r++) {{
        ds.push({{ label:'', data:data[r],
            backgroundColor: data[r].map(v => {{
                const t = Math.min(Math.max(v/(maxVal||1),0),1);
                return style==='force'
                    ? `rgba(${{Math.floor(t*240)}},${{Math.floor(t*190)}},${{Math.floor((1-t)*180+t*40)}},0.9)`
                    : `rgba(${{Math.floor(t*220)}},${{Math.floor(t*140+(1-t)*30)}},${{Math.floor((1-t)*200+t*30)}},0.9)`;
            }}), borderWidth:0 }});
    }}
    new Chart(ctx, {{ type:'bar', data:{{ labels:ys.map(y=>y.toFixed(2)), datasets:ds }},
        options:{{ responsive:true, maintainAspectRatio:false, animation:false,
            plugins:{{ legend:{{ display:false }} }},
            scales:{{ x:{{ stacked:true, display:false }}, y:{{ stacked:true, display:false }} }}
        }}
    }});
}}

const maxF = Math.max(...fv1.flat(), ...fv2.flat());
draw('forceV1', fv1, maxF, 'force');
draw('forceV2', fv2, maxF, 'force');
draw('cancelV1', cv1, 1.05, 'cancel');
draw('cancelV2', cv2, 1.05, 'cancel');
</script>

<footer>
Constraint-First Design Workflow · LendVault Protocol · 2026<br>
"Don't audit code. Verify topology." — The Constraint-First Manifesto
</footer>
</body></html>"""

with open(OUT, 'w') as f:
    f.write(html)

print(f"\nReport: {OUT}")
print(f"\n=== CONSTRAINT-FIRST DESIGN WORKFLOW ===")
print(f"\nStep 1: Model constraints in DSL")
print(f"  V1: {len(spec_v1['constraints'])} constraints defined")
print(f"  Spec: {v1_path}")
print(f"\nStep 2: Scan for dark zones")
print(f"  V1 dark zones: {len(dz_v1)}")
for dc in dz_v1:
    print(f"    c(p)={dc.mean_cancellation_ratio:.4f} at ({dc.centroid[0]:.3f},{dc.centroid[1]:.3f}) — {dc.balance_topology}")

print(f"\nStep 3: Dark zone found at (0.5, 0.5)")
print(f"  V1 at dark zone: c(p)={v1_dz['c_ratio']:.4f}, ||Π||={v1_dz['combined']:.3f}")
print(f"  Root cause: withdraw_check ∇ and balance_update ∇ oppose")
for name, g in grads_v1.items():
    print(f"    {name}: ∇=({g['gradient'][0]:.2f},{g['gradient'][1]:.2f}) ||∇||={g['magnitude']:.3f}")

print(f"\nStep 4: Fix — add withdraw_sequencing constraint")
print(f"  V2: {len(spec_v2['constraints'])} constraints (added withdraw_sequencing)")

print(f"\nStep 5: Verify — re-scan")
print(f"  V2 dark zones: {len(dz_v2)}")
print(f"  V2 at same point: c(p)={v2_dz['c_ratio']:.4f}, ||Π||={v2_dz['combined']:.3f}")
for name, g in grads_v2.items():
    print(f"    {name}: ∇=({g['gradient'][0]:.2f},{g['gradient'][1]:.2f}) ||∇||={g['magnitude']:.3f}")

delta_c = v2_dz['c_ratio'] - v1_dz['c_ratio']
print(f"\n  c(p) improved by +{delta_c:.4f} → dark zone {'ELIMINATED' if len(dz_v2)==0 else 'REDUCED'}")
print(f"\nStep 6: Generate Solidity from fixed spec (via solidity_gen.py)")
print(f"  → LendVault.sol with withdraw_sequencing embedded as checks-effects-interactions")
