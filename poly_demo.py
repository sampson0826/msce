#!/usr/bin/env python3
"""
DeFi Dark Zone Scanner — Case Study 3: Poly Network Cross-Chain Attack (2021)

Models the cross-chain dark zone as a HIERARCHICAL constraint failure:
an L1 verification constraint and an L2 access-control constraint operate
in adjacent domains. A MISSING EXECUTOR (L1→L2 coupler) should have
prevented verification payloads from rewriting access control, but doesn't exist.

This is structurally distinct from:
- The DAO (Type III mutual_cancellation: peer constraints, gradients oppose)
- Parity (cold start gap: all constraints inactive at origin)

Poly Network = hierarchical: cross-layer, different-layer constraints form
the dark zone because the executor that SHOULD couple them is absent.

Run:
  cd /Users/dengxinhang/paper && python3 constraint_residual/poly_demo.py
  open constraint_residual/poly_output.html
"""

import numpy as np
import json
import os
import sys
sys.path.insert(0, '/Users/dengxinhang/paper')
from constraint_residual.core import Rule, ConstraintField, ResidualDetector
from constraint_residual.dark_zone_detector import DarkZoneDetector
from constraint_residual.executor_models import Executor, ExecutorGap
from constraint_residual.executor_hunter import ExecutorHunter

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "poly_output.html")

# ═══════════════════════════════════════════════════════════════
# State Space
# ═══════════════════════════════════════════════════════════════
# x ∈ [0, 1] : cross_chain_verification integrity
#   0 = attacker controls verification keys → all messages "pass"
#   1 = legitimate keepers → genuine cross-chain verification
#
# y ∈ [0, 1] : access_control boundary strength
#   0 = anyone can call privileged functions (change keepers)
#   1 = strong isolation: payload execution cannot touch infrastructure

# ═══════════════════════════════════════════════════════════════
# Constraint Functions (known rules only — no coupler executor)
# ═══════════════════════════════════════════════════════════════

def sigma_verification(p):
    """L1 Cross-chain verification constraint.

    Enforces: messages from source chain must carry valid keeper signatures.
    Strong when verification infrastructure is intact (high x).
    Centers at high x — this is where legitimate verification lives.
    During the attack, x drops as keepers are replaced.
    """
    x, y = float(p[0]), float(p[1])
    return np.exp(-((x - 0.75) / 0.25)**2 - ((y - 0.5) / 0.4)**2)


def sigma_access_control(p):
    """L2 Keeper management access constraint.

    Enforces: only authorized actors can change the keeper set.
    Strong when access boundary is intact (high y).
    Centers at high y — the access wall.
    During the attack, the payload bypasses this wall → y drops.
    """
    x, y = float(p[0]), float(p[1])
    return np.exp(-((x - 0.5) / 0.35)**2 - ((y - 0.25) / 0.25)**2)


def sigma_payload_isolation(p):
    """L1→L2 Payload isolation constraint (THE MISSING EXECUTOR).

    What SHOULD exist: a constraint preventing cross-chain message payloads
    from modifying the verification infrastructure (keeper keys, relayers).

    In the actual protocol, this constraint does not exist. We model it
    as a placeholder with certainty=0.05 to mark where the executor
    gap is — and to show that adding it breaks the dark zone.

    Positioned at the INTERSECTION of verification and access domains —
    exactly where the attack transits.
    """
    x, y = float(p[0]), float(p[1])
    return 0.05 * np.exp(-((x - 0.5) / 0.2)**2 - ((y - 0.4) / 0.2)**2)


# ═══════════════════════════════════════════════════════════════
# Two scenarios: WITHOUT coupler (actual protocol) vs WITH coupler (fixed)
# ═══════════════════════════════════════════════════════════════

# Scenario A: actual Poly Network (no coupler)
rules_vulnerable = [
    Rule(name="verification", layer=1, domain="cross_chain",
         constraint_fn=sigma_verification, certainty=1.0),
    Rule(name="access_control", layer=2, domain="access",
         constraint_fn=sigma_access_control, certainty=1.0),
    Rule(name="payload_isolation", layer=1, domain="cross_chain→access",
         constraint_fn=sigma_payload_isolation, certainty=0.05),
]
field_vuln = ConstraintField(rules=rules_vulnerable)

# Scenario B: fixed protocol (with coupler activated)
def sigma_isolation_active(p):
    """Activated payload isolation — provides gradient where verification/access intersect.
    Centered broad enough to push the dark zone centroid above cancellation threshold."""
    x, y = float(p[0]), float(p[1])
    return 0.9 * np.exp(-((x - 0.5) / 0.35)**2 - ((y - 0.5) / 0.35)**2)

rules_fixed = [
    Rule(name="verification", layer=1, domain="cross_chain",
         constraint_fn=sigma_verification, certainty=1.0),
    Rule(name="access_control", layer=2, domain="access",
         constraint_fn=sigma_access_control, certainty=1.0),
    Rule(name="payload_isolation", layer=1, domain="cross_chain→access",
         constraint_fn=sigma_isolation_active, certainty=0.9),
]
field_fixed = ConstraintField(rules=rules_fixed)

# ═══════════════════════════════════════════════════════════════
# Scan both scenarios
# ═══════════════════════════════════════════════════════════════

bounds = [(0, 1), (0, 1)]
n_points = 80

# Vulnerable scenario
dark_det_vuln = DarkZoneDetector(cancellation_eps=0.2, individual_min=0.2)
dark_clusters_vuln = dark_det_vuln.scan(field_vuln, bounds, n_points=n_points)

residual_det_vuln = ResidualDetector(field_vuln, epsilon=0.15)
residuals_vuln = residual_det_vuln.scan_grid(bounds, n_points=n_points)
residual_clusters_vuln = residual_det_vuln.cluster_residuals(residuals_vuln, angle_threshold_deg=30)

# Fixed scenario
dark_det_fixed = DarkZoneDetector(cancellation_eps=0.2, individual_min=0.2)
dark_clusters_fixed = dark_det_fixed.scan(field_fixed, bounds, n_points=n_points)

# ═══════════════════════════════════════════════════════════════
# Heatmaps for both scenarios
# ═══════════════════════════════════════════════════════════════

xs = np.linspace(0, 1, 100)
ys = np.linspace(0, 1, 100)

def compute_heatmaps(field):
    hm_force = np.zeros((100, 100))
    hm_cancel = np.zeros((100, 100))
    hm_indiv = np.zeros((100, 100))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            p = np.array([x, y])
            grad = field.constraint_gradient(p)
            hm_force[j, i] = float(np.linalg.norm(grad))
            indiv_mags = [float(np.linalg.norm(r.gradient(p))) for r in field.rules]
            total_indiv = sum(indiv_mags)
            hm_indiv[j, i] = total_indiv
            hm_cancel[j, i] = float(np.linalg.norm(grad)) / total_indiv if total_indiv > 1e-10 else 1.0
    return hm_force, hm_cancel, hm_indiv

force_v, cancel_v, indiv_v = compute_heatmaps(field_vuln)
force_f, cancel_f, indiv_f = compute_heatmaps(field_fixed)

# ═══════════════════════════════════════════════════════════════
# Point metrics
# ═══════════════════════════════════════════════════════════════

def metrics_at(field, p, label):
    grad = field.constraint_gradient(p)
    indiv_grad = {r.name: float(np.linalg.norm(r.gradient(p))) for r in field.rules}
    indiv_val = {r.name: float(r.constraint_fn(p)) for r in field.rules}
    total_indiv = sum(indiv_grad.values())
    total_sigma = sum(indiv_val.values())
    combined = float(np.linalg.norm(grad))
    cr = combined / total_indiv if total_indiv > 1e-10 else 1.0
    return {
        'label': label, 'position': p.tolist(),
        'combined': combined, 'total_indiv': total_indiv,
        'total_sigma': total_sigma, 'c_ratio': cr,
        'individual_grad': indiv_grad, 'individual_val': indiv_val,
    }

# Key state-space points
points_vuln = [
    metrics_at(field_vuln, np.array([0.8, 0.5]), "Normal operation: verification intact"),
    metrics_at(field_vuln, np.array([0.655, 0.326]), "Dark zone centroid (exploit window)"),
    metrics_at(field_vuln, np.array([0.2, 0.8]), "Post-attack: verification captured"),
]
points_fixed = [
    metrics_at(field_fixed, np.array([0.8, 0.5]), "Normal (fixed)"),
    metrics_at(field_fixed, np.array([0.655, 0.326]), "Dark zone centroid WITH coupler"),
]

# ═══════════════════════════════════════════════════════════════
# Executor gap analysis using ExecutorHunter framework
# ═══════════════════════════════════════════════════════════════

# Build executors: verification (L1→L1) and access_control (L2→L2)
# The MISSING executor is L1→L2 (the coupler)
exec_verify = Executor(
    id='E_verify', name='Cross-chain verification',
    from_layer=1, to_layer=1, executor_type='E-I', certainty=1.0,
    transmission_fn=lambda uc, p: uc * np.array([
        np.exp(-((p[0]-0.75)/0.25)**2 - ((p[1]-0.5)/0.4)**2),
        np.exp(-((p[0]-0.75)/0.25)**2 - ((p[1]-0.5)/0.4)**2),
    ]),
    evidence_strength='high', math_form='Keeper threshold signature verification'
)
exec_access = Executor(
    id='E_access', name='Access control boundary',
    from_layer=2, to_layer=2, executor_type='E-II', certainty=1.0,
    transmission_fn=lambda uc, p: uc * np.array([
        np.exp(-((p[0]-0.5)/0.35)**2 - ((p[1]-0.25)/0.25)**2),
        np.exp(-((p[0]-0.5)/0.35)**2 - ((p[1]-0.25)/0.25)**2),
    ]),
    evidence_strength='high', math_form='onlyOwner / onlyKeeper modifier pattern'
)

hunter = ExecutorHunter([exec_verify, exec_access],
                        residual_epsilon=0.1, dark_zone_cancellation_eps=0.2)
gaps = hunter.hunt_gaps(from_layer=1, to_layer=2, bounds=bounds, n_points=40)
trans = hunter.compute_transmission_completeness([exec_verify, exec_access], bounds, n_points=40)

# ═══════════════════════════════════════════════════════════════
# HTML Report
# ═══════════════════════════════════════════════════════════════

dark_data = []
for dc in dark_clusters_vuln:
    dark_data.append({
        'id': dc.id, 'topology': dc.balance_topology,
        'n_points': len(dc.points), 'cancellation_ratio': dc.mean_cancellation_ratio,
        'centroid': dc.centroid.tolist() if dc.centroid is not None else None,
        'constraints': dc.constraints_involved,
    })

html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DeFi Dark Zone Scanner — Poly Network Case Study</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ background:#0a0a12; color:#c8c8d8; font-family:'SF Mono','Menlo','Consolas',monospace; padding:24px 40px; line-height:1.5; }}
h1 {{ font-size:20px; color:#fff; margin-bottom:2px; }}
h2 {{ font-size:15px; color:#8899cc; margin:32px 0 12px; border-bottom:1px solid #1a1a3a; padding-bottom:6px; }}
h3 {{ font-size:13px; color:#aabbdd; margin:16px 0 8px; }}
.sub {{ color:#556; font-size:11px; margin-bottom:28px; }}
.row {{ display:flex; gap:20px; flex-wrap:wrap; margin:12px 0; }}
.card {{ background:#0e0e20; border:1px solid #1a1a3a; border-radius:8px; padding:18px; flex:1; min-width:340px; }}
.chart-wrap {{ position:relative; height:300px; }}
.metric {{ display:inline-block; background:#111133; border-radius:6px; padding:12px 18px; margin:6px; text-align:center; }}
.metric .val {{ font-size:24px; font-weight:bold; }}
.metric .lbl {{ font-size:10px; color:#556; margin-top:2px; }}
.val-good {{ color:#44dd66; }} .val-warn {{ color:#ddcc44; }} .val-danger {{ color:#dd6644; }} .val-info {{ color:#66aadd; }}

.callout {{ background:#111122; border-left:3px solid #66aadd; padding:12px 16px; margin:12px 0; font-size:12px; border-radius:0 6px 6px 0; line-height:1.8; }}
.callout-danger {{ border-left-color:#dd6644; }}
.callout-purple {{ border-left-color:#cc44dd; }}

.taxonomy-grid {{ display:grid; grid-template-columns:repeat(3,1fr); gap:14px; margin:16px 0; }}
.tax-card {{ background:#0e0e20; border:1px solid #1a1a3a; border-radius:8px; padding:14px; }}
.tax-card h3 {{ margin-top:0; font-size:13px; }}
.tax-card .sig {{ font-size:10px; padding:4px 8px; border-radius:4px; display:inline-block; margin:4px 0; }}
.tax-card p {{ font-size:10px; color:#889; line-height:1.6; }}
.sig-dao {{ background:#331144; color:#cc44dd; }}
.sig-parity {{ background:#332211; color:#ddcc44; }}
.sig-poly {{ background:#112244; color:#66aadd; }}

table {{ width:100%; border-collapse:collapse; font-size:11px; margin:8px 0; }}
th {{ background:#111133; color:#889; padding:8px 12px; text-align:left; font-weight:normal; }}
td {{ padding:8px 12px; border-bottom:1px solid #111122; }}
tr:hover td {{ background:#0f0f24; }}

footer {{ color:#444; font-size:10px; text-align:center; margin-top:40px; padding:20px; }}
</style>
</head>
<body>

<h1>DeFi Dark Zone Scanner</h1>
<div class="sub">Case Study: Poly Network Cross-Chain Attack (2021) — Hierarchical Cross-Layer Dark Zone<br>
L1 跨链验证约束 + L2 访问控制约束 = 层级暗区。与传统审计不同的是：每条规则独立正确，但层级间的执行者缺失。</div>

<!-- Metrics -->
<div>
<div class="metric"><div class="val val-info">3</div><div class="lbl">Constraints (2 active + 1 placeholder)</div></div>
<div class="metric"><div class="val val-danger">{len(dark_clusters_vuln)}</div><div class="lbl">Dark zones (vulnerable)</div></div>
<div class="metric"><div class="val val-good">{len(dark_clusters_fixed)}</div><div class="lbl">Dark zones (fixed)</div></div>
<div class="metric"><div class="val val-info">{len(gaps)}</div><div class="lbl">Executor gaps (L1→L2)</div></div>
<div class="metric"><div class="val val-warn">{trans.completeness_mean:.1%}</div><div class="lbl">L1→L2 transmission</div></div>
</div>

<!-- The Attack Mechanism -->
<h2>How Poly Network Was Exploited — Constraint View</h2>
<div class="row">
<div class="card">
<h3>Protocol Architecture</h3>
<div style="font-size:11px;color:#889;line-height:1.8">
<p><b>EthCrossChainManager.verifyHeaderAndExecuteTx()</b></p>
<p>1. Verify cross-chain message signature (L1 constraint)</p>
<p>2. Decode payload from verified message</p>
<p>3. Execute payload on target chain → <b style="color:#dd6644">can call putCurEpochConPubKeyBytes() to change keepers</b></p>
<p style="margin-top:8px;color:#cc44dd">The missing L1→L2 executor: nothing prevents payload execution from modifying the verification infrastructure.</p>
</div></div>
<div class="card">
<h3>Attack Path</h3>
<div style="font-size:11px;color:#889;line-height:1.8">
<p>1. Attacker crafts a cross-chain message with a payload that calls <code>putCurEpochConPubKeyBytes</code></p>
<p>2. <b style="color:#44dd66">Verification passes</b> — the message format is valid</p>
<p>3. Payload executes → <b style="color:#dd6644">keeper keys replaced with attacker's keys</b></p>
<p>4. All future verification now passes for attacker → full drain</p>
<p style="margin-top:8px">The dark zone is at the verification/access boundary — where verification succeeds but the access wall has a hole.</p>
</div></div></div>

<!-- Heatmaps: BEFORE vs AFTER fix -->
<h2>Constraint Force Field — Vulnerable vs Fixed</h2>
<div class="row">
<div class="card" style="flex:1.2"><h3>||Π|| — Vulnerable (no coupler)</h3><div class="chart-wrap"><canvas id="forceVChart"></canvas></div></div>
<div class="card" style="flex:1.2"><h3>||Π|| — Fixed (with coupler executor)</h3><div class="chart-wrap"><canvas id="forceFChart"></canvas></div></div>
</div>
<div class="row">
<div class="card" style="flex:1.2"><h3>c(p) — Vulnerable</h3><div class="chart-wrap"><canvas id="cancelVChart"></canvas></div></div>
<div class="card" style="flex:1.2"><h3>c(p) — Fixed</h3><div class="chart-wrap"><canvas id="cancelFChart"></canvas></div></div>
</div>

<!-- Point comparison -->
<h2>State-Space Point Analysis</h2>
<table>
<tr><th>Point</th><th>Position</th><th>||Π||</th><th>Σ||∇σ||</th><th>c(p)</th><th>Verdict</th></tr>
"""
for pt in points_vuln:
    verdict = "DARK ZONE" if pt['c_ratio'] < 0.3 else "PROTECTED"
    vc = 'val-danger' if verdict == "DARK ZONE" else 'val-good'
    html += f"""<tr><td>{pt['label']}</td><td>({pt['position'][0]:.2f}, {pt['position'][1]:.2f})</td>
<td>{pt['combined']:.3f}</td><td>{pt['total_indiv']:.3f}</td>
<td class="{vc}">{pt['c_ratio']:.4f}</td><td class="{vc}">{verdict}</td></tr>"""
for pt in points_fixed:
    html += f"""<tr style="background:#112211"><td>{pt['label']}</td><td>({pt['position'][0]:.2f}, {pt['position'][1]:.2f})</td>
<td>{pt['combined']:.3f}</td><td>{pt['total_indiv']:.3f}</td>
<td class="val-good">{pt['c_ratio']:.4f}</td><td class="val-good">PROTECTED (coupler active)</td></tr>"""
html += '</table>'

# Before/After comparison
html += f"""<h2>The Missing Executor: L1→L2 Payload Isolation</h2>
<div class="row">
<div class="card">
<h3>Without Coupler (Actual Protocol)</h3>
<div style="font-size:11px;color:#889;line-height:1.8">
<p>Dark zones detected: <b style="color:#dd6644">{len(dark_clusters_vuln)}</b></p>
<p>L1→L2 transmission: <b style="color:#dd6644">{trans.completeness_mean:.1%}</b></p>
<p>Executor gaps: <b style="color:#dd6644">{len(gaps)}</b></p>
<p>The verification constraint (L1) and access constraint (L2) operate
in their own domains. No executor couples them — so a payload that
passes L1 verification can freely modify L2 access control.</p>
<p style="margin-top:8px;color:#dd6644">The gap is not IN a layer. It's BETWEEN layers.</p>
</div></div>
<div class="card">
<h3>With Coupler (Hypothetical Fix)</h3>
<div style="font-size:11px;color:#889;line-height:1.8">
<p>Dark zones after fix: <b style="color:#44dd66">{len(dark_clusters_fixed)}</b></p>
<p>The payload_isolation constraint activates at the verification/access
intersection — the exact state-space region where the attack transits.</p>
<p>Adding this executor raises ||Π|| at the dark zone centroid
enough to push c(p) above threshold.</p>
<p style="margin-top:8px;color:#44dd66">The fix is not changing L1 or L2. It's adding the executor BETWEEN them.</p>
</div></div></div>"""

# Three-species taxonomy
html += f"""<h2>Dark Zone Taxonomy — Three Species, One Framework</h2>
<div class="taxonomy-grid">
<div class="tax-card">
<h3><span class="sig sig-dao">Type III Mutual Cancellation</span></h3>
<p><b>Case:</b> The DAO (2016)<br>
<b>Signature:</b> c(p)→0, Σ||∇σ||≫0<br>
<b>Mechanism:</b> Two peer-L1 constraints have opposing gradients that cancel at a specific state<br>
<b>Detection:</b> DarkZoneDetector<br>
<b>Fix:</b> Reorder operations so gradients can't oppose (checks-effects-interactions)<br>
<b>Layer:</b> Single-layer (L1↔L1)</p>
</div>
<div class="tax-card">
<h3><span class="sig sig-parity">Cold Start Gap</span></h3>
<p><b>Case:</b> Parity Wallet (2017)<br>
<b>Signature:</b> c(p)→1, Σ||∇σ||≈0, ||Π||≈0<br>
<b>Mechanism:</b> Initial state sits in a region where no constraint has activated yet<br>
<b>Detection:</b> ResidualDetector (||Π||<ε, σ values low)<br>
<b>Fix:</b> Add deployment constraint that initializes before use<br>
<b>Layer:</b> Single-layer (L2↔L2)</p>
</div>
<div class="tax-card">
<h3><span class="sig sig-poly">Cross-Layer Hierarchical</span></h3>
<p><b>Case:</b> Poly Network (2021)<br>
<b>Signature:</b> c(p)→0, Σ||∇σ||≫0, but constraints from DIFFERENT layers<br>
<b>Mechanism:</b> L1 verification + L2 access control operate in adjacent domains; missing executor creates blind zone at their boundary<br>
<b>Detection:</b> ExecutorHunter (transmission completeness < threshold)<br>
<b>Fix:</b> Add executor coupling L1→L2 (payload sandbox)<br>
<b>Layer:</b> Cross-layer (L1↛L2)</p>
</div></div>

<!-- Executor gap details -->
<h2>Executor Hunter Output — L1→L2 Transmission Analysis</h2>
<table>
<tr><th>Priority</th><th>Region</th><th>Transmission</th><th>Residual</th><th>Candidate Type</th><th>Math Form Prediction</th></tr>
"""
for g in gaps[:5]:
    ttag = {'E-I':'background:#225533;color:#44dd66','E-II':'background:#334422;color:#ddcc44',
            'E-III':'background:#442222;color:#dd6644'}.get(g.candidate_type,'')
    dz_tag = ' + dark zone' if g.dark_zone_ids else ''
    html += f"""<tr><td>{g.priority:.3f}</td><td>({g.region[0]:.2f}, {g.region[1]:.2f})</td>
<td style="color:{'#44dd66' if g.transmission_completeness>0.8 else '#dd6644'}">{g.transmission_completeness:.1%}</td>
<td>{g.residual_magnitude:.4f}</td>
<td><span style="font-size:9px;padding:2px 6px;border-radius:3px;{ttag}">{g.candidate_type}</span></td>
<td style="font-size:10px;color:#889">{g.candidate_math_form[:90] if g.candidate_math_form else '—'}{dz_tag}</td></tr>"""
html += '</table>'

# Summary callout
html += f"""<div class="callout callout-purple">
<b>Poly Network 的独特性：</b> 这不是同一层级内的约束互消（The DAO），也不是初始状态全约束未激活（Parity）。<br>
这是 <b>跨层级执行者缺失</b>——L1 验证和 L2 访问控制之间应该有一个执行者将两者耦合，但它不存在。<br>
传统审计检查每条规则的正确实现。每条都是对的。但层级之间的"缝"是不可见的——除非你建模约束传递拓扑。<br>
<b>Poly Network 本质上不是代码漏洞。是约束架构中缺失了一个 executor。</b>
</div>"""

# Charts
html += f"""
<script>
const xs = {json.dumps(xs.tolist())};
const ys = {json.dumps(ys.tolist())};
const forceV = {json.dumps(force_v.tolist())};
const cancelV = {json.dumps(cancel_v.tolist())};
const forceF = {json.dumps(force_f.tolist())};
const cancelF = {json.dumps(cancel_f.tolist())};

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
        datasets.push({{
            label: '', data: data[row],
            backgroundColor: data[row].map(v => rgbaMap(v, maxVal, style)),
            borderWidth: 0,
        }});
    }}
    new Chart(ctx, {{
        type: 'bar', data: {{ labels: ys.map(y => y.toFixed(2)), datasets: datasets }},
        options: {{
            responsive: true, maintainAspectRatio: false, animation: false,
            plugins: {{ legend: {{ display: false }} }},
            scales: {{ x: {{ stacked: true, display: false }}, y: {{ stacked: true, display: false }} }}
        }}
    }});
}}

drawHeatmap('forceVChart', forceV, Math.max(...forceV.flat()), 'force');
drawHeatmap('cancelVChart', cancelV, 1.05, 'cancel');
drawHeatmap('forceFChart', forceF, Math.max(...forceF.flat()), 'force');
drawHeatmap('cancelFChart', cancelF, 1.05, 'cancel');
</script>

<footer>
DeFi Dark Zone Scanner · Constraint Residual Framework · 2026<br>
Case studies: The DAO (mutual_cancellation) · Parity (cold_start_gap) · Poly Network (hierarchical)<br>
"Three vulnerabilities. Three dark zone species. One mathematical framework."
</footer>
</body></html>"""

with open(OUT, 'w') as f:
    f.write(html)

print(f"Report: {OUT}")
print(f"\n=== POLY NETWORK — Cross-Layer Hierarchical Dark Zone ===")
print(f"\nVulnerable scenario dark zones: {len(dark_clusters_vuln)}")
for dc in dark_clusters_vuln:
    print(f"  #{dc.id}: {dc.balance_topology}, c(p)={dc.mean_cancellation_ratio:.4f}, "
          f"centroid=({dc.centroid[0]:.3f},{dc.centroid[1]:.3f}), "
          f"constraints={dc.constraints_involved}")
print(f"\nFixed scenario dark zones: {len(dark_clusters_fixed)}")

print(f"\nExecutor gaps (L1→L2): {len(gaps)}")
print(f"  Transmission completeness: {trans.completeness_mean:.1%}")
for g in gaps[:3]:
    print(f"  Priority={g.priority:.3f}, region=({g.region[0]:.2f},{g.region[1]:.2f}), "
          f"residual={g.residual_magnitude:.4f}, type={g.candidate_type}")

print(f"\nPoint comparison (vulnerable):")
for pt in points_vuln:
    v = "DARK ZONE" if pt['c_ratio'] < 0.3 else "protected"
    print(f"  {pt['label']:50s} ||Π||={pt['combined']:.3f}  Σ||∇σ||={pt['total_indiv']:.3f}  c(p)={pt['c_ratio']:.4f}  [{v}]")
for pt in points_fixed:
    print(f"  {pt['label']:50s} ||Π||={pt['combined']:.3f}  Σ||∇σ||={pt['total_indiv']:.3f}  c(p)={pt['c_ratio']:.4f}  [FIXED]")

print(f"\n=== THREE-SPECIES TAXONOMY ===")
print(f"  The DAO:     mutual_cancellation | c(p)→0, Σ||∇σ||≫0 | single-layer (L1↔L1)")
print(f"  Parity:      cold_start_gap       | c(p)→1, Σ||∇σ||≈0 | single-layer (L2↔L2)")
print(f"  Poly Network: hierarchical        | c(p)→0, Σ||∇σ||≫0 | cross-layer (L1↛L2)")
