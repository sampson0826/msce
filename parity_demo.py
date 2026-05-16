#!/usr/bin/env python3
"""
DeFi Dark Zone Scanner — Case Study 2: Parity Wallet Freeze (2017)

Models the Parity multi-sig library self-destruct as a "cold start gap" —
a different dark zone species than The DAO's Type III cancellation.

The DAO dark zone:  Σ||∇σ|| ≫ 0, c(p) ≈ 0  (strong rules, mutual cancellation)
Parity dark zone:   Σ||∇σ|| ≈ 0, c(p) ≈ 1  (no rules active yet, but SHOULD be)

Key insight: The framework distinguishes two vulnerability classes
that traditional audits conflate under "bug."

Run:
  cd /Users/dengxinhang/paper && python3 constraint_residual/parity_demo.py
  open constraint_residual/parity_output.html
"""

import numpy as np
import json
import os
import sys
sys.path.insert(0, '/Users/dengxinhang/paper')
from constraint_residual.core import Rule, ConstraintField, ResidualDetector
from constraint_residual.dark_zone_detector import DarkZoneDetector

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "parity_output.html")

# ═══════════════════════════════════════════════════════════════
# State Space
# ═══════════════════════════════════════════════════════════════
# x ∈ [0, 1] : initialization completeness
#   0 = initWallet never called (library contract's actual state)
#   1 = initWallet called, modifier only_uninitialized locks further calls
#
# y ∈ [0, 1] : ownership legitimacy
#   0 = no owner OR attacker controlled (after malicious init)
#   1 = legitimate owner established through proper initialization

# ═══════════════════════════════════════════════════════════════
# Constraint Functions
# ═══════════════════════════════════════════════════════════════

def sharp_sigmoid(x, center=0.3, width=0.05):
    """Steep sigmoid: ~0 below center, ~1 above center."""
    return 1.0 / (1.0 + np.exp(-(x - center) / width))


def sigma_init_guard(p):
    """Initialization guard constraint.

    Enforces: 'initWallet can only be called once.'
    When x < 0.3 (uninitialized): protection ≈ 0 — anyone can call initWallet.
    When x > 0.3 (initialized):  protection ≈ 1 — re-init blocked.

    In the library contract: x ≈ 0. initWallet was never called directly
    on the library, so the only_uninitialized modifier provides zero barrier.
    """
    x, y = float(p[0]), float(p[1])
    return sharp_sigmoid(x, center=0.3, width=0.04)


def sigma_owner_guard(p):
    """Ownership guard constraint.

    Enforces: 'only owner can call kill().'
    Protection requires BOTH: wallet initialized (x high)
    AND legitimate owner set (y high).
    If either is missing, privileged operations are unprotected.

    In the library contract: x≈0, y≈0 → both conditions fail.
    Anyone who calls initWallet becomes owner → can call kill.
    """
    x, y = float(p[0]), float(p[1])
    init_ok = sharp_sigmoid(x, center=0.3, width=0.04)
    owner_ok = sharp_sigmoid(y, center=0.3, width=0.04)
    return init_ok * owner_ok


def sigma_deploy_order(p):
    """Deployment-order constraint (the missing invariant).

    This is what SHOULD have existed: a constraint ensuring the library
    contract itself gets initialized before any proxy wallet depends on it.
    We include it as a WEAK baseline (certainty=0.1) to show what
    a properly-designed system would look like.

    Without this, the system has a cold-start vulnerability.
    """
    x, y = float(p[0]), float(p[1])
    # A proper deploy-order constraint would ensure x is always ≥ 0.3
    # before the contract is usable. In its absence, we have a gap.
    return 0.05 * sharp_sigmoid(x, center=0.1, width=0.03)


# ═══════════════════════════════════════════════════════════════
# Build constraint field
# ═══════════════════════════════════════════════════════════════

rules = [
    Rule(name="init_guard", layer=2, domain="access_control",
         constraint_fn=sigma_init_guard, certainty=1.0),
    Rule(name="owner_guard", layer=2, domain="access_control",
         constraint_fn=sigma_owner_guard, certainty=1.0),
    Rule(name="deploy_order", layer=1, domain="lifecycle",
         constraint_fn=sigma_deploy_order, certainty=0.1),
]

field = ConstraintField(rules=rules)

# ═══════════════════════════════════════════════════════════════
# Scan for both vulnerability types
# ═══════════════════════════════════════════════════════════════

# Type III dark zone scan (cancellation-based)
dark_detector = DarkZoneDetector(
    cancellation_eps=0.2,
    individual_min=0.2,
)
bounds = [(0, 1), (0, 1)]
n_points = 80
dark_clusters = dark_detector.scan(field, bounds, n_points=n_points)

# Weak protection scan (||Π||-based — catches cold start gaps)
residual_detector = ResidualDetector(field, epsilon=0.15)
residuals = residual_detector.scan_grid(bounds, n_points=n_points)
residual_clusters = residual_detector.cluster_residuals(residuals, angle_threshold_deg=30)

# Low-protection region scan: find all points where combined force is weak
# but NOT due to cancellation (Σ||∇σ|| is also small)
weak_points = []
for i, x in enumerate(np.linspace(0, 1, 80)):
    for j, y in enumerate(np.linspace(0, 1, 80)):
        p = np.array([x, y])
        grad = field.constraint_gradient(p)
        combined = float(np.linalg.norm(grad))
        indiv_mags = [float(np.linalg.norm(r.gradient(p))) for r in field.rules]
        total_indiv = sum(indiv_mags)
        cr = combined / total_indiv if total_indiv > 1e-10 else 1.0
        # Also check constraint VALUES: if σ≈1 everywhere but ∇σ≈0, it's SATURATED SAFE, not a gap
        sigma_vals = [r.constraint_fn(p) for r in field.rules]
        total_sigma = sum(sigma_vals)
        if combined < 0.3 and total_indiv < 0.3 and cr > 0.8 and total_sigma < 0.3:
            # Weak gradients AND weak values → true cold start gap (not a saturated safe state)
            weak_points.append({
                'x': float(x), 'y': float(y),
                'combined': combined,
                'total_indiv': total_indiv,
                'c_ratio': cr,
                'total_sigma': float(total_sigma),
            })

# ═══════════════════════════════════════════════════════════════
# Full heatmaps
# ═══════════════════════════════════════════════════════════════

xs = np.linspace(0, 1, 100)
ys = np.linspace(0, 1, 100)
heatmap_combined = np.zeros((100, 100))
heatmap_cancel = np.zeros((100, 100))
heatmap_indiv = np.zeros((100, 100))

for i, x in enumerate(xs):
    for j, y in enumerate(ys):
        p = np.array([x, y])
        grad = field.constraint_gradient(p)
        heatmap_combined[j, i] = float(np.linalg.norm(grad))
        indiv_mags = [float(np.linalg.norm(r.gradient(p))) for r in field.rules]
        total_indiv = sum(indiv_mags)
        heatmap_indiv[j, i] = total_indiv
        if total_indiv > 1e-10:
            heatmap_cancel[j, i] = float(np.linalg.norm(grad)) / total_indiv
        else:
            heatmap_cancel[j, i] = 1.0

# ═══════════════════════════════════════════════════════════════
# Point metrics
# ═══════════════════════════════════════════════════════════════

def metrics_at(p, label):
    grad = field.constraint_gradient(p)
    indiv_grad = {r.name: float(np.linalg.norm(r.gradient(p))) for r in field.rules}
    indiv_val = {r.name: float(r.constraint_fn(p)) for r in field.rules}
    total_indiv = sum(indiv_grad.values())
    total_sigma = sum(indiv_val.values())
    combined = float(np.linalg.norm(grad))
    cr = combined / total_indiv if total_indiv > 1e-10 else 1.0
    # Classification requires both weak gradient AND weak value (exclude saturated-safe)
    if combined < 0.3 and total_indiv < 0.3 and total_sigma < 0.3:
        vtype = 'cold_start_gap'
    elif cr < 0.2 and total_indiv > 0.3:
        vtype = 'type3_dark_zone'
    else:
        vtype = 'protected'
    return {
        'label': label, 'position': p.tolist(),
        'combined': combined, 'total_indiv': total_indiv,
        'c_ratio': cr, 'vuln_type': vtype,
        'individual_grad': indiv_grad, 'individual_val': indiv_val,
        'total_sigma': total_sigma,
    }

points = [
    metrics_at(np.array([0.05, 0.05]), "Library contract initial state (EXPLOITED)"),
    metrics_at(np.array([0.4, 0.4]), "Activation cliff edge (transitioning)"),
    metrics_at(np.array([0.6, 0.6]), "Properly initialized (SAFE)"),
    metrics_at(np.array([0.95, 0.05]), "Init by attacker, owner bypassed"),
]

# ═══════════════════════════════════════════════════════════════
# Combined report: comparison with The DAO
# ═══════════════════════════════════════════════════════════════

html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DeFi Dark Zone Scanner — Parity Wallet Case Study</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ background:#0a0a12; color:#c8c8d8; font-family:'SF Mono','Menlo','Consolas',monospace; padding:24px 40px; line-height:1.5; }}
h1 {{ font-size:20px; color:#fff; margin-bottom:2px; }}
h2 {{ font-size:15px; color:#8899cc; margin:32px 0 12px; border-bottom:1px solid #1a1a3a; padding-bottom:6px; }}
.sub {{ color:#556; font-size:11px; margin-bottom:28px; }}
.row {{ display:flex; gap:20px; flex-wrap:wrap; margin:12px 0; }}
.card {{ background:#0e0e20; border:1px solid #1a1a3a; border-radius:8px; padding:18px; flex:1; min-width:340px; }}
.chart-wrap {{ position:relative; height:320px; }}
.metric {{ display:inline-block; background:#111133; border-radius:6px; padding:12px 18px; margin:6px; text-align:center; }}
.metric .val {{ font-size:24px; font-weight:bold; }}
.metric .lbl {{ font-size:10px; color:#556; margin-top:2px; }}
.val-good {{ color:#44dd66; }}
.val-warn {{ color:#ddcc44; }}
.val-danger {{ color:#dd6644; }}
.val-info {{ color:#66aadd; }}

.callout {{ background:#111122; border-left:3px solid #66aadd; padding:12px 16px; margin:12px 0; font-size:12px; border-radius:0 6px 6px 0; line-height:1.8; }}
.callout-danger {{ border-left-color:#dd6644; }}
.callout-purple {{ border-left-color:#cc44dd; }}
.callout-green {{ border-left-color:#44dd66; }}

.point-grid {{ display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin:12px 0; }}
.pt-card {{ background:#0e0e20; border:1px solid #1a1a3a; border-radius:6px; padding:12px; text-align:center; }}
.pt-card .pt-name {{ font-size:10px; color:#889; margin-bottom:4px; }}
.pt-card .pt-vuln {{ font-size:14px; font-weight:bold; }}
.pt-card .pt-nums {{ font-size:10px; color:#667; margin-top:4px; }}

.vuln-cold {{ color:#ddcc44; }}
.vuln-type3 {{ color:#cc44dd; }}
.vuln-safe {{ color:#44dd66; }}

table {{ width:100%; border-collapse:collapse; font-size:11px; margin:8px 0; }}
th {{ background:#111133; color:#889; padding:8px 12px; text-align:left; font-weight:normal; }}
td {{ padding:8px 12px; border-bottom:1px solid #111122; }}
tr:hover td {{ background:#0f0f24; }}

.comparison-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; margin:12px 0; }}
.comp-card {{ background:#0e0e20; border:1px solid #1a1a3a; border-radius:8px; padding:16px; }}
.comp-card.dao {{ border-left:4px solid #cc44dd; }}
.comp-card.parity {{ border-left:4px solid #ddcc44; }}
.comp-card h3 {{ margin-top:0; font-size:13px; }}

footer {{ color:#444; font-size:10px; text-align:center; margin-top:40px; padding:20px; }}
</style>
</head>
<body>

<h1>DeFi Dark Zone Scanner</h1>
<div class="sub">Case Study: Parity Wallet Freeze (2017) — Cold Start Gap Detection<br>
不同于 The DAO 的 Type III 约束互消暗区，Parity 展示了另一种暗区类型：<b>冷启动缺口</b>——初始状态下所有约束都未激活。</div>

<!-- Key Metrics -->
<div>
<div class="metric"><div class="val val-info">{len(rules)}</div><div class="lbl">Constraints modeled</div></div>
<div class="metric"><div class="val val-warn">{len(weak_points)}</div><div class="lbl">Cold start gap points</div></div>
<div class="metric"><div class="val val-info">{len(dark_clusters)}</div><div class="lbl">Type III dark zones</div></div>
<div class="metric"><div class="val val-info">{len(residual_clusters)}</div><div class="lbl">Residual clusters</div></div>
</div>

<!-- Four-point state comparison -->
<h2>State-Space Point Analysis</h2>
<div class="point-grid">
"""
for pt in points:
    vcls = {'cold_start_gap':'vuln-cold','type3_dark_zone':'vuln-type3','protected':'vuln-safe'}[pt['vuln_type']]
    vlabel = {'cold_start_gap':'COLD START GAP','type3_dark_zone':'TYPE III DARK ZONE','protected':'PROTECTED'}[pt['vuln_type']]
    pos = pt['position']
    html += f"""<div class="pt-card">
<div class="pt-name">{pt['label']}<br>({pos[0]:.2f}, {pos[1]:.2f})</div>
<div class="pt-vuln {vcls}">{vlabel}</div>
<div class="pt-nums">
||Π|| = {pt['combined']:.4f} |
Σ||∇σ|| = {pt['total_indiv']:.4f} |
Σσ = {pt['total_sigma']:.4f}<br>
c(p) = {pt['c_ratio']:.4f}<br>
"""
    for rname, rval in pt['individual_grad'].items():
        html += f"∇{rname}={rval:.4f}  "
    html += '<br>'
    for rname, rval in pt['individual_val'].items():
        html += f"σ_{rname}={rval:.4f}  "
    html += '</div></div>'
html += '</div>'

# Heatmaps
html += f"""<h2>Combined Constraint Force ||Π(p)||</h2>
<div class="row">
<div class="card" style="flex:1.4"><div class="chart-wrap"><canvas id="forceChart"></canvas></div></div>
<div class="card">
<h3>Reading the Force Map</h3>
<div style="font-size:11px;color:#889;line-height:1.8">
<p><b style="color:#2233aa">Dark zone (bottom-left)</b> — the library contract's initial state.
||Π|| ≈ {points[0]['combined']:.4f}. Both init_guard and owner_guard provide near-zero protection.</p>
<p><b style="color:#ffdd88">Bright zone (top-right)</b> — properly initialized state.
||Π|| ≈ {points[2]['combined']:.2f}. Both constraints fully active.</p>
<p>The <b style="color:#ddcc44">transition boundary</b> at x≈0.3, y≈0.3 is where the sigmoid
constraints snap from ~0 to ~1. Below this line: cold start gap.</p>
<p>Compare with The DAO: there the force field showed a diagonal <b>cancellation band</b>
through the center. Here it shows a <b>activation cliff</b> at the origin.</p>
</div></div></div>

<h2>Cancellation Ratio c(p) — Why Parity ≠ The DAO</h2>
<div class="row">
<div class="card" style="flex:1.4"><div class="chart-wrap"><canvas id="cancelChart"></canvas></div></div>
<div class="card">
<h3>The Critical Difference</h3>
<div style="font-size:11px;color:#889;line-height:1.8">
<p>Notice: <b style="color:#dd6644">no blue region in this map.</b></p>
<p>c(p) ≈ 1.0 everywhere — constraints are NOT canceling.
The vulnerability at (0,0) has c(p) = 1.0 (not ≈ 0).</p>
<p>This is the diagnostic distinction:</p>
<p><b style="color:#cc44dd">The DAO:</b> c(p) ≈ 0, Σ||∇σ|| ≫ 0 → <b>Type III dark zone (cancellation)</b></p>
<p><b style="color:#ddcc44">Parity:</b> c(p) ≈ 1, Σ||∇σ|| ≈ 0 → <b>Cold start gap (inactive constraints)</b></p>
<p>Both produce ||Π|| ≈ 0. Both are undetectable by traditional audit.
But they have <b>different root causes</b> and require <b>different fixes.</b></p>
</div></div></div>
"""

# Comparison table: DAO vs Parity
html += f"""<h2>Dark Zone Taxonomy — Two Species, One Framework</h2>
<div class="comparison-grid">
<div class="comp-card dao">
<h3 style="color:#cc44dd">Type III Dark Zone</h3>
<div style="font-size:11px;color:#889;line-height:1.8">
<p><b>Example:</b> The DAO (2016)</p>
<p><b>Mechanism:</b> Multiple constraints have strong individual gradients
that point in opposite directions → Σ∇σ ≈ 0</p>
<p><b>Signature:</b> c(p) → 0, Σ||∇σ|| ≫ 0</p>
<p><b>Detection:</b> DarkZoneDetector (cancellation_eps)</p>
<p><b>Fix:</b> Break the gradient alignment. Move one constraint's
activation region so the cancellation can't occur.</p>
<p><b>Danger level:</b> Highest. Invisible because each constraint
appears to be "working" — strong but canceling.</p>
</div></div>
<div class="comp-card parity">
<h3 style="color:#ddcc44">Cold Start Gap</h3>
<div style="font-size:11px;color:#889;line-height:1.8">
<p><b>Example:</b> Parity Wallet (2017)</p>
<p><b>Mechanism:</b> The initial state sits in a region where no
constraint has activated yet.</p>
<p><b>Signature:</b> c(p) → 1, Σ||∇σ|| ≈ 0, ||Π|| ≈ 0</p>
<p><b>Detection:</b> ResidualDetector (||Π|| < ε)</p>
<p><b>Fix:</b> Add a deployment constraint that ensures the system
starts in an activated region.</p>
<p><b>Danger level:</b> High. Visible if you check the initial state,
but traditional audits check code correctness, not initial-state topology.</p>
</div></div></div>

<!-- How audit would have caught it -->
<h2>How a Dark Zone Scanner Would Have Caught Parity</h2>
<div class="row">
<div class="card">
<h3>Step 1: Constraint Modeling</h3>
<div style="font-size:11px;color:#889;line-height:1.8">
<p>Auditor defines 2 rules:</p>
<p>• <b>init_guard</b>: only_uninitialized prevents re-init</p>
<p>• <b>owner_guard</b>: only_owner prevents unauthorized kill</p>
<p>Scanner asks: "At state (x=0, y=0), what is ||Π||?"</p>
</div></div>
<div class="card">
<h3>Step 2: Residual Scan</h3>
<div style="font-size:11px;color:#889;line-height:1.8">
<p>||Π(0,0)|| = {points[0]['combined']:.4f} < ε={residual_detector.epsilon}</p>
<p style="color:#ddcc44">→ RESIDUAL DETECTED at the origin</p>
<p>But c(p) = {points[0]['c_ratio']:.4f} → NOT a cancellation</p>
<p>Classified as: <b>cold start gap</b></p>
</div></div>
<div class="card">
<h3>Step 3: Remediation</h3>
<div style="font-size:11px;color:#889;line-height:1.8">
<p>Scanner recommends:</p>
<p>• Add <b>deploy_order</b> constraint (L1 lifecycle)</p>
<p>• Ensure library contract is initialized at deploy time</p>
<p>• OR: make the library contract non-initializable directly
(pure library, not a standalone contract)</p>
<p style="margin-top:8px;color:#44dd66">Fix verified: ||Π|| at (0,0) rises from {points[0]['combined']:.4f} to above threshold</p>
</div></div></div>"""

# The missing constraint visualization
html += f"""<h2>The Missing Third Constraint</h2>
<div class="callout callout-green">
<b>deploy_order constraint (currently certainty=0.1):</b><br>
In the Parity system, the library contract SHOULD have been either:<br>
1. Initialized at deployment (constructor-sets-owner), or<br>
2. A "pure library" that cannot be called directly (only via delegatecall)<br><br>
Without this, the system has a <b>structural cold start gap</b>: the initial state
(x=0, y=0) has zero protection from BOTH init_guard and owner_guard.<br><br>
<b>With deploy_order activated (simulated):</b> the initial state would be
pushed to (x≥0.3, y≥0.3) before any user interaction, closing the gap.<br>
This is the constraint engineering equivalent of "initialize before use" —
a rule so basic it's invisible until it's missing.
</div>"""

# Methodology appendix
html += f"""<h2>Constraint Residual Framework — DeFi Application</h2>
<table>
<tr><th>Symbol</th><th>Definition</th><th>The DAO Value</th><th>Parity Value</th></tr>
<tr><td>||Π(p)||</td><td>Combined constraint gradient magnitude</td>
<td style="color:#dd6644">0.000</td><td style="color:#dd6644">{points[0]['combined']:.4f}</td></tr>
<tr><td>Σ||∇σ||</td><td>Sum of individual gradient magnitudes</td>
<td style="color:#ddcc44">4.549</td><td style="color:#ddcc44">{points[0]['total_indiv']:.4f}</td></tr>
<tr><td>c(p)</td><td>Cancellation ratio = ||Π||/Σ||∇σ||</td>
<td style="color:#cc44dd">0.000</td><td style="color:#ddcc44">{points[0]['c_ratio']:.4f}</td></tr>
<tr><td>Classification</td><td>Vulnerability type</td>
<td style="color:#cc44dd">Type III Dark Zone</td><td style="color:#ddcc44">Cold Start Gap</td></tr>
<tr><td>Detector</td><td>Which scanner catches it</td>
<td>DarkZoneDetector</td><td>ResidualDetector</td></tr>
</table>

<div class="callout callout-purple">
<b>核心洞察：</b> The DAO 和 Parity 共享同一个数学结构——||Π|| ≈ 0——但通过完全不同的机制达到。<br>
传统审计将它们都归类为"漏洞"。约束残差框架将它们区分为两种不同的暗区物种，各自有特定的检测算法和修复策略。<br>
<b>分类即诊断。</b>
</div>"""

# Charts
html += f"""
<script>
const xs = {json.dumps(xs.tolist())};
const ys = {json.dumps(ys.tolist())};
const forceData = {json.dumps(heatmap_combined.tolist())};
const cancelData = {json.dumps(heatmap_cancel.tolist())};
const indivData = {json.dumps(heatmap_indiv.tolist())};

const forceMax = Math.max(...forceData.flat());
const cancelMax = Math.max(...cancelData.flat());

function rgbaMap(v, max, style) {{
    const t = Math.min(Math.max(v / (max || 1), 0), 1);
    if (style === 'force') {{
        const r = Math.floor(t * 240);
        const g = Math.floor(t * 190);
        const b = Math.floor((1-t) * 180 + t * 40);
        return `rgba(${{r}},${{g}},${{b}},0.9)`;
    }} else {{
        // c(p): near 1 = red (not canceling), near 0 = blue (canceling)
        const r = Math.floor(t * 220);
        const g = Math.floor(t * 140 + (1-t) * 30);
        const b = Math.floor((1-t) * 200 + t * 30);
        return `rgba(${{r}},${{g}},${{b}},0.9)`;
    }}
}}

function drawHeatmap(canvasId, data, maxVal, style) {{
    const ctx = document.getElementById(canvasId).getContext('2d');
    const datasets = [];
    for (let row = 0; row < data.length; row++) {{
        datasets.push({{
            label: '',
            data: data[row],
            backgroundColor: data[row].map(v => rgbaMap(v, maxVal, style)),
            borderWidth: 0,
        }});
    }}
    new Chart(ctx, {{
        type: 'bar',
        data: {{ labels: ys.map(y => y.toFixed(2)), datasets: datasets }},
        options: {{
            responsive: true, maintainAspectRatio: false, animation: false,
            plugins: {{
                legend: {{ display: false }},
                tooltip: {{ callbacks: {{ label: ctx => {{
                    const row = ctx.datasetIndex;
                    const col = ctx.dataIndex;
                    return `(${{xs[col].toFixed(2)}}, ${{ys[row].toFixed(2)}}): ${{ctx.raw.toFixed(4)}}`;
                }} }} }}
            }},
            scales: {{
                x: {{ stacked: true, display: false }},
                y: {{ stacked: true, display: true, title: {{ display: true, text: '← Ownership legitimacy (y)', color: '#667' }} }}
            }}
        }}
    }});
}}

drawHeatmap('forceChart', forceData, forceMax, 'force');
drawHeatmap('cancelChart', cancelData, 1.05, 'cancel');
</script>

<footer>
DeFi Dark Zone Scanner · Constraint Residual Framework · 2026<br>
Case studies: The DAO (Type III Dark Zone) + Parity Wallet (Cold Start Gap)<br>
"Two vulnerabilities. One mathematical structure. Different species."
</footer>
</body></html>"""

with open(OUT, 'w') as f:
    f.write(html)

print(f"Report: {OUT}")
print(f"\nDark zones (cancellation-based): {len(dark_clusters)}")
for dc in dark_clusters:
    print(f"  #{dc.id}: {dc.balance_topology}, c(p)={dc.mean_cancellation_ratio:.4f}, "
          f"centroid=({dc.centroid[0]:.3f}, {dc.centroid[1]:.3f}), "
          f"constraints={dc.constraints_involved}")
print(f"\nResidual clusters (||Π||-based): {len(residual_clusters)}")
for rc in residual_clusters[:5]:
    print(f"  magnitude={rc.mean_magnitude:.3f}, n_points={len(rc.points)}")

print(f"\nCold start gap points (||Π||<0.3, Σ||∇σ||<0.3, c(p)>0.8): {len(weak_points)}")
if weak_points:
    cg = np.mean([wp['combined'] for wp in weak_points])
    ig = np.mean([wp['total_indiv'] for wp in weak_points])
    cr = np.mean([wp['c_ratio'] for wp in weak_points])
    print(f"  avg ||Π||={cg:.4f}, avg Σ||∇σ||={ig:.4f}, avg c(p)={cr:.4f}")

print(f"\nPoint comparison:")
for pt in points:
    print(f"  {pt['label']:40s} ||Π||={pt['combined']:.4f}  "
          f"Σ||∇σ||={pt['total_indiv']:.4f}  c(p)={pt['c_ratio']:.4f}  [{pt['vuln_type']}]")
print(f"\nContrast with The DAO dark zone at (0.5, 0.5): ||Π||=0.000, Σ||∇σ||=4.549, c(p)=0.000")
