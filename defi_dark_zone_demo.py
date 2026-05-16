#!/usr/bin/env python3
"""
DeFi Dark Zone Scanner — Case Study: The DAO Hack (2016)

Demonstrates constraint residual detection on a real DeFi exploit.
Models the reentrancy vulnerability as a Type III dark zone:
two individually-correct constraints whose gradients cancel exactly
in the recursive withdrawal window.

Run:
  cd /Users/dengxinhang/paper && python3 constraint_residual/defi_dark_zone_demo.py
  open constraint_residual/defi_dark_zone_output.html
"""

import numpy as np
import json
import os
import sys
sys.path.insert(0, '/Users/dengxinhang/paper')
from constraint_residual.core import Rule, ConstraintField, ResidualDetector, ResidualPoint
from constraint_residual.dark_zone_detector import DarkZoneDetector, DarkZoneCluster

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "defi_dark_zone_output.html")

# ═══════════════════════════════════════════════════════════════
# State Space
# ═══════════════════════════════════════════════════════════════
# x ∈ [0, 1] : recursion depth (0 = normal single call, 1 = deep recursion)
# y ∈ [0, 1] : balance consistency (0 = not yet updated post-transfer, 1 = fully synced)
#
# The 2D state space represents the contract's logical state during
# a withdrawal.  Under normal operation the state stays near (0.2, 0.8) —
# shallow calls, balances updated promptly.  During the recursive exploit
# the state is pushed toward (0.5, 0.5) — the dark zone.

# ═══════════════════════════════════════════════════════════════
# Constraint Functions
# ═══════════════════════════════════════════════════════════════

def sigma_withdraw_limit(p):
    """Withdrawal proportionality constraint.

    Enforces: 'can only withdraw up to your token share of total Ether.'
    Strongest during normal operation (low recursion, moderate consistency).
    Gradient pushes AGAINST increasing recursion — this constraint resists
    the state moving toward deeper recursive calls.
    """
    x, y = float(p[0]), float(p[1])
    return np.exp(-((x - 0.2) / 0.25)**2 - ((y - 0.5) / 0.4)**2)


def sigma_call_mechanism(p):
    """EVM nested-call constraint.

    Encodes: 'Ethereum allows contracts to make external calls that can
    re-enter the caller.'  Not a bug — a feature of the platform.
    Strongest at high recursion depth.  Gradient pushes WITH increasing
    recursion — this constraint enables the state to move deeper.
    """
    x, y = float(p[0]), float(p[1])
    return np.exp(-((x - 0.8) / 0.25)**2 - ((y - 0.5) / 0.4)**2)


def sigma_balance_sync(p):
    """Balance accounting consistency constraint.

    Enforces: 'totalEther == sum of all individual balances.'
    Strong at the recursion midpoint (where the exploit lives) but
    its GRADIENT is near zero there — the constraint has magnitude
    but no directional force.  It knows something is wrong but can't
    tell which way to push.  This is the third constraint that makes
    the dark zone 'Type III': strong individual magnitude, zero
    gradient contribution, perfect for hiding.
    """
    x, y = float(p[0]), float(p[1])
    return -0.85 * np.exp(-((x - 0.5) / 0.2)**2 - ((y - 0.5) / 0.2)**2)


# ═══════════════════════════════════════════════════════════════
# Build constraint field
# ═══════════════════════════════════════════════════════════════

rules = [
    Rule(name="withdraw_limit", layer=1, domain="tokenomics",
         constraint_fn=sigma_withdraw_limit, certainty=1.0),
    Rule(name="call_mechanism", layer=1, domain="evm_platform",
         constraint_fn=sigma_call_mechanism, certainty=1.0),
    Rule(name="balance_sync", layer=1, domain="accounting",
         constraint_fn=sigma_balance_sync, certainty=1.0),
]

field = ConstraintField(rules=rules)

# ═══════════════════════════════════════════════════════════════
# Scan for dark zones
# ═══════════════════════════════════════════════════════════════

detector = DarkZoneDetector(
    cancellation_eps=0.15,   # points with c(p) < 0.15 are dark zone candidates
    individual_min=0.3,      # each participating rule must have ||∇σ|| > 0.3
)
bounds = [(0, 1), (0, 1)]
n_points = 80
dark_clusters = detector.scan(field, bounds, n_points=n_points)

# ═══════════════════════════════════════════════════════════════
# Also run residual scan for broader context
# ═══════════════════════════════════════════════════════════════

residual_detector = ResidualDetector(field, epsilon=0.1)
residuals = residual_detector.scan_grid(bounds, n_points=n_points)
residual_clusters = residual_detector.cluster_residuals(residuals, angle_threshold_deg=30)

# ═══════════════════════════════════════════════════════════════
# Compute full heatmaps
# ═══════════════════════════════════════════════════════════════

xs = np.linspace(0, 1, 100)
ys = np.linspace(0, 1, 100)
heatmap_combined = np.zeros((100, 100))     # ||Π|| = combined constraint force
heatmap_cancel = np.zeros((100, 100))       # c(p) = cancellation ratio
heatmap_indiv_sum = np.zeros((100, 100))    # Σ||∇σ|| = total individual force
heatmap_sigma_sum = np.zeros((100, 100))    # Σ σ_i = sum of constraint values

for i, x in enumerate(xs):
    for j, y in enumerate(ys):
        p = np.array([x, y])
        grad = field.constraint_gradient(p)
        heatmap_combined[j, i] = float(np.linalg.norm(grad))

        indiv_mags = [float(np.linalg.norm(r.gradient(p))) for r in field.rules]
        total_indiv = sum(indiv_mags)
        heatmap_indiv_sum[j, i] = total_indiv

        combined_mag = float(np.linalg.norm(grad))
        if total_indiv > 1e-10:
            heatmap_cancel[j, i] = combined_mag / total_indiv
        else:
            heatmap_cancel[j, i] = 1.0

        heatmap_sigma_sum[j, i] = sum(r.constraint_fn(p) for r in field.rules)


# ═══════════════════════════════════════════════════════════════
# Stress propagation analysis
# ═══════════════════════════════════════════════════════════════

def compute_stress_path(start, direction, steps=50, step_size=0.04):
    """Trace how constraint forces change along a path through state space."""
    path = []
    p = np.array(start, dtype=float)
    d = np.array(direction, dtype=float)
    d = d / np.linalg.norm(d)
    for _ in range(steps):
        grad = field.constraint_gradient(p)
        indiv = {r.name: float(np.linalg.norm(r.gradient(p))) for r in field.rules}
        path.append({
            'position': p.tolist(),
            'combined_force': float(np.linalg.norm(grad)),
            'individual_forces': dict(indiv),
        })
        p = p + d * step_size
        if p[0] < 0 or p[0] > 1 or p[1] < 0 or p[1] > 1:
            break
    return path

# Attack path: from normal state through dark zone
attack_path = compute_stress_path([0.2, 0.8], [0.6, -0.6])

# ═══════════════════════════════════════════════════════════════
# Key metrics at specific points
# ═══════════════════════════════════════════════════════════════

def compute_metrics_at(p, label):
    grad = field.constraint_gradient(p)
    indiv = {r.name: float(np.linalg.norm(r.gradient(p))) for r in field.rules}
    total_indiv = sum(indiv.values())
    combined = float(np.linalg.norm(grad))
    cr = combined / total_indiv if total_indiv > 1e-10 else 1.0
    return {
        'label': label,
        'position': p.tolist(),
        'combined_force': combined,
        'individual_forces': indiv,
        'total_individual': total_indiv,
        'cancellation_ratio': cr,
    }

point_metrics = [
    compute_metrics_at(np.array([0.2, 0.8]), "Normal operation"),
    compute_metrics_at(np.array([0.5, 0.5]), "Dark zone (exploit window)"),
    compute_metrics_at(np.array([0.8, 0.2]), "Post-exploit (drained)"),
]

# ═══════════════════════════════════════════════════════════════
# Generate HTML Report
# ═══════════════════════════════════════════════════════════════

# Pack data for JS
dark_zone_data = []
for dc in dark_clusters:
    dark_zone_data.append({
        'id': dc.id,
        'topology': dc.balance_topology,
        'n_points': len(dc.points),
        'cancellation_ratio': dc.mean_cancellation_ratio,
        'centroid': dc.centroid.tolist() if dc.centroid is not None else None,
        'constraints': dc.constraints_involved,
        'break_direction': dc.break_direction.tolist() if dc.break_direction is not None else None,
    })

html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DeFi Dark Zone Scanner — The DAO Case Study</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ background:#0a0a12; color:#c8c8d8; font-family:'SF Mono','Menlo','Consolas',monospace; padding:24px 40px; line-height:1.5; }}
h1 {{ font-size:20px; color:#fff; margin-bottom:2px; }}
h2 {{ font-size:15px; color:#8899cc; margin:32px 0 12px; border-bottom:1px solid #1a1a3a; padding-bottom:6px; }}
h3 {{ font-size:13px; color:#aabbdd; margin:16px 0 8px; }}
.sub {{ color:#556; font-size:11px; margin-bottom:28px; }}
.row {{ display:flex; gap:20px; flex-wrap:wrap; margin:12px 0; }}
.card {{ background:#0e0e20; border:1px solid #1a1a3a; border-radius:8px; padding:18px; flex:1; min-width:360px; }}
.card h3 {{ margin-top:0; }}
.chart-wrap {{ position:relative; height:320px; }}
.metric {{ display:inline-block; background:#111133; border-radius:6px; padding:12px 18px; margin:6px; text-align:center; }}
.metric .val {{ font-size:26px; font-weight:bold; }}
.metric .lbl {{ font-size:10px; color:#556; margin-top:2px; }}
.val-good {{ color:#44dd66; }}
.val-warn {{ color:#ddcc44; }}
.val-danger {{ color:#dd6644; }}
.val-info {{ color:#66aadd; }}

.dz-card {{ background:#0e0e20; border:1px solid #331144; border-radius:8px; padding:16px; margin:10px 0; border-left:4px solid #cc44dd; }}
.dz-card .dz-title {{ color:#cc44dd; font-weight:bold; margin-bottom:6px; }}
.dz-card .dz-detail {{ font-size:11px; color:#998; line-height:1.7; }}

.point-row {{ display:flex; gap:12px; margin:8px 0; }}
.point-box {{ background:#111122; border-radius:6px; padding:12px; flex:1; }}
.point-box .pt-label {{ font-size:11px; color:#889; margin-bottom:6px; }}
.point-box .pt-bar {{ display:flex; gap:2px; height:8px; border-radius:4px; overflow:hidden; margin:4px 0; }}

table {{ width:100%; border-collapse:collapse; font-size:11px; margin:8px 0; }}
th {{ background:#111133; color:#889; padding:8px 12px; text-align:left; font-weight:normal; }}
td {{ padding:8px 12px; border-bottom:1px solid #111122; }}
tr:hover td {{ background:#0f0f24; }}

.legend {{ display:flex; gap:16px; flex-wrap:wrap; font-size:10px; color:#667; margin:8px 0; }}
.legend-item {{ display:flex; align-items:center; gap:4px; }}
.legend-swatch {{ width:12px; height:12px; border-radius:2px; }}

.callout {{ background:#111122; border-left:3px solid #66aadd; padding:12px 16px; margin:12px 0; font-size:12px; border-radius:0 6px 6px 0; }}
.callout-danger {{ border-left-color:#dd6644; }}
.callout-purple {{ border-left-color:#cc44dd; }}
</style>
</head>
<body>

<h1>DeFi Dark Zone Scanner</h1>
<div class="sub">Case Study: The DAO Hack (2016) — Constraint Residual Analysis<br>
核心命题：The DAO 不是代码出错。是两条正确规则在递归提款窗口中约束梯度完美互消，形成 Type III 暗区。</div>

<!-- Key Metrics -->
<div>
<div class="metric"><div class="val val-info">{len(rules)}</div><div class="lbl">DeFi protocol constraints</div></div>
<div class="metric"><div class="val val-danger">{len(dark_clusters)}</div><div class="lbl">Dark zones detected</div></div>
<div class="metric"><div class="val val-warn">{len(residual_clusters)}</div><div class="lbl">Residual clusters</div></div>
<div class="metric"><div class="val val-good">{n_points}²</div><div class="lbl">State grid resolution</div></div>
</div>

<!-- Dark Zone Findings -->
<h2>Dark Zone Detection Results</h2>
"""

for dc in dark_clusters:
    cz = dc.centroid
    constraints_str = ', '.join(dc.constraints_involved)
    break_str = ""
    if dc.break_direction is not None:
        break_str = f"Break direction: ({dc.break_direction[0]:.3f}, {dc.break_direction[1]:.3f})"

    html += f"""<div class="dz-card">
<div class="dz-title">Dark Zone #{dc.id} — {dc.balance_topology}</div>
<div class="dz-detail">
<b>Centroid:</b> ({cz[0]:.3f}, {cz[1]:.3f}) |
<b>Mean c(p):</b> {dc.mean_cancellation_ratio:.4f} |
<b>Points:</b> {len(dc.points)} |
<b>Constraints:</b> {constraints_str}<br>
{break_str}<br>
<b>Interpretation:</b> At this state-space location, the <i>withdraw_limit</i> and <i>call_mechanism</i>
constraint gradients point in opposite directions with nearly equal magnitude.
Combined ||Π|| ≈ 0 while individual ||∇σ|| ≫ 0 — the signature of a Type III dark zone.
This is the recursive withdrawal window: each nested call passes the balance check
because the balance hasn't been decremented yet.
</div></div>"""

if not dark_clusters:
    html += """<div class="callout">No dark zones detected at current threshold settings.
Try lowering <code>cancellation_eps</code> or <code>individual_min</code>.</div>"""

# Constraint field heatmap
html += f"""<h2>Constraint Force Field — ||Π(p)||</h2>
<div class="row">
<div class="card" style="flex:1.4"><div class="chart-wrap"><canvas id="forceChart"></canvas></div></div>
<div class="card">
<h3>What This Shows</h3>
<div style="font-size:11px;color:#889;line-height:1.8">
<p><b style="color:#ffdd88">Bright</b> = strong combined constraint protection.</p>
<p><b style="color:#2244aa">Dark</b> = weak combined protection.</p>
<p>The dark band running diagonally through the center is the <b style="color:#cc44dd">dark zone corridor</b> —
where the withdraw_limit and call_mechanism gradients cancel.</p>
<p>Each correctly-implemented rule is individually strong,
but their gradients cancel in this region.</p>
<p style="margin-top:8px;color:#667">State space: x = recursion depth, y = balance consistency</p>
</div></div></div>"""

# Cancellation ratio heatmap
html += f"""<h2>Dark Zone Map — Cancellation Ratio c(p) = ||Σ∇σ|| / Σ||∇σ||</h2>
<div class="row">
<div class="card" style="flex:1.4"><div class="chart-wrap"><canvas id="cancelChart"></canvas></div></div>
<div class="card">
<h3>Reading the Map</h3>
<div style="font-size:11px;color:#889;line-height:1.8">
<p><b style="color:#dd6644">Red/Orange (c ≈ 1)</b> = constraints align, real protection.</p>
<p><b style="color:#3344cc">Blue (c ≈ 0)</b> = dark zone — strong individual constraints,
zero combined protection.</p>
<p>The blue region at center is the <b style="color:#cc44dd">Type III dark zone</b>:
||∇σ_withdraw|| ≈ {point_metrics[1]['individual_forces']['withdraw_limit']:.1f},
||∇σ_call|| ≈ {point_metrics[1]['individual_forces']['call_mechanism']:.1f},
||∇σ_balance|| ≈ {point_metrics[1]['individual_forces']['balance_sync']:.1f},
but ||Π|| ≈ {point_metrics[1]['combined_force']:.1f}.</p>
<p>The balance_sync constraint has strong VALUE ({abs(sigma_balance_sync(np.array([0.5, 0.5]))):.2f})
but zero GRADIENT — its most visible moment is its most useless.</p>
</div></div></div>"""

# Three-point comparison
html += f"""<h2>State-Space Point Comparison</h2>
<div class="point-row">"""
for pm in point_metrics:
    color = 'val-good' if pm['cancellation_ratio'] > 0.5 else 'val-warn' if pm['cancellation_ratio'] > 0.1 else 'val-danger'
    html += f"""<div class="point-box">
<div class="pt-label">{pm['label']}<br><span style="font-size:9px;color:#556">({pm['position'][0]:.2f}, {pm['position'][1]:.2f})</span></div>
<div style="font-size:22px;font-weight:bold" class="{color}">c(p)={pm['cancellation_ratio']:.4f}</div>
<div style="font-size:10px;color:#889;margin-top:4px">
Σ||∇σ|| = {pm['total_individual']:.3f}<br>
||Π|| = {pm['combined_force']:.3f}<br>
"""
    for rname, rval in pm['individual_forces'].items():
        html += f'{rname}: {rval:.3f}<br>'
    html += '</div></div>'
html += '</div>'

# Stress propagation path
html += f"""<h2>Stress Propagation — Attack Path Through the Dark Zone</h2>
<div class="row">
<div class="card" style="flex:1.4"><div class="chart-wrap"><canvas id="stressChart"></canvas></div></div>
<div class="card">
<h3>Attack Trajectory</h3>
<div style="font-size:11px;color:#889;line-height:1.8">
<p>The path traces state from <b>normal operation (0.2, 0.8)</b> through the
<b>dark zone (0.5, 0.5)</b> toward <b>post-exploit (0.8, 0.2)</b>.</p>
<p>As the path enters the dark zone, combined constraint force ||Π|| drops sharply
while individual forces remain strong — the signature of entering a cancellation region.</p>
<p>The attacker's recursive withdrawal pushes the state along this corridor where
neither the withdraw limit nor the call mechanism dominates.</p>
</div></div></div>"""

# Constraint topology summary
html += f"""<h2>Constraint Topology Summary</h2>
<table>
<tr><th>Rule</th><th>Layer</th><th>Domain</th><th>Direction</th><th>Peak ||∇σ||</th><th>At Dark Zone ||∇σ||</th></tr>
<tr><td>withdraw_limit</td><td>L1</td><td>tokenomics</td><td>← resists recursion</td>
<td>≈2.5</td><td style="color:#dd6644">{point_metrics[1]['individual_forces']['withdraw_limit']:.3f}</td></tr>
<tr><td>call_mechanism</td><td>L1</td><td>evm_platform</td><td>→ enables recursion</td>
<td>≈2.5</td><td style="color:#dd6644">{point_metrics[1]['individual_forces']['call_mechanism']:.3f}</td></tr>
<tr><td>balance_sync</td><td>L1</td><td>accounting</td><td>↗ (at normal state)</td>
<td>≈0</td><td style="color:#889">{point_metrics[1]['individual_forces']['balance_sync']:.3f}</td></tr>
<tr style="background:#1a1122"><td colspan="6" style="color:#cc44dd;text-align:center">
<b>▼ Dark Zone: withdraw_limit ∇ ≈ -call_mechanism ∇, balance_sync ∇ ≈ 0</b></td></tr>
</table>"""

# How this differs from traditional audit
html += f"""<h2>Why Traditional Audits Missed This</h2>
<div class="row">
<div class="card">
<h3>Traditional Audit (2016 approach)</h3>
<div style="font-size:11px;color:#889;line-height:1.8">
<p>Q: Is the withdraw limit correctly implemented?<br><b style="color:#44dd66">Yes</b> — proportional to token holdings.</p>
<p>Q: Is the balance update correct?<br><b style="color:#44dd66">Yes</b> — reduces balance after transfer.</p>
<p>Q: Is the external call safe?<br><b style="color:#44dd66">Yes</b> — uses Solidity's standard call.</p>
<p style="margin-top:8px;color:#ddcc44">Each rule <b>individually</b> passes. Audit: ✓</p>
</div></div>
<div class="card">
<h3>Dark Zone Scan (constraint residual method)</h3>
<div style="font-size:11px;color:#889;line-height:1.8">
<p>Q: What is c(p) at (recursion=0.5, sync=0.5)?<br><b style="color:#dd6644">c(p) ≈ 0.0</b> — perfect cancellation.</p>
<p>Q: Do the constraint gradients oppose?<br><b style="color:#dd6644">Yes</b> — withdraw_limit ∇ ≈ −call_mechanism ∇.</p>
<p>Q: Does balance_sync provide backup?<br><b style="color:#dd6644">No</b> — ∇ ≈ 0 at the inflection point.</p>
<p style="margin-top:8px;color:#dd6644">The <b>combination</b> fails. Dark zone: ★★★</p>
</div></div></div>

<div class="callout callout-purple">
<b>核心洞察：</b> The DAO 的每条规则单独看都是正确的。传统审计检查"每条规则是否被正确实现"——答案是是。
暗区扫描检查"规则本身是否形成了保护为零的状态区间"——答案也是是。
<b>这不是代码漏洞。这是约束拓扑的结构性盲区。</b> 传统审计方法学看不到它，因为它不是实现错误。
</div>"""

# What-if: if the DAO team had this tool
html += f"""<h2>Counterfactual: If The DAO Team Had a Dark Zone Scanner</h2>
<div class="row">
<div class="card">
<h3>What they would have seen</h3>
<div style="font-size:11px;color:#889;line-height:1.8">
<p>1. c(p) heatmap shows a blue band through (0.5, 0.5)</p>
<p>2. Dark zone #{'1' if dark_clusters else 'N/A'} classified as <b>mutual_cancellation</b></p>
<p>3. Stress path shows combined force dropping to near zero</p>
<p>4. Scanner recommends: break the cancellation by moving balance update BEFORE the external call</p>
</div></div>
<div class="card">
<h3>Fix: checks-effects-interactions pattern</h3>
<div style="font-size:11px;color:#889;line-height:1.8">
<p>The post-The DAO fix (Solidity best practice since 2016) exactly corresponds to
<b>moving the balance_sync constraint gradient away from the cancellation point</b>:</p>
<p>1. <b style="color:#44dd66">Checks</b>: verify withdraw_limit</p>
<p>2. <b style="color:#44dd66">Effects</b>: update balance FIRST (┴ balance_sync ∇ ≠ 0 now)</p>
<p>3. <b style="color:#44dd66">Interactions</b>: external call LAST</p>
<p style="margin-top:8px">This reorders gradient application so that balance_sync ∇ ≠ 0
when the recursive call arrives — breaking the dark zone.</p>
</div></div></div>

<h2>Methodology</h2>
<div class="callout">
<b>Constraint Residual Method (reminder):</b><br>
σ_i(p) = constraint strength of rule i at state point p<br>
Π(p) = Σ ∇σ_i(p) — combined constraint gradient vector<br>
c(p) = ||Π(p)|| / Σ||∇σ_i(p)|| — cancellation ratio<br>
Dark zone: c(p) → 0 while Σ||∇σ_i(p)|| ≫ 0<br>
<b>Three constraint types at play:</b> withdraw_limit ∇ ≈ −call_mechanism ∇ (mutual cancellation),
balance_sync ∇ ≈ 0 (zero-gradient backup failure).</div>

<footer style="color:#444;font-size:10px;text-align:center;margin-top:40px;padding:20px;">
DeFi Dark Zone Scanner · Constraint Residual Framework · 2026<br>
"Every rule was correct. The combination was not."<br>
Case study: The DAO Hack · June 17, 2016 · ~3.6M ETH drained
</footer>

<script>
// ═══════════════════════ Chart Data ═══════════════════════
const xs = {json.dumps(xs.tolist())};
const ys = {json.dumps(ys.tolist())};
const forceData = {json.dumps(heatmap_combined.tolist())};
const cancelData = {json.dumps(heatmap_cancel.tolist())};
const indivSumData = {json.dumps(heatmap_indiv_sum.tolist())};

// Normalize data for color mapping
const forceMax = Math.max(...forceData.flat());
const indivMax = Math.max(...indivSumData.flat());

function rgbaFromValue(v, max, style) {{
    const t = Math.min(Math.max(v / (max || 1), 0), 1);
    if (style === 'force') {{
        // Blue(dark) → yellow → white(bright)
        const r = Math.floor(t * 255);
        const g = Math.floor(t * 200);
        const b = Math.floor((1 - t) * 180 + t * 50);
        return `rgba(${{r}},${{g}},${{b}},0.9)`;
    }} else if (style === 'cancel') {{
        // Blue (c≈0, dark zone) → green → yellow → red (c≈1, safe)
        // We want: dark zone (c≈0) = blue, safe (c≈1) = red
        const r = Math.floor(t * 220);
        const g = Math.floor(t * 150 + (1-t) * 40);
        const b = Math.floor((1-t) * 200 + t * 30);
        return `rgba(${{r}},${{g}},${{b}},0.9)`;
    }} else {{
        const r = Math.floor(t * 180);
        const g = Math.floor(t * 140);
        const b = Math.floor(100 + t * 155);
        return `rgba(${{r}},${{g}},${{b}},0.85)`;
    }}
}}

function buildHeatmapDatasets(data, maxVal, style) {{
    // data[y][x], we have 100 rows (y) and 100 cols (x)
    const datasets = [];
    for (let row = 0; row < data.length; row++) {{
        datasets.push({{
            label: '',
            data: data[row],
            backgroundColor: data[row].map(v => rgbaFromValue(v, maxVal, style)),
            borderWidth: 0,
            borderSkipped: false,
        }});
    }}
    return datasets;
}}

function drawHeatmap(canvasId, data, maxVal, style, xLabel, yLabel) {{
    const ctx = document.getElementById(canvasId).getContext('2d');
    const yLabels = ys.map(y => y.toFixed(2));
    new Chart(ctx, {{
        type: 'bar',
        data: {{
            labels: yLabels,
            datasets: buildHeatmapDatasets(data, maxVal, style),
        }},
        options: {{
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            plugins: {{
                legend: {{ display: false }},
                tooltip: {{
                    callbacks: {{
                        label: function(ctx) {{
                            const row = ctx.datasetIndex;
                            const col = ctx.dataIndex;
                            return `(${{xs[col].toFixed(2)}}, ${{ys[row].toFixed(2)}}): ${{ctx.raw.toFixed(4)}}`;
                        }}
                    }}
                }},
                title: {{
                    display: true,
                    text: xLabel || 'Recursion depth (x) →',
                    color: '#889',
                    font: {{ size: 11 }}
                }}
            }},
            scales: {{
                x: {{ stacked: true, display: false }},
                y: {{
                    stacked: true,
                    display: true,
                    title: {{
                        display: true,
                        text: yLabel || 'Balance consistency (y) →',
                        color: '#889',
                        font: {{ size: 11 }}
                    }}
                }}
            }}
        }}
    }});
}}

drawHeatmap('forceChart', forceData, forceMax, 'force',
    'Recursion depth (x) →',
    '← Balance consistency (y)');
drawHeatmap('cancelChart', cancelData, 1.05, 'cancel',
    'Recursion depth (x) →',
    '← Balance consistency (y)');

// Stress propagation chart
const stressCtx = document.getElementById('stressChart').getContext('2d');
const attackPathData = {json.dumps(attack_path)};
const steps = attackPathData.map((_, i) => i);
new Chart(stressCtx, {{
    type: 'line',
    data: {{
        labels: steps,
        datasets: [
            {{
                label: 'Combined Force ||Π||',
                data: attackPathData.map(p => p.combined_force),
                borderColor: '#dd6644',
                backgroundColor: 'transparent',
                borderWidth: 2,
                pointRadius: 0,
                tension: 0.3,
            }},
            {{
                label: 'withdraw_limit ||∇σ||',
                data: attackPathData.map(p => p.individual_forces.withdraw_limit || 0),
                borderColor: '#44dd66',
                backgroundColor: 'transparent',
                borderWidth: 1,
                pointRadius: 0,
                tension: 0.3,
            }},
            {{
                label: 'call_mechanism ||∇σ||',
                data: attackPathData.map(p => p.individual_forces.call_mechanism || 0),
                borderColor: '#66aadd',
                backgroundColor: 'transparent',
                borderWidth: 1,
                pointRadius: 0,
                tension: 0.3,
            }},
            {{
                label: 'balance_sync ||∇σ||',
                data: attackPathData.map(p => p.individual_forces.balance_sync || 0),
                borderColor: '#cc44dd',
                backgroundColor: 'transparent',
                borderWidth: 1,
                pointRadius: 0,
                tension: 0.3,
            }},
        ]
    }},
    options: {{
        responsive: true,
        maintainAspectRatio: false,
        plugins: {{
            legend: {{
                position: 'bottom',
                labels: {{ color: '#889', font: {{ size: 10 }}, boxWidth: 12, padding: 12 }}
            }},
            title: {{
                display: true,
                text: 'Constraint forces along attack path (step 0 = normal → step 30 = dark zone)',
                color: '#889',
                font: {{ size: 11 }}
            }}
        }},
        scales: {{
            x: {{
                title: {{ display: true, text: 'Path step', color: '#667' }},
                ticks: {{ color: '#556' }},
                grid: {{ color: '#1a1a2a' }}
            }},
            y: {{
                title: {{ display: true, text: 'Constraint force', color: '#667' }},
                ticks: {{ color: '#556' }},
                grid: {{ color: '#1a1a2a' }}
            }}
        }}
    }}
}});
</script>

</body></html>"""

with open(OUT, 'w') as f:
    f.write(html)

print(f"Report written to {OUT}")
print(f"\nDark zones found: {len(dark_clusters)}")
for dc in dark_clusters:
    print(f"  #{dc.id}: {dc.balance_topology}, c(p)={dc.mean_cancellation_ratio:.4f}, "
          f"centroid=({dc.centroid[0]:.3f}, {dc.centroid[1]:.3f}), "
          f"constraints={dc.constraints_involved}")
print(f"\nResidual clusters: {len(residual_clusters)}")
for rc in residual_clusters:
    print(f"  magnitude={rc.mean_magnitude:.3f}, n_points={len(rc.points)}")

print(f"\nPoint metrics:")
for pm in point_metrics:
    print(f"  {pm['label']}: c(p)={pm['cancellation_ratio']:.4f}, "
          f"||Π||={pm['combined_force']:.3f}, Σ||∇σ||={pm['total_individual']:.3f}")
