#!/usr/bin/env python3
"""
Interactive Constraint Field Explorer

Generates a self-contained HTML file that lets anyone explore:
- 7 protocols (The DAO, Parity, Poly Network, LendVault V1, LendVault V2, Aave V3 vuln, Aave V3 fixed)
- Real-time c(p) and ||Π|| heatmaps
- Click-to-probe state-space point analysis
- Before/after dark zone comparison
- Four-species taxonomy reference

Run:  cd /Users/dengxinhang/paper && python3 constraint_residual/explorer.py
Open: constraint_residual/explorer.html
"""

import numpy as np
import json
import os
import sys
sys.path.insert(0, '/Users/dengxinhang/paper')
from constraint_residual.dsl.compiler import load_protocol
from constraint_residual.dark_zone_detector import DarkZoneDetector

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "explorer.html")
DSL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dsl", "protocols")

# ═══════════════════════════════════════════════════════════════
# Protocol definitions — all 7
# ═══════════════════════════════════════════════════════════════

PROTOCOLS = {
    'the_dao': {
        'file': 'the_dao.yaml',
        'name': 'The DAO (2016)',
        'type': 'mutual_cancellation',
        'exploit': '3.6M ETH (~$50M)',
        'narrative': 'Recursive withdrawal. withdraw_limit ∇ ≈ −call_mechanism ∇ at midpoint.',
        'fix': 'checks-effects-interactions',
        'dark_point': [0.5, 0.5],
        'normal_point': [0.2, 0.8],
    },
    'parity_wallet': {
        'file': 'parity_wallet.yaml',
        'name': 'Parity Wallet (2017)',
        'type': 'cold_start_gap',
        'exploit': '$300M frozen',
        'narrative': 'Library contract uninitialized. init_guard and owner_guard inactive at origin.',
        'fix': 'constructor initialization',
        'dark_point': [0.05, 0.05],
        'normal_point': [0.6, 0.6],
    },
    'poly_network': {
        'file': 'poly_network.yaml',
        'name': 'Poly Network (2021)',
        'type': 'hierarchical',
        'exploit': '$600M',
        'narrative': 'Missing L1→L2 coupler. Verification payloads could modify keeper keys.',
        'fix': 'payload isolation executor',
        'dark_point': [0.655, 0.326],
        'normal_point': [0.8, 0.5],
    },
    'lendvault_v1': {
        'file': 'lendvault_v1.yaml',
        'name': 'LendVault V1 (vulnerable)',
        'type': 'mutual_cancellation',
        'exploit': 'Hypothetical — designed with intentional dark zone',
        'narrative': 'Withdraw check and balance update gradients oppose at (0.5, 0.5).',
        'fix': 'add withdraw_sequencing constraint',
        'dark_point': [0.5, 0.5],
        'normal_point': [0.8, 0.8],
    },
    'lendvault_v2': {
        'file': 'lendvault_v2.yaml',
        'name': 'LendVault V2 (fixed)',
        'type': 'protected',
        'exploit': 'Dark zone eliminated by constraint fix (linear ramp)',
        'narrative': 'Linear ramp ∇=(0,5) in y-axis breaks x-axis mutual cancellation.',
        'fix': 'already fixed',
        'dark_point': [0.5, 0.5],
        'normal_point': [0.8, 0.8],
    },
    'aave_v3': {
        'file': 'aave_v3.yaml',
        'name': 'Aave v3 (vulnerable)',
        'type': 'hostile_asymmetry',
        'exploit': 'CVSS 9.8 — Oracle staleness (Spark/AaveOracle)',
        'narrative': 'oracle_validator ∇ ≈ −price_corridor ∇ at gray area. L2 constraints cancel.',
        'fix': 'linear timestamp_validator (∇=(0,8) in y-axis)',
        'dark_point': [0.536, 0.704],
        'normal_point': [0.8, 0.9],
    },
    'aave_v3_fixed': {
        'file': 'aave_v3_fixed.yaml',
        'name': 'Aave v3 (fixed)',
        'type': 'protected',
        'exploit': 'Dark zone eliminated by linear timestamp_validator',
        'narrative': 'Linear ramp ∇=(0,8) breaks oracle_validator↔price_corridor cancellation.',
        'fix': 'already fixed',
        'dark_point': [0.536, 0.704],
        'normal_point': [0.8, 0.9],
    },
}

N_POINTS = 64  # heatmap resolution

# ═══════════════════════════════════════════════════════════════
# Pre-compute all data
# ═══════════════════════════════════════════════════════════════

print("Pre-computing constraint fields...")
all_data = {}

for key, info in PROTOCOLS.items():
    yaml_path = os.path.join(DSL_DIR, info['file'])
    field, spec = load_protocol(yaml_path)

    # Heatmaps
    xs = np.linspace(0, 1, N_POINTS)
    ys = np.linspace(0, 1, N_POINTS)
    hm_force = np.zeros((N_POINTS, N_POINTS))
    hm_cancel = np.zeros((N_POINTS, N_POINTS))

    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            p = np.array([x, y])
            grad = field.constraint_gradient(p)
            hm_force[j, i] = float(np.linalg.norm(grad))
            indiv = sum(float(np.linalg.norm(r.gradient(p))) for r in field.rules)
            hm_cancel[j, i] = float(np.linalg.norm(grad)) / indiv if indiv > 1e-10 else 1.0

    # Dark zone detection
    detector = DarkZoneDetector(cancellation_eps=0.15, individual_min=0.3)
    bounds = [(0, 1), (0, 1)]
    dark_clusters = detector.scan(field, bounds, n_points=64)

    # Constraint info
    constraints = []
    for c in spec.get('constraints', []):
        constraints.append({
            'name': c['name'],
            'layer': c.get('layer', '?'),
            'domain': c.get('domain', '?'),
            'fn': c.get('fn', '?'),
            'description': c.get('description', '')[:120],
        })

    # Compute metrics at key points
    def metrics_at(pt):
        p = np.array(pt)
        grad = field.constraint_gradient(p)
        combined = float(np.linalg.norm(grad))
        indiv_grad = {r.name: float(np.linalg.norm(r.gradient(p))) for r in field.rules}
        indiv_val = {r.name: float(r.constraint_fn(p)) for r in field.rules}
        total_indiv = sum(indiv_grad.values())
        cr = combined / total_indiv if total_indiv > 1e-10 else 1.0
        return {
            'combined': combined, 'total_indiv': total_indiv,
            'total_sigma': sum(indiv_val.values()), 'c_ratio': cr,
            'individual_grad': indiv_grad, 'individual_val': indiv_val,
        }

    dark_metrics = metrics_at(info['dark_point'])
    normal_metrics = metrics_at(info['normal_point'])

    # Dark zone clusters
    dz_data = []
    for dc in dark_clusters:
        dz_data.append({
            'centroid': dc.centroid.tolist() if dc.centroid is not None else None,
            'c_ratio': dc.mean_cancellation_ratio,
            'topology': dc.balance_topology,
            'n_points': len(dc.points),
            'constraints': dc.constraints_involved,
        })

    all_data[key] = {
        'info': info,
        'force_heatmap': hm_force.tolist(),
        'cancel_heatmap': hm_cancel.tolist(),
        'dark_zones': dz_data,
        'dark_metrics': dark_metrics,
        'normal_metrics': normal_metrics,
        'constraints': constraints,
        'force_max': float(np.max(hm_force)),
    }
    print(f"  {info['name']}: {len(dark_clusters)} dark zones, force_max={all_data[key]['force_max']:.1f}")

# ═══════════════════════════════════════════════════════════════
# Generate Interactive HTML
# ═══════════════════════════════════════════════════════════════

print(f"\nGenerating interactive explorer...")

html = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DeFi Dark Zone Scanner — Interactive Explorer</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0a0a12;color:#c8c8d8;font-family:'SF Mono','Menlo','Consolas',monospace;padding:20px 32px;line-height:1.5}
h1{font-size:18px;color:#fff;margin-bottom:2px}
.sub{color:#556;font-size:10px;margin-bottom:16px}
.topbar{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:12px 0}
.topbar select,.topbar button{background:#111133;color:#c8c8d8;border:1px solid #1a1a3a;border-radius:6px;padding:8px 14px;font-family:inherit;font-size:12px;cursor:pointer}
.topbar select:hover,.topbar button:hover{background:#1a1a44}
.topbar button.active{background:#1a1a44;border-color:#66aadd}
.row{display:flex;gap:14px;flex-wrap:wrap;margin:10px 0}
.card{background:#0e0e20;border:1px solid #1a1a3a;border-radius:8px;padding:14px;flex:1;min-width:320px}
.card h3{font-size:12px;color:#aabbdd;margin-bottom:8px}
.chart-wrap{position:relative;height:280px;cursor:crosshair}
.metric-row{display:flex;gap:8px;flex-wrap:wrap;margin:4px 0}
.metric{background:#111133;border-radius:5px;padding:8px 14px;text-align:center;flex:1;min-width:80px}
.metric .val{font-size:18px;font-weight:bold}
.metric .lbl{font-size:9px;color:#556;margin-top:2px}
.val-danger{color:#dd6644}.val-warn{color:#ddcc44}.val-good{color:#44dd66}.val-info{color:#66aadd}
.probe-info{font-size:10px;color:#889;line-height:1.7}
.probe-info b{color:#aabbdd}
.grad-bar{display:flex;gap:2px;height:6px;border-radius:3px;overflow:hidden;margin:2px 0}
.grad-seg{border-radius:0}
table{width:100%;border-collapse:collapse;font-size:10px;margin:6px 0}
th{background:#111133;color:#889;padding:6px 10px;text-align:left;font-weight:normal}
td{padding:6px 10px;border-bottom:1px solid #111122}
.callout{background:#111122;border-left:3px solid #66aadd;padding:10px 14px;margin:8px 0;font-size:11px;border-radius:0 5px 5px 0;line-height:1.7}
footer{color:#444;font-size:9px;text-align:center;margin-top:28px;padding:14px}
</style>
</head>
<body>

<h1>DeFi Dark Zone Scanner</h1>
<div class="sub">Constraint Residual Framework — Interactive Explorer<br>
Click on any heatmap to probe the constraint field at that state-space point.</div>

<div class="topbar">
<select id="protocolSelect"></select>
<button id="btnNormal" class="active">← Normal state</button>
<button id="btnDark">→ Dark zone</button>
<span style="font-size:10px;color:#667;margin-left:4px" id="pointLabel">Probe at (0.50, 0.50)</span>
</div>

<div class="metric-row" id="metricRow"></div>

<div class="row">
<div class="card" style="flex:1.2">
<h3>Constraint Force ||Π(p)|| <span style="color:#667;font-weight:normal">— combined protection strength</span></h3>
<div class="chart-wrap"><canvas id="forceChart"></canvas></div>
</div>
<div class="card" style="flex:1.2">
<h3>Cancellation Ratio c(p) <span style="color:#667;font-weight:normal">— ||Σ∇σ|| / Σ||∇σ||</span></h3>
<div class="chart-wrap"><canvas id="cancelChart"></canvas></div>
</div>
<div class="card" style="max-width:320px">
<h3>Constraint Gradient Breakdown</h3>
<div id="gradientBreakdown" class="probe-info">Select a point on the heatmap.</div>
<div style="margin-top:8px">
<h3>Dark Zones</h3>
<div id="darkZoneList" class="probe-info"></div>
</div>
</div>
</div>

<div class="row">
<div class="card">
<h3>Constraint Inventory</h3>
<table><thead><tr><th>Name</th><th>Layer</th><th>Domain</th><th>Fn</th></tr></thead>
<tbody id="constraintTable"></tbody></table>
</div>
<div class="card">
<h3>Dark Zone Taxonomy</h3>
<table>
<tr><td style="color:#cc44dd">mutual_cancellation</td><td>c→0, Σ||∇σ||≫0</td><td>Peer constraints cancel</td></tr>
<tr><td style="color:#ddcc44">cold_start_gap</td><td>c→1, Σ||∇σ||≈0</td><td>All constraints inactive</td></tr>
<tr><td style="color:#66aadd">hierarchical</td><td>c→0, cross-layer</td><td>Missing executor L1↛L2</td></tr>
<tr><td style="color:#dd6644">hostile_asymmetry</td><td>c→0, same-direction</td><td>Protocol constraints weakened by external factor</td></tr>
</table>
<div class="callout" style="margin-top:12px" id="narrativeBox"></div>
</div>
</div>

<footer>DeFi Dark Zone Scanner · Constraint Residual Framework · 2026<br>"Every rule was correct. The combination was not."</footer>

<script>
// ═══════════════════════ Data ═══════════════════════
const ALL_DATA = """ + json.dumps(all_data) + r""";
const N = """ + str(N_POINTS) + r""";
const xs = Array.from({length:N}, (_,i) => i/(N-1));
const ys = Array.from({length:N}, (_,i) => i/(N-1));

let currentProtocol = 'the_dao';
let currentPoint = [0.5, 0.5];
let forceChart, cancelChart;

// ═══════════════════════ Init ═══════════════════════
function init() {
    const sel = document.getElementById('protocolSelect');
    for (const [key, data] of Object.entries(ALL_DATA)) {
        const opt = document.createElement('option');
        opt.value = key;
        opt.textContent = data.info.name;
        if (key === currentProtocol) opt.selected = true;
        sel.appendChild(opt);
    }
    sel.addEventListener('change', () => {
        currentProtocol = sel.value;
        currentPoint = [...ALL_DATA[currentProtocol].info.normal_point];
        updateAll();
    });
    document.getElementById('btnNormal').addEventListener('click', () => {
        currentPoint = [...ALL_DATA[currentProtocol].info.normal_point];
        updateAll();
    });
    document.getElementById('btnDark').addEventListener('click', () => {
        currentPoint = [...ALL_DATA[currentProtocol].info.dark_point];
        updateAll();
    });
    updateAll();
}

// ═══════════════════════ Heatmaps ═══════════════════════
function rgbaMap(v, max, style) {
    const t = Math.min(Math.max(v/(max||1), 0), 1);
    if (style === 'force') {
        return `rgba(${Math.floor(t*240)},${Math.floor(t*190)},${Math.floor((1-t)*180+t*40)},0.9)`;
    } else {
        return `rgba(${Math.floor(t*220)},${Math.floor(t*140+(1-t)*30)},${Math.floor((1-t)*200+t*30)},0.9)`;
    }
}

function buildDatasets(data, maxVal, style) {
    return data.map((row, i) => ({
        label: '', data: row,
        backgroundColor: row.map(v => rgbaMap(v, maxVal, style)),
        borderWidth: 0,
    }));
}

function drawHeatmaps() {
    const data = ALL_DATA[currentProtocol];
    const forceMax = data.force_max;

    const forceCtx = document.getElementById('forceChart').getContext('2d');
    const cancelCtx = document.getElementById('cancelChart').getContext('2d');

    const baseOpts = {
        responsive: true, maintainAspectRatio: false, animation: false,
        plugins: {
            legend: { display: false },
            tooltip: { enabled: false }
        },
        scales: {
            x: { stacked: true, display: false },
            y: { stacked: true, display: false }
        },
        onClick: (evt) => {
            const canvas = evt.chart.canvas;
            const rect = canvas.getBoundingClientRect();
            const scaleX = canvas.width / rect.width;
            const scaleY = canvas.height / rect.height;
            const cx = (evt.native.offsetX || (evt.x - rect.left)) * scaleX;
            const cy = (evt.native.offsetY || (evt.y - rect.top)) * scaleY;
            const chartArea = evt.chart.chartArea;
            const rx = (cx - chartArea.left) / (chartArea.right - chartArea.left);
            const ry = 1 - (cy - chartArea.top) / (chartArea.bottom - chartArea.top);
            currentPoint = [
                Math.max(0, Math.min(1, rx)),
                Math.max(0, Math.min(1, ry))
            ];
            updateAll();
        }
    };

    if (forceChart) forceChart.destroy();
    if (cancelChart) cancelChart.destroy();

    forceChart = new Chart(forceCtx, {
        type: 'bar',
        data: { labels: ys.map(y => y.toFixed(2)), datasets: buildDatasets(data.force_heatmap, forceMax, 'force') },
        options: baseOpts
    });
    cancelChart = new Chart(cancelCtx, {
        type: 'bar',
        data: { labels: ys.map(y => y.toFixed(2)), datasets: buildDatasets(data.cancel_heatmap, 1.05, 'cancel') },
        options: baseOpts
    });
}

// ═══════════════════════ Bilinear interpolation for probe point ═══════════════════════
function interpolate(heatmap, px, py) {
    const fx = px * (N - 1);
    const fy = py * (N - 1);
    const ix = Math.floor(fx), iy = Math.floor(fy);
    const ixn = Math.min(ix + 1, N - 1), iyn = Math.min(iy + 1, N - 1);
    const dx = fx - ix, dy = fy - iy;
    const v00 = heatmap[iy][ix], v10 = heatmap[iy][ixn];
    const v01 = heatmap[iyn][ix], v11 = heatmap[iyn][ixn];
    return (1-dx)*(1-dy)*v00 + dx*(1-dy)*v10 + (1-dx)*dy*v01 + dx*dy*v11;
}

function computeProbeMetrics(px, py) {
    const data = ALL_DATA[currentProtocol];
    const force = interpolate(data.force_heatmap, px, py);
    const cancel = interpolate(data.cancel_heatmap, px, py);
    return { force, cancel };
}

// ═══════════════════════ Update ═══════════════════════
function updateAll() {
    const data = ALL_DATA[currentProtocol];
    const [px, py] = currentPoint;
    const metrics = computeProbeMetrics(px, py);

    // Metric row
    const isDark = metrics.cancel < 0.2;
    const color = isDark ? 'val-danger' : metrics.cancel < 0.5 ? 'val-warn' : 'val-good';
    const status = isDark ? 'DARK ZONE' : metrics.cancel < 0.4 ? 'MARGINAL' : 'PROTECTED';
    document.getElementById('metricRow').innerHTML = `
        <div class="metric"><div class="val val-info">(${px.toFixed(3)}, ${py.toFixed(3)})</div><div class="lbl">Probe position</div></div>
        <div class="metric"><div class="val">${metrics.force.toFixed(3)}</div><div class="lbl">||Π(p)||</div></div>
        <div class="metric"><div class="val ${color}">${metrics.cancel.toFixed(4)}</div><div class="lbl">c(p)</div></div>
        <div class="metric"><div class="val ${color}">${status}</div><div class="lbl">Classification</div></div>
    `;

    // Point label
    document.getElementById('pointLabel').textContent =
        `Probe at (${px.toFixed(2)}, ${py.toFixed(2)}) — c(p)=${metrics.cancel.toFixed(4)}`;

    // Gradient breakdown — use pre-computed dark/normal metrics as reference
    let gradHtml = `<p>Probe: <b>(${px.toFixed(3)}, ${py.toFixed(3)})</b></p>
<p>||Π|| = <b>${metrics.force.toFixed(3)}</b> | c(p) = <b style="color:${isDark?'#dd6644':'#44dd66'}">${metrics.cancel.toFixed(4)}</b></p>
<p style="margin-top:6px;color:#667">Constraints (pre-computed at dark & normal points):</p>
`;

    const dm = data.dark_metrics;
    if (dm) {
        gradHtml += `<p style="color:#cc44dd;margin-top:4px">At dark zone (${data.info.dark_point[0].toFixed(2)},${data.info.dark_point[1].toFixed(2)}):</p>`;
        for (const [name, val] of Object.entries(dm.individual_grad)) {
            const sigma = dm.individual_val[name];
            const sigColor = sigma < 0.1 ? '#dd6644' : sigma > 0.7 ? '#44dd66' : '#ddcc44';
            gradHtml += `<div class="grad-bar" style="width:${Math.min(100,val*25)}%"><div class="grad-seg" style="width:100%;background:${sigColor};opacity:0.7"></div></div>
            <span style="font-size:9px">∇${name} = ${val.toFixed(3)} | σ = ${sigma.toFixed(3)}</span><br>`;
        }
    }

    const nm = data.normal_metrics;
    if (nm) {
        gradHtml += `<p style="color:#44dd66;margin-top:6px">At normal state (${data.info.normal_point[0].toFixed(2)},${data.info.normal_point[1].toFixed(2)}):</p>`;
        for (const [name, val] of Object.entries(nm.individual_grad)) {
            gradHtml += `<span style="font-size:9px">∇${name} = ${val.toFixed(3)} </span>`;
        }
        gradHtml += `<br><span style="font-size:9px;color:#667">c(p) = ${nm.c_ratio.toFixed(4)}</span>`;
    }
    document.getElementById('gradientBreakdown').innerHTML = gradHtml;

    // Dark zone list
    let dzHtml = '';
    if (data.dark_zones.length === 0) {
        dzHtml = '<p style="color:#44dd66">✓ No dark zones detected</p>';
        if (data.info.type === 'cold_start_gap') {
            dzHtml += '<p style="color:#ddcc44;font-size:9px">Cold start gap — detected by ||Π||<ε, not c(p)≈0</p>';
        }
        if (data.info.type === 'hostile_asymmetry') {
            dzHtml += '<p style="color:#dd6644;font-size:9px">Hostile asymmetry — protocol constraints weakened by oracle modulation</p>';
        }
    } else {
        for (const dz of data.dark_zones) {
            const cz = dz.centroid;
            dzHtml += `<div style="background:#1a1122;padding:8px;margin:4px 0;border-radius:4px;border-left:3px solid #cc44dd">
<b style="color:#cc44dd">${dz.topology}</b> c(p)=${dz.c_ratio.toFixed(4)}<br>
<span style="font-size:9px;color:#998">at (${cz[0].toFixed(3)},${cz[1].toFixed(3)}) · ${dz.n_points} pts · ${dz.constraints.join(', ')}</span></div>`;
        }
    }
    document.getElementById('darkZoneList').innerHTML = dzHtml;

    // Constraint table
    let ctHtml = '';
    for (const c of data.constraints) {
        ctHtml += `<tr><td>${c.name}</td><td>${c.layer}</td><td>${c.domain}</td><td>${c.fn}</td></tr>`;
    }
    document.getElementById('constraintTable').innerHTML = ctHtml;

    // Narrative
    document.getElementById('narrativeBox').innerHTML = `
<b>${data.info.name}</b> — <span style="color:#889">${data.info.type}</span><br>
<b>Exploit:</b> ${data.info.exploit}<br>
<b>Narrative:</b> ${data.info.narrative}<br>
<b>Fix:</b> ${data.info.fix}`;

    // Buttons
    document.getElementById('btnNormal').classList.toggle('active',
        Math.abs(px - data.info.normal_point[0]) < 0.02 && Math.abs(py - data.info.normal_point[1]) < 0.02);
    document.getElementById('btnDark').classList.toggle('active',
        Math.abs(px - data.info.dark_point[0]) < 0.02 && Math.abs(py - data.info.dark_point[1]) < 0.02);

    drawHeatmaps();
}

init();
</script>
</body></html>"""

with open(OUT, 'w') as f:
    f.write(html)

print(f"\nExplorer: {OUT}")
print("Ready to open.")
