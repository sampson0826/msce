#!/usr/bin/env python3
"""
DeFi Dark Zone Scanner — Full Pipeline (MVP)

Solidity → Constraint Extraction → DSL Spec → Dark Zone Scan → Ranking → Report

Usage:
  cd /Users/dengxinhang/paper

  # Scan a protocol from its DSL spec:
  python3 constraint_residual/dsl/pipeline.py --spec dsl/protocols/the_dao.yaml

  # Extract constraints from Solidity, then scan:
  python3 constraint_residual/dsl/pipeline.py --solidity dsl/generated/the_dao.sol

  # Scan all three and generate unified report:
  python3 constraint_residual/dsl/pipeline.py --all
"""

import numpy as np
import json
import os
import re
import sys
import argparse
sys.path.insert(0, '/Users/dengxinhang/paper')
from constraint_residual.dsl.compiler import (
    load_protocol, scan_protocol, compile_field, metrics_at_point,
)
from constraint_residual.core import Rule, ConstraintField
from constraint_residual.dark_zone_detector import DarkZoneDetector

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
os.makedirs(OUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# Phase 1: Solidity → Constraint Extraction
# ═══════════════════════════════════════════════════════════════

REQUIRE_PATTERN = re.compile(
    r'require\s*\(\s*([^,;]+?)\s*(?:,\s*"[^"]*")?\s*\)\s*;',
    re.MULTILINE
)

VARIABLE_PATTERN = re.compile(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b')

CONSTRAINT_HEURISTICS = [
    # (pattern regex, constraint_type, layer, domain, fn_type, fn_params_builder)
    (r'balance|amount|token|deposit|withdraw', 'tokenomics', 'L1',
     'gaussian', lambda vars: {'center': [0.3, 0.5], 'width': [0.3, 0.4]}),
    (r'owner|onlyOwner|msg\.sender\s*==', 'access_control', 'L2',
     'sigmoid', lambda vars: {'axis': 'x', 'center': 0.3, 'width': 0.05}),
    (r'lock|reentran|guard|mutex', 'reentrancy', 'L1',
     'sigmoid', lambda vars: {'axis': 'x', 'center': 0.5, 'width': 0.05}),
    (r'init|initialized|constructor', 'lifecycle', 'L1',
     'sigmoid', lambda vars: {'axis': 'x', 'center': 0.5, 'width': 0.1}),
    (r'health|collateral|liquidat|solvency', 'risk', 'L2',
     'gaussian', lambda vars: {'center': [0.5, 0.5], 'width': [0.3, 0.3]}),
    (r'keeper|verify|signature|proof|cross.chain', 'cross_chain', 'L1',
     'gaussian', lambda vars: {'center': [0.7, 0.5], 'width': [0.3, 0.4]}),
    (r'price|oracle|feed', 'oracle', 'L2',
     'gaussian', lambda vars: {'center': [0.5, 0.5], 'width': [0.3, 0.3]}),
    (r'time|deadline|expir|delay|timelock', 'temporal', 'L2',
     'sigmoid', lambda vars: {'axis': 'y', 'center': 0.3, 'width': 0.1}),
]


def extract_requires(solidity_code):
    """Extract all require() statements with their conditions and line numbers."""
    results = []
    for m in REQUIRE_PATTERN.finditer(solidity_code):
        condition = m.group(1).strip()
        line_num = solidity_code[:m.start()].count('\n') + 1
        results.append({
            'condition': condition,
            'line': line_num,
        })
    return results


def classify_require(condition):
    """Heuristically classify a require condition into constraint parameters."""
    variables = VARIABLE_PATTERN.findall(condition)
    # Filter Solidity keywords
    keywords = {'require', 'assert', 'revert', 'if', 'else', 'for', 'while',
                'return', 'true', 'false', 'address', 'uint', 'int', 'bool',
                'bytes', 'string', 'mapping', 'struct', 'enum', 'contract',
                'function', 'modifier', 'event', 'public', 'private', 'internal',
                'external', 'view', 'pure', 'payable', 'memory', 'storage',
                'calldata', 'msg', 'block', 'tx', 'abi', 'now'}
    variables = [v for v in variables if v not in keywords and len(v) > 1]

    # Text for heuristic matching
    text = condition.lower()

    for pattern, domain, layer, fn_type, params_builder in CONSTRAINT_HEURISTICS:
        if re.search(pattern, text):
            return {
                'domain': domain,
                'layer': layer,
                'fn_type': fn_type,
                'params': params_builder(variables),
                'variables': variables,
            }

    # Default: generic constraint
    return {
        'domain': 'general',
        'layer': 'L1',
        'fn_type': 'gaussian',
        'params': {'center': [0.5, 0.5], 'width': [0.4, 0.4]},
        'variables': variables,
    }


def generate_dsl_spec(contract_name, solidity_code):
    """Generate a draft DSL YAML spec from Solidity require() statements."""
    requires = extract_requires(solidity_code)
    constraints = []

    for i, req in enumerate(requires):
        cls = classify_require(req['condition'])
        name = f"c{i+1}_{cls['domain']}"
        c_spec = {
            'name': name,
            'layer': cls['layer'],
            'domain': cls['domain'],
            'fn': cls['fn_type'],
        }
        c_spec.update(cls['params'])
        c_spec['description'] = f"Extracted from require(): {req['condition'][:100]} (line {req['line']})"

        if cls['fn_type'] == 'sigmoid':
            # Ensure sigmoid has required fields
            if 'axis' not in c_spec:
                c_spec['axis'] = 'x'
                c_spec['center'] = c_spec.get('center', 0.5)
                c_spec['width'] = c_spec.get('width', 0.1)
            # Remove gaussian fields
            c_spec.pop('center', None)
            c_spec.pop('width', None)
            c_spec['axis'] = c_spec.get('axis', 'x')
            c_spec['center'] = c_spec['params']['center'] if isinstance(c_spec.get('params', {}).get('center'), (int, float)) else 0.5
            c_spec['width'] = c_spec['params']['width'] if isinstance(c_spec.get('params', {}).get('width'), (int, float)) else 0.1
            c_spec.pop('params', None)
        else:
            if 'params' in c_spec:
                del c_spec['params']

        # Clean up: ensure gaussian has center/width, sigmoid has axis/center/width
        if c_spec['fn'] == 'sigmoid':
            if 'axis' not in c_spec:
                c_spec = {**c_spec, 'axis': 'x', 'center': 0.5, 'width': 0.1}
        elif c_spec['fn'] == 'gaussian':
            if 'center' not in c_spec:
                c_spec['center'] = [0.5, 0.5]
            if 'width' not in c_spec:
                c_spec['width'] = [0.4, 0.4]

        constraints.append(c_spec)

    # Build the full spec dict
    spec = {
        'protocol': contract_name,
        'description': f'Auto-extracted from Solidity source. {len(constraints)} constraints found.',
        'state_space': [
            {'name': 'x_dim', 'range': [0, 1], 'description': 'Primary state dimension (auto-extracted)'},
            {'name': 'y_dim', 'range': [0, 1], 'description': 'Secondary state dimension (auto-extracted)'},
        ],
        'constraints': constraints,
        'scan': {'cancellation_eps': 0.2, 'individual_min': 0.2},
        'dark_zone_type': 'unknown (auto-extracted)',
        'cancellation_signature': 'auto-detected',
    }
    return spec


# ═══════════════════════════════════════════════════════════════
# Phase 2: Dark Zone Ranking
# ═══════════════════════════════════════════════════════════════

def rank_dark_zones(field, dark_clusters, normal_point=np.array([0.5, 0.8])):
    """Rank dark zones by exploitability = ease_of_reach × damage_potential.

    ease_of_reach: inverse of state-space distance from normal operation
    damage_potential: combined constraint force drop when entering the dark zone
    """
    if not dark_clusters:
        return []

    normal_grad = field.constraint_gradient(normal_point)
    normal_force = float(np.linalg.norm(normal_grad))

    ranked = []
    for dc in dark_clusters:
        cz = dc.centroid if dc.centroid is not None else np.zeros(2)

        # Ease: how close is the dark zone to normal operation?
        distance = float(np.linalg.norm(cz - normal_point))
        max_possible_distance = np.sqrt(2)  # 2D unit square diagonal
        ease = 1.0 - (distance / max_possible_distance)

        # Damage: relative force drop compared to normal
        dz_force = float(np.linalg.norm(field.constraint_gradient(cz)))
        force_drop = max(0, normal_force - dz_force)
        damage = force_drop / (normal_force + 1e-10)

        # Exploitability score
        exploitability = ease * damage

        ranked.append({
            'cluster': dc,
            'ease': ease,
            'damage': damage,
            'exploitability': exploitability,
            'distance': distance,
        })

    ranked.sort(key=lambda r: r['exploitability'], reverse=True)
    return ranked


# ═══════════════════════════════════════════════════════════════
# Phase 3: Report Generation
# ═══════════════════════════════════════════════════════════════

def generate_report(protocol_name, field, spec, scan_result, ranked_zones,
                    solidity_path=None, extracted_from_solidity=False):
    """Generate an HTML report for a single protocol scan."""

    # Heatmaps
    xs = np.linspace(0, 1, 60)
    ys = np.linspace(0, 1, 60)
    hm_force = np.zeros((60, 60))
    hm_cancel = np.zeros((60, 60))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            p = np.array([x, y])
            grad = field.constraint_gradient(p)
            hm_force[j, i] = float(np.linalg.norm(grad))
            indiv = sum(float(np.linalg.norm(r.gradient(p))) for r in field.rules)
            hm_cancel[j, i] = float(np.linalg.norm(grad)) / indiv if indiv > 1e-10 else 1.0

    # Point metrics
    normal_pt = metrics_at_point(field, np.array([0.5, 0.8]), "Reference (normal operation)")
    dz_pts = []
    for r in ranked_zones[:3]:
        cz = r['cluster'].centroid
        if cz is not None:
            dz_pts.append(metrics_at_point(field, cz, f"DZ centroid ({cz[0]:.2f},{cz[1]:.2f})"))

    constraints_html = ""
    for c in spec.get('constraints', []):
        constraints_html += f"<tr><td>{c['name']}</td><td>{c.get('layer','?')}</td>"
        constraints_html += f"<td>{c.get('domain','?')}</td><td>{c.get('fn','?')}</td></tr>"

    ranking_html = ""
    if ranked_zones:
        ranking_html = """<h2>Dark Zone Ranking — Exploitability Score</h2>
<table>
<tr><th>Rank</th><th>Centroid</th><th>c(p)</th><th>Ease</th><th>Damage</th><th>Exploitability</th><th>Topology</th></tr>"""
        for i, r in enumerate(ranked_zones[:10]):
            dc = r['cluster']
            cz = dc.centroid
            color = '#dd6644' if r['exploitability'] > 0.3 else '#ddcc44' if r['exploitability'] > 0.1 else '#44dd66'
            ranking_html += f"""<tr>
<td>#{i+1}</td>
<td>({cz[0]:.3f}, {cz[1]:.3f})</td>
<td>{dc.mean_cancellation_ratio:.4f}</td>
<td>{r['ease']:.3f}</td><td>{r['damage']:.3f}</td>
<td style="color:{color};font-weight:bold">{r['exploitability']:.3f}</td>
<td>{dc.balance_topology}</td></tr>"""
        ranking_html += '</table>'
    else:
        ranking_html = '<p style="color:#889">No dark zones detected at current threshold.</p>'

    source_note = ""
    if extracted_from_solidity:
        source_note = f'<div class="callout" style="border-left-color:#ddcc44"><b>Constraints auto-extracted from Solidity.</b> Draft spec — auditor review recommended. {len(spec.get("constraints",[]))} require() statements mapped to constraint functions heuristically.</div>'
    else:
        source_note = f'<div class="callout" style="border-left-color:#44dd66"><b>Constraints loaded from verified DSL spec.</b> {len(spec.get("constraints",[]))} hand-modeled constraints.</div>'

    dark_zone_count = scan_result.n_dark_zones
    dz_status = "DARK ZONES FOUND" if dark_zone_count > 0 else "NO DARK ZONES DETECTED"
    dz_color = "#dd6644" if dark_zone_count > 0 else "#44dd66"

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="UTF-8">
<title>Dark Zone Scan: {protocol_name}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ background:#0a0a12; color:#c8c8d8; font-family:'SF Mono','Menlo',monospace; padding:24px 40px; line-height:1.5; }}
h1 {{ font-size:20px; color:#fff; }}
h2 {{ font-size:15px; color:#8899cc; margin:28px 0 10px; border-bottom:1px solid #1a1a3a; padding-bottom:6px; }}
.sub {{ color:#556; font-size:11px; margin-bottom:20px; }}
.row {{ display:flex; gap:16px; flex-wrap:wrap; margin:12px 0; }}
.card {{ background:#0e0e20; border:1px solid #1a1a3a; border-radius:8px; padding:16px; flex:1; min-width:300px; }}
.chart-wrap {{ position:relative; height:260px; }}
.metric {{ display:inline-block; background:#111133; border-radius:6px; padding:12px 18px; margin:6px; text-align:center; }}
.metric .val {{ font-size:22px; font-weight:bold; }}
.metric .lbl {{ font-size:10px; color:#556; }}
table {{ width:100%; border-collapse:collapse; font-size:11px; margin:8px 0; }}
th {{ background:#111133; color:#889; padding:8px 12px; text-align:left; font-weight:normal; }}
td {{ padding:8px 12px; border-bottom:1px solid #111122; }}
tr:hover td {{ background:#0f0f24; }}
.callout {{ background:#111122; border-left:3px solid #66aadd; padding:12px 16px; margin:12px 0; font-size:12px; border-radius:0 6px 6px 0; line-height:1.8; }}
footer {{ color:#444; font-size:10px; text-align:center; margin-top:32px; padding:16px; }}
</style></head>
<body>
<h1>DeFi Dark Zone Scanner</h1>
<div class="sub">Protocol: <b>{protocol_name}</b> | Constraints: {len(spec.get('constraints',[]))} | Source: {'Solidity extraction' if extracted_from_solidity else 'DSL spec'}</div>

{source_note}

<div>
<div class="metric"><div class="val" style="color:{dz_color}">{dark_zone_count}</div><div class="lbl">{dz_status}</div></div>
<div class="metric"><div class="val val-info">{len(spec.get('constraints',[]))}</div><div class="lbl">Constraints</div></div>
<div class="metric"><div class="val val-info">{len(ranked_zones)}</div><div class="lbl">Ranked zones</div></div>
</div>

<h2>Constraint Force Field — ||Π(p)||</h2>
<div class="row">
<div class="card" style="flex:1.5"><div class="chart-wrap"><canvas id="forceChart"></canvas></div></div>
<div class="card">
<h3>Reading the Map</h3>
<div style="font-size:10px;color:#889;line-height:1.7">
<p>Bright = strong combined constraint protection.</p>
<p>Dark = weak protection — potential dark zone.</p>
<p>Scan resolution: 60×60 grid over [0,1]² state space.</p>
</div></div></div>

<h2>Cancellation Ratio — c(p)</h2>
<div class="row">
<div class="card" style="flex:1.5"><div class="chart-wrap"><canvas id="cancelChart"></canvas></div></div>
<div class="card"><div style="font-size:10px;color:#889;line-height:1.7">
<p>Blue (c≈0) = dark zone — strong individual constraints, zero combined.</p>
<p>Red (c≈1) = no cancellation — constraints align or only one is active.</p>
</div></div></div>

{ranking_html}

<h2>Constraint Inventory</h2>
<table><tr><th>Name</th><th>Layer</th><th>Domain</th><th>Function Type</th></tr>{constraints_html}</table>

<h2>Key State-Space Points</h2>
<table><tr><th>Point</th><th>Position</th><th>||Π||</th><th>Σ||∇σ||</th><th>c(p)</th></tr>"""
    for pt in [normal_pt] + dz_pts:
        cr = pt['c_ratio']
        color = '#dd6644' if cr < 0.2 else '#ddcc44' if cr < 0.5 else '#44dd66'
        html += f"""<tr><td>{pt['label']}</td><td>({pt['position'][0]:.3f},{pt['position'][1]:.3f})</td>
<td>{pt['combined']:.3f}</td><td>{pt['total_indiv']:.3f}</td><td style="color:{color}">{cr:.4f}</td></tr>"""
    html += '</table>'

    # Charts
    html += f"""
<script>
const xs = {json.dumps(xs.tolist())};
const ys = {json.dumps(ys.tolist())};
const forceD = {json.dumps(hm_force.tolist())};
const cancelD = {json.dumps(hm_cancel.tolist())};

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
draw('forceChart', forceD, Math.max(...forceD.flat()), 'force');
draw('cancelChart', cancelD, 1.05, 'cancel');
</script>
<footer>DeFi Dark Zone Scanner · Pipeline MVP · 2026</footer>
</body></html>"""

    return html


# ═══════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════

def run_pipeline_from_spec(yaml_path, output_name=None):
    """Run full pipeline from a DSL YAML spec file."""
    field, spec = load_protocol(yaml_path)
    result = scan_protocol(field, spec, n_points=80)

    # Dark zone detection with ranking
    detector = DarkZoneDetector(
        cancellation_eps=spec.get('scan', {}).get('cancellation_eps', 0.2),
        individual_min=spec.get('scan', {}).get('individual_min', 0.2),
    )
    bounds = [(0, 1), (0, 1)]
    dark_clusters = detector.scan(field, bounds, n_points=80)
    ranked = rank_dark_zones(field, dark_clusters)

    name = output_name or spec.get('protocol', 'unknown')
    html = generate_report(name, field, spec, result, ranked,
                           extracted_from_solidity=False)
    report_path = os.path.join(OUT_DIR, f'{name}_report.html')
    with open(report_path, 'w') as f:
        f.write(html)

    return {
        'protocol': name,
        'n_constraints': len(spec.get('constraints', [])),
        'n_dark_zones': len(dark_clusters),
        'n_ranked': len(ranked),
        'top_exploitability': ranked[0]['exploitability'] if ranked else 0,
        'report': report_path,
        'dark_zone_type': spec.get('dark_zone_type', 'unknown'),
    }


def run_pipeline_from_solidity(sol_path, output_name=None):
    """Run full pipeline from a Solidity file."""
    with open(sol_path) as f:
        code = f.read()

    name = output_name or os.path.splitext(os.path.basename(sol_path))[0]
    spec = generate_dsl_spec(name, code)
    field = compile_field(spec)

    detector = DarkZoneDetector(cancellation_eps=0.2, individual_min=0.2)
    bounds = [(0, 1), (0, 1)]
    dark_clusters = detector.scan(field, bounds, n_points=60)
    ranked = rank_dark_zones(field, dark_clusters)

    # Create a scan-like result
    class SimpleResult:
        n_dark_zones = len(dark_clusters)
        dark_zone_centroids = [dc.centroid.tolist() if dc.centroid is not None else None for dc in dark_clusters]
        dark_zone_c_ratios = [dc.mean_cancellation_ratio for dc in dark_clusters]
        dark_zone_topologies = [dc.balance_topology for dc in dark_clusters]
    result = SimpleResult()

    html = generate_report(name, field, spec, result, ranked,
                           solidity_path=sol_path, extracted_from_solidity=True)
    report_path = os.path.join(OUT_DIR, f'{name}_report.html')
    with open(report_path, 'w') as f:
        f.write(html)

    return {
        'protocol': name,
        'n_constraints': len(spec.get('constraints', [])),
        'n_dark_zones': len(dark_clusters),
        'n_ranked': len(ranked),
        'top_exploitability': ranked[0]['exploitability'] if ranked else 0,
        'report': report_path,
        'dark_zone_type': 'auto-extracted',
    }


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeFi Dark Zone Scanner Pipeline')
    parser.add_argument('--spec', help='Path to DSL YAML spec')
    parser.add_argument('--solidity', help='Path to Solidity contract')
    parser.add_argument('--all', action='store_true', help='Run on all known protocols')
    args = parser.parse_args()

    results = []

    if args.all:
        dsl_dir = os.path.dirname(os.path.abspath(__file__))
        proto_dir = os.path.join(dsl_dir, 'protocols')
        for name in ['the_dao', 'parity_wallet', 'poly_network']:
            yp = os.path.join(proto_dir, f'{name}.yaml')
            if os.path.exists(yp):
                r = run_pipeline_from_spec(yp)
                results.append(r)
    elif args.spec:
        r = run_pipeline_from_spec(args.spec)
        results.append(r)
    elif args.solidity:
        r = run_pipeline_from_solidity(args.solidity)
        results.append(r)
    else:
        # Default: run on all DSL specs
        dsl_dir = os.path.dirname(os.path.abspath(__file__))
        proto_dir = os.path.join(dsl_dir, 'protocols')
        for name in ['the_dao', 'parity_wallet', 'poly_network']:
            yp = os.path.join(proto_dir, f'{name}.yaml')
            if os.path.exists(yp):
                r = run_pipeline_from_spec(yp)
                results.append(r)

    print(f"\n{'='*60}")
    print(f"DeFi Dark Zone Scanner — Pipeline Results")
    print(f"{'='*60}")
    for r in results:
        print(f"\n  {r['protocol']} ({r['dark_zone_type']}):")
        print(f"    Constraints: {r['n_constraints']}")
        print(f"    Dark zones:  {r['n_dark_zones']}")
        print(f"    Ranked:      {r['n_ranked']}")
        if r['top_exploitability'] > 0:
            print(f"    Top exploitability: {r['top_exploitability']:.3f}")
        print(f"    Report:      {r['report']}")

    # Open all reports
    for r in results:
        os.system(f"open {r['report']}")
