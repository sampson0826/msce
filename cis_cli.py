#!/usr/bin/env python3
"""
CIS — Constraint Invisibility Scanner CLI

Phase 1 product: DeFi Dark Zone Scanner

Usage:
  cis scan <contract.sol>              Scan a Solidity contract
  cis scan <directory/>                Scan all .sol files in directory
  cis scan --dsl <protocol.yaml>       Analyze a DSL protocol definition
  cis compare <a.sol> <b.sol>          Compare two contracts
  cis compare --dsl <a.yaml> <b.yaml>  Compare two DSL protocols
  cis report                            Generate HTML report from scan cache
  cis score <contract.sol>              Quick security score only
  cis domains <contract.sol>            Show domain coverage analysis

Examples:
  cis scan contracts/Pool.sol
  cis scan contracts/
  cis scan --dsl dsl/protocols/aave_v3.yaml
  cis compare --dsl dsl/protocols/aave_v3.yaml dsl/protocols/aave_v3_fixed.yaml
"""

import sys, os, json, time, argparse, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from dataclasses import dataclass, field as dc_field
from typing import Optional

from constraint_residual.core import ConstraintField, Rule
from constraint_residual.cis_core import CISAnalyzer
from constraint_residual.dark_zone_detector import DarkZoneDetector
from constraint_residual.solidity_extractor import extract_constraints_from_contract
from constraint_residual.dsl.compiler import load_protocol

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cis_cache")
REPORT_HTML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cis_report.html")

ALL_DOMAINS = ['access_control', 'oracle', 'tokenomics', 'security', 'risk', 'accounting', 'general']

# ═══════════════════════════════════════════════════════════════
# Core scanning logic
# ═══════════════════════════════════════════════════════════════

def scan_contract(filepath: str, n_grid: int = 64, max_per_domain: int = 3) -> dict:
    """Full A.1-A.7 scan of a Solidity file."""
    fname = os.path.basename(filepath)
    t0 = time.time()

    field = extract_constraints_from_contract(filepath, max_per_domain=max_per_domain)
    if not field.rules:
        return {'error': f'No require() statements found in {fname}', 'rules': 0, 'file': fname}

    analyzer = CISAnalyzer(field, bounds=[(0, 1), (0, 1)], n_points=n_grid)
    report = analyzer.full_analysis(fname)

    detector = DarkZoneDetector(cancellation_eps=0.15, individual_min=0.25)
    dz = detector.scan(field, [(0, 1), (0, 1)], n_points=n_grid)

    domains_present = sorted(set(r.domain for r in field.rules))
    domains_missing = sorted(set(ALL_DOMAINS) - set(domains_present))

    score = _compute_score(report, domains_missing, len(dz))

    return {
        'file': fname,
        'path': filepath,
        'rules': len(field.rules),
        'domains_present': domains_present,
        'domains_missing': domains_missing,
        'coverage': len(domains_present) / len(ALL_DOMAINS),
        'dark_zones': len(dz),
        'dz_details': [{'centroid': (float(d.centroid[0]), float(d.centroid[1])),
                         'c_ratio': float(d.mean_cancellation_ratio),
                         'topology': d.balance_topology} for d in dz],
        'e1_fraction': report.e1_fraction,
        'e2_fraction': report.e2_fraction,
        'e3_fraction': report.e3_fraction,
        'unconstrained': report.unconstrained_fraction,
        'structural_score': report.structural_score,
        'max_condition': float(np.max(report.riemannian.condition_number)),
        'score': score,
        'elapsed': time.time() - t0,
        'has_structural_gap': report.helmholtz.has_structural_gap,
    }


def scan_dsl(yaml_path: str, n_grid: int = 64) -> dict:
    """Full A.1-A.7 scan of a DSL YAML protocol definition."""
    fname = os.path.basename(yaml_path)
    t0 = time.time()

    field, spec = load_protocol(yaml_path)
    if not field.rules:
        return {'error': f'No constraints in {fname}', 'rules': 0}

    analyzer = CISAnalyzer(field, bounds=[(0, 1), (0, 1)], n_points=n_grid)
    report = analyzer.full_analysis(fname)

    detector = DarkZoneDetector(cancellation_eps=0.15, individual_min=0.25)
    dz = detector.scan(field, [(0, 1), (0, 1)], n_points=n_grid)

    domains_present = sorted(set(r.domain for r in field.rules))
    domains_missing = sorted(set(ALL_DOMAINS) - set(domains_present))

    score = _compute_score(report, domains_missing, len(dz))

    return {
        'file': fname,
        'path': yaml_path,
        'rules': len(field.rules),
        'domains_present': domains_present,
        'domains_missing': domains_missing,
        'coverage': len(domains_present) / len(ALL_DOMAINS),
        'dark_zones': len(dz),
        'dz_details': [{'centroid': (float(d.centroid[0]), float(d.centroid[1])),
                         'c_ratio': float(d.mean_cancellation_ratio),
                         'topology': d.balance_topology} for d in dz],
        'e1_fraction': report.e1_fraction,
        'e2_fraction': report.e2_fraction,
        'e3_fraction': report.e3_fraction,
        'unconstrained': report.unconstrained_fraction,
        'structural_score': report.structural_score,
        'max_condition': float(np.max(report.riemannian.condition_number)),
        'score': score,
        'elapsed': time.time() - t0,
        'has_structural_gap': report.helmholtz.has_structural_gap,
    }


def _compute_score(report, domains_missing: list, dz_count: int) -> float:
    """CIS Security Score: 0-100."""
    score = 100.0
    score -= len(domains_missing) * 8.0
    score -= report.e1_fraction * 500.0
    score -= report.unconstrained_fraction * 30.0
    score -= dz_count * 5.0
    max_cond = float(np.max(report.riemannian.condition_number))
    if max_cond > 1000:
        score -= 5.0 * np.log10(max_cond / 1000)
    score += report.structural_score * 10.0
    return round(max(0.0, min(100.0, score)), 1)


# ═══════════════════════════════════════════════════════════════
# Output formatting
# ═══════════════════════════════════════════════════════════════

RISK_COLORS = {
    'critical': '\033[91m',  # red
    'high': '\033[93m',      # yellow
    'medium': '\033[96m',    # cyan
    'low': '\033[92m',       # green
    'reset': '\033[0m',
}

def _risk_level(score: float) -> str:
    if score < 25: return 'critical'
    if score < 50: return 'high'
    if score < 70: return 'medium'
    return 'low'

def print_result(r: dict):
    """Pretty-print a scan result."""
    if 'error' in r:
        print(f"\n{'─'*60}")
        print(f"  Target: {r['file']}")
        print(f"  {RISK_COLORS['critical']}Error: {r['error']}{RISK_COLORS['reset']}")
        return
    risk = _risk_level(r['score'])
    c = RISK_COLORS.get(risk, '')
    r_c = RISK_COLORS['reset']

    print(f"\n{'─'*60}")
    print(f"  Target: {r['file']}")
    print(f"  Rules extracted: {r['rules']}")
    print(f"  Domains covered: {r['domains_present']} ({r['coverage']:.0%})")
    if r['domains_missing']:
        print(f"  {c}Domains missing: {r['domains_missing']}{r_c}")
    print(f"  Dark zones: {r['dark_zones']}")
    for d in r.get('dz_details', []):
        print(f"    → ({d['centroid'][0]:.3f}, {d['centroid'][1]:.3f}) "
              f"c(p)={d['c_ratio']:.4f} [{d['topology']}]")
    print(f"  E-I (structural):  {r['e1_fraction']:.1%}")
    print(f"  E-II (scalar):     {r['e2_fraction']:.1%}")
    print(f"  E-III (boundary):  {r['e3_fraction']:.1%}")
    print(f"  Unconstrained:     {r['unconstrained']:.1%}")
    print(f"  Max condition #:   {r['max_condition']:.1f}")
    print(f"  Structural score:  {r['structural_score']:.4f}")
    print(f"  {c}CIS Security Score: {r['score']}/100 [{risk.upper()}]{r_c}")
    print(f"  Elapsed: {r['elapsed']:.1f}s")


def print_compare(r1: dict, r2: dict):
    """Pretty-print a comparison table."""
    print(f"\n{'='*70}")
    print(f"  CIS Comparative Analysis")
    print(f"{'='*70}")
    print(f"  {'Metric':<35} {r1['file']:<20} {r2['file']:<20}")
    print(f"  {'-'*65}")

    pairs = [
        ('Rules', 'rules'),
        ('Domains covered', lambda r: f"{len(r['domains_present'])}/7"),
        ('Domains missing', lambda r: ','.join(r['domains_missing']) if r['domains_missing'] else '—'),
        ('Dark zones', 'dark_zones'),
        ('E-I (structural)', lambda r: f"{r['e1_fraction']:.1%}"),
        ('Unconstrained', lambda r: f"{r['unconstrained']:.1%}"),
        ('Max condition #', lambda r: f"{r['max_condition']:.1f}"),
        ('Structural score', lambda r: f"{r['structural_score']:.4f}"),
    ]

    for label, key in pairs:
        if callable(key):
            v1, v2 = key(r1), key(r2)
        else:
            v1, v2 = r1[key], r2[key]
        print(f"  {label:<35} {str(v1):<20} {str(v2):<20}")

    c1 = RISK_COLORS.get(_risk_level(r1['score']), '')
    c2 = RISK_COLORS.get(_risk_level(r2['score']), '')
    r_c = RISK_COLORS['reset']
    print(f"  {'CIS Security Score':<35} {c1}{r1['score']}/100{r_c:<14} {c2}{r2['score']}/100{r_c}")

    diff = r2['score'] - r1['score']
    if diff > 0:
        print(f"\n  → {r2['file']} scores {diff:.1f} points HIGHER")
    elif diff < 0:
        print(f"\n  → {r1['file']} scores {abs(diff):.1f} points HIGHER")
    else:
        print(f"\n  → Scores tied")


# ═══════════════════════════════════════════════════════════════
# Batch scan + HTML report
# ═══════════════════════════════════════════════════════════════

def scan_directory(dirpath: str) -> list:
    """Scan all .sol files in a directory."""
    results = []
    for fpath in sorted(glob.glob(os.path.join(dirpath, '*.sol'))):
        print(f"\n  Scanning {os.path.basename(fpath)}...")
        r = scan_contract(fpath)
        print_result(r)
        results.append(r)
    return results


def generate_html_report(results: list, outpath: str = None):
    """Generate HTML report from scan results."""
    if outpath is None:
        outpath = REPORT_HTML

    rows = ''
    for r in results:
        if 'error' in r:
            rows += f"""<tr>
<td><b>{r['file']}</b></td>
<td style="color:#dd4466">ERROR</td><td>—</td><td>0</td><td>0/7</td><td>—</td><td>—</td><td>—</td><td>—</td>
</tr>"""
            continue
        risk = _risk_level(r['score'])
        color = {'critical': '#dd4466', 'high': '#ddcc44', 'medium': '#66aadd', 'low': '#44dd66'}.get(risk, '#889')
        rows += f"""<tr>
<td><b>{r['file']}</b></td>
<td style="color:{color};font-weight:bold">{r['score']}/100</td>
<td style="color:{color}">{risk.upper()}</td>
<td>{r['rules']}</td>
<td>{len(r['domains_present'])}/7</td>
<td>{r['dark_zones']}</td>
<td>{r['e1_fraction']:.1%}</td>
<td>{r['unconstrained']:.1%}</td>
<td>{r['max_condition']:.1f}</td>
</tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8">
<title>CIS — Constraint Invisibility Scanner</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ background:#0a0a12; color:#c8c8d8; font-family:'SF Mono','Menlo',monospace; padding:24px 32px; line-height:1.5; }}
h1 {{ font-size:18px; color:#fff; margin-bottom:4px; }}
h2 {{ font-size:14px; color:#8899cc; margin:24px 0 8px; }}
.sub {{ color:#556; font-size:10px; margin-bottom:20px; }}
table {{ width:100%; border-collapse:collapse; font-size:11px; }}
th {{ background:#111133; color:#889; padding:8px 12px; text-align:left; font-weight:normal; }}
td {{ padding:8px 12px; border-bottom:1px solid #111122; }}
tr:hover td {{ background:#0f0f24; }}
.bar {{ display:inline-block; height:10px; border-radius:2px; margin-right:4px; }}
.bar-critical {{ background:#dd4466; }}
.bar-high {{ background:#ddcc44; }}
.bar-medium {{ background:#66aadd; }}
.bar-low {{ background:#44dd66; }}
.callout {{ background:#111122; border-left:3px solid #66aadd; padding:10px 14px; margin:8px 0; font-size:11px; border-radius:0 4px 4px 0; }}
</style></head>
<body>

<h1>CIS — Constraint Invisibility Scanner</h1>
<div class="sub">A.1-A.7 Full Pipeline · DeFi Protocol Security Assessment</div>

<h2>Scan Results ({len(results)} targets)</h2>
<table>
<tr><th>Protocol</th><th>Score</th><th>Risk</th><th>Rules</th><th>Domains</th><th>DZ</th><th>E-I</th><th>Unconstrained</th><th>Max Cond</th></tr>
{rows}
</table>

<h2>Score Distribution</h2>
{_score_bars_html(results)}

<div class="callout">
<b>Legend:</b><br>
<b>CIS Security Score</b>: 0-100 scale. &lt;25 = CRITICAL, 25-50 = HIGH, 50-70 = MEDIUM, &gt;70 = LOW risk<br>
<b>Dark Zones (DZ)</b>: regions where c(p)→0 — protection cancels out<br>
<b>E-I</b>: structural gaps requiring new constraint types (unfixable by tuning)<br>
<b>Unconstrained</b>: fraction of parameter space with λ_min(g)→0<br>
<b>Max Cond</b>: condition number of Riemannian metric — instability indicator
</div>

</body></html>"""

    with open(outpath, 'w') as f:
        f.write(html)
    return outpath


def _score_bars_html(results: list) -> str:
    bars = ''
    valid = [r for r in results if 'score' in r]
    for r in sorted(valid, key=lambda x: x['score'], reverse=True):
        cls = 'bar-' + _risk_level(r['score'])
        w = max(1, int(r['score'] / 2))
        bars += f'<div style="margin:4px 0;font-size:10px">'
        bars += f'{r["file"]:<40s} '
        bars += f'<span class="bar {cls}" style="width:{w}px"></span> '
        bars += f'<b>{r["score"]:.0f}/100</b></div>\n'
    return bars


# ═══════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='CIS — Constraint Invisibility Scanner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    subs = parser.add_subparsers(dest='cmd')

    # cis scan
    scan_p = subs.add_parser('scan', help='Scan a contract or directory')
    scan_p.add_argument('target', help='Solidity file, directory, or --dsl YAML')
    scan_p.add_argument('--dsl', action='store_true', help='Target is a DSL YAML file')
    scan_p.add_argument('--grid', type=int, default=64, help='Grid resolution (default: 64)')
    scan_p.add_argument('--html', action='store_true', help='Generate HTML report')

    # cis compare
    cmp_p = subs.add_parser('compare', help='Compare two protocols')
    cmp_p.add_argument('target_a', help='First contract/YAML')
    cmp_p.add_argument('target_b', help='Second contract/YAML')
    cmp_p.add_argument('--dsl', action='store_true', help='Targets are DSL YAML files')

    # cis report
    rep_p = subs.add_parser('report', help='Generate HTML report from scan cache')

    # cis score
    score_p = subs.add_parser('score', help='Quick security score only')
    score_p.add_argument('target', help='Solidity file')
    score_p.add_argument('--dsl', action='store_true', help='Target is a DSL YAML file')

    args = parser.parse_args()

    if args.cmd == 'scan':
        if args.dsl:
            r = scan_dsl(args.target, n_grid=args.grid)
        elif os.path.isdir(args.target):
            results = scan_directory(args.target)
            if args.html:
                out = generate_html_report(results)
                print(f"\nReport: {out}")
            return
        else:
            r = scan_contract(args.target, n_grid=args.grid)
        print_result(r)
        if args.html and not args.dsl and not os.path.isdir(args.target):
            out = generate_html_report([r])
            print(f"\nReport: {out}")

    elif args.cmd == 'compare':
        if args.dsl:
            r1, r2 = scan_dsl(args.target_a), scan_dsl(args.target_b)
        else:
            r1, r2 = scan_contract(args.target_a), scan_contract(args.target_b)
        print_compare(r1, r2)

    elif args.cmd == 'score':
        if args.dsl:
            r = scan_dsl(args.target)
        else:
            r = scan_contract(args.target)
        risk = _risk_level(r['score'])
        c = RISK_COLORS.get(risk, '')
        print(f"{c}{r['score']}/100 [{risk.upper()}]{RISK_COLORS['reset']} — {r['file']}")

    elif args.cmd == 'report':
        # Scan cached results or contracts directory
        contracts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'contracts')
        if os.path.isdir(contracts_dir):
            results = scan_directory(contracts_dir)
        else:
            print("No contracts directory found and no scan cache.")
            return
        out = generate_html_report(results)
        print(f"\nReport: {out}")
        os.system(f"open {out}")

    else:
        # Default: scan the contracts directory
        contracts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'contracts')
        if os.path.isdir(contracts_dir):
            print("CIS — Constraint Invisibility Scanner\n")
            print("Usage: cis {scan,compare,report,score} ...")
            print(f"\nFound contracts directory with {len(glob.glob(contracts_dir+'/*.sol'))} files.")
            print("Run 'cis scan contracts/' to analyze all contracts.\n")
            results = scan_directory(contracts_dir)
            out = generate_html_report(results)
            print(f"\nReport: {out}")
            os.system(f"open {out}")
        else:
            parser.print_help()


if __name__ == '__main__':
    main()
