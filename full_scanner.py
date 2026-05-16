#!/usr/bin/env python3
"""
Full Protocol Security Scanner — AST + CIS Combined Pipeline

For a given protocol (set of Solidity contracts):
  1. AST-level extraction: functions, modifiers, validation calls, require patterns
  2. Validation gap detection: missing checkX calls, missing require themes
  3. CIS A.1-A.7 parameter-space analysis
  4. Cross-contract constraint flow mapping
  5. Composite security assessment

Usage:
  python3 full_scanner.py <protocol_name> <contract_dir_or_files...>
"""

import sys, os, json
sys.path.insert(0, '/Users/dengxinhang/paper')
import numpy as np
from dataclasses import dataclass, field
from constraint_residual.ast_extractor import (
    analyze_protocol, analyze_contract, print_protocol_report
)
from constraint_residual.core import ConstraintField, Rule
from constraint_residual.solidity_extractor import extract_constraints_from_contract
from constraint_residual.cis_core import CISAnalyzer
from constraint_residual.dark_zone_detector import DarkZoneDetector

ALL_DOMAINS = ['access_control', 'oracle', 'tokenomics', 'security', 'risk', 'accounting', 'general']


@dataclass
class FullScanResult:
    protocol: str
    contracts: list
    ast_result: dict
    total_functions: int
    total_state_changers: int
    validation_gaps: list
    require_gaps: list
    unguarded_functions: list
    cis_rules: int
    cis_domains_present: list
    cis_domains_missing: list
    cis_dark_zones: int
    cis_unconstrained: float
    cis_e1_fraction: float
    cis_max_condition: float
    cis_score: float
    risk_level: str


def full_scan(protocol_name: str, contract_paths: list[str], n_grid: int = 64) -> FullScanResult:
    """Run the complete AST + CIS pipeline on a protocol."""

    # ═══ Phase 1: AST-level constraint extraction ═══
    ast_result = analyze_protocol(contract_paths)

    total_funcs = sum(len(a.functions) for a in ast_result['contracts'].values())
    total_sc = sum(a.flow['total_state_changers'] for a in ast_result['contracts'].values())

    # Collect unguarded functions (only external/public — internal/private not in attack surface)
    # Also skip post-validation callbacks (Compound v2 *Verify convention) and
    # proxy passthrough contracts (*Delegator pattern — delegate all logic to impl)
    CALLBACK_SUFFIXES = ['Verify']
    PROXY_SUFFIXES = ['Delegator']
    unguarded = []
    for name, a in ast_result['contracts'].items():
        # Skip proxy passthrough contracts entirely
        if any(name.endswith(s) for s in PROXY_SUFFIXES):
            continue
        for f in a.functions:
            if f['has_state_change'] and f.get('visibility') in ('public', 'external'):
                if any(f['name'].endswith(s) for s in CALLBACK_SUFFIXES):
                    continue
                checks = (len(f['requires']) + len(f['validations'])
                         + len(f['weak_checks']) + len(f['modifiers'])
                         + len(f.get('cross_checks', [])))
                # Internal passthrough (delegates to *Internal/*Fresh)
                has_delegated = (f.get('internal_calls') and len(f.get('requires', [])) == 0
                                and len(f.get('validations', [])) == 0)
                if checks == 0 and not has_delegated:
                    unguarded.append(f'{name}.{f["name"]}')

    # ═══ Phase 2: CIS parameter-space analysis ═══
    all_rules = []
    for fpath in contract_paths:
        if not os.path.exists(fpath):
            continue
        try:
            field = extract_constraints_from_contract(fpath, max_per_domain=5)
            all_rules.extend(field.rules)
        except Exception:
            pass

    if all_rules:
        combined = ConstraintField(rules=all_rules)
        analyzer = CISAnalyzer(combined, bounds=[(0, 1), (0, 1)], n_points=n_grid)
        report = analyzer.full_analysis(protocol_name)

        detector = DarkZoneDetector(cancellation_eps=0.15, individual_min=0.25)
        dz = detector.scan(combined, [(0, 1), (0, 1)], n_points=n_grid)

        domains_present = sorted(set(r.domain for r in combined.rules))
        domains_missing = sorted(set(ALL_DOMAINS) - set(domains_present))

        score = 100.0
        score -= len(domains_missing) * 8.0
        score -= report.e1_fraction * 500.0
        score -= report.unconstrained_fraction * 30.0
        score -= len(dz) * 5.0
        max_cond = float(np.max(report.riemannian.condition_number))
        if max_cond > 1000:
            score -= 5.0 * np.log10(max_cond / 1000)
        score += report.structural_score * 10.0
        score = round(max(0.0, min(100.0, score)), 1)

        if score < 25:
            risk = 'CRITICAL'
        elif score < 50:
            risk = 'HIGH'
        elif score < 70:
            risk = 'MEDIUM'
        else:
            risk = 'LOW'
    else:
        report = None
        dz = []
        domains_present = []
        domains_missing = ALL_DOMAINS
        score = 0.0
        risk = 'UNKNOWN'
        max_cond = float('inf')

    return FullScanResult(
        protocol=protocol_name,
        contracts=[os.path.basename(p) for p in contract_paths],
        ast_result=ast_result,
        total_functions=total_funcs,
        total_state_changers=total_sc,
        validation_gaps=ast_result['validation_gaps'],
        require_gaps=ast_result['require_gaps'],
        unguarded_functions=unguarded,
        cis_rules=len(all_rules),
        cis_domains_present=domains_present,
        cis_domains_missing=domains_missing,
        cis_dark_zones=len(dz),
        cis_unconstrained=report.unconstrained_fraction if report else 1.0,
        cis_e1_fraction=report.e1_fraction if report else 0,
        cis_max_condition=max_cond,
        cis_score=score,
        risk_level=risk,
    )


def print_full_report(result: FullScanResult):
    """Comprehensive security assessment report."""
    c_red = '\033[91m'
    c_yel = '\033[93m'
    c_cya = '\033[96m'
    c_grn = '\033[92m'
    c_rst = '\033[0m'

    risk_color = {'CRITICAL': c_red, 'HIGH': c_yel, 'MEDIUM': c_cya, 'LOW': c_grn}.get(result.risk_level, c_rst)

    print(f"\n{'='*70}")
    print(f"  FULL SECURITY SCAN: {result.protocol}")
    print(f"{'='*70}")
    print(f"  Contracts: {result.contracts}")
    print(f"  Total functions: {result.total_functions} "
          f"({result.total_state_changers} state-changing)")

    # ── AST Findings ──
    print(f"\n  ── Code-Level Constraint Analysis ──")

    high_gaps = [g for g in result.validation_gaps if g['severity'] == 'HIGH']
    high_rgaps = [g for g in result.require_gaps if g['severity'] == 'HIGH']

    if high_gaps:
        print(f"\n  {c_red}⚠ VALIDATION CALL GAPS (HIGH): {len(high_gaps)}{c_rst}")
        for g in high_gaps:
            print(f"    {c_red}[{g['severity']}]{c_rst} {g['contract']}: "
                  f"'{g['validation_function']}' called by {g['called_by']}")
            print(f"         MISSING from: {g['missing_from']}")

    if high_rgaps:
        print(f"\n  {c_red}⚠ REQUIRE PATTERN GAPS (HIGH): {len(high_rgaps)}{c_rst}")
        for g in high_rgaps:
            print(f"    {c_red}[{g['severity']}]{c_rst} {g['contract']}: "
                  f"'{g['require_theme']}' enforced by {g['enforced_by']}")
            print(f"         MISSING from: {g['missing_from']}")

    med_gaps = [g for g in result.validation_gaps if g['severity'] != 'HIGH']
    med_rgaps = [g for g in result.require_gaps if g['severity'] != 'HIGH']

    if med_gaps or med_rgaps:
        print(f"\n  {c_yel}⚠ MEDIUM severity gaps: {len(med_gaps) + len(med_rgaps)}{c_rst}")
        for g in med_gaps + med_rgaps:
            key = g.get('validation_function') or g.get('require_theme')
            print(f"    {g['contract']}: '{key}' missing from {g['missing_from']}")

    if result.unguarded_functions:
        print(f"\n  {c_yel}⚠ UNGUARDED (no checks at all): {len(result.unguarded_functions)}{c_rst}")
        for u in result.unguarded_functions:
            print(f"    {u}")

    # ── CIS Findings ──
    print(f"\n  ── Parameter-Space CIS Analysis ──")
    print(f"  CIS Rules extracted:     {result.cis_rules}")
    print(f"  Domains present:         {result.cis_domains_present}")
    if result.cis_domains_missing:
        print(f"  {c_red}Domains missing:          {result.cis_domains_missing}{c_rst}")
    print(f"  Dark zones:              {result.cis_dark_zones}")
    print(f"  Unconstrained space:     {result.cis_unconstrained:.1%}")
    print(f"  E-I (structural):        {result.cis_e1_fraction:.1%}")
    print(f"  Max condition #:         {result.cis_max_condition:.1f}")

    # ── Overall Assessment ──
    print(f"\n  ── Overall Assessment ──")
    print(f"  {risk_color}CIS Security Score: {result.cis_score}/100 [{result.risk_level}]{c_rst}")

    # Risk explanation
    if result.risk_level in ('CRITICAL', 'HIGH'):
        print(f"\n  Risk factors:")
        if result.cis_domains_missing:
            print(f"    • Missing constraint domains: {result.cis_domains_missing}")
        if high_gaps:
            print(f"    • Validation call gaps: {len(high_gaps)} HIGH severity")
        if result.cis_unconstrained > 0.2:
            print(f"    • {result.cis_unconstrained:.0%} of parameter space unconstrained")
        if result.cis_dark_zones > 0:
            print(f"    • {result.cis_dark_zones} dark zone(s) detected")


# ═══════════════════════════════════════════════════════════════
# Batch scanning + discovery
# ═══════════════════════════════════════════════════════════════

def discover_protocols(base_dir: str) -> dict[str, list[str]]:
    """Walk contracts directory and discover all protocols.

    Each subdirectory is one protocol. Top-level .sol files are treated
    as separate single-contract protocols. Nested subdirectories (e.g.
    contracts/discovered/Benqi_avalanche/) are also discovered. Empty files skipped.
    """
    base_dir = os.path.abspath(base_dir)
    protocols = {}

    def collect_sol(dirpath: str) -> list[str]:
        return sorted([
            os.path.join(dirpath, f)
            for f in os.listdir(dirpath)
            if f.endswith('.sol') and os.path.getsize(os.path.join(dirpath, f)) > 0
        ])

    for entry in sorted(os.listdir(base_dir)):
        full_path = os.path.join(base_dir, entry)
        if os.path.isdir(full_path):
            sol_files = collect_sol(full_path)
            if sol_files:
                protocols[entry] = sol_files
            # Recurse one level for nested dirs (e.g. contracts/discovered/Benqi_avalanche/)
            # Skip known non-protocol subdirectories (test/, script/, lib/, audit/, node_modules/)
            NON_PROTOCOL_DIRS = {'test', 'script', 'lib', 'audit', 'node_modules', 'broadcast', 'cache', 'out', 'forge-std'}
            for subentry in sorted(os.listdir(full_path)):
                if subentry in NON_PROTOCOL_DIRS:
                    continue
                sub_path = os.path.join(full_path, subentry)
                if os.path.isdir(sub_path):
                    sub_files = collect_sol(sub_path)
                    if sub_files:
                        protocols[subentry] = sub_files
        elif entry.endswith('.sol') and os.path.getsize(full_path) > 0:
            protocol_name = entry.replace('.sol', '')
            protocols[protocol_name] = [full_path]

    return protocols


def batch_scan(base_dir: str, n_grid: int = 64) -> list[FullScanResult]:
    """Discover and scan all protocols in base_dir/discovered/. Returns ranked results.

    Reference protocols in the parent directory are excluded from batch scans.
    Use full_scan() directly to analyze specific reference protocols.
    """
    discovered_dir = os.path.join(base_dir, 'discovered')
    if os.path.isdir(discovered_dir):
        protocols = discover_protocols(discovered_dir)
    else:
        protocols = discover_protocols(base_dir)
    if not protocols:
        print(f"No protocols found in {base_dir}")
        return []

    results = []
    for name, paths in protocols.items():
        print(f"\n  Scanning {name} ({len(paths)} file{'s' if len(paths) > 1 else ''})...")
        try:
            result = full_scan(name, paths, n_grid=n_grid)
            results.append(result)
            print(f"    Score: {result.cis_score:.1f} [{result.risk_level}], "
                  f"{len([g for g in result.validation_gaps if g['severity']=='HIGH'])} HIGH gaps, "
                  f"{len(result.unguarded_functions)} unguarded")
        except Exception as e:
            print(f"    ERROR: {e}")
    return results


# ═══════════════════════════════════════════════════════════════
# Exploitability ranking
# ═══════════════════════════════════════════════════════════════

def compute_exploitability_score(result: FullScanResult) -> float:
    """Composite score ranking protocols by likelihood of finding an exploitable bug.

    High-weight signals: validation call gaps (Euler-class), unguarded functions.
    Medium-weight: oracle domain missing + gap combo.
    Low-weight: CIS structural metrics.
    """
    s = 0.0

    high_val_gaps = len([g for g in result.validation_gaps if g.get('severity') == 'HIGH'])
    high_req_gaps = len([g for g in result.require_gaps if g.get('severity') == 'HIGH'])
    med_gaps = len([g for g in result.validation_gaps + result.require_gaps
                    if g.get('severity') == 'MEDIUM'])

    s += high_val_gaps * 40.0       # Euler-class: validation call missing
    s += high_req_gaps * 25.0       # Require theme gap (oracle etc.)
    s += med_gaps * 10.0
    s += len(result.unguarded_functions) * 15.0

    # Oracle gap + validation gap combo = systemic blind spot (compound-class)
    if (high_val_gaps > 0 or high_req_gaps > 0) and 'oracle' in result.cis_domains_missing:
        s += 25.0

    # CIS structural (lower weight — harder to action)
    s += (1.0 - result.cis_score / 100.0) * 20.0
    s += result.cis_unconstrained * 30.0
    s += result.cis_dark_zones * 8.0
    s += len(result.cis_domains_missing) * 5.0

    return round(s, 1)


def generate_investigation_leads(results: list[FullScanResult], top_n: int = 5) -> list[dict]:
    """Rank findings and annotate top leads with investigation guidance."""
    ranked = sorted(results, key=compute_exploitability_score, reverse=True)
    leads = []
    for r in ranked[:top_n]:
        lead = {
            'protocol': r.protocol,
            'contracts': r.contracts,
            'exploitability_score': compute_exploitability_score(r),
            'cis_score': r.cis_score,
            'risk_level': r.risk_level,
            'patterns': [],
            'action_items': [],
        }
        for g in r.validation_gaps:
            if g.get('severity') == 'HIGH':
                lead['patterns'].append({
                    'type': 'euler_class_missing_validation',
                    'validation_fn': g.get('validation_function', g.get('require_theme', '?')),
                    'called_by': g.get('called_by', []),
                    'missing_from': g.get('missing_from', []),
                })
                lead['action_items'].append(
                    f"Manual review: does skipping {g.get('validation_function')} in "
                    f"{g.get('missing_from')} enable state manipulation (like Euler donateToReserves)?"
                )
        for g in r.require_gaps:
            if g.get('severity') == 'HIGH':
                lead['patterns'].append({
                    'type': 'missing_require_theme',
                    'theme': g.get('require_theme', '?'),
                    'missing_from': g.get('missing_from', []),
                })
        for ug in r.unguarded_functions:
            lead['patterns'].append({
                'type': 'unguarded_state_changer',
                'function': ug,
            })
            lead['action_items'].append(
                f"Check if {ug} is called from guarded entrypoints or is an unprotected external"
            )
        if 'oracle' in r.cis_domains_missing:
            lead['patterns'].append({
                'type': 'compound_class_oracle_gap',
                'detail': 'Zero oracle-domain constraints — price feed unchecked'
            })
            lead['action_items'].append(
                "Check all oracle price reads for: != 0, staleness, deviation bounds"
            )
        leads.append(lead)
    return leads


def save_results_json(results: list[FullScanResult], output_path: str):
    """Save ranked scan results as structured JSON."""
    ranked = sorted(results, key=compute_exploitability_score, reverse=True)
    data = []
    for r in ranked:
        data.append({
            'protocol': r.protocol,
            'contracts': r.contracts,
            'exploitability_score': compute_exploitability_score(r),
            'cis_score': r.cis_score,
            'risk_level': r.risk_level,
            'total_functions': r.total_functions,
            'total_state_changers': r.total_state_changers,
            'high_validation_gaps': [
                {'fn': g.get('validation_function', g.get('require_theme', '?')),
                 'called_by': g.get('called_by', []),
                 'missing_from': g.get('missing_from', [])}
                for g in r.validation_gaps if g.get('severity') == 'HIGH'
            ],
            'high_require_gaps': [
                {'theme': g.get('require_theme', '?'),
                 'enforced_by': g.get('enforced_by', []),
                 'missing_from': g.get('missing_from', [])}
                for g in r.require_gaps if g.get('severity') == 'HIGH'
            ],
            'medium_gaps': len([g for g in r.validation_gaps + r.require_gaps
                                if g.get('severity') == 'MEDIUM']),
            'unguarded_functions': r.unguarded_functions,
            'cis_dark_zones': r.cis_dark_zones,
            'unconstrained_fraction': r.cis_unconstrained,
            'domains_missing': r.cis_domains_missing,
            'domains_present': r.cis_domains_present,
        })
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to {output_path}")


def print_batch_summary(results: list[FullScanResult]):
    """Print ranked batch summary table sorted by exploitability."""
    ranked = sorted(results, key=compute_exploitability_score, reverse=True)

    c_red = '\033[91m'
    c_yel = '\033[93m'
    c_cya = '\033[96m'
    c_grn = '\033[92m'
    c_rst = '\033[0m'

    print(f"\n\n{'='*90}")
    print(f"  BATCH SCAN RESULTS — RANKED BY EXPLOITABILITY")
    print(f"{'='*90}")
    print(f"  {'#':>3} {'Protocol':<22} {'Exp.S':>7} {'CIS':>6} {'Risk':>9} {'Gaps':>6} {'Unc':>7} {'Domains':>8}")
    print(f"  {'-'*80}")
    for i, r in enumerate(ranked):
        risk_c = {'CRITICAL': c_red, 'HIGH': c_yel, 'MEDIUM': c_cya, 'LOW': c_grn}.get(r.risk_level, '')
        exp_s = compute_exploitability_score(r)
        high_gaps = len([g for g in r.validation_gaps if g.get('severity') == 'HIGH'])
        print(f"  {i+1:>3} {r.protocol:<22} {exp_s:>7.1f} {risk_c}{r.cis_score:>6.1f} {r.risk_level:>9}{c_rst} "
              f"{high_gaps:>5} {r.cis_unconstrained:>6.1%} "
              f"{len(r.cis_domains_present)}/{7-len(r.cis_domains_missing)+len(r.cis_domains_present)}")

    # Investigation leads
    leads = generate_investigation_leads(results, top_n=5)
    if leads:
        print(f"\n{'='*90}")
        print(f"  TOP 5 INVESTIGATION LEADS")
        print(f"{'='*90}")
        for i, lead in enumerate(leads):
            print(f"\n  [{i+1}] {c_red if lead['risk_level'] in ('CRITICAL','HIGH') else c_yel}"
                  f"{lead['protocol']}{c_rst} (Exploitability: {lead['exploitability_score']:.1f}, "
                  f"CIS: {lead['cis_score']:.1f} [{lead['risk_level']}])")
            if lead['patterns']:
                for p in lead['patterns']:
                    print(f"    • {p['type']}: {p.get('detail', p.get('validation_fn', p.get('function', p.get('theme', ''))))}")
            if lead['action_items']:
                for a in lead['action_items'][:3]:
                    print(f"    → {a}")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    contracts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'contracts')

    results = batch_scan(contracts_dir)

    if results:
        print_batch_summary(results)
        save_results_json(results, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                'scan_results.json'))
