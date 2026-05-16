"""
Compound vs Aave v3 — Comparative CIS Security Analysis

Runs full A.1-A.7 pipeline on both protocols extracted from real Solidity,
computes CIS Security Score, and generates comparison HTML.
"""

import sys, os, json
sys.path.insert(0, '/Users/dengxinhang/paper')
import numpy as np
from constraint_residual.solidity_extractor import extract_constraints_from_contract
from constraint_residual.cis_core import CISAnalyzer
from constraint_residual.dark_zone_detector import DarkZoneDetector
from constraint_residual.core import ConstraintField

CONTRACTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'contracts')
N_GRID = 64

def domain_coverage_gap(field, expected_domains):
    """Measure domain coverage gap — domains expected but not present."""
    present = set(r.domain for r in field.rules)
    missing = set(expected_domains) - present
    return {
        'present': sorted(present),
        'missing': sorted(missing),
        'coverage': len(present) / len(expected_domains) if expected_domains else 1.0,
        'gap_count': len(missing),
    }

ALL_EXPECTED_DOMAINS = ['access_control', 'oracle', 'tokenomics', 'security', 'risk', 'accounting']

print("=" * 70)
print("Compound vs Aave v3 — CIS Comparative Security Assessment")
print("=" * 70)

results = {}
for name, contract_file in [('Aave v3', 'Pool.sol'), ('Aave v3', 'AaveOracle.sol'),
                              ('Compound', 'Comptroller.sol'), ('Compound', 'PriceOracle.sol')]:
    fpath = os.path.join(CONTRACTS_DIR, contract_file)
    if not os.path.exists(fpath):
        continue
    print(f"\n--- {name} :: {contract_file} ---")
    field = extract_constraints_from_contract(fpath)

    coverage = domain_coverage_gap(field, ALL_EXPECTED_DOMAINS)
    print(f"  Domains present: {coverage['present']}")
    print(f"  Domains missing: {coverage['missing']}")
    print(f"  Coverage: {coverage['coverage']:.1%} ({len(field.rules)} rules)")

    if name not in results:
        results[name] = {'rules': [], 'coverage_gaps': []}
    results[name]['rules'].extend(field.rules)
    results[name]['coverage_gaps'].append(coverage)

# Build combined fields
print("\n" + "=" * 70)
print("Full A.1-A.7 CIS Pipeline Analysis")
print("=" * 70)

for name in results:
    combined = ConstraintField(rules=results[name]['rules'])
    results[name]['field'] = combined

    coverage = domain_coverage_gap(combined, ALL_EXPECTED_DOMAINS)
    results[name]['coverage'] = coverage

    print(f"\n{'='*50}")
    print(f"  {name}: {len(combined.rules)} total rules, "
          f"{coverage['coverage']:.1%} domain coverage")
    if coverage['missing']:
        print(f"  MISSING DOMAINS: {coverage['missing']}")

    analyzer = CISAnalyzer(combined, bounds=[(0, 1), (0, 1)], n_points=N_GRID)
    report = analyzer.full_analysis(name)
    results[name]['cis'] = report
    analyzer.print_summary()

    # Dark zone scan
    detector = DarkZoneDetector(cancellation_eps=0.15, individual_min=0.25)
    dz = detector.scan(combined, [(0, 1), (0, 1)], n_points=N_GRID)
    results[name]['dz'] = dz

    print(f"\n  Dark zones: {len(dz)}")
    for d in dz:
        print(f"    centroid=({d.centroid[0]:.4f},{d.centroid[1]:.4f}) "
              f"c(p)={d.mean_cancellation_ratio:.4f} type={d.balance_topology}")

# ═══════════════════════════════════════════════════════════════
# CIS Security Score (compound-specific weights)
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("CIS Security Score Comparison")
print("=" * 70)

def compute_cis_score(report, coverage, dz_count):
    """CIS Security Score: 0-100, lower = more vulnerable.

    Calibrated so a protocol with perfect coverage, 0 dark zones, and low
    unconstrained space scores ~95. Real DeFi protocols typically score 40-75.
    """
    score = 100.0

    # Domain coverage: each missing domain = -8 (6 domains total → max -48)
    score -= coverage['gap_count'] * 8.0

    # E-I structural gap: each 1% E-I = -5 (structural gaps are fundamentally unfixable)
    score -= report.e1_fraction * 500.0

    # Unconstrained space: each 1% = -0.3 (up to -30 for fully unconstrained)
    score -= report.unconstrained_fraction * 30.0

    # Dark zones: each = -5
    score -= dz_count * 5.0

    # Condition number penalty: log-scaled, meaningful above 1e3
    max_cond = float(np.max(report.riemannian.condition_number))
    if max_cond > 1000:
        score -= 5.0 * np.log10(max_cond / 1000)

    # Structural score bonus: 0→1, add up to 10
    score += report.structural_score * 10.0

    return max(0.0, min(100.0, score))

scores = {}
for name in ['Aave v3', 'Compound']:
    r = results[name]
    score = compute_cis_score(r['cis'], r['coverage'], len(r['dz']))
    scores[name] = score

    print(f"\n  {name}:")
    print(f"    Coverage gap:        {r['coverage']['gap_count']} missing domains "
          f"({r['coverage']['missing']})")
    print(f"    E-I structural:      {r['cis'].e1_fraction:.2%}")
    print(f"    Unconstrained:       {r['cis'].unconstrained_fraction:.2%}")
    print(f"    Dark zones:          {len(r['dz'])}")
    print(f"    Max condition #:     {float(np.max(r['cis'].riemannian.condition_number)):.1f}")
    print(f"    Structural score:    {r['cis'].structural_score:.4f}")
    print(f"    CIS Security Score:  {score:.1f}/100")

# ═══════════════════════════════════════════════════════════════
# Comparative table
# ═══════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("Comparative Summary Table")
print(f"{'='*70}")
print(f"{'Metric':<40} {'Aave v3':>12} {'Compound':>12}")
print("-" * 66)

metrics = [
    ('Rules extracted', lambda r: len(r['field'].rules)),
    ('Domains covered', lambda r: len(r['coverage']['present'])),
    ('Domains missing', lambda r: r['coverage']['gap_count']),
    ('Dark zones', lambda r: len(r['dz'])),
    ('E-I (structural)', lambda r: f"{r['cis'].e1_fraction:.1%}"),
    ('E-II (scale/scalar)', lambda r: f"{r['cis'].e2_fraction:.1%}"),
    ('E-III (boundary)', lambda r: f"{r['cis'].e3_fraction:.1%}"),
    ('Unconstrained', lambda r: f"{r['cis'].unconstrained_fraction:.1%}"),
    ('Max condition #', lambda r: f"{float(np.max(r['cis'].riemannian.condition_number)):.1f}"),
    ('Structural score', lambda r: f"{r['cis'].structural_score:.4f}"),
    ('CIS Security Score', lambda r: f"{scores[name_from_scores(r)]:.1f}/100"),
]

def name_from_scores(r):
    for n, v in results.items():
        if v is r:
            return n
    return '?'

for label, fn in metrics[:-1]:
    a_val = fn(results['Aave v3'])
    c_val = fn(results['Compound'])
    print(f"{label:<40} {str(a_val):>12} {str(c_val):>12}")

# Score separately (needs scores dict)
a_score = scores['Aave v3']
c_score = scores['Compound']
print(f"{'CIS Security Score':<40} {f'{a_score:.1f}/100':>12} {f'{c_score:.1f}/100':>12}")

print(f"\n{'='*70}")
print("Key Findings:")
print(f"{'='*70}")

a_missing = results['Aave v3']['coverage']['missing']
c_missing = results['Compound']['coverage']['missing']

if 'oracle' in a_missing and 'oracle' not in c_missing:
    print("  [CRITICAL] Aave v3 missing oracle domain — oracle staleness blind spot")
elif 'oracle' in a_missing and 'oracle' in c_missing:
    print("  [HIGH] Both protocols missing oracle constraints — common DeFi blind spot")

if scores['Aave v3'] < scores['Compound']:
    diff = scores['Compound'] - scores['Aave v3']
    print(f"  Compound scores {diff:.1f} points higher — better constraint coverage")
elif scores['Compound'] < scores['Aave v3']:
    diff = scores['Aave v3'] - scores['Compound']
    print(f"  Aave v3 scores {diff:.1f} points higher — better constraint coverage")

print(f"\n  Domain coverage comparison:")
print(f"    Aave v3:   {results['Aave v3']['coverage']['present']}")
print(f"    Compound:  {results['Compound']['coverage']['present']}")

print(f"\n{'='*70}")
print("Protocol Ranking:")
ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
for rank, (name, score) in enumerate(ranked, 1):
    bar = '█' * int(score / 5) + '░' * (20 - int(score / 5))
    print(f"  #{rank} {name:<30s} [{bar}] {score:.1f}/100")
