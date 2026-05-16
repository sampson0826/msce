"""
Generate HTML scan reports for both case studies.

Run:
  python3 -m constraint_residual.defi_adapter.generate_reports
"""

import sys
import os
sys.path.insert(0, '/Users/dengxinhang/paper')

from constraint_residual.core import ConstraintField
from constraint_residual.defi_adapter.cases.euler import (
    build_euler_rules, build_euler_mapper, build_exploit_path,
)
from constraint_residual.defi_adapter.cases.leveraged_staking import (
    build_leveraged_staking_rules,
)
from constraint_residual.defi_adapter.unified_scanner import UnifiedScanner
from constraint_residual.defi_adapter.report_generator import generate_html

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')


def generate_euler_report():
    print("Generating Euler Finance report...")
    rules = build_euler_rules()
    mapper = build_euler_mapper()
    exploit_path = build_exploit_path()

    scanner = UnifiedScanner(
        "Euler Finance (2023.03 — $197M Exploit)",
        type3_cancellation_eps=0.15,
        type2_condition_threshold=100.0,
    )
    for rule in rules:
        scanner.add_rule(rule)

    # 2D scan: health_factor × donation_ratio (fix discount at 0.05)
    bounds_2d = [(0.5, 2.0), (0.0, 1.0)]
    report = scanner.scan(bounds_2d, n_points=60)

    print(report.summary())

    # Highlight exploit path points on the visualization
    highlights = {}
    for i, (label, pt) in enumerate(zip(exploit_path.labels, exploit_path.points)):
        highlights[label] = pt[:2]  # Only first 2 dims for 2D viz

    field = ConstraintField(rules=rules)
    html = generate_html(report, field, mapper, highlight_points=highlights)

    out_path = os.path.join(OUT_DIR, 'euler_scan_report.html')
    with open(out_path, 'w') as f:
        f.write(html)
    print(f"Report saved to {out_path}")
    return out_path


def generate_leveraged_staking_report():
    print("\nGenerating Leveraged Staking report...")
    rules = build_leveraged_staking_rules()

    scanner = UnifiedScanner(
        "Cross-Protocol: Lido stETH + Lending Protocol (Leveraged Staking)",
        type3_cancellation_eps=0.15,
        type2_condition_threshold=100.0,
    )
    for rule in rules:
        scanner.add_rule(rule)

    bounds = [(1.0, 5.0), (0.92, 1.02)]
    report = scanner.scan(bounds, n_points=60)

    print(report.summary())

    # Highlight danger zone points
    highlights = {
        "Safe(2x,peg=1.0)": [2.0, 1.000],
        "Warning(3x,peg=0.97)": [3.0, 0.970],
        "Danger(4x,peg=0.95)": [4.0, 0.950],
    }

    field = ConstraintField(rules=rules)
    html = generate_html(report, field, highlight_points=highlights)

    out_path = os.path.join(OUT_DIR, 'leveraged_staking_scan_report.html')
    with open(out_path, 'w') as f:
        f.write(html)
    print(f"Report saved to {out_path}")
    return out_path


if __name__ == "__main__":
    path1 = generate_euler_report()
    path2 = generate_leveraged_staking_report()
    print(f"\nDone. Open the reports:")
    print(f"  open {path1}")
    print(f"  open {path2}")
