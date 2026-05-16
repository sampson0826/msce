"""
Solidity Constraint Extractor — Regex-based constraint mining

Extracts require() statements from Solidity contracts via regex,
classifies them by domain/type, and maps to constraint functions
for CIS analysis.

Approach:
  1. Regex extract all require(condition, ...) statements
  2. Classify each condition into a constraint domain
  3. Map to constraint function parameters
  4. Build ConstraintField

This is more robust than solc AST for contracts with unresolved imports.
"""

import re, os, sys
sys.path.insert(0, '/Users/dengxinhang/paper')
import numpy as np
from constraint_residual.core import Rule, ConstraintField
from constraint_residual.dsl.compiler import _build_gaussian, _build_sigmoid


def extract_requires(source: str) -> list[str]:
    """Extract all require() condition strings from Solidity source.

    Handles:
      require(condition, "error");
      require(condition, Errors.CONSTANT);
      require(
          complex_condition,
          error
      );
    """
    requires = []

    # Match require(...) — handle single-line and multi-line
    # Find require( then match to the matching )
    pattern = r'\brequire\s*\((.*?)\)\s*;'
    # Use DOTALL for multi-line
    matches = re.finditer(pattern, source, re.DOTALL)

    for m in matches:
        full = m.group(1).strip()
        # Split condition from error argument at the LAST comma at top level
        # Simple heuristic: split at comma that's followed by Errors., ", or 0x
        parts = re.split(r',\s*(?=Errors\.|"[^"]*"|\'[^\']*\'|0x)', full, maxsplit=1)
        condition = parts[0].strip()
        if condition:
            requires.append(condition)

    # Also extract if-revert patterns: if (cond) revert Error();
    revert_pattern = r'if\s*\(([^{}]+?)\)\s*(?:\{\s*)?(?:revert\s+[^;]+;)'
    for m in re.finditer(revert_pattern, source, re.DOTALL):
        cond = m.group(1).strip()
        if cond:
            requires.append(cond)

    # Also extract _validate(condition, ...) — custom require wrappers (GMX, etc.)
    validate_pattern = r'_validate\s*\(\s*(.+?)\s*,\s*[^)]+\)\s*;'
    for m in re.finditer(validate_pattern, source, re.DOTALL):
        cond = m.group(1).strip()
        if cond:
            requires.append(cond)

    return requires


def extract_oracle_signals(source: str) -> list[str]:
    """Detect oracle-related patterns beyond require() conditions.

    Compound v2 forks often validate prices via external oracle contract calls
    (oracle.getUnderlyingPrice, priceOracle.getAssetPrice) rather than inline
    require() with 'price' in the condition. This extracts those signals.
    """
    signals = []

    # Oracle contract method calls: oracle.getUnderlyingPrice, priceOracle.getPrice, etc.
    oracle_calls = re.findall(
        r'(?:oracle|priceOracle|assetPrice|priceFeed)\s*\.\s*'
        r'(?:get\w*[Pp]rice|latest\w+|update\w*)',
        source, re.IGNORECASE)
    if oracle_calls:
        signals.append('oracle_price_call')

    # Staleness / freshness checks
    staleness = re.findall(
        r'(?:staleness|updatedAt|latestRoundData|getRoundData|'
        r'priceDeviation|maxPriceDeviation)',
        source, re.IGNORECASE)
    if staleness:
        signals.append('oracle_staleness_check')

    # Price deviation bounds
    deviation = re.findall(
        r'(?:priceDeviation|maxDeviation|deviation\s*[<>=])',
        source, re.IGNORECASE)
    if deviation:
        signals.append('oracle_deviation_bound')

    # Zero-price guards: require(price != 0), if (price == 0) revert, etc.
    zero_guards = re.findall(
        r'(?:require|revert|if)\s*\([^)]*'
        r'(?:price|getPrice|getUnderlyingPrice|getAssetPrice)[^)]*'
        r'(?:!=\s*0|==\s*0|>\s*0)',
        source, re.IGNORECASE)
    if zero_guards:
        signals.append('oracle_zero_guard')

    return signals


# ═══════════════════════════════════════════════════════════════
# Condition → Constraint Function Mapping
# ═══════════════════════════════════════════════════════════════

def classify_require(condition: str) -> dict:
    """Classify a require condition into constraint domain and function parameters.

    Each classification returns: domain, fn_type, center, width, scale, axis
    """
    c = condition.lower()

    # ── Access Control ──
    access_kw = ['msg.sender', 'onlypool', 'onlybridge', 'aclmanager',
                 'ispooladmin', 'isassetlistingadmin', 'caller_not',
                 'pooladmin', 'onlyowner', 'owner']
    if any(kw in c for kw in access_kw):
        return {
            'domain': 'access_control',
            'fn_type': 'sigmoid',
            'axis': 'x',
            'center': 0.7,
            'width': 0.05,
            'scale': 2.0,
        }

    # ── Oracle / Price (including staleness) ──
    oracle_kw = ['price', 'oracle', 'latestanswer', 'latestrounddata',
                 'getassetprice', 'updatedat', 'staleness']
    if any(kw in c for kw in oracle_kw):
        return {
            'domain': 'oracle',
            'fn_type': 'gaussian',
            'center': [0.5, 0.85],
            'width': [0.35, 0.18],
            'scale': 1.0,
        }

    # ── Health Factor / Collateral Ratio ──
    hf_kw = ['healthfactor', 'health_factor', 'liquidation',
             'collateral', 'ltv', 'liquidationthreshold']
    if any(kw in c for kw in hf_kw):
        return {
            'domain': 'tokenomics',
            'fn_type': 'gaussian',
            'center': [0.2, 0.5],
            'width': [0.15, 0.35],
            'scale': 1.5,
        }

    # ── Borrow/Supply Amounts and Limits ──
    amount_kw = ['borrow', 'supply', 'debt', 'amount', 'repay',
                 'withdraw', 'mint', 'burn', '> 0', '>0', '!= 0']
    if any(kw in c for kw in amount_kw):
        return {
            'domain': 'tokenomics',
            'fn_type': 'gaussian',
            'center': [0.3, 0.5],
            'width': [0.2, 0.3],
            'scale': 1.2,
        }

    # ── Reentrancy / State Lock ──
    reent_kw = ['reentrancy', 'locked', 'nonreentrant', '_status']
    if any(kw in c for kw in reent_kw):
        return {
            'domain': 'security',
            'fn_type': 'sigmoid',
            'axis': 'y',
            'center': 0.5,
            'width': 0.05,
            'scale': 3.0,
        }

    # ── Reserve State / Configuration ──
    reserve_kw = ['reserve', 'active', 'frozen', 'paused', 'configuration']
    if any(kw in c for kw in reserve_kw):
        return {
            'domain': 'risk',
            'fn_type': 'gaussian',
            'center': [0.5, 0.5],
            'width': [0.3, 0.3],
            'scale': 1.0,
        }

    # ── Balance / Transfer Checks ──
    bal_kw = ['balance', 'transfer', 'safetransfer', 'send', 'allowance']
    if any(kw in c for kw in bal_kw):
        return {
            'domain': 'accounting',
            'fn_type': 'gaussian',
            'center': [0.5, 0.5],
            'width': [0.25, 0.25],
            'scale': 1.0,
        }

    # ── Flash Loan ──
    flash_kw = ['flashloan', 'flash_loan', 'premium', 'fee']
    if any(kw in c for kw in flash_kw):
        return {
            'domain': 'security',
            'fn_type': 'gaussian',
            'center': [0.5, 0.3],
            'width': [0.3, 0.3],
            'scale': 1.5,
        }

    # ── Default ──
    return {
        'domain': 'general',
        'fn_type': 'gaussian',
        'center': [0.5, 0.5],
        'width': [0.3, 0.3],
        'scale': 0.5,
    }


def build_rule(classification: dict, name: str, layer: int = 1) -> Rule:
    """Build a Rule from a constraint classification."""
    fn_type = classification['fn_type']
    scale = classification['scale']

    if fn_type == 'gaussian':
        fn = _build_gaussian(classification['center'], classification['width'], scale)
    else:
        fn = _build_sigmoid(classification['axis'], classification['center'],
                            classification['width'], scale)
    return Rule(name=name, layer=layer, domain=classification['domain'], constraint_fn=fn)


def extract_constraints_from_contract(filepath: str, max_per_domain: int = 3) -> ConstraintField:
    """Extract constraints from a Solidity contract file.

    Args:
        filepath: Path to .sol file
        max_per_domain: Maximum constraints per domain to avoid over-representation

    Returns:
        ConstraintField with extracted rules
    """
    source = open(filepath).read()
    conditions = extract_requires(source)
    oracle_signals = extract_oracle_signals(source)

    print(f"  Found {len(conditions)} require() statements in {os.path.basename(filepath)}")

    # Classify and build constraints
    rules_by_domain = {}
    rule_list = []

    for i, cond in enumerate(conditions):
        cls = classify_require(cond)
        domain = cls['domain']

        # Limit per domain
        if domain not in rules_by_domain:
            rules_by_domain[domain] = 0
        if rules_by_domain[domain] >= max_per_domain:
            continue
        rules_by_domain[domain] += 1

        name = f"{domain}_{i}"
        rule = build_rule(cls, name)
        rule_list.append(rule)

    # Oracle signals from function calls (not require() keywords)
    # If oracle domain is empty but signals exist, inject oracle constraints
    if 'oracle' not in rules_by_domain and oracle_signals:
        oracle_cls = {
            'domain': 'oracle',
            'fn_type': 'gaussian',
            'center': [0.5, 0.85],
            'width': [0.35, 0.18],
            'scale': 0.8,  # slightly lower confidence than explicit require()
        }
        for j in range(min(2, max_per_domain)):
            rule = build_rule(oracle_cls, f"oracle_signal_{j}")
            rule_list.append(rule)
            rules_by_domain['oracle'] = rules_by_domain.get('oracle', 0) + 1

    # Print summary
    for domain, count in sorted(rules_by_domain.items()):
        print(f"    {domain}: {count} constraints")

    return ConstraintField(rules=rule_list)


# ═══════════════════════════════════════════════════════════════
# Main — extract from Aave v3 and run CIS analysis
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    from constraint_residual.cis_core import CISAnalyzer
    from constraint_residual.dark_zone_detector import DarkZoneDetector

    contracts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'contracts')

    print("=== Real Solidity Constraint Extraction & CIS Analysis ===\n")

    all_rules = []
    for fname in ['Pool.sol', 'AaveOracle.sol']:
        fpath = os.path.join(contracts_dir, fname)
        if not os.path.exists(fpath):
            print(f"  {fname} not found, skipping")
            continue
        print(f"Extracting from {fname}...")
        field = extract_constraints_from_contract(fpath)
        all_rules.extend(field.rules)
        print()

    if not all_rules:
        print("No constraints extracted!")
        sys.exit(1)

    # Build combined constraint field
    combined_field = ConstraintField(rules=all_rules)
    print(f"Total constraints: {len(all_rules)}")
    domain_counts = {}
    for r in all_rules:
        domain_counts[r.domain] = domain_counts.get(r.domain, 0) + 1
    print("Domains:", dict(sorted(domain_counts.items())))

    # ── Run CIS Analysis ──
    print(f"\n{'='*60}")
    print("Running CIS Core analysis on extracted constraints...")
    analyzer = CISAnalyzer(combined_field, bounds=[(0, 1), (0, 1)], n_points=60)
    report = analyzer.full_analysis("Aave v3 (real code extract)")
    analyzer.print_summary()

    # ── Dark Zone Detection ──
    print(f"\n{'='*60}")
    print("Dark Zone Detection:")
    detector = DarkZoneDetector(cancellation_eps=0.15, individual_min=0.25)
    dz = detector.scan(combined_field, [(0, 1), (0, 1)], n_points=64)
    print(f"  Dark zones: {len(dz)}")
    for d in dz:
        print(f"    centroid=({d.centroid[0]:.4f},{d.centroid[1]:.4f}) c(p)={d.mean_cancellation_ratio:.4f} type={d.balance_topology}")
