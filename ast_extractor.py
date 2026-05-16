"""
Heuristic Solidity Constraint Extractor — no compiler dependency.

Extracts richer constraint information than regex alone:
  1. require() statements (with modifier context)
  2. Modifier definitions and their constraint content
  3. Internal validation calls (checkX, requireX, validateX, verifyX)
  4. if/return error patterns (weaker than require)
  5. State-changing functions that DON'T call validation
  6. Cross-contract constraint transmission edges

This is an AST-like extraction using structural pattern matching,
not a full Solidity parser — but it catches patterns the regex misses.
"""

import re
import os
from collections import defaultdict
from dataclasses import dataclass, field


# ═══════════════════════════════════════════════════════════════
# Known modifier constraint registry — maps common DeFi modifier
# names to (domain, theme, strength) when modifier is not defined
# in the scanned file (e.g. from imported contracts).
# Strength: 1=weak hint, 4=strong constraint (like onlyOwner)
# ═══════════════════════════════════════════════════════════════
KNOWN_MODIFIER_STRENGTH = {
    # Security: reentrancy guards
    'nonReentrant':       ('security', 'reentrancy_guard', 3),
    'reentrancyGuard':    ('security', 'reentrancy_guard', 3),
    'lock':               ('security', 'reentrancy_guard', 2),
    'nonreentrant':       ('security', 'reentrancy_guard', 3),
    'noReentrancy':       ('security', 'reentrancy_guard', 3),
    'noreentrancy':       ('security', 'reentrancy_guard', 3),
    # Access control (strongest constraints)
    'onlyOwner':          ('access_control', 'owner_check', 4),
    'onlyAdmin':          ('access_control', 'admin_check', 4),
    'onlyGovernor':       ('access_control', 'governor_check', 4),
    'onlyGovernance':     ('access_control', 'governor_check', 4),
    'onlyRole':           ('access_control', 'role_check', 3),
    'onlyPool':           ('access_control', 'pool_check', 4),
    'onlyPoolAdmin':      ('access_control', 'admin_check', 4),
    'onlyVault':          ('access_control', 'vault_check', 3),
    'onlyProxy':          ('access_control', 'proxy_check', 3),
    'onlyFactory':        ('access_control', 'factory_check', 3),
    'onlyKeeper':         ('access_control', 'keeper_check', 2),
    'onlyLiquidator':     ('access_control', 'liquidator_check', 2),
    'onlyRouter':         ('access_control', 'router_check', 3),
    'onlyInitializing':   ('access_control', 'init_guard', 3),
    'onlyMinter':         ('access_control', 'minter_check', 3),
    'onlyBurner':         ('access_control', 'burner_check', 3),
    'onlyBorrower':       ('access_control', 'borrower_check', 2),
    'onlySupplier':       ('access_control', 'supplier_check', 2),
    # State guards
    'whenNotPaused':      ('risk', 'pause_guard', 4),
    'whenPaused':         ('risk', 'pause_guard', 4),
    'onlyActive':         ('risk', 'active_guard', 3),
    'notFrozen':          ('risk', 'freeze_guard', 3),
    'onlyEOA':            ('security', 'eoa_check', 2),
    # Supply / capacity guards
    'notExceedsCap':      ('tokenomics', 'supply_cap', 3),
    'withinLimits':       ('tokenomics', 'limit_check', 3),
    'belowMaxMintPerBlock': ('tokenomics', 'mint_cap', 3),
    'belowMaxRedeemPerBlock': ('tokenomics', 'redeem_cap', 3),
    # Staking / cooldown
    'ensureCooldownOff':  ('tokenomics', 'cooldown_check', 3),
    'ensureCooldownOn':   ('tokenomics', 'cooldown_check', 3),
    # General validators
    'notZero':            ('general', 'nonzero_check', 1),
    'validAddress':       ('general', 'address_check', 1),
}

# ═══════════════════════════════════════════════════════════════
# Pattern definitions
# ═══════════════════════════════════════════════════════════════

# Modifier definition
RE_MODIFIER = re.compile(
    r'modifier\s+(\w+)\s*\((.*?)\)\s*\{',
    re.DOTALL)

# Function definition (captures name, params, modifiers, visibility)
RE_FUNCTION = re.compile(
    r'function\s+(\w+)\s*\((.*?)\)\s*'
    r'(.*?)'
    r'(external|internal|public|private)?\s*'
    r'(\{|returns)',
    re.DOTALL)

# Compound v2 post-validation callback suffixes — functions named *Verify are
# called by the comptroller AFTER validation, not entry points that need guards.
CALLBACK_SUFFIXES = ['Verify']

# require() with full error message
RE_REQUIRE = re.compile(
    r'require\s*\((.*?)\)\s*;',
    re.DOTALL)

# Cross-contract validation calls — comptroller.*Allowed / checkMembership patterns
RE_CROSS_CONTRACT_CHECK = re.compile(
    r'\b(?:comptroller|controller|joetroller)\.\s*'
    r'(?:get\w+)?(?:check\w+|mintAllowed|borrowAllowed|redeemAllowed|'
    r'transferAllowed|liquidateBorrowAllowed|seizeAllowed|repayBorrowAllowed)'
    r'\s*\(',
    re.DOTALL)

# Internal implementation passthrough — function delegates to *Internal/*Fresh/*Implement
RE_INTERNAL_PASSTHROUGH = re.compile(
    r'\b\w+(?:Internal|Fresh|Implement|Actual)\s*\(',
    re.DOTALL)

# External contract calls — any contract.function() pattern (E-I call graph)
# Captures: comptroller.claimReward(...), joetroller.checkMembership(...), oracle.getPrice(...)
RE_EXTERNAL_CALL = re.compile(
    r'\b(?:(\w+)\s*\(\s*\w+\s*\))?\.\s*'   # optional cast: Joetroller(addr).
    r'(\w+)\s*\(',
    re.DOTALL)

# Internal validation call — functions with check/validate/verify/require/auth prefix
# Also catches _validate, _onlyGov, _onlyRole, _auth (underscore-prefixed guards)
RE_VALIDATION_CALL = re.compile(
    r'\b(check\w+|require\w+|validate\w+|verify\w+|ensure\w+|assert\w+'
    r'|_validate|_only\w*[Gg]ov|_only\w*[Rr]ole|_\w*auth\w*)'
    r'\s*\(([^)]*)\)',
    re.DOTALL)

# if (...) revert Error(); or if (...) { revert Error(); } — modern Solidity constraint
RE_IF_REVERT = re.compile(
    r'if\s*\(([^{}]+?)\)\s*'
    r'(?:\{\s*)?'
    r'(revert\s+[^;]+;)',
    re.DOTALL)

# if (cond) { ... } else { revert ... } — constraint in else branch
RE_ELSE_REVERT = re.compile(
    r'if\s*\(([^{}]+?)\)\s*'
    r'\{[^}]*\}'
    r'\s*else\s*'
    r'(?:\{\s*)?'
    r'(revert\s+[^;]+;)',
    re.DOTALL)

# Legacy if/return error patterns — weaker than require
RE_IF_RETURN_ERROR = re.compile(
    r'if\s*\(([^{}]+?)\)\s*\{\s*'
    r'(return\s+[^;]+Error[^;]*)',
    re.DOTALL)

# State variable declarations
RE_STATE_VAR = re.compile(
    r'(uint(?:256|128|64|32|16|8)?|int(?:256|128|64|32|16|8)?|bool|address|bytes(?:32)?|string)\s+'
    r'(public|internal|private|constant|immutable)?\s*'
    r'(\w+)\s*[=;]')

# Event emissions (constraint logging)
RE_EVENT = re.compile(
    r'emit\s+(\w+)\s*\(')

# Inheritance
RE_INHERIT = re.compile(
    r'contract\s+(\w+)\s+is\s+([^{]+)\{')

# Contract declaration
RE_CONTRACT = re.compile(
    r'contract\s+(\w+)\s*(?:is\s+([^{]+))?\{')

# ═══════════════════════════════════════════════════════════════
# Extraction functions
# ═══════════════════════════════════════════════════════════════

def _extract_block(source: str, start_pos: int) -> str:
    """Extract a brace-balanced block starting from the first {."""
    depth = 0
    in_block = False
    for i in range(start_pos, len(source)):
        if source[i] == '{':
            depth += 1
            in_block = True
        elif source[i] == '}':
            depth -= 1
            if in_block and depth == 0:
                return source[start_pos:i+1]
    return ''


def _extract_function_body(source: str, match_end: int) -> str:
    """Extract the body of a function after the function signature."""
    return _extract_block(source, match_end)


def extract_modifiers(source: str) -> list[dict]:
    """Extract all modifier definitions with their constraint content."""
    modifiers = []
    for m in RE_MODIFIER.finditer(source):
        name = m.group(1)
        params = m.group(2).strip()
        body = _extract_block(source, m.end() - 1)

        # Extract requires inside the modifier body
        requires = [r.group(1).strip() for r in RE_REQUIRE.finditer(body)]
        requires += [r.group(1).strip() for r in RE_IF_REVERT.finditer(body)]
        requires += [r.group(1).strip() for r in RE_ELSE_REVERT.finditer(body)]

        modifiers.append({
            'name': name,
            'params': params,
            'requires': requires,
            'body': body,
        })
    return modifiers


def _strip_solidity_comments(source: str) -> str:
    """Remove Solidity comments to prevent regex matching inside comment bodies.

    Handles /* block */, /** natspec */, and // line comments.
    Replaces comment bodies with spaces to preserve character offsets.
    """
    result = []
    i = 0
    n = len(source)
    while i < n:
        # Line comment
        if source[i:i+2] == '//' and (i == 0 or source[i-1] != ':'):
            j = source.find('\n', i)
            if j == -1:
                j = n
            result.append(' ' * (j - i))
            i = j
        # Block comment (including NatSpec /** */)
        elif source[i:i+2] == '/*':
            j = source.find('*/', i + 2)
            if j == -1:
                result.append(source[i])
                i += 1
            else:
                result.append(' ' * (j + 2 - i))
                i = j + 2
        else:
            result.append(source[i])
            i += 1
    return ''.join(result)


def extract_functions(source: str) -> list[dict]:
    """Extract all function definitions with their modifiers, requires, and validation calls."""
    functions = []
    source = _strip_solidity_comments(source)

    # Match function declarations
    func_pattern = re.compile(
        r'\bfunction\s+(\w+)\s*\((.*?)\)\s*'  # name(params)
        r'([^{]*)'  # modifiers + returns
        r'\{',
        re.DOTALL)

    for m in func_pattern.finditer(source):
        name = m.group(1)
        params = m.group(2).strip()
        decorators = m.group(3).strip()
        body = _extract_block(source, m.end() - 1)

        # Parse modifiers from decorators (handle onlyRole(ROLE) etc.)
        mods = []
        vis = 'internal'
        state_mutability = None
        # Extract modifier names with optional parenthesized arguments
        mod_part_pattern = re.compile(r'(\w+)\s*(?:\([^)]*\))?')
        for mp in mod_part_pattern.finditer(decorators):
            mname = mp.group(1)
            if mname in ('external', 'internal', 'public', 'private'):
                vis = mname
            elif mname in ('view', 'pure'):
                state_mutability = mname
            elif mname in ('payable', 'virtual', 'override', 'returns'):
                pass  # keywords
            elif mname == 'return':
                pass
            else:
                mods.append(mname)

        # Extract requires (both require() and if-revert patterns)
        requires = []
        for r in RE_REQUIRE.finditer(body):
            cond = r.group(1).strip()
            # Split condition from error message
            parts = re.split(r',\s*(?=Errors\.|"[^"]*"|\'[^\']*\'|0x)', cond, maxsplit=1)
            requires.append(parts[0].strip())
        # Also extract if-revert as full constraints
        for r in RE_IF_REVERT.finditer(body):
            cond = r.group(1).strip()
            requires.append(cond)
        for r in RE_ELSE_REVERT.finditer(body):
            cond = r.group(1).strip()
            requires.append(cond)

        # Extract validation calls
        validations = []
        for v in RE_VALIDATION_CALL.finditer(body):
            vname = v.group(1)
            vargs = v.group(2).strip()
            validations.append({'name': vname, 'args': vargs})

        # Cross-contract validation calls (comptroller.mintAllowed etc.)
        cross_checks = []
        for cc in RE_CROSS_CONTRACT_CHECK.finditer(body):
            cross_checks.append(cc.group(0))

        # Internal passthrough calls (delegates to *Internal/*Fresh)
        internal_calls = []
        for ic in RE_INTERNAL_PASSTHROUGH.finditer(body):
            internal_calls.append(ic.group(0))

        # External contract calls — for E-I call graph propagation
        # Captures patterns like: comptroller.fn(...), joetroller.claimReward(...)
        external_calls = []
        for ec in RE_EXTERNAL_CALL.finditer(body):
            context = ec.group(0)  # full match: "Joetroller(addr).claimReward("
            callee_fn = ec.group(2)  # just the function name
            # Filter noise: skip library calls, built-ins, low-level calls
            if callee_fn in ('add', 'sub', 'mul', 'div', 'require', 'revert',
                            'assert', 'emit', 'abi', 'keccak256', 'address',
                            'transfer', 'send', 'balanceOf', 'allowance', 'approve',
                            'safeTransfer', 'safeTransferFrom', 'balanceOfUnderlying',
                            'borrowBalanceCurrent', 'borrowBalanceStored', 'exchangeRateStored',
                            'totalSupply', 'totalBorrows', 'totalReserves'):
                continue
            # Only capture calls to known contract references (not address(this).fn())
            if 'address(' not in context and 'this.' not in context:
                external_calls.append({
                    'callee_fn': callee_fn,
                    'context': context.strip(),
                })

        # Extract if/return error patterns (legacy — weaker than require/revert)
        weak_checks = []
        for w in RE_IF_RETURN_ERROR.finditer(body):
            weak_checks.append({
                'condition': w.group(1).strip(),
                'action': w.group(2).strip(),
            })

        # Check for state changes (view/pure functions cannot modify state)
        if state_mutability in ('view', 'pure'):
            has_state_change = False
            has_transfer = False
            has_balance_change = False
        else:
            has_transfer = bool(re.search(r'\.transfer\(|\.send\(|safeTransfer', body))
            has_balance_change = bool(re.search(r'balance.*=|\.balance\s*=\s*|mint|burn|deposit|withdraw|liquidate', body))
            has_state_change = has_transfer or has_balance_change

        functions.append({
            'name': name,
            'params': params,
            'visibility': vis,
            'modifiers': mods,
            'requires': requires,
            'validations': validations,
            'weak_checks': weak_checks,
            'cross_checks': cross_checks,
            'internal_calls': internal_calls,
            'external_calls': external_calls,
            'has_state_change': has_state_change,
            'has_transfer': has_transfer,
            'has_balance_change': has_balance_change,
        })

    return functions


def extract_inheritance(source: str) -> list[str]:
    """Extract parent contract names from inheritance."""
    parents = []
    ct = RE_CONTRACT.search(source)
    if ct and ct.group(2):
        parents = [p.strip().split()[0] for p in ct.group(2).split(',')]
    return parents


def extract_state_variables(source: str) -> list[dict]:
    """Extract state variable declarations."""
    vars_list = []
    # Rough extraction: match type + name at contract level
    pattern = re.compile(
        r'(uint\d*|int\d*|bool|address|bytes\d*|string|mapping\s*\([^)]+\))\s+'
        r'(?:public|internal|private|constant|immutable)?\s*'
        r'(\w+)\s*(?:=\s*([^;]+))?;')
    for m in pattern.finditer(source):
        vars_list.append({
            'type': m.group(1),
            'name': m.group(2),
            'default': (m.group(3) or '').strip(),
        })
    return vars_list


# ═══════════════════════════════════════════════════════════════
# Constraint flow analysis
# ═══════════════════════════════════════════════════════════════

def analyze_constraint_flow(functions: list[dict], modifiers: list[dict]) -> dict:
    """Analyze constraint transmission between functions and validation checks.

    Returns a dict with:
      - guarded_functions: functions that call validation
      - unguarded_state_changers: state-changing functions that DON'T call validation
      - modifier_coverage: which modifiers provide what constraints
    """
    modifier_requires = {m['name']: m['requires'] for m in modifiers}

    validation_names = set()
    for f in functions:
        for v in f['validations']:
            validation_names.add(v['name'])

    guarded = []
    unguarded = []

    for f in functions:
        total_checks = (len(f['requires']) + len(f['validations'])
                       + len(f['weak_checks']))
        # Cross-contract validation (comptroller.mintAllowed etc.) — indirect check
        total_checks += len(f.get('cross_checks', []))
        # Internal passthrough (delegates to *Internal/*Fresh) — indirect check
        if f.get('internal_calls') and len(f.get('requires', [])) == 0 and len(f.get('validations', [])) == 0:
            total_checks += 1  # treat as "delegated to implementation"
        # File-defined modifiers
        for mod in f['modifiers']:
            total_checks += len(modifier_requires.get(mod, []))
        # Known modifier fallback — modifiers not defined in this file
        for mod in f['modifiers']:
            if mod not in modifier_requires:
                known = KNOWN_MODIFIER_STRENGTH.get(mod)
                if known:
                    total_checks += known[2]  # strength value

        if f['has_state_change']:
            if total_checks == 0:
                # Only external/public functions are part of attack surface
                if f['visibility'] in ('public', 'external'):
                    # Skip post-validation callbacks (Compound v2 *Verify convention)
                    is_callback = any(f['name'].endswith(s) for s in CALLBACK_SUFFIXES)
                    if not is_callback:
                        unguarded.append({
                        'name': f['name'],
                        'modifiers': f['modifiers'],
                        'visibility': f['visibility'],
                        'validation_calls': [v['name'] for v in f['validations']],
                        'requires': f['requires'],
                        'weak_checks': f['weak_checks'],
                    })
            else:
                guarded.append({
                    'name': f['name'],
                    'check_count': total_checks,
                    'validation_calls': [v['name'] for v in f['validations']],
                    'modifiers': f['modifiers'],
                })

    return {
        'guarded_state_changers': guarded,
        'unguarded_state_changers': unguarded,
        'validation_functions': sorted(validation_names),
        'modifier_constraints': modifier_requires,
        'total_functions': len(functions),
        'total_state_changers': len([f for f in functions if f['has_state_change']]),
    }


# ═══════════════════════════════════════════════════════════════
# Cross-contract analysis
# ═══════════════════════════════════════════════════════════════

@dataclass
class ContractAnalysis:
    name: str
    filepath: str
    functions: list[dict] = field(default_factory=list)
    modifiers: list[dict] = field(default_factory=list)
    state_vars: list[dict] = field(default_factory=list)
    inheritance: list[str] = field(default_factory=list)
    flow: dict = field(default_factory=dict)
    validation_gaps: list = field(default_factory=list)
    require_gaps: list = field(default_factory=list)


def analyze_contract(filepath: str) -> ContractAnalysis:
    """Extract full constraint topology from a Solidity file."""
    source = open(filepath).read()

    name = os.path.basename(filepath).replace('.sol', '')
    ct = RE_CONTRACT.search(source)
    if ct:
        name = ct.group(1)

    functions = extract_functions(source)
    modifiers = extract_modifiers(source)
    state_vars = extract_state_variables(source)
    inheritance = extract_inheritance(source)
    flow = analyze_constraint_flow(functions, modifiers)
    validation_gaps = find_validation_gaps(functions)
    require_gaps = find_require_gaps(functions)

    return ContractAnalysis(
        name=name,
        filepath=filepath,
        functions=functions,
        modifiers=modifiers,
        state_vars=state_vars,
        inheritance=inheritance,
        flow=flow,
        validation_gaps=validation_gaps,
        require_gaps=require_gaps,
    )


def _categorize_require(condition: str) -> str:
    """Categorize a require() condition into a constraint theme."""
    c = condition.lower()

    if any(kw in c for kw in ['msg.sender', 'owner', 'admin', 'only', 'caller', 'acl']):
        return 'access_control'
    if any(kw in c for kw in ['price', 'oracle', 'staleness', 'twap', 'latest']):
        return 'oracle'
    if any(kw in c for kw in ['balance', 'transfer', 'send', 'allowance', '>=', '> 0', '!= 0']):
        return 'balance_check'
    if any(kw in c for kw in ['borrow', 'debt', 'supply', 'collateral', 'ltv', 'liquidation', 'health']):
        return 'health_factor'
    if any(kw in c for kw in ['paused', 'frozen', 'active', 'guardian']):
        return 'state_guard'
    if any(kw in c for kw in ['reentrancy', 'locked', 'nonreentrant']):
        return 'reentrancy_guard'
    if any(kw in c for kw in ['mint', 'cap', 'limit', 'supply', 'max']):
        return 'supply_cap'
    if any(kw in c for kw in ['initialized', 'migrate', 'upgrade']):
        return 'init_guard'
    if any(kw in c for kw in ['address(0)', '!= 0x', '!= address']):
        return 'null_check'
    return 'general'


def _classify_function_role(func: dict) -> str:
    """Classify a function into a security role based on caller identity and state impact.

    Roles determine which constraint themes are expected. Cross-role comparison
    (e.g. admin vs user_entry) produces meaningless noise — an admin pausing
    function should not be compared with a user borrowing function.

    Role definitions:
      user_entry    — callable by any external actor, modifies user positions
      policy_hook   — called by OToken to validate before/after state changes
      admin_write   — modifies protocol parameters, requires admin access
      admin_pause   — pauses/unpauses markets, requires pauseGuardian or admin
      system_upgrade — protocol upgrade/initialization functions
      reward        — claim/distribute reward tokens, no position change
    """
    name = func['name']
    requires_text = ' '.join(str(r) for r in func.get('requires', []))
    validations = func.get('validations', [])
    vis = func.get('visibility', 'internal')
    is_stateful = func.get('has_state_change', False)

    # ── Admin access patterns in the function's own require() statements ──
    has_admin_check = any(p in requires_text for p in [
        'msg.sender == admin', 'msg.sender != admin',
        'adminOrInitializing()', 'only admin',
        'msg.sender == marketCapGuardian',
    ])
    has_pause_check = any(p in requires_text for p in [
        'pauseGuardian', 'only pause guardian',
    ])

    # ── Policy hook: called by OToken, not by users directly ──
    if name.endswith('Allowed') or name.endswith('Verify'):
        return 'policy_hook'

    # ── System upgrade: become/accept pattern ──
    if name in ('_become', '_acceptAdmin', '_acceptImplementation',
                '_setPendingAdmin', '_setComptroller', '_setPauseGuardian',
                '_setMarketCapGuardian'):
        return 'system_upgrade'

    # ── Explicit reward claim functions ──
    if name.startswith('claim') or name == '_grantXcn':
        return 'reward'

    # ── State-changing functions with admin checks ──
    if is_stateful and has_admin_check:
        if has_pause_check:
            return 'admin_pause'
        # Functions that both check admin AND set protocol parameters
        if any(p in name for p in ['_set', '_support', '_grant', '_initialize', '_reduce']):
            return 'admin_write'
        return 'admin_write'

    # ── Admin-ish by name pattern (fallback) ──
    if name.startswith('_set') or name.startswith('_support') or name.startswith('_grant'):
        return 'admin_write'

    # ── State-changing external/public functions without admin checks → user entry ──
    if is_stateful and vis in ('public', 'external'):
        return 'user_entry'

    return 'other'


# Constraint themes that are meaningful ONLY within specific roles.
# Themes NOT listed for a role are considered irrelevant noise.
ROLE_THEME_EXPECTATIONS = {
    'user_entry': {
        'expected': {'health_factor', 'oracle', 'balance_check', 'supply_cap',
                     'access_control', 'state_guard', 'reentrancy_guard'},
        'high_severity': {'health_factor', 'oracle'},
    },
    'policy_hook': {
        'expected': {'state_guard', 'oracle', 'supply_cap', 'balance_check'},
        'high_severity': {'oracle', 'state_guard'},
    },
    'admin_write': {
        'expected': {'access_control'},
        'high_severity': set(),  # Missing admin check is caught differently
    },
    'admin_pause': {
        'expected': {'access_control', 'state_guard'},
        'high_severity': set(),
    },
    'system_upgrade': {
        'expected': {'access_control'},
        'high_severity': set(),
    },
    'reward': {
        'expected': {'state_guard', 'balance_check'},
        'high_severity': set(),
    },
}


def find_require_gaps(functions: list[dict]) -> list[dict]:
    """Role-aware require() gap detection.

    Functions are grouped by security role (user_entry, admin_write, etc.).
    Gaps are only detected WITHIN the same role — comparing a user-entry
    function against an admin function is structurally invalid.
    """
    state_changers = [f for f in functions if f['has_state_change']
                      and f.get('visibility') in ('public', 'external')
                      and not any(f['name'].endswith(s) for s in CALLBACK_SUFFIXES)]
    if len(state_changers) < 2:
        return []

    # Group by role
    role_groups: dict[str, list[dict]] = {}
    for f in state_changers:
        role = _classify_function_role(f)
        if role == 'other':
            continue
        role_groups.setdefault(role, []).append(f)

    # Within each role group, find theme gaps
    all_gaps = []
    for role, group_fns in role_groups.items():
        if len(group_fns) < 2:
            continue
        expectations = ROLE_THEME_EXPECTATIONS.get(role, {})

        # Map themes per function within this role
        func_themes = {}
        for f in group_fns:
            themes = set()
            for r in f['requires']:
                themes.add(_categorize_require(r))
            func_themes[f['name']] = themes

        all_theme_set = set()
        for ts in func_themes.values():
            all_theme_set.update(ts)

        for theme in all_theme_set:
            if theme in ('general', 'null_check', 'init_guard'):
                continue

            # Only flag themes that are expected for this role
            expected_set = expectations.get('expected', set())
            if theme not in expected_set:
                continue

            callers = [name for name, ts in func_themes.items() if theme in ts]
            non_callers = [name for name, ts in func_themes.items() if theme not in ts]

            if len(callers) >= len(group_fns) * 0.5 and len(non_callers) > 0:
                high_set = expectations.get('high_severity', set())
                severity = 'HIGH' if theme in high_set else 'MEDIUM'
                all_gaps.append({
                    'require_theme': theme,
                    'enforced_by': callers,
                    'missing_from': non_callers,
                    'severity': severity,
                    'role': role,
                })

    return all_gaps


def find_validation_gaps(functions: list[dict]) -> list[dict]:
    """Role-aware validation call gap detection.

    Only compares functions within the same security role. The Euler-class
    bug (donateToReserves missing checkLiquidity) lives here — a user_entry
    function missing a validation call that ALL its role-peers have.
    """
    state_changers = [f for f in functions if f['has_state_change']
                      and f.get('visibility') in ('public', 'external')
                      and not any(f['name'].endswith(s) for s in CALLBACK_SUFFIXES)]
    if len(state_changers) < 2:
        return []

    # Group by role
    role_groups: dict[str, list[dict]] = {}
    for f in state_changers:
        role = _classify_function_role(f)
        if role == 'other':
            continue
        role_groups.setdefault(role, []).append(f)

    all_gaps = []
    for role, group_fns in role_groups.items():
        if len(group_fns) < 2:
            continue

        # Validation calls within this role group
        val_calls_per_func = {}
        for f in group_fns:
            val_calls_per_func[f['name']] = set(v['name'] for v in f['validations'])

        all_vals = set()
        for vc in val_calls_per_func.values():
            all_vals.update(vc)

        for val_fn in all_vals:
            callers = [name for name, vc in val_calls_per_func.items() if val_fn in vc]
            non_callers = [name for name, vc in val_calls_per_func.items() if val_fn not in vc]

            # HIGH severity: all role-peers have it, exactly one is missing (Euler pattern)
            if len(callers) >= len(group_fns) - 1 and len(non_callers) == 1:
                severity = 'HIGH'
            elif len(callers) >= len(group_fns) * 0.5 and len(non_callers) > 0:
                severity = 'MEDIUM'
            else:
                continue

            all_gaps.append({
                'validation_function': val_fn,
                'called_by': callers,
                'missing_from': non_callers,
                'severity': severity,
                'role': role,
            })

    return all_gaps


def propagate_cross_contract_constraints(analyses: dict, max_hops: int = 3) -> dict:
    """E-I cross-contract constraint propagation (multi-hop).

    When function A calls B which calls C, constraints from B and C propagate
    back to A. Uses iterative fixed-point propagation up to max_hops levels.

    Returns: {contract_name.function_name: {'validations': set, 'require_themes': set}}
    """
    # Build lookup: (contract_name, fn_name) → function dict
    fn_lookup = {}
    for cname, a in analyses.items():
        for f in a.functions:
            fn_lookup[(cname, f['name'])] = f

    # Initialize: each function's "effective" constraints = direct + propagated
    effective = {}
    for cname, a in analyses.items():
        for f in a.functions:
            key = f'{cname}.{f["name"]}'
            effective[key] = {
                'validations': set(v['name'] for v in f['validations']),
                'require_themes': set(_categorize_require(r) for r in f['requires']),
            }

    # Iterative propagation: repeat until fixed point or max_hops
    for hop in range(max_hops):
        changed = False
        new_effective = {k: {'validations': set(v['validations']),
                             'require_themes': set(v['require_themes'])}
                        for k, v in effective.items()}

        for cname, a in analyses.items():
            for f in a.functions:
                caller_key = f'{cname}.{f["name"]}'

                for ec in f.get('external_calls', []):
                    target_fn = ec['callee_fn']
                    # Find target function in any analyzed contract
                    for (tcname, tfn), tf in fn_lookup.items():
                        if tfn == target_fn:
                            callee_key = f'{tcname}.{tfn}'
                            callee_effective = effective.get(callee_key, {})
                            # Propagate callee's effective constraints (includes its own propagated)
                            before_v = len(new_effective[caller_key]['validations'])
                            before_t = len(new_effective[caller_key]['require_themes'])
                            new_effective[caller_key]['validations'].update(
                                callee_effective.get('validations', set()))
                            new_effective[caller_key]['require_themes'].update(
                                callee_effective.get('require_themes', set()))
                            if (len(new_effective[caller_key]['validations']) > before_v or
                                len(new_effective[caller_key]['require_themes']) > before_t):
                                changed = True

        effective = new_effective
        if not changed:
            break  # Fixed point reached

    # Compute "propagated" = effective minus direct (what was gained through propagation)
    propagated = {}
    for cname, a in analyses.items():
        for f in a.functions:
            key = f'{cname}.{f["name"]}'
            direct_v = set(v['name'] for v in f['validations'])
            direct_t = set(_categorize_require(r) for r in f['requires'])
            eff = effective.get(key, {'validations': set(), 'require_themes': set()})
            propagated[key] = {
                'validations': eff['validations'] - direct_v,
                'require_themes': eff['require_themes'] - direct_t,
            }

    return propagated


def _find_validation_gaps_protocol_level(analyses: dict, propagated: dict) -> list[dict]:
    """Find validation gaps with E-I cross-contract constraint propagation.

    A function is NOT considered to have a gap if it delegates the check to
    an external function (propagated constraints cover the gap).
    """
    all_gaps = []
    CALLBACK_SUFFIXES = ['Verify']

    for cname, a in analyses.items():
        functions = a.functions
        state_changers = [f for f in functions if f['has_state_change']
                          and f.get('visibility') in ('public', 'external')
                          and not any(f['name'].endswith(s) for s in CALLBACK_SUFFIXES)]
        if len(state_changers) < 2:
            continue

        role_groups = {}
        for f in state_changers:
            role = _classify_function_role(f)
            if role == 'other':
                continue
            role_groups.setdefault(role, []).append(f)

        for role, group_fns in role_groups.items():
            if len(group_fns) < 2:
                continue

            val_calls_per_func = {}
            for f in group_fns:
                direct = set(v['name'] for v in f['validations'])
                # Include propagated constraints from E-I call graph
                prop_key = f'{cname}.{f["name"]}'
                indirect = propagated.get(prop_key, {}).get('validations', set())
                val_calls_per_func[f['name']] = direct | indirect

            all_vals = set()
            for vc in val_calls_per_func.values():
                all_vals.update(vc)

            for val_fn in all_vals:
                if val_fn.startswith('[external]'):
                    continue  # skip placeholder markers
                callers = [name for name, vc in val_calls_per_func.items() if val_fn in vc]
                non_callers = [name for name, vc in val_calls_per_func.items() if val_fn not in vc]

                if len(callers) >= len(group_fns) - 1 and len(non_callers) == 1:
                    severity = 'HIGH'
                elif len(callers) >= len(group_fns) * 0.5 and len(non_callers) > 0:
                    severity = 'MEDIUM'
                else:
                    continue

                all_gaps.append({
                    'contract': cname,
                    'validation_function': val_fn,
                    'called_by': callers,
                    'missing_from': non_callers,
                    'severity': severity,
                    'role': role,
                })

    return all_gaps


def _find_require_gaps_protocol_level(analyses: dict, propagated: dict) -> list[dict]:
    """Find require theme gaps with E-I cross-contract constraint propagation."""
    all_gaps = []
    CALLBACK_SUFFIXES = ['Verify']

    for cname, a in analyses.items():
        functions = a.functions
        state_changers = [f for f in functions if f['has_state_change']
                          and f.get('visibility') in ('public', 'external')
                          and not any(f['name'].endswith(s) for s in CALLBACK_SUFFIXES)]
        if len(state_changers) < 2:
            continue

        role_groups = {}
        for f in state_changers:
            role = _classify_function_role(f)
            if role == 'other':
                continue
            role_groups.setdefault(role, []).append(f)

        for role, group_fns in role_groups.items():
            if len(group_fns) < 2:
                continue
            expectations = ROLE_THEME_EXPECTATIONS.get(role, {})

            func_themes = {}
            for f in group_fns:
                direct = set(_categorize_require(r) for r in f['requires'])
                prop_key = f'{cname}.{f["name"]}'
                indirect = propagated.get(prop_key, {}).get('require_themes', set())
                func_themes[f['name']] = direct | indirect

            all_theme_set = set()
            for ts in func_themes.values():
                all_theme_set.update(ts)

            for theme in all_theme_set:
                if theme in ('general', 'null_check', 'init_guard'):
                    continue
                expected_set = expectations.get('expected', set())
                if theme not in expected_set:
                    continue

                callers = [name for name, ts in func_themes.items() if theme in ts]
                non_callers = [name for name, ts in func_themes.items() if theme not in ts]

                if len(callers) >= len(group_fns) * 0.5 and len(non_callers) > 0:
                    high_set = expectations.get('high_severity', set())
                    severity = 'HIGH' if theme in high_set else 'MEDIUM'
                    all_gaps.append({
                        'contract': cname,
                        'require_theme': theme,
                        'enforced_by': callers,
                        'missing_from': non_callers,
                        'severity': severity,
                        'role': role,
                    })

    return all_gaps


def analyze_protocol(contract_paths: list[str]) -> dict:
    """Analyze constraint topology across multiple contracts.

    Returns:
      - per-contract analysis
      - cross-contract validation call graph
      - unguarded state changers across all contracts
    """
    analyses = {}
    all_validation_fns = set()
    all_unguarded = []
    all_validation_gaps = []
    all_require_gaps = []

    for fpath in contract_paths:
        if not os.path.exists(fpath):
            continue
        a = analyze_contract(fpath)
        analyses[a.name] = a
        all_validation_fns.update(a.flow['validation_functions'])
        for u in a.flow['unguarded_state_changers']:
            all_unguarded.append({
                'contract': a.name,
                'function': u['name'],
                'visibility': u['visibility'],
                'requires': u['requires'],
                'weak_checks': u['weak_checks'],
            })

    # ── E-I: Cross-contract constraint propagation ──
    propagated = propagate_cross_contract_constraints(analyses)

    # Re-run gap detection with propagated constraints
    all_validation_gaps = _find_validation_gaps_protocol_level(analyses, propagated)
    all_require_gaps = _find_require_gaps_protocol_level(analyses, propagated)

    return {
        'contracts': analyses,
        'cross_validation_targets': sorted(all_validation_fns),
        'total_unguarded': len(all_unguarded),
        'unguarded_details': all_unguarded,
        'validation_gaps': all_validation_gaps,
        'require_gaps': all_require_gaps,
        'total_contracts': len(analyses),
        'propagated_constraints': {k: {'validations': list(v['validations']),
                                       'require_themes': list(v['require_themes'])}
                                  for k, v in propagated.items()},
    }


# ═══════════════════════════════════════════════════════════════
# Report generation
# ═══════════════════════════════════════════════════════════════

def print_protocol_report(result: dict):
    """Pretty-print a cross-contract constraint flow analysis."""
    print(f"\n{'='*70}")
    print(f"Cross-Contract Constraint Flow Analysis")
    print(f"{'='*70}")
    print(f"  Contracts analyzed: {result['total_contracts']}")
    print(f"  Validation targets: {result['cross_validation_targets']}")

    for name, a in result['contracts'].items():
        f = a.flow
        print(f"\n  ── {name} ──")
        print(f"    Functions: {f['total_functions']} "
              f"({f['total_state_changers']} state-changing)")
        print(f"    Modifiers: {list(f['modifier_constraints'].keys())}")
        print(f"    Guarded state-changers: {len(f['guarded_state_changers'])}")

        if f['unguarded_state_changers']:
            print(f"    ⚠ UNGUARDED state-changers: {len(f['unguarded_state_changers'])}")
            for u in f['unguarded_state_changers']:
                print(f"      • {u['name']} ({u['visibility']})")
                if u['requires']:
                    print(f"        requires: {u['requires'][:1]}...")
                if u['weak_checks']:
                    print(f"        weak checks: {u['weak_checks'][:1]}...")
        else:
            print(f"    ✓ All state-changers have constraint coverage")

    if result['validation_gaps']:
        print(f"\n  {'='*50}")
        print(f"  ⚠ VALIDATION CALL GAPS: {len(result['validation_gaps'])}")
        print(f"  {'='*50}")
        for g in result['validation_gaps']:
            sev_color = '\033[91m' if g['severity'] == 'HIGH' else '\033[93m'
            print(f"  {sev_color}[{g['severity']}]\033[0m {g['contract']}: "
                  f"'{g['validation_function']}' called by {g['called_by']}")
            print(f"       MISSING from: {g['missing_from']}")

    if result['require_gaps']:
        print(f"\n  {'='*50}")
        print(f"  ⚠ REQUIRE PATTERN GAPS: {len(result['require_gaps'])}")
        print(f"  {'='*50}")
        for g in result['require_gaps']:
            sev_color = '\033[91m' if g['severity'] == 'HIGH' else '\033[93m'
            print(f"  {sev_color}[{g['severity']}]\033[0m {g['contract']}: "
                  f"'{g['require_theme']}' enforced by {g['enforced_by']}")
            print(f"       MISSING from: {g['missing_from']}")

    if result['total_unguarded'] > 0:
        print(f"\n  ⚠ UNGUARDED STATE-CHANGERS: {result['total_unguarded']}")
        for u in result['unguarded_details']:
            print(f"    {u['contract']}.{u['function']} ({u['visibility']})")
            if u['requires']:
                print(f"      has require: {u['requires'][0][:100]}")
            if u['weak_checks']:
                print(f"      has weak check: {u['weak_checks'][0]['condition'][:100]}")
    else:
        print(f"\n  ✓ No fully-unguarded functions found")


# ═══════════════════════════════════════════════════════════════
# Main — scan Euler
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import sys
    euler_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'contracts', 'euler')

    if os.path.isdir(euler_dir):
        paths = [os.path.join(euler_dir, f) for f in os.listdir(euler_dir)
                 if f.endswith('.sol')]
        result = analyze_protocol(paths)
        print_protocol_report(result)

        # Focus: EToken.sol unguarded functions
        for name, a in result['contracts'].items():
            if 'EToken' in name:
                print(f"\n{'='*50}")
                print(f"  Deep dive: {name}")
                print(f"{'='*50}")
                for f in a.functions:
                    if f['has_state_change']:
                        val_calls = [v['name'] for v in f['validations']]
                        mods = f['modifiers']
                        checks = len(f['requires']) + len(val_calls) + len(f['weak_checks'])
                        status = '✓' if checks > 0 else '⚠ UNGUARDED'
                        print(f"  {status} {f['name']}: {len(f['requires'])}R + "
                              f"{len(val_calls)}V + {len(f['weak_checks'])}W "
                              f"[{','.join(val_calls)}] [{','.join(mods)}]")
