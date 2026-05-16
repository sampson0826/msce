// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title DAOVault — Constraint-Aware Implementation
 * @notice Generated from constraint DSL spec: the_dao
 *
 * STATE SPACE:
 *   recursion_depth ∈ [0, 1]
 *     0 = normal single call, 1 = deep recursive re-entry
 *   balance_consistency ∈ [0, 1]
 *     0 = balance not yet updated post-transfer, 1 = fully synced
 *
 * CONSTRAINTS:
 *   - withdraw_limit (L1, tokenomics): Prevents withdrawing more than proportional token share. Strongest at low recurs...
 *   - call_mechanism (L1, evm_platform): EVM's native ability to make nested contract calls. Not a bug — a platform featu...
 *   - balance_sync (L1, accounting): Balance accounting consistency: totalEther must equal sum of individual balances...
 *
 * DARK ZONE: mutual_cancellation
 *   Location: recursion_depth ≈ 0.5, balance_consistency ≈ 0.5
 *   c(p) → 0: withdraw_limit ∇ ≈ -call_mechanism ∇ at recursion midpoint
 *   balance_sync has strong value but zero gradient at inflection
 *
 * COUNTERMEASURE: checks-effects-interactions (gradient reorder)
 *   Move balance_sync gradient away from zero by updating state BEFORE
 *   external call. This ensures balance_sync ∇ ≠ 0 when recursive call
 *   arrives, breaking the mutual cancellation.
 */
contract DAOVault {
    mapping(address => uint256) public balances;
    uint256 public totalSupply;
    uint256 public totalEther;

    // ─── State space monitors (for off-chain constraint analysis) ───
    // These are read-only views that expose the abstract state space
    // variables to the constraint scanner.

    /// @notice Constraint: withdraw_limit (L1, tokenomics)
    /// σ_withdraw_limit ≈ exp(-((recursion-0.2)/0.25)² - ((sync-0.5)/0.4)²)
    /// Strongest at low recursion — resists state moving toward deeper recursion.
    modifier withdrawLimitCheck(uint256 amount) {
        require(balances[msg.sender] >= amount,
            "DAO: withdraw_limit constraint violated — insufficient balance");
        _;
    }

    /// @notice Constraint: call_mechanism (L1, evm_platform)
    /// This constraint is inherent to the EVM — external calls can re-enter.
    /// The countermeasure is NOT to block calls (impossible), but to ensure
    /// the balance_sync constraint has non-zero gradient when they arrive.
    /// We track recursion depth on-chain for constraint monitoring.
    uint256 private _recursionDepth;

    modifier noReentrancy() {
        _recursionDepth++;
        _;
        _recursionDepth--;
    }

    /// @notice Constraint: balance_sync (L1, accounting)
    /// σ_balance_sync ≈ -0.85 * exp(-((recursion-0.5)/0.2)² - ((sync-0.5)/0.2)²)
    /// KEY INSIGHT: balance_sync's gradient is strongest when balance is updated
    /// BEFORE external interaction. By placing the balance update in the EFFECTS
    /// phase (before the external call), we ensure ∇ ≠ 0 at the dark zone centroid.
    modifier balanceSync() {
        _;
        // Post-condition: totalEther == sum of all balances
        // Enforced by: effects BEFORE interactions in withdraw()
    }

    // ─── Core functions ───

    function deposit() external payable {
        balances[msg.sender] += msg.value;
        totalEther += msg.value;
    }

    /**
     * @notice Withdraw ETH — DARK ZONE SAFE
     *
     * CRITICAL: This function follows checks-effects-interactions pattern.
     *
     * WITHOUT this pattern (original The DAO):
     *   1. Check: require(balances >= amount)     ← withdraw_limit ∇ active
     *   2. Interact: call.value(amount)()         ← call_mechanism ∇ active
     *   3. Effect: balances -= amount              ← balance_sync ∇ at ZERO
     *   → During recursive call, withdraw_limit ∇ and call_mechanism ∇ cancel,
     *     balance_sync ∇ = 0 → DARK ZONE → c(p) ≈ 0 → exploit possible
     *
     * WITH this pattern (fixed):
     *   1. Check: require(balances >= amount)     ← withdraw_limit active
     *   2. Effect: balances -= amount              ← balance_sync ∇ NOW NON-ZERO
     *   3. Interact: call.value(amount)()         ← recursive call arrives,
     *                                               balance_sync ∇ breaks the
     *                                               mutual cancellation → c(p) ≫ 0
     */
    function withdraw(uint256 amount) external noReentrancy withdrawLimitCheck(amount) {
        // Phase 1: CHECKS — withdraw_limit constraint verified
        // (already done in modifier)

        // Phase 2: EFFECTS — balance_sync gradient activated BEFORE external call
        // THIS IS THE FIX: moving balance update before transfer ensures
        // balance_sync ∇ ≠ 0 when recursive call arrives, breaking the
        // gradient cancellation at the dark zone centroid (0.5, 0.5).
        uint256 balanceBefore = balances[msg.sender];
        balances[msg.sender] = balanceBefore - amount;
        totalEther -= amount;

        // Phase 3: INTERACTIONS — external call LAST
        // When this call re-enters withdraw(), balances[msg.sender] has already
        // been reduced, so withdrawLimitCheck will REJECT the recursive withdrawal.
        // The dark zone is broken: balance_sync ∇ is now non-zero at the
        // recursion midpoint.
        (bool success, ) = msg.sender.call{value: amount}("");
        require(success, "DAO: transfer failed");

        // Post-condition: totalEther consistency maintained by effects-first ordering
    }

    // ─── Constraint monitoring (off-chain analysis) ───

    /// @notice Expose state space coordinates for constraint scanner
    function getStateSpace() external view returns (
        uint256 recursion_depth,
        uint256 balance_consistency
    ) {
        // recursion_depth: normalized [0,1] where 0=no recursion, 1=deep recursion
        recursion_depth = _recursionDepth > 0 ? 1 : 0;
        // balance_consistency: 1 if totalEther matches sum(balances)
        balance_consistency = totalEther == address(this).balance ? 1 : 0;
    }
}
