// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title MultiSigWallet — Constraint-Aware Implementation
 * @notice Generated from constraint DSL spec: parity_wallet
 *
 * STATE SPACE:
 *   initialization_status ∈ [0, 1]
 *     0 = uninitialized (initWallet never called), 1 = properly initialized
 *   ownership_legitimacy ∈ [0, 1]
 *     0 = no owner or attacker controls, 1 = legitimate owner established
 *
 * CONSTRAINTS:
 *   - init_guard (L2, access_control): only_uninitialized modifier prevents re-initialization. Protection ≈ 0 when unin...
 *   - owner_guard (L2, access_control): only_owner modifier. Requires BOTH initialization AND legitimate owner. If eithe...
 *   - deploy_order (L1, lifecycle): MISSING CONSTRAINT. Should ensure library contract is initialized at deploy time...
 *
 * DARK ZONE: cold_start_gap
 *   Location: initialization_status ≈ 0, ownership_legitimacy ≈ 0
 *   c(p) → 1, Σ||∇σ|| ≈ 0, ||Π|| ≈ 0
 *   Both init_guard and owner_guard are inactive at the origin.
 *
 * COUNTERMEASURE: constructor initialization + deploy_order constraint
 *   The constructor sets owner and marks initialized, pushing the initial
 *   state from (0, 0) to (1, 1).  The deploy_order constraint (L1 lifecycle)
 *   ensures the library pattern cannot leave a contract in uninitialized state.
 *
 * ORIGINAL VULNERABILITY:
 *   The WalletLibrary contract was deployed as a standalone contract but
 *   never had initWallet() called on it directly.  At (x=0, y=0):
 *   - init_guard σ ≈ 0 (anyone can call initWallet)
 *   - owner_guard σ ≈ 0 (no owner to restrict kill())
 *   Combined ||Π|| ≈ 0 — zero protection at the initial state.
 *
 * FIX:
 *   Constructor initializes ownership at deploy time.  The initWallet()
 *   function is replaced by constructor logic.  The library contract
 *   cannot exist in an uninitialized state — the cold start gap is closed.
 */
contract MultiSigWallet {
    // ─── State variables ───
    address[] public owners;
    uint256 public required;
    bool public initialized;

    // ─── Constraint: init_guard (L2, access_control) ───
    // σ_init ≈ sigmoid((x - 0.3) / 0.04)
    // Protection ≈ 0 when x < 0.3, ≈ 1 when x > 0.3.
    //
    // FIX: Constructor sets initialized = true at deploy time.
    // The contract CANNOT exist at x < 0.3 — the cold start gap is
    // eliminated by construction.

    modifier onlyUninitialized() {
        require(!initialized,
            "Parity: init_guard — contract already initialized");
        _;
    }

    // ─── Constraint: owner_guard (L2, access_control) ───
    // σ_owner ≈ sigmoid(x) * sigmoid(y)
    // Protection requires BOTH initialization AND legitimate owner.
    //
    // FIX: Since constructor sets both x=1 and y=1, owner_guard σ ≈ 1
    // at the initial state.  No window where y < 0.3.

    modifier onlyOwner() {
        bool isOwner = false;
        for (uint256 i = 0; i < owners.length; i++) {
            if (owners[i] == msg.sender) {
                isOwner = true;
                break;
            }
        }
        require(isOwner,
            "Parity: owner_guard — caller is not an owner");
        _;
    }

    // ─── Constraint: deploy_order (L1, lifecycle) ───
    // THE MISSING CONSTRAINT — now activated.
    // Ensures contract cannot be used before initialization.
    // Activated by constructor.

    modifier onlyInitialized() {
        require(initialized,
            "Parity: deploy_order — contract not yet initialized");
        _;
    }

    // ─── Constructor — closes the cold start gap ───
    //
    // KEY: The constructor pushes the initial state from (0, 0) to (1, 1).
    // After construction:
    //   x = initialization_status = 1 → init_guard σ ≈ 1
    //   y = ownership_legitimacy = 1 → owner_guard σ ≈ 1
    //   ||Π|| at initial state: from 0.271 → above threshold
    //
    // This is the deploy_order constraint in action: the system CANNOT
    // exist in an uninitialized state.

    constructor(address[] memory _owners, uint256 _required) {
        require(_owners.length > 0, "Parity: need at least one owner");
        require(_required > 0 && _required <= _owners.length,
            "Parity: required must be between 1 and owner count");

        owners = _owners;
        required = _required;
        initialized = true;  // ← COLD START GAP CLOSED
    }

    // ─── Wallet operations ───

    function execute(address to, uint256 value, bytes memory data)
        external onlyInitialized onlyOwner
    {
        // All privileged operations require:
        // 1. Contract is initialized (deploy_order + init_guard)
        // 2. Caller is an owner (owner_guard)
        //
        // With constructor initialization, both constraints are active
        // from block 0.  No cold start gap.
        (bool success, ) = to.call{value: value}(data);
        require(success, "Parity: external call failed");
    }

    // ─── Self-destruct (the function exploited in 2017) ───
    //
    // In the original Parity: anyone could call initWallet() on the library
    // contract, become owner, then call kill().  Cold start gap: both
    // constraints inactive at (x=0, y=0).
    //
    // In this version: constructor sets owner.  onlyOwner modifier prevents
    // unauthorized kill().  The deploy_order constraint ensures initialization
    // happens at deploy time, not via a separate transaction.

    function kill() external onlyInitialized onlyOwner {
        selfdestruct(payable(msg.sender));
    }

    // ─── Constraint monitoring ───

    function getStateSpace() external view returns (
        uint256 initialization_status,
        uint256 ownership_legitimacy
    ) {
        initialization_status = initialized ? 1 : 0;
        // ownership_legitimacy: 1 if owners array is non-empty and initialized
        ownership_legitimacy = (initialized && owners.length > 0) ? 1 : 0;
    }
}
