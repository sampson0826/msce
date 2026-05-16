"""
DeFi Adapter — bridges the constraint residual engine to real smart contract analysis.

Layers:
  constraint_templates.py — reusable σ_i factories for common DeFi patterns
  state_mapper.py         — EVM storage layout ↔ engine state space
  scanner.py              — full pipeline orchestration
  cases/                  — historical exploit case studies
"""

from constraint_residual.defi_adapter.constraint_templates import (
    ConstraintTemplate,
    BarrierType,
    make_barrier,
    balance_conservation,
    collateral_health,
    reentrancy_guard,
    allowance_check,
    liquidation_permission,
    supply_cap,
    access_control,
    timelock_delay,
)

from constraint_residual.defi_adapter.state_mapper import (
    StateMapper,
    StateDimension,
    ContractState,
)

from constraint_residual.defi_adapter.scanner import DeFiScanner, ScanReport
