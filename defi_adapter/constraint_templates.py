"""
Constraint templates for common DeFi security patterns.

Each template is a factory that produces Rule objects for the constraint residual
engine. The key design decision: DeFi constraints are fundamentally DISCRETE
(boolean assertions like require(balance >= amount)), but the engine needs
CONTINUOUS differentiable σ(p) functions.

Bridge: use barrier/sigmoid functions that encode "distance from violation" as
constraint tension. At violation boundary, ∇σ is maximal — the constraint is
"screaming." Far from boundary, ∇σ → 0 — the constraint is "quiet."

A dark zone forms when multiple constraints have strong but opposing gradients
at the same point: each constraint is individually screaming, but they cancel.
"""

import numpy as np
from enum import Enum
from typing import Callable, Optional
from dataclasses import dataclass

from constraint_residual.core import Rule


# ---------------------------------------------------------------------------
# Barrier functions — the bridge from discrete to continuous
# ---------------------------------------------------------------------------

class BarrierType(Enum):
    SIGMOID = "sigmoid"        # smooth, differentiable everywhere
    SOFTPLUS = "softplus"      # one-sided, ~0 on safe side, linear growth on violation side
    GAUSSIAN = "gaussian"      # peak at threshold, decays both sides
    EXPONENTIAL = "exponential"  # sharp on violation side


def make_barrier(
    x: np.ndarray,
    threshold: float,
    barrier_type: BarrierType = BarrierType.SIGMOID,
    steepness: float = 10.0,
    invert: bool = False,
) -> float:
    """Compute constraint strength from a ratio variable and its threshold.

    Args:
        x: ratio value(s) — e.g. health_factor, allowance/amount, balance/supply
        threshold: the boundary value where constraint activates
        barrier_type: shape of the activation curve
        steepness: sharpness of transition (higher = more like discrete boolean)
        invert: if True, σ → 1 when x is BELOW threshold (for minimum requirements)

    Returns:
        σ ∈ [0, 1] — constraint strength at this state point
    """
    x0 = float(x[0]) if isinstance(x, np.ndarray) else float(x)
    diff = x0 - threshold
    if invert:
        diff = -diff

    if barrier_type == BarrierType.SIGMOID:
        return float(1.0 / (1.0 + np.exp(-steepness * diff)))
    elif barrier_type == BarrierType.SOFTPLUS:
        return float(np.log(1.0 + np.exp(steepness * diff))) / steepness
    elif barrier_type == BarrierType.GAUSSIAN:
        return float(np.exp(-steepness * diff ** 2))
    elif barrier_type == BarrierType.EXPONENTIAL:
        return float(np.exp(-steepness * max(-diff, 0)))
    return 0.0


def make_barrier_gradient(
    x: np.ndarray,
    threshold: float,
    barrier_type: BarrierType = BarrierType.SIGMOID,
    steepness: float = 10.0,
    invert: bool = False,
    dim: int = 0,
) -> np.ndarray:
    """Analytic gradient of barrier w.r.t. state vector p.

    Assumes x depends only on p[dim].
    """
    grad = np.zeros_like(x, dtype=float)
    x0 = float(x[dim])
    diff = x0 - threshold
    if invert:
        diff = -diff

    if barrier_type == BarrierType.SIGMOID:
        sig = 1.0 / (1.0 + np.exp(-steepness * diff))
        d = steepness * sig * (1.0 - sig)
    elif barrier_type == BarrierType.SOFTPLUS:
        d = steepness / (1.0 + np.exp(-steepness * diff)) / steepness
        d = 1.0 / (1.0 + np.exp(-steepness * diff))
    elif barrier_type == BarrierType.GAUSSIAN:
        d = -2.0 * steepness * diff * np.exp(-steepness * diff ** 2)
    elif barrier_type == BarrierType.EXPONENTIAL:
        if diff < 0:
            d = steepness * np.exp(-steepness * (-diff))
        else:
            d = 0.0
    else:
        d = 0.0

    if invert:
        d = -d

    grad[dim] = d
    return grad


# ---------------------------------------------------------------------------
# Constraint template dataclass
# ---------------------------------------------------------------------------

@dataclass
class ConstraintTemplate:
    """Factory specification for a DeFi constraint function.

    Rather than requiring users to write lambda functions directly, templates
    capture the semantic type of a DeFi constraint and generate the appropriate
    σ(p) and ∇σ(p) from parameters.
    """
    pattern: str           # e.g. "collateral_health", "reentrancy_guard"
    param_dim: int         # which state dimension this constraint reads
    threshold: float       # the threshold value
    barrier_type: BarrierType = BarrierType.SIGMOID
    steepness: float = 10.0
    invert: bool = False   # True if constraint activates BELOW threshold
    domain: str = "defi"


def template_to_rule(t: ConstraintTemplate, name: str, layer: int = 2) -> Rule:
    """Convert a constraint template into an engine Rule."""
    def constraint_fn(p: np.ndarray) -> float:
        return make_barrier(p, t.threshold, t.barrier_type, t.steepness, t.invert)

    def gradient_fn(p: np.ndarray) -> np.ndarray:
        return make_barrier_gradient(p, t.threshold, t.barrier_type, t.steepness, t.invert, t.param_dim)

    return Rule(
        name=name,
        layer=layer,
        domain=t.domain,
        constraint_fn=constraint_fn,
        gradient_fn=gradient_fn,
        certainty=0.95,
    )


# ---------------------------------------------------------------------------
# Standard DeFi constraint factories
# ---------------------------------------------------------------------------

def balance_conservation(param_dim: int = 0, tolerance: float = 0.001) -> ConstraintTemplate:
    """Σ balances == totalSupply — fundamental accounting invariant.

    σ → 1 when the sum of individual balances diverges from total supply.
    This is almost always a hard constraint (steep). When it's violated,
    the protocol is in an inconsistent state.
    """
    return ConstraintTemplate(
        pattern="balance_conservation",
        param_dim=param_dim,
        threshold=tolerance,
        barrier_type=BarrierType.GAUSSIAN,
        steepness=500.0,
        invert=False,
    )


def collateral_health(param_dim: int = 0, ltv: float = 0.8) -> ConstraintTemplate:
    """collateral * LTV >= debt — position must be over-collateralized.

    σ → 1 when health factor approaches 1.0 from above.
    This is the primary constraint in lending protocols.
    """
    return ConstraintTemplate(
        pattern="collateral_health",
        param_dim=param_dim,
        threshold=1.0,  # health factor = 1.0 is the liquidation boundary
        barrier_type=BarrierType.SIGMOID,
        steepness=8.0,
        invert=True,  # activate when health_factor < 1.0
    )


def reentrancy_guard(param_dim: int = 0) -> ConstraintTemplate:
    """Reentrancy lock — prevents recursive calls.

    σ → 1 when lock is engaged (1). σ → 0 when unlocked (0).
    Discrete by nature, approximated with very steep sigmoid.
    """
    return ConstraintTemplate(
        pattern="reentrancy_guard",
        param_dim=param_dim,
        threshold=0.5,
        barrier_type=BarrierType.SIGMOID,
        steepness=50.0,  # very steep — nearly discrete
        invert=False,
    )


def allowance_check(param_dim: int = 0) -> ConstraintTemplate:
    """allowance >= amount — spender has sufficient approval.

    σ → 1 when amount approaches or exceeds allowance.
    """
    return ConstraintTemplate(
        pattern="allowance_check",
        param_dim=param_dim,
        threshold=1.0,  # allowance/amount ratio
        barrier_type=BarrierType.SIGMOID,
        steepness=10.0,
        invert=True,  # activate when ratio < 1.0
    )


def liquidation_permission(param_dim: int = 0) -> ConstraintTemplate:
    """Liquidation is only allowed when health factor < 1.0.

    σ → 1 when health factor > 1.0 (liquidation blocked).
    σ → 0 when health factor < 1.0 (liquidation allowed — constraint released).

    This is the complement of collateral_health: one screams when position is
    unhealthy, the other screams when liquidation is NOT permitted.
    """
    return ConstraintTemplate(
        pattern="liquidation_permission",
        param_dim=param_dim,
        threshold=1.0,
        barrier_type=BarrierType.SIGMOID,
        steepness=8.0,
        invert=False,  # activate when health_factor > 1.0 (block liquidation)
    )


def supply_cap(param_dim: int = 0, cap: float = 1.0) -> ConstraintTemplate:
    """total_borrow <= supply_cap — protocol-level borrowing limit.

    σ → 1 when utilization approaches the cap.
    """
    return ConstraintTemplate(
        pattern="supply_cap",
        param_dim=param_dim,
        threshold=cap,
        barrier_type=BarrierType.SOFTPLUS,
        steepness=15.0,
        invert=True,  # activate when utilization > cap
    )


def access_control(param_dim: int = 0) -> ConstraintTemplate:
    """Only authorized callers can execute restricted functions.

    Discrete: 1 if authorized, 0 if not. Steep sigmoid approximation.
    """
    return ConstraintTemplate(
        pattern="access_control",
        param_dim=param_dim,
        threshold=0.5,
        barrier_type=BarrierType.SIGMOID,
        steepness=100.0,  # essentially discrete
        invert=False,
    )


def timelock_delay(param_dim: int = 0, delay: float = 1.0) -> ConstraintTemplate:
    """Operation must wait a minimum delay period.

    σ → 1 before delay has elapsed. σ → 0 after.
    """
    return ConstraintTemplate(
        pattern="timelock_delay",
        param_dim=param_dim,
        threshold=delay,
        barrier_type=BarrierType.SOFTPLUS,
        steepness=5.0,
        invert=True,  # activate before delay elapsed
    )


# ---------------------------------------------------------------------------
# Custom multi-dimensional constraint factory
# ---------------------------------------------------------------------------

def make_multidim_constraint(
    name: str,
    fn: Callable[[np.ndarray], float],
    grad_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    layer: int = 2,
    domain: str = "defi",
) -> Rule:
    """Create a custom Rule for multi-dimensional DeFi constraints.

    For constraints that depend on multiple state dimensions (e.g.,
    liquidation bonus = f(collateral_ratio, debt_amount)), use this
    to define the constraint function directly.
    """
    return Rule(
        name=name,
        layer=layer,
        domain=domain,
        constraint_fn=fn,
        gradient_fn=grad_fn,
        certainty=0.9,
    )


# ---------------------------------------------------------------------------
# Compound constraint: a constraint formed by interaction of multiple rules
# ---------------------------------------------------------------------------

def make_interaction_constraint(
    name: str,
    rules: list[Rule],
    combine_fn: Callable[[list[float]], float],
    layer: int = 2,
) -> Rule:
    """Create a constraint that measures interaction between other constraints.

    This is the key abstraction for cross-protocol dark zone detection.
    An interaction constraint captures how two or more rules from different
    protocols interact in a shared state space.
    """
    def constraint_fn(p: np.ndarray) -> float:
        values = [r.constraint_fn(p) for r in rules]
        return combine_fn(values)

    return Rule(
        name=name,
        layer=layer,
        domain="cross_protocol",
        constraint_fn=constraint_fn,
        gradient_fn=None,  # numerical gradient
        certainty=0.7,
    )
