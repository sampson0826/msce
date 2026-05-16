"""
State mapper — bridges EVM contract storage to the engine's abstract state space.

In the constraint residual engine, p is an np.ndarray of float coordinates.
In a DeFi protocol, the actual state is:
  - Storage slots (uint256 values at specific slots)
  - Nested mappings (e.g., balances[user] at keccak256(user, slot))
  - Transient state (e.g., reentrancy lock, block.timestamp)

This module defines:
  StateDimension: metadata for one dimension of the engine state space
  ContractState: a snapshot of a contract's state at a block
  StateMapper: converts between EVM state and engine coordinates
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable
from enum import Enum


class DimensionType(Enum):
    RATIO = "ratio"             # normalized to [0, 1+] range (e.g., health factor)
    BALANCE = "balance"         # raw token amount
    BOOLEAN = "boolean"         # 0 or 1
    TIMESTAMP = "timestamp"     # block timestamp
    ADDRESS = "address"         # normalized address (for access control)
    CUSTOM = "custom"           # user-defined normalization


@dataclass
class StateDimension:
    """One axis of the constraint residual state space.

    Maps an EVM storage location (or derived value) to a dimension index
    in the engine's state vector p.

    Attributes:
        index: position in p array
        name: human-readable label
        source: where this value comes from — 'storage:<slot>', 'derived:<expr>', 'input:<param>'
        dim_type: semantic type, drives normalization
        raw_min, raw_max: bounds in raw EVM units (for normalization)
        normalize_fn: optional custom normalization function
        current_value: latest known value (populated from chain data)
    """
    index: int
    name: str
    source: str
    dim_type: DimensionType = DimensionType.RATIO
    raw_min: float = 0.0
    raw_max: float = 1.0
    normalize_fn: Optional[Callable[[float], float]] = None
    current_value: Optional[float] = None

    def normalize(self, raw_value: float) -> float:
        """Convert a raw EVM value to engine coordinate."""
        if self.normalize_fn is not None:
            return self.normalize_fn(raw_value)
        if self.dim_type == DimensionType.RATIO:
            return raw_value  # already a ratio
        elif self.dim_type == DimensionType.BOOLEAN:
            return 1.0 if raw_value > 0 else 0.0
        elif self.dim_type == DimensionType.BALANCE:
            if self.raw_max == 0:
                return raw_value
            return raw_value / self.raw_max
        elif self.dim_type == DimensionType.TIMESTAMP:
            if self.raw_max == self.raw_min:
                return 0.0
            return (raw_value - self.raw_min) / (self.raw_max - self.raw_min)
        return raw_value

    def denormalize(self, coord: float) -> float:
        """Convert engine coordinate back to raw EVM value."""
        if self.normalize_fn is not None:
            return coord  # irreversible without inverse fn
        if self.dim_type == DimensionType.RATIO:
            return coord
        elif self.dim_type == DimensionType.BOOLEAN:
            return 1.0 if coord > 0.5 else 0.0
        elif self.dim_type == DimensionType.BALANCE:
            return coord * self.raw_max
        elif self.dim_type == DimensionType.TIMESTAMP:
            return coord * (self.raw_max - self.raw_min) + self.raw_min
        return coord


@dataclass
class ContractState:
    """A snapshot of a contract's state at a specific block.

    Populated from chain data (RPC, archive node, or manual construction).
    """
    address: str
    block_number: int
    dimensions: list[StateDimension]
    raw_values: dict[str, float] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def to_engine_point(self) -> np.ndarray:
        """Convert this contract state to an engine state vector p."""
        p = np.zeros(len(self.dimensions), dtype=float)
        for dim in self.dimensions:
            raw = self.raw_values.get(dim.source, 0.0)
            p[dim.index] = dim.normalize(raw)
        return p

    def update_raw_value(self, source: str, value: float):
        """Set a raw value from chain data."""
        self.raw_values[source] = value


class StateMapper:
    """Manages the mapping between a protocol's EVM state and engine coordinates.

    Usage:
        mapper = StateMapper()
        mapper.add_dimension(StateDimension(0, "health_factor", "derived:collateral*ltv/debt",
                                            DimensionType.RATIO))
        mapper.add_dimension(StateDimension(1, "reentrancy_lock", "storage:0x5",
                                            DimensionType.BOOLEAN))
        ...
        p = mapper.build_point({"storage:0x5": 0, "derived:health_factor": 1.2})
        # p = array([1.2, 0.0, ...])
    """

    def __init__(self):
        self.dimensions: list[StateDimension] = []
        self._source_index: dict[str, int] = {}

    def add_dimension(self, dim: StateDimension):
        self.dimensions.append(dim)
        self._source_index[dim.source] = dim.index

    @property
    def n_dims(self) -> int:
        return len(self.dimensions)

    def build_point(self, raw_values: dict[str, float]) -> np.ndarray:
        """Construct an engine state vector from raw EVM values.

        Args:
            raw_values: dict mapping source strings to raw values
        Returns:
            np.ndarray of shape (n_dims,)
        """
        p = np.zeros(self.n_dims, dtype=float)
        for dim in self.dimensions:
            raw = raw_values.get(dim.source, dim.current_value or 0.0)
            p[dim.index] = dim.normalize(raw)
        return p

    def build_bounds(self) -> list[tuple[float, float]]:
        """Generate scan bounds from dimension definitions."""
        bounds = []
        for dim in self.dimensions:
            lo = dim.normalize(dim.raw_min)
            hi = dim.normalize(dim.raw_max)
            if lo == hi:
                lo, hi = 0.0, 2.0  # default range for undefined bounds
            if lo > hi:
                lo, hi = hi, lo
            bounds.append((lo, hi))
        return bounds

    def explain_point(self, p: np.ndarray) -> dict[str, float]:
        """Convert engine coordinates back to human-readable values."""
        explained = {}
        for dim in self.dimensions:
            coord = p[dim.index]
            explained[dim.name] = dim.denormalize(coord)
        return explained
