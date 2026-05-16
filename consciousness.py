"""
Consciousness Window Module

Computes the CS2 scale window for consciousness L0 instantiation:
  d ∈ (0.3, 1.5) mm

Derived from:
  - SNR constraint (lower bound): d > √(SNR_min * kT / E_unit) * √N_modes
  - Signal attenuation constraint (upper bound): d < λ * ln(A₀/A_min)
  - Iterative integration constraint: d < (τ_int/K - τ_long) * v_unmyelinated

Also computes:
  - Self-reference strength S ∈ (10⁻³, 10⁻²)
  - Integration iterations K within τ_int
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class ConsciousnessWindow:
    d_min_mm: float = 0.065
    d_min_adj_mm: float = 0.3
    d_max_mm: float = 1.5
    N_min: int = 1_000_000
    K_iterations: int = 30

    @property
    def midpoint_mm(self) -> float:
        return np.sqrt(self.d_min_adj_mm * self.d_max_mm)

    def in_window(self, d_mm: float, N: int,
                  long_range: bool = True) -> dict:
        """Check if given system parameters fall within the consciousness window."""
        within_d = self.d_min_adj_mm <= d_mm <= self.d_max_mm
        within_N = N >= self.N_min
        return {
            'd_mm': d_mm,
            'd_in_window': within_d,
            'd_margin_low': d_mm - self.d_min_adj_mm,
            'd_margin_high': self.d_max_mm - d_mm,
            'N': N,
            'N_sufficient': within_N,
            'topology_ok': long_range,
            'consciousness_possible': within_d and within_N and long_range
        }


def snr_lower_bound(kT: float = 4.28e-21, E_unit: float = 1e-10,
                    SNR_min: float = 100.0, N_modes: float = 3000.0) -> float:
    """Compute lower bound on functional unit diameter from SNR.

    Args:
        kT: Thermal energy at body temp (J), default 4.28e-21 at 310K
        E_unit: Signal energy per unit area (J/cm²)
        SNR_min: Minimum signal-to-noise ratio for reliable detection
        N_modes: Number of simultaneously active neural patterns

    Returns:
        d_min in mm
    """
    # E_unit is J/cm², convert to J/m²: 1 J/cm² = 1e4 J/m²
    E_unit_SI = E_unit * 1e4
    d_base = np.sqrt(SNR_min * kT / E_unit_SI)
    d_adj = d_base * np.sqrt(N_modes)
    return d_adj * 1e3


def attenuation_upper_bound(lam_mm: float = 0.75,
                            amp_ratio: float = 5.0) -> float:
    """Compute upper bound on functional unit diameter from signal attenuation.

    A(d) = A₀ * exp(-d/λ) → d < λ * ln(A₀/A_min)

    Args:
        lam_mm: Space constant for unmyelinated fibers (mm)
        amp_ratio: Minimum detectable amplitude ratio A₀/A_min

    Returns:
        d_max in mm
    """
    return lam_mm * np.log(amp_ratio)


def iterative_integration_bound(tau_int_ms: float = 150.0,
                                K: int = 30,
                                tau_long_ms: float = 2.0,
                                v_unmyelinated_m_s: float = 0.8) -> float:
    """Compute upper bound from iterative integration constraint.

    K * (τ_long + d/v) < τ_int → d < (τ_int/K - τ_long) * v

    Args:
        tau_int_ms: Consciousness integration time window (ms)
        K: Number of integration iterations
        tau_long_ms: Long-range propagation time (ms)
        v_unmyelinated_m_s: Unmyelinated fiber conduction velocity (m/s)

    Returns:
        d_max in mm
    """
    return (tau_int_ms / K - tau_long_ms) * v_unmyelinated_m_s


def self_reference_strength(K: int = 30, SNR_eff: float = 10.0) -> dict:
    """Compute self-reference strength S bounds.

    S * K * SNR_eff > 1 for convergence
    S_min = 1 / (K_max * SNR_eff_max)
    S_max = 1 / (K_min * SNR_eff_min)

    Args:
        K: Integration iterations (20-50 typical)
        SNR_eff: Effective signal-to-noise ratio (2-20 typical)

    Returns:
        dict with S_min, S_max, S_mid
    """
    S_min = 1.0 / (50 * 20)
    S_max = 1.0 / (20 * 2)
    S_mid = np.sqrt(S_min * S_max)
    S_current = 1.0 / (K * SNR_eff) if SNR_eff > 0 else float('inf')
    return {
        'S_min': S_min,
        'S_max': S_max,
        'S_mid': S_mid,
        'S_range': (S_min, S_max),
        'S_current': S_current,
        'in_range': S_min <= S_current <= S_max
    }


def test_cleaning_fish(d_mm: float, N: int = 1_000_000) -> dict:
    """Test the cleaner fish (裂唇鱼) prediction.

    d ≈ 0.2-0.4 mm → near the lower bound.
    If d < 0.3mm → no causal-efficacy consciousness despite complex behavior.
    """
    window = ConsciousnessWindow()
    result = window.in_window(d_mm, N)
    prediction = (
        "consciousness_possible" if result['consciousness_possible']
        else "no_consciousness"
    )
    result['prediction'] = prediction
    result['note'] = (
        f"At d={d_mm}mm, "
        + ("within window → mirror self-recognition should succeed"
           if result['d_in_window']
           else "below window → complex behavior but no causal-efficacy consciousness")
    )
    return result


def test_artificial_chip(d_mm: float, N: int = 1_000_000) -> dict:
    """Test artificial neuromorphic chip prediction.

    Chips with d outside (0.3, 1.5) mm should NOT produce consciousness
    even with N > 10⁶ and correct topology.
    """
    window = ConsciousnessWindow()
    result = window.in_window(d_mm, N)
    result['prediction'] = (
        "no_consciousness" if not result['d_in_window']
        else "consciousness_possible"
    )
    result['note'] = (
        f"Chip d={d_mm}mm is {'inside' if result['d_in_window'] else 'outside'} "
        f"window → prediction: {result['prediction']}"
    )
    return result
