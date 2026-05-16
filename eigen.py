"""
Eigen Threshold Module

Computes the error threshold for biological L0 activation:
  q^N > 1/σ

Where:
  q = per-monomer replication fidelity
  N = sequence length
  σ = superiority (A_max / A_avg) ∈ (1.1, 10) typically

Also computes phase transition observables:
  - Sequence entropy S(q,N) vs q^N
  - Transition width Δ(q^N) ≈ 1/√M
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class EigenResult:
    q: float
    N: int
    sigma: float
    q_pow_N: float
    threshold: float
    activated: bool

    @property
    def margin(self) -> float:
        return self.q_pow_N - self.threshold


def eigen_threshold(q: float, N: int, sigma: float = 2.0) -> EigenResult:
    """Compute whether biological L0 is activated for given parameters.

    Args:
        q: Per-monomer replication fidelity (0 < q ≤ 1)
        N: Sequence length (number of monomers)
        sigma: Superiority of master sequence over mutants (default 2.0)

    Returns:
        EigenResult with activation status
    """
    q_pow_N = q ** N
    threshold = 1.0 / sigma
    return EigenResult(
        q=q, N=N, sigma=sigma,
        q_pow_N=q_pow_N,
        threshold=threshold,
        activated=q_pow_N > threshold
    )


def sequence_entropy(q: float, N: int, M: int = 1000) -> float:
    """Estimate sequence entropy S for given replication parameters.

    S ≈ ln(N_eff) where N_eff is effective number of sequences.
    Below threshold: S ≈ N * ln(2) (uniform over sequence space)
    Above threshold: S → 0 (master sequence dominates)

    Args:
        q: fidelity
        N: sequence length
        M: population size (for transition width)
    """
    q_pow_N = q ** N
    sigma_c = 2.0
    threshold = 1.0 / sigma_c

    if q_pow_N < threshold:
        return N * np.log(2)
    else:
        excess = (q_pow_N - threshold) / threshold
        if excess > 1.0:
            return 0.0
        else:
            return N * np.log(2) * (1.0 - excess)


def transition_width(M: int) -> float:
    """Estimate phase transition width Δ(q^N) ≈ 1/√M."""
    return 1.0 / np.sqrt(M)


def critical_length(q: float, sigma: float = 2.0) -> float:
    """Maximum sequence length N_max that still maintains information.

    From q^N > 1/σ:
        N < ln(1/σ) / ln(q) = -ln(σ) / ln(q)
    """
    if q >= 1.0:
        return float('inf')
    return -np.log(sigma) / np.log(q)


def phase_diagram(q_range: tuple[float, float], N_range: tuple[int, int],
                  sigma: float = 2.0, n_q: int = 100) -> dict:
    """Generate phase diagram data for plotting.

    Returns dict with 'q_vals', 'N_vals', 'Z' (activation matrix).
    """
    q_vals = np.linspace(q_range[0], q_range[1], n_q)
    N_vals = np.arange(N_range[0], N_range[1] + 1)
    Z = np.zeros((len(N_vals), len(q_vals)))

    for i, N in enumerate(N_vals):
        for j, q in enumerate(q_vals):
            Z[i, j] = 1.0 if q**N > 1.0 / sigma else 0.0

    return {
        'q_vals': q_vals.tolist(),
        'N_vals': N_vals.tolist(),
        'Z': Z.tolist(),
        'phase_boundary': {N: np.exp(np.log(1.0/sigma)/N) for N in N_vals
                          if np.exp(np.log(1.0/sigma)/N) <= 1.0}
    }


def verify_sigma_bounds():
    """Verify σ ∈ (1.1, 10) for functional sequences.
    This constraint is derived in the 稳固版 document.
    """
    tests = []
    for sigma in [1.05, 1.1, 2.0, 5.0, 10.0, 15.0]:
        q_test = 0.99
        for N in [50, 100, 200, 500]:
            result = eigen_threshold(q_test, N, sigma)
            tests.append({
                'sigma': sigma, 'N': N,
                'q^N': result.q_pow_N,
                'threshold': result.threshold,
                'activated': result.activated
            })
    return tests
