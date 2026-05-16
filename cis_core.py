"""
CIS Core v1 — Complete A.1-A.7 Constraint Analysis Pipeline

Implements the full Constraint Invisibility Scanner mathematical framework
on top of ConstraintField.

A.1  Π(p) = Σ∇σ_i(p)                          — constraint residual vector field
A.2  c(p) = ||Π|| / Σ||∇σ||                     — cancellation ratio
A.3  g_ij = Σ(∂σ_k/∂x_i)(∂σ_k/∂x_j)            — Riemannian metric
A.4  Π = -∇φ + J∇ψ (Helmholtz decomposition)    — executor type classification
A.5  ∂ρ/∂t + ∇·Π = 0 → ∇·Π ≠ 0                — missing executor location
A.6  Executor type classification: E-I, E-II, E-III
A.7  Constraint decay law: S_{n+1} = S_n · (1-β)

Usage:
  from constraint_residual.cis_core import CISAnalyzer
  analyzer = CISAnalyzer(field, bounds=[(0,1),(0,1)], n_points=100)
  report = analyzer.full_analysis()
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ═══════════════════════════════════════════════════════════════
# A.1: Constraint Residual Vector Field
# ═══════════════════════════════════════════════════════════════

@dataclass
class VectorField2D:
    """2D vector field Π(x,y) = (u(x,y), v(x,y)) on a regular grid."""
    xs: np.ndarray       # (N,) x-coordinates
    ys: np.ndarray       # (N,) y-coordinates
    u: np.ndarray        # (N, N) x-component
    v: np.ndarray        # (N, N) y-component
    magnitude: np.ndarray  # (N, N) ||Π||

    @property
    def n_points(self) -> int:
        return len(self.xs)

    @property
    def dx(self) -> float:
        return self.xs[1] - self.xs[0]

    @property
    def dy(self) -> float:
        return self.ys[1] - self.ys[0]


# ═══════════════════════════════════════════════════════════════
# A.2: Cancellation Ratio
# ═══════════════════════════════════════════════════════════════

def compute_cancellation_ratio(combined_magnitude: np.ndarray,
                                individual_magnitudes: np.ndarray) -> np.ndarray:
    """c(p) = ||Σ∇σ_i|| / Σ||∇σ_i||"""
    total_indiv = np.sum(individual_magnitudes, axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        cr = np.where(total_indiv > 1e-10,
                      combined_magnitude / total_indiv,
                      1.0)  # cold start → c=1
    return cr


# ═══════════════════════════════════════════════════════════════
# A.3: Riemannian Metric g_ij
# ═══════════════════════════════════════════════════════════════

@dataclass
class RiemannianAnalysis:
    """Analysis of constraint metric g_ij = Σ(∂σ_k/∂x_i)(∂σ_k/∂x_j)."""
    g11: np.ndarray       # Σ(∂σ/∂x)²
    g12: np.ndarray       # Σ(∂σ/∂x)(∂σ/∂y)
    g22: np.ndarray       # Σ(∂σ/∂y)²
    eval_min: np.ndarray  # minimum eigenvalue of g at each point
    eval_max: np.ndarray  # maximum eigenvalue
    zero_direction: np.ndarray  # (N,N,2) eigenvector for eval_min → unconstrained direction
    condition_number: np.ndarray  # eval_max/eval_min → ∞ where one direction dominates

    def unconstrained_mask(self, eps: float = 0.05) -> np.ndarray:
        """Points where the minimum eigenvalue is near zero → truly unconstrained."""
        return self.eval_min < eps


def compute_riemannian_metric(individual_grads: np.ndarray) -> RiemannianAnalysis:
    """Compute g_ij from individual constraint gradients.

    Args:
        individual_grads: (K, N, N, 2) — gradient of each constraint at each grid point

    Returns:
        RiemannianAnalysis with metric tensor components and eigendecomposition.
    """
    K, N, M, _ = individual_grads.shape
    assert M == N, "Grid must be square"

    # g_ij = Σ_k (∂σ_k/∂x_i)(∂σ_k/∂x_j)
    g11 = np.sum(individual_grads[..., 0] * individual_grads[..., 0], axis=0)  # Σ(dx)²
    g12 = np.sum(individual_grads[..., 0] * individual_grads[..., 1], axis=0)  # Σ(dx·dy)
    g22 = np.sum(individual_grads[..., 1] * individual_grads[..., 1], axis=0)  # Σ(dy)²

    # Eigenvalues of 2×2 symmetric matrix g = [[g11, g12], [g12, g22]]
    # λ = (g11+g22 ± sqrt((g11-g22)² + 4g12²)) / 2
    trace = g11 + g22
    det = g11 * g22 - g12 * g12
    discriminant = np.sqrt(np.maximum((g11 - g22)**2 + 4 * g12**2, 0))

    eval_max = (trace + discriminant) / 2
    eval_min = (trace - discriminant) / 2

    # Eigenvector for eval_min (unconstrained direction)
    # For [[g11-λ, g12], [g12, g22-λ]] with eigenvalue λ = eval_min
    # eigenvector = (g22-λ_min, -g12) or (-g12, g11-λ_min)
    zero_dir = np.zeros((N, N, 2))
    # Use (g22-λ_min, -g12) when g12 dominates; otherwise use g11-g22 difference
    dx_zero = g22 - eval_min
    dy_zero = -g12

    # Normalize
    norm = np.sqrt(dx_zero**2 + dy_zero**2)
    with np.errstate(divide='ignore', invalid='ignore'):
        zero_dir[..., 0] = np.where(norm > 1e-10, dx_zero / norm, 0.0)
        zero_dir[..., 1] = np.where(norm > 1e-10, dy_zero / norm, 1.0)  # default to y

    # Condition number
    with np.errstate(divide='ignore', invalid='ignore'):
        cond = np.where(eval_min > 1e-10, eval_max / eval_min, np.inf)

    return RiemannianAnalysis(
        g11=g11, g12=g12, g22=g22,
        eval_min=eval_min, eval_max=eval_max,
        zero_direction=zero_dir,
        condition_number=cond,
    )


# ═══════════════════════════════════════════════════════════════
# A.4: Helmholtz Decomposition Π = -∇φ + J∇ψ
# ═══════════════════════════════════════════════════════════════

@dataclass
class HelmholtzDecomposition:
    """Π = -∇φ + J∇ψ where J = [[0,-1],[1,0]] is 90° rotation.

    In 2D: Π = (u, v) = (-∂φ/∂x - ∂ψ/∂y, -∂φ/∂y + ∂ψ/∂x)

    - ∇×Π = ∇²ψ → the curl source = solenoidal part (E-I: structural gaps)
    - ∇·Π = -∇²φ → the divergence source = irrotational part (E-II/III: parametric gaps)

    Executor classification:
      ∇×Π ≠ 0 → E-I (structural/mathematical, cannot be absorbed by a scalar constraint)
      ∇×Π = 0, ∇·Π ≠ 0 → E-II or E-III (can be fixed by adding a scalar constraint)
      ∇×Π = 0, ∇·Π = 0 → harmonic field, no executor needed
    """
    phi: np.ndarray          # scalar potential (irrotational part)
    psi: np.ndarray          # stream function (solenoidal part)
    divergence: np.ndarray   # ∇·Π
    curl: np.ndarray         # ∇×Π (scalar in 2D)
    irrotational_u: np.ndarray  # -∇φ (conservative part of Π)
    irrotational_v: np.ndarray
    solenoidal_u: np.ndarray   # J∇ψ (rotational part of Π)
    solenoidal_v: np.ndarray
    executor_type: np.ndarray   # (N,N) int: 0=none, 1=E-I, 2=E-II, 3=E-III
    structural_score: np.ndarray  # |∇×Π| / (|∇·Π| + |∇×Π| + ε)

    @property
    def has_structural_gap(self) -> bool:
        """True if any point has non-zero curl → E-I executor needed."""
        return bool(np.any(np.abs(self.curl) > 1e-6))


def _solve_poisson_2d(rhs: np.ndarray, dx: float, dy: float,
                       max_iter: int = 5000, tol: float = 1e-6) -> np.ndarray:
    """Solve ∇²u = rhs on a 2D grid with Dirichlet BC (u=0 at boundary)
    using Successive Over-Relaxation (SOR) — numpy only, no scipy dependency.

    For N×N with N≤100, this converges in <1000 iterations.
    """
    N, M = rhs.shape
    u = np.zeros((N, M))

    # SOR parameters
    # Optimal ω for 2D Dirichlet Poisson on N×N grid
    rho = np.cos(np.pi / N)
    omega = 2.0 / (1.0 + np.sqrt(1.0 - rho**2))  # optimal SOR

    dx2, dy2 = dx * dx, dy * dy
    # Coefficients for the 5-point stencil
    # u[i,j] = (dx²dy²·rhs[i,j] + dy²(u[i+1]+u[i-1]) + dx²(u[j+1]+u[j-1])) / (2dx²+2dy²)
    cx = 1.0 / dx2
    cy = 1.0 / dy2
    c0 = -2.0 * (cx + cy)

    for it in range(max_iter):
        u_old = u.copy()
        # Interior points (red-black ordering for efficiency)
        for i in range(1, N - 1):
            for j in range(1, M - 1):
                # Jacobi update: u_new = (rhs - (u_x+ + u_x-) - (u_y+ + u_y-)) / c0
                lap = cx * (u[i+1, j] + u[i-1, j]) + cy * (u[i, j+1] + u[i, j-1])
                u_new = (rhs[i, j] - lap) / c0
                # SOR relaxation
                u[i, j] = (1 - omega) * u[i, j] + omega * u_new

        # Convergence check
        diff = np.max(np.abs(u - u_old))
        if diff < tol:
            break

    return u


def helmholtz_decompose(field: VectorField2D) -> HelmholtzDecomposition:
    """Compute Helmholtz decomposition of Π = (u, v).

    Π = -∇φ + J∇ψ
    ∇·Π = -∇²φ  → solve ∇²φ = -∇·Π
    ∇×Π = ∇²ψ   → solve ∇²ψ = ∇×Π
    """
    u, v = field.u, field.v
    dx, dy = field.dx, field.dy

    # Compute divergence and curl via central differences
    # du/dx, dv/dy
    div = np.zeros_like(u)
    du_dx = np.zeros_like(u)
    dv_dy = np.zeros_like(v)
    dv_dx = np.zeros_like(v)
    du_dy = np.zeros_like(u)

    # Central differences for interior points
    if u.shape[0] > 2 and u.shape[1] > 2:
        du_dx[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2 * dx)
        du_dy[1:-1, :] = (u[2:, :] - u[:-2, :]) / (2 * dy)
        dv_dx[:, 1:-1] = (v[:, 2:] - v[:, :-2]) / (2 * dx)
        dv_dy[1:-1, :] = (v[2:, :] - v[:-2, :]) / (2 * dy)

        # One-sided at boundaries
        du_dx[:, 0] = (u[:, 1] - u[:, 0]) / dx
        du_dx[:, -1] = (u[:, -1] - u[:, -2]) / dx
        du_dy[0, :] = (u[1, :] - u[0, :]) / dy
        du_dy[-1, :] = (u[-1, :] - u[-2, :]) / dy
        dv_dx[:, 0] = (v[:, 1] - v[:, 0]) / dx
        dv_dx[:, -1] = (v[:, -1] - v[:, -2]) / dx
        dv_dy[0, :] = (v[1, :] - v[0, :]) / dy
        dv_dy[-1, :] = (v[-1, :] - v[-2, :]) / dy
    else:
        # Very small grid, use one-sided everywhere
        du_dx[:, :-1] = (u[:, 1:] - u[:, :-1]) / dx
        du_dy[:-1, :] = (u[1:, :] - u[:-1, :]) / dy
        dv_dx[:, :-1] = (v[:, 1:] - v[:, :-1]) / dx
        dv_dy[:-1, :] = (v[1:, :] - v[:-1, :]) / dy

    div = du_dx + dv_dy
    curl = dv_dx - du_dy  # scalar curl in 2D

    # Solve Poisson equations via FFT
    phi = _solve_poisson_2d(-div, dx, dy)
    psi = _solve_poisson_2d(curl, dx, dy)

    # Reconstruct components: Π_irrot = -∇φ, Π_solen = J∇ψ
    # Compute gradients of φ and ψ
    irrot_u = np.zeros_like(u)
    irrot_v = np.zeros_like(v)
    sol_u = np.zeros_like(u)
    sol_v = np.zeros_like(v)

    if phi.shape[0] > 2 and phi.shape[1] > 2:
        # dφ/dx
        irrot_u[:, 1:-1] = -(phi[:, 2:] - phi[:, :-2]) / (2 * dx)
        irrot_u[:, 0] = -(phi[:, 1] - phi[:, 0]) / dx
        irrot_u[:, -1] = -(phi[:, -1] - phi[:, -2]) / dx
        # dφ/dy
        irrot_v[1:-1, :] = -(phi[2:, :] - phi[:-2, :]) / (2 * dy)
        irrot_v[0, :] = -(phi[1, :] - phi[0, :]) / dy
        irrot_v[-1, :] = -(phi[-1, :] - phi[-2, :]) / dy

        # J∇ψ: (-dψ/dy, dψ/dx)
        sol_u[:, 1:-1] = -(psi[:, 2:] - psi[:, :-2]) / (2 * dx)  # J rotates: -dψ/dy
        # Wait, J∇ψ = J(∂ψ/∂x, ∂ψ/∂y) = (-∂ψ/∂y, ∂ψ/∂x)
        # Actually J = [[0, -1], [1, 0]], so J∇ψ = (-∂ψ/∂y, ∂ψ/∂x)
        # sol_u = -dψ/dy, sol_v = dψ/dx
        sol_u[1:-1, :] = -(psi[2:, :] - psi[:-2, :]) / (2 * dy)  # -dψ/dy
        sol_u[0, :] = -(psi[1, :] - psi[0, :]) / dy
        sol_u[-1, :] = -(psi[-1, :] - psi[-2, :]) / dy
        # dψ/dx
        sol_v[:, 1:-1] = (psi[:, 2:] - psi[:, :-2]) / (2 * dx)  # dψ/dx
        sol_v[:, 0] = (psi[:, 1] - psi[:, 0]) / dx
        sol_v[:, -1] = (psi[:, -1] - psi[:, -2]) / dx

    # Executor type classification at each point
    # E-I (structural): |curl| >> |div|
    # E-II/E-III (parametric): |div| dominates, curl ≈ 0
    abs_div = np.abs(div)
    abs_curl = np.abs(curl)
    total = abs_div + abs_curl + 1e-10

    structural_score = abs_curl / total

    # Type at each grid point:
    # 0: harmonic (both ≈ 0)
    # 1: E-I (curl dominant, structural)
    # 2: E-II (div dominant, scalar/scale dependent)
    # 3: E-III (div dominant but weaker, boundary condition)
    executor_type = np.zeros_like(div, dtype=int)
    EPS = 1e-8
    # E-I: curl > div and curl > eps
    mask_ei = (abs_curl > abs_div) & (abs_curl > EPS)
    executor_type[mask_ei] = 1
    # E-II: div > curl and div is strongly localized (large magnitude)
    mask_eii = (abs_div >= abs_curl) & (abs_div > 0.5)
    executor_type[mask_eii] = 2
    # E-III: div > curl but weaker, more distributed
    mask_eiii = (abs_div >= abs_curl) & (abs_div > EPS) & (abs_div <= 0.5)
    executor_type[mask_eiii] = 3

    return HelmholtzDecomposition(
        phi=phi, psi=psi,
        divergence=div, curl=curl,
        irrotational_u=irrot_u, irrotational_v=irrot_v,
        solenoidal_u=sol_u, solenoidal_v=sol_v,
        executor_type=executor_type,
        structural_score=structural_score,
    )


# ═══════════════════════════════════════════════════════════════
# A.5: Constraint Continuity Equation
# ═══════════════════════════════════════════════════════════════

@dataclass
class ContinuityAnalysis:
    """∂ρ/∂t + ∇·Π = 0 → ∇·Π ≠ 0 locates missing executors."""
    divergence: np.ndarray        # ∇·Π at each point
    divergence_peaks: list        # local maxima of |∇·Π|
    source_regions: np.ndarray    # boolean mask: ∇·Π > threshold (constraint sources)
    sink_regions: np.ndarray      # boolean mask: ∇·Π < -threshold (constraint sinks)
    missing_executor_candidates: list  # (x, y, score) for top divergence peaks


def _local_maxima(data: np.ndarray, footprint_size: int = 3) -> np.ndarray:
    """Find local maxima in 2D array (numpy only)."""
    from functools import reduce
    N, M = data.shape
    hs = footprint_size // 2
    result = np.ones_like(data, dtype=bool)
    for di in range(-hs, hs + 1):
        for dj in range(-hs, hs + 1):
            if di == 0 and dj == 0:
                continue
            shifted = np.roll(np.roll(data, di, axis=0), dj, axis=1)
            result &= (data >= shifted)
    # Boundary points excluded
    result[:hs, :] = False
    result[-hs:, :] = False
    result[:, :hs] = False
    result[:, -hs:] = False
    return result


def _label_regions(mask: np.ndarray) -> tuple:
    """Label connected components (4-connectivity) in a boolean mask.
    Returns (labeled_array, n_labels). Pure numpy + simple flood-fill.
    """
    N, M = mask.shape
    labeled = np.zeros((N, M), dtype=int)
    n_labels = 0

    for i in range(N):
        for j in range(M):
            if mask[i, j] and labeled[i, j] == 0:
                n_labels += 1
                # Flood fill
                stack = [(i, j)]
                while stack:
                    ci, cj = stack.pop()
                    if 0 <= ci < N and 0 <= cj < M and mask[ci, cj] and labeled[ci, cj] == 0:
                        labeled[ci, cj] = n_labels
                        stack.extend([(ci+1, cj), (ci-1, cj), (ci, cj+1), (ci, cj-1)])
    return labeled, n_labels


def _center_of_mass(region_mask: np.ndarray) -> tuple:
    """Compute center of mass of a boolean region."""
    total = np.sum(region_mask)
    if total == 0:
        return (0.0, 0.0)
    ys, xs = np.where(region_mask)
    return (float(np.mean(ys)), float(np.mean(xs)))


def analyze_continuity(div: np.ndarray, xs: np.ndarray, ys: np.ndarray,
                        n_peaks: int = 5) -> ContinuityAnalysis:
    """Find peaks in |∇·Π| — these locate missing executors."""
    abs_div = np.abs(div)
    N = abs_div.shape[0]

    # Find local maxima
    local_max_mask = _local_maxima(abs_div)
    # Only keep significant peaks (top 10%)
    significant = abs_div > np.percentile(abs_div, 90)
    peak_mask = local_max_mask & significant

    labeled, n_labels = _label_regions(peak_mask)

    peaks = []
    for li in range(1, n_labels + 1):
        region = labeled == li
        score = float(np.max(abs_div[region]))
        com = _center_of_mass(region)
        if com[0] > 0 and com[1] > 0:
            py, px = com
            peaks.append((float(xs[int(px)]), float(ys[int(py)]), score))

    peaks.sort(key=lambda x: x[2], reverse=True)
    peaks = peaks[:n_peaks]

    threshold = float(np.percentile(div, 75))
    sources = div > threshold
    sinks = div < -threshold

    return ContinuityAnalysis(
        divergence=div,
        divergence_peaks=peaks,
        source_regions=sources,
        sink_regions=sinks,
        missing_executor_candidates=peaks,
    )


# ═══════════════════════════════════════════════════════════════
# A.7: Constraint Decay Law
# ═══════════════════════════════════════════════════════════════

def constraint_decay(base_strength: float, n_layers: int, beta: float = 0.25) -> np.ndarray:
    """S_{n+1} = S_n · (1-β) — constraint strength decays across layers.

    Args:
        base_strength: S_0 (strength at source layer)
        n_layers: number of layers to propagate
        beta: decay rate per layer (default 0.25)

    Returns:
        Array of strengths at each layer [S_0, S_1, ..., S_n]
    """
    strengths = np.zeros(n_layers + 1)
    strengths[0] = base_strength
    for i in range(n_layers):
        strengths[i + 1] = strengths[i] * (1 - beta)
    return strengths


# ═══════════════════════════════════════════════════════════════
# Full CIS Analysis Pipeline
# ═══════════════════════════════════════════════════════════════

@dataclass
class CISReport:
    """Complete CIS analysis report for a protocol."""
    protocol_name: str
    n_constraints: int
    n_dark_zones: int
    dark_zone_centroids: list
    dark_zone_c_ratios: list

    # A.1
    vector_field: VectorField2D

    # A.2
    cancellation_ratio: np.ndarray  # grid

    # A.3
    riemannian: RiemannianAnalysis

    # A.4
    helmholtz: HelmholtzDecomposition

    # A.5
    continuity: ContinuityAnalysis

    # Summary statistics
    e1_fraction: float  # fraction of grid points classified as E-I
    e2_fraction: float
    e3_fraction: float
    structural_score: float  # global structural score (max)
    unconstrained_fraction: float  # fraction of grid where g^{-1} has zero eigenvalue

    # Top findings
    top_missing_executor_locations: list  # (x, y, type, score)


class CISAnalyzer:
    """Complete Constraint Invisibility Scanner analyzer."""

    def __init__(self, field, bounds=None, n_points=100):
        """
        Args:
            field: ConstraintField instance
            bounds: [(xmin, xmax), (ymin, ymax)]
            n_points: grid resolution per axis
        """
        self.field = field
        self.bounds = bounds or [(0, 1), (0, 1)]
        self.n_points = n_points
        self.report: Optional[CISReport] = None

    def full_analysis(self, protocol_name: str = "unknown") -> CISReport:
        """Run the complete A.1-A.7 analysis pipeline."""
        N = self.n_points
        xs = np.linspace(self.bounds[0][0], self.bounds[0][1], N)
        ys = np.linspace(self.bounds[1][0], self.bounds[1][1], N)
        dx = xs[1] - xs[0]
        dy = ys[1] - ys[0]

        # ── Compute all gradients on the grid ──
        K = len(self.field.rules)
        u = np.zeros((N, N))      # Π_x
        v = np.zeros((N, N))      # Π_y
        indiv_mag = np.zeros((N, N))  # Σ||∇σ_i||
        individual_grads = np.zeros((K, N, N, 2))  # each constraint's gradient

        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                p = np.array([x, y])
                for ki, rule in enumerate(self.field.rules):
                    g = rule.gradient(p)
                    individual_grads[ki, j, i] = g
                    u[j, i] += g[0]
                    v[j, i] += g[1]
                    indiv_mag[j, i] += float(np.linalg.norm(g))

        magnitude = np.sqrt(u**2 + v**2)

        # A.1: Vector field
        vf = VectorField2D(xs=xs, ys=ys, u=u, v=v, magnitude=magnitude)

        # A.2: Cancellation ratio
        cr = compute_cancellation_ratio(magnitude, indiv_mag)

        # A.3: Riemannian metric
        riemannian = compute_riemannian_metric(individual_grads)

        # A.4: Helmholtz decomposition
        helmholtz = helmholtz_decompose(vf)

        # A.5: Continuity analysis
        continuity = analyze_continuity(helmholtz.divergence, xs, ys)

        # ── Dark zone detection from cancellation ratio ──
        dark_mask = (cr < 0.15) & (indiv_mag > 0.25) & (magnitude > 1e-6)
        labeled, n_dz = _label_regions(dark_mask)

        dz_centroids, dz_c_ratios = [], []
        for i in range(1, n_dz + 1):
            region = labeled == i
            if np.sum(region) > 5:
                com = _center_of_mass(region)
                py, px = com
                dz_centroids.append((float(xs[int(px)]), float(ys[int(py)])))
                dz_c_ratios.append(float(np.mean(cr[region])))

        # ── Summary statistics ──
        e1_frac = float(np.mean(helmholtz.executor_type == 1))
        e2_frac = float(np.mean(helmholtz.executor_type == 2))
        e3_frac = float(np.mean(helmholtz.executor_type == 3))
        structural_score = float(np.max(helmholtz.structural_score))
        unconstr_frac = float(np.mean(riemannian.unconstrained_mask()))

        # Top missing executor locations
        candidates = []
        for x, y, score in continuity.missing_executor_candidates:
            # Find the executor type at or near this location
            ix = int(np.argmin(np.abs(xs - x)))
            iy = int(np.argmin(np.abs(ys - y)))
            etype = helmholtz.executor_type[iy, ix]
            type_names = {0: 'none', 1: 'E-I (structural)', 2: 'E-II (scalar/scale)', 3: 'E-III (boundary)'}
            candidates.append({
                'position': (x, y),
                'type': type_names.get(etype, f'type_{etype}'),
                'divergence_score': score,
                'structural_score': float(helmholtz.structural_score[iy, ix]),
            })

        self.report = CISReport(
            protocol_name=protocol_name,
            n_constraints=K,
            n_dark_zones=n_dz,
            dark_zone_centroids=dz_centroids,
            dark_zone_c_ratios=dz_c_ratios,
            vector_field=vf,
            cancellation_ratio=cr,
            riemannian=riemannian,
            helmholtz=helmholtz,
            continuity=continuity,
            e1_fraction=e1_frac,
            e2_fraction=e2_frac,
            e3_fraction=e3_frac,
            structural_score=structural_score,
            unconstrained_fraction=unconstr_frac,
            top_missing_executor_locations=candidates,
        )
        return self.report

    def print_summary(self):
        """Print a human-readable summary of the CIS analysis."""
        if self.report is None:
            print("Run full_analysis() first.")
            return
        r = self.report
        print(f"\n{'='*60}")
        print(f"CIS Analysis Report: {r.protocol_name}")
        print(f"{'='*60}")
        print(f"\nConstraints analyzed: {r.n_constraints}")
        print(f"Dark zones detected: {r.n_dark_zones}")
        for i, (cz, cr) in enumerate(zip(r.dark_zone_centroids, r.dark_zone_c_ratios)):
            print(f"  [{i+1}] centroid=({cz[0]:.4f}, {cz[1]:.4f})  c(p)={cr:.4f}")

        print(f"\n── Helmholtz Decomposition ──")
        print(f"  E-I (structural):      {r.e1_fraction:.1%}  ← requires new constraint type")
        print(f"  E-II (scalar/scale):   {r.e2_fraction:.1%}  ← can be absorbed into existing")
        print(f"  E-III (boundary):      {r.e3_fraction:.1%}  ← context-dependent")
        print(f"  Global structural score: {r.structural_score:.4f}")
        print(f"  Has structural gap:      {r.helmholtz.has_structural_gap}")

        print(f"\n── Riemannian Metric ──")
        print(f"  Unconstrained fraction: {r.unconstrained_fraction:.1%}")
        print(f"  Min eigenvalue range:   [{np.min(r.riemannian.eval_min):.4f}, {np.max(r.riemannian.eval_min):.4f}]")
        print(f"  Max condition number:   {np.max(r.riemannian.condition_number):.1f}")

        print(f"\n── Continuity Analysis ──")
        print(f"  Divergence range: [{np.min(r.helmholtz.divergence):.3f}, {np.max(r.helmholtz.divergence):.3f}]")
        print(f"  Top missing executor candidates:")
        for i, c in enumerate(r.top_missing_executor_locations):
            print(f"    [{i+1}] ({c['position'][0]:.3f}, {c['position'][1]:.3f}) "
                  f"type={c['type']} div={c['divergence_score']:.3f}")

        print(f"\n── A.7 Decay Law ──")
        strengths = constraint_decay(1.0, 4)
        print(f"  Layer 0→4: " + " → ".join(f"{s:.3f}" for s in strengths))
