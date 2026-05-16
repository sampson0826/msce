"""
Constraint DSL Compiler — v0

Compiles declarative YAML protocol specs into executable ConstraintField objects.

Supported constraint functions:
  gaussian:  exp(-((x-cx)/wx)^2 - ((y-cy)/wy)^2)  [scale, negative]
  sigmoid:   1/(1+exp(-(axis_val-center)/width))     [scale]
  product:   multiplies two sub-constraint functions

Usage:
  from constraint_residual.dsl.compiler import load_protocol, scan_protocol
  field, spec = load_protocol("dsl/protocols/the_dao.yaml")
  result = scan_protocol(field, spec)
"""

import numpy as np
import yaml
import os
from dataclasses import dataclass, field
from typing import Optional

from constraint_residual.core import Rule, ConstraintField
from constraint_residual.dark_zone_detector import DarkZoneDetector

# ═══════════════════════════════════════════════════════════════
# Function builders
# ═══════════════════════════════════════════════════════════════

def _build_gaussian(center, width, scale=1.0):
    """Build a Gaussian constraint function."""
    cx, cy = center[0], center[1]
    wx, wy = width[0], width[1]
    def fn(p):
        x, y = float(p[0]), float(p[1])
        return scale * np.exp(-((x - cx) / wx)**2 - ((y - cy) / wy)**2)
    return fn


def _build_sigmoid(axis, center, width, scale=1.0):
    """Build a sigmoid constraint function along one axis."""
    ax_idx = 0 if axis == 'x' else 1
    def fn(p):
        val = float(p[ax_idx])
        return scale / (1.0 + np.exp(-(val - center) / width))
    return fn


def _build_linear(axis, center, scale=1.0):
    """Build a linear ramp constraint — constant gradient, no saturation.

    Key property: ∇σ is constant everywhere, making it ideal for
    breaking gradient cancellations that other constraints (with
    position-varying gradients) cannot fully eliminate.
    """
    ax_idx = 0 if axis in ('x', 'health_factor') else 1
    def fn(p):
        return scale * (float(p[ax_idx]) - center)
    return fn


def _build_product(factors_spec):
    """Build a product of multiple sub-constraint functions."""
    sub_fns = [_build_constraint_fn(spec) for spec in factors_spec]
    def fn(p):
        result = 1.0
        for f in sub_fns:
            result *= f(p)
        return result
    return fn


def _build_constraint_fn(spec):
    """Build a single constraint function from its spec dict."""
    fn_type = spec.get('fn', 'gaussian')
    scale = spec.get('scale', 1.0)

    if fn_type == 'gaussian':
        return _build_gaussian(spec['center'], spec['width'], scale)
    elif fn_type == 'sigmoid':
        return _build_sigmoid(spec['axis'], spec['center'], spec['width'], scale)
    elif fn_type == 'linear':
        return _build_linear(spec.get('axis', 'y'), spec['center'], scale)
    elif fn_type == 'product':
        return _build_product(spec['factors'])
    else:
        raise ValueError(f"Unknown constraint fn type: {fn_type}")


# ═══════════════════════════════════════════════════════════════
# Compiler
# ═══════════════════════════════════════════════════════════════

def layer_to_int(layer_str):
    """Convert layer string to integer. L-1=-1, L0=0, L1=1, ..."""
    return int(layer_str[1:]) if layer_str.startswith('L') else int(layer_str)


def compile_field(spec):
    """Compile a protocol spec dict into a ConstraintField."""
    rules = []
    for cs in spec.get('constraints', []):
        fn = _build_constraint_fn(cs)
        rule = Rule(
            name=cs['name'],
            layer=layer_to_int(cs.get('layer', 'L1')),
            domain=cs.get('domain', ''),
            constraint_fn=fn,
            certainty=cs.get('certainty', 1.0),
        )
        rules.append(rule)
    return ConstraintField(rules=rules)


def load_protocol(yaml_path):
    """Load a YAML protocol spec and compile into a ConstraintField.

    Returns: (ConstraintField, spec_dict)
    """
    with open(yaml_path) as f:
        spec = yaml.safe_load(f)
    field = compile_field(spec)
    return field, spec


# ═══════════════════════════════════════════════════════════════
# Scanning
# ═══════════════════════════════════════════════════════════════

@dataclass
class ScanResult:
    protocol: str
    dark_zone_type: str
    signature: str
    n_dark_zones: int
    dark_zone_centroids: list
    dark_zone_c_ratios: list
    dark_zone_topologies: list
    point_metrics: list  # list of (label, position, c_ratio, combined, total_indiv)


def scan_protocol(field, spec, bounds=None, n_points=80):
    """Run dark zone detection on a compiled protocol.

    Returns ScanResult with detection outcomes.
    """
    if bounds is None:
        bounds = [(0, 1), (0, 1)]

    scan_cfg = spec.get('scan', {})
    detector = DarkZoneDetector(
        cancellation_eps=scan_cfg.get('cancellation_eps', 0.15),
        individual_min=scan_cfg.get('individual_min', 0.2),
    )
    dark_clusters = detector.scan(field, bounds, n_points=n_points)

    centroids = []
    c_ratios = []
    topologies = []
    for dc in dark_clusters:
        centroids.append(dc.centroid.tolist() if dc.centroid is not None else None)
        c_ratios.append(dc.mean_cancellation_ratio)
        topologies.append(dc.balance_topology)

    return ScanResult(
        protocol=spec.get('protocol', 'unknown'),
        dark_zone_type=spec.get('dark_zone_type', 'unknown'),
        signature=spec.get('cancellation_signature', ''),
        n_dark_zones=len(dark_clusters),
        dark_zone_centroids=centroids,
        dark_zone_c_ratios=c_ratios,
        dark_zone_topologies=topologies,
        point_metrics=[],
    )


def metrics_at_point(field, p, label):
    """Compute constraint metrics at a specific state-space point."""
    grad = field.constraint_gradient(p)
    combined = float(np.linalg.norm(grad))
    indiv_grad = {r.name: float(np.linalg.norm(r.gradient(p))) for r in field.rules}
    indiv_val = {r.name: float(r.constraint_fn(p)) for r in field.rules}
    total_indiv = sum(indiv_grad.values())
    total_sigma = sum(indiv_val.values())
    cr = combined / total_indiv if total_indiv > 1e-10 else 1.0
    return {
        'label': label, 'position': p.tolist(),
        'combined': combined, 'total_indiv': total_indiv,
        'total_sigma': total_sigma, 'c_ratio': cr,
        'individual_grad': indiv_grad, 'individual_val': indiv_val,
    }
