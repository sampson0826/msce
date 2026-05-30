"""MSCE — Multi-Source Constraint Engine.

Detect hidden conflicts between scientific theories and observational constraints.
"""

__version__ = "0.1.0"
__author__ = "Deng Xinhang & MSCE Collaboration"

from .core import analyze, check, load_hubble_data
from .visualize import heatmap, confidence_bars

__all__ = ["analyze", "check", "load_hubble_data", "heatmap", "confidence_bars"]
