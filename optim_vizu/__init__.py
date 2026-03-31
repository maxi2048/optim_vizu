"""
optim_vizu – Visualisierung von Optimierungsverfahren

Hauptfunktionen:
- optimize: Führt eine Optimierung durch
- compare:  Vergleicht mehrere Optimierungsverfahren
"""

from .optiviz import optimize, compare, OptimizeResult

__all__ = [
    "optimize",
    "compare",
    "OptimizeResult",
]

__version__ = "1.0.0"