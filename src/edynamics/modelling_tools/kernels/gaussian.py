# gaussian.py
"""
Gaussian (squared–exponential) kernel.

w(d) = exp(-½ (d / θ)²)

• θ > 0 is the bandwidth / length-scale.
• Works with any NumPy array shape and preserves dtype casting.
"""

from __future__ import annotations
from typing import Union

import numpy as np

from .kernel import Kernel


class Gaussian(Kernel):
    def __init__(self, theta: float, dim: int):
        super().__init__(theta)
        self.dim = dim

    def weigh(self, distance_matrix):
        d = np.asarray(distance_matrix, dtype=float)
        # normalized Gaussian kernel in R^dim
        norm = (2*np.pi)**(-self.dim/2) * (1/self.theta**self.dim)
        return norm * np.exp(-0.5 * (d/self.theta)**2)

    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # optional, for symmetry with base class
        return f"GaussianKernel(theta={self.theta})"

    __str__ = __repr__
