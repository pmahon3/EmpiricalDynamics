"""Each kernel implements `weigh(distance_matrix)` and produces a
non-negative array of the same shape."""
from __future__ import annotations

import numpy as np
import pytest

from edynamics.modelling_tools.kernels import (
    Gaussian, Exponential, Epanechnikov, Tricubic,
)


@pytest.fixture
def distances():
    # (3, 5) distance matrix with one identical-point row (zeros)
    return np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.5, 1.0, 1.5, 2.0, 2.5],
        [1.0, 2.0, 3.0, 4.0, 5.0],
    ])


def test_gaussian_shape_and_signs(distances):
    k = Gaussian(theta=1.0, dim=3)
    w = k.weigh(distances)
    assert w.shape == distances.shape
    assert np.all(w >= 0)
    # Weight at d=0 should be the largest at the kernel's centre
    assert w[0, 0] >= w[1, 0]


def test_gaussian_monotone_in_distance():
    k = Gaussian(theta=1.0, dim=3)
    # Within a fixed row, larger d -> smaller weight
    w = k.weigh(np.array([[0.5, 1.0, 1.5, 2.0]]))
    assert np.all(np.diff(w[0]) < 0)


def test_exponential_shape(distances):
    k = Exponential(theta=1.0)
    w = k.weigh(distances)
    assert w.shape == distances.shape
    assert np.all(w >= 0)
    # The Exponential normalises by row-0's average distance.  The
    # row-0 row has avg=0; the kernel handles this by returning weight
    # 1.0 (np.inf in the denominator).
    assert np.allclose(w[0], 1.0)


def test_epanechnikov_compactly_supported():
    k = Epanechnikov(theta=1.0)
    # d=1 should be at the kernel's edge; d>theta should be zero
    w = k.weigh(np.array([[0.0, 0.5, 0.99, 1.5, 2.0]]))
    assert w[0, 0] > 0
    assert w[0, -2] == 0
    assert w[0, -1] == 0


def test_tricubic_compactly_supported():
    k = Tricubic(theta=1.0)
    w = k.weigh(np.array([[0.0, 0.5, 0.9, 1.0, 1.5]]))
    assert w[0, 0] > 0
    # Tricubic is zero outside |d/theta| <= 1
    assert w[0, -1] == 0
