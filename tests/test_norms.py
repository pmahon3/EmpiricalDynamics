"""Minkowski norm direct test (it's only exercised indirectly elsewhere)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from edynamics.modelling_tools.embeddings import Embedding
from edynamics.modelling_tools.norms import Minkowski


def test_minkowski_p2_euclidean(small_embedding: Embedding):
    """Default Minkowski(p=2) is Euclidean.  distance_matrix returns
    (n_lib, n_query) of float distances."""
    times = small_embedding.library_times[:5]
    points = small_embedding.get_points(times).values   # (5, d)
    n = Minkowski(p=2)
    D = n.distance_matrix(
        embedding=small_embedding, points=points, times=times,
    )
    assert D.shape == (5, 5)
    # Diagonal is zero (same library point as query)
    assert np.allclose(np.diag(D), 0.0)
    assert (D >= 0).all()


def test_minkowski_p1_l1(small_embedding: Embedding):
    """Minkowski(p=1) is the L1 / city-block distance."""
    times = small_embedding.library_times[:3]
    points = small_embedding.get_points(times).values
    D1 = Minkowski(p=1).distance_matrix(
        embedding=small_embedding, points=points, times=times,
    )
    D2 = Minkowski(p=2).distance_matrix(
        embedding=small_embedding, points=points, times=times,
    )
    # L1 dominates L2 (for d > 1, sum |xi| >= sqrt(sum xi^2))
    assert (D1 >= D2 - 1e-10).all()


def test_minkowski_equality():
    assert Minkowski(p=2) == Minkowski(p=2)
    assert Minkowski(p=2) != Minkowski(p=1)
    assert Minkowski(p=2) != "not-a-norm"
