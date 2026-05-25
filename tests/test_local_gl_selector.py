"""LocalGLSelector picks per-anchor (theta*, sigma*) from grids.

Smoke test: it fits without error, returns torch tensors of the right
shape, and populates the WLS object's anchor_times + best_*_vals.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from edynamics.modelling_tools.embeddings import Embedding
from edynamics.modelling_tools.estimators import LocalGLSelector
from edynamics.modelling_tools.kernels import Gaussian
from edynamics.modelling_tools.projectors import WeightedLeastSquares


@pytest.fixture
def coarse_grid():
    """Small grids: fits in seconds rather than minutes."""
    return np.logspace(-0.5, 1.0, 6)   # 6 values from ~0.32 to 10


@pytest.fixture
def wls(small_embedding: Embedding) -> WeightedLeastSquares:
    """A fresh WLS with Gaussian drift + residual kernels."""
    return WeightedLeastSquares(
        kernel=Gaussian(theta=1.0, dim=small_embedding.dimension),
        residual_kernel=Gaussian(theta=1.0, dim=small_embedding.dimension),
    )


def test_local_gl_selector_fits(small_embedding: Embedding,
                                 wls: WeightedLeastSquares,
                                 coarse_grid):
    """Pick a handful of anchors (cheap) and run a fit."""
    anchors = small_embedding.library_times[::200]   # ~5 anchors
    assert len(anchors) >= 2

    selector = LocalGLSelector(
        theta_grid=coarse_grid, sigma_grid=coarse_grid,
        lwls=wls, C=2.0,
    )
    theta_star, sigma_star = selector.fit(small_embedding, anchors)

    # Output shape
    assert isinstance(theta_star, torch.Tensor)
    assert isinstance(sigma_star, torch.Tensor)
    assert theta_star.shape == (len(anchors),)
    assert sigma_star.shape == (len(anchors),)

    # No NaNs
    assert not torch.isnan(theta_star).any()
    assert not torch.isnan(sigma_star).any()

    # Selected values lie in the grid range
    assert (theta_star >= coarse_grid.min()).all()
    assert (theta_star <= coarse_grid.max()).all()
    assert (sigma_star >= coarse_grid.min()).all()
    assert (sigma_star <= coarse_grid.max()).all()


def test_local_gl_selector_populates_wls(small_embedding: Embedding,
                                          wls: WeightedLeastSquares,
                                          coarse_grid):
    """After fit, the WLS object has anchor_times set."""
    anchors = small_embedding.library_times[::200]
    selector = LocalGLSelector(
        theta_grid=coarse_grid, sigma_grid=coarse_grid,
        lwls=wls, C=2.0,
    )
    selector.fit(small_embedding, anchors)

    assert wls.anchor_times is not None
    assert len(wls.anchor_times) == len(anchors)
