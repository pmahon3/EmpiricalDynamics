"""WLS end-to-end: requires a LocalGLSelector.fit() before .project()."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from edynamics.modelling_tools.embeddings import Embedding
from edynamics.modelling_tools.estimators import LocalGLSelector
from edynamics.modelling_tools.kernels import Gaussian
from edynamics.modelling_tools.projectors import WeightedLeastSquares


@pytest.fixture
def fitted_wls(small_embedding: Embedding) -> WeightedLeastSquares:
    """A WLS whose anchor_times + best_*_vals have been populated."""
    wls = WeightedLeastSquares(
        kernel=Gaussian(theta=1.0, dim=small_embedding.dimension),
        residual_kernel=Gaussian(theta=1.0, dim=small_embedding.dimension),
    )
    anchors = small_embedding.library_times[::200]
    grid = np.logspace(-0.5, 1.0, 6)
    LocalGLSelector(theta_grid=grid, sigma_grid=grid,
                    lwls=wls, C=2.0).fit(small_embedding, anchors)
    return wls


def test_project_raises_before_fit(small_embedding: Embedding):
    """Calling .project() without a prior LocalGLSelector.fit() should
    raise a RuntimeError -- the API contract enforces it."""
    wls = WeightedLeastSquares(
        kernel=Gaussian(theta=1.0, dim=small_embedding.dimension),
        residual_kernel=Gaussian(theta=1.0, dim=small_embedding.dimension),
    )
    qry = small_embedding.get_points(small_embedding.library_times[-5:])
    with pytest.raises(RuntimeError):
        wls.project(embedding=small_embedding, points=qry,
                    steps=1, step_size=1)


def test_project_returns_rose_result(small_embedding: Embedding,
                                      fitted_wls: WeightedLeastSquares):
    """Single-step projection returns a RoseResult with predictions."""
    qry = small_embedding.get_points(small_embedding.library_times[-5:])
    result = fitted_wls.project(
        embedding=small_embedding, points=qry,
        steps=1, step_size=1,
    )
    # RoseResult has .predictions, .coefficients, .covariances
    assert hasattr(result, "predictions")
    assert isinstance(result.predictions, pd.DataFrame)
    # n_points * steps rows; d=3 columns
    assert result.predictions.shape == (5 * 1, 3)


def test_project_multistep_shape(small_embedding: Embedding,
                                  fitted_wls: WeightedLeastSquares):
    """Multi-step projection returns (n_points * steps) rows."""
    qry = small_embedding.get_points(small_embedding.library_times[-10:])
    H = 5
    result = fitted_wls.project(
        embedding=small_embedding, points=qry,
        steps=H, step_size=1,
    )
    assert result.predictions.shape == (10 * H, 3)


def test_rose_result_evaluate(small_embedding: Embedding,
                               fitted_wls: WeightedLeastSquares):
    """RoseResult.evaluate() computes MSE/RMSE/MAE/Skill."""
    # Need predictions whose 'valid' times exist in the embedding
    qry = small_embedding.get_points(small_embedding.library_times[-20:-5])
    result = fitted_wls.project(
        embedding=small_embedding, points=qry,
        steps=1, step_size=1,
    )
    metrics = result.evaluate(small_embedding)
    assert isinstance(metrics, pd.DataFrame)
    # Should produce some columns named MSE, RMSE, MAE, Skill (or similar)
    cols = {c.lower() for c in metrics.columns}
    assert any(k in cols for k in {"mse", "rmse", "mae", "skill"})
