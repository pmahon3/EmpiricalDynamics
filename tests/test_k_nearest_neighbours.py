"""KNearestNeighbours.project() smoke tests.

Until this commit the only KNN exercise was the import line in
test_smoke_imports; the projector itself was entirely untested
(29% coverage).
"""
from __future__ import annotations

import pandas as pd
import pytest

from edynamics.modelling_tools.embeddings import Embedding
from edynamics.modelling_tools.projectors import KNearestNeighbours


def test_knn_project_single_step(small_embedding: Embedding):
    """KNN one-step projection on Lorenz: returns predictions over
    every query in a (reference_time, prediction_time) MultiIndex."""
    projector = KNearestNeighbours(k=5)
    qry = small_embedding.get_points(small_embedding.library_times[-5:])
    preds = projector.project(
        embedding=small_embedding, points=qry,
        steps=1, step_size=1,
    )
    assert isinstance(preds, pd.DataFrame)
    # MultiIndex (origin, valid)
    assert preds.index.nlevels == 2
    assert preds.shape == (5 * 1, 3)
    # Predictions are finite
    assert preds.notna().all().all()


def test_knn_project_multistep(small_embedding: Embedding):
    """h-step projection yields (n_pts * h) rows."""
    projector = KNearestNeighbours(k=5)
    qry = small_embedding.get_points(small_embedding.library_times[-10:])
    H = 5
    preds = projector.project(
        embedding=small_embedding, points=qry,
        steps=H, step_size=1,
    )
    assert preds.shape == (10 * H, 3)
    assert preds.notna().all().all()


def test_knn_k_none_uses_dimension_plus_one(small_embedding: Embedding):
    """k=None should be adaptive to embedding.dimension+1 and reset
    after the call so the projector instance stays reusable."""
    projector = KNearestNeighbours(k=None)
    qry = small_embedding.get_points(small_embedding.library_times[-3:])
    preds = projector.project(
        embedding=small_embedding, points=qry,
        steps=1, step_size=1,
    )
    assert preds.shape == (3, 3)
    # KNN's project() sets k=dim+1 internally and then reverts to None
    assert projector.k is None
