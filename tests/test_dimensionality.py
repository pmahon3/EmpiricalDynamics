"""dimensionality() sweep test.

Production use (MITACS processing/dimensions/process.py) selects an
embedding dimension per day-type by argmax of the rho(d) curve this
function produces.  Until this commit it was untested (28% coverage).
"""
from __future__ import annotations

import pandas as pd
import pytest

from edynamics.modelling_tools.embeddings import Embedding
from edynamics.modelling_tools.estimators import dimensionality


def test_dimensionality_returns_curve(small_embedding: Embedding):
    """Sweep max_dimensions=5 on the Lorenz embedding.  Returns a
    DataFrame with one rho per dimension, sorted by dimension."""
    times = small_embedding.library_times[-30:]
    df = dimensionality(
        embedding=small_embedding,
        target="X",
        times=times,
        max_dimensions=5,
        steps=1,
        step_size=1,
        verbose=True,        # disable tqdm output during tests
    )
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (5, 1)
    assert list(df.columns) == ["rho"]
    # Index = embedding dimensions 1..5
    assert list(df.index) == [1, 2, 3, 4, 5]
    # rho should be finite for every dimension
    assert df["rho"].notna().all()
    # And in [-1, 1] (it's a Pearson correlation)
    assert (df["rho"] >= -1.0).all()
    assert (df["rho"] <= 1.0).all()


def test_dimensionality_restores_observers(small_embedding: Embedding):
    """The function mutates embedding.observers during the sweep; it
    must restore the original observers on the way out."""
    original = list(small_embedding.observers)
    times = small_embedding.library_times[-20:]
    dimensionality(
        embedding=small_embedding,
        target="X", times=times,
        max_dimensions=3, steps=1, step_size=1,
        verbose=True,
    )
    assert list(small_embedding.observers) == original


def test_dimensionality_target_must_be_in_data(small_embedding: Embedding):
    """A non-existent target should produce a clear failure (currently
    KeyError on the embedding's data lookup)."""
    times = small_embedding.library_times[-10:]
    with pytest.raises((KeyError, AttributeError)):
        dimensionality(
            embedding=small_embedding,
            target="NOT_A_COLUMN", times=times,
            max_dimensions=2, steps=1, step_size=1,
            verbose=True,
        )
