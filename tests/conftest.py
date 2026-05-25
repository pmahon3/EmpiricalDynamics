"""Shared pytest fixtures.

A small Lorenz-derived embedding fixture is the most useful piece of
shared infrastructure; every projector/estimator test starts from one.
"""
from __future__ import annotations

import pandas as pd
import pytest

from edynamics.data_sets.lorenz import lorenz_data
from edynamics.modelling_tools.embeddings import Embedding
from edynamics.modelling_tools.observers import Lag


@pytest.fixture(scope="session")
def lorenz_df() -> pd.DataFrame:
    """Full Lorenz trajectory as a DataFrame indexed by datetime."""
    return lorenz_data()


@pytest.fixture(scope="session")
def small_embedding(lorenz_df: pd.DataFrame) -> Embedding:
    """A compiled 3-coordinate Lorenz embedding suitable for fast tests.

    Uses 0-lag observers on (X, Y, Z) and a library of ~900 anchors.
    """
    observers = [Lag("X", 0), Lag("Y", 0), Lag("Z", 0)]
    library_times = lorenz_df.index[100:1000]
    emb = Embedding(
        data=lorenz_df,
        observers=observers,
        library_times=library_times,
        compile_block=True,
    )
    return emb
