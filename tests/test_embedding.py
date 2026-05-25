"""Embedding fixture is the foundation of everything else.  Spot-check it."""
from __future__ import annotations

import pandas as pd
import pytest

from edynamics.modelling_tools import Embedding, Lag


def test_embedding_compiles(small_embedding: Embedding):
    blk = small_embedding.block
    assert isinstance(blk, pd.DataFrame)
    assert len(blk) > 0
    # 0-lag on X, Y, Z -> 3 columns
    assert blk.shape[1] == 3


def test_embedding_dimension(small_embedding: Embedding):
    assert small_embedding.dimension == 3


def test_embedding_library_times_match_block(small_embedding: Embedding):
    blk = small_embedding.block
    # block rows correspond to library_times (modulo lag-warmup; for tau=0
    # observers it should match exactly)
    assert blk.index.equals(small_embedding.library_times)


def test_embedding_with_lag(lorenz_df: pd.DataFrame):
    """A delay-coordinate embedding (tau=-1, -2) on a single variable."""
    observers = [Lag("X", -i) for i in range(3)]
    lib = lorenz_df.index[100:200]
    emb = Embedding(
        data=lorenz_df, observers=observers,
        library_times=lib, compile_block=True,
    )
    assert emb.dimension == 3
    assert emb.block.shape[1] == 3
    # All three columns derive from X but at different lags
    assert len(emb.block) > 0


def test_set_library_changes_library_times(lorenz_df: pd.DataFrame):
    observers = [Lag("X", 0)]
    emb = Embedding(
        data=lorenz_df, observers=observers,
        library_times=lorenz_df.index[100:200], compile_block=True,
    )
    new_lib = lorenz_df.index[300:400]
    emb.set_library(new_lib)
    assert emb.library_times.equals(new_lib)


def test_set_observers_replaces(lorenz_df: pd.DataFrame):
    observers = [Lag("X", 0)]
    emb = Embedding(
        data=lorenz_df, observers=observers,
        library_times=lorenz_df.index[100:200], compile_block=True,
    )
    new_obs = [Lag("X", 0), Lag("Y", 0)]
    emb.set_observers(new_obs, compile_block=True)
    assert list(emb.observers) == new_obs
    # Block now has 2 columns
    assert emb.block.shape[1] == 2


def test_set_observers_rejects_unknown_variable(lorenz_df: pd.DataFrame):
    """An observer referencing a column not in data must raise."""
    observers = [Lag("X", 0)]
    emb = Embedding(
        data=lorenz_df, observers=observers,
        library_times=lorenz_df.index[100:200], compile_block=True,
    )
    bad = [Lag("NOT_A_COLUMN", 0)]
    with pytest.raises(AttributeError):
        emb.set_observers(bad)


def test_get_k_nearest_neighbours(small_embedding: Embedding):
    """KDTree-backed kNN over the embedding block."""
    import numpy as np
    pt = small_embedding.block.iloc[0].values  # a state in the library
    idxs = small_embedding.get_k_nearest_neighbours(point=pt, knn=5)
    assert len(idxs) == 5
    # The first idx should be the query point itself (distance 0)
    assert idxs[0] == 0
