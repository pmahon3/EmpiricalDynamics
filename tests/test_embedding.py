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
