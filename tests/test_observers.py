"""Concrete Observer subclasses: Lag, LagMovingAverage, ColumnObserver.

Lag is exercised indirectly through every Embedding fixture.  This
file covers the bits the embedding-level tests don't:

  - LagMovingAverage construction + __eq__ + __hash__ (verifies the
    fix in commit f96da84: __hash__ was missing 'return').
  - ColumnObserver constructor + observe() over a DataFrame.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from edynamics.modelling_tools.observers import (
    ColumnObserver, Lag, LagMovingAverage, Observer,
)


# ---------------------------------------------------------------------------
# Lag
# ---------------------------------------------------------------------------
def test_lag_equality_and_hash():
    a = Lag("X", -1)
    b = Lag("X", -1)
    c = Lag("Y", -1)
    d = Lag("X", -2)
    assert a == b
    assert a != c
    assert a != d
    # Equal observers must hash equal (set/dict semantics rely on this)
    assert hash(a) == hash(b)
    # And the observer must be hashable -- placing it in a set must succeed
    s = {a, b, c, d}
    assert len(s) == 3   # a == b, so 4 -> 3


# ---------------------------------------------------------------------------
# LagMovingAverage
# ---------------------------------------------------------------------------
def test_lag_moving_average_construction():
    obs = LagMovingAverage("X", q=3, tau=-1)
    assert obs.variable_name == "X"
    assert obs.q == 3
    assert obs.tau == -1


def test_lag_moving_average_equality_and_hash():
    """Regression: __hash__ used to compute hash but forget to return
    it (silently returned None -> instances unhashable).  Fixed in
    f96da84."""
    a = LagMovingAverage("X", q=3, tau=-1)
    b = LagMovingAverage("X", q=3, tau=-1)
    c = LagMovingAverage("X", q=3, tau=-2)
    assert a == b
    assert a != c
    # The bug: __hash__ returned None.  Now it must be int.
    assert isinstance(hash(a), int)
    assert hash(a) == hash(b)
    s = {a, b, c}
    assert len(s) == 2


# ---------------------------------------------------------------------------
# ColumnObserver
# ---------------------------------------------------------------------------
def test_column_observer_construction():
    obs = ColumnObserver(observation_name="X_proxy", variable_name="X")
    assert obs.variable_name == "X"
    assert obs.observation_name == "X_proxy"


def test_column_observer_observe(lorenz_df: pd.DataFrame):
    """observe(data, times) returns the named column at the requested times."""
    obs = ColumnObserver(observation_name="X_proxy", variable_name="X")
    times = lorenz_df.index[100:105]
    out = obs.observe(data=lorenz_df, times=times)
    assert isinstance(out, pd.Series)
    assert len(out) == 5
    # Same values as the original column at those times
    assert np.allclose(out.values, lorenz_df["X"].loc[times].values)


def test_column_observer_equality_and_hash():
    a = ColumnObserver(observation_name="alpha", variable_name="X")
    b = ColumnObserver(observation_name="beta", variable_name="X")  # different obs_name, same var
    c = ColumnObserver(observation_name="gamma", variable_name="Y")
    assert a == b    # equality is by variable_name only
    assert a != c
    assert hash(a) == hash(b)
    assert isinstance(hash(a), int)
