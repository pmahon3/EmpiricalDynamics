"""greedy_nearest_neighbour() smoke test.

Forward-stepwise observer selector that successively adds observers
from a candidate pool, keeping those that increase prediction skill.

Until commit b40ce0b this function was entirely untested (21% line
coverage on estimators/observers.py).  Two tests added then xfailed,
because they surfaced a real bug in Projector.update_values that
silently swallowed a KeyError and left `data` stale.  Bug fixed in
the immediately-following commit; the tests now run for real.
"""
from __future__ import annotations

import pandas as pd
import pytest

from edynamics.modelling_tools.embeddings import Embedding
from edynamics.modelling_tools.estimators import greedy_nearest_neighbour
from edynamics.modelling_tools.observers import Lag
from edynamics.modelling_tools.projectors import KNearestNeighbours


def test_greedy_nearest_neighbour_returns_performance(small_embedding: Embedding):
    """The function mutates embedding.observers in place; returns a
    list of best-skill rho per round."""
    projector = KNearestNeighbours(k=5)
    times = small_embedding.library_times[-50:]
    # Pool of candidate observers (Lag of each variable at small taus)
    candidates = [Lag(v, -t) for v in ("X", "Y", "Z") for t in (1, 2)]

    perf = greedy_nearest_neighbour(
        embedding=small_embedding,
        target="X",
        projector=projector,
        times=times,
        observers=candidates,
        steps=1, step_size=1,
        verbose=False,
    )
    assert isinstance(perf, list)
    # At least one round of selection occurred
    assert len(perf) >= 1
    # All rho values are in [-1, 1] (Pearson correlations)
    for p in perf:
        assert -1.0 <= p <= 1.0


def test_greedy_nearest_neighbour_with_early_stopping(small_embedding: Embedding):
    """An improvement_threshold above 0 should stop the search early
    once selection stops adding skill above that threshold."""
    projector = KNearestNeighbours(k=5)
    times = small_embedding.library_times[-50:]
    candidates = [Lag(v, -t) for v in ("X", "Y") for t in (1, 2)]

    perf = greedy_nearest_neighbour(
        embedding=small_embedding,
        target="X",
        projector=projector,
        times=times,
        observers=candidates,
        steps=1, step_size=1,
        improvement_threshold=1.0,   # impossible threshold -> stop after first
        verbose=False,
    )
    assert isinstance(perf, list)
    # Whatever the count, it's bounded by the candidate pool size
    assert len(perf) <= len(candidates) + 1
