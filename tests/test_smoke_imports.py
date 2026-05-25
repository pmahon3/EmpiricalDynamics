"""Smoke test: the public API surface imports cleanly.

If this file fails, the package isn't installed correctly.  Run first
when triaging.
"""


def test_top_level_imports():
    from edynamics.modelling_tools import Embedding, Lag           # noqa: F401


def test_observers():
    from edynamics.modelling_tools.observers import (              # noqa: F401
        Observer, Lag, LagMovingAverage, ColumnObserver,
    )


def test_kernels():
    from edynamics.modelling_tools.kernels import (                # noqa: F401
        Kernel, Gaussian, Exponential, Epanechnikov, Tricubic, constant,
    )


def test_norms():
    from edynamics.modelling_tools.norms import Norm, Minkowski    # noqa: F401


def test_projectors():
    from edynamics.modelling_tools.projectors import (             # noqa: F401
        Projector, WeightedLeastSquares, KNearestNeighbours,
    )


def test_estimators():
    from edynamics.modelling_tools.estimators import (             # noqa: F401
        LocalGLSelector, dimensionality,
    )


def test_data_sets():
    from edynamics.data_sets import lorenz_data                    # noqa: F401
    from edynamics import lorenz_data as top                       # noqa: F401
