# edynamics

Empirical dynamic modelling toolkit: delay-embedding state-space
reconstruction with locally-weighted projection methods, organised as
a small set of composable building blocks (embeddings, observers,
kernels, norms, projectors, estimators).

## Install

From PyPI:

```bash
pip install edynamics
```

From a checkout, into a Python 3.10+ environment:

```bash
pip install -e .
```

Dependencies (`numpy`, `pandas`, `scipy`, `tqdm`, `ray`, `torch`)
are declared in `pyproject.toml` and installed automatically.

If a build environment can't reach PyPI (e.g. air-gapped or
Compute Canada CVMFS with pre-installed wheels), preinstall the
runtime deps and add `--no-build-isolation`:

```bash
pip install numpy pandas scipy tqdm ray torch wheel setuptools
pip install -e . --no-build-isolation
```

### Note on torch

`torch` is a heavy dependency (~1 GB). It is required because two
modules (`projectors/weighted_least_squares.py`,
`estimators/local_gl_selector.py`) use it for vectorised batched
operations. If you only need `Embedding`, `Lag`, `Minkowski`, or the
non-WLS kernels, the install will still pull `torch` because
declaring it as runtime-optional would mask the import failure on
modules that depend on it.

## Upgrading from 0.3.x → 0.4.0

`0.4.0` introduces breaking changes to the
`WeightedLeastSquares` projector. See [CHANGELOG.md](CHANGELOG.md)
for the full migration. The two largest:

- `WeightedLeastSquares` now requires you to first run
  `LocalGLSelector.fit()` on a library to populate per-anchor
  `(θ*, σ*)` before calling `.project()`. The old `global_theta=...`
  constructor argument is gone.
- The result type returned by `.project()` was renamed `WLSResult`
  → `RoseResult`. The DataFrame access pattern (`result.predictions`)
  is unchanged; the new `.evaluate(embedding)` method computes
  per-lead error and persistence-skill metrics.

Minor PEP 8 fix: `kernels.constant` (the class) renamed to
`kernels.Constant`.

## What's in it

- `edynamics.modelling_tools.embeddings.Embedding` — the delay
  embedding container; binds raw data to a set of `Observer`s
  (e.g. `Lag`) and exposes a `block` of state-vector rows indexed
  by time.
- `edynamics.modelling_tools.observers` — `Observer` (ABC), `Lag`,
  `LagMovingAverage`, `ColumnObserver`.
- `edynamics.modelling_tools.kernels` — `Kernel` (ABC), `Constant`,
  `Gaussian`, `Exponential`, `Epanechnikov`, `Tricubic`.
- `edynamics.modelling_tools.norms` — `Norm` (ABC), `Minkowski`.
- `edynamics.modelling_tools.projectors.WeightedLeastSquares` —
  locally-weighted least-squares projection with separate drift
  (`theta`) and residual (`sigma`) bandwidths, selected per anchor
  by `LocalGLSelector`.
- `edynamics.modelling_tools.projectors.KNearestNeighbours` — simplex
  projection.
- `edynamics.modelling_tools.estimators.LocalGLSelector` — per-anchor
  joint `(theta, sigma)` grid search via the
  Goldenshluger-Lepski criterion.
- `edynamics.modelling_tools.estimators.dimensionality` — embedding-
  dimension prediction-skill sweep.
- `edynamics.data_sets.lorenz_data` — Lorenz attractor trajectory
  (for testing and examples).

## Minimal example

```python
import pandas as pd
from edynamics.modelling_tools import Embedding, Lag
from edynamics.modelling_tools.projectors import WeightedLeastSquares
from edynamics.modelling_tools.kernels import Gaussian
from edynamics.modelling_tools.estimators import LocalGLSelector

# 1. Build an embedding from a 1-D time series
data = pd.DataFrame(...)                # DatetimeIndex x one column
lags = [Lag(variable_name="x", tau=-i) for i in range(3)]
emb  = Embedding(data=data, observers=lags, library_times=data.index[3:-1])
emb.compile()

# 2. Set up the projector with drift + residual kernels
wls = WeightedLeastSquares(
    kernel=Gaussian(theta=1.0, dim=emb.dimension),
    residual_kernel=Gaussian(theta=1.0, dim=emb.dimension),
)

# 3. Select per-anchor (theta*, sigma*)
import numpy as np
sel = LocalGLSelector(
    theta_grid=np.logspace(-1, 1.5, 30),
    sigma_grid=np.logspace(-1, 1.5, 30),
    lwls=wls, C=2.0,
)
sel.fit(emb, library_times=emb.library_times[::10])

# 4. Project / forecast
qry = emb.get_points(emb.library_times[-5:])
result = wls.project(embedding=emb, points=qry, steps=1, step_size=1)
print(result.predictions)
print(result.evaluate(emb))             # MSE / RMSE / MAE / Skill per lead
```

See `examples/wls_lorenz_demo.py` for the full end-to-end Lorenz
demo with plotting.

## Development

Tests:

```bash
pip install pytest
pytest tests/
```

The suite covers 87 % of source lines; new contributions should
include matching tests.

## License

MIT — see [LICENSE](LICENSE).
