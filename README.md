# edynamics

Empirical dynamic modelling toolkit: delay-embedding state-space
reconstruction with locally-weighted projection methods, organised as
a small set of composable building blocks (embeddings, observers,
kernels, norms, projectors, estimators).

## Install

From a checkout, into a Python 3.10+ environment:

```bash
pip install -e .
```

Dependencies (`numpy`, `pandas`, `scipy`, `tqdm`, `ray`, `torch`)
are declared in `pyproject.toml` and installed automatically.

If a build environment can't reach PyPI (e.g. air-gapped or
Compute-Canada CVMFS with pre-installed wheels), preinstall the
runtime deps and add `--no-build-isolation`:

```bash
pip install numpy pandas scipy tqdm ray torch wheel setuptools
pip install -e . --no-build-isolation
```

## What's in it

- `edynamics.modelling_tools.embeddings.Embedding` — the delay
  embedding container; binds raw data to a set of `Observer`s
  (e.g. `Lag`) and exposes a `block` of state-vector rows indexed
  by time.
- `edynamics.modelling_tools.observers.Lag` — the canonical
  delay-coordinate observer.
- `edynamics.modelling_tools.kernels` — `Gaussian`, `Exponential`,
  `Epanechnikov`, `Tricubic`, `constant`.
- `edynamics.modelling_tools.norms.Minkowski`.
- `edynamics.modelling_tools.projectors.WeightedLeastSquares` —
  locally-weighted least-squares projection with separate drift
  (`theta`) and residual (`sigma`) bandwidths, selected per anchor
  by `LocalGLSelector`.
- `edynamics.modelling_tools.projectors.KNearestNeighbours`.
- `edynamics.modelling_tools.estimators.LocalGLSelector` — per-anchor
  joint `(theta, sigma)` grid search over a generalised-log-likelihood
  objective.
- `edynamics.modelling_tools.estimators.dimensionality` — embedding-
  dimension prediction-skill sweep.

## Minimal example

```python
import pandas as pd
from edynamics.modelling_tools import Embedding, Lag
from edynamics.modelling_tools.projectors import WeightedLeastSquares
from edynamics.modelling_tools.kernels import Gaussian

# 1. Build an embedding from a 1-D time series
data = pd.DataFrame(...)                # DatetimeIndex x one column
lags = [Lag(variable_name="x", tau=-i) for i in range(3)]
emb  = Embedding(data=data, observers=lags, library_times=data.index[3:-1])
emb.compile()

# 2. Set up a projector + fit per-anchor bandwidths
from edynamics.modelling_tools.estimators import LocalGLSelector
wls = WeightedLeastSquares(
    kernel=Gaussian(theta=1.0, dim=emb.dimension),
    residual_kernel=Gaussian(theta=1.0, dim=emb.dimension),
)
sel = LocalGLSelector(theta_grid=..., sigma_grid=..., lwls=wls, C=2.0)
sel.fit(emb, library_times=emb.library_times[::10])

# 3. Project / forecast
result = wls.project(emb, ...)          # returns a RoseResult
metrics = result.evaluate(emb)
```

## License

MIT — see `LICENSE`.
