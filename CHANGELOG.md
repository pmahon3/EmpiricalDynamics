# Changelog

All notable changes to `edynamics` are recorded here.  The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
the project follows [SemVer](https://semver.org/) with the caveat
that anything below `1.0.0` may break minor-to-minor.

## [0.4.0] — 2026-05-25

A substantial reshape of the projector + estimator layers, plus a
package-wide cleanup and a real test suite (0 → 44 tests, 87 % line
coverage).  Several breaking API changes; see the migration notes.

### Breaking

- **`WeightedLeastSquares` constructor + use pattern.** WLS now
  exposes a dual-bandwidth interface (drift `θ` and residual `σ`):

      WeightedLeastSquares(*, norm, kernel, residual_kernel)

  Per-anchor `(θ*, σ*)` must be selected by a `LocalGLSelector.fit()`
  pass over the library before `.project()` is called.  The old
  `global_theta=...` constructor argument and the single-kernel
  workflow are gone.  Old call:

      wls = WeightedLeastSquares(norm=Minkowski(p=2),
                                  kernel=Exponential(theta=1.5))

  New call:

      from edynamics.modelling_tools.kernels import Gaussian
      from edynamics.modelling_tools.estimators import LocalGLSelector
      wls = WeightedLeastSquares(
          kernel=Gaussian(theta=1.0, dim=embedding.dimension),
          residual_kernel=Gaussian(theta=1.0, dim=embedding.dimension),
      )
      LocalGLSelector(theta_grid=..., sigma_grid=..., lwls=wls).fit(
          embedding, embedding.library_times[::10])
      result = wls.project(embedding=embedding, points=qry,
                            steps=1, step_size=1)

- **`WLSResult` → `RoseResult`.**  The return type of
  `WeightedLeastSquares.project()` was renamed.  It carries the same
  `predictions` DataFrame plus a new `.evaluate(embedding)` method
  that returns per-lead MSE / RMSE / MAE / Skill-vs-persistence.

- **`class constant` → `class Constant`** (PEP 8 conformance).  Any
  `from edynamics.modelling_tools.kernels import constant` becomes
  `import Constant`.

- **`convergent_cross_mapping` is no longer importable from
  `edynamics`.**  The implementation moved to `sketches/` while it
  remains in prototyping; it is not part of the installed package.
  Tracked separately and may return as a stable API once settled.

- **`estimators/nonlinearity.py` removed.**  The single-θ
  `nonlinearity()` sweep was superseded by the joint `(θ, σ)` grid
  in `LocalGLSelector`.

### Added

- `kernels.Gaussian(theta, dim)` — normalised squared-exponential
  kernel.
- `estimators.LocalGLSelector` — per-anchor `(θ, σ)` selection via
  the Goldenshluger-Lepski criterion; populates `WLS.anchor_times`,
  `.best_theta_vals`, `.best_sigma_vals` for later forecast use.
- `observers.ColumnObserver(observation_name, variable_name)` — a
  passthrough observer that reads one column from a DataFrame.
- `RoseResult.evaluate(embedding)` — per-lead error + persistence
  skill metrics.
- 44-test pytest suite covering every kernel, observer, norm,
  projector, and estimator (87 % line coverage).
- `examples/wls_lorenz_demo.py` — end-to-end Lorenz dual-bandwidth
  demo (was previously the inline `__main__` block at the bottom of
  `weighted_least_squares.py`).

### Fixed

- **`LagMovingAverage.__hash__`** silently returned `None` (missing
  `return` on the `hash((self.variable_name, self.tau, self.q))`
  call).  Instances were therefore unhashable in any code path that
  used Python dict/set semantics.
- **`Embedding.__hash__`** would raise `TypeError` if ever called
  (the implementation tried to hash a `pd.DataFrame`).  Now
  explicitly `__hash__ = None` to mark the class unhashable at the
  ABC layer.
- **`WeightedLeastSquares.__init__`** defaulted `kernel=Kernel(theta=1.0)`,
  which is an abstract class and would have `TypeError`-ed on any
  caller that passed `kernel=None`.  Default is now
  `Exponential(theta=1.0)`.
- **`Projector.update_values`** swallowed a `KeyError` when a newly-
  added observer referenced a variable not yet present in the
  predictions frame, leaving `data` stale for the rest of the
  iteration.  The pattern is now an explicit branch on the
  variable's presence in `predictions.columns`.
- **`mappers/cross_mapping.py`** passed an unknown `max_time=` argument
  to `Embedding.get_k_nearest_neighbours()` (the parameter doesn't
  exist).  Removed.  (Module subsequently moved to `sketches/`.)

### Performance

- **`Embedding.compile()` / `Embedding.get_points()`** rebuild the
  embedding block via a one-shot column-dict + single DataFrame
  construction, eliminating the per-column DataFrame fragmentation
  cost and the implicit copy `cKDTree` was doing on the DataFrame
  input.  Pass `.values` to `cKDTree` directly.
- **`WLS._wls_multi_step`** collects per-step forecasts into a row
  buffer and constructs the predictions DataFrame once, instead of
  `.loc[(t1, t2)] = x_next` per step inside the inner loop.  Library
  block sliced once into an ndarray; subsequent steps advance via
  integer offsets rather than `.iloc[1:]` rebuilds.  Distance
  computation in the inner loop bypasses the `Norm.distance_matrix`
  indirection and operates on the already-materialised slice
  directly.

### Removed

- `projectors/_lsh.py` (vendored LSH from gamboviol/lsh) — never
  imported anywhere.
- `utils/ProgressBar.py`, `utils/ProgressBarActor.py` — never
  imported.  Use `tqdm` directly if you need progress feedback.
- `Embedding.get_ball_point()` — never called.  The underlying
  `cKDTree.query_ball_point` is still reachable via
  `embedding.distance_tree.query_ball_point(...)` if needed.
- `estimators.iterative_bootstrap()` — dead prototype, never
  successfully ran (had a `ColumnObserver(c)` constructor-arity bug);
  recoverable from git history (commit `b40ce0b^`).
- `setup.py` (everything is now in `pyproject.toml`).
- `requirements.txt` (deps live in `pyproject.toml`).

### Renamed (internal; no public-import-path change)

- `observers/observer.py` → `observers/base.py`
- `observers/observers.py` → `observers/concrete.py`
- `LISCENSE` → `LICENSE`

### Packaging

- `pyproject.toml` is now the single source of build + project
  metadata; `setup.py` is gone.
- `wheel` added to `[build-system].requires` so `pip install -e .`
  works on environments without `wheel` pre-installed.
- Runtime deps removed from `[build-system].requires` (previous
  inclusion forced pip to re-download `numpy`/`pandas`/`scipy`/`ray`
  into an isolated build env on every install, which flaked on
  Compute Canada CVMFS).
- `torch>=2.0` declared as a runtime dep (was implicitly required
  by `weighted_least_squares.py` and `local_gl_selector.py` but
  never listed).
- `[project.urls]` set so PyPI displays a Homepage link.

## [0.3.14] — 2024-01-27

(First documented release.  Pre-changelog work.)
