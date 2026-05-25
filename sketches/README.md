# sketches/

Unmaintained prototype code, kept out of the public package surface
but not deleted because it's still being thought about.

Contents are **not** part of the installed `edynamics` package: they
live outside `src/edynamics/` so `pip install -e .` does not expose
them via `from edynamics. ...` imports. To run a sketch you have to
add `sketches/` to `PYTHONPATH` or invoke the file directly.

## Current contents

- `cross_mapping_prototype/` — convergent cross-mapping (Sugihara
  2012-style causality detection). A working sketch, untested, with
  one TODO (`# todo: generalize Norm.distance function to handle
  this use case`). Originally lived at
  `src/edynamics/modelling_tools/mappers/`. Moved here once it was
  determined to be still-in-prototyping and not yet stable enough
  to expose as a public API.

## Policy

If a sketch matures into a stable feature with tests, move it back
into `src/edynamics/`. If it's clearly abandoned, delete it; it's
recoverable from git history if needed.
