"""Demo: iterative_bootstrap() lifts an embedding via PF-Koopman.

Was the `if __name__ == "__main__"` block at the bottom of
src/edynamics/modelling_tools/estimators/local_gl_selector.py.

Run with:
    python examples/local_gl_iterative_bootstrap_demo.py
"""
import numpy as np

from edynamics.data_sets.lorenz import lorenz_data
from edynamics.modelling_tools.estimators.local_gl_selector import iterative_bootstrap
from edynamics.modelling_tools.kernels import Gaussian
from edynamics.modelling_tools.observers import Lag


def main() -> None:
    # 1) Generate Lorenz data
    data = lorenz_data()

    # 2) Initial embedding: a 3-dim delay of X, Y, Z (no delay)
    observers = [Lag("X", 0), Lag("Y", 0), Lag("Z", 0)]
    library_times = data.index[100:1000]   # skip transients

    # 3) PF-GL grids
    theta_grid = np.linspace(0.1, 5.0, 20)
    sigma_grid = np.linspace(0.1, 5.0, 20)

    # 4) Run the PF-Koopman bootstrap
    init_theta = 1.0
    init_sigma = 1.0

    lifted_embedding = iterative_bootstrap(
        data=data,
        observers=observers,
        library_times=library_times,
        theta_grid=theta_grid,
        sigma_grid=sigma_grid,
        wls_kernel=Gaussian(theta=init_theta, dim=7),
        wls_residual_kernel=Gaussian(theta=init_sigma, dim=7),
    )

    # 5) Inspect the lifted embedding
    print("Lifted embedding block (first 5 rows):")
    print(lifted_embedding.block.head())


if __name__ == "__main__":
    main()
