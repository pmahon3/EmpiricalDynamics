"""End-to-end demo: dual-bandwidth WLS on Lorenz data.

Was the `if __name__ == "__main__"` block at the bottom of
src/edynamics/modelling_tools/projectors/weighted_least_squares.py.
Moved out of the library so the runtime path doesn't drag in
matplotlib + a TkAgg backend on every import.

Run with:
    python examples/wls_lorenz_demo.py
"""
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("TkAgg")

from edynamics.data_sets.lorenz import lorenz_data
from edynamics.modelling_tools.embeddings import Embedding
from edynamics.modelling_tools.estimators import LocalGLSelector
from edynamics.modelling_tools.kernels import Gaussian
from edynamics.modelling_tools.observers import Lag
from edynamics.modelling_tools.projectors import WeightedLeastSquares


def plot_bandwidth_trajectory(anchors, theta_star, sigma_star):
    """Plot theta(t) and sigma(t) over time for the given anchors and
    selected bandwidths."""
    times = np.array(anchors)
    theta = theta_star.cpu().numpy() if hasattr(theta_star, "cpu") else np.array(theta_star)
    sigma = sigma_star.cpu().numpy() if hasattr(sigma_star, "cpu") else np.array(sigma_star)
    plt.figure(figsize=(12, 6))
    plt.plot(times, theta, marker="o", markersize=2, label=r"$\theta(t)$")
    plt.plot(times, sigma, marker="x", markersize=2, label=r"$\sigma(t)$")
    plt.xlabel("Time")
    plt.ylabel("Bandwidth value")
    plt.title("Time series of selected drift (θ) and diffusion (σ) bandwidths")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()


def main() -> None:
    # 1) Generate the Lorenz data
    data = lorenz_data()

    # 2) Define observers: a 0-lag observer for each coordinate
    observers = [Lag("X", 0), Lag("Y", 0), Lag("Z", 0)]

    # 3) Choose library times (skip first 100 to let dynamics settle)
    library_times = data.index[100:1000]

    # 4) Build and compile the embedding
    embedding = Embedding(
        data=data,
        observers=observers,
        library_times=library_times,
        compile_block=True,
    )

    # 5) Select a subset of anchors to tune (every 5th library point)
    anchors = embedding.library_times[::5]

    # 6) Instantiate the WLS projector (no global θ/σ)
    wls = WeightedLeastSquares(
        kernel=Gaussian(theta=1.0, dim=embedding.dimension),
        residual_kernel=Gaussian(theta=1.0, dim=embedding.dimension),
    )

    # 7) Create a coarse candidate grid for testing
    theta_grid = np.logspace(-1, 1.5, 60)  # ~0.1 to ~31.6
    sigma_grid = np.logspace(-1, 1.5, 60)

    selector = LocalGLSelector(
        theta_grid=theta_grid,
        sigma_grid=sigma_grid,
        lwls=wls,
        C=2.0,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )

    # 8) Fit the selector to find (θ*, σ*) at each anchor
    theta_star, sigma_star = selector.fit(embedding, anchors)

    # 8.1) Plot the trajectory of (θ*, σ*) over time
    plot_bandwidth_trajectory(anchors, theta_star, sigma_star)
    plt.savefig("./bandwidth_trajectory.png", dpi=300)

    print("Selected θ values:", theta_star)
    print("Selected σ values:", sigma_star)

    # Sanity-check: no NaNs
    assert not torch.isnan(theta_star).any(), "Found NaN in θ*"
    assert not torch.isnan(sigma_star).any(), "Found NaN in σ*"

    # 9) Forecasting: pick the last 200 points for prediction
    H = 5
    forecast_times = data.index[-200:-H]
    qry = embedding.get_points(forecast_times)

    # 10) Run multi-step forecast with the pointwise-tuned WLS
    result = wls.project(
        embedding=embedding,
        points=qry,
        steps=H,
        step_size=1,
    )

    # 11) Inspect the first few predictions
    print(result.predictions.head())

    # 12) Check shape: should be (len(forecast_times) * H, 3)
    expected_rows = len(forecast_times) * H
    assert result.predictions.shape == (expected_rows, 3)

    # 13) Build truth DataFrame at the exact valid times via the embedding
    valid_times = result.predictions.index.get_level_values(1)
    # get_points returns a DataFrame indexed by those times,
    # with exactly the same columns as the predictions
    truth = embedding.get_points(valid_times)

    # 14) Evaluate forecast skill (MSE, RMSE, MAE, Skill vs persistence)
    metrics = result.evaluate(embedding)
    print("\nForecast Skill Metrics by Lead:\n", metrics)


if __name__ == "__main__":
    main()
