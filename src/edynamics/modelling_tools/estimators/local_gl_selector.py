from typing import Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from edynamics.modelling_tools.embeddings import Embedding
from edynamics.modelling_tools.projectors import WeightedLeastSquares


class LocalGLSelector:
    """
    Selects local (θ, σ) for each anchor point via the pointwise Goldenshluger–Lepski criterion.

    Parameters
    ----------
    theta_grid : Sequence[float]
        Candidate values for the drift bandwidth θ.
    sigma_grid : Sequence[float]
        Candidate values for the diffusion bandwidth σ.
    lwls : WeightedLeastSquares
        An instance of the weighted least-squares projector.
    C : float, default=2.0
        Penalty constant in the GL criterion.
    device : torch.device or str, default='cpu'
        Device for torch tensors.
    dtype : torch.dtype, default=torch.float64
        Data type for torch tensors.
    """

    def __init__(
            self,
            theta_grid: Sequence[float],
            sigma_grid: Sequence[float],
            lwls: WeightedLeastSquares,
            C: float = 2.0,
            device: torch.device = torch.device('cpu'),
            dtype: torch.dtype = torch.float64
    ):
        self.lwls = lwls
        self.C = C
        self.device = device
        self.dtype = dtype

        # Candidate grids as torch tensors
        self.thetas = torch.tensor(theta_grid, device=self.device, dtype=self.dtype)
        self.sigmas = torch.tensor(sigma_grid, device=self.device, dtype=self.dtype)

        # To be filled by fit()
        self.anchor_times: pd.DatetimeIndex = pd.DatetimeIndex([])
        self.theta_star: torch.Tensor = torch.tensor([])
        self.sigma_star: torch.Tensor = torch.tensor([])

    def fit(
            self,
            embedding: Embedding,
            library_times: Sequence[pd.Timestamp]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fit the selector on the given embedding and anchor times.

        Returns
        -------
        theta_star : torch.Tensor of shape (n_anchors,)
        sigma_star : torch.Tensor of shape (n_anchors,)
        """
        # Record anchors
        self.anchor_times = pd.DatetimeIndex(library_times)
        self.lwls.anchor_times = self.anchor_times

        n_anchors = len(self.anchor_times)
        n_theta = len(self.thetas)
        n_sigma = len(self.sigmas)
        d = embedding.dimension

        # Prepare output tensors
        self.theta_star = torch.empty(n_anchors, device=self.device, dtype=self.dtype)
        self.sigma_star = torch.empty(n_anchors, device=self.device, dtype=self.dtype)

        # Full embedding block and numpy array for distance calcs
        block = embedding.block
        points_np = block.values  # shape (T, d)

        for idx, t in enumerate(self.anchor_times):
            # Anchor state vector
            x_anchor = block.loc[t].values  # (d,)

            # Precompute distances from anchor to all points
            dists_all = np.linalg.norm(points_np - x_anchor, axis=1)  # (T,)

            best_score = np.inf
            best_theta = None
            best_sigma = None

            # Loop over θ candidates
            for theta in self.thetas.cpu().numpy():
                # Drift weights
                self.lwls.kernel.theta = float(theta)

                # Prepare leave-one-out block (drop anchor)
                mask = block.index != t
                block_lo = block.loc[mask]
                dists_lo = dists_all[mask]  # (T-1,)

                # X_full / Y_full for one-step forecast
                X_full = block_lo.iloc[:-1].values  # (N, d)
                Y_full = block_lo.iloc[1:].values  # (N, d)
                dists_lo = dists_lo[:-1]  # align with X_full

                # Compute weights and effective sample size
                w = self.lwls.kernel.weigh(dists_lo)  # (N,)
                n_eff = w.sum()
                if n_eff <= 0:
                    # No effective points; skip this θ
                    continue

                # Weighted least-squares fit
                WX = (w[:, None] * X_full)  # (N, d)
                WY = (w[:, None] * Y_full)  # (N, d)
                C_mat = np.linalg.lstsq(WX, WY, rcond=None)[0]  # (d, d)

                # Residuals and their norms
                resid = WY - WX @ C_mat  # (N, d)
                rnorms = np.linalg.norm(resid, axis=1)  # (N,)

                # Loop over σ candidates
                for sigma in self.sigmas.cpu().numpy():
                    self.lwls.residual_kernel.theta = float(sigma)
                    g = self.lwls.residual_kernel.weigh(rnorms)  # (N,)
                    E = np.dot(w, g)

                    penalty = self.C * (theta ** (-d) + sigma ** (-d)) / n_eff
                    score = E + penalty

                    if score < best_score:
                        best_score = score
                        best_theta = float(theta)
                        best_sigma = float(sigma)

            # Store best for this anchor
            self.theta_star[idx] = best_theta
            self.sigma_star[idx] = best_sigma

        # Write back into LWLS for forecasting
        self.lwls.best_theta_vals = self.theta_star.cpu().numpy()
        self.lwls.best_sigma_vals = self.sigma_star.cpu().numpy()

        return self.theta_star, self.sigma_star


