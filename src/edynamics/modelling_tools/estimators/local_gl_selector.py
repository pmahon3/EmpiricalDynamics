import torch
import numpy as np
import pandas as pd
from typing import Sequence, Tuple, List

from scipy.stats import multivariate_normal

from edynamics import lorenz_data
from edynamics.modelling_tools import Lag
from edynamics.modelling_tools.embeddings import Embedding
from edynamics.modelling_tools.kernels import Kernel, Gaussian
from edynamics.modelling_tools.observers import ColumnObserver, Observer
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


def iterative_bootstrap(
        data: pd.DataFrame,
        observers: List[Observer],
        library_times: pd.DatetimeIndex,
        theta_grid: np.ndarray,
        sigma_grid: np.ndarray,
        wls_kernel: Kernel,
        wls_residual_kernel: Kernel,
        C_pf: float = 2.0,
        max_iters: int = 5,
        tol_dial: float = 1e-3,
        min_gap: float = 2.0,
        holdout_frac: float = 0.1
) -> Embedding:
    """
    Perform the PF–Koopman bootstrap loop to lift the embedding via residual modes.

    Parameters
    ----------
    data
      The raw time series.
    observers
      Initial list of Observer (e.g. delays).
    library_times
      Times to use for the embedding / PF anchor set.
    theta_grid, sigma_grid
      1D arrays of candidate bandwidths for the PF GL selector.
    C_pf
      Penalty constant in the PF GL criterion.
    max_iters
      Maximum number of bootstrap iterations.
    tol_dial
      Stop if |θ_k - θ_{k-1}| and |σ_k - σ_{k-1}| both < tol_dial.
    min_gap
      Stop if spectral gap λ_r / λ_{r+1} < min_gap.
    holdout_frac
      Fraction of library_times to hold out for optional forecasting check.

    Returns
    -------
    embedding
      The final lifted Embedding.
      :param wls_residual_kernel:
    """
    # 1) initial embedding
    embedding = Embedding(data=data, observers=observers,
                          library_times=library_times, compile_block=True)
    anchors = embedding.library_times

    # optional hold‐out split
    n_hold = int(len(anchors) * holdout_frac)
    holdout = anchors[-n_hold:] if n_hold > 0 else pd.DatetimeIndex([])
    train_anchors = anchors[:-n_hold] if n_hold > 0 else anchors

    # initialize PF selector and WLS
    wls = WeightedLeastSquares(kernel=wls_kernel, residual_kernel=wls_residual_kernel)
    selector = LocalGLSelector(theta_grid, sigma_grid, lwls=wls, C=C_pf)

    prev_theta, prev_sigma = None, None

    for k in range(max_iters):
        # --- PF step: fit local GL on train anchors ---
        selector.fit(embedding, train_anchors)
        theta_star = selector.theta_star
        sigma_star = selector.sigma_star

        # check dial convergence
        if prev_theta is not None:
            dθ = np.max(np.abs(theta_star - prev_theta))
            dσ = np.max(np.abs(sigma_star - prev_sigma))
            if dθ < tol_dial and dσ < tol_dial:
                print(f"[iter {k}] dials converged (dθ={dθ:.2e}, dσ={dσ:.2e}) → stop")
                break
        prev_theta, prev_sigma = theta_star.copy_(), sigma_star.copy_()

        # --- build PF matrix on anchors ---
        X = embedding.block.loc[train_anchors].values  # (J, d)
        J = len(train_anchors)
        P = np.zeros((J, J), float)
        for i in range(J):
            # set this anchor's θ,σ
            wls.kernel.theta = float(theta_star[i])
            wls.residual_kernel.theta = float(sigma_star[i])
            # weights from drift kernel only
            dists = np.linalg.norm(X - X[i: i + 1], axis=1)
            w = wls.kernel.weigh(dists)
            if w.sum() <= 0:
                P[i, :] = 1.0 / J
            else:
                P[i, :] = w / w.sum()

        # --- eigen‐decomposition and spectral‐gap r ---
        eigvals, eigvecs = np.linalg.eig(P.T)
        # sort by descending |λ|
        idx = np.argsort(-np.abs(eigvals))
        eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
        gaps = np.abs(eigvals[:-1] / (eigvals[1:] + 1e-16))
        r = int(np.argmax(gaps) + 1)
        gap_val = gaps[r - 1]
        print(f"[iter {k}] spectral gap at r={r} is {gap_val:.2f}")

        if gap_val < min_gap:
            print(f"[iter {k}] gap {gap_val:.2f} < {min_gap} → stop")
            break

        # --- lifting: build new features ψ₁,…,ψᵣ ---
        psi = eigvecs[:, :r].real  # (J, r)
        modes_df = pd.DataFrame(psi, index=train_anchors,
                                columns=[f"psi_{k + 1}" for k in range(r)])

        # augment the embedding.block only on the train anchors
        aug = embedding.block.loc[train_anchors].copy()
        for col in modes_df.columns:
            aug[col] = modes_df[col]

        # --- optional: hold‐out forecast error check (not implemented here) ---

        # --- rebuild embedding on train anchors with new observers ---
        new_obs = [ColumnObserver(c) for c in aug.columns]
        embedding = Embedding(data=aug,
                              observers=new_obs,
                              library_times=train_anchors,
                              compile_block=True)

    return embedding


