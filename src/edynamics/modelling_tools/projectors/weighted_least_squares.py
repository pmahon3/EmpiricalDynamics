from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

from edynamics.modelling_tools.embeddings import Embedding
from edynamics.modelling_tools.kernels import Exponential, Kernel
from edynamics.modelling_tools.norms import Minkowski, Norm
from .projector import Projector


@dataclass
class WLSResult:
    predictions: pd.DataFrame
    coefficients: Optional[torch.Tensor] = None   # (n_pts, steps, d, d)
    covariances:  Optional[torch.Tensor] = None   # (n_pts, steps, d, d)


class WeightedLeastSquares(Projector):
    """
    Locally weighted least-squares (S‐map style), now with an optional
    `global_theta` that, if provided, forces every point to use that θ,
    bypassing both “offline‐computed best‐θ” and “kernel‐voting”.
    """

    def __init__(
        self,
        norm: Norm | None   = None,
        kernel: Kernel | None = None,
        *,
        global_theta: Optional[float] = None
    ) -> None:
        super().__init__(
            norm   = norm   or Minkowski(p=2),
            kernel = kernel or Exponential(theta=0.0),
        )

        # --- New: if user sets `global_theta`, we will ignore per-anchor θ/voting ---
        self.global_theta: Optional[float] = global_theta

        # The following are set only if you call fit_best_thetas(...)
        self.anchor_embeddings: Optional[np.ndarray] = None  # (N, d)
        self.anchor_times:      Optional[pd.DatetimeIndex] = None
        self.best_theta_vals:   Optional[np.ndarray] = None  # (N,)
        self.voting_kernel:     Optional[Kernel] = None      # for new‐point θ‐prediction

    # -------------------------------------------------------------------------
    def fit_best_thetas(
        self,
        embedding: Embedding,
        anchor_times: pd.DatetimeIndex,
        residuals_arr: np.ndarray,
        theta_grid: np.ndarray,
        voting_kernel: Kernel
    ) -> None:
        """
        Offline step: populate
          • self.anchor_embeddings  (N, d)
          • self.anchor_times       (N,)
          • self.best_theta_vals    (N,)
          • self.voting_kernel
        so that later .project(...) knows how to pick θ for anchors (lookup) or new points (vote).
        """
        # (1) Gather anchor embeddings (N, d)
        anchor_df = embedding.get_points(anchor_times)
        if anchor_df.shape[0] != len(anchor_times):
            raise ValueError(
                "Some anchor_times not found in embedding. "
                "Ensure your embedding is compiled over exactly those times."
            )
        self.anchor_embeddings = anchor_df.values   # array shape (N, d)
        self.anchor_times      = anchor_times

        # (2) Compute ‖r‖ over residuals_arr (N, K, d) → (N, K)
        res_norms = np.linalg.norm(residuals_arr, axis=2)  # shape (N, K)

        # (3) Find argmin along axis=1 → index (N,), then lookup θ
        best_idx = np.nanargmin(res_norms, axis=1)         # shape (N,)
        self.best_theta_vals = theta_grid[best_idx]        # shape (N,)

        # (4) Store the kernel used for voting new points
        self.voting_kernel = voting_kernel

    # -------------------------------------------------------------------------
    def project(
        self,
        *,
        embedding:  Embedding,
        points:     pd.DataFrame,   # index = times, values = embedding vectors
        steps:      int,
        step_size:  int,
        leave_out:  bool = True,
        return_coefficients: bool = False,
        use_innovations:     bool = False,
        rng: Optional[np.random.Generator] = None,
    ) -> WLSResult:
        """
        For each row in `points`:
          1) If self.global_theta is not None:  θ_i = self.global_theta
          2) else if timestamp ∈ self.anchor_times:  θ_i = self.best_theta_vals[...] (lookup)
          3) else:  (new point) → vote θ via self.voting_kernel over self.anchor_embeddings

        Then set self.kernel.theta = θ_i, run one-step WLS, collect predictions & coefficients.
        """
        # Ensure anchors & voting info are available if we need them
        if self.global_theta is None:
            if self.anchor_embeddings is None or self.best_theta_vals is None or self.voting_kernel is None:
                raise RuntimeError(
                    "You must call fit_best_thetas(...) first, or set global_theta in the constructor."
                )

        # 1) Build the multi‐index (Current_Time, Prediction_Time) exactly as before
        indices = self.build_prediction_index(
            frequency = embedding.frequency,
            index     = points.index,
            steps     = steps,
            step_size = step_size,
        )

        d     = embedding.block.shape[1]
        n_pts = len(points)

        # Prepare storage for coefficients & covariances
        coeff_t = torch.empty((n_pts, steps, d, d), dtype=torch.float64) \
                  if return_coefficients else None
        cov_t   = torch.empty((n_pts, steps, d, d), dtype=torch.float64) \
                  if use_innovations else None

        all_pred: List[pd.DataFrame] = []
        rng = rng or np.random.default_rng()

        # 2) Loop over each point, pick θ_i according to the rules, then call WLS sub‐step
        for i, (timestamp, point_vals) in enumerate(zip(points.index, points.values)):
            # --- Decide which θ to use ---
            if self.global_theta is not None:
                # Option A: user has forced a single global θ for all points
                θ_i = float(self.global_theta)

            else:
                # Option B: local anchor lookup or kernel‐voting
                if timestamp in self.anchor_times:
                    # anchor → look up
                    idx_anchor = int(np.where(self.anchor_times == timestamp)[0][0])
                    θ_i = float(self.best_theta_vals[idx_anchor])
                else:
                    # new point → vote
                    x_t = point_vals  # (d,)
                    diffs = self.anchor_embeddings - x_t[None, :]  # shape (N, d)
                    dists = np.linalg.norm(diffs, axis=1)           # shape (N,)
                    w = self.voting_kernel.weigh(dists)             # shape (N,)
                    if np.all(w == 0):
                        w = np.ones_like(w) / len(w)
                    else:
                        w = w / np.sum(w)
                    θ_i = float(np.dot(w, self.best_theta_vals))

            # --- Set the kernel to this θ_i ---
            self.kernel.theta = θ_i

            # --- Prepare per‐point slices for coefficients and covariances ---
            idx_slice  = indices[i*steps:(i+1)*steps]
            coeff_view = coeff_t[i] if coeff_t is not None else None
            cov_view   = cov_t[i]   if cov_t   is not None else None

            # --- Run the one‐step (or multi‐step) WLS for this point & θ_i ---
            df_single = self._wls_multi_step(
                embedding   = embedding,
                point0      = point_vals.copy(),  # (d,)
                indices     = idx_slice,
                steps       = steps,
                step_size   = step_size,
                leave_out   = leave_out,
                coeff_store = coeff_view,
                cov_store   = cov_view,
                stochastic  = use_innovations,
                rng         = rng,
            )
            all_pred.append(df_single)

        # 3) Concatenate per‐point predictions into a single DataFrame
        preds = pd.DataFrame(index=indices, columns=embedding.block.columns)
        for df in all_pred:
            preds.loc[df.index] = df.values

        return WLSResult(
            predictions = preds,
            coefficients= coeff_t,
            covariances = cov_t
        )

    # -------------------------------------------------------------------------
    def _wls_multi_step(
        self,
        *,
        embedding:   Embedding,
        point0:      np.ndarray,
        indices:     pd.MultiIndex,
        steps:       int,
        step_size:   int,
        leave_out:   bool,
        coeff_store: Optional[torch.Tensor],
        cov_store:   Optional[torch.Tensor],
        stochastic:  bool,
        rng:         np.random.Generator,
    ) -> pd.DataFrame:
        """
        (Unchanged from your prior implementation.)
        """
        predictions = pd.DataFrame(index=indices,
                                   columns=embedding.block.columns,
                                   dtype=float)

        current_time, prediction_time = indices[0]

        # library selection (leave‐one‐out or not)
        if leave_out:
            lib_times = embedding.library_times[~embedding.library_times.isin(indices.droplevel(0))]
        else:
            lib_times = embedding.library_times[embedding.library_times <= current_time]

        valid = lib_times.intersection(embedding.block.index)
        block = embedding.block.loc[valid]

        X_full = block.iloc[:-step_size]
        Y_full = block.iloc[ step_size:]

        x = point0.copy()

        for j in range(steps):
            dist = self.norm.distance_matrix(embedding, x[None, :], valid)[:-step_size]
            w    = self.kernel.weigh(dist)
            WX, Wy = w * X_full.values, w * Y_full.values

            C = np.linalg.lstsq(WX, Wy, rcond=None)[0]  # (d, d)

            # compute raw residuals
            resid = Wy - WX @ C  # shape (n_neighbors, d)

            # ✱ compute weighted covariance with residual_kernel
            Σ = self._local_cov_from_weighted(
                WX=WX, Wy=Wy, C=C, W=W.sum(),
                rids=resid,
                residual_kernel=self.residual_kernel
            )

            if cov_store is not None:
                cov_store[j] = torch.as_tensor(Σ)

            x_next = x @ C
            if stochastic:
                noise = rng.multivariate_normal(mean=np.zeros_like(x_next), cov=Σ)
                x_next += noise

            predictions.loc[(current_time, prediction_time)] = x_next

            if coeff_store is not None:
                coeff_store[j] = torch.as_tensor(C)

            # advance
            if j < steps - 1:
                x = x_next
                prediction_time = indices[j+1][-1]
                X_full = X_full.iloc[1:]
                Y_full = Y_full.iloc[1:]
                valid   = valid[1:]

        return predictions

    # -------------------------------------------------------------------------
    @staticmethod
    def _local_cov_from_weighted(
            WX: np.ndarray,
            Wy: np.ndarray,
            C: np.ndarray,
            W: float,
            *,
            rids: np.ndarray,  # ✱ the raw residuals (n_neighbors, d)
            residual_kernel: Kernel  # ✱ the new kernel
    ) -> np.ndarray:
        """
        Compute Σ = (1/W_r) ∑ w_drift_i * g_resid(||rids[i]||) * (rids[i] rids[i].T)
        where:
          - WX, Wy, W are from the drift fit,
          - rids are the raw residual vectors,
          - residual_kernel.weigh(distances) gives g_resid.
        """
        # drift weights W_drift were already built into WX/Wy normalization

        # compute residual weights g_resid
        rnorms = np.linalg.norm(rids, axis=1)  # (n_neighbors,)
        w_resid = residual_kernel.weigh(rnorms)  # (n_neighbors,)
        W_resid = w_resid.sum()

        # center residuals by weighted mean
        mean_r = np.average(rids, axis=0, weights=w_resid)
        r_centered = rids - mean_r[None, :]

        # weighted covariance
        return (r_centered * w_resid[:, None]).T @ r_centered / W_resid
