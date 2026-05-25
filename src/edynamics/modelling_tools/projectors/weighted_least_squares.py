from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import torch

from edynamics.modelling_tools.embeddings import Embedding
from edynamics.modelling_tools.kernels import Exponential, Kernel
from edynamics.modelling_tools.norms import Minkowski, Norm
from edynamics.modelling_tools.projectors import Projector


@dataclass
class RoseResult:
    predictions: pd.DataFrame  # MultiIndex: (origin_time, lead_time) → state-vector
    coefficients: Optional[torch.Tensor] = None
    covariances: Optional[torch.Tensor] = None
    resid_means: Optional[torch.Tensor] = None
    resid_eigvals: Optional[torch.Tensor] = None

    def evaluate(
        self,
        embedding: Embedding,
        persistence: bool = True
    ) -> pd.DataFrame:
        """
        Compute MSE, RMSE, MAE, and Skill vs persistence by integer lead h,
        pulling true states directly via embedding.get_points().
        """
        preds = self.predictions
        # Extract origin & valid timestamps
        origins = preds.index.get_level_values(0)
        valids  = preds.index.get_level_values(1)

        # Pull prediction & true arrays in lock-step
        P  = preds.values                                  # (N, d)
        X  = embedding.get_points(valids).values           # (N, d)
        X0 = embedding.get_points(origins).values          # (N, d)

        # Infer single‐step interval from the first forecast
        step = valids[0] - origins[0]
        # Compute integer lead for each row
        leads = ((valids - origins) / step).astype(int)

        records = []
        for h in np.unique(leads):
            idx = np.where(leads == h)[0]
            Ph, Xh = P[idx], X[idx]

            E = Ph - Xh
            mse  = np.mean((E**2).sum(axis=1))
            rmse = np.sqrt(mse)
            mae  = np.mean(np.abs(E).sum(axis=1))

            skill = np.nan
            if persistence and h > 0:
                P0h = X0[idx]
                Ep  = P0h - Xh
                mse_p = np.mean((Ep**2).sum(axis=1))
                skill = 1 - mse/mse_p if mse_p > 0 else np.nan

            records.append({
                "lead":  h,
                "MSE":   mse,
                "RMSE":  rmse,
                "MAE":   mae,
                "Skill": skill,
            })

        return pd.DataFrame.from_records(records).set_index("lead")



class WeightedLeastSquares(Projector):
    """
    Locally weighted least-squares with *dual* bandwidths (θ, σ).

    Drift‐kernel:       self.kernel  (θ)
    Residual‐kernel:    self.residual_kernel  (σ)

    You must have run LocalGLSelector.fit(…) first, which
    populates self.anchor_times, self.best_theta_vals, self.best_sigma_vals.
    """

    def __init__(
            self,
            *,
            norm: Norm | None = None,
            kernel: Kernel | None = None,
            residual_kernel: Kernel | None = None,
    ) -> None:
        super().__init__(
            norm=norm or Minkowski(p=2),
            kernel=kernel or Exponential(theta=1.0),
        )
        self.residual_kernel: Kernel = residual_kernel or Exponential(theta=1.0)

        # To be filled by LocalGLSelector
        self.anchor_times: pd.DatetimeIndex | None = None
        self.best_theta_vals: np.ndarray | None = None
        self.best_sigma_vals: np.ndarray | None = None

    def project(
            self,
            *,
            embedding: Embedding,
            points: pd.DataFrame,
            steps: int,
            step_size: int,
            leave_out: bool = True,
            return_coefficients: bool = False,
            return_residual_stats: bool = False,
            use_innovations: bool = False,
            rng: Optional[np.random.Generator] = None,
    ) -> RoseResult:
        """
        Multi-step forecast. At each query time t:
         1) Find nearest anchor in self.anchor_times
         2) θ ← self.best_theta_vals[idx], σ ← self.best_sigma_vals[idx]
         3) Run the standard WLS‐multi‐step with those kernels.
        """
        if self.anchor_times is None or self.best_theta_vals is None or self.best_sigma_vals is None:
            raise RuntimeError(
                "anchor_times and best_*_vals must be set (via LocalGLSelector) before calling project().")

        # build output index
        indices = self.build_prediction_index(
            frequency=embedding.frequency,
            index=points.index,
            steps=steps,
            step_size=step_size,
        )

        d, n_pts = embedding.block.shape[1], len(points)
        coeff_t = torch.empty((n_pts, steps, d, d), dtype=torch.float64) if return_coefficients else None
        cov_t = torch.empty((n_pts, steps, d, d), dtype=torch.float64) if use_innovations else None
        mu_t = torch.empty((n_pts, steps, d), dtype=torch.float64) if return_residual_stats else None
        eig_t = torch.empty((n_pts, steps, d), dtype=torch.float64) if return_residual_stats else None

        rng = rng or np.random.default_rng()
        all_pred: List[pd.DataFrame] = []

        # Precompute numpy array of anchor times for distance
        anchor_np = np.array(self.anchor_times.values, dtype="datetime64[ns]")

        for i, (tstamp, x_np) in enumerate(zip(points.index, points.values)):
            # 1) find nearest anchor index
            #    convert both to numpy datetime64 for vectorized abs
            t64 = np.datetime64(tstamp)
            deltas = np.abs(anchor_np - t64)
            idx_anchor = int(np.argmin(deltas))

            # 2) pick θ & σ
            theta_i = float(self.best_theta_vals[idx_anchor])
            sigma_i = float(self.best_sigma_vals[idx_anchor])

            # 3) set kernels
            self.kernel.theta = theta_i
            self.residual_kernel.theta = sigma_i

            # 4) run multi-step WLS
            block_slice = indices[i * steps: (i + 1) * steps]
            df_one = self._wls_multi_step(
                embedding=embedding,
                point0=x_np.copy(),
                indices=block_slice,
                steps=steps,
                step_size=step_size,
                leave_out=leave_out,
                coeff_store=coeff_t[i] if coeff_t is not None else None,
                cov_store=cov_t[i] if cov_t is not None else None,
                mu_store=mu_t[i] if mu_t is not None else None,
                eig_store=eig_t[i] if eig_t is not None else None,
                stochastic=use_innovations,
                rng=rng,
            )
            all_pred.append(df_one)

        # assemble
        preds = pd.DataFrame(index=indices, columns=embedding.block.columns, dtype=float)
        for df in all_pred:
            preds.loc[df.index] = df.values

        return RoseResult(
            predictions=preds,
            coefficients=coeff_t,
            covariances=cov_t,
            resid_means=mu_t,
            resid_eigvals=eig_t
        )

    # ---------------------------------------------------------------------
    # Internal multi‑step routine
    # ---------------------------------------------------------------------
    def _wls_multi_step(
            self,
            *,
            embedding: Embedding,
            point0: np.ndarray,
            indices: pd.MultiIndex,
            steps: int,
            step_size: int,
            leave_out: bool,
            coeff_store: Optional[torch.Tensor],
            cov_store: Optional[torch.Tensor],
            mu_store: Optional[torch.Tensor],
            eig_store: Optional[torch.Tensor],
            stochastic: bool,
            rng: np.random.Generator,
    ) -> pd.DataFrame:
        """Same core algorithm with residual‑stats capture."""
        predictions = pd.DataFrame(index=indices, columns=embedding.block.columns, dtype=float)

        current_time, prediction_time = indices[0]
        if leave_out:
            lib_times = embedding.library_times[~embedding.library_times.isin(indices.droplevel(0))]
        else:
            lib_times = embedding.library_times[embedding.library_times <= current_time]
        valid = lib_times.intersection(embedding.block.index)
        block = embedding.block.loc[valid]
        X_full = block.iloc[:-step_size]
        Y_full = block.iloc[step_size:]

        x = point0.copy()
        for j in range(steps):
            dist = self.norm.distance_matrix(embedding, x[None, :], valid)[:-step_size]
            w = self.kernel.weigh(dist)
            WX, Wy = w * X_full.values, w * Y_full.values
            C = np.linalg.lstsq(WX, Wy, rcond=None)[0]
            # residuals and stats -------------------------------------
            resid = Wy - WX @ C
            Sigma, mu, eigvals = self._local_stats_from_weighted(rids=resid)
            if cov_store is not None:
                cov_store[j] = torch.as_tensor(Sigma)
            if mu_store is not None:
                mu_store[j] = torch.as_tensor(mu)
            if eig_store is not None:
                eig_store[j] = torch.as_tensor(eigvals)
            # forecast -------------------------------------------------
            x_next = x @ C
            if stochastic:
                noise_ = rng.multivariate_normal(mean=np.zeros_like(x_next), cov=Sigma)
                x_next += noise_
            predictions.loc[(current_time, prediction_time)] = x_next
            if coeff_store is not None:
                coeff_store[j] = torch.as_tensor(C)
            # advance --------------------------------------------------
            if j < steps - 1:
                x = x_next
                prediction_time = indices[j + 1][-1]
                X_full = X_full.iloc[1:]
                Y_full = Y_full.iloc[1:]
                valid = valid[1:]
        return predictions

    # ------------------------------------------------------------------
    def _local_stats_from_weighted(
            self,
            *,
            rids: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute (Sigma, mu, eigvals) using the *current* residual_kernel weights."""
        rnorms = np.linalg.norm(rids, axis=1)
        w_resid = self.residual_kernel.weigh(rnorms)
        W_resid = w_resid.sum()
        mu = np.average(rids, axis=0, weights=w_resid)
        r_center = rids - mu[None, :]
        Sigma = (r_center * w_resid[:, None]).T @ r_center / (W_resid + 1e-12)
        eigvals = np.linalg.eigvalsh(Sigma)
        return Sigma, mu, eigvals


