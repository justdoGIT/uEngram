"""Window features -> whitened latent z in R^d for the uEngram engine (numpy).

Pipeline, fitted on a commissioning run declared healthy:

1. Thermal residual. A one-pole RC model C dT/dt = P - (T - T_amb)/R with
   power constant inside each window of length K gives *exactly*, for window
   means,  T_w = c0 + c1 T_{w-1} + c2 P_w + c3 P_{w-1}  with
   c1 = rho = exp(-K/tau), c2 = (1-g)R, c3 = (g-rho)R, c0 = (1-rho)T_amb,
   g = (tau/K)(1-rho). Steady-state gain (c2+c3)/(1-c1) = R. Fitted by trimmed
   least squares; the feature is T_w minus the prediction. A window whose
   residual exceeds `reject_k` robust scales does not become context for the
   next one (innovation gating), so one hot window does not pull the model.
2. Robust standardisation (median, IQR/1.349) with the scale floored at each
   feature's smallest meaningful change, and median imputation of missing values.
3. PCA to d-1 dimensions (Hotelling subspace) plus one coordinate for the
   squared prediction error Q = |x - x_hat|^2 of the discarded subspace,
   taken as sqrt(Q) and robustly standardised. sqrt keeps this coordinate
   linear in the deviation, like the PCA coordinates, so D^2 grows as the
   square of a deviation in either subspace (a cube root, Wilson-Hilferty,
   is closer to Gaussian under the null but grows only as deviation^(2/3) and
   let a memory thrash on an idle node score below its power change); under
   the null sqrt(2Q/g) is close to N(sqrt(2h-1), 1) (Fisher), and the alarm
   threshold is calibrated conformally anyway. Without Q, a change confined
   to directions that were quiet during commissioning would be invisible.
4. Noise whitening with the robust within-state noise covariance from
   consecutive-window differences plus the resolution floor, so correlated
   features count once and the engine's diagonal Gaussian math applies.
"""
from __future__ import annotations

import numpy as np

from ..encoders import Whitener
from ..stats import robust_noise_cov, robust_noise_sigma
from . import schema
from .features import WindowBatch

TRIM_ITERS, TRIM_K, MAD_K = 6, 4.0, 1.4826
SPE_FLOOR = 0.1


class ThermalModel:
    def __init__(self, K: int, reject_k: float = 5.0):
        self.K, self.reject_k = K, reject_k

    def _pairs(self, seq: WindowBatch):
        ok = (np.diff(seq.t0) == self.K) & np.isfinite(seq.temp_mean[1:]) & np.isfinite(seq.temp_mean[:-1]) \
            & np.isfinite(seq.power_mean[1:]) & np.isfinite(seq.power_mean[:-1])
        X = np.column_stack([np.ones(ok.sum()), seq.temp_mean[:-1][ok], seq.power_mean[1:][ok],
                             seq.power_mean[:-1][ok]])
        return X, seq.temp_mean[1:][ok]

    def fit(self, seqs: list[WindowBatch]) -> "ThermalModel":
        parts = [self._pairs(s) for s in seqs]
        X = np.vstack([p[0] for p in parts])
        y = np.concatenate([p[1] for p in parts])
        if len(y) < 20:
            raise ValueError("too few consecutive windows to fit the thermal model")
        keep = np.ones(len(y), bool)
        for _ in range(TRIM_ITERS):
            c, *_ = np.linalg.lstsq(X[keep], y[keep], rcond=None)
            r = y - X @ c
            scale = MAD_K * np.median(np.abs(r[keep]))
            keep = np.abs(r) < TRIM_K * max(scale, 1e-9)
        self.c, self.rscale = c, max(scale, 1e-3)
        return self

    def params(self) -> dict:
        c0, c1, c2, c3 = self.c
        out = {"c": [float(v) for v in self.c], "rscale_c": float(self.rscale)}
        if 0.0 < c1 < 1.0:
            out.update(tau_s=float(-self.K / np.log(c1)), R_c_per_w=float((c2 + c3) / (1 - c1)),
                       T_amb_c=float(c0 / (1 - c1)))
        return out

    def residuals(self, seq: WindowBatch, state):
        """state = (t0, T context, P) of the previous window, or None."""
        r = np.full(len(seq), np.nan)
        for i in range(len(seq)):
            T, P = seq.temp_mean[i], seq.power_mean[i]
            ctx = T
            if state is not None and seq.t0[i] - state[0] == self.K and np.isfinite(T) and np.isfinite(P):
                pred = self.c @ np.array([1.0, state[1], P, state[2]])
                r[i] = T - pred
                if abs(r[i]) > self.reject_k * self.rscale:
                    ctx = pred
            state = (seq.t0[i], ctx, P)
        return r, state


class HostEncoder:
    def __init__(self, d: int = 16, K: int = 32):
        if d % 2:
            raise ValueError("d must be even for the exact chi-square tail")
        self.d, self.K = d, K
        self.names = schema.feature_names()
        self.modality = np.array(schema.feature_modality())
        self.i_thermal = self.names.index("thermal_residual")
        self.floors = np.array(schema.feature_floors())

    # -- fit ---------------------------------------------------------------
    def fit(self, seqs: list[WindowBatch]) -> "HostEncoder":
        self.thermal = ThermalModel(self.K).fit(seqs)
        F = np.vstack([self._with_thermal(s, None)[0] for s in seqs])
        groups = np.concatenate([np.full(len(s), i) for i, s in enumerate(seqs)])
        self.med = np.nanmedian(F, axis=0)
        q75, q25 = np.nanpercentile(F, [75, 25], axis=0)
        self.scale = np.maximum((q75 - q25) / 1.349, self.floors)
        X, _ = self.standardize(F)
        self.pca = Whitener().fit(X, self.d - 1)
        Zp = self.pca.transform(X)
        q = np.sqrt(self._spe(X))
        self.q_med = float(np.median(q))
        self.q_scale = max(MAD_K * float(np.median(np.abs(q - self.q_med))), 1e-6)
        Y = np.column_stack([Zp, (q - self.q_med) / self.q_scale])
        P = self.pca.V / np.sqrt(self.pca.lam)  # x-space -> PC scores
        floor = np.zeros((self.d, self.d))
        floor[:-1, :-1] = P.T @ np.diag((self.floors / self.scale) ** 2) @ P
        floor[-1, -1] = SPE_FLOOR ** 2
        lam, U = np.linalg.eigh(robust_noise_cov(Y, 1, groups, floor=floor))
        self.W = (U / np.sqrt(lam)) @ U.T
        self.W_inv = (U * np.sqrt(lam)) @ U.T
        Z = Y @ self.W
        self.sigma0_emp = robust_noise_sigma(Z, 1, groups)
        # Whitening declared the noise (robust + floor) to be the identity, so
        # sigma0 is 1 by construction; a larger empirical value is kept.
        self.sigma0 = np.maximum(self.sigma0_emp, 1.0)
        tot = float(np.sum(np.var(X, axis=0, ddof=1)))  # same normalisation as np.cov in the PCA
        self.fit_report = {
            "windows": int(len(F)), "features": len(self.names),
            "pca_var_share": float(np.sum(self.pca.lam) / tot) if tot > 0 else float("nan"),
            "imputed_share": float(np.mean(np.isnan(F))), "thermal": self.thermal.params(),
            "sigma0_emp_median": float(np.median(self.sigma0_emp)),
        }
        return self

    def _with_thermal(self, seq: WindowBatch, state):
        r, state = self.thermal.residuals(seq, state)
        F = seq.F.copy()
        F[:, self.i_thermal] = r
        return F, state

    def standardize(self, F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = (F - self.med) / self.scale
        miss = ~np.isfinite(X)
        X[miss] = 0.0  # training median
        return X, miss.sum(axis=1)

    def _spe(self, X: np.ndarray) -> np.ndarray:
        Xc = X - self.pca.m
        e = Xc - (Xc @ self.pca.V) @ self.pca.V.T
        return np.sum(e * e, axis=1)

    # -- encode ------------------------------------------------------------
    def encode(self, seq: WindowBatch, state=None):
        """-> (Z [W,d], X [W,p] standardised features, n_imputed [W], new thermal state)."""
        F, state = self._with_thermal(seq, state)
        X, n_imp = self.standardize(F)
        q = np.sqrt(self._spe(X))
        Y = np.column_stack([self.pca.transform(X), (q - self.q_med) / self.q_scale])
        return Y @ self.W, X, n_imp, state

    def attribution(self, x: np.ndarray, z: np.ndarray, mu: np.ndarray, var: np.ndarray) -> np.ndarray:
        """Per-feature contributions to D^2 = sum (z-mu)^2/var; they sum to D^2.

        Partial decomposition through the linear encoder: with g = W (z-mu)/var
        in pre-whitening coordinates, a PCA coordinate's term is split over
        features by x - x_mu (x_mu = the slot mean's preimage in the PCA
        subspace), and the Q coordinate's term in proportion to each
        feature's squared residual. Single terms can be negative.
        """
        gz = (z - mu) / var
        dY = (z - mu) @ self.W_inv
        gY = self.W @ gz
        mu_pc = (mu @ self.W_inv)[:-1]
        dx = (x - self.pca.m) - (mu_pc * np.sqrt(self.pca.lam)) @ self.pca.V.T
        P = self.pca.V / np.sqrt(self.pca.lam)
        c = dx * (P @ gY[:-1])
        xc = x - self.pca.m
        e2 = (xc - (xc @ self.pca.V) @ self.pca.V.T) ** 2
        c_spe = dY[-1] * gY[-1]
        return c + c_spe * (e2 / e2.sum() if e2.sum() > 0 else np.full(len(x), 1.0 / len(x)))
