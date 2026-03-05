import numpy as np
from UtilisSet.optimization_layout.waterlevel_nmf_completion import reconstruct_from_mask

# Synthetic demo
T, N = 120, 50
rng = np.random.default_rng(0)
U = rng.random((T, 4))
V = rng.random((N, 4))
X_full = U @ V.T

# Choose monitored nodes (columns) fully observed
monitored = rng.choice(N, size=10, replace=False)
W = np.zeros_like(X_full, dtype=float)
W[:, monitored] = 1.0
X_obs_nan = X_full.copy()
X_obs_nan[W == 0] = np.nan

X_hat, metrics, res = reconstruct_from_mask(X_obs_nan, rank=5, alpha=1e-2, n_restarts=3, max_iter=2000, tol=1e-6)
print("Reconstruction shape:", X_hat.shape)
print("Observed RMSE:", metrics["RMSE_observed"])
print("Best seed:", res.seed, "Iters:", res.n_iter)
