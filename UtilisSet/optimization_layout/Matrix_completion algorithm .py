import numpy as np
import matplotlib.pyplot as plt

def update_biases(X, U, V, bu, bv, beta, mask):
    for i in range(len(bu)):
        obs = mask[i, :]
        if np.any(obs):
            pred = U[i, :] @ V[obs, :].T + bv[obs]
            bu[i] = (np.sum(X[i, obs] - pred)) / (np.sum(obs) + beta)

    for j in range(len(bv)):
        obs = mask[:, j]
        if np.any(obs):
            pred = U[obs, :] @ V[j, :].T + bu[obs]
            bv[j] = (np.sum(X[obs, j] - pred)) / (np.sum(obs) + beta)
    return bu, bv

def nmf_objective(
        X: np.ndarray,
        U: np.ndarray,
        V: np.ndarray,
        mask= None,
) -> float:
    if mask is None:
        mask = ~np.isnan(X)
    X_obs = np.where(mask, X, 0.0)

    recon = U @ V.T
    resid = np.where(mask, X_obs - recon, 0.0)
    data_term = np.sum(resid ** 2)

    return data_term

def als_nmf_with_bias(
    X, k=5, mask=None,
    max_iter=100, tol=1e-3,
    alpha=1e-3,
    beta=1e-3,
    random_state=0,
):
    rng = np.random.default_rng(random_state)
    M, N = X.shape
    if mask is None:
        mask = ~np.isnan(X)
    X_filled = np.where(mask, X, 0.0)

    # 初始化
    U = rng.random((M, k))
    V = rng.random((N, k))
    bu = np.zeros(M)
    bv = np.zeros(N)
    mu = np.nanmean(np.where(mask, X, np.nan))

    Ik = np.eye(k)

    def proj_nonneg(a):
        a[a < 0] = 0.0
        return a

    for it in range(max_iter):
        for j in range(N):
            Oj = mask[:, j]
            if not np.any(Oj):
                continue
            Xj = X[Oj, j]
            Uj = U[Oj, :]
            A = Uj.T @ Uj
            b = Uj.T @ Xj

            lambda_reg = 1e-4
            A.flat[::A.shape[0] + 1] += lambda_reg

            vj = np.linalg.solve(A, b)
            V[j, :] = proj_nonneg(vj)

        for i in range(M):
            Oi = mask[i, :]
            if not np.any(Oi):
                continue
            Xi = X[i, Oi]
            Vi = V[Oi, :]
            A = Vi.T @ Vi
            b = Vi.T @ Xi

            lambda_reg = 1e-4
            A.flat[::A.shape[0] + 1] += lambda_reg

            ui = np.linalg.solve(A, b)
            U[i, :] = proj_nonneg(ui)

        data_term = nmf_objective(X, U, V)
        reg_uv = 0.5 * alpha * (np.sum(U ** 2) + np.sum(V ** 2))
        reg_bias = 0.5 * beta * (np.sum(bv**2) +np.sum(bu**2))
        loss = data_term
        bu,bv = update_biases(X,U,V,bu,bv,beta,mask)
        # print(f"iter: {it}, loss: {loss:.3f}")
        if loss < tol:
            break
    return U, V

if __name__ == '__main__':

    rng = np.random.default_rng(42)
    M, N, k = 80, 60, 5
    U_true = rng.random((M, k))
    V_true = rng.random((N, k))
    X_full = U_true @ V_true.T

    monitored = rng.choice(N, size=10, replace=False)
    W = np.zeros_like(X_full, dtype=float)
    W[:, monitored] = 1.0
    X_obs_nan = X_full.copy()
    X_obs_nan[W == 0] = np.nan

    U_hat, V_hat = als_nmf_with_bias(X_obs_nan,k=k, max_iter=300)

