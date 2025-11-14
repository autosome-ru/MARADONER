from time import time
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import pandas as pd
from tqdm import tqdm
import scipy.linalg.lapack as lapack
import numpy as np
import scipy.stats as st
from collections import defaultdict
from sklearn.model_selection import RepeatedKFold
from koptimizer import KOptimizer
from jax import config
from os import environ
import jax.numpy as jnp


from jax.numpy.linalg import eigh, svd



def chol_inv(x: np.array):
    """
    Calculate invserse of matrix using Cholesky decomposition.

    Parameters
    ----------
    x : np.array
        Data with columns as variables and rows as observations.

    Raises
    ------
    np.linalg.LinAlgError
        Rises when matrix is either ill-posed or not PD.

    Returns
    -------
    c : np.ndarray
        x^(-1).

    """
    c, info = lapack.dpotrf(x)
    if info:
        raise np.linalg.LinAlgError
    lapack.dpotri(c, overwrite_c=1)
    inds = np.tri(len(c), k=-1, dtype=bool)
    c[inds] = c.T[inds]
    return c


def lowrank_decomposition(X: np.ndarray, rel_eps=1e-10):
    q, s, v = [np.array(t) for t in svd(X)]
    max_sv = max(s)
    n = len(s)
    for r in range(n):
        if s[r] / max_sv < rel_eps:
            r -= 1
            break
    r += 1
    s = s[:r] ** 0.5
    null_q = q[:, r:]
    null_v = v[r:]
    q = q[:, :r]
    v = v[:r]
    return (q, s ** 2, v), (null_q, null_v)


def _nullspace_eigh(X):
    if type(X) not in (tuple, list):
        X = jnp.array(X)
        P0 = jnp.identity(X.shape[0], dtype=float) - X @ jnp.linalg.inv(X.T @ X) @ X.T
    else:
        Q = X[0]
        Q = jnp.array(Q)
        I = jnp.identity(Q.shape[0], dtype=float)
        P0 = I - Q @ Q.T
    return np.array(eigh(P0))


def estimate_groupwise_variance(Y: list[np.ndarray], B: np.ndarray, groups: list[tuple]) -> np.ndarray:
    """
    Estimate the diagonal variance matrix using the Restricted Maximum Likelihood method.

    Parameters
    ----------
    Y : List[np.ndarray]
        List of size g (number of groups), where each item is an array of shape (p, n), where p is a number of promoters and n is a number of samples.
    B : np.ndarray or tuple[np.ndarray, np.ndarray, np.ndarray]
        Loadings matrix or an SVD decomposition as returned by lowrank decomposition.

    Returns
    -------
    var_reml : np.ndarray
        Array of estimated variances.
    """
    if type(B) in (jnp.ndarray, np.ndarray):
        eigs, q = _nullspace_eigh(B)
        for j in range(len(eigs)):
            if eigs[j] > 0.8:
                break
        P = q[:, j:].T
    else:
        P = B[0].T
    P = jnp.array(P)
    Y = jnp.array(Y)
    var_reml = jnp.array([jnp.mean((P @ Y.at[..., inds].get()) ** 2) for inds in groups])
    return var_reml


def _estimate_transform_matrices(B_svd: tuple[np.ndarray, np.ndarray, np.ndarray], U_mults: np.ndarray):
    res = list()
    Q, D, V = B_svd
    dvt = jnp.array(D.reshape(-1, 1) * V)
    for mult in tqdm(U_mults):
        mult = jnp.array(mult)
        mx = dvt * mult ** 2 @ dvt.T
        qdv, _ = lowrank_decomposition(mx, rel_eps=1e-12)
        res.append((qdv[0].T, qdv[1]))
    return res


def estimate_sigma(Y: list[np.ndarray], B_svd: tuple[np.ndarray, np.ndarray, np.ndarray], variances: np.ndarray, U_mults: np.ndarray,
                   return_transformed=True, per_group=True, groups=None) -> np.ndarray:
    if not per_group and groups is not None:
        n = sum(map(len, groups))
        var = np.empty(n, dtype=float)
        for inds, v in zip(groups, variances):
            var[inds] = v
        variances = var
    if len(U_mults) < Y.shape[1]:
        n = sum(map(len, groups))
        mults = np.empty((n, U_mults.shape[-1]), dtype=float)
        for inds, mult in zip(groups, U_mults):
            mults[inds] = mult
        U_mults = mults
    P, eigs, _ = B_svd
    Yt = Y.T @ P
    trs = _estimate_transform_matrices(B_svd, U_mults)
    Yt = [P @ Y[..., np.newaxis] for Y, (P, _) in zip(Yt, trs)]
    eigs = [eigs for _, eigs in trs]
    aux = Yt
    bounds = (0.0, variances.max() * 10)

    def loglik(Y: list[jnp.ndarray], sigma: float, S: jnp.ndarray, eigs: list[jnp.ndarray]):
        loglik = 0
        for Y, sigma_g, eigs in zip(Y, S, eigs):
            R = sigma * eigs + sigma_g
            Y = (R.reshape(-1, 1)) ** (-0.5) * Y
            loglik += -jnp.einsum('ij,ij->', Y, Y)
            loglik += -Y.shape[1] * jnp.log(R).sum()
        return loglik

    if not per_group:
        def f(sigma): return -loglik(Yt, sigma, variances, eigs)
        r = minimize_scalar(f)
        return (r.x, aux) if return_transformed else r.x

    if groups is None:
        res = np.empty(Y.shape[1])
        for i in tqdm(list(range(Y.shape[1]))):
            def f(sigma): return -loglik(Yt[i:i + 1], sigma, variances[i:i + 1], eigs[i:i+1])
            r = minimize_scalar(f, bounds=bounds)
            res[i] = r.x
    else:
        res = np.empty(len(groups))
        for i in tqdm(list(range(len(res)))):
            inds = groups[i]
            def f(sigma): return -loglik([jnp.array(Yt[j]) for j in inds], sigma, jnp.array(variances[i:i + 1].repeat(len(inds))), [eigs[j] for j in inds])
            r = minimize_scalar(f, bounds=bounds)
            res[i] = r.x

    return (res, aux) if return_transformed else res


def estimate_u_map(Y, B_svd, variances, sigmas: np.ndarray, U_mults: np.ndarray, tau=1, groups=None,
                   k_fold=4, n_rep=1, B_orig=None):
    if groups is not None:
        Yt = np.empty((len(groups), len(Y)), dtype=float)
        U_mults_t = np.empty((len(groups), U_mults.shape[-1]), dtype=float)
        n = np.empty((len(groups),), dtype=float)
        for i, inds in enumerate(groups):
            n[i] = len(inds)
            Yt[i] = Y[:, inds].sum(axis=-1)
            U_mults_t[i] = U_mults[inds].mean(axis=0)
        U_mults = U_mults_t
        Y = Yt
    if len(U_mults) > Y.shape[1]:
        n = sum(map(len, groups))
        mults = np.empty((n, U_mults.shape[-1]), dtype=float)
        for inds, mult in zip(groups, U_mults):
            mults[inds] = mult
        U_mults = mults
    Q, D, V = [jnp.array(t) for t in B_svd]
    Vt = V.T * D
    BB = Vt @ Vt.T
    inds = jnp.diag_indices_from(BB)

    def _est(Y, Q, BB, tau: float) -> np.ndarray:
        nonlocal n
        Y = (Y @ Q) @ Vt.T
        U = list()
        U_std = list()
        stds = list()
        stds2 = list()
        for Y, sigma_g, sigma, mult, nt in zip(Y, variances, sigmas, U_mults, n):
            ratio = jnp.exp(jnp.log(sigma_g) - jnp.log(sigma) - jnp.log(tau))
            cov = nt * BB 
            cov = cov.at[inds].add(ratio * mult ** -2)
            cov = cov / sigma_g
            cov = jnp.linalg.pinv(cov, hermitian=True)
            U.append(cov @ Y[..., np.newaxis] / sigma_g )
            stds.append(cov.diagonal() ** 0.5)
            s = stds[-1]
            cor = (1/s) * cov * (1/s).reshape(-1,1)
            d, q = jnp.linalg.eigh(cor)
            p = q * (1/(d ** 0.5)) @ q.T
            W = p * (1/s).reshape(1,-1)

            U_std.append(W@U[-1])
            cov2 = cov @ ((BB * mult.reshape(1,-1) ** 2 @ BB) / sigma + BB / sigma_g) @ cov 
            stds2.append(cov2.diagonal() ** 0.5)
        return np.concatenate(U, axis=1), np.concatenate(U_std, axis=1), np.array(stds).T, np.array(stds2).T
    return _est(Y, Q, BB, tau)


def preprocess_data(Y, B, U_mult, std_y=False, std_b=False, b_cutoff=1e-9, weights=None, _means_est=False):
    if std_y:
        Y = Y / Y.std(axis=0, keepdims=True)
    n_samples = Y.shape[1]
    if _means_est:
        Ym = Y - Y.mean(axis=0, keepdims=True)
    if n_samples < 3:
        Y = Y - Y.mean(axis=0, keepdims=True)
    else:
        Y = Y - Y.mean(axis=0, keepdims=True) - Y.mean(axis=1, keepdims=True) + Y.mean()
    B = B.copy()
    B[B < b_cutoff] = 0.0
    min_b, max_b = B.min(axis=0, keepdims=True), B.max(axis=0, keepdims=True)
    inds = ((max_b - min_b) > 1e-4).flatten()
    inds[:] = True
    B = B[:, inds]
    min_b = min_b[:, inds]
    max_b = max_b[:, inds]
    U_mult = U_mult[:, inds]
    # B -= B.mean(axis=0, keepdims=True)
    std1 = B.std(axis=1, keepdims=True)
    if std_b:
        B /= std1
    if weights is not None:
        weights = weights.reshape(-1, 1) ** -0.5
        Y = weights * Y
        B = weights * B
    if _means_est:
        Y = (Y, Ym)
    return Y, B, U_mult, inds



