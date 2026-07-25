import numpy as np
import jax.numpy as jnp
import jax
import scipy.linalg.lapack as lapack
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from dataclasses import dataclass, replace
from functools import partial
from .meta_optimizer import MetaOptimizer
from sklearn.model_selection import RepeatedKFold
from collections import defaultdict
from enum import Enum
from .utils import read_init, ProjectData, logger_print, openers
import dill
from .linalg import lowrank_decomposition, ones_nullspace, ones_nullspace_transform, \
                    ones_nullspace_transform_transpose, null_space_transform, LowrankDecomposition
from .loading_context import estimate_context, LoadingContext, LoadingMultipliers, estimate_multipliers
from .drist import DRIST
from . import lowrank as _lowrank


def _error_lowrank_W(error_variance):
    """Return the (s, k) low-rank factor W of the error covariance, or None if absent.
    Defensive against old pickles that predate the `lowrank` field."""
    W = getattr(error_variance, 'lowrank', None)
    if W is None:
        return None
    W = np.asarray(W)
    return W if W.ndim == 2 and W.shape[1] > 0 else None


def _activity_lowrank_V(motif_variance):
    """Return the (s, k) low-rank factor V of the activity covariance, or None if absent."""
    V = getattr(motif_variance, 'lowrank', None)
    if V is None:
        return None
    V = np.asarray(V)
    return V if V.ndim == 2 and V.shape[1] > 0 else None


def error_covariance(error_variance, group_inds_inv) -> np.ndarray:
    """Dense per-sample error covariance Theta = diag(D[group]) + W W^T (s x s)."""
    d = np.asarray(error_variance.variance)[group_inds_inv]
    Theta = np.diag(d)
    W = _error_lowrank_W(error_variance)
    if W is not None:
        Theta = Theta + W @ W.T
    return Theta

class GOFStat(str, Enum):
    fov = 'fov'
    corr = 'corr'

class GOFStatMode(str, Enum):
    residual = 'residual'
    total = 'total'



@dataclass
class TransformedData:
    Y: np.ndarray
    B: np.ndarray
    group_inds: list
    group_inds_inv: np.ndarray
    K : np.ndarray = None
    L: np.ndarray = None
    # Covariate design matrix, shape (c, s). First row is the intercept (ones).
    X: np.ndarray = None


def covariate_projection(D_n: jnp.ndarray, X: jnp.ndarray) -> jnp.ndarray:
    """Build the projection-aware inverse used in the Sigma/G REML step:

        H_X^T (H_X D H_X^T)^{-1} H_X  ==  D^{-1} - D^{-1} X^T (X D^{-1} X^T)^{-1} X D^{-1},

    where H_X is any semi-orthogonal complement of the row space of X (which always
    contains 1_s as its intercept row). Projecting out X simultaneously removes the
    promoter-mean term mu_p 1_s^T (via 1_s) and the covariate mean term B M X (via
    the full row space of X). When X is just the intercept row this reduces exactly
    to the classic D^{-1} - D^{-1} 1 1^T D^{-1} / (1^T D^{-1} 1) construction.

    Parameters
    ----------
    D_n : (s,) sample-wise error variances (already expanded over samples).
    X   : (c, s) covariate design matrix.
    """
    X = jnp.asarray(X, dtype=D_n.dtype)
    Di = 1.0 / D_n                      # (s,) = diag(D^{-1})
    XDi = X * Di                        # (c, s) = X D^{-1}
    XDiX = XDi @ X.T                    # (c, c) = X D^{-1} X^T
    return jnp.diag(Di) - XDi.T @ jnp.linalg.solve(XDiX, XDi)


def variance_dof(X, group_inds) -> dict:
    """Degrees-of-freedom budget for the (Sigma, G) REML step.

    That step estimates the group variances nu from the sample-space residual left
    after projecting out the covariate row space X (which always contains the
    intercept 1_s). The projection consumes rank(X) of the s sample dimensions, so
    only ``dof = s - rank(X)`` remain, while ``g - 1`` group variances are free (one
    is pinned to fix the Sigma/G scale indeterminacy). When ``g - 1 > dof`` those
    variances are not identifiable and the Fisher information is singular.

    Returns a dict with the budget and an ``ok`` flag.
    """
    X = np.asarray(X, dtype=float)
    s = X.shape[1]
    rank_x = int(np.linalg.matrix_rank(X)) if X.size else 0
    dof = s - rank_x
    g = len(group_inds)
    n_free = max(g - 1, 0)
    return {'n_samples': s, 'n_covariates': X.shape[0], 'rank_x': rank_x,
            'dof': dof, 'n_groups': g, 'n_free_nu': n_free, 'ok': n_free <= dof}


def warn_variance_dof(X, group_inds, verbose: bool = True) -> dict:
    """Emit an explicit warning when the variance parameters are over-determined.

    Prints an actionable diagnosis instead of leaving the user to decode a singular
    Fisher information matrix later on (an indefinite FIM here surfaces downstream as
    "Failed to compute inverse using Cholesky decomposition" during ``export``).
    """
    info = variance_dof(X, group_inds)
    if info['ok']:
        return info
    n_cov = info['n_covariates'] - 1                      # excluding the intercept
    logger_print(
        '\n[warning] Degrees-of-freedom problem: the group variances are not identifiable.\n'
        f'    samples                              : {info["n_samples"]}\n'
        f'    covariate design rank                : {info["rank_x"]} '
        f'(intercept + {n_cov} covariate(s))\n'
        f'    residual sample d.o.f. (s - rank)    : {info["dof"]}\n'
        f'    free group variances to estimate     : {info["n_free_nu"]} '
        f'({info["n_groups"]} groups, one pinned for scale)\n'
        f'  Estimating {info["n_free_nu"]} group variances needs at least that many residual '
        f'dimensions, but only {info["dof"]} remain.\n'
        '  Consequences: the Fisher information for (Sigma, nu) is singular, so nu estimates and\n'
        '  their standard errors (params/group_variances.tsv) are unreliable, and "maradoner export"\n'
        '  will report failures to invert it and fall back to a pseudo-inverse.\n'
        '  Remedies: supply --sample-groups so samples are pooled into fewer, larger groups '
        '(replicates\n'
        '  sharing a variance), and/or reduce the number of covariates passed to "create".\n',
        verbose)
    return info


def motif_mean_matrix(motif_mean, X) -> np.ndarray:
    """Per-sample motif-activity mean in motif space: M @ X, shape (m, s).

    Works for both the legacy 1-D mu_m (treated as an intercept-only M) and the
    covariate matrix M of shape (m, c)."""
    M = np.asarray(motif_mean.mean if hasattr(motif_mean, 'mean') else motif_mean)
    if M.ndim == 1:
        M = M.reshape(-1, 1)
    return M @ np.asarray(X)
    
@dataclass(frozen=True)
class ErrorVarianceEstimates:
    variance: np.ndarray
    promotor: np.ndarray
    fim: np.ndarray
    loglik: float
    loglik_start: float
    # Optional low-rank correction to the error column-covariance: the full error
    # covariance is Theta = diag(variance[group]) + lowrank @ lowrank.T, with `lowrank`
    # of shape (s, k).  None / k == 0 recovers the classic diagonal group-wise D.
    lowrank: np.ndarray = None

@dataclass(frozen=True)
class MotifVarianceEstimates:
    motif: np.ndarray
    group: np.ndarray
    fim: np.ndarray
    fixed_group: int
    loglik: float
    loglik_start: float
    # Optional low-rank correction to the activity column-covariance: the full activity
    # covariance is G = diag(group[group]) + lowrank @ lowrank.T, with `lowrank` of shape
    # (s, k).  None / k == 0 recovers the classic diagonal group-wise G.
    lowrank: np.ndarray = None

@dataclass(frozen=True)
class MotifMeanEstimates:
    mean: np.ndarray
    fim: np.ndarray

@dataclass(frozen=True)
class PromoterMeanEstimates:
    mean: np.ndarray

@dataclass(frozen=True)
class SampleMeanEstimates:
    mean: np.ndarray

@dataclass(frozen=True)
class FitResult:
    error_variance: ErrorVarianceEstimates
    motif_variance: MotifVarianceEstimates 
    motif_mean: MotifMeanEstimates 
    promoter_mean: PromoterMeanEstimates
    sample_mean: SampleMeanEstimates
    group_names: list
    clustering: np.ndarray = None
    clustered_B: np.ndarray = None
    promoter_inds_to_drop: list = None
    loading_context: LoadingContext = None
    drist: DRIST = None
    loading_multipliers: LoadingMultipliers = None
    


def transform_data(data, std_y=False, helmert=True, weights=None,
                   loading_context: LoadingContext = None, drist: DRIST = None,
                   loading_multipliers: LoadingMultipliers = None) -> TransformedData:
    try:
        B = data.B_orig
        Y = data.Y_orig
        group_inds = data.original_inds
    except:
        B = data.B
        Y = data.Y
        group_inds = data.group_inds
    if drist is not None:
        B = drist.transform(B)
    if loading_context is not None:
        B = loading_context.apply_correction(B, B, L=data.L)
    if loading_multipliers is not None:
        B = loading_multipliers.apply_correction(B)
    if weights is not None:
        B = B * weights.reshape(-1, 1)
        Y = Y * weights.reshape(-1, 1)
        if weights.std()  == 0:
            weights = None
    if helmert:
        if weights is None:
            # F_p = ones_nullspace(len(Y))
            # Y = F_p @ Y
            # B = F_p @ B
            Y = ones_nullspace_transform(Y)
            B = ones_nullspace_transform(B)
        else:
            weights = weights.reshape(-1, 1)
            weights /= np.linalg.norm(weights)
            Y = null_space_transform(weights, Y)
            B = null_space_transform(weights, B)
    group_inds_inv = list()
    d = dict()
    for i, items in enumerate(group_inds):
        for j in items:
            d[j] = i
    for i in sorted(d.keys()):
        group_inds_inv.append(d[i])
    group_inds_inv = np.array(group_inds_inv)
    X = getattr(data, 'X', None)
    if X is None:
        X = np.ones((1, Y.shape[1]), dtype=float)
    return TransformedData(Y=Y, B=B,
                           group_inds=group_inds,
                           group_inds_inv=group_inds_inv,
                           L=data.L,
                           X=np.asarray(X, dtype=float))

def loglik_error(d: jnp.ndarray, Qn_Y: jnp.ndarray, group_inds_inv: jnp.ndarray) -> float:
    d = d.at[group_inds_inv].get()
    logdet_D = jnp.log(d).sum()
    d = 1 / d
    logdet_FDF = logdet_D + jnp.log(d.mean())
    K = d * d.sum()
    xi = jnp.exp(logdet_D - logdet_FDF - jnp.log(len(d)))
    Y1 = Qn_Y * K
    Y2 = Qn_Y @ d.reshape(-1, 1)
    m = len(Qn_Y)
    return xi * (jnp.einsum('ij,ij->', Y1, Qn_Y) - (Y2.T @ Y2).flatten()[0]) + m * logdet_FDF

def loglik_error_full(x: jnp.ndarray, Y: jnp.ndarray, Q_C: jnp.ndarray, group_inds_inv: jnp.ndarray, D_fix_val: float, D_fix_ind: int) -> float:
    p, s = Y.shape
    r = Q_C.shape[1]
    x = x ** 2
    D = x.at[:-p].get() + 1e-4
    if D_fix_ind is not None:
        D = jnp.insert(D, D_fix_ind, D_fix_val)
    S = x.at[-p:].get() + 1e-3
    S = 1 / S
    D = 1 / D
    D = D.at[group_inds_inv].get()
    w_D = D.sum()
    M = Q_C.T * S @ Q_C
    YD = Y * D
    YS = Y.T * S
    YD = YD - jnp.outer(YD.sum(axis=-1), D / w_D)
    YS = YS - YS @ Q_C @ jnp.linalg.inv(M) @ Q_C.T * S
    vec = jnp.einsum('ij,ji->', YD, YS)
    logdet_D = (p-r) * (-jnp.log(D).sum()  + jnp.log(w_D))
    logdet_S = (s-1) * (jnp.linalg.slogdet(M)[1] - jnp.log(S).sum())
    logdet = logdet_D + logdet_S
    return vec + logdet
    
def loglik_error_grad(d: jnp.ndarray, Qn_Y: jnp.ndarray, group_inds_inv: jnp.ndarray,
                      group_inds: jnp.ndarray) -> jnp.ndarray:
    d = d.at[group_inds_inv].get()
    logdet_D = jnp.log(d).sum()
    d = 1 / d
    logdet_FDF = logdet_D + jnp.log(d.mean())
    K = d * d.sum()
    xi = jnp.exp(logdet_D - logdet_FDF - jnp.log(len(d)))
    Y1 = Qn_Y * K
    Y2 = Qn_Y @ d.reshape(-1, 1) 
    Z = Y1 - Y2 @ d.reshape(1, -1)
    g = len(Qn_Y) * xi * (K - d ** 2) - xi ** 2 * jnp.einsum('ji,ji->i', Z, Z)
    return jnp.array([g[ind].sum() for ind in group_inds])

def loglik_motifs(x: jnp.ndarray, Z: jnp.ndarray, BTB: jnp.ndarray,
                  D_product_inv: jnp.ndarray, group_inds_inv: jnp.ndarray,
                  G_fix_ind=None, G_fix_val=1.0, drop_sigma=False, _motif_zero=None) -> float:
    if drop_sigma:
        x = jnp.append(jnp.ones(len(BTB)), x)
    Sigma = x.at[:len(BTB)].get() ** 0.5
    if _motif_zero is not None:
        Sigma = Sigma.at[_motif_zero].set(0)
    G = x.at[len(BTB):].get()
    if G_fix_ind is not None:
        G = jnp.insert(G, G_fix_ind, G_fix_val)
    G = G ** 0.5
    G = G.at[group_inds_inv].get()
    D_A, Q_A = jnp.linalg.eigh(G.reshape(-1, 1) * D_product_inv * G)
    D_B, Q_B = jnp.linalg.eigh(Sigma.reshape(-1, 1) * BTB * Sigma)
    D_A = jnp.where(D_A > 0, D_A, 0.0)
    D_B = jnp.where(D_B > 0, D_B, 0.0)
    cov = jnp.kron(D_A, D_B) + 1
    logdet = jnp.log(cov).sum()
    v = (Q_B.T * Sigma @ Z * G @ Q_A).flatten('F')
    loglik = -(v ** 2 / cov).sum() + logdet
    return loglik 

def loglik_motifs_grad(x: jnp.ndarray, Z: jnp.ndarray, BTB: jnp.ndarray,
                  D_product_inv: jnp.ndarray, group_inds_inv: jnp.ndarray,
                  group_inds: jnp.ndarray, G_fix_ind=None, G_fix_val=1.0,
                  drop_sigma=False,
                  _motif_zero=None) -> float:
    if drop_sigma:
        x = jnp.append(jnp.ones(len(BTB)), x)
    Sigma = x.at[:len(BTB)].get() ** 0.5
    if _motif_zero is not None:
        Sigma = Sigma.at[_motif_zero].set(0)
    G = x.at[len(BTB):].get()
    if G_fix_ind is not None:
        G = jnp.insert(G, G_fix_ind, G_fix_val)
    G = G ** 0.5
    G = G.at[group_inds_inv].get()
    D_A, Q_A = jnp.linalg.eigh(G.reshape(-1, 1) * D_product_inv * G)
    D_B, Q_B = jnp.linalg.eigh(Sigma.reshape(-1, 1) * BTB * Sigma)
    D_A = jnp.where(D_A > 0, D_A, 0.0)
    D_B = jnp.where(D_B > 0, D_B, 0.0)
    s = 1 / (jnp.kron(D_A, D_B) + 1)
    M = s.reshape((len(Q_B), Q_A.shape[1]), order='F')
    Lambda_base = (Q_B.T * Sigma @ Z * G @ Q_A) * M
    
    # Derivative w.r.t. to tau/Sigma
    Lambda = Q_B @ Lambda_base
    vec_term = jnp.einsum('ij,ij->i', Lambda, Lambda) / (Sigma ** 2)
    Z = (Q_B.T * Sigma @ BTB) * Q_B.T
    det_term = (s.reshape(1, -1) @ jnp.kron(D_A.reshape(-1,1), Z)).flatten() / Sigma
    grad_tau = -vec_term + det_term
    # Derivative w.r.t to nu/G
    Lambda = Lambda_base @ Q_A.T
    vec_term = jnp.einsum('ji,ji->i', Lambda, Lambda) / (G ** 2)
    Z = (Q_A.T * G @ D_product_inv) * Q_A.T
    det_term = (s.reshape(1, -1) @ jnp.kron(Z, D_B.reshape(-1,1))).flatten() / G
    grad_nu = -vec_term + det_term
    grad_nu = jnp.array([grad_nu.at[inds].get().sum() for inds in group_inds])
    if G_fix_ind is not None:
        grad_nu = jnp.delete(grad_nu, G_fix_ind)
    grad = jnp.append(grad_tau, grad_nu)
    if _motif_zero is not None:
        grad = grad.at[_motif_zero].set(0)
    if drop_sigma:
        grad = grad[len(BTB):]
    return grad 

def loglik_motifs_fim(x: jnp.ndarray, BTB: jnp.ndarray, 
                      D_product_inv: jnp.ndarray, group_inds_inv: jnp.ndarray,
                      group_inds: jnp.ndarray, G_fix_ind=None, G_fix_val=1.0,
                      drop_sigma=False) -> float:
    if drop_sigma:
        x = jnp.append(jnp.ones(len(BTB)), x)
    Sigma = x.at[:len(BTB)].get() ** 0.5
    G = x.at[len(BTB):].get()
    if G_fix_ind is not None:
        G = jnp.insert(G, G_fix_ind, G_fix_val)
    G = G ** 0.5
    G = G.at[group_inds_inv].get()
    D_A, Q_A = jnp.linalg.eigh(G.reshape(-1, 1) * D_product_inv * G)
    D_B, Q_B = jnp.linalg.eigh(Sigma.reshape(-1, 1) * BTB * Sigma)
    D_A = jnp.where(D_A > 0, D_A, 0.0)
    D_B = jnp.where(D_B > 0, D_B, 0.0)
    s = 1 / (jnp.kron(D_A, D_B) + 1)
    indices = jnp.arange(len(s), dtype=int)

    indices = len(G) * (indices % len(Sigma)) + indices // len(Sigma)
    s_permuted = s.at[indices].get()
    BTCQ = BTB * Sigma @ Q_B
    D_prod_Q = D_product_inv * G @ Q_A
   
    group_loadings = np.zeros((len(G), len(group_inds)), dtype=int)
    for i, indices in enumerate(group_inds):
        group_loadings[indices, i] = 1
    group_loadings = jnp.array(group_loadings)
    indices = jnp.arange(0, len(s), dtype=int).reshape((len(G), len(Sigma)))
    
    @jax.jit
    def f_tau(k, mx):
        ind = indices.at[k].get()
        S_k = s.at[ind].get()
        Lambda_k = BTCQ * S_k @ Q_B.T
        return mx + D_A.at[k].get() ** 2 * Lambda_k * Lambda_k.T

    FIM_tau = jnp.zeros((len(Sigma), len(Sigma)), dtype=float)
    FIM_tau = jax.lax.fori_loop(0, len(D_A), f_tau, FIM_tau) / 2
    FIM_tau = FIM_tau * jnp.outer(1 / Sigma, 1 / Sigma)
    @jax.jit
    def f_nu(k, mx):
        ind = indices.at[k].get()
        S_k = s_permuted.at[ind].get()
        Gamma_k = D_prod_Q * S_k @ Q_A.T
        return mx + D_B.at[k].get() ** 2 * Gamma_k * Gamma_k.T
    
    indices = indices.reshape(indices.shape[::-1])
    FIM_nu = jnp.zeros((len(G), len(G)), dtype=float)
    FIM_nu = jax.lax.fori_loop(0, len(D_B), f_nu, FIM_nu) / 2
    FIM_nu = FIM_nu * jnp.outer(1 / G, 1 / G)
    FIM_nu = group_loadings.T @ FIM_nu @ group_loadings
    indices = jnp.arange(0, len(s), dtype=int)
    indices_mod = indices % len(Sigma)
    indices_div = indices // len(Sigma)
    indices = jnp.array(list(np.ndindex((len(Sigma), len(G)))))
    zeta = s ** 2 * D_A.at[indices_div].get() * D_B.at[indices_mod].get()
    Psi = BTCQ * Q_B
    K = D_prod_Q
    Theta = K * Q_A 

    @jax.jit
    def f_tau_nu(ind):
        i, j = ind
        psi_i = Psi.at[i, indices_mod].get()
        theta_j = Theta.at[j, indices_div].get()
        return (zeta * psi_i * theta_j).sum()
    
    FIM_tau_nu = jnp.zeros((len(Sigma), len(G)), dtype=float)
    FIM_tau_nu = FIM_tau_nu.at[*indices.T].set(jax.lax.map(f_tau_nu, indices, batch_size=32))
    FIM_tau_nu = FIM_tau_nu * jnp.outer(1 / Sigma, 1 / G) / 2
    FIM_tau_nu = FIM_tau_nu @ group_loadings
    
    if G_fix_ind is not None:
        FIM_nu = jnp.delete(jnp.delete(FIM_nu, G_fix_ind, axis=0), G_fix_ind, axis=1)
        FIM_tau_nu = jnp.delete(FIM_tau_nu, G_fix_ind, axis=1)
    if drop_sigma:
        FIM_tau = jnp.identity(FIM_tau.shape[0])
        FIM_tau_nu = jnp.zeros_like(FIM_tau_nu)
    FIM = jnp.block([[FIM_tau, FIM_tau_nu],
                     [FIM_tau_nu.T, FIM_nu]])
    return FIM


def calc_error_variance_fim(data: TransformedData, error_variance: jnp.ndarray):
    d = 1 / jnp.array(error_variance).at[data.group_inds_inv].get()
    d = d / d.sum() ** 0.5
    D_product_inv = jnp.outer(-d, d)
    D_product_inv = jnp.fill_diagonal(D_product_inv,
                                      D_product_inv.diagonal() + d * d.sum(),
                                      inplace=False )
    fim = D_product_inv * D_product_inv.T / 2
    group_inds = data.group_inds
    group_loadings = np.zeros((len(d), len(group_inds)), dtype=int)
    for i, indices in enumerate(group_inds):
        group_loadings[indices, i] = 1
    group_loadings = jnp.array(group_loadings)
    return group_loadings.T @ fim @ group_loadings

def estimate_error_variance(data: TransformedData, B_decomposition: LowrankDecomposition,
                            error_lowrank: int = 0, verbose=False) -> ErrorVarianceEstimates:
    p = B_decomposition.Q.shape[0]
    Y = B_decomposition.null_space_transform(data.Y)
    d0 = jnp.array([np.var(Y[:, inds]) for inds in data.group_inds])

    fun = partial(loglik_error, Qn_Y=Y, group_inds_inv=data.group_inds_inv)
    grad = partial(loglik_error_grad, Qn_Y=Y, group_inds_inv=data.group_inds_inv,
                   group_inds=data.group_inds)
    fun = jax.jit(fun)
    grad = jax.jit(grad)
    opt = MetaOptimizer(fun, grad,  num_steps_momentum=15,
                        )
    res = opt.optimize(d0)
    if verbose:
        print('-' * 15)
        print(res)
        print('-' * 15)

    sigma = np.array(res.x)
    loglik = res.fun
    W = None
    if error_lowrank and error_lowrank > 0:
        # Jointly refine the group variances together with an (s x k) low-rank update to
        # the error column-covariance via the same REML deviance (Woodbury / Gram form).
        G_Y = np.asarray(Y).T @ np.asarray(Y)          # s x s sufficient statistic
        ptilde = Y.shape[0]
        sigma, W, loglik = _lowrank.estimate_error_lowrank(
            G_Y, ptilde, data.group_inds_inv, data.group_inds,
            k=int(error_lowrank), sigma0=np.array(res.x), verbose=verbose)
        if verbose:
            print(f'[error-lowrank k={error_lowrank}] REML deviance '
                  f'{res.fun:.3f} -> {loglik:.3f} (gain {res.fun - loglik:+.3f})')

    fim = calc_error_variance_fim(data, sigma)
    return ErrorVarianceEstimates(sigma, np.ones(p),
                                  np.array(fim),
                                  loglik_start=res.start_loglik,
                                  loglik=loglik,
                                  lowrank=(np.asarray(W) if W is not None and W.shape[1] else None))


def estimate_error_variance_full(data: TransformedData,
                                 B_decomposition: LowrankDecomposition,
                                 error_variance: ErrorVarianceEstimates,
                                 original_data: TransformedData = None,
                                 verbose=False) -> ErrorVarianceEstimates:
    # Y = B_decomposition.null_space_transform(data.Y)
    Y = data.Y
    d0 = error_variance.variance 
    D_fix_ind = np.argmin(d0)
    D_fix_val = d0[D_fix_ind]
    d0 = np.delete(d0, D_fix_ind)
    fun = partial(loglik_error_full, Y=Y, Q_C=B_decomposition.Q, group_inds_inv=data.group_inds_inv, D_fix_val=D_fix_val, D_fix_ind=D_fix_ind)
    fun = jax.jit(jax.value_and_grad(fun, argnums=0))
    from scipy.optimize import minimize
    if original_data is not None:
        Y0 = original_data.Y
        Y0 = Y0 - Y0.mean(axis=0, keepdims=True) - Y0.mean(axis=1, keepdims=True) + Y0.mean()
        D = error_variance.variance
        D = D[original_data.group_inds_inv]
        Y0 = Y0 * D ** (-0.5)
        prom_x0 = Y0.var(axis=1)
    else:
        prom_x0 = jnp.ones(len(data.Y))
    x0 = jnp.append(d0, prom_x0) ** 0.5
    res = minimize(fun, x0, jac=True,
                   method='TNC'#'L-BFGS-B', 
                   # options={'maxiter': 10000},
                   )
    if verbose:
        print('-' * 15)
        print(res)
        print('-' * 15)
    x = res.x ** 2
    D = x[:-len(data.Y)] 
    D = np.insert(D, D_fix_ind, D_fix_val)
    S = x[-len(data.Y):]# + 1e-4
    
    # Transforming to weights
    S = 1 - S / (original_data.Y * D[original_data.group_inds_inv] ** (-0.5)).var(axis=1)
    S = 1.0 / np.clip(S, 1e-8, 1.0)
    
    fim = error_variance.fim # TODO
    
    return ErrorVarianceEstimates(np.array(D), np.array(S), np.array(fim),
                                  loglik_start=error_variance.loglik_start,
                                  loglik=res.fun)

def estimate_promoter_mean(data: TransformedData,
                            B_decomposition: LowrankDecomposition,
                            error_variance: ErrorVarianceEstimates,
                            verbose=False) -> PromoterMeanEstimates:
    
    D = error_variance.variance[data.group_inds_inv]
    Y = jnp.array(data.Y)
    Q_C = jnp.array(B_decomposition.Q)
    W = _error_lowrank_W(error_variance)
    if W is None:
        dvec = (1.0 / D).reshape(-1, 1)              # Theta^{-1} 1_s  (diagonal case)
    else:
        # GLS weighting with the full error covariance Theta = D + W W^T.
        dvec = (_lowrank.theta_inv(D, W) @ jnp.ones((len(D), 1)))
    w = float(jnp.sum(dvec))                         # 1_s^T Theta^{-1} 1_s  (dvec = Theta^{-1} 1)
    mean = Y @ dvec
    mean = mean - Q_C @ (Q_C.T @ mean)
    weights = error_variance.promotor ** -0.5
    if np.std(weights) > 1e-12:
        q = weights / np.linalg.norm(weights)
        

        decomp_null = LowrankDecomposition(
            Q=q.reshape(-1, 1), 
            S=np.array([]), 
            V=np.array([])
        )
        

        mean_2d = mean.reshape(-1, 1)
        mean_2d = decomp_null.adjoint_null_space_transform(mean_2d)
        mean = mean_2d.flatten()
    else:

        mean = ones_nullspace_transform_transpose(mean)
    mean = mean / w
    return PromoterMeanEstimates(mean)

def _estimate_motif_variance_mom(Y, B, ind_fix, fix_value, eps=1e-14):
    # Gamma = (B^T B)^-1
    BTB = B.T @ B
    try:
        Gamma = np.linalg.inv(BTB)
    except np.linalg.LinAlgError:
        raise ValueError("B must have full column rank to be invertible.")
        
    # W = (B^T B)^-1 B^T
    W = Gamma @ B.T
    Z = W @ Y
    
    # E[Z_ij^2] = sigma_i^2 * g_j + Gamma_ii
    # We estimate sigma_i^2 * g_j by subtracting the known noise bias Gamma_ii.
    Gamma_diag = np.diag(Gamma)
    
    # S_ij = Z_ij^2 - Gamma_ii
    S = np.square(Z) - Gamma_diag[:, None]
    
    #  R_i approx sum(g) * sigma_i^2
    R = np.sum(S, axis=1)
    # C_j approx sum(sigma^2) * g_j
    C = np.sum(S, axis=0)
    
    # Using the ratio of column sums: g_j / g_fixed = C_j / C_fixed
    if C[ind_fix] == 0:
        scale_factor = 0
    else:
        scale_factor = fix_value / C[ind_fix]
        
    g_est = C * scale_factor
    
    # sigma_i^2 = R_i / sum(g)
    sum_g_est = np.sum(g_est)
    
    if sum_g_est == 0:
        sigma_sq_est = np.zeros_like(R)
    else:
        sigma_sq_est = R / sum_g_est
        
    return np.clip(sigma_sq_est, eps, float('inf')), np.clip(g_est, eps, float('inf'))

def estimate_motif_variance(data: TransformedData, B_decomposition: LowrankDecomposition,
                             error_variance: ErrorVarianceEstimates,
                             original_data: TransformedData = None,
                             activity_lowrank: int = 0,
                             verbose=False) -> MotifVarianceEstimates:
    multiplier = 1e-2
    D = jnp.array(error_variance.variance)
    j = jnp.argsort(D)[len(D) // 2]
    fix = D.at[j].get() * multiplier
    BTB = B_decomposition.V.T * B_decomposition.S ** 2 @ B_decomposition.V
    d = 1 / D.at[data.group_inds_inv].get()
    
    if original_data is not None:
        Y = original_data.Y
        B = original_data.B
        Y = Y - Y.mean(axis=0, keepdims=True) - Y.mean(axis=1, keepdims=True) + Y.mean()
        Y = Y * d ** 0.5
        B = B - B.mean(axis=0, keepdims=True)
        Sigma0, G0 = _estimate_motif_variance_mom(Y, B, j, 1e-1)
        G0 = G0 / d 
        G0 = np.array([G0[inds].mean() for inds in original_data.group_inds])
        scaler = fix / G0[j]
        G0 = G0 * scaler
        Sigma0 = Sigma0 / scaler
        G0 = np.delete(G0, j)
    else:
        Sigma0, G0 = jnp.ones(len(BTB), dtype=float), np.repeat(fix, len(D) - 1)

    # H_X^T (H_X Theta H_X^T)^{-1} H_X : projects out the full covariate row space X
    # (which contains 1_s), thereby removing both mu_p 1_s^T and the B M X mean.
    # With a low-rank error correction, Theta = D + W W^T (else Theta = D).
    W_err = _error_lowrank_W(error_variance)
    D_n = D.at[data.group_inds_inv].get()
    if W_err is None:
        D_product_inv = covariate_projection(D_n, data.X)
    else:
        D_product_inv = _lowrank.covariate_projection_theta(D_n, W_err, data.X)
    Z = data.B.T @ data.Y @ D_product_inv

    x0 = jnp.append(Sigma0, G0)
    fun = partial(loglik_motifs, Z=Z, BTB=BTB, D_product_inv=D_product_inv,
                  group_inds_inv=data.group_inds_inv, G_fix_ind=j, G_fix_val=fix)
    grad = partial(loglik_motifs_grad, Z=Z, BTB=BTB, D_product_inv=D_product_inv,
                  group_inds_inv=data.group_inds_inv, group_inds=data.group_inds,
                  G_fix_ind=j, G_fix_val=fix)
    fun = jax.jit(fun)
    grad = jax.jit(grad)
    opt = MetaOptimizer(fun, grad, num_steps_momentum=50 if original_data is None else 0)
    res = opt.optimize(x0)
    if not np.isfinite(res.fun):
        print(res)
        print(res.x)
    if verbose:
        print('-' * 15)
        print(res)
        print('-' * 15)
    Sigma = res.x[:len(BTB)]
    G = res.x[len(BTB):]
    
    G = jnp.insert(G, j, fix)
    fim = partial(loglik_motifs_fim, BTB=BTB, D_product_inv=D_product_inv,
                  group_inds_inv=data.group_inds_inv, group_inds=data.group_inds,
                  G_fix_ind=j, G_fix_val=fix)
    f = fim(res.x)
    eig = np.linalg.eigvalsh(f).min()
    print('FIM min eig', eig)
    if eig < 0:
        eig = list()
        epsilons =  [1e-23, 1e-18, 1e-15, 1e-12, 1e-9, 1e-8,
                     1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
        for eps in epsilons:
            x = res.x.copy()
            x = x.at[:len(BTB)].set(jnp.clip(x.at[:len(BTB)].get(), eps, float('inf')))
            f = fim(x)
            eig.append(np.linalg.eigvalsh(f).min())
            print(eps, eig[-1])
            if eig[-1] > 0:
                break
        i = np.argmax(eig)
        eps = epsilons[i]
        x = res.x.copy()
        x = x.at[:len(BTB)].set(jnp.clip(x.at[:len(BTB)].get(), eps, float('inf')))
        best_eig = eig[i]
        fim = fim(x)
        if best_eig <= 0:
            # Regularisation could not restore positive-definiteness: the information
            # matrix is genuinely rank deficient. Spell out why rather than leaving the
            # user with a bare negative eigenvalue (and a Cholesky failure in `export`).
            info = variance_dof(data.X, data.group_inds)
            n_bad = int(np.sum(np.linalg.eigvalsh(np.asarray(fim)) <= 0))
            logger_print(
                f'\n[warning] The Fisher information for (Sigma, nu) is singular: {n_bad} of '
                f'{len(np.asarray(fim))} directions are not\n  identified (regularisation did not '
                'restore positive-definiteness).', True)
            if not info['ok']:
                logger_print(
                    f'  Cause: only {info["dof"]} residual sample d.o.f. ({info["n_samples"]} '
                    f'samples - rank {info["rank_x"]} covariate design) are available for '
                    f'{info["n_free_nu"]} free\n  group variances -- see the degrees-of-freedom '
                    'warning above.\n', True)
            else:
                logger_print(
                    '  The degrees-of-freedom budget looks adequate, so this is more likely a '
                    'numerical or\n  convergence issue. Treat the variance parameters\' standard '
                    'errors with caution.\n', True)
    else:
        fim = f
    V_act = None
    loglik = res.fun
    if activity_lowrank and activity_lowrank > 0:
        # Refine (Sigma, nu, V) jointly with a rank-k correction to the activity covariance
        # G = diag(nu) + V V^T, using the safe-eigh REML (nan-free gradient even when grouped).
        from . import lowrank_g as _lrg
        Sigma_lr, nu_lr, V_act, loglik = _lrg.estimate_activity_lowrank(
            np.asarray(Z), np.asarray(BTB), np.asarray(D_product_inv),
            data.group_inds_inv, data.group_inds, k=int(activity_lowrank),
            Sigma0=np.asarray(Sigma), nu0=np.asarray(G),
            fix_ind=int(j), fix_val=float(fix), verbose=verbose)
        Sigma = Sigma_lr
        G = nu_lr
        if verbose:
            print(f'[activity-lowrank k={activity_lowrank}] Sigma/G REML deviance '
                  f'{res.fun:.3f} -> {loglik:.3f} (gain {res.fun - loglik:+.3f})')
    return MotifVarianceEstimates(motif=np.array(Sigma), group=np.array(G), fim=np.array(fim),
                                  fixed_group=j, loglik_start=res.start_loglik,
                                  loglik=loglik,
                                  lowrank=(np.asarray(V_act) if V_act is not None
                                           and np.asarray(V_act).shape[1] else None))


def estimate_motif_variance_identity(data: TransformedData, B_decomposition: LowrankDecomposition,
                                     error_variance: ErrorVarianceEstimates,
                                     verbose=False) -> MotifVarianceEstimates:
    D = jnp.array(error_variance.variance)
    BTB = B_decomposition.V.T * B_decomposition.S ** 2 @ B_decomposition.V
    W_err = _error_lowrank_W(error_variance)
    D_n = D.at[data.group_inds_inv].get()
    if W_err is None:
        D_product_inv = covariate_projection(D_n, data.X)
    else:
        D_product_inv = _lowrank.covariate_projection_theta(D_n, W_err, data.X)

    Z = data.B.T @ data.Y @ D_product_inv
    x0 = np.repeat(0.1, len(D))
    fun = partial(loglik_motifs, Z=Z, BTB=BTB, D_product_inv=D_product_inv,
                  group_inds_inv=data.group_inds_inv, drop_sigma=True)
    grad = partial(loglik_motifs_grad, Z=Z, BTB=BTB, D_product_inv=D_product_inv,
                  group_inds_inv=data.group_inds_inv, group_inds=data.group_inds,
                  drop_sigma=True)
    fun = jax.jit(fun)
    grad = jax.jit(grad)
    opt = MetaOptimizer(fun, grad, num_steps_momentum=50)
    res = opt.optimize(x0)
    
    if verbose:
        print('-' * 15)
        print(res)
        print('-' * 15)
    Sigma = np.ones(len(BTB))
    G = res.x
    fim = partial(loglik_motifs_fim, BTB=BTB, D_product_inv=D_product_inv,
                  group_inds_inv=data.group_inds_inv, group_inds=data.group_inds,
                  drop_sigma=True)
    fim = fim(res.x)
    eig = jnp.linalg.eigh(fim)[0].min()
    print('FIM min eig', eig)
    return MotifVarianceEstimates(motif=np.array(Sigma), group=np.array(G), fim=np.array(fim),
                                  fixed_group=None, loglik_start=res.start_loglik,
                                  loglik=res.fun)

def estimate_motif_mean(data: TransformedData, B_decomposition: LowrankDecomposition,
                         error_variance: ErrorVarianceEstimates,
                         motif_variance: MotifVarianceEstimates,
                         promoter_mean: PromoterMeanEstimates) -> MotifMeanEstimates:
    """BLUE/GLS estimate of the motif-mean matrix M (m x c) in the model
    U ~ MN(M X, Sigma, G). Generalizes the classic mu_m (c == 1, X == 1_s^T).

    Solves vec(M) = (W^T S^{-1} W)^{-1} W^T S^{-1} vec(Ytilde) with W = Xt^T (x) B and
    S = Ghat (x) (B Sigma B^T) + I, exploiting the eigenstructure of sqrt(Sigma) B^T B
    sqrt(Sigma) so that no n*p sized objects are ever formed. The stored Fisher
    information `fim` is mc x mc with column-major vec(M) ordering, so the intercept
    (mu_m) sub-block is fim[:m, :m].
    """
    D = jnp.array(error_variance.variance)
    Sigma = jnp.array(motif_variance.motif)
    G = jnp.array(motif_variance.group)
    mu_p = jnp.array(promoter_mean.mean)
    X = jnp.asarray(data.X, dtype=float)            # (c, n)
    c = X.shape[0]
    m = len(Sigma)

    d = (D ** 0.5).at[data.group_inds_inv].get()    # (n,) sqrt(D)
    Ghat = (G / D).at[data.group_inds_inv].get()    # (n,) nu/D per sample

    BTB = B_decomposition.V.T * B_decomposition.S ** 2 @ B_decomposition.V
    sig = jnp.sqrt(Sigma)
    A = sig.reshape(-1, 1) * BTB                     # sqrt(Sigma) B^T B  (m, m)

    weights = error_variance.promotor ** -0.5
    if np.std(weights) > 1e-12:
        # Weighted (promoter-variance) case: Householder transform matching weights.
        q = weights / np.linalg.norm(weights)
        mu_p_transformed = null_space_transform(q.reshape(-1, 1), mu_p.reshape(-1, 1))
    else:
        # Standard case: Helmert transform.
        mu_p_transformed = ones_nullspace_transform(mu_p.reshape(-1, 1))
    Y_tilde = (data.Y - mu_p_transformed) / d       # (p-1, n)  =  (Yhat - mu_p) sqrt(D^-1)
    Xt = X / d                                       # (c, n)    =  X sqrt(D^-1)

    D_B, Q_B = jnp.linalg.eigh(sig.reshape(-1, 1) * BTB * sig)   # eig of sqrt(Sig) B^TB sqrt(Sig)
    At_QB = A.T @ Q_B                                # (m, m), columns a_k
    BTY = data.B.T @ Y_tilde                         # (m, n)
    QY = Q_B.T * sig @ BTY                           # Q_B^T sqrt(Sig) B^T Ytilde  (m, n)

    XDX = Xt @ Xt.T                                  # (c, c) = X D^{-1} X^T
    w_jk = Ghat.reshape(-1, 1) / (1.0 + Ghat.reshape(-1, 1) * D_B.reshape(1, -1))   # (n, m)
    # Phi[k] = sum_j w_jk[j,k] xt_j xt_j^T   (c x c per eigen-direction k)
    Phi = jnp.einsum('an,nk,bn->kab', Xt, w_jk, Xt)                                 # (m, c, c)

    # Hessian (== Fisher information), assembled block-wise in column-major vec(M)
    # ordering: H[a*m + i, b*m + j].
    H_blocks = [[XDX[a, b] * BTB - (At_QB * Phi[:, a, b]) @ At_QB.T
                 for b in range(c)] for a in range(c)]
    H = jnp.block(H_blocks)                          # (mc, mc)

    # Right-hand side g, shape (m, c).
    g1 = BTY @ Xt.T                                  # (m, c)
    T = w_jk.T * QY                                  # (m, n)
    g2 = At_QB @ (T @ Xt.T)                          # (m, c)
    Grhs = g1 - g2                                   # (m, c)
    g_flat = Grhs.T.reshape(-1)                      # column-major: index a*m + i

    vecM = jnp.linalg.pinv(H) @ g_flat
    M = vecM.reshape(c, m).T                         # (m, c)
    return MotifMeanEstimates(np.array(M), np.array(H))

def estimate_mixture(Y: np.ndarray, B: np.ndarray, pi_mode=None, max_iter=100, tol=1e-5) -> np.ndarray:
    """
    EM algorithm for the corrected mixture model:
    Signal: Y_i ~ N(X_i U, D)
    Noise:  Y_i ~ N(mu, Sigma)
    D and Sigma are diagonal s x s matrices shared across their respective gene sets.
    """
    p, s = Y.shape
    # Y = Y - Y.mean(axis=1, keepdims=True) - Y.mean(axis=0, keepdims=True) + Y.mean()
    # X = B - B.mean(axis=0, keepdims=True)
    m = B.shape[1]
    # X = [B, 1] to allow for a baseline promoter mean in the signal model
    X = np.hstack([B, np.ones((p, 1))])
    
    Y_c = Y - Y.mean(axis=1, keepdims=True)
    var_y = (Y_c ** 2).mean(axis=1)
    
    # 2. Prevent division by zero
    var_y = np.clip(var_y, 1e-12, None)
    
    # 3. Create a soft weighting curve
    # E.g., genes near the median variance get weight ~1.0.
    # Genes with tiny variance (housekeeping) get lower weight.
    med_var = np.median(var_y)
    
    # This specific formula smoothly downweights low-variance genes
    # while keeping normal/high variance genes near 1.0.
    gamma = 1.0 - np.exp(-var_y / med_var)
    
    # Clip to avoid complete zeroes, which cause matrix conditioning issues
    gamma = np.clip(gamma, 1e-3, 1.0)
    
    return gamma
    # Old smart code for estimating gamma
    # 1. Initialize parameters
    pi = 0.5 if pi_mode is None or isinstance(pi_mode, tuple) else float(pi_mode)
    
    # Initial guess for U and mu
    U = np.linalg.solve(X.T @ X + 1e-5 * np.eye(X.shape[1]), X.T @ Y)
    mu = np.mean(Y, axis=0)
    
    # Initial guess for Variances (D and Sigma)
    v_floor = np.var(Y) * 1e-6
    D = np.var(Y - X @ U, axis=0) + v_floor
    Sigma = np.var(Y - mu[None, :], axis=0) + v_floor
    
    loglik_old = -np.inf
    
    for it in range(max_iter):
        # --- E-step ---
        # Signal log-density
        res_sig = Y - X @ U
        # (Y - XU)^2 / D -> sum over samples
        ll_sig = -0.5 * np.sum(res_sig**2 / D[None, :], axis=1) - 0.5 * np.sum(np.log(D))
        
        # Noise log-density
        res_noise = Y - mu[None, :]
        ll_noise = -0.5 * np.sum(res_noise**2 / Sigma[None, :], axis=1) - 0.5 * np.sum(np.log(Sigma))
        
        # Posterior Responsibility gamma_i = Prob(Signal | Y_i)
        l_pi = np.log(pi + 1e-15)
        l_1_pi = np.log(1 - pi + 1e-15)
        
        s1 = l_pi + ll_sig
        s2 = l_1_pi + ll_noise
        
        # Log-sum-exp trick for stability
        max_s = np.maximum(s1, s2)
        log_sum = max_s + np.log(np.exp(s1 - max_s) + np.exp(s2 - max_s))
        gamma = np.exp(s1 - log_sum)
        
        current_loglik = np.sum(log_sum)
        if it > 0 and 0 <= (current_loglik - loglik_old) < tol:
            break
        print(it, current_loglik, (current_loglik - loglik_old))
        loglik_old = current_loglik

        # --- M-step ---
        sum_gamma = np.sum(gamma)
        sum_1_gamma = p - sum_gamma
        
        # A. Update Signal Model (U and D)
        # Weight genes by gamma_i
        # U = (X^T diag(gamma) X)^-1 (X^T diag(gamma) Y)
        Xt_W = X.T * gamma
        U = np.linalg.solve(Xt_W @ X + 1e-8 * np.eye(X.shape[1]), Xt_W @ Y)
        
        # D is diagonal sample-wise variance
        D = np.sum(gamma[:, None] * (Y - X @ U)**2, axis=0) / (sum_gamma + 1e-12)
        D = np.clip(D, v_floor, None)
        
        # B. Update Noise Model (mu and Sigma)
        # mu is shared across genes in the noise model
        w_noise = (1 - gamma)[:, None]
        mu = np.sum(w_noise * Y, axis=0) / (sum_1_gamma + 1e-12)
        
        # Sigma is diagonal sample-wise variance
        Sigma = np.sum(w_noise * (Y - mu[None, :])**2, axis=0) / (sum_1_gamma + 1e-12)
        Sigma = np.clip(Sigma, v_floor, None)
        
        # C. Update mixing proportion pi
        if pi_mode is None:
            pi = sum_gamma / p
        elif isinstance(pi_mode, tuple):
            a, b = pi_mode
            pi = (sum_gamma + a - 1) / (p + a + b - 2)
        pi = np.clip(pi, 1e-5, 1.0)

    return gamma

def estimate_error_variance_weighted(data: TransformedData, 
                                     gamma: np.ndarray,
                                     verbose=False) -> ErrorVarianceEstimates:
    """
    Estimates sample-wise error variances D using gene-wise weights Gamma^-1.
    We transform the system by sqrt(gamma_i) so that the new error is MN(0, I, D).
    """
    # gamma_i is the probability of being signal. 
    # We treat Gamma_ii = 1/gamma_i as the variance.
    # Therefore, weights for whitening are sqrt(gamma_i).
    weights = np.sqrt(gamma)
    
    # 1. Manually whiten the data
    Y_weighted = data.Y * weights.reshape(-1, 1)
    B_weighted = data.B * weights.reshape(-1, 1)
    
    # 2. Re-calculate the low-rank decomposition for the whitened B
    # This is crucial because the null space changes when rows are scaled
    B_weighted_aug = np.append(B_weighted, weights.reshape(-1, 1), axis=1)
    B_decomp = lowrank_decomposition(B_weighted_aug)
    
    # 3. Project the weighted Y into the null space of weighted B
    Qn_Y = B_decomp.null_space_transform(Y_weighted)
    
    # 4. Standard estimation on the transformed data
    # The loglik_error function assumes Gamma=I, which is now true for (weights * Y)
    d0 = jnp.array([np.var(Qn_Y[:, inds]) for inds in data.group_inds])

    fun = partial(loglik_error, Qn_Y=Qn_Y, group_inds_inv=data.group_inds_inv)
    grad = partial(loglik_error_grad, Qn_Y=Qn_Y, group_inds_inv=data.group_inds_inv,
                   group_inds=data.group_inds)
    
    fun = jax.jit(fun)
    grad = jax.jit(grad)
    
    opt = MetaOptimizer(fun, grad, num_steps_momentum=15)
    res = opt.optimize(d0)
    
    if verbose:
        print(f"Weighted Error Variance Fit: {res.fun}")

    fim = calc_error_variance_fim(data, res.x)
    
    # We return the calculated Gamma = 1/gamma as the 'promoter' variance component
    # so that subsequent steps (motif variance, etc.) use the correct weighting.
    return ErrorVarianceEstimates(
        variance=np.array(res.x), 
        promotor=1.0 / (gamma + 1e-12), # This is Gamma_ii
        fim=np.array(fim),
        loglik_start=res.start_loglik,
        loglik=res.fun
    )

def estimate_sample_mean(data: TransformedData, error_variance: ErrorVarianceEstimates, 
                         motif_variance: MotifVarianceEstimates, promoter_mean: PromoterMeanEstimates,
                         motif_mean: MotifMeanEstimates):
    Y = data.Y
    B = data.B
    Y = Y - promoter_mean.mean.reshape(-1, 1) - B @ motif_mean_matrix(motif_mean, data.X)

    Y = jnp.asarray(Y)
    B = jnp.asarray(B)
    Sigma = jnp.asarray(motif_variance.motif)
    G = jnp.asarray(motif_variance.group)
    D = jnp.asarray(error_variance.variance)
    G = G.at[data.group_inds_inv].get()
    D = D.at[data.group_inds_inv].get()
    a_vec = (error_variance.promotor ** (-0.5))

    p, m = B.shape
    sqrt_Sigma = np.sqrt(Sigma).reshape(1, -1)
    C = B * sqrt_Sigma
    U, S, _ = jnp.linalg.svd(C, full_matrices=False)
    S_sq = S ** 2


    a = a_vec.T @ U
    sum_Y = a_vec.T @ Y
    a_sq_norm = np.sum(a_vec ** 2)


    UT_Y = U.T @ Y
    a_sq = a ** 2
    sum_a_sq = np.sum(a_sq)

    a_UT_Y = a[:, np.newaxis] * UT_Y


    num_part1 = np.sum(a_UT_Y / (G[np.newaxis, :] * S_sq[:, np.newaxis] + D[np.newaxis, :]), axis=0)
    num_part2 = (sum_Y - np.sum(a_UT_Y, axis=0)) / D
    numerator = num_part1 + num_part2


    denom_part1 = np.sum(a_sq[:, np.newaxis] / (G[np.newaxis, :] * S_sq[:, np.newaxis] + D[np.newaxis, :]), axis=0)
    denom_part2 = (a_sq_norm - sum_a_sq) / D
    denominator = denom_part1 + denom_part2

    mu = numerator / denominator

    return SampleMeanEstimates(np.array(mu).reshape(-1, 1))

@dataclass(frozen=True)
class ActivitiesPrediction:
    U: np.ndarray
    U_raw: np.ndarray
    filtered_motifs: np.ndarray
    tau_groups: dict
    clustering: tuple[np.ndarray, np.ndarray] = None
    _cov: tuple[np.ndarray, np.ndarray, np.ndarray,
                np.ndarray, np.ndarray, np.ndarray] = None
    # When the error covariance carries a low-rank correction, the per-group posterior
    # covariances are no longer the simple diagonal-D form below; they are precomputed
    # (full-Theta, see lowrank.blup_group_cov) and stored here.
    _cov_lowrank: list = None

    def cov(self) -> np.ndarray:
        cov_lr = getattr(self, '_cov_lowrank', None)
        if cov_lr is not None:
            # compact factors (M_eig, W_weights): cov_i = M_eig diag(W_weights[i]) M_eig^T
            M_eig, W_weights = cov_lr
            for i in range(len(W_weights)):
                yield _lowrank.expand_group_cov(M_eig, W_weights[i])
            return
        assert self._cov is not None
        Q_hat, S, sigma, nu, n, tau_mult = self._cov
        for sigma, nu, n, tau_mult in zip(sigma, nu, n, tau_mult):
            tau = nu / sigma * tau_mult
            D = n * S + 1 / tau
            D = 1 / D * sigma
            D = D ** 0.5
            Q_hat2 = Q_hat * D
            c = np.array(Q_hat2 @ Q_hat2.T, dtype=float)
            yield c

def predict_activities(data: TransformedData, fit: FitResult,
                       filter_motifs=True, filter_order=5,
                       tau_search=True, tau_left=0.1,  tau_right=1.0, tau_num=15,
                       clustering_search=False, k_min=0.1, k_max=0.9, k_num=6, 
                       cv_repeats=3, cv_splits=5,
                       pinv=False) -> ActivitiesPrediction:

    def _sol(BT_Y_sum, Q_hat, S, sigma, nu, n: int, tau_mult=1.0, BT_B=None):
        tau = nu / sigma * tau_mult
        if pinv:
            tau_mult = np.clip(tau_mult - 1, 0.0, a_max=float('inf'))
            sol = jnp.linalg.pinv(BT_B + tau_mult * jnp.identity(len(BT_B))) @ BT_Y_sum
        else:
            D = ( n * S + 1 / tau) ** (-0.5)
            Q_hat = Q_hat * D
            sol = Q_hat @ Q_hat.T @ BT_Y_sum
        return sol

    Sigma = fit.motif_variance.motif
    G = fit.motif_variance.group
    D = fit.error_variance.variance
    group_inds = data.group_inds
    a_vec = fit.error_variance.promotor ** -0.5
    mu_p = (a_vec * fit.promoter_mean.mean.flatten()).reshape(-1, 1)
    mu_s = fit.sample_mean.mean.reshape(-1, 1)
    B = data.B
    Y = data.Y
    # Subtract the per-sample mean motif effect B M X (covariate-aware; reduces to
    # B mu_m 1_s^T when there are no covariates).
    Y = Y - mu_p - B @ motif_mean_matrix(fit.motif_mean, data.X) - np.outer(a_vec, mu_s.flatten())


    if filter_motifs:
        inds = np.log10(Sigma) >= (np.median(np.log10(Sigma)) - filter_order)
        B = B[:, inds]
        Sigma = Sigma[inds]
        filtered_motifs = np.where(~inds)[0]
    else:
        filtered_motifs = list()
    clusters = defaultdict(list)
    if clustering_search:
        from tqdm import tqdm
        for n_cluster in tqdm([10, 25, 50, 75, 100, 150, 200, 500, B.shape[1]]):
            if n_cluster == B.shape[1]:
                Bc = B
                Sigma_c = Sigma
            else:
                Bc, c = cluster_data(B, mode=ClusteringMode.KMeans, num_clusters=n_cluster)
                Sigma_c = c * Sigma @ c.T 
                Sigma_c = Sigma_c.diagonal()
            rkf = RepeatedKFold(n_repeats=cv_repeats, random_state=1, n_splits=cv_splits)
            for train_inds, test_inds in rkf.split(Y):
                B_train = Bc[train_inds]
                B_test = Bc[test_inds]
                Y_train = Y[train_inds]
                Y_test = Y[test_inds]

                BT_Y = B_train.T @ Y_train
                if not pinv:
                    B_train = B_train * Sigma ** 0.5
                BT_B = B_train.T @ B_train
                S, Q_hat = jnp.linalg.eigh(BT_B)
                Q_hat = (Sigma ** 0.5).reshape(-1, 1) * Q_hat
                for i, (inds, sigma, nu) in enumerate(zip(group_inds, D, G)):
                    BT_Y_sub = BT_Y[:, inds]
                    U = _sol(BT_Y_sub, Q_hat, S, sigma, nu, 1, tau_mult=1, BT_B=BT_B)
                    diff = ((Y_test[:, inds] - B_test @ U[:, np.argsort(inds)]) ** 2).mean()
                    clusters[n_cluster].append(diff)
        clusters = {n: np.mean(v) for n, v in clusters.items()}
    else:
        clusters = {B.shape[1]: 0} 
    best_clust = min(clusters, key=lambda x: clusters[x])
    if best_clust == B.shape[1]:
        clust = None
        pass
    else:
        B, clust = cluster_data(B, mode=ClusteringMode.KMeans, num_clusters=best_clust)
        Sigma = c * Sigma @ c.T
        Sigma = Sigma.diagonal()
    tau_groups = defaultdict(lambda: defaultdict(list))
    if tau_search:
        from tqdm import tqdm
        # stats = defaultdict(float)
        rkf = RepeatedKFold(n_repeats=cv_repeats, random_state=1, n_splits=cv_splits)
        for train_inds, test_inds in tqdm(list(rkf.split(Y))):
            B_train = B[train_inds]
            B_test = B[test_inds]
            Y_train = Y[train_inds]
            Y_test = Y[test_inds]
            BT_Y = B_train.T @ Y_train
            if not pinv:
                B_train = B_train * Sigma ** 0.5
            BT_B = B_train.T @ B_train
            S, Q_hat = jnp.linalg.eigh(BT_B)
            Q_hat = (Sigma ** 0.5).reshape(-1, 1) * Q_hat
            for tau in np.linspace(tau_left, tau_right, num=tau_num):
                # pi = jnp.linalg.pinv(B)
                for i, (inds, sigma, nu) in enumerate(zip(group_inds, D, G)):
                    # all_inds.extend(inds)
                    BT_Y_sub = BT_Y[:, inds]
                    U = _sol(BT_Y_sub, Q_hat, S, sigma, nu, 1, tau_mult=tau, BT_B=BT_B)
                    diff = ((Y_test[:, inds] - B_test @ U[:, np.argsort(inds)]) ** 2).mean()
                    tau_groups[i][tau].append(diff)
        tau_groups = {g: min(v, key=lambda x: np.mean(v[x])) for g, v in tau_groups.items()}
    else:
        tau_groups = {i: 1.0 for i in range(len(group_inds))}
    W_err = _error_lowrank_W(fit.error_variance)
    V_act = _activity_lowrank_V(fit.motif_variance)
    if V_act is not None and len(filtered_motifs):
        # V lives in sample space (s x k); independent of motif filtering -> unchanged.
        pass
    B_unscaled = B                                   # keep loadings before sqrt(Sigma)
    BT_Y = B.T @ Y
    if not pinv:
        B = B * Sigma ** 0.5
    BT_B = B.T @ B
    S, Q_hat = jnp.linalg.eigh(BT_B)
    Q_hat = (Sigma ** 0.5).reshape(-1, 1) * Q_hat

    sizes = [len(inds) for inds in group_inds]
    tau_mults = [tau_groups[i] for i in range(len(group_inds))]

    cov_lowrank = None
    if W_err is None and V_act is None:
        # -------- classic diagonal-D BLUP (unchanged) --------
        U = list()
        U0 = list()
        all_inds = list()
        for i, (inds, sigma, nu) in enumerate(zip(group_inds, D, G)):
            tau = tau_groups[i]
            all_inds.extend(inds)
            BT_Y_sub = BT_Y[:, inds]
            U_pred = _sol(BT_Y_sub.sum(axis=-1, keepdims=True), Q_hat, S,
                          sigma, nu, len(inds), tau_mult=tau,
                          BT_B=BT_B)
            U.append(U_pred)
            U0.append(_sol(BT_Y_sub, Q_hat, S, sigma, nu, 1, tau_mult=tau, BT_B=BT_B))
        U = np.concatenate(U, axis=-1)
        U0 = np.concatenate(U0, axis=-1)[:, np.argsort(all_inds)]
    else:
        # -------- full-Theta / full-G BLUP with the low-rank corrections --------
        # Joint (sample-correlated) posterior means; reduces to the loop above at W=V=0.
        tau_arr = np.asarray(tau_mults, dtype=float)
        U0, U = _lowrank.blup_activities(Y, B_unscaled, np.asarray(Sigma), group_inds,
                                         np.asarray(D), np.asarray(G), W_err,
                                         tau_mult=tau_arr, V_act=V_act)
        # per-group posterior covariances (for export z-scores / ANOVA / contrasts), stored
        # as compact factors and expanded lazily.  The activity low-rank V enters the prior
        # only through its diagonal here (the point estimates above use the full G).
        cov_lowrank = _lowrank.blup_group_cov_factors(group_inds, np.asarray(D), np.asarray(G),
                                                      W_err, np.asarray(Sigma), B_unscaled,
                                                      tau_mult=tau_arr)
    return ActivitiesPrediction(U, U_raw=U0,
                                filtered_motifs=filtered_motifs,
                                tau_groups=tau_groups,
                                clustering=(B, clust) if clust is not None else None,
                                _cov=(Q_hat, S, D, G, sizes, tau_mults),
                                _cov_lowrank=cov_lowrank)


def null_space_transform_jax(Q: jax.Array, Y: jax.Array) -> jax.Array:
    p, r = Q.shape
    A = Q
    Y_transformed = Y

    def _householder_loop_body(j, state):
        A, Y_transformed = state
        col_j = A[:, j]
        mask = (jnp.arange(p) >= j)
        x_padded = col_j * mask
        sign_x_j = jnp.where(col_j[j] >= 0, 1.0, -1.0)
        alpha = -sign_x_j * jnp.linalg.norm(x_padded)
        v = x_padded.at[j].add(-alpha)
        v_norm_sq = jnp.dot(v, v)
        tau = jnp.where(v_norm_sq < 1e-24, 0.0, 2.0 / v_norm_sq)

        def update_A_func(A_in):
            w_A = jnp.dot(v, A_in)
            update_A = tau * jnp.outer(v, w_A)
            col_mask = (jnp.arange(r) > j)
            return A_in - update_A * col_mask
        
        A = jax.lax.cond(j + 1 < r, update_A_func, lambda A_in: A_in, A)
        
        w_Y = jnp.dot(v, Y_transformed)
        update_Y = tau * jnp.outer(v, w_Y)
        Y_transformed = Y_transformed - update_Y

        return A, Y_transformed

    initial_state = (A, Y_transformed)
    _, final_Y = jax.lax.fori_loop(0, r, _householder_loop_body, initial_state)
    return final_Y[r:, :]


class ClusteringMode(str, Enum):
    none = 'none'
    KMeans = 'KMeans'
    NMF = 'NMF'

def cluster_data(B: np.ndarray, mode=ClusteringMode.none, num_clusters=200,
                 keep_motifs=False)->ProjectData:
    def trs(B, labels, n):
        mx = np.zeros((n, B.shape[1]))
        for i, v in enumerate(labels):
            mx[v, i] = 1
        return mx
    if mode == ClusteringMode.none:
        return B, None
    if mode == ClusteringMode.KMeans:
        km = KMeans(n_clusters=num_clusters, n_init=10)
        km = km.fit(B.T)
        W = km.cluster_centers_.T 
        H = trs(B, km.labels_, num_clusters); 
    else:
        model = NMF(n_components=num_clusters, max_iter=1000)
        W = model.fit_transform(B)
        H = model.components_
    if not keep_motifs:
        B = W
        clustering = H
    else:
        B = W @ H
        clustering = None
    return B, clustering

from sklearn.decomposition import PCA, NMF

def construct_x(B, inds, n_pca: int = 4, n_nmf: int = 8):
    """
    Constructs feature matrices X_train and X_test from B using PCA, NMF, 
    and specific feature engineering rules.

    Parameters:
    B (np.ndarray): Input data matrix (n_samples, n_features).
    inds (array-like): Indices of the test samples.
    n_pca (int): Number of PCA components.
    n_nmf (int): Number of NMF components.

    Returns:
    X_train (np.ndarray): Training feature matrix.
    X_test (np.ndarray): Test feature matrix.
    """
    
    # 1. PCA Decomposition
    # pca.fit_transform returns (n_samples, n_pca)
    pca = PCA(n_components=n_pca)
    X_pca = pca.fit_transform(B)
    
    # 2. NMF Decomposition
    # nmf.fit_transform returns W matrix (n_samples, n_nmf)
    # We assume B is non-negative as required by NMF.
    nmf = NMF(n_components=n_nmf, init='nndsvda', random_state=0, max_iter=1000)
    W = nmf.fit_transform(B)
    
    # 3. Process NMF components
    # Divide by their sum (Probabilistic interpretation)
    # Sum along the components axis (axis=1). Keep dimensions for broadcasting.
    # We add a small epsilon to the denominator to avoid division by zero.
    W_sum = np.sum(W, axis=1, keepdims=True)
    W_norm = W / (W_sum + 1e-9)
    
    # 4. Create the extra component
    # "sum of NMF components in the power {-1}"
    # Grammatically interpreted as: Sum(Components^(-1)) -> element-wise inverse, then sum.
    # Equivalent to sum(1/W) along axis 1.
    # We add epsilon to W to prevent division by zero.
    W_inv_sum = np.sum(1.0 / (W + 1e-4), axis=1, keepdims=True)
    
    # 5. Concatenate all features
    # Order: PCA, NMF (raw), NMF (normalized), Extra component
    # Shape: (p, n_pca + n_nmf + n_nmf + 1)
    X = np.hstack([X_pca, W, W_norm, W_inv_sum, np.ones((len(X_pca), 1))])
    
    # 6. Split into Train and Test
    # inds are the test indices.
    # We create a mask to separate train and test data.
    n_samples = B.shape[0]
    mask = np.zeros(n_samples, dtype=bool)
    mask[inds] = True
    
    X_test = X[mask]
    X_train = X[~mask]
    
    return X_train, X_test

def fit(project: str, clustering: ClusteringMode,
        num_clusters: int, test_chromosomes: list, 
        gpu: bool, gpu_decomposition: bool, x64=True, true_mean=None, motif_variance: bool = True,
        promoter_variance: bool = False, test_promoters_filename: str = None,
        context_r: int = 0, drist: bool = False, error_lowrank: int = 0,
        activity_lowrank: int = 0,
        verbose=True, dump=True) -> ActivitiesPrediction:
    if x64:
        jax.config.update("jax_enable_x64", True)
    data = read_init(project)
    fmt = data.fmt
    group_names = data.group_names
    if clustering != clustering.none:
        logger_print('Clustering data...', verbose)
    promoter_names_train = data.promoter_names
    data.B, clustering = cluster_data(data.B, mode=clustering, 
                                      num_clusters=num_clusters)
    
    if test_promoters_filename:
        with open(test_promoters_filename, 'r') as f:
            test_chromosomes = filter(lambda x: len(x), map(lambda x: x.strip(), f.readlines()))
            test_chromosomes = set(test_chromosomes)
            promoter_inds_to_drop = [i for i, p in enumerate(data.promoter_names) 
                                     if p in test_chromosomes]
    elif test_chromosomes:
        import re
        pattern = re.compile(r'chr([0-9XYM]+|\d+)')

        test_chromosomes = set(test_chromosomes)
        promoter_inds_to_drop = [i for i, p in enumerate(data.promoter_names) 
                                 if pattern.search(p).group() in test_chromosomes]
    else:
        promoter_inds_to_drop = None
    promoter_names_test = list(np.array(promoter_names_train)[promoter_inds_to_drop])
    promoter_names_train = list(np.array(promoter_names_train)[np.setdiff1d(np.arange(0, len(promoter_names_train), True), 
                                                                            np.array(promoter_inds_to_drop))])
    # print('Loadings seqs [train]')
    # from .genread import read_seqs
    # X_train = read_seqs(promoter_names_train, 'susScr11.fa')
    # print('Loadings seqs [test]')
    # X_test = read_seqs(promoter_names_test, 'susScr11.fa')
    # data, data_test = split_data(data, promoter_inds_to_drop)
    drist = False
    if drist:
        logger_print('Imma DRISTing right now...', verbose)
        drist = DRIST(verbose=True, init=None, share_function=False).fit(data.B, data.Y)
    else:
        drist = None
    
    if context_r > 1:
        if data.L is not None:
            context_r = data.L.shape[1]
        logger_print('Estimating motif loading matrix context vectors...', verbose)
        loading_context = estimate_context(data.Y, data.B, r=context_r, gpu=gpu, L=data.L)
    else:
        loading_context = None
    # logger_print('Estimating motif loading matrix multipliers...', verbose)
    # loading_multipliers = estimate_multipliers(data.Y, data.B, gpu=gpu,)
    loading_multipliers = None
    logger_print('Transforming data...', verbose)
    data_orig = transform_data(data, helmert=False, loading_context=loading_context,
                               drist=drist, loading_multipliers=loading_multipliers)
    # data_test= transform_data(data_test, helmert=False, loading_context=loading_context,
    #                            drist=drist, loading_multipliers=loading_multipliers)

    if gpu_decomposition:
        device = jax.devices()
    else:
        device = jax.devices('cpu')
    device = next(iter(device))

    logger_print('Computing low-rank decompositions of the loading matrix...', verbose)
    with jax.default_device(device):
        B = np.append(data_orig.B, np.ones((len(data_orig.B), 1)), axis=1)
        B_decomposition_orig = lowrank_decomposition(B)
    if gpu:
        device = jax.devices()
    else:
        device = jax.devices('cpu')
    device = next(iter(device))

    with jax.default_device(device):

        logger_print('Estimating error variances...', verbose)
        error_variance = estimate_error_variance(data_orig, B_decomposition_orig,
                                                  error_lowrank=error_lowrank,
                                                  verbose=verbose)
        if error_lowrank and error_lowrank > 0:
            W_err = _error_lowrank_W(error_variance)
            kk = 0 if W_err is None else W_err.shape[1]
            logger_print(f'  error covariance: diagonal D + rank-{kk} update '
                         f'(REML deviance {error_variance.loglik:.2f}).', verbose)
        if promoter_variance:
            if error_lowrank and error_lowrank > 0:
                logger_print('  [warning] --error-lowrank is not jointly estimated with '
                             '--promoter-variance; the low-rank error correction is ignored '
                             'when promoter variances are estimated.', verbose)
            # logger_print('Estimating FULL error variances...', verbose)
            # error_variance = estimate_error_variance_full(data_orig, B_decomposition_orig, error_variance,
            #                                               original_data=data_orig,
            #                                               verbose=verbose)
            logger_print('Estimating mixture...', verbose)
            gamma = estimate_mixture(data_orig.Y, data_orig.B, pi_mode=0.9)
            logger_print('Estimating FULL error variances...', verbose)
            error_variance = estimate_error_variance_weighted(data_orig, gamma,
                                                              verbose=verbose)
            
            
            data_orig = transform_data(data, helmert=False, weights=error_variance.promotor ** (-0.5), 
                                       loading_context=loading_context, drist=drist,
                                       loading_multipliers=loading_multipliers)
            data = transform_data(data, helmert=True, weights=error_variance.promotor ** (-0.5),
                                  loading_context=loading_context, drist=drist,
                                  loading_multipliers=loading_multipliers)
        else:
            data_orig = transform_data(data, helmert=False, loading_context=loading_context, drist=drist,
                                       loading_multipliers=loading_multipliers)
            data = transform_data(data, helmert=True, loading_context=loading_context, drist=drist,
                                  loading_multipliers=loading_multipliers)
        B_decomposition = lowrank_decomposition(data.B)
    
        logger_print('Estimating promoter-wise means...', verbose)
        promoter_mean = estimate_promoter_mean(data, B_decomposition,
                                               error_variance=error_variance)
        
        logger_print('Estimating variances of motif activities...', verbose)
        # Check the sample-space d.o.f. budget up-front: if the covariate design plus the
        # number of groups over-determines the group variances, say so explicitly here
        # rather than letting it surface as a singular information matrix during export.
        warn_variance_dof(data.X, data.group_inds, verbose=verbose)
        if motif_variance:
            motif_variance = estimate_motif_variance(data, B_decomposition,
                                                      error_variance=error_variance,
                                                      original_data=data_orig,
                                                      activity_lowrank=activity_lowrank,
                                                      verbose=verbose)
        else:
            if activity_lowrank and activity_lowrank > 0:
                logger_print('  [warning] --activity-lowrank requires per-motif variances; '
                             'ignored because --no-motif-variance was set.', verbose)
            motif_variance = estimate_motif_variance_identity(data, B_decomposition,
                                                              error_variance=error_variance,
                                                              verbose=verbose)
        if activity_lowrank and activity_lowrank > 0 and motif_variance.lowrank is not None:
            logger_print(f'  activity covariance: diagonal G + rank-'
                         f'{motif_variance.lowrank.shape[1]} update '
                         f'(REML deviance {motif_variance.loglik:.2f}).', verbose)
        
        logger_print('Estimating motif means...', verbose)
        motif_mean = estimate_motif_mean(data, B_decomposition, error_variance=error_variance,
                                          motif_variance=motif_variance,
                                          promoter_mean=promoter_mean)
        logger_print('Estimating sample means...', verbose)
        sample_mean = estimate_sample_mean(data_orig, error_variance=error_variance, 
                                           motif_variance=motif_variance, motif_mean=motif_mean,
                                           promoter_mean=promoter_mean)

    promoter_mean = PromoterMeanEstimates(promoter_mean.mean.flatten() * error_variance.promotor ** 0.5)
    res = FitResult(error_variance=error_variance, motif_variance=motif_variance,
                    motif_mean=motif_mean, promoter_mean=promoter_mean,
                    sample_mean=sample_mean, clustering=clustering,
                    group_names=group_names, promoter_inds_to_drop=promoter_inds_to_drop,
                    loading_context=loading_context, drist=drist, loading_multipliers=loading_multipliers)    
    if dump:
        with openers[fmt](f'{project}.fit.{fmt}', 'wb') as f:
            dill.dump(res, f)
    return res

def split_data(data: ProjectData, inds: list) -> tuple[ProjectData, ProjectData]:
    if not inds:
        return data, None
    B_d = np.delete(data.B, inds, axis=0)
    B = data.B[inds]
    Y_d = np.delete(data.Y, inds, axis=0)
    Y = data.Y[inds]
    if data.L is not None:
        L_d = np.delete(data.L, inds, axis=0)
        L = data.L[inds]
    else:
        L_d = None
        L = None
    promoter_names_d = np.delete(data.promoter_names, inds)
    promoter_names = list(np.array(data.promoter_names)[inds])
    # Splitting is on the promoter (row) axis; the covariate design X (sample axis)
    # is identical for both halves and must be carried over.
    data_d = ProjectData(Y=Y_d, B=B_d, K=data.K, weights=data.weights,
                         group_inds=data.group_inds, group_names=data.group_names,
                         motif_names=data.motif_names, promoter_names=promoter_names_d,
                         motif_postfixes=data.motif_postfixes, sample_names=data.sample_names,
                         L=L_d, X=getattr(data, 'X', None),
                         covariate_names=getattr(data, 'covariate_names', None),
                         fmt=data.fmt)
    data = ProjectData(Y=Y, B=B, K=data.K, weights=data.weights,
                         group_inds=data.group_inds, group_names=data.group_names,
                         motif_names=data.motif_names, promoter_names=promoter_names,
                         motif_postfixes=data.motif_postfixes, sample_names=data.sample_names,
                         L=L, X=getattr(data, 'X', None),
                         covariate_names=getattr(data, 'covariate_names', None),
                         fmt=data.fmt)
    return data_d, data

def align_fit_to_promoters(fit: FitResult, inds, n_full: int) -> FitResult:
    """Restrict the per-promoter estimates in ``fit`` to the *training* promoters.

    ``fit`` is estimated over the FULL promoter set (the in-``fit`` train/test split at the
    top of :func:`fit` is disabled), whereas ``predict``/``gof``/``export`` evaluate on the
    training half returned by ``split_data(data, inds)`` -- i.e. with the held-out promoter
    indices ``inds`` removed.  The per-promoter arrays ``error_variance.promotor`` and
    ``promoter_mean.mean`` must therefore be sliced the very same way, otherwise they no
    longer line up row-for-row with the (B, Y) of the split data and broadcasting fails
    (e.g. ``B * weights.reshape(-1, 1)`` in :func:`transform_data`).

    ``np.delete(arr, inds)`` mirrors exactly how ``split_data`` slices B/Y, so the result is
    row-aligned by construction.  Arrays that are not full length (a fit that already trained
    on the training split, or degenerate/scalar arrays) are left untouched, which makes this
    safe to call unconditionally and idempotent.
    """
    if not inds:
        return fit
    inds = np.asarray(inds)
    error_variance = fit.error_variance
    promotor = getattr(error_variance, 'promotor', None)
    if promotor is not None:
        promotor = np.asarray(promotor)
        if promotor.ndim == 1 and promotor.shape[0] == n_full:
            error_variance = replace(error_variance,
                                     promotor=np.delete(promotor, inds, axis=0))
    promoter_mean = fit.promoter_mean
    mean = np.asarray(promoter_mean.mean).flatten()
    if mean.shape[0] == n_full:
        promoter_mean = replace(promoter_mean, mean=np.delete(mean, inds, axis=0))
    return replace(fit, error_variance=error_variance, promoter_mean=promoter_mean)

def predict(project: str, filter_motifs: bool, filter_order: int,
            tau_search: bool, cv_repeats: int, cv_splits: int,
            tau_left: float, tau_right: float, tau_num: int, pinv: bool,
            gpu: bool, x64=True,
            dump=True):
    if x64:
        jax.config.update("jax_enable_x64", True)
    data = read_init(project)
    fmt = data.fmt
    with openers[fmt](f'{project}.fit.{fmt}', 'rb') as f:
        fit: FitResult = dill.load(f)
    loading_context = fit.loading_context
    loading_multipliers = fit.loading_multipliers
    drist = fit.drist
    n_full = len(data.promoter_names)
    data, _ = split_data(data, fit.promoter_inds_to_drop)
    # Per-promoter estimates in `fit` span the full promoter set; keep only the training
    # entries so they stay row-aligned with the split data used below and in predict_activities.
    fit = align_fit_to_promoters(fit, fit.promoter_inds_to_drop, n_full)
    data = transform_data(data, helmert=False, weights=fit.error_variance.promotor ** (-0.5),
                          loading_context=loading_context, drist=drist, loading_multipliers=loading_multipliers)
    if gpu:
        device = jax.devices()
    else:
        device = jax.devices('cpu')
    device = next(iter(device))
    with jax.default_device(device):
        activities = predict_activities(data, fit, tau_search=tau_search,
                                        cv_repeats=cv_repeats, cv_splits=cv_splits,
                                        tau_left=tau_left, tau_right=tau_right, tau_num=tau_num, 
                                        pinv=pinv,
                                        filter_motifs=filter_motifs, 
                                        filter_order=filter_order)
    if dump:
        with openers[fmt](f'{project}.predict.{fmt}', 'wb') as f:
            dill.dump(activities, f)
    return activities

@dataclass(frozen=True)
class FOVResult:
    total: float
    promoter: np.ndarray
    sample: np.ndarray
    
@dataclass(frozen=True)
class TestResult:
    train: tuple[FOVResult]
    test: tuple[FOVResult]
    grouped: bool

def _groupify(X: np.ndarray, groups: list[np.ndarray]) -> np.ndarray:
    res = list()
    for inds in groups:
        res.append(X[:, inds].mean(axis=-1, keepdims=True))
    return np.concatenate(res, axis=-1)

def compute_mu_mle(data: TransformedData, fit: FitResult):
    mu_s = fit.sample_mean.mean.reshape(-1, 1)
    Y = data.Y - mu_s.T
    Y = Y - data.B @ motif_mean_matrix(fit.motif_mean, data.X)
    
    Sigma = fit.motif_variance.motif
    G = fit.motif_variance.group
    D = fit.error_variance.variance
    groups = data.group_inds_inv
    G = G[groups]
    D = D[groups]
    # Compute B√Σ using broadcasting
    B_tilde = data.B * jnp.sqrt(Sigma[None, :])
    
    # Economy-size SVD (p x k), k = min(p, m)
    U, s, _ = jnp.linalg.svd(B_tilde, full_matrices=False)
    s_sq = s**2
    
    # Compute residual space components first
    sum_Y_over_d = Y @ (1/D)  # \sum_i Y_i/D_ii
    sum_1_over_d = jnp.sum(1/D)  # \sum_i 1/D_ii
    
    # Projection and residual calculation
    proj = U @ (U.T @ sum_Y_over_d)
    mu_residual = (sum_Y_over_d - proj) / sum_1_over_d
    
    # Compute signal space components
    UTY = U.T @ Y  # k x s
    
    # Create inverse factor matrix (s x k)
    inv_factors = 1 / (G[:, None] * s_sq[None, :] + D[:, None])
    
    # Compute weighted sums
    sum_term1 = jnp.sum(UTY.T * inv_factors, axis=0)  # Sum over observations
    a_j = jnp.sum(inv_factors, axis=0)  # Normalization factors
    
    # Combine components
    mu_signal = U @ (sum_term1 / a_j)
    mu_hat = mu_signal + mu_residual
    
    return mu_hat

def _cor(a, b, axis=1):
    a_centered = a - a.mean(axis=axis, keepdims=True)
    b_centered = b - b.mean(axis=axis, keepdims=True)
    numerator = np.sum(a_centered * b_centered, axis=axis)
    denominator = np.sqrt(np.sum(a_centered**2, axis=axis) * np.sum(b_centered**2, axis=axis))
    return numerator / denominator

def calculate_fov(project: str, use_groups: bool, gpu: bool, 
                  stat_type: GOFStat, stat_mode: GOFStatMode, weights: bool = True,
                  x64=True, verbose=True, dump=True):
    def calc_fov(data: TransformedData, fit: FitResult,
                 activities: ActivitiesPrediction, mu_p=None, a_vec=None) -> tuple[FOVResult]:
        def sub(Y, effects) -> FOVResult:
            if stat_type == stat_type.fov:
                Y1 = Y - effects
                Y = Y - Y.mean()
                Y1 = Y1 - Y1.mean()
                Y = Y ** 2
                Y1 = Y1 ** 2
                prom = 1 - Y1.mean(axis=1) / Y.mean(axis=1)
                sample = 1 - Y1.mean(axis=0) / Y.mean(axis=0)
                total = 1 - Y1.mean() / Y.mean()
            elif stat_type == stat_type.corr:
                total = np.corrcoef(Y.flatten(), effects.flatten())[0, 1]
                prom = _cor(Y, effects, axis=1)
                sample = _cor(Y, effects, axis=0)
            return FOVResult(total, prom, sample)
        B = data.B
        drops = activities.filtered_motifs
        if mu_p is None:
            mu_p = fit.promoter_mean.mean
        mu_s = fit.sample_mean.mean.reshape(-1, 1)
        mu_p = mu_p.reshape(-1,1)
        Y = data.Y
        if weights:
            if a_vec is None:
                a_vec = fit.error_variance.promotor ** (-0.5)
                if len(a_vec) != len(Y):
                    a_vec = np.ones(len(Y)) * np.median(a_vec)
            Y = a_vec.reshape(-1, 1) * Y
            B = a_vec.reshape(-1, 1) * B
            mu_p = a_vec.reshape(-1, 1) * mu_p
        else:
            a_vec = jnp.ones(len(Y))
        d1 = mu_p.reshape(-1, 1) + jnp.outer(a_vec, mu_s.flatten())
        # Per-sample mean motif effect B M X (a_vec already folded into B above).
        d2 = B @ motif_mean_matrix(fit.motif_mean, data.X)
        # Y1 = Y0 - mu_p.reshape(-1, 1) - mu_s.reshape(1, -1)
        if use_groups:
            U = activities.U
            groups = data.group_inds
            Y = _groupify(Y, groups)
            d1 = _groupify(d1, groups)
            d2 = _groupify(d2, groups)
        else:
            U = activities.U_raw
        if activities.clustering is not None:
            d3 = activities.clustering[0] @ U
        else:
            d3 = np.delete(B, drops, axis=1) @ U
        if stat_mode == stat_mode.residual:
            stat_0 = sub(Y, d1 + d2 + d3)
            stat_1 = sub(Y - d1, d2 + d3)
            stat_2 = sub(Y - d1 - d2, d3)
        elif stat_mode == stat_mode.total:
            stat_0 = sub(Y, d1)
            stat_1 = sub(Y, d1 + d2)
            stat_2 = sub(Y, d1 + d2 + d3)
        return stat_0, stat_1, stat_2
    data = read_init(project)
    fmt = data.fmt
    with openers[fmt](f'{project}.fit.{fmt}', 'rb') as f:
        fit : FitResult = dill.load(f)
        loading_context = fit.loading_context
        drist = fit.drist
        loading_multipliers = fit.loading_multipliers
    with openers[fmt](f'{project}.predict.{fmt}', 'rb') as f:
        activities : ActivitiesPrediction = dill.load(f)
    n_full = len(data.promoter_names)
    data, data_test = split_data(data, fit.promoter_inds_to_drop)
    # Per-promoter estimates span the full promoter set; restrict them to the training
    # promoters so they line up with the training `data` used by calc_fov below.
    fit_train = align_fit_to_promoters(fit, fit.promoter_inds_to_drop, n_full)
    if x64:
        jax.config.update("jax_enable_x64", True)
    data = transform_data(data, helmert=False, loading_context=loading_context, drist=drist,
                          loading_multipliers=loading_multipliers)
    if data_test is not None:
        data_test = transform_data(data_test, helmert=False, loading_context=loading_context, drist=drist,
                                   loading_multipliers=loading_multipliers)
    if gpu:
        device = jax.devices()
    else:
        device = jax.devices('cpu')
    device = next(iter(device))
    with jax.default_device(device):
        if data_test is not None:
            try:
                with openers[fmt](f'{project}.promoter_mean.{fmt}', 'rb') as f:
                    mu_p : np.ndarray = dill.load(f)
            except FileNotFoundError:
                raise FileNotFoundError("To compute GOFs on the testing set, you must first run estimate-promoter-means with an appropriate setting.")
            test_FOV = calc_fov(data=data_test, fit=fit, activities=activities,
                                mu_p=mu_p)
        train_FOV = calc_fov(data=data, fit=fit_train, activities=activities)
    if data_test is None:
        test_FOV = None
        mu_p = None
    res = TestResult(train_FOV, test_FOV, grouped=use_groups)
    with openers[fmt](f'{project}.fov.{fmt}', 'wb') as f:
        dill.dump(res, f)
    return res
