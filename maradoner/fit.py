import numpy as np
import jax.numpy as jnp
import jax
import scipy.linalg.lapack as lapack
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from dataclasses import dataclass, replace
from functools import partial
from .meta_optimizer import MetaOptimizer, OptimizerResult, STATIONARITY_TOL
from sklearn.model_selection import RepeatedKFold
from collections import defaultdict
from enum import Enum
from .utils import read_init, ProjectData, logger_print, openers
from .utils import InputError, warn, load_stage, dump_stage, stage_filename, remove_stage, apply_clustering
import os
import re
import dill
from .linalg import lowrank_decomposition, ones_nullspace, ones_nullspace_transform, \
                    ones_nullspace_transform_transpose, null_space_transform, LowrankDecomposition, \
                    check_design_rank
from .loading_context import estimate_context, LoadingContext, LoadingMultipliers, estimate_multipliers
from .drist import DRIST, LoadingDRIST
from . import lowrank as _lowrank
from . import lowrank_sigma as _ls
from . import mean_prior as _mean_prior
from .lowrank_sigma import motif_covariance, motif_root, lmul, rmul


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


def _decorrelate_samples(D_n, nu_n, W=None, V=None):
    """Transform of the samples for Theta = diag(D_n) + W W^T and G = diag(nu_n) + V V^T.
    Returns (T, Ghat, T_back): T = Theta^{-1/2} R with R the eigenvectors of
    Theta^{-1/2} G Theta^{-1/2} and Ghat their eigenvalues, so that T^T Theta T = I and
    T^T G T = diag(Ghat); T_back = Theta^{1/2} R maps the parameters of the columns of
    Y T back to the samples."""
    Theta = np.diag(np.asarray(D_n, dtype=np.float64))
    if W is not None:
        Theta = Theta + W @ W.T
    G = np.diag(np.asarray(nu_n, dtype=np.float64))
    if V is not None:
        G = G + V @ V.T
    lt, Qt = np.linalg.eigh(Theta)
    Theta_isqrt = (Qt / np.sqrt(lt)) @ Qt.T
    Ghat, R = np.linalg.eigh(Theta_isqrt @ G @ Theta_isqrt)
    return Theta_isqrt @ R, np.clip(Ghat, 0.0, None), (Qt * np.sqrt(lt)) @ Qt.T @ R


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
    # Optional names of the motifs (columns of B), of the sample groups and of the
    # covariates (rows of X); used only to name them in warnings and errors.
    motif_names: list = None
    group_names: list = None
    covariate_names: list = None


# The scale of the motif variances tau and of the group variances nu is not identified
# separately (only their products are): it is fixed by setting nu of the reference group
# (the one with the median error variance) to this fraction of its error variance.
NU_REFERENCE_FRACTION = 1e-2
# Starting values of variances are floored at this fraction of the typical positive
# starting value: the optimiser works with square roots, where zero is always stationary.
START_REL_FLOOR = 1e-3
# A variance below this fraction of its natural scale is treated as sitting at zero.
ZERO_REL_TOL = 1e-8
# Group variances of activities further than this factor from the reference one are
# reported as poorly identified.
NU_RATIO_WARN = 1e3
# Variances estimated as zero are stored as this (negligible) fraction of their natural
# scale, as the code using them takes their logarithms and reciprocals.
ZERO_REL_VALUE = 1e-20
# A variance at zero is released and the fit restarted if the expected decrease of the
# objective (-2 logL) exceeds this.
KKT_GAIN_TOL = 1e-3
# The optimiser of the variances of motif activities: 'fisher' for Fisher scoring and L-BFGS-B in
# the variances (_optimize_motif_variances), 'meta' for the MetaOptimizer alone.
MOTIF_VARIANCE_OPTIMIZER = 'fisher'


def covariate_projection(D_n: jnp.ndarray, X: jnp.ndarray, names=None) -> jnp.ndarray:
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
    names : optional names of the rows of X, used in the error raised when they are
            linearly dependent.
    """
    check_design_rank(X, names)
    X = jnp.asarray(X, dtype=D_n.dtype)
    Di = 1.0 / D_n                      # (s,) = diag(D^{-1})
    XDi = X * Di                        # (c, s) = X D^{-1}
    XDiX = XDi @ X.T                    # (c, c) = X D^{-1} X^T
    return jnp.diag(Di) - XDi.T @ jnp.linalg.solve(XDiX, XDi)


def _n_identifiable_variances(X, group_inds) -> int:
    """Number of linearly independent matrices among H_X J_g H_X^T, g = 1..#groups, where
    H_X projects the covariate row space X out of the sample space and J_g is the diagonal
    indicator of the samples of group g. Group-wise variances enter the restricted
    likelihood only through sum_g v_g H_X J_g H_X^T, hence they are identifiable if and
    only if this number equals the number of groups."""
    X = np.asarray(X, dtype=float)
    s = X.shape[1]
    proj = np.eye(s) - np.linalg.pinv(X) @ X
    loadings = np.zeros((s, len(group_inds)))
    for i, inds in enumerate(group_inds):
        loadings[inds, i] = 1.0
    # Gram matrix of the vectorised H_X J_g H_X^T
    gram = loadings.T @ (proj ** 2) @ loadings
    eig = np.linalg.eigvalsh(gram)
    if not len(eig) or eig.max() <= 0:
        return 0
    return int(np.sum(eig > len(eig) * 1e-10 * eig.max()))


def variance_dof(X, group_inds) -> dict:
    """Degrees-of-freedom budget for the (Sigma, G) REML step.

    That step estimates the group variances nu from the sample-space residual left
    after projecting out the covariate row space X (which always contains the
    intercept 1_s). The projection consumes rank(X) of the s sample dimensions, so
    only ``dof = s - rank(X)`` remain. The projected sample covariance is a symmetric
    ``dof x dof`` matrix to which every group contributes one fixed matrix scaled by its
    variance; the group variances are identifiable exactly when these matrices are
    linearly independent (which requires ``g <= dof (dof + 1) / 2``). ``n_free_nu`` is
    ``g - 1`` since one nu is pinned to fix the Sigma/G scale indeterminacy.

    Returns a dict with the budget, an ``ok`` flag for the group variances nu and an
    ``ok_sigma`` flag for the error variances (the same condition with the intercept as
    the only covariate, as in the REML step that estimates them).
    """
    X = np.asarray(X, dtype=float)
    s = X.shape[1]
    rank_x = int(np.linalg.matrix_rank(X)) if X.size else 0
    dof = s - rank_x
    g = len(group_inds)
    n_free = max(g - 1, 0)
    n_ident = _n_identifiable_variances(X, group_inds)
    n_ident_sigma = _n_identifiable_variances(np.ones((1, s)), group_inds)
    return {'n_samples': s, 'n_covariates': X.shape[0], 'rank_x': rank_x,
            'dof': dof, 'n_groups': g, 'n_free_nu': n_free,
            'n_identifiable': n_ident, 'n_identifiable_sigma': n_ident_sigma,
            'ok': n_ident >= g, 'ok_sigma': n_ident_sigma >= g}


def warn_variance_dof(X, group_inds, verbose: bool = True) -> dict:
    """Emit an explicit warning when the variance parameters are over-determined.

    Prints an actionable diagnosis instead of leaving the user to decode a singular
    Fisher information matrix later on (an indefinite FIM here surfaces downstream as
    "Failed to compute inverse using Cholesky decomposition" during ``export``).
    The warning is printed whatever ``verbose`` is: it concerns the validity of results.
    """
    info = variance_dof(X, group_inds)
    remedy = ('What to do: pool samples into fewer, larger groups with "--sample-groups" of '
              '"maradoner create" (replicates then share one variance)')
    if not info['ok_sigma']:
        warn(f'The error variances of the {info["n_groups"]} sample groups are not identifiable '
             f'from {info["n_samples"]} samples: after removing the sample-wise means only '
             f'{info["n_identifiable_sigma"]} independent combination(s) of them can be '
             'estimated, the split between the groups is arbitrary. Group variances, their '
             'standard errors and everything derived from them are unreliable. '
             f'{remedy}, or add samples.')
    if info['ok']:
        return info
    n_cov = info['n_covariates'] - 1                      # excluding the intercept
    warn('The variances of motif activities of the sample groups (nu) are not identifiable: '
         f'{info["n_samples"]} samples minus the rank {info["rank_x"]} of the covariate design '
         f'(intercept + {n_cov} covariate(s)) leave {info["dof"]} residual sample dimension(s), '
         f'which determine only {info["n_identifiable"]} independent combination(s) of the '
         f'{info["n_groups"]} group variances. The Fisher information for (tau, nu) is '
         'singular: nu, their standard errors (params/group_variances.tsv) and the tests '
         f'built on them are unreliable. {remedy}'
         + (', and/or pass fewer covariates to "maradoner create".' if n_cov else '.'))
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
    # Per-motif likelihood-ratio statistics (length m, >= 0) for H0: tau_i = 0, with the
    # other variance parameters held at their estimates. None when not computed (old
    # pickles, --no-motif-variance).
    lrt: np.ndarray = None
    # Optional structure of the covariance of activities between motifs (fit --motif-families /
    # --motif-lowrank): S = diag(motif) + F (diag(family_variance) + family_lowrank
    # family_lowrank^T) F^T with F = families (m, f) of zeros and ones, or the identity if
    # families is None (then family_variance is None too). Only the family variances marked
    # in family_free are estimated (a family of one motif has none: it is a part of tau), and
    # they follow nu in fim. family_lrt holds the likelihood-ratio statistics of
    # H0: family_variance_j = 0. All None: the motifs are independent, S = diag(motif).
    families: np.ndarray = None
    family_names: list = None
    family_variance: np.ndarray = None
    family_lowrank: np.ndarray = None
    family_free: np.ndarray = None
    family_lrt: np.ndarray = None

@dataclass(frozen=True)
class MotifMeanEstimates:
    mean: np.ndarray
    fim: np.ndarray
    # (m, m) covariance that the unknown promoter baselines mu_p contribute to the
    # intercept column of M: only mu_p + B mu_m is identified, so the part of mu_p lying
    # in col(B) is absorbed by mu_m. It is added to the intercept block of pinv(fim) when
    # standard errors are reported. None when not computed (old pickles).
    # It equals omega^2 (B_c^T B_c)^+ with B_c the transformed loadings and omega^2 the
    # variance of the promoter baselines, taken as i.i.d. random (see estimate_motif_mean
    # for omega^2 and for the form it takes with --promoter-variance).
    baseline_cov: np.ndarray = None
    # Prior on the intercept column of M (fit --motif-mean-prior, see mean_prior.py): its kind,
    # the REML estimates of its variances and of the common mean mu_0, the covariance of the
    # prediction error of the posterior means (m, m; it replaces the intercept block of
    # pinv(fim) + baseline_cov) and the GLS estimate before the shrinkage. None without a prior.
    prior: str = None
    prior_variances: dict = None
    prior_mean: float = None
    posterior_cov: np.ndarray = None
    gls_intercept: np.ndarray = None
    # With a prior on the coefficients of the covariates as well: the covariance of the
    # prediction error of each row of M (m, c, c). None otherwise.
    posterior_blocks: np.ndarray = None

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
    if all(hasattr(data, a) for a in ('B_orig', 'Y_orig', 'original_inds')):
        B = data.B_orig
        Y = data.Y_orig
        group_inds = data.original_inds
    else:
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
            weights = weights.reshape(-1, 1) / np.linalg.norm(weights)
            Y = null_space_transform(weights, Y)
            B = null_space_transform(weights, B)
    _check_groups(group_inds, Y.shape[1], getattr(data, 'sample_names', None),
                  getattr(data, 'group_names', None))
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
                           X=np.asarray(X, dtype=float),
                           motif_names=getattr(data, 'motif_names', None),
                           group_names=getattr(data, 'group_names', None),
                           covariate_names=getattr(data, 'covariate_names', None))

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
    det_term = ((M @ D_A) @ Z) / Sigma
    grad_tau = -vec_term + det_term
    # Derivative w.r.t to nu/G
    Lambda = Lambda_base @ Q_A.T
    vec_term = jnp.einsum('ji,ji->i', Lambda, Lambda) / (G ** 2)
    Z = (Q_A.T * G @ D_product_inv) * Q_A.T
    det_term = ((D_B @ M) @ Z) / G
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

def _motifs_spectral(x: jnp.ndarray, BTB: jnp.ndarray, D_product_inv: jnp.ndarray,
                     group_inds_inv: jnp.ndarray, G_fix_ind=None, G_fix_val=1.0,
                     drop_sigma=False, G_lowrank=None) -> tuple:
    """Quantities shared by the value, the gradient and the Fisher information of
    loglik_motifs. With N = B^T B, P = D_product_inv, T = diag(tau) and G = R R^T the
    sample covariance of activities (diag(nu) per sample, plus G_lowrank G_lowrank^T
    if given): eigenpairs (D_B, Q_B) of T^1/2 N T^1/2 and (D_A, Q_A) of R^T P R,
    BTCQ = N T^1/2 Q_B, PGQ = P R Q_A and Mi[l, k] = 1 / (1 + D_B[l] D_A[k]).
    None of the expressions built on them divides by sqrt(tau) or sqrt(nu), so they stay
    valid for parameters that are exactly zero."""
    m = len(BTB)
    if drop_sigma:
        x = jnp.append(jnp.ones(m), x)
    sig = x[:m] ** 0.5
    nu = x[m:]
    if G_fix_ind is not None:
        nu = jnp.insert(nu, G_fix_ind, G_fix_val)
    nu = nu[group_inds_inv]
    if G_lowrank is None:
        R = jnp.diag(nu ** 0.5)
        A = (nu ** 0.5).reshape(-1, 1) * D_product_inv * nu ** 0.5
    else:
        # A rectangular root: no Cholesky factor, hence valid for nu that are zero.
        R = jnp.concatenate([jnp.diag(nu ** 0.5), jnp.asarray(G_lowrank)], axis=1)
        A = R.T @ D_product_inv @ R
    D_A, Q_A = jnp.linalg.eigh(A)
    D_B, Q_B = jnp.linalg.eigh(sig.reshape(-1, 1) * BTB * sig)
    D_A = jnp.where(D_A > 0, D_A, 0.0)
    D_B = jnp.where(D_B > 0, D_B, 0.0)
    BTCQ = BTB * sig @ Q_B
    RQ = R @ Q_A
    PGQ = D_product_inv @ RQ
    Mi = 1 / (D_B.reshape(-1, 1) * D_A + 1)
    return sig, D_A, Q_A, D_B, Q_B, BTCQ, RQ, PGQ, Mi

def loglik_motifs_value_and_grad(x: jnp.ndarray, Z: jnp.ndarray, BTB: jnp.ndarray,
                                 D_product_inv: jnp.ndarray, group_inds_inv: jnp.ndarray,
                                 num_groups: int, G_fix_ind=None, G_fix_val=1.0,
                                 drop_sigma=False) -> tuple:
    """loglik_motifs and loglik_motifs_grad from a single pair of eigendecompositions,
    without Kronecker products. The gradient is also correct at tau_i = 0 and nu_g = 0."""
    sig, D_A, Q_A, D_B, Q_B, BTCQ, RQ, PGQ, Mi = _motifs_spectral(
        x, BTB, D_product_inv, group_inds_inv, G_fix_ind=G_fix_ind, G_fix_val=G_fix_val,
        drop_sigma=drop_sigma)
    ZQ = Z @ RQ
    QZ = Q_B.T * sig @ Z
    Lambda_base = (QZ @ RQ) * Mi
    loglik = -(Lambda_base ** 2 / Mi).sum() - jnp.log(Mi).sum()
    # Derivative w.r.t. to tau/Sigma
    E = ZQ - BTCQ @ (Lambda_base * D_A)
    grad_tau = -(E ** 2).sum(axis=1) + BTB.diagonal() * D_A.sum() - BTCQ ** 2 @ (Mi @ D_A ** 2)
    # Derivative w.r.t to nu/G
    E = QZ - (Lambda_base * D_B.reshape(-1, 1)) @ PGQ.T
    grad_nu = -(E ** 2).sum(axis=0) + D_product_inv.diagonal() * D_B.sum() \
              - PGQ ** 2 @ (D_B ** 2 @ Mi)
    grad_nu = jax.ops.segment_sum(grad_nu, group_inds_inv, num_segments=num_groups)
    if G_fix_ind is not None:
        grad_nu = jnp.delete(grad_nu, G_fix_ind)
    grad = jnp.append(grad_tau, grad_nu)
    if drop_sigma:
        grad = grad[len(BTB):]
    return loglik, grad

# Compiled once per problem size (the data are arguments, not constants of the program).
_loglik_motifs_value_and_grad = jax.jit(loglik_motifs_value_and_grad,
                                        static_argnames=('num_groups', 'G_fix_ind', 'drop_sigma'))

@jax.jit
def _fim_quadratic_block(N: jnp.ndarray, U: jnp.ndarray, d: jnp.ndarray,
                         e: jnp.ndarray) -> jnp.ndarray:
    # 1/2 sum_k e_k^2 W_k o W_k with W_k = N - e_k U diag(1 / (1 + e_k d)) U^T
    def body(k, mx):
        W_k = N - e[k] * (U / (1 + e[k] * d)) @ U.T
        return mx + e[k] ** 2 * W_k * W_k
    return jax.lax.fori_loop(0, len(e), body, jnp.zeros_like(N)) / 2

def loglik_motifs_fim(x: jnp.ndarray, BTB: jnp.ndarray, 
                      D_product_inv: jnp.ndarray, group_inds_inv: jnp.ndarray,
                      group_inds: jnp.ndarray, G_fix_ind=None, G_fix_val=1.0,
                      drop_sigma=False, G_lowrank=None) -> float:
    """Fisher information of (tau, free nu) in the model of loglik_motifs, i.e.
    1/2 tr(V^-1 dV_i V^-1 dV_j) for the covariance V of the projected data. With G_lowrank
    (s x k) the sample covariance of activities is diag(nu) + G_lowrank G_lowrank^T and
    the information is the one of (tau, nu) with G_lowrank held fixed."""
    sig, D_A, Q_A, D_B, Q_B, BTCQ, RQ, PGQ, Mi = _motifs_spectral(
        x, BTB, D_product_inv, group_inds_inv, G_fix_ind=G_fix_ind, G_fix_val=G_fix_val,
        drop_sigma=drop_sigma, G_lowrank=G_lowrank)
   
    group_loadings = np.zeros((len(group_inds_inv), len(group_inds)), dtype=int)
    for i, indices in enumerate(group_inds):
        group_loadings[indices, i] = 1
    group_loadings = jnp.array(group_loadings)

    # The block of tau is replaced by the identity with drop_sigma.
    FIM_tau = None if drop_sigma else _fim_quadratic_block(jnp.asarray(BTB), BTCQ, D_B, D_A)
    FIM_nu = _fim_quadratic_block(jnp.asarray(D_product_inv), PGQ, D_A, D_B)
    FIM_nu = group_loadings.T @ FIM_nu @ group_loadings
    FIM_tau_nu = BTCQ ** 2 @ Mi ** 2 @ (PGQ ** 2).T / 2
    FIM_tau_nu = FIM_tau_nu @ group_loadings
    
    if G_fix_ind is not None:
        FIM_nu = jnp.delete(jnp.delete(FIM_nu, G_fix_ind, axis=0), G_fix_ind, axis=1)
        FIM_tau_nu = jnp.delete(FIM_tau_nu, G_fix_ind, axis=1)
    if drop_sigma:
        FIM_tau = jnp.identity(len(BTB))
        FIM_tau_nu = jnp.zeros_like(FIM_tau_nu)
    FIM = jnp.block([[FIM_tau, FIM_tau_nu],
                     [FIM_tau_nu.T, FIM_nu]])
    return FIM

def loglik_motifs_lrt(tau: np.ndarray, G_root: np.ndarray, Z: np.ndarray, BTB: np.ndarray,
                      D_product_inv: np.ndarray) -> np.ndarray:
    """Likelihood-ratio statistics of H0: tau_i = 0, one per motif:
    lrt[i] = loglik_motifs(tau with tau_i := 0) - loglik_motifs(tau), all the other
    parameters being held at the supplied values (loglik_motifs is -2 logL).
    G_root is any matrix R with s rows and R R^T equal to the sample covariance of the
    activities (diag(sqrt(nu)) per sample for a diagonal G).

    Setting tau_i to zero removes motif i from the model, which is a rank-one downdate
    of (T^-1 + alpha_k N)^-1 for every eigenvalue alpha_k of R^T P R. Hence, with
    R_k = (I + alpha_k T^1/2 N T^1/2)^-1 and lambda_k = R_k T^1/2 Z R q_k,
        lrt[i] = sum_k lambda_k[i]^2 / R_k[i, i] + log R_k[i, i],
    which needs no eigendecomposition beyond those of the full model."""
    tau = np.asarray(tau, dtype=np.float64)
    G_root = np.asarray(G_root, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)
    BTB = np.asarray(BTB, dtype=np.float64)
    P = np.asarray(D_product_inv, dtype=np.float64)
    sig = tau ** 0.5
    D_A, Q_A = np.linalg.eigh(G_root.T @ P @ G_root)
    D_B, Q_B = np.linalg.eigh(sig.reshape(-1, 1) * BTB * sig)
    D_A = np.where(D_A > 0, D_A, 0.0)
    D_B = np.where(D_B > 0, D_B, 0.0)
    Mi = 1 / (D_B.reshape(-1, 1) * D_A + 1)
    Lambda = Q_B @ ((Q_B.T * sig @ Z @ G_root @ Q_A) * Mi)
    R_diag = np.clip(Q_B ** 2 @ Mi, np.finfo(np.float64).tiny, 1.0)
    lrt = (Lambda ** 2 / R_diag + np.log(R_diag)).sum(axis=1)
    return np.clip(lrt, 0.0, None)


def calc_error_variance_fim(data: TransformedData, error_variance: jnp.ndarray,
                            n_rows: int = 1, lowrank: np.ndarray = None):
    """Fisher information of the group-wise error variances in the REML step that
    estimates them: n_rows / 2 * sum_{i in g, j in h} P_ij^2 with
    P = H^T (H Theta H^T)^-1 H, H the complement of 1_s. n_rows is the number of rows of
    the REML-projected data (promoters minus the rank of the fixed-effect design), each
    of which is an independent draw from N(0, H Theta H^T). With lowrank = W (s x k),
    Theta = D + W W^T and the information is the one of the variances with W held fixed."""
    if lowrank is not None and np.asarray(lowrank).shape[1] > 0:
        d = jnp.array(error_variance).at[data.group_inds_inv].get()
        D_product_inv = _lowrank.covariate_projection_theta(d, lowrank, jnp.ones((1, len(d))))
    else:
        d = 1 / jnp.array(error_variance).at[data.group_inds_inv].get()
        d = d / d.sum() ** 0.5
        D_product_inv = jnp.outer(-d, d)
        D_product_inv = jnp.fill_diagonal(D_product_inv,
                                          D_product_inv.diagonal() + d * d.sum(),
                                          inplace=False )
    fim = n_rows * D_product_inv * D_product_inv.T / 2
    group_inds = data.group_inds
    group_loadings = np.zeros((len(d), len(group_inds)), dtype=int)
    for i, indices in enumerate(group_inds):
        group_loadings[indices, i] = 1
    group_loadings = jnp.array(group_loadings)
    return group_loadings.T @ fim @ group_loadings

def _null_space_gram(Y: np.ndarray, Q: np.ndarray) -> tuple:
    """Gram matrix (s x s) of Q_N^T Y, where Q_N spans the orthogonal complement of the
    column space of the semi-orthogonal Q, and the number of rows of Q_N^T Y. The REML
    objective of the error variances depends on the data through this matrix only, so
    the (p - r) x s matrix Q_N^T Y itself is never needed."""
    Y = np.asarray(Y, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    ptilde = Q.shape[0] - Q.shape[1]
    if ptilde <= 0:
        raise InputError(f'No residual degrees of freedom are left to estimate error variances: '
                         f'the loading matrix has rank {Q.shape[1]} (intercept included) for '
                         f'{Q.shape[0]} promoters. Use fewer motifs (or "--clustering" of '
                         '"maradoner fit"), or more promoters.')
    Y = Y - Q @ (Q.T @ Y)
    return Y.T @ Y, ptilde

def _optimize_error_variance(G_Y: np.ndarray, ptilde: int, group_inds_inv: np.ndarray,
                             group_inds: list):
    d0 = np.array([G_Y.diagonal()[inds].sum() / (ptilde * len(inds)) for inds in group_inds])
    if not np.all(np.isfinite(d0)) or np.any(d0 <= 0):
        bad = [i + 1 for i in np.where(~(d0 > 0))[0]]
        raise RuntimeError('Estimation of error variances failed: the residual variance of '
                           f'sample group(s) #{", #".join(map(str, bad[:5]))} is zero or not '
                           'finite after removing the motif loadings (constant or non-finite '
                           'expression values?).')

    def value_and_grad(d):
        return _lowrank.reml_deviance_and_grad(d, None, G_Y, ptilde, group_inds_inv,
                                               group_inds)[:2]
    opt = MetaOptimizer(lambda d: value_and_grad(d)[0], lambda d: value_and_grad(d)[1],
                        num_steps_momentum=15, value_and_grad=value_and_grad, polish=True,
                        name='error variances', warn_convergence=False)
    res = opt.optimize(d0)
    # Fisher scoring from the optimiser's solution. The information matrix is available
    # in closed form, so the iteration converges to rounding precision in a few steps;
    # the later stages are very sensitive to the error variances.
    loadings = np.zeros((len(group_inds_inv), len(group_inds)))
    for i, inds in enumerate(group_inds):
        loadings[inds, i] = 1.0
    x, value = res.x, res.fun
    for _ in range(20):
        q = 1 / x[group_inds_inv]
        P = np.diag(q) - np.outer(q, q) / q.sum()
        fim = ptilde / 2 * loadings.T @ (P * P) @ loadings
        try:
            step = np.linalg.solve(2 * fim, value_and_grad(x)[1])
        except np.linalg.LinAlgError:
            break
        x_new = x - step
        if not np.all(np.isfinite(x_new)) or np.any(x_new <= 0):
            break
        value_new = value_and_grad(x_new)[0]
        if not value_new <= value + 1e-12 * abs(value):
            break
        x, value = x_new, min(value, value_new)
        if np.max(np.abs(step) / x) < 1e-13:
            break
    res.x, res.fun = x, float(value)
    res.stationarity = float(np.max(np.abs(x * value_and_grad(x)[1])))
    res.success = res.success or res.stationarity <= STATIONARITY_TOL
    opt.warn_if_not_converged(res)
    return res

def estimate_error_variance(data: TransformedData, B_decomposition: LowrankDecomposition,
                            error_lowrank: int = 0, verbose=False) -> ErrorVarianceEstimates:
    p = B_decomposition.Q.shape[0]
    G_Y, ptilde = _null_space_gram(data.Y, B_decomposition.Q)
    res = _optimize_error_variance(G_Y, ptilde, data.group_inds_inv, data.group_inds)
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
        sigma, W, loglik = _lowrank.estimate_error_lowrank(
            G_Y, ptilde, data.group_inds_inv, data.group_inds,
            k=int(error_lowrank), sigma0=np.array(res.x), verbose=verbose)
        if verbose:
            print(f'[error-lowrank k={error_lowrank}] REML deviance '
                  f'{res.fun:.3f} -> {loglik:.3f} (gain {res.fun - loglik:+.3f})')

    fim = calc_error_variance_fim(data, sigma, n_rows=ptilde, lowrank=W)
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

def _estimate_motif_variance_mom(Y, B, ind_fix, fix_value, eps=1e-14, rcond=1e-10):
    # Gamma = (B^T B)^+ -- the Moore-Penrose inverse rather than a plain `inv`.
    # This routine only builds the method-of-moments warm start x0 for the REML
    # optimiser, so a rank-deficient B is not worth dying over. `inv` is also the
    # wrong tool here even when it succeeds: on a near-singular B^T B it returns a
    # garbage inverse whose blown-up diagonal poisons the E[Z^2] - Gamma_ii bias
    # correction for *every* motif, not just the offending ones. With the
    # pseudo-inverse the unidentified directions simply get Gamma_ii == 0 and
    # Z_i == 0, hence a zero moment estimate that is clipped to the `eps` floor.
    BTB = B.T @ B
    Gamma = np.linalg.pinv(BTB, rcond=rcond, hermitian=True)

    # W = (B^T B)^+ B^T
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
    
    # Using the ratio of column sums: g_j / g_fixed = C_j / C_fixed, where ind_fix holds
    # the indices of the samples of the reference group and C_fixed is their average.
    # The moments are bias corrected, hence they can be negative: the ratios are then
    # meaningless and all g_j are set equal instead (usable = False).
    c_fix = float(np.mean(C[ind_fix]))
    usable = c_fix > 0 and np.sum(C) > 0
    if usable:
        g_est = C * (fix_value / c_fix)
    else:
        g_est = np.full_like(C, fix_value)
    
    # sigma_i^2 = R_i / sum(g)
    sigma_sq_est = R / np.sum(g_est)
        
    return sigma_sq_est, g_est, usable

def _motif_variance_start(original_data: TransformedData, D: np.ndarray, j: int, fix: float,
                          BTB: np.ndarray, verbose=False) -> tuple:
    """Starting values (tau, nu of all groups) for the (Sigma, G) fit by the method of
    moments, with nu[j] = fix. Non-positive or extreme moment estimates are replaced by
    values relative to the scale of the others, never by an absolute constant."""
    group_inds = original_data.group_inds
    d = 1 / D[original_data.group_inds_inv]
    Y = original_data.Y
    B = original_data.B
    Y = Y - Y.mean(axis=0, keepdims=True) - Y.mean(axis=1, keepdims=True) + Y.mean()
    Y = Y * d ** 0.5
    B = B - B.mean(axis=0, keepdims=True)
    Sigma0, G0, usable = _estimate_motif_variance_mom(Y, B, np.asarray(group_inds[j]), 1e-1)
    G0 = G0 / d 
    G0 = np.array([G0[inds].mean() for inds in group_inds])
    scaler = fix / G0[j]
    G0 = G0 * scaler
    Sigma0 = Sigma0 / scaler
    # Group variances further than max_ratio from the reference one are not credible as
    # moment estimates; tau is rescaled so that sum_j tau_i nu_j / D_j keeps its value.
    max_ratio = 1e2
    total = (G0[original_data.group_inds_inv] * d).sum()
    G0 = np.clip(G0, fix / max_ratio, fix * max_ratio)
    total_clipped = (G0[original_data.group_inds_inv] * d).sum()
    Sigma0 = Sigma0 * total / total_clipped
    positive = Sigma0 > 0
    if positive.any():
        Sigma0 = np.clip(Sigma0, START_REL_FLOOR * np.median(Sigma0[positive]), None)
    else:
        # No motif shows any signal above the noise: start where a motif explains about
        # as much variance as the noise of a single promoter.
        usable = False
        Sigma0 = np.full(len(BTB), 1.0 / max(BTB.diagonal().mean() * total_clipped / len(d),
                                             np.finfo(float).tiny))
    if not usable:
        logger_print('  The method-of-moments starting values for the variances of motif '
                     'activities are not usable (no signal above the noise level in the '
                     'reference group); a neutral start is used instead.', verbose)
    return Sigma0, G0

def warn_constant_loadings(B, motif_names=None, tol=1e-12, verbose=True) -> np.ndarray:
    """Report motifs whose loadings do not vary across promoters.

    The model is fitted on H_p Y and H_p B, so a column of B that is constant across
    promoters is mapped to exactly zero: the motif explains nothing, its tau_k and
    mu_m,k are unidentified, and B^T B is singular. The usual source is the ECDF/ESF
    transform in `create`, which collapses a column holding one or two distinct values
    onto a single level -- and since which promoters survive the low-expression filter
    depends on the samples in the project, the very same motif panel can be fine for
    one set of samples and degenerate for another.
    """
    B = np.asarray(B)
    spread = B.max(axis=0) - B.min(axis=0)
    scale = np.maximum(np.abs(B).max(axis=0), 1.0)
    bad = np.where(spread <= tol * scale)[0]
    if not len(bad):
        return bad
    if motif_names is not None:
        names = [str(motif_names[i]) for i in bad]
    else:
        names = [str(i) for i in bad]
    shown = ', '.join(names[:10]) + (', ...' if len(names) > 10 else '')
    warn(f'{len(bad)} motif(s) have loadings that are constant across all {B.shape[0]} promoters: '
         f'{shown}. Promoter-wise centering annihilates such columns, so these motifs carry no '
         'information: their variance and mean estimates are arbitrary and B^T B is rank '
         'deficient. Re-create the project with a current "maradoner create", which drops them '
         'at the transform step.')
    return bad

def _zero_variances(x: np.ndarray, m: int, fix: float) -> np.ndarray:
    # Boolean mask of the variances (tau, free nu) sitting at their lower bound, zero.
    scale = np.append(np.full(m, np.max(x[:m], initial=0.0)),
                      np.full(len(x) - m, max(np.max(x[m:], initial=0.0), fix)))
    return x <= ZERO_REL_TOL * scale

def _kkt_violations(x: np.ndarray, grad: np.ndarray, fim: np.ndarray, m: int,
                    fix: float) -> np.ndarray:
    """Indices of the variances that sit at zero although the likelihood increases when
    they leave it. A Newton step from zero decreases the objective (-2 logL, whose
    expected Hessian is twice the Fisher information) by about grad^2 / (4 * fim_ii)."""
    info = np.asarray(fim).diagonal()
    with np.errstate(divide='ignore', invalid='ignore'):
        gain = np.where((grad < 0) & (info > 0), grad ** 2 / (4 * info), 0.0)
    return np.where(_zero_variances(x, m, fix) & (gain > KKT_GAIN_TOL))[0]

def _check_motifs_objective(value: float, Z: np.ndarray, BTB: np.ndarray, BY: np.ndarray,
                            stage: str):
    """loglik_motifs cannot be lower than -tr((B^T B)^+ B^T Y P Y^T B), its value when
    every direction of the loadings is fully explained. A lower value means that the
    optimiser went to parameters (variance ratios of 1e12 and beyond) at which the
    likelihood can no longer be evaluated accurately."""
    lower = -float(np.trace(np.linalg.pinv(np.asarray(BTB), hermitian=True) @ Z @ BY.T))
    if not np.isfinite(value) or value < lower - 1e-6 * abs(lower):
        raise RuntimeError(f'Estimation of {stage} failed: the optimiser diverged to parameter '
                           'values at which the likelihood cannot be evaluated accurately '
                           f'(objective {value}, its theoretical minimum is {lower}).')

def _names_or_numbers(names, inds) -> list:
    if names is not None:
        return [str(names[i]) for i in inds]
    return [f'#{i + 1}' for i in inds]

def _list_some(items, limit=5) -> str:
    items = list(items)
    return ', '.join(items[:limit]) + (f' ... and {len(items) - limit} more'
                                       if len(items) > limit else '')

def _fim_scaled_eigenvalues(fim) -> np.ndarray:
    """Eigenvalues of an information matrix scaled to a unit diagonal. The scaling makes
    them independent of the units of the parameters (nu follows the units of the data,
    tau does not), and it does not change whether the matrix is singular. NaNs are
    returned if they cannot be computed."""
    fim = np.asarray(fim, dtype=np.float64)
    if not np.all(np.isfinite(fim)):
        return np.full(len(fim), np.nan)
    info = fim.diagonal()
    scale = np.where(info > 0, info, 1.0) ** -0.5
    try:
        return np.linalg.eigvalsh(fim * scale * scale.reshape(-1, 1))
    except np.linalg.LinAlgError:
        return np.full(len(fim), np.nan)

def _fim_is_singular(fim, verbose=False) -> tuple:
    # (singular, number of unidentified directions); singular means not positive definite
    # up to rounding errors.
    eig = _fim_scaled_eigenvalues(fim)
    if not np.all(np.isfinite(eig)):
        return True, len(eig)
    tol = len(eig) * 1e-12 * np.abs(eig).max(initial=0.0)
    logger_print('Smallest eigenvalue of the Fisher information of the variances (scaled to '
                 f'a unit diagonal): {eig.min()}', verbose)
    return bool(eig.min() <= tol), int(np.sum(eig <= tol))

def _optimize_motif_variances(problem, opt: MetaOptimizer, x0: np.ndarray, j: int, fix: float,
                              verbose: bool = False) -> OptimizerResult:
    """Variances of motif activities x = (tau, nu without nu[j]) by Fisher scoring and L-BFGS-B
    in the variances themselves (lowrank_sigma.estimate_motif_covariance without families).
    If they stop short of an optimum, the MetaOptimizer opt goes on from their result."""
    m = problem.m
    x0 = np.asarray(x0, dtype=np.float64)
    if MOTIF_VARIANCE_OPTIMIZER == 'meta':
        return opt.optimize(x0)
    try:
        est = _ls.estimate_motif_covariance(problem, 0, x0[:m], np.insert(x0[m:], j, fix), None, j, fix,
                                            verbose=verbose, silent=True, stage=opt.name,
                                            tag='motif variances')
    except RuntimeError as e:
        logger_print(f'[motif variances] {e}', verbose)
        est = None
    if est is not None and est['converged']:
        x = np.append(est['tau'], np.delete(est['nu'], j))
        stages = None if est['dev_fisher'] is None else (('Fisher scoring', est['dev_fisher']),)
        return OptimizerResult(est['dev0'], est['dev0'], est['dev0'], est['dev'], est['grad_norm'], x,
                               success=True, message=est['message'], stationarity=est['stationarity'],
                               stages=stages)
    logger_print('[motif variances] no optimum reached, the general optimiser goes on.', verbose)
    start = None if est is None else est['dev0']
    if est is not None:
        # The MetaOptimizer works with square roots, and zero is a stationary point for them.
        x0 = np.append(est['tau'], np.delete(est['nu'], j))
        x0[:m] = np.maximum(x0[:m], START_REL_FLOOR * np.max(x0[:m], initial=0.0))
        x0[m:] = np.maximum(x0[m:], START_REL_FLOOR * max(np.max(x0[m:], initial=0.0), fix))
    res = opt.optimize(x0)
    if start is not None:
        res.start_loglik = start
    return res


def estimate_motif_variance(data: TransformedData, B_decomposition: LowrankDecomposition,
                             error_variance: ErrorVarianceEstimates,
                             original_data: TransformedData = None,
                             activity_lowrank: int = 0,
                             verbose=False, motif_names=None,
                             group_names=None, families: np.ndarray = None,
                             family_names: list = None,
                             motif_lowrank: int = 0) -> MotifVarianceEstimates:
    """REML estimates of the variances of motif activities tau (`motif`) and of their
    group-wise multipliers nu (`group`). motif_names / group_names are used only to
    name motifs and groups in warnings. With families (m x f, zeros and ones), motif_lowrank
    or both, the covariance between motifs gets the structure of lowrank_sigma, estimated
    after (and starting from) the model with independent motifs."""
    multiplier = NU_REFERENCE_FRACTION
    D = np.asarray(error_variance.variance, dtype=np.float64)
    j = int(np.argsort(D)[len(D) // 2])
    fix = float(D[j] * multiplier)
    BTB = B_decomposition.V.T * B_decomposition.S ** 2 @ B_decomposition.V
    m = len(BTB)
    if motif_names is None:
        motif_names = getattr(data, 'motif_names', None)
    if group_names is None:
        group_names = getattr(data, 'group_names', None)
    
    if original_data is not None:
        Sigma0, G0 = _motif_variance_start(original_data, D, j, fix, np.asarray(BTB),
                                           verbose=verbose)
        G0 = np.delete(G0, j)
    else:
        Sigma0, G0 = jnp.ones(len(BTB), dtype=float), np.repeat(fix, len(D) - 1)

    # H_X^T (H_X Theta H_X^T)^{-1} H_X : projects out the full covariate row space X
    # (which contains 1_s), thereby removing both mu_p 1_s^T and the B M X mean.
    # With a low-rank error correction, Theta = D + W W^T (else Theta = D).
    W_err = _error_lowrank_W(error_variance)
    D_n = jnp.asarray(D)[data.group_inds_inv]
    covariate_names = getattr(data, 'covariate_names', None)
    if W_err is None:
        D_product_inv = covariate_projection(D_n, data.X, names=covariate_names)
    else:
        D_product_inv = _lowrank.covariate_projection_theta(D_n, W_err, data.X,
                                                            names=covariate_names)
    BY = np.asarray(data.B.T @ data.Y)
    Z = BY @ D_product_inv

    x0 = np.append(Sigma0, G0)
    fun = partial(loglik_motifs, Z=Z, BTB=BTB, D_product_inv=D_product_inv,
                  group_inds_inv=data.group_inds_inv, G_fix_ind=j, G_fix_val=fix)
    grad = partial(loglik_motifs_grad, Z=Z, BTB=BTB, D_product_inv=D_product_inv,
                  group_inds_inv=data.group_inds_inv, group_inds=data.group_inds,
                  G_fix_ind=j, G_fix_val=fix)
    value_and_grad = partial(_loglik_motifs_value_and_grad, Z=jnp.asarray(Z),
                             BTB=jnp.asarray(BTB), D_product_inv=jnp.asarray(D_product_inv),
                             group_inds_inv=jnp.asarray(data.group_inds_inv),
                             num_groups=len(data.group_inds), G_fix_ind=j, G_fix_val=fix)
    fun = jax.jit(fun)
    grad = jax.jit(grad)
    opt = MetaOptimizer(fun, grad, num_steps_momentum=50 if original_data is None else 0,
                        value_and_grad=value_and_grad, polish=True, warn_convergence=False,
                        scaling_set=[slice(0, m), slice(m, None)],
                        name='variances of motif activities')
    # The same deviance, its gradient and the Fisher information through the eigendecomposition
    # of K = C^T diag(tau) C (B^T B = C C^T); see lowrank_sigma.
    problem = _ls.MotifCovarianceProblem(Z, B_decomposition, D_product_inv, data.group_inds_inv,
                                         data.group_inds)
    res = _optimize_motif_variances(problem, opt, x0, j, fix, verbose)
    s = len(data.group_inds_inv)

    def unpack(x):
        x = np.asarray(x, dtype=np.float64)
        return x[:m], np.zeros(m), np.zeros((m, 0)), np.insert(x[m:], j, fix), np.zeros((s, 0))

    def fim(x):
        return problem.fim(*unpack(x), fix_ind=j)

    def gradient(x):
        g_tau, _, _, g_nu, _ = problem.value_and_grad(*unpack(x))[1]
        return np.append(g_tau, np.delete(g_nu, j))
    # A variance at zero, although the likelihood increases away from it, is released and the
    # optimisation starts again.
    for _ in range(3):
        grad_x = gradient(res.x)
        f = fim(res.x)
        violations = _kkt_violations(res.x, grad_x, f, m, fix)
        if not len(violations):
            break
        logger_print(f'  {len(violations)} variance(s) stopped at zero although the likelihood '
                     'increases away from it; restarting the optimisation from a perturbed '
                     'point.', verbose)
        x = res.x.copy()
        x[violations] = START_REL_FLOOR * np.where(violations < m, np.max(x[:m]),
                                                   max(np.max(x[m:], initial=0.0), fix))
        res_restart = _optimize_motif_variances(problem, opt, x, j, fix, verbose)
        if not res_restart.fun < res.fun:
            break
        res_restart.start_loglik = res.start_loglik
        res = res_restart
    _check_motifs_objective(res.fun, np.asarray(Z), BTB, BY, opt.name)
    opt.warn_if_not_converged(res)
    if verbose:
        print('-' * 15)
        print(res)
        print('-' * 15)
    Sigma = res.x[:len(BTB)]
    G = res.x[len(BTB):]
    
    G = np.insert(G, j, fix)
    zero = _zero_variances(res.x, m, fix)
    zero_tau = np.where(zero[:m])[0]
    zero_nu = np.where(np.insert(zero[m:], j, False))[0]
    if len(zero_tau) == m:
        warn('The fit of the variances of motif activities is degenerate: all of them were '
             'estimated as zero, i.e. no motif activity varies between samples beyond the '
             'noise. Activities, their tests and the group variances are meaningless for '
             'this fit. Check that expression and loadings refer to the same promoters and '
             'that there are enough samples.')
    elif 2 * len(zero_tau) > m:
        warn(f'{len(zero_tau)} of {m} variances of motif activities were estimated as zero '
             f'({_list_some(_names_or_numbers(motif_names, zero_tau))}): the data carry little '
             'signal about most motifs. With the default filter "maradoner predict" does not '
             'predict activities for them; consider fewer motifs ("--clustering") or '
             '"--no-motif-variance".')
    if len(zero_nu):
        warn('The variance of motif activities of sample group(s) '
             f'{_list_some(_names_or_numbers(group_names, zero_nu))} was estimated as zero: '
             'predicted activities of these groups are shrunk to the motif means and their '
             'tests are uninformative. The groups may be too small or too noisy; consider '
             'merging them with similar groups ("--sample-groups").')
    large_nu = np.where(G > NU_RATIO_WARN * fix)[0]
    if len(large_nu):
        warn('The variance of motif activities of sample group(s) '
             f'{_list_some(_names_or_numbers(group_names, large_nu))} is more than '
             f'{NU_RATIO_WARN:g} times that of the reference group '
             f'{_names_or_numbers(group_names, [j])[0]}: the group variances are poorly '
             'identified (weak signal or too few samples per group), and so are the '
             'activities and tests of these groups.')
    eig = _fim_scaled_eigenvalues(f)
    if not np.all(np.isfinite(eig)) or eig.min() < -1e-8 * np.abs(eig).max():
        # A numerically indefinite information matrix: evaluate it at slightly positive
        # variances and keep the best-conditioned one.
        eig = list()
        epsilons =  [1e-23, 1e-18, 1e-15, 1e-12, 1e-9, 1e-8,
                     1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
        for eps in epsilons:
            x = res.x.copy()
            x[:len(BTB)] = np.clip(x[:len(BTB)], eps, float('inf'))
            f = fim(x)
            eig.append(np.linalg.eigvalsh(f).min())
            logger_print(f'Fisher information with variances not below {eps}: '
                         f'smallest eigenvalue {eig[-1]}', verbose)
            if eig[-1] > 0:
                break
        i = np.argmax(eig)
        eps = epsilons[i]
        x = res.x.copy()
        x[:len(BTB)] = np.clip(x[:len(BTB)], eps, float('inf'))
        best_eig = eig[i]
        fim = fim(x)
    else:
        fim = f
    V_act = None
    loglik = res.fun
    G_root = np.diag(G[data.group_inds_inv] ** 0.5)
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
        if np.asarray(V_act).shape[1]:
            # The information and the likelihood ratios below refer to the refined model,
            # with the low-rank factor V held fixed.
            V_act = np.asarray(V_act)
            fim = problem.fim(np.asarray(Sigma), np.zeros(m), np.zeros((m, 0)), np.asarray(G), V_act,
                              fix_ind=j)
            G_root = np.append(np.diag(G[data.group_inds_inv] ** 0.5), V_act, axis=1)
    structure = None
    if families is not None or (motif_lowrank and motif_lowrank > 0):
        # The covariance between motifs S = diag(tau) + F (diag(omega) + L L^T) F^T, refined
        # jointly with nu (and V) from the model with independent motifs.
        problem = _ls.MotifCovarianceProblem(Z, B_decomposition, D_product_inv, data.group_inds_inv,
                                             data.group_inds, F=families)
        V_start = None if V_act is None or not np.asarray(V_act).shape[1] else np.asarray(V_act)
        est = _ls.estimate_motif_covariance(problem, int(motif_lowrank or 0), Sigma0=Sigma, nu0=G,
                                            V0=V_start, fix_ind=j, fix_val=fix, verbose=verbose)
        if verbose:
            print(f'[motif covariance] Sigma/G REML deviance {loglik:.3f} -> {est["dev"]:.3f} '
                  f'(gain {loglik - est["dev"]:+.3f})')
        Sigma, G, loglik = est['tau'], est['nu'], est['dev']
        V_fit = est['V']
        V_act = V_fit if V_fit.shape[1] else None
        fim = problem.fim(Sigma, est['omega'], est['L'], G, V_fit, fix_ind=j)
        lrt, family_lrt = problem.lrt(Sigma, est['omega'], est['L'], G, V_fit)
        structure = dict(families=None if families is None else np.asarray(families, dtype=np.float64),
                         family_names=family_names,
                         family_variance=None if families is None else est['omega'],
                         family_lowrank=est['L'] if est['L'].shape[1] else None,
                         family_free=None if families is None else problem.omega_free,
                         family_lrt=family_lrt)
    singular, n_bad = _fim_is_singular(fim, verbose=verbose)
    if singular:
        # The information matrix is rank deficient. Spell out why rather than leaving the
        # user with a bare negative eigenvalue (and a Cholesky failure in `export`).
        info = variance_dof(data.X, data.group_inds)
        if not info['ok']:
            cause = (f'Cause: only {info["dof"]} residual sample d.o.f. ({info["n_samples"]} '
                     f'samples - rank {info["rank_x"]} covariate design) are available for '
                     f'{info["n_groups"]} group variances -- see the warning above.')
        elif len(B_decomposition.S) < m:
            cause = (f'Cause: the loading matrix has rank {len(B_decomposition.S)} for {m} '
                     'motifs, so the variances of collinear motifs cannot be told apart.')
        elif structure is not None and structure['family_free'] is not None:
            cause = ('Cause: the motif families may overlap so that the variance of a family is a '
                     'combination of the variances of other families (e.g. families {a, b, c}, '
                     '{a, b}, {a, c}, {b, c}), or the motifs of a family may have collinear loadings.')
        else:
            cause = ('The degrees-of-freedom budget looks adequate, so this is more likely a '
                     'numerical or convergence issue.')
        warn(f'The Fisher information for (tau, nu) is singular: {n_bad} of '
             f'{len(np.asarray(fim))} directions are not identified. {cause} Treat the '
             'standard errors of the variance parameters and the tests based on them with '
             'caution.')
    if structure is None:
        lrt = loglik_motifs_lrt(Sigma, G_root, Z, BTB, D_product_inv)
    lrt[Sigma <= ZERO_REL_TOL * np.max(Sigma)] = 0.0
    Sigma = np.maximum(Sigma, ZERO_REL_VALUE * np.max(Sigma))
    G = np.maximum(G, ZERO_REL_VALUE * fix)
    return MotifVarianceEstimates(motif=np.array(Sigma), group=np.array(G), fim=np.array(fim),
                                  fixed_group=j, loglik_start=res.start_loglik,
                                  loglik=loglik,
                                  lowrank=(np.asarray(V_act) if V_act is not None
                                           and np.asarray(V_act).shape[1] else None),
                                  lrt=lrt, **(structure or dict()))


def estimate_motif_variance_identity(data: TransformedData, B_decomposition: LowrankDecomposition,
                                     error_variance: ErrorVarianceEstimates,
                                     verbose=False) -> MotifVarianceEstimates:
    D = jnp.array(error_variance.variance)
    BTB = B_decomposition.V.T * B_decomposition.S ** 2 @ B_decomposition.V
    W_err = _error_lowrank_W(error_variance)
    D_n = D.at[data.group_inds_inv].get()
    covariate_names = getattr(data, 'covariate_names', None)
    if W_err is None:
        D_product_inv = covariate_projection(D_n, data.X, names=covariate_names)
    else:
        D_product_inv = _lowrank.covariate_projection_theta(D_n, W_err, data.X,
                                                            names=covariate_names)

    BY = np.asarray(data.B.T @ data.Y)
    Z = BY @ D_product_inv
    # Start where the motifs explain about as much variance as the noise of one promoter
    x0 = np.asarray(D, dtype=np.float64) / max(float(np.mean(np.diagonal(BTB))),
                                               np.finfo(float).tiny)
    fun = partial(loglik_motifs, Z=Z, BTB=BTB, D_product_inv=D_product_inv,
                  group_inds_inv=data.group_inds_inv, drop_sigma=True)
    grad = partial(loglik_motifs_grad, Z=Z, BTB=BTB, D_product_inv=D_product_inv,
                  group_inds_inv=data.group_inds_inv, group_inds=data.group_inds,
                  drop_sigma=True)
    value_and_grad = partial(_loglik_motifs_value_and_grad, Z=jnp.asarray(Z),
                             BTB=jnp.asarray(BTB), D_product_inv=jnp.asarray(D_product_inv),
                             group_inds_inv=jnp.asarray(data.group_inds_inv),
                             num_groups=len(data.group_inds), drop_sigma=True)
    fun = jax.jit(fun)
    grad = jax.jit(grad)
    opt = MetaOptimizer(fun, grad, num_steps_momentum=50, value_and_grad=value_and_grad,
                        polish=True, name='group variances of motif activities')
    res = opt.optimize(x0)
    _check_motifs_objective(res.fun, np.asarray(Z), BTB, BY, opt.name)
    
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
    _fim_is_singular(fim, verbose=verbose)
    G = np.maximum(G, ZERO_REL_VALUE * np.max(G))
    return MotifVarianceEstimates(motif=np.array(Sigma), group=np.array(G), fim=np.array(fim),
                                  fixed_group=None, loglik_start=res.start_loglik,
                                  loglik=res.fun)

def estimate_motif_mean(data: TransformedData, B_decomposition: LowrankDecomposition,
                         error_variance: ErrorVarianceEstimates,
                         motif_variance: MotifVarianceEstimates,
                         promoter_mean: PromoterMeanEstimates,
                         verbose: bool = True) -> MotifMeanEstimates:
    """BLUE/GLS estimate of the motif-mean matrix M (m x c) in the model
    U ~ MN(M X, Sigma, G). Generalizes the classic mu_m (c == 1, X == 1_s^T).

    Solves vec(M) = (W^T S^{-1} W)^{-1} W^T S^{-1} vec(Ytilde) with W = Xt^T (x) B and
    S = Ghat (x) (B Sigma B^T) + I, exploiting the eigenstructure of sqrt(Sigma) B^T B
    sqrt(Sigma) so that no n*p sized objects are ever formed. With the low-rank corrections
    Theta = diag(D) + W W^T (--error-lowrank) and G = diag(nu) + V V^T (--activity-lowrank),
    the samples are decorrelated first, and Ghat holds the eigenvalues of
    Theta^{-1/2} G Theta^{-1/2}. Sigma may be the full covariance between motifs
    (--motif-families, --motif-lowrank). The stored Fisher
    information `fim` is mc x mc with column-major vec(M) ordering, so the intercept
    (mu_m) sub-block is fim[:m, :m].

    Only mu_p + B mu_m is identified, and the promoter means are estimated under the
    minimum-norm convention (orthogonal to the transformed loadings B_c). The intercept
    column of M therefore estimates mu_m + (B_c^T B_c)^+ B_c^T xi, where xi are the
    transformed promoter baselines, and pinv(fim) does not cover that second term.
    Regarding the baselines as random with i.i.d. entries of variance omega^2, its
    covariance is stored as `baseline_cov`:
        omega^2 (B_c^T B_c)^+ K (B_c^T B_c)^+,   K = B_c^T N A^2 N^T B_c,
    where A = diag(promoter weights) and N is the transform applied to promoters, so
    that K = B_c^T B_c and baseline_cov = omega^2 (B_c^T B_c)^+ without promoter weights.
    omega^2 is estimated from the promoter means, whose squared norm has the expectation
    omega^2 tr((I - P_Bc) N A^2 N^T) + (p - 1 - rank B_c) / (1^T Theta^-1 1); the last
    term is the sampling noise of the row means, already accounted for by pinv(fim).
    """
    W_err = _error_lowrank_W(error_variance)
    V_act = _activity_lowrank_V(motif_variance)
    D = jnp.array(error_variance.variance)
    Sigma = jnp.array(motif_variance.motif)
    G = jnp.array(motif_variance.group)
    mu_p = jnp.array(promoter_mean.mean)
    X = jnp.asarray(data.X, dtype=float)            # (c, n)
    c = X.shape[0]
    m = len(Sigma)
    # The covariance between motifs: diag(Sigma), or a full matrix (fit --motif-families /
    # --motif-lowrank). The symmetric root of the matrix takes the place of sqrt(Sigma) below.
    S_motif = motif_covariance(motif_variance)

    if W_err is None and V_act is None:
        d = (D ** 0.5).at[data.group_inds_inv].get()    # (n,) sqrt(D)
        Ghat = (G / D).at[data.group_inds_inv].get()    # (n,) nu/D per sample
        T = None
    else:
        # The columns of Ytilde T are independent with covariances Ghat_j B Sigma B^T + I.
        gii = np.asarray(data.group_inds_inv)
        T, Ghat, _ = _decorrelate_samples(np.asarray(D)[gii], np.asarray(G)[gii], W_err, V_act)
        T, Ghat = jnp.asarray(T), jnp.asarray(Ghat)   # (n, n), (n,)

    BTB = B_decomposition.V.T * B_decomposition.S ** 2 @ B_decomposition.V
    if np.ndim(S_motif) == 1:
        sig = jnp.sqrt(Sigma)
        A = sig.reshape(-1, 1) * BTB                 # sqrt(Sigma) B^T B  (m, m)
    else:
        sig = jnp.asarray(motif_root(S_motif))
        A = sig @ BTB

    weights = error_variance.promotor ** -0.5
    if np.std(weights) > 1e-12:
        # Weighted (promoter-variance) case: Householder transform matching weights.
        q = weights / np.linalg.norm(weights)
        mu_p_transformed = null_space_transform(q.reshape(-1, 1), mu_p.reshape(-1, 1))
    else:
        # Standard case: Helmert transform.
        mu_p_transformed = ones_nullspace_transform(mu_p.reshape(-1, 1))
    if T is None:
        Y_tilde = (data.Y - mu_p_transformed) / d       # (p-1, n)  =  (Yhat - mu_p) sqrt(D^-1)
        Xt = X / d                                       # (c, n)    =  X sqrt(D^-1)
    else:
        Y_tilde = (data.Y - mu_p_transformed) @ T        # (p-1, n)
        Xt = X @ T                                       # (c, n)

    if sig.ndim == 1:
        D_B, Q_B = jnp.linalg.eigh(sig.reshape(-1, 1) * BTB * sig)   # eig of sqrt(Sig) B^TB sqrt(Sig)
    else:
        D_B, Q_B = jnp.linalg.eigh(A @ sig)
    At_QB = A.T @ Q_B                                # (m, m), columns a_k
    BTY = data.B.T @ Y_tilde                         # (m, n)
    QY = (Q_B.T * sig if sig.ndim == 1 else Q_B.T @ sig) @ BTY   # Q_B^T sqrt(Sig) B^T Ytilde  (m, n)

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

    vecM = jnp.linalg.pinv(H, hermitian=True) @ g_flat
    M = vecM.reshape(c, m).T                         # (m, c)

    BTB = np.asarray(BTB, dtype=np.float64)
    BTB_pinv = np.linalg.pinv(BTB, hermitian=True)
    n_rows = data.B.shape[0]
    dof = n_rows - len(B_decomposition.S)
    D_n = np.asarray(D)[data.group_inds_inv]
    W_err = _error_lowrank_W(error_variance)
    if W_err is None:
        w = float(np.sum(1.0 / D_n))
    else:
        w = float(np.sum(_lowrank.theta_inv(D_n, W_err)))
    if np.std(weights) > 1e-12:
        decomp_null = LowrankDecomposition(Q=q.reshape(-1, 1), S=np.array([]), V=np.array([]))
        B_promoters = decomp_null.adjoint_null_space_transform(np.asarray(data.B, dtype=np.float64))
        K = (B_promoters * weights.reshape(-1, 1) ** 2).T @ B_promoters
        trace = float(np.sum(weights ** 2 * (1 - q ** 2)) - np.trace(BTB_pinv @ K))
        baseline_cov = BTB_pinv @ K @ BTB_pinv
    else:
        trace = float(dof)
        baseline_cov = BTB_pinv
    rss = float(np.sum(np.asarray(mu_p_transformed) ** 2))
    omega_sq = max((rss - dof / w) / trace, 0.0) if trace > 0 and dof > 0 else 0.0
    baseline_cov = omega_sq * (baseline_cov + baseline_cov.T) / 2
    return MotifMeanEstimates(np.array(M), np.array(H), baseline_cov=baseline_cov)

def estimate_mixture(Y: np.ndarray, B: np.ndarray, pi_mode=None, max_iter=100, tol=1e-5) -> np.ndarray:
    """
    Heuristic promoter weights used by `fit --promoter-variance`:
    gamma_i = 1 - exp(-v_i / median(v)), clipped to [1e-3, 1], with v_i the variance of
    promoter i across samples. Promoters that hardly vary are down-weighted. This is not
    an estimate of promoter-wise error variances (neither the REML estimator of the paper
    nor the EM below, which is not reached); B, pi_mode, max_iter and tol are not used.

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

    Note: gamma comes from the heuristic of `estimate_mixture`, so this is a
    reweighting of promoters followed by the usual REML for D. It is not the REML
    estimator of promoter-wise variances K of the model E ~ MN(0, K, D) described in
    the paper, and the stored `promotor` = 1 / gamma is not an estimate of K.
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
    G_Y, ptilde = _null_space_gram(Y_weighted, B_decomp.Q)
    
    # 4. Standard estimation on the transformed data
    # The REML objective assumes Gamma=I, which is now true for (weights * Y)
    res = _optimize_error_variance(G_Y, ptilde, data.group_inds_inv, data.group_inds)
    
    if verbose:
        print(f"Weighted Error Variance Fit: {res.fun}")

    fim = calc_error_variance_fim(data, res.x, n_rows=ptilde)
    
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
    W_err = _error_lowrank_W(error_variance)
    V_act = _activity_lowrank_V(motif_variance)
    T_back = None
    if W_err is not None or V_act is not None:
        # The sample means of the columns of Y T, independent with unit error variances and
        # the activity multipliers Ghat, are estimated, then mapped back to the samples.
        T, G, T_back = _decorrelate_samples(D, G, W_err, V_act)
        Y = Y @ T
        D = np.ones(len(G))

    p, m = B.shape
    S_motif = motif_covariance(motif_variance)
    if np.ndim(S_motif) == 1:
        sqrt_Sigma = np.sqrt(Sigma).reshape(1, -1)
        sqrt_Sigma = np.asarray(sqrt_Sigma, dtype=np.float64).ravel()
    else:
        # the symmetric root of the full covariance between motifs
        sqrt_Sigma = motif_root(S_motif)
    # With C = B sqrt(Sigma) = U S V^T, the sums over the left singular vectors u_k that
    # the estimator needs are sums of (S_k a^T u_k) (S_k u_k^T y) = (v_k^T C^T a) (v_k^T C^T y),
    # so the m x m eigendecomposition of C^T C replaces the SVD of the p x m matrix.
    if sqrt_Sigma.ndim == 1:
        S_sq, V = np.linalg.eigh(sqrt_Sigma.reshape(-1, 1) * np.asarray(B.T @ B, dtype=np.float64) * sqrt_Sigma)
    else:
        S_sq, V = np.linalg.eigh(sqrt_Sigma @ np.asarray(B.T @ B, dtype=np.float64) @ sqrt_Sigma)
    S_sq = np.clip(S_sq, 0.0, None)
    G = np.asarray(G, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)

    if sqrt_Sigma.ndim == 1:
        a = V.T @ (sqrt_Sigma * np.asarray(B.T @ a_vec, dtype=np.float64))
    else:
        a = V.T @ (sqrt_Sigma @ np.asarray(B.T @ a_vec, dtype=np.float64))
    sum_Y = np.asarray(a_vec.T @ Y, dtype=np.float64)
    a_sq_norm = np.sum(a_vec ** 2)

    VT_Y = V.T @ lmul(sqrt_Sigma, np.asarray(B.T @ Y, dtype=np.float64))
    a_sq = a ** 2

    a_VT_Y = a[:, np.newaxis] * VT_Y

    # 1 / (G S^2 + D) - 1 / D = -S^2 shrink
    shrink = G[np.newaxis, :] / (D[np.newaxis, :] * (G[np.newaxis, :] * S_sq[:, np.newaxis] + D[np.newaxis, :]))
    numerator = sum_Y / D - np.sum(a_VT_Y * shrink, axis=0)

    denominator = a_sq_norm / D - np.sum(a_sq[:, np.newaxis] * shrink, axis=0)

    mu = numerator / denominator
    if T_back is not None:
        mu = T_back @ mu

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
    # When the error covariance carries a low-rank correction, or in `--pinv` mode, the
    # per-group posterior covariances are no longer the simple diagonal-D form below; their
    # compact factors (Q, W) are precomputed (full-Theta, see lowrank.blup_group_cov) and
    # stored here.
    _cov_lowrank: list = None

    def cov_factors(self) -> tuple[np.ndarray, np.ndarray]:
        """Compact form of the per-group posterior covariances: returns (Q, W) with Q of
        shape (m, m) and W of shape (g, m) such that the covariance of the i-th group is
        Q @ diag(W[i]) @ Q.T. All groups share Q, so sums of covariances / precisions and
        their diagonals never need a dense m x m matrix per group."""
        cov_lr = getattr(self, '_cov_lowrank', None)
        if cov_lr is not None:
            M_eig, W_weights = cov_lr
            return np.asarray(M_eig, dtype=float), np.asarray(W_weights, dtype=float)
        if self._cov is None:
            raise ValueError('Posterior covariances are not available in this prediction.')
        Q_hat, S, sigma, nu, n, tau_mult = self._cov
        S = np.asarray(S, dtype=float)
        W = np.empty((len(n), len(S)), dtype=float)
        for i, (sigma_i, nu_i, n_i, tau_mult_i) in enumerate(zip(sigma, nu, n, tau_mult)):
            tau = nu_i / sigma_i * tau_mult_i
            W[i] = sigma_i / (n_i * S + 1 / tau)
        return np.asarray(Q_hat, dtype=float), W

    def cov(self) -> np.ndarray:
        """Yield the dense (m, m) posterior covariance of each group in turn."""
        Q, W = self.cov_factors()
        for w in W:
            yield (Q * w) @ Q.T

    def var(self) -> np.ndarray:
        """Posterior variances (diagonals of `cov()`), shape (g, m)."""
        Q, W = self.cov_factors()
        return W @ (Q ** 2).T

def _inv_spectrum(ev: np.ndarray, pinv: bool = False) -> np.ndarray:
    """Elementwise inverse of the eigenvalues `ev` of a symmetric matrix (one column per
    matrix). With `pinv` the Moore-Penrose convention is followed: eigenvalues below
    max(m) * eps times the largest one of their column are treated as zero."""
    if not pinv:
        return 1.0 / ev
    cutoff = ev.shape[0] * np.finfo(float).eps * np.abs(ev).max(axis=0, keepdims=True)
    keep = np.abs(ev) > cutoff
    return np.where(keep, 1.0 / np.where(keep, ev, 1.0), 0.0)


def _spectral_solve(Q_hat: np.ndarray, S: np.ndarray, rhs: np.ndarray, n: np.ndarray,
                    reg: np.ndarray, pinv: bool = False) -> np.ndarray:
    """Q_hat diag(1 / (n S + reg)) Q_hat^T rhs, column by column of `rhs`, each column with
    its own (n, reg). Costs two (m, m) x (m, c) products rather than an m^3 one per column."""
    ev = np.asarray(n, dtype=float).reshape(1, -1) * S.reshape(-1, 1) \
        + np.asarray(reg, dtype=float).reshape(1, -1)
    return Q_hat @ (_inv_spectrum(ev, pinv) * (Q_hat.T @ rhs))


def _cv_tau_losses(B: np.ndarray, Y: np.ndarray, Sigma: np.ndarray, D: np.ndarray, G: np.ndarray,
                   group_inds: list, taus: np.ndarray, cv_repeats: int, cv_splits: int,
                   pinv: bool = False) -> np.ndarray:
    """Cross-validated mean squared prediction error of every group (rows) for every tau
    multiplier in `taus` (columns). The training Gram matrix and B^T Y of a fold are the full
    ones minus the test rows' contribution, a fold is decomposed once, and all groups and
    samples are handled with matrix products; the error on the held-out rows is evaluated
    from sufficient statistics: |y - B_test u|^2 = y^T y - 2 u^T B_test^T y + u^T (B_test^T B_test) u.
    Sigma is the vector of motif variances or the full covariance between motifs."""
    m = B.shape[1]
    s = Y.shape[1]
    g = len(group_inds)
    gid = np.empty(s, dtype=int)
    for i, inds in enumerate(group_inds):
        gid[inds] = i
    sizes = np.array([len(inds) for inds in group_inds], dtype=float)
    if pinv:
        sq = np.ones(m)
    elif np.ndim(Sigma) == 1:
        sq = np.sqrt(np.asarray(Sigma, dtype=float))
    else:
        sq = motif_root(Sigma)
    with np.errstate(divide='ignore'):
        ratio = (np.asarray(D, dtype=float) / np.asarray(G, dtype=float))[gid]
    BT_B = B.T @ B
    BT_Y = B.T @ Y
    losses = np.zeros((g, len(taus)))
    n_folds = 0
    rkf = RepeatedKFold(n_repeats=cv_repeats, random_state=1, n_splits=cv_splits)
    for train_inds, test_inds in rkf.split(Y):
        B_test = B[test_inds]
        Y_test = Y[test_inds]
        G_test = B_test.T @ B_test
        BT_Y_test = B_test.T @ Y_test
        yy = np.einsum('ij,ij->j', Y_test, Y_test)
        if sq.ndim == 1:
            S, Q = jnp.linalg.eigh(sq.reshape(-1, 1) * (BT_B - G_test) * sq.reshape(1, -1))
        else:
            S, Q = jnp.linalg.eigh(sq @ (BT_B - G_test) @ sq)
        S = np.asarray(S, dtype=float)
        Q = np.asarray(Q, dtype=float)
        C = Q.T @ lmul(sq, BT_Y - BT_Y_test)
        for k, tau in enumerate(taus):
            reg = np.full(s, max(tau - 1.0, 0.0)) if pinv else ratio / tau
            U = lmul(sq, Q @ (_inv_spectrum(S.reshape(-1, 1) + reg.reshape(1, -1), pinv) * C))
            sse = yy - 2 * np.einsum('ij,ij->j', U, BT_Y_test) + np.einsum('ij,ij->j', U, G_test @ U)
            losses[:, k] += np.bincount(gid, weights=sse, minlength=g) / (len(test_inds) * sizes)
        n_folds += 1
    return losses / n_folds


def _group_mean_cov_factors(group_inds: list, s: int, D_group, G_group, W, V, Sigma, B,
                            tau_mult=None):
    """Compact factors (M_eig, W_weights) of the posterior covariance of the group-mean
    activity, i.e. of the mean over a group's samples of the per-sample activities, under
    the error covariance Theta = diag(D[group]) + W W^T and the activity sample-covariance
    G = diag(nu[group]) + V V^T (the model behind `lowrank.blup_activities` when V is given).
    The covariance of group i is M_eig @ diag(W_weights[i]) @ M_eig.T.

    Whitening both sides turns the posterior precision of the standardised activities into
    I + P^2 (x) C^T C with C = B sqrt(Sigma), P^2 = R^-1 G R^-T and Theta = R R^T; its
    spectral decomposition gives every group's block in closed form. Sigma may also be the full
    covariance between motifs; its symmetric root then takes the place of sqrt(Sigma)."""
    B = np.asarray(B, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    g = len(group_inds)
    gii = np.zeros(s, dtype=int)
    for gi, inds in enumerate(group_inds):
        gii[inds] = gi
    tau_mult = np.ones(g) if tau_mult is None else np.asarray(tau_mult, dtype=float)
    nu_n = (np.asarray(G_group, dtype=float) * tau_mult)[gii]
    Theta = np.diag(np.asarray(D_group, dtype=float)[gii])
    if W is not None and np.asarray(W).shape[1] > 0:
        Theta = Theta + np.asarray(W) @ np.asarray(W).T
    Gs = np.diag(nu_n)
    if V is not None and np.asarray(V).shape[1] > 0:
        Gs = Gs + np.asarray(V) @ np.asarray(V).T
    R = np.linalg.cholesky(Theta)
    Rinv = np.linalg.solve(R, np.eye(s))
    Gw = Rinv @ Gs @ Rinv.T
    lam_G, Q_G = np.linalg.eigh(0.5 * (Gw + Gw.T))
    lam_G = np.clip(lam_G, 0.0, None)
    sig = np.sqrt(Sigma) if Sigma.ndim == 1 else motif_root(Sigma)
    if sig.ndim == 1:
        A = sig.reshape(-1, 1) * (B.T @ B) * sig.reshape(1, -1)
    else:
        A = sig @ (B.T @ B) @ sig
    lam_B, Q_B = np.linalg.eigh(0.5 * (A + A.T))
    lam_B = np.clip(lam_B, 0.0, None)
    avg = np.zeros((g, s))
    for i, inds in enumerate(group_inds):
        avg[i, inds] = 1.0 / len(inds)
    T = (avg @ R) @ Q_G
    W_weights = (lam_G.reshape(1, -1) * T ** 2) @ (1.0 / (1.0 + lam_G.reshape(-1, 1) * lam_B.reshape(1, -1)))
    return lmul(sig, Q_B), W_weights


def predict_activities(data: TransformedData, fit: FitResult,
                       filter_motifs=True, filter_order=5,
                       tau_search=False, tau_left=0.1,  tau_right=1.0, tau_num=10,
                       clustering_search=False, k_min=0.1, k_max=0.9, k_num=6, 
                       cv_repeats=3, cv_splits=5,
                       pinv=False) -> ActivitiesPrediction:

    def _sol(BT_Y_sum, Q_hat, S, sigma, nu, n: int, tau_mult=1.0, BT_B=None):
        tau = nu / sigma * tau_mult
        if pinv:
            tau_mult = np.clip(tau_mult - 1, 0.0, a_max=float('inf'))
            sol = jnp.linalg.pinv(n * BT_B + tau_mult * jnp.identity(len(BT_B))) @ BT_Y_sum
        else:
            D = 1 / (n * S + 1 / tau)
            sol = Q_hat @ (D.reshape(-1, 1) * (Q_hat.T @ BT_Y_sum))
        return sol

    # The covariance between motifs: the vector of their variances, or a full matrix with
    # "fit --motif-families" / "--motif-lowrank".
    Sigma = motif_covariance(fit.motif_variance)
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


    if not np.isfinite(Sigma).all():
        raise InputError('The fit contains non-finite motif variances, so activities cannot be predicted. '
                         'Re-run "maradoner fit" on this project.')
    if filter_motifs:
        with np.errstate(divide='ignore'):
            log_sigma = np.log10(_ls.motif_variances(Sigma))
        inds = log_sigma >= (np.median(log_sigma) - filter_order)
        if not inds.any():
            raise InputError(f'--filter-order {filter_order} filters out all {len(Sigma)} motifs. '
                             'Use a larger value, or --no-filter-motifs.')
        B = B[:, inds]
        Sigma = _ls.subset_covariance(Sigma, inds)
        filtered_motifs = np.where(~inds)[0]
    else:
        filtered_motifs = np.zeros(0, dtype=int)
    clusters = defaultdict(list)
    if clustering_search and np.ndim(Sigma) == 2:
        raise InputError('The search for the number of clusters is not available for a fit with a covariance '
                         'between motifs (--motif-families / --motif-lowrank).')
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
        Sigma = clust * Sigma @ clust.T
        Sigma = Sigma.diagonal()
    tau_groups = defaultdict(lambda: defaultdict(list))
    if tau_search:
        # stats = defaultdict(float)
        # pi = jnp.linalg.pinv(B)
        # all_inds.extend(inds)
        taus = np.linspace(tau_left, tau_right, num=tau_num)
        losses = _cv_tau_losses(B, Y, Sigma, D, G, group_inds, taus, cv_repeats, cv_splits, pinv=pinv)
        tau_groups = {i: taus[int(np.argmin(losses[i]))] for i in range(len(group_inds))}
    else:
        tau_groups = {i: 1.0 for i in range(len(group_inds))}
    W_err = _error_lowrank_W(fit.error_variance)
    V_act = _activity_lowrank_V(fit.motif_variance)
    if V_act is not None and len(filtered_motifs):
        # V lives in sample space (s x k); independent of motif filtering -> unchanged.
        pass
    B_unscaled = B                                   # keep loadings before sqrt(Sigma)
    BT_Y = B.T @ Y
    # BT_B and its eigenvectors in the scaled (sqrt(Sigma) B) coordinates; for --pinv the
    # plain, unscaled B^T B. With a full covariance between motifs its symmetric root takes
    # the place of sqrt(Sigma).
    sq = np.ones(B.shape[1]) if pinv else motif_root(Sigma)
    if sq.ndim == 1:
        BT_B = sq.reshape(-1, 1) * (B.T @ B) * sq.reshape(1, -1)
    else:
        BT_B = sq @ (B.T @ B) @ sq
    S, Q_hat = jnp.linalg.eigh(BT_B)
    S = np.asarray(S, dtype=float)
    Q_hat = lmul(sq, np.asarray(Q_hat, dtype=float))

    sizes = [len(inds) for inds in group_inds]
    tau_mults = [tau_groups[i] for i in range(len(group_inds))]

    cov_lowrank = None
    if W_err is None and V_act is None:
        # -------- classic diagonal-D BLUP --------
        D_arr = np.asarray(D, dtype=float)
        G_arr = np.asarray(G, dtype=float)
        tau_arr = np.asarray(tau_mults, dtype=float)
        n_arr = np.asarray(sizes, dtype=float)
        with np.errstate(divide='ignore'):
            reg = np.clip(tau_arr - 1, 0.0, None) if pinv else D_arr / (G_arr * tau_arr)
        gid = np.empty(Y.shape[1], dtype=int)
        for i, inds in enumerate(group_inds):
            gid[inds] = i
        BT_Y_sum = np.stack([BT_Y[:, inds].sum(axis=-1) for inds in group_inds], axis=1)
        U = _spectral_solve(Q_hat, S, BT_Y_sum, n_arr, reg, pinv)
        U0 = _spectral_solve(Q_hat, S, BT_Y, np.ones(len(gid)), reg[gid], pinv)
        if pinv:
            # Covariance of the pseudoinverse (ridge) solution of group i: sigma_i (n_i B^T B + lam_i I)^+
            ev = S.reshape(-1, 1) * n_arr.reshape(1, -1) + reg.reshape(1, -1)
            cov_lowrank = (Q_hat, (D_arr.reshape(1, -1) * _inv_spectrum(ev, True)).T)
    else:
        # -------- full-Theta / full-G BLUP with the low-rank corrections --------
        # Joint (sample-correlated) posterior means; reduces to the loop above at W=V=0.
        if pinv:
            warn('--pinv is ignored: the fit has a low-rank correction of the error or activity '
                 'covariance, and activities are predicted with the full covariance model.')
        tau_arr = np.asarray(tau_mults, dtype=float)
        U0, U = _lowrank.blup_activities(Y, B_unscaled, np.asarray(Sigma), group_inds,
                                         np.asarray(D), np.asarray(G), W_err,
                                         tau_mult=tau_arr, V_act=V_act)
        # per-group posterior covariances (for export z-scores / ANOVA / contrasts), stored
        # as compact factors and expanded lazily. With the activity low-rank V the reported
        # group value is the mean of the per-sample activities, and so is the covariance.
        if V_act is None:
            cov_lowrank = _lowrank.blup_group_cov_factors(group_inds, np.asarray(D), np.asarray(G),
                                                          W_err, np.asarray(Sigma), B_unscaled,
                                                          tau_mult=tau_arr)
        else:
            cov_lowrank = _group_mean_cov_factors(group_inds, Y.shape[1], D, G, W_err, V_act,
                                                  Sigma, B_unscaled, tau_mult=tau_arr)
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
                 keep_motifs=False) -> tuple[np.ndarray, np.ndarray]:
    """Cluster the motifs (columns of B). Returns the cluster profiles W (rows of B x
    clusters) and the membership / mixing matrix H (clusters x motifs), B ~ W H;
    (B, None) for `ClusteringMode.none`."""
    def trs(B, labels, n):
        mx = np.zeros((n, B.shape[1]))
        for i, v in enumerate(labels):
            mx[v, i] = 1
        return mx
    if mode == ClusteringMode.none:
        return B, None
    if num_clusters is None or num_clusters < 1:
        raise InputError(f'--num-clusters must be at least 1 (got {num_clusters}).')
    if num_clusters > B.shape[1]:
        raise InputError(f'--num-clusters ({num_clusters}) exceeds the number of motifs ({B.shape[1]}). '
                         f'Choose a value between 1 and {B.shape[1]}.')
    if mode == ClusteringMode.NMF and (B < 0).any():
        raise InputError(f'--clustering NMF needs non-negative loadings, but the loading matrix has '
                         f'{int((B < 0).sum())} negative entries (minimum {B.min():.3g}). '
                         'Use --clustering KMeans, or create the project with a non-negative '
                         'loading transformation (esf, ecdf or scale).')
    if mode == ClusteringMode.KMeans:
        km = KMeans(n_clusters=num_clusters, n_init=10, random_state=0)
        km = km.fit(B.T)
        W = km.cluster_centers_.T 
        H = trs(B, km.labels_, num_clusters); 
    else:
        model = NMF(n_components=num_clusters, max_iter=1000, random_state=0)
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

def _names_list(names, limit: int = 5) -> str:
    """'a, b, c, d, e ... and 7 more' -- the first `limit` names of a (long) list."""
    names = [str(x) for x in names]
    text = ', '.join(names[:limit])
    if len(names) > limit:
        text += f' ... and {len(names) - limit} more'
    return text


def _label(names, i: int) -> str:
    return str(names[i]) if names is not None and i < len(names) else f'#{i + 1}'


def _check_groups(group_inds, n_samples: int, sample_names=None, group_names=None) -> None:
    """Every sample must belong to exactly one group."""
    membership = [list() for _ in range(n_samples)]
    for gi, inds in enumerate(group_inds):
        for j in np.asarray(inds, dtype=int).ravel():
            if not 0 <= j < n_samples:
                raise InputError(f'Group {_label(group_names, gi)} refers to sample number {j}, but the '
                                 f'project has only {n_samples} samples. Re-create the project.')
            membership[j].append(gi)
    empty = [gi for gi, inds in enumerate(group_inds) if len(inds) == 0]
    if empty:
        raise InputError(f'{len(empty)} group(s) contain no samples: '
                         f'{_names_list([_label(group_names, gi) for gi in empty])}. Check the groups given with --sample-groups.')
    more = [j for j, g in enumerate(membership) if len(g) > 1]
    none = [j for j, g in enumerate(membership) if len(g) == 0]
    if more:
        details = [f'{_label(sample_names, j)} (groups {_names_list([_label(group_names, g) for g in membership[j]], 3)})'
                   for j in more[:5]]
        raise InputError(f'{len(more)} sample(s) are assigned to more than one group: '
                         f'{"; ".join(details)}{" ..." if len(more) > 5 else ""}. Every sample must belong to '
                         'exactly one group; check the groups given with --sample-groups.')
    if none:
        raise InputError(f'{len(none)} sample(s) belong to no group: {_names_list([_label(sample_names, j) for j in none])}. '
                         'Every sample must belong to exactly one group; check the groups given with --sample-groups.')


def validate_project(data: ProjectData, n_test: int = 0, rank: bool = True) -> None:
    """Raise an `InputError` if the project cannot be fitted: non-finite values, fewer than two
    samples, an invalid group structure, covariates that use up all samples, or more motifs
    than the promoters can support (`rank`, after `n_test` promoters have been held out)."""
    Y = np.asarray(data.Y)
    B = np.asarray(data.B)
    if Y.ndim != 2 or B.ndim != 2 or len(Y) != len(B):
        raise InputError('The project is corrupt: the expression and loading tables have inconsistent shapes '
                         f'({Y.shape} and {B.shape}). Run "maradoner create" again.')
    p, s = Y.shape
    m = B.shape[1]
    if p == 0:
        raise InputError('The project contains no promoters.')
    if m == 0:
        raise InputError('The project contains no motifs (the loading matrix has no columns). '
                         'Check the loading files given to "maradoner create".')
    if s < 2:
        raise InputError(f'The project has {s} sample, but the model estimates motif activities from the '
                         'variation across samples and needs at least 2. For a single sample use '
                         '"maradoner bayes_linreg" instead.')
    sample_names = getattr(data, 'sample_names', None)
    promoter_names = getattr(data, 'promoter_names', None)
    motif_names = getattr(data, 'motif_names', None)
    _check_groups(data.group_inds, s, sample_names, getattr(data, 'group_names', None))
    for what, X, columns, column_names, hint in (
            ('expression table', Y, 'sample', sample_names, 'Remove or impute them (for example drop the affected promoters)'),
            ('loading matrix', B, 'motif', motif_names, 'Remove or fix the affected motifs or promoters')):
        finite = np.isfinite(X)
        if not finite.all():
            bad = ~finite
            cols = np.flatnonzero(bad.any(axis=0))
            rows = np.flatnonzero(bad.any(axis=1))
            raise InputError(f'The {what} contains {int(bad.sum())} non-finite value(s) (NaN or infinity), in '
                             f'{len(cols)} {columns}(s) ({_names_list([_label(column_names, j) for j in cols])}) and '
                             f'{len(rows)} promoter(s) ({_names_list([_label(promoter_names, i) for i in rows])}). '
                             f'{hint} and run "maradoner create" again.')
    X = getattr(data, 'X', None)
    if X is not None:
        X = np.asarray(X, dtype=float)
        rank_x = int(np.linalg.matrix_rank(X)) if X.size else 0
        if rank_x >= s:
            raise InputError(f'The covariate design (the intercept and {X.shape[0] - 1} covariate(s), rank {rank_x}) '
                             f'uses up all {s} samples, so no degrees of freedom are left to estimate the variances. '
                             'Use fewer covariates or more samples.')
    if s == 2 and len(data.group_inds) == 2:
        # With two samples the sample-space error covariance has a single free entry, so the
        # variances of two separate groups cannot be told apart.
        raise InputError('The 2 samples form 2 separate groups, but with 2 samples only the sum of the two error '
                         'variances is identifiable. Put both samples in one group with --sample-groups.')
    if rank:
        n_free = p - 1 - min(p - 1, m)
        if n_free <= 0:
            # the error variances are estimated from what remains of the expression after
            # removing the promoter means and everything the loadings span
            r = int(np.linalg.matrix_rank(B - B.mean(axis=0, keepdims=True)))
            n_free = p - 1 - r
        if n_free <= 0:
            held = f' (after holding out {n_test} test promoters)' if n_test else ''
            raise InputError(f'Too few promoters for {m} motifs: {p} promoters remain{held}, and once the promoter '
                             f'means and the span of the loadings are removed nothing is left to estimate the error '
                             f'variances from. Use fewer motifs, more promoters or a smaller test set, or reduce the '
                             f'motifs with --clustering KMeans --num-clusters K, with K well below {p - 1}.')


_CHROM_RE = re.compile(r'(?<![A-Za-z0-9])chr(?P<c>[0-9]+|MT|[XYM])(?![A-Za-z0-9])(?!_[A-Za-z])', re.IGNORECASE)
_CONTIG_RE = re.compile(r'(?<![A-Za-z0-9])chr(?P<c>(?:[0-9]+|MT|[XYM]|Un)_[A-Za-z][A-Za-z0-9_]*)', re.IGNORECASE)


def _normalize_chromosome(name: str) -> str:
    """'chr3', 'Chr3', '3' -> '3'; 'chrx', 'X' -> 'X'."""
    name = str(name).strip()
    if name[:3].lower() == 'chr':
        name = name[3:]
    name = name.upper()
    return str(int(name)) if name.isdigit() else name


def _chromosome_of(promoter_name: str):
    """(normalised chromosome, as written) named in a promoter name like 'chr3:1000..1100,+',
    or None."""
    m = _CHROM_RE.search(promoter_name) or _CONTIG_RE.search(promoter_name)
    return None if m is None else (_normalize_chromosome(m.group('c')), 'chr' + m.group('c'))


def select_test_promoters(promoter_names, test_chromosomes=None, test_promoters_filename=None,
                          verbose: bool = True):
    """Indices (sorted Python ints) of the promoters to hold out of the fit as a test set:
    those listed in `test_promoters_filename` (one name per line) together with those on one of
    `test_chromosomes` ('3', 'chr3', 'X' or 'chrX', in any case; the chromosome is read from
    the promoter name). None if neither is given."""
    if isinstance(test_chromosomes, str):
        test_chromosomes = [test_chromosomes]
    given = bool(test_chromosomes)
    test_chromosomes = [c for x in (test_chromosomes or ()) for c in str(x).split(',') if c.strip()]
    if given and not test_chromosomes:
        raise InputError('--test-chromosomes needs a chromosome like 3, X or chr3 (got an empty value).')
    if not test_chromosomes and not test_promoters_filename:
        return None
    names = [str(x) for x in promoter_names]
    n = len(names)
    inds = set()
    if test_promoters_filename:
        path = str(test_promoters_filename)
        try:
            with open(path, 'r') as f:
                requested = [x.strip() for x in f.read().splitlines()]
        except FileNotFoundError:
            raise InputError(f'The test promoter file "{path}" was not found.')
        except (OSError, UnicodeDecodeError) as e:
            raise InputError(f'The test promoter file "{path}" cannot be read as a text file with one promoter name per line ({e}).')
        requested = list(dict.fromkeys(x for x in requested if x))
        if not requested:
            raise InputError(f'The test promoter file "{path}" contains no promoter names '
                             '(one promoter name per line is expected).')
        known = set(names)
        found = [x for x in requested if x in known]
        if not found:
            raise InputError(f'None of the {len(requested)} names in the test promoter file "{path}" is a promoter '
                             f'of this project (names in the file: {_names_list(requested)}; promoters of the project: '
                             f'{_names_list(names, 3)}). The names must match exactly.')
        missing = [x for x in requested if x not in known]
        if missing:
            warn(f'{len(missing)} of the {len(requested)} names in "{path}" are not promoters of this project '
                 f'and are ignored: {_names_list(missing)}.', verbose)
        found = set(found)
        inds.update(i for i, name in enumerate(names) if name in found)
    if test_chromosomes:
        parsed = [_chromosome_of(name) for name in names]
        chromosomes = [None if c is None else c[0] for c in parsed]
        shown = {c[0]: c[1] for c in parsed if c is not None}
        present = sorted(shown, key=lambda c: (not c.isdigit(), int(c) if c.isdigit() else 0, c))
        if not present:
            raise InputError(f'--test-chromosomes needs promoter names that contain their chromosome, like '
                             f'"chr3:1000..1100,+", but none of the {n} promoter names does '
                             f'(for example {_names_list(names, 3)}). Use --test-promoters-filename instead.')
        requested = [_normalize_chromosome(c) for c in test_chromosomes]
        absent = [c for c in dict.fromkeys(requested) if c not in present]
        if absent:
            raise InputError(f'No promoter lies on chromosome {_names_list(absent)} (requested with --test-chromosomes). '
                             f'The chromosomes found in the promoter names are: {_names_list([shown[c] for c in present], 12)}.')
        no_chromosome = [name for name, c in zip(names, chromosomes) if c is None]
        if no_chromosome:
            warn(f'{len(no_chromosome)} promoter name(s) do not contain a chromosome and stay in the training set: '
                 f'{_names_list(no_chromosome)}.', verbose)
        wanted = set(requested)
        inds.update(i for i, c in enumerate(chromosomes) if c in wanted)
    if len(inds) >= n - 1:
        raise InputError(f'The test set holds {len(inds)} of the {n} promoters, which leaves too few '
                         'promoters for fitting the model.')
    inds = sorted(int(i) for i in inds)
    logger_print(f'Holding out {len(inds)} of {n} promoters as a test set; {n - len(inds)} promoters are used for fitting.',
                 verbose)
    return inds


def _remove_outdated_stages(project: str, stages, fmt: str, verbose: bool = True) -> None:
    """Delete the project's stage files that were made from a previous version of the stage
    just written, and say so."""
    for stage in stages:
        filename = stage_filename(project, stage, fmt)
        if os.path.isfile(filename):
            remove_stage(project, stage, fmt)
            logger_print(f'Removed outdated "{filename}" (it belongs to the previous results of this project).', verbose)


def fit(project: str, clustering: ClusteringMode,
        num_clusters: int, test_chromosomes: list, 
        gpu: bool, gpu_decomposition: bool, x64=True, true_mean=None, motif_variance: bool = True,
        promoter_variance: bool = False, test_promoters_filename: str = None,
        context_r: int = 0, drist: bool = False, error_lowrank: int = 0,
        activity_lowrank: int = 0, motif_families: str = None, motif_lowrank: int = 0,
        motif_mean_prior: str = 'none', covariate_prior: str = 'none', verbose=True,
        dump=True) -> FitResult:
    if x64:
        jax.config.update("jax_enable_x64", True)
    if (error_lowrank or 0) < 0 or (activity_lowrank or 0) < 0 or (motif_lowrank or 0) < 0:
        raise InputError('--error-lowrank, --activity-lowrank and --motif-lowrank are ranks and cannot '
                         'be negative.')
    data = read_init(project)
    validate_project(data, rank=False)
    check_design_rank(data.X, data.covariate_names)
    warn_constant_loadings(data.B, data.motif_names, verbose=verbose)
    families = family_names = None
    if motif_families is not None or motif_lowrank:
        if clustering != ClusteringMode.none:
            raise InputError('--motif-families and --motif-lowrank describe the covariance between motifs '
                             'and cannot be combined with --clustering: the latter replaces the motifs by '
                             'clusters.')
        if not motif_variance:
            warn('--motif-families and --motif-lowrank require per-motif variances; ignored because '
                 '--no-motif-variance was set.')
        elif motif_families is not None:
            families, family_names = _ls.read_families(motif_families, data.motif_names, verbose=verbose)
            if not motif_lowrank and not np.any(families.sum(axis=0) > 1):
                warn('Every motif family has a single motif, and the variance of such a family cannot '
                     'be told apart from that of its motif: the families are ignored. Pass families of '
                     'several motifs, or --motif-lowrank for a covariance between them.')
                families = family_names = None
    motif_mean_prior = str(getattr(motif_mean_prior, 'value', motif_mean_prior) or 'none')
    if motif_mean_prior not in _mean_prior.KINDS:
        raise InputError(f'Unknown --motif-mean-prior "{motif_mean_prior}": choose one of '
                         f'{", ".join(_mean_prior.KINDS)}.')
    mean_prior_families = None
    if motif_mean_prior == 'families':
        if motif_families is None or clustering != ClusteringMode.none:
            raise InputError('--motif-mean-prior families needs --motif-families and cannot be combined '
                             'with --clustering.')
        mean_prior_families = families if families is not None else \
            _ls.read_families(motif_families, data.motif_names, verbose=False)[0]
    covariate_prior = str(getattr(covariate_prior, 'value', covariate_prior) or 'none')
    if covariate_prior not in _mean_prior.COVARIATE_PRIORS:
        raise InputError(f'Unknown --covariate-prior "{covariate_prior}": choose one of '
                         f'{", ".join(_mean_prior.COVARIATE_PRIORS)}.')
    if covariate_prior != 'none' and motif_mean_prior == 'none':
        raise InputError('--covariate-prior needs --motif-mean-prior: the coefficients of the covariates get '
                         'a prior of the same kind.')
    fmt = data.fmt
    group_names = data.group_names
    promoter_inds_to_drop = select_test_promoters(data.promoter_names, test_chromosomes,
                                                  test_promoters_filename, verbose=verbose)
    deferred_transforms = getattr(data, 'deferred_transforms', None)
    if deferred_transforms and clustering != ClusteringMode.none:
        # The cluster profiles stored in the fit are made of the transformed loadings.
        train = np.setdiff1d(np.arange(len(data.B)), np.asarray(promoter_inds_to_drop or [], dtype=int))
        logger_print('Fitting the DRIST transform of the loadings on the training promoters...', verbose)
        data.B = LoadingDRIST(deferred_transforms).fit(np.asarray(data.B)[train],
                                                       np.asarray(data.Y)[train]).transform(data.B)
        deferred_transforms = None
    if clustering != ClusteringMode.none:
        n_train = len(data.promoter_names) - len(promoter_inds_to_drop or ())
        if num_clusters is not None and num_clusters >= n_train - 1:
            raise InputError(f'--num-clusters ({num_clusters}) is too large for {n_train} promoters used for '
                             f'fitting: it must be smaller than {n_train - 1}.')
        logger_print('Clustering data...', verbose)
    promoter_names_train = data.promoter_names
    data.B, clustering = cluster_data(data.B, mode=clustering, 
                                      num_clusters=num_clusters)
    # cluster profiles of all promoters of the project, made before the train/test split
    clustered_B = data.B if clustering is not None else None
    if clustering is not None:
        data.motif_names = [f'cluster_{i + 1}' for i in range(data.B.shape[1])]
    
    promoter_names_test = list(np.array(promoter_names_train)[promoter_inds_to_drop or []])
    promoter_names_train = list(np.array(promoter_names_train)[np.setdiff1d(np.arange(0, len(promoter_names_train), True), 
                                                                            np.array(promoter_inds_to_drop or [], dtype=int))])
    # print('Loadings seqs [train]')
    # from .genread import read_seqs
    # X_train = read_seqs(promoter_names_train, 'susScr11.fa')
    # print('Loadings seqs [test]')
    # X_test = read_seqs(promoter_names_test, 'susScr11.fa')
    data, data_test = split_data(data, promoter_inds_to_drop)
    del data_test
    validate_project(data, n_test=len(promoter_inds_to_drop or ()))
    drist = False
    if drist:
        logger_print('Imma DRISTing right now...', verbose)
        drist = DRIST(verbose=True, init=None, share_function=False).fit(data.B, data.Y)
    else:
        drist = None
    if deferred_transforms:
        logger_print('Fitting the DRIST transform of the loadings on the training promoters...', verbose)
        drist = LoadingDRIST(deferred_transforms).fit(data.B, data.Y)
    
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
        B = np.empty((len(data_orig.B), data_orig.B.shape[1] + 1))
        B[:, :-1] = data_orig.B
        B[:, -1] = 1.0
        B_decomposition_orig = lowrank_decomposition(B, names=list(data.motif_names) + ['intercept'])
        del B
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
        del B_decomposition_orig
        if error_lowrank and error_lowrank > 0:
            W_err = _error_lowrank_W(error_variance)
            kk = 0 if W_err is None else W_err.shape[1]
            logger_print(f'  error covariance: diagonal D + rank-{kk} update '
                         f'(REML deviance {error_variance.loglik:.2f}).', verbose)
        if promoter_variance:
            if error_lowrank and error_lowrank > 0:
                warn('--error-lowrank is not jointly estimated with '
                     '--promoter-variance; the low-rank error correction is ignored '
                     'when promoter variances are estimated.')
            logger_print('Note: --promoter-variance is a heuristic, mixture-based reweighting of the promoters, '
                         'not the REML estimator of promoter variances described in the paper.', verbose)
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
            data = transform_data(data, helmert=True, loading_context=loading_context, drist=drist,
                                  loading_multipliers=loading_multipliers)
        # numerically dependent loadings have been reported for the decomposition above
        B_decomposition = lowrank_decomposition(data.B, verbose=False)
    
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
                                                      families=families, family_names=family_names,
                                                      motif_lowrank=motif_lowrank,
                                                      verbose=verbose)
        else:
            if activity_lowrank and activity_lowrank > 0:
                warn('--activity-lowrank requires per-motif variances; '
                     'ignored because --no-motif-variance was set.')
            motif_variance = estimate_motif_variance_identity(data, B_decomposition,
                                                              error_variance=error_variance,
                                                              verbose=verbose)
        if promoter_variance and getattr(motif_variance, 'lrt', None) is not None:
            # The heuristic promoter weights do not define a likelihood: the statistics would
            # not be likelihood ratios (their null distribution is far from the nominal one).
            motif_variance = replace(motif_variance, lrt=None)
        if activity_lowrank and activity_lowrank > 0 and motif_variance.lowrank is not None:
            logger_print(f'  activity covariance: diagonal G + rank-'
                         f'{motif_variance.lowrank.shape[1]} update '
                         f'(REML deviance {motif_variance.loglik:.2f}).', verbose)
        if getattr(motif_variance, 'families', None) is not None or \
           getattr(motif_variance, 'family_lowrank', None) is not None:
            omega = getattr(motif_variance, 'family_variance', None)
            L_fam = getattr(motif_variance, 'family_lowrank', None)
            parts = list()
            if omega is not None:
                parts.append(f'{int(np.sum(omega > 0))} of {int(np.sum(motif_variance.family_free))} '
                             'family variances above zero')
            if L_fam is not None:
                what = 'families' if motif_variance.families is not None else 'motifs'
                parts.append(f'rank-{L_fam.shape[1]} covariance between {what}')
            logger_print(f'  covariance between motifs: diagonal Sigma + {" + ".join(parts) or "nothing"} '
                         f'(REML deviance {motif_variance.loglik:.2f}).', verbose)
        
        logger_print('Estimating motif means...', verbose)
        motif_mean = estimate_motif_mean(data, B_decomposition, error_variance=error_variance,
                                          motif_variance=motif_variance,
                                          promoter_mean=promoter_mean, verbose=verbose)
        if motif_mean_prior != 'none':
            motif_mean, delta = _mean_prior.shrink_motif_means(motif_mean, motif_mean_prior,
                                                               motif_variance=motif_variance,
                                                               families=mean_prior_families,
                                                               verbose=verbose, covariates=covariate_prior,
                                                               covariate_names=getattr(data_orig, 'covariate_names', None))
            # The promoter means take back what the prior removed from B mu_m: mu_p + B M X,
            # and with it everything estimated from the residuals, stays the same.
            promoter_mean = PromoterMeanEstimates(np.asarray(promoter_mean.mean).reshape(-1) +
                                                  np.asarray(data_orig.B) @ delta)
        # the Helmert-transformed copies and the decomposition are no longer needed
        del data, B_decomposition
        logger_print('Estimating sample means...', verbose)
        sample_mean = estimate_sample_mean(data_orig, error_variance=error_variance, 
                                           motif_variance=motif_variance, motif_mean=motif_mean,
                                           promoter_mean=promoter_mean)

    promoter_mean = PromoterMeanEstimates(promoter_mean.mean.flatten() * error_variance.promotor ** 0.5)
    res = FitResult(error_variance=error_variance, motif_variance=motif_variance,
                    motif_mean=motif_mean, promoter_mean=promoter_mean,
                    sample_mean=sample_mean, clustering=clustering, clustered_B=clustered_B,
                    group_names=group_names, promoter_inds_to_drop=promoter_inds_to_drop,
                    loading_context=loading_context, drist=drist, loading_multipliers=loading_multipliers)    
    if dump:
        dump_stage(res, project, 'fit', fmt)
        _remove_outdated_stages(project, ('predict', 'fov', 'promoter_mean', 'promvar'), fmt,
                                verbose=verbose)
    return res

def split_data(data: ProjectData, inds: list) -> tuple[ProjectData, ProjectData]:
    """Split the promoters into (training, test): the promoters with the indices `inds` form
    the test set. The test part is None if `inds` is None or empty."""
    if inds is None or len(inds) == 0:
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
                         covariate_terms=getattr(data, 'covariate_terms', None),
                         fmt=data.fmt)
    data = ProjectData(Y=Y, B=B, K=data.K, weights=data.weights,
                         group_inds=data.group_inds, group_names=data.group_names,
                         motif_names=data.motif_names, promoter_names=promoter_names,
                         motif_postfixes=data.motif_postfixes, sample_names=data.sample_names,
                         L=L, X=getattr(data, 'X', None),
                         covariate_names=getattr(data, 'covariate_names', None),
                         covariate_terms=getattr(data, 'covariate_terms', None),
                         fmt=data.fmt)
    return data_d, data

def align_fit_to_promoters(fit: FitResult, inds, n_full: int) -> FitResult:
    """Restrict the per-promoter estimates in ``fit`` to the *training* promoters.

    :func:`fit` estimates everything on the training promoters, so its per-promoter arrays
    (``error_variance.promotor``, ``promoter_mean.mean``) already have the length of the
    training set. A fit made by an older version estimated them over the FULL promoter set
    (``n_full`` entries), whereas ``predict``/``gof``/``export`` evaluate on the training half
    returned by ``split_data(data, inds)`` -- i.e. with the held-out promoter indices ``inds``
    removed.  Such arrays must be sliced the very same way, otherwise they no longer line up
    row-for-row with the (B, Y) of the split data and broadcasting fails (e.g.
    ``B * weights.reshape(-1, 1)`` in :func:`transform_data`).

    ``np.delete(arr, inds)`` mirrors exactly how ``split_data`` slices B/Y, so the result is
    row-aligned by construction.  Arrays that are not full length (a fit that already trained
    on the training split, or degenerate/scalar arrays) are left untouched, which makes this
    safe to call unconditionally and idempotent.
    """
    if inds is None or len(inds) == 0:
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

def _check_fit_matches_project(data: ProjectData, fit: FitResult) -> None:
    """The fit must have been made for this project's motifs (or clusters), groups and training promoters."""
    n_motifs = len(np.asarray(fit.motif_variance.motif))
    if n_motifs != data.B.shape[1]:
        raise InputError(f'The fit was made for {n_motifs} motifs, but the project has {data.B.shape[1]}. '
                         'Run "maradoner fit" again.')
    if len(fit.error_variance.variance) != len(data.group_inds):
        raise InputError(f'The fit was made for {len(fit.error_variance.variance)} sample groups, but the project '
                         f'has {len(data.group_inds)}. Run "maradoner fit" again.')
    for what, arr in (('error variances', fit.error_variance.promotor), ('means', fit.promoter_mean.mean)):
        if arr is not None and np.size(arr) not in (1, len(data.Y)):
            raise InputError(f'The fit holds promoter {what} for {np.size(arr)} promoters, but the project has '
                             f'{len(data.Y)} promoters to use. Run "maradoner fit" again.')

def _validate_predict_options(filter_motifs, filter_order, tau_search, cv_repeats, cv_splits,
                              tau_left, tau_right, tau_num) -> None:
    if filter_motifs and (np.isnan(filter_order) or filter_order < 0):
        raise InputError(f'--filter-order must not be negative (got {filter_order}): a negative value would keep only '
                         'the motifs whose variance lies above the median.')
    if not tau_search:
        return
    if cv_repeats < 1:
        raise InputError(f'--cv-repeats must be at least 1 (got {cv_repeats}).')
    if cv_splits < 2:
        raise InputError(f'--cv-splits must be at least 2 (got {cv_splits}).')
    if tau_num < 1:
        raise InputError(f'--tau-num must be at least 1 (got {tau_num}).')
    if not (tau_left > 0 and tau_right > 0):
        raise InputError(f'--tau-left and --tau-right must be positive (got {tau_left} and {tau_right}).')
    if tau_left > tau_right or (tau_left == tau_right and tau_num > 1):
        raise InputError(f'--tau-left ({tau_left}) must be smaller than --tau-right ({tau_right}).')

def predict(project: str, filter_motifs: bool, filter_order: int,
            tau_search: bool, cv_repeats: int, cv_splits: int,
            tau_left: float, tau_right: float, tau_num: int, pinv: bool,
            gpu: bool, x64=True,
            dump=True, verbose=True):
    if x64:
        jax.config.update("jax_enable_x64", True)
    _validate_predict_options(filter_motifs, filter_order, tau_search, cv_repeats, cv_splits,
                              tau_left, tau_right, tau_num)
    data = read_init(project)
    fmt = data.fmt
    fit: FitResult = load_stage(project, 'fit', fmt)
    data = apply_clustering(data, fit)
    validate_project(data, rank=False)
    loading_context = fit.loading_context
    loading_multipliers = fit.loading_multipliers
    drist = fit.drist
    n_full = len(data.promoter_names)
    data, _ = split_data(data, fit.promoter_inds_to_drop)
    # Fits made by older versions hold per-promoter estimates for the full promoter set; keep
    # only the training entries so they stay row-aligned with the split data used below and
    # in predict_activities.
    fit = align_fit_to_promoters(fit, fit.promoter_inds_to_drop, n_full)
    _check_fit_matches_project(data, fit)
    if tau_search and cv_splits > len(data.Y):
        raise InputError(f'--cv-splits ({cv_splits}) exceeds the number of promoters used for prediction ({len(data.Y)}).')
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
        dump_stage(activities, project, 'predict', fmt)
        _remove_outdated_stages(project, ('fov', 'promoter_mean', 'promvar'), fmt, verbose=verbose)
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
        if drops is None:
            drops = []
        if activities.clustering is None and B.shape[1] - len(drops) != activities.U_raw.shape[0]:
            raise InputError(f'The predicted activities cover {activities.U_raw.shape[0]} motifs, but the project '
                             f'and the fit have {B.shape[1] - len(drops)} (after filtering). The predict results '
                             'are out of date: run "maradoner predict" again.')
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
    fit : FitResult = load_stage(project, 'fit', fmt)
    loading_context = fit.loading_context
    drist = fit.drist
    loading_multipliers = fit.loading_multipliers
    activities : ActivitiesPrediction = load_stage(project, 'predict', fmt)
    data = apply_clustering(data, fit)
    validate_project(data, rank=False)
    n_full = len(data.promoter_names)
    data, data_test = split_data(data, fit.promoter_inds_to_drop)
    # Fits made by older versions hold per-promoter estimates for the full promoter set;
    # restrict them to the training promoters so they line up with the training `data`
    # used by calc_fov below.
    fit_train = align_fit_to_promoters(fit, fit.promoter_inds_to_drop, n_full)
    _check_fit_matches_project(data, fit_train)
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
            mu_p : np.ndarray = load_stage(project, 'promoter_mean', fmt, required=False)
            if mu_p is None:
                raise InputError(f'To compute the statistics for the test promoters, first run '
                                 f'"maradoner estimate-promoter-mean {project}" (after "maradoner predict").')
            # Held-out promoters have no weights of their own: they all get the median
            # weight of the training promoters.
            a_test = np.full(len(data_test.Y), np.median(np.asarray(fit.error_variance.promotor, dtype=float) ** (-0.5)))
            test_FOV = calc_fov(data=data_test, fit=fit, activities=activities,
                                mu_p=mu_p, a_vec=a_test)
        train_FOV = calc_fov(data=data, fit=fit_train, activities=activities)
    if data_test is None:
        test_FOV = None
        mu_p = None
    res = TestResult(train_FOV, test_FOV, grouped=use_groups)
    if dump:
        dump_stage(res, project, 'fov', fmt)
    return res
