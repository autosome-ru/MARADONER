import numpy as np
import jax.numpy as jnp
import jax
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from dataclasses import dataclass
from functools import partial
from .meta_optimizer import MetaOptimizer
from sklearn.model_selection import RepeatedKFold
from collections import defaultdict
from enum import Enum
from scipy.sparse.linalg import LinearOperator, lsmr
from scipy.linalg import pinvh as pinvh_scp
from .utils import read_init, ProjectData, logger_print, openers
import dill


class GOFStat(str, Enum):
    fov = 'fov'
    corr = 'corr'

class GOFStatMode(str, Enum):
    residual = 'residual'
    total = 'total'


@dataclass(frozen=True)
class LowrankDecomposition:
    Q: np.ndarray
    S: np.ndarray
    V: np.ndarray
    null_Q: np.ndarray

@dataclass
class TransformedData:
    Y: np.ndarray
    B: np.ndarray
    group_inds: list
    group_inds_inv: np.ndarray
    K : np.ndarray = None
    
@dataclass(frozen=True)
class ErrorVarianceEstimates:
    variance: np.ndarray
    fim: np.ndarray
    loglik: float
    loglik_start: float

@dataclass(frozen=True)
class MotifVarianceEstimates:
    motif: np.ndarray
    group: np.ndarray
    fim: np.ndarray
    fixed_group: int
    loglik: float
    loglik_start: float

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
    
def ones_nullspace(n: int):
    res = np.zeros((n - 1, n), dtype=float)
    for i in range(1, n):
        norm = (1 / i + 1) ** 0.5
        res[i - 1, :i] = -1 / i / norm
        res[i - 1, i] = 1 / norm
    return res

def lowrank_decomposition(X: np.ndarray, rel_eps=1e-9) -> LowrankDecomposition:
    svd = jnp.linalg.svd
    q, s, v = [np.array(t) for t in svd(X)]
    max_sv = max(s)
    n = len(s)
    for r in range(n):
        if s[r] / max_sv < rel_eps:
            r -= 1
            break
    r += 1
    s = s[:r]
    null_q = q[:, r:]
    q = q[:, :r]
    v = v[:r]
    return LowrankDecomposition(q, s, v, null_q)

def transform_data(data, std_y=False, std_b=False, helmert=True) -> TransformedData:
    try:
        B = data.B_orig
        Y = data.Y_orig
        group_inds = data.original_inds
    except:
        B = data.B
        Y = data.Y
        group_inds = data.group_inds
    if std_b:
        B /= B.std(axis=0, keepdims=True)
    if helmert:
        F_p = ones_nullspace(len(Y))
        Y = F_p @ Y
        B = F_p @ B
    group_inds_inv = list()
    d = dict()
    for i, items in enumerate(group_inds):
        for j in items:
            d[j] = i
    for i in sorted(d.keys()):
        group_inds_inv.append(d[i])
    group_inds_inv = np.array(group_inds_inv)
    return TransformedData(Y=Y, B=B, 
                           group_inds=group_inds,
                           group_inds_inv=group_inds_inv)

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
                  G_fix_ind=None, G_fix_val=1.0) -> float:
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
    cov = jnp.kron(D_A, D_B) + 1
    logdet = jnp.log(cov).sum()
    v = (Q_B.T * Sigma @ Z * G @ Q_A).flatten('F')
    loglik = -(v ** 2 / cov).sum() + logdet
    return loglik 

def loglik_motifs_naive(x: jnp.ndarray, Y_Fn: jnp.ndarray, B, BTB, FDF, Fn,
                        group_inds_inv):
    Sigma = x.at[:len(BTB)].get() 
    G = x.at[len(BTB):].get() 
    G = G.at[-1].set(1.0)
    G = G.at[group_inds_inv].get()
    mx = jnp.kron(Fn * G @ Fn.T, B * Sigma @ B.T)
    mx = mx + jnp.kron(FDF, jnp.identity(len(B)))
    Y_Fn = Y_Fn.reshape((-1, 1), order='F')
    return (Y_Fn.T @ jnp.linalg.inv(mx) @ Y_Fn).flatten()[0] + jnp.linalg.slogdet(mx)[1]

def loglik_motifs_grad(x: jnp.ndarray, Z: jnp.ndarray, BTB: jnp.ndarray,
                  D_product_inv: jnp.ndarray, group_inds_inv: jnp.ndarray,
                  group_inds: jnp.ndarray, G_fix_ind=None, G_fix_val=1.0) -> float:
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
    return grad 


def loglik_motifs_fim(x: jnp.ndarray, BTB: jnp.ndarray, 
                      D_product_inv: jnp.ndarray, group_inds_inv: jnp.ndarray,
                      group_inds: jnp.ndarray, G_fix_ind=None, G_fix_val=1.0) -> float:
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
    indices = (indices % len(G)) * len(Sigma) + indices // len(G)
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
    Theta = D_prod_Q * Q_A
    # V = B^TCQ_B, K^T = D_prod_Q

    @jax.jit
    def f_tau_nu(ind):
        i, j = ind
        psi_i = Psi.at[i, indices_mod].get()
        theta_j = Theta.at[j, indices_div].get()
        return (zeta * psi_i * theta_j).sum()
    
    FIM_tau_nu = jnp.zeros((len(Sigma), len(G)), dtype=float)
    FIM_tau_nu = FIM_tau_nu.at[*indices.T].set(jax.vmap(f_tau_nu)(indices))
    FIM_tau_nu = FIM_tau_nu * jnp.outer(1 / Sigma, 1 / G) / 2
    FIM_tau_nu = FIM_tau_nu @ group_loadings
    
    if G_fix_ind is not None:
        FIM_nu = jnp.delete(jnp.delete(FIM_nu, G_fix_ind, axis=0), G_fix_ind, axis=1)
        FIM_tau_nu = jnp.delete(FIM_tau_nu, G_fix_ind, axis=1)
    FIM = jnp.block([[FIM_tau, FIM_tau_nu],
                     [FIM_tau_nu.T, FIM_nu]])
    return FIM


def estimate_error_variance(data: TransformedData, B_decomposition: LowrankDecomposition,
                             verbose=False) -> ErrorVarianceEstimates:
    Y = B_decomposition.null_Q.T @ data.Y
    d0 = jnp.array([np.var(Y[:, inds]) for inds in data.group_inds])

    fun = partial(loglik_error, Qn_Y=Y, group_inds_inv=data.group_inds_inv)
    grad = partial(loglik_error_grad, Qn_Y=Y, group_inds_inv=data.group_inds_inv,
                   group_inds=data.group_inds)
    fun = jax.jit(fun)
    grad = jax.jit(grad)
    opt = MetaOptimizer(fun, grad,  num_steps_momentum=5)
    res = opt.optimize(d0)
    if verbose:
        print('-' * 15)
        print(res)
        print('-' * 15)
    fim = jax.jacrev(grad)(res.x)
    return ErrorVarianceEstimates(np.array(res.x), np.array(fim),
                                  loglik_start=res.start_loglik,
                                  loglik=res.fun)

def estimate_promoter_mean(data: TransformedData,
                            B_decomposition: LowrankDecomposition,
                            error_variance: ErrorVarianceEstimates,
                            verbose=False) -> PromoterMeanEstimates:
    
    D = error_variance.variance[data.group_inds_inv]
    Y = jnp.array(data.Y)
    F_p = jnp.array(ones_nullspace(len(Y) + 1))
    Q_N = jnp.array(B_decomposition.null_Q)
    w = (1 / D).sum()
    mean = Y @ (1 / D.reshape(-1, 1))
    mean = Q_N.T @ mean
    mean = Q_N @ mean
    mean = F_p.T @ mean
    mean = mean / w
    return PromoterMeanEstimates(mean)

def estimate_motif_variance(data: TransformedData, B_decomposition: LowrankDecomposition,
                             error_variance: ErrorVarianceEstimates,
                             verbose=False) -> MotifVarianceEstimates:
    D = jnp.array(error_variance.variance)
    BTB = B_decomposition.V.T * B_decomposition.S ** 2 @ B_decomposition.V
    d = 1 / D.at[data.group_inds_inv].get()
    d = d / d.sum() ** 0.5
    D_product_inv = jnp.outer(-d, d)
    D_product_inv = jnp.fill_diagonal(D_product_inv,
                                      D_product_inv.diagonal() + d * d.sum(),
                                      inplace=False )
    Z = data.B.T @ data.Y @ D_product_inv
    j = jnp.argsort(D)[len(D) // 2]
    _fix = D.at[j].get()
    for multiplier in [1e-2, 1, 10]:
        fix = _fix * multiplier
        x0 = jnp.append(jnp.ones(len(BTB), dtype=float), np.repeat(fix, len(D) - 1))
        fun = partial(loglik_motifs, Z=Z, BTB=BTB, D_product_inv=D_product_inv,
                      group_inds_inv=data.group_inds_inv, G_fix_ind=j, G_fix_val=fix)
        grad = partial(loglik_motifs_grad, Z=Z, BTB=BTB, D_product_inv=D_product_inv,
                      group_inds_inv=data.group_inds_inv, group_inds=data.group_inds,
                      G_fix_ind=j, G_fix_val=fix)
        fun = jax.jit(fun)
        grad = jax.jit(grad)
        opt = MetaOptimizer(fun, grad, num_steps_momentum=60,
                            # scaling_set=(slice(len(BTB)), slice(len(BTB), None))
                            )
        try:
            res = opt.optimize(x0)
        except ValueError as E:
            print(E)
            print('Failed for mult =', multiplier)
            continue
        if not np.isfinite(res.fun):
            print(res)
            print(res.x)
            break
        
        break
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
    fim = fim(res.x)
    return MotifVarianceEstimates(np.array(Sigma), np.array(G), np.array(fim),
                                  fixed_group=j, loglik_start=res.start_loglik,
                                  loglik=res.fun)

def estimate_motif_mean(data: TransformedData, B_decomposition: LowrankDecomposition,
                         error_variance: ErrorVarianceEstimates,
                         motif_variance: MotifVarianceEstimates,
                         promoter_mean: PromoterMeanEstimates) -> MotifMeanEstimates:
    D = jnp.array(error_variance.variance)
    Sigma = jnp.array(motif_variance.motif)
    G = jnp.array(motif_variance.group)
    mu_p = jnp.array(promoter_mean.mean)
    
    d = D ** 0.5
    d = d.at[data.group_inds_inv].get()
    g = G ** 0.5
    g = g.at[data.group_inds_inv].get()
    R = G ** 0.5 / D
    r = R.at[data.group_inds_inv].get()

    BTB = B_decomposition.V.T * B_decomposition.S ** 2 @ B_decomposition.V
    A = jnp.sqrt(Sigma).reshape(-1, 1) * BTB
    Fp = ones_nullspace(len(data.Y) + 1)
    Y_tilde = (data.Y - Fp @ mu_p.reshape(-1, 1)) / d
    Y_hat = jnp.sqrt(Sigma).reshape(-1,1) *  data.B.T @ Y_tilde * g / d
    D_B, Q_B = jnp.linalg.eigh(jnp.sqrt(Sigma).reshape(-1, 1) * BTB * jnp.sqrt(Sigma))
    At_QB = A.T @ Q_B
    w = (1 / d ** 2).sum()
    hat_g = (d / g) ** 2
    S = 1 / (hat_g.reshape(-1,1) + D_B) / d.reshape(-1,1) ** 2
    S = S.T
    mx = w * BTB - At_QB * S.sum(axis=-1) @ At_QB.T
    b = data.B.T @ (Y_tilde @ (1 / d).reshape(-1, 1))
    b = b - At_QB @ ((S / r) * (Q_B.T @ Y_hat)).sum(axis=-1, keepdims=True) 
    fim = mx
    mx = jnp.linalg.pinv(mx)
    mu_m = mx @ b
    return MotifMeanEstimates(np.array(mu_m), np.array(fim))

def estimate_sample_mean(data: TransformedData, error_variance: ErrorVarianceEstimates, 
                         motif_variance: MotifVarianceEstimates, promoter_mean: PromoterMeanEstimates,
                         motif_mean: MotifMeanEstimates):
    Y = data.Y
    B = data.B
    Y = Y - promoter_mean.mean.reshape(-1, 1) - B @ motif_mean.mean.reshape(-1, 1)
    
    Y = jnp.asarray(Y)
    B = jnp.asarray(B)
    Sigma = jnp.asarray(motif_variance.motif)
    G = jnp.asarray(motif_variance.group)
    D = jnp.asarray(error_variance.variance)
    G = G.at[data.group_inds_inv].get()
    D = D.at[data.group_inds_inv].get()

    p, m = B.shape

    sqrt_Sigma = jnp.sqrt(Sigma).reshape(1, -1)
    C = B * sqrt_Sigma

    U, S, _ = jnp.linalg.svd(C, full_matrices=False)

    a = jnp.sum(U, axis=0)
    UT_Y = U.T @ Y          # [m, s]
    sum_Y = jnp.sum(Y, axis=0)  # [s] 

    S_sq = S ** 2
    a_sq = a ** 2
    sum_a_sq = jnp.sum(a_sq)
    a_UT_Y = a[:, None] * UT_Y  # [m, s]

    denom_part1 = jnp.sum(a_sq[:, None] / (G[None, :] * S_sq[:, None] + D[None, :]), axis=0)
    denom_part2 = (p - sum_a_sq) / D
    denominator = denom_part1 + denom_part2

    num_part1 = jnp.sum(a_UT_Y / (G[None, :] * S_sq[:, None] + D[None, :]), axis=0)
    num_part2 = (sum_Y - jnp.sum(a_UT_Y, axis=0)) / D
    numerator = num_part1 + num_part2

    mu = numerator / denominator
    return SampleMeanEstimates(np.array(mu).reshape(-1, 1))

@dataclass(frozen=True)
class ActivitiesPrediction:
    U: np.ndarray
    U_decor: np.ndarray
    U_o: np.ndarray
    stds: np.ndarray
    filtered_motifs: np.ndarray
    tau_groups: dict
    clustering: tuple[np.ndarray, np.ndarray] = None

def predict_activities(data: TransformedData, fit: FitResult,
                       filter_motifs=True, filter_order=5,
                       tau_search=True, tau_left=0.1,  tau_right=1.0, tau_num=15,
                       clustering_search=False, k_min=0.1, k_max=0.9, k_num=6, 
                       cv_repeats=3, cv_splits=5,
                       pinv=False) -> ActivitiesPrediction:

    def _sol(BT_Y_sum, BT_B, Sigma, sigma, nu, n: int, standardize=False, tau_mult=1.0):
        tau = nu / sigma * tau_mult
        Sigma_hat = Sigma * tau
        Z = n * BT_B
        Z[np.diag_indices_from(Z)] += 1 / Sigma_hat 
        
        if pinv:
            tau_mult = np.clip(tau_mult - 1, 0.0, a_max=float('inf'))
            sol = jnp.linalg.pinv(BT_B + tau_mult * jnp.identity(len(BT_B))) @ BT_Y_sum
        else:
            sol = jax.scipy.linalg.solve(Z, BT_Y_sum,
                                         assume_a='sym')
        if standardize:
            Z = np.linalg.pinv(Z, hermitian=True)
    
            Z = sigma * Z 
            D = Z.diagonal() ** (-0.5)
            C = D.reshape(-1, 1) * Z * D
            eigh = jnp.linalg.eigh(C)
            T = D.reshape(-1, 1) * (eigh.eigenvectors * (1 / eigh.eigenvalues ** 0.5) @ eigh.eigenvectors.T)
            U_decor = T @ sol
            return sol, U_decor, Z.diagonal() ** 0.5
        return sol

    Sigma = fit.motif_variance.motif
    G = fit.motif_variance.group
    D = fit.error_variance.variance
    group_inds = data.group_inds
    
    mu_p = fit.promoter_mean.mean.reshape(-1, 1)
    mu_m = fit.motif_mean.mean.reshape(-1, 1)
    mu_s = fit.sample_mean.mean.reshape(-1, 1)
    B = data.B
    Y = data.Y
    Y = Y - mu_p - B @ mu_m - mu_s.T
    # print(B.shape, Y.shape)
    if filter_motifs:
        inds = np.log10(Sigma) >= (np.median(np.log10(Sigma)) - filter_order)
        B = B[:, inds]
        Sigma = Sigma[inds]
        mu_m = mu_m[inds]
        filtered_motifs = np.where(~inds)[0]
    else:
        filtered_motifs = list()
    # print(B.shape, Y.shape)
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
                BT_B = B_train.T @ B_train
                BT_Y = B_train.T @ Y_train
                for i, (inds, sigma, nu) in enumerate(zip(group_inds, D, G)):
                    BT_Y_sub = BT_Y[:, inds]
                    U = _sol(BT_Y_sub, BT_B, Sigma_c, sigma, nu, 1, tau_mult=1)
                    diff = ((Y_test[:, inds] - B_test @ U[:, np.argsort(inds)]) ** 2).mean()
                    clusters[n_cluster].append(diff)
        clusters = {n: np.mean(v) for n, v in clusters.items()}
    else:
        clusters = {B.shape[1]: 0} 
    print(clusters, B.shape)
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
            BT_B = B_train.T @ B_train
            BT_Y = B_train.T @ Y_train
            for tau in np.linspace(tau_left, tau_right, num=tau_num):
                # pi = jnp.linalg.pinv(B)
                for i, (inds, sigma, nu) in enumerate(zip(group_inds, D, G)):
                    # all_inds.extend(inds)
                    BT_Y_sub = BT_Y[:, inds]
                    U = _sol(BT_Y_sub, BT_B, Sigma, sigma, nu, 1, tau_mult=tau)
                    diff = ((Y_test[:, inds] - B_test @ U[:, np.argsort(inds)]) ** 2).mean()
                    tau_groups[i][tau].append(diff)
        tau_groups = {g: min(v, key=lambda x: np.mean(v[x])) for g, v in tau_groups.items()}
    else:
        tau_groups = {i: 1.0 for i in range(len(group_inds))}
    BT_B = B.T @ B
    BT_Y = B.T @ Y
    U = list()
    U_independent = list()
    U_stds = list()
    U0 = list()
    # pi = jnp.linalg.pinv(B)
    all_inds = list()
    for i, (inds, sigma, nu) in enumerate(zip(group_inds, D, G)):
        tau = tau_groups[i]
        all_inds.extend(inds)
        BT_Y_sub = BT_Y[:, inds]
        U_pred, U_decor, std = _sol(BT_Y_sub.sum(axis=-1, keepdims=True), BT_B, Sigma,
                                    sigma, nu, len(inds), standardize=True, tau_mult=tau)
        U1 = _sol(BT_Y_sub, BT_B, Sigma, sigma, nu, 1, tau_mult=tau)
        # U1 = pi @ Y[:, inds]
        U0.append(U1)
        U.append(U_pred)
        U_independent.append(U_decor)
        U_stds.append(std.reshape(-1, 1))
    U = np.concatenate(U, axis=-1)
    U_independent = np.concatenate(U_independent, axis=-1)
    U_stds = np.concatenate(U_stds, axis=-1)
    U0 = np.concatenate(U0, axis=-1)[:, np.argsort(all_inds)]
    return ActivitiesPrediction(U, U_decor=U_independent, U_o=U0,
                                stds=U_stds,
                                filtered_motifs=filtered_motifs,
                                tau_groups=tau_groups,
                                clustering=(B, clust) if clust is not None else None)

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

def fit(project: str, clustering: ClusteringMode,
        num_clusters: int, test_chromosomes: list,
        gpu: bool, gpu_decomposition: bool, x64=True, true_mean=None,
        verbose=True, dump=True) -> ActivitiesPrediction:
    if x64:
        jax.config.update("jax_enable_x64", True)
    data = read_init(project)
    fmt = data.fmt
    group_names = data.group_names
    if clustering != clustering.none:
        logger_print('Clustering data...', verbose)
    data.B, clustering = cluster_data(data.B, mode=clustering, 
                                      num_clusters=num_clusters)
    if test_chromosomes:
        test_chromosomes = tuple([c + '_' for c in test_chromosomes])
        promoter_inds_to_drop = [i for i, p in enumerate(data.promoter_names) if p.startswith(test_chromosomes)]
        data.Y = np.delete(data.Y, promoter_inds_to_drop, axis=0)
        data.B = np.delete(data.B, promoter_inds_to_drop, axis=0)
    else:
        promoter_inds_to_drop = None
    logger_print('Transforming data...', verbose)
    data_orig = transform_data(data, helmert=False)
    data = transform_data(data, helmert=True)
    if gpu_decomposition:
        device = jax.devices()
    else:
        device = jax.devices('cpu')
    device = next(iter(device))

    logger_print('Computing low-rank decompositions of the loading matrix...', verbose)
    with jax.default_device(device):
        B_decomposition = lowrank_decomposition(data.B)
    if gpu:
        device = jax.devices()
    else:
        device = jax.devices('cpu')
    device = next(iter(device))
    # print(data.B.shape, data_orig.B.shape)
    with jax.default_device(device):

        logger_print('Estimating error variances...', verbose)
        error_variance = estimate_error_variance(data, B_decomposition, 
                                                  verbose=verbose)
        
        logger_print('Estimating promoter-wise means...', verbose)
        promoter_mean = estimate_promoter_mean(data, B_decomposition,
                                               error_variance=error_variance)
        
        logger_print('Estimating variances of motif activities...', verbose)
        motif_variance = estimate_motif_variance(data, B_decomposition,
                                                  error_variance=error_variance,
                                                  verbose=verbose)
        
        logger_print('Estimating motif means...', verbose)
        motif_mean = estimate_motif_mean(data, B_decomposition, error_variance=error_variance,
                                          motif_variance=motif_variance,
                                          promoter_mean=promoter_mean)
        logger_print('Estimating sample means...', verbose)
        sample_mean = estimate_sample_mean(data_orig, error_variance=error_variance, 
                                           motif_variance=motif_variance, motif_mean=motif_mean,
                                           promoter_mean=promoter_mean)
        # logger_print('Predicting motif activities...', verbose)
        # U = predict_activities(data, B_decomposition, error_variance, motif_variance, promoter_mean, motif_mean,
        #                        cov_mode=cov_mode)
    
    res = FitResult(error_variance=error_variance, motif_variance=motif_variance,
                    motif_mean=motif_mean, promoter_mean=promoter_mean,
                    sample_mean=sample_mean, clustering=clustering,
                    group_names=group_names, promoter_inds_to_drop=promoter_inds_to_drop)    
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
    promoter_names_d = np.delete(data.promoter_names, inds)
    promoter_names = list(np.array(data.promoter_names)[inds])
    data_d = ProjectData(Y=Y_d, B=B_d, K=data.K, weights=data.weights,
                         group_inds=data.group_inds, group_names=data.group_names,
                         motif_names=data.motif_names, promoter_names=promoter_names_d,
                         motif_postfixes=data.motif_postfixes,
                         fmt=data.fmt)
    data = ProjectData(Y=Y, B=B, K=data.K, weights=data.weights,
                         group_inds=data.group_inds, group_names=data.group_names,
                         motif_names=data.motif_names, promoter_names=promoter_names,
                         motif_postfixes=data.motif_postfixes,
                         fmt=data.fmt)
    return data_d, data

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
        fit = dill.load(f)
    data, _ = split_data(data, fit.promoter_inds_to_drop)
    data = transform_data(data, helmert=False)
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
    U_m = fit.motif_mean.mean.reshape(-1, 1)
    mu_s = fit.sample_mean.mean.reshape(-1, 1)
    Y = data.Y - mu_s.T
    Y = Y - data.B @ U_m
    
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
                  stat_type: GOFStat, stat_mode: GOFStatMode, x64=True,
                  verbose=True, dump=True):
    def calc_fov(data: TransformedData, fit: FitResult,
                 activities: ActivitiesPrediction, mu_p=None) -> tuple[FOVResult]:
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
        U_m = fit.motif_mean.mean.reshape(-1, 1)
        if mu_p is None:
            mu_p = fit.promoter_mean.mean
        mu_p = mu_p.reshape(-1, 1)
        mu_s = fit.sample_mean.mean.reshape(-1, 1)
        
        Y = data.Y
        d1 = mu_p.reshape(-1, 1) + mu_s.reshape(1, -1)
        d2 = B @ U_m
        d2 = d2.repeat(len(mu_s), -1)
        # Y1 = Y0 - mu_p.reshape(-1, 1) - mu_s.reshape(1, -1)
        if use_groups:
            U = activities.U
            groups = data.group_inds
            Y = _groupify(Y, groups)
            d1 = _groupify(d1, groups)
            d2 = _groupify(d2, groups)
        else:
            U = activities.U_o
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
        fit = dill.load(f)
    with openers[fmt](f'{project}.predict.{fmt}', 'rb') as f:
        activities = dill.load(f)
    data, data_test = split_data(data, fit.promoter_inds_to_drop)
    if x64:
        jax.config.update("jax_enable_x64", True)
    data = transform_data(data, helmert=False)
    if data_test is not None:
        data_test = transform_data(data_test, helmert=False)
    if gpu:
        device = jax.devices()
    else:
        device = jax.devices('cpu')
    device = next(iter(device))
    with jax.default_device(device):

        if data_test is not None:
            drops = activities.filtered_motifs
            U = activities.U_o
            U_m = fit.motif_mean.mean.reshape(-1, 1)
            mu_s = fit.sample_mean.mean.reshape(-1, 1)
            Y = data_test.Y - mu_s.T
            Y = Y - data_test.B @ U_m
            Y = Y - np.delete(data_test.B, drops, axis=1) @ U
            D = (1 / fit.error_variance.variance)[data_test.group_inds_inv].reshape(-1, 1)
            mu_p = Y @ D / (D.sum())
            # mu_p = compute_mu_mle(data_test, fit)
            test_FOV = calc_fov(data=data_test, fit=fit, activities=activities,
                                mu_p=mu_p)
        train_FOV = calc_fov(data=data, fit=fit, activities=activities)
    if data_test is None:
        test_FOV = None
    res = TestResult(train_FOV, test_FOV, grouped=use_groups)
    with openers[fmt](f'{project}.fov.{fmt}', 'wb') as f:
        dill.dump(res, f)
    return res
        
        
        