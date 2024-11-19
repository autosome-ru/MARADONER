import numpy as np
import jax.numpy as jnp
from scipy.optimize import minimize_scalar, minimize
import jax.scipy as jscipy
import scipy
import jax
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from dataclasses import dataclass
from functools import partial
from .meta_optimizer import MetaOptimizer
from sklearn.model_selection import RepeatedKFold
from collections import defaultdict
from enum import Enum
import matplotlib.pyplot as plt
from scipy.sparse.linalg import LinearOperator, lsmr
from .utils import read_init, ProjectData, logger_print, openers
import dill


class CenteringMethod(str, Enum):
    single = 'single'
    group = 'group'
    sample_average = 'sample_average'
    
class Regularization(str, Enum):
    none = 'none'
    rank = 'rank'

class SigmaStructure(str, Enum):
    identity = 'identity'
    diag = 'diag'

def ones_nullspace(n: int):
    res = np.zeros((n - 1, n), dtype=float)
    for i in range(1, n):
        norm = (1 / i + 1) ** 0.5
        res[i - 1, :i] = -1 / i / norm
        res[i - 1, i] = 1 / norm
    return res

@dataclass(frozen=True)
class LowrankDecomposition:
    Q: np.ndarray
    S: np.ndarray
    V: np.ndarray
    null_Q: np.ndarray

@dataclass(frozen=True)
class TransformedData:
    Y: np.ndarray
    Y_orig: np.ndarray
    B: np.ndarray
    B_orig: np.ndarray
    group_inds: list
    original_inds: list
    centering_method: CenteringMethod

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

def loglik_identity_sigma(nu, YYT_diagonal, D_z, sigma, n_hat):
    Z = nu * D_z + sigma
    return jnp.sum(YYT_diagonal * (1 / Z)) + n_hat * jnp.log(Z).sum()

def loglik(Sigma, nu, sigma, YT_Y_diag_sum, BT_B, YT_B):
    loglik = YT_Y_diag_sum / sigma
    Sigma_hat = (nu * Sigma) ** 0.5
    YT_B = YT_B * Sigma_hat
    AT_A = Sigma_hat.reshape(-1, 1) * BT_B * Sigma_hat
    Z_sub = AT_A + jnp.identity(AT_A.shape[0]) * sigma
    cfac = jscipy.linalg.cho_factor(Z_sub)
    sol = jscipy.linalg.cho_solve(cfac, YT_B.T)
    loglik -= jnp.einsum('ij,ji->', sol, YT_B) / sigma
    loglik += len(YT_B) * 2 * jnp.log(cfac[0].diagonal()).sum()
    return loglik

def loglik_grad(Sigma, nu, sigma, YT_Y_diag_sum, BT_B, YT_B):
    n_hat = len(YT_B)
    Sigma_hat = (nu * Sigma) ** 0.5
    YT_B = YT_B * Sigma_hat
    AT_A = Sigma_hat.reshape(-1, 1) * BT_B * Sigma_hat
    Z_sub = jnp.linalg.inv(AT_A + jnp.identity(AT_A.shape[0]) * sigma) / Sigma_hat
    R = YT_B @ Z_sub
    g = n_hat * jnp.einsum('ij,ij->i', BT_B * Sigma_hat, Z_sub) - jnp.einsum('ij,ji->i', R.T, R)
    g = nu * g
    g_nu = jnp.sum(Sigma / nu * g)
    return g, g_nu

def loglik_groups(x: jnp.ndarray, YT_Y_diag_sums: jnp.ndarray, BT_B, YT_B, sigmas, group_inds: list):
    loglik_all = 0
    x = jnp.abs(x) 
    Sigma = x.at[:len(BT_B)].get() #* 0.1
    nus = x.at[len(BT_B):].get()
    for (nu, sigma, YT_Y_diag_sum, inds) in zip(nus, sigmas, YT_Y_diag_sums, group_inds):
        YT_B_g = YT_B.at[inds].get()
        loglik_all += loglik(Sigma, nu, sigma, YT_Y_diag_sum, BT_B, YT_B_g)
    return loglik_all

def loglik_grad_groups(x: jnp.ndarray, YT_Y_diag_sums: jnp.ndarray, BT_B, YT_B, sigmas, group_inds: list):
    grad_all = jnp.zeros_like(x)
    x = jnp.abs(x) 
    Sigma = x.at[:len(BT_B)].get() #* 0.1
    nus = x.at[len(BT_B):].get()
    for i, (nu, sigma, YT_Y_diag_sum, inds) in enumerate(zip(nus, sigmas, YT_Y_diag_sums, group_inds)):
        YT_B_g = YT_B.at[inds].get()
        g, g_nu = loglik_grad(Sigma, nu, sigma, YT_Y_diag_sum, BT_B, YT_B_g)
        grad_all = grad_all.at[i + len(Sigma)].add(g_nu)
        grad_all = grad_all.at[:len(BT_B)].add(g)
    return grad_all

def estimate_motif_variances(data: TransformedData, B_decomposition: LowrankDecomposition, 
                             sigmas, nus):
    scale_f = np.median(nus) ** 0.5
    scale_f = 1
    YT = data.Y.T
    YT = jnp.array(YT)
    S = B_decomposition.S / scale_f
    sigmas /= scale_f ** 2
    BT_B = B_decomposition.V.T * S ** 2 @ B_decomposition.V
    YT_B = YT @ B_decomposition.Q * S  @ B_decomposition.V
    YT_B = jnp.array(YT_B)
    BT_B = jnp.array(BT_B)
    YT_Y_diag_sums = list()
    for inds in data.group_inds:
        Yg = YT.at[inds].get()
        YT_Y_diag_sums.append(jnp.einsum('ij,ij->', Yg, Yg))
    YT_Y_diag_sums = jnp.array(YT_Y_diag_sums)
    Sigma = jnp.repeat(1.0, len(BT_B))
    x0 = jnp.concat((Sigma , nus / scale_f ** 2))
    fun = partial(loglik_groups,  YT_Y_diag_sums=YT_Y_diag_sums, BT_B=BT_B, YT_B=YT_B, 
                  sigmas=sigmas, group_inds=data.group_inds)
    fun = jax.jit(fun)
    grad = partial(loglik_grad_groups,  YT_Y_diag_sums=YT_Y_diag_sums, BT_B=BT_B, YT_B=YT_B, 
                  sigmas=sigmas, group_inds=data.group_inds)
    grad = jax.jit(grad)
    inds = jnp.zeros_like(x0, dtype=bool)
    inds = inds.at[:-len(data.group_inds)].set(True)
    opt = MetaOptimizer(fun, grad, scaling_set=inds)
    sol = opt.optimize(x0)
    Sigma = jnp.array(sol.x[:-len(sigmas)])
    nus = np.array(sol.x[-len(sigmas):])
    return Sigma, nus

def _calc_z_reml(sigma: jnp.ndarray, F: jnp.ndarray, indices: list) -> jnp.ndarray:
    inds = np.zeros(max(F.shape), dtype=int)
    for i, ind in enumerate(indices):
        inds[ind] = i
    inds = jnp.array(inds)
    D = sigma.at[inds].get()
    return F * D @ F.T

def loglik_reml(sigma: jnp.ndarray, YT: jnp.ndarray, F: jnp.ndarray,
                inds: list) -> jnp.ndarray:
    Z = _calc_z_reml(sigma, F, inds)
    cfac = jscipy.linalg.cho_factor(Z)
    sol = jscipy.linalg.cho_solve(cfac, YT)
    loglik = jnp.einsum('ij,ij->', sol, YT)
    loglik += YT.shape[1] * 2 * jnp.log(cfac[0].diagonal()).sum()
    return loglik

def loglik_reml_grad(sigma: jnp.ndarray, YT: jnp.ndarray, F: jnp.ndarray,
                     inds: list) -> jnp.ndarray:
    Z = _calc_z_reml(sigma, F, inds)
    A = jscipy.linalg.solve(Z, F, assume_a='pos')
    Y_A = YT.T @ A
    diag = YT.shape[1] * jnp.einsum('ji,ji->i', F, A) -\
           jnp.einsum('ji,ji->i', Y_A, Y_A)
    g = jnp.array([diag.at[ind].get().sum() for ind in inds])
    return g

def estimate_error_variances(data: TransformedData,
                             B_decomposition: LowrankDecomposition,
                             min_n_to_center=2) -> np.ndarray:
    YT = data.Y.T @ B_decomposition.null_Q# B_decomposition.null_Q.T @ data.Y
    rp = YT.shape[1] # len(Y)
    p = len(data.Y) + 1
    if data.centering_method == CenteringMethod.group:
         return np.array([(YT[inds] ** 2).mean()
                          for inds in data.group_inds])
    x0 = jnp.repeat(YT.var(axis=1).mean(), len(data.original_inds))
    F = jnp.array(ones_nullspace(YT.shape[0] + 1))
    fun = partial(loglik_reml, YT=YT, F=F, inds=data.original_inds)
    grad = partial(loglik_reml_grad, YT=YT, F=F, inds=data.group_inds)
    fun = jax.jit(fun)
    grad = jax.jit(grad)
    opt = MetaOptimizer(fun, grad,  num_steps_momentum=5)
    res = opt.optimize(x0)
    return np.array(res.x)

def estimate_nu(data: TransformedData, B_decomposition: LowrankDecomposition,
                sigmas: np.ndarray) -> np.ndarray:
    res = list()
    Y = data.Y
    Y = B_decomposition.Q.T @ Y
    D_z = B_decomposition.S ** 2
    for (sigma, inds) in zip(sigmas, data.group_inds):
        Yg = Y[:, inds]
        YYT_diagonal = jnp.einsum('ij,ij->i', Yg, Yg)
        f = partial(loglik_identity_sigma, YYT_diagonal=YYT_diagonal, D_z=D_z,
                    sigma=sigma, n_hat=Yg.shape[1])
        sol = minimize_scalar(f, bounds=(0, sigma * 10))
        res.append(sol.x)
    return np.array(res)
    

def transform_data(data, std_y=False, std_b=False, 
                   weights=None, min_n_to_center=2,
                   centering_method=CenteringMethod.single) -> TransformedData:
    try:
        B = data.B_orig
        Y = data.Y_orig
        group_inds = data.original_inds
    except:
        B = data.B
        Y = data.Y
        group_inds = data.group_inds
    original_inds = group_inds
    if std_b:
        B /= B.std(axis=0, keepdims=True)
    null_space = ones_nullspace(max(Y.shape))
    n = len(Y)
    Y_orig = Y
    B_orig = B
    if centering_method in (CenteringMethod.group, CenteringMethod.single):
        Y = null_space[:n - 1, :n] @ Y
        B = null_space[:n - 1, :n] @ B
    if centering_method == CenteringMethod.group:
        m = max(map(len, group_inds)) + 1
        null_space = null_space[:m - 1, :m]
        new_Y = list()
        new_inds = list()
        n = 0
        for i, ind in enumerate(group_inds):
            subY = Y[:, ind]
            tn = len(ind)
            if tn >= min_n_to_center:
                subY = subY @ null_space[:tn-1, :tn].T
                tn -= 1
            if std_y:
                subY /= subY.std()
            new_Y.append(subY)
            new_inds.append(np.arange(n, n + tn, dtype=int))
            n += tn
        Y = np.concatenate(new_Y, axis=1)
        group_inds = new_inds
    elif centering_method == CenteringMethod.single:
        m = Y.shape[1]
        Y = Y @ null_space[:m - 1, :m].T
    elif centering_method == CenteringMethod.sample_average:
        Y = Y - Y.mean(axis=0, keepdims=True) - Y.mean(axis=1, keepdims=True) + Y.mean()
        B = B - B.mean(axis=0, keepdims=True)

    if weights is not None:
        weights = weights.reshape(-1, 1) ** -0.5
        Y = weights * Y
        B = weights * B
    return TransformedData(Y=Y, B=B,  Y_orig=Y_orig, B_orig=B_orig,
                           group_inds=group_inds,
                           original_inds=original_inds,
                           centering_method=centering_method)

def estimate_sample_means(data: TransformedData, B_orig_decomposition: LowrankDecomposition,
                          promoter_means: np.ndarray) -> np.ndarray:
    null_Q = B_orig_decomposition.null_Q
    group_inds = data.original_inds
    if len(promoter_means.shape) < 2:
        Y = data.Y_orig - promoter_means.reshape(-1, 1)
    else:
        for inds, m in zip(group_inds, promoter_means.T):
            Y = data.Y_orig
            Y[:, inds] -= m
    return (null_Q.T @ Y).mean(axis=0)

def estimate_promoter_means(data: TransformedData,
                            B_decomposition: LowrankDecomposition,
                            sigmas: np.ndarray,
                            exact: bool,
                            single=True) -> np.ndarray:
    Y_part = jnp.array(data.Y_part)
    F_p = jnp.array(ones_nullspace(len(Y_part) + 1))
    Q_N = jnp.array(B_decomposition.null_Q)
    Y_part = (Y_part.T @ Q_N)#(Q_N.T @ Y_part).T
    group_inds = data.original_inds
    res = list()
    
    if single:
        r_sigma = jnp.zeros(Y_part.shape[0])
        for sigma, ind in zip(sigmas, group_inds):
            r_sigma = r_sigma.at[ind].set(1 / sigma)
        d = r_sigma.sum()
        b = (Y_part.T * r_sigma).sum(axis=-1, keepdims=True) / d
        if exact:
            A = Q_N.T @ F_p
            del F_p; del Q_N
            return (jnp.linalg.pinv(A) @ b).flatten()
        Q_NT = Q_N.T#; F_pT = F_p.T
        operator = LinearOperator(shape=(Q_N.shape[1], F_p.shape[1]),
                                  matvec=lambda x: Q_NT @ (F_p @ x),
                                  rmatvec=lambda x: F_p.T @ (Q_N @ x))
        return lsmr(operator, b, show=True, atol=1e-8, btol=1e-8)[0]
    A = Q_N.T @ F_p
    del F_p; del Q_N
    if exact:
        A = jnp.linalg.pinv(A)
        for ind in group_inds:
            res.append(A @ Y_part[ind].mean(axis=0, keepdims=True).T)
    else:
        Y_part = np.array(Y_part)
        A = np.array(A)
        x0 = np.zeros(A.shape[-1])
        for ind in group_inds:
            sol = lsmr(A, Y_part[ind].mean(axis=0).reshape(-1,1), x0=x0)[0]
            res.append(sol.reshape(-1,1))
    return np.concatenate(res, axis=1)

@dataclass(frozen=True)
class ActivitiesPrediction:
    U: np.ndarray
    U_decor: np.ndarray
    U_o: np.ndarray
    stds: np.ndarray
    FOV_train: np.ndarray
    FOV_test: np.ndarray
    filtered_motifs: np.ndarray

class CovarianceMode(str, Enum):
    posterior = 'posterior'
    MAP = 'MAP'

def predict_activities(data: TransformedData, B_decomposition: LowrankDecomposition,
                       Sigma, sigmas, nus, filter_motifs=True,
                       filter_order=4, cv_search=True, nu_search_min=0.5,
                       nu_search_max=2.5, nu_search_steps=21, cv_splits=5,
                       cv_repeats=5,
                       cov_mode=CovarianceMode.MAP) -> ActivitiesPrediction:
    
    def _sol(BT_Y_sum, BT_B, sigma, nu, n: int, standardize=False):
        tau = nu / sigma
        Sigma_hat = Sigma * tau
        Z = n * BT_B
        Z[np.diag_indices_from(Z)] += 1 / Sigma_hat 

        sol = jscipy.linalg.solve(Z, BT_Y_sum,
                                  assume_a='her')
        if standardize:
            Z = np.linalg.pinv(Z, hermitian=True)
            if cov_mode == CovarianceMode.posterior:
                Z = sigma * Z
            else:
                Z =  n * Z @ (BT_B * Sigma_hat @ BT_B + BT_B * sigma) @ Z 
            D = Z.diagonal() ** (-0.5)
            C = D.reshape(-1, 1) * Z * D
            eigh = jnp.linalg.eigh(C)
            T = D.reshape(-1, 1) * (eigh.eigenvectors * (1 / eigh.eigenvalues ** 0.5) @ eigh.eigenvectors.T)
            U_decor = T @ sol
            return sol, U_decor, Z.diagonal() ** 0.5
        return sol
    
    Y = data.Y
    group_inds = data.original_inds
    S = B_decomposition.S; V = B_decomposition.V; Q = B_decomposition.Q
    
    if filter_motifs:
        inds = np.log10(Sigma) >= (np.median(np.log10(Sigma)) - filter_order)
        V = V[:, inds]
        Sigma = Sigma[inds]
        filtered_motifs = np.where(~inds)[0]
    else:
        filtered_motifs = None

    B = Q * S @ V
    mults_train = defaultdict(lambda: defaultdict(list))
    mults_test = defaultdict(lambda: defaultdict(list))
    rkf = RepeatedKFold(n_splits=cv_splits, n_repeats=cv_repeats)
    if cv_search:
        mults = list(np.linspace(nu_search_min, nu_search_max, nu_search_steps))
        mults += [nu_search_max * 2, nu_search_max * 4]    
        mults = sorted(mults + [1.0])
    else:
        mults = [1.0]

    for train_inds, test_inds in rkf.split(Y):
        Ytest = Y[test_inds]
        Btest = B[test_inds]
        Ytr = Y[train_inds]
        Btr = B[train_inds]
        BT_B = Btr.T @ Btr
        BT_Y = Btr.T @ Ytr
        for i, (inds, sigma, nu) in enumerate(zip(group_inds, sigmas, nus)):
            BT_Y_sum = BT_Y[:, inds]#.sum(axis=-1, keepdims=True)
            Ytrs = Ytr[:, inds]
            Ytests = Ytest[:, inds]
            for mult in mults:
                sol = _sol(BT_Y_sum, BT_B, sigma, nu * mult, n=1)
                diff_test = ((Ytests - Btest @ sol) ** 2).sum(axis=0)
                diff_train = ((Ytrs - Btr @ sol) ** 2).sum(axis=0)
                FOV_train = 1. - diff_train / (Ytrs ** 2).sum(axis=0)
                FOV_test = 1. - diff_test / (Ytests ** 2).sum(axis=0)
                mults_train[i][mult].append(FOV_train.mean())
                mults_test[i][mult].append(FOV_test.mean())
    FOV_test = np.zeros_like(nus)
    FOV_train = np.zeros_like(nus)
    for i in range(len(nus)):
        for m in mults:
            mults_train[i][m] = np.mean(mults_train[i][m])
            mults_test[i][m] = np.mean(mults_test[i][m])
    for i in range(len(nus)):
        test = mults_test[i]
        train = mults_train[i]
        k = max(test, key=lambda x: test[x])
        FOV_train[i] = train[k]
        FOV_test[i] = test[k]
        nus[i] = nus[i] * k
    BT_B = B.T @ B
    BT_Y = B.T @ Y
    U = list()
    U_independent = list()
    U_stds = list()
    U0 = list()
    for inds, sigma, nu in zip(group_inds, sigmas, nus):
        BT_Y_sub = BT_Y[:, inds]
        U_pred, U_decor, std = _sol(BT_Y_sub.sum(axis=-1, keepdims=True), BT_B,
                                    sigma, nu, len(inds), standardize=True)
        U1 = _sol(BT_Y_sub, BT_B, sigma, nu, 1)
        U0.append(U1)
        U.append(U_pred)
        U_independent.append(U_decor)
        U_stds.append(std.reshape(-1, 1))
    U = np.concatenate(U, axis=-1)
    U_independent = np.concatenate(U_independent, axis=-1)
    U_stds = np.concatenate(U_stds, axis=-1)
    return ActivitiesPrediction(U, U_decor=U_independent, U_o=np.concatenate(U0, axis=-1),
                                stds=U_stds, FOV_test=FOV_test, FOV_train=FOV_train,
                                filtered_motifs=filtered_motifs)

class ClusteringMode(str, Enum):
    none = 'none'
    KMeans = 'KMeans'
    NMF = 'NMF'

def cluster_data(data: ProjectData, mode=ClusteringMode.none, num_clusters=200,
                 keep_motifs=False)->ProjectData:
    def trs(B, labels, n):
        mx = np.zeros((n, B.shape[1]))
        for i, v in enumerate(labels):
            mx[v, i] = 1
        return mx
    if mode == ClusteringMode.none:
        return data, None
    loadings = data.B
    motif_expression = data.K
    if mode == ClusteringMode.KMeans:
        km = KMeans(n_clusters=num_clusters, n_init=10)
        km = km.fit(loadings.T)
        W = km.cluster_centers_.T 
        H = trs(loadings, km.labels_, num_clusters); 
    else:
        model = NMF(n_components=num_clusters, max_iter=1000)
        W = model.fit_transform(loadings)
        H = model.components_
    if not keep_motifs:
        loadings = W
        if motif_expression is not None:
            data.K = H @ motif_expression
        clustering = H
    else:
        loadings = W @ H
        clustering = None
    data.B = loadings
    data.K = motif_expression
    return data, clustering

@dataclass(frozen=True)
class FitResult:
    activities: ActivitiesPrediction
    Sigma: np.ndarray
    nu: np.ndarray
    sigma: np.ndarray
    clustering: np.ndarray
    group_names: list

def fit(project: str, sigma_structure: SigmaStructure, clustering: ClusteringMode,
        num_clusters: int, cov_mode: CovarianceMode, nu_cv_search: bool,
        nu_cv_splits: int, nu_cv_repeats: int, nu_search_min: float,
        nu_search_max: float, nu_search_steps: int, regularization: Regularization,
        alpha: float, temperature: float, exact_promoter_means_solver: bool,
        gpu: bool, gpu_decomposition: bool, x64=True, true_mean=None,
        verbose=True, dump=True) -> ActivitiesPrediction:
    if x64:
        jax.config.update("jax_enable_x64", True)
    data = read_init(project)
    fmt = data.fmt
    promoter_names = data.promoter_names
    group_names = data.group_names
    motif_names = data.motif_names
    if clustering != clustering.none:
        logger_print('Clustering data...', verbose)
    data, clustering = cluster_data(data, mode=clustering, 
                                    num_clusters=num_clusters)

    logger_print('Transforming data...', verbose)
    data = transform_data(data, centering_method=CenteringMethod.single)
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

    with jax.default_device(device):
        logger_print('Estimating error variances...', verbose)
        sigmas = estimate_error_variances(data, B_decomposition)
        data = transform_data(data, centering_method=CenteringMethod.sample_average)
        del B_decomposition
        B_decomposition = lowrank_decomposition(data.B)
        # data = transform_data(data, centering_method=CenteringMethod.group)
        logger_print('Estimating "nu" parameters/total variances of motif activities...', verbose)
        nus = estimate_nu(data, B_decomposition, sigmas)
        if sigma_structure == SigmaStructure.diag:
            logger_print('Estimating individual variances and re-estimating "nu"...', verbose)
            Sigma, nus = estimate_motif_variances(data, B_decomposition, sigmas, nus)
        else:
            Sigma = jnp.ones(data.B.shape[-1])
        logger_print('Predicting motif activities...', verbose)
        # data = transform_data(data, centering_method=CenteringMethod.sample_average)
        # del B_decomposition
        # B_decomposition = lowrank_decomposition(data.B)
        U = predict_activities(data, B_decomposition, Sigma, sigmas, nus, 
                               cov_mode=cov_mode, cv_search=nu_cv_search, cv_repeats=nu_cv_repeats,
                               cv_splits=nu_cv_splits, nu_search_min=nu_search_min,
                               nu_search_max=nu_search_max, nu_search_steps=nu_search_steps)
    
    res = FitResult(activities=U, Sigma=Sigma, nu=nus, sigma=sigmas,
                    clustering=clustering, group_names=group_names)    
    if dump:
        with openers[fmt](f'{project}.fit.{fmt}', 'wb') as f:
            dill.dump(res, f)
    return res
