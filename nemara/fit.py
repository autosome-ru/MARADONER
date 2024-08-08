#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from .utils import logger_print, openers, get_init_file
from scipy.optimize import minimize_scalar
from copy import deepcopy
import multiprocessing as mp
import scipy.linalg.lapack as lapack
import numpy as np
import jax.numpy as jnp
from functools import partial
from tqdm import tqdm
import json
import dill
import os
from jax.numpy.linalg import eigh, svd
from jax import config
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from .koptimizer import KOptimizer
from collections import defaultdict
from sklearn.model_selection import RepeatedKFold
from itertools import product
from enum import Enum



if __name__.startswith('__mp_'):
    config.update("jax_platform_name", "cpu")
config.update("jax_enable_x64", True)

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


def estimate_groupwise_variance(Y: list[np.ndarray], B_null: tuple) -> np.ndarray:
    """
    Estimate groupwise variance

    Parameters
    ----------
    Y : np.ndarray
        Array of shape (p, n), where p is a number of promoters and n is a number of samples.
    B : tuple[np.ndarray, np.ndarray, np.ndarray]
        Nullspace of the B matrix.

    Returns
    -------
    var_reml : np.ndarray
        Array of estimated variances.
    """
    P = B_null[0].T
    P = jnp.array(P)
    Y = jnp.array(Y)
    var_reml = jnp.mean((P @ Y) ** 2) 
    return var_reml


def _estimate_transform_matrices(B_svd: tuple[np.ndarray, np.ndarray, np.ndarray], U_mults: np.ndarray):
    Q, D, V = B_svd
    dvt = jnp.array(D.reshape(-1, 1) * V)
    mult = jnp.array(U_mults)#.mean(axis=-1)
    mx = dvt * mult ** 2 @ dvt.T
    qdv, _ = lowrank_decomposition(mx, rel_eps=1e-12)
    return qdv[0].T, qdv[1]


def estimate_sigma(Y: np.ndarray, B_svd: tuple[np.ndarray, np.ndarray, np.ndarray], sigma_e: float,  U_mults: np.ndarray, aux=None) -> np.ndarray:
    P, eigs, _ = B_svd
    Yt = Y.T @ P
    trs = _estimate_transform_matrices(B_svd, U_mults)
    Yt = trs[0] @ Yt[..., np.newaxis]
    eigs = trs[1]
    bounds = (0.0, sigma_e * 10)

    def loglik(sigma: float):
        loglik = 0
        for Y in Yt:
            R = sigma * eigs + sigma_e
            Y = (R.reshape(-1, 1)) ** (-0.5) * Y
            loglik += -jnp.einsum('ij,ij->', Y, Y)
            loglik += -Y.shape[1] * jnp.log(R).sum()
        return -loglik
    return minimize_scalar(loglik, bounds=bounds).x


def transform_data(Y, B, U_mult, std_y=False, std_b=False, b_cutoff=1e-9, weights=None, _means_est=False):
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
    U_mult = U_mult[inds]
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

def cluster_data(prj, mode=None, num_clusters=200, keep_motifs=False):
    def trs(B, labels, n):
        mx = np.zeros((n, B.shape[1]))
        for i, v in enumerate(labels):
            mx[v, i] = 1
        return mx
    prj = deepcopy(prj)
    if not mode or str(mode) == 'none':
        return prj
    loadings = prj['loadings']
    motif_expression = prj['motif_expression']
    if mode == 'K-Means':
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
        motif_expression = H @ motif_expression
        prj['clustering'] = H
    else:
        loadings = W @ H
    prj['loadings'] = loadings
    prj['motif_expression'] = motif_expression
    return prj
    

def preprocess_data(prj, rel_eps=1e-9, inds_train=None):
    Y = prj['expression']
    B = prj['loadings']
    U_mult = prj['motif_expression']
    if U_mult is None:
        U_mult = np.ones((B.shape[1], Y.shape[1]), dtype=float)
    Y, B, U_mult, inds = transform_data(Y, B, U_mult)
    if inds_train is not None:
        qdv, null = lowrank_decomposition(B[inds_train], rel_eps=rel_eps)
    else:
        qdv, null = lowrank_decomposition(B, rel_eps=rel_eps)
    res = {'expression': Y, 'loadings': B, 'motif_expression': U_mult,
           'loadings_svd': qdv, 'loadings_null': null}
    return res

def estimate_motif_variances(t, B, regul: str, alpha: float):
    Y, motif_expression, sigma_e, sigma_u = t
    opt_pre = KOptimizer(B, alpha=0, penalty=regul, hotstart_iters=400)
    opt = KOptimizer(B, alpha=alpha, penalty=regul, hotstart_iters=600)
    scale_factor = sigma_u
    motif_expression = motif_expression.flatten()
    if regul and regul.endswith('prior'):
        opt.x0 = jnp.append(motif_expression, 1.0)
        prior = motif_expression
    else:
        opt.x0 = motif_expression
        prior = None
    opt_pre.x0 = opt.x0
    for res in opt_pre.optimize(Y, sigma_e, prior=prior, scale_factor=scale_factor):
        pass
    opt.x0 = res.get_params() ** 2
    for res in opt.optimize(Y, sigma_e, prior=prior, scale_factor=scale_factor):
        pass
    if regul and regul.endswith('prior'):
        motif_expression = res.get_params()[:-1]
    else:
        motif_expression = res.get_params()
    return motif_expression

def estimate_variance(t, B, B_null: tuple, B_svd: tuple):
    Y, motif_expression = t
    sigma_e = estimate_groupwise_variance(Y, B_null)
    sigma_u = estimate_sigma(Y, B_svd, sigma_e, U_mults=motif_expression)
    res = {'sigma_e': sigma_e, 'sigma_u': sigma_u}
    return res

def _calc_aux(B_svd):
    Q, D, V = [jnp.array(t) for t in B_svd]
    Vt = V.T * D
    BB = Vt @ Vt.T
    return Vt, BB, Q

def estimate_u_map(t, B_svd, tau=1, aux=None):
    Y, U_mults, sigma_g, sigma_u = t
    n = Y.shape[1]
    Y = Y.sum(axis=-1, keepdims=True).T
    if aux is None:
        aux = _calc_aux(B_svd)
    Vt, BB, Q = aux
    inds = jnp.diag_indices_from(BB)
    Y = (Y @ Q) @ Vt.T
    ratio = jnp.exp(jnp.log(sigma_g) - jnp.log(sigma_u) - jnp.log(tau))
    cov = n * BB 
    cov = cov.at[inds].add(ratio * U_mults ** -2)
    cov = cov / sigma_g
    cov = jnp.linalg.pinv(cov, hermitian=True)
    U = cov @ Y.T / sigma_g 
    U = U.T
    stds = cov.diagonal() ** 0.5
    s = stds
    cor = (1/s) * cov * (1/s).reshape(-1,1)
    d, q = jnp.linalg.eigh(cor)
    p = q * (1/(d ** 0.5)) @ q.T
    W = p * (1/s).reshape(1,-1)
    U_std = W @ U[-1]
    return {'U_raw': U.flatten(), 'U_std': U_std.flatten(), 'stds': stds}
    

def fit(project: str, regul: str = None, alpha: float = 0, estimate_motif_vars=False, tau=1,
        clustering=None, n_clusters=200,
        verbose=True, dump=True, n_jobs=1):
    if type(project) is str:
        filename, fmt = get_init_file(project)
        with openers[fmt](filename, 'rb') as f:
            init = dill.load(f)
    else:
        init = project
    groups = init['groups']
    data = cluster_data(init, mode=clustering, num_clusters=n_clusters)
    if clustering:
        clustering = data['clustering']
    data = preprocess_data(data)
    Y = data['expression']
    U_mults = data['motif_expression']
    
    B = data['loadings']
    B_null = data['loadings_null']
    B_svd = data['loadings_svd']
    u_aux = _calc_aux(B_svd)
    items = [[Y[..., inds], U_mults[..., inds].mean(axis=-1)] for inds in groups.values()]
    est_vars_f = partial(estimate_variance, B=B, B_null=B_null, B_svd=B_svd)
    est_motif_vars_f = partial(estimate_motif_variances, B=B, regul=regul, alpha=alpha)
    est_random_effects_f = partial(estimate_u_map, B_svd=B_svd, tau=tau, aux=u_aux)
    sigmas_e = list()
    sigmas_u = list()
    motif_sigmas = list()
    residuals_groups = list()
    total_var_groups = list()
    Us = list()
    if n_jobs == -1:
        n_jobs = max(1, mp.cpu_count() - 1)
    logger_print('Estimating variance parameters...', verbose)
    if n_jobs > 1:
        ctx = mp.get_context("spawn")
        p = ctx.Pool(n_jobs)
        mapper = p.imap
    else:
        mapper = map
    for i, res in enumerate(mapper(est_vars_f, items)):
        sigmas_e.append(res['sigma_e'])
        sigmas_u.append(res['sigma_u'])
        items[i].extend([sigmas_e[-1], sigmas_u[-1]])
    if estimate_motif_vars:
        logger_print('Estimating motif variances (will take a lot of time)...', verbose)
        for i, res in enumerate(mapper(est_motif_vars_f, items)):
            motif_sigmas.append(res)
            items[i][1] = motif_sigmas[-1]
            items[i][-1] = 1
    logger_print('Estimating random effects...', verbose)
    for res in mapper(est_random_effects_f, items):
        Us.append(res)
    if n_jobs > 1:
        p.close()
    res = {'sigma_e': np.array(sigmas_e), 'sigma_u': np.array(sigmas_u)}
    if estimate_motif_vars:
        res['sigma_u'] = np.array(motif_sigmas).T
    for t, u_d in zip(items, Us):
        U = u_d['U_raw']
        Y = t[0].mean(axis=-1, keepdims=1)
        total_var_groups.append((Y ** 2).sum())
        residuals_groups.append(((Y - B @ U.reshape(-1, 1)) ** 2).sum())
    res['FOV'] = 1 - np.sum(residuals_groups) / np.sum(total_var_groups)
    res['FOV_groups'] = 1 - np.array(residuals_groups) / np.array(total_var_groups)
    d = defaultdict(list)
    for u_d in Us:
        for key, val in u_d.items():
            d[key].append(val)
    for key, val in d.items():
        d[key] = np.array(val).T
    res.update(d)
    if clustering is not None:
        res['clustering'] = clustering
    
    if dump:
        t = os.path.split(filename)
        folder = t[0]
        filename = t[1]
        project_name = '.'.join(filename.split('.')[:-2])
        with openers[fmt](os.path.join(folder, f'{project_name}.fit.{fmt}'), 'wb') as f:
            dill.dump(res, f)
    return res

def _calc_fov(B_train, B_test, items, Y_test, groups, u):
    Us = list()
    total_var_groups_train = list()
    residuals_groups_train = list()
    total_var_groups_test = list()
    residuals_groups_test = list()
    for res in u:
        Us.append(res)
    for t, Y_test, u_d in zip(items, [Y_test[..., inds] for inds in groups.values()], Us):
        U = u_d['U_raw']
        Y_train = t[0].mean(axis=-1, keepdims=1)
        Y_test = Y_test.mean(axis=-1, keepdims=1)
        total_var_groups_train.append((Y_train ** 2).sum())
        residuals_groups_train.append(((Y_train - B_train @ U.reshape(-1, 1)) ** 2).sum())
        total_var_groups_test.append((Y_test ** 2).sum())
        residuals_groups_test.append(((Y_test - B_test @ U.reshape(-1, 1)) ** 2).sum())
    train_fov_groups = 1 - np.array(residuals_groups_train) / np.array(total_var_groups_train)
    test_fov_groups = 1 - np.array(residuals_groups_test) / np.array(total_var_groups_test)
    train_fov = 1 - np.sum(residuals_groups_train) / np.sum(total_var_groups_train)
    test_fov = 1 - np.sum(residuals_groups_test) / np.sum(total_var_groups_test)
    return {'train_fov': train_fov, 'test_fov': test_fov, 'train_fov_groups': train_fov_groups, 'test_fov_groups': test_fov_groups}

def cross_validate(project: str, n_splits=4, n_repeats=1, regul: list[str] = None, alpha: list[str] = 1, tau: list[float] = 1,
                   clustering: list[str] = None, n_clusters: list[int] = 200,  estimate_motif_vars: list[bool] = False,
                   verbose=True, dump=True, seed=1, n_jobs=1):
    def listconv(x, string=False):
        if type(x) in (str, ) or not np.iterable(x):
            if string:
                if issubclass(type(x), Enum):
                    x = x.value
            return [x]
        if string:
            x = [x.value if issubclass(type(x), Enum) else str(x) for x in x]
            x = [x if x != 'none' else None for x in x]
        return x
    regul = listconv(regul, string=True)
    alpha = listconv(alpha)
    tau = listconv(tau)
    clustering = listconv(clustering, string=True)
    n_clusters = listconv(n_clusters)
    estimate_motif_vars = sorted(listconv(estimate_motif_vars))
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)

    if type(project) is str:
        filename, fmt = get_init_file(project)
        with openers[fmt](filename, 'rb') as f:
            init = dill.load(f)
    else:
        init = project
    groups = init['groups']
    res = dict()
    fov_names = ('train_fov', 'test_fov', 'train_fov_groups', 'test_fov_groups')
    for n in fov_names:
        res[n] = defaultdict(list)
    params = dict()
    if n_jobs == -1:
        n_jobs = max(1, mp.cpu_count() - 1)
    if n_jobs > 1:
        ctx = mp.get_context("forkserver")
        p = ctx.Pool(n_jobs)
        mapper = p.imap
    else:
        mapper = map
    for cluster_mode, n_clusts in product(clustering, n_clusters):
        params['clustering'] = cluster_mode; params['n_clusters'] = n_clusts
        data = cluster_data(init, mode=cluster_mode, num_clusters=n_clusts)
        for i, (train_inds, test_inds) in enumerate(rkf.split(init['expression'])):
            logger_print(f'[Cluster = {"None" if not cluster_mode else cluster_mode}, n = {n_clusts}] Fold #{i+1}/{n_splits*n_repeats}...', verbose)
            data = preprocess_data(data, inds_train=train_inds)
            Y_train = data['expression'][train_inds]
            Y_test = data['expression'][test_inds]
            U_mults = data['motif_expression']
            
            B_train = data['loadings'][train_inds]
            B_test = data['loadings'][test_inds]
            B_null = data['loadings_null']
            B_svd = data['loadings_svd']
            u_aux = _calc_aux(B_svd)
            items = [[Y_train[..., inds], U_mults[..., inds].mean(axis=-1)] for inds in groups.values()]
            est_vars_f = partial(estimate_variance, B=B_train, B_null=B_null, B_svd=B_svd)
            for i, ests in enumerate(mapper(est_vars_f, items)):
                items[i].extend([ests['sigma_e'], ests['sigma_u']])
            for est_motifs in estimate_motif_vars:
                params['estimate_motif_variances'] = est_motifs
                if est_motifs:
                    tmp_vars = [it[-1] for it in items]
                    for reg, a in product(regul, alpha):
                        params['regul'] = reg
                        params['alpha'] = a
                        est_motif_vars_f = partial(estimate_motif_variances, B=B_train, regul=reg, alpha=a)
                        for i, ests in enumerate(mapper(est_motif_vars_f, items)):
                            items[i][1] = ests
                            items[i][-1] = 1
                        for t in tau:
                            params['tau'] = t
                            est_random_effects_f = partial(estimate_u_map, B_svd=B_svd, tau=t, aux=u_aux)
                            fovs = _calc_fov(B_train, B_test, items, Y_test, groups, mapper(est_random_effects_f, items))
                            for n in fov_names:
                                res[n][str(tuple(sorted(params.items(), key=lambda x: x[0])))[1:-1]].append(fovs[n])
                        for i in range(len(items)):
                            items[i][-1] = tmp_vars[-1]
                else:
                    try:
                        del params['regul']
                        del params['alpha']
                    except KeyError:
                        pass
                    for t in tau:
                        params['tau'] = t
                        est_random_effects_f = partial(estimate_u_map, B_svd=B_svd, aux=u_aux, tau=t)
                        fovs = _calc_fov(B_train, B_test, items, Y_test, groups, mapper(est_random_effects_f, items))
                        for n in fov_names:
                            res[n][str(tuple(sorted(params.items(), key=lambda x: x[0])))[1:-1]].append(fovs[n])                        
    fov_means = defaultdict(lambda: defaultdict(float))
    fov_stds = defaultdict(lambda: defaultdict(float))
    for fov_type, params_d in res.items():
        for params, fovs in params_d.items():
            if np.iterable(fovs[0]):
                fovs = np.vstack(fovs)
                fov_means[fov_type][params] = list(fovs.mean(axis=0))
                fov_stds[fov_type][params] = list(fovs.std(axis=0))
            else:
                fov_means[fov_type][params] = np.mean(fovs)
                fov_stds[fov_type][params] = np.std(fovs)
    if n_jobs > 1:
        p.close()
    res = {'mean': fov_means, 'std': fov_stds}
    if dump:
        t = os.path.split(filename)
        folder = t[0]
        filename = t[1]
        project_name = '.'.join(filename.split('.')[:-2])
        with open(os.path.join(folder, f'{project_name}.cv.json'), 'w') as f:
            json.dump(res, f)
    return res            
        

# if __name__ == '__main__':
#     project = 'syntest'
#     from time import time
#     t0 = time()
#     t = cv(project, n_jobs=1, dump=True)
#     print(time() - t0)