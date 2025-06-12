# -*- coding: utf-8 -*-
import numpy as np
import jax.numpy as jnp
import jax 
from .utils import read_init, openers, ProjectData
from .fit import FOVResult, ActivitiesPrediction, FitResult
from scipy.optimize import minimize_scalar, minimize
import os
import dill
from pandas import DataFrame as DF
from scipy.stats import norm
from functools import partial
from tqdm import tqdm


def estimate_promoter_prior_variance(data: ProjectData, activities: ActivitiesPrediction,
                                     fit: FitResult, top=0.90, eps=1e-6):
    B = data.B
    Y = data.Y
    group_inds = data.group_inds
    Y = Y - fit.promoter_mean.mean.reshape(-1, 1) - fit.sample_mean.mean.reshape(1, -1)
    Y = Y -  B @ fit.motif_mean.mean.reshape(-1, 1)
    Y = np.concatenate([Y[:, inds].mean(axis=1, keepdims=True) - B @ U.reshape(-1, 1)
                        for inds, U in zip(group_inds, activities.U.T)],
                       axis=1)
    
    var = (Y**2).mean(axis=1)
    var = var[var > eps]
    inds = np.argsort(var)
    inds = inds[:int(len(inds) * top)]
    return np.var(var[inds])

def estimate_promoter_variance(project_name: str, prior_top=0.90):
 
    def fun(sigma, y: jnp.ndarray, b: jnp.ndarray, s: int,
          prior_mean: float, prior_var: float):
        if jnp.iterable(sigma):
            sigma = sigma[0]
        theta = prior_var / prior_mean
        alpha = prior_var / theta ** 2
        penalty = sigma / theta - (alpha - 1) * jnp.log(sigma)
        return y / (b + sigma) + s * jnp.log(b + sigma) + penalty
    data = read_init(project_name)
    fmt = data.fmt
    with openers[fmt](f'{project_name}.fit.{fmt}', 'rb') as f:
        fit: FitResult = dill.load(f)
    with openers[fmt](f'{project_name}.predict.{fmt}', 'rb') as f:
        activities: ActivitiesPrediction = dill.load(f)
    B = data.B
    Y = data.Y
    group_inds = data.group_inds
    prior_var = estimate_promoter_prior_variance(data, activities, fit,
                                                 top=prior_top)
    print('Piror standard deviation:', prior_var ** 0.5)
    prior_means = fit.error_variance.variance
    
    Y = Y - fit.promoter_mean.mean.reshape(-1, 1) - fit.sample_mean.mean.reshape(1, -1)
    Y = Y - B @ fit.motif_mean.mean.reshape(-1, 1)
    Y = Y ** 2
    B_hat = B ** 2 * fit.motif_variance.motif
    B_hat = B_hat.sum(axis=1)
    var = list()
    for inds, prior_mean, nu in tqdm(list(zip(group_inds, prior_means, fit.motif_variance.group))):
        Yt = Y[:, inds].sum(axis=1)
        s = len(inds)
        f_ = jax.jit(partial(fun, prior_mean=prior_mean, prior_var=prior_var, s=s))
        g_ = jax.jit(jax.grad(f_))
        var_g = list()
        for y, b in zip(Yt, B_hat * nu):
            res = minimize(partial(f_, b=b, y=y), x0=jnp.array([prior_mean]),
                           method='SLSQP', bounds=[(0, None)],
                           jac=partial(g_, b=b, y=y))
            var_g.append(res.x[0] ** 2)
        var.append(var_g)
    var = np.array(var, dtype=float).T
    with openers[fmt](f'{project_name}.promvar.{fmt}', 'wb') as f:
        dill.dump(var, f)
    return var
    

def grn(project_name: str,  output: str, use_hdf=False, save_stat=True,
        prior_h1=1/100):
    data = read_init(project_name)
    fmt = data.fmt
    with openers[fmt](f'{project_name}.fit.{fmt}', 'rb') as f:
        fit: FitResult = dill.load(f)
    with openers[fmt](f'{project_name}.predict.{fmt}', 'rb') as f:
        activities: ActivitiesPrediction = dill.load(f)
    
    dtype = np.float32
    B = data.B.astype(dtype)
    Y = data.Y.astype(dtype)
    group_inds = data.group_inds
    group_names = data.group_names
    nus = fit.motif_variance.group.astype(dtype)
    motif_names = data.motif_names
    prom_names = data.promoter_names
    U = activities.U_raw.astype(dtype)
    motif_mean = fit.motif_mean.mean.flatten().astype(dtype)
    motif_variance = fit.motif_variance.motif.astype(dtype)
    promoter_mean = fit.promoter_mean.mean.astype(dtype)
    sample_mean = fit.sample_mean.mean.astype(dtype)
    
    try:
        with openers[fmt](f'{project_name}.promvar.{fmt}', 'rb') as f:
            promvar: np.ndarray = dill.load(f)
    except FileNotFoundError:
        print('WARNING')
        print('It seems that promoter variances were not estimated prior to running GRN.')
        print('All promoter-wise variances will be assumed to be equal to the average error variance.')
        print('Consider estimating promoter-wise variances before running GRN in the future.')
        promvar = np.zeros((len(B), len(group_names)))
        for i, sigma in enumerate(fit.error_variance.variance):
            promvar[:, i] = sigma
    
    Y = Y - promoter_mean.reshape(-1, 1) - sample_mean.reshape(1, -1)
    Y = Y - B @ motif_mean.reshape(-1, 1)
    
    if activities.filtered_motifs is not None:
        motif_names = np.delete(motif_names, activities.filtered_motifs)
        B = np.delete(B, activities.filtered_motifs, axis=1)
        motif_mean = np.delete(motif_mean, activities.filtered_motifs)
        motif_variance = np.delete(motif_variance, activities.filtered_motifs)
    
    BM = B * motif_mean
    BM = BM[..., None]
    # BU = BU[..., None]
    B_hat = B ** 2 * motif_variance
    B_hat = B_hat.sum(axis=1, keepdims=True) - B_hat
    B_pow = B ** 2
    
    folder_stat = os.path.join(output, 'lr')
    folder_belief = os.path.join(output, 'belief')
    if save_stat:
        os.makedirs(folder_stat, exist_ok=True)
    os.makedirs(folder_belief, exist_ok=True)
    for sigma, nu, name, inds in zip(promvar.T[..., None], nus,  group_names, group_inds):
        # if name != 'anconeus':
        #     continue
        print(name)
        var = (B_hat * nu + sigma)
        Y_ = Y[:, inds][..., None, :] + BM
        # theta = U[:, inds][..., None, :] + BM
        theta = B[..., None] * U[:, inds] + BM
        loglr = 2 * B * (Y_ * theta).sum(axis=-1) - B_pow * (theta ** 2).sum(axis=-1)
        del Y_
        del theta
        loglr = loglr / (2 * var)
        del var
        lr = np.exp(loglr)
        belief = lr * prior_h1 / ((1 - prior_h1) + lr * prior_h1)
        inds = sigma.flatten() > 1e-3
        lr = lr[inds]
        belief = belief[inds]
        belief = belief.astype(np.half)
        
        proms = list(np.array(prom_names)[inds])
        
        # pvalue = n.sf(lr) * (theta > 0) + n.cdf(lr) * (theta <= 0)
        if use_hdf:
            if save_stat:
                lr = lr.astype(np.half)
                filename = os.path.join(folder_stat, f'{name}.hdf')
                DF(data=lr, index=proms, columns=motif_names).to_hdf(filename, key='zscore', mode='w', complevel=4)
            filename = os.path.join(folder_belief, f'{name}.hdf')
            DF(data=belief, index=proms, columns=motif_names).to_hdf(filename, key='lrt', mode='w', complevel=4)
        else:
            if save_stat:
                lr = lr.astype(np.half)
                filename = os.path.join(folder_stat, f'{name}.tsv')
                DF(data=lr, index=proms, columns=motif_names).to_csv(filename, sep='\t',
                                                                          float_format='%.3f')
            filename = os.path.join(folder_belief, f'{name}.tsv')
            DF(data=belief, index=proms, columns=motif_names).to_csv(filename, sep='\t',
                                                                          float_format='%.3f')    
        
        