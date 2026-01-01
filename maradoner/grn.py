# -*- coding: utf-8 -*-
import numpy as np
import jax.numpy as jnp
import jax 
from .utils import read_init, openers, ProjectData
from .fit import ActivitiesPrediction, FitResult, split_data
from scipy.optimize import minimize
import os
import dill
from pandas import DataFrame as DF
from functools import partial
from tqdm import tqdm
from scipy.special import digamma
from scipy.interpolate import interp1d
from scipy.stats import f as f_dist


def estimate_promoter_prior_variance(data: ProjectData, activities: ActivitiesPrediction,
                                     fit: FitResult, top=0.90, eps=1e-6):
    B = data.B
    Y = data.Y
    group_inds = data.group_inds
    Y = Y - fit.promoter_mean.mean.reshape(-1, 1) - fit.sample_mean.mean.reshape(1, -1)
    Y = Y -  B @ fit.motif_mean.mean.reshape(-1, 1)
    if activities.filtered_motifs is not None and len(activities.filtered_motifs):
        B = np.delete(B, activities.filtered_motifs, axis=1)
    Y = np.concatenate([Y[:, inds].mean(axis=1, keepdims=True) - B @ U.reshape(-1, 1)
                        for inds, U in zip(group_inds, activities.U.T)],
                       axis=1)
    
    var = (Y**2).mean(axis=1)
    var = var[var > eps]
    inds = np.argsort(var)
    inds = inds[:int(len(inds) * top)]
    return np.var(var[inds])

def scalable_gaussian_smoother(x, y, span=0.1, n_grid=200):
    x_min, x_max = x.min(), x.max()
    x_grid = np.linspace(x_min, x_max, n_grid)
    y_grid = np.zeros(n_grid)
    sigma = (x_max - x_min) * span
    
    for i in range(n_grid):
        dists = x - x_grid[i]
        weights = np.exp(-0.5 * (dists / sigma)**2)
        sum_weights = np.sum(weights)
        if sum_weights > 0:
            y_grid[i] = np.sum(weights * y) / sum_weights
    return x_grid, y_grid


def estimate_variance_loess(Y: np.ndarray, M: np.ndarray, span: float = 0.2) -> np.ndarray:
    """
    Estimates the diagonal elements of the variance matrix D using Empirical Bayes.
    
    Parameters:
    -----------
    Y : np.ndarray
        Observed data matrix of shape (p_features, s_samples).
    M : np.ndarray
        Expected mean matrix of shape (p_features, s_samples).
    span : float
        The fraction of data range used for the Gaussian kernel smoother.
        
    Returns:
    --------
    np.ndarray
        A vector of length p containing the posterior variance estimates.
    """
    p, s = Y.shape
    d_obs = float(s) + 1
    
    residuals = Y - M
    obs_vars = np.mean(residuals**2, axis=1) + 1e-12
    # print(Y.shape)
    # obs_vars = np.var(residuals, axis=1, ddof=0)
    # mad = np.median(np.abs(residuals), axis=1)
    # robust_vars = mad ** 2 / 2
    # obs_vars = robust_vars
    # inds = 
    # print(obs_vars.mean(), obs_vars.max(), obs_vars.std())
    # inds = np.argsort(obs_vars)
    # print(obs_vars[inds[:int(len(inds)*0.9)]].mean())
    
    # Use mean of M (or Y) as the abundance metric for the trend
    abundance = np.mean(M, axis=1)
    
    # We work on log(variance) vs mean. 
    # E[log(s^2)] != log(E[s^2]). It is biased downwards by digamma(d/2) - log(d/2).
    log_obs_vars = np.log(obs_vars)
    
    bias = digamma(d_obs / 2.0) - np.log(d_obs / 2.0)
    
    corrected_log_vars = log_obs_vars - bias
    
    

    x_trend, y_trend = scalable_gaussian_smoother(abundance, corrected_log_vars, span=span, n_grid=100)
 
    trend_interpolator = interp1d(x_trend, y_trend, bounds_error=False, fill_value="extrapolate")
    # The trend now represents the unbiased log(sigma^2), so we can exponentiate directly
    prior_vars_s0 = np.exp(trend_interpolator(abundance))
    
    # Optimize hyperparameter d0 
    # The statistic F = obs_vars / prior_vars follows F(d_obs, d0)
    F_stats = obs_vars / prior_vars_s0
    
    def objective_with_gradient(x):
        d0 = x[0]
        if d0 <= 1e-5: d0 = 1e-5
        
     
        nll = -np.sum(f_dist.logpdf(F_stats, dfn=d_obs, dfd=d0))
        
        # grad
        psi_term = digamma((d_obs + d0) / 2.0) - digamma(d0 / 2.0)
        log_term = np.log(d0 / (d0 + d_obs * F_stats))
        rational_term = (d_obs * (F_stats - 1)) / (d0 + d_obs * F_stats)
        
        # Gradient per gene: 0.5 * (psi + log + rational)
        grad = -0.5 * np.sum(psi_term + log_term + rational_term)
        
        return nll, np.array([grad])

    res = minimize(objective_with_gradient, 
                   [10.0], 
                   method='L-BFGS-B', 
                   jac=True, 
                   bounds=[(0.01, 1e5)])
    
    d0_hat = res.x[0]
    
    if d0_hat > 1e4: 
        return prior_vars_s0

    # posterior mean
    numerator = (d_obs * obs_vars) + (d0_hat * prior_vars_s0)
    denominator = d_obs + d0_hat
    
    return numerator / denominator

def estimate_promoter_variance(project_name: str, span=0.1):
 
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
    data, _ = split_data(data, fit.promoter_inds_to_drop)
    B = data.B
    Y = data.Y
    group_inds = data.group_inds
    
    Y = Y - fit.sample_mean.mean.reshape(1, -1)
    M = fit.promoter_mean.mean.reshape(-1, 1)
    M = M + B @ fit.motif_mean.mean.reshape(-1, 1) 
    if activities.filtered_motifs is not None:
        M = M + np.delete(B, activities.filtered_motifs, axis=1) @ activities.U_raw
    else:
        M = M + B @ activities.U_raw
    if fit.error_variance.promotor is None:
        var_mean = 1
    else:
        var_mean =fit.error_variance.promotor.mean()
    var = list()
    for inds, nu in (pbar := tqdm(list(zip(group_inds, fit.motif_variance.group)))):
        Y_ = Y[:, inds] / nu ** 0.5
        M_ = M[:, inds] / nu ** 0.5
        var_g = estimate_variance_loess(Y_, M_, span=span)
        var_g = var_g / var_g.mean() * var_mean
        var.append(var_g)
    var = np.array(var, dtype=float).T
    with openers[fmt](f'{project_name}.promvar.{fmt}', 'wb') as f:
        dill.dump(var, f)
    return var

def estimate_promoter_variance_(project_name: str, prior_top=0.90):
 
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
    if fit.error_variance.promotor is None or fit.error_variance.promotor.var() == 0:
        prior_var = estimate_promoter_prior_variance(data, activities, fit,
                                                     top=prior_top)
    else:
        prior_var = fit.error_variance.promotor
        
    print('Piror standard deviation:', prior_var ** 0.5)
    prior_means = fit.error_variance.variance
    Y = Y - fit.promoter_mean.mean.reshape(-1, 1) - fit.sample_mean.mean.reshape(1, -1)
    Y = Y - B @ fit.motif_mean.mean.reshape(-1, 1)
    Y = Y ** 2
    B_hat = B ** 2 * fit.motif_variance.motif
    B_hat = B_hat.sum(axis=1)
    var = list()
    jax.config.update('jax_enable_x64', True)
    jax.config.update('jax_platform_name', 'cpu') 
    for inds, prior_mean, nu in tqdm(list(zip(group_inds, prior_means, fit.motif_variance.group))):
        Yt = Y[:, inds].sum(axis=1)
        s = len(inds)
        f_ = jax.jit(partial(fun, prior_mean=prior_mean, prior_var=prior_var, s=s))
        g_ = jax.jit(jax.grad(f_))
        var_g = list()
        for y, b in zip(Yt, B_hat * nu):
            res = minimize(partial(f_, b=b, y=y), x0=jnp.array([prior_mean]),
                           method='SLSQP', bounds=[(0, None)],
                           jac=partial(g_, b=b, y=y)
                           )
            var_g.append(res.x[0] ** 2)
        var.append(var_g)
        break
    var = np.array(var, dtype=float).T
    with openers[fmt](f'{project_name}.promvar.{fmt}', 'wb') as f:
        dill.dump(var, f)
    return var
    

def bayesian_fdr_control(p0, alpha=0.05):
    """
    Control Bayesian FDR using sorted posterior probabilities of H0.
    
    Args:
        p0: Array of posterior probabilities P(H0|X) for each hypothesis.
        alpha: Target FDR level (e.g., 0.05).
    
    Returns:
        discoveries: Boolean array (True = reject H0).
        threshold: Rejection threshold for P(H0|X).
    """
    p0 = np.asarray(p0)
    if p0.size == 0:
        return np.zeros_like(p0, dtype=bool), -np.inf
    
    # Sort in ascending order
    sorted_p0 = np.sort(p0)
    m = len(sorted_p0)
    
    # Compute cumulative FDR = (sum_{i=1}^k p_i) / k
    cum_sum = np.cumsum(sorted_p0)
    fdr_k = cum_sum / np.arange(1, m + 1)
    
    # Find largest k where FDR(k) <= alpha
    valid_indices = np.where(fdr_k <= alpha)[0]  # 0-based indices
    if len(valid_indices) == 0:
        k = 0
    else:
        k = valid_indices[-1] + 1  # Convert to 1-based index
    
    # Set threshold and discover
    if k > 0:
        threshold = sorted_p0[k - 1]  # k-th element (0-indexed at k-1)
        discoveries = p0 <= threshold
    else:
        threshold = -np.inf
        discoveries = np.zeros_like(p0, dtype=bool)
    
    return discoveries, threshold

def grn(project_name: str,  output: str, use_hdf=False, save_stat=True,
        fdr_alpha=0.05, prior_h1=1/2, include_mean: bool = True):
    data = read_init(project_name)
    fmt = data.fmt
    with openers[fmt](f'{project_name}.fit.{fmt}', 'rb') as f:
        fit: FitResult = dill.load(f)
    with openers[fmt](f'{project_name}.predict.{fmt}', 'rb') as f:
        activities: ActivitiesPrediction = dill.load(f)
    data, _ = split_data(data, fit.promoter_inds_to_drop)
    dtype = np.float64
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
    
    promvar = np.zeros((len(B), len(group_names)))
    for i, sigma in enumerate(fit.error_variance.variance):
        promvar[:, i] = sigma * fit.error_variance.promotor
    
    Y = Y - promoter_mean.reshape(-1, 1) - sample_mean.reshape(1, -1)
    Y = Y - B @ motif_mean.reshape(-1, 1)
    
    if activities.filtered_motifs is not None:
        motif_names = np.delete(motif_names, activities.filtered_motifs)
        B = np.delete(B, activities.filtered_motifs, axis=1)
        motif_mean = np.delete(motif_mean, activities.filtered_motifs)
        motif_variance = np.delete(motif_variance, activities.filtered_motifs)
    
    BM = B * motif_mean
    BM = BM[..., None]
    B_hat = B ** 2 * motif_variance
    B_hat = B_hat.sum(axis=1, keepdims=True) - B_hat

    
    folder_stat = os.path.join(output, 'lr')
    folder_belief = os.path.join(output, 'belief')
    if save_stat:
        os.makedirs(folder_stat, exist_ok=True)
    os.makedirs(folder_belief, exist_ok=True)
    for sigma, nu, name, inds in (pbar := tqdm(list(zip(promvar.T[..., None], nus,  group_names, group_inds)), dynamic_ncols=True)):
        pbar.set_postfix_str(name)
        # var = (B_hat * nu + sigma)
        var = sigma
        
        Y_ = Y[:, inds][..., None, :] 
        theta = B[..., None] * U[:, inds] 
        if include_mean:
            Y_ = Y_ + BM
            theta = theta + BM
            

        loglr = 2 * (Y_ * theta).sum(axis=-1) - (theta ** 2).sum(axis=-1)
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

        sorted_beliefs = belief.flatten() 
        sorted_beliefs = sorted_beliefs[sorted_beliefs > 0.5]
        sorted_beliefs = np.sort(sorted_beliefs)[::-1]
        sorted_inbeliefs = 1 - sorted_beliefs
        
        cumulative_fdr = np.cumsum(sorted_inbeliefs) / (np.arange(len(sorted_inbeliefs)) + 1)
        # print(fdr_alpha)
        try:
            k = np.min(np.where(cumulative_fdr <= fdr_alpha)[0])
            fdr_threshold = sorted_beliefs[k]
            # print(k, fdr_threshold)
        except ValueError:
            fdr_threshold = 1.0
        if '/' in name:
            print('Slash character detected in the group name. It will be replaced with an underscore when saving results.')
            name = name.replace('/', '_')
        filename = os.path.join(folder_belief, f'{name}.txt')
        with open(filename, 'w') as f:
            f.write(f'{fdr_threshold}')

        
        
        proms = list(np.array(prom_names)[inds])
        if use_hdf:
            if save_stat:
                lr = lr.astype(np.half)
                filename = os.path.join(folder_stat, f'{name}.hdf')
                DF(data=lr, index=proms, columns=motif_names).to_hdf(filename, key='lrt', mode='w', complevel=4, 
                                                                     complib='blosc')
            filename = os.path.join(folder_belief, f'{name}.hdf')
            DF(data=belief, index=proms, columns=motif_names).to_hdf(filename, key='belief', mode='w', complevel=4, 
                                                                    complib='blosc')
        else:
            if save_stat:
                lr = lr.astype(np.half)
                filename = os.path.join(folder_stat, f'{name}.tsv')
                DF(data=lr, index=proms, columns=motif_names).to_csv(filename, sep='\t',
                                                                          float_format='%.3f')
            filename = os.path.join(folder_belief, f'{name}.tsv')
            DF(data=belief, index=proms, columns=motif_names).to_csv(filename, sep='\t',
                                                                          float_format='%.3f')
        