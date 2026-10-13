# -*- coding: utf-8 -*-
import numpy as np
import jax.numpy as jnp
import jax 
from .utils import read_init, openers, ProjectData
from .utils import InputError, warn, load_stage, apply_clustering
from .export_readme import write_readme
from .fit import ActivitiesPrediction, FitResult, split_data, transform_data, motif_mean_matrix, align_fit_to_promoters
from scipy.optimize import minimize
from scipy.special import expit
import os
import dill
from pandas import DataFrame as DF
from functools import partial
from tqdm import tqdm
from scipy.special import digamma
from scipy.interpolate import interp1d
from scipy.stats import f as f_dist


def held_out_inds(fit) -> list:
    """Row indices (in the full project) of the promoters held out by `fit` as a testing
    set; an empty list if there are none."""
    inds = getattr(fit, 'promoter_inds_to_drop', None)
    if inds is None:
        return list()
    return [int(i) for i in inds]

def filtered_motifs(activities) -> np.ndarray:
    """Indices of the motifs dropped by `predict`; an empty array if there are none."""
    filtered = getattr(activities, 'filtered_motifs', None)
    if filtered is None:
        return np.zeros(0, dtype=int)
    return np.asarray(filtered, dtype=int).reshape(-1)

def safe_filenames(names: list, what: str = 'group') -> list:
    """File names for the tables that are written one per item of `names`: path separators
    are replaced by underscores and names that would then coincide get a numeric suffix.
    Only the file names change; a warning lists them."""
    names = [str(n) for n in names]
    seps = {'/', os.sep} | ({os.altsep} if os.altsep else set())
    def clean(name):
        for sep in seps:
            name = name.replace(sep, '_')
        return name
    res = [clean(n) for n in names]
    used = {r for r, n in zip(res, names) if r == n}
    renamed = list()
    for i, (r, n) in enumerate(zip(res, names)):
        if r == n:
            continue
        k = 2
        while res[i] in used:
            res[i] = f'{r}_{k}'
            k += 1
        used.add(res[i])
        renamed.append(f'"{n}" -> "{res[i]}"')
    if renamed:
        shown = ', '.join(renamed[:5]) + (f', ... and {len(renamed) - 5} more' if len(renamed) > 5 else '')
        warn(f'{len(renamed)} {what} name(s) contain a path separator and cannot be used as file '
             f'names; their files are named differently: {shown}.')
    return res

def covariate_filenames(names: list) -> list:
    """File names for the tables that are written one per covariate or per term of the
    covariate design. The colon of an interaction ("age:sex_M") is not allowed in file
    names of every system and is replaced by a dot; the rest is as in safe_filenames.
    Names that coincide after that get a numeric suffix."""
    res = safe_filenames([str(n).replace(':', '.') for n in names], what='covariate')
    used = set()
    for i, name in enumerate(res):
        k = 2
        while res[i] in used:
            res[i] = f'{name}_{k}'
            k += 1
        used.add(res[i])
    return res

def load_fit(project_name: str, data: ProjectData) -> tuple:
    """Load the fit of a project and bring `data` in line with it (cluster profiles
    instead of motifs if the fit was made with "--clustering"). Returns (data, fit)."""
    # Stage files written by older versions hold jax arrays: they are restored in single
    # precision unless x64 is enabled beforehand.
    jax.config.update('jax_enable_x64', True)
    fit = load_stage(project_name, 'fit', data.fmt)
    data = apply_clustering(data, fit)
    n_motifs = len(np.asarray(fit.motif_variance.motif))
    n_test = len(held_out_inds(fit))
    n_proms = len(data.promoter_names)
    n_fit_proms = len(np.asarray(fit.promoter_mean.mean).reshape(-1))
    problem = None
    if n_motifs != len(data.motif_names):
        problem = f'{n_motifs} motifs, but the project has {len(data.motif_names)}'
    elif list(fit.group_names) != list(data.group_names):
        problem = f'{len(fit.group_names)} sample groups that differ from those of the project'
    elif len(np.asarray(fit.sample_mean.mean).reshape(-1)) != len(data.sample_names):
        problem = (f'{len(np.asarray(fit.sample_mean.mean).reshape(-1))} samples, but the project has '
                   f'{len(data.sample_names)}')
    elif n_fit_proms not in (n_proms, n_proms - n_test) or (n_test and max(held_out_inds(fit)) >= n_proms):
        problem = f'{n_fit_proms} promoters ({n_test} more held out), but the project has {n_proms}'
    if problem:
        raise InputError(f'The fit of the project "{project_name}" is out of sync with the project: it '
                         f'holds {problem}. Re-run "maradoner fit {project_name}".')
    return data, fit

def load_predict(project_name: str, data: ProjectData, fit: FitResult, required: bool = True):
    """Load the predicted activities of a project (None if absent and not `required`) and
    make sure that they belong to the given (already loaded) fit."""
    activities = load_stage(project_name, 'predict', data.fmt, required=required)
    if activities is None:
        return None
    expected = (len(data.motif_names) - len(filtered_motifs(activities)), len(fit.group_names))
    if tuple(np.shape(activities.U)) != expected or np.shape(activities.U_raw)[-1] != len(data.sample_names):
        raise InputError(f'The predicted activities of the project "{project_name}" are out of sync '
                         f'with its fit (activities of {np.shape(activities.U)[0]} motifs in '
                         f'{np.shape(activities.U)[-1]} groups were found, {expected[0]} motifs in '
                         f'{expected[1]} groups are expected). Re-run "maradoner predict {project_name}".')
    return activities


def estimate_promoter_prior_variance(data: ProjectData, activities: ActivitiesPrediction,
                                     fit: FitResult, top=0.90, eps=1e-6):
    B = data.B
    Y = data.Y
    group_inds = data.group_inds
    Y = Y - fit.promoter_mean.mean.reshape(-1, 1) - fit.sample_mean.mean.reshape(1, -1)
    Y = Y -  B @ motif_mean_matrix(fit.motif_mean, data.X)
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
    n_full = len(data.promoter_names)
    data, _ = split_data(data, fit.promoter_inds_to_drop)
    # Restrict per-promoter estimates to the training promoters kept by split_data above.
    fit = align_fit_to_promoters(fit, fit.promoter_inds_to_drop, n_full)
    # helmert=False: the per-promoter estimates used below (promoter_mean, promotor)
    # are length p, so the promoter axis must not be collapsed to p-1 here.
    data = transform_data(data, helmert=False, loading_context=fit.loading_context,
                          drist=fit.drist, loading_multipliers=fit.loading_multipliers)
    B = data.B
    Y = data.Y
    group_inds = data.group_inds

    Y = Y - fit.sample_mean.mean.reshape(1, -1)
    M = fit.promoter_mean.mean.reshape(-1, 1)
    M = M + B @ motif_mean_matrix(fit.motif_mean, data.X)
    if activities.filtered_motifs is not None:
        M = M + np.delete(B, activities.filtered_motifs, axis=1) @ activities.U_raw
    else:
        M = M + B @ activities.U_raw
    if fit.error_variance.promotor is None:
        var_mean = 1
    else:
        var_mean =fit.error_variance.promotor.mean()
    var = list()
    for inds, nu in (pbar := tqdm(list(zip(group_inds, fit.motif_variance.group)), disable=None)):
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
    Y = Y - B @ motif_mean_matrix(fit.motif_mean, data.X)
    Y = Y ** 2
    B_hat = B ** 2 * fit.motif_variance.motif
    B_hat = B_hat.sum(axis=1)
    var = list()
    jax.config.update('jax_enable_x64', True)
    jax.config.update('jax_platform_name', 'cpu') 
    for inds, prior_mean, nu in tqdm(list(zip(group_inds, prior_means, fit.motif_variance.group)), disable=None):
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

def covariate_rows(covariates: list, covariate_names: list) -> list:
    """Rows of the design matrix X that belong to the covariates named in `covariates`, in
    the order of X and without repeats. "all" stands for every covariate but the intercept
    (unless a covariate is named so)."""
    names = [str(n) for n in covariate_names]
    rows = set()
    unknown = list()
    for name in map(str, covariates):
        if name in names:
            rows.add(names.index(name))
        elif name == 'all':
            rows.update(range(1, len(names)))
        else:
            unknown.append(name)
    if len(names) < 2 and (unknown or not rows):
        raise InputError('The project has no covariates, so there are no covariate effects to test. '
                         'They are supplied with "maradoner create ... --covariates <table>".')
    if unknown:
        raise InputError(f'Unknown covariate(s): {", ".join(unknown)}. The covariates of the project are '
                         f'{", ".join(names[1:])} (and "intercept" for the mean activity; "all" stands for '
                         'every covariate but the intercept).')
    return sorted(rows)

def grn(project_name: str,  output: str, use_hdf=True, save_stat=False,
        fdr_alpha=0.05, prior_h1=1/2, include_mean: bool = True, min_variance: float = 1e-3,
        covariates: list = None):
    if not 0 < prior_h1 < 1:
        raise InputError('--prior-h1 is the prior probability that a motif contributes to the '
                         f'expression of a promoter and must lie strictly between 0 and 1, but {prior_h1:g} '
                         'was given.')
    if not 0 < fdr_alpha < 1:
        raise InputError(f'The FDR level must lie strictly between 0 and 1, but {fdr_alpha:g} was given.')
    data = read_init(project_name)
    fmt = data.fmt
    data, fit = load_fit(project_name, data)
    activities: ActivitiesPrediction = load_predict(project_name, data, fit)
    n_full = len(data.promoter_names)
    test_inds = held_out_inds(fit)
    data, _ = split_data(data, test_inds)
    # Restrict per-promoter estimates to the training promoters kept by split_data above.
    fit = align_fit_to_promoters(fit, test_inds, n_full)
    # Capture name fields off the ProjectData before `data` is reassigned to a
    # TransformedData (which only carries arrays, not name lists).
    group_names = data.group_names
    motif_names = data.motif_names
    prom_names = data.promoter_names
    covariate_names = getattr(data, 'covariate_names', None)
    # helmert=False: GRN works in the raw per-promoter space. Y, B, promoter_mean,
    # error_variance.promotor and prom_names are all length p and must stay aligned;
    # the default Helmert transform projects out the promoter mean and drops a row
    # (p -> p-1), desyncing them (shape (p,) into shape (p-1,) broadcast errors).
    # No `weights` either: unlike `predict`, GRN never whitens by the promoter
    # standard deviation -- it carries the per-promoter variance explicitly in
    # `promvar` below, and subtracts the unwhitened promoter mean.
    data = transform_data(data, helmert=False, loading_context=fit.loading_context,
                          drist=fit.drist, loading_multipliers=fit.loading_multipliers)
    dtype = np.float64
    B = np.asarray(data.B, dtype=dtype)
    Y = data.Y.astype(dtype)
    group_inds = data.group_inds
    nus = fit.motif_variance.group.astype(dtype)
    U = activities.U_raw.astype(dtype)
    # Per-sample mean activities M X: B M X is removed from the residual, and M X is the
    # mean part of the contribution of a motif tested below.
    MX = motif_mean_matrix(fit.motif_mean, data.X).astype(dtype)
    BMX = (B @ MX).astype(dtype)
    # With `covariates`, a covariate is tested in place of a group: the contribution of a
    # motif is the part of M X that is due to the covariate, M[:, c] X[c], in all samples.
    # Neither the deviations nor the other rows of X are a part of it.
    MX_cov = None
    if covariates:
        _M = np.asarray(fit.motif_mean.mean)
        _M = _M.reshape(-1, 1) if _M.ndim == 1 else _M
        covariate_names = covariate_names or ['intercept'] + [f'cov_{i}' for i in range(1, _M.shape[1])]
        cov_rows = covariate_rows(covariates, covariate_names)
        covariates = [str(covariate_names[i]) for i in cov_rows]
        # (c, m, s)
        MX_cov = (_M[:, cov_rows].T[..., None] * np.asarray(data.X)[cov_rows][:, None]).astype(dtype)
        if not include_mean:
            warn('"--no-means" has no effect together with "--covariate": only the effects of '
                 f'the covariate(s) {", ".join(covariates)} are tested.')
    # (the loading matrix is the largest object here: keep a single copy of it)
    del data
    _M = np.asarray(fit.motif_mean.mean)
    motif_mean = (_M[:, 0] if _M.ndim > 1 else _M).flatten().astype(dtype)
    motif_variance = fit.motif_variance.motif.astype(dtype)
    promoter_mean = fit.promoter_mean.mean.astype(dtype)
    sample_mean = fit.sample_mean.mean.astype(dtype)

    promvar = np.zeros((len(B), len(group_names)))
    for i, sigma in enumerate(fit.error_variance.variance):
        promvar[:, i] = sigma * fit.error_variance.promotor

    Y = Y - promoter_mean.reshape(-1, 1) - sample_mean.reshape(1, -1)
    Y = Y - BMX
    
    filtered = filtered_motifs(activities)
    R_cov = None
    if MX_cov is not None:
        # A motif dropped by `predict` has no deviations, but it has effects of covariates
        # like any other motif: all motifs are tested against a covariate. The residual of
        # the full model is the same for all covariates and is computed once, with the
        # loadings of the motifs that have deviations.
        B_kept = np.delete(B, filtered, axis=1) if len(filtered) else B
        if B_kept.shape[1] != U.shape[0]:
            raise InputError(f'Loading matrix has {B_kept.shape[1]} motifs after filtering but the '
                             f'predicted activities have {U.shape[0]}. The "{project_name}.predict" '
                             'file is out of sync with the loading matrix -- re-run "maradoner predict".')
        R_cov = Y - B_kept @ U
        del B_kept
    elif len(filtered):
        motif_names = list(np.delete(np.asarray(motif_names, dtype=object), filtered))
        B = np.delete(B, filtered, axis=1)
        MX = np.delete(MX, filtered, axis=0)
        if MX_cov is not None:
            MX_cov = np.delete(MX_cov, filtered, axis=1)
        motif_mean = np.delete(motif_mean, filtered)
        motif_variance = np.delete(motif_variance, filtered)
    if R_cov is None and B.shape[1] != U.shape[0]:
        raise InputError(f'Loading matrix has {B.shape[1]} motifs after filtering but the '
                         f'predicted activities have {U.shape[0]}. The "{project_name}.predict" '
                         'file is out of sync with the loading matrix -- re-run "maradoner predict".')

    # B_hat = B ** 2 * motif_variance
    # B_hat = B_hat.sum(axis=1, keepdims=True) - B_hat

    
    folder_stat = os.path.join(output, 'lr')
    folder_belief = os.path.join(output, 'belief')
    if save_stat:
        os.makedirs(folder_stat, exist_ok=True)
    os.makedirs(folder_belief, exist_ok=True)
    file_names = safe_filenames(group_names)
    tasks = list(zip(promvar.T[..., None], nus,  group_names, file_names, group_inds))
    sample_weights = None
    if MX_cov is not None:
        # One table per covariate over all samples. The error variance of a sample is that
        # of its group: a sample enters with the weight sigma_h / sigma_g, where sigma_h is
        # the harmonic mean of the variances over samples, and sigma_h takes the place of
        # the variance of a group.
        folder_stat = os.path.join(folder_stat, 'covariates')
        folder_belief = os.path.join(folder_belief, 'covariates')
        if save_stat:
            os.makedirs(folder_stat, exist_ok=True)
        os.makedirs(folder_belief, exist_ok=True)
        n_samples = Y.shape[1]
        sample_weights = np.zeros(n_samples, dtype=dtype)
        for sigma_g, inds_g in zip(fit.error_variance.variance, group_inds):
            sample_weights[inds_g] = 1 / sigma_g
        sigma_h = 1 / sample_weights.mean()
        sample_weights *= sigma_h
        sigma = (sigma_h * np.asarray(fit.error_variance.promotor) * np.ones(len(B))).reshape(-1, 1)
        tasks = [(sigma, None, name, file_name, np.arange(n_samples))
                 for name, file_name in zip(covariates, covariate_filenames(covariates))]
    logit_prior = np.log(prior_h1) - np.log1p(-prior_h1)
    # The likelihood ratios are stored in single precision: larger ones are capped.
    loglr_max = 88.72
    n_capped = 0
    skipped = dict()
    for n_task, (sigma, nu, name, file_name, inds) in enumerate(pbar := tqdm(tasks, dynamic_ncols=True, disable=None)):
        pbar.set_postfix_str(name)
        # var = (B_hat * nu + sigma)
        var = sigma

        U_g = U[:, inds]                                  # (m, n_g)
        # Contribution of motif m to promoter p in sample n: theta_pmn = B_pm V_mn, where
        # V is the deviation U of the activity (+ its mean M X if the means are tested too).
        if MX_cov is None:
            V_g = U_g + MX[:, inds] if include_mean else U_g
        else:
            # ... or the part of the mean that is due to the covariate.
            V_g = MX_cov[n_task] * sample_weights ** 0.5
        # Residual of the full model; without motif m it becomes R_pn + theta_pmn, so
        # loglr_pm = log L(full) - log L(without m) = sum_n [(R_pn + theta_pmn)^2 - R_pn^2] / (2 var_p)
        #          = [2 B_pm (R V^T)_pm + B_pm^2 sum_n V_mn^2] / (2 var_p).
        # Only (p, m) objects are ever formed, never the (p, m, n_g) tensor.
        if MX_cov is None:
            R_g = Y[:, inds] - B @ U_g                    # (p, n_g)
        else:
            R_g = R_cov * sample_weights ** 0.5
        loglr = R_g @ V_g.T
        del R_g
        loglr *= 2
        loglr += B * (V_g ** 2).sum(axis=1)
        loglr *= B
        del U_g, V_g
        loglr /= 2 * var
        del var
        # Posterior probability of the full model; odds = likelihood ratio x prior odds,
        # in the logit scale so that large log-ratios cannot overflow.
        belief = loglr + logit_prior
        expit(belief, out=belief)

        
        inds = sigma.flatten() > min_variance
        if not inds.all():
            skipped[name] = int((~inds).sum())
            loglr = loglr[inds]
            belief = belief[inds]

        sorted_beliefs = belief[belief > 0.5]
        sorted_beliefs.sort()
        sorted_beliefs = sorted_beliefs[::-1]
        sorted_inbeliefs = 1 - sorted_beliefs
        
        # Bayesian FDR of the k most credible pairs: the mean of their 1 - belief. It grows
        # with k, so the rule is the largest k that keeps it at or below fdr_alpha:
        # mean_{i <= k} (1 - belief_i) <= fdr_alpha  <=>  sum_{i <= k} (1 - belief_i - fdr_alpha) <= 0.
        # cumulative_fdr = np.cumsum(sorted_inbeliefs) / (np.arange(len(sorted_inbeliefs)) + 1)
        sorted_inbeliefs -= fdr_alpha
        cumulative_fdr = np.cumsum(sorted_inbeliefs, out=sorted_inbeliefs)
        cumulative_fdr = cumulative_fdr <= 0
        # print(fdr_alpha)
        if cumulative_fdr.any():
            k = len(cumulative_fdr) - 1 - np.argmax(cumulative_fdr[::-1])
            fdr_threshold = sorted_beliefs[k]
            # print(k, fdr_threshold)
        else:
            fdr_threshold = 1.0
        del sorted_beliefs, sorted_inbeliefs, cumulative_fdr
        belief = belief.astype(np.half)
        # The threshold is applied by the user to the stored (rounded) beliefs: round it the
        # same way, so that "belief >= threshold" selects the pairs found above.
        fdr_threshold = float(np.half(fdr_threshold))
        if not use_hdf:
            # (the tsv tables keep three decimals)
            fdr_threshold = float(f'{fdr_threshold:.3f}')
        filename = os.path.join(folder_belief, f'{file_name}.txt')
        with open(filename, 'w') as f:
            f.write(f'{fdr_threshold}')

        if save_stat:
            n_capped += int((loglr > loglr_max).sum())
            np.minimum(loglr, loglr_max, out=loglr)
            np.exp(loglr, out=loglr)
            lr = loglr.astype(np.float32)
        del loglr
        
        proms = list(np.array(prom_names)[inds])
        if use_hdf:
            if save_stat:
                filename = os.path.join(folder_stat, f'{file_name}.hdf')
                DF(data=lr, index=proms, columns=motif_names).to_hdf(filename, key='lrt', mode='w', complevel=4, 
                                                                     complib='blosc')
            filename = os.path.join(folder_belief, f'{file_name}.hdf')
            DF(data=belief, index=proms, columns=motif_names).to_hdf(filename, key='belief', mode='w', complevel=4, 
                                                                    complib='blosc')
        else:
            if save_stat:
                filename = os.path.join(folder_stat, f'{file_name}.tsv')
                DF(data=lr, index=proms, columns=motif_names).to_csv(filename, sep='\t',
                                                                          float_format='%.6g')
            filename = os.path.join(folder_belief, f'{file_name}.tsv')
            DF(data=belief, index=proms, columns=motif_names).to_csv(filename, sep='\t',
                                                                          float_format='%.3f')
    if skipped:
        shown = ', '.join(f'{n} in {g}' for g, n in list(skipped.items())[:5])
        what = 'group' if MX_cov is None else 'covariate'
        shown += f', ... and {len(skipped) - 5} more {what}s' if len(skipped) > 5 else ''
        warn(f'Promoters whose error variance does not exceed {min_variance:g} are left out of the '
             f'tables of the respective {what} ({shown} of {len(prom_names)} promoters).')
    if n_capped:
        warn(f'{n_capped} likelihood ratio(s) exceed the range of the single-precision tables in '
             f'"{folder_stat}" and are stored as {np.exp(loglr_max):.3g}; their beliefs are not affected.')
    if MX_cov is not None:
        # (the tables of groups that may be in the folder keep their description)
        write_readme(output, grn_prior=float(prior_h1), grn_fdr=float(fdr_alpha), grn_min_variance=float(min_variance))
        return
    write_readme(output, grn_prior=float(prior_h1), grn_means=bool(include_mean), grn_fdr=float(fdr_alpha),
                 grn_min_variance=float(min_variance))
        