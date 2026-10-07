#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pandas import DataFrame as DF
# add dot
from .utils import read_init, openers, ProjectData, subs_zeros
from .utils import InputError, warn, load_stage, apply_clustering
from .fit import FOVResult, ActivitiesPrediction, FitResult, split_data, transform_data, motif_mean_matrix, align_fit_to_promoters
from .fit import variance_dof
from .grn import grn, safe_filenames, filtered_motifs, held_out_inds, load_fit, load_predict
from .export_readme import write_readme, register_file
from scipy.stats import norm, chi2, multivariate_normal, Covariance
from scipy.special import erf, ndtr
from scipy.linalg import eigh, lapack, cholesky, solve
from statsmodels.stats import multitest
import numpy as np
from enum import Enum
from tqdm import tqdm
import multiprocessing as mp
import re
from functools import partial
from scipy.integrate import quad
import math
import time
import dill
import os


class Standardization(str, Enum):
    full = 'full'
    std = 'std'
    
class ANOVAType(str, Enum):
    positive = 'positive'
    negative = 'negative'
    
def chol_inv(x: np.array):
    """
    Calculate invserse of matrix using Ctestholesky decomposition.

    Parameters
    ----------
    x : np.array
        Data with columns as variables and rows as observations.
    return_chol : bool
        Returns cholesky decomposition if True.

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
    lapack.dpotri(c, overwrite_c=True)
    mx = c + c.T - np.diag(c.diagonal())
    return mx

    
def _first_names(names, limit=5) -> str:
    names = [str(n) for n in names]
    if len(names) <= limit:
        return ', '.join(names)
    return ', '.join(names[:limit]) + f', ... and {len(names) - limit} more'


class Information():
    eps = 1e-10
    # The singularity diagnosis below is emitted once per run and kind of matrix (`label`):
    # the same rank-deficient information matrix backs several tables of the report.
    _warned_singular = set()
    # A callable returning the explanation of a singular information matrix for the
    # project at hand. Set by export_results, which knows the design and the loadings.
    _singular_cause = None

    def __init__(self, fim: np.ndarray, slc=None, use_preconditioner=False, filter_items=None,
                 names=None, label=None):
        self.filter_items = filter_items
        if filter_items is not None:
            fim = np.delete(fim, filter_items, axis=0)
            fim = np.delete(fim, filter_items, axis=1)
            if names is not None:
                names = np.delete(np.asarray(names, dtype=object), filter_items)
        # Names of the parameters (rows of the matrix), used to say which of them are not
        # identified when the matrix is singular.
        self.names = names
        self.label = label
        # Computed on demand by cholesky_transform, the only consumer.
        self.square_root_inv = None
        precond = 1 / fim.diagonal() ** 0.5
        if not use_preconditioner:
            precond[:] = 1
        fim = precond.reshape(-1, 1) * fim * precond
        self.fim = fim
        self.precond = precond
        self.slice = slice(None, None) if slc is None else slc
        self._fim_inv = None

    def _report_singular(self, w: np.ndarray, v: np.ndarray):
        label = self.label if self.label else 'model parameters'
        if label in Information._warned_singular:
            return
        first = not Information._warned_singular
        Information._warned_singular.add(label)
        bad = w <= 1e-9 * np.abs(w).max()
        if not bad.any():
            bad[np.argmin(w)] = True
        n_bad = int(bad.sum())
        msg = (f'The information matrix of the {label} is not positive-definite: {n_bad} of '
               f'{len(w)} directions are not identified (eigenvalues range from {w.min():.3e} to '
               f'{w.max():.3e}). Falling back to a pseudo-inverse.')
        if self.names is not None and len(self.names) == len(w):
            # Share of each parameter in the unidentified directions (does not depend on
            # the choice of the basis of that subspace).
            involved = np.where((v[:, bad] ** 2).sum(axis=1) > 1e-2)[0]
            msg += f' Parameters involved: {_first_names(np.asarray(self.names, dtype=object)[involved])}.'
        if first:
            cause = Information._singular_cause
            msg += ' ' + (cause() if cause is not None else
                          'Possible causes are linearly dependent motif loadings or too few '
                          'samples for the number of sample groups and covariates.')
        msg += ' Standard errors, correlations and tests of the affected parameters are unreliable.'
        warn(msg)

    def _inv(self, x: np.ndarray):
        x = np.array(x)
        # t = np.linalg.eigh(x)
        try:
            x = chol_inv(x)
        except Exception:
            assert np.allclose(x, x.T), x - x.T
            w, v = np.linalg.eigh(x)
            self._report_singular(w, v)
            # x = np.linalg.pinv(x, hermitian=True)
            x = v * (1/np.clip(w, self.eps, float('inf'))) @ v.T
        return x

    def _inv_fim(self):
        # The inverse backs standardize(), covariance() and correlation(): computed once.
        if self._fim_inv is None:
            self._fim_inv = self._inv(self.fim)
        return self._fim_inv

    def _square_root_inv(self, x: np.ndarray, slc=None, corr=True):
        x = self._inv(x)
        if corr:
            istd =  1 / x.diagonal() ** 0.5
            x = istd.reshape(-1, 1) * x * istd
        if slc is not None:
            x = x[slc, slc]
        try:
            x = cholesky(x)
        except:
            w, v = np.linalg.eigh(x)
            # Clip before the square root: a rank-deficient (or merely round-off
            # indefinite) matrix has tiny negative eigenvalues whose sqrt would be NaN
            # and would silently poison the exported standard errors.
            x = v * np.clip(w, 0.0, None) ** (1 / 2) @ v.T
        return x
    
    def standardize(self, x: np.ndarray, 
                    mode: Standardization=Standardization.std,
                    return_std=True):
        if self.filter_items is not None:
            x = np.delete(x, self.filter_items)
        x = x / self.precond[self.slice]
        cov = self._inv_fim()
        cov = cov[self.slice, self.slice]
        std = cov.diagonal() ** 0.5
        if mode == mode.std:
            x /= std
        elif mode == mode.full:
            istd = 1 / std
            cor = istd.reshape(-1, 1) * cov * istd
            e = np.linalg.eigh(cor)
            T = istd.reshape(-1,1) * e[1] * e[0] ** (-0.5) @ e[1].T
            x = T @ x.reshape(-1, 1)
        if return_std:
            return x.flatten(), std * self.precond[self.slice]
        return x.flatten()
    
    def covariance(self):
        cov = self._inv_fim()
        cov = cov[self.slice, self.slice]
        return self.precond[self.slice].reshape(-1, 1) * cov * self.precond[self.slice]

    def correlation(self):
        cov = self.covariance()
        d = cov.diagonal() ** (-0.5)
        return d.reshape(-1, 1) * cov * d

    def cholesky_transform(self, x: np.ndarray):
        if self.square_root_inv is None:
            self.square_root_inv = self._square_root_inv(self.fim, self.slice, corr=True)
        return self.square_root_inv.T @ x
        



def export_fov(fovs: tuple[FOVResult], folder: str,
               promoter_names: list[str], sample_names: list[str]):
    os.makedirs(folder, exist_ok=True)
    cols = ['null', 'means', 'motif_means']
    fov_null, fov_means, fov_motif_means = fovs
    total = [fov_null.total, fov_means.total, fov_motif_means.total]
    DF(total, index=cols, columns=['FOV']).T.to_csv(os.path.join(folder, 'total.tsv'), sep='\t')
    promoters = [fov_null.promoter[:, None], fov_means.promoter[:, None], fov_motif_means.promoter[:, None]]
    promoters = np.concatenate(promoters, axis=-1)
    DF(promoters, index=promoter_names, columns=cols).to_csv(os.path.join(folder, 'promoters.tsv'), sep='\t')
    samples = [fov_null.sample[:, None], fov_means.sample[:, None], fov_motif_means.sample[:, None]]
    samples = np.concatenate(samples, axis=-1)
    DF(samples, index=sample_names, columns=cols).to_csv(os.path.join(folder, 'samples.tsv'), sep='\t')




def _posterior_anova(activities: ActivitiesPrediction, groups=None):
    """Posterior ANOVA from the factors of the group covariances, cov_i = Q diag(w_i) Q^T.
    All groups share Q, so the pooled (GLS) mean is found in that basis,

        mean = Q [sum_i (Q^-1 u_i) / w_i] / [sum_i 1 / w_i],

    in O(g m^2) time and O(m^2 + g m) memory: no m x m matrix is built per group."""
    Q, W = activities.cov_factors()
    U = np.asarray(activities.U, dtype=float)
    if groups is not None:
        groups = np.asarray(groups, dtype=int)
        W = W[groups]
        U = U[:, groups]
    m, g = U.shape
    bad_inds = np.zeros(m, dtype=bool)
    if g < 2:
        nans = np.full(m, np.nan)
        return np.zeros(m), nans, nans.copy(), bad_inds
    sv = np.linalg.svd(Q, compute_uv=False) if m else np.ones(1)
    if np.all(np.isfinite(W)) and np.all(W > 0) and sv[-1] > sv[0] * m * np.finfo(float).eps * 1e2:
        iw = 1 / W
        T = np.linalg.solve(Q, U)
        mean = Q @ ((T * iw.T).sum(axis=1) / iw.sum(axis=0))
        var = (Q ** 2) @ W.T
    else:
        # A singular Q (e.g. a motif variance of exactly zero) has no common basis for
        # the precisions: pool them with pseudo-inverses, one group at a time.
        total_prec = np.zeros((m, m))
        mean = np.zeros(m)
        var = np.empty((m, g))
        for i, (w, u) in enumerate(zip(W, U.T)):
            cov = (Q * w) @ Q.T
            prec = np.linalg.pinv(cov, hermitian=True)
            total_prec += prec
            mean += prec @ u
            var[:, i] = cov.diagonal()
        mean = np.linalg.pinv(total_prec, hermitian=True) @ mean
    istds = np.array([1 / subs_zeros(v ** 0.5) for v in var.T]).T
    stats = (((U - mean.reshape(-1, 1)) * istds) ** 2).sum(axis=-1)
    pvalues = chi2.sf(stats, g - 1)
    fdr = multitest.multipletests(pvalues, alpha=0.05, method='fdr_by')[1]
    return stats, pvalues, fdr, bad_inds


def posterior_anova(activities: ActivitiesPrediction, fit: FitResult,
                    B: np.ndarray, corr_stat=False, map_cov=False,
                    groups=None):
    if not map_cov:
        return _posterior_anova(activities, groups=groups)
    precs = list()
    istds = list()
    covs = list()
    mean = 0.0
    bad_inds = np.zeros(activities.U.shape[0], dtype=bool)
    # for cov, U, nu in zip(activities.cov(), activities.U.T, fit.motif_variance.group):
    #     mot = fit.motif_variance.motif
    #     mot = np.delete(mot, activities.filtered_motifs)
    #     ind = mot * nu < cov.diagonal() + 1e-9
    #     bad_inds[ind] = True
    # mot = fit.motif_variance.motif
    # mot = np.delete(mot, activities.filtered_motifs)[~bad_inds]
    motif_variance = fit.motif_variance.motif
    if activities.filtered_motifs is not None:
        motif_variance = np.delete(motif_variance, activities.filtered_motifs)
        B = np.delete(B, activities.filtered_motifs, axis=1)
    U = activities.U
    if map_cov:
        # fit.motif_variance.m
        BTB = B.T @ B
        BTB_s = BTB * motif_variance ** 0.5
        BTB_s = BTB_s @ BTB_s.T
    for i, (cov, U, sigma, n, nu) in enumerate(zip(activities.cov(), U.T, 
                          activities._cov[-2], 
                          fit.error_variance.variance, fit.motif_variance.group)):
        # cov = cov[~bad_inds, ~bad_inds]
        # cov = cov[..., ~bad_inds]
        # cov = cov[~bad_inds]
        if groups and i not in groups:
            continue
        if map_cov:
            D = BTB_s * nu  + np.identity(len(BTB)) * sigma
            cov = cov @ D @ cov.T * n / sigma ** 2 
        covs.append(cov)
        # U = U[~bad_inds]
        # prec = np.linalg.inv(np.diag(mot * nu) - cov)
        prec = np.linalg.pinv(cov, hermitian=True)
        mean += prec @ U
        precs.append(prec)
    total_prec = sum(precs)
    total_cov = np.linalg.pinv(total_prec, hermitian=True)
    mean = total_cov @ mean
    stats = activities.U[~bad_inds] - mean.reshape(-1, 1)
    # if corr_stat:
    #     istd = 1 / total_cov.diagonal() ** 0.5
    #     total_cor = istd.reshape(-1, 1) * total_cov * istd
    #     stats = total_cor @ stats
    #     total_cov = total_cor @ total_cov @ total_cor
    # stats = (1 / total_cov.diagonal().reshape(-1, 1)) ** 0.5 * stats
    istds = [1 / subs_zeros(c.diagonal() ** 0.5) for c in covs]
    istds = np.array(istds).T 
    if groups:
        stats = stats[:, np.array(groups)]
    # print(stats, stats.shape, np.array(groups))
    stats = stats * istds
    stats = stats ** 2
    stats = stats.sum(axis=-1)
    # print(stats)
    pvalues = chi2.sf(stats, len(precs) - 1)
    fdr = multitest.multipletests(pvalues, alpha=0.05, method='fdr_by')[1] 
    return stats, pvalues, fdr, bad_inds
    

def calc_log_counts(data: ProjectData, fit: FitResult, activities: ActivitiesPrediction, groups: bool = True,
                    mu_p: np.ndarray = None):
    sample_names = data.sample_names
    group_names = data.group_names
    promoter_names = data.promoter_names
    data = transform_data(data, helmert=False, loading_context=fit.loading_context, 
                          drist=fit.drist, loading_multipliers=fit.loading_multipliers)
    B = data.B
    # Per-sample mean motif effect B M X (covariate-aware; p x s).
    mu_m = B @ motif_mean_matrix(fit.motif_mean, data.X)
    B = np.delete(B, filtered_motifs(activities), axis=1)
    mu_s = fit.sample_mean.mean.reshape(1, -1)
    if mu_p is None:
        mu_p = fit.promoter_mean.mean.reshape(-1, 1)
    mu_p = mu_p.reshape(-1, 1)
    # if len(mu_p) != len(B):
    #     Y = data.Y  - mu_s - mu_m - B @ activities.U_raw
    #     D = (1 / fit.error_variance.variance)[data.group_inds_inv].reshape(-1, 1)
    #     mu_p = (Y @ D / (D.sum())).reshape(-1, 1)
    if not groups:
        cols = sample_names
        U = activities.U_raw
    else:
        lt = list()
        mu_m_lt = list()
        mu_s = mu_s.flatten()
        for inds in data.group_inds:
            lt.append(np.mean(mu_s[inds]))
            mu_m_lt.append(mu_m[:, inds].mean(axis=1))
        mu_s = np.array(lt).reshape(1, -1)
        mu_m = np.array(mu_m_lt).T          # p x g, group-averaged covariate effect
        cols = group_names
        U = activities.U
    log_counts = mu_s + mu_m + B @ U + mu_p
    return DF(log_counts, index=promoter_names, columns=cols)
    


def export_log_counts(output_folder: str, data: ProjectData, fit: FitResult,
                      activities: ActivitiesPrediction, group: bool = True,
                      mu_p_test: np.ndarray = None):
    os.makedirs(output_folder, exist_ok=True)
    test_inds = held_out_inds(fit)
    data_train, data_test = split_data(data, test_inds)
    # A no-op when the per-promoter estimates are already restricted to the training set.
    fit = align_fit_to_promoters(fit, test_inds, len(data.promoter_names))
    log_counts = calc_log_counts(data_train, fit, activities, groups=group)
    log_counts.to_csv(os.path.join(output_folder, 'train.tsv'), sep='\t')
    if data_test and mu_p_test is not None:
        log_counts = calc_log_counts(data_test, fit, activities, groups=group,
                                     mu_p=mu_p_test)
        log_counts.to_csv(os.path.join(output_folder, 'test.tsv'), sep='\t')
    

def motif_mean_covariance(motif_mean, motif_names=None, covariate_names=None, verbose=True) -> tuple:
    """Standard errors of the motif-mean matrix M and the covariance of its intercept
    column mu_m (the average motif activities). Returns (std, cov): std has the shape
    (m, c) of M, cov is (m, m).

    The covariance of vec(M) is the inverse of the FULL mc x mc Fisher matrix. Inverting
    each covariate's m x m diagonal block in isolation ignores correlations *between*
    covariates and severely underestimates the standard errors when covariates are
    collinear (e.g. sex vs sex-specific male_age/female_age). vec(M) is stored
    column-major, so coefficient (motif i, covariate a) is index a*m + i.
    The intercept column additionally carries the uncertainty of the promoter baselines
    (only mu_p + B mu_m is identified): the fit supplies it as `baseline_cov`, which is
    added to the intercept block."""
    M = np.asarray(motif_mean.mean)
    if M.ndim == 1:
        M = M.reshape(-1, 1)
    m, c = M.shape
    names = None
    if motif_names is not None:
        if covariate_names is None or len(covariate_names) != c:
            covariate_names = ['intercept'] + [f'cov_{i}' for i in range(1, c)]
        names = [f'mean of {n}' if a == 0 else f'effect of {cname} on {n}'
                 for a, cname in enumerate(covariate_names) for n in motif_names]
    fim = np.asarray(motif_mean.fim, dtype=np.float64)
    w, v = np.linalg.eigh(fim)
    cutoff = len(w) * np.finfo(float).eps * np.abs(w).max()
    if w.min() <= cutoff:
        Information(fim, names=names, label='motif means')._report_singular(w, v)
    # Pseudo-inverse: for a singular matrix this is the covariance of the minimum-norm
    # estimate that the fit returns (unidentified directions contribute nothing).
    keep = np.abs(w) > cutoff
    cov_full = (v[:, keep] / w[keep]) @ v[:, keep].T
    var = np.clip(np.diag(cov_full), 0.0, None).reshape(c, m).T
    cov = cov_full[:m, :m]
    baseline_cov = getattr(motif_mean, 'baseline_cov', None)
    if baseline_cov is None:
        warn('This fit was made by a MARADONER version that did not account for the uncertainty of '
             'the promoter baselines in the standard errors of the motif means: these standard '
             'errors are too small and the tests based on them are too optimistic. Re-run '
             '"maradoner fit" (and "predict") to correct this.', verbose)
    else:
        cov = cov + np.asarray(baseline_cov, dtype=np.float64)
        var[:, 0] = np.clip(np.diag(cov), 0.0, None)
    return var ** 0.5, cov


def _explain_singular_information(data: ProjectData, fit: FitResult) -> str:
    """Finds out why an information matrix of this project is singular: `data` are the
    training promoters, `fit` the fit aligned with them."""
    info = variance_dof(data.X, data.group_inds)
    if not info['ok']:
        return (f'Cause: the group variances are over-determined. After accounting for the covariates, '
                f'{info["dof"]} sample dimensions remain ({info["n_samples"]} samples, design of rank '
                f'{info["rank_x"]}), but {info["n_free_nu"]} group variances have to be estimated '
                f'({info["n_groups"]} sample groups). Pool samples into fewer, larger groups with '
                '"create --sample-groups" and/or use fewer covariates.')
    B = transform_data(data, helmert=False, loading_context=fit.loading_context,
                       drist=fit.drist, loading_multipliers=fit.loading_multipliers).B
    B = B - B.mean(axis=0, keepdims=True)
    w, v = np.linalg.eigh(B.T @ B)
    null = w <= 1e-9 * max(w.max(), 0.0)
    if null.any():
        involved = np.where((v[:, null] ** 2).sum(axis=1) > 1e-2)[0]
        names = np.asarray(data.motif_names, dtype=object)[involved]
        return (f'Cause: the loadings of {len(involved)} motif(s) are linearly dependent across the '
                f'promoters (duplicated, collinear or constant columns of the loading matrix): '
                f'{_first_names(names)}. Their individual variances and means are not identifiable, only '
                'combinations of them are. Remove the redundant motifs (e.g. "create --motif-filename") '
                'or merge them ("fit --clustering").')
    return ('Neither a shortage of samples nor linearly dependent motif loadings explain it, so this '
            'is likely a numerical or convergence problem of the fit.')


def export_results(project_name: str, output_folder: str,
                   std_mode: Standardization, 
                   anova_mode: ANOVAType=ANOVAType.positive,
                   weighted_zscore=False,
                   alpha=0.05,
                   export_B: bool = False,
                   export_counts: bool = False,
                   counts_grouped: bool = True,
                   n_jobs=6,
                   zero_margin: float = None):
    
    def calc_z_test(x):
        if anova_mode == ANOVAType.negative:
            # Lower tail P(T <= t) of the Wald statistic t = tau / std under tau = 0,
            # where T = |N(0, 1)|. A small value says that the estimate is unusually close
            # to zero relative to its standard error. It is NOT a p-value of a test showing
            # that the variance is zero: no such test exists without an equivalence margin.
            pval = erf(x / np.sqrt(2))
        else:
            # One-sided: tau = 0 lies on the boundary of the parameter space, so only
            # large positive tau / std speak against it.
            pval = norm.sf(x)
        return pval

    if zero_margin is not None and not zero_margin > 0:
        raise InputError(f'The equivalence margin of the zero-mean test must be positive, but {zero_margin:g} '
                         'was given.')
    data = read_init(project_name)
    fmt = data.fmt
    motif_names = data.motif_names
    prom_names = data.promoter_names
    if export_B:
        B = data.B
        B = DF(B, index=prom_names, columns=motif_names)
        os.makedirs(output_folder, exist_ok=True)
        B.to_csv(os.path.join(output_folder, 'B.tsv'), sep='\t')
    # del data
    data, fit = load_fit(project_name, data)
    motif_names = data.motif_names
    act: ActivitiesPrediction = load_predict(project_name, data, fit)
    filtered = filtered_motifs(act)
    if not len(act.U):
        raise InputError(f'All {len(motif_names)} motifs of the project "{project_name}" were filtered out by '
                         '"predict", so there is nothing to export. Re-run "maradoner predict" with a larger '
                         '"--filter-order" or with "--no-filter-motifs".')
    test_inds = held_out_inds(fit)
    prom_names_test = None
    if test_inds:
        prom_names_test = np.array(prom_names)[test_inds]
        prom_names = np.delete(prom_names, test_inds)
    # Per-promoter estimates in `fit` may span the full promoter set; restrict them to the
    # training promoters so they line up with the (trimmed) `prom_names` written below.
    fit = align_fit_to_promoters(fit, test_inds, len(data.promoter_names))
    group_names = fit.group_names
    motif_names_filtered = np.delete(motif_names, filtered)

    # Everything the requested tables depend on is checked before anything is written.
    mu_p_test = load_stage(project_name, 'promoter_mean', fmt, required=False)
    if mu_p_test is not None:
        mu_p_test = np.asarray(mu_p_test, dtype=float)
        if mu_p_test.ndim == 0 and test_inds:
            # Older versions stored a single zero for "--mean-mode null".
            mu_p_test = np.full(len(test_inds), float(mu_p_test))
        mu_p_test = mu_p_test.reshape(-1)
        if len(mu_p_test) != len(test_inds) or not test_inds:
            warn(f'Ignoring "{project_name}.promoter_mean.{fmt}": it holds means of {len(mu_p_test)} '
                 f'promoter(s), whereas the current fit has {len(test_inds)} test promoter(s). It was '
                 f'left by an earlier fit; re-run "maradoner estimate-promoter-mean {project_name}" '
                 'if the fit has a testing set.')
            mu_p_test = None
    if export_counts and test_inds and mu_p_test is None:
        raise InputError('Expression predictions for the test promoters need their promoter-wise means: '
                         f'run "maradoner estimate-promoter-mean {project_name}" first (see its '
                         '"--mean-mode" option).')
    fov = load_stage(project_name, 'fov', fmt, required=False)
    if fov is not None:
        n_stat = len(group_names) if fov.grouped else len(data.sample_names)
        if len(fov.train[0].promoter) != len(prom_names) or len(fov.train[0].sample) != n_stat or \
           (fov.test is not None and len(fov.test[0].promoter) != len(test_inds)):
            warn(f'Ignoring "{project_name}.fov.{fmt}": it does not match the current fit (it was '
                 f'left by an earlier one). Re-run "maradoner gof {project_name}" to export the '
                 'goodness-of-fit statistics.')
            fov = None

    os.makedirs(output_folder, exist_ok=True)

    Information._warned_singular = set()
    def singular_cause():
        return _explain_singular_information(split_data(data, test_inds)[0], fit)
    Information._singular_cause = singular_cause

    error_variance = fit.error_variance.variance
    # For backwards comptatability # TODO: rename promotor to promoter
    promoter_variance = getattr(fit.error_variance, 'promotor', None)
    error_variance_fim = Information(fit.error_variance.fim, names=[f'sigma of {g}' for g in group_names],
                                     label='error variances')
    error_variance_stat, error_variance_std = error_variance_fim.standardize(error_variance, 
                                                                             mode=Standardization.std)
    
    motif_variance = fit.motif_variance.motif
    excluded_motif_group = fit.motif_variance.fixed_group
    if excluded_motif_group is not None:
        excluded_motif_group = int(excluded_motif_group)
    variance_names = [f'tau of {n}' for n in motif_names] + \
                     [f'nu of {g}' for i, g in enumerate(group_names) if i != excluded_motif_group]
    motif_variance_fim = Information(fit.motif_variance.fim, slice(None, len(motif_names_filtered)), 
                                     filter_items=filtered, names=variance_names,
                                     label='variances of motif activities')
    motif_variance_stat, motif_variance_std = motif_variance_fim.standardize(motif_variance, 
                                                                             mode=Standardization.std)
    # With "fit --no-motif-variance" all motif variances are fixed to one: there is nothing
    # to estimate or to test per motif.
    motif_variance_estimated = excluded_motif_group is not None
    if not motif_variance_estimated:
        motif_variance_std = np.full(len(motif_variance_std), np.nan)
    
    motif_group_variance = fit.motif_variance.group
    motif_group_variance_fim = Information(fit.motif_variance.fim, slice(len(motif_names), None),
                                           names=variance_names, label='variances of motif activities')
    motif_group_variance_std = motif_group_variance_fim.covariance().diagonal() ** 0.5
    
    
    # motif_mean.mean is the (m, c) matrix M (U ~ MN(M X, Sigma, G)); column 0 is
    # the intercept mu_m (average activity). Its Fisher block is fim[:m, :m].
    M_matrix = np.asarray(fit.motif_mean.mean)
    if M_matrix.ndim == 1:
        M_matrix = M_matrix.reshape(-1, 1)
    n_motifs = M_matrix.shape[0]
    covariate_names = getattr(data, 'covariate_names', None) or \
        ['intercept'] + [f'cov_{i}' for i in range(1, M_matrix.shape[1])]
    motif_mean = M_matrix[:, 0]
    # Joint standard errors of M and the covariance of its intercept column.
    joint_std, motif_mean_cov = motif_mean_covariance(fit.motif_mean, motif_names, covariate_names)
    motif_mean_std = joint_std[:, 0]

    promoter_mean = fit.promoter_mean.mean.flatten()
    # del fit
    
    
    folder = os.path.join(output_folder, 'params')
    os.makedirs(folder, exist_ok=True)
    promvar = load_stage(project_name, 'promvar', fmt, required=False)
    if promvar is not None and np.shape(promvar) == (len(prom_names), len(group_names)):
        DF(promvar, index=prom_names, columns=group_names).to_csv(os.path.join(folder, 'promoter_variances.tsv'), sep='\t')
    if excluded_motif_group is not None:
        motif_group_variance_std = np.insert(motif_group_variance_std, excluded_motif_group, np.nan)
    DF(np.array([error_variance, error_variance_std, motif_group_variance, motif_group_variance_std]).T,
                index=group_names,
                columns=['sigma', 'sigma_std', 'nu', 'nu_std']).to_csv(os.path.join(folder, 'group_variances.tsv'),
                                                             sep='\t')
    s = 'motif\ttau\tstd\n' + '\n'.join(f'{a}\t{b}\t{c}' for a, b, c in zip(motif_names_filtered,
                                                                            np.delete(motif_variance, filtered),
                                                                            motif_variance_std))
    with open(os.path.join(folder, 'motif_variances.tsv'), 'w') as f:
        f.write(s)
    s = 'motif\ttau\n' + '\n'.join(f'{a}\t{b}' for a, b in zip(np.array(motif_names)[filtered],
                                                               motif_variance[filtered]
                                                              ))
    with open(os.path.join(folder, 'filtered_motif_variances.tsv'), 'w') as f:
        f.write(s)
    
    if fit.loading_context is not None:
        R = fit.loading_context.R.T
        DF(R, index=motif_names).to_csv(os.path.join(folder, 'motif_context.tsv'), sep='\t')
    
    if promoter_variance is None or promoter_variance.var() == 0:
        DF(promoter_mean, index=prom_names, columns=['mean']).to_csv(os.path.join(folder, 'promoter_means.tsv'),
                                                                     sep='\t')
    else:
        # "fit --promoter-variance" does not estimate promoter variances: it reweights the
        # promoters heuristically. `weight` is the weight of the squared residuals of a
        # promoter in the fit (its rows of Y and B are multiplied by sqrt(weight)).
        DF(np.array([promoter_mean, 1 / promoter_variance]).T, index=prom_names,
               columns=['mean', 'weight']).to_csv(os.path.join(folder, 'promoter.tsv'),
                                               sep='\t')
    DF(np.array([motif_mean, motif_mean_std]).T,
       index=motif_names, columns=['mean', 'std']).to_csv(os.path.join(folder, 'motif_means.tsv'),
                                                          sep='\t')

    # Per-covariate motif-mean effects: one TSV per covariate (intercept included)
    # in params/covariates/, with columns effect, std, z, pvalue, fdr. The effect for
    # the intercept is the average motif activity mu_m; for other covariates it is the
    # change in motif activity per unit (z-scored numeric / encoded categorical).
    cov_folder = os.path.join(folder, 'covariates')
    os.makedirs(cov_folder, exist_ok=True)
    # Clear stale per-covariate tables so a re-run with different covariates does not
    # leave behind files for covariates no longer in the model.
    for fname in os.listdir(cov_folder):
        if fname.endswith('.tsv'):
            os.remove(os.path.join(cov_folder, fname))
    cov_file_names = safe_filenames([covariate_names[a] if a < len(covariate_names) else f'cov_{a}'
                                     for a in range(M_matrix.shape[1])], what='covariate')
    for a in range(M_matrix.shape[1]):
        eff = M_matrix[:, a]
        eff_std = joint_std[:, a]            # joint SE (accounts for covariate collinearity)
        z = eff / eff_std
        pvalue = 2 * norm.sf(np.abs(z))
        fdr = multitest.multipletests(pvalue, method='fdr_bh')[1]
        cname = cov_file_names[a]
        DF({'effect': eff, 'std': eff_std, 'z': z, 'pvalue': pvalue, 'fdr': fdr},
           index=motif_names).to_csv(os.path.join(cov_folder, f'{cname}.tsv'), sep='\t')
    
                                                          
    if mu_p_test is not None:
        DF(mu_p_test, index=prom_names_test, columns=['mean']).to_csv(os.path.join(folder, 'test_promoter_means.tsv'),
                                                                      sep='\t')                                                      
    DF(np.asarray(fit.sample_mean.mean).reshape(-1), index=data.sample_names,
       columns=['mean']).to_csv(os.path.join(folder, 'sample_means.tsv'), sep='\t')
    folder = os.path.join(folder, 'correlations')
    os.makedirs(folder, exist_ok=True)
    # (kept for backward compatibility; the labelled table is params/sample_means.tsv)
    DF(fit.sample_mean.mean).to_csv(os.path.join(folder, 'sample_means.tsv'),
                                                                      sep='\t')
    d = motif_mean_cov.diagonal() ** (-0.5)
    DF(d.reshape(-1, 1) * motif_mean_cov * d, index=motif_names, columns=motif_names).to_csv(os.path.join(folder, 'motif_means.tsv'),
                                                                      sep='\t')
    DF(motif_variance_fim.correlation(), index=motif_names_filtered, columns=motif_names_filtered).to_csv(os.path.join(folder, 'motif_variances.tsv'),
                                                                      sep='\t')
    _group_names = group_names
    if excluded_motif_group is not None:
        _group_names = np.delete(_group_names, excluded_motif_group)
    DF(motif_group_variance_fim.correlation(), index=_group_names, columns=_group_names).to_csv(os.path.join(folder, 'motif_group_variances.tsv'),
                                                                      sep='\t')

    DF(error_variance_fim.correlation(), index=group_names, columns=group_names).to_csv(os.path.join(folder, 'error_variances.tsv'),
                                                                      sep='\t')
    
    
    folder = output_folder

    folder = os.path.join(output_folder, 'tests', 'prediction_based')
    os.makedirs(folder, exist_ok=True)

    if len(group_names) < 2:
        warn('The project has a single sample group: the posterior ANOVA compares groups, so its '
             'p-values in tests/prediction_based/anova.tsv are empty.')
    stat, pvalue, fdr, bad_inds = posterior_anova(act, fit, B=data.B)
    motif_names_filtered = np.array(motif_names_filtered)[~bad_inds]
    anova = DF([stat, pvalue, fdr], columns=motif_names_filtered, index=['stat', 'p-value', 'FDR']).T
    anova.to_csv(os.path.join(folder, 'anova.tsv'), sep='\t')

    folder = os.path.join(output_folder, 'tests', 'asymptotics_based')
    os.makedirs(folder, exist_ok=True)
    
    if not motif_variance_estimated:
        warn('The model was fitted with "--no-motif-variance": all motifs share one variance, so there '
             'are no motif-specific variances to test or to give standard errors for. '
             'tests/asymptotics_based/anova.tsv is not written and the "std" column of '
             'params/motif_variances.tsv is empty.')
        if os.path.isfile(os.path.join(folder, 'anova.tsv')):
            os.remove(os.path.join(folder, 'anova.tsv'))
        anova_kind = None
    else:
        wald = motif_variance_stat
        wald_pval = calc_z_test(wald)
        wald_fdrs = multitest.multipletests(wald_pval, alpha=0.05, method='fdr_bh')[1]
        anova_ass, pval, fdrs = wald, wald_pval, wald_fdrs
        # lrt = 2 * fit.motif_variance.logratios
        # lrt_pvalues = chi2.sf(lrt, 1)
        # lrt_fdr = multitest.multipletests(lrt_pvalues, alpha=0.05, method='fdr_bh')[1]
        lrt = getattr(fit.motif_variance, 'lrt', None)
        if anova_mode == ANOVAType.negative:
            warn('"--anova-mode negative": the p-value column of tests/asymptotics_based/anova.tsv is '
                 'the probability of a Wald statistic as small as the observed one if the variance '
                 'were zero. Small values flag variances that are small compared to their standard '
                 'errors; they are not evidence that a variance is zero.')
        elif lrt is not None:
            # Likelihood-ratio test of H0: tau_i = 0. The null value lies on the boundary
            # of the parameter space, hence the reference distribution of the statistic is
            # the mixture 0.5 chi2_0 + 0.5 chi2_1 (chi2_0 is a point mass at zero).
            lrt = np.clip(np.asarray(lrt, dtype=float), 0.0, None)
            pval = np.where(lrt > 0, 0.5 * chi2.sf(lrt, 1), 1.0)
            # The FDR is controlled over all motifs of the model: those left out of the
            # table by "predict" (negligible variance) are hypotheses that were tested too.
            fdrs = multitest.multipletests(pval, alpha=0.05, method='fdr_bh')[1]
            anova_ass, pval, fdrs = (np.delete(t, filtered) for t in (lrt, pval, fdrs))
        else:
            dof = variance_dof(data.X, data.group_inds)['dof']
            warn('This fit holds no likelihood-ratio statistics of the motif variances (it was made by '
                 'an older MARADONER version, or with "--promoter-variance", whose heuristic promoter '
                 'weights do not define a likelihood), so tests/asymptotics_based/anova.tsv falls back to '
                 'the one-sided Wald test. That test has little power: with the '
                 f'{dof} residual degrees of freedom of the samples its statistic cannot exceed about '
                 f'{np.sqrt(max(dof, 0) / 2):.2f}, i.e. its p-values cannot fall below '
                 f'{norm.sf(np.sqrt(max(dof, 0) / 2)):.2g}. Re-run "maradoner fit" (without '
                 '"--promoter-variance") to obtain the likelihood-ratio test.')
        anova_kind = 'negative' if anova_mode == ANOVAType.negative else ('lrt' if lrt is not None else 'wald')
        anova_ass = DF(np.array([anova_ass, pval, fdrs, wald, wald_pval, wald_fdrs]).T, index=motif_names_filtered,
                       columns=['stat', 'p-value', 'FDR', 'wald_stat', 'wald_p-value', 'wald_fdr'])
        anova_ass.to_csv(os.path.join(folder, 'anova.tsv'), sep='\t')
    
    sign = motif_mean.flatten() / motif_mean_std
    neg = norm.cdf(sign)
    pos = norm.sf(sign)
    if zero_margin is None:
        # `zero` is 1 - (two-sided p-value) = P(|Z| <= |stat|) for a standard normal Z: a
        # small value says that the mean is unusually close to zero relative to its
        # standard error. It is NOT a p-value of a test showing that the mean is zero (it
        # is uniformly distributed when the mean is exactly zero, whatever the precision).
        zero = erf(np.abs(sign) / np.sqrt(2))
    else:
        # Equivalence (TOST) test of H0: |mean| >= margin against H1: |mean| < margin: a
        # small p-value is evidence that the mean lies within the margin around zero.
        zero = ndtr((np.abs(motif_mean.flatten()) - zero_margin) / motif_mean_std)

    neg_fdr = multitest.multipletests(neg, alpha=0.05, method='fdr_bh')[1]
    pos_fdr = multitest.multipletests(pos, alpha=0.05, method='fdr_bh')[1]
    zero_fdr = multitest.multipletests(zero, alpha=0.05, method='fdr_bh')[1]
    sign_ass = DF(np.array([sign, zero, zero_fdr, neg, neg_fdr, pos, pos_fdr]).T, columns=['stat', 
                                                                                           'zero', 'fdr_zero',
                                                                                           'pvalue_neg', 'fdr_neg',
                                                                                           'pvalue_pos', 'fdr_pos'],
                  index=motif_names)
    sign_ass.to_csv(os.path.join(folder, 'sign.tsv'), sep='\t')
    
    folder = os.path.join(output_folder, 'activities')
    os.makedirs(folder, exist_ok=True)
    U = list()
    stds = list()
    for u, var in zip(act.U.T, act.var()):
        std = subs_zeros(var ** 0.5)
        u = u / std
        U.append(u)
        stds.append(std)
    U = np.array(U).T
    DF(U, index=motif_names_filtered, columns=group_names).to_csv(os.path.join(folder, 'activity.tsv'), sep='\t')
    # activity.tsv holds the standardized activities (z-scores); these two are the group
    # activities themselves (not standardized) and their posterior standard deviations.
    DF(act.U, index=motif_names_filtered, columns=group_names).to_csv(os.path.join(folder, 'activity_group_raw.tsv'), sep='\t')
    DF(np.array(stds).T, index=motif_names_filtered, columns=group_names).to_csv(os.path.join(folder, 'activity_group_raw_std.tsv'), sep='\t')
    U = U ** 2
    if weighted_zscore:
        # Squared z-score of the inverse-variance weighted mean of the group activities:
        # (sum_g u_g / s_g^2) / sqrt(sum_g 1 / s_g^2).
        iv = 1 / np.array(stds).T ** 2
        U_total = (np.asarray(act.U) * iv).sum(axis=1, keepdims=True) ** 2 / iv.sum(axis=1, keepdims=True)
    else:
        U_total = U.mean(axis=1, keepdims=True)
    
    U = np.hstack((U_total, U)) ** 0.5
    DF(U, index=motif_names_filtered,
       columns=['overall'] + list(group_names)).to_csv(os.path.join(folder, 'z_score.tsv'), sep='\t')
    DF(act.U_raw, index=motif_names_filtered, columns=data.sample_names).to_csv(os.path.join(folder, 'activity_raw.tsv'), sep='\t')
    
    
            
    if fov is not None:
        train = fov.train
        test = fov.test
        folder = os.path.join(output_folder, 'fov')
        if fov.grouped:
            sample_names = data.group_names
        else:
            sample_names = data.sample_names
        export_fov(train, os.path.join(folder, 'train'), promoter_names=prom_names,
                   sample_names=sample_names)
        if test is not None:
            export_fov(test, os.path.join(folder, 'test'), promoter_names=prom_names_test,
                       sample_names=sample_names)
    
    
    
    if export_counts:
        folder = os.path.join(output_folder, 'log_counts')
        export_log_counts(folder, data, fit, act, counts_grouped, mu_p_test=mu_p_test)

    from . import __version__
    write_readme(output_folder, project=os.path.basename(os.fspath(project_name)), version=__version__,
                 n_motifs=len(motif_names), n_filtered=len(filtered), anova=anova_kind,
                 zero_margin=zero_margin, weighted_zscore=weighted_zscore,
                 fov_grouped=bool(fov is not None and fov.grouped), counts_grouped=counts_grouped,
                 fov_mara=False)
        
            
def export_posterior_anova(project_name: str, filename: str,
                           groups: list[str]):
    

    data = read_init(project_name)
    fmt = data.fmt
    data, fit = load_fit(project_name, data)
    motif_names = data.motif_names
    prom_names = data.promoter_names
    if held_out_inds(fit):
        prom_names = np.delete(prom_names, held_out_inds(fit))
    group_names = list(fit.group_names)
    act: ActivitiesPrediction = load_predict(project_name, data, fit)
    motif_names_filtered = np.delete(motif_names, filtered_motifs(act))
    
    groups = list(dict.fromkeys(str(g) for g in groups))
    available = f'Sample groups of the project "{project_name}": {_first_names(group_names, 20)}.'
    unknown = [g for g in groups if g not in group_names]
    if unknown:
        raise InputError(f'Unknown sample group(s): {_first_names(unknown)}. {available}')
    if len(groups) < 2:
        raise InputError(f'ANOVA compares at least two different sample groups, but {len(groups)} '
                         f'was given. {available}')
    output_folder = os.path.split(filename)[0]
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
    groups = [i for i, n in enumerate(group_names) if n in groups]
    stat, pvalue, fdr, bad_inds = posterior_anova(act, fit, B=data.B, groups=groups)
    motif_names_filtered = np.array(motif_names_filtered)[~bad_inds]
    anova = DF([stat, pvalue, fdr], columns=motif_names_filtered, index=['stat', 'p-value', 'FDR']).T
    anova.to_csv(filename, sep='\t')
    register_file(filename, 'anova', [group_names[i] for i in groups])

import ast
import sympy

def validate(node):
    """
    Validates that the AST only contains allowed operations: +, -, *, unary -, names, and numeric constants.
    """
    if isinstance(node, ast.Expression):
        return validate(node.body)
    if isinstance(node, ast.BinOp):
        validate(node.left)
        validate(node.right)
        allowed_ops = (ast.Add, ast.Sub, ast.Mult, ast.Div)
        if not isinstance(node.op, allowed_ops):
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
    elif isinstance(node, ast.UnaryOp):
        validate(node.operand)
        if not isinstance(node.op, ast.USub):
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
    elif isinstance(node, ast.Name):
        pass
    elif isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)):
            raise ValueError("Non-numeric constant")
    else:
        raise ValueError(f"Unsupported node type: {type(node).__name__}")



def parse_linear_expression(expr_str):
    """
    Parses a linear algebraic expression string with only +, -, *, parentheses, and constants.
    Raises ValueError for unsupported operations (e.g., ^, /, sin) or non-linear terms (e.g., a*b, a^2).
    Returns a dictionary of variable coefficients.
    
    Example:
    >>> parse_linear_expression("(my_variable_1 * 2 - my_variable_2) - (3 * a - b)")
    {'a': -3, 'b': 1, 'my_variable_1': 2, 'my_variable_2': -1}
    
    >>> parse_linear_expression("(my_variable_1 - my_variable_2) - (a - b) * 2")
    {'a': -2, 'b': 2, 'my_variable_1': 1, 'my_variable_2': -1}
    """
    preprocessed = expr_str.replace(' ', '').replace('^', '**')
    try:
        tree = ast.parse(preprocessed, mode='eval')
    except SyntaxError as e:
        raise ValueError(f"Invalid expression syntax: {e}")
    
    validate(tree)
    
    expr = sympy.sympify(preprocessed)
    expr = sympy.expand(expr)
    symbols = sorted(expr.free_symbols, key=str)
    coeffs = {str(s): int(expr.coeff(s)) if expr.coeff(s).is_integer else float(expr.coeff(s)) 
              for s in symbols}
    
    # Ensure it's linear
    if symbols:
        poly = expr.as_poly(symbols)
        if poly.total_degree() > 1:
            raise ValueError("Expression must be linear (no products of variables or powers >1)")
    
    return dict(sorted(coeffs.items(), key=lambda x: x[0]))


def parse_contrast(expression: str, group_names: list) -> dict:
    """
    Parses a contrast: a linear combination of sample groups with numeric coefficients,
    written with +, -, *, / and parentheses. Returns a dictionary {group name: coefficient}.

    The expression is split into tokens against the actual group names (the longest name
    that fits is taken), so names with spaces, dashes or a leading digit can be written as
    they are; any name can also be quoted with backticks or quotes. Nothing is handed to
    Python or SymPy for evaluation.

    Example:
    >>> parse_contrast("(a + b) / 2 - `T-cell`", ['a', 'b', 'T-cell'])
    {'a': 0.5, 'b': 0.5, 'T-cell': -1.0}
    """
    expression = str(expression)
    group_names = [str(n) for n in group_names]
    def fail(reason: str):
        raise InputError(f'Cannot interpret the contrast "{expression}": {reason} A contrast is a '
                         'linear combination of sample groups, e.g. "treated - control" or '
                         '"(a + b) / 2 - c"; names with spaces, operators or a leading digit can be '
                         f'quoted with backticks. Sample groups of this project: {_first_names(group_names, 20)}.')
    def is_word(c: str):
        return c.isalnum() or c == '_'
    names = sorted(group_names, key=len, reverse=True)
    number = re.compile(r'(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?')
    word = re.compile(r'\w+')
    tokens = list()
    i, n = 0, len(expression)
    while i < n:
        c = expression[i]
        if c.isspace():
            i += 1
            continue
        if c in '`\'"':
            j = expression.find(c, i + 1)
            if j < 0:
                fail(f'the quote {c} at position {i + 1} is never closed.')
            tokens.append(('name', expression[i + 1:j]))
            i = j + 1
            continue
        num = number.match(expression, i)
        name = None
        if not (i and is_word(expression[i - 1]) and is_word(c)):
            for candidate in names:
                j = i + len(candidate)
                if expression.startswith(candidate, i) and not (j < n and is_word(candidate[-1])
                                                                 and is_word(expression[j])):
                    name = candidate
                    break
        # A group whose name reads as a number has to be quoted.
        if name is not None and not (num and num.end() >= i + len(name)):
            tokens.append(('name', name))
            i += len(name)
        elif num:
            tokens.append(('number', float(num.group())))
            i = num.end()
        elif c in '+-*/()':
            tokens.append((c, c))
            i += 1
        elif word.match(expression, i):
            j = word.match(expression, i).end()
            tokens.append(('name', expression[i:j]))
            i = j
        else:
            fail(f'unexpected character "{c}" at position {i + 1}.')
    unknown = list(dict.fromkeys(t for kind, t in tokens if kind == 'name' and t not in group_names))
    if unknown:
        fail(f'unknown sample group(s) {_first_names(unknown)}.')

    # A value is a pair (coefficients of the groups, constant term).
    def scale(value, k):
        return {g: v * k for g, v in value[0].items()}, value[1] * k
    def add(a, b, sign):
        coeffs = dict(a[0])
        for g, v in b[0].items():
            coeffs[g] = coeffs.get(g, 0.0) + sign * v
        return coeffs, a[1] + sign * b[1]
    pos = 0
    def peek():
        return tokens[pos][0] if pos < len(tokens) else None
    def factor():
        nonlocal pos
        if pos >= len(tokens):
            fail('it ends where a group name or a number is expected.')
        kind, token = tokens[pos]
        pos += 1
        if kind in ('+', '-'):
            return scale(factor(), -1.0 if kind == '-' else 1.0)
        if kind == 'number':
            return dict(), token
        if kind == 'name':
            return {token: 1.0}, 0.0
        if kind == '(':
            value = expression_()
            if peek() != ')':
                fail('a closing parenthesis is missing.')
            pos += 1
            return value
        fail(f'"{token}" stands where a group name or a number is expected.')
    def term():
        nonlocal pos
        value = factor()
        while peek() in ('*', '/'):
            op = tokens[pos][0]
            pos += 1
            other = factor()
            if op == '*':
                if value[0] and other[0]:
                    fail('it is not linear, as two sample groups are multiplied.')
                value = scale(other, value[1]) if other[0] else scale(value, other[1])
            else:
                if other[0]:
                    fail('it is not linear, as it divides by a sample group.')
                if other[1] == 0:
                    fail('it divides by zero.')
                value = scale(value, 1.0 / other[1])
        return value
    def expression_():
        nonlocal pos
        value = term()
        while peek() in ('+', '-'):
            op = tokens[pos][0]
            pos += 1
            value = add(value, term(), -1.0 if op == '-' else 1.0)
        return value
    coeffs, const = expression_()
    if pos < len(tokens):
        fail(f'"{tokens[pos][1]}" stands where an operator (+, -, *, /) is expected.')
    if abs(const) > 1e-12:
        fail(f'it contains the constant term {const:g}, which has no meaning in a comparison of groups.')
    coeffs = {g: coeffs[g] for g in group_names if abs(coeffs.get(g, 0.0)) > 1e-12}
    if not coeffs:
        fail('all its coefficients are zero, so it compares nothing.')
    return coeffs


def export_contrast(project_name: str,  output_folder: str,
                    contrasts: str, filename_postfix: str = None):
    if filename_postfix is None:
        postfix = str()
    else:
        postfix = '_' + safe_filenames([filename_postfix], what='postfix')[0]
    data = read_init(project_name)
    fmt = data.fmt
    data, fit = load_fit(project_name, data)
    motif_names = data.motif_names
    prom_names = data.promoter_names
    act: ActivitiesPrediction = load_predict(project_name, data, fit)
    del data
    if held_out_inds(fit):
        prom_names = np.delete(prom_names, held_out_inds(fit))
    group_names = list(fit.group_names)
    motif_names_filtered = np.delete(motif_names, filtered_motifs(act))
    contrasts = parse_contrast(contrasts, group_names)
    variances = act.var()
    coeffs = np.zeros_like(variances)
    inds = np.zeros(len(variances), dtype=bool)
    for c, v in contrasts.items():
        i = group_names.index(c)
        coeffs[i] = v
        inds[i] = True
    U = act.U.T
    U = U[inds]
    variances = variances[inds]
    coeffs = coeffs[inds]
    variances = variances * coeffs ** 2
    z_stat = (U * coeffs).sum(axis=0) / variances.sum(axis=0) ** 0.5
    # z_stat =  (U_b - U_a) / (variances_a + variances_b) ** 0.5
    pval = 2 * norm.sf(np.abs(z_stat))
    fdr = multitest.multipletests(pval, alpha=0.05, method='fdr_by')[1]
    data = np.array([z_stat, pval, fdr]).T
    filename = os.path.join(output_folder, f'contrast{postfix}.tsv')
    os.makedirs(output_folder, exist_ok=True)
    DF(data, columns=['z_stat', 'pval', 'fdr'],
       index=motif_names_filtered).to_csv(filename, sep='\t')
    write_readme(output_folder)

         
def export_loadings_product(project_name: str, output_folder: str,
                            use_hdf: bool = True, intercepts: bool = True,
                            tsv_truncation=4):
    

    data = read_init(project_name)
    fmt = data.fmt
    data, fit = load_fit(project_name, data)
    motif_names = data.motif_names
    act: ActivitiesPrediction = load_predict(project_name, data, fit)
    # The activities refer to the training promoters and to the loadings as the model
    # saw them.
    data, _ = split_data(data, held_out_inds(fit))
    prom_names = data.promoter_names
    # del data
    group_names = safe_filenames(fit.group_names)
    
    output_folder = os.path.join(output_folder, 'loadings-product')
    os.makedirs(output_folder, exist_ok=True)
    
    U = act.U
    B = transform_data(data, helmert=False, loading_context=fit.loading_context,
                       drist=fit.drist, loading_multipliers=fit.loading_multipliers).B
    # Intercept (average) motif activity mu_m = M[:, 0]; covariate components are
    # group-specific and not folded into this static loadings-product tensor.
    mu = np.asarray(fit.motif_mean.mean)
    mu = mu[:, 0] if mu.ndim > 1 else mu

    filtered = filtered_motifs(act)
    if len(filtered):
        motif_names = np.delete(motif_names, filtered)
        B = np.delete(B, filtered, axis=1)
        mu = np.delete(mu, filtered)
    BM = B * mu
    for name, U in zip(group_names, U.T):
        effect = B * U
        if intercepts:
            effect += BM
        if use_hdf:
            effect = effect.astype(np.half)
            filename = os.path.join(output_folder, f'{name}.hdf')
            DF(data=effect, index=prom_names, columns=motif_names).to_hdf(filename, key='lrt', mode='w', complevel=4)
        else:
            filename = os.path.join(output_folder, f'{name}.tsv')
            DF(data=effect, index=prom_names, columns=motif_names).to_csv(filename, sep='\t',
                                                                          float_format=f'%.{tsv_truncation}f')  
    write_readme(os.path.dirname(output_folder), lp_intercepts=intercepts)
        
        
    