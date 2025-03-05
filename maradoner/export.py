#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pandas import DataFrame as DF
# add dot
from .utils import read_init, openers
from .fit import FOVResult
from scipy.stats import norm, chi2, multivariate_normal, Covariance
from scipy.linalg import eigh, lapack, cholesky, solve
from statsmodels.stats import multitest
import numpy as np
from enum import Enum
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
from scipy.integrate import quad
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

    
class Information():
    eps = 1e-10
    
    def __init__(self, fim: np.ndarray, slc=None, use_preconditioner=False):
        self.square_root_inv = self._square_root_inv(fim, slc, corr=True)
        precond = 1 / fim.diagonal() ** 0.5
        if not use_preconditioner:
            precond[:] = 1
        fim = precond.reshape(-1, 1) * fim * precond
        self.fim = fim
        self.precond = precond
        self.slice = slice(None, None) if slc is None else slc
    
    def _inv(self, x: np.ndarray):
        try:
            x = chol_inv(x)
        except:
            print('alarm')
            x = np.linalg.eigh(x)
            x = x[1] * (1/np.clip(x[0], self.eps, float('inf'))) @ x[1].T
        return x
    
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
            x = np.linalg.eigh(x)
            x = x[1] * x[0] ** (1 / 2) @ x[1].T
        return x
    
    def standardize(self, x: np.ndarray, 
                    mode: Standardization=Standardization.std,
                    return_std=True):
        x = x / self.precond[self.slice]
        cov = self._inv(self.fim)
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
        cov = self._inv(self.fim)
        cov = cov[self.slice, self.slice]
        return self.precond[self.slice].reshape(-1, 1) * cov * self.precond[self.slice]

    def correlation(self):
        cov = self.covariance()
        d = cov.diagonal() ** (-0.5)
        return d.reshape(-1, 1) * cov * d
    
    def cholesky_transform(self, x: np.ndarray):
        if self.square_root_inv is None:
            self.square_root_inv = self.square_root_inv(self.fim, self.slice, corr=True)
        return self.square_root_inv.T @ x
        


def _corrected_numerical(x, mvn, n: int):
    x = np.abs(x)
    return 1.0 - mvn.cdf(np.repeat(x, n), lower_limit=-x)

def _corrected_sampled(x, information: Information, num_samples: int, m: int,
                       num_repeats=1):
    x = np.abs(x)
    c = 0
    n = 0
    for _ in range(num_repeats):
        t = np.abs(information.cholesky_transform(norm.rvs(size=(m, num_samples))))
        c += np.any(t > x, axis=0).sum()
        n += num_samples
    return c / n

def corrected_z_test(stat: np.ndarray, information: Information,
                     numerical: bool, num_samples: int,
                     n_jobs: int) -> np.ndarray:
    if numerical:
        raise NotImplementedError
    
    num_samples = int(num_samples)
    f = partial(_corrected_sampled, information=information, num_samples=num_samples,
                m=len(stat), num_repeats=1)
    
    if n_jobs > 1:
        with mp.Pool(n_jobs) as p:
            corrected = np.array(list(p.map(f , stat)))
    else:
        corrected = np.array(list(map(f, stat)))
    return corrected


def weird_test(mu, shift=0, eps=1e-12, std=None):
    if std is None:
        std = np.ones_like(mu)
    
    def log_integrand(u, mu, mu_k, std, std_k):
        return norm.logpdf(u, loc=mu_k, scale=std_k) + norm.logcdf((u - mu) / std_k).sum()
    
    def integrand(u, mu, mu_k, std, std_k):
        return np.exp(log_integrand(u, mu, mu_k, std, std_k) + shift)
    
    argmax = np.zeros_like(mu, dtype=float)
    for k in tqdm(list(range(len(mu)))):
        argmax[k] = quad(lambda x: integrand(x, np.delete(mu, k), mu[k], np.delete(std, k), std[k]), 
                         -np.inf, np.inf, epsabs=eps, epsrel=eps)[0]
    result = np.zeros_like(argmax)
    inds = np.arange(len(result), dtype=int)
    return argmax
    for k in range(len(mu)):
        result[k] = argmax[np.delete(inds, k)].sum()
    return result * np.exp(-shift)

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
    
    

def export_results(project_name: str, output_folder: str,
                   std_mode: Standardization, 
                   anova_mode: ANOVAType=ANOVAType.positive,
                   compute_corrected_pvalues=False,
                   corrected_numerical=False,
                   corrected_num_samples=1e5,
                   alpha=0.05,
                   n_jobs=6):
    
    def calc_z_test(x):
        if anova_mode == ANOVAType.negative:
            import mpmath
            mpmath.mp.dps = 500
            pval = np.array([float(2 * mpmath.ncdf(t) - 1) for t in x])
        else:
            pval = 2 * norm.sf(np.abs(x))
        return pval

    data = read_init(project_name)
    fmt = data.fmt
    motif_names = data.motif_names
    prom_names = data.promoter_names
    # del data
    with openers[fmt](f'{project_name}.fit.{fmt}', 'rb') as f:
        fit = dill.load(f)
    if fit.promoter_inds_to_drop:
        prom_names = np.delete(prom_names, fit.promoter_inds_to_drop)
    group_names = fit.group_names
    with openers[fmt](f'{project_name}.predict.{fmt}', 'rb') as f:
        act = dill.load(f)
    if act.filtered_motifs is not None:
        motif_names_filtered = np.delete(motif_names, act.filtered_motifs)
    else:
        motif_names_filtered = motif_names
    
    os.makedirs(output_folder, exist_ok=True)
    
    error_variance = fit.error_variance.variance
    error_variance_fim = Information(fit.error_variance.fim)
    error_variance_stat, error_variance_std = error_variance_fim.standardize(error_variance, 
                                                                             mode=Standardization.std)
    
    motif_variance = fit.motif_variance.motif
    motif_variance_fim = Information(fit.motif_variance.fim, slice(None, len(motif_names)))
    motif_variance_stat, motif_variance_std = motif_variance_fim.standardize(motif_variance, 
                                                                             mode=Standardization.std)
    
    motif_group_variance = fit.motif_variance.group
    excluded_motif_group = fit.motif_variance.fixed_group
    
    motif_group_variance_fim = Information(fit.motif_variance.fim, slice(len(motif_names), None))
    motif_group_variance_std = motif_group_variance_fim.covariance().diagonal() ** 0.5
    
    
    motif_mean = fit.motif_mean.mean.flatten()
    motif_mean_fim = Information(fit.motif_mean.fim)
    motif_mean_stat, motif_mean_std = motif_mean_fim.standardize(motif_mean,
                                                 mode=Standardization.std)
    
    promoter_mean = fit.promoter_mean.mean.flatten()
    # del fit
    
    
    folder = os.path.join(output_folder, 'params')
    os.makedirs(folder, exist_ok=True)
    if excluded_motif_group is not None:
        motif_group_variance_std = np.insert(motif_group_variance_std, excluded_motif_group, np.nan)
    DF(np.array([error_variance, error_variance_std, motif_group_variance, motif_group_variance_std]).T,
                index=group_names,
                columns=['sigma', 'sigma_std', 'nu', 'nu_std']).to_csv(os.path.join(folder, 'group_variances.tsv'),
                                                             sep='\t')
    s = 'motif\ttau\tstd\n' + '\n'.join(f'{a}\t{b}\t{c}' for a, b, c in zip(motif_names,
                                                                            motif_variance,
                                                                            motif_variance_std))
    with open(os.path.join(folder, 'motif_variances.tsv'), 'w') as f:
        f.write(s)
    DF(promoter_mean, index=prom_names, columns=['mean']).to_csv(os.path.join(folder, 'promoter_means.tsv'),
                                                                 sep='\t')
    DF(np.array([motif_mean, motif_mean_std]).T,
       index=motif_names, columns=['mean', 'std']).to_csv(os.path.join(folder, 'motif_means.tsv'),
                                                          sep='\t')
    folder = os.path.join(folder, 'correlations')
    os.makedirs(folder, exist_ok=True)
    DF(motif_mean_fim.correlation(), index=motif_names, columns=motif_names).to_csv(os.path.join(folder, 'motif_means.tsv'),
                                                                      sep='\t')
    DF(motif_variance_fim.correlation(), index=motif_names, columns=motif_names).to_csv(os.path.join(folder, 'motif_variances.tsv'),
                                                                      sep='\t')
    _group_names = group_names
    if excluded_motif_group is not None:
        _group_names = np.delete(_group_names, excluded_motif_group)
    DF(motif_group_variance_fim.correlation(), index=_group_names, columns=_group_names).to_csv(os.path.join(folder, 'motif_group_variances.tsv'),
                                                                      sep='\t')
    # DF(motif_cor_cross, index=motif_names, columns=_group_names).to_csv(os.path.join(folder, 'motif_cross.tsv'),
    #                                                                   sep='\t')
    DF(error_variance_fim.correlation(), index=group_names, columns=group_names).to_csv(os.path.join(folder, 'error_variances.tsv'),
                                                                      sep='\t')
    
    
    folder = output_folder
    U_raw, U_decor, stds = act.U, act.U_decor, act.stds

    if std_mode == Standardization.full:
        U = U_decor
    else:
        U = U_raw / stds
    folder = os.path.join(output_folder, 'activities')
    os.makedirs(folder, exist_ok=True)
    DF(U_raw, index=motif_names_filtered, columns=group_names).to_csv(os.path.join(folder, 'activity_raw.tsv'), sep='\t')
    DF(U, index=motif_names_filtered, columns=group_names).to_csv(os.path.join(folder, 'activity.tsv'), sep='\t')
    DF(stds, index=motif_names_filtered, columns=group_names).to_csv(os.path.join(folder, 'activity_stds.tsv'), sep='\t')
    
    folder = os.path.join(output_folder, 'tests', 'prediction_based')
    os.makedirs(folder, exist_ok=True)
    z_test = 2 * norm.sf(np.abs(U))#calc_z_test(U)
    z_test_fdr = [multitest.multipletests(z_test[:, i], alpha=alpha, method='fdr_bh')[1] for i in range(z_test.shape[1])]
    z_test_fdr = np.array(z_test_fdr).T
    z_test = DF(z_test, index=motif_names_filtered, columns=group_names)
    z_test.to_csv(os.path.join(folder, 'z_test.tsv'), sep='\t')
    z_test = DF(z_test_fdr, index=motif_names_filtered, columns=group_names)
    z_test.to_csv(os.path.join(folder, 'z_test_fdr.tsv'), sep='\t')
    stat = (U ** 2).sum(axis=1)
    anova = chi2.sf(stat, df=U.shape[1])
    fdrs = multitest.multipletests(anova, alpha=0.05, method='fdr_bh')[1]
    anova = DF([stat, anova, fdrs], columns=motif_names_filtered, index=['stat', 'p-value', 'FDR']).T
    anova.to_csv(os.path.join(folder, 'anova.tsv'), sep='\t')
    stat = (U ** 2).min(axis=1)
    off_test = -np.expm1(U.shape[1]*chi2.logsf(stat, df=1))
    fdrs = multitest.multipletests(off_test, alpha=0.05, method='fdr_bh')[1]
    off_test = DF([stat, off_test, fdrs], columns=motif_names_filtered, index=['stat', 'p-value', 'FDR']).T
    off_test.to_csv(os.path.join(folder, 'off_test.tsv'), sep='\t')
    
    folder = os.path.join(output_folder, 'tests', 'asymptotics_based')
    os.makedirs(folder, exist_ok=True)
    
    anova_ass = motif_variance_stat
    pval = calc_z_test(anova_ass)
    # anova_ass = motif_variance_stat * motif_variance_std 
    # pval = weird_test(anova_ass, std=motif_variance_std)
    fdrs = multitest.multipletests(pval, alpha=0.05, method='fdr_bh')[1]
    if compute_corrected_pvalues:
        corrected_pval = corrected_z_test(anova_ass, motif_variance_fim, numerical=corrected_numerical,
                                          num_samples=corrected_num_samples,
                                          n_jobs=n_jobs)
        anova_ass = DF(np.array([anova_ass, pval, fdrs, corrected_pval]).T, index=motif_names, columns=['stat', 'p-value', 'FDR', 'corrected-p-value'])
    else:
        anova_ass = DF(np.array([anova_ass, pval, fdrs]).T, index=motif_names, columns=['stat', 'p-value', 'FDR'])
    anova_ass.to_csv(os.path.join(folder, 'anova.tsv'), sep='\t')
    
    sign = motif_mean.flatten() / motif_mean_std
    neg = norm.cdf(sign)
    pos = norm.sf(sign)
    zero = chi2.cdf(sign ** 2, df=1)

    neg_fdr = multitest.multipletests(neg, alpha=0.05, method='fdr_bh')[1]
    pos_fdr = multitest.multipletests(pos, alpha=0.05, method='fdr_bh')[1]
    zero_fdr = multitest.multipletests(zero, alpha=0.05, method='fdr_bh')[1]
    sign_ass = DF(np.array([sign, zero, zero_fdr, neg, neg_fdr, pos, pos_fdr]).T, columns=['stat', 
                                                                                           'zero', 'fdr_zero',
                                                                                           'pvalue_neg', 'fdr_neg',
                                                                                           'pvalue_pos', 'fdr_pos'],
                  index=motif_names)
    sign_ass.to_csv(os.path.join(folder, 'sign.tsv'), sep='\t')
    
    if os.path.isfile(f'{project_name}.fov.{fmt}'):
        with open(f'{project_name}.fov.{fmt}', 'rb') as f:
            fov = dill.load(f)
            train = fov.train
            test = fov.test
        folder = os.path.join(output_folder, 'fov')
        if fov.grouped:
            sample_names = data.group_names
        else:
            sample_names = [None] * len(train[0].sample)
            for i, inds in enumerate(data.group_inds):
                name = data.group_names[i]
                for k, j in enumerate(inds):
                    sample_names[j] = f'{name}_{k+1}'
        promoter_names_train = np.delete(data.promoter_names, fit.promoter_inds_to_drop)
        export_fov(train, os.path.join(folder, 'train'), promoter_names=promoter_names_train,
                   sample_names=sample_names)
        if test is not None:
            promoter_names_test = np.array(data.promoter_names)[fit.promoter_inds_to_drop]
            export_fov(test, os.path.join(folder, 'test'), promoter_names=promoter_names_test,
                       sample_names=sample_names)
        
            
    
    return {'z-test': z_test, 'anova': anova, 'off_test': off_test,
            'anova_ass': anova_ass, 'sign_ass': sign_ass}
