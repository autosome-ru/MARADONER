#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pandas import DataFrame as DF
# add dot
from .utils import read_init, openers
from scipy.stats import norm, chi2
from statsmodels.stats import multitest
import numpy as np
from enum import Enum
import dill
import os

class Standardization(str, Enum):
    full = 'full'
    std = 'std'


def export_results(project_name: str, output_folder: str,
                   std_mode: Standardization, alpha=0.05):
    data = read_init(project_name)
    fmt = data.fmt
    motif_names = data.motif_names
    # prom_names = data.promoter_names
    del data
    with openers[fmt](f'{project_name}.fit.{fmt}', 'rb') as f:
        fit = dill.load(f)
    group_names = fit.group_names
    if fit.activities.filtered_motifs is not None:
        motif_names = np.delete(motif_names, fit.activities.filtered_motifs)
    
    os.makedirs(output_folder, exist_ok=True)
    
    Sigma = fit.Sigma
    sigmas = fit.sigma
    # promoter_means = fit.promoter_means
    nu = fit.nu
    act = fit.activities
    del fit
  
    DF(np.array([sigmas, nu]).T, index=group_names,
                columns=['sigma', 'nu']).to_csv(os.path.join(output_folder, 'params_groups.tsv'), sep='\t')
    s = 'motif\ttau\n' + '\n'.join(f'{a}\t{b}' for a, b in zip(motif_names, Sigma))
    with open(os.path.join(output_folder, 'Sigma.tsv'), 'w') as f:
        f.write(s)
    U_raw, U_decor, stds = act.U, act.U_decor, act.stds
    FOV_train, FOV_test = act.FOV_train, act.FOV_test
    FOV_train = list(FOV_train)
    FOV_test = list(FOV_test)
    DF(np.array([FOV_train + [np.mean(FOV_train)], FOV_test + [np.mean(FOV_test)]]).T, index=group_names + ['all'],
       columns=['train', 'test']).to_csv(os.path.join(output_folder, 'FOV_groups.tsv'), sep='\t')
    # if len(promoter_means.shape) < 2:
    #     DF(promoter_means, index=prom_names,
    #        columns=['mean']).to_csv(os.path.join(output_folder, 'promoter_mean.tsv'), sep='\t')
    # else:
    #     DF(promoter_means, index=prom_names,
    #        columns=group_names).to_csv(os.path.join(output_folder, 'promoter_mean.tsv'), sep='\t')
    if std_mode == Standardization.full:
        U = U_decor
    else:
        U = U_raw / stds
    
    DF(U_raw, index=motif_names, columns=group_names).to_csv(os.path.join(output_folder, 'U_raw.tsv'), sep='\t')
    DF(U, index=motif_names, columns=group_names).to_csv(os.path.join(output_folder, 'U.tsv'), sep='\t')
    DF(stds, index=motif_names, columns=group_names).to_csv(os.path.join(output_folder, 'stds.tsv'), sep='\t')
    z_test = 2 * norm.sf(np.abs(U))
    z_test_fdr = [multitest.multipletests(z_test[:, i], alpha=alpha, method='fdr_bh')[1] for i in range(z_test.shape[1])]
    z_test_fdr = np.array(z_test_fdr).T
    z_test = DF(z_test, index=motif_names, columns=group_names)
    z_test.to_csv(os.path.join(output_folder, 'z_test.tsv'), sep='\t')
    z_test = DF(z_test_fdr, index=motif_names, columns=group_names)
    z_test.to_csv(os.path.join(output_folder, 'z_test_fdr.tsv'), sep='\t')
    anova = chi2.sf((U ** 2).sum(axis=1), df=U.shape[1])
    fdrs = multitest.multipletests(anova, alpha=0.05, method='fdr_bh')[1]
    anova = DF([anova, fdrs], columns=motif_names, index=['p-value', 'FDR']).T
    anova.to_csv(os.path.join(output_folder, 'anova.tsv'), sep='\t')
    off_test = -np.expm1(U.shape[1]*chi2.logsf((U ** 2).min(axis=1), df=1))
    fdrs = multitest.multipletests(off_test, alpha=0.05, method='fdr_bh')[1]
    off_test = DF([off_test, fdrs], columns=motif_names, index=['p-value', 'FDR']).T
    off_test.to_csv(os.path.join(output_folder, 'off_test.tsv'), sep='\t')
    
    return {'z-test': z_test, 'anova': anova, 'off_test': off_test}
