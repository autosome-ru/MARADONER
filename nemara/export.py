#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pandas import DataFrame as DF
from .utils import get_init_file, openers
from scipy.stats import norm, chi2
from statsmodels.stats import multitest
import numpy as np
import json
import dill
import os


def export_results(project_name: str, output_folder: str, std_mode='full', alpha=0.05):
    filename, fmt = get_init_file(project_name)
    with openers[fmt](filename, 'rb') as f:
        init = dill.load(f)
        promoters = init['promoter_names']
        motifs = init['motif_names']
        group_names = list(init['groups'].keys())
        del init
    folder, filename = os.path.split(filename)
    project_name = '.'.join(filename.split('.')[:-2])
    fit_filename = os.path.join(folder, f'{project_name}.fit.{fmt}')
    if not os.path.isfile(fit_filename):
        raise FileNotFoundError(f'The project {project_name} was not fit.')
    with openers[fmt](fit_filename, 'rb') as f:
        fit = dill.load(f)
    
    os.makedirs(output_folder, exist_ok=True)
    try:
        DF(np.array([fit['sigma_e'], fit['sigma_u']]).T, index=group_names, columns=['sigma_e', 'sigma_u']).to_csv(os.path.join(output_folder, 'sigma.tsv'),
                                                                                                       sep='\t')
    except ValueError:
        DF(fit['sigma_e'], index=group_names, columns=['sigma_e']).to_csv(os.path.join(output_folder, 'sigma.tsv'), sep='\t')
        DF(fit['sigma_u'], index=motifs, columns=group_names).to_csv(os.path.join(output_folder, 'motif_sigma.tsv'), sep='\t')
    with open(os.path.join(output_folder, 'FOV.txt'), 'w') as f:
        f.write(str(fit['FOV']))
    DF(fit['FOV_groups'].reshape(1,-1), columns=group_names, index=['FOV']).to_csv(os.path.join(output_folder, 'FOV_groups.tsv'), sep='\t')
    
    cv_filename = os.path.join(folder, f'{project_name}.cv.json')
    if os.path.isfile(cv_filename):
        with open(cv_filename, 'r') as f:
            d = json.load(f)
        means = d['mean']
        stds = d['std']
        stats = means['test_fov']
        best = max(stats, key=lambda x:stats[x])
        subfolder = os.path.join(output_folder, 'best_cv')
        os.makedirs(subfolder, exist_ok=True)
        with open(os.path.join(subfolder, 'params.txt'), 'w') as f:
            f.write(best)
        fov_train = means['train_fov'][best]
        fov_test = means['test_fov'][best]
        std_train = stds['train_fov'][best]
        std_test = stds['test_fov'][best]
        fov_train_groups = means['train_fov_groups'][best]
        fov_test_groups = means['test_fov_groups'][best]
        DF([[fov_train, std_train], [fov_test, std_test]], index=['train', 'test'], columns=['FOV', 'FOV std']).to_csv(os.path.join(subfolder, 'FOV.tsv'),
                                                                                                                       sep='\t')
        DF([fov_train_groups, fov_test_groups], columns=group_names, index=['train', 'test']).to_csv(os.path.join(subfolder, 'FOV_groups.tsv'),
                                                                                                     sep='\t')
        
    
    DF(fit['U_raw'], index=motifs, columns=group_names).to_csv(os.path.join(output_folder, 'activities_raw.tsv'), sep='\t')
    if std_mode == 'full':
        U = fit['U_std']
    else:
        U = fit['U_raw'] / fit['stds']
    DF(fit['stds'], index=motifs, columns=group_names).to_csv(os.path.join(output_folder, 'stds.tsv'), sep='\t')
    DF(U, index=motifs, columns=group_names).to_csv(os.path.join(output_folder, 'activities_std.tsv'), sep='\t')
    
    z_test = 2 * norm.sf(np.abs(U))
    z_test_fdr = [multitest.multipletests(z_test[:, i], alpha=alpha, method='fdr_bh')[1] for i in range(z_test.shape[1])]
    z_test_fdr = np.array(z_test_fdr).T
    z_test = DF(z_test, index=motifs, columns=group_names)
    z_test.to_csv(os.path.join(output_folder, 'z_test.tsv'), sep='\t')
    z_test = DF(z_test_fdr, index=motifs, columns=group_names)
    z_test.to_csv(os.path.join(output_folder, 'z_test_fdr.tsv'), sep='\t')
    anova = chi2.sf((U ** 2).sum(axis=1), df=U.shape[1])
    fdrs = multitest.multipletests(anova, alpha=0.05, method='fdr_bh')[1]
    anova = DF([anova, fdrs], columns=motifs, index=['p-value', 'FDR']).T
    anova.to_csv(os.path.join(output_folder, 'anova.tsv'), sep='\t')
    off_test = -np.expm1(U.shape[1]*norm.logsf(U.min(axis=1)))
    fdrs = multitest.multipletests(off_test, alpha=0.05, method='fdr_bh')[1]
    off_test = DF([off_test, fdrs], columns=motifs, index=['p-value', 'FDR']).T
    off_test.to_csv(os.path.join(output_folder, 'off_test.tsv'), sep='\t')
    
    return {'z-test': z_test, 'anova': anova, 'off_test': off_test}
