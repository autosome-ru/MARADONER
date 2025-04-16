#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pandas import DataFrame as DF
# add dot
from ..utils import read_init, openers
from ..fit import FOVResult, ActivitiesPrediction, FitResult, transform_data
from scipy.stats import norm, chi2, multivariate_normal, Covariance
from statsmodels.stats import multitest
import numpy as np
from enum import Enum
import dill
import os


class Standardization(str, Enum):
    full = 'full'
    std = 'std'


class ANOVAType(str, Enum):
    positive = 'positive'
    negative = 'negative'


def export_fov(fovs: tuple[FOVResult], folder: str,
               promoter_names: list[str], sample_names: list[str]):
    os.makedirs(folder, exist_ok=True)
    cols = ['null', 'means', 'motif_means']
    fov_null, fov_means, fov_motif_means = fovs
    total = [fov_null.total, fov_means.total, fov_motif_means.total]
    DF(total, index=cols, columns=['FOV']).T.to_csv(
        os.path.join(folder, 'total.tsv'), sep='\t')
    promoters = [fov_null.promoter[:, None],
                 fov_means.promoter[:, None], fov_motif_means.promoter[:, None]]
    promoters = np.concatenate(promoters, axis=-1)
    DF(promoters, index=promoter_names, columns=cols).to_csv(
        os.path.join(folder, 'promoters.tsv'), sep='\t')
    samples = [fov_null.sample[:, None],
               fov_means.sample[:, None], fov_motif_means.sample[:, None]]
    samples = np.concatenate(samples, axis=-1)
    DF(samples, index=sample_names, columns=cols).to_csv(
        os.path.join(folder, 'samples.tsv'), sep='\t')


def export_results(project_name: str, output_folder: str):
    data = read_init(project_name)
    fmt = data.fmt
    motif_names = data.motif_names
    prom_names = data.promoter_names
    sample_names = data.sample_names
    # del data
    with openers[fmt](f'{project_name}.old.fit.{fmt}', 'rb') as f:
        fit: FitResult = dill.load(f)
    if fit.promoter_inds_to_drop:
        prom_names = np.delete(prom_names, fit.promoter_inds_to_drop)
    group_names = fit.group_names
    with openers[fmt](f'{project_name}.old.predict.{fmt}', 'rb') as f:
        act: ActivitiesPrediction = dill.load(f)

    error_variance = fit.error_variance.variance
    motif_variance = fit.motif_variance.variance

    U = act.U
    U_var = act.variance
    
    U = U / U_var ** 0.5

    # U_grouped = list()
    # U_var_grouped = list()
    # for ind in data.group_inds:
    #     U_grouped.append(U[:, ind].mean(axis=-1))
    #     U_var_grouped.append(U_var[ind].mean(axis=-1))
    # U_grouped = np.array(U_grouped).T
    # U_var_grouped = np.array(U_var_grouped).T
    
    os.makedirs(output_folder, exist_ok=True)
    DF(np.array([error_variance, motif_variance]).T, index=sample_names, 
       columns=['sigma', 'tau']).to_csv(os.path.join(output_folder, 'params.tsv'), sep='\t')
    U_total = U.mean(axis=1, keepdims=True) # / (1 / U_var ** 0.5).sum(axis=1, keepdims=True)
    act = np.hstack((U_total, U))
    DF(act, index=motif_names, 
       columns=['overall'] + list(sample_names)).to_csv(os.path.join(output_folder, 'activities.tsv'), 
                                    sep='\t')
    
    z = U ** 2
    U_total = z.mean(axis=1, keepdims=True) #/ (1 / U_var ** 0.5).sum(axis=1, keepdims=True)
    z = np.hstack((U_total, z))
    z = z ** 0.5
    DF(z, index=motif_names, 
       columns=['overall'] + list(sample_names)).to_csv(os.path.join(output_folder, 'z_scores.tsv'), 
                                    sep='\t')
    

