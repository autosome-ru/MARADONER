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
    cols = ['stat']
    fov_null,  = fovs
    total = [fov_null.total,]
    DF(total, index=cols, columns=['FOV']).T.to_csv(
        os.path.join(folder, 'total.tsv'), sep='\t')
    promoters = [fov_null.promoter[:, None],
                 ]
    promoters = np.concatenate(promoters, axis=-1)
    DF(promoters, index=promoter_names, columns=cols).to_csv(
        os.path.join(folder, 'promoters.tsv'), sep='\t')
    samples = [fov_null.sample[:, None],
               ]
    samples = np.concatenate(samples, axis=-1)
    DF(samples, index=sample_names, columns=cols).to_csv(
        os.path.join(folder, 'samples.tsv'), sep='\t')


def export_results(project_name: str, output_folder: str,
                   export_B: bool = True):
    data = read_init(project_name)
    fmt = data.fmt
    motif_names = data.motif_names
    prom_names = data.promoter_names
    sample_names = data.sample_names
    if export_B:
        B = data.B
        B = DF(B, index=prom_names, columns=motif_names)
        os.makedirs(output_folder, exist_ok=True)
        B.to_csv(os.path.join(output_folder, 'B.tsv'), sep='\t')
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
    U0 = act.U
    act = np.hstack((U_total, U))
    DF(act, index=motif_names, 
       columns=['overall'] + list(sample_names)).to_csv(os.path.join(output_folder, 'activities.tsv'), 
                                    sep='\t')
    DF(U0, index=motif_names, 
       columns= list(sample_names)).to_csv(os.path.join(output_folder, 'activities_raw.tsv'), 
                                           sep='\t')
    
    z = U ** 2
    U_total = z.mean(axis=1, keepdims=True) #/ (1 / U_var ** 0.5).sum(axis=1, keepdims=True)
    z = np.hstack((U_total, z))
    z = z ** 0.5
    DF(z, index=motif_names, 
       columns=['overall'] + list(sample_names)).to_csv(os.path.join(output_folder, 'z_scores.tsv'), 
                                    sep='\t')
    if os.path.isfile(f'{project_name}.old.fov.{fmt}'):
        with open(f'{project_name}.old.fov.{fmt}', 'rb') as f:
            fov = dill.load(f)
            if type(fov) in (tuple, list):
                print('n', len(fov))
                fov = fov[0][-1]
            print(type(fov))
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
        if fit.promoter_inds_to_drop:
            promoter_names_train = np.delete(data.promoter_names, fit.promoter_inds_to_drop)
        else:
            promoter_names_train = data.promoter_names
        export_fov(train, os.path.join(folder, 'train'), promoter_names=promoter_names_train,
                   sample_names=sample_names)
        if test is not None:
            promoter_names_test = np.array(data.promoter_names)[fit.promoter_inds_to_drop]
            export_fov(test, os.path.join(folder, 'test'), promoter_names=promoter_names_test,
                       sample_names=sample_names)
    

