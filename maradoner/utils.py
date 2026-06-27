# -*- coding: utf-8 -*-
from dataclasses import dataclass
import scipy.stats as st
import numpy as np
import dill
import lzma
import gzip
import bz2
import os
import re


openers = {'lzma': lzma.open,
           'gzip': gzip.open,
           'bz2': bz2.open,
           'raw': open}

def logger_print(msg: str, verbose: bool = True):
    if verbose:
        print(msg)

def get_init_file(path: str):
    folder, name = os.path.split(path)
    for file in os.listdir(folder if folder else None):
        if file.startswith(f'{name}.init.') and file.endswith(tuple(openers.keys())) and os.path.isfile(os.path.join(folder, file)):
            return os.path.join(folder, file), file.split('.')[-1]

def get_init_files(path: str):
    files = list()
    folder, name = os.path.split(path)
    ptrn = re.compile(name + r'.init.\d+.\w+')
    for file in os.listdir(folder if folder else None):
        m = ptrn.fullmatch(file)
        if m is not None and (m.start() == 0) and (m.end() == len(file)):
            files.append(file)
    return [os.path.join(folder, x) for x in sorted(files, key=lambda x: int(x.split('.')[-2]))]


def variance_explained(Y, B, U, groups: list, group_average=True, weights=None, contract=True, prom_diffs=False) -> float:
    if weights is not None:
        weights = weights.reshape(-1, 1) ** 0.5
        Y = weights * Y
        B = weights * B
    if group_average:
        Yn = np.empty((len(Y), len(groups)), dtype=float)
        for i, inds in enumerate(groups):
            Yn[:, i] = Y[:, inds].mean(axis=1)
        if contract:
            return 1 - np.sum((Yn - B @ U) ** 2) / np.sum(Yn ** 2)
        if prom_diffs:
            return 1 - np.sum((Yn - B @ U) ** 2, axis=1) / np.sum(Yn ** 2, axis=1), \
                   1 - (Yn - B @ U) ** 2 / Yn ** 2
        return 1 - np.sum((Yn - B @ U) ** 2, axis=0) / np.sum(Yn ** 2, axis=0)
    n = sum(map(len, groups))
    Un = np.empty((len(U), n), dtype=float)
    for i, inds in enumerate(groups):
        Un[:, inds] = U[:, i:i+1]
    if contract:
        return 1 - np.sum((Y - B @ Un) ** 2) / np.sum(Y ** 2)
    if prom_diffs:
        return 1 - (Yn - B @ U) ** 2 / Yn ** 2
    return 1 - np.sum((Yn - B @ U) ** 2, axis=0) / np.sum(Yn ** 2, axis=0)

def subs_zeros(X: np.ndarray) -> np.ndarray:
    X = np.array(X)
    inds = X == 0
    m = np.abs(X[~inds]).min()
    X[inds] = m
    return X

@dataclass
class ProjectData:
    Y: np.ndarray
    B: np.ndarray
    K: np.ndarray
    weights: np.ndarray
    group_inds: list
    group_names: list
    sample_names: list
    motif_names: list
    promoter_names: list
    motif_postfixes: list
    fmt: str
    L: np.ndarray = None
    # Covariate design matrix X of shape (c, s): each column is a sample, each
    # row a covariate. The first row is always the intercept (all ones). When no
    # covariates are supplied, X is just the intercept row (c == 1), which makes
    # the model identical to the classic MARADONER (M X == mu_m 1_s^T).
    X: np.ndarray = None
    covariate_names: list = None


def read_init(project_name: str) -> ProjectData:
    if type(project_name) is str:
        filename, fmt = get_init_file(project_name)
        with openers[fmt](filename, 'rb') as f:
            init = dill.load(f)
    else:
        init = project_name
    group_names = sorted(init['groups'])
    group_inds = list()
    for name in group_names:
        group_inds.append(np.array(init['groups'][name]))
    import pandas as pd
    L = None
    # L = pd.read_csv('/home/georgy/Загрузки/cluster32_nmf.tsv', sep='\t', index_col=0)
    # L = L.loc[init['promoter_names']]
    # L = L.values
    sample_names = init['sample_names']
    X = init.get('covariates', None)
    covariate_names = init.get('covariate_names', None)
    if X is None:
        # No covariates supplied: design matrix is just the intercept row of ones,
        # i.e. the classic mu_m 1_s^T term.
        X = np.ones((1, len(sample_names)), dtype=float)
        covariate_names = ['intercept']
    r = ProjectData(
        Y=init['expression'],
        B=init['loadings'],
        K=init.get('motif_expression', None),
        weights=init.get('weights', None),
        motif_names=init['motif_names'],
        promoter_names=init['promoter_names'],
        motif_postfixes=init['motif_postfixes'],
        group_names=group_names,
        sample_names=sample_names,
        group_inds=group_inds,
        L=L,
        X=X,
        covariate_names=covariate_names,
        fmt=fmt
        )
    return r