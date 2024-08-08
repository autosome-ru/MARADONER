# -*- coding: utf-8 -*-
import scipy.stats as st
import numpy as np
import lzma
import gzip
import bz2
import os
import re


openers = {'lzma': lzma.open,
           'gzip': gzip.open,
           'bz2': bz2.open,
           'raw': open}

def logger_print(msg: str, verbose: bool):
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


def generate_data(p: int, m: int, g: int, min_samples: int, max_samples: int, g_std_a=0.5, g_std_b=0.5,
                  re_per_sample=False, sigma=1, U_mult_noise=0.1):
    g_samples = [np.random.randint(min_samples, max_samples) for _ in range(g)]
    g_std = st.gamma.rvs(1, 1, size=g)
    B = np.random.rand(p, m)
    K = st.wishart.rvs(df=p, scale=np.identity(m))
    stds = K.diagonal() ** 0.5
    stds = 1 / stds
    K = np.clip(stds.reshape(-1, 1) * K * stds, -1, 1)
    K = np.identity(len(K))
    U_exp = np.random.rand(g, m, 1) * 1 + 0.05
    # U_exp[:] = 1
    mean_p = st.norm.rvs(size=(p, 1))
    mean_g = st.norm.rvs(size=(g,))
    Us = list()
    Ys = list()
    inds = list()
    for i, (n_samples, m_g, std, U_mult) in enumerate(zip(g_samples, mean_g, g_std, U_exp)):
        sub_inds = np.empty(n_samples, dtype=int)
        sub_inds = list()
        for j in range(n_samples):
            Us.append(st.matrix_normal(rowcov=K, colcov=sigma * np.identity(1)).rvs())
            Ys.append((st.norm.rvs(loc=0, scale=std, size=(p, 1)) + mean_p + m_g) + B @ (U_mult * Us[-1]))
            sub_inds.append(len(Ys) - 1)
        inds.append(sub_inds)
    Ys = np.concatenate(Ys, axis=1)
    Us = np.concatenate(Us, axis=1)
    return K, Ys, B, Us, g_std, (mean_p, mean_g), U_exp[..., 0], inds