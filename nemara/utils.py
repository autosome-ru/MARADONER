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
