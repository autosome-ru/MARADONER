# -*- coding: utf-8 -*-
from contextlib import contextmanager
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

class InputError(ValueError):
    """An error the user can fix themselves: malformed input data, an invalid option
    value, or a project that is not in the state a command needs. The CLI prints the
    message alone, without a traceback."""


def logger_print(msg: str, verbose: bool = True):
    if verbose:
        print(msg)

def warn(msg: str, verbose: bool = True):
    logger_print(f'[warning] {msg}', verbose)

@contextmanager
def spinner(description: str):
    """Transient console spinner shown while a command runs. It is always stopped on
    exit, also when the command fails, so the terminal cursor is never left hidden."""
    from rich.progress import Progress, SpinnerColumn, TextColumn
    p = Progress(SpinnerColumn(speed=0.5), TextColumn("[progress.description]{task.description}"), transient=True)
    p.add_task(description=description, total=None)
    p.start()
    try:
        yield p
    finally:
        p.stop()

def get_init_file(path: str):
    folder, name = os.path.split(path)
    if folder and not os.path.isdir(folder):
        return None
    for file in os.listdir(folder if folder else None):
        if file.startswith(f'{name}.init.') and file.endswith(tuple(openers.keys())) and os.path.isfile(os.path.join(folder, file)):
            return os.path.join(folder, file), file.split('.')[-1]

# Commands that produce each of the project's intermediate files.
_STAGE_COMMANDS = {'init': 'create', 'fit': 'fit', 'predict': 'predict', 'fov': 'gof',
                   'promoter_mean': 'estimate-promoter-mean', 'old.fit': 'mara fit',
                   'old.predict': 'mara predict', 'old.fov': 'mara gof'}

def _load_pickle(filename: str, fmt: str, stage: str, project_name: str):
    """dill.load of a project file. A file that cannot be read back (truncated by an
    interrupted run, edited, or not written by MARADONER at all) is the user's to
    recreate, so it is reported as an InputError naming the command that does it."""
    try:
        with openers[fmt](filename, 'rb') as f:
            return dill.load(f)
    except (FileNotFoundError, PermissionError, IsADirectoryError, MemoryError):
        raise
    except Exception as e:
        cmd = _STAGE_COMMANDS.get(stage, stage)
        reason = str(e).strip() or type(e).__name__
        args = ' ...' if stage == 'init' else ''
        raise InputError(f'"{filename}" is damaged or was not written by MARADONER ({reason}). '
                         f'Re-run "maradoner {cmd} {project_name}{args}" to recreate it.') from e

def stage_filename(project_name: str, stage: str, fmt: str) -> str:
    return f'{project_name}.{stage}.{fmt}'

def load_stage(project_name: str, stage: str, fmt: str, required: bool = True):
    """Load `<project>.<stage>.<fmt>` with the compression-aware opener. A missing file
    is reported together with the command that creates it (or None is returned if it is
    not `required`)."""
    filename = stage_filename(project_name, stage, fmt)
    if not os.path.isfile(filename):
        if not required:
            return None
        cmd = _STAGE_COMMANDS.get(stage, stage)
        raise InputError(f'"{filename}" not found: run "maradoner {cmd} {project_name}" first.')
    return _load_pickle(filename, fmt, stage, project_name)

def dump_stage(obj, project_name: str, stage: str, fmt: str):
    # Written to a temporary file first: an interrupted run must not leave a truncated
    # file behind under the real name.
    filename = stage_filename(project_name, stage, fmt)
    temporary = filename + '.tmp'
    try:
        with openers[fmt](temporary, 'wb') as f:
            dill.dump(obj, f)
        os.replace(temporary, filename)
    except BaseException:
        if os.path.isfile(temporary):
            os.remove(temporary)
        raise

def remove_stage(project_name: str, stage: str, fmt: str):
    filename = stage_filename(project_name, stage, fmt)
    if os.path.isfile(filename):
        os.remove(filename)

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
    # Terms of the covariate design, see create.build_covariates. None for projects made
    # by versions that did not store them.
    covariate_terms: list = None


def apply_clustering(data: ProjectData, fit) -> ProjectData:
    """Make `data` match a fit produced with `fit --clustering`: the loading matrix is
    replaced by the stored cluster profiles (one column per cluster, rows aligned with
    the full promoter set of the project) and motif names by cluster names, so that
    every downstream command works at the cluster level. A no-op for ordinary fits."""
    if getattr(fit, 'clustering', None) is None:
        return data
    clustered_B = getattr(fit, 'clustered_B', None)
    if clustered_B is None:
        raise InputError('This project was fitted with "--clustering" by a MARADONER version '
                         'that did not store the cluster profiles. Re-run "maradoner fit" to '
                         'use the fit with this command.')
    if len(clustered_B) != len(data.B):
        raise InputError(f'The fit holds cluster profiles for {len(clustered_B)} promoters, but '
                         f'the project has {len(data.B)}: the fit is out of sync with the '
                         'project. Re-run "maradoner fit".')
    data.B = clustered_B
    data.motif_names = [f'cluster_{i + 1}' for i in range(clustered_B.shape[1])]
    return data


def read_init(project_name: str) -> ProjectData:
    if isinstance(project_name, (str, os.PathLike)):
        project_name = os.fspath(project_name)
        found = get_init_file(project_name)
        if found is None:
            raise InputError(f'No MARADONER project "{project_name}" found (expected a file '
                             f'"{project_name}.init.<{"|".join(openers)}>"). Run '
                             f'"maradoner create {project_name} ..." first, or check the '
                             'project name and path.')
        filename, fmt = found
        init = _load_pickle(filename, fmt, 'init', project_name)
        required = ('expression', 'loadings', 'groups', 'sample_names', 'motif_names',
                    'promoter_names', 'motif_postfixes')
        if not isinstance(init, dict) or any(k not in init for k in required):
            raise InputError(f'"{filename}" is not a MARADONER project file. Re-run '
                             f'"maradoner create {project_name} ..." to recreate it.')
    else:
        init = project_name
        fmt = 'raw'
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
    covariate_terms = init.get('covariate_terms', None)
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
        covariate_terms=covariate_terms,
        fmt=fmt
        )
    return r