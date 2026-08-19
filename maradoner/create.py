from .utils import logger_print, openers
from .dataset_filter import filter_lowexp
from .drist import DRIST
import multiprocessing
import scipy.stats as st
import datatable as dt
import pandas as pd
import numpy as np
import dill
import json
import os
import re


def drist_it(B: pd.DataFrame, Y: pd.DataFrame, test_chromosomes: list[str] = None,
             share_function: bool = False, optimizer='jacobi'):
    if test_chromosomes:
        pattern = re.compile(r'chr([0-9XYM]+|\d+)')
        
        test_chromosomes = set(test_chromosomes)
        mask = [pattern.search(p).group() in test_chromosomes for i, p in enumerate(Y.index)]
        mask = ~np.array(mask, dtype=bool)
    else:
        mask = np.ones(len(B), dtype=bool)
    Y = Y.values
    Y = Y - Y.mean(axis=1, keepdims=True)
    Bt = B.values[mask, :]
    Y = Y[mask, :]
    drist = DRIST(max_iter=1000, verbose=True, share_function=share_function,
                  optimizer=optimizer)
    B.values[mask, :] = drist.fit_transform(Bt, Y)
    if not np.all(mask):
        B.values[~mask, :] = drist.transform(B.values[~mask, :])
        
    B = B - B.min()
    return B


def transform_loadings(df, mode: str, zero_cutoff=1e-9, prom_inds=None, Y=None,
                       test_chromosomes: list[str] = None):
    # Subset promoters *before* testing for constant columns: a motif can vary across
    # the full promoter set yet be constant among the promoters that survive the
    # low-expression filter, in which case its ECDF/ESF is degenerate.
    if prom_inds is not None:
        df = df.loc[prom_inds]
    stds = df.std()
    drop_inds = (stds == 0) | np.isnan(stds)
    df = df.loc[:, ~drop_inds]
    # if not mode or mode == 'none':
    #     df[df < zero_cutoff] = 0
    #     df = (df - df.min(axis=None)) / (df.max(axis=None) - df.min(axis=None))
    if mode == 'ecdf':
        for j in range(len(df.columns)):
            v = df.iloc[:, j]
            df.iloc[:, j] = st.ecdf(v).cdf.evaluate(v)
    elif mode in ('esf',):
        for j in range(len(df.columns)):
            v = df.iloc[:, j]
            v = st.ecdf(v).sf.evaluate(v)
            uniq = np.unique(v)
            # sf hits exactly 0 at the column maximum; clip it to the next smallest
            # value so that -log stays finite.
            t = uniq[1] if len(uniq) > 1 else 1.0 / len(v)
            v[v < t] = t
            df.iloc[:, j] = -np.log(v)
        # if mode == 'drist':
        #     df = drist_it(df, Y, test_chromosomes=test_chromosomes)
    elif mode.startswith('drist'):
        df = drist_it(df, Y, test_chromosomes=test_chromosomes,
                      share_function=mode.endswith('un'))
    elif mode == 'none':
        pass
    elif mode:
        raise Exception('Unknown transformation mode ' + str(mode))
    return df

def build_covariates(covariates_filename: str, sample_names: list, n_jobs: int = 1,
                     verbose: bool = True):
    """Read a covariate/metadata table and build the design matrix X of shape
    (c, s) aligned to ``sample_names``.

    The first column of the file is treated as the sample identifier (index).
    Remaining columns are covariates. Numeric covariates are z-scored; categorical
    covariates are one-hot encoded (first level dropped) and centred. An intercept
    row of ones is always prepended, so the returned X always has at least one row.
    Samples missing from the table are imputed with the column mean (numeric) /
    most-frequent level (categorical).

    Returns (X, covariate_names).
    """
    df = dt.fread(covariates_filename, nthreads=n_jobs).to_pandas()
    df = df.set_index(df.columns[0])
    # Metadata files may contain (many) duplicate rows; keep the first occurrence
    # per sample id.
    df = df[~df.index.duplicated(keep='first')]
    sample_names = list(sample_names)
    n_missing = len([s for s in sample_names if s not in df.index])
    if n_missing:
        logger_print(f'Warning: {n_missing}/{len(sample_names)} samples are absent from the '
                     'covariate table; their covariates will be imputed.', verbose)
    # Align (and create NaN rows for missing samples).
    df = df.reindex(sample_names)
    rows = [np.ones(len(sample_names), dtype=float)]
    names = ['intercept']
    for col in df.columns:
        series = df[col]
        numeric = pd.to_numeric(series, errors='coerce')
        is_numeric = numeric.notna().sum() >= series.notna().sum() and series.notna().any()
        if is_numeric:
            v = numeric.astype(float)
            v = v.fillna(v.mean())
            std = v.std()
            if std == 0 or not np.isfinite(std):
                logger_print(f'Warning: dropping constant covariate "{col}".', verbose)
                continue
            v = (v - v.mean()) / std
            rows.append(v.values.astype(float))
            names.append(str(col))
        else:
            s = series.astype('object')
            mode = s.dropna()
            mode = mode.mode().iloc[0] if len(mode) else ''
            s = s.fillna(mode)
            dummies = pd.get_dummies(s, prefix=str(col), drop_first=True)
            for dcol in dummies.columns:
                v = dummies[dcol].astype(float)
                if v.std() == 0:
                    continue
                v = v - v.mean()
                rows.append(v.values.astype(float))
                names.append(str(dcol))
    X = np.vstack(rows)
    return X, names


def create_project(project_name: str, promoter_expression_filename: str, loading_matrix_filenames: list[str],
                   motif_expression_filenames=None, loading_matrix_transformations=None, sample_groups=None, motif_postfixes=None,
                   promoter_filter_lowexp_cutoff=0.95, promoter_filter_plot_filename=None, promoter_filter_max=True,
                   promoter_filter_component_limit=0.6,
                   sample_groups_subset=False,  motif_names_filename=None, covariates_filename=None,
                   n_jobs:float = 0.5, compression='raw', dump=True, verbose=True):
    if not os.path.isfile(promoter_expression_filename):
        raise FileNotFoundError(f'Promoter expression file {promoter_expression_filename} not found.')
    if type(loading_matrix_filenames) is str:
        loading_matrix_filenames = [loading_matrix_filenames]
    for mx_name in loading_matrix_filenames:
        if not os.path.isfile(mx_name):
            raise FileNotFoundError(f'Loading matrix file {mx_name} not found.')
    if motif_expression_filenames:
        if type(motif_expression_filenames) is str:
            motif_expression_filenames = [motif_expression_filenames]
        for exp_name in motif_expression_filenames:
            if not os.path.isfile(exp_name):
                raise FileNotFoundError(f'Motif expresion file {exp_name} not found.')
    if type(sample_groups) is str:
        with open(sample_groups, 'r') as f:
            if sample_groups.endswith('.json'):
                sample_groups = json.load(f)
            else:
                sample_groups = dict()
                for line in f:
                    items = line.split()
                    sample_groups[items[0]] = items[1:]
    if motif_names_filename is not None:
        with open(motif_names_filename, 'r') as f:
            motif_names = list()
            for line in f:
                line = line.strip().split()
                for item in line:
                    if item:
                        motif_names.append(item)
    else:
        motif_names = None
    cpu_count = multiprocessing.cpu_count()
    if n_jobs < 1 and n_jobs > 0:
        n_jobs = max(1, int(n_jobs * cpu_count))
    elif n_jobs <= 0:
        n_jobs = cpu_count
    logger_print('Reading dataset...', verbose)
    promoter_expression = dt.fread(promoter_expression_filename, nthreads=n_jobs).to_pandas()
    promoter_expression = promoter_expression.set_index(promoter_expression.columns[0])
    
    if sample_groups:
        if sample_groups_subset:
            cols = set(promoter_expression.columns)
            to_rem = list()
            for group, samples in sample_groups.items():
                samples = set(samples) & cols
                if not samples:
                    to_rem.append(group)
                else:
                    sample_groups[group] = list(samples)
            for group in to_rem:
                del sample_groups[group]
        cols = set()
        for vals in sample_groups.values():
            cols.update(vals)
        cols = list(cols)
        promoter_expression = promoter_expression[cols]
    
    proms = promoter_expression.index
    sample_names = promoter_expression.columns
    loading_matrices = [dt.fread(f, nthreads=n_jobs).to_pandas() for f in loading_matrix_filenames]
    loading_matrices = [df.set_index(df.columns[0]).loc[proms] for df in loading_matrices]
    if loading_matrix_transformations is None or type(loading_matrix_transformations) is str:
        loading_matrix_transformations = [loading_matrix_transformations] * len(loading_matrices)
    else:
        if len(loading_matrix_transformations) == 1:
            loading_matrix_transformations = [loading_matrix_transformations[0]] * len(loading_matrices)
        elif len(loading_matrix_transformations) != len(loading_matrices):
            raise Exception(f'Total number of loading matrices is {len(loading_matrices)}, but the number of transformations is '
                            f'{len(loading_matrix_transformations)}.')
    
    logger_print('Filtering promoters of low expression...', verbose)
    inds, weights = filter_lowexp(promoter_expression, cutoff=promoter_filter_lowexp_cutoff, fit_plot_filename=promoter_filter_plot_filename,
                                  max_mode=promoter_filter_max, component_limit=promoter_filter_component_limit)
    promoter_expression = promoter_expression.loc[inds]
    logger_print(f'Kept {int(np.sum(inds))} of {len(inds)} promoters.', verbose)
    proms = promoter_expression.index
    test_chromosomes  = list() # ['chr2', 'chr15']
    loading_matrices = [transform_loadings(df, mode, prom_inds=inds, test_chromosomes=test_chromosomes,
                                           Y=promoter_expression) for df, mode in zip(loading_matrices, loading_matrix_transformations)]
    if motif_postfixes is not None:
        for mx, postfix in zip(loading_matrices, motif_postfixes):
            mx.columns = [f'{c}_{postfix}' for c in mx.columns]
    if motif_expression_filenames:
        motif_expression = [dt.fread(f, nthreads=n_jobs).to_pandas() for f in motif_expression_filenames]
        motif_expression = [df.set_index(df.columns[0]) for df in motif_expression]
        if motif_postfixes is not None:
            for mx, postfix in zip(motif_expression, motif_postfixes):
                mx.index = [f'{c}_{postfix}' for c in mx.index]
        if sample_groups:
            if len(set(motif_expression[0].columns) & set(sample_groups)) == len(sample_groups):
                for i in range(len(motif_expression)):
                    mx = motif_expression[i]
                    for group, cols in sample_groups.items():
                        for col in cols:
                            mx[col] = mx[group]
                    mx = mx.drop(sorted(sample_groups), axis=1)     
        motif_expression = [df.loc[mx.columns, sample_names] for df, mx in zip(motif_expression, loading_matrices)]
        motif_expression = pd.concat(motif_expression, axis=0)
    else:
        motif_expression = None
    loading_matrices = pd.concat(loading_matrices, axis=1)
    if motif_names is not None:
        motif_names = list(set(motif_names) & set(loading_matrices.columns))
        loading_matrices = loading_matrices[motif_names]
    proms = list(promoter_expression.index)
    sample_names = list(promoter_expression.columns)
    motif_names = list(loading_matrices.columns)
    loading_matrices = loading_matrices.values
    promoter_expression = promoter_expression.values
    if motif_expression is not None:
        motif_expression = motif_expression.values
    if not sample_groups:
        sample_groups = {n: [i] for i, n in enumerate(sample_names)}
    else:
        sample_groups = {n: sorted([sample_names.index(i) for i in inds]) for n, inds in sample_groups.items()}
    if covariates_filename:
        logger_print('Building covariate design matrix...', verbose)
        covariates, covariate_names = build_covariates(covariates_filename, sample_names,
                                                       n_jobs=n_jobs, verbose=verbose)
    else:
        covariates, covariate_names = None, None
    res = {'expression': promoter_expression,
           'loadings': loading_matrices,
           'motif_expression': motif_expression,
           'motif_postfixes': motif_postfixes,
           'promoter_names': proms,
           'sample_names': sample_names,
           'motif_names': motif_names,
           'weights': weights,
           'covariates': covariates,
           'covariate_names': covariate_names,
           'groups': sample_groups}
    if dump:
        folder = os.path.split(project_name)[0]
        name = os.path.split(project_name)[-1]
        for file in os.listdir(folder if folder else None):
            if file.startswith(f'{name}.') and file.endswith(tuple(openers.keys())):
                os.remove(os.path.join(folder, file))
        logger_print('Saving project...', verbose)
        with openers[compression](f'{project_name}.init.{compression}', 'wb') as f:
            dill.dump(res, f)
    return res
