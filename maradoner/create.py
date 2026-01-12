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
    stds = df.std()
    drop_inds = (stds == 0) | np.isnan(stds)
    if prom_inds is not None:
        df = df.loc[prom_inds, ~drop_inds]
    else:
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
            t = np.unique(v)[1]
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

def create_project(project_name: str, promoter_expression_filename: str, loading_matrix_filenames: list[str],
                   motif_expression_filenames=None, loading_matrix_transformations=None, sample_groups=None, motif_postfixes=None,
                   promoter_filter_lowexp_cutoff=0.95, promoter_filter_plot_filename=None, promoter_filter_max=True,
                   motif_names_filename=None, n_jobs:float = 0.5, compression='raw', dump=True, verbose=True):
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
                                  max_mode=promoter_filter_max)
    promoter_expression = promoter_expression.loc[inds]
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
    res = {'expression': promoter_expression, 
           'loadings': loading_matrices,
           'motif_expression': motif_expression,
           'motif_postfixes': motif_postfixes,
           'promoter_names': proms,
           'sample_names': sample_names,
           'motif_names': motif_names,
           'weights': weights,
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
