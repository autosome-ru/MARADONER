# -*- coding: utf-8 -*-
import os
import sys

# BLAS/OpenMP runtimes read their thread limits once, when numpy/scipy/jax are loaded, so
# the limit has to be in the environment before anything below is imported. MARADONER
# works on many small matrices: with one thread per core the pools mostly spin and a fit
# takes several times longer than single-threaded, hence the default of 1.
_THREAD_VARS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')

def _requested_threads(argv: list):
    """Value of the global --threads option (or MARADONER_THREADS) if it is a positive
    integer, None otherwise."""
    value = os.environ.get('MARADONER_THREADS')
    for i, arg in enumerate(argv):
        if not arg.startswith('-'):
            break
        if arg == '--threads' and i + 1 < len(argv):
            value = argv[i + 1]
            break
        if arg.startswith('--threads='):
            value = arg.split('=', 1)[1]
            break
    try:
        value = int(value)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None

def _limit_threads(argv: list):
    """Returns the thread limit enforced by MARADONER, or None if the user's own
    environment variables are in charge."""
    n = _requested_threads(argv)
    if n is None:
        if any(var in os.environ for var in _THREAD_VARS):
            return None
        n = 1
    for var in _THREAD_VARS:
        os.environ[var] = str(n)
    return n

def _limit_xla_threads(n: int):
    """XLA does not read the variables above: it sizes its thread pool by the number of CPUs
    the process may run on when its backend starts. The backend is started with n CPUs
    allowed, and every thread is then allowed all of them again, so that the n threads of
    the pool run on any core."""
    if not hasattr(os, 'sched_setaffinity') or not os.path.isdir('/proc/self/task'):
        return
    allowed = os.sched_getaffinity(0)
    if n >= len(allowed):
        return
    os.sched_setaffinity(0, sorted(allowed)[:n])
    try:
        import jax
        jax.devices()
    finally:
        for tid in os.listdir('/proc/self/task'):
            try:
                os.sched_setaffinity(int(tid), allowed)
            except OSError:
                pass

_thread_limit = _limit_threads(sys.argv[1:])
# Only a limit asked for with --threads or MARADONER_THREADS applies to XLA as well.
if _thread_limit is not None and _requested_threads(sys.argv[1:]) is not None:
    _limit_xla_threads(_thread_limit)

from . import check_packages
check_packages()
from enum import Enum
from click import Context
from typer import Typer, Option, Argument, Exit
from typer.core import TyperGroup
from typing import List
from rich import print as rprint
from rich.console import Console
from rich.markup import escape
from jax import __version__ as jax_version
from scipy import __version__ as scipy_version
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from .create import create_project
from pathlib import Path
from .fit import fit, ClusteringMode, calculate_fov, predict, GOFStat, GOFStatMode
from .promoter_mean_test import FOVMeanMode, CovariateModel, estimate_promoter_mean
from .grn import grn
from .synthetic_data import generate_dataset
from time import time
from dill import __version__ as dill_version
from .export import export_results, export_loadings_product, Standardization, ANOVAType
from .export import export_contrast, export_covariate_contrast, export_posterior_anova
from .select_rank import select_activity_rank
from .bayes_linreg import bayes_linreg
from . import __version__ as project_version
from .select import select_motifs_single
from .utils import InputError, spinner, warn
import json

# Opt-in persistent cache of compiled JAX programs: most of the run time of small and
# medium projects is their compilation, which is the same from run to run.
if os.environ.get('MARADONER_JAX_CACHE'):
    try:
        import jax
        jax.config.update('jax_compilation_cache_dir', os.environ['MARADONER_JAX_CACHE'])
        jax.config.update('jax_persistent_cache_min_compile_time_secs', 0)
        jax.config.update('jax_persistent_cache_min_entry_size_bytes', -1)
    except (AttributeError, KeyError):
        pass

# logging.getLogger("jax._src.xla_bridge").addFilter(logging.Filter("No GPU/TPU found, falling back to CPU."))
# logging.getLogger("jax._src.xla_bridge").addFilter(logging.Filter("An NVIDIA GPU may be present on this machine, but a CUDA-enabled jaxlib is not installed. Falling back to cpu."))

__all__ = ['main']

class Compression(str, Enum):
    lzma = 'lzma'
    gzip = 'gzip'
    bz2 = 'bz2'
    raw = 'raw'

class MotifMeanPrior(str, Enum):
    none = 'none'
    ridge = 'ridge'
    families = 'families'
    covariance = 'covariance'

class CovariatePrior(str, Enum):
    none = 'none'
    shared = 'shared'
    separate = 'separate'

class LoadingTransform(str, Enum):
    none = 'none'
    scale = 'scale'
    ecdf = 'ecdf'
    esf = 'esf'
    drist = 'drist'
    dristun = 'dristun'


class OrderCommands(TyperGroup):
  def list_commands(self, ctx: Context):
    """Return list of commands in the order appear."""
    return list(self.commands)    # get commands using self.commands

_DO_NOT_UPDATE_HISTORY = False
_HISTORY_KEYS = ('jax', 'maradoner', 'scipy', 'dill', 'name')

def update_history(name: str, command: str, **kwargs):
    if _DO_NOT_UPDATE_HISTORY:
        return
    filename = f'{name}.json'
    try:
        with open(filename, 'r') as f:
            d = json.load(f)
        # A history file holds version strings and one dictionary of options per command;
        # anything else of this name (say, a sample groups file) is the user's own.
        if type(d) is not dict or any(type(v) is not dict and k not in _HISTORY_KEYS for k, v in d.items()):
            raise ValueError
    except FileNotFoundError:
        d = dict()
    except (ValueError, OSError):
        # Not a history file written by MARADONER: leave it alone rather than clobber it or
        # crash after the actual work has been done.
        warn(f'Could not record the "{command}" run: "{filename}" exists, but is not a '
             'MARADONER history file.')
        return
    if command == 'create':
        d.clear()
        d['jax'] = jax_version
        d['maradoner'] = project_version
        d['scipy'] = scipy_version
        d['dill'] = dill_version
        d['name'] = name
    elif command == 'fit':
        for k in ('test', 'test_binom', 'difftest', 'combine', 'export', 'plot'):
            if k in d:
                del d[k]
        for k in list(d):
            if k.startswith('export'):
                del d[k]
    d[command] = kwargs
    try:
        with open(filename, 'w') as f:
            json.dump(d, f, indent=4, default=str)
    except OSError as e:
        warn(f'Could not record the "{command}" run in "{filename}": {e.strerror}.')

def _done(t0: float):
    rprint(f'[green][bold]✔️[/bold] Done![/green]\t time: {time() - t0:.2f} s.')

def _parse_n_jobs(n_jobs: float):
    if n_jobs == -1 or 0 < n_jobs < 1:
        return n_jobs
    if n_jobs >= 1 and float(n_jobs).is_integer():
        return int(n_jobs)
    raise InputError(f'--n-jobs must be -1 (all threads), a fraction between 0 and 1 (of all threads) '
                     f'or a positive whole number of threads, but {n_jobs:g} was given.')

doc = f'''
[bold]MARADONER[/bold] version {project_version}: [bold]M[/bold]otif [bold]A[/bold]ctivity [bold]R[/bold]esponse [bold]A[/bold]nalysis [bold]Done R[/bold]ight
\b\n
\b\n
A typical [bold]MARADONER[/bold] session consists of sequential runs of [bold cyan]create[/bold cyan], [bold cyan]fit[/bold cyan],  [bold cyan]predict[/bold cyan] and, finally, \
[bold cyan]export[/bold cyan] commands. [bold]MARADONER[/bold] accepts files in the tabular format (.tsv or .csv, they also can come in gzipped-flavours), \
and requires the following inputs:
[bold dark_orange]•[/bold dark_orange] Promoter expression table of shape [blue]p[/blue] x [blue]s[/blue], where [blue]p[/blue] is a number of promoters and \
[blue]s[/blue] is a number of samples;
[bold dark_orange]•[/bold dark_orange] Matrix of loading coefficients of motifs onto promoters of shape [blue]p[/blue] x [blue]m[/blue], where [blue]m[/blue] \
is a number of motifs;
[bold dark_orange]•[/bold dark_orange] [i](Optional)[/i] Matrix of motif expression levels in log2 scale per sample of shape [blue]m[/blue] x [blue]s[/blue];
[bold dark_orange]•[/bold dark_orange] [i](Optional)[/i] JSON dictionary or a text file that collects samples into groups (if not supplied, it is assumed that \
each sample is a group of its own).
[red]Note:[/red] all tabular files must have named rows and columns.
All of the input files are supplied once at the [cyan]create[/cyan] stage. All of the commands are very customizable via numerous options, more \
details can be found in their respective helps, e.g.:
[magenta]>[/magenta] [cyan]maradoner fit --help[/cyan]

\b\n
If you found a bug or have any questions, feel free to contact us via
a) e-mail: [blue]iam@georgy.top[/blue] b) issue-tracker at [blue]github.com/autosome-ru/maradoner[/blue]
'''
app = Typer(rich_markup_mode='rich', cls=OrderCommands, add_completion=False, help=doc)

def _print_version(value: bool):
    if value:
        print(f'maradoner {project_version}')
        raise Exit()

@app.callback()
def _global_options(version: bool = Option(False, '--version', callback=_print_version, is_eager=True,
                                           help='Show the version and exit.'),
                    threads: int = Option(None, '--threads', min=1, envvar='MARADONER_THREADS',
                                          help='Number of threads for linear algebra (BLAS/OpenMP). Must precede the command name, '
                                               'e.g. [cyan]maradoner --threads 4 fit name[/cyan]. The default is 1 unless the '
                                               'OMP_NUM_THREADS family of environment variables is set: [bold]MARADONER[/bold] '
                                               'mostly works with many small matrices, where more threads usually make it slower.')):
    # The environment variables are set at import time (see the top of this file); this
    # covers the case when numpy was loaded before this module.
    limit = threads if threads else _thread_limit
    if limit:
        try:
            from threadpoolctl import threadpool_limits
            threadpool_limits(limits=limit)
        except ImportError:
            pass

from .mara.main import app_old
app.add_typer(app_old, name='mara', help='Run classical (IS)MARA framework. For testing purposes.')

help_str = 'Initialize [bold]MARADONER[/bold] project initial files: do parsing and filtering of the input data.'

@app.command('create', help=help_str)
def _create(name: str = Argument(..., help='Project name. [bold]MARADONER[/bold] will produce files for internal usage that start with [cyan]'
                                            'name[/cyan].'),
            expression: Path = Argument(..., help='A path to the promoter expression table. Expression values are assumed to be in a log-scale.'),
            loading: List[Path] = Argument(..., help='A list (if applicable, separated by space) of filenames containing loading matrices. '),
            loading_transform: List[LoadingTransform] = Option([LoadingTransform.esf], '--loading-transform', '-t',
                                                               help='A type of transformation to apply to loading '
                                                                'matrices. [dark_orange]ecdf[/dark_orange] substitutes values in the table with empirical CDF,'
                                                                ' [dark_orange]esf[/dark_orange] with negative logarithm of the empirical survival function. '
                                                                '[dark_orange]scale[/dark_orange] just rescales all entries to [0, 1] range. '
                                                                '[dark_orange]drist[/dark_orange] will attempt to find an optimal monotonous transform of each column. '
                                                                '[dark_orange]dristun[/dark_orange] is [dark_orange]drist[/dark_orange] but the transform function is same for all columns/motifs. '
                                                                'Either one transformation for all loading matrices or one per matrix (repeat the option). '
                                                                'Note that [dark_orange]esf[/dark_orange] and [dark_orange]ecdf[/dark_orange] make a column with only two distinct values '
                                                                '(e.g. 0/1 tables) constant: use [dark_orange]none[/dark_orange] or [dark_orange]scale[/dark_orange] for those.'),
            motif_expression: List[Path] = Option(None, help='A list of paths (of length equal to the number of loading matrices) of motif expression'
                                                  ' tables. All expression values are assumed to be in log2-scale. They are stored in the project, '
                                                  'but currently not used by any of the estimators.'),
            sample_groups: Path = Option(None, help='Either a JSON dictionary or a text file with a mapping between groups and sample names they'
                                          ' contain. If a text file, each line must start with a group name followed by space-separated sample names. '
                                          'A sample can belong to one group only; samples of the expression table that are in no group are dropped (with a warning).'),
            sample_groups_subset: bool = Option(False, help='Do not raise an exception if [dark_orange]sample-groups[/dark_orange] contains columns not present in the data.'),
            filter_lowexp_w: float = Option(0.9, help='Low-expression filter: a two-component max-normal mixture is fitted to the promoter '
                                            'expression, and a promoter is dropped iff its posterior probability of belonging to the low '
                                            'component exceeds [dark_orange]w[/dark_orange]. The closer [dark_orange]w[/dark_orange] to 1, the more promoters '
                                            'are left in the dataset; 1 disables the filter. The filter is also skipped (with a warning) if the '
                                            'fitted low-expression component holds most of the promoters.'),
            filter_max_mode: bool = Option(True, help='Use max-mode of filtering. Max-mode keeps promoters that are active at least for some samples.'
                                                       ' If disabled, filtration using GMM on the averages will be ran instead.'),
            filter_plot: bool = Option(True, help='Expression plot with a fitted mixture that is used for filtering.'),
            loading_postfix: List[str] = Option(None, '--loading-postfix', '-p',
                                                help='String postfixes that will be appended to the motifs from each of the supplied loading matrices '
                                                     '(one per matrix).'),
            motif_filename: Path = Option(None, '--motif-filename', help='If provided, then only motifs with names present in this file will be kept. '
                                                                         'Names are separated by whitespace and are matched after the postfixes have been appended.'),
            covariates: Path = Option(None, '--covariates', help='A path to a tabular file of sample covariates (first column: sample ids, '
                                      'remaining columns: covariates). If supplied, motif activity means are modeled as M X (with an '
                                      'intercept), i.e. covariates such as sex or age are accounted for. Numeric covariates are z-scored, '
                                      'categorical ones are one-hot encoded. Every sample must have a row; missing cells are imputed (with a warning). '
                                      'A column is numeric if all of its values are numbers, so write categories as text (batch_1, not 1).'),
            formula: str = Option(None, '--formula', help='A formula that chooses columns of the [dark_orange]covariates[/dark_orange] table and adds their '
                                  'interactions: "+" joins terms, "a:b" is an interaction, "a*b" stands for "a + b + a:b". For example, '
                                  '[dark_orange]"age * sex + batch"[/dark_orange]. By default every column of the table is used and there are no interactions.'),
            n_jobs: float = Option(0.5, help='Number of threads to use when reading datatables. If -1, maximal number of threads are used. If it is fractional in range (0, 1), '
                                   'the fraction maximal number is used. A whole number >= 1 is a number of threads.'),
            compression: Compression = Option(Compression.raw.value, help='Compression method used to store results.')):
    if type(compression) is Compression:
        compression = str(compression.value)
    if sample_groups:
        sample_groups = str(sample_groups)
    n_jobs = _parse_n_jobs(n_jobs)
    loading = list(map(str, loading))
    loading_transform = [x.value if issubclass(type(x), Enum) else str(x) for x in loading_transform]
    t0 = time()
    if filter_plot:
        filter_plot = f'{name}.filter-plot.png'
    if covariates:
        covariates = str(covariates)
    with spinner("Initializing project..."):
        r = create_project(name, expression, loading_matrix_filenames=loading, motif_expression_filenames=motif_expression,
                           loading_matrix_transformations=loading_transform, sample_groups=sample_groups,
                           sample_groups_subset=sample_groups_subset,
                           promoter_filter_lowexp_cutoff=filter_lowexp_w,
                           promoter_filter_plot_filename=filter_plot,
                           promoter_filter_max=filter_max_mode,
                           compression=compression,
                           motif_postfixes=loading_postfix,
                           motif_names_filename=motif_filename,
                           covariates_filename=covariates,
                           covariates_formula=formula,
                           n_jobs=n_jobs,
                           verbose=True)
    p, s = r['expression'].shape
    m = r['loadings'].shape[1]
    g = len(r['groups'])
    rprint(f'Number of promoters: {p}, number of motifs: {m}, number of samples: {s}, number of groups: {g}')
    if r.get('covariates') is not None:
        cov_names = [c for c in r['covariate_names'] if c != 'intercept']
        rprint(f'Number of covariates (excl. intercept): {len(cov_names)} ({escape(", ".join(map(str, cov_names)))})')
    _done(t0)


@app.command('fit', help='Estimate variance parameters and motif activities.')
def _fit(name: str = Argument(..., help='Project name.'),
          clustering: ClusteringMode = Option(ClusteringMode.none, help='Clustering method: motifs are replaced by clusters of motifs with '
                                              'similar loadings, and all the downstream commands then report clusters instead of motifs.'),
          num_clusters: int = Option(200, min=1, help='Number of clusters if [dark_orange]clustering[/dark_orange] is not [dark_orange]none[/dark_orange]. '
                                               'Must not exceed the number of motifs.'),
          test_promoters_filename: Path = Option(None, '--test-promoters-filename', help='Path to a text file containing gene/promoter names,'
                                                                                          ' separated by a newline character, to be used as a testing set.'),
          test_chromosomes: List[str] = Option(None, '--test-chromosomes', '-t', help='Chromosome whose promoters are held out of the fit as a testing set, '
                                               'e.g. 3, X or chr3 (promoter names must then contain the chromosome, like in chr3:1000-2000). '
                                               'Repeat the option to hold out several chromosomes. Note that the low-expression filter of '
                                               '[cyan]create[/cyan] has already looked at all promoters, the held-out ones included.'),
          motif_variance: bool = Option(True,  help='Estimate individual motif variances or assume them all equal. '
                                                  'The latter makes method more similar to (IS)MARA approach and might be beneficial when the data is small.'),
          promoter_variance: bool = Option(False, help='[i](Experimental)[/i] Reweight promoters by a heuristic estimate of their individual error '
                                                       'variances (a mixture-based reweighting, not a likelihood-based estimate) instead of '
                                                       'assuming them all equal.'),
          error_lowrank: int = Option(0, '--error-lowrank', min=0, help='Rank [dark_orange]k[/dark_orange] of a low-rank correction to the error '
                                      'covariance: the error column-covariance becomes [cyan]Theta = D + W W^T[/cyan] with [cyan]W[/cyan] of shape '
                                      '[blue]s x k[/blue], estimated jointly with [cyan]D[/cyan] inside the REML stage. Captures correlated '
                                      '(e.g. batch / unmodeled-covariate) structure across samples that the diagonal [cyan]D[/cyan] misses. '
                                      '[dark_orange]0[/dark_orange] (default) disables it and reproduces the classic diagonal model exactly. '
                                      'Must be smaller than the number of samples minus one.'),
          activity_lowrank: int = Option(0, '--activity-lowrank', min=0, help='Rank [dark_orange]k[/dark_orange] of a low-rank correction to the motif '
                                         'activity covariance: [cyan]G = diag(nu) + V V^T[/cyan] with [cyan]V[/cyan] of shape [blue]s x k[/blue], '
                                         'estimated jointly with [cyan]Sigma[/cyan] and [cyan]nu[/cyan]. Captures latent sample-factors along '
                                         'which motif activities co-vary. [dark_orange]0[/dark_orange] (default) disables it. Requires per-motif variances; '
                                         'must be smaller than the number of samples minus the number of covariates (intercept included).'),
          motif_families: Path = Option(None, '--motif-families', help='Families of motifs with co-varying activities: the covariance '
                                        'between motifs becomes [cyan]Sigma + F Omega F^T[/cyan], with a variance per family in '
                                        '[cyan]Omega[/cyan]. A JSON dictionary [blue]{"family": ["motif", ...]}[/blue], or a text file with '
                                        'a family per line ([blue]family motif motif ...[/blue]) or a motif per line ([blue]motif family[/blue]).'),
          motif_lowrank: int = Option(0, '--motif-lowrank', min=0, help='Rank [dark_orange]k[/dark_orange] of a covariance between motif '
                                      'families: [cyan]Omega[/cyan] becomes [cyan]Omega + L L^T[/cyan]. Without [purple]--motif-families[/purple] '
                                      'the covariance between motifs is [cyan]Sigma + L L^T[/cyan].'),
          motif_mean_prior: MotifMeanPrior = Option(MotifMeanPrior.none, '--motif-mean-prior',
                                      help='Prior on the mean activities of the motifs, with its variances estimated by REML: '
                                      '[dark_orange]ridge[/dark_orange] (a variance shared by all motifs), '
                                      '[dark_orange]families[/dark_orange] (on top of it, a part common to the motifs of a family, with one variance for all families; needs '
                                      '[purple]--motif-families[/purple]) or [dark_orange]covariance[/dark_orange] (a multiple of the '
                                      'fitted covariance between motif activities on top of it). Without a prior, the means of many '
                                      'similar motifs fit the levels of the training promoters and predict those of other promoters '
                                      'poorly.'),
          covariate_prior: CovariatePrior = Option(CovariatePrior.none, '--covariate-prior',
                                      help='The prior of [purple]--motif-mean-prior[/purple] for the effects of the covariates on '
                                      'the motif activities (the columns of [cyan]M[/cyan] other than the intercept) as well: '
                                      '[dark_orange]shared[/dark_orange] (one set of variances for all covariates) or '
                                      '[dark_orange]separate[/dark_orange] (one set per covariate). By default only the means are '
                                      'shrunk, and the effects of the covariates fit the training promoters as freely as the means '
                                      'without a prior.'),
          gpu: bool = Option(False, help='Use GPU if available for most of computations.'),
          gpu_decomposition: bool = Option(False, help='Use GPU if available for the SVD decomposition of the loading matrix.'),
          x64: bool = Option(True, help='Use high precision algebra.')):
    """
    Estimate the variance parameters and the mean parameters of the model for the given project.
    """

    t0 = time()
    with spinner("Fitting model to the data..."):
        fit(name, clustering=clustering, num_clusters=num_clusters,
            gpu=gpu, test_chromosomes=test_chromosomes,
            motif_variance=motif_variance,
            promoter_variance=promoter_variance,
            test_promoters_filename=test_promoters_filename,
            error_lowrank=error_lowrank,
            activity_lowrank=activity_lowrank,
            motif_families=motif_families, motif_lowrank=motif_lowrank,
            motif_mean_prior=motif_mean_prior.value, covariate_prior=covariate_prior.value,
            gpu_decomposition=gpu_decomposition, x64=x64)
    _done(t0)

__select_rank_doc = 'Choose the number of factors for [purple]--activity-lowrank[/purple] of [bold cyan]fit[/bold cyan]. '\
                    'Models with 1, 2, ... factors are fitted to the project one after another. A factor is accepted if it lowers the REML '\
                    'deviance more than a factor does on datasets simulated from the model without it, where there is nothing to find: '\
                    'the p-value is the probability of such a gain on one more simulated dataset, with the simulated gains taken for normal. '\
                    'The command needs a fitted project and changes nothing in it.'

@app.command('select-rank', help=__select_rank_doc)
def _select_rank(name: str = Argument(..., help='Project name.'),
                 max_rank: int = Option(5, '--max-rank', min=1, help='The largest number of factors to try.'),
                 null: int = Option(5, '--null', min=0, help='Number of simulated datasets per factor, at least 3. Each one costs 2 model fits. '
                                    'With [dark_orange]0[/dark_orange] only the gains are reported, for all ranks up to [dark_orange]max-rank[/dark_orange].'),
                 alpha: float = Option(0.05, '--alpha', help='A factor is accepted if its p-value is below this level.'),
                 full: bool = Option(False, '--full', help='Try all ranks up to [dark_orange]max-rank[/dark_orange] instead of stopping at the first '
                                     'factor that is not accepted.'),
                 seed: int = Option(0, help='Seed of the random number generator used for the simulated datasets.'),
                 output: Path = Option(None, '--output', help='A file to save the table to (tab-separated).')):
    t0 = time()
    with spinner("Choosing the number of factors..."):
        res = select_activity_rank(name, max_rank=max_rank, n_null=null, seed=seed, full=full, alpha=alpha)
    t = Table('Factors', 'REML deviance', 'Gain', 'Gain on simulated data\nmean (min .. max)', 'p-value', 'Accepted',
              title='Factors of the covariance of activities between samples')
    lines = ['rank\tdeviance\tgain\tnull_mean\tnull_min\tnull_max\tpvalue\taccepted']
    for step in res['steps']:
        gain, sim = step['gain'], step['null']
        row = [str(step['rank']), f'{step["deviance"]:.3f}', '' if gain is None else f'{gain:.3f}']
        line = [str(step['rank']), f'{step["deviance"]:.6f}', '' if gain is None else f'{gain:.6f}']
        if sim:
            row += [f'{sum(sim) / len(sim):.3f} ({min(sim):.3f} .. {max(sim):.3f})', f'{step["pvalue"]:.2g}',
                    'yes' if step['accepted'] else 'no']
            line += [f'{sum(sim) / len(sim):.6f}', f'{min(sim):.6f}', f'{max(sim):.6f}', f'{step["pvalue"]:.6g}',
                     'yes' if step['accepted'] else 'no']
        else:
            row += ['', '', '']
            line += ['', '', '', '', '']
        t.add_row(*row)
        lines.append('\t'.join(line))
    rprint(t)
    if res['rank'] is not None:
        rprint(f'Number of factors supported by the data: [bold]{res["rank"]}[/bold]' +
               (f' (fit it with [cyan]maradoner fit {escape(name)} --activity-lowrank {res["rank"]}[/cyan])' if res['rank'] else ''))
        if res['rank'] == res['steps'][-1]['rank'] and res['steps'][-1]['accepted'] and not res['capped']:
            rprint(f'All {res["rank"]} factors that were tried are accepted: there may be more of them, see [purple]--max-rank[/purple].')
    if output is not None:
        with open(output, 'w') as f:
            f.write('\n'.join(lines) + '\n')
    _done(t0)

__bayes_linreg_doc = '''Fit a single-sample Bayesian linear regression instead of the [bold]MARADONER[/bold] model.

The [bold]MARADONER[/bold] model reads activities off the variation [i]across samples[/i]. With a single \
sample there is nothing to vary, the REML objective for the error variances is flat and [cyan]fit[/cyan] \
is not identifiable. This command fits a different, single-sample model where the replication comes from \
the [blue]p[/blue] promoters instead:
[cyan]y = mu 1_p + B u + e[/cyan], [cyan]e ~ N(0, sigma^2 I_p)[/cyan], [cyan]u ~ N(0, Sigma)[/cyan],
with a scalar mean [cyan]mu[/cyan], a scalar error variance [cyan]sigma^2[/cyan] and a diagonal \
[cyan]Sigma[/cyan] of per-motif activity variances. The variance components are estimated by REML \
([cyan]mu[/cyan] is profiled out), motif activities are the posterior mean [cyan]E\\[u|y][/cyan] -- i.e. \
automatic relevance determination.

Each sample is fitted independently. Unlike the usual chain, this command goes straight from a project \
to result files: no [cyan]fit[/cyan]/[cyan]predict[/cyan]/[cyan]export[/cyan] in between, the output \
folder is supplied here.'''

@app.command('bayes_linreg', help=__bayes_linreg_doc,
             short_help='Single-sample alternative model: y = mu 1_p + B u + e. Writes results directly.')
def _bayes_linreg(name: str = Argument(..., help='Project name.'),
                  output_folder: Path = Argument(..., help='Output folder. Results are written here immediately.'),
                  alpha: float = Option(0.05, help='FDR alpha for the Benjamini-Hochberg correction of motif activity p-values.'),
                  stderr: bool = Option(True, help='Compute asymptotic standard errors of the variance components from the REML '
                                                   'observed information. Components shrunk to zero sit on the parameter-space '
                                                   'boundary and are reported as NaN. Note that the reported std of the activities is '
                                                   'conditional on the estimated variances and therefore understates their uncertainty.'),
                  promoters: bool = Option(False, help='Also write [dark_orange]promoters.tsv[/dark_orange] with the fitted value per promoter '
                                                       'per sample. Off by default: it is the only per-locus output and everything '
                                                       'else lives in the two summary tables.'),
                  x64: bool = Option(True, help='Use high precision algebra.')):
    __doc__ = __bayes_linreg_doc
    if not 0 < alpha < 1:
        raise InputError(f'--alpha is a false discovery rate and must lie strictly between 0 and 1, but {alpha:g} was given.')
    t0 = time()
    with spinner("Fitting single-sample Bayesian linear regression..."):
        bayes_linreg(name, output_folder, alpha=alpha, stderr=stderr,
                     promoters=promoters, x64=x64)
    update_history(name, 'bayes_linreg', output_folder=str(output_folder), alpha=alpha,
                   stderr=stderr, promoters=promoters)
    _done(t0)

__gof_doc = '''3-rd col:
[cyan]pred = mu_p 1_s^T + 1_p mu_s^T + B mu_m 1_s^T + B U[/cyan]
2-nd col:
[cyan]pred = mu_p 1_s^T + 1_p mu_s^T + B mu_m 1_s^T[/cyan]
1-st col:
[cyan]pred = mu_p 1_s^T + 1_p mu_s^T[/cyan]'''

@app.command('gof', help='Estimate GOFs given test/train data split. Statistics for the testing set are provided only if promoters were held out '
                         'in [cyan]fit[/cyan] ([dark_orange]--test-chromosomes[/dark_orange] or [dark_orange]--test-promoters-filename[/dark_orange]) '
                         'and [cyan]estimate-promoter-mean[/cyan] was run after [cyan]predict[/cyan].')
def _gof(name: str = Argument(..., help='Project name.'),
         use_groups: bool = Option(False, help='Compute statistic for samples aggregated across groups.'),
         stat_type: GOFStat = Option(GOFStat.fov, help='Statistic type to compute'),
         stat_mode: GOFStatMode = Option(GOFStatMode.total, help='Whether to compute stats for residuals or accumulate their effects'),
         weights: bool = Option(True, help='Whether to use weighted statistics. Applicable only if promoter variance was estimated.'),
         gpu: bool = Option(False, help='Use GPU if available for most of computations.'),
         x64: bool = Option(True, help='Use high precision algebra.')):
    """
    Compute goodness-of-fit statistics for the nested parts of the model for the given project.
    """

    t0 = time()
    with spinner("Calculating FOVs..."):
        res = calculate_fov(name, stat_mode=stat_mode, stat_type=stat_type, use_groups=use_groups,
                            weights=weights,
                            gpu=gpu, x64=x64)
    if stat_type == GOFStat.corr:
        title = 'Pearson correlation'
    else:
        title = 'Fraction of variance explained'
    if stat_mode == GOFStatMode.residual:
        title += ' (Residual)'
    else:
        title += ' (Total)'
    t = Table('Set\n[cyan]Y = [/cyan]',
              'centering\n[cyan]mu_p 1_s^T + 1_p mu_s^T[/cyan]',
              '+ mean motif activity\n[cyan]+ B M X[/cyan]',
              '+ group-specific motif activity\n[cyan]+ B U[/cyan]',
              title=title)
    row = [f'{t.total:.6f}' for t in res.train]

    t.add_row('train', *row)
    if res.test is not None:
        row = [f'{t.total:.6f}' for t in res.test]
        t.add_row('test', *row)
    rprint(t)
    # rprint(__gof_doc)
    _done(t0)

@app.command('estimate-promoter-mean', help='Estimate promoter-specific mean expression values for a testing subset.')
def _estimate_promoter_mean(name: str = Argument(..., help='Project name.'),
                            mean_mode: FOVMeanMode = Option(FOVMeanMode.null, help='Promoter-wise mean imputation method used for a testing set. '\
                                                                                  '[dark_orange]null[/dark_orange] substitutes mu_p with zeros. '\
                                                                                  '[dark_orange]gls[/dark_orange] uses a simple GLS estimator using only sample-wise variances after accounting for other effects - [red]leakage[/red]-prone. '\
                                                                                  '[dark_orange]knn[/dark_orange] employs KNN using principal components of the loading matrix and the total predicted effect matrices as features. '\
                                                                                  'Needs a fit with held-out promoters and must be re-run after each [cyan]predict[/cyan].'),
                            knn_n: int = Option(256, help='Number of nearest neighbours to use if [dark_orange]mean-mode=knn[/dark_orange].'),
                            pca_b: int = Option(64, help='Number of PC of the loading matrix to use if [dark_orange]mean-mode=knn[/dark_orange] (-1 uses all).'),
                            pca_z: int = Option(3, help='Number of PC of the total effect matrix to use if [dark_orange]mean-mode=knn[/dark_orange] (-1 uses all).'),
                            covariate: Path = Option(None, help='A path to a tsv file containing covariates. If present, will enhance mean predictions using GAMs '
                                                                                '(ignored if [dark_orange]mean-mode=gls[/dark_orange]).'),
                            covariate_model: CovariateModel = Option(CovariateModel.gam, help='How the promoter means are learned from the covariates: '
                                                                                '[dark_orange]gam[/dark_orange] fits a spline per covariate (for a few covariates); '
                                                                                '[dark_orange]ridge[/dark_orange], [dark_orange]boosting[/dark_orange] (gradient boosting) and '
                                                                                '[dark_orange]ensemble[/dark_orange] (the average of the two) take large tables, '
                                                                                'such as the output of [cyan]sequence-features[/cyan].')):
    """
    Estimate promoter-specific means of the held-out (testing) promoters for the given project.
    """

    t0 = time()
    with spinner("Estimating promoter-specific mean expression values..."):
        res = estimate_promoter_mean(name,
                                    mean_mode=mean_mode,
                                    knn_n=knn_n,
                                    pca_b=pca_b,
                                    pca_z=pca_z,
                                    covariate=covariate,
                                    covariate_model=covariate_model.value)
    _done(t0)

@app.command('sequence-features', help='Count k-mers, GC and CpG in bins around the TSS: a promoter covariate table for '
                                       '[cyan]estimate-promoter-mean --covariate[/cyan].')
def _sequence_features(fasta: Path = Argument(..., help='FASTA of promoter sequences in the orientation of the promoter, all with the TSS '
                                                        'at the same position (e.g. [cyan]bedtools getfasta -s -name+[/cyan] of windows '
                                                        'from 1000 bp upstream to 500 bp downstream). The promoter name is the part of the record name before "::".'),
                       output: Path = Argument(..., help='Output tsv file, one row per promoter.'),
                       tss: int = Option(1000, help='0-based position of the TSS in every sequence.'),
                       bins: str = Option('-1000,-500,-250,-50,50,250,500', help='Comma-separated bin edges, in bp relative to the TSS.'),
                       k: int = Option(5, help='Length of the k-mers.')):
    """
    Build sequence features of promoters: k-mer counts, GC content and CpG o/e per bin.
    """
    from .seqfeatures import make_sequence_features
    t0 = time()
    try:
        edges = [int(x) for x in str(bins).split(',') if x.strip()]
    except ValueError:
        raise InputError(f'--bins must be comma-separated whole numbers, but "{bins}" was given.')
    with spinner('Counting k-mers...'):
        table = make_sequence_features(str(fasta), str(output), tss=tss, bins=edges, k=k)
    rprint(f'{table.shape[0]} promoters, {table.shape[1]} features.')
    _done(t0)

@app.command('predict', help='Estimate deviations of motif activities from their means.')
def _predict(name: str = Argument(..., help='Project name.'),
         filter_motifs: bool = Option(True, help='Do not predict deviations from motifs whose variance is low.'),
         filter_order: int = Option(5, help='Motif variance is considered low if it is [dark_orange]filter-order[/dark_orange] (>= 0) orders of magnitude smaller than a median motif variance.'),
         tau_search: bool = Option(False, help='Search for tau multiplier using CV'),
         cv_repeats: int = Option(3, help='CV repeats in [dark_orange]RepeatedKFold[/dark_orange]'),
         cv_splits: int = Option(5, help='CV splits in [dark_orange]RepeatedKFold[/dark_orange]'),
         tau_left: float = Option(0.1, help='Left bound for tau search'),
         tau_right: float = Option(1.0, help='Right bound for tau search. Values > 1 imply that we believe that variance could be underestimated'),
         tau_num: int = Option(10, help='Number of splits in np.linspace(tau_left, tau_right, tau_num)'),
         pinv: bool = Option(False, help='Use plain Moore-Penrose pseudoinverse with Tikhonov regularization for tau > 1. Applicable for testing purposes; '
                                         'ignored for fits with a low-rank correction.'),
         gpu: bool = Option(False, help='Use GPU if available for most of computations.'),
         x64: bool = Option(True, help='Use high precision algebra.')):
    """
    Predict the group-specific deviations of motif activities from their means for the given project.
    """

    t0 = time()
    with spinner("Predicting motif activities..."):
        predict(name, tau_search=tau_search, cv_splits=cv_splits, cv_repeats=cv_repeats,
                tau_left=tau_left, tau_right=tau_right, tau_num=tau_num, pinv=pinv,
                filter_motifs=filter_motifs, filter_order=filter_order, gpu=gpu, x64=x64)
    _done(t0)

@app.command('generate', help='Generate synthetic dataset for testing purposes.')
def _generate(output_folder: Path = Argument(..., help='Output folder.'),
                p: int = Option(2000, help='Number of promoters.'),
                m: int = Option(500, help='Number of motifs.'),
                g: int = Option(10, help='Number of groups.'),
                min_samples: int = Option(5, help='Minimal number of observations per each group.'),
                max_samples: int = Option(6, help='Maximal number of observations per each group.'),
                non_signficant_motifs_fraction: float = Option(0.20, '--non-significant-motifs-fraction', '--non-signficant-motifs-fraction',
                                                               help='Fraction of non-significant motifs (the second, misspelt, name of the '
                                                                    'option is kept for backward compatibility).'),
                sigma_rel: float = Option(1e-1, help='Ratio of nu to sigma. The higher this value, the higher is FOV.'),
                means: bool = Option(True, help='Non-zero groupwise and promoter-wise intercepts'),
                n_covariates: int = Option(0, help='Number of covariates (beyond the intercept) to simulate. If > 0, the '
                                           'motif activity means follow M X and a covariates.tsv (plus ground-truth M_gt.tsv) is written.'),
                cov_effect: float = Option(1.0, help='Scale of the simulated covariate effects (columns of M beyond the intercept).'),
                error_lowrank: int = Option(0, '--error-lowrank', help='Inject a rank-[dark_orange]k[/dark_orange] correction W W^T into the error '
                                            'covariance (correlated noise across samples). Ground-truth W is written to W_err_gt.tsv.'),
                error_lowrank_scale: float = Option(1.0, help='Per-column scale of the injected error factor W, relative to mean sqrt(D).'),
                activity_lowrank: int = Option(0, '--activity-lowrank', help='Inject a rank-[dark_orange]k[/dark_orange] correction V V^T into the '
                                               'activity covariance G (correlated motif activities across samples). Ground-truth V -> V_act_gt.tsv.'),
                activity_lowrank_scale: float = Option(1.0, help='Per-column scale of the injected activity factor V, relative to mean sqrt(nu).'),
                families: int = Option(0, '--families', help='Number of motif families, among which the active motifs are split at random '
                                       '(families.txt, family_variances_gt.tsv).'),
                family_scale: float = Option(1.0, help='Variances omega of the families relative to the mean tau.'),
                motif_lowrank: int = Option(0, '--motif-lowrank', help='Inject a rank-[dark_orange]k[/dark_orange] covariance L L^T between the '
                                            'families, or between the motifs without --families (L_motif_gt.tsv).'),
                motif_lowrank_scale: float = Option(1.0, help='Entries of L relative to the square root of the mean tau.'),
                windows: str = Option(None, '--windows', help='Comma-separated names of windows around the TSS, e.g. '
                                      '[blue]core,up,down[/blue]: each motif gets a column of loadings per window, the windows of '
                                      'a motif form a family, and an active motif acts through all its windows together, through '
                                      'one of them or both (windows_gt.tsv). With [purple]--families[/purple] the motifs form families '
                                      'of a second level as well. families.txt holds both levels.'),
                seed: int = Option(1, help='Random seed.')
            ):
    t0 = time()
    with spinner("Generating synthetic dataset..."):
        generate_dataset(folder=output_folder, p=p, m=m, g=g, min_samples=min_samples, max_samples=max_samples,
                         non_signficant_motifs_fraction=non_signficant_motifs_fraction, sigma_rel=sigma_rel,
                         means=means,
                         B_cor=0, U_cor=0, E_cor=0, motif_cor=0,
                         n_covariates=n_covariates, cov_effect=cov_effect,
                         error_lowrank=error_lowrank, error_lowrank_scale=error_lowrank_scale,
                         activity_lowrank=activity_lowrank, activity_lowrank_scale=activity_lowrank_scale,
                         families=families, family_scale=family_scale,
                         motif_lowrank=motif_lowrank, motif_lowrank_scale=motif_lowrank_scale,
                         windows=[w.strip() for w in windows.split(',')] if windows else None,
                         seed=seed)
    _done(t0)

@app.command('export', help='Extract motif activities, parameter estimates, FOVs and statistical tests.')
def _export(name: str = Argument(..., help='Project name.'),
            output_folder: Path = Argument(..., help='Output folder.'),
            std_mode: Standardization = Option(Standardization.full, help='[i](Has no effect, kept for backward compatibility)[/i] Whether to standardize activities with plain variances or also decorrelate them.'),
            anova_mode: ANOVAType = Option(ANOVAType.positive, help='[dark_orange]positive[/dark_orange] tests whether the variance of a motif is greater than zero. '
                                           '[dark_orange]negative[/dark_orange] reports the opposite (lower) tail of the Wald statistic instead: a small value means that the '
                                           'estimate is unusually close to zero, which is not by itself evidence that a motif does not vary.'),
            zero_margin: float = Option(None, help='If given, the [dark_orange]zero[/dark_orange] column of sign.tsv becomes an equivalence test of the mean motif '
                                                   'activity lying within [-margin, margin]; by default it is just the lower tail of the |z| statistic.'),
            weighted_zscore: bool = Option(False, help='Reciprocal variance weighted Z-scores'),
            alpha: float = Option(0.05, help='FDR level of the classes in tests/families.tsv. The exported FDR values are adjusted p-values and do not depend on it.'),
            log_counts: bool = Option(False, help='Export expression value predictions.'),
            log_counts_grouped: bool = Option(True, help='Whether to export expressions wrt to groups or samples.'),
            loadings_product: bool = Option(False, help='Export loading matrix-activity 3D tensor. This will produce num_of_groups tabular files.'),
            lp_hdf: bool = Option(True, help='Each loadings-product table will be stored in hdf format (occupies much less space than plain tsv) using float16 precision.'),
            lp_intercepts: bool = Option(True, help='Include motif means in the 3D tensor.'),
            lp_tsv_truncation: int = Option(4, min=0, help='Number of digits after a floating point to truncate. Decreases the output size of a tabular if [dark_orange]lp-hdf[/dark_orange] is disabled.')):
    if not 0 < alpha < 1:
        raise InputError(f'--alpha is a false discovery rate and must lie strictly between 0 and 1, but {alpha:g} was given.')
    t0 = time()
    with spinner("Exporting results..."):
        export_results(name, output_folder, std_mode=std_mode, anova_mode=anova_mode, alpha=alpha,
                       weighted_zscore=weighted_zscore, export_counts=log_counts, counts_grouped=log_counts_grouped,
                       zero_margin=zero_margin)

    if loadings_product:
        with spinner("Exporting results..."):
            export_loadings_product(name, output_folder, use_hdf=lp_hdf, intercepts=lp_intercepts,
                                    tsv_truncation=lp_tsv_truncation)

    _done(t0)

@app.command('anova', help='Run posterior ANOVA across groups of interest.')
def _anova(name: str = Argument(..., help='Project name.'),
            filename: Path = Argument(..., help='Path of the output table (tsv).'),
            groups: List[str] = Argument(..., help='Names of the groups to compare (at least two).')):
    t0 = time()
    with spinner("Computing posterior ANOVA..."):
        export_posterior_anova(name, filename, groups)
    _done(t0)


__select_motif_doc = 'Selects best motif variants when the project was created from multiple loading matrices, each with an unique postfix. '\
                     'Best motif variant is selected by comparing their standardized contribution/z-score of both intercepts and deviations. '\
                     'The resulting list of motif names with the appropriate postfixes will be then stored in [dark_orange]filename[/dark_orange]. '\
                     'It is assumed that [dark_orange]filename[/dark_orange] will be supplied as [dark_orange]motif-filename[/dark_orange] to the [bold cyan]create[/bold cyan].'
@app.command('select-motifs',
             help=__select_motif_doc)
def _select_motifs(name: str = Argument(..., help='Project name'),
                   filename: Path = Argument(..., help='Filename where a list of best motif variants will be stored')):
    t0 = time()
    with spinner("Selecting motifs..."):
        select_motifs_single(name, filename)
    _done(t0)


__grn_doc = '\bTests each promoter against each motif per each group. Some people call it a GRN.\n'\
            'It compares H_0 against H_1:\n'\
            ' [cyan]H_0: y = b u + b μ 1_s^T + e,[/cyan]\n'\
            " [cyan]H_1: y = (b u)' + (b μ)' 1_s^T + e,[/cyan]\n"\
            "where y is a vector of gene expression observations, apostrophe denotes the effect of the motif of interest being excluded from the model. "\
            "If [purple]--no-means[/purple] is supplied, H_1 instead is\n"\
            " [cyan]H_1: y = (b u)' + [dark_orange]b μ [/dark_orange]1_s^T + e,[/cyan]\n "\
            "so we measure only effect that is due to the deviation u. "\
            "With [purple]--covariate[/purple], a covariate is tested in place of a group: the motif of interest loses "\
            "only the part of its mean activity that is due to the covariate (its effect times the covariate), "\
            "and all samples are used."
@app.command('grn',
             help=__grn_doc)
def _grn(name: str = Argument(..., help='Project name'),
         folder: Path = Argument(..., help='Output folder where results will be stored. In total, expect number_of_groups tables of size'
                                           ' comparable to the expression file size.'),
         hdf: bool = Option(True, help='Use HDF format instead of plain tsv files. Typically eats much less space'),
         stat: bool = Option(False, help='Save statistics alongside probabilities.'),
         prior_h1: float = Option(1/2, help='Prior belief on the expected fraction of motifs active per promoter (strictly between 0 and 1).'),
         means: bool = Option(True, help='Test motif-specific means along with activities deviations, otherwise only the latter are tested.'),
         covariate: List[str] = Option(None, '--covariate', '-c',
                                       help='Test the effect of a covariate on motif activities instead of the activities themselves: '
                                            'the contribution of a motif is its loading times the effect times the covariate. '
                                            'All samples are used, and a single table is written to the [dark_orange]covariates[/dark_orange] '
                                            'subfolder. Names are those listed by [bold cyan]create[/bold cyan]; the option can be supplied '
                                            'multiple times for a table per covariate, and [dark_orange]all[/dark_orange] stands for all '
                                            'covariates. [purple]--means[/purple]/[purple]--no-means[/purple] are of no effect then.')):
    t0 = time()
    with spinner("Building GRN..."):
        grn(name, output=folder, use_hdf=hdf, save_stat=stat, prior_h1=prior_h1, include_mean=means,
            covariates=covariate)
    _done(t0)

__estimate_promvar_doc = 'Estimates each promoter variance for each group using Empirical Bayesian shrinkage.'\
                         ' The nature of the approach is similar to the one introduced in limma-voom. A very recommended step before computing GRN.'

@app.command('contrast',
             help='Perform differential test given a contrasts vector.'
                 ' The contrast vector can be any algebraic string with variables being group names.'
                 ' For example, "[dark_orange]healthy - ill[/dark_orange]", "[dark_orange](group_a + group_b + group_c) / 3 - group_d[/dark_orange]".'
                 ' It computes a weighted algebraic sum of motif activities and then performs a series of Z-tests for each motif.')
def _contrast(name: str = Argument(..., help='Project name'),
                                contrast: str = Argument(...,
                                                          help='An algebraic expression defining the contrast vector.'),
                                output_folder: Path = Argument(..., help='Output folder.'),
                                postfix: str = Option(None, help='Postfix to add to the results file.'),
                                covariates: bool = Option(False, '--covariates', help='The variables of the contrast are covariates (as [bold cyan]create[/bold cyan] '
                                                          'lists them) instead of groups, and their effects on motif activities are compared: '
                                                          '"[dark_orange]tissue_liver - tissue_muscle[/dark_orange]". The reference level of a categorical '
                                                          'covariate can be named too, its effect is zero. The result is saved as '
                                                          '[dark_orange]covariate_contrast.tsv[/dark_orange].')):
    t0 = time()
    with spinner("Performing and saving a pairwise test..."):
        if covariates:
            export_covariate_contrast(name, output_folder, contrast, postfix)
        else:
            export_contrast(name, output_folder, contrast, postfix)
    _done(t0)


def _fail(message: str, code: int = 1):
    # soft_wrap: no hard line breaks inside paths when the output goes to a file or a pipe
    Console(stderr=True, soft_wrap=True).print(f'[bold red]Error:[/bold red] {escape(message)}', highlight=False)
    sys.exit(code)

def main():
    debug = bool(os.environ.get('MARADONER_DEBUG'))
    for stream in (sys.stdout, sys.stderr):
        # Terminals and pipes that are not UTF-8 must not turn a finished run into a crash
        # on the first non-ASCII character of a message.
        if (getattr(stream, 'encoding', None) or '').lower().replace('-', '') != 'utf8' and hasattr(stream, 'reconfigure'):
            stream.reconfigure(errors='replace')
    try:
        app()
    except InputError as e:
        if debug:
            raise
        _fail(str(e))
    except OSError as e:
        # Files and folders that cannot be read or written: missing, not permitted, a file
        # where a folder is expected, a full disk, a name that is too long, ...
        if debug:
            raise
        if isinstance(e, FileExistsError) and e.filename is not None:
            _fail(f'"{e.filename}" exists, but it is a file: a folder was expected there.')
        _fail(f'{e.strerror}: "{e.filename}".' if e.filename is not None and e.strerror else str(e))
    except MemoryError:
        if debug:
            raise
        _fail('Out of memory. Consider a smaller number of motifs/promoters, or run the command on a machine with more RAM.')
    except Exception:
        Console(stderr=True).print('[bold red]Unexpected internal error.[/bold red] If your inputs look right, this is likely a bug in '
                                   'MARADONER: please report it at github.com/autosome-ru/maradoner together with the traceback below.',
                                   highlight=False)
        raise
