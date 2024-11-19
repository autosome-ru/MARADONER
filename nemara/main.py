# -*- coding: utf-8 -*-
from . import check_packages
from enum import Enum
from click import Context
from typer import Typer, Option, Argument
from typer.core import TyperGroup
from typing import List
from rich import print as rprint
from jax import __version__ as jax_version
from scipy import __version__ as scipy_version
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from .create import create_project
from pathlib import Path
from .fit import fit, Regularization, SigmaStructure, CovarianceMode, ClusteringMode# cross_validate
from .synthetic_data import generate_dataset
from time import time
from dill import __version__ as dill_version
import logging
from .export import export_results, Standardization
from . import __version__ as project_version
import json

# logging.getLogger("jax._src.xla_bridge").addFilter(logging.Filter("No GPU/TPU found, falling back to CPU."))
# logging.getLogger("jax._src.xla_bridge").addFilter(logging.Filter("An NVIDIA GPU may be present on this machine, but a CUDA-enabled jaxlib is not installed. Falling back to cpu."))

__all__ = ['main']

class Compression(str, Enum):
    lzma = 'lzma'
    gzip = 'gzip'
    bz2 = 'bz2'
    raw = 'raw'

class LoadingTransform(str, Enum):
    none = 'none'
    ecdf = 'ecdf'
    esf = 'esf'

class OrderCommands(TyperGroup):
  def list_commands(self, ctx: Context):
    """Return list of commands in the order appear."""
    return list(self.commands)    # get commands using self.commands

_DO_NOT_UPDATE_HISTORY = False

def update_history(name: str, command: str, **kwargs):
    if _DO_NOT_UPDATE_HISTORY:
        return
    try:
        with open(f'{name}.json', 'r') as f:
            d = json.load(f)
    except FileNotFoundError:
        d = dict()
    if command == 'create':
        d.clear()
        d['jax'] = jax_version
        d['nemara'] = project_version
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
    with open(f'{name}.json', 'w') as f:
        json.dump(d, f, indent=4)
    
doc = f'''
[bold]NeMARA[/bold] version {project_version}: Placeholder Name 
\b\n
\b\n
A typical [bold]NeMARA[/bold] session consists of sequential runs of [bold cyan]create[/bold cyan], [bold cyan]fit[/bold cyan],  and, finally, \
[bold cyan]export[/bold cyan] commands. [bold]NeMARA[/bold] accepts files in the tabular format (.tsv or .csv, they also can come in gzipped-flavours), \
and requires the following inputs:
[bold orange]•[/bold orange] Promoter expression table of shape [blue]p[/blue] x [blue]s[/blue], where [blue]p[/blue] is a number of promoters and \
[blue]s[/blue] is a number of samples;
[bold orange]•[/bold orange] Matrix of loading coefficients of motifs onto promoters of shape [blue]p[/blue] x [blue]m[/blue], where [blue]m[/blue] \
is a number of motifs;
[bold orange]•[/bold orange] [i](Optional)[/i] Matrix of motif expression levels in log2 scale per sample of shape [blue]m[/blue] x [blue]s[/blue];
[bold orange]•[/bold orange] [i](Optional)[/i] JSON dictionary or a text file that collects samples into groups (if not supplied, it is assumed that \
each sample is a group of its own).
[red]Note:[/red] all tabular files must have named rows and columns.
All of the input files are supplied once at the [cyan]create[/cyan] stage. All of the commands are very customizable via numerous options, more \
details can be found in their respective helps, e.g.:
[magenta]>[/magenta] [cyan]nemara fit --help[/cyan]
The [cyan]fit[/cyan] is especially option-heavy, many of which impact the power of the [bold]NeMARA[/bold]. To asses which set of hyperparameters is \
the most suitable, [bold]NeMARA[/bold] allows to do cross-validation via the [cyan]cv[/cyan] command. 
\b\n
If you found a bug or have any questions, feel free to contact us via
a) e-mail: [blue]iam@georgy.top[/blue] b) issue-tracker at [blue]github.com/autosome-ru/neMARA[/blue]
'''
app = Typer(rich_markup_mode='rich', cls=OrderCommands, add_completion=False, help=doc)

help_str = 'Initialize [bold]NeMARA[/bold] project initial files: do parsing and filtering of the input data.'

@app.command('create', help=help_str)
def _create(name: str = Argument(..., help='Project name. [bold]NeMARA[/bold] will produce files for internal usage that start with [cyan]'
                                            'name[/cyan].'),
            expression: Path = Argument(..., help='A path to the promoter expression table. Expression values are assumed to be in a log-scale.'),
            loading: List[Path] = Argument(..., help='A list (if applicable, separated by space) of filenames containing loading matrices. '),
            loading_transform: List[LoadingTransform] = Option([LoadingTransform.none], '--loading-transform', '-t',
                                                               help='A type of transformation to apply to loading '
                                                                'matrices. [orange]ecdf[/orange] substitutes values in the table with empricical CDF,'
                                                                ' [orange]esf[/orange] with negative logarithm of the empirical survival function.'),
            motif_expression: List[Path] = Option(None, help='A list of paths (of length equal to the number of loading matrices) of motif expression'
                                                  ' tables. All expression values are assumed to be in log2-scale.'),
            sample_groups: Path = Option(None, help='Either a JSON dictionary or a text file with a mapping between groups and sample names they'
                                          ' contain. If a text file, each line must start with a group name followed by space-separated sample names.'),
            filter_lowexp_w: float = Option(0.9, help='Truncation boundary for filtering out low-expressed promoters. The closer [orange]w[/orange]'
                                            ' to 1, the more promoters will be left in the dataset.'),
            filter_plot: Path = Option(None, help='Expression plot with a fitted mixture that is used for filtering.'),
            loading_postfix: List[str] = Option(None, '--loading-postfix', '-p', 
                                                help='String postfixes will be appeneded to the motifs from each of the supplied loading matrices'),
            compression: Compression = Option(Compression.raw.value, help='Compression method used to store results.')):
    if type(compression) is Compression:
        compression = str(compression.value)
    if sample_groups:
        sample_groups = str(sample_groups)
    loading = list(map(str, loading))
    loading_transform = [x.value if issubclass(type(x), Enum) else str(x) for x in loading_transform]
    t0 = time()
    p = Progress(SpinnerColumn(speed=0.5), TextColumn("[progress.description]{task.description}"), transient=True)
    p.add_task(description="Initializing project...", total=None)
    p.start()
    r = create_project(name, expression, loading_matrix_filenames=loading, motif_expression_filenames=motif_expression, 
                       loading_matrix_transformations=loading_transform, sample_groups=sample_groups, 
                       promoter_filter_lowexp_cutoff=filter_lowexp_w,
                       promoter_filter_plot_filename=filter_plot,                       
                       compression=compression, 
                       motif_postfixes=loading_postfix, verbose=False)
    p.stop()
    dt = time() - t0
    p, s = r['expression'].shape
    m = r['loadings'].shape[1]
    g = len(r['groups'])
    rprint(f'Number of promoters: {p}, number of motifs: {m}, number of samples: {s}, number of groups: {g}')
    rprint(f'[green][bold]✔️[/bold] Done![/green]\t time: {dt:.2f} s.')
    

@app.command('fit', help='Estimate variance parameters and motif activities.')
def _fit(name: str = Argument(..., help='Project name.'),
          clustering: ClusteringMode = Option(ClusteringMode.none, help='Clustering method.'),
          num_clusters: int = Option(200, help='Number of clusters if [orange]clustering[/orange] is not [orange]none[/orange].'),
          sigma_structure: SigmaStructure = Option(SigmaStructure.identity, help='If [orange]diag[/orange], estimates individual variance parameters.'),
          cov_mode: CovarianceMode = Option(CovarianceMode.posterior, help='Type of covariance matrix estimates to use for standardization'),
          nu_cv_search: bool = Option(False, help='Try to improve "nu" estimates using CV at the activities estimation stage.'),
          nu_cv_splits: int = Option(5, help='Number of CV splits.'),
          nu_cv_repeats: int = Option(1, help='Number of CV repeats'), 
          nu_search_min: float = Option(0.5, help='Minimal tau multiplier for CV'),
          nu_search_max: float = Option(3.0, help='Maximal tau multiplier for CV'),
          nu_search_steps: int = Option(21, help='Number of stems between [cyan]nu_search_min[/cyan] and [cyan]nu_search_max[/cyan]'),
          regul: Regularization = Option(Regularization.none, help='Regularization for motif variances estimates. Both regularizaiton types rely on the'
                                        ' motif expression info.'),
          alpha: float = Option(1.0, help='Regularization strength.'),
          temperature: float = Option(100.0, help='Temperature hyperparameter for [orange]rank[/orange] regularization.'),
          exact_promoter_means_solver: bool = Option(False, help='Whether to use exact solution via Moore-Penrose inverse or an iterative method.'),
          gpu: bool = Option(False, help='Use GPU if available for most of computations.'), 
          gpu_decomposition: bool = Option(False, help='Use GPU if available or SVD decomposition.'), 
          x64: bool = Option(True, help='Use high precision algebra.')):
    """
    Fit a a mixture model parameters to data for the given project.
    """

    t0 = time()
    p = Progress(SpinnerColumn(speed=0.5), TextColumn("[progress.description]{task.description}"), transient=True)
    p.add_task(description="Fitting model to the data...", total=None)
    p.start()
    fit(name, sigma_structure=sigma_structure, clustering=clustering, num_clusters=num_clusters,
        cov_mode=cov_mode, nu_cv_search=nu_cv_search, nu_cv_splits=nu_cv_splits,
        nu_cv_repeats=nu_cv_repeats, nu_search_min=nu_search_min, nu_search_max=nu_search_max, nu_search_steps=nu_search_steps,
        regularization=regul, alpha=alpha, temperature=temperature,
        exact_promoter_means_solver=exact_promoter_means_solver, gpu=gpu,
        gpu_decomposition=gpu_decomposition, x64=x64)
    p.stop()
    dt = time() - t0
    rprint(f'[green][bold]✔️[/bold] Done![/green]\t time: {dt:.2f} s.')


# @app.command('cv', help='Cross-validate model hyperparameters.')
# def _cv(name: str = Argument(..., help='Project name.'),
#         n_splits: int = Option(4, help='Number of CV splits.'),
#         n_repeats: int = Option(1, help='Number of CV repeats.'),
#         clustering: List[Clustering] = Option([Clustering.none], '--clustering', '-n',
#                                               help='Clustering method.'),
#         n_clusters: List[int] = Option([200], '--n-clusters', '-n',
#                                        help='Number of clusters if [orange]clustering[/orange] is not [orange]none[/orange].'),
#         tau: list[float] = Option([1.0], '--tau', '-t',
#                                   help='Tau parameter that controls overfitting. The higher it is, the more variance will be explained by the model.'),
#         motif_variances: List[bool] = Option([False], '--motif-variance', '-e',
#                                              help='Estimate individual motif variances. Takes plenty of time.'),
#         regul: List[Regularization] = Option([Regularization.none], '--regul', '-r',
#                                              help='Regularization for motif variances estimates. Both regularizaiton types rely on the'
#                                                  ' motif expression info.'),
#         alpha: List[float] = Option([1.0], '--alpha', '-a', help='Regularization strength.'),
#         n_jobs: int = Option(1, help='Number of jobs to be run at parallel, -1 will use all available threads. [red]Improves performance only if'
#                             ' there is a plenty of cores as JAX uses multi-processing by deafult.[/red] Parallelization is done across groups.')):
#     """
#     Fit a a mixture model parameters to data for the given project.
#     """
#     t0 = time()
#     rprint('Starting CV...')
#     raise NotImplemented
#     # cross_validate(name, n_splits=n_splits, n_repeats=n_repeats,
#     #                regul=regul, alpha=alpha, estimate_motif_vars=motif_variances, tau=tau, clustering=clustering, n_clusters=n_clusters,
#     #                 n_jobs=n_jobs, verbose=True)
#     dt = time() - t0
#     rprint(f'[green][bold]✔️[/bold] Done![/green]\t time: {dt:.2f} s.')

@app.command('generate', help='Generate synthetic dataset for testing purporses.')
def _generate(output_folder: Path = Argument(..., help='Output folder.'),
                p: int = Option(2000, help='Number of promoters.'),
                m: int = Option(500, help='Number of motifs.'),
                g: int = Option(10, help='Number of groups.'),
                min_samples: int = Option(5, help='Minimal number of observations per each group.'),
                max_samples: int = Option(6, help='Maximal number of observations per each group.'),
                non_signficant_motifs_fraction: float = Option(0.20, help='Fraction of non-significant motifs.'),
                sigma_rel: float = Option(1e-1, help='Ratio of nu to sigma. The higher this value, the higher is FOV.'),
                means: bool = Option(True, help='Non-zero groupwise and promoter-wise intercepts'),
                seed: int = Option(1, help='Random seed.')
            ):
    t0 = time()
    pr = Progress(SpinnerColumn(speed=0.5), TextColumn("[progress.description]{task.description}"), transient=True)
    pr.add_task(description="Generating synthetic dataset...", total=None)
    pr.start()
    generate_dataset(folder=output_folder, p=p, m=m, g=g, min_samples=min_samples, max_samples=max_samples,
                     non_signficant_motifs_fraction=non_signficant_motifs_fraction, sigma_rel=sigma_rel,
                     means=means,seed=seed)
    pr.stop()
    dt = time() - t0
    rprint(f'[green][bold]✔️[/bold] Done![/green]\t time: {dt:.2f} s.')

@app.command('export', help='Extract motif activities, parameter estimates, FOVs and statistical tests.')
def _export(name: str = Argument(..., help='Project name.'),
            output_folder: Path = Argument(..., help='Output folder.'),
            std_mode: Standardization = Option(Standardization.full, help='Whether to standardize activities with plain variances or also decorrelate them.'),
            alpha: float = Option(0.05, help='FDR alpha.')):
    t0 = time()
    p = Progress(SpinnerColumn(speed=0.5), TextColumn("[progress.description]{task.description}"), transient=True)
    p.add_task(description="Fitting model to the data...", total=None)
    p.start()
    export_results(name, output_folder, std_mode=std_mode, alpha=alpha)
    p.stop()
    dt = time() - t0
    rprint(f'[green][bold]✔️[/bold] Done![/green]\t time: {dt:.2f} s.')


def main():
    check_packages()
    app()
