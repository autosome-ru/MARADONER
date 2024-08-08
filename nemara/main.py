# -*- coding: utf-8 -*-
from . import check_packages
from enum import Enum
from click import Context
from typer import Typer, Option, Argument
from typer.core import TyperGroup
from typing import List
from rich import print as rprint
from betanegbinfit import __version__ as bnb_version
from jax import __version__ as jax_version
from scipy import __version__ as scipy_version
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from .create import create_project
from pathlib import Path
from .fit import fit, cross_validate
from time import time
from dill import __version__ as dill_version
import logging
from .export import export_results
from . import __version__ as project_version
import json

logging.getLogger("jax._src.xla_bridge").addFilter(logging.Filter("No GPU/TPU found, falling back to CPU."))
logging.getLogger("jax._src.xla_bridge").addFilter(logging.Filter("An NVIDIA GPU may be present on this machine, but a CUDA-enabled jaxlib is not installed. Falling back to cpu."))

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

class Clustering(str, Enum):
    none = 'none'
    kmeans = 'K-Means'
    nmf = 'NMF'

class Regularization(str, Enum):
    none = 'none'
    l2 = 'l2-prior'
    ranked = 'ranked'

class Standardization(str, Enum):
    full = 'full'
    std = 'std'


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
            filter_lowexp_w: float = Option(0.95, help='Truncation boundary for filtering out low-expressed promoters. The closer [orange]w[/orange]'
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
          cv: bool = Option(False, help='Use hyperparameters from the [cyan]cv[/cyan] call.'),
          clustering: Clustering = Option(Clustering.none, help='Clustering method.'),
          n_clusters: int = Option(200, help='Number of clusters if [orange]clustering[/orange] is not [orange]none[/orange].'),
          tau: float = Option(1.0, help='Tau parameter that controls overfitting. The higher it is, the more variance will be explained by the model.'),
          motif_variances: bool = Option(False, help='Estimate individual motif variances. Takes plenty of time.'),
          regul: Regularization = Option(Regularization.none, help='Regularization for motif variances estimates. Both regularizaiton types rely on the'
                                        ' motif expression info.'),
          alpha: float = Option(1.0, help='Regularization strength.'),
          n_jobs: int = Option(1, help='Number of jobs to be run at parallel, -1 will use all available threads. [red]Improves performance only if'
                              ' there is a plenty of cores as JAX uses multi-processing by deafult.[/red] Parallelization is done across groups.')):
    """
    Fit a a mixture model parameters to data for the given project.
    """
    if clustering == Clustering.none:
        clustering = None
    else:
        clustering = clustering.value
    if regul == Regularization.none:
        regul = None
    else:
        regul = regul.value
    t0 = time()
    p = Progress(SpinnerColumn(speed=0.5), TextColumn("[progress.description]{task.description}"), transient=True)
    p.add_task(description="Fitting model to the data...", total=None)
    p.start()
    fit(name, regul=regul, alpha=alpha, estimate_motif_vars=motif_variances, tau=tau, clustering=clustering, n_clusters=n_clusters,
        n_jobs=n_jobs, verbose=False)
    p.stop()
    dt = time() - t0
    rprint(f'[green][bold]✔️[/bold] Done![/green]\t time: {dt:.2f} s.')


@app.command('cv', help='Cross-validate model hyperparameters.')
def _cv(name: str = Argument(..., help='Project name.'),
        n_splits: int = Option(4, help='Number of CV splits.'),
        n_repeats: int = Option(1, help='Number of CV repeats.'),
        clustering: List[Clustering] = Option([Clustering.none], '--clustering', '-n',
                                              help='Clustering method.'),
        n_clusters: List[int] = Option([200], '--n-clusters', '-n',
                                       help='Number of clusters if [orange]clustering[/orange] is not [orange]none[/orange].'),
        tau: list[float] = Option([1.0], '--tau', '-t',
                                  help='Tau parameter that controls overfitting. The higher it is, the more variance will be explained by the model.'),
        motif_variances: List[bool] = Option([False], '--motif-variance', '-e',
                                             help='Estimate individual motif variances. Takes plenty of time.'),
        regul: List[Regularization] = Option([Regularization.none], '--regul', '-r',
                                             help='Regularization for motif variances estimates. Both regularizaiton types rely on the'
                                                 ' motif expression info.'),
        alpha: List[float] = Option([1.0], '--alpha', '-a', help='Regularization strength.'),
        n_jobs: int = Option(1, help='Number of jobs to be run at parallel, -1 will use all available threads. [red]Improves performance only if'
                            ' there is a plenty of cores as JAX uses multi-processing by deafult.[/red] Parallelization is done across groups.')):
    """
    Fit a a mixture model parameters to data for the given project.
    """
    t0 = time()
    rprint('Starting CV...')
    cross_validate(name, n_splits=n_splits, n_repeats=n_repeats,
                   regul=regul, alpha=alpha, estimate_motif_vars=motif_variances, tau=tau, clustering=clustering, n_clusters=n_clusters,
                    n_jobs=n_jobs, verbose=True)
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
    export_results(name, output_folder, std_mode=std_mode.value, alpha=alpha)
    p.stop()
    dt = time() - t0
    rprint(f'[green][bold]✔️[/bold] Done![/green]\t time: {dt:.2f} s.')


def main():
    check_packages()
    app()
