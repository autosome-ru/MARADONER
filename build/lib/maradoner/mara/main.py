# -*- coding: utf-8 -*-
from enum import Enum
from click import Context
from typer import Typer, Option, Argument
from typer.core import TyperGroup
from typing import List
from rich import print as rprint
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from pathlib import Path
from .fit import fit, ClusteringMode, calculate_fov, predict, GOFStat, TauMode, TauEstimation
from time import time
from .export import export_results


class OrderCommands(TyperGroup):
  def list_commands(self, ctx: Context):
    """Return list of commands in the order appear."""
    return list(self.commands)    # get commands using self.commands

app_old = Typer(rich_markup_mode='rich', cls=OrderCommands, add_completion=False)


@app_old.command('fit', help='Estimate variance parameters.')
def _fit(name: str = Argument(..., help='Project name.'),
         tau_mode: TauMode = Option(TauMode.mara, help='MARA or ISMARA model.'),
         tau_estimation: TauEstimation = Option(TauEstimation.fixed, help='Estimation method for tau'),
         tau_fix: float = Option(0.1, help='A value for tau to be fixed to given that [cyan]tau_estimation[/cyan] is [orange]fixed[/orange].'),
        clustering: ClusteringMode = Option(ClusteringMode.none, help='Clustering method.'),
        num_clusters: int = Option(200, help='Number of clusters if [orange]clustering[/orange] is not [orange]none[/orange].'),
        test_chromosomes: List[str] = Option(None, '--test-chromosomes', '-t', help='Test chromosomes'),
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
    fit(name, tau_mode=tau_mode, tau_estimation=tau_estimation, tau_fix=tau_fix, 
        clustering=clustering, num_clusters=num_clusters,
        gpu=gpu, test_chromosomes=test_chromosomes,
        gpu_decomposition=gpu_decomposition, x64=x64)
    p.stop()
    dt = time() - t0
    rprint(f'[green][bold]✔️[/bold] Done![/green]\t time: {dt:.2f} s.')

@app_old.command('gof', help='Estimate GOFs given test/train data split. Provides test info only if [orange]test-chromosomes[/orange] is not None in [cyan]fit[/cyan].')
def _gof(name: str = Argument(..., help='Project name.'),
         # use_groups: bool = Option(False, help='Compute statistic for sammples aggragated across groups.'), 
         keep_motifs: Path = Option(None, help='Table with 2 columns: motif and status'),
         stat_type: GOFStat = Option(GOFStat.fov, help='Statistic type to compute'),
         gpu: bool = Option(False, help='Use GPU if available for most of computations.'), 
         x64: bool = Option(True, help='Use high precision algebra.')):
    """
    Fit a a mixture model parameters to data for the given project.
    """

    t0 = time()
    p = Progress(SpinnerColumn(speed=0.5), TextColumn("[progress.description]{task.description}"), transient=True)
    p.add_task(description="Calculating FOVs...", total=None)
    p.start()
    res = calculate_fov(name, stat_type=stat_type, keep_motifs=keep_motifs, gpu=gpu, x64=x64)
    for name, res in res:
        if name:
            print(name)
        if stat_type == GOFStat.corr:
            title = 'Pearson correlation'
        else:
            title = 'Fraction of variance explained'
        if name:
            title = f'({name}) {title}'
        t = Table('Set', 'stat',
                  title=title)
        row = [f'{t.total:.6f}' for t in res.train]
        t.add_row('train', *row)
        if res.test is not None:
            row = [f'{t.total:.6f}' for t in res.test]
            t.add_row('test', *row)
        rprint(t)
    p.stop()
    dt = time() - t0
    rprint(f'[green][bold]✔️[/bold] Done![/green]\t time: {dt:.2f} s.')

@app_old.command('predict', help='Estimate deviations of motif activities from their means.')
def _predict(name: str = Argument(..., help='Project name.'),
             gpu: bool = Option(False, help='Use GPU if available for most of computations.'), 
             x64: bool = Option(True, help='Use high precision algebra.')):
    """
    Fit a a mixture model parameters to data for the given project.
    """

    t0 = time()
    p = Progress(SpinnerColumn(speed=0.5), TextColumn("[progress.description]{task.description}"), transient=True)
    p.add_task(description="Predicting motif activities...", total=None)
    p.start()
    predict(name, gpu=gpu, x64=x64)
    p.stop()
    dt = time() - t0
    rprint(f'[green][bold]✔️[/bold] Done![/green]\t time: {dt:.2f} s.')


@app_old.command('export', help='Extract motif activities, parameter estimates.')
def _export(name: str = Argument(..., help='Project name.'),
            output_folder: Path = Argument(..., help='Output folder.')):
    t0 = time()
    p = Progress(SpinnerColumn(speed=0.5), TextColumn("[progress.description]{task.description}"), transient=True)
    p.add_task(description="Exporting results...", total=None)
    p.start()
    export_results(name, output_folder)
    p.stop()
    dt = time() - t0
    rprint(f'[green][bold]✔️[/bold] Done![/green]\t time: {dt:.2f} s.')
