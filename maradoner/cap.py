# -*- coding: utf-8 -*-
"""Expression predicted from motif activities with an optional saturating curve on top of it."""
from dataclasses import dataclass, replace
import os
import numpy as np
from pandas import DataFrame as DF, read_csv
from scipy.optimize import minimize
from .fit import transform_data, motif_mean_matrix
from .grn import load_fit, load_predict, held_out_inds, filtered_motifs
from .line_gain import clean_activities
from .utils import read_init, InputError, warn, load_stage, dump_stage


@dataclass(frozen=True)
class SaturationCurves:
    """Per-sample curves y = level + ceiling * tanh(slope * (p - center) / ceiling), with p the
    prediction of the model. An infinite ceiling stands for no curve: y = p."""
    sample_names: list
    center: np.ndarray
    level: np.ndarray
    ceiling: np.ndarray
    slope: np.ndarray
    n_promoters: int
    n_folds: int

    def __call__(self, P: np.ndarray) -> np.ndarray:
        x = self.slope * (P - self.center)
        res = self.level + x
        capped = np.isfinite(self.ceiling)
        res[:, capped] = self.level[capped] + self.ceiling[capped] * np.tanh(x[:, capped] / self.ceiling[capped])
        return res


def fit_curve(p: np.ndarray, y: np.ndarray) -> tuple:
    """Least squares (center, level, ceiling, slope) of one sample. There is no curve if the
    prediction does not grow with the expression: the slope would go to zero."""
    center, mean, sd = p.mean(), y.mean(), y.std()
    x = p - center
    cross = x @ (y - mean)
    if sd == 0 or cross <= 0:
        return 0.0, 0.0, np.inf, 1.0
    slope = cross / (x @ x)
    # Level and ceiling are searched in units of the standard deviation of the expression.
    x, z = x / sd, (y - mean) / sd

    def loss(theta):
        level, ceiling, slope = theta[0], np.exp(theta[1]), np.exp(theta[2])
        u = slope * x / ceiling
        t = np.tanh(u)
        e = level + ceiling * t - z
        d = ceiling * u * (1 - t ** 2)
        return (e ** 2).mean(), 2 * np.array([e.mean(), (e * (ceiling * t - d)).mean(), (e * d).mean()])

    bounds = [(None, None), (np.log(1e-3), np.log(1e3)), (-7.0, 7.0)]
    start = np.array([0.0, np.log(3.0), np.clip(np.log(slope), -7.0, 7.0)])
    theta = minimize(loss, start, jac=True, method='L-BFGS-B', bounds=bounds, options=dict(maxiter=1000)).x
    return center, mean + sd * theta[0], sd * np.exp(theta[1]), np.exp(theta[2])


def _load_project(project: str) -> tuple:
    data = read_init(project)
    data, fit = load_fit(project, data)
    if getattr(fit, 'clustering', None) is not None:
        raise InputError(f'The project "{project}" was fitted with --clustering, and its expression cannot be '
                         'predicted: the activities belong to clusters of motifs.')
    return data, fit, load_predict(project, data, fit)


def _predicted(data, fit, activities, rows=None) -> np.ndarray:
    """mu_s + B M X + B U. The promoter means are left out, as a new promoter has none."""
    data = transform_data(data, helmert=False, loading_context=fit.loading_context, drist=fit.drist,
                          loading_multipliers=fit.loading_multipliers)
    B = np.asarray(data.B) if rows is None else np.asarray(data.B)[rows]
    return (np.asarray(fit.sample_mean.mean).reshape(1, -1) + B @ motif_mean_matrix(fit.motif_mean, data.X)
            + np.delete(B, filtered_motifs(activities), axis=1) @ np.asarray(activities.U_raw))


def _held_out_predictions(folds: list, data) -> tuple:
    """Predictions of the fold projects for the promoters they held out, on the rows of data.
    Returns their sum and the number of folds that held out each promoter."""
    index = {str(name): i for i, name in enumerate(data.promoter_names)}
    Y = np.asarray(data.Y)
    total, counts = np.zeros(Y.shape), np.zeros(len(Y), dtype=int)
    for fold in folds:
        fold_data, fold_fit, activities = _load_project(fold)
        if list(fold_data.sample_names) != list(data.sample_names):
            raise InputError(f'The fold project "{fold}" has other samples than the main project.')
        held = held_out_inds(fold_fit)
        if not held:
            raise InputError(f'The fold project "{fold}" holds out no promoters. Fit it with '
                             '--test-promoters-filename or --test-chromosomes.')
        rows = [index.get(str(fold_data.promoter_names[i]), -1) for i in held]
        if min(rows) < 0:
            raise InputError(f'The fold project "{fold}" holds out promoters that the main project does not have.')
        if not np.allclose(np.asarray(fold_data.Y)[held], Y[rows]):
            raise InputError(f'The expression of the fold project "{fold}" differs from that of the main project.')
        np.add.at(total, rows, _predicted(fold_data, fold_fit, activities, held))
        np.add.at(counts, rows, 1)
    return total, counts


def _fov(P: np.ndarray, Y: np.ndarray) -> float:
    return 1 - ((Y - P) ** 2).sum() / ((Y - Y.mean(axis=0)) ** 2).sum()


def fit_cap(project: str, folds: list = None) -> tuple:
    """Fit one saturating curve per sample on the training promoters of the project: on the
    predictions of the fold projects if they are given, on those of the project itself if not.
    Motif activities stay as they are. Returns the curves and the FOVs without and with them."""
    data, fit, activities = _load_project(project)
    train = np.ones(len(data.Y), dtype=bool)
    train[held_out_inds(fit)] = False
    if folds:
        P, counts = _held_out_predictions(folds, data)
        missing = int((train & (counts == 0)).sum())
        if missing:
            warn(f'{missing} of the {int(train.sum())} training promoters are held out by no fold project and take '
                 'no part in the curves.')
        train &= counts > 0
        P = P[train] / counts[train].reshape(-1, 1)
    else:
        P = _predicted(data, fit, activities, np.flatnonzero(train))
    Y = np.asarray(data.Y)[train]
    if len(Y) < 10:
        raise InputError(f'{len(Y)} promoters are too few for the curves.')
    params = np.array([fit_curve(P[:, j], Y[:, j]) for j in range(Y.shape[1])])
    curves = SaturationCurves(sample_names=list(data.sample_names), center=params[:, 0], level=params[:, 1],
                              ceiling=params[:, 2], slope=params[:, 3], n_promoters=len(Y), n_folds=len(folds or ()))
    plain = [name for name, c in zip(data.sample_names, curves.ceiling) if not np.isfinite(c)]
    if plain:
        warn(f'No curve for {len(plain)} sample(s), as the predictions do not grow with the expression: {plain[:5]}.')
    dump_stage(curves, project, 'cap', data.fmt)
    return curves, _fov(P, Y), _fov(curves(P), Y), P.std(axis=0)


def _read_loadings(path: str, data):
    """A copy of the project with the promoters of a loading table."""
    table = read_csv(path, sep='\t', index_col=0)
    missing = [m for m in data.motif_names if m not in table.columns]
    if missing:
        raise InputError(f'The loading table "{path}" has no column for {len(missing)} motif(s) of the project, '
                         f'for example {missing[:5]}.')
    try:
        B = table[list(data.motif_names)].to_numpy(dtype=float)
    except ValueError:
        raise InputError(f'The loading table "{path}" holds values that are not numbers.') from None
    if not np.isfinite(B).all():
        raise InputError(f'The loading table "{path}" holds missing or infinite values.')
    return replace(data, B=B, Y=np.zeros((len(B), data.Y.shape[1])), L=None,
                   promoter_names=[str(x) for x in table.index])


def predict_expression(project: str, output: str, loadings: str = None, promoters: str = None,
                       cap: bool = False, line_gain: bool = False) -> tuple:
    """Write the predicted expression of a set of promoters: 'test', 'train', 'all' or, with
    loadings, the promoters of that table. By default, the held-out promoters if there are
    any, and all promoters if not. With line_gain, the part of the activities specific to
    each sample is multiplied by its k from the stage line_gain. Returns the shape of the table."""
    data, fit, activities = _load_project(project)
    if line_gain:
        activities = replace(activities, U_raw=clean_activities(project, data, activities))
    if loadings is not None:
        if promoters is not None:
            raise InputError('--promoters selects promoters of the project and cannot be combined with --loadings.')
        if fit.loading_context is not None:
            raise InputError('--loadings cannot be used with a fit that corrects loadings by their context.')
        data = _read_loadings(loadings, data)
        rows = None
    else:
        held = held_out_inds(fit)
        promoters = promoters or ('test' if held else 'all')
        if promoters == 'test' and not held:
            raise InputError(f'The project "{project}" holds out no promoters. Use --promoters train or all.')
        rows = dict(test=held, train=np.delete(np.arange(len(data.Y)), held), all=np.arange(len(data.Y)))[promoters]
    P = _predicted(data, fit, activities, rows)
    if cap:
        curves = load_stage(project, 'cap', data.fmt, required=False)
        if curves is None:
            raise InputError(f'The project "{project}" has no curves. Run "maradoner cap {project}" first.')
        P = curves(P)
    names = np.asarray(data.promoter_names) if rows is None else np.asarray(data.promoter_names)[rows]
    folder = os.path.dirname(os.fspath(output))
    if folder:
        os.makedirs(folder, exist_ok=True)
    DF(P, index=names, columns=data.sample_names).rename_axis('promoter').to_csv(output, sep='\t', float_format='%.6g')
    return P.shape
