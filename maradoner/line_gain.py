# -*- coding: utf-8 -*-
"""Gains of the part of the activities specific to a group, for a library read with a defect.

A group of a library read with a defect can show its part of the activities, U_g = (U - M X)_g,
enlarged: multiplied by a gain g >= 1 (form 'gain') or with an independent part added to it (form
'added'). The fit of that library sees only the size of the result, so the defect is identified by a
prior tying the sizes of the clean group parts across groups: log nu_g ~ N(mu, s^2), with mu the median
and s the sd of the log sizes of the groups unlikely to be defective. A library read without the defect
is then predicted with U_g multiplied by

    k_g = 1 - p_g + p_g E[(nu_g / size_g)^t | nu_g <= size_g],

p_g the probability that group g was read with the defect, t = 1/2 for a gain (k = E[1/g]) and t = 1
for an added part. The size of a group part is its activity variance nu (size 'nu') or the variance of
B U_g over the training promoters (size 'part')."""
from dataclasses import dataclass
import hashlib
import numpy as np
from pandas import read_csv, to_numeric
from scipy.special import log_ndtr
from .fit import transform_data
from .grn import load_fit, load_predict, held_out_inds, filtered_motifs
from .utils import read_init, InputError, warn, load_stage, dump_stage

SIZES = ('nu', 'part')
FORMS = ('gain', 'added')


@dataclass(frozen=True)
class LineGains:
    """Multipliers k of the group parts of the activities for a library read without the defect. Per
    group: the defect probability p, the size of the group part and k; per sample: k of its group.
    typical and spread are exp(mu) and s of the prior; reference marks the groups it was fitted to."""
    group_names: list
    sample_names: list
    p: np.ndarray
    size: np.ndarray
    k: np.ndarray
    k_sample: np.ndarray
    reference: np.ndarray
    typical: float
    spread: float
    size_kind: str
    form: str
    threshold: float
    checksum: str


def _checksum(U_raw) -> str:
    return hashlib.sha1(np.ascontiguousarray(np.asarray(U_raw, dtype=np.float64)).tobytes()).hexdigest()


def defect_probabilities(path: str, group_names: list) -> np.ndarray:
    """Probability of the defect per group from a csv or tsv table with one row per replicate: the
    product of its column p_def over the rows of the group, named in a column 'group' or 'line'. A group
    absent from the table gets 0."""
    try:
        table = read_csv(path, sep=None, engine='python')
    except (OSError, ValueError) as e:
        raise InputError(f'The defect table "{path}" cannot be read: {e}') from None
    name = next((c for c in ('group', 'line') if c in table.columns), None)
    if name is None or 'p_def' not in table.columns:
        raise InputError(f'The defect table "{path}" needs a column "p_def" and a column "group" or "line".')
    p_def = to_numeric(table['p_def'], errors='coerce').to_numpy(dtype=float)
    if not np.all((p_def >= 0) & (p_def <= 1)):
        raise InputError(f'The column p_def of the defect table "{path}" holds values that are not probabilities.')
    p = table.assign(p_def=p_def).groupby(table[name].astype(str))['p_def'].prod()
    groups = [str(g) for g in group_names]
    unknown = sorted(set(p.index) - set(groups))
    if unknown:
        warn(f'{len(unknown)} name(s) of the defect table are not groups of the project and are skipped: {unknown[:5]}.')
    return np.array([float(p.get(g, 0.0)) for g in groups])


def defect_multiplier(log_size, mu: float, s: float, t: float) -> np.ndarray:
    """E[(nu / size)^t] for a defective group, log nu ~ N(mu, s^2) cut at log size, as the defect only
    enlarges the group part. Tends to 1 for a size far below the typical one."""
    log_size = np.asarray(log_size, dtype=float)
    if s <= 0:
        return np.minimum(1.0, np.exp(t * (mu - log_size)))
    z = (log_size - mu) / s
    return np.exp(t * (mu - log_size) + (t * s) ** 2 / 2 + log_ndtr(z - t * s) - log_ndtr(z))


def gains(size, p, form: str = 'gain', threshold: float = 0.5) -> tuple:
    """k per group from the sizes of the group parts and the defect probabilities p. The prior of the
    clean sizes is fitted to the groups with p below threshold. Returns k, the mask of those groups,
    exp(mu) and s."""
    if form not in FORMS:
        raise InputError(f'Unknown form of the defect "{form}": one of {", ".join(FORMS)}.')
    size, p = np.asarray(size, dtype=float), np.asarray(p, dtype=float)
    reference = p < threshold
    if reference.sum() < 3:
        raise InputError(f'{int(reference.sum())} group(s) have a defect probability below {threshold}, and at '
                         'least 3 are needed for the prior of the clean sizes.')
    if not np.all(np.isfinite(size) & (size > 0)):
        raise InputError('The part of the activities specific to a group is zero in some group: no gain can be '
                         'estimated for it.')
    logs = np.log(size)
    mu, s = float(np.median(logs[reference])), float(np.std(logs[reference], ddof=1))
    k = 1 - p + p * defect_multiplier(logs, mu, s, 0.5 if form == 'gain' else 1.0)
    return k, reference, float(np.exp(mu)), s


def _transform(data, fit):
    return transform_data(data, helmert=False, loading_context=fit.loading_context, drist=fit.drist,
                          loading_multipliers=fit.loading_multipliers)


def part_sizes(transformed, fit, activities, kind: str = 'nu') -> np.ndarray:
    """Size of the part of the activities specific to each group of the fit (in the order of
    fit.group_names): the activity variance nu of the group (kind 'nu'), or the variance of B U_raw over
    the training promoters, averaged over the samples of the group (kind 'part')."""
    group_of = np.asarray(transformed.group_inds_inv)
    if kind == 'nu':
        v = np.asarray(fit.motif_variance.group, dtype=float).reshape(-1)[group_of]
        lowrank = getattr(fit.motif_variance, 'lowrank', None)
        if lowrank is not None and np.size(lowrank):
            v = v + (np.asarray(lowrank, dtype=float) ** 2).sum(axis=1)
    elif kind == 'part':
        B = np.asarray(transformed.B)
        B = np.delete(np.delete(B, held_out_inds(fit), axis=0), filtered_motifs(activities), axis=1)
        v = (B @ np.asarray(activities.U_raw, dtype=float)).var(axis=0)
    else:
        raise InputError(f'Unknown size of the group part "{kind}": one of {", ".join(SIZES)}.')
    return np.array([v[group_of == g].mean() for g in range(len(fit.group_names))])


def line_gain(project: str, defects: str, size: str = 'nu', form: str = 'gain', threshold: float = 0.5) -> LineGains:
    """Estimate k for the groups of a fitted and predicted project from a table of defect probabilities
    (see defect_probabilities) and store it as the stage line_gain of the project."""
    data = read_init(project)
    data, fit = load_fit(project, data)
    activities = load_predict(project, data, fit)
    transformed = _transform(data, fit)
    p = defect_probabilities(defects, fit.group_names)
    sizes = part_sizes(transformed, fit, activities, size)
    k, reference, typical, spread = gains(sizes, p, form, threshold)
    res = LineGains(group_names=[str(g) for g in fit.group_names], sample_names=[str(s) for s in data.sample_names],
                    p=p, size=sizes, k=k, k_sample=k[np.asarray(transformed.group_inds_inv)], reference=reference,
                    typical=typical, spread=spread, size_kind=size, form=form, threshold=float(threshold),
                    checksum=_checksum(activities.U_raw))
    dump_stage(res, project, 'line_gain', data.fmt)
    return res


def clean_activities(project: str, data, activities) -> np.ndarray:
    """U_raw of the project with the part of each sample multiplied by its k from the stage line_gain:
    the activities of a library read without the defect."""
    res = load_stage(project, 'line_gain', data.fmt)
    if res.checksum != _checksum(activities.U_raw) or list(res.sample_names) != [str(s) for s in data.sample_names]:
        raise InputError(f'The line gains of the project "{project}" were estimated for other activities. '
                         f'Re-run "maradoner line-gain {project}".')
    return np.asarray(activities.U_raw, dtype=float) * np.asarray(res.k_sample, dtype=float).reshape(1, -1)
