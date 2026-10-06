from .utils import read_init, logger_print, openers
from .utils import InputError, warn, subs_zeros
from .grn import load_fit, load_predict, filtered_motifs
from .export import motif_mean_covariance
from collections import defaultdict
import numpy as np
import dill




def select_motifs_single(project_name: str, filename: str):
    data = read_init(project_name)
    fmt = data.fmt
    # postfixes = set(data.motif_postfixes)
    postfixes = [str(p) for p in (data.motif_postfixes or list())]
    if not postfixes:
        raise InputError(f'The project "{project_name}" was created without "--loading-postfix", so its '
                         'motifs have no variants to choose from. "select-motifs" is meant for projects '
                         'made from several loading matrices, each with its own postfix '
                         '("maradoner create ... -p <postfix> -p <postfix>").')
    data, fit = load_fit(project_name, data)
    if getattr(fit, 'clustering', None) is not None:
        raise InputError(f'The project "{project_name}" was fitted with "--clustering": its model is '
                         'built on clusters of motifs, so motif variants cannot be compared. Re-run '
                         '"maradoner fit" without "--clustering" to select motifs.')
    names = data.motif_names
    activities = load_predict(project_name, data, fit, required=False)
    del data

    # z-score of the mean activity of a motif: the intercept component mu_m = M[:, 0]
    # over its standard error.
    M = np.asarray(fit.motif_mean.mean)
    M = M[:, 0] if M.ndim > 1 else M
    std = motif_mean_covariance(fit.motif_mean, names)[0][:, 0]
    motif_score = (M / std) ** 2



    if activities is not None:
        # z-scores of the deviations from that mean, one per group: the predicted
        # activity over its posterior standard deviation.
        z = np.asarray(activities.U) / np.array([subs_zeros(v ** 0.5) for v in activities.var()]).T
        inds = np.arange(0, len(motif_score))
        inds = np.delete(inds, filtered_motifs(activities))
        motif_score[inds] += (z ** 2).mean(axis=-1)
    else:
        warn('No predicted motif activities were found (run "maradoner predict" to use them): the '
             'motif variants are compared by their mean activities only.')
    # Both parts are on the z-scale: score = sqrt(z_mean^2 + average over groups of z_group^2).
    motif_score = motif_score ** 0.5

    names_base = defaultdict(list)
    for i, name in enumerate(names):
        # The variant of a motif is told by the postfix `create` appended to its name.
        postfix = max((p for p in postfixes if name.endswith('_' + p)), key=len, default=None)
        base = name if postfix is None else name[:-len(postfix) - 1]
        names_base[base].append(i)
    names = [names[max(vals, key=lambda x: motif_score[x])] for vals in names_base.values()]
    with open(filename, 'w') as f:
        f.write('\n'.join(names))







