from .utils import read_init, logger_print, openers
from collections import defaultdict
import numpy as np
import dill




def select_motifs_single(project_name: str, filename: str):
    data = read_init(project_name)
    fmt = data.fmt
    names = data.motif_names
    # postfixes = set(data.motif_postfixes)
    del data
    
    with openers[fmt](f'{project_name}.fit.{fmt}', 'rb') as f:
        motif_score = dill.load(f).motif_mean
        # Use the intercept (average activity) component mu_m = M[:, 0] and its
        # Fisher block fim[:m, :m].
        M = np.asarray(motif_score.mean)
        M = M[:, 0] if M.ndim > 1 else M
        m = len(M)
        std = np.linalg.pinv(motif_score.fim[:m, :m], hermitian=True).diagonal() ** 0.5
        motif_score = M / std ** 0.5
    
    
    
    try:
        with openers[fmt](f'{project_name}.predict.{fmt}', 'rb') as f:
            U = dill.load(f)
            filtered = U.filtered_motifs
            U = U.U.mean(axis=-1)
        inds = np.arange(0, len(motif_score))
        inds = np.delete(inds, filtered)
        motif_score[inds] += U
    except FileNotFoundError:
        logger_print('Warning: no motif activities prediction found, using only means')
    motif_score = np.abs(motif_score).flatten()
    
    names_base = defaultdict(list)
    postfixes_d = dict()
    for i, name in enumerate(names):
        name = name.split('_')
        base = '_'.join(name[:-1])
        names_base[base].append(i)
        postfixes_d[i] = name[-1]
    names = list()
    for name, vals in names_base.items():
        i = max(vals, key=lambda x: motif_score[x])
        names.append(name + '_' + postfixes_d[i])
    with open(filename, 'w') as f:
        f.write('\n'.join(names))
        
        
        
    
    
    
    