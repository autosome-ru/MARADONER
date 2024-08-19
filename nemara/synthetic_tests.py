# -*- coding: utf-8 -*-
from collections import namedtuple
import scipy.stats as st
import numpy as np
import pandas as pd
import random
import json
import os

GenerationResult = namedtuple('GenerationResult', 'Y, B, U, U_mults, K, std_g, mean_p, mean_m, mean_g, group_inds, insignificant_inds')

def generate_data(p: int, m: int, g: int, min_samples: int, max_samples: int, g_std_a=0.5, g_std_b=0.5,
                  re_per_sample=False, sigma=1, U_mult_noise=0.1, noisy_cov=False, mean_motifs=True,
                  motif_variances=True, motif_variances_min=0.05, motif_variances_scale=1,
                  fraction_significant_motifs=1.0):
    g_samples = [np.random.randint(min_samples, max_samples) for _ in range(g)]
    g_std = st.gamma.rvs(1, 1, size=g)
    B = np.random.rand(p, m)
    if noisy_cov:
        K = st.wishart.rvs(df=p, scale=np.identity(m))
        stds = K.diagonal() ** 0.5
        stds = 1 / stds
        K = np.clip(stds.reshape(-1, 1) * K * stds, -1, 1)
    else:
        K = np.identity(m)
    U_exp = np.random.rand(g, m, 1) * motif_variances_scale + motif_variances_min
    # U_exp[:] = 1
    mean_p = st.norm.rvs(size=(p, 1))
    mean_g = st.norm.rvs(size=(g,))
    Us = list()
    Ys = list()
    inds = list()
    insignificant_motifs = np.ones(m, dtype=bool)
    insignificant_motifs[np.random.choice(np.arange(m), size=int(m * (1-fraction_significant_motifs)), replace=False)] = False
    
    if mean_motifs:
        mean_m = st.norm.rvs(size=(m, 1))
        mean_m[insignificant_motifs] = 0
    else:
        mean_m = np.zeros((m, 1), dtype=float)
    u_gen = st.matrix_normal(rowcov=K, colcov=sigma * np.identity(1), mean=mean_m)
    for i, (n_samples, m_g, std, U_mult) in enumerate(zip(g_samples, mean_g, g_std, U_exp)):
        sub_inds = np.empty(n_samples, dtype=int)
        sub_inds = list()
        for j in range(n_samples):
            U = u_gen.rvs()
            U[insignificant_motifs] = 0
            Us.append(U)
            Ys.append((st.norm.rvs(loc=0, scale=std, size=(p, 1)) + mean_p + m_g) + B @ (U_mult * Us[-1]))
            sub_inds.append(len(Ys) - 1)
        inds.append(sub_inds)
    Ys = np.concatenate(Ys, axis=1)
    Us = np.concatenate(Us, axis=1)
    res = GenerationResult(Y=Ys, B=B, U=Us, U_mults=U_exp[..., 0], std_g=g_std, mean_p=mean_p, mean_m=mean_m,
                           mean_g=mean_g, group_inds=inds, insignificant_inds=insignificant_motifs, K=K)
    return res


def generate_dataset(folder: str, p: int, m: int, g: int, min_samples: int, max_samples: int, 
                     fraction_significant_motifs: float, mean_motifs: bool, sigma: float, g_std_a: float,
                     g_std_b: float, motif_variances: float, motif_variances_min: float,
                     motif_variances_scale: float, seed: int):
    random.seed(seed)
    np.random.seed(seed)
    res = generate_data(p=p, m=m, g=g, min_samples=min_samples, max_samples=max_samples,
                        fraction_significant_motifs=fraction_significant_motifs, 
                        g_std_a=g_std_a, g_std_b=g_std_b, sigma=sigma, mean_motifs=mean_motifs,
                        motif_variances=motif_variances, motif_variances_min=motif_variances_min,
                        motif_variances_scale=motif_variances_scale)
    inds = res.group_inds
    Ys = res.Y; B = res.B; Us = res.U; std_g = res.std_g; U_exp = res.U_mults
    insignificant_inds = res.insignificant_inds
    colnames = np.empty(shape=sum(map(len, inds)), dtype=object)
    groups = dict()
    for i, inds in enumerate(inds):
        cols = [f'col_{i + 1}' for i in inds]
        groups[f'group_{i + 1}'] = cols
        colnames[inds] = cols
    proms = [f'prom_{i}' for i in range(1, p + 1)]
    motifs = [f'motif_{i}' for i in range(1, m + 1)]
    for i in np.where(insignificant_inds)[0]:
        motifs[i] = f'inactive_{motifs[i]}'
    
    Y = pd.DataFrame(Ys, columns=colnames, index=proms)
    B = pd.DataFrame(B, index=proms, columns=motifs)
    U_exp = pd.DataFrame(U_exp.T, columns=groups.keys(), index=motifs)
    U_gt = pd.DataFrame(Us, index=motifs, columns=colnames)
    g_gt = pd.DataFrame(std_g, index=groups, columns=['sigma_g'])
    os.makedirs(folder, exist_ok=1)
    expression_filename = os.path.join(folder, 'expression.tsv')
    loadings_filename = os.path.join(folder, 'loadings.tsv')
    motif_expression_filename = os.path.join(folder, 'motif_expression.tsv')
    groups_filename = os.path.join(folder, 'groups.json')
    U_gt_filename = os.path.join(folder, 'activities.tsv')
    g_gt_filename = os.path.join(folder, 'sigma_g.tsv')
    Y.to_csv(expression_filename, sep='\t')
    B.to_csv(loadings_filename, sep='\t')
    U_exp.to_csv(motif_expression_filename, sep='\t')
    U_gt.to_csv(U_gt_filename, sep='\t')
    g_gt.to_csv(g_gt_filename, sep='\t')
    
    mean_m = pd.DataFrame(res.mean_m.flatten(), columns=['mean'], index=motifs)
    mean_m.to_csv(os.path.join(folder, 'activity_means.tsv'), sep='\t')
    mean_p = pd.DataFrame(res.mean_p.flatten(), columns=['mean'], index=proms)
    mean_p.to_csv(os.path.join(folder, 'promoter_means.tsv'), sep='\t')
    mean_g = pd.DataFrame(res.mean_g.flatten(), columns=['mean'], index=groups)
    mean_g.to_csv(os.path.join(folder, 'group_means.tsv'), sep='\t')
    
    with open(groups_filename, 'w') as f:
        json.dump(groups, f)

