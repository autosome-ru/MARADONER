from dataclasses import dataclass
from scipy.stats import Covariance
from scipy.linalg import cholesky
import scipy.stats as st
import numpy as np
import pandas as pd
import random
import json
import os


@dataclass(frozen=True)
class GeneratedData:
    Y: np.ndarray
    B: np.ndarray
    U: np.ndarray
    Sigma: np.ndarray
    sigmas: np.ndarray
    nu: np.ndarray
    mean_p: np.ndarray
    mean_s: np.ndarray
    mean_m: np.ndarray
    significant_motifs: np.ndarray
    group_inds: list

def generate_data(p: int, m: int, g: int, min_samples: int, max_samples: int,
                  sigma_rel=1e-1, non_signficant_motifs_fraction=0.25,
                  means=True, B_cor=0, U_cor=0, E_cor=0, motif_cor=0) -> GeneratedData:
    g_samples = [np.random.randint(min_samples, max_samples) for _ in range(g)]
    U_cor = motif_cor
    g_std = st.gamma.rvs(1, 1, size=g) 
    # g_std[:] = 1
    sigmas = g_std ** 2 * sigma_rel
    B = np.random.rand(p, m)
    if B_cor or motif_cor:
        c = np.zeros((m, m))
        c[:] = B_cor if B_cor else motif_cor
        np.fill_diagonal(c,1.0)
        B = st.multivariate_normal(cov=c).rvs(size=p)
        
    # B /= B.var()
    K = st.wishart.rvs(df=p, scale=np.identity(m))
    K = np.identity(len(K))
    if U_cor:
        K = np.zeros((m, m))
        K[:] = U_cor
        np.fill_diagonal(K, 1.0)
    # stds = K.diagonal() ** 0.5
    # stds = 1 / stds
    # K = np.clip(stds.reshape(-1, 1) * K * stds, -1, 1)
        # print(U_mult, U_mult.shape)
    
        # U_mult = np.abs(st.multivariate_normal(cov=c).rvs() + 1.5).reshape(-1, 1) / 2
    # U_mult[:] = 1.0 ###
    significant_motifs = np.ones(m, dtype=bool)
    if non_signficant_motifs_fraction:
        significant_motifs[np.random.choice(np.arange(m), size=int(m * non_signficant_motifs_fraction), replace=False)] = False
        if B_cor or motif_cor:
            nm = (~significant_motifs).sum()
            c = np.zeros((nm, nm))
            c[:] = B_cor if B_cor else motif_cor
            np.fill_diagonal(c,1.0)
            B[:, ~significant_motifs] = st.multivariate_normal(cov=c).rvs(size=(p))
    
    U_mult = np.random.rand(m, 1) * 1 + 0.05
    if motif_cor or B_cor:
        if motif_cor:
            c = np.zeros((m, m))
            c[:] = motif_cor
            np.fill_diagonal(c,1.0)
        else:
            c = B.T @ B
            # c = c + np.identity(len(c)) * 1e-6
            d = 1 / c.diagonal() ** 0.5
            c = d.reshape(-1, 1) * c * d
        mvn = st.multivariate_normal(cov=c)
        U_mult = mvn.rvs().flatten()
        # print(U_mult)
        U_mult = st.uniform.ppf(st.norm.cdf(U_mult)) * 1 + 0.05
        U_mult = U_mult.reshape(-1, 1)
    mean_p = st.norm.rvs(size=(p, 1))
    # mean_m = st.norm.rvs(size=())
    if not means:
        mean_p[:] = 0
    Us = list()
    Ys = list()
    means_g = list()
    inds = list()
    mean_motifs = st.norm.rvs(size=(m, 1))
    mean_m = B @ mean_motifs
    if E_cor:
        c = np.zeros((p, p))
        c[:] = E_cor
        np.fill_diagonal(c, 1)
        c = cholesky(c)
        c = Covariance.from_cholesky(c)
        mvn_e = st.multivariate_normal(cov=c)
    for i, (n_samples, std, sigma) in enumerate(zip(g_samples, g_std, sigmas)):
        sub_inds = np.empty(n_samples, dtype=int)
        sub_inds = list()
        mean_g = st.norm.rvs(size=(n_samples,))
        if not means:
            mean_g[:] = 0
        means_g.append(mean_g)
        for j in range(n_samples):
            m_g = mean_g[j]
            U = st.matrix_normal(rowcov=K, colcov=sigma * np.identity(1)).rvs()
            U[~significant_motifs] = 0
            Us.append(U)
            E = st.norm.rvs(loc=0, scale=std, size=(p, 1))
            if E_cor:
                E = mvn_e.rvs().reshape(-1, 1)
            Ys.append((E + mean_p + mean_m + m_g) + B @ (U_mult * Us[-1]))
            sub_inds.append(len(Ys) - 1)
        inds.append(sub_inds)
    Ys = np.concatenate(Ys, axis=1)
    Us = np.concatenate(Us, axis=1)
    means_g = np.array(means_g)
    U_mult[~significant_motifs] = 0
    res = GeneratedData(Y=Ys, B=B, Sigma=U_mult[..., 0] ** 2, sigmas=g_std ** 2, nu=sigmas, U = Us,
                        mean_p=mean_p, mean_s=means_g,
                        mean_m=mean_motifs,
                        significant_motifs=significant_motifs,
                        group_inds=list(map(np.array, inds)))
    return res

def generate_dataset(folder: str, p: int, m: int, g: int, min_samples: int, max_samples: int, 
                     non_signficant_motifs_fraction: float, sigma_rel: float,
                     means: bool, B_cor: float, U_cor: float, E_cor:float, 
                     motif_cor: float, seed: int):
    random.seed(seed)
    np.random.seed(seed)
    res = generate_data(p=p, m=m, g=g, min_samples=min_samples, max_samples=max_samples,
                        non_signficant_motifs_fraction=non_signficant_motifs_fraction, 
                        sigma_rel=sigma_rel, means=means, B_cor=B_cor, U_cor=U_cor, E_cor=E_cor,
                        motif_cor=motif_cor)
    inds = res.group_inds
    Ys = res.Y; B = res.B; Us = res.U; std_g = res.sigmas; Sigma = res.Sigma 
    nu = res.nu
    insignificant_inds = ~res.significant_motifs
    colnames = np.empty(shape=sum(map(len, inds)), dtype=object)
    sample_names = list()
    groups = dict()
    for i, inds in enumerate(inds):
        cols = [f'col_{i + 1}' for i in inds]
        groups[f'group_{i + 1}'] = cols
        colnames[inds] = cols
        sample_names.extend(cols)
    proms = [f'prom_{i}' for i in range(1, p + 1)]
    motifs = [f'motif_{i}' for i in range(1, m + 1)]
    for i in np.where(insignificant_inds)[0]:
        motifs[i] = f'inactive_{motifs[i]}'
    
    Y = pd.DataFrame(Ys, columns=colnames, index=proms)
    B = pd.DataFrame(B, index=proms, columns=motifs)
    U_gt = pd.DataFrame(Us, index=motifs, columns=colnames)
    g_gt = pd.DataFrame(std_g, index=groups, columns=['sigma'])
    os.makedirs(folder, exist_ok=1)
    expression_filename = os.path.join(folder, 'expression.tsv')
    loadings_filename = os.path.join(folder, 'loadings.tsv')
    groups_filename = os.path.join(folder, 'groups.json')
    U_gt_filename = os.path.join(folder, 'activities.tsv')
    g_gt_filename = os.path.join(folder, 'sigma.tsv')
    Sigma_filename = os.path.join(folder, 'Sigma.tsv')
    Y.to_csv(expression_filename, sep='\t')
    B.to_csv(loadings_filename, sep='\t')
    U_gt.to_csv(U_gt_filename, sep='\t')
    g_gt.to_csv(g_gt_filename, sep='\t')
    
    s = '\n'.join(f'{a}\t{b}' for a, b in zip(motifs, Sigma))
    s = 'motif\ttau\n' + s
    with open(Sigma_filename, 'w') as f:
        f.write(s)
    
    mean_p = pd.DataFrame(res.mean_p.flatten(), columns=['mean'], index=proms)
    mean_p.to_csv(os.path.join(folder, 'promoter_means.tsv'), sep='\t')
    mean_s = pd.DataFrame(res.mean_s.flatten(), columns=['mean'], index=sample_names)
    mean_s.to_csv(os.path.join(folder, 'sample_means.tsv'), sep='\t')
    mean_m = pd.DataFrame(res.mean_m.flatten(), columns=['mean'], index=motifs)
    mean_m.to_csv(os.path.join(folder, 'motif_means.tsv'), sep='\t')
    
    with open(groups_filename, 'w') as f:
        json.dump(groups, f)
    return res