from dataclasses import dataclass
from scipy.stats import Covariance
from scipy.linalg import cholesky
import scipy.stats as st
import numpy as np
import pandas as pd
import random
import json
import os
from .utils import InputError


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
    M: np.ndarray = None       # (m, c) motif-mean matrix, U ~ MN(M X, Sigma, G)
    X: np.ndarray = None       # (c, n) covariate design (first row = intercept ones)
    W_err: np.ndarray = None   # (s, k) ground-truth low-rank error factor, or None
    V_act: np.ndarray = None   # (s, k) ground-truth low-rank activity factor, or None
    F: np.ndarray = None       # (m, f) motif families, or None
    omega: np.ndarray = None   # (f,) variances of the families, or None
    L_motif: np.ndarray = None # (f, k) covariance factor between families (motifs if F is None)


def generate_data_cov(p: int, m: int, g: int, min_samples: int, max_samples: int,
                      sigma_rel: float = 1e-1, non_signficant_motifs_fraction: float = 0.20,
                      means: bool = True, n_covariates: int = 0, cov_effect: float = 1.0,
                      error_lowrank: int = 0, error_lowrank_scale: float = 1.0,
                      activity_lowrank: int = 0, activity_lowrank_scale: float = 1.0,
                      families: int = 0, family_scale: float = 1.0,
                      motif_lowrank: int = 0, motif_lowrank_scale: float = 1.0,
                      windows: list = None):
    """Covariate-aware synthetic data generator for the model

        Y = 1_p mu_s^T + mu_p 1_s^T + B U + E,
        U ~ MN(M X, Sigma, G),  E ~ MN(0, I_p, D + W W^T).

    X is (c, n) with c = 1 + n_covariates (first row is the intercept). M is (m, c):
    column 0 is the per-motif average activity mu_m, the rest are covariate effects.
    Returns a GeneratedData with the ground-truth M and X attached. Handles variable
    group sizes (fixes the ragged-array issue of the legacy generator).

    error_lowrank > 0 injects a rank-k correction W W^T (W is s x k) into the error
    column-covariance, i.e. correlated noise shared across samples (think batch/technical
    effects). W is scaled by error_lowrank_scale * mean(sqrt(D)) per column and projected
    off 1_s (the identifiable part). The ground-truth W is returned for evaluation.

    families > 0 splits the active motifs at random among that many families, and the
    covariance between motifs becomes Sigma + F (diag(omega) + L L^T) F^T, with omega of the
    order family_scale * mean(tau) and L (families x motif_lowrank) of the order
    motif_lowrank_scale * sqrt(mean(tau)). Without families, motif_lowrank > 0 gives
    Sigma + L L^T with L (m x motif_lowrank) zero for the inactive motifs.

    windows (a list of K names) gives each of the m motifs K columns of loadings, one per window
    around the TSS, with a correlation of 0.5 between the windows of a motif; the data then have
    m K columns, motif i holding columns i K, ..., i K + K - 1. The windows of a motif form a
    family. An active motif acts through all its windows together (a family variance), through
    one window (the tau of that window) or both, at random; the variances are those of an active
    motif above, the family variance times family_scale. families > 0 then splits the active
    motifs among that many families of a second level, each holding all windows of its motifs.
    F lists the families of windows first (one per motif, inactive motifs included).
    """
    for name, value in (('promoters (--p)', p), ('motifs (--m)', m), ('groups (--g)', g)):
        if value < 1:
            raise InputError(f'The number of {name} must be at least 1, but {value} was given.')
    if not 1 <= min_samples <= max_samples:
        raise InputError('The numbers of samples per group must satisfy 1 <= --min-samples <= '
                         f'--max-samples, but --min-samples {min_samples} and --max-samples {max_samples} '
                         'were given.')
    if not 0 <= non_signficant_motifs_fraction <= 1:
        raise InputError('The fraction of non-significant motifs must lie between 0 and 1, but '
                         f'{non_signficant_motifs_fraction:g} was given.')
    if not sigma_rel > 0:
        raise InputError(f'--sigma-rel is a ratio of variances and must be positive, but {sigma_rel:g} was given.')
    for name, value in (('--n-covariates', n_covariates), ('--error-lowrank', error_lowrank),
                        ('--activity-lowrank', activity_lowrank),
                        ('--error-lowrank-scale', error_lowrank_scale),
                        ('--activity-lowrank-scale', activity_lowrank_scale),
                        ('--families', families), ('--family-scale', family_scale),
                        ('--motif-lowrank', motif_lowrank), ('--motif-lowrank-scale', motif_lowrank_scale)):
        if value < 0:
            raise InputError(f'{name} cannot be negative, but {value:g} was given.')
    group_sizes = [np.random.randint(min_samples, max_samples + 1) for _ in range(g)]
    n = int(sum(group_sizes))
    for name, value in (('--n-covariates', n_covariates + 1), ('--error-lowrank', error_lowrank),
                        ('--activity-lowrank', activity_lowrank)):
        if value >= n:
            raise InputError(f'{name} is too large for the {n} samples that were generated: the covariates '
                             '(plus the intercept) and the ranks of the low-rank corrections must each be '
                             'smaller than the number of samples. Use more groups or samples per group.')
    group_inds = []
    t = 0
    for gs in group_sizes:
        group_inds.append(np.arange(t, t + gs))
        t += gs
    group_inds_inv = np.zeros(n, dtype=int)
    for gi, inds in enumerate(group_inds):
        group_inds_inv[inds] = gi

    # Group variances: D (error) and nu (motif activity multiplier); nu/D == sigma_rel.
    g_std = st.gamma.rvs(1, 1, size=g)
    D_group = g_std ** 2                      # error variance per group
    nu_group = g_std ** 2 * sigma_rel         # G multiplier per group

    if windows:
        windows = list(windows)
        if len(set(windows)) != len(windows) or not all(windows):
            raise InputError(f'The names of the windows must be distinct and non-empty, but {windows} were given.')
        K = len(windows)
        B = 0.5 * np.repeat(np.random.rand(p, m), K, axis=1) + 0.5 * np.random.rand(p, m * K)
    else:
        B = np.random.rand(p, m)

    # Motif variances Sigma (tau); a fraction of motifs are inactive (Sigma = 0).
    U_mult = np.random.rand(m) * 1.0 + 0.05
    significant = np.ones(m, dtype=bool)
    if non_signficant_motifs_fraction:
        idx = np.random.choice(np.arange(m), size=int(m * non_signficant_motifs_fraction),
                               replace=False)
        significant[idx] = False
    U_mult[~significant] = 0.0
    Sigma = U_mult ** 2

    # Families of the active motifs and the covariance between motifs S.
    F = omega = L_motif = None
    S_motif = None
    if windows:
        # Per active motif: 0 all windows together, 1 one window, 2 both.
        mean_tau = Sigma[significant].mean() if significant.any() else 1.0
        pattern = np.random.randint(0, 3, size=m)
        window = np.random.randint(0, K, size=m)
        tau = np.zeros(m * K)
        omega_w = np.where(significant & (pattern != 1), Sigma * family_scale, 0.0)
        one = significant & (pattern != 0)
        tau[np.arange(m)[one] * K + window[one]] = Sigma[one]
        F = np.kron(np.eye(m), np.ones((K, 1)))
        omega = omega_w
        if families:
            F_c = np.zeros((m, int(families)))
            active = np.where(significant)[0]
            F_c[active, np.random.randint(0, int(families), size=len(active))] = 1.0
            F = np.hstack([F, np.repeat(F_c, K, axis=0)])
            omega = np.concatenate([omega_w, (np.random.rand(int(families)) + 0.5) * family_scale * mean_tau])
        if motif_lowrank and motif_lowrank >= F.shape[1]:
            raise InputError('--motif-lowrank must be smaller than the number of families.')
        Psi = np.diag(omega)
        if motif_lowrank:
            L_motif = st.norm.rvs(size=(F.shape[1], int(motif_lowrank))) * motif_lowrank_scale * mean_tau ** 0.5
            L_motif[:m][~significant] = 0.0
            Psi += L_motif @ L_motif.T
        m = m * K
        significant = np.repeat(significant, K)
        Sigma = tau
        S_motif = np.diag(Sigma) + F @ Psi @ F.T
        w, v = np.linalg.eigh(S_motif)
        S_half = (v * np.sqrt(np.clip(w, 0.0, None))) @ v.T
    elif families or motif_lowrank:
        if motif_lowrank and motif_lowrank >= (families or m):
            raise InputError('--motif-lowrank must be smaller than the number of families (of motifs '
                             'without --families).')
        mean_tau = Sigma[significant].mean() if significant.any() else 1.0
        n_fam = int(families) if families else m
        Psi = np.zeros((n_fam, n_fam))
        if families:
            active = np.where(significant)[0]
            F = np.zeros((m, n_fam))
            F[active, np.random.randint(0, n_fam, size=len(active))] = 1.0
            omega = (np.random.rand(n_fam) + 0.5) * family_scale * mean_tau
            Psi += np.diag(omega)
        if motif_lowrank:
            L_motif = st.norm.rvs(size=(n_fam, int(motif_lowrank))) * motif_lowrank_scale * mean_tau ** 0.5
            if not families:
                L_motif[~significant] = 0.0
            Psi += L_motif @ L_motif.T
        S_motif = np.diag(Sigma) + (Psi if F is None else F @ Psi @ F.T)
        w, v = np.linalg.eigh(S_motif)
        S_half = (v * np.sqrt(np.clip(w, 0.0, None))) @ v.T

    # Intercept (average) motif activities mu_m and covariate-effect columns -> M.
    c = 1 + max(0, int(n_covariates))
    M = np.zeros((m, c), dtype=float)
    M[:, 0] = st.norm.rvs(size=m)             # mu_m
    if c > 1:
        M[:, 1:] = st.norm.rvs(size=(m, c - 1)) * cov_effect
    M[~significant, :] = 0.0

    # Covariate design X: intercept + standardized continuous covariates.
    X = np.ones((c, n), dtype=float)
    if c > 1:
        Xc = st.norm.rvs(size=(c - 1, n))
        Xc = (Xc - Xc.mean(axis=1, keepdims=True)) / Xc.std(axis=1, keepdims=True)
        X[1:] = Xc

    mean_p = st.norm.rvs(size=p) if means else np.zeros(p)
    mean_s = st.norm.rvs(size=n) if means else np.zeros(n)

    # Sample-wise motif activities U = M X + noise (column cov G, row cov Sigma).
    # G = diag(nu_n) by default; activity_lowrank injects G = diag(nu_n) + V_true V_true^T.
    nu_n = nu_group[group_inds_inv]
    V_act = None
    Nz = st.norm.rvs(size=(m, n))
    if activity_lowrank and activity_lowrank > 0:
        ka = int(activity_lowrank)
        scale = activity_lowrank_scale * np.sqrt(nu_n.mean())
        V_act = st.norm.rvs(size=(n, ka)) * scale
        V_act = V_act - V_act.mean(axis=0, keepdims=True)
        G_full = np.diag(nu_n) + V_act @ V_act.T
        wv, Qv = np.linalg.eigh(G_full)
        G_half = (Qv * np.sqrt(np.clip(wv, 0, None))) @ Qv.T
        U0 = np.sqrt(Sigma).reshape(-1, 1) * (Nz @ G_half) if S_motif is None else S_half @ (Nz @ G_half)
    elif S_motif is None:
        U0 = (np.sqrt(Sigma).reshape(-1, 1) * Nz) * np.sqrt(nu_n).reshape(1, -1)
    else:
        U0 = (S_half @ Nz) * np.sqrt(nu_n).reshape(1, -1)
    U = M @ X + U0
    U[~significant, :] = 0.0

    D_n = D_group[group_inds_inv]
    E = st.norm.rvs(size=(p, n)) * np.sqrt(D_n).reshape(1, -1)
    # Optional low-rank (correlated) error component:  E += C W^T,  C ~ N(0, I) (p x k).
    W_err = None
    if error_lowrank and error_lowrank > 0:
        k = int(error_lowrank)
        scale = error_lowrank_scale * np.mean(np.sqrt(D_n))
        W_err = st.norm.rvs(size=(n, k)) * scale
        W_err = W_err - W_err.mean(axis=0, keepdims=True)        # identifiable part (off 1_s)
        E = E + st.norm.rvs(size=(p, k)) @ W_err.T

    Y = mean_p.reshape(-1, 1) + mean_s.reshape(1, -1) + B @ U + E

    return GeneratedData(Y=Y, B=B, U=U, Sigma=Sigma, sigmas=D_group, nu=nu_group,
                         mean_p=mean_p.reshape(-1, 1), mean_s=mean_s, mean_m=M[:, 0],
                         significant_motifs=significant, group_inds=group_inds, M=M, X=X,
                         W_err=W_err, V_act=V_act, F=F, omega=omega, L_motif=L_motif)

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
                     means: bool, B_cor: float = 0, U_cor: float = 0, E_cor: float = 0,
                     motif_cor: float = 0, seed: int = 1,
                     n_covariates: int = 0, cov_effect: float = 1.0,
                     error_lowrank: int = 0, error_lowrank_scale: float = 1.0,
                     activity_lowrank: int = 0, activity_lowrank_scale: float = 1.0,
                     families: int = 0, family_scale: float = 1.0,
                     motif_lowrank: int = 0, motif_lowrank_scale: float = 1.0,
                     windows: list = None):
    random.seed(seed)
    if not 0 <= seed < 2 ** 32:
        raise InputError(f'--seed must lie between 0 and {2 ** 32 - 1}, but {seed} was given.')
    if p * max(m, g * max_samples) > 2 ** 31:
        raise InputError(f'The requested dataset is too large to generate ({p} promoters x {m} motifs, '
                         f'up to {g * max_samples} samples).')
    np.random.seed(seed)
    res = generate_data_cov(p=p, m=m, g=g, min_samples=min_samples, max_samples=max_samples,
                            non_signficant_motifs_fraction=non_signficant_motifs_fraction,
                            sigma_rel=sigma_rel, means=means,
                            n_covariates=n_covariates, cov_effect=cov_effect,
                            error_lowrank=error_lowrank, error_lowrank_scale=error_lowrank_scale,
                            activity_lowrank=activity_lowrank,
                            activity_lowrank_scale=activity_lowrank_scale,
                            families=families, family_scale=family_scale,
                            motif_lowrank=motif_lowrank, motif_lowrank_scale=motif_lowrank_scale,
                            windows=windows)
    inds = res.group_inds
    Ys = res.Y; B = res.B; Us = res.U; std_g = res.sigmas; Sigma = res.Sigma
    insignificant_inds = ~res.significant_motifs
    colnames = np.empty(shape=sum(map(len, inds)), dtype=object)
    sample_names = list()
    groups = dict()
    for i, sub in enumerate(inds):
        cols = [f'col_{j + 1}' for j in sub]
        groups[f'group_{i + 1}'] = cols
        colnames[sub] = cols
        sample_names.extend(cols)
    proms = [f'prom_{i}' for i in range(1, p + 1)]
    if windows:
        bases = [f'motif_{i}' for i in range(1, m + 1)]
        bases = [f'inactive_{b}' if insignificant_inds[i * len(windows)] else b for i, b in enumerate(bases)]
        motifs = [f'{b}_{w}' for b in bases for w in windows]
    else:
        motifs = [f'motif_{i}' for i in range(1, m + 1)]
        for i in np.where(insignificant_inds)[0]:
            motifs[i] = f'inactive_{motifs[i]}'

    Y = pd.DataFrame(Ys, columns=colnames, index=proms)
    B = pd.DataFrame(B, index=proms, columns=motifs)
    U_gt = pd.DataFrame(Us, index=motifs, columns=colnames)
    g_gt = pd.DataFrame(std_g, index=list(groups), columns=['sigma'])
    os.makedirs(folder, exist_ok=1)
    Y.to_csv(os.path.join(folder, 'expression.tsv'), sep='\t')
    B.to_csv(os.path.join(folder, 'loadings.tsv'), sep='\t')
    U_gt.to_csv(os.path.join(folder, 'activities.tsv'), sep='\t')
    g_gt.to_csv(os.path.join(folder, 'sigma.tsv'), sep='\t')
    # Group multipliers nu of the activity variances: Var(U_kn) = tau_k * nu_g. A fit fixes the
    # scale of nu by pinning it for one group, so only nu_g * tau_k and the ratios of nu are
    # comparable with this table and Sigma.tsv.
    pd.DataFrame(res.nu, index=list(groups), columns=['nu']).to_csv(os.path.join(folder, 'nu.tsv'), sep='\t')

    s = 'motif\ttau\n' + '\n'.join(f'{a}\t{b}' for a, b in zip(motifs, Sigma))
    with open(os.path.join(folder, 'Sigma.tsv'), 'w') as f:
        f.write(s)

    pd.DataFrame(res.mean_p.flatten(), columns=['mean'], index=proms).to_csv(
        os.path.join(folder, 'promoter_means.tsv'), sep='\t')
    pd.DataFrame(np.asarray(res.mean_s).flatten(), columns=['mean'], index=sample_names).to_csv(
        os.path.join(folder, 'sample_means.tsv'), sep='\t')
    pd.DataFrame(res.mean_m.flatten(), columns=['mean'], index=motifs).to_csv(
        os.path.join(folder, 'motif_means.tsv'), sep='\t')

    # Covariate ground truth and the design file consumable by `create --covariates`.
    if res.X is not None and res.X.shape[0] > 1:
        cov_names = [f'cov_{i}' for i in range(1, res.X.shape[0])]
        # covariates.tsv: rows = samples, columns = covariates (raw, pre-standardization).
        cov_df = pd.DataFrame(res.X[1:].T, index=[colnames[j] for j in range(len(colnames))],
                              columns=cov_names)
        cov_df.index.name = 'sample_id'
        cov_df.to_csv(os.path.join(folder, 'covariates.tsv'), sep='\t')
        # Ground-truth M (motif x [intercept, covariates]).
        pd.DataFrame(res.M, index=motifs, columns=['intercept'] + cov_names).to_csv(
            os.path.join(folder, 'M_gt.tsv'), sep='\t')

    # Ground-truth low-rank error factor (s x k), for evaluation of --error-lowrank.
    if res.W_err is not None:
        pd.DataFrame(res.W_err, index=sample_names,
                     columns=[f'errfac_{i+1}' for i in range(res.W_err.shape[1])]).to_csv(
            os.path.join(folder, 'W_err_gt.tsv'), sep='\t')
    # Ground-truth low-rank activity factor (s x k), for evaluation of --activity-lowrank.
    if res.V_act is not None:
        pd.DataFrame(res.V_act, index=sample_names,
                     columns=[f'actfac_{i+1}' for i in range(res.V_act.shape[1])]).to_csv(
            os.path.join(folder, 'V_act_gt.tsv'), sep='\t')

    # Motif families (a valid file for fit --motif-families) and the ground truth of the
    # covariance between motifs.
    if res.F is not None:
        fam_names = [f'family_{j + 1}' for j in range(res.F.shape[1])]
        if windows:
            # A family of windows is named after its motif; the families of the second level follow.
            # The class of a motif in windows_gt.tsv refers to its windows, beyond the second level.
            fam_names = bases + [f'family_{j + 1}' for j in range(res.F.shape[1] - m)]
            K = len(windows)
            cluster = res.F[::K, m:]
            omega_w = res.omega[:m]
            tau = res.Sigma.reshape(m, K)
            classes = {(True, True): 'family+individual', (True, False): 'family',
                       (False, True): 'individual', (False, False): 'none'}
            kind = [classes[o > 0, bool(t.any())] for o, t in zip(omega_w, tau)]
            pd.DataFrame({'class': kind, 'omega': omega_w,
                          'individual': [','.join(f'{b}_{w}' for w, t_ in zip(windows, t) if t_ > 0)
                                         for b, t in zip(bases, tau)],
                          'family': [fam_names[m + int(np.argmax(c))] if c.any() else '' for c in cluster]},
                         index=pd.Index(bases, name='motif')).to_csv(os.path.join(folder, 'windows_gt.tsv'), sep='\t')
        lines = [' '.join([fam] + [motifs[i] for i in np.where(res.F[:, j])[0]])
                 for j, fam in enumerate(fam_names) if res.F[:, j].any()]
        with open(os.path.join(folder, 'families.txt'), 'w') as f:
            f.write('\n'.join(lines) + '\n')
        pd.DataFrame({'omega': res.omega, 'motifs': res.F.sum(axis=0).astype(int)}, index=fam_names).to_csv(
            os.path.join(folder, 'family_variances_gt.tsv'), sep='\t')
    if res.L_motif is not None:
        rows = [f'family_{j + 1}' for j in range(len(res.L_motif))] if res.F is not None else motifs
        pd.DataFrame(res.L_motif, index=rows,
                     columns=[f'factor_{i + 1}' for i in range(res.L_motif.shape[1])]).to_csv(
            os.path.join(folder, 'L_motif_gt.tsv'), sep='\t')

    with open(os.path.join(folder, 'groups.json'), 'w') as f:
        json.dump(groups, f)
    return res