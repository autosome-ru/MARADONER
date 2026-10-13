# -*- coding: utf-8 -*-
"""Choice of the number of factors of "fit --activity-lowrank": the covariance of
activities between samples is G = diag(nu) + V V^T with V of k columns, and k is
increased while a new column lowers the REML deviance more than it does on data
simulated from the model without that column."""
# The smallest number of simulated datasets from which the spread of their gains is estimated.
MIN_NULL = 3
import numpy as np
import jax
import jax.numpy as jnp
from scipy.stats import t as t_dist
from .utils import read_init, InputError, logger_print, warn
from .fit import transform_data, split_data, align_fit_to_promoters, covariate_projection, \
                 _error_lowrank_W, _activity_lowrank_V, NU_REFERENCE_FRACTION
from .linalg import lowrank_decomposition
from . import lowrank as _lowrank
from . import lowrank_g as _lrg


def _sym_root(A: np.ndarray) -> np.ndarray:
    """A root R of a symmetric non-negative definite matrix: R R^T = A."""
    w, v = np.linalg.eigh(0.5 * (A + A.T))
    return v * np.sqrt(np.clip(w, 0.0, None))

def _new_column(Z, P, nu_n, V):
    """A starting point for one more column of V: the leading direction of what the
    current G = diag(nu_n) + V V^T leaves unexplained in the sample covariance of Z.
    The recipe of lowrank_g._init_V."""
    s = len(nu_n)
    S = Z.T @ Z
    G = np.diag(nu_n) + V @ V.T
    S = S / max(np.trace(S), 1e-12) * np.trace(G)
    J = np.eye(s) - np.ones((s, s)) / s
    R = J @ (S - G) @ J
    vals, vecs = np.linalg.eigh(0.5 * (R + R.T))
    floor = 1e-8 * max(float(np.mean(nu_n)), np.finfo(float).tiny)
    return vecs[:, -1:] * np.sqrt(max(vals[-1], floor)) * 0.3

class _RankProblem():
    """The inputs of the REML deviance of (Sigma, nu, V), see fit.estimate_motif_variance."""

    def __init__(self, Z, BTB, P, gii, group_inds, fix_ind, fix_val):
        self.Z, self.BTB, self.P = np.asarray(Z), np.asarray(BTB), np.asarray(P)
        self.gii, self.group_inds = np.asarray(gii), group_inds
        self.fix_ind, self.fix_val = int(fix_ind), float(fix_val)

    def fit(self, k: int, Sigma0, nu0, V0, Z=None, silent: bool = True) -> tuple:
        """REML fit of the model with k factors started at (Sigma0, nu0, V0). If V0 has
        fewer than k columns, columns are added to it one by one (it has k - 1 of them
        unless an earlier fit fell back to a smaller model). Returns
        (Sigma, nu, V, deviance)."""
        Z = self.Z if Z is None else Z
        V0 = np.zeros((len(self.gii), 0)) if V0 is None else np.asarray(V0)[:, :k]
        while V0.shape[1] < k:
            V0 = np.append(V0, _new_column(Z, self.P, np.asarray(nu0)[self.gii], V0), axis=1)
        Sigma, nu, V, dev = _lrg.estimate_activity_lowrank(
            Z, self.BTB, self.P, self.gii, self.group_inds, k=k, Sigma0=Sigma0, nu0=nu0,
            fix_ind=self.fix_ind, fix_val=self.fix_val, V0=V0, silent=silent)
        return np.asarray(Sigma), np.asarray(nu), np.asarray(V), float(dev)

    def simulate(self, Sigma, nu, V, root_BTB, root_theta, rng) -> np.ndarray:
        """Z = B^T Y P for data drawn from the model with the given parameters. B^T Y is
        BTB U + B^T E with U ~ MN(0, Sigma, G) and B^T E ~ MN(0, BTB, Theta); the means
        do not matter, as P removes them."""
        m = len(Sigma)
        root_G = np.append(np.diag(np.asarray(nu)[self.gii] ** 0.5), V, axis=1)
        U = (np.asarray(Sigma) ** 0.5).reshape(-1, 1) * rng.standard_normal((m, root_G.shape[1])) @ root_G.T
        E = root_BTB @ rng.standard_normal((root_BTB.shape[1], root_theta.shape[1])) @ root_theta.T
        return (self.BTB @ U + E) @ self.P


def null_pvalue(gain: float, null: list) -> float:
    """The probability of a gain of at least `gain` on one more simulated dataset, if the
    simulated gains `null` are a sample from a normal distribution: the upper tail of
    the predictive t distribution with len(null) - 1 degrees of freedom."""
    null = np.asarray(null, dtype=np.float64)
    n = len(null)
    scale = np.std(null, ddof=1) * (1 + 1 / n) ** 0.5
    if not scale > 0:
        return float(gain <= null.mean())
    return float(t_dist.sf((gain - null.mean()) / scale, n - 1))

def select_activity_rank(project: str, max_rank: int = 5, n_null: int = 5, seed: int = 0,
                         full: bool = False, alpha: float = 0.05, verbose: bool = True) -> dict:
    """Fits the models with 0, 1, ... factors to the project and tells how many of them
    are supported by the data. The step from k - 1 to k factors is judged by the gain,
    the decrease of the REML deviance. The same step is made on `n_null` datasets
    simulated from the fitted model with k - 1 factors, where the k-th factor has
    nothing to find, and the factor is accepted if its gain is unlikely among the
    simulated ones: null_pvalue is below `alpha`. The search stops at the first factor
    that is not accepted (unless `full`). With n_null = 0 only the gains are computed,
    for all ranks up to `max_rank`.

    Returns a dictionary with the chosen `rank` (None if n_null = 0), `steps`, a list of
    dictionaries with the keys rank, deviance, gain, null (simulated gains), pvalue and
    accepted, and `capped`: whether max_rank had to be lowered to what the samples allow."""
    from .grn import load_fit, held_out_inds
    jax.config.update('jax_enable_x64', True)
    if max_rank < 1:
        raise InputError(f'--max-rank is the largest number of factors to try and must be at least 1, but '
                         f'{max_rank} was given.')
    if n_null < 0 or 0 < n_null < MIN_NULL:
        raise InputError(f'--null is a number of simulated datasets: either 0 (no simulations) or at least '
                         f'{MIN_NULL}, as the spread of their gains has to be estimated. {n_null} was given.')
    if not 0 < alpha < 1:
        raise InputError(f'--alpha is a significance level and must lie strictly between 0 and 1, but {alpha:g} '
                         'was given.')
    data = read_init(project)
    data, fit = load_fit(project, data)
    if fit.motif_variance.fixed_group is None:
        raise InputError('The factors of "--activity-lowrank" need variances of motif activities, but the '
                         f'project "{project}" was fitted with "--no-motif-variance". Re-run "maradoner fit" '
                         'without that option.')
    n_full = len(data.promoter_names)
    test_inds = held_out_inds(fit)
    data, _ = split_data(data, test_inds)
    fit = align_fit_to_promoters(fit, test_inds, n_full)
    error_variance = fit.error_variance
    weights = None
    promotor = getattr(error_variance, 'promotor', None)
    if promotor is not None and np.var(promotor) > 0:
        # "fit --promoter-variance": the same weights of promoters as in the fit
        weights = np.asarray(promotor) ** (-0.5)
    logger_print('Transforming data...', verbose)
    data = transform_data(data, helmert=True, weights=weights, loading_context=fit.loading_context,
                          drist=fit.drist, loading_multipliers=fit.loading_multipliers)
    B_decomposition = lowrank_decomposition(data.B, verbose=False)
    # The same quantities as in fit.estimate_motif_variance.
    D = np.asarray(error_variance.variance, dtype=np.float64)
    j = fit.motif_variance.fixed_group
    j = int(np.argsort(D)[len(D) // 2]) if j is None else int(j)
    fix = float(D[j] * NU_REFERENCE_FRACTION)
    BTB = np.asarray(B_decomposition.V.T * B_decomposition.S ** 2 @ B_decomposition.V)
    W_err = _error_lowrank_W(error_variance)
    D_n = jnp.asarray(D)[data.group_inds_inv]
    if W_err is None:
        P = covariate_projection(D_n, data.X)
    else:
        P = _lowrank.covariate_projection_theta(D_n, W_err, data.X)
    P = np.asarray(P)
    Z = np.asarray(data.B.T @ data.Y) @ P
    problem = _RankProblem(Z, BTB, P, data.group_inds_inv, data.group_inds, j, fix)
    s = Z.shape[1]
    dof = int(np.linalg.matrix_rank(P, hermitian=True))
    capped = max_rank >= dof
    if capped:
        warn(f'After the covariates are removed, {dof} degrees of freedom are left among the {s} samples: '
             f'at most {dof - 1} factors can be tried, not {max_rank}.')
        max_rank = dof - 1
    del data, B_decomposition

    logger_print('Fitting the model without factors...', verbose)
    Sigma, nu = np.asarray(fit.motif_variance.motif, dtype=np.float64), np.asarray(fit.motif_variance.group, dtype=np.float64)
    nu = nu.copy()
    nu[j] = fix
    # (a fit made with --activity-lowrank holds the variances of another model: refit)
    Sigma, nu, V, dev = problem.fit(0, Sigma, nu, None)
    rng = np.random.default_rng(seed)
    root_BTB = root_theta = None
    if n_null:
        root_BTB = _sym_root(BTB)
        root_theta = np.diag(np.asarray(D_n) ** 0.5)
        if W_err is not None:
            root_theta = np.append(root_theta, np.asarray(W_err), axis=1)
    steps = [dict(rank=0, deviance=dev, gain=None, null=None, pvalue=None, accepted=None)]
    rank = 0 if n_null else None
    for k in range(1, max_rank + 1):
        logger_print(f'Fitting the model with {k} factor(s)...', verbose)
        Sigma_k, nu_k, V_k, dev_k = problem.fit(k, Sigma, nu, V, silent=False)
        gain = max(dev - dev_k, 0.0)
        null, pvalue, accepted = None, None, None
        if n_null:
            null = list()
            for i in range(n_null):
                logger_print(f'  simulated dataset {i + 1} of {n_null}...', verbose)
                Z_sim = problem.simulate(Sigma, nu, V, root_BTB, root_theta, rng)
                Sigma_s, nu_s, V_s, dev_s = problem.fit(k - 1, Sigma, nu, V, Z=Z_sim)
                dev_s_k = problem.fit(k, Sigma_s, nu_s, V_s, Z=Z_sim)[-1]
                null.append(max(dev_s - dev_s_k, 0.0))
            pvalue = null_pvalue(gain, null)
            accepted = bool(pvalue < alpha)
        steps.append(dict(rank=k, deviance=dev_k, gain=gain, null=null, pvalue=pvalue, accepted=accepted))
        if n_null and accepted and rank == k - 1:
            rank = k
        if n_null and not accepted and not full:
            break
        if dev_k < dev:
            Sigma, nu, V, dev = Sigma_k, nu_k, V_k, dev_k
    return dict(rank=rank, steps=steps, n_null=n_null, capped=capped)
