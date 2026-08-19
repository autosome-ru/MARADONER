# -*- coding: utf-8 -*-
"""Single-sample Bayesian linear regression: an alternative to the MARADONER model.

MARADONER reads motif activities off the *variation across samples*: the model
``Y = B U + E`` with ``U ~ MN(mu_m sqrt(d^-1)^T, Sigma, G_hat)`` splits expression
into an activity part (varying over samples, with per-motif variance ``Sigma`` and
per-sample variance ``nu``) and an error part. With a single sample there is no
across-sample variation left to split -- the REML objective for the error
variances is identically flat -- so that fit is not identifiable.

This module implements a deliberately *different* model that is identifiable from
one sample, because there the replication comes from the ``p`` promoters rather
than from the samples::

    y = mu * 1_p + B u + e,    e ~ N(0, sigma^2 I_p),    u ~ N(0, Sigma)

* ``mu``      -- scalar mean expression level (the single fixed effect);
* ``sigma^2`` -- scalar error variance;
* ``Sigma``   -- diagonal ``m x m`` matrix of per-motif activity variances, as in
  MARADONER.

Marginally ``y ~ N(mu 1_p, V)`` with ``V = B Sigma B^T + sigma^2 I_p``. The
variance components ``(sigma^2, Sigma)`` are estimated by REML exactly in the
spirit of MARADONER -- ``mu`` is profiled out as the fixed effect, and the
resulting restricted likelihood is optimised with the same ``MetaOptimizer``
(squared reparameterisation, so the variances stay non-negative). Motif
activities are then reported as the posterior mean ``E[u | y]``, which makes this
an automatic-relevance-determination / sparse-Bayesian ridge regression.

Everything is reduced to ``m x m`` algebra by the matrix determinant lemma and
the Woodbury identity, so no ``p x p`` matrix is ever formed:

    V = sigma^2 I_p + Bt Bt^T,          Bt = B diag(sqrt(Sigma))
    K = sigma^2 I_m + Bt^T Bt           (m x m, positive definite)
    log|V|      = (p - m) log sigma^2 + log|K|
    x^T V^-1 z  = (x^T z - (Bt^T x)^T K^-1 (Bt^T z)) / sigma^2

so a single ``m x m`` Cholesky per likelihood evaluation is all that is needed.
"""
from .meta_optimizer import MetaOptimizer
from .utils import read_init, logger_print
from statsmodels.stats import multitest
from functools import partial
import jax.scipy.linalg as jsl
import scipy.stats as st
import jax.numpy as jnp
import pandas as pd
import numpy as np
import scipy.linalg
import jax
import os


# Additive floor on sigma^2 so that the Cholesky of K stays defined even if the
# optimiser walks onto the sigma^2 == 0 boundary. Scaled by var(y) at fit time.
_SIGMA2_FLOOR = 1e-10


def _neg2_reml(theta: jnp.ndarray, C: jnp.ndarray, by: jnp.ndarray, b1: jnp.ndarray,
               yy: float, y1: float, p: int, floor: float) -> float:
    """-2 * restricted log-likelihood (up to the constant ``(p-1) log 2 pi``).

    ``theta`` packs ``[sigma^2, Sigma_1, ..., Sigma_m]``, all non-negative.
    ``C = B^T B``, ``by = B^T y``, ``b1 = B^T 1_p``, ``yy = y^T y``, ``y1 = 1_p^T y``.
    """
    m = C.shape[0]
    sigma2 = theta[0] + floor
    s = jnp.sqrt(theta[1:])
    K = sigma2 * jnp.eye(m) + s[:, None] * C * s[None, :]
    L = jsl.cholesky(K, lower=True)
    logdet_V = (p - m) * jnp.log(sigma2) + 2.0 * jnp.log(jnp.diag(L)).sum()

    ay = s * by
    a1 = s * b1
    Ky = jsl.cho_solve((L, True), ay)
    K1 = jsl.cho_solve((L, True), a1)

    yVy = (yy - ay @ Ky) / sigma2          # y^T V^-1 y
    y1V = (y1 - ay @ K1) / sigma2          # 1_p^T V^-1 y
    o1V = (p - a1 @ K1) / sigma2           # 1_p^T V^-1 1_p
    rVr = yVy - y1V ** 2 / o1V             # residual quadratic form at mu_hat
    # REML criterion: log|V| + log|X^T V^-1 X| + r^T V^-1 r, with X = 1_p.
    return logdet_V + jnp.log(o1V) + rVr


def _posterior(theta: np.ndarray, C: np.ndarray, by: np.ndarray, b1: np.ndarray,
               y1: float, p: int, floor: float) -> dict:
    """Closed-form ``mu_hat`` and the posterior of ``u`` at the REML estimates.

    ``E[u|y]   = diag(s) K^-1 diag(s) B^T r``  and
    ``Cov[u|y] = sigma^2 diag(s) K^-1 diag(s)``, with ``r = y - mu_hat 1_p``.
    """
    m = C.shape[0]
    sigma2 = float(theta[0]) + floor
    S = np.asarray(theta[1:], dtype=float)
    s = np.sqrt(S)
    K = sigma2 * np.eye(m) + s[:, None] * C * s[None, :]
    cf = scipy.linalg.cho_factor(K, lower=True)

    a1 = s * b1
    ay = s * by
    K1 = scipy.linalg.cho_solve(cf, a1)
    o1V = (p - a1 @ K1) / sigma2            # 1_p^T V^-1 1_p
    y1V = (y1 - ay @ K1) / sigma2           # 1_p^T V^-1 y
    mu = y1V / o1V
    mu_std = float(np.sqrt(1.0 / o1V))

    br = by - mu * b1                       # B^T r
    u = s * scipy.linalg.cho_solve(cf, s * br)
    Kinv_diag = np.diag(scipy.linalg.cho_solve(cf, np.eye(m)))
    u_var = sigma2 * S * Kinv_diag
    return {'mu': float(mu), 'mu_std': mu_std, 'sigma2': sigma2, 'Sigma': S,
            'u': u, 'u_std': np.sqrt(np.clip(u_var, 0.0, None)), 'Bt_r': br}


def _variance_stderr(theta: np.ndarray, obj, rel_tol: float = 1e-8) -> np.ndarray:
    """Asymptotic standard errors of the variance components.

    The REML observed information is ``H / 2`` where ``H`` is the Hessian of
    ``-2 l_R``. Components driven to (numerically) zero by the ARD shrinkage sit on
    the boundary of the parameter space, where the asymptotic normal approximation
    does not hold; those are excluded from the Hessian and reported as NaN.
    """
    theta = np.asarray(theta, dtype=float)
    scale = max(float(np.max(theta)), 1e-300)
    interior = theta > rel_tol * scale
    interior[0] = True                                    # sigma^2 is always interior
    idx = np.flatnonzero(interior)
    base = jnp.array(theta)

    def sub(z):
        return obj(base.at[jnp.array(idx)].set(z))

    H = np.asarray(jax.hessian(sub)(jnp.array(theta[idx])), dtype=float)
    info = H / 2.0
    info = 0.5 * (info + info.T)
    cov = np.linalg.pinv(info, hermitian=True)
    out = np.full(len(theta), np.nan)
    out[idx] = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    return out


def _fit_one(y: np.ndarray, B: np.ndarray, C: np.ndarray, b1: np.ndarray,
             compute_stderr: bool, verbose: bool) -> dict:
    """REML fit of the single-sample model for one expression column."""
    p, m = B.shape
    by = B.T @ y
    yy = float(y @ y)
    y1 = float(y.sum())
    var_y = float(np.var(y))
    floor = _SIGMA2_FLOOR * max(var_y, 1e-12)

    # Start from an even split of var(y) between error and motif activities: the
    # activity part contributes tr(B Sigma B^T)/p = sum_j Sigma_j C_jj / p on average.
    sigma2_0 = 0.5 * var_y
    trC = float(np.sum(np.diag(C)))
    Sigma_0 = 0.5 * var_y * p / max(trC, 1e-12)
    theta0 = np.concatenate([[sigma2_0], np.full(m, Sigma_0)])

    obj = jax.jit(partial(_neg2_reml, C=jnp.array(C), by=jnp.array(by), b1=jnp.array(b1),
                          yy=yy, y1=y1, p=p, floor=floor))
    grad = jax.jit(jax.grad(obj))

    # Same squared reparameterisation MARADONER uses for variance components, so the
    # variances stay non-negative and the ARD shrinkage can park them exactly at zero.
    from scipy.optimize import minimize

    def f(x):
        return float(obj(jnp.array(x) ** 2))

    def g(x):
        x = np.asarray(x, dtype=float)
        return 2.0 * x * np.asarray(grad(jnp.array(x) ** 2), dtype=float)

    sol = minimize(f, np.sqrt(theta0), jac=g, method='L-BFGS-B')
    theta = np.asarray(sol.x, dtype=float) ** 2
    neg2_reml = float(sol.fun)
    grad_norm = float(np.linalg.norm(sol.jac))

    # Polish with MARADONER's own MetaOptimizer (rmsprop momentum + TNC) from that
    # point, and keep whichever objective is lower. Separate scalers for sigma^2 and
    # Sigma, which live on very different scales.
    try:
        opt = MetaOptimizer(obj, grad, num_steps_momentum=15,
                            scaling_set=[slice(0, 1), slice(1, m + 1)])
        res = opt.optimize(jnp.array(theta))
        if np.isfinite(res.fun) and res.fun < neg2_reml:
            theta = np.asarray(res.x, dtype=float)
            neg2_reml = float(res.fun)
            grad_norm = float(res.grad_norm)
        if verbose:
            print('-' * 15)
            print(res)
            print('-' * 15)
    except Exception as exc:                              # pragma: no cover - safety net
        logger_print(f'  MetaOptimizer polish skipped ({exc}); keeping the L-BFGS-B fit.',
                     verbose)

    theta = np.clip(theta, 0.0, None)
    post = _posterior(theta, C, by, b1, y1, p, floor)

    # REML log-likelihood including the constant: -0.5 * (obj + (p - 1) log 2 pi).
    post['loglik'] = -0.5 * (neg2_reml + (p - 1) * np.log(2 * np.pi))
    post['grad_norm'] = grad_norm

    if compute_stderr:
        se = _variance_stderr(theta, obj)
        post['sigma2_std'] = float(se[0])
        post['Sigma_std'] = se[1:]
    else:
        post['sigma2_std'] = np.nan
        post['Sigma_std'] = np.full(m, np.nan)
    return post


def bayes_linreg(project_name: str, output_folder: str, alpha: float = 0.05,
                 stderr: bool = True, promoters: bool = False, x64: bool = True,
                 verbose: bool = True) -> dict:
    """Fit ``y = mu 1_p + B u + e`` per sample by REML and write results.

    Every sample of the project is fitted independently (the model has no
    across-sample term), so with the usual single-sample input this is one fit.
    Results are written straight into ``output_folder`` -- there is no separate
    ``fit`` / ``predict`` / ``export`` chain for this command -- as two flat tables,
    ``motifs.tsv`` and ``samples.tsv`` (plus ``promoters.tsv`` if requested).
    """
    if x64:
        jax.config.update('jax_enable_x64', True)

    data = read_init(project_name)
    Y = np.asarray(data.Y, dtype=float)
    B = np.asarray(data.B, dtype=float)
    p, m = B.shape
    s = Y.shape[1]
    motif_names = list(data.motif_names)
    promoter_names = list(data.promoter_names)
    sample_names = list(data.sample_names)
    logger_print(f'Promoters: {p}, motifs: {m}, samples: {s}.', verbose)

    # Shared across samples: only B enters these.
    C = B.T @ B
    b1 = B.sum(axis=0)
    diagC = np.diag(C)

    motif_frames = list()
    sample_rows = list()
    fitted_cols = dict()

    for i, sample in enumerate(sample_names):
        logger_print(f'Fitting sample "{sample}" ({i + 1}/{s})...', verbose)
        y = Y[:, i]
        res = _fit_one(y, B, C, b1, compute_stderr=stderr, verbose=verbose)

        u = res['u']
        u_std = res['u_std']
        z = np.divide(u, u_std, out=np.zeros_like(u), where=u_std > 0)
        pval = 2.0 * st.norm.sf(np.abs(z))
        fdr = multitest.multipletests(pval, alpha=alpha, method='fdr_bh')[1]

        fitted = res['mu'] + B @ u
        resid = y - fitted
        # Fractions of variance are always taken against the plain "predict the mean
        # of y" baseline: y.mean() is what minimises this sum, so using the GLS
        # estimate mu_hat instead would silently inflate every FOV below.
        y_c = y - y.mean()
        ss_tot = max(float(y_c @ y_c), 1e-300)
        # Contribution of a single motif: the increase in residual sum of squares
        # when its effect alone is removed, relative to that same total.
        Bt_res = res['Bt_r'] - C @ u
        fov = (2.0 * u * Bt_res + u ** 2 * diagC) / ss_tot

        frame = pd.DataFrame({'activity': u, 'std': u_std, 'z': z, 'p_value': pval,
                              'fdr_bh': fdr, 'variance': res['Sigma'],
                              'variance_std': res['Sigma_std'], 'fov': fov},
                             index=motif_names)
        frame = frame.sort_values('z', key=lambda c: -c.abs())
        if s > 1:
            frame.insert(0, 'sample', sample)
        motif_frames.append(frame)
        fitted_cols[sample] = fitted

        n_nonzero = int(np.sum(res['Sigma'] > 1e-8 * max(res['Sigma'].max(), 1e-300)))
        n_sig = int(np.sum(fdr < alpha))
        sample_rows.append({
            'sample': sample,
            'mu': res['mu'], 'mu_std': res['mu_std'],
            'sigma2': res['sigma2'], 'sigma2_std': res['sigma2_std'],
            'fov': 1.0 - float(resid @ resid) / ss_tot,
            'reml_loglik': res['loglik'], 'grad_norm': res['grad_norm'],
            'n_nonzero_variance': n_nonzero, 'n_significant': n_sig,
        })
        logger_print(f'  mu = {res["mu"]:.4f}, sigma^2 = {res["sigma2"]:.4f}, '
                     f'FOV = {sample_rows[-1]["fov"]:.4f}, '
                     f'{n_nonzero}/{m} motifs with non-zero variance, '
                     f'{n_sig} significant at FDR {alpha}.', verbose)

    output_folder = str(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    # Two flat files, nothing stored twice: everything per-motif in one table, ranked
    # by |z| (strongest first), and everything per-sample in the other. z / p_value /
    # fdr_bh do follow from activity and std, but they are what the table is read and
    # sorted by, so they stay.
    motifs = pd.concat(motif_frames) if s > 1 else motif_frames[0]
    motifs.index.name = 'motif'
    motifs.to_csv(os.path.join(output_folder, 'motifs.tsv'), sep='\t')

    samples = pd.DataFrame(sample_rows).set_index('sample')
    samples.to_csv(os.path.join(output_folder, 'samples.tsv'), sep='\t')

    written = ['motifs.tsv', 'samples.tsv']
    if promoters:
        # The only per-locus output, and the only thing not recoverable from the two
        # tables above without the loading matrix. Off by default.
        pd.DataFrame(fitted_cols, index=promoter_names).rename_axis('promoter').to_csv(
            os.path.join(output_folder, 'promoters.tsv'), sep='\t')
        written.append('promoters.tsv')

    logger_print(f'Written to {output_folder}: {", ".join(written)}.', verbose)
    return {'motifs': motifs, 'samples': samples}
