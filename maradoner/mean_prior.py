"""A prior on the mean activities of the motifs (fit --motif-mean-prior).

Only mu_p + B mu_m is identified, and the GLS estimate of mu_m (the intercept column of M)
is the regression of the promoter levels on the loadings. With many similar motifs it fits
the promoter baselines of the training promoters and predicts the levels of other promoters
poorly. A prior

    mu_m ~ N(1 mu_0, P),
    ridge:      P = kappa^2 I,
    families:   P = kappa_f^2 F F^T + kappa^2 I,
    covariance: P = kappa_s^2 S + kappa^2 I,

with F the families of motifs and S the fitted covariance between motif activities, shrinks
the estimate. The GLS estimate is taken as mu_hat ~ N(mu_m, C): C is its covariance, the
inverse of the Fisher information plus the covariance of the promoter baselines. Then
mu_hat ~ N(1 mu_0, P + C). The variances in P are estimated by REML with mu_0 profiled out,
and mu_m is replaced by its posterior mean. The covariance stored with it is that of the
prediction error, the uncertainty of mu_0 included (Henderson):

    C - C H C,   H = V^-1 - V^-1 1 (1^T V^-1 1)^-1 1^T V^-1,   V = P + C.

Only the intercept column of M is changed. With covariates, the covariances between it and
the other columns are those of the GLS estimate.
"""
from dataclasses import replace
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from .utils import InputError, logger_print
from .lowrank_sigma import motif_covariance

KINDS = ('none', 'ridge', 'families', 'covariance')


def gls_covariance(motif_mean) -> np.ndarray:
    """Covariance C of the GLS estimate of the intercept column of M: the pseudo-inverse of the
    Fisher information (its intercept block) plus the covariance of the promoter baselines."""
    M = np.asarray(motif_mean.mean)
    m = M.shape[0]
    fim = np.asarray(motif_mean.fim, dtype=np.float64)
    w, v = np.linalg.eigh((fim + fim.T) / 2)
    keep = np.abs(w) > len(w) * np.finfo(float).eps * np.abs(w).max()
    C = ((v[:, keep] / w[keep]) @ v[:, keep].T)[:m, :m]
    if getattr(motif_mean, 'baseline_cov', None) is not None:
        C = C + np.asarray(motif_mean.baseline_cov, dtype=np.float64)
    return (C + C.T) / 2


def prior_terms(kind: str, m: int, families: np.ndarray = None, S: np.ndarray = None) -> list:
    """The matrices whose sum, each times a variance, is P. The identity comes last."""
    if kind == 'ridge':
        return [np.eye(m)]
    if kind == 'families':
        F = np.asarray(families, dtype=np.float64)
        return [F @ F.T, np.eye(m)]
    if kind == 'covariance':
        S = np.diag(S) if np.ndim(S) == 1 else np.asarray(S, dtype=np.float64)
        return [(S + S.T) / 2, np.eye(m)]
    raise InputError(f'Unknown prior on the motif means "{kind}": choose one of {", ".join(KINDS)}.')


def _reml(log_var, terms, C, mu):
    """REML deviance of mu ~ N(1 mu_0, P + C) up to a constant, and the Cholesky factor."""
    V = C + sum(np.exp(t) * T for t, T in zip(log_var, terms))
    try:
        cf = cho_factor(V, lower=True)
    except np.linalg.LinAlgError:
        return np.inf, None
    Vi_1 = cho_solve(cf, np.ones(len(mu)))
    a = float(np.sum(Vi_1))
    mu0 = float(Vi_1 @ mu) / a
    r = mu - mu0
    dev = 2 * np.sum(np.log(np.diag(cf[0]))) + np.log(a) + float(r @ cho_solve(cf, r))
    return dev, cf


def estimate_prior(mu: np.ndarray, C: np.ndarray, terms: list) -> tuple:
    """REML estimates of the variances of P (log scale) and the deviance at them. Several
    starting points are tried, since a variance may sit at its lower bound."""
    scale = max(float(np.var(mu)), 1e-12)
    starts, bounds = list(), list()
    for T in terms:
        s = np.log(scale / max(float(np.mean(np.diag(T))), 1e-12))
        bounds.append((s - 30.0, s + 10.0))
        starts.append(s - np.log(2.0))
    trials = [starts]
    for i in range(len(terms) - 1):
        low = list(starts)
        low[i] = bounds[i][0]
        trials.append(low)
    best = None
    for x0 in trials:
        res = minimize(lambda x: _reml(x, terms, C, mu)[0], np.array(x0), method='L-BFGS-B', bounds=bounds)
        if best is None or res.fun < best.fun:
            best = res
    return best.x, float(best.fun)


def posterior(mu: np.ndarray, C: np.ndarray, terms: list, log_var: np.ndarray) -> tuple:
    """Posterior mean of mu_m, the covariance of its prediction error and mu_0."""
    P = sum(np.exp(t) * T for t, T in zip(log_var, terms))
    _, cf = _reml(log_var, terms, C, mu)
    Vi_1 = cho_solve(cf, np.ones(len(mu)))
    a = float(np.sum(Vi_1))
    mu0 = float(Vi_1 @ mu) / a
    mean = mu0 + P @ cho_solve(cf, mu - mu0)
    C_Vi_1 = C @ Vi_1
    cov = C - C @ cho_solve(cf, C) + np.outer(C_Vi_1, C_Vi_1) / a
    return mean, (cov + cov.T) / 2, mu0


def shrink_motif_means(motif_mean, kind: str, motif_variance=None, families: np.ndarray = None,
                       verbose: bool = True) -> tuple:
    """Replaces the intercept column of M by its posterior mean under the prior of the given kind.
    Returns the new MotifMeanEstimates and delta = old - new intercept column (m,): the
    promoter means take B delta back, so that mu_p + B M X stays the same."""
    M = np.asarray(motif_mean.mean, dtype=np.float64)
    flat = M.ndim == 1
    M = M.reshape(len(M), -1).copy()
    m = M.shape[0]
    S = motif_covariance(motif_variance) if kind == 'covariance' else None
    if kind == 'families' and families is None:
        raise InputError('--motif-mean-prior families needs the families of motifs: pass --motif-families.')
    terms = prior_terms(kind, m, families=families, S=S)
    mu = M[:, 0].copy()
    C = gls_covariance(motif_mean)
    log_var, dev = estimate_prior(mu, C, terms)
    mean, cov, mu0 = posterior(mu, C, terms, log_var)
    names = {'ridge': ['kappa^2'], 'families': ['kappa_f^2', 'kappa^2'],
             'covariance': ['kappa_s^2', 'kappa^2']}[kind]
    variances = dict(zip(names, np.exp(log_var).tolist()))
    logger_print(f'  prior on the motif means ({kind}): ' +
                 ', '.join(f'{k} = {v:.3g}' for k, v in variances.items()) +
                 f', mu_0 = {mu0:.3g}; the norm of the means is {np.linalg.norm(mean):.3g} '
                 f'(GLS: {np.linalg.norm(mu):.3g}).', verbose)
    delta = mu - mean
    M[:, 0] = mean
    res = replace(motif_mean, mean=M[:, 0] if flat else M, prior=kind,
                  prior_variances=variances, prior_mean=mu0, posterior_cov=cov, gls_intercept=mu)
    return res, delta


def _selftest():
    """Ridge: the posterior mean equals the ridge regression with a free common mean, and REML
    recovers the prior variance. Families: P = 0 for the family part gives the ridge."""
    rng = np.random.default_rng(1)
    p, m, omega, kappa = 1500, 120, 20.0, 0.3
    base = rng.normal(size=(p, 20))
    B = np.abs(base @ rng.normal(size=(20, m)) + 0.5 * rng.normal(size=(p, m)))   # correlated loadings
    mu_true = 0.2 + kappa * rng.normal(size=m)
    y = B @ mu_true + omega * rng.normal(size=p)
    BTB = B.T @ B
    mu_hat = np.linalg.solve(BTB, B.T @ y)
    C = omega ** 2 * np.linalg.inv(BTB)
    terms = prior_terms('ridge', m)
    log_var, _ = estimate_prior(mu_hat, C, terms)
    mean, cov, mu0 = posterior(mu_hat, C, terms, log_var)
    lam = omega ** 2 / np.exp(log_var[0])
    # ridge on mu - mu_0 1 with mu_0 free: minimise |y - B mu|^2 + lam |mu - mu_0 1|^2
    A = np.block([[BTB + lam * np.eye(m), -lam * np.ones((m, 1))],
                  [-lam * np.ones((1, m)), lam * m * np.ones((1, 1))]])
    sol = np.linalg.solve(A, np.concatenate([B.T @ y, [0.0]]))
    err_ridge = np.max(np.abs(sol[:m] - mean))
    print(f'posterior mean vs ridge with a free common mean: {err_ridge:.2e}')
    print(f'kappa: {np.exp(log_var[0] / 2):.3f} (true {kappa}), mu_0: {mu0:.3f} (true 0.2)')
    print(f'error of the means: GLS {np.sqrt(np.mean((mu_hat - mu_true) ** 2)):.3f}, '
          f'posterior {np.sqrt(np.mean((mean - mu_true) ** 2)):.3f}; '
          f'mean posterior sd {np.sqrt(np.mean(np.diag(cov))):.3f}')
    F = np.kron(np.eye(m // 4), np.ones((4, 1)))
    lv_f, dev_f = estimate_prior(mu_hat, C, prior_terms('families', m, families=F))
    _, dev_r = estimate_prior(mu_hat, C, terms)
    print(f'families on data without them: kappa_f^2 {np.exp(lv_f[0]):.2e}, deviance {dev_f:.3f} vs ridge {dev_r:.3f}')
    assert err_ridge < 1e-8 and abs(np.exp(log_var[0] / 2) - kappa) < 0.1 and dev_f <= dev_r + 1e-6


if __name__ == '__main__':
    _selftest()
