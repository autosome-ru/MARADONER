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

By default only the intercept column of M is changed, and the covariances between it and the
other columns are those of the GLS estimate. The coefficients of the covariates (the other
columns of M) have the same problem as the intercept: a regression on many similar loadings fits
the training promoters and predicts others poorly. With a prior on them (covariates 'shared' or
'separate'), every column a of M gets a prior

    M[:, a] ~ N(1 mu_a, P_a)

of the same kind: the intercept with a separate set of variances, and the other columns with
one set of variances for all of them ('shared') or one per column ('separate'). The variances
of a column are estimated by REML from its GLS estimate and the block of its covariance (the
covariances between columns are left out there). The posterior means and the covariance of
their prediction error are computed jointly for vec(M), as above with

    V = blockdiag(P_a) + C_full,   1 replaced by I_c kron 1_m,

C_full the covariance of the GLS estimate of vec(M) (pinv(fim), and the covariance of the
promoter baselines in the intercept block).
"""
from dataclasses import replace
import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.optimize import minimize
from .utils import InputError, logger_print
from .lowrank_sigma import motif_covariance

KINDS = ('none', 'ridge', 'families', 'covariance')
# Whether the coefficients of the covariates get a prior, and with which variances.
COVARIATE_PRIORS = ('none', 'shared', 'separate')


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


def gls_covariance_full(motif_mean) -> np.ndarray:
    """Covariance of the GLS estimate of vec(M) (column-major, (m c, m c)): the pseudo-inverse of
    the Fisher information, with the covariance of the promoter baselines added to the
    intercept block."""
    m = np.asarray(motif_mean.mean).shape[0]
    fim = np.asarray(motif_mean.fim, dtype=np.float64)
    w, v = np.linalg.eigh((fim + fim.T) / 2)
    keep = np.abs(w) > len(w) * np.finfo(float).eps * np.abs(w).max()
    v = v[:, keep]
    C = (v / w[keep]) @ v.T
    del v
    if getattr(motif_mean, 'baseline_cov', None) is not None:
        C[:m, :m] += np.asarray(motif_mean.baseline_cov, dtype=np.float64)
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


def estimate_prior(mu, C, terms: list) -> tuple:
    """REML estimates of the variances of P (log scale) and the deviance at them. Several
    starting points are tried, since a variance may sit at its lower bound. mu and C may be
    lists of the columns and of their covariances sharing the variances: the deviance is then
    the sum over the columns, with a common mean per column."""
    pairs = list(zip(mu, C)) if isinstance(mu, (list, tuple)) else [(mu, C)]
    scale = max(float(np.mean([np.var(x) for x, _ in pairs])), 1e-12)
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
        res = minimize(lambda x: sum(_reml(x, terms, C_, mu_)[0] for mu_, C_ in pairs), np.array(x0),
                       method='L-BFGS-B', bounds=bounds)
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


def joint_posterior(M: np.ndarray, C: np.ndarray, P: list) -> tuple:
    """Posterior means of M (m, c) under vec(M) ~ N(A mu, blockdiag(P)), A = I_c kron 1_m, with
    the GLS estimate M and its covariance C ((m c, m c), column-major; overwritten), mu
    estimated by GLS (REML). Returns the means (m, c), mu (c,), the intercept block (m, m) of the
    covariance of the prediction error (Henderson: C - C V^-1 C + C V^-1 A (A^T V^-1 A)^-1
    A^T V^-1 C, V = blockdiag(P) + C) and its blocks per motif (m, c, c)."""
    m, c = M.shape
    blk = [slice(a * m, (a + 1) * m) for a in range(c)]
    y = M.reshape(-1, order='F')
    # The parts of C needed after it is overwritten.
    C00 = C[blk[0], blk[0]].copy()
    Cd = np.stack([np.stack([np.diag(C[blk[a], blk[b]]) for b in range(c)], axis=-1) for a in range(c)], axis=1)
    A = np.kron(np.eye(c), np.ones((m, 1)))
    V = C.copy()
    for a in range(c):
        V[blk[a], blk[a]] += P[a]
    L, _ = cho_factor(V, lower=True, overwrite_a=True, check_finite=False)
    del V
    ViA = cho_solve((L, True), A, check_finite=False)
    AViA = A.T @ ViA
    mu = np.linalg.solve(AViA, ViA.T @ y)
    Vir = cho_solve((L, True), y - A @ mu, check_finite=False)
    mean = np.stack([mu[a] + P[a] @ Vir[blk[a]] for a in range(c)], axis=1)
    CViA = C @ ViA
    G = CViA @ np.linalg.inv(AViA)
    Z = solve_triangular(L, C, lower=True, overwrite_b=True, check_finite=False)   # L^-1 C
    del L
    # C V^-1 C = Z^T Z
    cov00 = C00 - Z[:, blk[0]].T @ Z[:, blk[0]] + G[blk[0]] @ CViA[blk[0]].T
    blocks = np.empty((m, c, c))
    for a in range(c):
        for b in range(a, c):
            d = Cd[:, a, b] - np.einsum('ki,ki->i', Z[:, blk[a]], Z[:, blk[b]]) + \
                np.einsum('ij,ij->i', G[blk[a]], CViA[blk[b]])
            blocks[:, a, b] = blocks[:, b, a] = d
    return mean, mu, (cov00 + cov00.T) / 2, blocks


def shrink_motif_means(motif_mean, kind: str, motif_variance=None, families: np.ndarray = None,
                       verbose: bool = True, covariates: str = 'none', covariate_names: list = None) -> tuple:
    """Replaces the intercept column of M by its posterior mean under the prior of the given kind,
    and with covariates 'shared' or 'separate' the other columns as well (see the module
    docstring). Returns the new MotifMeanEstimates and delta = old - new intercept column (m,):
    the promoter means take B delta back, so that mu_p + B M X stays the same in the intercept.
    The changes of the other columns stay in the residuals."""
    M = np.asarray(motif_mean.mean, dtype=np.float64)
    flat = M.ndim == 1
    M = M.reshape(len(M), -1).copy()
    m, c = M.shape
    if covariates not in COVARIATE_PRIORS:
        raise InputError(f'Unknown prior on the covariates of the motif means "{covariates}": choose one of '
                         f'{", ".join(COVARIATE_PRIORS)}.')
    S = motif_covariance(motif_variance) if kind == 'covariance' else None
    if kind == 'families' and families is None:
        raise InputError('--motif-mean-prior families needs the families of motifs: pass --motif-families.')
    terms = prior_terms(kind, m, families=families, S=S)
    names = {'ridge': ['kappa^2'], 'families': ['kappa_f^2', 'kappa^2'],
             'covariance': ['kappa_s^2', 'kappa^2']}[kind]
    mu = M[:, 0].copy()
    if c == 1 or covariates == 'none':
        C = gls_covariance(motif_mean)
        log_var, dev = estimate_prior(mu, C, terms)
        mean, cov, mu0 = posterior(mu, C, terms, log_var)
        variances = dict(zip(names, np.exp(log_var).tolist()))
        M[:, 0] = mean
        blocks = None
    else:
        if covariate_names is None or len(covariate_names) != c:
            covariate_names = ['intercept'] + [f'covariate {a}' for a in range(1, c)]
        C = gls_covariance_full(motif_mean)
        col = lambda a: C[a * m:(a + 1) * m, a * m:(a + 1) * m]
        log_vars = [estimate_prior(mu, col(0), terms)[0]]
        if covariates == 'shared':
            lv = estimate_prior([M[:, a] for a in range(1, c)], [col(a) for a in range(1, c)], terms)[0]
            log_vars += [lv] * (c - 1)
        else:
            log_vars += [estimate_prior(M[:, a], col(a), terms)[0] for a in range(1, c)]
        P = [sum(np.exp(t) * T for t, T in zip(lv, terms)) for lv in log_vars]
        new, mus, cov, blocks = joint_posterior(M, C, P)
        del C
        mean, mu0 = new[:, 0], float(mus[0])
        variances = dict(zip(names, np.exp(log_vars[0]).tolist()))
        if covariates == 'shared':
            variances.update({f'{k} (covariates)': v for k, v in zip(names, np.exp(log_vars[1]).tolist())})
        else:
            for a in range(1, c):
                variances.update({f'{k} ({covariate_names[a]})': v
                                  for k, v in zip(names, np.exp(log_vars[a]).tolist())})
        norms = ', '.join(f'{covariate_names[a]} {np.linalg.norm(new[:, a]):.3g} (GLS: {np.linalg.norm(M[:, a]):.3g})'
                          for a in range(1, c))
        logger_print(f'  prior on the coefficients of the covariates ({covariates}): the norms of the '
                     f'columns are {norms}.', verbose)
        M[:, 1:] = new[:, 1:]
        M[:, 0] = mean
    logger_print(f'  prior on the motif means ({kind}): ' +
                 ', '.join(f'{k} = {v:.3g}' for k, v in variances.items()) +
                 f', mu_0 = {mu0:.3g}; the norm of the means is {np.linalg.norm(mean):.3g} '
                 f'(GLS: {np.linalg.norm(mu):.3g}).', verbose)
    delta = mu - mean
    res = replace(motif_mean, mean=M[:, 0] if flat else M, prior=kind,
                  prior_variances=variances, prior_mean=mu0, posterior_cov=cov, gls_intercept=mu,
                  posterior_blocks=blocks)
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
    # The joint posterior of several columns vs the dense formulas, and with one column vs posterior().
    m, c = 25, 3
    R = rng.normal(size=(m * c, m * c))
    C = R @ R.T / (m * c) + 0.1 * np.eye(m * c)
    Mh = rng.normal(size=(m, c))
    P = [np.exp(rng.normal()) * prior_terms('ridge', m)[0] + 0.3 * np.ones((m, m)) * a for a in range(c)]
    mean, mu, cov00, blocks = joint_posterior(Mh, C.copy(), P)
    A = np.kron(np.eye(c), np.ones((m, 1)))
    Pf = np.zeros((m * c, m * c))
    for a in range(c):
        Pf[a * m:(a + 1) * m, a * m:(a + 1) * m] = P[a]
    Vi = np.linalg.inv(Pf + C)
    y = Mh.reshape(-1, order='F')
    mu_d = np.linalg.solve(A.T @ Vi @ A, A.T @ Vi @ y)
    mean_d = (A @ mu_d + Pf @ Vi @ (y - A @ mu_d)).reshape(c, m).T
    cov_d = C - C @ Vi @ C + C @ Vi @ A @ np.linalg.solve(A.T @ Vi @ A, A.T @ Vi @ C)
    blocks_d = np.einsum('aibi->iab', cov_d.reshape(c, m, c, m))
    err = max(np.abs(mean - mean_d).max(), np.abs(mu - mu_d).max(), np.abs(cov00 - cov_d[:m, :m]).max(),
              np.abs(blocks - blocks_d).max())
    m1, mu1, c1, _ = joint_posterior(Mh[:, :1], C[:m, :m].copy(), P[:1])
    terms = prior_terms('ridge', m)
    lv = np.log([float(P[0][0, 0] - P[0][0, 1])])
    pm, pc, p0 = posterior(Mh[:, 0], C[:m, :m], terms, lv)
    err1 = max(np.abs(m1[:, 0] - pm).max(), abs(mu1[0] - p0), np.abs(c1 - pc).max())
    print(f'joint posterior vs dense: {err:.2e}; one column vs posterior(): {err1:.2e}')
    assert err < 1e-8 and err1 < 1e-8


if __name__ == '__main__':
    _selftest()
