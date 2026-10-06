import jax.numpy as jnp
from jax import jit, grad
from jax.scipy.stats import norm
from jax.scipy.special import logsumexp, logit, expit
import pandas as pd
import numpy as np
from scipy.optimize import minimize, brentq
import scipy.stats as st
from functools import partial
from sklearn.mixture import GaussianMixture
from .utils import InputError, warn

try:
    from jax import enable_x64 as _enable_x64
except ImportError:     # older jax
    from jax.experimental import enable_x64 as _enable_x64

def compute_leftmost_probability(Y):
    Y = Y.reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, random_state=0)
    gmm.fit(Y)
    
    means = gmm.means_.flatten()
    leftmost_component_index = np.argmin(means)
    probas = gmm.predict_proba(Y)
    leftmost_probs = probas[:, leftmost_component_index]
    
    return leftmost_probs, gmm

def normax_logpdf(x: jnp.ndarray, mu: float, sigma: float, n: int):
    x = (x - mu) / sigma
    return jnp.log(n) - jnp.log(sigma) + norm.logpdf(x) + (n - 1) * norm.logcdf(x)

def normax_mode(mu: float, sigma: float, n: int) -> float:
    """Mode of the density of the maximum of n independent N(mu, sigma^2) variables
    (float64, from the stationarity condition -z + (n - 1) pdf(z) / cdf(z) = 0)."""
    def g(z):
        return -z + (n - 1) * np.exp(st.norm.logpdf(z) - st.norm.logcdf(z))
    return float(mu + sigma * brentq(g, -40.0, 40.0))

def logmixture(x: jnp.ndarray, mus: jnp.ndarray, sigmas: jnp.ndarray, w: float, n: int):
    logpdf1 = normax_logpdf(x, mus[0], sigmas[0], n)
    logpdf2 = normax_logpdf(x, mus[1], sigmas[1], n)
    w = jnp.array([w, 1 - w]).reshape(-1,1)
    logpdf = jnp.array([logpdf1, logpdf2])
    return logsumexp(logpdf, b=w, axis=0)


def transform(params, forward=True):
    mu = params[:2]
    sigma = params[2:4]
    w = params[-1:]
    if forward:
        sigma = sigma ** 2
        w = expit(w)
    else:
        sigma = sigma ** 0.5
        w = logit(w)
    return jnp.concatenate([mu, sigma, w])

def loglik(params: jnp.ndarray, x: jnp.ndarray, n: int):
    params = transform(params)
    mu = params[:2]
    sigma = params[2:4]
    w = params[-1]
    return -logmixture(x, mu, sigma, w, n).sum()

def filter_lowexp(expression: pd.DataFrame, cutoff=0.9, component_limit=0.6, max_mode=True,
                  fit_plot_filename=None, plot_dpi=200):
    if not (isinstance(cutoff, (int, float, np.integer, np.floating)) and 0 < cutoff <= 1):
        raise InputError('The low-expression filter parameter w must be a number with 0 < w <= 1 (a promoter is '
                         'dropped iff its posterior probability of belonging to the low-expression component '
                         f'exceeds w; w = 1 keeps all promoters), but {cutoff} was given.')
    n_total = len(expression)
    if cutoff >= 1:
        return np.ones(n_total, dtype=bool), np.ones(n_total)
    if not np.isfinite(expression.to_numpy(dtype=float)).all():
        raise InputError('The expression table contains missing or infinite values, so the low-expression '
                         'filter cannot be fitted.')
    if n_total < 10:
        warn(f'Low-expression filter skipped: {n_total} promoters are too few to fit the mixture; all '
             f'{n_total} promoters are kept.')
        return np.ones(n_total, dtype=bool), np.ones(n_total)
    expression = (expression - expression.mean()) / expression.std()
    # A sample with constant expression has no standardised values; it cannot inform the fit.
    expression = expression.loc[:, expression.notna().all(axis=0)]
    if expression.shape[1] == 0:
        warn(f'Low-expression filter skipped: the expression of every sample is constant; all {n_total} '
             'promoters are kept.')
        return np.ones(n_total, dtype=bool), np.ones(n_total)
    if not max_mode:
        expression = expression.mean(axis=1).values
        probs, gmm = compute_leftmost_probability(expression)
        inds = probs <= cutoff
        if fit_plot_filename:
            import matplotlib.pyplot as plt
            from matplotlib.collections import LineCollection
            import seaborn as sns
            x = np.array(sorted(expression))
            pdf = np.exp(gmm.score_samples(x[:, None]))
            points = np.array([x, pdf]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            plt.figure(dpi=plot_dpi, )
            sns.histplot(expression, stat='density', color='grey')
            lc = LineCollection(segments, cmap='winter')
            lc.set_array(probs[np.argsort(expression)])
            lc.set_linewidth(3)
            line = plt.gca().add_collection(lc)
            plt.colorbar(line)
            plt.xlabel('Standardized expression')
            plt.tight_layout()
            plt.savefig(fit_plot_filename)
            plt.close()
        return inds, 1 - probs
        
    expression_max = expression.max(axis=1).values
    n = expression.shape[1]

    # The fit is done in double precision whatever the global jax setting is: in float32 the
    # finite-difference steps below are lost in rounding and the optimiser stalls.
    with _enable_x64(True):
        mu = [-1.0, 0.0]
        sigmas = [1.0, 1.0]
        w = [0.5]
        x0 = jnp.array(mu + sigmas + w)
        x0 = transform(x0, False)
        fun = jit(partial(loglik, x=expression_max, n=n))
        jac = jit(grad(fun))
        res = minimize(fun, x0, jac=jac)
        params = np.asarray(transform(res.x), dtype=float)
    mu = params[:2]
    sigma = params[2:4]
    w = float(params[-1])
    if not (np.isfinite(params).all() and np.isfinite(res.fun) and 0 < w < 1):
        warn('Low-expression filter skipped: the two-component mixture could not be fitted to the expression '
             f'of the promoters; all {n_total} promoters are kept. Check that the expression is on a log scale.')
        return np.ones(n_total, dtype=bool), np.ones(n_total)

    # The low component is the one whose density peaks at the smaller expression.
    mode1 = normax_mode(mu[0], sigma[0], n)
    mode2 = normax_mode(mu[1], sigma[1], n)
    if mode1 > mode2:
        mu = mu[::-1]
        sigma = sigma[::-1]
        w = 1 - w
    
    inds = np.argsort(expression_max)
    inds_inv = np.empty_like(inds, dtype=int)
    inds_inv[inds] = np.arange(len(inds))
    x = expression_max[inds]
    with _enable_x64(True):
        logpdf1 = np.asarray(normax_logpdf(x, mu[0], sigma[0], n=n))
        logpdf2 = np.asarray(normax_logpdf(x, mu[1], sigma[1], n=n))
    lw1 = np.log(w) + logpdf1
    lw2 = np.log1p(-w) + logpdf2
    ws = np.exp(lw1 - np.logaddexp(lw1, lw2))
    if not np.isfinite(ws).all():
        warn('Low-expression filter skipped: the posterior probabilities of the fitted mixture are not finite; '
             f'all {n_total} promoters are kept.')
        return np.ones(n_total, dtype=bool), np.ones(n_total)
    
    degenerate = False
    if float(w) > component_limit:
        ws[:] = 1.0
        warn(f'Low-expression filter skipped: the fitted low-expression component holds {w:.0%} of the promoters '
             f'(more than {component_limit:.0%}), so it cannot be told apart from the bulk of the data; all '
             f'{n_total} promoters are kept.')
    else:
        ws = 1 - ws
        high = ws >= 0.5
        if high.any() and (~high).any():
            if x[high].mean() < x[~high].mean():
                ws = 1 - ws
        else:
            degenerate = True
        j = np.argmax(ws)
        l = np.argmin(ws)
        ws[j:] = 1.0
        ws[:l] = 0.0
    
    ok = np.flatnonzero(ws >= 1.0-cutoff)
    k = int(ok[0]) if len(ok) else len(ws)
    if degenerate:
        which = 'is not the more probable component for any promoter' if high.all() else \
                'is the more probable component for every promoter'
        warn(f'Low-expression filter: the low-expression component of the fitted mixture (weight {w:.1%}) {which}, '
             'so the data have no clearly separated class of unexpressed promoters. Only the promoters of the '
             f'lowest expression were removed ({n_total - (len(ws) - k)} of {n_total}); use --filter-lowexp-w 1 to '
             'keep them all.')
    if fit_plot_filename:
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        import seaborn as sns
        with _enable_x64(True):
            pdf = np.asarray(jnp.exp(logmixture(x, mu, sigma, w, n=n)))
        points = np.array([x, pdf]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        plt.figure(dpi=plot_dpi, )
        sns.histplot(expression_max, stat='density', color='grey')
        lc = LineCollection(segments, cmap='winter')
        lc.set_array(ws)
        lc.set_linewidth(3)
        line = plt.gca().add_collection(lc)
        plt.colorbar(line)
        plt.xlabel('Standardized expression')
        plt.tight_layout()
        plt.savefig(fit_plot_filename)
        plt.close()
    ws = ws[inds_inv]
    inds = np.ones(len(expression), dtype=bool)
    inds[:k] = False
    # print(inds)
    # inds[:] = 1
    inds = inds[inds_inv]
    return inds, ws
