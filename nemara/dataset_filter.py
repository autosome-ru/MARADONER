import jax.numpy as jnp
from jax import jit, grad
from jax.scipy.stats import norm
from jax.scipy.special import logsumexp, logit, expit
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from functools import partial


def normax_logpdf(x: jnp.ndarray, mu: float, sigma: float, n: int):
    x = (x - mu) / sigma
    return jnp.log(n) - jnp.log(sigma) + norm.logpdf(x) + (n - 1) * norm.logcdf(x)

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

def filter_lowexp(expression: pd.DataFrame, cutoff=0.95, fit_plot_filename=None, plot_dpi=200):
    expression = (expression - expression.mean()) / expression.std()
    
    expression_max = expression.max(axis=1).values
    
    mu = [-1.0, 0.0]
    sigmas = [1.0, 1.0]
    w = [0.5]
    x0 = jnp.array(mu + sigmas + w)
    x0 = transform(x0, False)
    fun = jit(partial(loglik, x=expression_max, n=expression.shape[1]))
    jac = jit(grad(fun))
    res = minimize(fun, x0, jac=jac)
    
    params = transform(res.x)
    mu = params[:2]
    sigma = params[2:4]
    w = params[-1]
    
    mode1 = minimize(lambda x: -normax_logpdf(x, mu[0], sigma[0], n=expression.shape[1]), x0=[0.0]).x
    mode2 = minimize(lambda x: -normax_logpdf(x, mu[1], sigma[1], n=expression.shape[1]), x0=[0.0]).x
    if mode1 > mode2:
        mu = mu[::-1]
        sigma = sigma[::-1]
        w = 1 - w
    
    inds = np.argsort(expression_max)
    inds_inv = np.empty_like(inds, dtype=int)
    inds_inv[inds] = np.arange(len(inds))
    x = expression_max[inds]
    logpdf1 = normax_logpdf(x, mu[0], sigma[0], n=expression.shape[1])
    logpdf2 = normax_logpdf(x, mu[1], sigma[1], n=expression.shape[1])
    pdf1 = jnp.exp(logpdf1)
    pdf2 = jnp.exp(logpdf2)
    ws = np.array(pdf1 / ((w * pdf1 + (1-w)*pdf2)) * w)
    # _inds = ws >= 0.5
    # if np.sum(pdf1 * x) >= np.sum(pdf2 * x):
    #     ws = 1 - ws
    j = np.argmax(ws)
    ws[:j] = 1.0
    l = np.argmin(ws)
    ws[l:] = 0.0
    for k in range(j, len(ws)):
        if ws[k] < cutoff:
            break
    if fit_plot_filename:
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        import seaborn as sns
        pdf = jnp.exp(logmixture(x, mu, sigma, w, n=expression.shape[1]))
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
    
    ws = ws[inds_inv]
    inds = np.zeros(len(expression), dtype=bool)
    inds[k:] = True
    # inds[:] = 1
    inds = inds[inds_inv]
    return inds, 1.0 - ws
