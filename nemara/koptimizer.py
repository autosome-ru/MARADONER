import jax.numpy as jnp
from jax.scipy.linalg import solve, cho_factor, cho_solve, cholesky
from jax import jit, grad, hessian
from jax.example_libraries.optimizers import optimizer, make_schedule
from functools import partial
from dataclasses import dataclass
from sklearn.model_selection import RepeatedKFold
from scipy.optimize import minimize

def rank_penalty(sigmas, temp: float, prior: jnp.ndarray):
    if prior is not None:
        sigmas = sigmas.at[jnp.argsort(prior)].get()
    sigmas = sigmas.reshape(-1,1)
    sigmas = (sigmas - sigmas.T) / (sigmas + sigmas.T)
    temp = 1e2
    sigmas = jnp.tanh(sigmas * temp)
    sigmas = sigmas.at[jnp.tril_indices_from(sigmas)].multiply(-1)
    ranks = jnp.arange(len(sigmas)).reshape(-1, 1)
    ranks = jnp.abs(ranks.T - ranks) / len(sigmas)
    sigmas = sigmas * ranks
    return sigmas.sum() 

rank_penalty_grad = grad(rank_penalty, argnums=0)
rank_penalty_hess = hessian(rank_penalty, argnums=0)


def apply_prior(loglik: float, sigmas, c, prior, alpha, temp=1., penalty='l2-prior'):
    if not penalty:
        return loglik
    if penalty == 'l2-prior':
        penalty = ((c * sigmas - prior) ** 2).sum() 
    elif penalty == 'l2':
        penalty = ((sigmas - c) ** 2).sum()
    elif penalty == 'ranked':
        penalty = rank_penalty(sigmas, temp=temp, prior=prior)
    return loglik + alpha * penalty

def apply_prior_grad(g: jnp.ndarray, sigmas, c, prior, alpha, penalty='l2-prior', sigma_u=1.0, temp=1.):
    if not penalty:
        return g
    if penalty == 'l2-prior':
        reg_s = 2 * alpha * (c * sigmas - prior) 
        c_term = (reg_s * sigmas).sum()
        reg_s = reg_s * (c * sigma_u)
    elif penalty == 'l2':
        reg_s = 2 * alpha * (sigmas - c) * sigma_u
        c_term = None
    elif penalty == 'ranked':
        reg_s = alpha * rank_penalty_grad(sigmas, temp=temp, prior=prior)
        c_term = None
    if c_term is not None:
        g = jnp.append(g + reg_s, (c_term))
    else: 
        g = g + reg_s
    return g

def apply_prior_hessian(hess: jnp.ndarray, sigmas, c, prior, alpha, penalty='l2-prior', diag=False, sigma_u=1.0, temp=1.):
    if not penalty:
        return hess
    if diag:
        if penalty == 'l2-prior':
            diag = 2 * alpha * c ** 2 * (sigma_u ** 2)
            c_term = 2 * alpha * (sigmas ** 2).sum()
        elif penalty == 'l2':
            diag = 2 * alpha * (sigma_u ** 2)
            c_term = None
        elif penalty == 'ranked':
            diag = 0
            c_term = None
            hess = hess + alpha * rank_penalty_hess(sigmas, temp=temp, prior=prior).diagonal()
        if c_term is not None:
            return jnp.append(hess + diag, c_term)
        return hess + diag
    inds = jnp.diag_indices_from(hess)
    if penalty == 'l2-prior':
        c_row = 2 * alpha * (2 * c * sigmas - prior) * sigma_u
        diag = 2 * alpha * c ** 2 * (sigma_u ** 2)
        c_term = 2 * alpha * (sigmas ** 2).sum()
    elif penalty == 'l2':
        diag = 2 * alpha * (sigma_u ** 2)
        c_term = None
    elif penalty == 'ranked':
        diag = 0
        c_term = None
        hess = hess + alpha * rank_penalty_hess(sigmas, temp=temp, prior=prior)
    if c_term is not None:
        hess = jnp.vstack((hess.at[inds].add(diag), c_row))
        hess = jnp.hstack((hess, jnp.append(c_row, c_term).reshape(-1,1)))
    else:
        hess = hess.at[inds].add(diag)
    return hess

def extract_params(params, penalty: str):
    if penalty in {'l2-prior',}:
        return params.at[:-1].get(), params.at[-1].get()
    return params, 0
    

def calc_sigma(sigmas, B, sigma_g):
    return B * sigmas.reshape((1,-1)) @ B.T + jnp.identity(len(B)) * sigma_g

def loglik(params, X, B, sigma_g, prior=None, alpha=1.0, penalty='l2-prior', sigma_u=1, temp=1.):
    res = 0
    sigmas, c = extract_params(params, penalty)
    sigmas = sigmas * sigma_u
    sigma = calc_sigma(sigmas, B, sigma_g)
    # sigma = jnp.linalg.inv(sigma)
    sigma = cho_factor(sigma)
    res = jnp.trace(cho_solve(sigma, X @ X.T))
    sigma = sigma[0]
    res += 2 * X.shape[1] * jnp.log(sigma.diagonal()).sum()
    res = apply_prior(res, sigmas, c, prior, alpha * X.shape[1], penalty=penalty, temp=temp)
    return res

def loglik_grad(params, X, B, sigma_g, prior=None, alpha=1.0, penalty='l2-prior'):
    sigmas, c = extract_params(params, penalty)
    sigma = calc_sigma(sigmas, B, sigma_g)
    Z = solve(sigma, B, assume_a='pos')
    t = Z.T @ X
    g = -jnp.einsum('ji,ji->j', t, t) + X.shape[1] * jnp.einsum('ji,ji->i', B, Z)
    g = apply_prior_grad(g, sigmas, c, prior, alpha * X.shape[1], penalty=penalty)
    return g
    
def _calc_Z(sigmas, B, sigma_g, aux_btb=None, b_factors=None):
    sqrt_sigmas = sigmas ** 0.5
    if b_factors is None:
        B_hat = B * sqrt_sigmas
        if aux_btb is None:
            sigma = jnp.linalg.inv(B_hat.T @ B_hat + jnp.identity(B.shape[1]) * sigma_g)
        else:
            sigma = jnp.linalg.inv(sqrt_sigmas.reshape(-1,1) * aux_btb * sqrt_sigmas + jnp.identity(B.shape[1]) * sigma_g)
        Z = B_hat @ sigma * (1 / sqrt_sigmas)
    else:
        W, H = b_factors
        H_hat = H * sqrt_sigmas
        L = cholesky(H_hat @ H_hat.T)
        A = W @ L
        sigma = jnp.linalg.inv(A.T @ A - jnp.identity(L.shape[0]) * sigma_g)
        Z = A @ sigma @ cho_solve((L, False), H)
    return Z

def loglik_hess(params, X, B, sigma_g, sigma_u=1.0, prior=None, alpha=1.0, penalty='l2-prior', return_grad=False,
                aux_btb=None, b_factors=None, diagonal_only=False, temp=1.):
    sigmas, c = extract_params(params, penalty)
    sigmas *= sigma_u
    Z = _calc_Z(sigmas, B, sigma_g, aux_btb, b_factors)
    V = Z.T @ X
    if not diagonal_only:
        Gamma = Z.T @ B
        VV = V @ V.T
        A = 2 * VV - X.shape[1] * Gamma
        hess = A * Gamma * (sigma_u ** 2)
        hess = apply_prior_hessian(hess, sigmas, c, prior, alpha * X.shape[1], penalty=penalty, sigma_u=sigma_u, temp=temp)
        if return_grad:
            g = Gamma.diagonal() * X.shape[1] - VV.diagonal()
            g = g * sigma_u
            g = apply_prior_grad(g, sigmas, c, prior, alpha * X.shape[1], penalty=penalty, sigma_u=sigma_u, temp=temp)
            return hess, g
        return hess
    else:
        vv_diag = jnp.einsum('ij,ij->i', V, V)
        gamma_diag = jnp.einsum('ij,ij->j', Z, B)
        hess = (2 * vv_diag - X.shape[1] * gamma_diag) * gamma_diag * (sigma_u ** 2)
        hess = apply_prior_hessian(hess, sigmas, c, prior, alpha * X.shape[1], penalty=penalty, diag=True, sigma_u=sigma_u, temp=temp)
        if return_grad:
            g = X.shape[1] * gamma_diag - vv_diag 
            g = g * sigma_u
            g = apply_prior_grad(g, sigmas, c, prior, alpha * X.shape[1], penalty=penalty, sigma_u=sigma_u, temp=temp)
            return hess, g
        return hess

def sufficient_not_pd(X) -> bool:
    return jnp.any(jnp.abs(X).sum(axis=1) - jnp.abs(X.diagonal()) + X.diagonal() <= 0)

@optimizer
def rmsprop(step_size, gamma=0.9, eps=1e-8):
  """Construct optimizer triple for RMSProp.

  Args:
    step_size: positive scalar, or a callable representing a step size schedule
      that maps the iteration index to a positive scalar.
      gamma: Decay parameter.
      eps: Epsilon parameter.

  Returns:
    An (init_fun, update_fun, get_params) triple.
  """
  step_size = make_schedule(step_size)
  def init(x0):
    avg_sq_grad = jnp.zeros_like(x0)
    return x0, avg_sq_grad
  def update(i, g, state):
    x, avg_sq_grad = state
    avg_sq_grad = avg_sq_grad * gamma + jnp.square(g) * (1. - gamma)
    # x = jnp.clip(x - step_size(i) * g / jnp.sqrt(avg_sq_grad + eps), left_bound)
    x = jnp.abs(x - step_size(i) * g / jnp.sqrt(avg_sq_grad + eps))
    return x, avg_sq_grad
  def get_params(state):
    x, _ = state
    return x
  return init, update, get_params

@dataclass
class OptimizerResult:
    regul_param: bool
    n_iter: int
    cur_obj: float
    best_obj: float
    grad_norm: float
    rel_change: float
    step_size: float
    best_x: jnp.ndarray
    scale_factor: float
    
    def __str__(self):
        res = [      f'Iteration: {self.n_iter}', f'Current objective: {self.cur_obj:.4f}',
               f'Best objective so far: {self.best_obj:.4f}', 
               'Current gradient norm: [NA]' if self.grad_norm is None else f'Current gradient norm: {self.grad_norm:.4e}',
               f'Crurrent step size: {self.step_size:.4e}' if self.step_size is not None else 'Current step size: [NA]',
               'Max. relative change in parameters values: [NA]' if self.rel_change is None else f'Max. relative change in parameters values: {self.rel_change:.4f}']
        return '\n'.join(res)
    
    def get_params(self):
        if self.regul_param:
            return (self.best_x[:-1] * self.scale_factor) ** 0.5
        return (self.best_x * self.scale_factor) ** 0.5
    

class KOptimizer():
    def __init__(self, B, B_factors=None,
                 n_step_iters=10, max_iter=500, hotstart_iters=500, gtol=1e-6, 
                 penalty=None, alpha=1e-2, prior=None,
                 hess_stabilizer=1e-4, linesearch_divisor=5, linesearch_post_mult=2,
                 stochastic_der=False, der_n_splits=5, der_n_repeats=2,
                 stochastic_loglik=True, loglik_n_splits=10, loglik_n_repeats=2,
                 ):
        if not stochastic_der and B_factors is None:
            self.aux_btb = B.T @ B
        else:
            self.aux_btb = None
        self.B = B
        self.B_factors = B_factors
        self.linesearch_divisor = linesearch_divisor
        self.linesearch_post_mult = linesearch_post_mult
        self.n_step_iters = n_step_iters; self.hostart_iters = hotstart_iters
        self.max_iter = max_iter; self.gtol = gtol
        self.penalty = penalty; self.alpha = alpha; self.prior = prior 
        self.hess_stabilizer = hess_stabilizer
        self.stochastic_der = stochastic_der
        self.rkf_der = RepeatedKFold(n_splits=der_n_splits, n_repeats=der_n_repeats, random_state=1)
        self.stochastic_loglik = stochastic_loglik
        self.rkf_loglik = RepeatedKFold(n_splits=loglik_n_splits, n_repeats=loglik_n_repeats, random_state=1)
        self.x0 = jnp.ones(B.shape[1], dtype=float) 
        if penalty in ('l2-prior', ):
            self.x0 = jnp.append(self.x0, 1.0)
        self._hessg = jit(partial(loglik_hess, return_grad=True), static_argnames=('penalty', 'return_grad', 
                                                                                   'diagonal_only',))
        self._loglik = jit(loglik, static_argnames=('penalty',))
    
    def hess_grad_fun(self, params, X, sigma_g: float, prior=None, alpha=None, penalty=None, sigma_u=1,
                      hess_diag=False):
        if prior is None:
            prior = self.prior
        if alpha is None:
            alpha = self.alpha
        if penalty is None:
            penalty = self.penalty
        hessg = partial(self._hessg, prior=prior, alpha=alpha, penalty=penalty, sigma_g=sigma_g, 
                        diagonal_only=hess_diag, sigma_u=sigma_u)
        B = self.B
        if not self.stochastic_der:
            h, g =  hessg(params, X=X, B=self.B, aux_btb=self.aux_btb, b_factors=self.B_factors)
        else:
            h = 0
            g = 0
            n = 0
            for _, inds in self.rkf_der.split(X):
                h_, g_ = hessg(params, X=X[inds], B=B[inds])
                h += h_
                g += g_
                n += 1
            h /= n
            g /= n
        return h, g
    
    def objective(self, params, X, sigma_g: float, prior=None, alpha=None, penalty=None, sigma_u=1, subset=None):
        if prior is None:
            prior = self.prior
        if alpha is None:
            alpha = self.alpha
        if penalty is None:
            penalty = self.penalty
        logl = partial(self._loglik, prior=prior, alpha=alpha, penalty=penalty, sigma_g=sigma_g, sigma_u=sigma_u)
        B = self.B
        if subset is not None:
            obj = logl(params, X=X[subset], B=B[subset])
        elif not self.stochastic_loglik:
            obj = logl(params, X=X, B=self.B)
        else:
            obj = 0
            n = 0
            for _, inds in self.rkf_loglik.split(X):
                obj += logl(params, X=X[inds], B=B[inds])
                n += 1
            obj /= n
        return obj

    def compute_subset(self, params, X, sigma_g: float, prior=None, alpha=None, penalty=None, sigma_u=1):
        if prior is None:
            prior = self.prior
        if alpha is None:
            alpha = self.alpha
        if penalty is None:
            penalty = self.penalty
        B = self.B
        logl = partial(self._loglik, prior=prior, alpha=alpha, penalty=penalty, sigma_g=sigma_g, sigma_u=sigma_u)
        objs = list()
        subsets = list()
        for _, inds in self.rkf_linesearch.split(X):
            objs.append(logl(params, X=X[inds], B=B[inds]))
            subsets.append(inds)
        objs = jnp.array(objs)
        objs = jnp.abs(objs - objs.mean())        
        return subsets[jnp.argmin(objs)]
    
    def newtonian_search(self, params, p, step_size: float, X, sigma_g, prior, alpha, penalty, g=None, h=None):
        num_steps = 1
        safety_step_size = self.step_size
        
        p_pos = p >= 0
        params_eps = params - 1e-10
        if jnp.all(p_pos):
            max_step_size = (params_eps / p).min()
            min_step_size = None
        elif jnp.all(~p_pos):
            min_step_size = (params_eps / p).max()
            max_step_size = None
        else:
            max_step_size = (params_eps[p_pos] / p[p_pos]).min()
            min_step_size = (params_eps[~p_pos] / p[~p_pos]).max()
        if step_size == 0:
            if len(h.shape) > 1:
                h = h.diagonal()
            step_size = ((p * g) ** 2).sum() / (2 * p ** 3 * g * h).sum()
            if step_size < 0:
                step_size = safety_step_size
            step_size = jnp.clip(step_size, min_step_size, max_step_size)
            num_steps -= 1
        hessg = partial(self.hess_grad_fun, X=X, sigma_g=sigma_g, prior=prior, alpha=alpha, penalty=penalty)
        while num_steps:
            h, g = hessg(params - step_size * p)
            step_size = ((p * g) ** 2).sum() / (2 * p ** 3 * g * h).sum()
            if step_size < 0:
                step_size = safety_step_size
            step_size = jnp.clip(step_size, min_step_size, max_step_size)
            num_steps -= 1
        return step_size
    
    def curvature_search(self, params, p, step_size: float, X, sigma_g, prior, alpha, penalty):
        hessg = partial(self.hess_grad_fun, X=X, sigma_g=sigma_g, prior=prior, alpha=alpha, penalty=penalty, hess_diag=True)
        f_best = hessg(params)[0].min()
        while step_size > 1e-14:
            f = hessg(jnp.abs(params - step_size * p))[0].min()
            if f < f_best:
                return step_size
            step_size /= self.linesearch_divisor
        return False
    
    def backtracking_search(self, params, p, step_size: float, X, sigma_g, prior, alpha, penalty, subset=None):
        obj_fun = partial(self.objective, X=X, sigma_g=sigma_g, prior=prior, alpha=alpha, penalty=penalty, subset=subset)
        f_best = obj_fun(params)
        while step_size > 1e-14:
            f = obj_fun(jnp.abs(params - step_size * p))
            if f < f_best:
                break
            step_size /= self.linesearch_divisor
        return False
        
    
    def hostart(self, hessg, obj, best_obj=None):
        hessg = partial(hessg, hess_diag=True)
        if best_obj is None:
            best_obj = obj(self.x0)
        lrs = [1e-2, 1e-3, 1e-4]
        x = self.x0
        for j, lr in enumerate(lrs):
            opt_init, opt_update, get_params = rmsprop(lr)
            opt_state = opt_init(x)
            prev_x = x
            n = 0
            n_no_change = 0
            while n < self.hostart_iters and n_no_change < 3:
                x = get_params(opt_state)
                h, g = hessg(x)
                opt_state = opt_update(n, g, opt_state)
                n += 1
                if not n % 20:
                    if jnp.abs((x - prev_x) / x).max() > 5e-2:
                        n_no_change = 0
                    else:
                        n_no_change += 1
                    prev_x = x
            # print(n)
                    
                    
        return x
            
        
    
    def optimize(self, X, sigma_g: float, prior=None, alpha=None, penalty=None, scale_factor=1):

        best_params = self.x0.copy() / scale_factor
        prev_best_params = best_params.copy() * 0
        params = best_params.copy()
        sigma_g = sigma_g / scale_factor
        X = X / scale_factor ** 0.5
        hess_stabilizer = self.hess_stabilizer
        

        n_step_iters = self.n_step_iters
        max_iter = self.max_iter
        gtol = self.gtol
        last_rel_change = None
        
        best_subset = None#self.compute_subset(params, X, sigma_g, prior=prior, alpha=alpha, penalty=penalty)
        obj_fun = partial(self.objective, X=X, sigma_g=sigma_g, prior=prior, alpha=alpha, penalty=penalty)
        hessg = partial(self.hess_grad_fun, X=X, sigma_g=sigma_g, prior=prior, alpha=alpha, penalty=penalty)
        
        best_params = self.hostart(hessg, obj_fun)
        last_rel_change = float(jnp.abs((best_params - params) / best_params).max())
        params = best_params
        
        extra_param = penalty == 'l2-prior'
        step_size = 1
        
        line_search = partial(self.backtracking_search, X=X, sigma_g=sigma_g, prior=prior, alpha=alpha, penalty=penalty, subset=best_subset)
        curvature_search = partial(self.curvature_search, X=X, sigma_g=sigma_g, prior=prior, alpha=alpha, penalty=penalty)
        best_obj = obj_fun(params)
        h, g = hessg(params)
        g_norm = jnp.linalg.norm(g)
        yield OptimizerResult(extra_param, 0, best_obj, best_obj, jnp.linalg.norm(g), last_rel_change, None, best_params, scale_factor=scale_factor)
        n_iter = 1
        while (n_iter < max_iter) and (g_norm > gtol):
            h, g = hessg(params)
            if sufficient_not_pd(h):
                p = g
                pd_hessian = False
                step_size = curvature_search(params, p=p, step_size=step_size * self.linesearch_post_mult * self.linesearch_divisor)
            else:
                eigs, q = jnp.linalg.eigh(h)
                min_eig = eigs.min()
                pd_hessian = min_eig > hess_stabilizer / 10
                eigs = eigs + max(0, hess_stabilizer - min_eig)
                p = (q  / eigs @ q.T) @ g
                if not pd_hessian:
                    step_size = line_search(params, p=p, step_size=1)
                else:
                    step_size = 1
            if not step_size:
                break
            params = jnp.abs(params - step_size * p)
            g_norm = jnp.linalg.norm(g)
            if not n_iter % n_step_iters:
                obj = obj_fun(params)
                if obj < best_obj:
                    best_obj = obj
                    prev_best_params = best_params
                    best_params = params
                    last_rel_change = float(jnp.abs((best_params - prev_best_params) / best_params).max())
                yield OptimizerResult(extra_param, n_iter, obj, best_obj, g_norm, last_rel_change, step_size, best_params, scale_factor=scale_factor)
            n_iter += 1
        if n_iter % n_step_iters:
            obj = obj_fun(params)
            if obj < best_obj:
                best_params = params
                best_obj = obj
                last_rel_change = float(jnp.abs((best_params - prev_best_params) / best_params).max())
            yield OptimizerResult(extra_param, n_iter, obj, best_obj, g_norm,
                                  last_rel_change, step_size, best_params, scale_factor=scale_factor)