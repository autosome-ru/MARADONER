from scipy.optimize import minimize
from jax.example_libraries.optimizers import rmsprop
from functools import partial
from dataclasses import dataclass
import jax.numpy as jnp
import numpy as np
import jax

@dataclass
class OptimizerResult:
    start_loglik: float
    init_loglik: float
    momentum_loglik: float
    fun: float
    grad_norm: float
    x: np.ndarray
    
    def __str__(self):
        res = [ f'Current objective: {self.fun:.4f}',
               f'Objective at a start: {self.start_loglik:.4f}', 
               f'Objective after a warm-up: {self.init_loglik:.4f}', 
               f'Objective after momentum optimization: {self.momentum_loglik:.4f}', 
               f'Current gradient norm: {self.grad_norm:.4e}'
              ]
        return '\n'.join(res)
        

class MetaOptimizer():
    def __init__(self, fun, grad, init_method='L-BFGS-B', methods=('TNC',), num_steps_momentum=60,
                 reparam='square', scaling_set=None,  skip_init=False,
                 momentum_lrs=(1e-2, 1e-3, 1e-4)):
        self.init_method = init_method
        self.scipy_methods = methods
        self.num_steps_momentum = num_steps_momentum
        self.reparam = reparam
        self._fun = fun
        self._grad = grad
        self.scaling_set = scaling_set
        self.lrs = momentum_lrs
        self.param_scalers = 1
        self.fun_scale = 1
        self.skip_init = skip_init

    def _reparam(self, x, param_scalers=1):
        x = jnp.array(x)
        x = x / param_scalers
        if self.reparam == 'abs':
            x = jnp.abs(x)
        elif self.reparam == 'square':
            x = x ** 2
        return x
    
    def _inverse_reparam(self, x, param_scalers=1):
        x = jnp.array(x)
        if self.reparam == 'abs':
            x = jnp.abs(x)
        elif self.reparam == 'square':
            x = jnp.abs(x) ** 0.5
        x = x * param_scalers
        return x

    def grad(self, x):
        g = self._grad(self._reparam(x, param_scalers=self.param_scalers))
        rg = jax.jacrev(self._reparam, argnums=0)
        return g * rg(x, 
                      param_scalers=self.param_scalers).sum(axis=0) * self.fun_scale
    
    def fun(self, x):
        return self._fun(self._reparam(x, 
                                       param_scalers=self.param_scalers)) * self.fun_scale
    
    def scipy_optimize(self, x0, methods: list, max_iter=None):
        if type(methods) is str:
            methods = [methods]
        best_sol = None
        for method in methods:
            fun = partial(self.fun)
            grad = partial(self.grad)
            sol = minimize(fun,
                           x0=self._inverse_reparam(x0, param_scalers=self.param_scalers),
                           method=method, jac=grad, 
                           options={'maxiter': max_iter} if max_iter else None)
            if best_sol is None or np.isnan(best_sol.fun) or best_sol.fun > sol.fun:
                best_sol = sol
        best_sol.x = self._reparam(best_sol.x, param_scalers=self.param_scalers)
        best_sol.fun /= self.fun_scale
        return best_sol
    
    def momentum_optimize(self, x0):
        lrs = self.lrs
        best_x = x0
        fun = partial(self.fun)
        grad = partial(self.grad)
        x = self._inverse_reparam(best_x,
                                  param_scalers=self.param_scalers)
        best_fun = fun(x)
        if self.num_steps_momentum <= 0:
            return x0, best_fun / self.fun_scale
        for j, lr in enumerate(lrs):
            opt_init, opt_update, get_params = rmsprop(lr)
            opt_state = opt_init(x)
            prev_x = self._inverse_reparam(best_x,
                                           param_scalers=self.param_scalers)
            n = 0
            n_no_change = 0
            while n < self.num_steps_momentum and n_no_change < 3:
                x = get_params(opt_state)
                opt_state = opt_update(n, grad(x), opt_state)
                n += 1
                if not n % 10:
                    lf = fun(x)
                    if best_fun is None or lf < best_fun:
                        best_fun = lf
                        best_x = self._reparam(x,
                                               param_scalers=self.param_scalers)
                if not n % 20:
                    if jnp.abs((x - prev_x) / x).max() > 5e-2:
                        n_no_change = 0
                    else:
                        n_no_change += 1
                    prev_x = x
        return best_x, best_fun / self.fun_scale


    def optimize(self, x0, ):
        x0 = jnp.array(x0)
        self.fun_scale = 1.0
        start_loglik = self.fun(x0)
        if not jnp.isfinite(start_loglik):
            raise ValueError("Can't compute loglikelihood at a starting point.")
        best_sol = None
        best_scale = None
        best_fun = start_loglik
        if not self.skip_init:
            for scale in (1.0, 1e1, 1e2):
                self.fun_scale = np.abs(1 / start_loglik  * scale)
                sol = self.scipy_optimize(x0, self.init_method, max_iter=10)
                if sol is None:
                    continue
                if np.isfinite(sol.fun) and sol.fun < best_fun:
                    best_sol = sol
                    best_scale = scale
                    best_fun = sol.fun
            if best_sol is None:
                raise ValueError('Numerical error in optimization')
        else:
            best_fun = float('inf')
        
        if best_fun > start_loglik:
            x = x0
            self.fun_scale = 1
            best_scale = 1
            init_loglik = start_loglik
        else:
            init_loglik = best_sol.fun
            x = best_sol.x
        if not self.scaling_set:
            self.param_scalers = np.abs(x).mean()
        else:
            self.param_scalers = np.ones_like(x)
            for slc in self.scaling_set:
                self.param_scalers[slc] = np.max(np.abs(x)[slc])
        self.fun_scale = np.abs(1 / init_loglik * best_scale)
        x, momentum_loglik = self.momentum_optimize(x)
        xm = x.copy()
        self.fun_scale = np.abs(1 / momentum_loglik * best_scale)
        if not self.scaling_set:
            self.param_scalers = np.abs(x).mean()
        else:
            self.param_scalers = np.ones_like(x)
            for slc in self.scaling_set:
                self.param_scalers[slc] = np.max(np.abs(x)[slc])
        sol = self.scipy_optimize(x, methods=self.scipy_methods)
        if np.isnan(sol.fun) or sol.fun > momentum_loglik:
            sol.fun = momentum_loglik
            sol.x = xm
            print('Eggog')
            print(sol)
        grad_norm = np.linalg.norm(sol.jac)
        loglik = sol.fun
        x = sol.x
        return OptimizerResult(start_loglik, init_loglik, momentum_loglik, loglik , grad_norm, x)
        
        