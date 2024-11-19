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
    def __init__(self, fun, grad, init_method='SLSQP', methods=('SLSQP', 'L-BFGS-B'), num_steps_momentum=500,
                 reparam='square', scaling_set=None, scaling_factors=(1.0,),
                 momentum_lrs=(1e-2, 1e-3, 1e-4)):
        self.init_method = init_method
        self.scipy_methods = methods
        self.num_steps_momentum = num_steps_momentum
        self.reparam = reparam
        self._fun = fun
        self._grad = grad
        self.scaling_set = scaling_set
        self.scaling_factors = scaling_factors
        self.lrs = momentum_lrs
        self.fun_scale = 1

    def _reparam(self, x, scale_factor=1):
        x = jnp.array(x)
        if self.reparam == 'abs':
            x = jnp.abs(x)
        elif self.reparam == 'square':
            x = x ** 2
        if self.scaling_set is None:
            x = x * scale_factor
        else:
            x = x.at[self.scaling_set].multiply(scale_factor)
        return x
    
    def _inverse_reparam(self, x, scale_factor=1):
        x = jnp.array(x)
        if self.reparam == 'abs':
            x = jnp.abs(x)
        elif self.reparam == 'square':
            x = jnp.abs(x) ** 0.5
        if self.scaling_set is None:
            x = x / scale_factor
        else:
            x = x.at[self.scaling_set].divide(scale_factor)
        return x

    # @jax.jit
    def grad(self, x, scale_factor=1):
        g = self._grad(self._reparam(x, scale_factor=scale_factor))
        rg = jax.jacrev(self._reparam, argnums=0)
        return g * rg(x, scale_factor=scale_factor).sum(axis=0) * self.fun_scale
    
    # @jax.jit
    def fun(self, x, scale_factor=1):
        return self._fun(self._reparam(x, scale_factor=1)) * self.fun_scale
    
    def scipy_optimize(self, x0, methods: list, max_iter=1000):
        if type(methods) is str:
            methods = [methods]
        best_sol = None
        best_scale = None
        for method in methods:
            for scale in self.scaling_factors:
                fun = partial(self.fun, scale_factor=scale)
                grad = partial(self.grad, scale_factor=scale)
                sol = minimize(fun, x0=self._inverse_reparam(x0, scale), method=method, jac=grad, 
                               options={'maxiter': max_iter})
                if best_sol is None or best_sol.fun > sol.fun:
                    best_sol = sol
                    best_scale = scale
        best_sol.x = self._reparam(best_sol.x, best_scale)
        best_sol.fun /= self.fun_scale
        return best_sol
    
    def momentum_optimize(self, x0):
        lrs = self.lrs
        best_x = x0
        for scale in self.scaling_factors:
            fun = partial(self.fun, scale_factor=scale)
            grad = partial(self.grad, scale_factor=scale)
            x = self._inverse_reparam(best_x, scale_factor=scale)
            best_fun = fun(x)
            if self.num_steps_momentum <= 0:
                return x0, best_fun / self.fun_scale
            for j, lr in enumerate(lrs):
                opt_init, opt_update, get_params = rmsprop(lr)
                opt_state = opt_init(x)
                prev_x = self._inverse_reparam(best_x)
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
                            best_x = self._reparam(x, scale_factor=scale)
                            # best_scale = scale
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
        best_sol = None
        best_scale = None
        for scale in (1.0, 1e1, 1e2, 1e3):
            self.fun_scale = 1 / start_loglik  * scale
            sol = self.scipy_optimize(x0, self.init_method, max_iter=20)
            if (np.isfinite(sol.fun) and best_sol is None) or (sol.fun < best_sol.fun):
                best_sol = sol
                best_scale = scale
        if best_sol is None:
            raise Exception('Numerical error in optimization')
        init_loglik = best_sol.fun
        x = best_sol.x
        self.fun_scale = 1 / init_loglik * best_scale
        x, momentum_loglik = self.momentum_optimize(x)
        self.fun_scale = 1 / momentum_loglik * best_scale
        sol = self.scipy_optimize(x, methods=self.scipy_methods)
        x = sol.x
        grad_norm = np.linalg.norm(sol.jac)
        loglik = sol.fun
        return OptimizerResult(start_loglik, init_loglik, momentum_loglik, loglik , grad_norm, x)
        
        