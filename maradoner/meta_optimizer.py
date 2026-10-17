from scipy.optimize import minimize
from jax.example_libraries.optimizers import rmsprop
from functools import partial
from dataclasses import dataclass
import jax.numpy as jnp
import numpy as np
import jax
from .utils import InputError, warn

# The optimum is regarded as reached, whatever the optimiser's own status says, when no
# parameter changes the objective (-2 logL) by more than this per unit change of its log.
STATIONARITY_TOL = 1e-2
# Stopping rules of the final refinement, relative to an objective and to parameters that
# are both scaled to order one.
POLISH_FTOL = 1e-13
POLISH_GTOL = 1e-9

@dataclass
class OptimizerResult:
    start_loglik: float
    init_loglik: float
    momentum_loglik: float
    fun: float
    grad_norm: float
    x: np.ndarray
    # Convergence diagnostics of the final optimisation step. `stationarity` is
    # max_i |x_i * df/dx_i|, the largest gradient with respect to the log-parameters,
    # which does not depend on the units of the data.
    success: bool = True
    message: str = ''
    stationarity: float = 0.0
    # (name, objective) after each stage of an optimiser other than this one; when given, they
    # are reported in place of the warm-up and the momentum stages.
    stages: tuple = None

    def __str__(self):
        res = [ f'Current objective: {self.fun:.4f}',
               f'Objective at a start: {self.start_loglik:.4f}',
               f'Objective after a warm-up: {self.init_loglik:.4f}',
               f'Objective after momentum optimization: {self.momentum_loglik:.4f}',
               f'Current gradient norm: {self.grad_norm:.4e}'
              ]
        if self.stages is not None:
            res = res[:2] + [f'Objective after {name}: {value:.4f}' for name, value in self.stages] + res[-1:]
        return '\n'.join(res)


class MetaOptimizer():
    def __init__(self, fun, grad, init_method='L-BFGS-B', methods=('TNC',), num_steps_momentum=60,
                 reparam='square', scaling_set=None,  skip_init=False,
                 momentum_lrs=(1e-2, 1e-3, 1e-4), value_and_grad=None, name=None,
                 warn_convergence=True, polish=False):
        """fun and grad (or value_and_grad, which returns both from a single evaluation
        and is preferred when given) take the parameter vector as a float64 numpy array.
        name is the estimation stage shown in warnings and error messages.
        polish=True adds a final bound-constrained refinement in the parameters themselves
        (see polish_optimize); it requires a gradient that is valid at zero parameters."""
        self.init_method = init_method
        self.scipy_methods = methods
        self.num_steps_momentum = num_steps_momentum
        self.reparam = reparam
        self._fun = fun
        self._grad = grad
        self._vg = value_and_grad
        self.name = name
        self.warn_convergence = warn_convergence
        self.polish = polish
        self.scaling_set = scaling_set
        self.lrs = momentum_lrs
        self.param_scalers = 1
        self.fun_scale = 1
        self.skip_init = skip_init
        self._cache_key = None
        self._cache_val = None
        self._cache_grad = None

    def _stage(self):
        return f'Estimation of {self.name}' if self.name else 'Optimisation'

    def _call(self, f, x):
        try:
            return f(x)
        except InputError:
            raise
        except Exception as e:
            raise RuntimeError(f'{self._stage()} failed while evaluating the objective '
                               f'({type(e).__name__}: {e}).') from e

    def _cached(self, x, need_grad: bool):
        # scipy asks for the objective and for its gradient at the same point in separate
        # calls; the last evaluation is kept so that the two share their computations.
        x = np.asarray(x, dtype=np.float64)
        key = x.tobytes()
        if key != self._cache_key:
            self._cache_key = key
            self._cache_val = None
            self._cache_grad = None
        if self._vg is not None:
            if self._cache_val is None or self._cache_grad is None:
                val, grad = self._call(self._vg, x)
                self._cache_val = float(val)
                self._cache_grad = np.asarray(grad, dtype=np.float64)
        elif need_grad:
            if self._cache_grad is None:
                self._cache_grad = np.asarray(self._call(self._grad, x), dtype=np.float64)
        elif self._cache_val is None:
            self._cache_val = float(self._call(self._fun, x))
        return self._cache_grad if need_grad else self._cache_val

    def _reparam(self, x, param_scalers=1):
        x = np.array(x, dtype=np.float64)
        x = x / param_scalers
        if self.reparam == 'abs':
            x = np.abs(x)
        elif self.reparam == 'square':
            x = x ** 2
        return x

    def _inverse_reparam(self, x, param_scalers=1):
        x = np.array(x, dtype=np.float64)
        if self.reparam == 'abs':
            x = np.abs(x)
        elif self.reparam == 'square':
            x = np.abs(x) ** 0.5
        x = x * param_scalers
        return x

    def _dreparam(self, x, param_scalers=1):
        # Derivative of _reparam: it acts element-wise, so its Jacobian is diagonal.
        x = np.asarray(x, dtype=np.float64)
        if self.reparam == 'abs':
            return np.sign(x) / param_scalers
        elif self.reparam == 'square':
            return 2.0 * x / param_scalers ** 2
        return np.ones_like(x) / param_scalers

    def _calc_scales(self, x):
        # Typical magnitude of the parameters: one for all of them, or one per slice of
        # scaling_set.
        def valid(scale):
            return float(scale) if np.isfinite(scale) and scale > 0 else 1.0
        x = np.abs(np.asarray(x, dtype=np.float64))
        if not self.scaling_set:
            return np.full_like(x, valid(x.mean()))
        scales = np.ones_like(x)
        for slc in self.scaling_set:
            scales[slc] = valid(np.max(x[slc], initial=0.0))
        return scales

    def _calc_scalers(self, x):
        # Scalers s such that the optimiser's variable _inverse_reparam(x, s) is of order
        # one whatever the units of x.
        scales = self._calc_scales(x)
        scalers = scales ** -0.5 if self.reparam == 'square' else 1.0 / scales
        return scalers if self.scaling_set else float(scalers[0]) if len(scalers) else 1.0

    @staticmethod
    def _fun_scale(loglik, scale=1.0):
        loglik = abs(float(loglik))
        if not np.isfinite(loglik) or loglik < 1e-300:
            return float(scale)
        return float(scale) / loglik

    def grad(self, x):
        x = np.asarray(x, dtype=np.float64)
        g = self._cached(self._reparam(x, param_scalers=self.param_scalers), True)
        return g * self._dreparam(x, param_scalers=self.param_scalers) * self.fun_scale

    def fun(self, x):
        return self._cached(self._reparam(x,
                                          param_scalers=self.param_scalers), False) * self.fun_scale

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

    def polish_optimize(self, x0, max_iter=None):
        """L-BFGS-B in the parameters themselves, each divided by its typical magnitude,
        with the bound x >= 0 when the reparametrisation implies it. Unlike the
        optimisers working with square roots, for which zero is always a stationary
        point, it can both reach the bound exactly and leave it."""
        x0 = np.asarray(x0, dtype=np.float64)
        scales = self._calc_scales(x0)
        fun_scale = self._fun_scale(self._cached(x0, False))

        def fun(u):
            return self._cached(u * scales, False) * fun_scale

        def grad(u):
            return self._cached(u * scales, True) * scales * fun_scale
        bounds = [(0.0, None)] * len(x0) if self.reparam in ('square', 'abs') else None
        if max_iter is None:
            max_iter = max(500, 2 * len(x0))
        sol = minimize(fun, x0 / scales, jac=grad, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': max_iter, 'maxfun': 2 * max_iter,
                                'ftol': POLISH_FTOL, 'gtol': POLISH_GTOL})
        if bounds is not None:
            # projected gradient: a positive derivative at the bound is not a defect
            sol.jac = np.where((sol.x <= 0) & (sol.jac > 0), 0.0, sol.jac)
        sol.x = sol.x * scales
        sol.fun /= fun_scale
        return sol

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
        # RMSProp with the constants of jax.example_libraries.optimizers.rmsprop
        gamma, eps = 0.9, 1e-8
        # For every learning rate: the best point is recorded every 10 steps, and the rate
        # is abandoned after 3 consecutive windows of 20 steps in which no parameter moved
        # by more than 5 %.
        for j, lr in enumerate(lrs):
            avg_sq_grad = np.zeros_like(x)
            prev_x = self._inverse_reparam(best_x,
                                           param_scalers=self.param_scalers)
            n = 0
            n_no_change = 0
            while n < self.num_steps_momentum and n_no_change < 3:
                g = grad(x)
                if not np.all(np.isfinite(g)):
                    break
                x_cur = x
                avg_sq_grad = avg_sq_grad * gamma + np.square(g) * (1.0 - gamma)
                x = x - lr * g / np.sqrt(avg_sq_grad + eps)
                n += 1
                if not n % 10:
                    lf = fun(x_cur)
                    if best_fun is None or lf < best_fun:
                        best_fun = lf
                        best_x = self._reparam(x_cur,
                                               param_scalers=self.param_scalers)
                if not n % 20:
                    with np.errstate(divide='ignore', invalid='ignore'):
                        change = np.abs((x_cur - prev_x) / x_cur)
                    if np.nanmax(change) > 5e-2:
                        n_no_change = 0
                    else:
                        n_no_change += 1
                    prev_x = x_cur
            x = x_cur if n else x
        return best_x, best_fun / self.fun_scale


    def optimize(self, x0, ):
        x0 = np.array(x0, dtype=np.float64)
        self.param_scalers = 1
        self.fun_scale = 1.0
        start_loglik = self._cached(x0, False)
        if not np.isfinite(start_loglik):
            raise RuntimeError(f'{self._stage()} failed: the objective is not finite '
                               f'({start_loglik}) at the starting point.')
        best_sol = None
        best_scale = None
        best_fun = start_loglik
        if not self.skip_init:
            for scale in (1.0, 1e1, 1e2):
                self.fun_scale = self._fun_scale(start_loglik, scale)
                sol = self.scipy_optimize(x0, self.init_method, max_iter=10)
                if sol is None:
                    continue
                if np.isfinite(sol.fun) and sol.fun <= best_fun and np.all(np.isfinite(sol.x)):
                    best_sol = sol
                    best_scale = scale
                    best_fun = sol.fun
        else:
            best_fun = float('inf')

        if best_sol is None or best_fun > start_loglik:
            x = x0
            self.fun_scale = 1
            best_scale = 1
            init_loglik = start_loglik
        else:
            init_loglik = best_sol.fun
            x = best_sol.x
        self.param_scalers = self._calc_scalers(x)
        self.fun_scale = self._fun_scale(init_loglik, best_scale)
        x, momentum_loglik = self.momentum_optimize(x)
        xm = x.copy()
        self.fun_scale = self._fun_scale(momentum_loglik, best_scale)
        self.param_scalers = self._calc_scalers(x)
        sol = self.scipy_optimize(x, methods=self.scipy_methods)
        success = bool(getattr(sol, 'success', True))
        message = str(getattr(sol, 'message', ''))
        if not np.isfinite(sol.fun) or sol.fun > momentum_loglik or not np.all(np.isfinite(sol.x)):
            sol.fun = momentum_loglik
            sol.x = xm
            sol.jac = self.grad(self._inverse_reparam(xm, param_scalers=self.param_scalers))
            success = False
            message = 'the last optimisation step could not improve the objective' + \
                      (f' ({message})' if message else '')
        if self.polish:
            pol = self.polish_optimize(sol.x)
            if np.isfinite(pol.fun) and pol.fun <= sol.fun and np.all(np.isfinite(pol.x)):
                sol.fun = pol.fun
                sol.x = pol.x
                sol.jac = pol.jac
                # status 1 of L-BFGS-B: iteration / evaluation limit
                success = bool(pol.success)
                message = str(pol.message)
        grad_norm = np.linalg.norm(sol.jac)
        loglik = float(sol.fun)
        x = np.asarray(sol.x, dtype=np.float64)
        if not np.isfinite(loglik) or not np.all(np.isfinite(x)):
            raise RuntimeError(f'{self._stage()} failed: the optimiser returned a non-finite '
                               'objective or non-finite parameters.')
        stationarity = float(np.max(np.abs(x * self._cached(x, True)), initial=0.0))
        if not np.isfinite(stationarity):
            stationarity = float('inf')
        res = OptimizerResult(start_loglik, init_loglik, momentum_loglik, loglik , grad_norm, x,
                              success=success or stationarity <= STATIONARITY_TOL,
                              message=message, stationarity=stationarity)
        if self.warn_convergence:
            self.warn_if_not_converged(res)
        return res

    def warn_if_not_converged(self, res: OptimizerResult):
        if res.success:
            return
        warn(f'{self._stage()}: the optimiser stopped before reaching an optimum '
             f'({res.message}; largest gradient with respect to a log-parameter: '
             f'{res.stationarity:.2e}). The estimates of this stage may be inaccurate.')


