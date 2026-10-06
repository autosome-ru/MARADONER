"""
Low-rank correction to the motif-activity covariance G:  G = diag(nu) + V V^T.

The Sigma/G REML log-likelihood (fit.loglik_motifs) eigendecomposes G in sample space.
For diagonal G this is eigh(diag(sqrt nu) P diag(sqrt nu)); for low-rank G it becomes
eigh(R^T P R) with R = chol(G).  The log-likelihood is invariant to rotations within
degenerate eigenspaces, so a degeneracy-safe eigh VJP (which zeros the 1/(lam_i-lam_j)
terms for coincident eigenvalues) yields the EXACT gradient while never producing the nan
that plain autodiff-through-eigh gives on grouped data (where P has repeated eigenvalues).
This mirrors why MARADONER hand-derives gradients, but localises the fix to the eigh.

Limitation of safe_eigh: dropping the 1/(lam_i-lam_j) terms is exact only when the
eigenvalues that are closer than _EIGH_TOL (in absolute units) coincide exactly and the
function does not depend on the choice of the basis within their eigenspace to first
order. For nearly coincident eigenvalues the derivative with respect to V is not exact.
estimate_activity_lowrank therefore uses loglik_G_value_and_grad, whose gradient is in
closed form and involves no differentiation through an eigendecomposition.
"""
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from .utils import InputError, warn

_EIGH_TOL = 1e-7


@jax.custom_vjp
def safe_eigh(A):
    return jnp.linalg.eigh(A)


def _safe_eigh_fwd(A):
    w, V = jnp.linalg.eigh(A)
    return (w, V), (w, V)


def _safe_eigh_bwd(res, ct):
    w, V = res
    w_bar, V_bar = ct
    diff = w[None, :] - w[:, None]                      # w_j - w_i
    safe = jnp.abs(diff) > _EIGH_TOL
    F = jnp.where(safe, 1.0 / jnp.where(safe, diff, 1.0), 0.0)   # 1/(w_j - w_i), 0 if degenerate
    inner = jnp.diag(w_bar) + F * (V.T @ V_bar)
    A_bar = V @ inner @ V.T
    return (0.5 * (A_bar + A_bar.T),)


safe_eigh.defvjp(_safe_eigh_fwd, _safe_eigh_bwd)


def loglik_G(Sigma, nu, V, Z, BTB, P, gii):
    """REML deviance for Sigma (m,), nu (g,), V (s,k); G = diag(nu[gii]) + V V^T.
    Reduces exactly to fit.loglik_motifs at V with 0 columns.  Uses a Cholesky factor of G
    (loglik value identical to the symmetric sqrt) and safe_eigh so the gradient is nan-free
    even when P has repeated eigenvalues (grouped data)."""
    sig = jnp.sqrt(Sigma)
    s = len(gii)
    nu_n = nu[gii]
    # The jitter is relative to the scale of G, so that the value does not depend on units.
    G_full = jnp.diag(nu_n) + (V @ V.T if V.shape[1] else jnp.zeros((s, s))) \
             + 1e-9 * jnp.mean(nu_n) * jnp.eye(s)
    R = jnp.linalg.cholesky(G_full)
    D_A, Q_A = safe_eigh(R.T @ P @ R)
    D_B, Q_B = safe_eigh(sig[:, None] * BTB * sig[None, :])
    D_A = jnp.clip(D_A, 0.0, None); D_B = jnp.clip(D_B, 0.0, None)
    cov = jnp.kron(D_A, D_B) + 1.0
    v = (Q_B.T * sig @ Z @ R @ Q_A).flatten('F')
    return -(v ** 2 / cov).sum() + jnp.log(cov).sum()


def loglik_G_value_and_grad(Sigma, nu, V, Z, BTB, P, gii, num_groups):
    """loglik_G (without its jitter) and its gradient with respect to Sigma (m,), nu (g,)
    and V (s, k), in closed form. G = R R^T with the rectangular root R = [diag(sqrt(nu)), V],
    so that neither a Cholesky factor nor positive nu are needed; with (D_A, Q_A) the
    eigenpairs of R^T P R and (D_B, Q_B) those of Sigma^1/2 BTB Sigma^1/2 the derivative
    with respect to the matrix G is
        -E^T E + P sum(D_B) - P R Q_A diag(sum_l D_B[l]^2 / (1 + D_B[l] D_A)) (P R Q_A)^T,
    E = Q_B^T Sigma^1/2 Z (I + D_B[l] G P)^-1 row by row. It is valid for coincident
    eigenvalues and for parameters that are exactly zero."""
    sig = jnp.sqrt(Sigma)
    R = jnp.concatenate([jnp.diag(jnp.sqrt(nu[gii])), V], axis=1)
    D_A, Q_A = jnp.linalg.eigh(R.T @ P @ R)
    D_B, Q_B = jnp.linalg.eigh(sig[:, None] * BTB * sig[None, :])
    D_A = jnp.clip(D_A, 0.0, None); D_B = jnp.clip(D_B, 0.0, None)
    Mi = 1 / (D_B[:, None] * D_A[None, :] + 1)
    RQ = R @ Q_A
    PGQ = P @ RQ
    BTCQ = BTB * sig @ Q_B
    QZ = Q_B.T * sig @ Z
    Lam = (QZ @ RQ) * Mi
    value = -(Lam ** 2 / Mi).sum() - jnp.log(Mi).sum()
    E = Z @ RQ - BTCQ @ (Lam * D_A)
    grad_Sigma = -(E ** 2).sum(axis=1) + BTB.diagonal() * D_A.sum() - BTCQ ** 2 @ (Mi @ D_A ** 2)
    E = QZ - (Lam * D_B[:, None]) @ PGQ.T
    grad_G = -E.T @ E + P * D_B.sum() - (PGQ * (D_B ** 2 @ Mi)) @ PGQ.T
    grad_nu = jax.ops.segment_sum(grad_G.diagonal(), gii, num_segments=num_groups)
    grad_V = 2 * grad_G @ V
    return value, grad_Sigma, grad_nu, grad_V


def _init_V(P, Z, BTB, Sigma, nu, gii, k):
    """Initialise V from the residual sample covariance of the whitened activity estimate."""
    gii = np.asarray(gii)
    s = len(gii)
    nu_n = np.asarray(nu)[gii]
    # crude per-sample activity proxy: Z already = B^T Y P (m x s); its column-cov minus
    # the expected diagonal gives the residual correlated structure.
    Zc = np.asarray(Z)
    S = Zc.T @ Zc
    S = S / max(np.trace(S), 1e-12) * np.sum(nu_n)
    J = np.eye(s) - np.ones((s, s)) / s
    R = J @ (S - np.diag(nu_n)) @ J
    R = 0.5 * (R + R.T)
    vals, vecs = np.linalg.eigh(R)
    idx = np.argsort(vals)[::-1][:k]
    floor = 1e-8 * max(float(np.mean(nu_n)), np.finfo(float).tiny)
    return vecs[:, idx] * np.sqrt(np.clip(vals[idx], floor, None))[None, :] * 0.3


def estimate_activity_lowrank(Z, BTB, P, gii, group_inds, k, Sigma0, nu0,
                              fix_ind, fix_val, maxiter=3000, verbose=False):
    """Joint REML fit of (Sigma, nu, V) for G = diag(nu) + V V^T, with the closed-form
    gradient of loglik_G_value_and_grad. One nu (fix_ind) is held at fix_val to pin the
    Sigma<->G scale. The variances are first optimised through their square roots and
    then directly under non-negativity bounds, so that they can reach zero exactly and
    leave it. If the refit does not improve on the diagonal model (Sigma0, nu0), the
    latter is returned with an (s, 0) V.
    Returns (Sigma, nu, V, dev)."""
    import jax
    jax.config.update("jax_enable_x64", True)
    from scipy.optimize import minimize
    stage = f'low-rank activity covariance (--activity-lowrank {k})'
    Z = jnp.asarray(Z); BTB = jnp.asarray(BTB); P = jnp.asarray(P)
    gii_j = jnp.asarray(gii)
    m = int(BTB.shape[0]); g = len(group_inds); s = len(gii)
    k = int(k)
    dof = int(np.linalg.matrix_rank(np.asarray(P), hermitian=True))
    if k >= dof:
        raise InputError(f'--activity-lowrank {k} cannot be estimated: after the covariates '
                         f'are removed only {dof} degrees of freedom are left among the {s} '
                         'samples, and the rank of the correction must be smaller than that. '
                         'Use a smaller rank or drop the option.')
    free = np.array([i for i in range(g) if i != fix_ind], dtype=int)
    Sigma0 = np.asarray(Sigma0, dtype=np.float64)
    nu0 = np.asarray(nu0, dtype=np.float64)
    fix_val = float(fix_val)
    V0 = _init_V(P, Z, BTB, Sigma0, nu0, gii, k)
    n_var = m + g - 1

    def assemble(x):
        x = np.asarray(x, dtype=np.float64)
        Sigma = x[:m]
        nu = np.full(g, fix_val)
        nu[free] = x[m:n_var]
        V = x[n_var:].reshape(s, k)
        return Sigma, nu, V

    vg_closed = jax.jit(partial(loglik_G_value_and_grad, Z=Z, BTB=BTB, P=P, gii=gii_j,
                                num_groups=g))

    def vg(x):
        try:
            f, g_Sigma, g_nu, g_V = vg_closed(*assemble(x))
        except Exception as e:
            raise RuntimeError(f'{stage} failed while evaluating the objective '
                               f'({type(e).__name__}: {e})') from e
        grad = np.concatenate([np.asarray(g_Sigma), np.asarray(g_nu)[free],
                               np.asarray(g_V).reshape(-1)])
        return float(f), np.asarray(grad, dtype=np.float64)

    def vg_sqrt(z):
        x = np.array(z, dtype=np.float64)
        x[:n_var] = x[:n_var] ** 2
        f, grad = vg(x)
        grad[:n_var] = grad[:n_var] * 2 * z[:n_var]
        return f, grad

    x0 = np.concatenate([Sigma0, nu0[free], V0.reshape(-1)])
    f_diag = vg(np.concatenate([Sigma0, nu0[free], np.zeros(s * k)]))[0]
    if not np.isfinite(vg(x0)[0]):
        raise RuntimeError(f'{stage}: the objective is not finite at the starting point.')
    # Zero is a stationary point of the square-root parametrisation whatever the gradient:
    # variances that are zero in the diagonal model start slightly above it.
    tiny = np.finfo(np.float64).tiny
    scale_Sigma = max(float(np.max(Sigma0)), tiny)
    z0 = x0.copy()
    z0[:m] = np.maximum(z0[:m], 1e-3 * scale_Sigma)
    z0[m:n_var] = np.maximum(z0[m:n_var], 1e-3 * fix_val)
    z0[:n_var] = z0[:n_var] ** 0.5
    res = minimize(vg_sqrt, z0, jac=True, method='L-BFGS-B', options=dict(maxiter=maxiter))
    x = np.array(res.x, dtype=np.float64)
    x[:n_var] = x[:n_var] ** 2
    dev = float(res.fun)
    if np.isfinite(dev) and dev != 0:
        # Bounded refinement in the variances themselves, brought to a comparable scale.
        scale = np.concatenate([np.repeat(scale_Sigma, m), np.repeat(max(fix_val, tiny), g - 1),
                                np.repeat(max(float(np.mean(nu0[gii])), tiny) ** 0.5, s * k)])
        fun_scale = 1.0 / abs(dev)

        def vg_scaled(u):
            f, grad = vg(u * scale)
            return fun_scale * f, fun_scale * grad * scale
        bounds = [(0.0, None)] * n_var + [(None, None)] * (s * k)
        sol = minimize(vg_scaled, x / scale, jac=True, method='L-BFGS-B', bounds=bounds,
                       options=dict(maxiter=max(500, 2 * len(x)), ftol=1e-13, gtol=1e-9))
        if np.isfinite(sol.fun) and sol.fun / fun_scale <= dev:
            x = sol.x * scale
            dev = float(sol.fun) / fun_scale
    Sigma, nu, V = assemble(x)
    if not (np.isfinite(dev) and np.all(np.isfinite(x))):
        raise RuntimeError(f'{stage}: the optimisation ended at a non-finite value.')
    if verbose:
        print(f'[activity-lowrank k={k}] deviance {dev:.4f} (converged={res.success})')
    if not dev < f_diag:
        warn(f'The rank-{k} correction to the covariance of activities between samples '
             '(--activity-lowrank) did not improve the likelihood; the model without the '
             'correction is kept.')
        return Sigma0, nu0, np.zeros((s, 0)), float(f_diag)
    if not res.success:
        warn(f'The optimisation of the {stage} did not converge ({res.message}); the '
             'estimates of the variances of motif activities may be inaccurate.')
    return Sigma, nu, V, dev


def _selftest():
    jax.config.update("jax_enable_x64", True)
    rng = np.random.default_rng(0)
    print("lowrank_g.py self-test")

    # ---- safe_eigh gradient vs finite differences (directional, non-degenerate) ----
    n = 6
    M = rng.standard_normal((n, n)); M = M + M.T
    cw = rng.standard_normal(n); cV = rng.standard_normal((n, n))

    def f(A):
        w, V = safe_eigh(A)
        return jnp.sum(cw * w) + jnp.sum(cV * (V @ jnp.diag(jnp.arange(1.0, n + 1)) @ V.T))
    g = np.asarray(jax.grad(f)(jnp.asarray(M)))
    # directional derivatives along random SYMMETRIC directions (convention-free)
    errs = []
    for _ in range(5):
        Delta = rng.standard_normal((n, n)); Delta = Delta + Delta.T
        ana = float(np.sum(g * Delta))
        eps = 1e-6
        fd = (float(f(jnp.asarray(M + eps * Delta))) - float(f(jnp.asarray(M - eps * Delta)))) / (2 * eps)
        errs.append(abs(ana - fd) / (abs(fd) + 1e-12))
    print(f"  safe_eigh grad vs fd (max rel over 5 dirs): {max(errs):.2e}")

    # ---- degenerate matrix: gradient must be finite (plain eigh -> nan) ----
    D = np.diag([1.0, 1.0, 1.0, 2.0, 2.0, 3.0])   # repeated eigenvalues
    gd = np.asarray(jax.grad(f)(jnp.asarray(D)))
    plain = np.asarray(jax.grad(lambda A: (lambda wV: jnp.sum(cw * wV[0]) +
            jnp.sum(cV * (wV[1] @ jnp.diag(jnp.arange(1.0, n + 1)) @ wV[1].T)))(jnp.linalg.eigh(A)))(jnp.asarray(D)))
    print(f"  degenerate: safe_eigh grad finite={np.isfinite(gd).all()}, "
          f"plain eigh grad finite={np.isfinite(plain).all()}")

    # ---- loglik_G gradient on GROUPED data: finite (autodiff would nan) + matches fd ----
    from .linalg import ones_nullspace_transform
    from . import lowrank as LR
    p, m, s, gk = 400, 6, 10, 2
    gii = np.array([0, 0, 0, 1, 1, 1, 2, 2, 3, 3])          # grouped -> P has repeated eigvals
    g = int(gii.max()) + 1
    B = rng.random((p, m)); Y = B @ rng.standard_normal((m, s)) + rng.standard_normal((p, s))
    Bh = ones_nullspace_transform(B); Yh = ones_nullspace_transform(Y)
    D_n = (rng.random(g) + 0.5)[gii]
    P = np.asarray(LR.covariate_projection_theta(D_n, None, np.ones((1, s))))
    Z = jnp.asarray(Bh.T @ Yh @ P); BTB = jnp.asarray(Bh.T @ Bh); Pj = jnp.asarray(P)
    Sigma = jnp.asarray(rng.random(m) + 0.3); nu = jnp.asarray(rng.random(g) + 0.3)
    Vv = jnp.asarray(rng.standard_normal((s, gk)) * 0.2)
    gfun = jax.grad(lambda Sg, Nu, Vx: loglik_G(Sg, Nu, Vx, Z, BTB, Pj, jnp.asarray(gii)),
                    argnums=(0, 1, 2))
    gS, gN, gV = [np.asarray(x) for x in gfun(Sigma, nu, Vv)]
    fin = all(np.isfinite(x).all() for x in (gS, gN, gV))
    # directional fd in V
    Delta = rng.standard_normal((s, gk)); eps = 1e-6
    ana = float(np.sum(gV * Delta))
    fp = float(loglik_G(Sigma, nu, jnp.asarray(np.asarray(Vv) + eps * Delta), Z, BTB, Pj, jnp.asarray(gii)))
    fm = float(loglik_G(Sigma, nu, jnp.asarray(np.asarray(Vv) - eps * Delta), Z, BTB, Pj, jnp.asarray(gii)))
    fd = (fp - fm) / (2 * eps)
    print(f"  loglik_G grouped: grad_finite={fin}, grad_V vs fd rel={abs(ana-fd)/(abs(fd)+1e-12):.2e}")


if __name__ == '__main__':
    _selftest()
