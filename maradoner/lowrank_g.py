"""
Low-rank correction to the motif-activity covariance G:  G = diag(nu) + V V^T.

The Sigma/G REML log-likelihood (fit.loglik_motifs) eigendecomposes G in sample space.
For diagonal G this is eigh(diag(sqrt nu) P diag(sqrt nu)); for low-rank G it becomes
eigh(R^T P R) with R = chol(G).  The log-likelihood is invariant to rotations within
degenerate eigenspaces, so a degeneracy-safe eigh VJP (which zeros the 1/(lam_i-lam_j)
terms for coincident eigenvalues) yields the EXACT gradient while never producing the nan
that plain autodiff-through-eigh gives on grouped data (where P has repeated eigenvalues).
This mirrors why MARADONER hand-derives gradients, but localises the fix to the eigh.
"""
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial

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
    G_full = jnp.diag(nu_n) + (V @ V.T if V.shape[1] else jnp.zeros((s, s))) + 1e-9 * jnp.eye(s)
    R = jnp.linalg.cholesky(G_full)
    D_A, Q_A = safe_eigh(R.T @ P @ R)
    D_B, Q_B = safe_eigh(sig[:, None] * BTB * sig[None, :])
    D_A = jnp.clip(D_A, 0.0, None); D_B = jnp.clip(D_B, 0.0, None)
    cov = jnp.kron(D_A, D_B) + 1.0
    v = (Q_B.T * sig @ Z @ R @ Q_A).flatten('F')
    return -(v ** 2 / cov).sum() + jnp.log(cov).sum()


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
    return vecs[:, idx] * np.sqrt(np.clip(vals[idx], 1e-8, None))[None, :] * 0.3


def estimate_activity_lowrank(Z, BTB, P, gii, group_inds, k, Sigma0, nu0,
                              fix_ind, fix_val, maxiter=3000, verbose=False):
    """Joint REML fit of (Sigma, nu, V) for G = diag(nu) + V V^T, using safe_eigh so the
    gradient is nan-free even on grouped data.  One nu (fix_ind) is held at fix_val to pin
    the Sigma<->G scale.  Returns (Sigma, nu, V, dev)."""
    import jax
    jax.config.update("jax_enable_x64", True)
    from scipy.optimize import minimize
    Z = jnp.asarray(Z); BTB = jnp.asarray(BTB); P = jnp.asarray(P)
    gii_j = jnp.asarray(gii)
    m = int(BTB.shape[0]); g = len(group_inds); s = len(gii)
    free = [i for i in range(g) if i != fix_ind]
    V0 = _init_V(P, Z, BTB, Sigma0, nu0, gii, k)

    def assemble(x):
        Sigma = jnp.asarray(x[:m]) ** 2
        nu_free = jnp.asarray(x[m:m + g - 1]) ** 2
        nu = jnp.zeros(g).at[jnp.asarray(free)].set(nu_free).at[fix_ind].set(fix_val)
        V = jnp.asarray(x[m + g - 1:]).reshape(s, k)
        return Sigma, nu, V

    @jax.jit
    def vg(x):
        f = lambda xx: loglik_G(*assemble(xx), Z, BTB, P, gii_j)
        return jax.value_and_grad(f)(x)

    x0 = np.concatenate([np.sqrt(Sigma0), np.sqrt(np.asarray(nu0)[free]), V0.reshape(-1)])
    res = minimize(lambda x: tuple(np.asarray(t, float) for t in vg(x)), x0,
                   jac=True, method='L-BFGS-B', options=dict(maxiter=maxiter))
    Sigma, nu, V = [np.asarray(t) for t in assemble(res.x)]
    if verbose:
        print(f'[activity-lowrank k={k}] deviance {res.fun:.4f} (converged={res.success})')
    return Sigma, nu, V, float(res.fun)


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
