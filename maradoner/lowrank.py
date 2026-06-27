"""
Diagonal-plus-low-rank correction to MARADONER's error covariance.

The error column-covariance becomes  Theta = D + W W^T, where D = diag(sigma_{group(i)})
is the existing group-wise diagonal (s x s) and W is an s x k low-rank factor estimated
inside the REML stage that fits D.  All routines reduce *exactly* to the diagonal case
when W is None or has zero columns, so the k == 0 (default) behaviour is unchanged.

Contents
--------
* projected_inverse(sigma, gii)            : P = H^T (H D H^T)^{-1} H and logdet|H D H^T|
* theta_inv / theta_logdet                 : dense Theta^{-1}, logdet|Theta| via Woodbury
* covariate_projection_theta(d, W, X)      : H_X^T (H_X Theta H_X^T)^{-1} H_X  (Sigma/G step)
* reml_deviance(sigma, W, G_Y, ptilde,gii) : REML deviance the D-stage minimises
* estimate_error_lowrank(...)              : joint REML fit of (sigma, W) from the Gram stat
* blup_solve_W(Y, B, Sigma, Gs, Theta)     : the shared dense BLUP/GLS solve (p x s)

Everything that touches the data does so only through the s x s Gram matrix
G_Y = (Q_N^T Y)^T (Q_N^T Y) in the variance step, or through small dense s x s / k x k
objects, so the cost is negligible (s is the number of samples).
"""
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from scipy.optimize import minimize


# ----------------------------------------------------------------------------- core ---
def projected_inverse(sigma, gii):
    """P = H^T (H D H^T)^{-1} H  (s x s) with D = diag(sigma[gii]); returns (P, logdet_M).
    H is the Helmert complement of 1_s, so P 1_s = 0 and logdet_M = logdet|H D H^T|."""
    d = jnp.asarray(sigma)[gii]
    d_inv = 1.0 / d
    xi = 1.0 / d_inv.sum()
    K = d_inv * d_inv.sum()
    P = xi * (jnp.diag(K) - jnp.outer(d_inv, d_inv))
    logdet_M = jnp.log(d).sum() + jnp.log(d_inv.mean())
    return P, logdet_M


def theta_inv(d, W):
    """Dense Theta^{-1} via Woodbury.  d: (s,) diagonal of D; W: (s,k) or None."""
    d = jnp.asarray(d)
    Dinv = 1.0 / d
    if W is None or jnp.asarray(W).shape[1] == 0:
        return jnp.diag(Dinv)
    W = jnp.asarray(W)
    k = W.shape[1]
    DiW = Dinv[:, None] * W
    C = jnp.eye(k) + W.T @ DiW
    return jnp.diag(Dinv) - DiW @ jnp.linalg.solve(C, DiW.T)


def theta_logdet(d, W):
    d = jnp.asarray(d)
    ld = jnp.log(d).sum()
    if W is None or jnp.asarray(W).shape[1] == 0:
        return ld
    W = jnp.asarray(W)
    k = W.shape[1]
    DiW = (1.0 / d)[:, None] * W
    C = jnp.eye(k) + W.T @ DiW
    return ld + jnp.linalg.slogdet(C)[1]


def covariate_projection_theta(d_n, W, X):
    """H_X^T (H_X Theta H_X^T)^{-1} H_X == Theta^{-1} - Theta^{-1} X^T (X Theta^{-1} X^T)^{-1} X Theta^{-1},
    Theta = diag(d_n) + W W^T.  Reduces to the diagonal covariate_projection when W is empty.
    d_n: (s,) per-sample error variances; X: (c, s) covariate design (first row ones)."""
    Ti = theta_inv(d_n, W)                       # (s,s)
    X = jnp.asarray(X, dtype=Ti.dtype)
    XTi = X @ Ti                                 # (c,s)
    XTiX = XTi @ X.T                             # (c,c)
    return Ti - XTi.T @ jnp.linalg.solve(XTiX, XTi)


# --------------------------------------------------------------------- REML deviance ---
def reml_deviance(sigma, W, G_Y, ptilde, gii):
    """REML deviance (-2 logL up to const) for Theta = D + W W^T in the D estimation
    stage.  Depends on the data only through the s x s Gram matrix G_Y = (Q_N^T Y)^T(Q_N^T Y)."""
    P, logdet_M = projected_inverse(sigma, gii)
    term1 = jnp.sum(P * G_Y)                                   # tr(P G_Y)
    if W is None or jnp.asarray(W).shape[1] == 0:
        return term1 + ptilde * logdet_M
    W = jnp.asarray(W)
    k = W.shape[1]
    PW = P @ W                                                 # (s,k)
    C = jnp.eye(k) + W.T @ PW                                  # I + W^T P W
    term2 = jnp.trace(jnp.linalg.solve(C, PW.T @ (G_Y @ PW)))
    logdet_C = jnp.linalg.slogdet(C)[1]
    return (term1 - term2) + ptilde * (logdet_M + logdet_C)


def reml_deviance_and_grad(sigma, W, G_Y, ptilde, gii, group_inds):
    """Exact analytic value+gradient of the low-rank-D REML deviance (numpy, no eigh, no
    autodiff -> no nans), in MARADONER's hand-derived style.  Returns (dev, grad_sigma (g,),
    grad_W (s,k)).  Reduces to the diagonal loglik_error / loglik_error_grad when W is empty.

    With P = D^{-1} - q q^T / w  (q = D^{-1} 1, w = 1^T D^{-1} 1),  A = P W,
    C = I_k + W^T A,  M2 = A^T G_Y A:
        dev = tr(P G_Y) - tr(C^{-1} M2) + ptilde (logdet|H D H^T| + logdet C).
    grad_W = 2[ A C^{-1}(M2 C^{-1} + ptilde I) - (P G_Y A) C^{-1} ].
    grad_d_i via  d tr(X P)/d d_i = (1/d_i^2)[-X_ii + 2 (X q)_i / w - q^T X q / w^2]  with
    X = G_Y + (X_a - X_b) + ptilde X_c  (the three P-sandwiched terms), plus the explicit
    logdet|H D H^T| derivative; then summed within each group."""
    sigma = np.asarray(sigma, float)
    G_Y = np.asarray(G_Y, float)
    gii = np.asarray(gii)
    d = sigma[gii]
    q = 1.0 / d
    w = q.sum()
    s = d.shape[0]
    P = np.diag(q) - np.outer(q, q) / w
    logdet_M = np.log(d).sum() + np.log(w) - np.log(s)
    term1 = float(np.sum(P * G_Y))                     # tr(P G_Y)

    has_lr = W is not None and np.asarray(W).shape[1] > 0
    if not has_lr:
        dev = term1 + ptilde * logdet_M
        X_tot = G_Y
        grad_W = np.zeros((s, 0))
    else:
        W = np.asarray(W, float)
        k = W.shape[1]
        A = P @ W                                      # (s,k)
        C = np.eye(k) + W.T @ A                        # (k,k)
        Cinv = np.linalg.inv(C)
        GA = G_Y @ A                                   # (s,k)
        M2 = A.T @ GA                                  # (k,k)
        dev = term1 - float(np.trace(Cinv @ M2)) + ptilde * (logdet_M + np.linalg.slogdet(C)[1])
        # grad_W = 2[ A C^{-1}(M2 C^{-1} + ptilde I) - P G_Y A C^{-1} ]   (note the P on P G_Y A)
        grad_W = 2.0 * (A @ (Cinv @ (M2 @ Cinv + ptilde * np.eye(k))) - (P @ GA) @ Cinv)
        X_a = W @ (Cinv @ M2 @ Cinv) @ W.T
        Sm = GA @ Cinv @ W.T
        X_b = Sm + Sm.T
        X_c = W @ Cinv @ W.T
        X_tot = G_Y + (X_a - X_b) + ptilde * X_c

    Xq = X_tot @ q
    qXq = float(q @ Xq)
    diagX = np.diag(X_tot)
    g_samp = (1.0 / d ** 2) * (-diagX + 2.0 * Xq / w - qXq / w ** 2) \
        + ptilde * (1.0 / d - 1.0 / (w * d ** 2))
    grad_sigma = np.array([g_samp[inds].sum() for inds in group_inds])
    return dev, grad_sigma, grad_W


def _init_W(sigma, G_Y, ptilde, gii, k):
    """Eigen init of W from the residual sample covariance (projected off 1_s)."""
    S = np.asarray(G_Y) / ptilde
    d = np.asarray(sigma)[np.asarray(gii)]
    s = S.shape[0]
    J = np.eye(s) - np.ones((s, s)) / s
    R = J @ (S - np.diag(d)) @ J
    R = 0.5 * (R + R.T)
    vals, vecs = np.linalg.eigh(R)
    idx = np.argsort(vals)[::-1][:k]
    return vecs[:, idx] * np.sqrt(np.clip(vals[idx], 1e-8, None))[None, :]


def estimate_error_lowrank(G_Y, ptilde, gii, group_inds, k, sigma0,
                           maxiter=4000, verbose=False):
    """Joint REML fit of (group variances sigma, low-rank W) given the Gram statistic.
    Uses the exact hand-derived gradient (no autodiff -> no nans), matching MARADONER's
    style.  sigma0: starting group variances (from the diagonal-only fit).  The optimiser
    works in sqrt(sigma) so sigma stays positive.  Returns (sigma, W, dev)."""
    G_Y = np.asarray(G_Y)
    gii = np.asarray(gii)
    g = len(group_inds)
    s = len(gii)
    if k == 0:
        dev = reml_deviance_and_grad(np.asarray(sigma0), None, G_Y, ptilde, gii, group_inds)[0]
        return np.asarray(sigma0), np.zeros((s, 0)), float(dev)

    W0 = _init_W(sigma0, G_Y, ptilde, gii, k)

    def fun_grad(x):
        root = x[:g]
        sigma = root ** 2
        W = x[g:].reshape(s, k)
        dev, gS, gW = reml_deviance_and_grad(sigma, W, G_Y, ptilde, gii, group_inds)
        grad = np.concatenate([gS * (2.0 * root), gW.reshape(-1)])   # chain rule sigma=root^2
        return float(dev), grad

    x0 = np.concatenate([np.sqrt(np.asarray(sigma0)), W0.reshape(-1)])
    res = minimize(fun_grad, x0, jac=True, method='L-BFGS-B', options=dict(maxiter=maxiter))
    sigma = res.x[:g] ** 2
    W = res.x[g:].reshape(s, k)
    if verbose:
        print(f'[lowrank-D] k={k}  deviance {res.fun:.4f}  (converged={res.success})')
    return np.asarray(sigma), np.asarray(W), float(res.fun)


# ----------------------------------------------------------------- BLUP / GLS solve ---
def blup_solve_W(Y, B, Sigma, Gs, Theta):
    """Return W_mat (p x s) = unvec[ (Gs (x) (B Sigma B^T) + Theta (x) I_p)^{-1} vec(Y) ].

    This is the shared core of every activity BLUP / sample-aware GLS in the model
    Y = B U + E, U ~ MN(., Sigma, .) with prior sample-covariance Gs (the s x s column
    covariance of the activity term in sample space) and error column-covariance Theta.
    Gs may be a length-s vector (diagonal, per-sample activities) or a full s x s matrix
    (e.g. the block matrix Z^T diag(nu) Z for per-group shared activities).  Uses dense
    s x s and m x m eigendecompositions only (s small, m = #motifs), never an n*p object.
    """
    Y = jnp.asarray(Y); B = jnp.asarray(B)
    Sigma = jnp.asarray(Sigma)
    Gs = jnp.asarray(Gs)
    Gs = jnp.diag(Gs) if Gs.ndim == 1 else Gs
    Theta = jnp.asarray(Theta)
    s = Y.shape[1]
    sig = jnp.sqrt(Sigma)

    R = jnp.linalg.cholesky(Theta)                      # Theta = R R^T  (lower)
    Rinv = jnp.linalg.solve(R, jnp.eye(s))              # R^{-1}
    RinvT = Rinv.T                                       # R^{-T}
    Gw = Rinv @ Gs @ RinvT                               # whitened prior sample-cov
    Gw = 0.5 * (Gw + Gw.T)
    Lam_G, Q_G = jnp.linalg.eigh(Gw)
    Lam_G = jnp.clip(Lam_G, 0.0, None)
    P = (Q_G * jnp.sqrt(Lam_G)) @ Q_G.T                  # symmetric sqrt of Gw

    C = B * sig[None, :]                                 # B sqrt(Sigma)  (p,m)
    BTB_w = sig[:, None] * (B.T @ B) * sig[None, :]      # sqrt(Sig) B^T B sqrt(Sig) (m,m)
    BTB_w = 0.5 * (BTB_w + BTB_w.T)
    Lam_B, Q_B = jnp.linalg.eigh(BTB_w)
    Lam_B = jnp.clip(Lam_B, 0.0, None)

    Zw = Y @ RinvT                                       # whiten data on sample axis
    T = C.T @ Zw @ P                                     # (m,s)
    Tspec = Q_B.T @ T @ Q_G
    Tdiv = Tspec / (Lam_B[:, None] * Lam_G[None, :] + 1.0)
    T2 = Q_B @ Tdiv @ Q_G.T                              # (m,s)
    Wc = Zw - C @ T2 @ P                                 # M_core^{-1} vec(Zw)
    return Wc @ Rinv                                     # unwhiten -> (p,s)


def blup_activities(Y, B, Sigma, group_inds, D_group, G_group, W,
                    tau_mult=None, V_act=None):
    """Compute per-sample (U_raw, m x s) and per-group (U, m x g) activity BLUPs for
    error covariance Theta = diag(D_group[gii]) + W W^T and activity sample-covariance
    G = diag(nu_{group(j)}) + V_act V_act^T.  tau_mult (per group) optionally rescales the
    prior variance nu (mirrors the existing tau-search); default 1.

    Per-sample uses the full prior sample-covariance G (diag + low-rank).  Per-group: when
    there is no activity low-rank it uses the block prior Z^T diag(nu) Z (one shared activity
    per group); when V_act is present (per-sample activity correlations) the per-group value
    is the group-average of the per-sample activities.  Reduces exactly to the diagonal-D
    `_sol` formulas in fit.py when W and V_act are empty."""
    Y = np.asarray(Y); B = np.asarray(B); Sigma = np.asarray(Sigma)
    g = len(group_inds)
    s = Y.shape[1]
    gii = np.zeros(s, dtype=int)
    for gi, inds in enumerate(group_inds):
        gii[inds] = gi
    if tau_mult is None:
        tau_mult = np.ones(g)
    tau_mult = np.asarray(tau_mult)
    d_n = np.asarray(D_group)[gii]
    Theta = np.diag(d_n)
    if W is not None and np.asarray(W).shape[1] > 0:
        Theta = Theta + np.asarray(W) @ np.asarray(W).T
    nu_grp = np.asarray(G_group) * tau_mult         # per-group prior variance
    nu_n = nu_grp[gii]                              # per-sample (its group's nu)
    has_V = V_act is not None and np.asarray(V_act).shape[1] > 0
    Gs_raw = np.diag(nu_n)
    if has_V:
        Va = np.asarray(V_act)
        Gs_raw = Gs_raw + Va @ Va.T                 # full per-sample activity sample-cov

    # --- per-sample activities: full prior sample-cov Gs_raw ---
    W_raw = np.asarray(blup_solve_W(Y, B, Sigma, Gs_raw, Theta))
    U_raw = Sigma[:, None] * ((B.T @ W_raw) @ Gs_raw)

    # --- per-group activities ---
    if has_V:
        # per-sample activities are the model; summarise a group by its mean activity
        U_grp = np.zeros((B.shape[1], g))
        for i, inds in enumerate(group_inds):
            U_grp[:, i] = U_raw[:, inds].mean(axis=1)
    else:
        Z = np.zeros((g, s))
        for i, inds in enumerate(group_inds):
            Z[i, inds] = 1.0
        Gs_block = Z.T @ np.diag(nu_grp) @ Z        # (s,s), rank g
        W_grp = np.asarray(blup_solve_W(Y, B, Sigma, Gs_block, Theta))
        BTWg = B.T @ W_grp
        U_grp = np.zeros((B.shape[1], g))
        for i, inds in enumerate(group_inds):
            U_grp[:, i] = Sigma * BTWg[:, inds].sum(axis=1) * nu_grp[i]
    return U_raw, U_grp


def blup_group_cov_factors(group_inds, D_group, G_group, W, Sigma, B, tau_mult=None):
    """Compact factors for the per-group posterior covariance of the (shared) activity
    under Theta = D + W W^T.  Returns (M_eig, W_weights) where the i-th group's covariance
    is  M_eig @ diag(W_weights[i]) @ M_eig^T.  Storing these (m x m + g x m) instead of g
    dense m x m matrices keeps the predict file small.

    The joint posterior precision over the g group activities is
        Prec = (Lam_nu^{-1} (x) Sigma^{-1}) + (Z Theta^{-1} Z^T) (x) (B^T B),
    Lam_nu = diag(nu * tau).  Simultaneously diagonalising the two Kronecker terms gives the
    i-th m x m diagonal block in closed form (no mg x mg object)."""
    B = np.asarray(B); Sigma = np.asarray(Sigma)
    g = len(group_inds)
    s = int(max(np.max(inds) for inds in group_inds)) + 1
    gii = np.zeros(s, dtype=int)
    for gi, inds in enumerate(group_inds):
        gii[inds] = gi
    if tau_mult is None:
        tau_mult = np.ones(g)
    nu_grp = np.asarray(G_group) * np.asarray(tau_mult)
    d_n = np.asarray(D_group)[gii]
    Ti = np.asarray(theta_inv(d_n, W))                       # (s,s)
    Z = np.zeros((g, s))
    for i, inds in enumerate(group_inds):
        Z[i, inds] = 1.0
    sq_nu = np.sqrt(nu_grp)
    Pg = (sq_nu[:, None] * (Z @ Ti @ Z.T)) * sq_nu[None, :]  # (g,g)
    Pg = 0.5 * (Pg + Pg.T)
    Lam_Pg, Q_g = np.linalg.eigh(Pg)
    Lam_Pg = np.clip(Lam_Pg, 0.0, None)
    sig = np.sqrt(Sigma)
    Am = sig[:, None] * (B.T @ B) * sig[None, :]             # (m,m)
    Am = 0.5 * (Am + Am.T)
    Lam_Am, Q_m = np.linalg.eigh(Am)
    Lam_Am = np.clip(Lam_Am, 0.0, None)
    M_eig = sig[:, None] * Q_m                               # Sigma^{1/2} Q_m  (m,m)
    G_eig = sq_nu[:, None] * Q_g                             # Lam_nu^{1/2} Q_g (g,g)
    inv_denom = 1.0 / (1.0 + Lam_Pg[:, None] * Lam_Am[None, :])   # (g_eig, m_eig)
    W_weights = (G_eig ** 2) @ inv_denom                    # (g, m)
    return M_eig, W_weights


def expand_group_cov(M_eig, w):
    """Reconstruct one group's m x m posterior covariance from compact factors."""
    return (np.asarray(M_eig) * np.asarray(w)[None, :]) @ np.asarray(M_eig).T


def blup_group_cov(group_inds, D_group, G_group, W, Sigma, B, tau_mult=None):
    """Convenience wrapper returning the list of g dense per-group covariances."""
    M_eig, W_weights = blup_group_cov_factors(group_inds, D_group, G_group, W, Sigma, B,
                                              tau_mult=tau_mult)
    return [expand_group_cov(M_eig, W_weights[i]) for i in range(W_weights.shape[0])]


# -------------------------------------------------------------------------- selftest ---
def _selftest():
    import numpy.random as npr
    jax.config.update("jax_enable_x64", True)
    rng = npr.default_rng(0)
    print("lowrank.py self-test")

    # ---- theta_inv / theta_logdet vs dense ----
    s, k = 12, 3
    d = rng.random(s) + 0.3
    W = rng.standard_normal((s, k)) * 0.5
    Theta = np.diag(d) + W @ W.T
    ti = np.asarray(theta_inv(d, W))
    print("  theta_inv  err:", np.linalg.norm(ti - np.linalg.inv(Theta)) / np.linalg.norm(np.linalg.inv(Theta)))
    print("  theta_logdet err:", abs(float(theta_logdet(d, W)) - np.linalg.slogdet(Theta)[1]))

    # ---- covariate_projection_theta vs dense Helmert-style ----
    c = 2
    X = np.vstack([np.ones(s), rng.standard_normal((c - 1, s))])
    Ti = np.linalg.inv(Theta)
    P_dense = Ti - Ti @ X.T @ np.linalg.solve(X @ Ti @ X.T, X @ Ti)
    P_eff = np.asarray(covariate_projection_theta(d, W, X))
    print("  cov_proj_theta err:", np.linalg.norm(P_eff - P_dense) / np.linalg.norm(P_dense))

    # ---- blup_solve_W vs brute-force dense Kronecker (small p,m,s) ----
    p, m, s = 9, 4, 6
    B = rng.standard_normal((p, m))
    Sigma = rng.random(m) + 0.2
    Gs = rng.random(s) + 0.1
    d = rng.random(s) + 0.3
    Wlr = rng.standard_normal((s, 2)) * 0.4
    Theta = np.diag(d) + Wlr @ Wlr.T
    Y = rng.standard_normal((p, s))
    Wmat = np.asarray(blup_solve_W(Y, B, Sigma, Gs, Theta))
    BS = B * Sigma[None, :]
    Lam = np.kron(np.diag(Gs), BS @ B.T) + np.kron(Theta, np.eye(p))
    w_bf = np.linalg.solve(Lam, Y.reshape(-1, order='F')).reshape(p, s, order='F')
    print("  blup_solve_W err:", np.linalg.norm(Wmat - w_bf) / np.linalg.norm(w_bf))

    # ---- blup vs the diagonal-D code path (W=0 should reproduce sqrtSig V .. formula) ----
    p, m, s = 30, 5, 8
    B = rng.standard_normal((p, m)); Sigma = rng.random(m) + 0.2
    d = rng.random(s) + 0.5; Gs = (rng.random(s) + 0.1)
    Theta = np.diag(d)
    Y = rng.standard_normal((p, s))
    Wmat = np.asarray(blup_solve_W(Y, B, Sigma, Gs, Theta))
    U_lr = Sigma[:, None] * (B.T @ Wmat) * Gs[None, :]
    # reference per-sample _sol with scalar sigma,nu (diagonal D)
    Bs = B * np.sqrt(Sigma)[None, :]
    Sval, V = np.linalg.eigh(Bs.T @ Bs)
    Qh = np.sqrt(Sigma)[:, None] * V
    U_ref = np.zeros((m, s))
    for j in range(s):
        tau = Gs[j] / d[j]
        scale = 1.0 / (Sval + 1.0 / tau)             # original _sol: no extra /d_j
        U_ref[:, j] = (Qh * scale) @ (Qh.T @ (B.T @ Y[:, j]))
    print("  blup vs diagonal _sol err:", np.linalg.norm(U_lr - U_ref) / np.linalg.norm(U_ref))

    # ---- per-group BLUP vs brute-force shared-activity joint posterior ----
    p, m, s = 14, 4, 9
    group_inds = [np.array([0, 1, 2]), np.array([3, 4]), np.array([5, 6, 7, 8])]
    g = len(group_inds)
    B = rng.standard_normal((p, m)); Sigma = rng.random(m) + 0.2
    D_group = rng.random(g) + 0.4; G_group = rng.random(g) + 0.1
    Wlr = rng.standard_normal((s, 2)) * 0.4
    gii = np.zeros(s, dtype=int)
    for gi, inds in enumerate(group_inds):
        gii[inds] = gi
    Theta = np.diag(D_group[gii]) + Wlr @ Wlr.T
    Y = rng.standard_normal((p, s))
    _, U_grp = blup_activities(Y, B, Sigma, group_inds, D_group, G_group, Wlr)
    # brute force: shared-activity model vec(Y) = (Z^T (x) B) vec(A) + E
    Z = np.zeros((g, s))
    for i, inds in enumerate(group_inds):
        Z[i, inds] = 1.0
    BS = B * Sigma[None, :]
    Lam = np.kron(Z.T @ np.diag(G_group) @ Z, BS @ B.T) + np.kron(Theta, np.eye(p))
    Phi = np.kron(Z.T, B)                                   # (ps, mg)
    Prior = np.kron(np.diag(G_group), np.diag(Sigma))       # (mg, mg)
    A_bf = (Prior @ Phi.T @ np.linalg.solve(Lam, Y.reshape(-1, order='F'))).reshape(m, g, order='F')
    print("  per-group BLUP vs brute-force err:",
          np.linalg.norm(U_grp - A_bf) / np.linalg.norm(A_bf))

    # ---- per-group posterior covariance vs brute-force dense precision ----
    tau_m = rng.random(g) * 0.5 + 0.8
    covs = blup_group_cov(group_inds, D_group, G_group, Wlr, Sigma, B, tau_mult=tau_m)
    nu_grp = G_group * tau_m
    Ti = np.linalg.inv(Theta)
    Prec = np.kron(np.diag(1.0 / nu_grp), np.diag(1.0 / Sigma)) + np.kron(Z @ Ti @ Z.T, B.T @ B)
    Cov_full = np.linalg.inv(Prec)
    errc = max(np.linalg.norm(covs[i] - Cov_full[i*m:(i+1)*m, i*m:(i+1)*m]) /
               np.linalg.norm(Cov_full[i*m:(i+1)*m, i*m:(i+1)*m]) for i in range(g))
    print("  per-group posterior cov vs brute-force err:", errc)

    # ---- per-group posterior covariance, W=0, vs original cov() formula ----
    covs0 = blup_group_cov(group_inds, D_group, G_group, None, Sigma, B, tau_mult=tau_m)
    Bs = B * np.sqrt(Sigma)[None, :]
    Sval, V = np.linalg.eigh(Bs.T @ Bs); Qh = np.sqrt(Sigma)[:, None] * V
    err0 = 0.0
    for i, inds in enumerate(group_inds):
        n = len(inds); tau = G_group[i] * tau_m[i] / D_group[i]
        scale = D_group[i] / (n * Sval + 1.0 / tau)
        c_ref = (Qh * scale) @ Qh.T
        err0 = max(err0, np.linalg.norm(covs0[i] - c_ref) / np.linalg.norm(c_ref))
    print("  per-group posterior cov (W=0) vs original cov() err:", err0)

    # ---- per-group BLUP, W=0, vs the original n_i `_sol` formula in fit.py ----
    Theta0 = np.diag(D_group[gii])
    _, U_grp0 = blup_activities(Y, B, Sigma, group_inds, D_group, G_group, None)
    Bs = B * np.sqrt(Sigma)[None, :]
    Sval, V = np.linalg.eigh(Bs.T @ Bs); Qh = np.sqrt(Sigma)[:, None] * V
    U_ref = np.zeros((m, g))
    for i, inds in enumerate(group_inds):
        n = len(inds); tau = G_group[i] / D_group[i]
        scale = 1.0 / (n * Sval + 1.0 / tau)
        U_ref[:, i] = (Qh * scale) @ (Qh.T @ (B.T @ Y[:, inds].sum(axis=1)))
    print("  per-group BLUP vs original n_i _sol err:",
          np.linalg.norm(U_grp0 - U_ref) / np.linalg.norm(U_ref))

    # ---- reml_deviance vs brute-force dense (reuse projected form) ----
    from .linalg import ones_nullspace
    s = 10
    gii = np.array([0, 0, 1, 1, 1, 2, 2, 2, 3, 3])
    group_inds = [np.where(gii == i)[0] for i in range(4)]
    G_Y = rng.standard_normal((40, s)); G_Y = G_Y.T @ G_Y
    H = ones_nullspace(s)
    for k in [0, 2, 3]:
        sigma = rng.random(4) + 0.3
        Wlr = rng.standard_normal((s, k)) if k else np.zeros((s, 0))
        de = float(reml_deviance(jnp.asarray(sigma), jnp.asarray(Wlr) if k else None,
                                 jnp.asarray(G_Y), 40, jnp.asarray(gii)))
        Th = np.diag(sigma[gii]) + Wlr @ Wlr.T
        HTH = H @ Th @ H.T
        db = np.trace(np.linalg.solve(HTH, H @ G_Y @ H.T)) + 40 * np.linalg.slogdet(HTH)[1]
        print(f"  reml_deviance k={k} err:", abs(de - db) / abs(db))

    # ---- hand-derived gradient vs finite differences AND vs the dev value ----
    for k in [0, 2, 3]:
        sigma = rng.random(4) + 0.4
        Wlr = rng.standard_normal((s, k)) * 0.3 if k else np.zeros((s, 0))
        dev, gS, gW = reml_deviance_and_grad(sigma, Wlr if k else None, G_Y, 40, gii, group_inds)
        dev_ref = float(reml_deviance(jnp.asarray(sigma), jnp.asarray(Wlr) if k else None,
                                      jnp.asarray(G_Y), 40, jnp.asarray(gii)))
        eps = 1e-6
        gS_fd = np.zeros(4)
        for i in range(4):
            sp = sigma.copy(); sp[i] += eps; sm = sigma.copy(); sm[i] -= eps
            fp = reml_deviance_and_grad(sp, Wlr if k else None, G_Y, 40, gii, group_inds)[0]
            fm = reml_deviance_and_grad(sm, Wlr if k else None, G_Y, 40, gii, group_inds)[0]
            gS_fd[i] = (fp - fm) / (2 * eps)
        eS = np.linalg.norm(gS - gS_fd) / (np.linalg.norm(gS_fd) + 1e-12)
        if k:
            gW_fd = np.zeros((s, k))
            for a in range(s):
                for b in range(k):
                    Wp = Wlr.copy(); Wp[a, b] += eps; Wm = Wlr.copy(); Wm[a, b] -= eps
                    fp = reml_deviance_and_grad(sigma, Wp, G_Y, 40, gii, group_inds)[0]
                    fm = reml_deviance_and_grad(sigma, Wm, G_Y, 40, gii, group_inds)[0]
                    gW_fd[a, b] = (fp - fm) / (2 * eps)
            eW = np.linalg.norm(gW - gW_fd) / (np.linalg.norm(gW_fd) + 1e-12)
        else:
            eW = 0.0
        print(f"  reml grad k={k}: dev err {abs(dev-dev_ref)/abs(dev_ref):.1e}, "
              f"grad_sigma fd-err {eS:.2e}, grad_W fd-err {eW:.2e}")


if __name__ == '__main__':
    _selftest()
