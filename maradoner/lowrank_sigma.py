# -*- coding: utf-8 -*-
"""
Structure of the covariance of activities between motifs: families and a low-rank part.

In the basic model the activities of different motifs are independent, U ~ MN(M X, Sigma, G)
with a diagonal Sigma = diag(tau). Here Sigma is replaced by

    S = diag(tau) + F (diag(omega) + L L^T) F^T,

with F (m x f) a fixed matrix of zeros and ones telling which motifs belong to which family,
omega (f) the variances shared by the motifs of a family, and an optional L (f x k) for a
covariance between families. Without families F is the identity (f = m) and omega is not used,
as it would be the same thing as tau: S = diag(tau) + L L^T. The variance omega_j of a family of
one motif is not used either, for the same reason.

The likelihood depends on S only through the r x r matrix K = C^T S C. Here B^T B = C C^T
(C = V^T diag(s) from the thin SVD Q diag(s) V of the loadings), and r is the rank of B:

    K = C^T diag(tau) C + C_F^T diag(omega) C_F + (C_F^T L)(C_F^T L)^T,    C_F = F^T C.

One r x r eigendecomposition per evaluation therefore does what that of
Sigma^1/2 B^T B Sigma^1/2 does in the diagonal model, whatever f and k are. With the eigenpairs
(d, Q_K) of K, (a, Q_A) of R^T P R (G = R R^T, P the projected inverse of the error covariance
between samples) and Mi[l, j] = 1 / (1 + d_l a_j), the REML deviance (-2 logL up to a constant) is

    dev = -sum_lj d_l What_lj^2 Mi_lj - sum_lj log Mi_lj,    What = Q_K^T W R Q_A,

with W = Q^T Y P the data in the column space of B (and Z = B^T Y P = C W). Its derivative
with respect to the matrix K is

    Gamma = Q_K [diag(Mi a) - Lam Lam^T] Q_K^T,    Lam = What o Mi,

so that d dev / d S = C Gamma C^T and

    d/d tau = diag(C Gamma C^T),   d/d omega = diag(C_F Gamma C_F^T),   d/d L = 2 C_F Gamma C_F^T L.

The Fisher information and the likelihood ratios of tau_i = 0 and omega_j = 0 have the same
structure: a parameter adds a rank-one term c c^T to K, with c a row of C or of C_F.
"""
import json
import os
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from .utils import InputError, warn, logger_print
from .meta_optimizer import STATIONARITY_TOL

# The optimisation also stops when the deviance falls by less than STALL_TOL over STALL_ITERATIONS
# iterations: differences of that size are immaterial for every likelihood ratio.
STALL_TOL = 1e-3
STALL_ITERATIONS = 100
# Fisher scoring goes on until the stationarity is at most this.
FISHER_TOL = 1e-4


# ------------------------------------------------------------- the covariance S itself ---
def motif_covariance(motif_variance):
    """The covariance of activities between motifs of a fit: the vector tau (m,) when it is
    diagonal, the dense (m, m) matrix S = diag(tau) + F (diag(omega) + L L^T) F^T otherwise."""
    tau = np.asarray(motif_variance.motif, dtype=np.float64)
    omega = getattr(motif_variance, 'family_variance', None)
    L = getattr(motif_variance, 'family_lowrank', None)
    F = getattr(motif_variance, 'families', None)
    has_omega = omega is not None and np.any(np.asarray(omega) > 0)
    has_L = L is not None and np.asarray(L).ndim == 2 and np.asarray(L).shape[1] > 0
    if not (has_omega or has_L):
        return tau
    f = len(tau) if F is None else np.asarray(F).shape[1]
    Psi = np.zeros((f, f))
    if has_omega:
        Psi += np.diag(np.asarray(omega, dtype=np.float64))
    if has_L:
        L = np.asarray(L, dtype=np.float64)
        Psi += L @ L.T
    if F is None:
        S = Psi
    else:
        F = np.asarray(F, dtype=np.float64)
        S = F @ Psi @ F.T
    S = S + np.diag(tau)
    return 0.5 * (S + S.T)


def motif_variances(Sigma) -> np.ndarray:
    """Variances of motif activities (the diagonal of S) from motif_covariance."""
    Sigma = np.asarray(Sigma)
    return Sigma if Sigma.ndim == 1 else np.diag(Sigma).copy()


def subset_covariance(Sigma, inds):
    """S restricted to the motifs inds (an index array or a boolean mask)."""
    Sigma = np.asarray(Sigma)
    if Sigma.ndim == 1:
        return Sigma[inds]
    inds = np.arange(len(Sigma))[inds]
    return Sigma[np.ix_(inds, inds)]


def motif_root(Sigma):
    """A square root of the motif covariance: the vector sqrt(tau) for a diagonal one, the
    symmetric root S^1/2 for a matrix. Being symmetric, the root can multiply from either
    side wherever the diagonal model multiplies by sqrt(tau) (see lmul and rmul)."""
    Sigma = np.asarray(Sigma, dtype=np.float64)
    if Sigma.ndim == 1:
        return Sigma ** 0.5
    w, v = np.linalg.eigh(0.5 * (Sigma + Sigma.T))
    return (v * np.sqrt(np.clip(w, 0.0, None))) @ v.T


def lmul(root, X):
    """root @ X, with a vector root standing for a diagonal matrix."""
    if np.ndim(root) == 1:
        return root.reshape(-1, 1) * X
    return root @ X


def rmul(X, root):
    """X @ root, with a vector root standing for a diagonal matrix."""
    if np.ndim(root) == 1:
        return X * root.reshape(1, -1)
    return X @ root


def cov_mul(Sigma, X):
    """S @ X for the motif covariance S as returned by motif_covariance."""
    if np.ndim(Sigma) == 1:
        return Sigma.reshape(-1, 1) * X if np.ndim(X) == 2 else Sigma * X
    return Sigma @ X


# ------------------------------------------------------------------------- families ---
def _show(items, n: int = 5) -> str:
    items = [str(i) for i in items]
    return ', '.join(items[:n]) + (f' ... and {len(items) - n} more' if len(items) > n else '')


def _load_families_file(path: str, motif_names: list) -> dict:
    """family -> list of motifs, from JSON ({"family": ["motif", ...]} or {"motif": "family"})
    or from text: either one family per line ("family motif motif ...") or one motif per line
    ("motif family"); the latter is recognised by its first column holding motif names."""
    if not os.path.exists(path):
        raise InputError(f'The motif families file "{path}" was not found.')
    if os.path.isdir(path):
        raise InputError(f'The motif families file "{path}" is a directory, not a file.')
    try:
        with open(path, 'rb') as f:
            text = f.read().decode('utf-8-sig')
    except UnicodeDecodeError:
        raise InputError(f'The motif families file "{path}" is not a text file.') from None
    stripped = text.strip()
    if not stripped:
        raise InputError(f'The motif families file "{path}" is empty.')
    known = set(map(str, motif_names))
    if stripped[0] in '{[':
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError as e:
            raise InputError(f'The motif families file "{path}" looks like JSON, but is not valid JSON: '
                             f'{e.msg} (line {e.lineno}, column {e.colno}).') from None
        if not isinstance(obj, dict):
            raise InputError(f'The motif families file "{path}" must hold a JSON object: either '
                             '{"family": ["motif", ...]} or {"motif": "family"}.')
        families = dict()
        for key, value in obj.items():
            if isinstance(value, str):
                families.setdefault(value, list()).append(str(key))
            elif isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value):
                families.setdefault(str(key), list()).extend(value)
            else:
                raise InputError(f'Entry "{key}" of the motif families file "{path}" must be a list of '
                                 'motif names (or the name of a family), but is '
                                 f'{type(value).__name__}.')
        return families
    lines = list()
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith('#'):
            lines.append(line.split())
    pairs = all(len(items) == 2 for items in lines)
    first_known = sum(items[0] in known for items in lines)
    rest_known = sum(any(i in known for i in items[1:]) for items in lines)
    families = dict()
    if pairs and first_known > rest_known:
        for motif, family in lines:
            families.setdefault(family, list()).append(motif)
        return families
    for lineno, items in enumerate(lines, 1):
        if len(items) == 1:
            hint = ' If the names are separated by commas, use spaces or tabs instead.' if ',' in items[0] else ''
            raise InputError(f'Family "{items[0]}" of the motif families file "{path}" has no motifs. Each '
                             'line must be a family name followed by the names of its motifs, or a motif '
                             f'name followed by the name of its family.{hint}')
        families.setdefault(items[0], list()).extend(items[1:])
    return families


def read_families(path: str, motif_names: list, verbose: bool = True) -> tuple:
    """The matrix F (m x f) of zeros and ones of the motif families listed in the file path,
    and the names of the families. A motif may belong to several families or to none. Motifs
    that the project does not have are ignored, families left without motifs are dropped,
    and families with the same motifs are merged."""
    motif_names = list(map(str, motif_names))
    families = _load_families_file(os.fspath(path), motif_names)
    index = {n: i for i, n in enumerate(motif_names)}
    unknown = sorted({mo for members in families.values() for mo in members if mo not in index})
    if unknown:
        warn(f'{len(unknown)} motif(s) of the motif families file are not in the project and were '
             f'ignored: {_show(unknown)}.')
    columns = dict()
    empty = list()
    for name, members in families.items():
        rows = sorted({index[mo] for mo in members if mo in index})
        if not rows:
            empty.append(name)
            continue
        key = tuple(rows)
        if key in columns:
            columns[key].append(name)
        else:
            columns[key] = [name]
    if empty:
        warn(f'{len(empty)} of the families have no motifs in the project and were dropped: {_show(empty)}.')
    merged = [names for names in columns.values() if len(names) > 1]
    if merged:
        warn(f'{len(merged)} group(s) of families consist of the same motifs and were merged (their '
             f'variances could not be told apart): {_show(["+".join(n) for n in merged])}.')
    if not columns:
        raise InputError(f'None of the motifs of the motif families file "{path}" are in the project '
                         f'(for example {_show(unknown, 3)} versus {_show(motif_names, 3)}).')
    F = np.zeros((len(motif_names), len(columns)))
    names = list()
    for j, (rows, fam) in enumerate(columns.items()):
        F[list(rows), j] = 1.0
        names.append('+'.join(fam))
    sizes = F.sum(axis=0)
    in_family = int((F.sum(axis=1) > 0).sum())
    logger_print(f'  Motif families: {F.shape[1]} families ({int((sizes > 1).sum())} of them with more '
                 f'than one motif) cover {in_family} of {len(motif_names)} motifs.', verbose)
    return F, names


# ---------------------------------------------------------------- B-space quantities ---
def b_space(B_decomposition) -> np.ndarray:
    """C (m x r) with B^T B = C C^T, from the thin SVD of the (transformed) loadings."""
    return np.asarray(B_decomposition.V, dtype=np.float64).T * np.asarray(B_decomposition.S, dtype=np.float64)


def b_space_data(Z, B_decomposition) -> np.ndarray:
    """W = Q^T Y P (r x s) from Z = B^T Y P = C W."""
    S = np.asarray(B_decomposition.S, dtype=np.float64)
    V = np.asarray(B_decomposition.V, dtype=np.float64)
    return (V @ np.asarray(Z, dtype=np.float64)) / S.reshape(-1, 1)


def _spectral(tau, omega, L, nu, V, C, CF, P, gii, has_omega):
    CFL = CF.T @ L
    K = (C.T * tau) @ C + CFL @ CFL.T
    if has_omega:
        K = K + (CF.T * omega) @ CF
    D_K, Q_K = jnp.linalg.eigh(K)
    return _spectral_samples(D_K, Q_K, nu, V, P, gii)


def _spectral_reduced(Phi, nu, V, P, gii):
    # _spectral for K = Phi Phi^T with Phi (r x n, n < r) the columns of K of the motifs and
    # families with nonzero variance (and C_F^T L): Q_K (r x n) spans the range of Phi, from its
    # QR decomposition and the eigendecomposition of R R^T. The eigenvalue of K is zero on the
    # complement of that span.
    Qp, Rp = jnp.linalg.qr(Phi)
    D_K, U = jnp.linalg.eigh(Rp @ Rp.T)
    return _spectral_samples(D_K, Qp @ U, nu, V, P, gii)


def _spectral_samples(D_K, Q_K, nu, V, P, gii):
    R = jnp.concatenate([jnp.diag(jnp.sqrt(nu[gii])), V], axis=1)
    D_A, Q_A = jnp.linalg.eigh(R.T @ P @ R)
    D_K = jnp.clip(D_K, 0.0, None)
    D_A = jnp.clip(D_A, 0.0, None)
    Mi = 1 / (D_K[:, None] * D_A[None, :] + 1)
    RQ = R @ Q_A
    return D_K, Q_K, D_A, RQ, Mi


def loglik_S_value_and_grad(tau, omega, L, nu, V, W, C, CF, P, gii, num_groups,
                            has_omega=True, identity=False):
    """REML deviance of (tau, omega, L, nu, V) and its gradient, see the module docstring.
    tau (m,), omega (f,), L (f, k), nu (g,), V (s, k_V): G = diag(nu[gii]) + V V^T. If
    identity is set, F is the identity matrix (CF is C). Reduces to fit.loglik_motifs at
    omega = 0 and L and V with no columns; valid for parameters equal to zero."""
    spec = _spectral(tau, omega, L, nu, V, C, CF, P, gii, has_omega)
    return _loglik_S_from_spectral(spec, omega, L, V, W, C, CF, P, gii, num_groups, has_omega, identity)


def _loglik_S_from_spectral(spec, omega, L, V, W, C, CF, P, gii, num_groups, has_omega, identity):
    D_K, Q_K, D_A, RQ, Mi = spec
    WR = W @ RQ
    What = Q_K.T @ WR
    Lam = What * Mi
    value = -(D_K[:, None] * What * Lam).sum() - jnp.log(Mi).sum()
    QL = Q_K @ Lam
    # Gamma = Q_K diag(Mi D_A) Q_K^T - QL QL^T is used through its factors: C Gamma C^T formed
    # as a matrix and multiplied elementwise by C makes XLA fuse the product into a slow loop.
    h = Mi @ D_A
    reduced = Q_K.shape[1] < C.shape[1]
    if reduced:
        # Q_K from _spectral_reduced: Gamma also has sum(D_A) (I - Q_K Q_K^T), and QL includes the
        # complement of Q_K (Mi is one on it).
        QL = QL + WR - Q_K @ What
        sA = D_A.sum()
        h = h - sA

    def quad(rows):
        # The diagonal of rows Gamma rows^T.
        E = rows @ Q_K
        res = E ** 2 @ h - ((rows @ QL) ** 2).sum(axis=1)
        return res + sA * (rows ** 2).sum(axis=1) if reduced else res
    grad_tau = quad(C)
    grad_omega = quad(CF) if has_omega else jnp.zeros_like(omega)
    Y = CF.T @ L
    GY = Q_K @ (h[:, None] * (Q_K.T @ Y)) - QL @ (QL.T @ Y)
    if reduced:
        GY = GY + sA * Y
    grad_L = 2 * CF @ GY
    # the sample side, as in lowrank_g.loglik_G_value_and_grad
    sD = jnp.sqrt(D_K)
    QZ = sD[:, None] * (Q_K.T @ W)
    PGQ = P @ RQ
    E = QZ - (sD[:, None] * Lam * D_K[:, None]) @ PGQ.T
    grad_G = -E.T @ E + P * D_K.sum() - (PGQ * (D_K ** 2 @ Mi)) @ PGQ.T
    grad_nu = jax.ops.segment_sum(grad_G.diagonal(), gii, num_segments=num_groups)
    grad_V = 2 * grad_G @ V
    return value, grad_tau, grad_omega, grad_L, grad_nu, grad_V


_value_and_grad = jax.jit(loglik_S_value_and_grad,
                          static_argnames=('num_groups', 'has_omega', 'identity'))


def _loglik_S_with_spectral(tau, omega, L, nu, V, W, C, CF, P, gii, num_groups, has_omega=True,
                            identity=False):
    # loglik_S_value_and_grad and the spectral pieces it was computed from.
    spec = _spectral(tau, omega, L, nu, V, C, CF, P, gii, has_omega)
    return _loglik_S_from_spectral(spec, omega, L, V, W, C, CF, P, gii, num_groups, has_omega,
                                   identity), spec


_value_grad_spectral = jax.jit(_loglik_S_with_spectral,
                               static_argnames=('num_groups', 'has_omega', 'identity'))


def _loglik_S_with_spectral_reduced(Phi, omega, L, nu, V, W, C, CF, P, gii, num_groups,
                                    has_omega=True, identity=False):
    # _loglik_S_with_spectral with K = Phi Phi^T (see _spectral_reduced).
    spec = _spectral_reduced(Phi, nu, V, P, gii)
    return _loglik_S_from_spectral(spec, omega, L, V, W, C, CF, P, gii, num_groups, has_omega,
                                   identity), spec


_value_grad_spectral_reduced = jax.jit(_loglik_S_with_spectral_reduced,
                                       static_argnames=('num_groups', 'has_omega', 'identity'))
_spectral_reduced_jit = jax.jit(_spectral_reduced)


@jax.jit
def _fim_block(N12, U1, U2, d, e):
    # 1/2 sum_k e_k^2 W_k o W_k with W_k = N12 - e_k U1 diag(1 / (1 + e_k d)) U2^T: the
    # information between parameters that add rank-one terms u u^T to one side of the
    # Kronecker covariance (cf. fit._fim_quadratic_block).
    def body(k, mx):
        W_k = N12 - e[k] * (U1 / (1 + e[k] * d)) @ U2.T
        return mx + e[k] ** 2 * W_k * W_k
    return jax.lax.fori_loop(0, len(e), body, jnp.zeros((U1.shape[0], U2.shape[0]))) / 2


def _kernel_factor(A, tol: float):
    # The eigenvalues w of A A^T (and of A^T A) down to tol times the largest, the eigenvectors
    # of A A^T and those of A^T A, from the Gram matrix A^T A (A has few columns).
    w, Vt = np.linalg.eigh(A.T @ A)
    keep = w > tol * max(w[-1], np.finfo(np.float64).tiny)
    w, Vt = w[keep][::-1], Vt[:, keep][:, ::-1]
    return w, A @ Vt / np.sqrt(w), Vt


@jax.jit
def _fim_block_lowrank(E1, E2, U, w):
    # 1/2 sum_l w_l W_l o W_l with W_l = E1 diag(U[:, l]) E2^T: _fim_block with its kernel
    # sum_k e_k^2 (1 / (1 + e_k d)) (1 / (1 + e_k d))^T factored as U diag(w) U^T.
    def body(l, mx):
        W_l = (E1 * U[:, l]) @ E2.T
        return mx + w[l] * W_l * W_l
    return jax.lax.fori_loop(0, U.shape[1], body, jnp.zeros((E1.shape[0], E2.shape[0]))) / 2


@partial(jax.jit, static_argnames=('size',))
def _fim_block_lowrank_sym(E, U, w, size: int = 1024):
    # _fim_block_lowrank(E, E, U, w) from the blocks of rows on and above the diagonal only.
    cuts = list(range(0, E.shape[0], size)) + [E.shape[0]]
    out = jnp.zeros((E.shape[0], E.shape[0]))
    for a in range(len(cuts) - 1):
        for b in range(a, len(cuts) - 1):
            ra, rb = slice(cuts[a], cuts[a + 1]), slice(cuts[b], cuts[b + 1])
            blk = _fim_block_lowrank(E[ra], E[rb], U, w)
            if b == a:
                blk = (blk + blk.T) / 2
            out = out.at[ra, rb].set(blk)
            if b > a:
                out = out.at[rb, ra].set(blk.T)
    return out


def _fim_sample_side(P, PGQ, D_A, D_K, Mi, tol: float) -> np.ndarray:
    """1/2 sum_a D_K[a]^2 W_a o W_a with W_a = P - D_K[a] PGQ diag(Mi[a]) PGQ^T (s x s): the
    information between the entries of the diagonal of G, as _fim_block(P, PGQ, PGQ, D_A, D_K).
    With N = P - PGQ_b diag(1 / D_A_b) PGQ_b^T over the columns b with D_A_b max(D_K) >= 1,
        W_a = N + PGQ diag(h_a) PGQ^T,  h_ak = Mi_ak / D_A_k for k in b, -D_K[a] Mi_ak otherwise.
    No term of W_a loses more than half of its size to another (D_K[a] D_A_k < 1 outside b),
    and the sum over a runs over the eigenvectors of the kernel sum_a D_K[a]^2 h_a h_a^T down
    to tol times the largest eigenvalue."""
    big = D_A * max(float(D_K.max()), 0.0) >= 1.0
    N = P - (PGQ[:, big] / D_A[big]) @ PGQ[:, big].T
    N = (N + N.T) / 2
    h = np.where(big, Mi / np.where(big, D_A, 1.0), -D_K[:, None] * Mi)
    H = D_K[:, None] * h
    res = (D_K ** 2).sum() * N * N + 2 * N * ((PGQ * (D_K @ H)) @ PGQ.T)
    # The kernel is factored for the columns of PGQ scaled to unit length, so that tol is
    # relative to the contributions of the terms.
    q2 = (PGQ ** 2).sum(axis=0)
    Hn = H * q2
    if Hn.any():
        w, _, Vt = _kernel_factor(Hn, tol)
        jPn = jnp.asarray(PGQ / np.sqrt(np.where(q2 > 0, q2, 1.0)))
        res = res + 2 * np.asarray(_fim_block_lowrank(jPn, jPn, jnp.asarray(Vt), jnp.asarray(w)))
    return (res + res.T) / 4


class MotifCovarianceProblem():
    """The fixed inputs of the REML deviance of the structured motif covariance: the loadings
    in B-space (C), the data (W), the projected inverse P of the error covariance between
    samples and the families (F; None for the identity).
    When the motifs and families with nonzero variance (and the columns of L) number at most
    reduce_fraction times the rank r of C, K is decomposed through those columns only (see
    _spectral_reduced), and their count is rounded up to a multiple of reduce_step (r / 16 and
    at least 64 by default) so that few sizes get compiled."""
    reduce_fraction = 0.7
    reduce_step = None

    def __init__(self, Z, B_decomposition, P, gii, group_inds, F=None, omega_free=None):
        self.C = b_space(B_decomposition)
        self.W = b_space_data(Z, B_decomposition)
        self.P = np.asarray(P, dtype=np.float64)
        self.gii = np.asarray(gii)
        self.group_inds = group_inds
        self.identity = F is None
        self.F = None if F is None else np.asarray(F, dtype=np.float64)
        self.CF = self.C if F is None else self.F.T @ self.C
        m, f = len(self.C), len(self.CF)
        if omega_free is None:
            omega_free = np.zeros(f, dtype=bool) if F is None else self.F.sum(axis=0) > 1
        self.omega_free = np.asarray(omega_free, dtype=bool)
        self.has_omega = bool(self.omega_free.any())
        self.m, self.f, self.g, self.s = m, f, len(group_inds), len(self.gii)
        self._jW, self._jC, self._jCF, self._jP = (jnp.asarray(x) for x in (self.W, self.C, self.CF, self.P))
        self._jgii = jnp.asarray(self.gii)
        self.r = self.C.shape[1]
        # The spectral pieces of the last evaluation and the parameters they belong to.
        self._spec_key, self._spec = None, None
        self._grams = dict()

    @staticmethod
    def _key(*params) -> tuple:
        return tuple((np.shape(x), np.asarray(x, dtype=np.float64).tobytes()) for x in params)

    def _reduction(self, tau, omega, L):
        # Phi (r x n) with K = Phi Phi^T for _spectral_reduced, or None when the full K is cheaper.
        tau, omega, L = (np.asarray(x, dtype=np.float64) for x in (tau, omega, L))
        act = np.where(tau > 0)[0]
        fam = np.where(omega > 0)[0] if self.has_omega else np.zeros(0, dtype=int)
        n = len(act) + len(fam) + L.shape[1]
        step = self.reduce_step or max(64, -(-self.r // 16))
        n_b = -(-max(n, 1) // step) * step
        if n > self.reduce_fraction * self.r or n_b >= self.r:
            return None
        Phi = np.zeros((self.r, n_b))
        Phi[:, :len(act)] = self.C[act].T * np.sqrt(tau[act])
        Phi[:, len(act):len(act) + len(fam)] = self.CF[fam].T * np.sqrt(omega[fam])
        Phi[:, len(act) + len(fam):n] = self.CF.T @ L
        return Phi

    def value_and_grad(self, tau, omega, L, nu, V):
        Phi = self._reduction(tau, omega, L)
        args = (jnp.asarray(omega), jnp.asarray(L), jnp.asarray(nu), jnp.asarray(V), self._jW, self._jC,
                self._jCF, self._jP, self._jgii)
        kw = dict(num_groups=self.g, has_omega=self.has_omega, identity=self.identity)
        if Phi is None:
            res, spec = _value_grad_spectral(jnp.asarray(tau), *args, **kw)
        else:
            res, spec = _value_grad_spectral_reduced(jnp.asarray(Phi), *args, **kw)
        self._spec_key, self._spec = self._key(tau, omega, L, nu, V), spec
        return float(res[0]), tuple(np.asarray(r, dtype=np.float64) for r in res[1:])

    def spectral(self, tau, omega, L, nu, V):
        """(D_K, Q_K, D_A, RQ, Mi) at the parameters, from the last evaluation of value_and_grad
        when it was at the same parameters. Q_K has fewer than r columns when K is decomposed
        through the columns of nonzero variance; its eigenvalue is zero on the rest."""
        key = self._key(tau, omega, L, nu, V)
        if key != self._spec_key:
            self._spec_key = key
            Phi = self._reduction(tau, omega, L)
            if Phi is None:
                self._spec = _spectral(*(jnp.asarray(x) for x in (tau, omega, L, nu, V)),
                                       self._jC, self._jCF, self._jP, self._jgii, self.has_omega)
            else:
                self._spec = _spectral_reduced_jit(jnp.asarray(Phi), jnp.asarray(nu), jnp.asarray(V),
                                                   self._jP, self._jgii)
        return tuple(np.asarray(x, dtype=np.float64) for x in self._spec)

    def _resolvent(self, Q_K, RQ, Mi):
        # QL = sum_j M_j w_j (r x s', w_j the columns of W RQ) and whether Q_K is reduced.
        WR = self.W @ RQ
        What = Q_K.T @ WR
        QL = Q_K @ (What * Mi)
        reduced = Q_K.shape[1] < self.r
        if reduced:
            QL = QL + WR - Q_K @ What
        return QL, reduced

    @staticmethod
    def _row_terms(rows, Q_K, Mi, reduced):
        # E = rows Q_K and the diagonals c^T M_j c (rows x s') for every row c.
        E = rows @ Q_K
        S = E ** 2 @ Mi
        if reduced:
            S = S + np.clip((rows ** 2).sum(axis=1) - (E ** 2).sum(axis=1), 0.0, None)[:, None]
        return E, S

    def _gram(self, name: str, A, B) -> np.ndarray:
        if name not in self._grams:
            self._grams[name] = A @ B.T
        return self._grams[name]

    def family_gradient(self, tau, omega, L, nu, V) -> np.ndarray:
        """C_F Gamma C_F^T (f x f): the derivative of the deviance with respect to the
        covariance F^T-side matrix Psi = diag(omega) + L L^T."""
        D_K, Q_K, D_A, RQ, Mi = self.spectral(tau, omega, L, nu, V)
        QL, reduced = self._resolvent(Q_K, RQ, Mi)
        if not reduced:
            Gamma = (Q_K * (Mi @ D_A)) @ Q_K.T - QL @ QL.T
            return self.CF @ Gamma @ self.CF.T
        sA = D_A.sum()
        EF, FL = self.CF @ Q_K, self.CF @ QL
        return (EF * (Mi @ D_A - sA)) @ EF.T + sA * self._gram('FF_all', self.CF, self.CF) - FL @ FL.T

    def information_diagonal(self, tau, omega, L, nu, V) -> tuple:
        """The diagonals of the Fisher information of tau and of omega (of all families)."""
        D_K, Q_K, D_A, RQ, Mi = self.spectral(tau, omega, L, nu, V)
        reduced = Q_K.shape[1] < self.r
        res = list()
        for rows in (self.C, self.CF):
            E = self._row_terms(rows, Q_K, Mi, reduced)[1]
            res.append(0.5 * (E ** 2 * D_A ** 2).sum(axis=1))
        return tuple(res)

    def fim(self, tau, omega, L, nu, V, fix_ind=None, kernel_tol: float = 1e-13) -> np.ndarray:
        """Fisher information of (tau, nu without nu[fix_ind], the free omega), with L and V
        held fixed: 1/2 tr(V^-1 dV_i V^-1 dV_j) for the covariance V of the projected data.
        The blocks of tau and omega are sums over the s eigenvalues e_k of the sample side of
        terms (E diag(M_k) E^T)^2, M_k = 1 / (1 + e_k D_K). The kernel sum_k e_k^2 M_k M_k^T has
        a small numerical rank (M is a Cauchy-like matrix), and the sums run over its
        eigenvectors down to kernel_tol times the largest eigenvalue; the block of nu likewise
        (see _fim_sample_side). kernel_tol = None sums over all eigenvalues."""
        D_K, Q_K, D_A, RQ, Mi = self.spectral(tau, omega, L, nu, V)
        reduced = Q_K.shape[1] < self.r
        Qs = Q_K * np.sqrt(D_K)
        U_C = self.C @ Qs
        PGQ = self.P @ RQ

        def complement(E1, E2, gram):
            # The terms of the eigenvectors of K outside Q_K (eigenvalue zero, M = 1) with
            # W_k = E1 diag(M_k) E2^T + N0, N0 = rows1 (I - Q_K Q_K^T) rows2^T.
            N0 = gram - E1 @ E2.T
            return ((D_A ** 2).sum() * N0 * N0 + 2 * N0 * ((E1 * (Mi @ D_A ** 2)) @ E2.T)) / 2
        loadings = np.zeros((self.s, self.g))
        for i, inds in enumerate(self.group_inds):
            loadings[inds, i] = 1.0
        jD_K, jD_A = jnp.asarray(D_K), jnp.asarray(D_A)
        if kernel_tol is not None:
            # The kernel is factored for the columns of E = C Q_K scaled to unit length, so
            # that kernel_tol is relative to the contributions of the terms.
            E_raw = self.C @ Q_K
            n2 = (E_raw ** 2).sum(axis=0)
            n2 = np.where(n2 > 0, n2, 1.0)
            w_t, U_t, _ = _kernel_factor(n2[:, None] * Mi * D_A, kernel_tol)
            jU_t, jw_t = jnp.asarray(U_t), jnp.asarray(w_t)
            E_C = jnp.asarray(E_raw / np.sqrt(n2))
            F_tt = np.asarray(_fim_block_lowrank_sym(E_C, jU_t, jw_t))
            if reduced:
                F_tt = F_tt + complement(E_raw, E_raw, self._gram('CC', self.C, self.C))
                F_tt = (F_tt + F_tt.T) / 2
            F_nn = _fim_sample_side(self.P, PGQ, D_A, D_K, Mi, kernel_tol)
        else:
            F_tt = np.asarray(_fim_block(jnp.asarray(self.C @ self.C.T), jnp.asarray(U_C), jnp.asarray(U_C), jD_K, jD_A))
            F_nn = np.asarray(_fim_block(jnp.asarray(self.P), jnp.asarray(PGQ), jnp.asarray(PGQ), jD_A, jD_K))
        F_nn = loadings.T @ F_nn @ loadings
        Mi2 = Mi ** 2
        F_tn = (U_C ** 2 @ Mi2 @ (PGQ ** 2).T / 2) @ loadings
        free_nu = np.array([i for i in range(self.g) if i != fix_ind], dtype=int)
        F_nn = F_nn[np.ix_(free_nu, free_nu)]
        F_tn = F_tn[:, free_nu]
        blocks = [[F_tt, F_tn], [F_tn.T, F_nn]]
        if self.has_omega:
            CF = self.CF[self.omega_free]
            U_F = CF @ Qs
            if kernel_tol is not None:
                EF_raw = CF @ Q_K
                E_F = jnp.asarray(EF_raw / np.sqrt(n2))
                F_oo = np.asarray(_fim_block_lowrank_sym(E_F, jU_t, jw_t))
                F_to = np.asarray(_fim_block_lowrank(E_C, E_F, jU_t, jw_t))
                if reduced:
                    F_oo = F_oo + complement(EF_raw, EF_raw, self._gram('FF', CF, CF))
                    F_oo = (F_oo + F_oo.T) / 2
                    F_to = F_to + complement(E_raw, EF_raw, self._gram('CF', self.C, CF))
            else:
                F_oo = np.asarray(_fim_block(jnp.asarray(CF @ CF.T), jnp.asarray(U_F), jnp.asarray(U_F), jD_K, jD_A))
                F_to = np.asarray(_fim_block(jnp.asarray(self.C @ CF.T), jnp.asarray(U_C), jnp.asarray(U_F), jD_K, jD_A))
            F_on = (U_F ** 2 @ Mi2 @ (PGQ ** 2).T / 2) @ loadings[:, free_nu]
            blocks = [[F_tt, F_tn, F_to], [F_tn.T, F_nn, F_on.T], [F_to.T, F_on, F_oo]]
        return np.block(blocks)

    def lrt(self, tau, omega, L, nu, V) -> tuple:
        """Likelihood-ratio statistics (dev under H0 minus dev) of tau_i = 0 for every motif
        and of omega_j = 0 for every family, the other parameters held at the supplied values.

        tau_i adds tau_i c c^T to K (c a row of C), so setting it to zero is a rank-one
        downdate: with M_j = (I + a_j K)^-1,
            lrt = sum_j tau_i (c^T M_j w_j)^2 / rho_j + log rho_j,    rho_j = 1 - a_j tau_i c^T M_j c.

        For omega_j the variances tau of the motifs of the family take over omega_j under H0:
        the variances of the motifs stay, only their covariance within the family goes. Were
        tau held fixed, the test would also reward the variance on the diagonal. tau explains
        that variance as well. This changes K by U diag(lam) U^T with U the rows of the family in
        C_F and of its t - 1 motifs in C, lam = omega_j (-1, 1, ..., 1), and
            lrt = sum_j -v_j^T (diag(1 / lam) + a_j G_j)^-1 v_j + log det(I + a_j diag(lam) G_j),
        G_j = U^T M_j U, v_j = U^T M_j w_j."""
        D_K, Q_K, D_A, RQ, Mi = self.spectral(tau, omega, L, nu, V)
        tau = np.asarray(tau, dtype=np.float64)
        QL, reduced = self._resolvent(Q_K, RQ, Mi)
        E, S = self._row_terms(self.C, Q_K, Mi, reduced)
        rho = 1 - tau.reshape(-1, 1) * S * D_A.reshape(1, -1)
        rho = np.clip(rho, np.finfo(np.float64).tiny, 1.0)
        mu = self.C @ QL
        lrt_tau = np.clip((tau.reshape(-1, 1) * mu ** 2 / rho + np.log(rho)).sum(axis=1), 0.0, None)
        if not self.has_omega:
            return lrt_tau, None
        lrt_omega = np.zeros(self.f)
        fams = np.where(self.omega_free & (np.asarray(omega) > 0))[0]
        if not len(fams):
            return lrt_tau, lrt_omega
        members = [np.where(self.F[:, j] != 0)[0] for j in fams]
        sizes = np.array([len(x) for x in members])
        EF, FL = self.CF[fams] @ Q_K, self.CF[fams] @ QL
        # The families with the same number of motifs are tested together.
        for size in np.unique(sizes):
            sel = np.where(sizes == size)[0]
            mem = np.array([members[i] for i in sel])
            t = size + 1
            rows = [self.CF[fams[sel]]] + [self.C[mem[:, a]] for a in range(size)]
            U = [EF[sel]] + [E[mem[:, a]] for a in range(size)]
            v = np.stack([FL[sel]] + [mu[mem[:, a]] for a in range(size)], axis=-1)
            G = np.empty((len(sel), len(D_A), t, t))
            for a in range(t):
                for b in range(a, t):
                    gab = (U[a] * U[b]) @ Mi
                    if reduced:
                        gab = gab + ((rows[a] * rows[b]).sum(axis=1) - (U[a] * U[b]).sum(axis=1))[:, None]
                    G[:, :, a, b] = gab
                    G[:, :, b, a] = gab
            lam = np.repeat(np.asarray(omega, dtype=np.float64)[fams[sel]].reshape(-1, 1), t, axis=1)
            lam[:, 0] = -lam[:, 0]
            A = D_A.reshape(1, -1, 1, 1) * G
            A[..., np.arange(t), np.arange(t)] += (1 / lam)[:, None, :]
            quad = np.einsum('fka,fka->fk', v, np.linalg.solve(A, v[..., None])[..., 0])
            logdet = np.linalg.slogdet(np.eye(t) + D_A.reshape(1, -1, 1, 1) * lam[:, None, :, None] * G)[1]
            lrt_omega[fams[sel]] = np.clip(logdet.sum(axis=1) - quad.sum(axis=1), 0.0, None)
        return lrt_tau, lrt_omega


# ------------------------------------------------------------------------ estimation ---
def fisher_scoring(problem: MotifCovarianceProblem, x0, n: int, assemble, vg, fix_ind: int, stationarity,
                   max_steps: int = 100, verbose: bool = False, tag: str = 'motif covariance',
                   tol: float = FISHER_TOL) -> tuple:
    """Fisher scoring for the variances (tau, the free nu, the free omega: the first n entries
    of x0) with L and V absent. The expected Hessian of the deviance is twice the Fisher
    information I. On real data the model is misspecified, the observed Hessian differs from
    2 I, and plain scoring converges only linearly near the optimum; a secant correction S, as
    in NL2SOL, collects that difference from successive gradients, and H = 2 I + S. S starts at
    zero and is cleared whenever the set of variances at zero changes, so scoring is plain until
    that set settles. A step solves (H + lam diag(2 I)) d = -grad over the variances that are
    free to move: a variance at zero whose gradient is positive stays at zero. A step is
    clipped at zero and accepted if it lowers the deviance; otherwise lam grows tenfold and
    the step is recomputed, and after an accepted step lam falls threefold. Stops when the
    stationarity of estimate_motif_covariance is at most tol (tighter than the L-BFGS-B stage:
    a step near the optimum costs one evaluation and one information matrix), after max_steps
    information matrices, or when lam exceeds 1e8. Returns the new x, the number of steps and
    of evaluations."""
    x = np.array(x0, dtype=np.float64)
    dev, grad = vg(x)
    lam, n_steps, n_ev = 1e-3, 0, 1
    dev_start = dev
    S, prev = np.zeros((n, n)), None
    for _ in range(max_steps):
        if stationarity(x, grad) <= tol:
            break
        F2 = 2 * problem.fim(*assemble(x), fix_ind=fix_ind)
        F2 = 0.5 * (F2 + F2.T)
        zero = x[:n] <= 0
        if prev is not None and not np.array_equal(zero, prev[2]):
            S[:] = 0.0
            prev = None
        if prev is not None:
            s_, y_ = x[:n] - prev[0], grad[:n] - prev[1]
            ys = y_ @ s_
            if ys > 1e-12 * np.linalg.norm(y_) * np.linalg.norm(s_):
                # The update of NL2SOL: S is sized down when it overstates the curvature along s_,
                # then H s_ = y_ is enforced with a symmetric rank-2 change of S.
                Ss = S @ s_
                sSs = s_ @ Ss
                if sSs != 0:
                    S *= min(1.0, abs(s_ @ (y_ - F2 @ s_)) / abs(sSs))
                    Ss = S @ s_
                res = y_ - F2 @ s_ - Ss
                S += (np.outer(res, y_) + np.outer(y_, res)) / ys - (res @ s_) * np.outer(y_, y_) / ys ** 2
        g = grad[:n]
        free = ~(zero & (g > 0))
        Hf = (F2 + S)[np.ix_(free, free)]
        dg = np.clip(np.diag(F2)[free], np.finfo(np.float64).tiny, None)
        n_steps += 1
        accepted = False
        while lam <= 1e8:
            try:
                step = np.linalg.solve(Hf + lam * np.diag(dg), -g[free])
            except np.linalg.LinAlgError:
                lam *= 10
                continue
            x_new = x.copy()
            x_new[np.where(free)[0]] = np.clip(x[free] + step, 0.0, None)
            dev_new, grad_new = vg(x_new)
            n_ev += 1
            if np.isfinite(dev_new) and dev_new < dev:
                prev = (x[:n].copy(), g.copy(), zero)
                x, dev, grad = x_new, dev_new, grad_new
                lam = max(lam / 3, 1e-9)
                accepted = True
                break
            lam *= 10
        if not accepted:
            break
    vg(x)
    logger_print(f'[{tag}] Fisher scoring: {n_steps} steps, {n_ev} evaluations, deviance '
                 f'{dev_start:.3f} -> {dev:.3f}', verbose)
    return x, n_steps, n_ev


def estimate_motif_covariance(problem: MotifCovarianceProblem, rank: int, Sigma0, nu0, V0,
                              fix_ind: int, fix_val: float, maxiter: int = 5000,
                              verbose: bool = False, silent: bool = False,
                              fisher_steps: int = 100, stage: str = None,
                              tag: str = 'motif covariance') -> dict:
    """Joint REML fit of (tau, nu, omega, L, V) starting from the model with a diagonal
    motif covariance (Sigma0, nu0, V0). One nu (fix_ind) is held at fix_val to pin the scale
    of S against G. With families and L, the model without L is fitted first, and its estimates
    are the start of tau, nu, omega and V; otherwise omega starts at a Newton step from zero.
    L starts along the directions in which the data have more covariance between families than
    the model, i.e. the leading eigenvectors of -d dev / d Psi at that start, at the scale that
    is the best of a grid. All variances are
    then optimised under non-negativity bounds, each measured in units of its standard error
    at the start (tau and omega share the diagonal of S, and the problem is badly conditioned
    otherwise). The optimum is regarded as reached when no variance changes the deviance by
    more than STATIONARITY_TOL per unit change of its log, and no entry of L or V does per
    change by its root mean square, or when the deviance has fallen by less than STALL_TOL over
    the last STALL_ITERATIONS iterations. If the result does not improve on the starting model,
    the latter is returned (with omega = 0 and an L with no columns), and if L does not improve
    on the model without it, that model is returned.
    Without L and V, at most fisher_steps steps of Fisher scoring come before L-BFGS-B (see
    fisher_scoring): with families, omega and the tau of the members of a family trade off
    nearly one for one, and L-BFGS-B scaled by the diagonal of the information crawls along that
    ridge. L-BFGS-B does not run if Fisher scoring ends at the optimum.
    stage names the estimation in messages, tag in the log lines.
    Returns a dict with tau, nu, omega, L, V, dev (the deviance), dev0 (that of the start),
    the numbers of iterations and evaluations of both stages, n_it and n_ev, dev_fisher (the
    deviance after Fisher scoring, None without it), and converged, stationarity, message of the
    last stage and grad_norm, the norm of the gradient with respect to the logs of the
    variances (zero for a variance at zero with a positive derivative)."""
    from scipy.optimize import minimize
    if stage is None:
        stage = 'covariance of activities between motifs (--motif-families / --motif-lowrank)'
    m, f, g, s = problem.m, problem.f, problem.g, problem.s
    k = int(rank)
    if k >= max(f, 1):
        what = 'motifs' if problem.identity else 'families'
        raise InputError(f'--motif-lowrank {k} cannot be estimated: there are {f} {what}, and the rank of '
                         'the covariance between them must be smaller than that. Use a smaller rank.')
    Sigma0 = np.asarray(Sigma0, dtype=np.float64)
    nu0 = np.asarray(nu0, dtype=np.float64).copy()
    nu0[fix_ind] = fix_val
    V0 = np.zeros((s, 0)) if V0 is None else np.asarray(V0, dtype=np.float64).reshape(s, -1)
    kv = V0.shape[1]
    free_nu = np.array([i for i in range(g) if i != fix_ind], dtype=int)
    free_om = np.where(problem.omega_free)[0]
    n_om = len(free_om)
    sizes = [m, len(free_nu), n_om, f * k, s * kv]
    offsets = np.cumsum([0] + sizes)
    n_var = offsets[3]

    def assemble(x):
        x = np.asarray(x, dtype=np.float64)
        tau = x[offsets[0]:offsets[1]]
        nu = np.full(g, fix_val)
        nu[free_nu] = x[offsets[1]:offsets[2]]
        omega = np.zeros(f)
        omega[free_om] = x[offsets[2]:offsets[3]]
        L = x[offsets[3]:offsets[4]].reshape(f, k)
        V = x[offsets[4]:offsets[5]].reshape(s, kv)
        return tau, omega, L, nu, V

    def pack(tau, omega, L, nu, V):
        return np.concatenate([tau, nu[free_nu], omega[free_om], np.asarray(L).reshape(-1),
                               np.asarray(V).reshape(-1)])

    last = dict()

    def vg(x):
        try:
            value, (g_tau, g_omega, g_L, g_nu, g_V) = problem.value_and_grad(*assemble(x))
        except Exception as e:
            raise RuntimeError(f'Estimation of the {stage} failed while evaluating the objective '
                               f'({type(e).__name__}: {e})') from e
        grad = np.concatenate([g_tau, g_nu[free_nu], g_omega[free_om], g_L.reshape(-1), g_V.reshape(-1)])
        last.update(x=np.array(x), grad=grad, value=float(value))
        return value, grad

    omega0 = np.zeros(f)
    L_zero = np.zeros((f, k))
    dev0 = vg(pack(Sigma0, omega0, L_zero, nu0, V0))[0]
    if not np.isfinite(dev0):
        raise RuntimeError(f'Estimation of the {stage}: the objective is not finite at the starting point.')
    n_it = n_ev = 0
    first = None
    if k and n_om:
        # The model without L is fitted first, and its estimates are the start of the model with L.
        first = estimate_motif_covariance(problem, 0, Sigma0, nu0, V0, fix_ind, fix_val, maxiter=maxiter,
                                          verbose=False, silent=True)
        n_it, n_ev = first['n_it'], first['n_ev']
        if first['improved']:
            Sigma0, nu0, V0, omega0 = first['tau'], first['nu'], first['V'], first['omega'].copy()
        else:
            first = None
    tiny = np.finfo(np.float64).tiny
    scale_tau = max(float(np.max(Sigma0)), tiny)
    if n_om and first is None:
        # A Newton step for omega from zero, the expected Hessian of the deviance being twice
        # the Fisher information.
        _, (_, g_omega, _, _, _) = problem.value_and_grad(Sigma0, omega0, L_zero, nu0, V0)
        info = problem.information_diagonal(Sigma0, omega0, L_zero, nu0, V0)[1]
        with np.errstate(divide='ignore', invalid='ignore'):
            step = np.where(info > 0, -g_omega / (2 * info), 0.0)
        omega0[free_om] = np.clip(step[free_om], 0.0, scale_tau)
    L0 = L_zero
    if k:
        # dev(t L) = dev(0) + t^2 tr(L^T A L) + O(t^4) with A = d dev / d Psi: the best directions
        # for the columns of L are the eigenvectors of A with the most negative eigenvalues.
        A = problem.family_gradient(Sigma0, omega0, L_zero, nu0, V0)
        w, v = np.linalg.eigh(0.5 * (A + A.T))
        w = -w[:k]
        v = v[:, :k] * np.sqrt(np.clip(w, 0.0, None) / max(w[0], tiny)).reshape(1, -1)
        if not np.any(v):
            v = np.eye(f, k)
        best = None
        for c in 10.0 ** np.arange(-3.0, 1.01, 0.5):
            L_c = v * (c * scale_tau ** 0.5)
            dev_c = vg(pack(Sigma0, omega0, L_c, nu0, V0))[0]
            if np.isfinite(dev_c) and (best is None or dev_c < best[0]):
                best = (dev_c, L_c)
        if best is not None:
            L0 = best[1]
    x0 = pack(Sigma0, omega0, L0, nu0, V0)
    scale = np.concatenate([np.repeat(scale_tau, m), np.repeat(max(fix_val, tiny), len(free_nu)),
                            np.repeat(scale_tau, n_om), np.repeat(scale_tau ** 0.5, f * k),
                            np.repeat(max(float(np.mean(nu0[problem.gii])), tiny) ** 0.5, s * kv)])
    # The standard error of a variance is about (2 I_ii)^-1/2, I the Fisher information.
    info_tau, info_om = problem.information_diagonal(*assemble(x0))
    info = np.concatenate([info_tau, np.zeros(len(free_nu)), info_om[free_om]])
    known = info > 0
    scale[:n_var][known] = (2 * info[known]) ** -0.5
    fun_scale = 1.0 / max(abs(dev0), 1.0)

    def stationarity_terms(x, grad) -> np.ndarray:
        # The change of the deviance per unit change of the log of every variance, per change of
        # an entry of L or V by their root mean square.
        var, g_var = x[:n_var], grad[:n_var]
        terms = [np.where(var > 0, np.abs(var * g_var), np.where(g_var < 0, -g_var * scale[:n_var], 0.0))]
        for a, b in ((offsets[3], offsets[4]), (offsets[4], offsets[5])):
            if b > a:
                rms = max(float(np.sqrt(np.mean(x[a:b] ** 2))), scale[a])
                terms.append(np.abs(grad[a:b]) * rms)
        return np.concatenate(terms)

    def stationarity(x, grad) -> float:
        return float(np.max(stationarity_terms(x, grad), initial=0.0))

    dev_fisher = None
    if fisher_steps and not k and not kv:
        x0, n_fs, n_fe = fisher_scoring(problem, x0, n_var, assemble, vg, fix_ind, stationarity,
                                        max_steps=fisher_steps, verbose=verbose and not silent, tag=tag)
        n_it += n_fs
        n_ev += n_fe
        dev_fisher = last['value']

    history = list()

    def callback(intermediate_result):
        x = intermediate_result.x * scale
        if 'x' not in last or not np.array_equal(last['x'], x):
            vg(x)
        history.append(last['value'])
        if stationarity(last['x'], last['grad']) <= STATIONARITY_TOL:
            raise StopIteration
        if len(history) > STALL_ITERATIONS and history[-STALL_ITERATIONS - 1] - history[-1] < STALL_TOL:
            last['stalled'] = True
            raise StopIteration

    def vg_scaled(u):
        value, grad = vg(u * scale)
        return fun_scale * value, fun_scale * grad * scale
    bounds = [(0.0, None)] * n_var + [(None, None)] * (len(x0) - n_var)
    x, dev, converged, message = x0, vg(x0)[0], False, ''
    # L-BFGS-B is skipped when Fisher scoring has reached the optimum.
    if dev_fisher is not None and stationarity(last['x'], last['grad']) <= STATIONARITY_TOL:
        converged, message = True, 'Fisher scoring converged'
    for _ in range(0 if converged else 3):
        history.clear()
        last['stalled'] = False
        sol = minimize(vg_scaled, x / scale, jac=True, method='L-BFGS-B', bounds=bounds,
                       callback=callback, options=dict(maxiter=maxiter, maxfun=2 * maxiter,
                                                       maxcor=100, ftol=1e-14, gtol=1e-10))
        n_it += int(sol.nit)
        n_ev += int(sol.nfev)
        improved = np.isfinite(sol.fun) and sol.fun / fun_scale < dev
        if improved:
            gain = dev - sol.fun / fun_scale
            x, dev = sol.x * scale, float(sol.fun) / fun_scale
        vg(x)
        converged = stationarity(last['x'], last['grad']) <= STATIONARITY_TOL or last['stalled']
        message = str(sol.message)
        # L-BFGS-B also stops on a relative decrease of the objective: a restart from the
        # solution with a fresh Hessian approximation goes on if the last run made progress.
        if converged or not improved or gain < 1e-6 * max(abs(dev), 1.0):
            break
    stat = stationarity(last['x'], last['grad'])
    grad_norm = float(np.linalg.norm(stationarity_terms(last['x'], last['grad'])))
    tau, omega, L, nu, V = assemble(x)
    if not (np.isfinite(dev) and np.all(np.isfinite(x))):
        raise RuntimeError(f'Estimation of the {stage} ended at a non-finite value.')
    if first is not None and not dev < first['dev']:
        # L brought nothing over the model without it.
        tau, omega, nu, V, dev = first['tau'], first['omega'], first['nu'], first['V'], first['dev']
        L, converged = np.zeros((f, 0)), True
    logger_print(f'[{tag}] {n_it} iterations, {n_ev} evaluations of the deviance '
                 f'(converged={converged}{", stalled" if last.get("stalled") else ""})', verbose)
    if not dev < dev0:
        if not silent:
            warn('The structure of the covariance of activities between motifs (--motif-families / '
                 '--motif-lowrank) did not improve the likelihood; the model with independent motifs '
                 'is kept.')
        return dict(tau=Sigma0, nu=nu0, omega=np.zeros(f), L=np.zeros((f, 0)), V=V0, dev=float(dev0),
                    dev0=float(dev0), improved=False, n_it=n_it, n_ev=n_ev, converged=False,
                    stationarity=float('inf'), grad_norm=float('nan'), message=message, dev_fisher=dev_fisher)
    if not converged and not silent:
        warn(f'The optimisation of the {stage} did not converge ({message}); its estimates may be '
             'inaccurate.')
    return dict(tau=tau, nu=nu, omega=omega, L=L, V=V, dev=float(dev), dev0=float(dev0), improved=True,
                n_it=n_it, n_ev=n_ev, converged=bool(converged), stationarity=stat, grad_norm=grad_norm,
                message=message, dev_fisher=dev_fisher)


# -------------------------------------------------------------------------- selftest ---
def _selftest():
    """Checks of the formulas against dense computations on a small problem."""
    jax.config.update('jax_enable_x64', True)
    from .linalg import ones_nullspace_transform, ones_nullspace, lowrank_decomposition
    from . import lowrank as LR
    from .fit import loglik_motifs_value_and_grad, loglik_motifs_lrt
    rng = np.random.default_rng(0)
    print('lowrank_sigma.py self-test')
    p, m, s, f, k, kv = 60, 7, 9, 3, 2, 1
    gii = np.array([0, 0, 0, 1, 1, 2, 2, 2, 3])
    g = gii.max() + 1
    group_inds = [np.where(gii == i)[0] for i in range(g)]
    B = rng.random((p, m))
    F = np.zeros((m, f))
    F[[0, 1, 2], 0] = 1; F[[3, 4], 1] = 1; F[[5], 2] = 1; F[2, 1] = 1
    tau = rng.random(m) + 0.2; tau[4] = 0.0
    omega = rng.random(f) + 0.3; omega[2] = 0.0
    L = rng.standard_normal((f, k)) * 0.5
    nu = rng.random(g) + 0.3
    V = rng.standard_normal((s, kv)) * 0.3
    d = rng.random(g) + 0.5
    S_true = np.diag(tau) + F @ (np.diag(omega) + L @ L.T) @ F.T
    w, v = np.linalg.eigh(S_true)
    G_true = np.diag(nu[gii]) + V @ V.T
    wg, vg = np.linalg.eigh(G_true)
    U_true = (v * np.sqrt(w)) @ rng.standard_normal((m, s)) @ (vg * np.sqrt(wg)).T
    Y = B @ U_true + rng.standard_normal((p, s)) * np.sqrt(d[gii])
    Bh = ones_nullspace_transform(B); Yh = ones_nullspace_transform(Y)
    X = np.vstack([np.ones(s), rng.standard_normal(s)])
    Pj = LR.covariate_projection_theta(jnp.asarray(d[gii]), None, X)
    P = np.asarray(Pj)
    Z = Bh.T @ Yh @ P
    dec = lowrank_decomposition(Bh, verbose=False)
    prob = MotifCovarianceProblem(Z, dec, P, gii, group_inds, F=F)

    # dense deviance of the data projected on the complement of X in sample space
    Hs = np.linalg.svd(np.eye(s) - np.linalg.pinv(X) @ X)[0][:, :s - 2].T
    def dense_dev(tau, omega, L, nu, V):
        S = np.diag(tau) + F @ (np.diag(omega) + L @ L.T) @ F.T
        G = np.diag(nu[gii]) + V @ V.T
        Cov = np.kron(Hs @ G @ Hs.T, Bh @ S @ Bh.T) + np.kron(Hs @ np.diag(d[gii]) @ Hs.T, np.eye(len(Bh)))
        y = (Yh @ Hs.T).reshape(-1, order='F')
        return float(y @ np.linalg.solve(Cov, y) + np.linalg.slogdet(Cov)[1])
    args = (tau, omega, L, nu, V)
    v0 = prob.value_and_grad(*args)[0]
    shift = dense_dev(*args) - v0
    args2 = (tau * 1.3, omega * 0.7, L * 1.1, nu * 0.9, V * 0.5)
    err = abs(dense_dev(*args2) - prob.value_and_grad(*args2)[0] - shift)
    print(f'  deviance vs dense (up to a constant): {err:.2e}')
    # the diagonal model
    vd, gd = loglik_motifs_value_and_grad(np.append(tau, np.delete(nu, 0)), jnp.asarray(Z),
                                          jnp.asarray(Bh.T @ Bh), Pj, jnp.asarray(gii), g,
                                          G_fix_ind=0, G_fix_val=nu[0])
    vs, (g_tau, _, _, g_nu, _) = prob.value_and_grad(tau, np.zeros(f), np.zeros((f, 0)), nu, np.zeros((s, 0)))
    print(f'  diagonal model vs fit.loglik_motifs: value {abs(float(vd) - vs):.2e}, gradient '
          f'{np.abs(np.asarray(gd) - np.append(g_tau, np.delete(g_nu, 0))).max():.2e}')
    # gradient vs finite differences
    value, grads = prob.value_and_grad(*args)
    eps = 1e-6
    worst = 0.0
    for a, (arg, grad) in enumerate(zip(args, (grads[0], grads[1], grads[2], grads[3], grads[4]))):
        for idx in np.ndindex(np.shape(arg)):
            if a == 1 and not prob.omega_free[idx[0]]:
                continue
            ap = [np.array(x, dtype=float) for x in args]; am = [np.array(x, dtype=float) for x in args]
            ap[a][idx] += eps; am[a][idx] -= eps
            fd = (prob.value_and_grad(*ap)[0] - prob.value_and_grad(*am)[0]) / (2 * eps)
            worst = max(worst, abs(fd - grad[idx]) / max(abs(fd), 1.0))
    print(f'  gradient vs finite differences: {worst:.2e}')
    # Fisher information vs the dense formula
    def dense_cov(tau, omega, L, nu, V):
        S = np.diag(tau) + F @ (np.diag(omega) + L @ L.T) @ F.T
        G = np.diag(nu[gii]) + V @ V.T
        return np.kron(Hs @ G @ Hs.T, Bh @ S @ Bh.T) + np.kron(Hs @ np.diag(d[gii]) @ Hs.T, np.eye(len(Bh)))
    Ci = np.linalg.inv(dense_cov(*args))
    G = np.diag(nu[gii]) + V @ V.T
    S = S_true
    dV = [np.kron(Hs @ G @ Hs.T, Bh @ np.outer(e, e) @ Bh.T) for e in np.eye(m)]
    dV += [np.kron(Hs @ np.diag((gii == i).astype(float)) @ Hs.T, Bh @ S @ Bh.T) for i in range(1, g)]
    dV += [np.kron(Hs @ G @ Hs.T, Bh @ np.outer(F[:, j], F[:, j]) @ Bh.T) for j in np.where(prob.omega_free)[0]]
    fim_dense = np.array([[0.5 * np.trace(Ci @ a @ Ci @ b) for b in dV] for a in dV])
    fim = prob.fim(*args, fix_ind=0)
    fim_exact = prob.fim(*args, fix_ind=0, kernel_tol=None)
    print(f'  Fisher information vs dense: {np.abs(fim - fim_dense).max() / np.abs(fim_dense).max():.2e}, '
          f'summed over all eigenvalues: {np.abs(fim_exact - fim_dense).max() / np.abs(fim_dense).max():.2e}')
    args0 = (tau, omega, L, np.where(np.arange(g) == 1, 0.0, nu), np.zeros((s, 0)))
    f0, f0_exact = prob.fim(*args0, fix_ind=0), prob.fim(*args0, fix_ind=0, kernel_tol=None)
    print(f'  Fisher information with nu[1] = 0 and no V vs summed over all eigenvalues: '
          f'{np.abs(f0 - f0_exact).max() / np.abs(f0_exact).max():.2e}')
    # K decomposed through the columns of nonzero variance vs the full eigendecomposition
    pz = (np.where(np.isin(np.arange(m), [1, 3, 4, 6]), 0.0, tau), np.where(np.arange(f) == 0, omega, 0.0),
          L[:, :1], nu, V)
    def everything(step, fraction):
        prob.reduce_step, prob.reduce_fraction, prob._spec_key = step, fraction, None
        vg_ = prob.value_and_grad(*pz)
        out = [np.array([vg_[0]])] + list(vg_[1]) + [prob.fim(*pz, fix_ind=0), prob.fim(*pz, fix_ind=0, kernel_tol=None)]
        out += [x for x in prob.lrt(*pz)] + list(prob.information_diagonal(*pz)) + [prob.family_gradient(*pz)]
        return out, prob.spectral(*pz)[1].shape[1]
    full, n_full = everything(None, -1.0)
    for step in (1, 3):
        red, n_red = everything(step, 1.0)
        err = max(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300) for a, b in zip(red, full))
        print(f'  {n_red} of {n_full} eigenvectors (reduce_step {step}) vs all: largest relative difference {err:.2e}')
    prob.reduce_step, prob.reduce_fraction, prob._spec_key = None, MotifCovarianceProblem.reduce_fraction, None
    # the fit with Fisher scoring before L-BFGS-B vs L-BFGS-B alone
    fits = [estimate_motif_covariance(prob, 0, tau, nu, None, 0, nu[0], fisher_steps=n, silent=True)
            for n in (0, 100)]
    print(f'  fit with Fisher scoring vs without: deviance {fits[1]["dev"] - fits[0]["dev"]:+.2e}, '
          f'evaluations {fits[1]["n_ev"]} vs {fits[0]["n_ev"]}')
    # likelihood ratios vs differences of the deviance
    lrt_tau, lrt_omega = prob.lrt(*args)
    ref_tau = [prob.value_and_grad(np.where(np.arange(m) == i, 0.0, tau), omega, L, nu, V)[0] - value
               for i in range(m)]
    ref_om = [prob.value_and_grad(tau + omega[j] * F[:, j], np.where(np.arange(f) == j, 0.0, omega), L, nu, V)[0]
              - value for j in range(f)]
    print(f'  LRT of tau vs deviance differences: {np.abs(lrt_tau - np.clip(ref_tau, 0, None)).max():.2e} '
          f'({int(np.sum(np.asarray(ref_tau) > 0))} of {m} positive), of omega: '
          f'{np.abs(lrt_omega - np.where(prob.omega_free, np.clip(ref_om, 0, None), 0)).max():.2e} '
          f'({int(np.sum(np.asarray(ref_om)[prob.omega_free] > 0))} of {int(prob.omega_free.sum())} positive)')
    R = np.append(np.diag(nu[gii] ** 0.5), V, axis=1)
    lrt_old = loglik_motifs_lrt(tau, R, Z, Bh.T @ Bh, P)
    lrt_new = MotifCovarianceProblem(Z, dec, P, gii, group_inds).lrt(tau, np.zeros(m), np.zeros((m, 0)), nu, V)[0]
    print(f'  LRT of tau, diagonal model, vs fit.loglik_motifs_lrt: {np.abs(lrt_old - lrt_new).max():.2e}')
    # the BLUP with a full motif covariance vs the dense posterior mean
    Theta = np.diag(d[gii])
    Gs = np.diag(nu[gii]) + V @ V.T
    Wm = np.asarray(LR.blup_solve_W(Y, B, S_true, Gs, Theta))
    U = S_true @ (B.T @ Wm) @ Gs
    Cov = np.kron(Gs, B @ S_true @ B.T) + np.kron(Theta, np.eye(p))
    U_ref = (np.kron(Gs, S_true @ B.T) @ np.linalg.solve(Cov, Y.reshape(-1, order='F'))).reshape(m, s, order='F')
    print(f'  BLUP with a full motif covariance vs dense: '
          f'{np.linalg.norm(U - U_ref) / np.linalg.norm(U_ref):.2e}')
    # the GLS of the motif means with S, G = diag(nu) + V V^T and Theta = diag(D) + W W^T
    from types import SimpleNamespace as NS
    from .fit import estimate_motif_mean
    W_e = rng.standard_normal((s, 1)) * 0.4
    mm = estimate_motif_mean(NS(Y=Yh, B=Bh, X=X, group_inds_inv=gii), dec,
                             NS(variance=d, promotor=np.ones(p), lowrank=W_e),
                             NS(motif=tau, group=nu, lowrank=V, families=F, family_variance=omega,
                                family_lowrank=L), NS(mean=np.zeros(p)), verbose=False)
    Th = np.diag(d[gii]) + W_e @ W_e.T
    Cov = np.kron(Gs, Bh @ S_true @ Bh.T) + np.kron(Th, np.eye(len(Bh)))
    Wd = np.kron(X.T, Bh)
    H = Wd.T @ np.linalg.solve(Cov, Wd)
    M_ref = np.linalg.solve(H, Wd.T @ np.linalg.solve(Cov, Yh.reshape(-1, order='F'))).reshape(m, -1, order='F')
    print(f'  motif means with S, V and W vs dense GLS: '
          f'{np.abs(mm.mean - M_ref).max() / np.abs(M_ref).max():.2e}, Fisher information '
          f'{np.abs(mm.fim - H).max() / np.abs(H).max():.2e}')
    # the GLS of the sample means, on the data without the Helmert transform
    from .fit import estimate_sample_mean
    Y_s = Y + rng.standard_normal(s).reshape(1, -1)
    mu = estimate_sample_mean(NS(Y=Y_s, B=B, X=X, group_inds_inv=gii),
                              NS(variance=d, promotor=np.ones(p), lowrank=W_e),
                              NS(motif=tau, group=nu, lowrank=V, families=F, family_variance=omega,
                                 family_lowrank=L), NS(mean=np.zeros(p)), mm).mean.ravel()
    Cov = np.kron(Gs, B @ S_true @ B.T) + np.kron(Th, np.eye(p))
    Wd = np.kron(np.eye(s), np.ones((p, 1)))
    y = (Y_s - B @ mm.mean @ X).reshape(-1, order='F')
    mu_ref = np.linalg.solve(Wd.T @ np.linalg.solve(Cov, Wd), Wd.T @ np.linalg.solve(Cov, y))
    print(f'  sample means with S, V and W vs dense GLS: {np.abs(mu - mu_ref).max() / np.abs(mu_ref).max():.2e}')


if __name__ == '__main__':
    _selftest()
