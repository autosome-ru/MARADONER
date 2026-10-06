import numpy as np
import jax.numpy as jnp
import jax
import scipy.linalg.lapack as lapack
from dataclasses import dataclass
from .utils import InputError, warn


def _list_names(names, limit=5) -> str:
    names = [str(n) for n in names]
    shown = ', '.join(names[:limit])
    if len(names) > limit:
        shown += f' ... and {len(names) - limit} more'
    return shown


def dependent_rows(X: np.ndarray, rel_tol=1e-7) -> list:
    """Indices of the rows of X that are (numerically: up to rel_tol of their norm)
    linear combinations of the rows above them."""
    X = np.asarray(X, dtype=np.float64)
    basis = np.zeros((0, X.shape[1]))
    dependent = list()
    for i, row in enumerate(X):
        norm = np.linalg.norm(row)
        resid = row - basis.T @ (basis @ row)
        resid = resid - basis.T @ (basis @ resid)
        resid_norm = np.linalg.norm(resid)
        if norm == 0 or resid_norm <= rel_tol * norm:
            dependent.append(i)
        else:
            basis = np.vstack([basis, resid / resid_norm])
    return dependent


def check_design_rank(X: np.ndarray, names=None):
    """Raise an InputError if the rows of the covariate design X (c x s, intercept first)
    are linearly dependent: the projection that removes the covariates is undefined then."""
    X = np.asarray(X, dtype=np.float64)
    if not np.all(np.isfinite(X)):
        raise InputError('The covariate design contains non-finite values (NaN or infinity); '
                         'fix the covariates table and re-run "maradoner create".')
    dependent = dependent_rows(X)
    if not dependent:
        return
    if names is not None and len(names) == len(X):
        shown = _list_names([names[i] for i in dependent])
    else:
        # Row 0 is the intercept, so row i is the i-th covariate.
        shown = _list_names([f'#{i}' for i in dependent])
    raise InputError(f'The covariates are linearly dependent: covariate(s) {shown} can be '
                     'written as a combination of the intercept and the preceding covariates '
                     f'(the design has rank {len(X) - len(dependent)} for {len(X)} rows, '
                     'intercept included), so their effects cannot be told apart. Remove '
                     'constant, duplicated or redundant covariates (e.g. keep k - 1 indicator '
                     'columns for a factor with k levels) and re-run "maradoner create".')


def null_space_transform(Q: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Compute V^T Y where V is the orthogonal complement to Q, using Householder 
    transformations via LAPACK's dormqr. 
    
    Parameters:
    Q (ndarray): p x r semi-orthogonal matrix where Q^T Q = I_r, r <= p. 
    Y (ndarray): p x n matrix. 
    
    Returns:
    VT_Y (ndarray): (p - r) x n matrix representing V^T Y (float64).
    """

    Y = np.array(Y, order='F', copy=True)

    p, r = Q.shape

    if r > p:
        raise ValueError(f"Number of columns r ({r}) cannot exceed number of rows p ({p}) in Q.")
        

    Q_copy = np.array(Q, order='F', dtype=np.float64)
    qr_a, tau, work_qr, info_qr = lapack.dgeqrf(Q_copy, overwrite_a=True)
    if info_qr != 0:
        raise RuntimeError(f"LAPACK dgeqrf failed with info = {info_qr}")
    

    lwork = -1
    # Use Z's shape here for the query, pass dummy Z
    _, work_query, _ = lapack.dormqr('L', 'T', qr_a, tau, np.empty_like(Y), lwork=lwork, overwrite_c=True)
    optimal_lwork = int(work_query[0].real)
    lwork = max(1, optimal_lwork)


    q_mult_y, work_actual, info_ormqr = lapack.dormqr('L', 'T', qr_a, tau, Y, 
                                                      lwork=lwork, overwrite_c=True)
    
    if info_ormqr != 0:
        raise RuntimeError(f"LAPACK dormqr failed with info = {info_ormqr} "
                           f"(Q shape: {Q.shape}, dtype: {Q.dtype}; "
                           f"Y shape: {Y.shape}, dtype: {Y.dtype}, "
                           f"order: {'F' if Y.flags.f_contiguous else 'C'}; lwork: {lwork})")

    VT_Y = q_mult_y[r:, :] 
    return VT_Y


@dataclass(frozen=True)
class LowrankDecomposition:
    Q: np.ndarray
    S: np.ndarray
    V: np.ndarray

    def null_space_transform(self, Y: np.ndarray) -> np.ndarray:
        """
        Compute V^T Y where V is the orthogonal complement to Q, using Householder 
        transformations via LAPACK's dormqr. 
        
        Parameters:
        Q (ndarray): p x r semi-orthogonal matrix where Q^T Q = I_r, r <= p. 
        Y (ndarray): p x n matrix. 
        
        Returns:
        VT_Y (ndarray): (p - r) x n matrix representing V^T Y (float64).
        """
        Y = np.array(Y, order='F', copy=True)
        Q = np.array(self.Q).astype(np.float64, copy=False)

        p, r = Q.shape

        if r > p:
            raise ValueError(f"Number of columns r ({r}) cannot exceed number of rows p ({p}) in Q.")
       
        Q_copy = np.array(Q, order='F', dtype=np.float64) 
        qr_a, tau, work_qr, info_qr = lapack.dgeqrf(Q_copy, overwrite_a=True)
        if info_qr != 0:
            raise RuntimeError(f"LAPACK dgeqrf failed with info = {info_qr}")
        # qr_a now contains R in upper triangle and reflectors below diagonal (overwritten Q_copy)
        

        lwork = -1
        _, work_query, _ = lapack.dormqr('L', 'T', qr_a, tau, np.empty_like(Y), lwork=lwork, overwrite_c=True)
        optimal_lwork = int(work_query[0].real)
        lwork = max(1, optimal_lwork)


        # Actual application
        q_mult_y, work_actual, info_ormqr = lapack.dormqr('L', 'T', qr_a, tau, Y, 
                                                          lwork=lwork, overwrite_c=True)
        
        if info_ormqr != 0:
            raise RuntimeError(f"LAPACK dormqr failed with info = {info_ormqr} "
                               f"(Q shape: {Q.shape}, dtype: {Q.dtype}; "
                               f"Y shape: {Y.shape}, dtype: {Y.dtype}, "
                               f"order: {'F' if Y.flags.f_contiguous else 'C'}; lwork: {lwork})")

        VT_Y = q_mult_y[r:, :] 
        return VT_Y
    
    
    def adjoint_null_space_transform(self, Z: np.ndarray) -> np.ndarray:
        """
        Computes V @ Z, the inverse of the null_space_transform.
        V is the orthogonal complement to Q. Z is typically the output of
        null_space_transform (i.e., Z = V^T Y).
        
        Parameters:
        Q (ndarray): The p x r semi-orthogonal matrix from the class instance.
        Z (ndarray): (p - r) x n matrix. 
        
        Returns:
        V_Z (ndarray): p x n matrix representing V @ Z (float64).
        """
        Q = np.array(self.Q).astype(np.float64, copy=False)
        p, r = Q.shape
        
        if Z.ndim != 2:
            raise ValueError("Input Z must be a 2D array.")
            
        if Z.shape[0] != p - r:
            raise ValueError(f"Input Z must have p-r = {p-r} rows, but got {Z.shape[0]}.")
        
        n = Z.shape[1]
        
        Z = np.asarray(Z, dtype=np.float64)
    

        Q_copy = np.array(Q, order='F', dtype=np.float64)
        qr_a, tau, work_qr, info_qr = lapack.dgeqrf(Q_copy, overwrite_a=True)
        if info_qr != 0:
            raise RuntimeError(f"LAPACK dgeqrf failed with info = {info_qr}")
            

        Y_padded = np.vstack([np.zeros((r, n)), Z])
        Y_padded = np.array(Y_padded, order='F', copy=True)
    

        lwork = -1
        _, work_query, _ = lapack.dormqr('L', 'N', qr_a, tau, Y_padded, lwork=lwork, overwrite_c=True)
        optimal_lwork = int(work_query[0].real)
        lwork = max(1, optimal_lwork)
    
        V_Z, work_actual, info_ormqr = lapack.dormqr('L', 'N', qr_a, tau, Y_padded, 
                                                      lwork=lwork, overwrite_c=True)
        
        if info_ormqr != 0:
            raise RuntimeError(f"LAPACK dormqr failed with info = {info_ormqr}")
    
        return V_Z


def ones_nullspace(n: int):
    res = np.zeros((n - 1, n), dtype=float)
    for i in range(1, n):
        norm = (1 / i + 1) ** 0.5
        res[i - 1, :i] = -1 / i / norm
        res[i - 1, i] = 1 / norm
    return res

def ones_nullspace_transform(x):
    n, m = x.shape
    if n <= 1:
        return np.zeros((0, m), dtype=float)
    
    x_float = x.astype(float)
    cumsum = np.cumsum(x_float, axis=0)  # cumsum[k] = sum of rows 0 to k
    
    i = np.arange(1, n)  # i = 1 to n-1
    sqrt_term = np.sqrt(i * (i + 1.0))
    
    coeff1 = -1.0 / sqrt_term
    coeff2 = np.sqrt(i / (i + 1.0))
    
    # Broadcast coefficients over columns
    Y = coeff1[:, None] * cumsum[:-1] + coeff2[:, None] * x_float[1:]
    
    return Y

def ones_nullspace_transform_transpose(X: np.ndarray) -> np.ndarray:
    p, m = X.shape
    total_n = p + 1
    
    if p == 0:
        return np.zeros((1, m), dtype=float)
    
    X_float = X.astype(float)
    
    i = np.arange(1, p + 1, dtype=float)
    sqrt_term = np.sqrt(i * (i + 1.0))
    
    coeff_pos = i / sqrt_term
    coeff_neg = -1.0 / sqrt_term
    
    weighted = coeff_neg[:, None] * X_float
    
    prefix_cum = np.cumsum(weighted, axis=0)
    total = prefix_cum[-1]
    
    suffix_before = total[None, :] - prefix_cum
    
    Y_body = coeff_pos[:, None] * X_float + suffix_before
    
    Y = np.zeros((total_n, m), dtype=float)
    Y[0] = total
    Y[1:] = Y_body
    
    return Y

def lowrank_decomposition(X: np.ndarray, rel_eps=1e-15, names=None,
                          verbose=True) -> LowrankDecomposition:
    """Thin SVD of X restricted to its numerical rank. Singular values not exceeding
    max(rel_eps, max(X.shape) * machine epsilon) times the largest one are rounding noise
    and are dropped; a warning then tells how many columns of X are linear combinations
    of the others (and which, by `names` if given)."""
    svd = jnp.linalg.svd
    q, s, v = [np.array(t) for t in svd(X, full_matrices=False)]
    if rel_eps is not None:
        n = len(s)
        max_sv = max(s) if n else 0.0
        rel_eps = max(rel_eps, max(np.shape(X)) * np.finfo(s.dtype).eps)
        r = int(np.sum(s > rel_eps * max_sv)) if max_sv > 0 else 0
        n_dropped = min(np.shape(X)) - r
        if n_dropped > 0 and np.shape(X)[0] >= np.shape(X)[1]:
            # Columns with a weight in the dropped right singular vectors are the ones
            # involved in the linear dependencies.
            involved = np.where(np.abs(v[r:]).max(axis=0) > 1e-6)[0]
            if names is not None and len(names) == np.shape(X)[1]:
                shown = _list_names([names[i] for i in involved])
            else:
                shown = _list_names([f'#{i + 1}' for i in involved])
            warn(f'The matrix being decomposed has rank {r}, but {np.shape(X)[1]} columns: '
                 f'{n_dropped} direction(s) are numerically linear combinations of the others '
                 f'and were dropped. Columns involved: {shown}. Parameters of the motifs '
                 'involved (variances, means, activities) are not separately identifiable; '
                 'remove or merge duplicated/collinear motifs to resolve this.', verbose)
        s = s[:r]
        q = q[:, :r]
        v = v[:r]
    return LowrankDecomposition(q, s, v)