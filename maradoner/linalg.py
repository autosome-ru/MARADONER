import numpy as np
import jax.numpy as jnp
import jax
import scipy.linalg.lapack as lapack
from dataclasses import dataclass


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
        print("--- Debug Info Before dormqr Failure ---")
        print(f"Q shape: {Q.shape}, dtype: {Q.dtype}")
        print(f"qr_a shape: {qr_a.shape}, dtype: {qr_a.dtype}, order: {'F' if qr_a.flags.f_contiguous else 'C'}")
        print(f"tau shape: {tau.shape}, dtype: {tau.dtype}")
        print(f"Y shape: {Y.shape}, dtype: {Y.dtype}, order: {'F' if Y.flags.f_contiguous else 'C'}")
        print(f"lwork: {lwork}")
        print("--- End Debug Info ---")
        raise RuntimeError(f"LAPACK dormqr failed with info = {info_ormqr}")

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
            print("--- Debug Info Before dormqr Failure ---")
            print(f"Q shape: {Q.shape}, dtype: {Q.dtype}")
            print(f"qr_a shape: {qr_a.shape}, dtype: {qr_a.dtype}, order: {'F' if qr_a.flags.f_contiguous else 'C'}")
            print(f"tau shape: {tau.shape}, dtype: {tau.dtype}")
            print(f"Y shape: {Y.shape}, dtype: {Y.dtype}, order: {'F' if Y.flags.f_contiguous else 'C'}")
            print(f"lwork: {lwork}")
            print("--- End Debug Info ---")
            raise RuntimeError(f"LAPACK dormqr failed with info = {info_ormqr}")

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
        
        Z = np.array(Z, dtype=np.float64, copy=False)
    

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

def lowrank_decomposition(X: np.ndarray, rel_eps=1e-15) -> LowrankDecomposition:
    svd = jnp.linalg.svd
    q, s, v = [np.array(t) for t in svd(X, full_matrices=False)]
    if rel_eps is not None:
        max_sv = max(s)
        n = len(s)
        for r in range(n):
            if s[r] / max_sv < rel_eps:
                r -= 1
                break
        r += 1
        s = s[:r]
        q = q[:, :r]
        v = v[:r]
    return LowrankDecomposition(q, s, v)