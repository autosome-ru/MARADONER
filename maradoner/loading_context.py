import numpy as np
from scipy.sparse.linalg import minres, LinearOperator, cg
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, NMF
import jax.numpy as jnp
import jax
from .linalg import lowrank_decomposition, ones_nullspace_transform, ones_nullspace_transform_transpose
from dataclasses import dataclass
from sklearn.neighbors import KNeighborsClassifier
from sklearn.mixture import GaussianMixture

@dataclass(frozen=True)
class ClusterResult:
    L: np.ndarray
    scaler: StandardScaler
    clusterer: KMeans = None
    pca: PCA = None
    K_beta = None


def cluster_data(X: np.ndarray,  num_clusters: int, cluster: ClusterResult=None,
                 pca_feats: int = -1, X2=None, random_state=42) -> ClusterResult:
    if cluster is None:
        pca = None
        scaler = None
        clusterer = None
    else:
        pca = cluster.pca
        scaler = cluster.scaler
        clusterer = cluster.clusterer
    if scaler is None:
        scaler = StandardScaler().fit(X)
    X = scaler.transform(X)
    if pca_feats > 0 and pca is None:
        pca = PCA(pca_feats).fit(X)
    if pca is not None:
        X = pca.transform(X)
    if clusterer is None:
        if X2 is not None:
            X2 = StandardScaler().fit_transform(X2)
            # km = KMeans(n_clusters=num_clusters, random_state=random_state).fit(X2)
            km = GaussianMixture(n_components=num_clusters).fit(X2)
            labels = km.predict(X2)
            clusterer = KNeighborsClassifier(1).fit(X, labels)
            print((clusterer.predict(X) == labels).mean())
        else:
            clusterer = KMeans(n_clusters=num_clusters, random_state=random_state)
            clusterer = clusterer.fit(X)
    # L = clusterer.predict_proba(X)
    labels = clusterer.predict(X)
    L = np.zeros((X.shape[0], num_clusters))
    for i, v in enumerate(labels):
        L[i, v] = 1
    return ClusterResult(L=L, scaler=scaler, clusterer=clusterer, pca=pca)


@dataclass(frozen=True)
class LoadingContext:
    R: np.ndarray 
    cluster_result: ClusterResult
    beta: np.ndarray = None
    
    def apply_correction(self, B: np.ndarray, X: np.ndarray, return_L: bool = False,
                         L: np.ndarray=None):
        R = self.R
        if L is None:
            L = cluster_data(X, R.shape[0], cluster=self.cluster_result).L
        if self.beta is not None:
            L = np.clip(LoadingContext.get_L_features(L) @ self.beta, 0.0, float('inf')).reshape(-1,1) * L
        if return_L:
            return B * (L @ R), L
        return B * (L @ R)
    
    @staticmethod
    def get_L_features(L: np.ndarray):
        sums = L.sum(axis=1, keepdims=True)
        probs = L / sums
        sums = 1 / sums
        ones = np.ones_like(sums)
        return np.hstack((probs, sums, ones))

@dataclass(frozen=True)
class LoadingMultipliers:
    beta: np.ndarray
    nmf: NMF
    
    def apply_correction(self, B: np.ndarray):
        W = self.nmf.transform(B)
        X = LoadingMultipliers.get_features(B, W)
        mult = np.clip(X @ self.beta, 0.0, float('inf'))
        return mult.reshape(-1, 1) *B
    
    @staticmethod
    def get_features(B: np.ndarray, W):
        W = W / W.sum(axis=1, keepdims=True)
        sums = B.mean(axis=1, keepdims=True)
        invsums = 1/B.mean(axis=1, keepdims=True)
        # sums = 1 / sums
        ones = np.ones_like(sums)
        return np.hstack((B, W, invsums, sums, ones))


def solve_beta_memory_efficient(QT_Y, B, L, R, U, K, D_diag, q, batch_size=128, ):
    """
    Solves for beta using blocked computation to handle large 'm' and CG for the system.
    
    Memory Complexity: O(p * s + p * batch_size) instead of O(p * m).
    """
    QT_Y = np.array(QT_Y)
    B = np.array(B)
    L = np.array(L)
    R = np.array(R)
    U = np.array(U)
    q = np.array(q)
    D_diag = np.array(D_diag)
    p, m = B.shape
    s = U.shape[1]
    n = K.shape[1] # Size of beta
    
    # =========================================================================
    # 1. Compute M = (B * (L_inner R)) U using Chunks
    #    Target size: p x s (fits in memory)
    #    Intermediate size: p x batch (fits in memory)
    # =========================================================================
    M = np.zeros((p, s))
    for start_idx in range(0, m, batch_size):
        end_idx = min(start_idx + batch_size, m)
        
        # Slices
        B_chunk = B[:, start_idx:end_idx]       # p x batch
        R_chunk = R[:, start_idx:end_idx]       # k x batch
        U_chunk = U[start_idx:end_idx, :]       # batch x s
        # Compute Partial M
        # 1. Inner product: L_inner @ R_chunk -> p x batch
        LR_chunk = L @ R_chunk
        # 2. Hadamard: B_chunk * LR_chunk -> p x batch
        Z_chunk = B_chunk * LR_chunk
        # 3. Accumulate: Z_chunk @ U_chunk -> p x s
        M += Z_chunk @ U_chunk
    # =========================================================================
    # 2. Prepare Gradient (RHS)
    #    g = K^T [ (QT_Y * D^-1) . M ]_row_sums
    # =========================================================================
    inv_D = 1.0 / D_diag
    
    # Efficient row-sum dot product
    # A_scaled = QT_Y * inv_D
    # v[i] = sum_j (QT_Y[i,j] * inv_D[j] * M[i,j])
    v = np.sum((QT_Y * inv_D[np.newaxis, :]) * M, axis=1) # Shape (p,)
    g = K.T @ v  # Shape (n,)
    # =========================================================================
    # 3. Define Linear Operator for Hessian H
    #    H = K^T [ (I - qq^T) . (M D^-1 M^T) ] K
    #      = K^T [ diag(k_vec) ] K  -  K^T [ V V^T ] K
    # =========================================================================
    
    # Precompute M_tilde = M * D^-0.5
    sqrt_inv_D = np.sqrt(inv_D)
    M_tilde = M * sqrt_inv_D[np.newaxis, :]
    # 1. Diagonal Component k_vec
    # k_vec[i] = || row i of M_tilde ||^2
    k_vec = np.sum(M_tilde**2, axis=1) # Shape (p,)
    # 2. Low-Rank Component W
    # V = diag(q) M_tilde = q[:, newaxis] * M_tilde
    # W = K^T V  (Size: n x s) -- usually small enough to store
    V = q[:, np.newaxis] * M_tilde
    W = K.T @ V 
    
    # Define Matrix-Vector Product H(x)
    def hessian_matvec(x):
        # x is shape (n,)
        
        # Part 1: H1 x = K^T (k_vec * (K x))
        Kx = K @ x           # (p,)
        Kx_weighted = k_vec * Kx
        h1 = K.T @ Kx_weighted # (n,)
        
        # Part 2: H2 x = W (W^T x)
        WTx = W.T @ x        # (s,)
        h2 = W @ WTx         # (n,)
        return h1 - h2

    A_op = LinearOperator((n, n), matvec=hessian_matvec)
    # =========================================================================
    # 4. Solve
    # =========================================================================
    beta_hat, exit_code = cg(A_op, g,)
    
    if exit_code != 0:
        print(f"CG Warning: Exit code {exit_code}")
    return beta_hat

def solve_for_beta_general(QT_Y, B, L, R, U, K, D_diag, q):
    """
    Solves Y = Q diag(K beta) (B * (LR)) U + E for beta.
    
    Parameters:
    - QT_Y:   Q^T @ Y [p x s]
    - B:      [p x m]
    - L, R:   Matrices forming the Hadamard component [p x k], [k x m]
    - U:      [m x s]
    - K:      Mapping for beta [p x n]
    - D_diag: Noise variances [s]
    - q:      Projection vector [p]
    """
    # 1. Inner structure M = (B * (LR)) U
    Z = B * (L @ R)
    M = Z @ U
    
    # 2. Gradient g = K^T d(QT_Y D^-1 M^T)
    A_scaled = QT_Y / D_diag
    v = jnp.sum(A_scaled * M, axis=1) # [p]
    g = K.T @ v # [n]
    
    # 3. Hessian H = K^T [ (I - qq^T) * (M D^-1 M^T) ] K
    M_tilde = M / np.sqrt(D_diag)
    
    # H1 = K^T diag(row_norms(M_tilde)^2) K
    k_vec = np.sum(M_tilde**2, axis=1)
    H1 = (K.T * k_vec) @ K # O(p * n^2)
    
    # H2 = K^T (q q^T * M_tilde M_tilde^T) K = (K^T (q * M_tilde)) (K^T (q * M_tilde))^T
    V = q[:, np.newaxis] * M_tilde
    W = K.T @ V # [n x s]
    H2 = W @ W.T # [n x n]
    
    H = H1 - H2
    
    # 4. Solve n x n system
    return np.linalg.solve(H, g)

def solve_r_m(Y_proj: np.ndarray, B: np.ndarray,
            L: np.ndarray, U: np.ndarray, QC: np.ndarray, x0=None,
            maxiter=500):
    """
    Solves for R.
    
    Y_proj: (p, s) - Projected expression matrix (I - QC QC^T) Y
    B: (p, m)
    L: (p, r) such that L.T @ L = I_r
    U: (m, s)
    QC: (p, k) - Orthogonal complement to Q, k is assumed to be much smaller than p
    """
    # print(Y_proj.shape, B.shape, L.shape, U.shape, QC.shape, x0.shape)
    p, m = B.shape
    r = L.shape[1]
    if x0 is not None:
        x0 = x0.flatten('F')
    
    S_U = U @ U.T
    
    # RHS: L^T (B . (Y_proj U^T))
    RHS_mat = L.T @ (B * (Y_proj @ U.T))
    b = RHS_mat.flatten(order='F')
    
    def matvec(v):
        R = v.reshape((r, m), order='F')
        
        # T1 = (B . (LR)) S_U
        T1 = (B * (L @ R)) @ S_U # (p, m)
        
        # Projection P_Q = I - QC QC^T
        # T2 = (I - QC QC^T) T1
        T2 = T1 - QC @ (QC.T @ T1) # (p, m) 
        
        T3 = L.T @ (B * T2) # (r, m)
        
        return T3.flatten(order='F')
    
    op = LinearOperator((r*m, r*m), matvec=matvec)
    solver = minres
    # solver = cg
    r_hat_vec, info = solver(op, b, maxiter=maxiter, x0=x0, )
    R_hat = r_hat_vec.reshape((r, m), order='F')
    if info:
        print('Error in solver', info)
    return R_hat
    # return B * (L @ R_hat)

def solve_r(Y_proj: np.ndarray, B: np.ndarray,
            L: np.ndarray, U: np.ndarray, QC: np.ndarray, x0=None,
            maxiter=200):
    """
    Solves for R (optimized for s << m version).
   
    Y_proj: (p, s) - Projected expression matrix (I - QC QC^T) Y
    B: (p, m)
    L: (p, r) such that L.T @ L = I_r
    U: (m, s)
    QC: (p, k) - Orthogonal complement to Q, k << p
    """
    p, m = B.shape
    r = L.shape[1]
    if x0 is not None:
        x0 = x0.flatten('F')
   
    # RHS: L^T (B .* (Y_proj @ U.T))  — already efficient O(p s m)
    rhs_temp = Y_proj @ U.T  # (p, m)
    RHS_mat = L.T @ (B * rhs_temp)  # (r, m)
    b = RHS_mat.flatten(order='F')
   
    def matvec(v):
        R = v.reshape((r, m), order='F')
       
        # Compute (B * (L @ R)) @ (U @ U.T) without forming U @ U.T
        LR = L @ R                    # (p, m)
        temp = B * LR                  # (p, m) element-wise
        temp_U = temp @ U               # (p, s)  — low-rank step 1
        T1 = temp_U @ U.T               # (p, m)  — low-rank step 2
       
        # Projection P_Q = I - QC QC^T
        coeff = QC.T @ T1               # (k, m)
        T2 = T1 - QC @ coeff           # (p, m)
       
        T3 = L.T @ (B * T2)             # (r, m)
        T3 += 1e-6 * R 
        return T3.flatten(order='F')
   
    op = LinearOperator((r*m, r*m), matvec=matvec)
    r_hat_vec, info = minres(op, b, maxiter=maxiter, x0=x0)
    R_hat = r_hat_vec.reshape((r, m), order='F')
    if info:
        print('Error in solver', info)
    return R_hat

def solve_R_bfgs(Y_proj, B, L, U, QC, x0=None, bounded: bool = False):
    """
    Solves for Non-Negative R minimizing ||(I-QC QC^T) (Y - (B.(LR))U)||^2
    
    Y_proj: (p, s)  -> Pre-projected Y: (I - QC QC^T) Y
    B:      (p, m)
    L:      (p, r_dim)
    U:      (m, s)
    QC:     (p, k)  -> Orthogonal complement matrix
    """
    p, m = B.shape
    r_dim = L.shape[1]
    from scipy.optimize import minimize
    if len(U.shape) == 1:
        U_vec = True
    else:
        U_vec = False
    # 1. Precompute Constants
    if U_vec:
        S_U = jnp.outer(U, U)
    else:
        S_U = U @ U.T # (m, m)
    
    # Compute Linear Term c (The RHS)
    # c = L^T (B . (Y_proj U^T))
    # Y_proj is already projected, so we use it directly.
    if U_vec:
        RHS_mat = L.T @ (B * (jnp.outer(Y_proj.sum(axis=1), U.T)))
    else:
        RHS_mat = L.T @ (B * (Y_proj @ U.T))
    c = RHS_mat.flatten(order='F')
    
    # 2. Define the Linear Operator H(v)
    # This is exactly the same operator as in the CG method
    def apply_H(v):
        R = v.reshape((r_dim, m), order='F')
        
        # Chain rule operations
        # Z = (B . (LR)) S_U
  
        Z = (B * (L @ R)) @ S_U 
        
        # Project: (I - QC QC^T) Z
        # If QC is None or empty, skip this.
        if QC is not None and QC.shape[1] > 0:
            P_Z = Z - QC @ (QC.T @ Z)
        else:
            P_Z = Z
            
        # Back to R space
        res = L.T @ (B * P_Z)
        return res.flatten(order='F')

    # 3. Define Objective and Gradient for L-BFGS-B
    # We minimize f(x) = 0.5 * x.T H x - x.T c
    # Gradient g(x) = H x - c
    
    def func_and_grad(v):
        Hv = apply_H(v)
        
        # Quadratic Objective: 0.5 * v^T (Hv) - v^T c
        f = 0.5 * np.dot(v, Hv) - np.dot(v, c)
        
        # Gradient: Hv - c
        g = Hv - c
        
        return f, g

    # 4. Set Bounds
    # List of (min, max) for every element. (0, None) means R >= 0.
    bounds = [(0, None)] * (r_dim * m)
    
    # Initial Guess (Zero or Random positive)
    if x0 is not None:
        x0 = x0.flatten('F')
    else:
        x0 = np.random.rand(r_dim * m) * 0.1
    

    res = minimize(
        func_and_grad,
        x0,
        method='L-BFGS-B',
        jac=True,           
        bounds=bounds if bounded else None,
        options={
            'maxiter': 500,
            'ftol': 1e-5,
            'gtol': 1e-5,
            'disp': True    # Print convergence info
        }
    )
    print(res)
    
    # 6. Reconstruct
    R_hat = res.x.reshape((r_dim, m), order='F')
    return R_hat

def solve_for_R(QT_Y, B, L, U, D_diag, q):
    """
    Solves Y = Q (B * (L R)) U + E for R.
    
    Parameters:
    - QT_Y:   Q^T @ Y [p x s]
    - B:      [p x m]
    - L:      [p x k]
    - U:      [m x s]
    - D_diag: [s]
    - q:      [p]
    - k_dim:  integer k (rows of R)
    """
    p, m = B.shape
    s = U.shape[1]
    k_dim = L.shape[1]
    
    # 1. Precompute Weight Matrix Omega_U [m x m]
    # Omega_U = U D^-1 U^T
    U_scaled = U / D_diag
    Omega_U = U_scaled @ U.T
    
    # 2. Compute Gradient G [k x m]
    # G = L^T [ B * (QT_Y D^-1 U^T) ]
    # T1 = QT_Y D^-1 U^T [p x m]
    T1 = (QT_Y / D_diag) @ U.T
    G = L.T @ (B * T1)
    g_vec = G.flatten(order='F') # Column-major vectorization
    
    # 3. Compute Hessian H1 (Identity Part)
    # Construct Z [p x mk]. 
    # To do this efficiently, we repeat L m times and multiply by repeated B cols
    # B_repeated columns: 0,0..0, 1,1..1, ...
    B_rep = np.repeat(B, k_dim, axis=1) # [p x mk] 
    # L_tiled columns: 0,1..k, 0,1..k, ...
    L_tile = np.tile(L, (1, m)) # [p x mk]
    
    Z_big = B_rep * L_tile
    
    # Gram matrix [mk x mk]
    # This is the heavy lifting: O(p * (mk)^2)
    Gram = Z_big.T @ Z_big 
    
    # Mask [mk x mk] formed by expanding Omega_U
    # Block (i,j) of Mask is scalar Omega_U[i,j] * Ones(k,k)
    # Use Kronecker product
    Mask = np.kron(Omega_U, np.ones((k_dim, k_dim)))
    
    H1 = Gram * Mask
    
    # 4. Compute Hessian H2 (Projection Part)
    # V [k x m], col i is L^T (B_i * q)
    # Can be computed as L^T (B * q_broadcast)
    V = L.T @ (B * q[:, np.newaxis])
    v_vec = V.flatten(order='F')
    
    # H2 is outer product of v_vec masked by Omega_U
    H2 = np.outer(v_vec, v_vec) * Mask
    
    # 5. Solve
    H = H1 - H2
    r_vec = np.linalg.solve(H, g_vec)
    
    # Reshape back to R [k x m]
    R_hat = r_vec.reshape((k_dim, m), order='F')
    
    return R_hat


def solve_nonnegative_R_FISTA(Y_proj, B, L, U, QC, x0=None, maxiter=200):
    p, m = B.shape
    r_dim = L.shape[1]
    S_U = U @ U.T
    
    # RHS
    RHS_mat = L.T @ (B * (Y_proj @ U.T))
    c = RHS_mat.ravel(order='F')
    
    def apply_H(v):
        R = v.reshape((r_dim, m), order='F')
        Z = (B * (L @ R)) @ S_U
        if QC is not None: Z -= QC @ (QC.T @ Z)
        return (L.T @ (B * Z)).ravel(order='F')

    # Estimate Step Size (Lipschitz Constant) via Power Method
    print("Estimating step size...")
    v = np.random.randn(r_dim * m)
    v /= np.linalg.norm(v)
    for _ in range(10):
        v_next = apply_H(v)
        L_const = np.linalg.norm(v_next)
        v = v_next / L_const
    step_size = 0.95 / L_const

    # FISTA Loop
    if x0 is None:
        x = np.zeros(r_dim * m)
        y = np.zeros(r_dim * m)
    else:
        x0 = x0.flatten('F')        
        x = x0.copy()
        y = x0.copy()
    t = 1.0
    
    print("Running FISTA...")
    for k in range(maxiter):
        # Gradient of Quadratic: Hx - c
        grad = apply_H(y) - c
        
        # Projected Gradient Step (ReLU)
        x_new = np.maximum(0, y - step_size * grad)
        
        # Momentum Update
        t_new = (1 + np.sqrt(1 + 4*t**2)) / 2
        y = x_new + ((t - 1) / t_new) * (x_new - x)
        
        # Check convergence
        if np.linalg.norm(x_new - x) / (np.linalg.norm(x) + 1e-9) < 1e-5:
            print(f"Converged at iteration {k}")
            break
            
        x = x_new
        t = t_new
        
    return x.reshape((r_dim, m), order='F')


def solve_U_mu(B, Y, d):
    """
    Solves Y = B * mu * 1^T + B * U + E
    where E ~ MN(0, I, diag(d))
    
    Parameters:
    Q, R : Reduced QR decomposition of B (shape p x m, m x m)
    Y    : Observation matrix (shape p x s)
    d    : Vector of length s (diagonal of the column variance matrix D)
    
    Returns:
    mu   : MLE of the common coefficient vector (shape m,)
    U    : MLE of the deviation matrix (shape m x s)
    """
    # 1. Solve for X (where X = mu * 1^T + U)
    # B*X = Y -> R*X = Q.T @ Y
    # This is the same as the unweighted case because d_j is a 
    # scalar multiplier for each independent column.
    X = jnp.linalg.pinv(B) @ Y
    
    # 2. Calculate weights (precisions) from the variance vector d
    # Higher variance means lower weight
    weights = 1.0 / d
    
    # 3. Solve for mu (Weighted Mean)
    # mu = sum(w_j * x_j) / sum(w_j)
    # In matrix form: (X @ weights) / sum(weights)
    mu = (X @ weights) / jnp.sum(weights)
    
    # 4. Solve for U (Residuals)
    # U = X - mu * 1^T
    U = X - mu[:, jnp.newaxis]
    
    return U, mu


def estimate_context(Y: np.ndarray, B: np.ndarray, r: int, rtol=5e-3, maxiter=60,
                     estimate_promoter_mean: bool = False, L: np.ndarray = None,
                     non_negative: bool = True, estimate_multipliers: bool = True,
                     compute_fovs: bool = True, gpu: bool = False) -> LoadingContext:
    def mom_variance(Y_proj) -> np.ndarray:
        # Rough estimate, but I guess it will do
        Y = Y_proj
        _, s = Y.shape
        H = Y - Y.mean(axis=0, keepdims=True) - Y.mean(axis=1, keepdims=True) + Y.mean()
        H = (H ** 2).mean(axis=0)
        return s / (s - 2) * H - H.mean() / (s-2)
    
    def estimate_mu_p(Y: jnp.ndarray, D: jnp.ndarray, Q_C: jnp.ndarray):
        # B annd Y are assumed to be left-Helemrt transformed
        w = (1 / D).sum()
        mean = Y @ (1 / D.reshape(-1, 1))
        mean = mean - Q_C @ (Q_C.T @ mean)
        mean = ones_nullspace_transform_transpose(mean)
        mean = mean / w
        return mean
    switched_method = False
    rchange = float('inf')
    change = rchange
    n_iter = 0
    if L is None:
        cluster_result = cluster_data(B, num_clusters=r, X2=Y, pca_feats=16)
        L = cluster_result.L
    else:
        cluster_result = None
    if gpu:
        device = jax.devices()
    else:
        device = jax.devices('cpu')
    device = next(iter(device))
    from time import time
    if estimate_promoter_mean:
        Y_center = ones_nullspace_transform(Y)
    else:
        Y_center = Y - Y.mean(axis=0, keepdims=True) - Y.mean(axis=1, keepdims=True) + Y.mean()
    # alphas = (0.1, 1.0)
    K = LoadingContext.get_L_features(L)
    with jax.default_device(device):
        B0 = jnp.array(B)
        B = jnp.array(B)
        ones_norm = jnp.ones((len(Y), 1))
        ones_norm = ones_norm / jnp.linalg.norm(ones_norm)
        R = jnp.ones((r, B0.shape[1]))
        R_norm_col = jnp.linalg.norm(R[0])
        if compute_fovs:
            if not estimate_promoter_mean:
                Bt = (B - B.mean(axis=0, keepdims=True))
                U = jnp.linalg.pinv(Bt) @ Y_center
                # ridge = RidgeCV(fit_intercept=False, 
                #                 alphas=alphas).fit(Bt, Y_center)
                # U = ridge.coef_.T
                # print(Bt.shape, Y_center.shape,)
                prev_fov = 1.0 - ((Y_center - Bt @ U) ** 2).mean() / (Y_center ** 2).mean()
                # print(ridge.score(Bt, Y_center))
                # print(np.corrcoef(Y_center.flatten(), (Bt @ U).flatten())[0, 1])
                # print(prev_fov)
            else:
                B_1 = jnp.hstack((B, ones_norm))
                Q = jnp.linalg.qr(B_1, mode='reduced')[0]
                Y_proj = Y - Q @ (Q.T @ Y)
                sample_variance = mom_variance(Y_proj)
                H_B =  ones_nullspace_transform(B)
                d = sample_variance.reshape(-1, 1) ** (-0.5)
                U, mu = solve_U_mu(H_B, Y_center, sample_variance)
                U = U + mu.reshape(-1, 1)
                fov = 1.0 - ((Y_center - H_B @ U) ** 2).mean() / (Y_center ** 2).mean()
                # change = np.abs(fov - prev_fov) / fov 
                prev_fov = fov
        print(f'Iter 0. Starting FOV: {prev_fov:.3f}')
        try:
            while (n_iter < maxiter) and ((rchange > rtol) or not switched_method):
                if (rchange < rtol * 5 or change < 0.01) and not switched_method:
                    print('Changed to L-BFGS-B')
                    switched_method = not switched_method
                # Step U
                t = time()
                # Estimating sample variances
                B_1 = jnp.hstack((B, ones_norm))
                Q = jnp.linalg.qr(B_1, mode='reduced')[0]
                Y_proj = Y - Q @ (Q.T @ Y)
                sample_variance = mom_variance(Y_proj)
                time_var = time() - t
                
                if not estimate_promoter_mean:
                    t = time()
                    # Estimating U
                    d = (1 / sample_variance ** 0.5).reshape(-1,1)
                    Y_proj = Y * d.T
                    Y_proj = lowrank_decomposition(d).null_space_transform(Y_proj.T).T
                    Bp = jnp.linalg.pinv(ones_nullspace_transform(B))
                    U = Bp @ ones_nullspace_transform(Y_proj)
                    # ridge = RidgeCV(fit_intercept=False, store_cv_results=True,
                    #                 alphas=alphas).fit( ones_nullspace_transform(B), ones_nullspace_transform(Y_proj))
                    # U = ridge.coef_.T
                    # print(ridge.alpha_,)
                    time_u = time() - t
                else:
                    # Estimating mu_p
                    t = time()
                    H_B = ones_nullspace_transform(B)
                    Q = jnp.linalg.qr(H_B, mode='reduced')[0]
                    mu_p = estimate_mu_p(Y_center, sample_variance, Q)
                    time_mu_p = time() - t
                    
                    # Estimating U, mu
                    t = time()
                    d = sample_variance.reshape(-1, 1) ** (-0.5)
                    U, mu = solve_U_mu(H_B, Y_center, sample_variance)
                    U = U + mu.reshape(-1, 1)
                    Y_proj = Y - mu_p.reshape(-1, 1)
                    Y_proj = Y_proj * d.T
                    U = U * d.T
                    time_u = time() - t
                    
                
                # Step R
                t = time()
                Y_proj = Y_proj - ones_norm @ (ones_norm.T @ Y_proj)
                
                if estimate_multipliers:
                    beta = solve_for_beta_general(Y_proj, B, L, R, U, K, np.ones(Y_proj.shape[1]).flatten(), ones_norm.flatten())
                    L_scaled = np.clip((K @ beta), 0, float('inf')).reshape(-1, 1) * L
                else:
                    L_scaled = L
                    beta = None
                
                R_prev = R
                if switched_method:
                    R = solve_R_bfgs(Y_proj=Y_proj, B=B0, L=L_scaled, U=U, QC=ones_norm, x0=R, bounded=non_negative)
                else:
                    # R = solve_for_R(Y_proj, B=B0, L=L, U=U, D_diag=np.ones_like(d)[:-1].flatten(),
                    #                 q=ones_norm)
                    R = solve_r(Y_proj=Y_proj, B=B0, L=L_scaled, U=U, QC=ones_norm, x0=R)
                    if non_negative:
                        R = jnp.clip(R, 0.0, None)
                # R = solve_nonnegative_R(Y_proj=Y_proj, B=B0, L=L, U=U, QC=ones_norm, x0=R)
                norms = jnp.linalg.norm(R, axis=0, keepdims=True)
                inds = norms < 1e-9
                norms = jnp.where(inds, 1, norms)
                R = R / norms * R_norm_col
                rchange = jnp.linalg.norm(R - R_prev) / jnp.linalg.norm(R)
                B = B0 * (L_scaled @ R)
                time_r = time() - t
        
                n_iter += 1
                
                # Measuring FOV
                if compute_fovs:
                    t = time()
                    if not estimate_promoter_mean:
                        Bt =  (B - B.mean(axis=0, keepdims=True))
                        U = jnp.linalg.pinv(Bt) @ Y_center
                        # ridge = RidgeCV(fit_intercept=False, store_cv_results=True,
                        #                 alphas=alphas).fit(Bt, Y_center)
                        # U = ridge.coef_.T
                    else:
                        prev_fov = fov
                        Bt =  ones_nullspace_transform(B)
                        U, mu = solve_U_mu(Bt, Y_center, sample_variance)
                        U = U + mu.reshape(-1, 1)
                    fov = 1.0 - ((Y_center - Bt @ U) ** 2).mean() / (Y_center ** 2).mean()
                    change = np.abs(fov - prev_fov) / fov 
                    prev_fov = fov
                    time_fov = time() - t
                    print(f'Iter {n_iter}. FOV: {fov:.3f} (rel. change {change*100:.2f}%), Rel chainge in R: {rchange*100:.3f}%. Time spent:')
                    if estimate_promoter_mean:
                        print(f'variance: {time_var:.2f} s.,  U: {time_u:.2f} s., mu_p: {time_mu_p:.2f} s., R: {time_r:.2f} s., FOV: {time_fov:.2f} s.')
                    else:
                        print(f'variance: {time_var:.2f} s.,  U: {time_u:.2f} s., R: {time_r:.2f} s., FOV: {time_fov:.2f} s.')
                else:
                    print(f'Iter {n_iter}. Rel chainge in R: {rchange*100:.3f}%. Time spent:')
                    if estimate_promoter_mean:
                        print(f'variance: {time_var:.2f} s.,  U: {time_u:.2f} s., mu_p: {time_mu_p:.2f} s., R: {time_r:.2f} s.')
                    else:
                        print(f'variance: {time_var:.2f} s.,  U: {time_u:.2f} s., R: {time_r:.2f} s.')
        except KeyboardInterrupt:
            pass
    return LoadingContext(R, cluster_result=cluster_result, beta=beta)

def estimate_multipliers(Y: np.ndarray, B: np.ndarray, maxiter=50, rtol=1e-3, gpu: bool = False) -> LoadingContext:
    def mom_variance(Y_proj) -> np.ndarray:
        # Rough estimate, but I guess it will do
        Y = Y_proj
        _, s = Y.shape
        H = Y - Y.mean(axis=0, keepdims=True) - Y.mean(axis=1, keepdims=True) + Y.mean()
        H = (H ** 2).mean(axis=0)
        return s / (s - 2) * H - H.mean() / (s-2)
    
    change = float('inf')
    n_iter = 0
    
    if gpu:
        device = jax.devices()
    else:
        device = jax.devices('cpu')
    device = next(iter(device))
    from time import time
   
    Y_center = Y - Y.mean(axis=0, keepdims=True) - Y.mean(axis=1, keepdims=True) + Y.mean()
    
    nmf = NMF(n_components=16, max_iter=1000).fit(B)
    W = nmf.transform(B)
    K = LoadingMultipliers.get_features(B, W)

    with jax.default_device(device):
        B = jnp.array(B)
        B0 = B.copy()
        ones_norm = jnp.ones((len(Y), 1))
        ones_norm = ones_norm / jnp.linalg.norm(ones_norm)
        L = jnp.ones((len(B), 1))
        R = jnp.ones((1, B.shape[1]))
        Bt = (B - B.mean(axis=0, keepdims=True))
        U = jnp.linalg.pinv(Bt) @ Y_center
        prev_fov = 1.0 - ((Y_center - Bt @ U) ** 2).mean() / (Y_center ** 2).mean()

            
        print(f'Iter 0. Starting FOV: {prev_fov:.3f}')
        try:
            while (n_iter < maxiter) and (change > rtol):
                # Step U
                t = time()
                # Estimating sample variances
                B_1 = jnp.hstack((B, ones_norm))
                Q = jnp.linalg.qr(B_1, mode='reduced')[0]
                Y_proj = Y - Q @ (Q.T @ Y)
                sample_variance = mom_variance(Y_proj)
                time_var = time() - t
                
            
                t = time()

                d = (1 / sample_variance ** 0.5).reshape(-1,1)
                Y_proj = Y * d.T
                Y_proj = lowrank_decomposition(d).null_space_transform(Y_proj.T).T
                Bp = jnp.linalg.pinv(ones_nullspace_transform(B))
                U = Bp @ ones_nullspace_transform(Y_proj)
  
                time_u = time() - t

                    
                
                # Step R
                t = time()
                Y_proj = Y_proj - ones_norm @ (ones_norm.T @ Y_proj)
                
              
                beta = solve_for_beta_general(Y_proj, B, L, R, U, K, np.ones(Y_proj.shape[1]).flatten(), ones_norm.flatten())
                # beta = beta / jnp.linalg.norm(beta)
                B = np.clip((K @ beta), 0, float('inf')).reshape(-1, 1) * B0
                time_r = time() - t
        
                n_iter += 1
                
               
                t = time()
                
                Bt =  (B - B.mean(axis=0, keepdims=True))
                U = jnp.linalg.pinv(Bt) @ Y_center

                fov = 1.0 - ((Y_center - Bt @ U) ** 2).mean() / (Y_center ** 2).mean()
                change = np.abs(fov - prev_fov) / fov 
                prev_fov = fov
                time_fov = time() - t
                print(f'Iter {n_iter}. FOV: {fov:.3f} (rel. change {change*100:.2f}%), Time spent:')
                print(f'variance: {time_var:.2f} s.,  U: {time_u:.2f} s., R: {time_r:.2f} s., FOV: {time_fov:.2f} s.')
        except KeyboardInterrupt:
            pass
    print(beta[-20:])
    return LoadingMultipliers(beta=beta, nmf=nmf)