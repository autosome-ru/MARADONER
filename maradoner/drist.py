import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.isotonic import IsotonicRegression, isotonic_regression
from sklearn.linear_model import LinearRegression
from joblib import Parallel, delayed

def _pava_kernel_serial(target, sort_idx, inv_sort_idx, weights):
    t_sorted = target[sort_idx]
    w_sorted = weights[sort_idx] if weights is not None else None
    f_new = isotonic_regression(t_sorted, sample_weight=w_sorted, increasing=True)
    return f_new[inv_sort_idx]

class DRIST(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, mode='col_hetero', fit_intercept=True, optimizer='jacobi',
                 init='rank', share_function=False, max_iter=200, tol=1e-4, 
                 learning_rate=0.25, verbose=False):
        """
        DRIST: Doubly Robust Isotonic Regression.
        
        Parameters
        ----------
        init : str or None, default=None
            Initialization method for the transformed feature matrix f(X).
            - None: Linear initialization (f(X) = X).
            - 'rank': Replaces values with their rank (0 to n-1).
            - 'svd': Reconstructs X using a Rank-1 SVD approximation (captures dominant global variance).
            - 'logesf': -log(Empirical Survival Function). Good for heavy-tailed data.
            - 'random': Standard normal random noise.
        """
        self.mode = mode
        self.fit_intercept = fit_intercept
        self.optimizer = optimizer
        self.init = init
        self.share_function = share_function
        self.max_iter = max_iter
        self.tol = tol
        self.learning_rate = learning_rate
        self.verbose = verbose
    
    def _initialize_f(self, X):
        """
        Generates the initial state of f(X) based on self.init strategy.
        """
        p, m = X.shape
        
        if self.init is None:
            return X.copy()
        
        elif self.init == 'random':
            return np.random.randn(p, m)
        
        elif self.init == 'rank':
            # Replace values with their rank (0..p-1)
            f_X = np.zeros_like(X)
            for j in range(m):
                # argsort of argsort gives the rank
                # We use 'ordinal' ranking (0 to p-1) for speed via pure numpy
                idx = np.argsort(X[:, j])
                inv = np.empty_like(idx)
                inv[idx] = np.arange(p)
                f_X[:, j] = inv
            return f_X
            
        elif self.init == 'logesf':
            # -log(1 - ECDF) approx -log(1 - rank/(n+1))
            f_X = np.zeros_like(X)
            for j in range(m):
                idx = np.argsort(X[:, j])
                inv = np.empty_like(idx)
                inv[idx] = np.arange(p)
                # Ranks 1..p
                ranks = inv + 1.0
                # Survival function: (p + 1 - rank) / (p + 1)
                # Negative Log: -log(p+1-rank) + const
                # We ignore constants/scaling as f_X is standardized later
                f_X[:, j] = -np.log(p + 1.0 - ranks)
            return f_X
            
        elif self.init == 'svd':
            U, S, Vt = np.linalg.svd(X, full_matrices=False)
            f_X = np.outer(U[:, 0] * S[0], Vt[0, :])
            return f_X
            
        else:
            raise ValueError(f"Unknown init method: {self.init}")

    def fit(self, X, y):
        X, y = check_X_y(X, y, multi_output=True, y_numeric=True)
        if y.ndim == 1: y = y.reshape(-1, 1)
        
        if self.share_function:
            return self._fit_jacobi(X, y)
            
        optimizer = self.optimizer
        if optimizer == 'auto':
            if y.shape[1] > 2:
                optimizer = 'jacobi'
            else:
                optimizer = 'cd'
                
        if optimizer == 'cd':
            return self._fit_cd(X, y)
        else:
            return self._fit_jacobi(X, y)
            
    def _fit_jacobi(self, X, y):
        X = np.asfortranarray(X)
        p, m = X.shape
        s = y.shape[1]
        
        # 1. Pre-sort
        if self.share_function:
            if self.verbose: print("Pre-sorting flattened matrix...")
            X_flat = X.ravel(order='F')
            self.sort_idx_ = np.argsort(X_flat)
            self.inv_sort_idx_ = np.empty_like(self.sort_idx_)
            self.inv_sort_idx_[self.sort_idx_] = np.arange(p * m)
        else:
            if self.verbose: print("Pre-sorting columns...")
            self.sort_idxs_ = np.zeros((p, m), dtype=np.int32, order='F')
            self.inv_sort_idxs_ = np.zeros((p, m), dtype=np.int32, order='F')
            for j in range(m):
                idx = np.argsort(X[:, j])
                self.sort_idxs_[:, j] = idx
                inv = np.empty_like(idx)
                inv[idx] = np.arange(p)
                self.inv_sort_idxs_[:, j] = inv

        # 2. Init F
        f_X = self._initialize_f(X)
        f_X = np.asfortranarray(f_X).astype(np.float32)
        
        # Standardize F
        if self.share_function:
            f_X = (f_X - np.mean(f_X)) / np.std(f_X)
        else:
            f_X = (f_X - f_X.mean(axis=0)) / f_X.std(axis=0)
        
        # OLS Init
        if self.fit_intercept:
            ols = LinearRegression(fit_intercept=True).fit(f_X, y)
            U = ols.coef_.T.astype(np.float32)
            beta0 = ols.intercept_.astype(np.float32)
        else:
            ols = LinearRegression(fit_intercept=False).fit(f_X, y)
            U = ols.coef_.T.astype(np.float32)
            beta0 = np.zeros(s, dtype=np.float32)
            
        d_diag = np.ones(p, dtype=np.float32)
        update_d = (self.mode == 'doubly_hetero')
        prev_loss = np.inf
        
        for it in range(self.max_iter):
            Y_eff = y - beta0
            
            # --- STEP 1: TARGET CALCULATION ---
            Z = Y_eff @ U.T
            
            if s < m:
                Pred_Z = (f_X @ U) @ U.T
                K_diag = np.sum(U**2, axis=1)
            else:
                K = U @ U.T
                Pred_Z = f_X @ K
                K_diag = np.diag(K).copy()
            
            K_diag[K_diag < 1e-12] = 1.0 
            Targets = (Z - Pred_Z) / K_diag[None, :] + f_X
            
            pava_w = (1.0/d_diag) if update_d else None
            
            # --- STEP 2: PAVA ---
            if self.share_function:
                # Global PAVA
                Targets_flat = Targets.ravel(order='F')
                if pava_w is not None:
                    w_flat = np.tile(pava_w, m)
                    w_sorted = w_flat[self.sort_idx_]
                else:
                    w_sorted = None
                
                t_sorted = Targets_flat[self.sort_idx_]
                f_flat_new = isotonic_regression(t_sorted, sample_weight=w_sorted, increasing=True)
                f_flat_restored = f_flat_new[self.inv_sort_idx_]
                F_pava = f_flat_restored.reshape((p, m), order='F')
                
            else:
                # Serial PAVA Loop
                F_pava = np.zeros_like(f_X)
                for j in range(m):
                    t_col = Targets[:, j]
                    idx = self.sort_idxs_[:, j]
                    inv = self.inv_sort_idxs_[:, j]
                    t_sorted = t_col[idx]
                    w_sorted = pava_w[idx] if pava_w is not None else None
                    f_iso = isotonic_regression(t_sorted, sample_weight=w_sorted, increasing=True)
                    F_pava[:, j] = f_iso[inv]

            # Damped Update
            f_X = (1.0 - self.learning_rate) * f_X + self.learning_rate * F_pava
            
            # --- STEP 3: UPDATE U ---
            if update_d:
                w_sqrt = 1.0 / np.sqrt(d_diag + 1e-8)
                X_w = f_X * w_sqrt[:, None]
                Y_w = Y_eff * w_sqrt[:, None]
            else:
                X_w = f_X
                Y_w = Y_eff
                
            XtX = X_w.T @ X_w
            XtX.flat[::m+1] += 1e-6
            XtY = X_w.T @ Y_w
            U = np.linalg.solve(XtX, XtY)

            # --- STEP 4: NORMALIZE ---
            if self.share_function:
                mu = np.mean(f_X)
                f_X -= mu
                std = np.std(f_X)
                if std < 1e-8: std = 1.0
                f_X /= std
                beta0 += mu * np.sum(U, axis=0)
                U *= std
            else:
                means = np.mean(f_X, axis=0)
                f_X -= means
                stds = np.std(f_X, axis=0)
                stds[stds < 1e-8] = 1.0
                f_X /= stds
                beta0 += means @ U
                U *= stds[:, None]
            
            if self.fit_intercept:
                Res = y - (f_X @ U)
                if update_d: beta0 = np.average(Res, axis=0, weights=pava_w)
                else: beta0 = np.mean(Res, axis=0)
            
            R_sq = (y - (f_X @ U + beta0))**2
            loss = np.sum(R_sq)
            if update_d:
                d_diag = np.mean(R_sq, axis=1)
                d_diag = np.clip(d_diag, 1e-6, 1e6)
                
            delta = (prev_loss - loss) / prev_loss if prev_loss != np.inf else 1.0
            if np.abs(delta) < self.tol: break
            if self.verbose and it % 10 == 0:
                print(f"Iter {it}: Loss={loss:.4e}")
            prev_loss = loss
            
        self.coef_ = U
        self.intercept_ = beta0
        self.fitted_ = True
        
        # --- FINAL TRANSFORMER FIT ---
        if update_d: pava_w = 1.0/d_diag
        else: pava_w = None
        
        Z = (y - beta0) @ U.T
        if s < m:
            Pred_Z = (f_X @ U) @ U.T
            K_diag = np.sum(U**2, axis=1)
        else:
            K = U @ U.T
            Pred_Z = f_X @ K
            K_diag = np.diag(K).copy()
            
        Targets = (Z - Pred_Z) / (K_diag[None, :] + 1e-12) + f_X
        
        if self.share_function:
            Targets_flat = Targets.ravel(order='F')
            X_flat = X.ravel(order='F')
            if pava_w is not None:
                w_flat = np.tile(pava_w, m)
            else:
                w_flat = None
            self.transformer_ = IsotonicRegression(increasing=True, out_of_bounds='clip')
            self.transformer_.fit(X_flat, Targets_flat, sample_weight=w_flat)
            x_trans = self.transformer_.transform(X_flat)
            self.f_mean_ = np.mean(x_trans)
            self.f_std_ = np.std(x_trans)
            if self.f_std_ < 1e-8: self.f_std_ = 1.0
        else:
            self.transformers_ = []
            self.f_means_ = np.zeros(m)
            self.f_stds_ = np.zeros(m)
            for j in range(m):
                iso = IsotonicRegression(increasing=True, out_of_bounds='clip')
                iso.fit(X[:, j], Targets[:, j], sample_weight=pava_w)
                self.transformers_.append(iso)
                x_trans = iso.transform(X[:, j])
                self.f_means_[j] = np.mean(x_trans)
                self.f_stds_[j] = np.std(x_trans) if np.std(x_trans) > 1e-8 else 1.0
                
        return self

    def _fit_cd(self, X, y):
        X = np.asfortranarray(X)
        p, m = X.shape; s = y.shape[1]
        
        # Pre-sort logic
        if self.share_function:
            if self.verbose: print("Pre-sorting flattened...")
            X_flat = X.ravel(order='F')
            self.sort_idx_ = np.argsort(X_flat)
            self.inv_sort_idx_ = np.empty_like(self.sort_idx_)
            self.inv_sort_idx_[self.sort_idx_] = np.arange(p * m)
        else:
            if self.verbose: print("Pre-sorting columns...")
            self.sort_idxs_ = np.zeros((p, m), dtype=np.int32, order='F')
            self.inv_sort_idxs_ = np.zeros((p, m), dtype=np.int32, order='F')
            for j in range(m):
                idx = np.argsort(X[:, j])
                self.sort_idxs_[:, j] = idx
                inv = np.empty_like(idx); inv[idx] = np.arange(p); self.inv_sort_idxs_[:, j] = inv

        # Init
        f_X = self._initialize_f(X)
        f_X = np.asfortranarray(f_X).astype(np.float32)
        if self.share_function: f_X = (f_X - np.mean(f_X)) / np.std(f_X)
        else: f_X = (f_X - np.mean(f_X, axis=0)) / np.std(f_X, axis=0)
        
        ols = LinearRegression(fit_intercept=self.fit_intercept).fit(f_X, y)
        U = ols.coef_.T.astype(np.float32)
        if s == 1: U = U.reshape(m, 1)
        beta0 = ols.intercept_.astype(np.float32) if self.fit_intercept else np.zeros(s, dtype=np.float32)
        
        Y_pred = f_X @ U + beta0
        d_diag = np.ones(p, dtype=np.float32)
        update_d = (self.mode == 'doubly_hetero')
        prev_loss = np.inf
        
        for it in range(self.max_iter):
            pava_w = (1.0/d_diag) if update_d else None
            
            if self.share_function:
                # --- SHARED CD LOGIC (Block CD: Update U col-by-col, then Global F) ---
                # 1. Update U (Iterative Coordinate Descent on U)
                for j in range(m):
                    u_j = U[j:j+1, :] 
                    f_j = f_X[:, j:j+1]
                    R_no_j = (y - Y_pred) + f_j @ u_j
                    
                    # Update u_j to fit R_no_j given current f_j
                    # u_j = (f_j' W R) / (f_j' W f_j)
                    # Simplified weights handling:
                    num = f_j.T @ R_no_j
                    den = np.sum(f_j**2) + 1e-12
                    u_j_new = num / den
                    
                    U[j, :] = u_j_new
                    Y_pred = y - (R_no_j - f_j @ u_j_new)
                
                # 2. Update F (Global PAVA on averaged residuals)
                # We want f_shared(B_ij) * U_j \approx Res_ij
                # This is equivalent to f_shared \approx Res_ij / U_j ??
                # Better: Use the Jacobi-style target construction for the global F
                Z = (y - beta0) @ U.T
                K_diag = np.sum(U**2, axis=1)
                Pred_Z = (f_X @ U) @ U.T if s < m else f_X @ (U @ U.T)
                
                K_diag[K_diag < 1e-12] = 1.0
                Targets = (Z - Pred_Z) / K_diag[None, :] + f_X
                
                Targets_flat = Targets.ravel(order='F')
                w_flat = np.tile(pava_w, m)[self.sort_idx_] if pava_w is not None else None
                t_sorted = Targets_flat[self.sort_idx_]
                
                f_flat = isotonic_regression(t_sorted, sample_weight=w_flat, increasing=True)
                F_pava = f_flat[self.inv_sort_idx_].reshape((p, m), order='F')
                
                f_X = (1.0 - self.learning_rate) * f_X + self.learning_rate * F_pava
                
                # Recalculate Y_pred fully after F update
                Y_pred = f_X @ U + beta0
                
            else:
                # --- STANDARD CD LOGIC ---
                for j in range(m):
                    u_j = U[j:j+1, :]
                    f_j = f_X[:, j:j+1]
                    R_no_j = (y - Y_pred) + f_j @ u_j
                    
                    numer = R_no_j @ u_j.T
                    denom = np.sum(u_j**2) + 1e-12
                    target_j = numer.ravel() / denom
                    
                    idx = self.sort_idxs_[:, j]
                    w_sorted = pava_w[idx] if pava_w is not None else None
                    f_new = isotonic_regression(target_j[idx], sample_weight=w_sorted, increasing=True)[self.inv_sort_idxs_[:, j]]
                    
                    mu = np.mean(f_new); f_new -= mu; std = np.std(f_new)
                    if std > 1e-8: f_new /= std
                    else: f_new[:] = 0.0
                    
                    num_u = f_new @ R_no_j
                    u_j_new = (num_u / p).ravel()
                    
                    f_X[:, j] = f_new; U[j, :] = u_j_new
                    Y_pred = y - (R_no_j - np.outer(f_new, u_j_new))
                
            # Intercept & Variance & Stopping
            if self.fit_intercept:
                Res = y - (f_X @ U)
                if update_d: beta0 = np.average(Res, axis=0, weights=pava_w)
                else: beta0 = np.mean(Res, axis=0)
                Y_pred = f_X @ U + beta0 # Update Y_pred with new beta
            
            loss = np.sum((y - Y_pred)**2)
            if update_d:
                d_diag = np.clip(np.mean((y - Y_pred)**2, axis=1), 1e-6, 1e6)
            
            delta = (prev_loss - loss) / prev_loss if prev_loss != np.inf else 1.0
            if np.abs(delta) < self.tol: break
            if self.verbose and it % 10 == 0: print(f"Iter {it}: Loss={loss:.4e}")
            prev_loss = loss
            
        self.coef_ = U; self.intercept_ = beta0; self.fitted_ = True
        
        if self.share_function:
            self._save_transformers_jacobi(X, y, f_X, U, beta0, pava_w)
        else:
            self._save_transformers_cd(X, y, f_X, U, beta0, pava_w, Y_pred)
            
        return self

    def transform(self, X):
        check_is_fitted(self, 'fitted_')
        X = check_array(X)
        X_trans = np.zeros_like(X, dtype=np.float32)
        
        if self.share_function:
            X_flat = X.ravel(order='F') if X.flags['F_CONTIGUOUS'] else X.flatten()
            f_flat = self.transformer_.transform(X_flat)
            X_trans = f_flat.reshape(X.shape)
            if not X.flags['F_CONTIGUOUS']: X_trans = np.asfortranarray(X_trans)
            X_trans = (X_trans - self.f_mean_) / self.f_std_
        else:
            for j in range(X.shape[1]):
                col = self.transformers_[j].transform(X[:, j])
                X_trans[:, j] = (col - self.f_means_[j]) / self.f_stds_[j]
        return X_trans

    def predict(self, X):
        f_X = self.transform(X)
        return f_X @ self.coef_ + self.intercept_