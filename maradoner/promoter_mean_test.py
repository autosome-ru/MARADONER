from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
import pandas as pd
from enum import Enum
import dill
import pygam
from .utils import read_init, openers
from .fit import FitResult, ActivitiesPrediction, transform_data, split_data

class FOVMeanMode(str, Enum):
    null = 'null'
    gls = 'gls'
    knn = 'knn'

def knn_predict(B: np.ndarray, Z: np.ndarray, mu_p: np.ndarray, test_inds: np.ndarray, 
                n_B: int = 8, n_Z: int = 8, n_neighbours: int = 64, covariate=None) -> np.ndarray:

    if n_B > 0:
        pca_b = PCA(n_components=n_B)
        B_reduced = pca_b.fit_transform(B)
    else:
        if n_B == -1:
            B_reduced = B
        else:
            B_reduced = None
    
    if n_Z > 0:
        pca_z = PCA(n_components=n_Z)
        Z_reduced = pca_z.fit_transform(Z)
        if B_reduced is None:
            comb = [Z_reduced, ]
        else:
            comb = [B_reduced, Z_reduced]
    else:
        if n_Z == -1:
            Z_reduced = Z
            if B_reduced is None:
                comb = [Z_reduced]
            else:
                comb = [B_reduced, Z_reduced]
        else:
            Z_reduced = None
            comb = [B_reduced, ]
    if covariate:
        comb.append(covariate)
    combined_features = np.hstack(comb)
    # combined_features = (combined_features - combined_features.mean(axis=0, keepdims=True)) / combined_features.std(axis=0, keepdims=True)
    p = combined_features.shape[0]
    all_indices = np.arange(p)
    train_inds = np.setdiff1d(all_indices, test_inds)
    
    X_train = combined_features[train_inds]
    y_train = mu_p
    X_test = combined_features[test_inds]
    
    reg = KNeighborsRegressor(n_neighbors=n_neighbours, weights='distance', )
    reg.fit(X_train, y_train)
    

    predictions = reg.predict(X_test)
    return predictions

def estimate_promoter_mean(project: str,
                           mean_mode: FOVMeanMode = FOVMeanMode.gls, knn_n=128, pca_b=64, pca_z=3,
                           covariate: str = None, num_splies=10):
    data = read_init(project)
    fmt = data.fmt
    prom_names = data.promoter_names
    with openers[fmt](f'{project}.fit.{fmt}', 'rb') as f:
        fit : FitResult = dill.load(f)
    with openers[fmt](f'{project}.predict.{fmt}', 'rb') as f:
        activities : ActivitiesPrediction = dill.load(f)
    B0 = transform_data(data, helmert=False).B
    data, data_test = split_data(data, fit.promoter_inds_to_drop)
    data = transform_data(data, helmert=False, )
    if data_test is not None:
        data_test = transform_data(data_test, helmert=False)
    drops = activities.filtered_motifs
    U = activities.U_raw
    U_m = fit.motif_mean.mean.reshape(-1, 1)
    mu_s = fit.sample_mean.mean.reshape(-1, 1)
    
    mu_p = fit.promoter_mean.mean.flatten()
    if covariate:
        ws = data.Y - mu_s.T - mu_p.reshape(-1, 1) - data.B @ U_m - np.delete(data.B, drops, axis=1) @ U
        ws = 1 / np.std(ws, axis=1)
        X = pd.read_csv(covariate, sep='\t', index_col=0)
        cols = X.columns
        X = X.loc[prom_names].values
        inds = np.arange(0, len(B0))
        inds_train = np.setdiff1d(inds, fit.promoter_inds_to_drop, assume_unique=True)
        X_train = X[inds_train]
        config = dict()
        for i, c in enumerate(cols):
            config[c] = pygam.s(i, n_splines=num_splies)
        # for i in range(len(cols) - 1):
        #     a, b = cols[i:i+2]
        #     config[f'{a}_{b}'] = pygam.te(pygam.s(i, n_splines=num_splies), pygam.s(i+1, n_splines=num_splies) )
        t = None
        for v in config.values():
            if t is None:
                t = v
            else:
                t += v
        gam = pygam.LinearGAM(t, max_iter=1000, tol=1e-5)
        gam.gridsearch(X_train, mu_p, progress=False)
        mu_p_d = gam.predict(X[fit.promoter_inds_to_drop]) 
        mu_p_d_train = gam.predict(X_train)
        print(1.0 - np.var(mu_p_d_train - mu_p) / np.var(mu_p))
    else:
        mu_p_d = 0
        mu_p_d_train = 0
    if mean_mode == mean_mode.gls:
        Y = data_test.Y - mu_s.T
        Y = Y - data_test.B @ U_m
        Y = Y - np.delete(data_test.B, drops, axis=1) @ U
        D = (1 / fit.error_variance.variance)[data_test.group_inds_inv].reshape(-1, 1)
        mu_p = Y @ D / (D.sum()) 
    elif mean_mode == mean_mode.knn:
        mu_p = mu_p - mu_p_d_train
        B = B0
        Z = B @ U_m + fit.sample_mean.mean.reshape(1, -1)
        B = np.delete(B, drops, axis=1)
        Z = Z + B @ U
        
        mu_p = knn_predict(B, Z, mu_p, fit.promoter_inds_to_drop,
                           n_neighbours=knn_n, n_Z=pca_z, n_B=pca_b) 
    elif mean_mode == mean_mode.null:
        mu_p = np.zeros_like(mu_p_d)
    mu_p = mu_p + mu_p_d
    with openers[fmt](f'{project}.promoter_mean.{fmt}', 'wb') as f:
        dill.dump(mu_p, f)