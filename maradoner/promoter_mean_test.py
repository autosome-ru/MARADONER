from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
import pandas as pd
from enum import Enum
import dill
import os
import pygam
from .utils import read_init, openers
from .utils import InputError, warn, dump_stage
from .fit import FitResult, ActivitiesPrediction, transform_data, split_data, motif_mean_matrix
from .fit import align_fit_to_promoters
from .grn import load_fit, load_predict, held_out_inds, filtered_motifs

class FOVMeanMode(str, Enum):
    null = 'null'
    gls = 'gls'
    knn = 'knn'

class CovariateModel(str, Enum):
    gam = 'gam'
    ridge = 'ridge'
    boosting = 'boosting'
    ensemble = 'ensemble'


def learn_promoter_means(X_train: np.ndarray, y: np.ndarray, X_test: np.ndarray, model: str,
                         cross_fit: bool = False) -> tuple:
    """Predicts promoter means from promoter covariates (e.g. the sequence features of
    "maradoner sequence-features") by ridge regression, gradient boosting or the average of the
    two (ensemble), trained on the training promoters. Returns the predictions for the test
    promoters and for the training ones; with cross_fit the latter are out-of-fold (5 folds)."""
    from sklearn.linear_model import RidgeCV
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import KFold
    model = CovariateModel(model)

    def one(Xa, ya, Xb):
        preds = list()
        if model in (CovariateModel.ridge, CovariateModel.ensemble):
            m, sd = Xa.mean(axis=0), Xa.std(axis=0)
            sd[sd == 0] = 1.0
            r = RidgeCV(alphas=np.logspace(-1, 6, 15)).fit((Xa - m) / sd, ya)
            preds.append(r.predict((Xb - m) / sd))
        if model in (CovariateModel.boosting, CovariateModel.ensemble):
            g = HistGradientBoostingRegressor(max_iter=3000, learning_rate=0.03, min_samples_leaf=40,
                                              l2_regularization=1.0, early_stopping=True,
                                              n_iter_no_change=50, random_state=0).fit(Xa, ya)
            preds.append(g.predict(Xb))
        return np.mean(preds, axis=0)

    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    pred_test = one(X_train, y, X_test)
    if cross_fit:
        pred_train = np.zeros(len(y))
        for a, b in KFold(5, shuffle=True, random_state=0).split(X_train):
            pred_train[b] = one(X_train[a], y[a], X_train[b])
    else:
        pred_train = one(X_train, y, X_train)
    return pred_test, pred_train

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
    if covariate is not None:
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

def read_promoter_covariates(filename: str, promoter_names: list) -> pd.DataFrame:
    """Reads the table of promoter covariates (promoters in rows, numeric covariates in
    columns) and returns its rows for `promoter_names`, in that order."""
    if not os.path.isfile(filename):
        raise InputError(f'Covariate file "{filename}" not found.')
    expected = ('A tab-separated table with promoter names in the first column and one numeric '
                'column per covariate is expected.')
    try:
        X = pd.read_csv(filename, sep='\t', index_col=0)
    except Exception as e:
        raise InputError(f'Cannot read the covariate file "{filename}": {e}. {expected}')
    if not X.shape[1]:
        raise InputError(f'The covariate file "{filename}" has no covariate columns. {expected}')
    def first(items):
        items = [str(i) for i in items]
        return ', '.join(items[:5]) + (f', ... and {len(items) - 5} more' if len(items) > 5 else '')
    X.index = X.index.astype(str)
    duplicated = X.index[X.index.duplicated()].unique()
    if len(duplicated):
        raise InputError(f'{len(duplicated)} promoter(s) occur more than once in the covariate file '
                         f'"{filename}": {first(duplicated)}.')
    present = set(X.index)
    missing = [p for p in promoter_names if str(p) not in present]
    if missing:
        raise InputError(f'{len(missing)} of the {len(promoter_names)} promoters of the project are absent '
                         f'from the covariate file "{filename}": {first(missing)}. Every promoter of the '
                         'project needs a row.')
    X = X.loc[[str(p) for p in promoter_names]]
    bad = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    if not bad:
        bad = [c for c in X.columns if not np.isfinite(X[c].values.astype(float)).all()]
    if bad:
        raise InputError(f'Column(s) {first(bad)} of the covariate file "{filename}" contain missing or '
                         f'non-numeric values for promoters of the project. {expected}')
    return X

def estimate_promoter_mean(project: str,
                           mean_mode: FOVMeanMode = FOVMeanMode.null, knn_n=256, pca_b=64, pca_z=3,
                           covariate: str = None, num_splies=10, covariate_model: str = 'gam'):
    mean_mode = FOVMeanMode(mean_mode)
    covariate_model = CovariateModel(getattr(covariate_model, 'value', covariate_model))
    if mean_mode == mean_mode.knn:
        if knn_n < 1:
            raise InputError(f'--knn-n must be a positive number of neighbours, but {knn_n} was given.')
        if pca_b < -1 or pca_z < -1 or (pca_b == 0 and pca_z == 0):
            raise InputError('--pca-b and --pca-z are numbers of principal components (-1: use the matrix '
                             f'itself, 0: do not use it) and cannot both be 0, but {pca_b} and {pca_z} were '
                             'given.')
    data = read_init(project)
    fmt = data.fmt
    data, fit = load_fit(project, data)
    prom_names = data.promoter_names
    activities : ActivitiesPrediction = load_predict(project, data, fit)
    test_inds = held_out_inds(fit)
    if not test_inds:
        raise InputError(f'The project "{project}" was fitted without a testing set, so there are no '
                         'held-out promoters whose means could be estimated. Hold promoters out with '
                         f'"maradoner fit {project} --test-chromosomes ..." or '
                         '"--test-promoters-filename ..." first.')
    if covariate and mean_mode == mean_mode.gls:
        warn('"--covariate" has no effect with "--mean-mode gls": the GLS estimate of a promoter mean is '
             'computed from the expression of that promoter and already contains whatever the '
             'covariates explain.')
        covariate = None
    covariate = read_promoter_covariates(str(covariate), prom_names) if covariate else None
    # Per-promoter estimates of fits made by older versions span all the promoters:
    # restrict them to the training ones.
    fit = align_fit_to_promoters(fit, test_inds, len(prom_names))
    B0 = transform_data(data, helmert=False, loading_context=fit.loading_context,
                        drist=fit.drist, loading_multipliers=fit.loading_multipliers).B
    data, data_test = split_data(data, test_inds)
    data = transform_data(data, helmert=False, loading_context=fit.loading_context,
                          drist=fit.drist, loading_multipliers=fit.loading_multipliers)
    if data_test is not None:
        data_test = transform_data(data_test, helmert=False, loading_context=fit.loading_context,
                                   drist=fit.drist, loading_multipliers=fit.loading_multipliers)
    drops = filtered_motifs(activities)
    U = activities.U_raw
    # Per-sample mean motif effect B M X (covariate-aware).
    BMX = data.B @ motif_mean_matrix(fit.motif_mean, data.X)
    mu_s = fit.sample_mean.mean.reshape(-1, 1)
    # The covariate/knn branches below use the fitted promoter means as the *training*
    # targets (aligned with `data`/`X_train`).
    mu_p = fit.promoter_mean.mean.flatten()
    if covariate is not None:
        ws = data.Y - mu_s.T - mu_p.reshape(-1, 1) - BMX - np.delete(data.B, drops, axis=1) @ U
        ws = 1 / np.std(ws, axis=1)
        X = covariate
        cols = X.columns
        X = X.values.astype(float)
        inds = np.arange(0, len(B0))
        inds_train = np.setdiff1d(inds, test_inds, assume_unique=True)
        X_train = X[inds_train]
    if covariate is not None and covariate_model != CovariateModel.gam:
        # The knn branch below fits the residual of the training means: it needs out-of-fold predictions.
        mu_p_d, mu_p_d_train = learn_promoter_means(X_train, mu_p, X[test_inds], covariate_model,
                                                    cross_fit=mean_mode == mean_mode.knn)
    elif covariate is not None:
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
        mu_p_d = gam.predict(X[test_inds]) 
        mu_p_d_train = gam.predict(X_train)
        # print(1.0 - np.var(mu_p_d_train - mu_p) / np.var(mu_p))
    else:
        mu_p_d = 0
        mu_p_d_train = 0
    if mean_mode == mean_mode.gls:
        Y = data_test.Y - mu_s.T
        Y = Y - data_test.B @ motif_mean_matrix(fit.motif_mean, data_test.X)
        Y = Y - np.delete(data_test.B, drops, axis=1) @ U
        D = (1 / fit.error_variance.variance)[data_test.group_inds_inv].reshape(-1, 1)
        mu_p = Y @ D / (D.sum()) 
    elif mean_mode == mean_mode.knn:
        mu_p = mu_p - mu_p_d_train
        B = B0
        # Z: the effect of the motifs (their mean activities and the deviations from them)
        # and of the samples on every promoter, as predicted from the loadings alone.
        U_m = motif_mean_matrix(fit.motif_mean, data.X)
        Z = B @ U_m + fit.sample_mean.mean.reshape(1, -1)
        B = np.delete(B, drops, axis=1)
        Z = Z + B @ U
        
        limits = {'--knn-n': (knn_n, len(mu_p), 'training promoters'),
                  '--pca-b': (pca_b, min(B.shape), 'motifs kept by "predict" and promoters'),
                  '--pca-z': (pca_z, min(Z.shape), 'samples and promoters')}
        defaults = {'--knn-n': 256, '--pca-b': 64, '--pca-z': 3}
        for option, (value, limit, what) in limits.items():
            if value > limit and value != defaults[option]:
                warn(f'{option} {value} exceeds what the project allows (the number of {what}): '
                     f'{limit} is used instead.')
        knn_n, pca_b, pca_z = [min(value, limit) for value, limit, _ in limits.values()]
        mu_p = knn_predict(B, Z, mu_p, test_inds,
                           n_neighbours=knn_n, n_Z=pca_z, n_B=pca_b) 
    elif mean_mode == mean_mode.null:
        mu_p = np.zeros(len(test_inds))
    # One mean per test promoter, in the order of the test promoters in the project. The
    # GLS estimate is used as it is (see the note on "--covariate" above).
    mu_p = np.asarray(mu_p + mu_p_d, dtype=float).reshape(-1)
    dump_stage(mu_p, project, 'promoter_mean', fmt)
    return mu_p