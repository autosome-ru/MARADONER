import numpy as np
import pandas as pd
import scipy.linalg
import scipy.stats as st
from statsmodels.stats import multitest


def _class_names(class_names) -> list:
    names = [str(name) for name in class_names]
    if len(names) < 2:
        raise ValueError('At least two class names are needed.')
    if any(not name.strip() for name in names):
        raise ValueError('Class names must not be empty.')
    if len(set(names)) != len(names):
        raise ValueError('Class names must be unique.')
    return names


def sum_to_zero_matrix(class_names) -> np.ndarray:
    """K x K orthonormal matrix: a shared column followed by K - 1 sum-to-zero
    Helmert contrasts, so that beta = T @ component_coefficients."""
    names = _class_names(class_names)
    K = len(names)
    shared = np.full(K, 1.0 / np.sqrt(K))
    contrasts = scipy.linalg.helmert(K, full=False).T
    return np.column_stack([shared, contrasts])


def component_names(class_names) -> list:
    """Names of the shared component followed by the Helmert contrast components."""
    names = _class_names(class_names)
    K = len(names)
    components = ['shared']
    if K == 2:
        components.append(f'{names[0]}_minus_{names[1]}')
        return components
    for j in range(K - 1):
        positive = '+'.join(names[:j + 1])
        components.append(f'{positive}_vs_{names[j + 1]}')
    return components


def build_joint_design(loadings, feature_names, class_names) -> tuple:
    """Joint class design. Each class block has K * m columns, one block of m base
    features per component, with the component coefficients related to the class
    coefficients by T."""
    class_names = _class_names(class_names)
    K = len(class_names)
    if len(loadings) != K:
        raise ValueError(f'Expected {K} class loadings, got {len(loadings)}.')
    arrays = [np.asarray(loading, dtype=np.float64) for loading in loadings]
    if any(array.ndim != 2 for array in arrays):
        raise ValueError('Each class loading must be a 2-D array.')
    m = arrays[0].shape[1]
    if m < 1 or any(array.shape[1] != m for array in arrays):
        raise ValueError('All class loadings must have the same positive number of columns.')
    if any(not np.all(np.isfinite(array)) for array in arrays):
        raise ValueError('The class loadings contain non-finite values.')
    feature_names = [str(name) for name in feature_names]
    if len(feature_names) != m:
        raise ValueError(f'Expected {m} feature names, got {len(feature_names)}.')
    T = sum_to_zero_matrix(class_names)
    blocks = [np.hstack([array * T[a, c] for c in range(K)]) for a, array in enumerate(arrays)]
    B = np.vstack(blocks)
    components = component_names(class_names)
    names = [f'{feature_names[f]}__{components[c]}' for c in range(K) for f in range(m)]
    variance_ties = [None if c == 0 else feature_names[f] for c in range(K) for f in range(m)]
    return B, names, variance_ties, T


def fixed_posterior(M, covariance, T) -> tuple:
    """Class means (m, K) and covariances (m, K, K) from a fixed-effect posterior in
    component coordinates. A 2-D M contributes its intercept column."""
    M = np.asarray(M, dtype=np.float64)
    covariance = np.asarray(covariance, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    if M.ndim == 2:
        if M.shape[1] < 1:
            raise ValueError('M must have at least one column.')
        M = M[:, 0]
    if M.ndim != 1:
        raise ValueError('M must have shape (K * m,) or (K * m, c).')
    K = T.shape[0]
    if T.shape != (K, K):
        raise ValueError('T must be a square class matrix.')
    if M.shape[0] % K:
        raise ValueError('The length of M must be a multiple of the number of classes.')
    m = M.shape[0] // K
    if covariance.shape != (K * m, K * m):
        raise ValueError(f'covariance must have shape {(K * m, K * m)}.')
    means = (T @ M.reshape(K, m)).T
    indices = np.arange(K * m).reshape(K, m)
    covariances = np.empty((m, K, K))
    for f in range(m):
        sub = covariance[np.ix_(indices[:, f], indices[:, f])]
        covariances[f] = T @ sub @ T.T
    covariances = 0.5 * (covariances + np.swapaxes(covariances, -1, -2))
    return means, covariances


def group_posterior(U, Q, W, T) -> tuple:
    """Class activities (m, K, g) and covariances (m, K, K, g) from component means
    U and the group covariances Q diag(W[g]) Q.T, without dense K*m matrices."""
    U = np.asarray(U, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    K = T.shape[0]
    if T.shape != (K, K):
        raise ValueError('T must be a square class matrix.')
    if U.ndim != 2 or Q.ndim != 2 or W.ndim != 2:
        raise ValueError('U, Q and W must be 2-D arrays.')
    if U.shape[0] % K:
        raise ValueError('The number of rows of U must be a multiple of the number of classes.')
    m = U.shape[0] // K
    g = U.shape[1]
    if Q.shape[0] != K * m:
        raise ValueError(f'Q must have {K * m} rows.')
    if W.shape != (g, Q.shape[1]):
        raise ValueError(f'W must have shape {(g, Q.shape[1])}.')
    U3 = U.reshape(K, m, g)
    means = np.einsum('ac,cfg->afg', T, U3).transpose(1, 0, 2)
    Q3 = Q.reshape(K, m, Q.shape[1])
    A = np.einsum('ac,cfr->far', T, Q3)
    covariances = np.einsum('fas,gs,fbs->fgab', A, W, A)
    covariances = np.moveaxis(covariances, 1, -1)
    covariances = 0.5 * (covariances + np.swapaxes(covariances, 1, 2))
    return means, covariances


def _group_arrays(means, covariances, group_names) -> tuple:
    means = np.asarray(means, dtype=np.float64)
    covariances = np.asarray(covariances, dtype=np.float64)
    if means.ndim == 2:
        if covariances.ndim != 3:
            raise ValueError('covariances must have shape (m, K, K).')
        means = means[:, :, None]
        covariances = covariances[:, :, :, None]
        groups = [None] if group_names is None else [str(name) for name in group_names]
        if group_names is not None and len(groups) != 1:
            raise ValueError('One group name is needed for 2-D means.')
    elif means.ndim == 3:
        if covariances.ndim != 4:
            raise ValueError('covariances must have shape (m, K, K, g) for grouped means.')
        if group_names is None or len(group_names) != means.shape[2]:
            raise ValueError('group_names must have one entry per group.')
        groups = [str(name) for name in group_names]
    else:
        raise ValueError('means must have shape (m, K) or (m, K, g).')
    if covariances.shape[:3] != (means.shape[0], means.shape[1], means.shape[1]):
        raise ValueError('The shape of covariances does not match means.')
    return means, covariances, groups


def _z_and_pvalue(effect, variance) -> tuple:
    std = np.sqrt(np.clip(variance, 0.0, None))
    z = np.zeros_like(std)
    zero = std == 0
    np.divide(effect, std, out=z, where=~zero)
    tolerance = 1e-10 * max(float(np.max(np.abs(effect), initial=0.0)),
                            np.finfo(float).tiny)
    identified = zero & (np.abs(effect) > tolerance)
    z[identified & (effect > 0)] = np.inf
    z[identified & (effect < 0)] = -np.inf
    pvalue = 2.0 * st.norm.sf(np.abs(z))
    return std, z, pvalue


def pairwise_contrasts(means, covariances, class_names, group_names=None) -> pd.DataFrame:
    """Long table of all class_1 - class_2 contrasts per feature, with the full
    covariance propagated and Benjamini-Hochberg FDR within each class pair and group."""
    class_names = _class_names(class_names)
    K = len(class_names)
    means, covariances, groups = _group_arrays(means, covariances, group_names)
    if means.shape[1] != K:
        raise ValueError(f'Expected {K} classes, got {means.shape[1]}.')
    m = means.shape[0]
    grouped = group_names is not None
    rows, feature_index = list(), list()
    for gi, group in enumerate(groups):
        for i in range(K):
            for j in range(i + 1, K):
                effect = means[:, i, gi] - means[:, j, gi]
                variance = (covariances[:, i, i, gi] + covariances[:, j, j, gi]
                            - 2.0 * covariances[:, i, j, gi])
                std, z, pvalue = _z_and_pvalue(effect, variance)
                fdr = multitest.multipletests(pvalue, method='fdr_bh')[1]
                for f in range(m):
                    rows.append({
                        'feature': f,
                        'class_1': class_names[i],
                        'class_2': class_names[j],
                        'effect': effect[f],
                        'std': std[f],
                        'z': z[f],
                        'pvalue': pvalue[f],
                        'fdr': fdr[f],
                        'group': group,
                    })
                    feature_index.append(f)
    columns = ['feature', 'class_1', 'class_2', 'effect', 'std', 'z', 'pvalue', 'fdr']
    if grouped:
        columns.append('group')
    return pd.DataFrame(rows, columns=columns, index=feature_index)


def activity_table(means, covariances, class_names, group_names=None) -> pd.DataFrame:
    """Long table of class activities and their standard deviations per feature."""
    class_names = _class_names(class_names)
    means, covariances, groups = _group_arrays(means, covariances, group_names)
    K = len(class_names)
    if means.shape[1] != K:
        raise ValueError(f'Expected {K} classes, got {means.shape[1]}.')
    m = means.shape[0]
    grouped = group_names is not None
    rows, feature_index = list(), list()
    for gi, group in enumerate(groups):
        for a in range(K):
            std = np.sqrt(np.clip(covariances[:, a, a, gi], 0.0, None))
            for f in range(m):
                rows.append({
                    'feature': f,
                    'class': class_names[a],
                    'activity': means[f, a, gi],
                    'std': std[f],
                    'group': group,
                })
                feature_index.append(f)
    columns = ['feature', 'class', 'activity', 'std']
    if grouped:
        columns.append('group')
    return pd.DataFrame(rows, columns=columns, index=feature_index)


def fixed_table(means, covariances, class_names) -> pd.DataFrame:
    """Long table of fixed-effect class means and their standard deviations per feature."""
    class_names = _class_names(class_names)
    means, covariances, groups = _group_arrays(means, covariances, None)
    K = len(class_names)
    if means.shape[1] != K:
        raise ValueError(f'Expected {K} classes, got {means.shape[1]}.')
    m = means.shape[0]
    rows, feature_index = list(), list()
    for a in range(K):
        std = np.sqrt(np.clip(covariances[:, a, a, 0], 0.0, None))
        for f in range(m):
            rows.append({
                'feature': f,
                'class': class_names[a],
                'mean': means[f, a, 0],
                'std': std[f],
            })
            feature_index.append(f)
    return pd.DataFrame(rows, columns=['feature', 'class', 'mean', 'std'], index=feature_index)
