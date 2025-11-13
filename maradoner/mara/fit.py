import numpy as np
import jax.numpy as jnp
import jax
from dataclasses import dataclass
from functools import partial
from sklearn.model_selection import RepeatedKFold
from collections import defaultdict
from scipy.optimize import minimize_scalar
from enum import Enum
from ..utils import read_init, ProjectData, logger_print, openers
from ..fit import LowrankDecomposition, lowrank_decomposition, TransformedData, \
    ClusteringMode, cluster_data, FOVResult, TestResult
import dill


class GOFStat(str, Enum):
    fov = 'fov'
    corr = 'corr'

class GOFStatMode(str, Enum):
    residual = 'residual'
    total = 'total'
    
class TauEstimation(str, Enum):
    fixed = 'fixed'
    mle = 'mle'

class TauMode(str, Enum):
    mara = 'mara'
    ismara = 'ismara'



@dataclass(frozen=True)
class ErrorVarianceEstimates:
    variance: np.ndarray

@dataclass(frozen=True)
class MotifVarianceEstimates:
    variance: np.ndarray


@dataclass(frozen=True)
class FitResult:
    error_variance: ErrorVarianceEstimates
    motif_variance: MotifVarianceEstimates
    B_decomposition: LowrankDecomposition
    group_names: list
    clustering: np.ndarray = None
    clustered_B: np.ndarray = None
    promoter_inds_to_drop: list = None


def transform_data(data, std_y=False, std_b=False) -> TransformedData:
    Y = data.Y - (data.Y.mean(axis=0, keepdims=True) + data.Y.mean(axis=1, keepdims=True) - data.Y.mean())
    B = data.B - data.B.mean(axis=0, keepdims=True)
    group_inds_inv = list()
    group_inds = data.group_inds
    d = dict()
    for i, items in enumerate(group_inds):
        for j in items:
            d[j] = i
    for i in sorted(d.keys()):
        group_inds_inv.append(d[i])
    group_inds_inv = np.array(group_inds_inv)
    return TransformedData(Y=Y, B=B, 
                           group_inds=group_inds,
                           group_inds_inv=group_inds_inv)



def estimate_error_variance(data: TransformedData,
                            B_decomposition: LowrankDecomposition) -> ErrorVarianceEstimates:
    # Y = B_decomposition.null_Q.T @ data.Y
    Y = B_decomposition.null_space_transform(data.Y)
    variance = (Y ** 2).mean(axis=0)
    return ErrorVarianceEstimates(variance)


def calc_tau(tau: float, error_variance: np.ndarray, mode: TauMode):
    if mode == mode.mara:
        taus = tau * np.ones_like(error_variance)
    else:
        taus = tau / error_variance
    return taus

def loglik_tau(tau: float, Sigma: np.ndarray, Y_hat: np.ndarray,
               error_variance: np.ndarray, mode: TauMode) -> float:
    vec = 0
    logdet = 0
    taus = calc_tau(tau, error_variance, mode)
    for sigma, tau, y in zip(error_variance, taus, Y_hat.T):
        S = tau / sigma * Sigma + 1
        vec += (y ** 2 / S).sum() * (tau / sigma ** 2)
        logdet += S.sum()
    return -vec + logdet
        
def estimate_motif_variance(data: TransformedData, B_decomposition: LowrankDecomposition,
                            error_variance: ErrorVarianceEstimates,
                            estimation_method: TauEstimation,
                            mode: TauMode,
                            fixed_value=0.1) -> MotifVarianceEstimates:
    if estimation_method == estimation_method.fixed:
        tau = calc_tau(fixed_value, error_variance.variance, mode)
        return MotifVarianceEstimates(tau)
    Sigma = B_decomposition.S ** 2
    Q = B_decomposition.V.T
    Y_hat = Q.T @ data.B.T @ data.Y
    fun = partial(loglik_tau, Sigma=Sigma, Y_hat=Y_hat, error_variance=error_variance.variance,
                  mode=mode)
    tau = calc_tau(minimize_scalar(fun, bounds=(0.0, error_variance.variance.max() * 10)).x, error_variance.variance, mode)
    return MotifVarianceEstimates(tau)
    

@dataclass(frozen=True)
class ActivitiesPrediction:
    U: np.ndarray
    variance: np.ndarray
    clustering: tuple[np.ndarray, np.ndarray] = None
            

def predict_activities(data: TransformedData, fit: FitResult, 
                       gpu=False, verbose=True) -> ActivitiesPrediction:
    U = list()
    variance = list()
    
    B_decomposition = fit.B_decomposition
    if gpu:
        device = jax.devices()
    else:
        device = jax.devices('cpu')
    device = next(iter(device))
    with jax.default_device(device):
        Y_hat = data.B.T @ data.Y
        Sigma = B_decomposition.S ** 2
        Q = B_decomposition.V.T
        for y, sigma, tau in zip(Y_hat.T, fit.error_variance.variance, fit.motif_variance.variance):
            Z = Q / (1 / tau + Sigma / sigma) @ Q.T
            u = Z @ y / sigma 
            variance.append(Z.diagonal())
            U.append(u)
    U = np.array(U).T
    variance = np.array(variance).T
    return ActivitiesPrediction(U, variance)


def fit(project: str, tau_mode: TauMode, tau_estimation: TauEstimation,
        tau_fix: float, clustering: ClusteringMode,
        num_clusters: int, test_chromosomes: list,
        gpu: bool, gpu_decomposition: bool, x64=True,
        verbose=True, dump=True) -> ActivitiesPrediction:
    if x64:
        jax.config.update("jax_enable_x64", True)
    data = read_init(project)
    fmt = data.fmt
    group_names = data.group_names
    if clustering != clustering.none:
        logger_print('Clustering data...', verbose)
    data.B, clustering = cluster_data(data.B, mode=clustering, 
                                      num_clusters=num_clusters)
    if test_chromosomes:
        import re
        pattern = re.compile(r'chr([0-9XYM]+|\d+)')
        
        test_chromosomes = set(test_chromosomes)
        promoter_inds_to_drop = [i for i, p in enumerate(data.promoter_names) 
                                 if pattern.search(p).group() in test_chromosomes]
        data.Y = np.delete(data.Y, promoter_inds_to_drop, axis=0)
        data.B = np.delete(data.B, promoter_inds_to_drop, axis=0)
    else:
        promoter_inds_to_drop = None
    logger_print('Transforming data...', verbose)
    data = transform_data(data)
    if gpu_decomposition:
        device = jax.devices()
    else:
        device = jax.devices('cpu')
    device = next(iter(device))

    logger_print('Computing low-rank decompositions of the loading matrix...', verbose)
    with jax.default_device(device):
        B_decomposition = lowrank_decomposition(data.B)
    if gpu:
        device = jax.devices()
    else:
        device = jax.devices('cpu')
    device = next(iter(device))
    # print(data.B.shape, data_orig.B.shape)
    with jax.default_device(device):

        logger_print('Estimating error variances...', verbose)
        error_variance = estimate_error_variance(data, B_decomposition)       
        
        logger_print('Estimating variances of motif activities...', verbose)
        motif_variance = estimate_motif_variance(data, B_decomposition,
                                                  error_variance=error_variance,
                                                  mode=tau_mode, estimation_method=tau_estimation,
                                                  fixed_value=tau_fix)
        
    
    res = FitResult(error_variance=error_variance, motif_variance=motif_variance,
                    clustering=clustering, B_decomposition=B_decomposition,
                    group_names=group_names, promoter_inds_to_drop=promoter_inds_to_drop)    
    if dump:
        with openers[fmt](f'{project}.old.fit.{fmt}', 'wb') as f:
            dill.dump(res, f)
    return res

def split_data(data: ProjectData, inds: list) -> tuple[ProjectData, ProjectData]:
    if not inds:
        return data, None
    B = data.B
    Y = data.Y
    # Y = Y -  (Y.mean(axis=0, keepdims=True) + Y.mean(axis=1, keepdims=True) - Y.mean())
    # B = B - B.mean(axis=0, keepdims=True)
    B_d = np.delete(B, inds, axis=0)
    B = B[inds]
    Y_d = np.delete(Y, inds, axis=0)
    Y = Y[inds]
    promoter_names_d = np.delete(data.promoter_names, inds)
    promoter_names = list(np.array(data.promoter_names)[inds])
    data_d = ProjectData(Y=Y_d, B=B_d, K=data.K, weights=data.weights,
                         group_inds=data.group_inds, group_names=data.group_names,
                         motif_names=data.motif_names, promoter_names=promoter_names_d,
                         motif_postfixes=data.motif_postfixes, sample_names=data.sample_names,
                         fmt=data.fmt)
    data = ProjectData(Y=Y, B=B, K=data.K, weights=data.weights,
                         group_inds=data.group_inds, group_names=data.group_names,
                         motif_names=data.motif_names, promoter_names=promoter_names,
                         motif_postfixes=data.motif_postfixes, sample_names=data.sample_names,
                         fmt=data.fmt)
    return data_d, data

def predict(project: str,  gpu: bool, x64=True, dump=True):
    if x64:
        jax.config.update("jax_enable_x64", True)
    data = read_init(project)
    fmt = data.fmt
    with openers[fmt](f'{project}.old.fit.{fmt}', 'rb') as f:
        fit = dill.load(f)
    data, _ = split_data(data, fit.promoter_inds_to_drop)
    data = transform_data(data)
    if gpu:
        device = jax.devices()
    else:
        device = jax.devices('cpu')
    device = next(iter(device))
    with jax.default_device(device):
        activities = predict_activities(data, fit)
    if dump:
        with openers[fmt](f'{project}.old.predict.{fmt}', 'wb') as f:
            dill.dump(activities, f)
    return activities

def _cor(a, b, axis=1):
    a_centered = a - a.mean(axis=axis, keepdims=True)
    b_centered = b - b.mean(axis=axis, keepdims=True)
    numerator = np.sum(a_centered * b_centered, axis=axis)
    denominator = np.sqrt(np.sum(a_centered**2, axis=axis) * np.sum(b_centered**2, axis=axis))
    return numerator / denominator

def calculate_fov(project: str, gpu: bool, 
                  stat_type: GOFStat, keep_motifs: str, x64=True,
                  verbose=True, dump=True):
    keep_motifs = None
    def calc_fov(data: TransformedData, fit: FitResult,
                 activities: ActivitiesPrediction, keep_motifs=None, Bs=None) -> tuple[FOVResult]:
        def sub(Y, effects) -> FOVResult:
            if stat_type == stat_type.fov:
                Y1 = Y - effects
                Y = Y - Y.mean()
                Y1 = Y1 - Y1.mean()
                Y = Y ** 2
                Y1 = Y1 ** 2
                prom = 1 - Y1.mean(axis=1) / Y.mean(axis=1)
                sample = 1 - Y1.mean(axis=0) / Y.mean(axis=0)
                total = 1 - Y1.mean() / Y.mean()
            elif stat_type == stat_type.corr:
                # Y = Y - Y.mean(axis=0, keepdims=True) - Y.mean(axis=1, keepdims=True) + Y.mean()
                total = np.corrcoef(Y.flatten(), effects.flatten())[0, 1]
                prom = _cor(Y, effects, axis=1)
                sample = _cor(Y, effects, axis=0)
            return FOVResult(total, prom, sample)
        m2 = Bs[1].mean(axis=0, keepdims=True)
        m0 = Bs[1].mean()
        
        # Bs = None
        if Bs is None:
            # data = transform_data(data)
            B = data.B if activities.clustering is None else activities.clustering[0]
            Y = data.Y
            U = activities.U
        else:
            B_ = Bs[0]
            B = data.B
            Y = data.Y
            U = np.linalg.pinv(Bs[0]) @ (Bs[1] - Bs[1].mean(axis=0, keepdims=True) - Bs[1].mean(axis=1, keepdims=True) + Bs[1].mean())
            # B = np.hstack((B, np.ones((len(B), 1))))
            # U = np.linalg.pinv(np.hstack((Bs[0], np.ones((len(Bs[0]), 1))))) @ Bs[1]
            # m0 = Y.mean(axis=1, keepdims=True)
            # m2 = 0
            # m0 = 0
        if keep_motifs is not None:
            B = B[:, keep_motifs]
            U = U[keep_motifs]
        # B = B - B.mean(axis=0, keepdims=True)
        d = B @ U
        # m2 = Y.mean(axis=0, keepdims=True)
        # m0 = Y.mean()
        m = (Y.mean(axis=1, keepdims=True) + m2 - m0)
        d = d + m
        stat_0 = sub(Y, d)
        return stat_0,
    data = read_init(project)
    fmt = data.fmt
    motif_names = data.motif_names
    if keep_motifs:
        import datatable as dt
        df = dt.fread(keep_motifs).to_pandas().groupby('status')
        keep_motifs = list()
        for name, motifs in df:
            inds = list()
            for mot in motifs.iloc[:, 0]:
                try:
                    i = motif_names.index(mot)
                    inds.append(i)
                except ValueError:
                    print(f'Motif {mot} not found in the project.')
            keep_motifs.append((name, np.array(inds, dtype=int)))
    else:
        keep_motifs = [(None, None)]
    with openers[fmt](f'{project}.old.fit.{fmt}', 'rb') as f:
        fit = dill.load(f)
    with openers[fmt](f'{project}.old.predict.{fmt}', 'rb') as f:
        activities = dill.load(f)
    data, data_test = split_data(data, fit.promoter_inds_to_drop)
    if x64:
        jax.config.update("jax_enable_x64", True)
    # data = transform_data(data, helmert=False)
    # if data_test is not None:
    #     data_test = transform_data(data_test, helmert=False)
    if gpu:
        device = jax.devices()
    else:
        device = jax.devices('cpu')
    device = next(iter(device))
    results = list()
    for status_name, motifs in keep_motifs:
        if status_name:
            status_name = f'{status_name} ({len(motifs)})'
        with jax.default_device(device):
    
            if data_test is not None:
                test_FOV = calc_fov(data=data_test, fit=fit, activities=activities, keep_motifs=motifs,
                                    Bs=(data.B, data.Y)
                                    )
            train_FOV = calc_fov(data=data, fit=fit, activities=activities, keep_motifs=motifs,
                                 Bs=(data.B, data.Y)
                                 )
        if data_test is None:
            test_FOV = None
        res = TestResult(train_FOV, test_FOV, grouped=False)
        results.append((status_name, res))
    with openers[fmt](f'{project}.old.fov.{fmt}', 'wb') as f:
        dill.dump(results, f)
    return results
        
        
        