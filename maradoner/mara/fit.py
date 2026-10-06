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
from ..utils import InputError, warn, load_stage, dump_stage, stage_filename, apply_clustering
from ..utils import remove_stage


def _remove_outdated_stages(project: str, stages, fmt: str, verbose: bool = True) -> None:
    for stage in stages:
        filename = stage_filename(project, stage, fmt)
        if os.path.isfile(filename):
            remove_stage(project, stage, fmt)
            logger_print(f'Removed outdated "{filename}" (it belongs to the previous results of this project).', verbose)
from ..fit import LowrankDecomposition, lowrank_decomposition, TransformedData, \
    ClusteringMode, cluster_data, FOVResult, TestResult
import dill
import os


def load_mara_stage(project: str, stage: str, fmt: str, required: bool = True):
    """Stage files of the classical MARA sub-app are stored as <project>.old.<stage>.<fmt>."""
    res = load_stage(project, f'old.{stage}', fmt, required=False)
    if res is None and required:
        command = 'gof' if stage == 'fov' else stage
        raise InputError(f'"{stage_filename(project, f"old.{stage}", fmt)}" not found: run '
                         f'"maradoner mara {command} {project}" first.')
    return res


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
        # log|V| = const + sum_k log(1 + tau s_k / sigma)
        logdet += np.log(S).sum()
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
    # (the default absolute tolerance of 1e-5 is coarse for the typical tau of 1e-3 .. 1e-2)
    tau = calc_tau(minimize_scalar(fun, bounds=(0.0, error_variance.variance.max() * 10), method='bounded',
                                   options=dict(xatol=1e-10)).x, error_variance.variance, mode)
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
        num_clusters: int, test_chromosomes: list, test_promoters_filename: str,
        gpu: bool, gpu_decomposition: bool, x64=True,
        verbose=True, dump=True) -> ActivitiesPrediction:
    if x64:
        jax.config.update("jax_enable_x64", True)
    if tau_estimation == TauEstimation.fixed and not (np.isfinite(tau_fix) and tau_fix > 0):
        raise InputError(f'--tau-fix must be a positive number, but {tau_fix:g} was given.')
    data = read_init(project)
    fmt = data.fmt
    group_names = data.group_names
    if clustering != clustering.none:
        if not 1 <= num_clusters <= data.B.shape[1]:
            raise InputError(f'--num-clusters must lie between 1 and the number of motifs '
                             f'({data.B.shape[1]}), but {num_clusters} was given.')
        logger_print('Clustering data...', verbose)
    data.B, clustering = cluster_data(data.B, mode=clustering, 
                                      num_clusters=num_clusters)
    # Cluster profiles of all the promoters: the later commands work with them (see
    # utils.apply_clustering).
    clustered_B = data.B if clustering is not None else None
    if test_promoters_filename:
        if not os.path.isfile(test_promoters_filename):
            raise InputError(f'File with the test promoters "{test_promoters_filename}" not found.')
        with open(test_promoters_filename, 'r') as f:
            test_chromosomes = filter(lambda x: len(x), map(lambda x: x.strip(), f.readlines()))
            test_chromosomes = set(test_chromosomes)
            promoter_inds_to_drop = [i for i, p in enumerate(data.promoter_names) 
                                     if p in test_chromosomes]
    elif test_chromosomes:
        import re
        pattern = re.compile(r'chr([0-9XYM]+|\d+)')

        test_chromosomes = set(c if str(c).startswith('chr') else f'chr{c}' for c in test_chromosomes)
        found = [pattern.search(p) for p in data.promoter_names]
        promoter_inds_to_drop = [i for i, c in enumerate(found)
                                 if c is not None and c.group() in test_chromosomes]
    else:
        promoter_inds_to_drop = None
    if promoter_inds_to_drop is not None and not 0 < len(promoter_inds_to_drop) < len(data.promoter_names):
        raise InputError(f'The requested testing set selects {len(promoter_inds_to_drop)} of the '
                         f'{len(data.promoter_names)} promoters of the project, but both the testing and the '
                         'training set must be non-empty. Check --test-chromosomes (promoter names must '
                         'contain the chromosome, e.g. chr3:1000-2000) or the names listed in '
                         '--test-promoters-filename.')
    # if promoter_inds_to_drop is not None:
    #     data.Y = np.delete(data.Y, promoter_inds_to_drop, axis=0)
    #     data.B = np.delete(data.B, promoter_inds_to_drop, axis=0)
    data, _ = split_data(data, promoter_inds_to_drop)
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
                    clustering=clustering, clustered_B=clustered_B, B_decomposition=B_decomposition,
                    group_names=group_names, promoter_inds_to_drop=promoter_inds_to_drop)    
    if dump:
        dump_stage(res, project, 'old.fit', fmt)
        _remove_outdated_stages(project, ('old.predict', 'old.fov'), fmt, verbose=verbose)
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
    fit = load_mara_stage(project, 'fit', fmt)
    data = apply_clustering(data, fit)
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
        dump_stage(activities, project, 'old.predict', fmt)
        _remove_outdated_stages(project, ('old.fov',), fmt)
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
        
        B_hat = Bs[0] - Bs[0].mean(axis=0, keepdims=True)
        Y_hat = Bs[1] - Bs[1].mean(axis=0, keepdims=True)
        mu_m = (np.linalg.pinv(B_hat) @ Y_hat).mean(axis=1, keepdims=True)
        # Bs = None
        if Bs is None:
            # data = transform_data(data)
            B = data.B if activities.clustering is None else activities.clustering[0]
            Y = data.Y
            U = activities.U
        else:
            B_ = Bs[0]
            # The activities of the fitted model: they were estimated with the loadings
            # centred by the column means of the training promoters (see transform_data),
            # so the loadings at hand are centred the same way.
            B = data.B - Bs[0].mean(axis=0, keepdims=True)
            Y = data.Y
            U = activities.U
            # B = np.hstack((B, np.ones((len(B), 1))))
            # U = np.linalg.pinv(np.hstack((Bs[0], np.ones((len(Bs[0]), 1))))) @ Bs[1]
            # m0 = Y.mean(axis=1, keepdims=True)
            # m2 = 0
            # m0 = 0
        m1 = Y.mean(axis=1, keepdims=True)
        m1 = B @ mu_m
        if keep_motifs is not None:
            B = B[:, keep_motifs]
            U = U[keep_motifs]
        
        # B = B - B.mean(axis=0, keepdims=True)
        d = B @ U
        # m2 = Y.mean(axis=0, keepdims=True)
        # m0 = Y.mean()
        m = (m1 + m2 - m0)
        d = d + m
        stat_0 = sub(Y, d)
        return stat_0,
    data = read_init(project)
    fmt = data.fmt
    fit = load_mara_stage(project, 'fit', fmt)
    activities = load_mara_stage(project, 'predict', fmt)
    data = apply_clustering(data, fit)
    motif_names = list(data.motif_names)
    if keep_motifs:
        import datatable as dt
        if not os.path.isfile(str(keep_motifs)):
            raise InputError(f'File "{keep_motifs}" (--keep-motifs) not found.')
        df = dt.fread(str(keep_motifs)).to_pandas()
        if df.shape[1] < 2 or 'status' not in df.columns[1:]:
            raise InputError(f'"{keep_motifs}" (--keep-motifs) must be a table with a header line, motif '
                             'names in the first column and a column named "status"; its columns are: '
                             f'{", ".join(map(str, df.columns))}.')
        df = df.groupby('status')
        keep_motifs = list()
        missing = list()
        for name, motifs in df:
            inds = list()
            for mot in motifs.iloc[:, 0]:
                try:
                    i = motif_names.index(mot)
                    inds.append(i)
                except ValueError:
                    missing.append(str(mot))
            keep_motifs.append((name, np.array(inds, dtype=int)))
        if missing:
            warn(f'{len(missing)} motif(s) of the --keep-motifs table are not in the project and are '
                 f'ignored: {", ".join(missing[:5])}' + (f', ... and {len(missing) - 5} more.' if len(missing) > 5 else '.'))
    else:
        keep_motifs = [(None, None)]
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
    dump_stage(results, project, 'old.fov', fmt)
    return results
        
        
        