"""Regression tests for degenerate (constant) loading columns.

A column of the loading matrix B that does not vary across promoters is
annihilated by the promoter-wise centering H_p that `fit` works in, so the motif
explains nothing, its tau and mu_m are unidentified and B^T B is singular. Such
columns are not only in the input data: the ECDF/ESF transform in `create`
*creates* them out of columns whose raw scores took one or two distinct values,
because the top survival-function level is clipped onto the next one.

Which promoters reach `transform_loadings` depends on the low-expression filter,
which depends on the expression matrix -- so the same motif panel can be fine for
one set of samples and degenerate for another. That is what made this show up as
"works on 15 cell lines, crashes on 12".

Run with pytest, or directly: python tests/test_degenerate_loadings.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from maradoner.create import transform_loadings
from maradoner.fit import _estimate_motif_variance_mom, warn_constant_loadings

N_PROM = 200


def _loadings() -> pd.DataFrame:
    rng = np.random.RandomState(0)
    return pd.DataFrame({
        # Varies properly: must survive every transform.
        'good': rng.rand(N_PROM),
        # Two distinct raw values, so std != 0 and the pre-transform check passes --
        # but ESF maps both onto the same level, producing a constant column.
        'two_values': np.array([1.0] * (N_PROM - 2) + [5.0, 5.0]),
        # Constant already in the raw scores: caught by the pre-transform check.
        'all_same': np.full(N_PROM, 3.0),
    })


def test_esf_drops_columns_it_collapses():
    out = transform_loadings(_loadings(), 'esf', verbose=False)
    assert list(out.columns) == ['good']
    assert out['good'].std() > 0


def test_ecdf_keeps_two_valued_columns():
    # ECDF maps the two raw levels onto two distinct CDF levels, so unlike ESF it does
    # not collapse the column; only the raw constant goes.
    out = transform_loadings(_loadings(), 'ecdf', verbose=False)
    assert list(out.columns) == ['good', 'two_values']


def test_none_transform_still_drops_raw_constants():
    out = transform_loadings(_loadings(), 'none', verbose=False)
    # 'two_values' does vary without a transform, so only the raw constant goes.
    assert list(out.columns) == ['good', 'two_values']


def test_transform_keeps_a_healthy_panel_intact():
    rng = np.random.RandomState(1)
    df = pd.DataFrame(rng.rand(N_PROM, 5), columns=[f'm{i}' for i in range(5)])
    out = transform_loadings(df.copy(), 'esf', verbose=False)
    assert list(out.columns) == list(df.columns)


def test_warn_constant_loadings_reports_indices():
    rng = np.random.RandomState(2)
    B = rng.rand(N_PROM, 4)
    B[:, 1] = 7.0                       # exactly constant
    assert list(warn_constant_loadings(B, ['a', 'b', 'c', 'd'], verbose=False)) == [1]
    assert not len(warn_constant_loadings(rng.rand(N_PROM, 3), verbose=False))


def test_mom_warm_start_survives_rank_deficient_B():
    """The method-of-moments warm start must not die on a singular B^T B.

    It only produces x0 for the REML optimiser, so unidentified motifs should fall
    back to the eps floor rather than raising `LinAlgError`/`ValueError`.
    """
    rng = np.random.RandomState(3)
    m, n = 12, 6
    B = rng.rand(N_PROM, m)
    B[:, 4] = 2.0                       # constant -> zero after centering
    B[:, 7] = B[:, 3]                   # exact duplicate -> collinear
    B = B - B.mean(axis=0, keepdims=True)
    assert np.linalg.matrix_rank(B) < m
    # Draw Y from the model itself, otherwise the moments S = Z^2 - Gamma_ii are pure
    # noise around zero and no motif gets a meaningful estimate.
    U = rng.randn(m, n) * np.sqrt(np.linspace(0.5, 2.0, m)).reshape(-1, 1)
    Y = B @ U + 0.3 * rng.randn(N_PROM, n)

    eps = 1e-14
    Sigma, G = _estimate_motif_variance_mom(Y, B, 0, 1e-1, eps=eps)
    assert Sigma.shape == (m,) and G.shape == (n,)
    assert np.isfinite(Sigma).all() and np.isfinite(G).all()
    assert Sigma[4] == eps               # the constant motif is pinned to the floor
    identified = np.delete(np.arange(m), 4)
    assert Sigma[identified].min() > 1.0  # everything else still gets a real estimate
    # The pseudo-inverse splits a collinear pair evenly instead of blowing up.
    assert np.isclose(Sigma[3], Sigma[7])


if __name__ == '__main__':
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            try:
                fn()
            except Exception as e:                       # noqa: BLE001
                failed += 1
                print(f'FAIL {name}: {type(e).__name__}: {e}')
            else:
                print(f'ok   {name}')
    sys.exit(1 if failed else 0)
