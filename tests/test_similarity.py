"""Unit tests of src/analysis/similarity.py on synthetic data with a known answer."""

import numpy as np
import pytest

from src.analysis.similarity import (canonical_correlations, dims_for, explained_variance, linear_cka,
                                     ridge_r2)

rng = np.random.default_rng(0)
X = rng.normal(size=(3000, 6))
NOISE = rng.normal(size=(3000, 4))


def test_cka_invariances_and_independence():
    rotation, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    assert linear_cka(X, X) == pytest.approx(1)
    assert linear_cka(X, 3 * X @ rotation + 5) == pytest.approx(1)
    assert linear_cka(X, NOISE) < 0.05


def test_canonical_correlations():
    mixed = X @ rng.normal(size=(6, 6))  # invertible linear map: every direction shared
    # the small ridge term of canonical_correlations lowers exact correlations very slightly
    assert canonical_correlations(X, mixed) == pytest.approx(np.ones(6), abs=1e-3)
    cc = canonical_correlations(X, NOISE)
    assert len(cc) == 4 and cc.max() < 0.15
    half = np.hstack([X[:, :2], NOISE[:, :2]])  # two shared directions out of four
    cc = canonical_correlations(X, half)
    assert cc[:2] == pytest.approx([1, 1], abs=1e-4) and cc[2:].max() < 0.15


def test_explained_variance_and_dims():
    ratios = explained_variance(X * np.array([10, 5, 1, 1, 1, 1]))
    assert ratios.sum() == pytest.approx(1) and np.all(np.diff(ratios) <= 1e-12)
    assert dims_for(np.array([0.5, 0.3, 0.2]), 0.8) == 2
    assert dims_for(np.array([0.5, 0.3, 0.2]), 0.81) == 3


def test_ridge_r2():
    y = X @ rng.normal(size=(6, 3))
    assert ridge_r2(X[:2000], y[:2000], X[2000:], y[2000:], lam=1e-9) == pytest.approx(1, abs=1e-6)
    assert abs(ridge_r2(X[:2000], NOISE[:2000], X[2000:], NOISE[2000:])) < 0.02
