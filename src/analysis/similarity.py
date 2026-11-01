"""Redundancy between two embeddings sampled at the same pixels (rows = pixels, columns = dimensions).

- linear CKA (Kornblith et al., ICML 2019): 1 = same geometry up to rotation and
  isotropic scaling, 0 = unrelated;
- CCA: canonical correlations, i.e. how many directions the two spaces share;
- intrinsic dimension: number of principal components for a share of the variance;
- linear R²: share of the variance of one embedding predicted by a ridge
  regression on the other, measured on held-out pixels.
"""

import numpy as np


def center(x):
    x = np.asarray(x, dtype=np.float64)
    return x - x.mean(axis=0)


def linear_cka(x, y):
    x, y = center(x), center(y)
    cross = np.linalg.norm(x.T @ y) ** 2
    return float(cross / (np.linalg.norm(x.T @ x) * np.linalg.norm(y.T @ y)))


def _inv_sqrt(c, eps=1e-12):
    w, v = np.linalg.eigh(c)
    return (v / np.sqrt(np.maximum(w, eps))) @ v.T


def canonical_correlations(x, y, reg=1e-6):
    """Sorted canonical correlations (min(dx, dy) values in [0, 1])."""
    x, y = center(x), center(y)
    n = len(x)
    cxx, cyy, cxy = x.T @ x / (n - 1), y.T @ y / (n - 1), x.T @ y / (n - 1)
    cxx += reg * np.trace(cxx) / len(cxx) * np.eye(len(cxx))
    cyy += reg * np.trace(cyy) / len(cyy) * np.eye(len(cyy))
    s = np.linalg.svd(_inv_sqrt(cxx) @ cxy @ _inv_sqrt(cyy), compute_uv=False)
    return np.clip(s, 0, 1)


def explained_variance(x):
    """Share of variance of each principal component, in decreasing order."""
    w = np.linalg.eigvalsh(np.cov(center(x), rowvar=False))[::-1]
    w = np.maximum(w, 0)
    return w / w.sum()


def dims_for(ratios, level):
    """Smallest number of components reaching `level` of the variance."""
    return int(np.searchsorted(np.cumsum(ratios), level) + 1)


def ridge_r2(x_train, y_train, x_test, y_test, lam=1e-3):
    """Held-out R² of y predicted linearly from x (all dimensions of y pooled)."""
    x_train, y_train = np.asarray(x_train, np.float64), np.asarray(y_train, np.float64)
    mx, my = x_train.mean(axis=0), y_train.mean(axis=0)
    xc, yc = x_train - mx, y_train - my
    gram = xc.T @ xc
    gram += lam * np.trace(gram) / len(gram) * np.eye(len(gram))
    w = np.linalg.solve(gram, xc.T @ yc)
    pred = (np.asarray(x_test, np.float64) - mx) @ w + my
    resid = ((np.asarray(y_test) - pred) ** 2).sum()
    total = ((np.asarray(y_test) - my) ** 2).sum()
    return float(1 - resid / total)
