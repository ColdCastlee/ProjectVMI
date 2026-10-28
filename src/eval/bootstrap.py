"""Patch bootstrap of the mIoU.

Each replicate draws the N test patches with replacement, sums their confusion
matrices and computes the mIoU of the summed matrix (never an average of
per-patch mIoUs: a patch holds only a few classes). For two variants evaluated
on the same patches, `paired_delta` uses the same draws for both and reports the
95% percentile interval of the difference; an improvement is claimed only when
that interval excludes 0.
"""

import numpy as np

from .metrics import miou


def draw_weights(n_patches, n_boot=1000, seed=0):
    """(n_boot, n_patches) counts: how many times each patch is drawn in each replicate."""
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_patches, np.full(n_patches, 1 / n_patches), size=n_boot)


def replicate_miou(cms, weights, classes=None):
    """mIoU of each replicate; cms is (N, K, K) per-patch confusion matrices."""
    totals = np.einsum("bn,nkl->bkl", weights, np.asarray(cms, dtype=np.int64))
    return miou(totals, classes)


def interval(values, level=0.95):
    tail = (1 - level) / 2 * 100
    lo, hi = np.nanpercentile(values, [tail, 100 - tail])
    return float(lo), float(hi)


def bootstrap_miou(cms, n_boot=1000, seed=0, classes=None, level=0.95):
    cms = np.asarray(cms)
    point = float(miou(cms.sum(axis=0), classes))
    reps = replicate_miou(cms, draw_weights(len(cms), n_boot, seed), classes)
    lo, hi = interval(reps, level)
    return {"miou": point, "ci_low": lo, "ci_high": hi, "std": float(np.nanstd(reps))}


def paired_delta(cms_a, cms_b, n_boot=1000, seed=0, classes=None, level=0.95):
    """Difference mIoU(b) - mIoU(a) on the same patches, with a paired bootstrap interval."""
    cms_a, cms_b = np.asarray(cms_a), np.asarray(cms_b)
    if cms_a.shape != cms_b.shape:
        raise ValueError(f"different shapes {cms_a.shape} vs {cms_b.shape}: not the same test patches")
    w = draw_weights(len(cms_a), n_boot, seed)
    delta = replicate_miou(cms_b, w, classes) - replicate_miou(cms_a, w, classes)
    point = float(miou(cms_b.sum(axis=0), classes) - miou(cms_a.sum(axis=0), classes))
    lo, hi = interval(delta, level)
    return {"delta": point, "ci_low": lo, "ci_high": hi, "share_le_0": float(np.mean(delta <= 0))}
