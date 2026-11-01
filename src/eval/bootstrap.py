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


def _check_same_patches(cms_list):
    shapes = {np.asarray(c).shape for c in cms_list}
    if len(shapes) != 1:
        raise ValueError(f"different shapes {shapes}: not the same test patches")


def bootstrap_miou_seeds(cms_list, n_boot=1000, seed=0, classes=None, level=0.95):
    """Mean mIoU over training seeds; cms_list holds one (N, K, K) array per seed, same test patches.

    The interval comes from the patch bootstrap (the same draws for every seed);
    std_seeds is the spread between seeds.
    """
    _check_same_patches(cms_list)
    w = draw_weights(len(cms_list[0]), n_boot, seed)
    reps = np.mean([replicate_miou(c, w, classes) for c in cms_list], axis=0)
    points = [float(miou(np.asarray(c).sum(axis=0), classes)) for c in cms_list]
    lo, hi = interval(reps, level)
    return {"miou": float(np.mean(points)), "std_seeds": float(np.std(points, ddof=1)) if len(points) > 1 else 0.0,
            "ci_low": lo, "ci_high": hi, "per_seed": points}


def paired_delta_seeds(cms_a, cms_b, n_boot=1000, seed=0, classes=None, level=0.95):
    """Mean over seeds of mIoU(b) - mIoU(a); cms_a and cms_b are lists with one (N, K, K) array per seed.

    The same patch draws are used for every seed and both variants.
    """
    _check_same_patches(list(cms_a) + list(cms_b))
    if len(cms_a) != len(cms_b):
        raise ValueError("not the same number of seeds")
    w = draw_weights(len(cms_a[0]), n_boot, seed)
    reps = np.mean([replicate_miou(b, w, classes) - replicate_miou(a, w, classes) for a, b in zip(cms_a, cms_b)],
                   axis=0)
    per_seed = [float(miou(np.asarray(b).sum(axis=0), classes) - miou(np.asarray(a).sum(axis=0), classes))
                for a, b in zip(cms_a, cms_b)]
    lo, hi = interval(reps, level)
    return {"delta": float(np.mean(per_seed)), "ci_low": lo, "ci_high": hi,
            "share_le_0": float(np.mean(reps <= 0)), "per_seed": per_seed}
