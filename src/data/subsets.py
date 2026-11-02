"""Low-label training subsets: a fraction of the training patches.

Rule fixed before any run. For a seed, the training patches are put in one order:
first a covering set, then all other patches in random order. Covering set:
classes 1-18 are taken from the rarest to the most common (rarity = number of
training patches where the class has at least `min_pixels` valid pixels), and for
each class not yet covered one of its patches is drawn at random. The subset at
fraction f is the first round(f * N) patches of that order (at least one), so
for a seed the subsets are nested (1% in 5% in 10% ...) and every variant of
both methods gets the same patches. When the budget is smaller than the covering
set (possible at 1%), the covering set is cut and the classes without any pixel
in the subset are reported.
"""

import numpy as np

from ..eval.metrics import CROPS, N_CLASSES

MIN_PIXELS = 100  # 1 ha at 10 m


def class_pixels(labels, valid, patches, n_classes=N_CLASSES):
    """(len(patches), n_classes) number of valid pixels of each class in each patch."""
    out = np.zeros((len(patches), n_classes), dtype=np.int64)
    for k, i in enumerate(patches):
        y = np.asarray(labels[i])[np.asarray(valid[i], dtype=bool)]
        out[k] = np.bincount(y, minlength=n_classes)[:n_classes]
    return out


def patch_order(pixels, seed, min_pixels=MIN_PIXELS):
    """Positions 0..N-1 of the patches in subset order, and the length of the covering set."""
    present = pixels >= min_pixels
    rng = np.random.default_rng(seed)
    rarity = present[:, CROPS].sum(axis=0)
    order, covered = [], np.zeros(pixels.shape[1], dtype=bool)
    for c in CROPS[np.argsort(rarity, kind="stable")]:
        candidates = np.flatnonzero(present[:, c])
        if covered[c] or len(candidates) == 0:
            continue
        k = int(rng.choice(candidates))
        order.append(k)
        covered |= present[k]
    rest = np.setdiff1d(np.arange(len(pixels)), order)
    return np.concatenate([np.asarray(order, dtype=np.int64), rng.permutation(rest)]), len(order)


def label_subset(labels, valid, train, frac, seed, min_pixels=MIN_PIXELS):
    """Sorted training patches for label fraction `frac` and `seed`, and a summary of the subset."""
    train = np.asarray(train)
    pixels = class_pixels(labels, valid, train)
    order, n_cover = patch_order(pixels, seed, min_pixels)
    n = len(train) if frac >= 1 else max(1, int(round(frac * len(train))))
    chosen = order[:n]
    got = pixels[chosen].sum(axis=0)
    info = {"label_frac": float(frac), "seed": int(seed), "n_patches": int(n), "n_train": int(len(train)),
            "n_cover": int(n_cover), "min_pixels": int(min_pixels),
            "missing_classes": [int(c) for c in CROPS if got[c] == 0], "class_pixels": got.tolist()}
    return np.sort(train[chosen]), info
