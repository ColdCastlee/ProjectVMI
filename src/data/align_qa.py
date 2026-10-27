"""Alignment check between an embedding and the PASTIS parcels.

If the embedding sits on the right grid, pixels of one parcel look alike, so the
share of variance left inside parcels is lowest at shift (0, 0). A shift (dx, dy)
pairs the label at (r, c) with the embedding pixel at (r + dy, c + dx).
"""

import numpy as np

SHIFTS = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]


def within_parcel_ratio(emb, inst, min_pixels=50):
    """emb (C, H, W) float, inst (H, W) parcel ids (0 = no parcel).

    Returns within-parcel variance / total variance over parcel pixels
    (lower = embedding boundaries match the parcels better), NaN if too few pixels.
    """
    valid = (inst > 0) & np.isfinite(emb).all(axis=0)
    if valid.sum() < min_pixels:
        return np.nan
    e = emb[:, valid].T.astype(np.float64)  # (n, C)
    _, inv = np.unique(inst[valid], return_inverse=True)
    counts = np.bincount(inv)
    means = np.zeros((counts.size, e.shape[1]))
    np.add.at(means, inv, e)
    means /= counts[:, None]
    within = ((e - means[inv]) ** 2).sum(axis=1).mean()
    total = e.var(axis=0).sum()
    return within / total if total > 0 else np.nan


def shift_scores(emb_margin, inst, m=1):
    """Ratios for SHIFTS; emb_margin is (C, H + 2m, W + 2m) around the patch."""
    h, w = inst.shape
    out = []
    for dx, dy in SHIFTS:
        win = emb_margin[:, m + dy : m + dy + h, m + dx : m + dx + w]
        out.append(within_parcel_ratio(win, inst))
    return np.array(out)


def best_shift(scores):
    if np.all(np.isnan(scores)):
        return (np.nan, np.nan)
    return SHIFTS[int(np.nanargmin(scores))]
