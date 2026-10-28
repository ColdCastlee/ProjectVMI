"""Segmentation metrics computed from confusion matrices (rows = ground truth, columns = prediction).

The main metric is the mIoU over the 19 classes 0-18 (background
included, void = 19 never counted), with overall accuracy, macro-F1 and the IoU of
each class. A class absent from both the ground truth and the predictions has no
IoU or F1 (NaN) and is left out of the means. The functions accept a single
(K, K) matrix or a stack (..., K, K), e.g. bootstrap replicates.
"""

import numpy as np

N_CLASSES = 19  # 0 background, 1-18 crops
VOID = 19
CROPS = np.arange(1, N_CLASSES)


def confusion_matrix(y_true, y_pred, valid=None, n_classes=N_CLASSES):
    """(K, K) int64 counts over the pixels where y_true != VOID (and valid, if given)."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    keep = y_true != VOID
    if valid is not None:
        keep &= np.asarray(valid, dtype=bool).ravel()
    t = y_true[keep].astype(np.int64)
    p = y_pred[keep].astype(np.int64)
    if t.size and (t.max() >= n_classes or p.min() < 0 or p.max() >= n_classes):
        raise ValueError(f"labels or predictions outside 0..{n_classes - 1}")
    return np.bincount(t * n_classes + p, minlength=n_classes**2).reshape(n_classes, n_classes)


def _tp_fp_fn(cm):
    cm = np.asarray(cm, dtype=np.float64)
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    return tp, cm.sum(axis=-2) - tp, cm.sum(axis=-1) - tp


def per_class_iou(cm):
    tp, fp, fn = _tp_fp_fn(cm)
    den = tp + fp + fn
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, tp / den, np.nan)


def per_class_f1(cm):
    tp, fp, fn = _tp_fp_fn(cm)
    den = 2 * tp + fp + fn
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, 2 * tp / den, np.nan)


def _nanmean(x):
    x = np.asarray(x, dtype=np.float64)
    count = np.isfinite(x).sum(axis=-1)
    total = np.where(np.isfinite(x), x, 0).sum(axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(count > 0, total / count, np.nan)


def miou(cm, classes=None):
    iou = per_class_iou(cm)
    return _nanmean(iou if classes is None else iou[..., np.asarray(classes)])


def crops_only(cm):
    """Drop the pixels whose ground truth is background (row 0), as in the RF pilot.

    Use with classes=CROPS. Crop pixels predicted as background still count as errors.
    """
    cm = np.array(cm, copy=True)
    cm[..., 0, :] = 0
    return cm


def summarize(cm, classes=None):
    """Scores of one (K, K) confusion matrix; `classes` restricts mIoU and macro-F1."""
    cm = np.asarray(cm)
    sel = np.arange(cm.shape[-1]) if classes is None else np.asarray(classes)
    total = cm.sum()
    return {
        "miou": float(miou(cm, sel)),
        "oa": float(np.trace(cm) / total) if total else float("nan"),
        "macro_f1": float(_nanmean(per_class_f1(cm)[sel])),
        "iou": per_class_iou(cm),
        "pixels": int(total),
    }
