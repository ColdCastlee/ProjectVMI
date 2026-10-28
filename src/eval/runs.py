"""Evaluation of a run on its test patches, and one line per run in runs.csv.

For every test patch three confusion matrices are kept (all valid pixels, parcel
boundary, parcel core) in <run_dir>/cm_per_patch.npz, so that any score and its
bootstrap interval can be recomputed later without the predictions.
Run names follow {method}_{variant}_{label%}_{split}_s{seed}, e.g. m1_gated_010_foldI_s1.
"""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..data.shards import code_version
from .metrics import N_CLASSES, confusion_matrix, summarize
from .regions import boundary_core

COLUMNS = (["run_id", "method", "variant", "label_frac", "split", "seed",
            "miou", "oa", "macro_f1", "miou_boundary", "miou_core"]
           + [f"iou_{k}" for k in range(N_CLASSES)]
           + ["n_params", "train_seconds", "n_test_patches", "commit", "finished_utc"])


def run_name(method, variant, label_frac, split, seed):
    return f"{method}_{variant}_{round(label_frac * 100):03d}_{split}_s{seed}"


def patch_cms(pred, label, inst, valid, width=1, n_classes=N_CLASSES):
    """(3, K, K) confusion matrices of one patch: all valid pixels, boundary, core."""
    boundary, core = boundary_core(inst, width)
    valid = np.asarray(valid, dtype=bool)
    return np.stack([
        confusion_matrix(label, pred, valid, n_classes),
        confusion_matrix(label, pred, valid & boundary, n_classes),
        confusion_matrix(label, pred, valid & core, n_classes),
    ])


class RunEvaluator:
    """Collects the per-patch confusion matrices of one run on its test patches."""

    def __init__(self, labels, instances, valid, width=1):
        self.labels, self.instances, self.valid, self.width = labels, instances, valid, width
        self.idx, self.cms = [], []

    def add(self, i, pred):
        self.idx.append(int(i))
        self.cms.append(patch_cms(pred, self.labels[i], self.instances[i], self.valid[i], self.width))

    def result(self):
        return np.asarray(self.idx), np.stack(self.cms)  # (N,), (N, 3, K, K)


def scores(cms):
    """Scores of the summed matrices; cms is (N, 3, K, K)."""
    total = cms.sum(axis=0)
    s_all, s_boundary, s_core = (summarize(total[k]) for k in range(3))
    row = {"miou": s_all["miou"], "oa": s_all["oa"], "macro_f1": s_all["macro_f1"],
           "miou_boundary": s_boundary["miou"], "miou_core": s_core["miou"]}
    row.update({f"iou_{k}": float(v) for k, v in enumerate(s_all["iou"])})
    return row


def log_run(runs_csv, run_dir, run, idx, cms, extra=None):
    """Save the per-patch matrices, append one line to runs_csv and return it.

    `run` holds run_id, method, variant, label_frac, split, seed, n_params, train_seconds.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(run_dir / "cm_per_patch.npz", idx=idx, all=cms[:, 0], boundary=cms[:, 1], core=cms[:, 2])
    row = {**run, **scores(cms), "n_test_patches": len(idx), "commit": code_version(),
           "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (run_dir / "run.json").write_text(json.dumps({**row, **(extra or {})}, indent=2, default=float))

    # one line per run_id: a rerun replaces the previous line of the same run
    runs_csv = Path(runs_csv)
    runs_csv.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    if runs_csv.exists():
        with open(runs_csv, newline="") as f:
            lines = [r for r in csv.DictReader(f) if r["run_id"] != row["run_id"]]
    lines.append({k: row.get(k, "") for k in COLUMNS})
    with open(runs_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(lines)
    return row


def load_cms(run_dir, region="all"):
    """(idx, (N, K, K)) per-patch matrices of a finished run; region: all, boundary or core."""
    with np.load(Path(run_dir) / "cm_per_patch.npz") as d:
        return d["idx"], d[region]
