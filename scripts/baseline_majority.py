"""Trivial baseline: predict the most frequent training class everywhere.

It is the lower bound of every table and an end-to-end check of the evaluation
code: per-patch confusion matrices (all / boundary / core), runs.csv, bootstrap.

Example:
    python scripts/baseline_majority.py --processed /content/drive/MyDrive/pj_vmi/processed \
        --split I --runs-dir /content/drive/MyDrive/pj_vmi/runs \
        --runs-csv /content/drive/MyDrive/pj_vmi/results/runs.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.splits import load_split, split_tag
from src.eval.bootstrap import bootstrap_miou
from src.eval.metrics import CROPS, N_CLASSES, crops_only, summarize
from src.eval.runs import RunEvaluator, log_run, run_name


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed", required=True)
    ap.add_argument("--split", default="I", help="I..V (official folds) or loto_<tile>")
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--runs-csv", required=True)
    args = ap.parse_args()

    processed = Path(args.processed)
    labels = np.load(processed / "labels.npy", mmap_mode="r")
    instances = np.load(processed / "instances.npy", mmap_mode="r")
    valid = np.load(processed / "valid_mask.npy", mmap_mode="r")
    part = load_split(processed, args.split)

    train = np.asarray(part["train"])
    counts = np.bincount(labels[train][valid[train]].ravel(), minlength=N_CLASSES)[:N_CLASSES]
    majority = int(np.argmax(counts))
    print(f"split {args.split}: {len(train)} train, {len(part['test'])} test patches; "
          f"majority class {majority} ({counts[majority] / counts.sum():.1%} of the valid train pixels)")

    evaluator = RunEvaluator(labels, instances, valid)
    for i in part["test"]:
        evaluator.add(i, np.full(labels.shape[1:], majority, dtype=np.int64))
    idx, cms = evaluator.result()

    tag = split_tag(args.split)
    run = {"run_id": run_name("base", "majority", 1.0, tag, 0), "method": "baseline", "variant": "majority",
           "label_frac": 1.0, "split": tag, "seed": 0, "n_params": 0, "train_seconds": 0}
    row = log_run(args.runs_csv, Path(args.runs_dir) / run["run_id"], run, idx, cms,
                  extra={"majority_class": majority})
    boot = bootstrap_miou(cms[:, 0])
    crops = summarize(crops_only(cms[:, 0].sum(axis=0)), CROPS)
    print(f"{run['run_id']}: mIoU {row['miou']:.4f} (95% CI {boot['ci_low']:.4f}-{boot['ci_high']:.4f}), "
          f"OA {row['oa']:.4f}, macro-F1 {row['macro_f1']:.4f}, "
          f"boundary {row['miou_boundary']:.4f}, core {row['miou_core']:.4f}")
    print(f"crops only (18 classes, background pixels left out): mIoU {crops['miou']:.4f}, OA {crops['oa']:.4f}")


if __name__ == "__main__":
    main()
