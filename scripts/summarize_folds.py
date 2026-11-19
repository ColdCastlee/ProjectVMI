"""Pooled summary over the official folds: every test patch counted once, one or more seeds.

The five official PASTIS folds have disjoint test sets that together cover the
2,433 patches. For every variant and seed, the per-patch confusion matrices of its
runs on folds I-V are put together; the pooled mIoU is the mIoU of their sum.
The patch bootstrap draws from all pooled patches, and paired differences use the
same draws for both variants. With several seeds, the pooled mIoU and the
differences are means over seeds, with the same patch draws for every seed (as in
summarize_runs.py). The per-fold mIoUs and differences are also kept, to see
whether a gain holds on every fold.

Outputs in --out: <name>_per_fold.csv, <name>_summary.csv, <name>_deltas.csv
(all valid pixels, parcel boundary and parcel core) and <name>_per_class.png.

Example:
    python scripts/summarize_folds.py --runs-dir /content/drive/MyDrive/pj_vmi/runs \
        --out /content/drive/MyDrive/pj_vmi/results/m1 --name fusion_unet_5folds \
        --variants base_unet-t19-w32 m1_unet-t19-ae19-w32 --pairs m1_unet-t19-ae19-w32:base_unet-t19-w32 \
        --seeds 0 1 2
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.summarize_runs import REGIONS, parse_pair, run_id_of
from src.eval.bootstrap import bootstrap_miou_seeds, paired_delta_seeds
from src.eval.metrics import CLASS_NAMES, miou, per_class_iou
from src.eval.runs import load_cms

FOLDS = ("I", "II", "III", "IV", "V")


def load_pooled(runs_dir, tag, splits, label_frac, seed):
    """Per-fold rows, pooled patch indices, fold of each patch and {region: (N, K, K)} of one variant."""
    rows, idx_parts, fold_parts, cm_parts = [], [], [], {r: [] for r in REGIONS}
    for split in splits:
        rid = run_id_of(tag, label_frac, split, seed)
        d = Path(runs_dir) / rid
        if not (d / "run.json").exists():
            raise FileNotFoundError(f"missing run {d / 'run.json'}")
        info = json.loads((d / "run.json").read_text())
        for r in REGIONS:
            idx, cm = load_cms(d, r)
            cm_parts[r].append(cm)
        idx_parts.append(idx)
        fold_parts.append(np.full(len(idx), split))
        rows.append({"variant": tag, "split": split, "seed": seed, "run_id": rid, "n_test_patches": len(idx),
                     "miou": info["miou"], "miou_boundary": info["miou_boundary"], "miou_core": info["miou_core"],
                     "best_val_miou": info.get("best_val_miou", info.get("val_miou", np.nan)),
                     "n_params": info["n_params"], "train_seconds": info["train_seconds"]})
    idx = np.concatenate(idx_parts)
    if len(np.unique(idx)) != len(idx):
        raise ValueError(f"{tag}: a test patch appears in more than one fold")
    return rows, idx, np.concatenate(fold_parts), {r: np.concatenate(cm_parts[r]) for r in REGIONS}


def summarize_folds(runs_dir, out, name="fusion_unet_5folds", variants=(), pairs=(), splits=FOLDS,
                    label_frac=1.0, seed=0, n_boot=1000, seeds=None):
    """seeds: training seeds to average (default: [seed]); the patch draws use the first one."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    variants, splits = list(variants), [str(s) for s in splits]
    seeds = [int(seed)] if seeds is None else [int(x) for x in seeds]
    draw_seed = seeds[0]
    pairs = [parse_pair(p) if isinstance(p, str) else tuple(p) for p in pairs]

    # 1. pool the folds of every (variant, seed); all must cover the same patches in the same order
    per_fold, cms, ref_idx, fold_of = [], {}, None, None
    for tag in variants:
        cms[tag] = {r: [] for r in REGIONS}
        for sd in seeds:
            rows, idx, folds, pooled = load_pooled(runs_dir, tag, splits, label_frac, sd)
            if ref_idx is None:
                ref_idx, fold_of = idx, folds
            elif not np.array_equal(idx, ref_idx):
                raise ValueError(f"{tag} seed {sd}: pooled test patches differ from those of {variants[0]}")
            for r in REGIONS:
                cms[tag][r].append(pooled[r])
            per_fold += rows
    per_fold = pd.DataFrame(per_fold)
    per_fold.to_csv(out / f"{name}_per_fold.csv", index=False)

    # 2. pooled mIoU (mean over seeds) with the patch-bootstrap interval, and the spread over folds
    summary = []
    for tag in variants:
        sub = per_fold[per_fold.variant == tag]
        row = {"variant": tag, "folds": len(splits), "seeds": len(seeds), "patches": len(ref_idx)}
        for r in REGIONS:
            b = bootstrap_miou_seeds(cms[tag][r], n_boot=n_boot, seed=draw_seed)
            key = "miou" if r == "all" else r
            row.update({f"{key}_pooled": b["miou"], f"{key}_ci_low": b["ci_low"], f"{key}_ci_high": b["ci_high"]})
            if r == "all":
                row["miou_std_seeds"] = b["std_seeds"]
        fold_means = sub.groupby("split").miou.mean()
        row.update({"miou_fold_mean": float(fold_means.mean()),
                    "miou_fold_std": float(fold_means.std(ddof=1)) if len(fold_means) > 1 else 0.0,
                    "n_params": int(sub.n_params.iloc[0]), "train_seconds_total": float(sub.train_seconds.sum())})
        summary.append(row)
    summary = pd.DataFrame(summary)
    summary.to_csv(out / f"{name}_summary.csv", index=False)

    # 3. paired differences on the pooled patches (mean over seeds), and per fold
    deltas = []
    for b, a in pairs:
        if b not in cms or a not in cms:
            raise KeyError(f"pair {b}:{a} uses a variant that is not in the list")
        for r in REGIONS:
            d = paired_delta_seeds(cms[a][r], cms[b][r], n_boot=n_boot, seed=draw_seed)
            by_fold = [float(np.mean([miou(cb[fold_of == s].sum(axis=0)) - miou(ca[fold_of == s].sum(axis=0))
                                      for ca, cb in zip(cms[a][r], cms[b][r])])) for s in splits]
            deltas.append({"variant": b, "versus": a, "region": r, "delta_pooled": d["delta"],
                           "ci_low": d["ci_low"], "ci_high": d["ci_high"], "share_le_0": d["share_le_0"],
                           "folds_positive": int(sum(x > 0 for x in by_fold)),
                           "per_fold": " ".join(f"{x:+.4f}" for x in by_fold),
                           "per_seed": " ".join(f"{x:+.4f}" for x in d["per_seed"])})
    deltas = pd.DataFrame(deltas)
    deltas.to_csv(out / f"{name}_deltas.csv", index=False)

    # 4. pooled IoU per class (mean over seeds)
    fig, ax = plt.subplots(figsize=(13, 4.5))
    width = 0.8 / max(len(variants), 1)
    for k, tag in enumerate(variants):
        iou = np.mean([per_class_iou(c.sum(axis=0)) for c in cms[tag]["all"]], axis=0)
        mean_miou = float(np.mean([miou(c.sum(axis=0)) for c in cms[tag]["all"]]))
        ax.bar(np.arange(len(iou)) + k * width, iou, width, label=f"{tag} (pooled mIoU {mean_miou:.3f})")
    ax.set_xticks(np.arange(len(CLASS_NAMES)) + 0.4 - width / 2, CLASS_NAMES, rotation=60, ha="right", fontsize=8)
    seed_txt = f"seed {seeds[0]}" if len(seeds) == 1 else f"mean of seeds {', '.join(map(str, seeds))}"
    ax.set(ylabel="IoU", ylim=(0, 1),
           title=f"{name}: pooled IoU per class over folds {', '.join(splits)} ({seed_txt}, {label_frac:.0%} labels)")
    ax.legend(fontsize=8, ncol=min(3, max(len(variants), 1)))
    fig.tight_layout()
    fig.savefig(out / f"{name}_per_class.png", dpi=130)
    plt.close(fig)
    return per_fold, summary, deltas


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="fusion_unet_5folds", help="prefix of the output files")
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--pairs", nargs="*", default=[], help="b:a pairs, delta = mIoU(b) - mIoU(a)")
    ap.add_argument("--splits", nargs="+", default=list(FOLDS))
    ap.add_argument("--label-frac", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="+", help="several training seeds to average (instead of --seed)")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    per_fold, summary, deltas = summarize_folds(args.runs_dir, args.out, args.name, args.variants, args.pairs,
                                                args.splits, args.label_frac, args.seed, args.n_boot, args.seeds)
    pd.set_option("display.width", 220)
    print(per_fold.drop(columns="run_id").round(4).to_string(index=False))
    print("\npooled over folds:")
    print(summary.round(4).to_string(index=False))
    if len(deltas):
        print("\npaired bootstrap on the pooled patches, delta = mIoU(variant) - mIoU(versus):")
        print(deltas.round(4).to_string(index=False))
    print(f"\nsaved in {args.out}")


if __name__ == "__main__":
    main()
