"""Summary tables, paired bootstrap differences and per-class IoU plot from saved runs.

Reads <runs_dir>/<run_id>/run.json and cm_per_patch.npz for every (variant, seed)
of a split and label fraction. Variants are written with their prefix, e.g.
lin_t19, base_mlp-t19, base_unet-t19-w32.

Outputs in --out: <name>_per_seed.csv, <name>_summary.csv, <name>_deltas.csv
(all valid pixels, parcel boundary and parcel core) and <name>_per_class.png.

Example:
    python scripts/summarize_runs.py --runs-dir /content/drive/MyDrive/pj_vmi/runs \
        --out /content/drive/MyDrive/pj_vmi/results/baselines --name baselines
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
from src.data.splits import split_tag
from src.eval.bootstrap import bootstrap_miou, bootstrap_miou_seeds, paired_delta_seeds
from src.eval.metrics import CLASS_NAMES, N_CLASSES
from src.eval.runs import load_cms, run_name

REGIONS = ("all", "boundary", "core")
DEFAULT_VARIANTS = [
    "lin_t19", "base_mlp-t19", "base_unet-t19-w32",
    "lin_ae19", "base_mlp-ae19", "base_unet-ae19-w32",
]
DEFAULT_PAIRS = [
    "base_unet-t19-w32:lin_t19",
    "base_unet-t19-w32:base_mlp-t19",
    "base_mlp-t19:lin_t19",
    "base_unet-ae19-w32:lin_ae19",
    "base_unet-ae19-w32:base_mlp-ae19",
    "base_mlp-ae19:lin_ae19",
    "base_unet-t19-w32:base_unet-ae19-w32",
]


def run_id_of(tag, label_frac, split, seed):
    if "_" not in tag:
        raise ValueError(f"expected <prefix>_<variant>, got {tag!r}")
    prefix, variant = tag.split("_", 1)
    return run_name(prefix, variant, label_frac, split_tag(split), seed)


def parse_pair(spec):
    if ":" not in spec:
        raise ValueError(f"expected <variant>:<versus>, got {spec!r}")
    return tuple(spec.split(":", 1))


def summarize_runs(runs_dir, out, name="baselines", variants=DEFAULT_VARIANTS, pairs=DEFAULT_PAIRS,
                   split="I", label_frac=1.0, seeds=(0, 1, 2), n_boot=1000):
    runs_dir, out = Path(runs_dir), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    variants = list(variants)
    pairs = [parse_pair(p) if isinstance(p, str) else tuple(p) for p in pairs]
    seeds = [int(s) for s in seeds]

    # 1. read each (variant, seed) and check that all runs use the same test patches
    cms = {tag: {r: [] for r in REGIONS} for tag in variants}
    ref_idx, per_seed = None, []
    for tag in variants:
        for seed in seeds:
            rid = run_id_of(tag, label_frac, split, seed)
            d = runs_dir / rid
            if not (d / "run.json").exists():
                raise FileNotFoundError(f"missing run {d / 'run.json'}")
            info = json.loads((d / "run.json").read_text())
            for r in REGIONS:
                idx, cm = load_cms(d, r)
                if ref_idx is None:
                    ref_idx = idx
                elif not np.array_equal(idx, ref_idx):
                    raise ValueError(f"{rid}: test patches differ from the first run")
                cms[tag][r].append(cm)
            boot = bootstrap_miou(cms[tag]["all"][-1], n_boot=n_boot, seed=seed)
            per_seed.append({
                "variant": tag, "seed": seed, "run_id": rid,
                "miou": info["miou"], "ci_low": boot["ci_low"], "ci_high": boot["ci_high"],
                "oa": info["oa"], "macro_f1": info["macro_f1"],
                "miou_boundary": info["miou_boundary"], "miou_core": info["miou_core"],
                "best_val_miou": info.get("best_val_miou", np.nan),
                "best_step": info.get("best_step", np.nan),
                "n_params": info["n_params"], "train_seconds": info["train_seconds"],
                **{f"iou_{c}": info[f"iou_{c}"] for c in range(N_CLASSES)},
            })
    per_seed = pd.DataFrame(per_seed)
    per_seed.to_csv(out / f"{name}_per_seed.csv", index=False)

    # 2. mean and patch-bootstrap interval over seeds
    summary = []
    for tag in variants:
        agg_all = bootstrap_miou_seeds(cms[tag]["all"], n_boot=n_boot)
        agg_bnd = bootstrap_miou_seeds(cms[tag]["boundary"], n_boot=n_boot)
        agg_cor = bootstrap_miou_seeds(cms[tag]["core"], n_boot=n_boot)
        sub = per_seed[per_seed.variant == tag]
        summary.append({
            "variant": tag, "seeds": len(sub),
            "miou_mean": agg_all["miou"], "miou_std": agg_all["std_seeds"],
            "ci_low": agg_all["ci_low"], "ci_high": agg_all["ci_high"],
            "oa_mean": float(sub.oa.mean()), "macro_f1_mean": float(sub.macro_f1.mean()),
            "boundary_mean": agg_bnd["miou"], "boundary_ci_low": agg_bnd["ci_low"],
            "boundary_ci_high": agg_bnd["ci_high"],
            "core_mean": agg_cor["miou"], "core_ci_low": agg_cor["ci_low"], "core_ci_high": agg_cor["ci_high"],
            "val_miou_mean": float(sub.best_val_miou.mean()),
            "n_params": int(sub.n_params.iloc[0]),
            "train_seconds_mean": float(sub.train_seconds.mean()),
        })
    summary = pd.DataFrame(summary)
    summary.to_csv(out / f"{name}_summary.csv", index=False)

    # 3. paired bootstrap differences on all valid pixels, boundaries and cores
    deltas = []
    for b, a in pairs:
        if b not in cms or a not in cms:
            continue
        for r in REGIONS:
            d = paired_delta_seeds(cms[a][r], cms[b][r], n_boot=n_boot)
            deltas.append({
                "variant": b, "versus": a, "region": r,
                "delta_mean": d["delta"], "ci_low": d["ci_low"], "ci_high": d["ci_high"],
                "share_le_0": d["share_le_0"],
                "per_seed": " ".join(f"{x:+.4f}" for x in d["per_seed"]),
            })
    deltas = pd.DataFrame(deltas)
    deltas.to_csv(out / f"{name}_deltas.csv", index=False)

    # 4. IoU per class, mean and spread over seeds
    iou_cols = [f"iou_{c}" for c in range(N_CLASSES)]
    fig, ax = plt.subplots(figsize=(13, 4.5))
    width = 0.8 / len(variants)
    for k, tag in enumerate(variants):
        sub = per_seed[per_seed.variant == tag][iou_cols]
        mean = summary.loc[summary.variant == tag, "miou_mean"].item()
        yerr = sub.std(ddof=1).fillna(0).values if len(sub) > 1 else np.zeros(N_CLASSES)
        ax.bar(np.arange(N_CLASSES) + k * width, sub.mean().values, width, yerr=yerr,
               capsize=1.5, label=f"{tag} (mIoU {mean:.3f})")
    ax.set_xticks(np.arange(N_CLASSES) + 0.4 - width / 2, CLASS_NAMES, rotation=60, ha="right", fontsize=8)
    ax.set(ylabel="IoU", ylim=(0, 1),
           title=f"{name}, split {split} ({label_frac:.0%} labels): IoU per class (mean and std over {len(seeds)} seeds)")
    ax.legend(fontsize=8, ncol=min(3, len(variants)))
    fig.tight_layout()
    fig.savefig(out / f"{name}_per_class.png", dpi=130)
    plt.close(fig)
    return per_seed, summary, deltas


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="baselines", help="prefix of the output files")
    ap.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS)
    ap.add_argument("--pairs", nargs="*", default=DEFAULT_PAIRS, help="b:a pairs for paired_delta_seeds")
    ap.add_argument("--split", default="I")
    ap.add_argument("--label-frac", type=float, default=1.0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    per_seed, summary, deltas = summarize_runs(
        args.runs_dir, args.out, args.name, args.variants, args.pairs,
        args.split, args.label_frac, args.seeds, args.n_boot,
    )
    pd.set_option("display.width", 220)
    cols = ["variant", "seed", "miou", "ci_low", "ci_high", "miou_boundary", "miou_core",
            "best_val_miou", "n_params", "train_seconds"]
    print(per_seed[cols].round(4).to_string(index=False))
    print("\nmean over seeds:")
    sum_cols = ["variant", "seeds", "miou_mean", "miou_std", "ci_low", "ci_high", "oa_mean",
                "macro_f1_mean", "boundary_mean", "core_mean", "val_miou_mean", "n_params"]
    print(summary[sum_cols].round(4).to_string(index=False))
    if len(deltas):
        print("\npaired bootstrap, delta = mIoU(variant) - mIoU(versus), averaged over seeds:")
        print(deltas.round(4).to_string(index=False))
    print(f"\nsaved in {args.out}")


if __name__ == "__main__":
    main()
