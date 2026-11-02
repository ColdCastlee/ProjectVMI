"""Pixel-wise linear heads on frozen embeddings (multinomial logistic regression), over several seeds.

Trains a per-pixel logistic regression head on TESSERA 2019, AlphaEarth 2019,
their concatenation (TESSERA 2019 + AlphaEarth 2019), and a dimensionality
control (TESSERA 2019 + 64 channels of Gaussian noise).

For each seed, all variants share the training pixels (sampled from the valid
pixels of every training patch) and the noise; the seed changes the pixel sample
and the noise. Same class weights (1 / sqrt of the class frequency), per-channel
standardisation fitted on the training pixels, same evaluation on every test
patch (shared valid mask, src/eval). Each (variant, seed) is one run: a line of
runs.csv and a folder with its per-patch confusion matrices.

Seeds are run one after the other and saved as they finish; a seed whose runs
already exist with the same max_iter is skipped, so the script can be started
again after a disconnection. The summary is then computed from the saved runs:
mean over seeds with a patch-bootstrap interval, and the paired difference of
every concatenation with the other variants, averaged over seeds with the same
patch draws.

Variants are written as sources joined by "+", sources being tessera_2019,
tessera_2018, ae_2019, ae_2018 or noise.

Example:
    python scripts/m1_linear_heads.py --processed /content/processed \
        --runs-dir /content/drive/MyDrive/pj_vmi/runs \
        --runs-csv /content/drive/MyDrive/pj_vmi/results/runs.csv \
        --out /content/drive/MyDrive/pj_vmi/results/m1 --seeds 0 1 2
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.embeddings import open_processed
from src.data.splits import load_split, split_tag
from src.eval.bootstrap import bootstrap_miou, bootstrap_miou_seeds, paired_delta_seeds
from src.eval.metrics import CLASS_NAMES, N_CLASSES
from src.eval.runs import RunEvaluator, load_cms, log_run, run_name

SHORT = {"tessera_2019": "t19", "tessera_2018": "t18", "ae_2019": "ae19", "ae_2018": "ae18", "noise": "noise"}
DEFAULT_VARIANTS = ["tessera_2019", "ae_2019", "tessera_2019+ae_2019", "tessera_2019+noise"]
NOISE_DIM = 64


class Sources:
    """Per-patch features (C, H, W) float32; noise is fixed per (seed, patch)."""

    def __init__(self, processed, names):
        self.get = {}
        for name in names:
            if name == "noise":
                continue
            get = open_processed(processed, name)
            if get is None:
                raise SystemExit(f"{name} not found in {processed}")
            self.get[name] = get

    def patch(self, sources, seed, i):
        parts = [np.random.default_rng([seed, int(i)]).standard_normal((NOISE_DIM, 128, 128), dtype=np.float32)
                 if s == "noise" else self.get[s](i) for s in sources]
        return np.concatenate(parts)


def class_weights(y):
    counts = np.bincount(y, minlength=N_CLASSES)
    w = np.zeros(N_CLASSES)
    w[counts > 0] = 1 / np.sqrt(counts[counts > 0] / counts.sum())
    sw = w[y]
    return sw / sw.mean()


def run_seed(seed, args, variants, tags, src, labels, instances, valid, train, test, names):
    """Train and evaluate every variant for one seed; the runs are saved as they finish."""
    rng = np.random.default_rng(seed)
    per_patch = int(np.ceil(args.train_pixels / len(train)))
    feats, ys = {n: [] for n in names}, []
    for k, i in enumerate(train):
        rows, cols = np.nonzero(valid[i])
        if len(rows) == 0:
            continue
        pick = rng.choice(len(rows), size=min(per_patch, len(rows)), replace=False)
        r, c = rows[pick], cols[pick]
        for n in names:
            feats[n].append(src.patch([n], seed, i)[:, r, c].T)
        ys.append(np.asarray(labels[i])[r, c])
        if (k + 1) % 500 == 0:
            print(f"  seed {seed}: {k + 1}/{len(train)} training patches read", flush=True)
    feats = {n: np.concatenate(v) for n, v in feats.items()}
    y = np.concatenate(ys).astype(np.int64)
    sw = class_weights(y)

    models = {}
    for v, tag in zip(variants, tags):
        x = np.hstack([feats[n] for n in v])
        scaler = StandardScaler().fit(x)
        clf = LogisticRegression(max_iter=args.max_iter)
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            clf.fit(scaler.transform(x), y, sample_weight=sw)
        seconds = time.time() - t0
        converged = bool(clf.n_iter_[0] < args.max_iter)
        models[tag] = (v, scaler, clf, seconds, converged)
        print(f"seed {seed} {tag}: {x.shape[1]} features, {int(clf.n_iter_[0])} iterations "
              f"({'converged' if converged else 'max_iter reached'}), {seconds:.0f} s", flush=True)
    del feats

    evaluators = {tag: RunEvaluator(labels, instances, valid) for tag in tags}
    for k, i in enumerate(test):
        cache = {n: src.patch([n], seed, i) for n in names}
        for tag, (v, scaler, clf, _, _) in models.items():
            x = np.concatenate([cache[n] for n in v]).reshape(-1, 128 * 128).T
            x = np.nan_to_num(scaler.transform(x))  # NaN only at no-data pixels, outside the valid mask
            evaluators[tag].add(i, clf.predict(x).reshape(128, 128))
        if (k + 1) % 250 == 0:
            print(f"  seed {seed}: {k + 1}/{len(test)} test patches predicted", flush=True)

    tag_split = split_tag(args.split)
    for tag, (v, scaler, clf, seconds, converged) in models.items():
        idx, cm = evaluators[tag].result()
        run = {"run_id": run_name("lin", tag, 1.0, tag_split, seed),
               "method": "baseline" if len(v) == 1 else "m1", "variant": tag, "label_frac": 1.0,
               "split": tag_split, "seed": seed, "n_params": int(clf.coef_.size + clf.intercept_.size),
               "train_seconds": round(seconds, 1)}
        log_run(args.runs_csv, Path(args.runs_dir) / run["run_id"], run, idx, cm,
                extra={"head": "logistic regression", "sources": v, "train_pixels": int(len(y)),
                       "max_iter": args.max_iter, "n_iter": int(clf.n_iter_[0]), "converged": converged,
                       "class_weights": "1/sqrt(freq)"})


def run_dir(args, tag, seed):
    return Path(args.runs_dir) / run_name("lin", tag, 1.0, split_tag(args.split), seed)


def finished(args, tag, seed):
    d = run_dir(args, tag, seed)
    if not ((d / "run.json").exists() and (d / "cm_per_patch.npz").exists()):
        return False
    return json.loads((d / "run.json").read_text()).get("max_iter") == args.max_iter


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed", required=True, help="directory with the arrays (copy them to /content first)")
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--runs-csv", required=True)
    ap.add_argument("--out", required=True, help="directory for the summary tables and the figure")
    ap.add_argument("--split", default="I")
    ap.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--train-pixels", type=int, default=300_000)
    ap.add_argument("--max-iter", type=int, default=1000)
    ap.add_argument("--rerun", action="store_true", help="train again seeds that are already finished")
    args = ap.parse_args()

    processed, out = Path(args.processed), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    variants = [v.split("+") for v in args.variants]
    names = sorted({s for v in variants for s in v})
    unknown = [n for n in names if n not in SHORT]
    if unknown:
        raise SystemExit(f"unknown sources {unknown}; use {list(SHORT)}")
    tags = ["-".join(SHORT[s] for s in v) for v in variants]
    labels = np.load(processed / "labels.npy", mmap_mode="r")
    instances = np.load(processed / "instances.npy", mmap_mode="r")
    valid = np.load(processed / "valid_mask.npy", mmap_mode="r")
    part = load_split(processed, args.split)
    train, test = np.asarray(part["train"]), np.asarray(part["test"])
    src = Sources(processed, names)

    # 1. one seed after the other, saved as it finishes
    for seed in args.seeds:
        if not args.rerun and all(finished(args, tag, seed) for tag in tags):
            print(f"seed {seed}: already finished, skipped")
            continue
        run_seed(seed, args, variants, tags, src, labels, instances, valid, train, test, names)

    # 2. summary from the saved runs
    cms, per_seed = {}, []
    for tag in tags:
        for seed in args.seeds:
            d = run_dir(args, tag, seed)
            info = json.loads((d / "run.json").read_text())
            _, cm = load_cms(d)
            cms.setdefault(tag, []).append(cm)
            boot = bootstrap_miou(cm, seed=seed)
            per_seed.append({"variant": tag, "seed": seed, "miou": info["miou"], "ci_low": boot["ci_low"],
                             "ci_high": boot["ci_high"], "oa": info["oa"], "macro_f1": info["macro_f1"],
                             "miou_boundary": info["miou_boundary"], "miou_core": info["miou_core"],
                             "n_iter": info["n_iter"], "converged": info["converged"],
                             "train_seconds": info["train_seconds"],
                             **{f"iou_{c}": info[f"iou_{c}"] for c in range(N_CLASSES)}})
    per_seed = pd.DataFrame(per_seed)
    per_seed.to_csv(out / "linear_heads_per_seed.csv", index=False)

    summary = []
    for tag in tags:
        agg = bootstrap_miou_seeds(cms[tag])
        sub = per_seed[per_seed.variant == tag]
        summary.append({"variant": tag, "seeds": len(sub), "miou_mean": agg["miou"], "miou_std": agg["std_seeds"],
                        "ci_low": agg["ci_low"], "ci_high": agg["ci_high"], "oa_mean": sub.oa.mean(),
                        "macro_f1_mean": sub.macro_f1.mean(), "boundary_mean": sub.miou_boundary.mean(),
                        "core_mean": sub.miou_core.mean(), "all_converged": bool(sub.converged.all())})
    summary = pd.DataFrame(summary)
    summary.to_csv(out / "linear_heads_summary.csv", index=False)

    deltas = []
    for b, vb in zip(tags, variants):
        if len(vb) == 1:
            continue
        for a in tags:
            if a != b:
                d = paired_delta_seeds(cms[a], cms[b])
                deltas.append({"variant": b, "versus": a, "delta_mean": d["delta"], "ci_low": d["ci_low"],
                               "ci_high": d["ci_high"], "share_le_0": d["share_le_0"],
                               "per_seed": " ".join(f"{x:+.4f}" for x in d["per_seed"])})
    deltas = pd.DataFrame(deltas)
    deltas.to_csv(out / "linear_heads_deltas.csv", index=False)

    # 3. IoU per class, mean and spread over seeds
    iou_cols = [f"iou_{c}" for c in range(N_CLASSES)]
    fig, ax = plt.subplots(figsize=(13, 4.5))
    width = 0.8 / len(tags)
    for k, tag in enumerate(tags):
        sub = per_seed[per_seed.variant == tag][iou_cols]
        mean = summary.loc[summary.variant == tag, "miou_mean"].item()
        ax.bar(np.arange(N_CLASSES) + k * width, sub.mean().values, width, yerr=sub.std().fillna(0).values,
               capsize=1.5, label=f"{tag} (mIoU {mean:.3f})")
    ax.set_xticks(np.arange(N_CLASSES) + 0.4 - width / 2, CLASS_NAMES, rotation=60, ha="right", fontsize=8)
    ax.set(ylabel="IoU", ylim=(0, 1),
           title=f"Linear heads, split {args.split}: IoU per class (mean and std over {len(args.seeds)} seeds)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "linear_heads_per_class.png", dpi=130)

    pd.set_option("display.width", 220)
    print(per_seed[["variant", "seed", "miou", "ci_low", "ci_high", "n_iter", "converged", "train_seconds"]]
          .round(4).to_string(index=False))
    print("\nmean over seeds:")
    print(summary.round(4).to_string(index=False))
    if len(deltas):
        print("\npaired bootstrap, delta = mIoU(variant) - mIoU(versus), averaged over seeds:")
        print(deltas.round(4).to_string(index=False))
    print(f"\nsaved in {out}")


if __name__ == "__main__":
    main()
