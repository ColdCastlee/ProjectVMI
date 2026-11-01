"""Pixel-wise linear heads on frozen embeddings (multinomial logistic regression).

Plan, section 5: B0 = TESSERA 2019 and B3 = AlphaEarth 2019 with a logistic
regression per pixel. The same head is also trained on TESSERA 2019 + AlphaEarth
2019 (concatenation) and on TESSERA 2019 + 64 channels of Gaussian noise (control:
does a gain come only from having more dimensions?). This is a first, linear look
at hypothesis H1a; the main Method 1 variants use the UNet head.

All variants share the training pixels (sampled from the valid pixels of every
training patch), the class weights (1 / sqrt of the class frequency), a
per-channel standardisation fitted on the training pixels, and the evaluation on
every test patch (shared valid mask, src/eval). Each variant is one line of
runs.csv; the paired bootstrap compares every concatenation with the other
variants on the same test patches.

Variants are written as sources joined by "+", sources being tessera_2019,
tessera_2018, ae_2019, ae_2018 or noise.

Example:
    python scripts/m1_linear_heads.py --processed /content/processed \
        --runs-dir /content/drive/MyDrive/pj_vmi/runs \
        --runs-csv /content/drive/MyDrive/pj_vmi/results/runs.csv \
        --out /content/drive/MyDrive/pj_vmi/results/m1
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.embeddings import open_processed  # noqa: E402
from src.data.splits import load_split, split_tag  # noqa: E402
from src.eval.bootstrap import bootstrap_miou, paired_delta  # noqa: E402
from src.eval.metrics import CLASS_NAMES, N_CLASSES  # noqa: E402
from src.eval.runs import RunEvaluator, log_run, run_name  # noqa: E402

SHORT = {"tessera_2019": "t19", "tessera_2018": "t18", "ae_2019": "ae19", "ae_2018": "ae18", "noise": "noise"}
DEFAULT_VARIANTS = ["tessera_2019", "ae_2019", "tessera_2019+ae_2019", "tessera_2019+noise"]
NOISE_DIM = 64


class Sources:
    """Per-patch features (C, H, W) float32 of each source; noise is fixed per patch and seed."""

    def __init__(self, processed, names, seed):
        self.seed = seed
        self.get = {}
        for name in names:
            if name == "noise":
                continue
            get = open_processed(processed, name)
            if get is None:
                raise SystemExit(f"{name} not found in {processed}")
            self.get[name] = get

    def patch(self, name, i):
        if name == "noise":
            rng = np.random.default_rng([self.seed, int(i)])
            return rng.standard_normal((NOISE_DIM, 128, 128), dtype=np.float32)
        return self.get[name](i)


def sample_train_pixels(src, names, labels, valid, train, n_pixels, rng):
    per_patch = int(np.ceil(n_pixels / len(train)))
    feats, y = {n: [] for n in names}, []
    for k, i in enumerate(train):
        rows, cols = np.nonzero(valid[i])
        if len(rows) == 0:
            continue
        pick = rng.choice(len(rows), size=min(per_patch, len(rows)), replace=False)
        r, c = rows[pick], cols[pick]
        for n in names:
            feats[n].append(src.patch(n, i)[:, r, c].T)
        y.append(np.asarray(labels[i])[r, c])
        if (k + 1) % 250 == 0:
            print(f"  {k + 1}/{len(train)} training patches read", flush=True)
    return {n: np.concatenate(v) for n, v in feats.items()}, np.concatenate(y).astype(np.int64)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed", required=True, help="directory with the arrays (copy them to /content first)")
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--runs-csv", required=True)
    ap.add_argument("--out", required=True, help="directory for the summary tables and the figure")
    ap.add_argument("--split", default="I")
    ap.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS)
    ap.add_argument("--train-pixels", type=int, default=300_000)
    ap.add_argument("--max-iter", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    processed, out = Path(args.processed), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    variants = [v.split("+") for v in args.variants]
    names = sorted({s for v in variants for s in v})
    unknown = [n for n in names if n not in SHORT]
    if unknown:
        raise SystemExit(f"unknown sources {unknown}; use {list(SHORT)}")
    labels = np.load(processed / "labels.npy", mmap_mode="r")
    instances = np.load(processed / "instances.npy", mmap_mode="r")
    valid = np.load(processed / "valid_mask.npy", mmap_mode="r")
    part = load_split(processed, args.split)
    train, test = np.asarray(part["train"]), np.asarray(part["test"])
    src = Sources(processed, names, args.seed)

    # 1. training pixels, shared by every variant
    rng = np.random.default_rng(args.seed)
    feats, y = sample_train_pixels(src, names, labels, valid, train, args.train_pixels, rng)
    counts = np.bincount(y, minlength=N_CLASSES)
    weights = np.zeros(N_CLASSES)
    weights[counts > 0] = 1 / np.sqrt(counts[counts > 0] / counts.sum())
    sample_weight = weights[y] / weights[y].mean()
    print(f"{len(y):,} training pixels from {len(train)} patches; classes present: {int((counts > 0).sum())}")

    # 2. one standardised logistic regression per variant
    models = {}
    for v in variants:
        tag = "-".join(SHORT[s] for s in v)
        x = np.hstack([feats[s] for s in v])
        scaler = StandardScaler().fit(x)
        clf = LogisticRegression(max_iter=args.max_iter)
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            clf.fit(scaler.transform(x), y, sample_weight=sample_weight)
        seconds = time.time() - t0
        models[tag] = (v, scaler, clf, seconds)
        print(f"{tag}: {x.shape[1]} features, {int(clf.n_iter_[0])} iterations "
              f"({'converged' if clf.n_iter_[0] < args.max_iter else 'max_iter reached'}), {seconds:.0f} s")

    # 3. prediction and evaluation on every test patch
    evaluators = {tag: RunEvaluator(labels, instances, valid) for tag in models}
    for k, i in enumerate(test):
        cache = {n: src.patch(n, i) for n in names}
        for tag, (v, scaler, clf, _) in models.items():
            x = np.concatenate([cache[s] for s in v]).reshape(-1, 128 * 128).T
            x = np.nan_to_num(scaler.transform(x))  # NaN only at no-data pixels, outside the valid mask
            evaluators[tag].add(i, clf.predict(x).reshape(128, 128))
        if (k + 1) % 100 == 0:
            print(f"  {k + 1}/{len(test)} test patches predicted", flush=True)

    # 4. runs.csv, bootstrap intervals, paired differences
    tag_split = split_tag(args.split)
    cms, rows = {}, []
    for tag, (v, scaler, clf, seconds) in models.items():
        idx, cm = evaluators[tag].result()
        cms[tag] = cm[:, 0]
        run = {"run_id": run_name("lin", tag, 1.0, tag_split, args.seed),
               "method": "baseline" if len(v) == 1 else "m1", "variant": tag, "label_frac": 1.0,
               "split": tag_split, "seed": args.seed, "n_params": int(clf.coef_.size + clf.intercept_.size),
               "train_seconds": round(seconds, 1)}
        row = log_run(args.runs_csv, Path(args.runs_dir) / run["run_id"], run, idx, cm,
                      extra={"head": "logistic regression", "sources": v, "train_pixels": int(len(y)),
                             "n_iter": int(clf.n_iter_[0]), "class_weights": "1/sqrt(freq)"})
        boot = bootstrap_miou(cm[:, 0], seed=args.seed)
        rows.append({"variant": tag, "features": int(clf.coef_.shape[1]), "miou": row["miou"],
                     "ci_low": boot["ci_low"], "ci_high": boot["ci_high"], "oa": row["oa"],
                     "macro_f1": row["macro_f1"], "miou_boundary": row["miou_boundary"],
                     "miou_core": row["miou_core"], "train_seconds": round(seconds, 1),
                     **{f"iou_{k}": row[f"iou_{k}"] for k in range(N_CLASSES)}})
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "linear_heads_summary.csv", index=False)

    deltas = []
    for b in models:
        if "-" not in b:
            continue
        for a in models:
            if a != b:
                d = paired_delta(cms[a], cms[b], seed=args.seed)
                deltas.append({"variant": b, "versus": a, **d})
    deltas = pd.DataFrame(deltas)
    deltas.to_csv(out / "linear_heads_deltas.csv", index=False)

    # 5. IoU per class
    fig, ax = plt.subplots(figsize=(13, 4.5))
    width = 0.8 / len(rows)
    for k, r in enumerate(rows):
        ax.bar(np.arange(N_CLASSES) + k * width, [r[f"iou_{c}"] for c in range(N_CLASSES)], width,
               label=f"{r['variant']} (mIoU {r['miou']:.3f})")
    ax.set_xticks(np.arange(N_CLASSES) + 0.4 - width / 2, CLASS_NAMES, rotation=60, ha="right", fontsize=8)
    ax.set(ylabel="IoU", ylim=(0, 1), title=f"Linear heads, split {args.split}: IoU per class")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "linear_heads_per_class.png", dpi=130)

    pd.set_option("display.width", 200)
    cols = ["variant", "features", "miou", "ci_low", "ci_high", "oa", "macro_f1", "miou_boundary", "miou_core",
            "train_seconds"]
    print(summary[cols].round(4).to_string(index=False))
    if len(deltas):
        print("\npaired bootstrap, delta = mIoU(variant) - mIoU(versus):")
        print(deltas.round(4).to_string(index=False))
    print(f"\nsaved in {out}")


if __name__ == "__main__":
    main()
