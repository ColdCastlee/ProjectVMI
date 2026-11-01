"""Method 1: how redundant are TESSERA and AlphaEarth? (plan, section 6: CKA and CCA on about 100,000 train pixels)

Samples valid pixels from the training patches of a split (the same pixels for
every embedding) and reports:
- for each pair of embeddings: linear CKA, canonical correlations (mean, first,
  number above 0.9 and 0.5), and the held-out linear R² in both directions
  (patches split 80/20, so that no patch is in both parts);
- for each embedding, and for the concatenation TESSERA 2019 + AlphaEarth 2019
  (each block scaled to unit total variance), the number of principal
  components for 90% and 95% of the variance.
Everything is computed on all valid pixels and on crop pixels only (labels 1-18).

Outputs in --out: m1_redundancy_pairs.csv, m1_redundancy_dims.csv,
m1_redundancy.png, m1_redundancy_info.json.

Example:
    python scripts/m1_redundancy.py --processed /content/drive/MyDrive/pj_vmi/processed \
        --out /content/drive/MyDrive/pj_vmi/results/m1
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.analysis.similarity import (canonical_correlations, center, dims_for,  # noqa: E402
                                     explained_variance, linear_cka, ridge_r2)
from src.data.embeddings import open_processed  # noqa: E402
from src.data.shards import code_version  # noqa: E402
from src.data.splits import load_split  # noqa: E402

NAMES = ["tessera_2019", "ae_2019", "tessera_2018", "ae_2018"]
PAIRS = [("tessera_2019", "ae_2019"), ("tessera_2018", "ae_2018"),
         ("tessera_2018", "tessera_2019"), ("ae_2018", "ae_2019")]
CONCAT = ("tessera_2019", "ae_2019")


def sample_pixels(processed, split, n_patches, per_patch, seed):
    labels = np.load(processed / "labels.npy", mmap_mode="r")
    valid = np.load(processed / "valid_mask.npy", mmap_mode="r")
    getters = {n: open_processed(processed, n) for n in NAMES}
    missing = [n for n, g in getters.items() if g is None]
    if missing:
        raise SystemExit(f"missing embeddings in {processed}: {missing}")

    rng = np.random.default_rng(seed)
    train = np.asarray(load_split(processed, split)["train"])
    patches = np.sort(rng.choice(train, size=min(n_patches, len(train)), replace=False))
    feats, lab, pid = {n: [] for n in NAMES}, [], []
    for k, i in enumerate(patches):
        rows, cols = np.nonzero(valid[i])
        pick = rng.choice(len(rows), size=min(per_patch, len(rows)), replace=False)
        r, c = rows[pick], cols[pick]
        for n, get in getters.items():
            feats[n].append(get(i)[:, r, c].T)
        lab.append(np.asarray(labels[i])[r, c])
        pid.append(np.full(len(pick), i))
        if (k + 1) % 50 == 0:
            print(f"  {k + 1}/{len(patches)} patches read", flush=True)
    return {n: np.concatenate(v) for n, v in feats.items()}, np.concatenate(lab), np.concatenate(pid)


def unit_variance(x):
    x = center(x)
    return x / np.sqrt((x ** 2).sum(axis=1).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="I", help="pixels are drawn from the training patches of this split")
    ap.add_argument("--n-patches", type=int, default=300)
    ap.add_argument("--pixels-per-patch", type=int, default=340)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    processed, out = Path(args.processed), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    feats, lab, pid = sample_pixels(processed, args.split, args.n_patches, args.pixels_per_patch, args.seed)
    crops = (lab >= 1) & (lab <= 18)
    print(f"{len(lab):,} pixels from {len(np.unique(pid))} train patches of split {args.split}; "
          f"crop pixels: {crops.mean():.1%}")

    rng = np.random.default_rng(args.seed + 1)
    patches = np.unique(pid)
    held_out = np.isin(pid, rng.choice(patches, size=len(patches) // 5, replace=False))

    pair_rows, dim_rows, cca_curves, ev_curves = [], [], {}, {}
    for subset, mask in [("all", np.ones(len(lab), bool)), ("crops", crops)]:
        fit, test = mask & ~held_out, mask & held_out
        for a, b in PAIRS:
            cc = canonical_correlations(feats[a][mask], feats[b][mask])
            pair_rows.append({
                "subset": subset, "a": a, "b": b, "pixels": int(mask.sum()),
                "cka": linear_cka(feats[a][mask], feats[b][mask]),
                "cca_mean": float(cc.mean()), "cca_first": float(cc[0]),
                "cca_n_above_0.9": int((cc > 0.9).sum()), "cca_n_above_0.5": int((cc > 0.5).sum()),
                "r2_b_from_a": ridge_r2(feats[a][fit], feats[b][fit], feats[a][test], feats[b][test]),
                "r2_a_from_b": ridge_r2(feats[b][fit], feats[a][fit], feats[b][test], feats[a][test]),
            })
            if subset == "all":
                cca_curves[f"{a} / {b}"] = cc
        blocks = {n: feats[n][mask] for n in NAMES}
        blocks["+".join(CONCAT)] = np.hstack([unit_variance(feats[n][mask]) for n in CONCAT])
        for name, x in blocks.items():
            ratios = explained_variance(x)
            dim_rows.append({"subset": subset, "embedding": name, "dim": x.shape[1],
                             "share_first_3": float(ratios[:3].sum()),
                             "k_90": dims_for(ratios, 0.90), "k_95": dims_for(ratios, 0.95)})
            if subset == "all":
                ev_curves[name] = np.cumsum(ratios)

    pairs, dims = pd.DataFrame(pair_rows), pd.DataFrame(dim_rows)
    pairs.to_csv(out / "m1_redundancy_pairs.csv", index=False)
    dims.to_csv(out / "m1_redundancy_dims.csv", index=False)
    info = {"split": args.split, "pixels": int(len(lab)), "patches": int(len(patches)),
            "held_out_patches": int(len(patches) // 5), "seed": args.seed, "code_commit": code_version()}
    (out / "m1_redundancy_info.json").write_text(json.dumps(info, indent=2))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for name, cc in cca_curves.items():
        ax1.plot(np.arange(1, len(cc) + 1), cc, label=name)
    ax1.set(xlabel="canonical direction", ylabel="canonical correlation", ylim=(0, 1.02),
            title="Shared directions (all valid pixels)")
    ax1.legend(fontsize=8)
    for name, cum in ev_curves.items():
        ax2.plot(np.arange(1, len(cum) + 1), cum, label=name)
    ax2.axhline(0.95, color="grey", lw=0.5, ls="--")
    ax2.set(xscale="log", xlabel="principal components", ylabel="cumulative variance", ylim=(0, 1.02),
            title="Intrinsic dimension")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "m1_redundancy.png", dpi=130)

    pd.set_option("display.width", 200)
    print(pairs.round(3).to_string(index=False))
    print()
    print(dims.round(3).to_string(index=False))
    print(f"\nsaved in {out}")


if __name__ == "__main__":
    main()
