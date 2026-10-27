"""Quality check of the aligned data (plan, section 3.6).

1. Table: summary of qa_<name>.csv and meta_qa.csv with one ok_<name> flag per
   embedding: no error, no-data <= --max-nodata, and no suspect shift. A shift
   is suspect when moving the embedding by one pixel lowers the within-parcel
   variance ratio by more than --max-shift-gain (relative). Sub-pixel offsets,
   which are systematic per tile (see subpixel_offset.py), give small gains and
   are not flagged.
2. Figures: for n patches per tile, Sentinel-2 RGB (clearest summer-2019 date),
   labels, and a 3-component PCA of each embedding, with parcel boundaries on
   top; plus the patches with the largest shift gain. One PCA is fitted per
   embedding on the pixels of all shown patches, so colours are comparable.

Example:
    python scripts/qa_alignment.py --pastis /content/drive/MyDrive/pj_vmi/pastis_subset \
        --processed /content/processed --out results/qa
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
from matplotlib.colors import ListedColormap  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.embeddings import open_processed  # noqa: E402
from src.data.pastis_zip import open_pastis  # noqa: E402
from src.data.selection import tile_sample  # noqa: E402

DEFAULT_NAMES = ["tessera_2019", "tessera_2018", "ae_2019", "ae_2018"]


def boundaries(inst):
    b = np.zeros(inst.shape, dtype=bool)
    b[:, :-1] |= inst[:, :-1] != inst[:, 1:]
    b[:-1, :] |= inst[:-1, :] != inst[1:, :]
    return b


def pca_rgb(embs, seed):
    flat = [e.reshape(e.shape[0], -1).T for e in embs]
    pool = np.concatenate(flat)
    pool = pool[np.isfinite(pool).all(axis=1)]
    rng = np.random.default_rng(seed)
    fit = pool[rng.choice(len(pool), size=min(50_000, len(pool)), replace=False)]
    pca = PCA(n_components=3, random_state=seed).fit(fit)
    lo, hi = np.percentile(pca.transform(fit), [2, 98], axis=0)
    out = []
    for e, f in zip(embs, flat):
        p = pca.transform(np.nan_to_num(f)).reshape(e.shape[1], e.shape[2], 3)
        out.append(np.clip((p - lo) / (hi - lo), 0, 1))
    return out, pca.explained_variance_ratio_


S2_NODATA = -10000  # THEIA fill value; whole dates can be filled with it


def s2_rgb(pz, pid, dates):
    x = pz.npy(f"DATA_S2/S2_{pid}.npy")  # (T, 10, H, W) int16: B2 B3 B4 B5 B6 B7 B8 B8A B11 B12
    d = np.array(dates)
    complete = ~(x == S2_NODATA).reshape(d.size, -1).any(axis=1)
    summer = np.where(complete & (d >= 20190601) & (d <= 20190831))[0]
    cand = summer if summer.size else np.where(complete)[0]
    if cand.size == 0:
        cand = np.arange(d.size)
    t = cand[np.argmin(x[cand, 0].reshape(cand.size, -1).mean(axis=1))]  # darkest blue = fewest clouds
    rgb = x[t, [2, 1, 0]].transpose(1, 2, 0).astype(np.float32)
    lo, hi = np.percentile(rgb, [2, 98])
    valid_min = int(x[x != S2_NODATA].min())
    return np.clip((rgb - lo) / (hi - lo + 1e-6), 0, 1), int(d[t]), valid_min


def label_cmap():
    colors = plt.get_cmap("tab20")(np.linspace(0, 1, 20))
    colors[0] = (0.85, 0.85, 0.85, 1)  # background
    colors[19] = (0, 0, 0, 1)  # void
    return ListedColormap(colors)


def plot_patches(ids, meta, labels, instances, embs, pz, dates, seed, title, path):
    rows = meta.index[meta.ID_PATCH.isin(ids)].tolist()
    names = list(embs)
    pcs = {name: pca_rgb([embs[name](i) for i in rows], seed) for name in names}
    ncol = 2 + len(names)
    fig, axes = plt.subplots(len(rows), ncol, figsize=(2.6 * ncol, 2.6 * len(rows)), squeeze=False)
    s2_mins = []
    for r, i in enumerate(rows):
        pid = int(meta.loc[i, "ID_PATCH"])
        edge = boundaries(instances[i])
        overlay = np.zeros((*edge.shape, 4))
        overlay[edge] = (1, 1, 1, 0.55)
        if pz.has(f"DATA_S2/S2_{pid}.npy"):
            rgb, date, s2_min = s2_rgb(pz, pid, dates[str(pid)])
            s2_mins.append(s2_min)
            panels = [(rgb, f"{pid} S2 {date}")]
        else:  # not in the PASTIS subset
            panels = [(np.zeros((*edge.shape, 3)), f"{pid} (no S2)")]
        panels += [(pcs[name][0][rows.index(i)], name) for name in names]
        axes[r, 0].imshow(panels[0][0])
        axes[r, 0].set_title(panels[0][1], fontsize=8)
        axes[r, 1].imshow(labels[i], cmap=label_cmap(), vmin=0, vmax=19, interpolation="nearest")
        axes[r, 1].contour(edge.astype(float), levels=[0.5], colors="k", linewidths=0.3)
        axes[r, 1].set_title(f"labels ({meta.loc[i, 'TILE']}, fold {meta.loc[i, 'Fold']})", fontsize=8)
        for c, (img, name) in enumerate(panels[1:], start=2):
            axes[r, c].imshow(img, interpolation="nearest")
            axes[r, c].set_title(f"PCA {name}", fontsize=8)
        for c in [0] + list(range(2, ncol)):
            axes[r, c].imshow(overlay, interpolation="nearest")
        for ax in axes[r]:
            ax.set_axis_off()
    var = ", ".join(f"{n}: {pcs[n][1].sum():.0%}" for n in names)
    fig.suptitle(f"{title} - PCA variance explained (3 comp.): {var}", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return s2_mins


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pastis", required=True, help="PASTIS.zip, PASTIS folder or subset (for the S2 images)")
    ap.add_argument("--processed", required=True)
    ap.add_argument("--out", default="results/qa")
    ap.add_argument("--names", nargs="*", default=DEFAULT_NAMES, help="embeddings to check (missing ones skipped)")
    ap.add_argument("--n-per-tile", type=int, default=5)
    ap.add_argument("--ids", type=int, nargs="*", help="show these ID_PATCH instead of a random draw")
    ap.add_argument("--max-nodata", type=float, default=0.05)
    ap.add_argument("--max-shift-gain", type=float, default=0.10,
                    help="relative drop of the ratio at the best 1-pixel shift above which a patch is suspect")
    ap.add_argument("--n-worst", type=int, default=8, help="patches shown in qa_largest_shift_gain.png")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    processed, out = Path(args.processed), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(processed / "meta.csv", index_col="idx")
    labels = np.load(processed / "labels.npy", mmap_mode="r")
    instances = np.load(processed / "instances.npy", mmap_mode="r")
    dates = json.loads((processed / "dates_s2.json").read_text())
    pz = open_pastis(args.pastis)
    embs = {n: f for n in args.names if (f := open_processed(processed, n)) is not None}
    print("embeddings found:", list(embs) or "none")

    # 1. numeric summary and flags
    meta_qa, summary = meta.copy(), []
    ratio_cols = ["ratio_+0_+0", "ratio_+1_+0", "ratio_-1_+0", "ratio_+0_+1", "ratio_+0_-1"]
    for name in embs:
        qa_file = processed / f"qa_{name}.csv"
        if not qa_file.exists():
            continue
        qa = pd.read_csv(qa_file, index_col="idx")
        error = qa["error"].fillna("").astype(str).str.len() > 0
        gain = (1 - qa[ratio_cols].min(axis=1) / qa["ratio_+0_+0"]).fillna(0)
        suspect = gain > args.max_shift_gain
        ok = ~error & (qa.nodata_frac <= args.max_nodata) & ~suspect
        meta_qa[f"ok_{name}"] = ok
        meta_qa[f"gain_{name}"] = gain.round(4)
        aligned = (qa.best_dx == 0) & (qa.best_dy == 0)
        summary.append({"name": name, "patches": len(qa), "errors": int(error.sum()),
                        "with_nodata": int((qa.nodata_frac > 0).sum()),
                        "nodata_gt_max": int((qa.nodata_frac > args.max_nodata).sum()),
                        "best_not_0": int((~aligned).sum()), "suspect_shift": int(suspect.sum()),
                        "gain_median": round(float(gain.median()), 4),
                        "ratio_median": round(float(qa["ratio_+0_+0"].median()), 4), "ok": int(ok.sum())})
    if summary:
        summary = pd.DataFrame(summary)
        print(summary.to_string(index=False))
        summary.to_csv(out / "qa_summary.csv", index=False)
        meta_qa.to_csv(processed / "meta_qa.csv", index_label="idx")

    # 2. figures: random patches per tile, plus the largest shift gains
    if args.ids:
        groups = {"selected": args.ids}
    else:
        groups = tile_sample(meta, args.n_per_tile, args.seed)  # same draw as extract_subset.py
    gain_cols = [c for c in meta_qa.columns if c.startswith("gain_")]
    if gain_cols:
        worst = meta_qa.assign(g=meta_qa[gain_cols].max(axis=1)).nlargest(args.n_worst, "g")
        groups["largest_shift_gain"] = worst.ID_PATCH.tolist()
    all_mins = []
    for gname, ids in groups.items():
        path = out / f"qa_{gname}.png"
        all_mins += plot_patches(ids, meta, labels, instances, embs, pz, dates, args.seed, gname, path)
        print(f"saved {path} ({len(ids)} patches)")
    if all_mins:
        print(f"S2 min() without the -10000 fill, over the shown patches: {min(all_mins)} "
              f"({sum(v < 0 for v in all_mins)}/{len(all_mins)} with negative reflectance; keep int16)")


if __name__ == "__main__":
    main()
