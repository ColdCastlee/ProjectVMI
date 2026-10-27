"""Sub-pixel offset between an embedding and the PASTIS parcels, per tile.

The QA of fetch_*.py stores, for each patch, the within-parcel variance ratio at
shifts -1, 0 and +1 pixel along x and along y. A parabola through these three
values has its minimum at the sub-pixel shift that would align the embedding
best (convention of src/data/align_qa.py: the label at (r, c) is paired with the
embedding at (r + dy, c + dx)). The median per tile is compared with the offset
expected if the PASTIS grid starts at the raw footprint corner rather than at
the 10 m-snapped one: dx = -snap_dx / 10, dy = snap_dy / 10.

The residual misregistration is reported for both choices of origin:
res_snapped = |(med_dx, med_dy)| and res_raw = |(med_dx - pred_dx, med_dy - pred_dy)|.

Inputs: merged qa_<name>.csv files and/or shard .npz files (e.g. a pilot shard).

Example:
    python scripts/subpixel_offset.py --processed /content/processed \
        --qa /content/processed/qa_tessera_2019.csv /content/processed/qa_tessera_2018.csv \
        --out results/subpixel_offset.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SHIFT_COUNTS = {"(0,0)": (0, 0), "(+1,0)": (1, 0), "(-1,0)": (-1, 0), "(0,+1)": (0, 1), "(0,-1)": (0, -1)}


def load_qa(path, meta):
    path = Path(path)
    if path.suffix == ".npz":
        d = np.load(path)
        qa = pd.DataFrame({k[3:]: d[k] for k in d.files if k.startswith("qa_")}, index=d["idx"])
        name = path.parent.name
    else:
        qa = pd.read_csv(path, index_col="idx")
        name = path.stem.removeprefix("qa_")
    qa = qa.drop(columns=[c for c in ["ID_PATCH"] if c in qa.columns])
    return name, qa.join(meta[["ID_PATCH", "TILE", "snap_dx", "snap_dy"]])


def vertex(f_minus, f_zero, f_plus):
    """Minimum of the parabola through (-1, f_minus), (0, f_zero), (1, f_plus); NaN if not convex."""
    den = f_minus - 2 * f_zero + f_plus
    with np.errstate(divide="ignore", invalid="ignore"):
        d = (f_minus - f_plus) / (2 * den)
    return np.where(den > 0, np.clip(d, -1, 1), np.nan)


def summarise(name, qa):
    qa = qa.assign(
        sub_dx=vertex(qa["ratio_-1_+0"], qa["ratio_+0_+0"], qa["ratio_+1_+0"]),
        sub_dy=vertex(qa["ratio_+0_-1"], qa["ratio_+0_+0"], qa["ratio_+0_+1"]),
    )
    rows = []
    for tile, g in qa.groupby("TILE"):
        pred_dx, pred_dy = -g.snap_dx.iloc[0] / 10, g.snap_dy.iloc[0] / 10
        med_dx, med_dy = float(np.nanmedian(g.sub_dx)), float(np.nanmedian(g.sub_dy))
        row = {"name": name, "tile": tile, "n": len(g),
               "pred_dx": pred_dx, "med_dx": med_dx, "pred_dy": pred_dy, "med_dy": med_dy,
               "res_snapped": float(np.hypot(med_dx, med_dy)),
               "res_raw": float(np.hypot(med_dx - pred_dx, med_dy - pred_dy))}
        for label, (dx, dy) in SHIFT_COUNTS.items():
            row[label] = int(((g.best_dx == dx) & (g.best_dy == dy)).sum())
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed", required=True, help="directory with meta.csv")
    ap.add_argument("--qa", nargs="+", required=True, help="qa_<name>.csv and/or shard .npz files")
    ap.add_argument("--out", default="results/subpixel_offset.csv")
    args = ap.parse_args()

    meta = pd.read_csv(Path(args.processed) / "meta.csv", index_col="idx")
    rows = []
    for path in args.qa:
        rows += summarise(*load_qa(path, meta))
    table = pd.DataFrame(rows)

    weighted = (table.assign(w_snapped=table.res_snapped * table.n, w_raw=table.res_raw * table.n)
                .groupby("name")[["n", "w_snapped", "w_raw"]].sum())
    weighted["mean_res_snapped"] = weighted.w_snapped / weighted.n
    weighted["mean_res_raw"] = weighted.w_raw / weighted.n

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    pd.set_option("display.width", 200)
    print(table.round(3).to_string(index=False))
    print("\nmean residual misregistration (pixels, weighted by patches):")
    print(weighted[["n", "mean_res_snapped", "mean_res_raw"]].round(3).to_string())
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
