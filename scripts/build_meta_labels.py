"""Build meta.csv, labels.npy, instances.npy, parcel_ids.npy, splits.json, dates_s2.json and
selection_seed42.json (the 25 patches of the RF pilot, prepare_batch.py) from PASTIS.zip
or an extracted PASTIS folder.

Row i of every array in --out is the i-th patch of meta.csv (sorted by ID_PATCH).

Example:
    python scripts/build_meta_labels.py --pastis /content/PASTIS.zip --out /content/processed
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.pastis_grid import SIZE, add_utm_grid
from src.data.pastis_zip import open_pastis
from src.data.selection import fold_selection, write_selection

# Official PASTIS cross-validation: name -> (train folds, val fold, test fold)
OFFICIAL_FOLDS = {
    "I": ([1, 2, 3], 4, 5),
    "II": ([2, 3, 4], 5, 1),
    "III": ([3, 4, 5], 1, 2),
    "IV": ([4, 5, 1], 2, 3),
    "V": ([5, 1, 2], 3, 4),
}
N_CLASSES = 20  # 0 background, 1-18 crops, 19 void
S2_NODATA = -10000  # THEIA fill value


def parse_dates(value):
    d = json.loads(value) if isinstance(value, str) else value
    return [int(d[k]) for k in sorted(d, key=int)]


def make_splits(meta, seed, val_frac):
    idx_of = lambda mask: meta.index[mask].tolist()
    splits = {"official": {}, "leave_one_tile_out": {}}
    for name, (train, val, test) in OFFICIAL_FOLDS.items():
        splits["official"][name] = {
            "train": idx_of(meta.Fold.isin(train)),
            "val": idx_of(meta.Fold == val),
            "test": idx_of(meta.Fold == test),
        }
    rng = np.random.default_rng(seed)
    for tile in sorted(meta.TILE.unique()):
        rest = np.array(idx_of(meta.TILE != tile))
        val = np.sort(rng.choice(rest, size=int(round(val_frac * len(rest))), replace=False))
        splits["leave_one_tile_out"][tile] = {
            "train": sorted(set(rest.tolist()) - set(val.tolist())),
            "val": val.tolist(),
            "test": idx_of(meta.TILE == tile),
        }
    return splits


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pastis", required=True, help="PASTIS.zip or an extracted PASTIS folder")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--ids", type=int, nargs="*", help="only these ID_PATCH (for tests)")
    ap.add_argument("--margin-px", type=int, default=5, help="margin of the lon/lat query bbox")
    ap.add_argument("--s2-sample", type=int, default=20, help="S2 files read fully to check min()")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--loto-val-frac", type=float, default=0.10)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pz = open_pastis(args.pastis)

    meta = pz.metadata()
    if args.ids:
        meta = meta[meta.ID_PATCH.isin(args.ids)].reset_index(drop=True)
    print(f"{len(meta)} patches, CRS of metadata: {meta.crs}")
    meta = add_utm_grid(meta, margin_px=args.margin_px)

    # Sentinel-2: dates from the metadata vs the time axis of each S2 file (header only)
    dates = {int(p): parse_dates(d) for p, d in zip(meta.ID_PATCH, meta["dates-S2"])}
    meta["n_dates_s2"] = [len(dates[int(p)]) for p in meta.ID_PATCH]
    s2_t, s2_dtype = [], []
    for pid in meta.ID_PATCH:
        rel = f"DATA_S2/S2_{pid}.npy"
        if pz.has(rel):
            shape, dtype = pz.npy_header(rel)
            s2_t.append(shape[0])
            s2_dtype.append(str(dtype))
        else:
            s2_t.append(-1)
            s2_dtype.append("missing")
    meta["s2_T"], meta["s2_dtype"] = s2_t, s2_dtype
    bad_t = meta[meta.s2_T != meta.n_dates_s2]
    print(f"S2 dtypes: {meta.s2_dtype.value_counts().to_dict()}; "
          f"patches whose S2 length != number of dates: {len(bad_t)}")

    # Labels, parcel instances (INSTANCE_ANNOTATIONS) and parcel ids (ANNOTATIONS/ParcelIDs,
    # used by the RF pilot scripts for the parcel mask and the train/test leakage check)
    n = len(meta)
    labels = np.zeros((n, SIZE, SIZE), dtype=np.uint8)
    instances = np.zeros((n, SIZE, SIZE), dtype=np.int32)
    parcel_ids = np.zeros((n, SIZE, SIZE), dtype=np.int32)
    for i, pid in enumerate(meta.ID_PATCH):
        labels[i] = pz.npy(f"ANNOTATIONS/TARGET_{pid}.npy")[0]  # channel 0 = semantic class
        inst = pz.npy(f"INSTANCE_ANNOTATIONS/INSTANCES_{pid}.npy")
        if inst.max() > np.iinfo(np.int32).max:
            raise ValueError(f"instance id too large in patch {pid}")
        instances[i] = inst
        parcel_ids[i] = pz.npy(f"ANNOTATIONS/ParcelIDs_{pid}.npy")
        if (i + 1) % 500 == 0:
            print(f"  labels {i + 1}/{n}")
    if labels.max() >= N_CLASSES:
        raise ValueError(f"unexpected label value {labels.max()}")

    # Full read of a few S2 files. Two kinds of negative values: small negative
    # reflectances (keep int16, never cast to uint16) and the THEIA fill -10000,
    # which can cover whole dates.
    rng = np.random.default_rng(args.seed)
    sample = rng.choice(meta.ID_PATCH[meta.s2_T > 0], size=min(args.s2_sample, n), replace=False)
    n_neg, n_fill_dates, valid_min = 0, 0, 0
    for pid in sample:
        x = pz.npy(f"DATA_S2/S2_{pid}.npy")
        fill = x == S2_NODATA
        valid_min = min(valid_min, int(x[~fill].min()))
        n_neg += int((x[~fill] < 0).any())
        n_fill_dates += int(fill.reshape(x.shape[0], -1).any(axis=1).sum())
    print(f"S2 check on {len(sample)} patches: min without fill = {valid_min}, "
          f"{n_neg} patches with negative reflectance, {n_fill_dates} dates containing {S2_NODATA}")

    # Pixel count per class and fold (compare with Fig. 3 of the PASTIS documentation)
    counts = pd.DataFrame(
        [np.bincount(labels[(meta.Fold == f).values].ravel(), minlength=N_CLASSES) for f in range(1, 6)],
        index=[f"fold_{f}" for f in range(1, 6)],
    )

    np.save(out / "labels.npy", labels)
    np.save(out / "instances.npy", instances)
    np.save(out / "parcel_ids.npy", parcel_ids)
    counts.to_csv(out / "class_pixel_counts.csv")
    with open(out / "dates_s2.json", "w") as f:
        json.dump(dates, f)
    with open(out / "splits.json", "w") as f:
        json.dump(make_splits(meta, args.seed, args.loto_val_frac), f)
    if meta.Fold.value_counts().min() >= 5 and meta.Fold.nunique() == 5:
        # the 25 patches of the RF pilot, prepare_batch.py (5 per fold, seed 42)
        write_selection(out / "selection_seed42.json", fold_selection(meta, 5, 42), year=None)
    cols = ["ID_PATCH", "Fold", "TILE", "epsg", "x0", "y1", "x0_raw", "y1_raw", "snap_dx", "snap_dy",
            "footprint_ok",
            "lon_min", "lat_min", "lon_max", "lat_max", "N_Parcel", "Parcel_Cover",
            "n_dates_s2", "s2_T", "s2_dtype"]
    meta[cols].to_csv(out / "meta.csv", index_label="idx")

    print("patches per fold:", meta.Fold.value_counts().sort_index().to_dict())
    print("patches per tile:", meta.TILE.value_counts().sort_index().to_dict())
    print("snap correction per tile (m):")
    print(meta.groupby("TILE")[["snap_dx", "snap_dy"]].agg(["min", "max"]).round(2))
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
