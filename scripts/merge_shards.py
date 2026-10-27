"""Merge the part_*.npz shards of one embedding into full arrays plus a QA table.

Outputs in --out (row i = patch i of meta.csv):
    TESSERA:    <name>_q.npy (N, 128, 128, 128) int8 and <name>_scale.npy (N, 128, 128) float32
    AlphaEarth: <name>.npy   (N, 64, 128, 128) float16
    qa_<name>.csv, <name>_info.json

Write to the local disk (/content) and copy to Drive afterwards: memory-mapped
writes straight to Drive are slow.

Example:
    python scripts/merge_shards.py --shard-dir /content/drive/MyDrive/pj_vmi/shards/tessera_2019 \
        --processed /content/processed --name tessera_2019
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shard-dir", required=True)
    ap.add_argument("--processed", required=True, help="directory with meta.csv")
    ap.add_argument("--name", required=True, help="e.g. tessera_2019, ae_2019")
    ap.add_argument("--out", help="output directory (default: --processed)")
    args = ap.parse_args()

    shard_dir, processed = Path(args.shard_dir), Path(args.processed)
    out = Path(args.out or processed)
    out.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(processed / "meta.csv", index_col="idx")
    n = len(meta)

    parts = sorted(shard_dir.glob("part_*.npz"))
    if not parts:
        raise SystemExit(f"no part_*.npz in {shard_dir}")
    with np.load(parts[0]) as d:
        keys = [k for k in d.files if k != "idx" and not k.startswith("qa_")]
        specs = {k: (d[k].shape[1:], d[k].dtype) for k in keys}

    arrays = {}
    for k, (shape, dtype) in specs.items():
        fname = f"{args.name}.npy" if len(keys) == 1 else f"{args.name}_{k}.npy"
        arrays[k] = np.lib.format.open_memmap(out / fname, mode="w+", dtype=dtype, shape=(n, *shape))
        print(f"writing {out / fname} {(n, *shape)} {dtype}")

    seen = np.zeros(n, dtype=int)
    qa_parts = []
    for p in parts:
        with np.load(p) as d:
            idx = d["idx"]
            for k in keys:
                arrays[k][idx] = d[k]
            seen[idx] += 1
            qa_parts.append(pd.DataFrame({k[3:]: d[k] for k in d.files if k.startswith("qa_")}, index=idx))
    for a in arrays.values():
        a.flush()

    qa = pd.concat(qa_parts).sort_index()
    qa.insert(0, "ID_PATCH", meta.loc[qa.index, "ID_PATCH"].values)
    qa.to_csv(out / f"qa_{args.name}.csv", index_label="idx")
    if (shard_dir / "info.json").exists():
        shutil.copy(shard_dir / "info.json", out / f"{args.name}_info.json")

    errors = qa["error"].fillna("").astype(str).str.len() > 0
    shifted = (qa.best_dx.fillna(0) != 0) | (qa.best_dy.fillna(0) != 0)
    print(f"{len(parts)} shards, {int((seen > 0).sum())}/{n} patches")
    print(f"missing: {int((seen == 0).sum())}, duplicated: {int((seen > 1).sum())}, errors: {int(errors.sum())}")
    print(f"patches with no-data pixels: {int((qa.nodata_frac > 0).sum())} "
          f"(max fraction {qa.nodata_frac.max():.3f})")
    print(f"best alignment shift != (0, 0): {int(shifted.sum())}; "
          f"not testable (too few parcel pixels): {int(qa.best_dx.isna().sum())}")
    print(f"within-parcel variance ratio at (0, 0): median {qa['ratio_+0_+0'].median():.3f}")
    if errors.any():
        print("first errors:\n", qa.loc[errors, ["ID_PATCH", "error"]].head(10).to_string())
    if (seen == 0).any():
        raise SystemExit("some patches are missing: rerun the fetch script (it resumes), then merge again")


if __name__ == "__main__":
    main()
