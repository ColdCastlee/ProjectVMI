"""Copy the PASTIS files of a few patches out of PASTIS.zip, for the steps run after the zip is gone.

Patches: the 25 of the RF pilot (selection_seed42.json), the QA
patches (n per tile) and any --ids. The output keeps the PASTIS layout, so the
RF pilot scripts and open_pastis() read it like the full dataset:

    <out>/PASTIS/metadata.geojson, NORM_S2_patch.json
    <out>/PASTIS/ANNOTATIONS/TARGET_<id>.npy, ParcelIDs_<id>.npy
    <out>/PASTIS/INSTANCE_ANNOTATIONS/INSTANCES_<id>.npy
    <out>/PASTIS/DATA_S2/S2_<id>.npy
    <out>/subset_ids.json

Example:
    python scripts/extract_subset.py --pastis /content/PASTIS.zip --processed /content/processed \
        --out /content/drive/MyDrive/pj_vmi/pastis_subset
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.pastis_zip import open_pastis
from src.data.selection import read_selection, tile_sample


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pastis", required=True, help="PASTIS.zip or an extracted PASTIS folder")
    ap.add_argument("--processed", required=True, help="directory with meta.csv and selection_seed42.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--qa-per-tile", type=int, default=5)
    ap.add_argument("--qa-seed", type=int, default=0)
    ap.add_argument("--ids", type=int, nargs="*", default=[], help="extra ID_PATCH")
    args = ap.parse_args()

    processed = Path(args.processed)
    meta = pd.read_csv(processed / "meta.csv", index_col="idx")
    groups = {
        "selection_seed42": [p["patch_id"] for p in read_selection(processed / "selection_seed42.json")],
        "qa": [p for ids in tile_sample(meta, args.qa_per_tile, args.qa_seed).values() for p in ids],
        "extra": list(args.ids),
    }
    ids = sorted({p for g in groups.values() for p in g})

    src = open_pastis(args.pastis)
    root = Path(args.out) / "PASTIS"
    files = ["metadata.geojson", "NORM_S2_patch.json"]
    for pid in ids:
        files += [f"ANNOTATIONS/TARGET_{pid}.npy", f"ANNOTATIONS/ParcelIDs_{pid}.npy",
                  f"INSTANCE_ANNOTATIONS/INSTANCES_{pid}.npy", f"DATA_S2/S2_{pid}.npy"]
    total = 0
    for rel in files:
        dst = root / rel
        if dst.exists():
            total += dst.stat().st_size
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        data = src.raw(rel)
        tmp = dst.with_name(dst.name + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(dst)
        total += len(data)
    (Path(args.out) / "subset_ids.json").write_text(json.dumps(groups, indent=2))
    print(f"{len(ids)} patches ({', '.join(f'{k}: {len(v)}' for k, v in groups.items())}), "
          f"{len(files)} files, {total / 1e6:.0f} MB in {root}")


if __name__ == "__main__":
    main()
