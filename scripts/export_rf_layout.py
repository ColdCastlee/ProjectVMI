"""Write aligned embeddings in the folder layout of the RF pilot scripts
(prepare_batch.py, train_batch.py, train_raw_baseline.py, train_fusion.py).

train_batch.py, train_raw_baseline.py and train_fusion.py then run unchanged,
after setting their ROOT to --root:

    <root>/PASTIS/metadata.geojson
    <root>/PASTIS/ANNOTATIONS/TARGET_<id>.npy, ParcelIDs_<id>.npy
    <root>/PASTIS/DATA_S2/S2_<id>.npy                     (skip with --no-s2)
    <root>/<dataset-dir>/selection.json                   (same format as theirs)
    <root>/<dataset-dir>/patch_<id>/aligned.tif           (float32, C x 128 x 128, NaN = no data)

One difference with their prepare_batch.py, on purpose: aligned.tif is the
integer crop on the 10 m-snapped grid (no bilinear resampling), so its transform
starts at (x0, y1) instead of the raw footprint corner (x0_raw, y1_raw).

Example (pilot shard, before the full run is merged):
    python scripts/export_rf_layout.py --pastis /content/drive/MyDrive/pj_vmi/pastis_subset \n        --processed /content/processed \
        --name tessera_2019 --shard /content/pilot/tessera_2019/pilot_0000.npz --root /content/rf_layout
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.embeddings import open_processed, open_shard  # noqa: E402
from src.data.pastis_grid import PIXEL  # noqa: E402
from src.data.pastis_zip import open_pastis  # noqa: E402
from src.data.selection import read_selection, write_selection  # noqa: E402


def default_dataset_dir(name):
    source, year = name.rsplit("_", 1)  # tessera_2019 -> TESSERA/batch_2019
    return f"{source.upper()}/batch_{year}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pastis", required=True, help="PASTIS.zip, PASTIS folder or subset")
    ap.add_argument("--processed", required=True, help="directory with meta.csv (and merged arrays)")
    ap.add_argument("--name", required=True, help="tessera_2019, tessera_2018, ae_2019, ...")
    ap.add_argument("--root", required=True, help="output root, i.e. the ROOT of their scripts")
    ap.add_argument("--shard", help="read the embeddings from this shard instead of the merged arrays")
    ap.add_argument("--selection", help="selection.json (default: <processed>/selection_seed42.json)")
    ap.add_argument("--dataset-dir", help="default: TESSERA/batch_<year> or AE/batch_<year>")
    ap.add_argument("--no-s2", action="store_true", help="do not copy the S2 time series")
    args = ap.parse_args()

    processed, root = Path(args.processed), Path(args.root)
    meta = pd.read_csv(processed / "meta.csv", index_col="idx")
    idx_of = {int(p): i for i, p in zip(meta.index, meta.ID_PATCH)}
    patches = read_selection(args.selection or processed / "selection_seed42.json")

    if args.shard:
        get, available = open_shard(args.shard)
    else:
        get = open_processed(processed, args.name)
        if get is None:
            raise SystemExit(f"no merged arrays for {args.name} in {processed}")
        available = list(meta.index)
    missing = [p["patch_id"] for p in patches if idx_of[p["patch_id"]] not in set(available)]
    if missing:
        raise SystemExit(f"patches not in the source: {missing}")

    pz = open_pastis(args.pastis)
    pastis = root / "PASTIS"
    (pastis / "ANNOTATIONS").mkdir(parents=True, exist_ok=True)
    (pastis / "DATA_S2").mkdir(parents=True, exist_ok=True)
    (pastis / "metadata.geojson").write_bytes(pz.raw("metadata.geojson"))

    data_dir = root / (args.dataset_dir or default_dataset_dir(args.name))
    data_dir.mkdir(parents=True, exist_ok=True)
    year = int(args.name.rsplit("_", 1)[1])
    write_selection(data_dir / "selection.json", patches, year)

    for p in patches:
        pid = p["patch_id"]
        i = idx_of[pid]
        for rel in [f"ANNOTATIONS/TARGET_{pid}.npy", f"ANNOTATIONS/ParcelIDs_{pid}.npy"] + (
                [] if args.no_s2 else [f"DATA_S2/S2_{pid}.npy"]):
            (pastis / rel).write_bytes(pz.raw(rel))

        emb = get(i)  # (C, 128, 128) float32, NaN = no data
        row = meta.loc[i]
        folder = data_dir / f"patch_{pid}"
        folder.mkdir(exist_ok=True)
        profile = {"driver": "GTiff", "height": emb.shape[1], "width": emb.shape[2], "count": emb.shape[0],
                   "dtype": "float32", "crs": f"EPSG:{int(row.epsg)}",
                   "transform": from_origin(row.x0, row.y1, PIXEL, PIXEL), "nodata": np.nan,
                   "compress": "deflate"}
        tmp = folder / "aligned.tmp.tif"
        with rasterio.open(tmp, "w", **profile) as dst:
            dst.write(emb)
        tmp.replace(folder / "aligned.tif")
        print(f"patch {pid} (fold {p['fold']}): {emb.shape}, valid {np.isfinite(emb).all(axis=0).mean():.2%}")
    print(f"written under {root}; set ROOT = Path(r'{root}') in the RF pilot scripts")


if __name__ == "__main__":
    main()
