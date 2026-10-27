"""Download TESSERA embeddings for the PASTIS patches, on the exact patch grid.

The PASTIS grid (UTM of the tile, 10 m, see src/data/pastis_grid.py) coincides
with the native TESSERA grid, so each patch is an integer crop of the stored
int8 embeddings: no resampling. Each shard keeps, per patch:
    q      int8    (128, 128, 128)  quantised embedding, channel first
    scale  float32 (128, 128)       dequantise with q * scale (NaN = no data)
plus QA values (no-data fraction, alignment shift test).

Example:
    python scripts/fetch_tessera.py --processed /content/processed --year 2019 \
        --shard-dir /content/drive/MyDrive/pj_vmi/shards/tessera_2019
"""

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from geotessera import GeoTesseraZarr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.align_qa import SHIFTS, best_shift, shift_scores  # noqa: E402
from src.data.pastis_grid import SIZE  # noqa: E402
from src.data.shards import code_version, run_sharded  # noqa: E402

N_DIM = 128
MARGIN = 1  # extra pixels around the patch, used by the shift test


def read_with_retry(gt, bbox, year, attempts=3):
    for k in range(attempts):
        try:
            return gt.read_region_quantized(bbox, year=year)
        except Exception:
            if k == attempts - 1:
                raise
            time.sleep(5 * 2**k)


def make_fetch(gt, meta, instances, year):
    m, n = MARGIN, SIZE + 2 * MARGIN

    def fetch(i):
        row = meta.loc[i]
        bbox = (row.lon_min, row.lat_min, row.lon_max, row.lat_max)
        q, s, tr, crs = read_with_retry(gt, bbox, year)
        if str(crs) != f"EPSG:{row.epsg}":
            raise ValueError(f"TESSERA zone {crs} != patch EPSG:{row.epsg}")
        col = (row.x0 - tr.c) / tr.a
        lin = (row.y1 - tr.f) / tr.e
        if abs(col - round(col)) > 1e-6 or abs(lin - round(lin)) > 1e-6:
            raise ValueError(f"patch corner not on the TESSERA grid (col {col:.3f}, row {lin:.3f})")
        r0, c0 = int(round(lin)) - m, int(round(col)) - m
        if r0 < 0 or c0 < 0 or r0 + n > q.shape[0] or c0 + n > q.shape[1]:
            raise ValueError(f"window outside the read region {q.shape[:2]}")
        qm = q[r0 : r0 + n, c0 : c0 + n].transpose(2, 0, 1)  # (128, n, n)
        sm = s[r0 : r0 + n, c0 : c0 + n]

        scores = shift_scores(qm.astype(np.float32) * sm[None], instances[i], m)
        dx, dy = best_shift(scores)
        qa = {"nodata_frac": float(np.isnan(sm[m:-m, m:-m]).mean()), "best_dx": dx, "best_dy": dy}
        qa.update({f"ratio_{a:+d}_{b:+d}": v for (a, b), v in zip(SHIFTS, scores)})
        data = {"q": np.ascontiguousarray(qm[:, m:-m, m:-m]), "scale": np.ascontiguousarray(sm[m:-m, m:-m])}
        return data, qa

    return fetch


def placeholder():
    return {"q": np.zeros((N_DIM, SIZE, SIZE), np.int8), "scale": np.full((SIZE, SIZE), np.nan, np.float32)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed", required=True, help="directory with meta.csv and instances.npy")
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--shard-dir", required=True)
    ap.add_argument("--shard-size", type=int, default=100)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--ids", type=int, nargs="*", help="pilot: only these ID_PATCH, one 'pilot' shard")
    ap.add_argument("--retry-errors", action="store_true", help="fetch again patches flagged with an error")
    ap.add_argument("--expect-dataset", default="1.1-dclimate", help="'' to accept any dataset")
    args = ap.parse_args()

    processed = Path(args.processed)
    meta = pd.read_csv(processed / "meta.csv", index_col="idx")
    instances = np.load(processed / "instances.npy", mmap_mode="r")

    gt = GeoTesseraZarr()
    ds = gt.dataset
    ds_name = getattr(ds, "name", None) or f"{ds.version}-{ds.variant}"
    print(f"TESSERA dataset: {ds_name}; years: {gt.years}")
    if args.expect_dataset and ds_name != args.expect_dataset:
        raise SystemExit(f"dataset {ds_name} != expected {args.expect_dataset}")
    if args.year not in gt.years:
        raise SystemExit(f"year {args.year} not available")

    shard_dir = Path(args.shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)
    info = {
        "source": "TESSERA", "dataset": ds_name, "year": args.year,
        "geotessera": getattr(sys.modules["geotessera"], "__version__", "?"),
        "python": platform.python_version(), "code_commit": code_version(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "grid": "UTM of the PASTIS tile, top-left snapped to 10 m, integer crop (no resampling)",
    }
    (shard_dir / "info.json").write_text(json.dumps(info, indent=2))

    if args.ids:
        indices = meta.index[meta.ID_PATCH.isin(args.ids)].tolist()
        prefix, shard_size = "pilot", len(indices)
    else:
        indices, prefix, shard_size = meta.index.tolist(), "part", args.shard_size
    fetch = make_fetch(gt, meta, instances, args.year)
    run_sharded(indices, fetch, placeholder, shard_dir, shard_size, args.workers, prefix,
                retry=args.retry_errors)
    print("done")


if __name__ == "__main__":
    main()
