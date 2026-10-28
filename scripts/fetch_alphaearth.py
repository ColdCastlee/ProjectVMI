"""Download AlphaEarth (Google Satellite Embedding, annual) for the PASTIS patches.

Each patch is requested with ee.data.computePixels on the PASTIS grid itself
(UTM of the tile, 10 m, top-left snapped to the 10 m lattice), so Earth Engine
returns the pixels already aligned. Each shard keeps, per patch:
    emb  float16 (64, 128, 128)  channel first, NaN where Earth Engine returns no data
plus QA values (no-data fraction, deviation from unit norm, alignment shift test).

Data: "The AlphaEarth Foundations Satellite Embedding dataset is produced by
Google and Google DeepMind." (CC-BY 4.0)

Example (after ee.Authenticate() in Colab):
    python scripts/fetch_alphaearth.py --project my-gee-project --processed /content/processed \
        --year 2019 --shard-dir /content/drive/MyDrive/pj_vmi/shards/ae_2019
"""

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import ee
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.align_qa import SHIFTS, best_shift, shift_scores
from src.data.pastis_grid import PIXEL, SIZE
from src.data.shards import code_version, run_sharded

COLLECTION = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
BANDS = [f"A{k:02d}" for k in range(64)]
MARGIN = 1  # extra pixels around the patch, used by the shift test


def pixel_grid(x0, y1, epsg, m=MARGIN):
    n = SIZE + 2 * m
    return {
        "dimensions": {"width": n, "height": n},
        "affineTransform": {
            "scaleX": PIXEL, "shearX": 0, "translateX": float(x0) - m * PIXEL,
            "shearY": 0, "scaleY": -PIXEL, "translateY": float(y1) + m * PIXEL,
        },
        "crsCode": f"EPSG:{int(epsg)}",
    }


def compute_with_retry(request, attempts=6):
    for k in range(attempts):
        try:
            return ee.data.computePixels(request)
        except Exception:  # quota / too many requests / transient HTTP errors
            if k == attempts - 1:
                raise
            time.sleep(min(60, 5 * 2**k))


def make_fetch(collection, meta, instances):
    m = MARGIN

    def fetch(i):
        row = meta.loc[i]
        region = ee.Geometry.Rectangle([row.lon_min, row.lat_min, row.lon_max, row.lat_max])
        image = collection.filterBounds(region).mosaic()
        request = {"expression": image, "fileFormat": "NUMPY_NDARRAY", "bandIds": BANDS,
                   "grid": pixel_grid(row.x0, row.y1, row.epsg)}
        arr = compute_with_retry(request)  # structured array (n, n), one field per band
        emb = np.stack([arr[b] for b in BANDS]).astype(np.float32)  # (64, n, n)
        norm = np.sqrt((emb**2).sum(axis=0))
        nodata = norm < 1e-6  # embeddings are unit vectors; all-zero = masked pixel
        emb[:, nodata] = np.nan

        scores = shift_scores(emb, instances[i], m)
        dx, dy = best_shift(scores)
        inner = np.s_[m:-m, m:-m]
        valid_norm = norm[inner][~nodata[inner]]
        qa = {"nodata_frac": float(nodata[inner].mean()), "best_dx": dx, "best_dy": dy,
              "norm_dev": float(np.abs(valid_norm - 1).mean()) if valid_norm.size else np.nan}
        qa.update({f"ratio_{a:+d}_{b:+d}": v for (a, b), v in zip(SHIFTS, scores)})
        return {"emb": np.ascontiguousarray(emb[:, m:-m, m:-m]).astype(np.float16)}, qa

    return fetch


def placeholder():
    return {"emb": np.full((len(BANDS), SIZE, SIZE), np.nan, np.float16)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", help="Google Cloud project registered for Earth Engine")
    ap.add_argument("--processed", required=True, help="directory with meta.csv and instances.npy")
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--shard-dir", required=True)
    ap.add_argument("--shard-size", type=int, default=100)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--ids", type=int, nargs="*", help="pilot: only these ID_PATCH, one 'pilot' shard")
    ap.add_argument("--retry-errors", action="store_true", help="fetch again patches flagged with an error")
    ap.add_argument("--dry-run", action="store_true", help="print the grid of the first patch and stop")
    args = ap.parse_args()

    processed = Path(args.processed)
    meta = pd.read_csv(processed / "meta.csv", index_col="idx")
    instances = np.load(processed / "instances.npy", mmap_mode="r")
    if args.ids:
        indices = meta.index[meta.ID_PATCH.isin(args.ids)].tolist()
        prefix, shard_size = "pilot", len(indices)
    else:
        indices, prefix, shard_size = meta.index.tolist(), "part", args.shard_size

    if args.dry_run:
        row = meta.loc[indices[0]]
        print(f"patch {row.ID_PATCH} ({row.TILE}):", json.dumps(pixel_grid(row.x0, row.y1, row.epsg), indent=2))
        return
    if not args.project:
        raise SystemExit("--project is required")

    ee.Initialize(project=args.project)
    collection = ee.ImageCollection(COLLECTION).filterDate(f"{args.year}-01-01", f"{args.year + 1}-01-01")
    n_images = collection.size().getInfo()
    if n_images == 0:
        raise SystemExit(f"no AlphaEarth image for {args.year}")
    first = meta.loc[indices[0]]
    first_props = (collection.filterBounds(ee.Geometry.Point([first.lon_min, first.lat_min]))
                   .first().toDictionary().getInfo())
    print(f"{n_images} images for {args.year}; properties of one image: {first_props}")

    shard_dir = Path(args.shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)
    info = {
        "source": "AlphaEarth", "collection": COLLECTION, "year": args.year, "n_images": n_images,
        "image_properties_example": first_props, "earthengine_api": ee.__version__,
        "python": platform.python_version(), "code_commit": code_version(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "grid": "computePixels on the PASTIS grid (UTM of the tile, top-left snapped to 10 m)",
        "attribution": "The AlphaEarth Foundations Satellite Embedding dataset is produced by "
                       "Google and Google DeepMind.",
    }
    (shard_dir / "info.json").write_text(json.dumps(info, indent=2))

    fetch = make_fetch(collection, meta, instances)
    run_sharded(indices, fetch, placeholder, shard_dir, shard_size, args.workers, prefix,
                retry=args.retry_errors)
    print("done")


if __name__ == "__main__":
    main()
