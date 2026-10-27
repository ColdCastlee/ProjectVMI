"""Compare two aligned versions of the same patches, e.g. integer crop vs the RF pilot's bilinear version.

Both folders hold patch_<id>/aligned.tif (layout of export_rf_layout.py and
of prepare_batch.py). For every common patch: distance between the two
grid origins, share of valid pixels, mean per-pixel cosine similarity at shift
(0, 0) and at +-1 pixel, and the best shift. Two versions of the same grid peak
at (0, 0); bilinear resampling to an origin a few metres off lowers that peak
below 1 (neighbouring pixels are mixed), mostly near parcel boundaries.

Example:
    python scripts/compare_alignment.py --a /content/rf_layout/TESSERA/batch_2019 \
        --b /content/drive/MyDrive/pj_vmi/rf_pilot/batch_2019 --out results/compare_alignment.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

SHIFTS = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]


def cosine_map(a, b):
    na, nb = np.linalg.norm(a, axis=0), np.linalg.norm(b, axis=0)
    return (a * b).sum(axis=0) / (na * nb + 1e-12)


def shifted_cosine(a, b, dx, dy):
    """Mean cosine between a[:, r, c] and b[:, r + dy, c + dx] over valid pixels."""
    h, w = a.shape[1:]
    ra, rb = slice(max(0, -dy), h - max(0, dy)), slice(max(0, dy), h - max(0, -dy))
    ca, cb = slice(max(0, -dx), w - max(0, dx)), slice(max(0, dx), w - max(0, -dx))
    cos = cosine_map(a[:, ra, ca], b[:, rb, cb])
    return float(np.nanmean(cos))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="folder with patch_<id>/aligned.tif (integer crop)")
    ap.add_argument("--b", required=True, help="folder with patch_<id>/aligned.tif (other version)")
    ap.add_argument("--out", default="results/compare_alignment.csv")
    args = ap.parse_args()

    a_dir, b_dir = Path(args.a), Path(args.b)
    ids = sorted({p.parent.name for p in a_dir.glob("patch_*/aligned.tif")}
                 & {p.parent.name for p in b_dir.glob("patch_*/aligned.tif")})
    if not ids:
        raise SystemExit("no common patch_<id>/aligned.tif")
    rows = []
    for name in ids:
        with rasterio.open(a_dir / name / "aligned.tif") as ra, rasterio.open(b_dir / name / "aligned.tif") as rb:
            a, b = ra.read().astype(np.float32), rb.read().astype(np.float32)
            same_crs = ra.crs == rb.crs
            dx_m, dy_m = rb.transform.c - ra.transform.c, rb.transform.f - ra.transform.f
        if a.shape != b.shape:
            rows.append({"patch": name, "error": f"shapes {a.shape} vs {b.shape}"})
            continue
        cos = [shifted_cosine(a, b, dx, dy) for dx, dy in SHIFTS]
        best = SHIFTS[int(np.nanargmax(cos))]
        rows.append({"patch": name, "same_crs": same_crs, "origin_dx_m": round(dx_m, 3), "origin_dy_m": round(dy_m, 3),
                     "valid_a": float(np.isfinite(a).all(axis=0).mean()),
                     "valid_b": float(np.isfinite(b).all(axis=0).mean()),
                     **{f"cos_{dx:+d}_{dy:+d}": round(c, 4) for (dx, dy), c in zip(SHIFTS, cos)},
                     "best_dx": best[0], "best_dy": best[1]})
    table = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(table.to_string(index=False))
    if "best_dx" in table:
        aligned = ((table.best_dx == 0) & (table.best_dy == 0)).sum()
        print(f"\n{aligned}/{len(table)} patches peak at (0, 0); "
              f"median cosine at (0, 0): {table['cos_+0_+0'].median():.4f}")


if __name__ == "__main__":
    main()
