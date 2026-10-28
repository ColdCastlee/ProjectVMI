"""Shared mask of the pixels used for training and evaluation by every variant of both methods.

valid = label != void (19) and the pixel has data in every listed embedding
(TESSERA no-data = NaN scale; AlphaEarth no-data = NaN). With one mask for all
variants, every comparison is made on exactly the same pixels.

Writes <processed>/valid_mask.npy (N, 128, 128) bool and prints how many pixels
each condition removes.

Example:
    python scripts/build_valid_mask.py --processed /content/drive/MyDrive/pj_vmi/processed
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

VOID = 19
DEFAULT_NAMES = ["tessera_2019", "tessera_2018", "ae_2019", "ae_2018"]


def finite_mask(processed, name, chunk=64):
    """(N, H, W) bool: pixels with data in embedding `name`."""
    scale_file = processed / f"{name}_scale.npy"
    if scale_file.exists():  # TESSERA: no-data is a NaN scale
        return np.isfinite(np.load(scale_file))
    qa_file = processed / f"qa_{name}.csv"
    emb = np.load(processed / f"{name}.npy", mmap_mode="r")
    if qa_file.exists() and pd.read_csv(qa_file).nodata_frac.max() == 0:
        print(f"{name}: no no-data pixel according to {qa_file.name}, array not read")
        return np.ones((emb.shape[0], *emb.shape[2:]), dtype=bool)
    out = np.empty((emb.shape[0], *emb.shape[2:]), dtype=bool)
    for start in range(0, emb.shape[0], chunk):
        out[start:start + chunk] = np.isfinite(emb[start:start + chunk]).all(axis=1)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed", required=True)
    ap.add_argument("--names", nargs="*", default=DEFAULT_NAMES)
    args = ap.parse_args()

    processed = Path(args.processed)
    labels = np.load(processed / "labels.npy")
    valid = labels != VOID
    total = labels.size
    print(f"pixels: {total:,}; void: {total - int(valid.sum()):,}")
    for name in args.names:
        has_data = finite_mask(processed, name)
        removed = int((valid & ~has_data).sum())
        valid &= has_data
        print(f"{name}: removes {removed:,} more pixels")
    np.save(processed / "valid_mask.npy", valid)
    print(f"valid pixels: {int(valid.sum()):,} ({valid.mean():.4%}); saved {processed / 'valid_mask.npy'}")


if __name__ == "__main__":
    main()
