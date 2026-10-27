"""Read aligned embeddings patch by patch, as (C, 128, 128) float32 with NaN = no data."""

from pathlib import Path

import numpy as np


def open_processed(processed, name):
    """From merged arrays: <name>_q.npy + <name>_scale.npy (TESSERA) or <name>.npy (AlphaEarth).

    Returns a function idx -> array, or None if the files are absent.
    """
    processed = Path(processed)
    q_file = processed / f"{name}_q.npy"
    if q_file.exists():
        q = np.load(q_file, mmap_mode="r")
        scale = np.load(processed / f"{name}_scale.npy", mmap_mode="r")
        return lambda i: q[i].astype(np.float32) * scale[i][None]
    f = processed / f"{name}.npy"
    if f.exists():
        e = np.load(f, mmap_mode="r")
        return lambda i: e[i].astype(np.float32)
    return None


def open_shard(path):
    """From one shard (.npz written by fetch_*.py), e.g. a pilot shard before merging.

    Returns (function idx -> array, list of the idx it contains).
    """
    d = np.load(path)
    pos = {int(i): k for k, i in enumerate(d["idx"])}
    if "q" in d.files:
        q, scale = d["q"], d["scale"]
        return (lambda i: q[pos[i]].astype(np.float32) * scale[pos[i]][None]), list(pos)
    e = d["emb"]
    return (lambda i: e[pos[i]].astype(np.float32)), list(pos)
