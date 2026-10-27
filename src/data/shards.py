"""Resumable download in shards: one .npz per block of patches, skipped if it already exists.

Shards are written to Google Drive so a Colab disconnection only loses the block
in progress. merge_shards.py assembles them into full arrays afterwards.
"""

import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


def code_version():
    """Short git commit of this repository, recorded with every download."""
    repo = Path(__file__).resolve().parents[2]
    try:
        out = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _safe_fetch(fetch, placeholder, i):
    t0 = time.time()
    try:
        data, qa = fetch(i)
        qa["error"] = ""
    except Exception as e:  # keep going; the patch is flagged in the QA table
        data, qa = placeholder(), {"error": f"{type(e).__name__}: {e}"[:300]}
    qa["seconds"] = time.time() - t0
    return data, qa


def save_shard(path, idx, results):
    arrays = {"idx": np.asarray(idx, dtype=np.int64)}
    for key in results[0][0]:
        arrays[key] = np.stack([data[key] for data, _ in results])
    for key in sorted({k for _, qa in results for k in qa}):
        values = [qa.get(key, np.nan) for _, qa in results]
        arrays["qa_" + key] = np.array(values, dtype=str if key == "error" else np.float64)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)  # a shard is either complete or absent


def retry_errors(path, fetch, placeholder, workers):
    """Fetch again the patches of an existing shard whose QA has an error."""
    with np.load(path) as d:
        arrays = {k: d[k] for k in d.files}
    bad = np.where(np.char.str_len(arrays["qa_error"]) > 0)[0]
    if bad.size == 0:
        return 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        redone = list(ex.map(lambda j: _safe_fetch(fetch, placeholder, int(arrays["idx"][j])), bad))
    keys = [k for k in arrays if k != "idx" and not k.startswith("qa_")]
    results = [({k: arrays[k][j] for k in keys},
                {k[3:]: arrays[k][j].item() for k in arrays if k.startswith("qa_")})
               for j in range(len(arrays["idx"]))]
    for j, res in zip(bad, redone):
        results[j] = res
    save_shard(path, arrays["idx"], results)
    return int(sum(bool(qa["error"]) for _, qa in redone))


def run_sharded(indices, fetch, placeholder, shard_dir, shard_size, workers, prefix="part",
                retry=False):
    shard_dir = Path(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)
    plan_file = shard_dir / f"{prefix}_plan.json"
    plan = {"n": len(indices), "shard_size": shard_size, "first": int(indices[0])}
    if plan_file.exists() and json.loads(plan_file.read_text()) != plan:
        raise SystemExit(f"{plan_file} differs from this run ({plan}); use another --shard-dir")
    plan_file.write_text(json.dumps(plan))

    blocks = [indices[s : s + shard_size] for s in range(0, len(indices), shard_size)]
    t_start, n_done = time.time(), 0
    for k, block in enumerate(blocks):
        path = shard_dir / f"{prefix}_{k:04d}.npz"
        if path.exists():
            if retry:
                left = retry_errors(path, fetch, placeholder, workers)
                print(f"[{prefix} {k + 1}/{len(blocks)}] retried errors, still failing: {left}", flush=True)
            continue
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(lambda i: _safe_fetch(fetch, placeholder, i), block))
        save_shard(path, block, results)
        n_done += len(block)
        n_left = sum(len(b) for b in blocks[k + 1 :])
        eta = (time.time() - t_start) / n_done * n_left / 60
        n_err = sum(bool(qa["error"]) for _, qa in results)
        first_err = next((qa["error"] for _, qa in results if qa["error"]), "")
        print(f"[{prefix} {k + 1}/{len(blocks)}] {len(block)} patches in {time.time() - t0:.0f} s, "
              f"errors: {n_err}, ETA {eta:.0f} min {first_err}", flush=True)
