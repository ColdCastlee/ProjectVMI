"""Patch selections shared by several scripts.

fold_selection: the 25 patches of the RF pilot (prepare_batch.py), reproduced
exactly. It draws per_fold patches per fold with random.Random(seed).sample on
the IDs of that fold sorted by ID_PATCH (folds 1 to 5, one generator).
random.sample only depends on the population size and the generator state, so
sampling the sorted IDs gives the same patches. selection.json keeps their
format: {"year", "seed", "patches": [{"patch_id", "fold"}]}.

tile_sample: the patches drawn for the QA figures (n per tile).
"""

import json
import random

import numpy as np


def fold_selection(meta, per_fold=5, seed=42):
    rng = random.Random(seed)
    chosen = []
    for fold in range(1, 6):
        ids = sorted(int(p) for p in meta.loc[meta.Fold == fold, "ID_PATCH"])
        chosen += [{"patch_id": p, "fold": fold} for p in rng.sample(ids, per_fold)]
    return chosen


def tile_sample(meta, n_per_tile=5, seed=0):
    rng = np.random.default_rng(seed)
    return {tile: [int(p) for p in rng.choice(sub.ID_PATCH.values, size=min(n_per_tile, len(sub)), replace=False)]
            for tile, sub in meta.groupby("TILE")}


def write_selection(path, patches, year, seed=42):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"year": year, "seed": seed, "patches": patches}, f, indent=2)


def read_selection(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)["patches"]
