"""Train / val / test patch indices from processed/splits.json (built by build_meta_labels.py).

Names: "I" to "V" for the official PASTIS folds, "loto_<tile>" for leave-one-tile-out
(e.g. "loto_t30uxv"). Indices are rows of meta.csv and of every processed array.
"""

import json
from pathlib import Path


def load_split(processed, name):
    splits = json.loads((Path(processed) / "splits.json").read_text())
    if name in splits["official"]:
        return splits["official"][name]
    if name.startswith("loto_") and name[5:] in splits["leave_one_tile_out"]:
        return splits["leave_one_tile_out"][name[5:]]
    raise KeyError(f"unknown split {name!r}; official: {list(splits['official'])}, "
                   f"leave-one-tile-out: {['loto_' + t for t in splits['leave_one_tile_out']]}")


def split_tag(name):
    """Short form used in run names: foldI, foldII, ..., loto_t30uxv."""
    return name if name.startswith("loto_") else f"fold{name}"
