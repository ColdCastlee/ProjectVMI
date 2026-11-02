"""Inputs and targets of the heads trained on frozen embeddings.

A variant is a list of sources stacked as input channels, e.g.
["tessera_2019", "ae_2019"] -> 128 + 64 = 192 channels. Sources:
- tessera_<year>: <name>_q.npy int8 (N, 128, H, W) and <name>_scale.npy float32 (N, H, W);
- ae_<year>: <name>.npy float16 (N, 64, H, W);
- noise: 64 channels of N(0, 1), fixed per (seed, patch) with the same generator
  as scripts/m1_linear_heads.py, so both heads see the same noise.

The Dataset returns every source in its stored dtype (2 MB per patch for TESSERA
instead of 8 MB in float32); `InputStack.decode` turns a batch into float32 on
the GPU and standardises every channel with a mean and std fitted on the valid
pixels of the training patches. The few no-data pixels (NaN, all outside the
shared valid mask) are set to 0 after standardisation, i.e. to the training
mean, so that no NaN spreads through the convolutions.

Targets: the label where the shared valid mask is true, IGNORE (19, the void
class) elsewhere, so that the loss and the metrics use exactly the same pixels.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..eval.metrics import N_CLASSES, VOID

IGNORE = VOID
NOISE = "noise"
NOISE_DIM = 64


def noise_patch(seed, i, size=128):
    return np.random.default_rng([seed, int(i)]).standard_normal((NOISE_DIM, size, size), dtype=np.float32)


class InputStack:
    """Reads, decodes and standardises the input channels of one variant."""

    def __init__(self, processed, sources, noise_seed=0, size=128):
        processed = Path(processed)
        self.sources, self.noise_seed, self.size = list(sources), int(noise_seed), size
        self.arrays, self.channels = {}, []
        for s in self.sources:
            if s == NOISE:
                self.channels.append(NOISE_DIM)
                continue
            q_file, f = processed / f"{s}_q.npy", processed / f"{s}.npy"
            if q_file.exists():
                self.arrays[s] = (np.load(q_file, mmap_mode="r"), np.load(processed / f"{s}_scale.npy", mmap_mode="r"))
            elif f.exists():
                self.arrays[s] = (np.load(f, mmap_mode="r"), None)
            else:
                raise FileNotFoundError(f"{s}: neither {q_file.name} nor {f.name} in {processed}")
            if self.arrays[s][0].shape[-2:] != (size, size):
                raise ValueError(f"{s}: patches of {self.arrays[s][0].shape[-2:]}, expected {size} x {size}")
            self.channels.append(self.arrays[s][0].shape[1])
        self.in_channels = sum(self.channels)
        self.mean = torch.zeros(self.in_channels)
        self.std = torch.ones(self.in_channels)
        self.fit_patches = None  # patches used by fit_stats

    def read(self, i):
        """Stored values of patch i as numpy arrays; keys are the sources and '<source>/scale' for TESSERA."""
        out = {}
        for s in self.sources:
            if s == NOISE:
                out[s] = noise_patch(self.noise_seed, i, self.size)
                continue
            emb, scale = self.arrays[s]
            out[s] = np.array(emb[int(i)])
            if scale is not None:
                out[f"{s}/scale"] = np.array(scale[int(i)])
        return out

    def to_float(self, raw):
        """(B, C, H, W) float32 before standardisation, from a batch of stored values (tensors)."""
        parts = []
        for s in self.sources:
            x = raw[s].float()
            if f"{s}/scale" in raw:
                x = x * raw[f"{s}/scale"].float().unsqueeze(1)
            parts.append(x)
        return torch.cat(parts, 1)

    def decode(self, raw, device=None):
        """Standardised (B, C, H, W) float32 input from a batch returned by the DataLoader."""
        if device is not None:
            raw = {k: v.to(device, non_blocking=True) for k, v in raw.items()}
        x = self.to_float(raw)
        mean = self.mean.to(x.device)[None, :, None, None]
        std = self.std.to(x.device)[None, :, None, None]
        return torch.nan_to_num((x - mean) / std, nan=0.0)

    def fit_stats(self, patches, valid, max_patches=400, seed=0):
        """Per-channel mean and std over the valid pixels of the training patches (at most max_patches of them)."""
        patches = np.asarray(patches)
        if len(patches) > max_patches:
            patches = np.random.default_rng(seed).choice(patches, max_patches, replace=False)
        total = torch.zeros(self.in_channels, dtype=torch.float64)
        total_sq = torch.zeros_like(total)
        n = 0
        for i in patches:
            x = self.to_float({k: torch.from_numpy(v)[None] for k, v in self.read(i).items()})[0].double()
            v = x[:, torch.from_numpy(np.asarray(valid[i], dtype=bool))]  # (C, valid pixels)
            total += v.sum(1)
            total_sq += (v * v).sum(1)
            n += v.shape[1]
        if n == 0:
            raise ValueError("no valid pixel in the patches used for the statistics")
        mean = total / n
        std = (total_sq / n - mean ** 2).clamp_min(0).sqrt()
        std[std < 1e-6] = 1.0
        self.mean, self.std = mean.float(), std.float()
        self.fit_patches = np.sort(patches)
        return self

    def state(self):
        return {"sources": self.sources, "noise_seed": self.noise_seed, "mean": self.mean, "std": self.std}

    def load_state(self, state):
        if list(state["sources"]) != self.sources:
            raise ValueError(f"statistics of {state['sources']}, not of {self.sources}")
        self.noise_seed = int(state["noise_seed"])
        self.mean, self.std = state["mean"].float(), state["std"].float()
        return self


class PatchDataset(Dataset):
    """(stored inputs, target, patch index) for every patch in `patches`."""

    def __init__(self, stack, labels, valid, patches):
        self.stack, self.labels, self.valid = stack, labels, valid
        self.patches = np.asarray(patches, dtype=np.int64)

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, k):
        i = int(self.patches[k])
        target = np.where(np.asarray(self.valid[i], dtype=bool), self.labels[i], IGNORE).astype(np.int64)
        return self.stack.read(i), target, i


def class_weights(labels, valid, patches, n_classes=N_CLASSES):
    """1 / sqrt(class frequency) over the valid pixels of `patches`, scaled so that the mean
    weight of a training pixel is 1 (as in scripts/m1_linear_heads.py); absent classes get 0.

    Returns (weights, pixel counts per class).
    """
    counts = np.zeros(n_classes, dtype=np.int64)
    for i in patches:
        y = np.asarray(labels[i])[np.asarray(valid[i], dtype=bool)]
        counts += np.bincount(y, minlength=n_classes)[:n_classes]
    freq = counts / counts.sum()
    w = np.zeros(n_classes)
    w[counts > 0] = 1 / np.sqrt(freq[counts > 0])
    return w / (freq * w).sum(), counts
