"""Batches of standardised inputs and targets for training and evaluation.

Training batches follow a fixed order: for epoch e the training patches are
shuffled by a generator seeded with (seed, e), and step s takes the
(s mod steps_per_epoch)-th group of `batch_size` patches of epoch
s // steps_per_epoch (an incomplete last group is dropped). The batch of a step
depends only on (seed, step), so a resumed run sees the same batches as a run
without interruption.

Two ways to feed the GPU, giving the same batches:
- DeviceData keeps the stored values (int8 + scale, float16) of a set of patches
  on the device, so that nothing is read from the disk during training (about
  8.3 GB for TESSERA + AlphaEarth on the 1,937 training and validation patches
  of a fold, on a 15 GB T4);
- a DataLoader over PatchDataset reads the patches from the disk at every step
  (slower with the 2 CPUs of Colab, but no GPU memory is used for the data).
"""

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import IGNORE, PatchDataset


class BatchOrder:
    """Positions (in the list of training patches) of the batch of each step."""

    def __init__(self, n, batch_size, seed):
        self.n, self.seed = n, int(seed)
        self.batch_size = min(batch_size, n)  # e.g. 15 patches at 1% labels
        self.per_epoch = n // self.batch_size
        self._epoch, self._perm = None, None

    def positions(self, step):
        epoch, k = divmod(step, self.per_epoch)
        if epoch != self._epoch:
            g = torch.Generator().manual_seed(self.seed * 1_000_003 + epoch)
            self._epoch, self._perm = epoch, torch.randperm(self.n, generator=g)
        return self._perm[k * self.batch_size:(k + 1) * self.batch_size]


class StepSampler:
    """Batch sampler for a DataLoader: the batches of steps start..stop-1, in the order of a BatchOrder."""

    def __init__(self, order, start, stop):
        self.order, self.start, self.stop = order, start, stop

    def __len__(self):
        return max(0, self.stop - self.start)

    def __iter__(self):
        for step in range(self.start, self.stop):
            yield self.order.positions(step).tolist()


class DeviceData:
    """Stored inputs and targets (label, or IGNORE outside the valid mask) of `patches`, kept on `device`."""

    def __init__(self, stack, labels, valid, patches, device, chunk=64):
        self.stack, self.device = stack, torch.device(device)
        self.patches = np.asarray(patches, dtype=np.int64)
        n = len(self.patches)
        self.raw = {}
        self.target = torch.empty((n, *labels.shape[1:]), dtype=torch.uint8, device=self.device)
        for start in range(0, n, chunk):
            idx = self.patches[start:start + chunk]
            for k, v in stack.read_many(idx).items():
                if k not in self.raw:
                    self.raw[k] = torch.empty((n, *v.shape[1:]), dtype=torch.from_numpy(v).dtype, device=self.device)
                self.raw[k][start:start + len(idx)] = torch.from_numpy(v).to(self.device)
            target = np.where(np.asarray(valid[idx], dtype=bool), labels[idx], IGNORE).astype(np.uint8)
            self.target[start:start + len(idx)] = torch.from_numpy(target).to(self.device)

    def __len__(self):
        return len(self.patches)

    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in [*self.raw.values(), self.target])

    def take(self, positions):
        """(standardised input, target) of the patches at `positions` of the list (not patch indices)."""
        pos = torch.as_tensor(positions, dtype=torch.long).to(self.device)
        raw = {k: v.index_select(0, pos) for k, v in self.raw.items()}
        return self.stack.decode(raw), self.target.index_select(0, pos).long()

    def batches(self, batch_size):
        """All patches in list order: (input, target, patch indices)."""
        for start in range(0, len(self), batch_size):
            stop = min(start + batch_size, len(self))
            x, y = self.take(torch.arange(start, stop))
            yield x, y, self.patches[start:stop]


def device_steps(data, order, start, stop):
    """Training batches (input, target) of steps start..stop-1 from a DeviceData."""
    for step in range(start, stop):
        yield data.take(order.positions(step))


def _loader(stack, labels, valid, patches, device, workers, **kw):
    # own generator: a DataLoader otherwise draws its base seed from the global generator,
    # which would change the dropout masks that follow
    return DataLoader(PatchDataset(stack, labels, valid, patches), num_workers=workers,
                      pin_memory=torch.device(device).type == "cuda", generator=torch.Generator().manual_seed(0), **kw)


def loader_steps(stack, labels, valid, patches, order, start, stop, device, workers=2):
    """Training batches (input, target) of steps start..stop-1, read from the disk by a DataLoader."""
    loader = _loader(stack, labels, valid, patches, device, workers, batch_sampler=StepSampler(order, start, stop))
    for raw, target, _ in loader:
        yield stack.decode(raw, device), target.to(device, non_blocking=True)


def loader_batches(stack, labels, valid, patches, device, batch_size, workers=2):
    """All patches in list order, read from the disk: (input, target, patch indices)."""
    for raw, target, idx in _loader(stack, labels, valid, patches, device, workers, batch_size=batch_size):
        yield stack.decode(raw, device), target.to(device, non_blocking=True), idx.numpy()
