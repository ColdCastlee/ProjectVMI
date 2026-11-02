"""Training of one run: a head on frozen inputs, from the training patches of a split.

Budget in optimizer steps, the same for every variant and every label fraction:
3,600 steps of 32 patches = 80 epochs of the 1,455 training patches of fold I
(at 1% labels the 15 patches are seen 3,600 times instead of 80). AdamW (lr 1e-4,
weight decay 1e-4), cosine schedule, mixed precision on GPU, cross-entropy
weighted by 1/sqrt(class frequency) of the labelled training pixels,
ignore_index 19 (void and pixels outside the shared valid mask). The validation
mIoU is computed every `eval_every` steps; the weights with the best validation
mIoU are evaluated on the test patches at the end.

The checkpoint (model, optimizer, scheduler, scaler, random states, best weights,
validation history) is written to the run folder every `checkpoint_every` steps.
With the fixed batch order of batches.py, a run stopped by a disconnection
continues from its last checkpoint and gives the same result as a run without
interruption (exactly on CPU; on GPU up to non-deterministic cuDNN kernels).
"""

import contextlib
import json
import math
import os
import time
import warnings
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..data.splits import load_split, split_tag
from ..data.subsets import label_subset
from ..eval.metrics import N_CLASSES, miou
from ..eval.runs import RunEvaluator, log_run, run_name
from ..models.unet import build_head, count_parameters
from .batches import BatchOrder, DeviceData, device_steps, loader_batches, loader_steps
from .data import IGNORE, InputStack, class_weights

# fields that do not change the result: a run may continue with other values
RUNTIME_FIELDS = ("data_on_device", "workers", "eval_batch_size", "checkpoint_every")


@dataclass
class TrainConfig:
    sources: list
    variant: str
    head: str = "unet"
    widths: tuple = (32, 64, 128)
    hidden: int = 128
    dropout: float = 0.1
    method: str = "baseline"
    prefix: str = "base"
    split: str = "I"
    label_frac: float = 1.0
    seed: int = 0
    steps: int = 3600
    eval_every: int = 45
    checkpoint_every: int = 225
    batch_size: int = 32
    eval_batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 1e-4
    amp: bool = True
    stats_patches: int = 400
    data_on_device: bool = True
    workers: int = 2

    def __post_init__(self):
        self.sources = list(self.sources)
        self.widths = tuple(int(w) for w in self.widths)
        self.label_frac, self.seed = float(self.label_frac), int(self.seed)

    @property
    def run_variant(self):
        return f"{self.variant}-w{self.widths[0]}" if self.head == "unet" else self.variant

    @property
    def run_id(self):
        return run_name(self.prefix, self.run_variant, self.label_frac, split_tag(self.split), self.seed)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds every CUDA device
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def autocast(device, enabled):
    return torch.autocast(device.type, dtype=torch.float16) if enabled else contextlib.nullcontext()


@torch.no_grad()
def predictions(model, batches, device, amp):
    """(predicted class, target, patch indices) for every batch (input, target, patch indices)."""
    model.eval()
    for x, y, idx in batches:
        with autocast(device, amp):
            pred = model(x).argmax(1)
        yield pred, y, idx


def val_miou(model, batches, device, amp):
    cm = torch.zeros(N_CLASSES * N_CLASSES, dtype=torch.long, device=device)
    for pred, y, _ in predictions(model, batches, device, amp):
        keep = y != IGNORE
        cm += torch.bincount(y[keep] * N_CLASSES + pred[keep], minlength=N_CLASSES * N_CLASSES)
    return float(miou(cm.view(N_CLASSES, N_CLASSES).cpu().numpy()))


def comparable(config):
    return {k: v for k, v in config.items() if k not in RUNTIME_FIELDS}


def save(obj, path):
    tmp = Path(f"{path}.tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def train_run(cfg, processed, runs_dir, runs_csv, device=None, stop_at=None, rerun=False, log_every=450,
              log=partial(print, flush=True)):
    """Train one run, keep the weights with the best validation mIoU, evaluate them on the test patches.

    Returns the run summary (the runs.csv line plus best_val_miou and best_step), or None when
    stopped at `stop_at` (the same call without stop_at continues from the checkpoint). A finished
    run (run.json present) is skipped unless rerun=True, which also discards its checkpoint.
    """
    # the first steps skipped by the GradScaler make the scheduler warn; the schedule is unchanged
    warnings.filterwarnings("ignore", message="Detected call of `lr_scheduler.step")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    amp = cfg.amp and device.type == "cuda"
    processed, run_dir = Path(processed), Path(runs_dir) / cfg.run_id
    ckpt_file = run_dir / "last.pt"
    if rerun:
        for f in ("run.json", "cm_per_patch.npz", "best.pt", "last.pt"):
            (run_dir / f).unlink(missing_ok=True)
    if (run_dir / "run.json").exists():
        log(f"{cfg.run_id}: already finished, skipped")
        return json.loads((run_dir / "run.json").read_text())
    run_dir.mkdir(parents=True, exist_ok=True)

    labels = np.load(processed / "labels.npy")
    valid = np.load(processed / "valid_mask.npy")
    instances = np.load(processed / "instances.npy", mmap_mode="r")
    part = load_split(processed, cfg.split)
    val, test = np.asarray(part["val"]), np.asarray(part["test"])
    train, subset = label_subset(labels, valid, part["train"], cfg.label_frac, cfg.seed)
    weights, _ = class_weights(labels, valid, train)

    set_seed(cfg.seed)
    stack = InputStack(processed, cfg.sources, noise_seed=cfg.seed, size=labels.shape[-1])
    model = build_head(cfg.head, stack.in_channels, N_CLASSES, cfg.widths, cfg.hidden, cfg.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, cfg.steps) / cfg.steps)))
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    config = asdict(cfg)
    state = {"step": 0, "seconds": 0.0, "loss_sum": 0.0, "loss_n": 0, "history": [],
             "best_miou": -1.0, "best_step": 0, "best_model": None}

    if ckpt_file.exists():
        ck = torch.load(ckpt_file, map_location="cpu", weights_only=False)
        if comparable(ck["config"]) != comparable(config):
            raise ValueError(f"{ckpt_file} was written with another configuration; use rerun to start again")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        stack.load_state(ck["stack"])
        stack.fit_patches = np.asarray(ck["fit_patches"])
        state = ck["state"]
        torch.set_rng_state(ck["rng_cpu"])
        if device.type == "cuda" and ck["rng_cuda"] is not None:
            torch.cuda.set_rng_state(ck["rng_cuda"])
        log(f"{cfg.run_id}: resumed from step {state['step']}")
    else:
        stack.fit_stats(train, valid, cfg.stats_patches, cfg.seed)
        log(f"{cfg.run_id}: {len(train)} training patches ({cfg.label_frac:.0%} of {subset['n_train']}), "
            f"{len(val)} validation, {len(test)} test; {stack.in_channels} input channels, "
            f"{count_parameters(model):,} parameters")

    order = BatchOrder(len(train), cfg.batch_size, cfg.seed)
    train_data = val_data = None
    if cfg.data_on_device:
        t0 = time.time()
        train_data = DeviceData(stack, labels, valid, train, device)
        val_data = DeviceData(stack, labels, valid, val, device)
        log(f"data on {device}: {(train_data.nbytes() + val_data.nbytes()) / 1e9:.1f} GB in {time.time() - t0:.0f} s")
        steps = device_steps(train_data, order, state["step"], cfg.steps)
        val_batches = partial(val_data.batches, cfg.eval_batch_size)
    else:
        steps = loader_steps(stack, labels, valid, train, order, state["step"], cfg.steps, device, cfg.workers)
        val_batches = partial(loader_batches, stack, labels, valid, val, device, cfg.eval_batch_size, cfg.workers)

    w = torch.tensor(weights, dtype=torch.float32, device=device)
    loss_sum = torch.tensor(state["loss_sum"], device=device)
    t0 = time.time()
    model.train()
    for x, y in steps:
        with autocast(device, amp):
            loss = F.cross_entropy(model(x), y, weight=w, ignore_index=IGNORE)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        loss_sum += loss.detach().float()
        state["loss_n"] += 1
        state["step"] += 1
        step = state["step"]

        if step % cfg.eval_every == 0 or step == cfg.steps:
            m = val_miou(model, val_batches(), device, amp)
            model.train()
            entry = {"step": step, "train_loss": loss_sum.item() / state["loss_n"], "val_miou": m,
                     "lr": sched.get_last_lr()[0]}
            state["history"].append(entry)
            loss_sum.zero_()
            state["loss_n"] = 0
            if m > state["best_miou"]:
                best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                state.update(best_miou=m, best_step=step, best_model=best)
            if step % log_every == 0 or step == cfg.steps:
                log(f"step {step:5d}  loss {entry['train_loss']:.4f}  val mIoU {m:.4f}  "
                    f"(best {state['best_miou']:.4f} at step {state['best_step']})")

        stopping = stop_at is not None and stop_at <= step < cfg.steps
        if step % cfg.checkpoint_every == 0 or step == cfg.steps or stopping:
            state["seconds"] += time.time() - t0
            t0 = time.time()
            state["loss_sum"] = loss_sum.item()
            save({"config": config, "model": model.state_dict(), "optimizer": opt.state_dict(),
                  "scheduler": sched.state_dict(), "scaler": scaler.state_dict(), "stack": stack.state(),
                  "fit_patches": stack.fit_patches, "state": state, "rng_cpu": torch.get_rng_state(),
                  "rng_cuda": torch.cuda.get_rng_state() if device.type == "cuda" else None}, ckpt_file)
        if stopping:
            log(f"{cfg.run_id}: stopped at step {step}; run the same command again to continue")
            return None

    # test: the weights with the best validation mIoU (the training data leave the GPU first)
    steps = val_batches = train_data = val_data = x = y = None
    if device.type == "cuda":
        torch.cuda.empty_cache()
    model.load_state_dict(state["best_model"])
    torch.save(state["best_model"], run_dir / "best.pt")
    evaluator = RunEvaluator(labels, instances, valid)
    test_batches = loader_batches(stack, labels, valid, test, device, cfg.eval_batch_size, cfg.workers)
    for pred, _, idx in predictions(model, test_batches, device, amp):
        for p, i in zip(pred.cpu().numpy(), idx):
            evaluator.add(i, p)
    ids, cms = evaluator.result()

    run = {"run_id": cfg.run_id, "method": cfg.method, "variant": cfg.run_variant, "label_frac": cfg.label_frac,
           "split": split_tag(cfg.split), "seed": cfg.seed, "n_params": count_parameters(model),
           "train_seconds": round(state["seconds"], 1)}
    best = {"best_val_miou": state["best_miou"], "best_step": state["best_step"]}
    extra = {**best, "config": config, "history": state["history"], "subset": subset,
             "train_patches": train.tolist(), "stats_patches": stack.fit_patches.tolist(),
             "class_weights": weights.tolist(), "torch": torch.__version__,
             "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"}
    row = log_run(runs_csv, run_dir, run, ids, cms, extra)
    ckpt_file.unlink(missing_ok=True)
    log(f"{cfg.run_id}: test mIoU {row['miou']:.4f} (boundary {row['miou_boundary']:.4f}, "
        f"core {row['miou_core']:.4f}); best validation mIoU {state['best_miou']:.4f} at step {state['best_step']}; "
        f"{state['seconds'] / 60:.1f} min of training")
    return {**row, **best}
