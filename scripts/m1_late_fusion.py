"""Late fusion (Method 1, M1-6): mean of the logits of two heads already trained on different inputs.

For every seed, the kept weights of run A (e.g. TESSERA 2019 + UNet) and run B
(e.g. AlphaEarth 2019 + UNet) of the same seed predict every patch; the class is
the argmax of the mean of their logits. Nothing is trained. The result is logged
like a trained run (one line of runs.csv, per-patch matrices of the test and
validation patches), so that scripts/summarize_runs.py compares it with the others.

Example:
    python scripts/m1_late_fusion.py --processed /content/processed \
        --runs-dir /content/drive/MyDrive/pj_vmi/runs --runs-csv /content/drive/MyDrive/pj_vmi/results/runs.csv \
        --a base_unet-t19-w32 --b base_unet-ae19-w32 --variant late-t19-ae19 --seeds 0 1 2
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.splits import load_split, split_tag
from src.eval.metrics import miou
from src.eval.runs import RunEvaluator, log_run, run_name, save_cms
from src.train.batches import loader_batches
from src.train.loop import autocast, load_trained


def run_id(tag, split, seed):
    prefix, variant = tag.split("_", 1)
    return run_name(prefix, variant, 1.0, split_tag(split), seed)


@torch.no_grad()
def late_cms(members, labels, instances, valid, patches, device, batch_size=32, workers=2):
    """Per-patch matrices of the argmax of the mean logits of `members` [(config, stack, model), ...]."""
    evaluator = RunEvaluator(labels, instances, valid)
    loaders = [loader_batches(stack, labels, valid, patches, device, batch_size, workers) for _, stack, _ in members]
    for batches in zip(*loaders):
        logits = 0
        for (cfg, _, model), (x, _, idx) in zip(members, batches):
            with autocast(device, cfg.amp and device.type == "cuda"):
                logits = logits + model(x).float()
        for p, i in zip((logits / len(members)).argmax(1).cpu().numpy(), batches[0][2]):
            evaluator.add(i, p)
    return evaluator.result()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed", required=True)
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--runs-csv", required=True)
    ap.add_argument("--a", required=True, help="first member, <prefix>_<variant>, e.g. base_unet-t19-w32")
    ap.add_argument("--b", required=True, help="second member, e.g. base_unet-ae19-w32")
    ap.add_argument("--variant", required=True, help="variant name of the fused run, e.g. late-t19-ae19")
    ap.add_argument("--method", default="m1")
    ap.add_argument("--split", default="I")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--rerun", action="store_true")
    args = ap.parse_args()

    processed, runs_dir = Path(args.processed), Path(args.runs_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = np.load(processed / "labels.npy")
    valid = np.load(processed / "valid_mask.npy")
    instances = np.load(processed / "instances.npy", mmap_mode="r")
    part = load_split(processed, args.split)
    test, val = np.asarray(part["test"]), np.asarray(part["val"])

    for seed in args.seeds:
        rid = run_name(args.method, args.variant, 1.0, split_tag(args.split), seed)
        out = runs_dir / rid
        if (out / "run.json").exists() and not args.rerun:
            print(f"{rid}: already finished, skipped")
            continue
        ids = [run_id(t, args.split, seed) for t in (args.a, args.b)]
        members = [load_trained(runs_dir / i, processed, device, valid) for i in ids]
        val_idx, val_cms = late_cms(members, labels, instances, valid, val, device)
        save_cms(out, val_idx, val_cms, "val")
        idx, cms = late_cms(members, labels, instances, valid, test, device)
        n_params = sum(json.loads((runs_dir / i / "run.json").read_text())["n_params"] for i in ids)
        run = {"run_id": rid, "method": args.method, "variant": args.variant, "label_frac": 1.0,
               "split": split_tag(args.split), "seed": seed, "n_params": n_params, "train_seconds": 0.0}
        row = log_run(args.runs_csv, out, run, idx, cms,
                      extra={"members": ids, "rule": "argmax of the mean of the logits",
                             "val_miou": float(miou(val_cms[:, 0].sum(axis=0)))})
        print(f"{rid}: test mIoU {row['miou']:.4f} (boundary {row['miou_boundary']:.4f}, core {row['miou_core']:.4f}); "
              f"members {ids[0]}, {ids[1]}")


if __name__ == "__main__":
    main()
