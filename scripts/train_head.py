"""Train heads on frozen inputs and evaluate them: one run per (config, split, label fraction, seed).

The YAML config gives the inputs and the head (configs/); the split, the label
fractions and the seeds are given here. Every run writes runs/<run_id>/
(run.json, cm_per_patch.npz, best.pt) and one line of runs.csv. A run stopped
by a disconnection continues from its last checkpoint when the same command is
started again; finished runs are skipped unless --rerun.

Example:
    python scripts/train_head.py --config configs/baselines/b2_unet_t19.yaml --processed /content/processed \
        --runs-dir /content/drive/MyDrive/pj_vmi/runs \
        --runs-csv /content/drive/MyDrive/pj_vmi/results/runs.csv --seeds 0 1 2 --label-fracs 1.0 0.01
"""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.train.loop import TrainConfig, train_run


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="YAML file with the inputs and the head")
    ap.add_argument("--processed", required=True, help="directory with the arrays (copied to /content first)")
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--runs-csv", required=True)
    ap.add_argument("--split", default="I")
    ap.add_argument("--label-fracs", type=float, nargs="+", default=[1.0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--steps", type=int, help="training budget in optimizer steps (default 3600)")
    ap.add_argument("--widths", type=int, nargs=3, help="UNet widths, e.g. 64 128 256")
    ap.add_argument("--data-on-cpu", action="store_true",
                    help="read the patches from the disk at every step instead of keeping them on the GPU")
    ap.add_argument("--stop-at", type=int, help="stop after this step; the run continues when started again")
    ap.add_argument("--rerun", action="store_true", help="train again finished runs, discarding checkpoints")
    args = ap.parse_args()

    base = yaml.safe_load(Path(args.config).read_text())
    overrides = {"steps": args.steps, "widths": args.widths}
    overrides = {k: v for k, v in overrides.items() if v is not None}
    if args.data_on_cpu:
        overrides["data_on_device"] = False

    rows = []
    for frac in args.label_fracs:
        for seed in args.seeds:
            cfg = TrainConfig(**{**base, **overrides, "split": args.split, "label_frac": frac, "seed": seed})
            row = train_run(cfg, args.processed, args.runs_dir, args.runs_csv, stop_at=args.stop_at, rerun=args.rerun)
            if row is None:
                return
            rows.append(row)

    print(f"\n{'run':40s} {'test mIoU':>9s} {'boundary':>9s} {'core':>9s} {'best val':>9s} {'minutes':>8s}")
    for r in rows:
        print(f"{r['run_id']:40s} {float(r['miou']):9.4f} {float(r['miou_boundary']):9.4f} "
              f"{float(r['miou_core']):9.4f} {float(r['best_val_miou']):9.4f} {float(r['train_seconds']) / 60:8.1f}")


if __name__ == "__main__":
    main()
