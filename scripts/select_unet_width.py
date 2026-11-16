"""Choice of the UNet width on the validation patches of a split (the test patches are never read).

Rule fixed before the runs. For every label fraction, the per-patch validation
matrices of the kept weights (cm_val_per_patch.npz) give the mean validation
mIoU over the seeds, and the paired patch bootstrap of
delta = mIoU(wide) - mIoU(narrow), averaged over the seeds with the same patch
draws. The wide UNet is chosen only if the 95% interval of delta is above 0 for
at least one label fraction and entirely below 0 for none; otherwise the narrow
one (4x fewer parameters, about 2x faster) is kept.

Outputs in --out: unet_width_<split>.json (the decision), unet_width_<split>_val.csv
(one line per run) and unet_width_<split>_val_curves.png.

Example:
    python scripts/select_unet_width.py --runs-dir /content/drive/MyDrive/pj_vmi/runs \
        --out /content/drive/MyDrive/pj_vmi/results/baselines
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.shards import code_version
from src.data.splits import split_tag
from src.eval.bootstrap import bootstrap_miou_seeds, paired_delta_seeds
from src.eval.metrics import miou
from src.eval.runs import load_cms, run_name

RULE = ("wide width only if the paired validation delta (wide - narrow, mean over seeds) has its 95% interval "
        "above 0 for at least one label fraction and entirely below 0 for none; otherwise the narrow width")


def choose(deltas):
    """True if the wide width is chosen; deltas: one dict (ci_low, ci_high) per label fraction."""
    better = any(d["ci_low"] > 0 for d in deltas)
    worse = any(d["ci_high"] < 0 for d in deltas)
    return better and not worse


def widths_of(first):
    return [first, 2 * first, 4 * first]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="I")
    ap.add_argument("--prefix", default="base")
    ap.add_argument("--variant", default="unet-t19")
    ap.add_argument("--widths", type=int, nargs=2, default=[32, 64], help="first width of the narrow and wide UNet")
    ap.add_argument("--label-fracs", type=float, nargs="+", default=[1.0, 0.01])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    runs_dir, out = Path(args.runs_dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = split_tag(args.split)
    narrow, wide = args.widths

    def run_dir(w, frac, seed):
        return runs_dir / run_name(args.prefix, f"{args.variant}-w{w}", frac, tag, seed)

    rows, cms, curves = [], {}, {}
    for w in (narrow, wide):
        for frac in args.label_fracs:
            cms[w, frac], curves[w, frac] = [], []
            for seed in args.seeds:
                d = run_dir(w, frac, seed)
                info = json.loads((d / "run.json").read_text())
                idx, cm = load_cms(d, part="val")
                if cms[w, frac] and not np.array_equal(idx, ref_idx):
                    raise ValueError(f"{d.name}: validation patches differ between runs")
                ref_idx = idx
                cms[w, frac].append(cm)
                h = pd.DataFrame(info["history"]).set_index("step")["val_miou"]
                curves[w, frac].append(h)
                rows.append({"width": w, "label_frac": frac, "seed": seed, "run_id": d.name,
                             "val_miou": float(miou(cm.sum(axis=0))), "best_val_miou": info["best_val_miou"],
                             "best_step": info["best_step"], "final_val_miou": float(h.iloc[-1]),
                             "n_params": info["n_params"], "train_minutes": info["train_seconds"] / 60})
    per_run = pd.DataFrame(rows)
    per_run.to_csv(out / f"unet_width_{tag}_val.csv", index=False)
    gap = (per_run.val_miou - per_run.best_val_miou).abs().max()

    per_frac = []
    for frac in args.label_fracs:
        a = bootstrap_miou_seeds(cms[narrow, frac], n_boot=args.n_boot)
        b = bootstrap_miou_seeds(cms[wide, frac], n_boot=args.n_boot)
        d = paired_delta_seeds(cms[narrow, frac], cms[wide, frac], n_boot=args.n_boot)
        per_frac.append({"label_frac": frac,
                         f"val_miou_w{narrow}": a["miou"], f"std_seeds_w{narrow}": a["std_seeds"],
                         f"val_miou_w{wide}": b["miou"], f"std_seeds_w{wide}": b["std_seeds"],
                         "delta": d["delta"], "ci_low": d["ci_low"], "ci_high": d["ci_high"],
                         "share_le_0": d["share_le_0"], "per_seed": d["per_seed"]})
    chosen = wide if choose(per_frac) else narrow

    decision = {"split": tag, "variant": args.variant, "seeds": args.seeds, "label_fracs": args.label_fracs,
                "candidates": {f"w{narrow}": widths_of(narrow), f"w{wide}": widths_of(wide)},
                "n_params": {f"w{w}": int(per_run[per_run.width == w].n_params.iloc[0]) for w in (narrow, wide)},
                "data": "validation patches only", "rule": RULE, "per_label_frac": per_frac,
                "chosen_width": widths_of(chosen), "chosen_variant": f"{args.variant}-w{chosen}",
                "commit": code_version(), "decided_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (out / f"unet_width_{tag}.json").write_text(json.dumps(decision, indent=2))

    fig, axes = plt.subplots(1, len(args.label_fracs), figsize=(6 * len(args.label_fracs), 3.8), squeeze=False)
    for ax, frac in zip(axes[0], args.label_fracs):
        for w in (narrow, wide):
            mat = pd.concat(curves[w, frac], axis=1)
            mean, std = mat.mean(axis=1), mat.std(axis=1)
            ax.plot(mat.index, mean, label=f"{'-'.join(map(str, widths_of(w)))}")
            ax.fill_between(mat.index, mean - std, mean + std, alpha=0.2)
        ax.set(xlabel="step", ylabel="validation mIoU",
               title=f"{args.variant}, {frac:.0%} labels (mean and std over {len(args.seeds)} seeds)")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / f"unet_width_{tag}_val_curves.png", dpi=120)
    plt.close(fig)

    pd.set_option("display.width", 200)
    print(per_run.drop(columns="run_id").round(4).to_string(index=False))
    print(f"\nlargest |validation mIoU from the matrices - best during training|: {gap:.4f}")
    print(f"\npaired bootstrap on the validation patches, delta = w{wide} - w{narrow}, mean over seeds:")
    print(pd.DataFrame(per_frac).drop(columns="per_seed").round(4).to_string(index=False))
    print(f"\nchosen: {decision['chosen_variant']} (widths {decision['chosen_width']})")
    print(f"saved in {out}")


if __name__ == "__main__":
    main()
