import json
from pathlib import Path

import numpy as np
import rasterio

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

ROOT = Path(r"G:\Vmi\project")
DATA = ROOT / "TESSERA/batch_2019"

SEED = 42
MAX_TRAIN_PIXELS = 5000

with open(DATA / "selection.json", encoding="utf-8") as f:
    selection = json.load(f)

rng = np.random.default_rng(SEED)

parts = {
    "train": {"X": [], "y": [], "parcels": set()},
    "val": {"X": [], "y": [], "parcels": set()},
    "test": {"X": [], "y": [], "parcels": set()},
}

# ---------- 1. 读取并按 Fold 分组 ----------
for item in selection["patches"]:
    patch_id = item["patch_id"]
    fold = item["fold"]

    split = "train" if fold <= 3 else "val" if fold == 4 else "test"

    with rasterio.open(DATA / f"patch_{patch_id}/aligned.tif") as src:
        features = src.read()

    labels = np.load(
        ROOT / f"PASTIS/ANNOTATIONS/TARGET_{patch_id}.npy"
    )[0]

    parcels = np.load(
        ROOT / f"PASTIS/ANNOTATIONS/ParcelIDs_{patch_id}.npy"
    )

    assert features.shape == (128, *labels.shape)
    assert labels.shape == parcels.shape == (128, 128)

    # 与前一次实验一致：只研究类别 1～18 的地块像素
    mask = (
        np.isfinite(features).all(axis=0)
        & (labels >= 1)
        & (labels <= 18)
        & (parcels > 0)
    )

    X = features.transpose(1, 2, 0)[mask]
    y = labels[mask]

    # 在抽样前检查所有有效地块，防止跨集合重复
    parts[split]["parcels"].update(
        int(value) for value in np.unique(parcels[mask])
    )

    if split == "train" and len(y) > MAX_TRAIN_PIXELS:
        idx = rng.choice(len(y), MAX_TRAIN_PIXELS, replace=False)
        X, y = X[idx], y[idx]

    parts[split]["X"].append(X)
    parts[split]["y"].append(y)

    print(f"{split:5s} | patch {patch_id} | {len(y)} 个像素")

# ---------- 2. 检查地块是否跨集合重复 ----------
for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
    overlap = parts[a]["parcels"] & parts[b]["parcels"]
    if overlap:
        raise ValueError(
            f"{a} 和 {b} 有 {len(overlap)} 个重复地块，"
            "先检查划分，暂不训练。"
        )

for name, part in parts.items():
    part["X"] = np.concatenate(part["X"])
    part["y"] = np.concatenate(part["y"])

    if len(part["y"]) == 0:
        raise ValueError(f"{name} 没有有效样本")

    classes, counts = np.unique(part["y"], return_counts=True)
    print(f"\n{name}：{part['X'].shape}")
    print("类别像素数：", dict(zip(classes.tolist(), counts.tolist())))

train = parts["train"]

# ---------- 3. 训练固定参数的基线 ----------
model = RandomForestClassifier(
    n_estimators=200,
    min_samples_leaf=3,
    class_weight="balanced",
    random_state=SEED,
    n_jobs=-1,
)

print("\n开始训练……", flush=True)
model.fit(train["X"], train["y"])

majority = int(np.bincount(train["y"]).argmax())

# 在看预测结果前，由所选标签确定同一套评价类别
eval_classes = np.unique(
    np.concatenate([part["y"] for part in parts.values()])
)

results = {
    "seed": SEED,
    "max_train_pixels_per_patch": MAX_TRAIN_PIXELS,
    "evaluation_classes": eval_classes.tolist(),
    "class_scope": "IDs 1-18; finite features; parcel ID > 0",
}

# ---------- 4. 验证与测试 ----------
for name in ["val", "test"]:
    part = parts[name]
    y = part["y"]
    pred = model.predict(part["X"])

    unseen = np.setdiff1d(np.unique(y), model.classes_)

    accuracy = accuracy_score(y, pred)
    macro_f1 = f1_score(
        y,
        pred,
        labels=eval_classes,
        average="macro",
        zero_division=0,
    )
    majority_accuracy = float(np.mean(y == majority))

    report = classification_report(
        y,
        pred,
        labels=eval_classes,
        digits=3,
        zero_division=0,
    )

    print(f"\n========== {name} ==========")
    print("训练中未见的类别：", unseen)
    print(f"准确率：{accuracy:.4f}")
    print(f"Macro-F1（统一类别表）：{macro_f1:.4f}")
    print(f"多数类基线准确率：{majority_accuracy:.4f}")
    print(report)

    results[name] = {
        "pixel_count": len(y),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "majority_accuracy": majority_accuracy,
        "unseen_classes": unseen.tolist(),
        "report": report,
        "confusion_matrix": confusion_matrix(
            y, pred, labels=eval_classes
        ).tolist(),
    }

with open(DATA / "baseline_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("\n结果已保存：", DATA / "baseline_results.json")