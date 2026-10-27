from pathlib import Path

import numpy as np
import rasterio
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, classification_report

root = Path(r"G:\Vmi\project")

# ---------- 1. 读取数据 ----------
feature_path = root / (
    "TESSERA/patch_10000_2019/"
    "tessera_2019_patch10000_aligned.tif"
)

with rasterio.open(feature_path) as src:
    features = src.read()  # (特征数, 高, 宽)

labels = np.load(
    root / "PASTIS/ANNOTATIONS/TARGET_10000.npy"
)[0]

parcels = np.load(
    root / "PASTIS/ANNOTATIONS/ParcelIDs_10000.npy"
)

assert features.shape[1:] == labels.shape == parcels.shape

# 这次先只研究编号 1～18 的作物类别
mask = (
    np.isfinite(features).all(axis=0)
    & (labels >= 1)
    & (labels <= 18)
    & (parcels > 0)
)

# 每行是一名样本：一个像素的 128 维特征
X = features.transpose(1, 2, 0)[mask]
y = labels[mask]
groups = parcels[mask]

print("输入 X：", X.shape)
print("标签 y：", y.shape)
print("地块数量：", len(np.unique(groups)))

# ---------- 2. 按地块划分 ----------
# 约 25% 的地块留作测试，不是 25% 的像素
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.25,
    random_state=42,
)

train_idx, test_idx = next(splitter.split(X, y, groups))

assert set(groups[train_idx]).isdisjoint(groups[test_idx])

print("训练像素：", len(train_idx))
print("测试像素：", len(test_idx))
print("训练类别：", np.unique(y[train_idx]))
print("测试类别：", np.unique(y[test_idx]))

unseen = np.setdiff1d(
    np.unique(y[test_idx]), np.unique(y[train_idx])
)
print("测试中出现、训练中没有的类别：", unseen)

# ---------- 3. 训练 ----------
model = RandomForestClassifier(
    n_estimators=200,
    min_samples_leaf=3,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

print("\n开始训练……")
model.fit(X[train_idx], y[train_idx])

# ---------- 4. 只在测试地块上评价 ----------
pred = model.predict(X[test_idx])

print("\n测试准确率：",
      f"{accuracy_score(y[test_idx], pred):.4f}")
print("测试 Macro-F1：",
      f"{f1_score(y[test_idx], pred, average='macro', zero_division=0):.4f}")

print("\n各类别结果：")
print(classification_report(
    y[test_idx], pred, digits=3, zero_division=0
))

# 多数类基线：全部预测为训练集中最多的类别
majority = np.bincount(y[train_idx]).argmax()
majority_pred = np.full(len(test_idx), majority)

print("多数类基线准确率：",
      f"{accuracy_score(y[test_idx], majority_pred):.4f}")

# ---------- 5. 仅显示测试像素 ----------
truth_map = np.full(labels.shape, np.nan)
pred_map = np.full(labels.shape, np.nan)

rows, cols = np.where(mask)
test_rows, test_cols = rows[test_idx], cols[test_idx]

truth_map[test_rows, test_cols] = y[test_idx]
pred_map[test_rows, test_cols] = pred

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

for ax, image, title in zip(
    axes,
    [truth_map, pred_map],
    ["Test parcels: ground truth", "Test parcels: prediction"],
):
    ax.imshow(
        image, cmap="tab20",
        vmin=-0.5, vmax=19.5,
        interpolation="nearest",
    )
    ax.set_title(title)
    ax.axis("off")

plt.tight_layout()
plt.show()