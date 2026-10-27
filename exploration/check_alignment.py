from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import rasterio

root = Path(r"G:\Vmi\project")

# 1. 卫星影像：取第 20 次观测，组合 RGB
images = np.load(root / "PASTIS/DATA_S2/S2_10000.npy")
rgb = images[20, [2, 1, 0]].transpose(1, 2, 0).astype(np.float32)

low, high = np.percentile(rgb, [2, 98])
rgb = np.clip((rgb - low) / max(high - low, 1e-6), 0, 1)

# 2. 类别标签与地块编号
labels = np.load(root / "PASTIS/ANNOTATIONS/TARGET_10000.npy")[0]
parcels = np.load(root / "PASTIS/ANNOTATIONS/ParcelIDs_10000.npy")

# 标出相邻像素的地块编号发生变化的位置
edges = np.zeros(parcels.shape, dtype=bool)
edges[1:, :] |= parcels[1:, :] != parcels[:-1, :]
edges[:, 1:] |= parcels[:, 1:] != parcels[:, :-1]

# 透明背景、红色边界
overlay = np.zeros((*edges.shape, 4))
overlay[edges] = [1, 0, 0, 0.7]

# 3. 已对齐的 TESSERA 特征：先显示一个通道
path = root / (
    "TESSERA/patch_10000_2019/"
    "tessera_2019_patch10000_aligned.tif"
)

with rasterio.open(path) as src:
    feature = src.read(1)

vmin, vmax = np.nanpercentile(feature, [2, 98])

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(rgb)
axes[0].imshow(overlay)
axes[0].set_title("Satellite RGB + parcel boundaries")

axes[1].imshow(
    labels, cmap="tab20", vmin=-0.5, vmax=19.5,
    interpolation="nearest"
)
axes[1].set_title("Class labels")

axes[2].imshow(feature, cmap="viridis", vmin=vmin, vmax=vmax)
axes[2].imshow(overlay)
axes[2].set_title("TESSERA channel 0 + parcel boundaries")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()