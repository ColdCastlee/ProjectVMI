from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling

folder = Path(r"G:\Vmi\project\TESSERA\patch_10000_2019")
source_path = folder / "tessera_2019_utm30.tif"
output_path = folder / "tessera_2019_patch10000_aligned.tif"

# 根据刚才转换出的左上角，建立 PASTIS 目标网格
# 左上角 X、左上角 Y、像素宽度、像素高度
target_transform = from_origin(
    624892.364, 5499520.621, 10, 10
)

with rasterio.open(source_path) as src:
    aligned = np.full((src.count, 128, 128), np.nan, dtype=np.float32)

    for i in range(src.count):
        reproject(
            source=rasterio.band(src, i + 1),
            destination=aligned[i],
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=np.nan,
            dst_transform=target_transform,
            dst_crs="EPSG:32630",
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )

    profile = src.profile.copy()
    profile.update(
        height=128,
        width=128,
        transform=target_transform,
        crs="EPSG:32630",
        dtype="float32",
        nodata=np.nan,
        compress="deflate",
    )

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(aligned)

valid = np.isfinite(aligned).all(axis=0)

print("对齐后的特征形状：", aligned.shape)
print("有效像素比例：", f"{valid.mean():.2%}")
print("保存位置：", output_path)