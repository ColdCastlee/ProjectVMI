import json
import random
import shutil
import subprocess
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling

ROOT = Path(r"G:\Vmi\project")
OUT = ROOT / "TESSERA" / "batch_2019"
OUT.mkdir(parents=True, exist_ok=True)

YEAR = 2019
PER_FOLD = 5
SEED = 42

with open(ROOT / "PASTIS/metadata.geojson", encoding="utf-8") as f:
    metadata = json.load(f)

# ---------- 1. 固定随机选择，每个 Fold 5 个区域 ----------
rng = random.Random(SEED)
selected = []

for fold in range(1, 6):
    candidates = sorted(
        [
            item for item in metadata["features"]
            if int(item["properties"]["Fold"]) == fold
        ],
        key=lambda item: int(item["properties"]["ID_PATCH"]),
    )
    selected.extend(rng.sample(candidates, PER_FOLD))

# 保存选择清单，后续训练使用同一批区域
manifest = {
    "year": YEAR,
    "seed": SEED,
    "patches": [
        {
            "patch_id": int(item["properties"]["ID_PATCH"]),
            "fold": int(item["properties"]["Fold"]),
        }
        for item in selected
    ],
}

with open(OUT / "selection.json", "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

cli = shutil.which("geotessera")
if cli is None:
    raise RuntimeError(
        "未找到 geotessera，请使用安装了它的 pytorch 环境运行。"
    )

print("共选择", len(selected), "个区域")
for item in manifest["patches"]:
    print(item)

# ---------- 2. 逐个下载并对齐 ----------
failures = []

for index, item in enumerate(selected, start=1):
    props = item["properties"]
    patch_id = int(props["ID_PATCH"])
    fold = int(props["Fold"])

    folder = OUT / f"patch_{patch_id}"
    folder.mkdir(exist_ok=True)

    output_path = folder / "aligned.tif"

    print(
        f"\n[{index}/{len(selected)}] "
        f"区域 {patch_id}，Fold {fold}",
        flush=True,
    )

    try:
        # 已完成的区域直接跳过，允许中断后重新运行
        if output_path.exists():
            with rasterio.open(output_path) as src:
                if (src.count, src.height, src.width) != (128, 128, 128):
                    raise ValueError("已有 aligned.tif 的形状不正确")
            print("已完成，跳过")
            continue

        # 确认有对应的类别标签
        label_path = (
            ROOT / "PASTIS/ANNOTATIONS" / f"TARGET_{patch_id}.npy"
        )
        labels = np.load(label_path)[0]
        if labels.shape != (128, 128):
            raise ValueError(f"标签形状异常：{labels.shape}")

        geometry = item["geometry"]

        # 两种几何类型的坐标嵌套层数不同
        if geometry["type"] == "Polygon":
            ring = geometry["coordinates"][0]

        elif geometry["type"] == "MultiPolygon":
            if len(geometry["coordinates"]) != 1:
                raise ValueError("该区域包含多个多边形，需要单独检查")
            ring = geometry["coordinates"][0][0]

        else:
            raise ValueError(
                f"不支持的几何类型：{geometry['type']}"
            )

        print("几何类型：", geometry["type"])
        print("边界顶点数：", len(ring))

        if len(ring) != 5:
            raise ValueError("该区域不是预期的四角边界")

        tile = props["TILE"].lower()
        zone = int(tile[1:3])
        target_crs = f"EPSG:{32600 + zone}"

        to_utm = Transformer.from_crs(
            "EPSG:2154", target_crs, always_xy=True
        )

        corners = np.array([
            to_utm.transform(x, y)
            for x, y in ring[:4]
        ])

        left, top = corners[0]

        expected = np.array([
            [left, top],
            [left + 1280, top],
            [left + 1280, top - 1280],
            [left, top - 1280],
        ])

        if not np.allclose(corners, expected, atol=0.1, rtol=0):
            raise ValueError("四角不符合预期网格，需要单独检查")

        target_transform = from_origin(left, top, 10, 10)

        # 用完整边界计算下载的经纬度范围
        to_lonlat = Transformer.from_crs(
            "EPSG:2154", "EPSG:4326", always_xy=True
        )

        # 加密边界，避免投影变换后遗漏边缘
        boundary = []
        for a, b in zip(ring[:-1], ring[1:]):
            for fraction in np.linspace(0, 1, 21):
                x = a[0] + fraction * (b[0] - a[0])
                y = a[1] + fraction * (b[1] - a[1])
                boundary.append(to_lonlat.transform(x, y))

        lons, lats = zip(*boundary)

        # 稍微扩大下载范围，为插值保留边缘
        margin = 0.0003
        bbox = [
            min(lons) - margin,
            min(lats) - margin,
            max(lons) + margin,
            max(lats) + margin,
        ]
        bbox_text = ",".join(map(str, bbox))

        raw_folder = folder / "raw"
        raw_folder.mkdir(exist_ok=True)

        subprocess.run(
            [
                cli,
                "download",
                f"--bbox={bbox_text}",
                "--year", str(YEAR),
                "--output", str(raw_folder),
            ],
            check=True,
        )

        tif_files = list(raw_folder.glob("*.tif"))
        if len(tif_files) != 1:
            raise ValueError(
                f"下载得到 {len(tif_files)} 个 TIFF，需要单独检查"
            )

        # ---------- 3. 地理对齐 ----------
        with rasterio.open(tif_files[0]) as src:
            if src.count != 128:
                raise ValueError(f"特征通道数异常：{src.count}")

            aligned = np.full(
                (128, 128, 128), np.nan, dtype=np.float32
            )

            for band in range(src.count):
                reproject(
                    source=rasterio.band(src, band + 1),
                    destination=aligned[band],
                    src_transform=src.transform,
                    src_crs=src.crs,
                    src_nodata=src.nodata,
                    dst_transform=target_transform,
                    dst_crs=target_crs,
                    dst_nodata=np.nan,
                    resampling=Resampling.bilinear,
                )

        valid = np.isfinite(aligned).all(axis=0)
        print(f"有效像素比例：{valid.mean():.2%}")

        if not valid.any():
            raise ValueError("对齐后没有有效像素")

        # 先写临时文件，写完再改名，避免把半成品当成成功
        temp_path = folder / "aligned.tmp.tif"

        with rasterio.open(
            temp_path,
            "w",
            driver="GTiff",
            height=128,
            width=128,
            count=128,
            dtype="float32",
            crs=target_crs,
            transform=target_transform,
            nodata=np.nan,
            compress="deflate",
        ) as dst:
            dst.write(aligned)

        temp_path.replace(output_path)
        print("完成：", output_path, flush=True)

    except Exception as exc:
        print("失败：", exc, flush=True)
        failures.append({
            "patch_id": patch_id,
            "fold": fold,
            "error": str(exc),
        })

with open(OUT / "failures.json", "w", encoding="utf-8") as f:
    json.dump(failures, f, indent=2, ensure_ascii=False)

print("\n批量处理结束")
print("成功或已完成：", len(selected) - len(failures))
print("失败：", len(failures))
print("输出目录：", OUT)