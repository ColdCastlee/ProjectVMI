import json
from collections import Counter

path = r"G:\Vmi\project\PASTIS\metadata.geojson"

with open(path, "r", encoding="utf-8") as f:
    metadata = json.load(f)

features = metadata["features"]

counts = Counter(
    int(item["properties"]["Fold"])
    for item in features
)

print("每个 Fold 的区域数量：")
for fold, count in sorted(counts.items()):
    print(f"Fold {fold}: {count} 个区域")

print("\n每个 Fold 的前 5 个区域编号：")
for fold in sorted(counts):
    ids = sorted(
        int(item["properties"]["ID_PATCH"])
        for item in features
        if int(item["properties"]["Fold"]) == fold
    )
    print(f"Fold {fold}: {ids[:5]}")