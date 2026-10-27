import json
from pyproj import Transformer

path = r"G:\Vmi\project\PASTIS\metadata.geojson"
ids = {30411, 30708, 20188, 20192, 20131}

with open(path, encoding="utf-8") as f:
    metadata = json.load(f)

for item in metadata["features"]:
    props = item["properties"]
    patch_id = int(props["ID_PATCH"])

    if patch_id not in ids:
        continue

    geometry = item["geometry"]
    tile = props["TILE"].lower()
    zone = int(tile[1:3])

    transformer = Transformer.from_crs(
        "EPSG:2154",
        f"EPSG:{32600 + zone}",
        always_xy=True,
    )

    if geometry["type"] == "Polygon":
        polygons = [geometry["coordinates"]]
    elif geometry["type"] == "MultiPolygon":
        polygons = geometry["coordinates"]
    else:
        print(patch_id, "未知类型：", geometry["type"])
        continue

    print(f"\n区域 {patch_id}，TILE={tile}")
    print("几何类型：", geometry["type"])

    for i, polygon in enumerate(polygons):
        ring = polygon[0]
        print(f"外边界 {i}：{len(ring)} 个顶点")

        for x, y in ring:
            easting, northing = transformer.transform(x, y)
            print(f"  {easting:.3f}, {northing:.3f}")