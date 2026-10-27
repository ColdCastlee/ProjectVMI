"""Pixel grid of the PASTIS patches.

metadata.geojson stores each patch footprint in Lambert-93 (EPSG:2154), but the
pixels live on the UTM grid of their Sentinel-2 tile (EPSG 326NN, NN = the zone
in the tile name, as in prepare_batch.py of the RF pilot). Re-projected to that
zone, every footprint is an axis-aligned 1280 m square (corners within 0.1 m)
whose top-left corner sits a few metres off the 10 m lattice, with a constant
offset per tile. Snapping that corner to the nearest multiple of 10 m gives the
grid that matches the TESSERA pixels exactly (shift test on one patch per tile).
The raw corner is kept as x0_raw / y1_raw: it is the origin used by the
bilinear alignment of the RF pilot.
"""

import geopandas as gpd
import numpy as np
from pyproj import Transformer

PIXEL = 10.0
SIZE = 128
CORNER_TOL = 0.1  # metres, same tolerance as prepare_batch.py


def tile_epsg(tile):
    """'t30uxv' -> 32630 (northern-hemisphere UTM zone of a Sentinel-2 tile)."""
    return 32600 + int(str(tile).lower()[1:3])


def load_metadata(path_or_file):
    """Read metadata.geojson, sorted by ID_PATCH like the official dataloader."""
    meta = gpd.read_file(path_or_file)
    return meta.sort_values("ID_PATCH").reset_index(drop=True)


def _check_square(geom, x0, y1, side):
    """The footprint must be one polygon whose 4 corners are those of the square."""
    if geom.geom_type == "MultiPolygon":
        if len(geom.geoms) != 1:
            return False
        geom = geom.geoms[0]
    corners = np.asarray(geom.exterior.coords)[:-1]
    if len(corners) != 4:
        return False
    expected = np.array([[x0, y1], [x0 + side, y1], [x0 + side, y1 - side], [x0, y1 - side]])
    dist = np.abs(corners[:, None, :] - expected[None, :, :]).max(axis=2)
    return bool((dist.min(axis=1) <= CORNER_TOL).all())


def add_utm_grid(meta, margin_px=5):
    """Add the UTM grid of every patch and a WGS84 bbox (with margin) for queries.

    New columns: epsg; x0_raw, y1_raw (top-left corner of the footprint, metres);
    x0, y1 (same corner snapped to 10 m: the grid used here); snap_dx, snap_dy
    (x0 - x0_raw, y1 - y1_raw); lon_min, lat_min, lon_max, lat_max (bbox of the
    128 x 128 window grown by margin_px pixels).
    """
    meta = meta.copy()
    meta["TILE"] = meta["TILE"].str.lower()
    meta["epsg"] = meta["TILE"].map(tile_epsg).astype(int)
    side = SIZE * PIXEL
    m = margin_px * PIXEL
    for epsg, sub in meta.groupby("epsg"):
        utm = sub.geometry.to_crs(epsg)
        b = utm.bounds
        ok = [_check_square(g, x, y, side) for g, x, y in zip(utm, b.minx, b.maxy)]
        meta.loc[sub.index, "footprint_ok"] = ok
        if not all(ok):
            bad = sub.ID_PATCH[[not v for v in ok]].tolist()
            print(f"WARNING EPSG:{epsg}: {len(bad)} footprints are not clean {side:.0f} m squares "
                  f"(flag footprint_ok=False): {bad[:10]}")
        x0 = np.round(b.minx.values / PIXEL) * PIXEL
        y1 = np.round(b.maxy.values / PIXEL) * PIXEL
        meta.loc[sub.index, "x0_raw"] = b.minx.values
        meta.loc[sub.index, "y1_raw"] = b.maxy.values
        meta.loc[sub.index, "x0"] = x0
        meta.loc[sub.index, "y1"] = y1
        meta.loc[sub.index, "snap_dx"] = x0 - b.minx.values
        meta.loc[sub.index, "snap_dy"] = y1 - b.maxy.values

        # Four corners of the enlarged window, then their lon/lat envelope.
        xs = np.stack([x0 - m, x0 + side + m, x0 - m, x0 + side + m], axis=1)
        ys = np.stack([y1 + m, y1 + m, y1 - side - m, y1 - side - m], axis=1)
        to_wgs84 = Transformer.from_crs(epsg, 4326, always_xy=True)
        lon, lat = to_wgs84.transform(xs.ravel(), ys.ravel())
        lon, lat = np.reshape(lon, xs.shape), np.reshape(lat, ys.shape)
        meta.loc[sub.index, "lon_min"] = lon.min(axis=1)
        meta.loc[sub.index, "lat_min"] = lat.min(axis=1)
        meta.loc[sub.index, "lon_max"] = lon.max(axis=1)
        meta.loc[sub.index, "lat_max"] = lat.max(axis=1)
    return meta
