"""Read PASTIS files from PASTIS.zip (without extracting 39 GB) or from an extracted folder.

Both give the same interface, with paths relative to the PASTIS root, e.g.
"ANNOTATIONS/TARGET_10000.npy". A folder can be the full dataset or a subset
written by scripts/extract_subset.py.
"""

import io
import zipfile
from pathlib import Path

import numpy as np

from .pastis_grid import load_metadata


def _header(f):
    major, _ = np.lib.format.read_magic(f)
    if major == 1:
        shape, _, dtype = np.lib.format.read_array_header_1_0(f)
    else:
        shape, _, dtype = np.lib.format.read_array_header_2_0(f)
    return shape, dtype


class PastisZip:
    def __init__(self, path, root="PASTIS"):
        self.zf = zipfile.ZipFile(path)
        self.root = root
        self.names = set(self.zf.namelist())

    def _member(self, rel):
        return f"{self.root}/{rel}"

    def has(self, rel):
        return self._member(rel) in self.names

    def raw(self, rel):
        """Bytes of a file, to copy it unchanged."""
        return self.zf.read(self._member(rel))

    def npy(self, rel):
        return np.load(io.BytesIO(self.raw(rel)))

    def npy_header(self, rel):
        """(shape, dtype) of a .npy file, reading only its header."""
        with self.zf.open(self._member(rel)) as f:
            return _header(f)

    def metadata(self):
        return load_metadata(io.BytesIO(self.raw("metadata.geojson")))


class PastisDir:
    def __init__(self, path):
        path = Path(path)
        self.root = path / "PASTIS" if (path / "PASTIS").is_dir() else path

    def has(self, rel):
        return (self.root / rel).exists()

    def raw(self, rel):
        return (self.root / rel).read_bytes()

    def npy(self, rel):
        return np.load(self.root / rel)

    def npy_header(self, rel):
        with open(self.root / rel, "rb") as f:
            return _header(f)

    def metadata(self):
        return load_metadata(self.root / "metadata.geojson")


def open_pastis(path):
    """PASTIS.zip or a folder (containing PASTIS/ or the PASTIS files themselves)."""
    return PastisDir(path) if Path(path).is_dir() else PastisZip(path)
