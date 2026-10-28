"""Boundary and core pixels of the parcels. Used for evaluation only, never as a model input.

A boundary pixel is a parcel pixel at most `width` pixels away from a change of
parcel id (another parcel or background); the core is the rest of the parcel.
The edge of the patch itself is not a parcel boundary. Background pixels belong
to neither region.
"""

import numpy as np
from scipy import ndimage


def id_changes(inst):
    """Pixels with a 4-neighbour of another id (both sides of every change)."""
    inst = np.asarray(inst)
    edge = np.zeros(inst.shape, dtype=bool)
    dx = inst[:, 1:] != inst[:, :-1]
    dy = inst[1:, :] != inst[:-1, :]
    edge[:, 1:] |= dx
    edge[:, :-1] |= dx
    edge[1:, :] |= dy
    edge[:-1, :] |= dy
    return edge


def boundary_core(inst, width=1):
    """(boundary, core) boolean masks of one patch of parcel ids (0 = no parcel)."""
    parcel = np.asarray(inst) > 0
    near = id_changes(inst)
    if width > 1:
        near = ndimage.binary_dilation(near, structure=np.ones((3, 3), bool), iterations=width - 1)
    boundary = near & parcel
    return boundary, parcel & ~boundary
