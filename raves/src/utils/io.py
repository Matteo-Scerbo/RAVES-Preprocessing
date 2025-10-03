import re
import numpy as np
from typing import Tuple

from raytracing import TriangleMesh

def sanitize(s: str) -> str:
    return re.sub(r'[^A-Za-z0-9]+', '_', s).strip('_')


def validate_inputs(path: str) -> bool:
    # TODO: validate OBJ
    # TODO: validate MTL
    # TODO: validate CSV
    # TODO: validate cross OBJ-MTL
    # TODO: validate cross OBJ-CSV

    # TODO: Check that all triangles in each patch have the same normal.

    return True


def load_mesh(path: str) -> Tuple[TriangleMesh, np.ndarray]:
    # TODO: parse OBJ
    vertices = np.zeros((0, 3), dtype=float)
    vert_triplets = np.zeros((0, 3), dtype=int)
    patch_ids = np.zeros(0, dtype=int)
    patch_materials = ['']

    mesh = TriangleMesh(vertices, vert_triplets, patch_ids)

    return mesh, patch_materials
