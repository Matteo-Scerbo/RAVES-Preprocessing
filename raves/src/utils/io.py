import re
import numpy as np
from typing import Tuple

from .raytracing import TriangleMesh

def sanitize(s: str) -> str:
    return re.sub(r'[^A-Za-z0-9]+', '_', s).strip('_')


def validate_inputs(folder_path: str) -> bool:
    try:
        mesh, patch_materials = load_mesh(folder_path + '/mesh.obj')
    except ValueError as e:
        print('Failed loading mesh. Error:', e)
        return False

    # TODO: Validate CSV.
    # TODO: Cross-validate OBJ-CSV.

    return True


def load_mesh(file_path: str) -> Tuple[TriangleMesh, np.ndarray]:
    vertex_list = list()
    face_triplet_list = list()
    face_material_list = list()
    current_material = None

    with open(file_path, 'r') as file:
        for line_idx, line in enumerate(file):
            if line_idx == 0 and line != 'mtllib mesh.mtl\n':
                raise ValueError('The first line of `mesh.obj` should be `mtllib mesh.mtl`. Instead, it is'
                                 + '\n\t' + line)

            # If there is a comment in this line, remove it (i.e. remove everything that follows a '#').
            comment_start = line.find('#')
            if comment_start != -1:
                line = line[:comment_start]

            # Separate the line into words.
            # Note that the default separator is any whitespace (including '\t', etc.)
            split_line = line.split()

            if len(split_line) == 0:
                # Ignore empty lines.
                continue

            if split_line[0] == 'v':
                if len(split_line) == 5:
                    print('`w` coordinates are ignored.')
                    split_line = split_line[:-1]

                if len(split_line) != 4:
                    raise ValueError('All vertex coordinates must have three dimensions.'
                                     + ' Bad line index: ' + str(line_idx) + ', bad line:\n\t' + line)

                vertex_list.append([float(c) for c in split_line[1:]])

            elif split_line[0] == 'usemtl':
                if len(split_line) != 2:
                    raise ValueError('`usemtl` lines should have only two words.'
                                     + ' Bad line index: ' + str(line_idx) + ', bad line:\n\t' + line)

                current_material = split_line[1]

            elif split_line[0] == 'f':
                if current_material is None:
                    raise ValueError('Face declaration encountered before material declaration.'
                                     + ' Bad line index: ' + str(line_idx) + ', bad line:\n\t' + line)

                if len(split_line) != 4:
                    raise ValueError('All faces must have three vertices (triangles only).'
                                     + ' Bad line index: ' + str(line_idx) + ', bad line:\n\t' + line)

                face_triplet_list.append([int(c) for c in split_line[1:]])
                face_material_list.append(current_material)

    vertices = np.array(vertex_list, dtype=float)
    vert_triplets = np.array(face_triplet_list, dtype=int)

    if np.any(vert_triplets < 1):
        raise ValueError('Vertex indices should start from 1.')
    if np.any(vert_triplets > vertices.shape[0]):
        raise ValueError('Vertex index out of bounds.')
    # Convert to 0-indexing.
    vert_triplets -= 1

    # Parse OBJ material names.
    patch_ids = list()
    patch_materials_dict = dict()
    for face_material in face_material_list:
        match = re.match(r'Patch_(\d+)_Mat_(.+)', face_material)
        patch_id = int(match.group(1)) - 1  # Convert to 0-indexing.
        patch_material = match.group(2)
        patch_ids.append(patch_id)
        if patch_id not in patch_materials_dict.keys():
            patch_materials_dict[patch_id] = patch_material
        elif patch_materials_dict[patch_id] != patch_material:
            raise ValueError('Each patch should only feature a single material.'
                             + ' Bad patch index: ' + str(patch_id))

    patch_ids = np.array(patch_ids, dtype=int)
    # Check that patch_ids is a proper range.
    if np.min(patch_ids) != 0 or np.max(patch_ids) != len(patch_materials_dict) - 1:
        raise ValueError('The patch indices should form a contiguous range.'
                         + ' Min ID: ' + str(np.min(patch_ids))
                         + ' Max ID: ' + str(np.max(patch_ids))
                         + ' Num ID: ' + str(len(patch_materials_dict)))

    patch_materials = [patch_materials_dict[i] for i in range(len(patch_materials_dict))]

    mesh = TriangleMesh(vertices, vert_triplets, patch_ids)

    # TODO: Check that all triangles in each patch have the same normal.

    # TODO: Validate `mesh.mtl`.
    # TODO: Cross-validate OBJ-MTL.

    return mesh, patch_materials
