import os
import re
import csv
import warnings
import numpy as np
from typing import Tuple, List, Dict, Set

from .raytracing import TriangleMesh


def is_clean_ascii(s: str) -> bool:
    """
    Allows no characters other than letters, digits, and underscores.
    """
    return bool(re.fullmatch(r'\w+', s, flags=re.ASCII))


def sanitize_ascii(s: str) -> str:
    """
    Removes any characters other than letters, digits, and underscores and replaces them with underscores.
    Also strips leading and trailing underscores, and collapses runs of multiple underscores.
    """
    return re.sub(r'[\W_]+', '_', s, flags=re.ASCII).strip('_')


def load_all_inputs(folder_path: str) -> Tuple[TriangleMesh, List[str], Dict[str, np.ndarray]]:
    # TODO: Fill out documentation properly.
    """

    Args:
        folder_path:

    Returns:

    """
    mesh, patch_materials = load_mesh(os.path.join(folder_path, 'mesh.obj'))
    material_coefficients = load_materials(os.path.join(folder_path, 'materials.csv'), set(patch_materials))

    return mesh, patch_materials, material_coefficients


def load_mesh(file_path: str) -> Tuple[TriangleMesh, List[str]]:
    # TODO: Fill out documentation properly.
    """

    Args:
        file_path:

    Returns:

    """
    vertex_list = list()
    face_triplet_list = list()
    face_material_list = list()
    current_material = None

    with open(file_path, mode='r') as file:
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

        if not is_clean_ascii(patch_material):
            raise ValueError('Material names should only contain ASCII letters, digits, or underscores.'
                             + ' Bad patch index: ' + str(patch_id) + '; bad material: ' + patch_material)

        patch_ids.append(patch_id)
        if patch_id not in patch_materials_dict.keys():
            patch_materials_dict[patch_id] = patch_material
        elif patch_materials_dict[patch_id] != patch_material:
            raise ValueError('Each patch should only feature a single material.'
                             + ' Bad patch index: ' + str(patch_id) + '; bad material: ' + patch_material)

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


def load_materials(file_path: str, expected_names: Set[str]) -> Dict[str, np.ndarray]:
    # TODO: Fill out documentation properly.
    """

    Args:
        file_path:
        expected_names:

    Returns:

    """
    material_coefficients = dict()

    # Material names will be added to this set when the absorption coefficients are read (first time the name appears in the file)
    # and removed when the scattering coefficients are read (second and last time the name appears in the file).
    expecting_scattering = set()

    with open(file_path, mode='r', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', skipinitialspace=True)

        first_row = next(reader, None)
        if first_row is not None:
            mat_name = first_row.pop(0)
            band_centers = np.array(first_row, dtype=float)
            if mat_name != 'Frequencies':
                raise ValueError('The first row of material.csv should start with the word "Frequencies" and contain the octave band center frequencies.')
            if len(first_row) == 0:
                warnings.warn('No octave band center frequencies are reported in material.csv. Using broadband mode (one band centered at 0).')
                band_centers = np.zeros(1)
            material_coefficients[mat_name] = band_centers
        else:
            raise ValueError('The first row of material.csv should start with the word "Frequencies" and contain the octave band center frequencies.')

        for row in reader:
            if row is None or len(row) == 0:
                continue

            mat_name = row.pop(0)
            coeffs = np.array(row, dtype=float)

            if mat_name not in material_coefficients.keys():
                # This is the first time the material name is encountered in the file. These are the absorption coefficients.
                expecting_scattering.add(mat_name)
                material_coefficients[mat_name] = np.zeros((2, len(band_centers)))

                if len(coeffs) == 1:
                    material_coefficients[mat_name][0] = coeffs[0]
                elif len(coeffs) == len(band_centers):
                    material_coefficients[mat_name][0] = coeffs
                else:
                    raise ValueError('Coefficient rows in material.csv should either contain a single value, or as many as there are octave bands.'
                                     + ' Bad material name: ' + mat_name + '; bad coefficients: ' + str(coeffs))
            elif mat_name in expecting_scattering:
                # This is the second time the material name is encountered in the file. These are the scattering coefficients.
                expecting_scattering.remove(mat_name)

                if len(coeffs) == 1:
                    material_coefficients[mat_name][1] = coeffs[0]
                elif len(coeffs) == len(band_centers):
                    material_coefficients[mat_name][1] = coeffs
                else:
                    raise ValueError('Coefficient rows in material.csv should either contain a single value, or as many as there are octave bands.'
                                     + ' Bad material name: ' + mat_name + '; bad coefficients: ' + str(coeffs))
            else:
                raise ValueError('Each material name should be encountered exactly twice in material.csv.'
                                 + ' This material name appears more than twice: ' + mat_name)

    # Check that expecting_scattering is empty.
    if len(expecting_scattering) != 0:
        raise ValueError('Each material name should be encountered exactly twice in material.csv.'
                         + ' These material names appear only once: ' + expecting_scattering)

    # Check that material_coefficients is not empty.
    if len(material_coefficients) < 2:
        raise ValueError('There should be at least three rows in material.csv.')

    # Check that all expected_names appear in the file.
    missing_names = expected_names.difference(material_coefficients.keys())
    if len(missing_names) != 0:
        raise ValueError('Not all expected material names were found in in material.csv.'
                         + ' These materials were not found: ' + str(missing_names))

    return material_coefficients
