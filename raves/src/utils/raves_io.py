import os
import re
import csv
import warnings
import numpy as np
import networkx as nx
from itertools import combinations
from collections import defaultdict
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


def merge_small_patches(vert_triplets: np.ndarray,
                        mesh: TriangleMesh,
                        patch_materials: List[str],
                        area_threshold: float,
                        ) -> None:
    # TODO: Fill out documentation properly.
    """
    patch_materials is modified in-place (elements are removed).
    Patch IDs are modified in-place inside "mesh".
    """

    # Compute patch_areas
    patch_areas = np.zeros(len(patch_materials))
    for p, a in zip(mesh.ID, mesh.area):
        patch_areas[p] += a

    group_partitions = list()
    for material in set(patch_materials):
        child_patches = set(p for p, m in enumerate(patch_materials) if m == material)

        # if len(child_patches) < 2:
        #     # Nothing to merge in this patch.
        #     # TODO: Add the default partition to group_partitions.
        #     continue
        #
        # if np.all([patch_areas[p] >= area_threshold for p in child_patches]):
        #     # Nothing to merge in this patch.
        #     # TODO: Add the default partition to group_partitions.
        #     continue

        # Build patch adjacency graph based on shared edges and co-planarity.
        edge_to_patches = defaultdict(set)
        child_triangles = np.where(np.isin(mesh.ID, list(child_patches)))[0]
        for t_idx in child_triangles:
            a, b, c = (int(x) for x in vert_triplets[t_idx])
            for u, v in ((a, b), (b, c), (c, a)):
                if v > u:
                    edge_to_patches[(u, v)].add(mesh.ID[t_idx])
                else:
                    edge_to_patches[(v, u)].add(mesh.ID[t_idx])

        G = nx.Graph()
        G.add_nodes_from(child_patches)
        for patches in edge_to_patches.values():
            assert len(patches.difference(child_patches)) == 0
            if len(patches) == 1:
                # This edge only pertains to one patch.
                continue
            for i, j in combinations(patches, 2):
                any_triangle_in_i = np.argwhere(mesh.ID == i).flatten()[0]
                any_triangle_in_j = np.argwhere(mesh.ID == j).flatten()[0]

                parallel_normals = np.isclose(np.dot(mesh.n[any_triangle_in_i],
                                                     mesh.n[any_triangle_in_j]),
                                              1.)
                same_offset = np.isclose(mesh.d0[any_triangle_in_i],
                                         mesh.d0[any_triangle_in_j])

                if parallel_normals and same_offset:
                    G.add_edge(i, j)

        # if nx.number_connected_components(G) == len(child_patches):
        #     # Nothing to merge in this patch.
        #     # TODO: Add the default partition to group_partitions.
        #     continue

        for comp_nodes in nx.connected_components(G):
            subG = G.subgraph(comp_nodes).copy()

            # Trivial partition: each patch alone (as frozensets, so the partition is hashable).
            trivial_partition = frozenset(frozenset([p]) for p in subG.nodes())

            # Build the full set of all legal partitions by iterative merging (only merge clusters that are adjacent in subG).
            legal_partitions = {trivial_partition}
            frontier = [trivial_partition]

            while len(frontier) > 0:
                clusters = list(frontier.pop())
                cluster_areas = [np.sum([patch_areas[a] for a in A]) for A in clusters]

                # Try all unordered pairs (A, B) once.
                for A_i in range(len(clusters)):
                    for B_i in range(len(clusters)):
                        if A_i == B_i:
                            continue

                        A = clusters[A_i]
                        B = clusters[B_i]

                        # If both clusters are already large, do not consider merging them.
                        if (cluster_areas[A_i] >= area_threshold) and (cluster_areas[B_i] >= area_threshold):
                            continue

                        # Check if the pair of clusters can be merged.
                        any_adjacent = False
                        for a in A:
                            if any((b in B) for b in subG.neighbors(a)):
                                any_adjacent = True
                                break
                        if not any_adjacent:
                            continue

                        new_part = {A.union(B)}.union(clusters[k]
                                                      for k in range(len(clusters))
                                                      if k not in (A_i, B_i))
                        new_part = frozenset(new_part)
                        if new_part not in legal_partitions:
                            legal_partitions.add(new_part)
                            frontier.append(new_part)

            if len(legal_partitions) == 1:
                group_partitions.append(list(legal_partitions)[0])
                continue

            # If at least one partition has A_min >= threshold:
            #    keep only partitions with A_min >= threshold
            # else:
            #    keep the partition(s) with the greatest A_min
            A_min = {partition:
                     np.min([np.sum([patch_areas[p] for p in cluster])
                             for cluster in partition])
                     for partition in legal_partitions}

            if np.any([a >= area_threshold for a in A_min.values()]):
                legal_partitions = [p for p, a in A_min.items()
                                    if a >= area_threshold]
            else:
                max_A_min = np.max([a for a in A_min.values()])
                legal_partitions = [p for p, a in A_min.items()
                                    if np.isclose(a, max_A_min)]

            if len(legal_partitions) == 1:
                group_partitions.append(legal_partitions[0])
                continue

            # If there is still more than one legal partition, choose one with the lowest maximum area.
            A_max = {partition:
                     np.max([np.sum([patch_areas[p] for p in cluster])
                             for cluster in partition])
                     for partition in legal_partitions}
            min_A_max = np.min([a for a in A_max.values()])
            legal_partitions = [p for p, a in A_max.items()
                                if np.isclose(a, min_A_max)]

            group_partitions.append(legal_partitions[0])

            # TODO: Choose one which maximizes the ratio of area over perimeter.
            #  R_min = {partition:
            #           np.min([subset_area(cluster) / subset_perimeter(cluster)
            #                   for cluster in partition])
            #           for partition in legal_partitions}

    full_cover = list()
    for partition in group_partitions:
        for merged_ids in partition:
            full_cover.extend(merged_ids)
    assert len(full_cover) == len(patch_materials), str((len(full_cover), len(patch_materials)))

    # Perform the chosen merging.
    kept_patch_ids = list()
    id_mapping = -np.ones(len(patch_materials), dtype=int)
    for partition in group_partitions:
        for merged_ids in partition:
            merged_ids = sorted(merged_ids)
            kept_patch_ids.append(merged_ids[0])

            id_mapping[merged_ids] = merged_ids[0]

    assert np.all(id_mapping >= 0), str(np.argwhere(id_mapping < 0))

    _, inverse_mapping = np.unique(id_mapping, return_inverse=True)

    # Remap materials (the [:] is what makes this in-place)
    mesh.ID[:] = inverse_mapping[mesh.ID]
    patch_materials[:] = [patch_materials[i] for i in kept_patch_ids]

    # Confirm patch_areas
    new_patch_areas = np.zeros(len(patch_materials))
    for p, a in zip(mesh.ID, mesh.area):
        new_patch_areas[p] += a


def load_all_inputs(folder_path: str, area_threshold: float = 0.) -> Tuple[TriangleMesh, List[str], Dict[str, np.ndarray]]:
    # TODO: Fill out documentation properly.
    """

    Args:
        folder_path:
        area_threshold:

    Returns:

    """
    mesh, patch_materials = load_mesh(folder_path, area_threshold)
    material_coefficients = load_materials(folder_path, set(patch_materials))

    return mesh, patch_materials, material_coefficients


def load_mesh(folder_path: str, area_threshold: float = 0.) -> Tuple[TriangleMesh, List[str]]:
    # TODO: Fill out documentation properly.
    """

    Args:
        folder_path:
        area_threshold:

    Returns:

    """
    vertex_list = list()
    face_triplet_list = list()
    face_material_list = list()
    current_material = None

    with open(os.path.join(folder_path, 'mesh.obj'), mode='r') as file:
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

    # Collapse duplicate vertices (within a millimeter of each other).
    keys = np.round(vertices * 1e3)
    _, keep_idx, old2new = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    vertices = vertices[keep_idx]
    vert_triplets = old2new[vert_triplets]

    # Create structure-of-arrays mesh (includes normal vectors, areas, etc).
    mesh = TriangleMesh(vertices, vert_triplets, patch_ids)

    # Check that all triangles in each patch are coplanar.
    for patch_id in np.unique(mesh.ID):
        for triangle_a in np.where(mesh.ID == patch_id)[0]:
            for triangle_b in np.where(mesh.ID == patch_id)[0]:
                if triangle_a == triangle_b:
                    continue

                parallel_normals = np.isclose(np.dot(mesh.n[triangle_a],
                                                     mesh.n[triangle_b]),
                                              1.)
                same_offset = np.isclose(mesh.d0[triangle_a],
                                         mesh.d0[triangle_b])

                if not (parallel_normals and same_offset):
                    raise ValueError('Patches should only contain coplanar triangles.'
                                     + ' Bad patch ID: ' + str(patch_id)
                                     + ' Bad triangle ID A: ' + str(triangle_a)
                                     + ' Bad triangle ID B: ' + str(triangle_b))

    if area_threshold > 0:
        old_num_patches = len(patch_materials)

        merge_small_patches(vert_triplets, mesh, patch_materials, area_threshold)

        # Check that all triangles in each patch are still coplanar.
        for patch_id in np.unique(mesh.ID):
            for triangle_a in np.where(mesh.ID == patch_id)[0]:
                for triangle_b in np.where(mesh.ID == patch_id)[0]:
                    if triangle_a == triangle_b:
                        continue

                    parallel_normals = np.isclose(np.dot(mesh.n[triangle_a],
                                                         mesh.n[triangle_b]),
                                                  1.)
                    same_offset = np.isclose(mesh.d0[triangle_a],
                                             mesh.d0[triangle_b])

                    if not (parallel_normals and same_offset):
                        raise ValueError('Patches should only contain coplanar triangles.'
                                         + ' Bad patch ID: ' + str(patch_id)
                                         + ' Bad triangle ID A: ' + str(triangle_a)
                                         + ' Bad triangle ID B: ' + str(triangle_b))

        new_num_patches = len(patch_materials)

        if new_num_patches != old_num_patches:
            # Save the modified mesh in a new folder.
            if '{}_patches'.format(old_num_patches) in folder_path:
                new_folder_path = folder_path.replace('{}_patches'.format(old_num_patches),
                                                      '{}_patches'.format(new_num_patches))
            else:
                new_folder_path = os.path.join(folder_path, '_{}_patches'.format(new_num_patches))

            if os.path.isdir(new_folder_path):
                warnings.warn('The following folder already exists, its contents may be overwritten:'
                              '\n\t' + new_folder_path)
            else:
                os.mkdir(new_folder_path)

            # Write the modified OBJ into the new folder.
            with open(os.path.join(new_folder_path, 'mesh.obj'), mode='w') as file:
                file.write('mtllib mesh.mtl\n\n')
                for i in range(32):
                    file.write('#')
                file.write(' Vertices\n\n')

                for vert_idx, vert_coords in enumerate(vertices):
                    vertex_line = 'v ' + str(vert_coords[0]) + ' ' + str(vert_coords[1]) + ' ' + str(vert_coords[2])

                    while len(vertex_line) < 31:
                        vertex_line += ' '
                    # Note: OBJ vertices are 1-indexed.
                    vertex_line += '# Vertex ' + str(vert_idx+1) + '\n'

                    file.write(vertex_line)

                file.write('\n')
                for i in range(32):
                    file.write('#')
                file.write(' Faces\n\n')

                for patch_id in range(new_num_patches):
                    patch_ID_str = 'Patch_' + str(patch_id+1) + '_Mat_' + patch_materials[patch_id]

                    file.write('usemtl ' + patch_ID_str + '\n')

                    for triangle_index, vert_triplet in enumerate(vert_triplets):
                        if patch_ids[triangle_index] == patch_id:
                            file.write('f ' + ' '.join([str(vert_idx+1) for vert_idx in vert_triplet]) + '\n')

            # Write the modified MTL into the new folder.
            with open(os.path.join(new_folder_path, 'mesh.mtl'), mode='w') as file:
                for patch_id in range(new_num_patches):
                    patch_ID_str = 'Patch_' + str(patch_id+1) + '_Mat_' + patch_materials[patch_id]

                    file.write('newmtl ' + patch_ID_str + '\n')
                    # TODO: Get appropriate color from original MTL.
                    c = float(patch_id+1) / new_num_patches
                    file.write('Kd {} {} {}\n'.format(c, c, c))
                    file.write('Ka {} {} {}\n'.format(c, c, c))
                    file.write('Ks {} {} {}\n'.format(c, c, c))
                    file.write('Ns 10\n')

            # Copy the old CSV there as well (no change needed).
            with open(os.path.join(folder_path, 'materials.csv'), mode='r') as old_file:
                content = old_file.read()
            with open(os.path.join(new_folder_path, 'materials.csv'), mode='w') as new_file:
                new_file.write(content)

            visualize_mesh(new_folder_path)

    exit()

    # TODO: Validate `mesh.mtl`.
    # TODO: Cross-validate OBJ-MTL.

    return mesh, patch_materials


def load_materials(folder_path: str, expected_names: Set[str]) -> Dict[str, np.ndarray]:
    # TODO: Fill out documentation properly.
    """

    Args:
        folder_path:
        expected_names:

    Returns:

    """
    material_coefficients = dict()

    # Material names will be added to this set when the absorption coefficients are read (first time the name appears in the file)
    # and removed when the scattering coefficients are read (second and last time the name appears in the file).
    expecting_scattering = set()

    with open(os.path.join(folder_path, 'materials.csv'), mode='r', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', skipinitialspace=True)

        first_row = next(reader, None)
        if first_row is not None:
            mat_name = first_row.pop(0)
            band_centers = np.array(first_row, dtype=float)
            if mat_name != 'Frequencies':
                raise ValueError('The first row of material.csv should start with the word "Frequencies" and contain the band center frequencies.')
            if len(first_row) == 0:
                warnings.warn('No band center frequencies are reported in material.csv. Using broadband mode (one band centered at 0).')
                band_centers = np.zeros(1)
            material_coefficients[mat_name] = band_centers
        else:
            raise ValueError('The first row of material.csv should start with the word "Frequencies" and contain the band center frequencies.')

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
                    raise ValueError('Coefficient rows in material.csv should either contain a single value, or as many as there are frequency bands.'
                                     + ' Bad material name: ' + mat_name + '; bad coefficients: ' + str(coeffs))
            elif mat_name in expecting_scattering:
                # This is the second time the material name is encountered in the file. These are the scattering coefficients.
                expecting_scattering.remove(mat_name)

                if len(coeffs) == 1:
                    material_coefficients[mat_name][1] = coeffs[0]
                elif len(coeffs) == len(band_centers):
                    material_coefficients[mat_name][1] = coeffs
                else:
                    raise ValueError('Coefficient rows in material.csv should either contain a single value, or as many as there are frequency bands.'
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


def visualize_mesh(folder_path: str, cull_back_faces: bool = True):
    # TODO: Fill out documentation properly.
    """

    Args:
        folder_path:
        cull_back_faces:

    Returns:

    """
    import pymeshlab
    import polyscope

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(os.path.join(folder_path, 'mesh.obj'))

    polyscope.set_verbosity(0)
    polyscope.set_use_prefs_file(False)
    polyscope.set_enable_render_error_checks(False)
    polyscope.set_give_focus_on_show(True)

    def disable_imgui_files():
        try:
            import polyscope.imgui as psim
            io = psim.GetIO()
            try:
                io.IniFilename = None   # disable imgui.ini
            except Exception as e:
                pass
            try:
                io.LogFilename = None   # disable imgui_log.txt
            except Exception as e:
                pass
        except Exception as e:
            pass
        finally:
            polyscope.clear_user_callback()

    polyscope.init()

    polyscope.set_user_callback(disable_imgui_files)

    ps_mesh = polyscope.register_surface_mesh(os.path.split(folder_path)[-1], ms.current_mesh().vertex_matrix(), ms.current_mesh().face_matrix())
    ps_mesh.add_color_quantity('face_colors', np.asarray(ms.current_mesh().face_color_matrix())[:, :3],
                               defined_on='faces', enabled=True)

    polyscope.set_up_dir('z_up')
    polyscope.set_navigation_style('turntable')
    if cull_back_faces:
        ps_mesh.set_back_face_policy('cull')

    polyscope.show()

    polyscope.remove_all_structures()
