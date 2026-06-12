import os
import shutil
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

import shapely
import shapely.plotting
from sklearn.cluster import KMeans

import polyscope

from raves.src.utils import load_mesh, load_mesh_as_arrays, visualize_mesh


# https://stackoverflow.com/a/26370192
def project_3D_to_2D(points_3D):
    # Must be a 2D array.
    assert len(points_3D.shape) == 2
    # The second dimension must be the 3D coordinates.
    assert points_3D.shape[1] == 3
    # The polygon must be triangulated (#vertices multiple of 3).
    assert points_3D.shape[0] % 3 == 0
    # There must be at least one triangle.
    num_tris = int(points_3D.shape[0] // 3)
    assert num_tris > 0

    # We need a 2D reference frame formed by two orthogonal vectors (X and Y),
    #  both tangent to the polygon's plane.
    # Many of the polygons are "malformed" in some way, with pairs of vertices
    #  very close to each other, so we need to act carefully.

    # Find the longest edge. It will be the X tangent, and its start will be the local origin.
    # Keeping track of orientation is important to avoid inverting the surface normal.
    local_origin = points_3D[0]
    tangent_x = points_3D[1] - local_origin
    max_len = np.linalg.norm(tangent_x)
    for i in range(1, points_3D.shape[0]):
        p_a = points_3D[i]
        p_b = points_3D[(i+1) % points_3D.shape[0]]
        tangent = p_b - p_a
        edge_len = np.linalg.norm(tangent)
        if edge_len > max_len:
            local_origin = p_a
            tangent_x = tangent
            max_len = edge_len
    tangent_x /= max_len

    # Next, find the surface normal "safely", using Newell's Method.
    # This is done separately for each triangle, to keep track of orientation.
    # The longest vector found is likely the most accurate, and it is kept.
    normal_candidates = list()

    for tri_i in range(num_tris):
        normal = np.zeros(3)
        # https://stackoverflow.com/a/53015210
        for i in range(3):
            p_a = points_3D[tri_i*3 + i]
            p_b = points_3D[tri_i*3 + ((i+1) % 3)]

            normal[0] += (p_a[1] - p_b[1]) * (p_a[2] + p_b[2])
            normal[1] += (p_a[2] - p_b[2]) * (p_a[0] + p_b[0])
            normal[2] += (p_a[0] - p_b[0]) * (p_a[1] + p_b[1])

        normal_candidates.append(normal)
    
    normal_lengths = [np.linalg.norm(normal)
                      for normal in normal_candidates]
    best_normal = np.argmax(normal_lengths)

    if normal_lengths[best_normal] == 0:
        raise ValueError('All points in the polygon are colinear.')
    else:
        normal = normal_candidates[best_normal] / normal_lengths[best_normal]

    # Finally, the orthogonal tangent Y is given by (X cross normal).
    tangent_y = np.cross(normal, tangent_x)
    tangent_y /= np.linalg.norm(tangent_y)

    basis = (local_origin, tangent_x, tangent_y)

    points_2D = [(np.dot(p - local_origin, tangent_x),
                  np.dot(p - local_origin, tangent_y))
                 for p in points_3D]

    # Test the reconstruction accuracy.
    reconstructed = project_2D_to_3D(points_2D, basis)
    errors = np.linalg.norm(reconstructed - points_3D,
                            axis=0)
    if np.any(errors > 1e-2):
        raise ValueError(f'Not all points are coplanar. Reconstruction errors: {errors}')

    return np.array(points_2D), basis


# https://stackoverflow.com/a/26370192
def project_2D_to_3D(points_2D, basis):
    local_origin, local_x, local_y = basis

    points_3D = [local_origin + p[0]*local_x + p[1]*local_y
                 for p in points_2D]

    return np.array(points_3D)


def isoperimetric_quotient(polygon):
    # "Narrowness" is evaluated through the isoperimetric quotient, defined as the ratio of
    #  the polygon's area and the area of the circle having the same perimeter as the polygon.
    area = polygon.area
    perimeter = polygon.length
    return 4 * np.pi * area / (perimeter**2)


def is_convex(polygon):
    if type(polygon) == shapely.MultiPolygon:
        return all(is_convex(p) for p in polygon.geoms)

    # A polygon with holes cannot be convex.
    if len(polygon.interiors) > 0:
        return False
    
    return np.isclose(polygon.area, polygon.convex_hull.area)


def split_polygon(polygon_2D, num_segments, sample_distance):
    if num_segments <= 1:
        return [polygon_2D]
    
    minx, miny, maxx, maxy = polygon_2D.bounds
    
    # Consider the polygon's maximum extent w.r.t. the 2D axes, and prepare a tight grid of sample points.
    X = np.arange(minx, maxx+sample_distance, sample_distance)
    Y = np.arange(miny, maxy+sample_distance, sample_distance)

    # Filter the sample points which fall within the polygon.
    samples = list()
    for x in X:
        for y in Y:
            p = (x, y)
            if polygon_2D.contains(shapely.Point(p)):
                samples.append(p)

    if len(samples) < 3 * num_segments:
        raise ValueError('Try finer sampling.')

    # Run K-means clustering to locate the Voronoi polygon centers.
    segment_centers = KMeans(n_clusters=num_segments).fit(samples).cluster_centers_

    # Find edges of Voronoi polygons.
    segment_edges = shapely.voronoi_polygons(shapely.MultiPoint(segment_centers),
                                             extend_to=polygon_2D)

    # "Cut out" individual polygon segments based on the Voronoi edges.
    return [shapely.intersection(edges, polygon_2D)
            for edges in segment_edges.geoms]


def segment_patch(all_vertices, all_faces, patch_triangle_idxs,
                  min_area_threshold, max_area_threshold, narrowness_threshold,
                  sample_distance):
    # Select only the vertices which form the patch.
    points_3D = all_vertices[all_faces[patch_triangle_idxs]]

    # Translate to 2D.
    coords_3D = points_3D.reshape((-1, 3))
    # https://stackoverflow.com/a/13531310
    try:
        coords_2D, basis = project_3D_to_2D(coords_3D)
    except ValueError as e:
        if str(e) != 'All points are colinear.':
            raise
        else:
            # If the patch had zero area, return nothing.
            return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
    points_2D = coords_2D.reshape((points_3D.shape[0], 3, 2))

    # Create shapely triangles.
    triangles_2D = [shapely.Polygon(tri)
                    for tri in points_2D]

    # Merge shapely triangles into polygon (may have holes).
    polygon_2D = shapely.union_all(triangles_2D)

    # if sample_distance < 1e-1:
    #     shapely.plotting.plot_polygon(polygon_2D)
    #     plt.show()

    # TODO: Somehow detect and correct thin gaps (mismatched edges).

    # The polygon will be split if its area is too large.
    # The narrowness constraint comes afterwards.
    area = polygon_2D.area

    if area < min_area_threshold:
        # The area is already below the minimum allowed; don't split any further.
        max_area_threshold = None
        narrowness_threshold = None
    
    if not is_convex(polygon_2D):
        # If the polygon has holes or is otherwise non-convex, the compactness approach seems to break.
        # TODO: Automatically split "holey" polygons into the minimum number of "dense" parts.
        narrowness_threshold = None
    
    num_segments = 1
    if max_area_threshold is not None:
        num_segments = max(int(np.ceil(area / max_area_threshold)), num_segments)

    small_segments = split_polygon(polygon_2D, num_segments, sample_distance)

    if narrowness_threshold is None:
        small_compact_segments = small_segments
    else:
        small_compact_segments = list()

        for small_segment in small_segments:
            compact_subsegments = [small_segment]
            prev_ipq = isoperimetric_quotient(small_segment)

            if prev_ipq < narrowness_threshold:
                # The component is not compact. Find a way to split it.
    
                # print('Area:', np.round(small_segment.area, 3))
                # print('ipq:', np.round(isoperimetric_quotient(small_segment), 3))

                for num_subsegments in range(2, 25):
                    if small_segment.area < min_area_threshold * num_subsegments:
                        # The split would be too small. Ignore it and proceed with the previous "level".
                        # print('Split would be too small.')
                        break
                    
                    candidate_compact_subsegments = split_polygon(small_segment, num_subsegments, sample_distance)

                    # print('Areas:', np.array([np.round(s.area, 3) for s in candidate_compact_subsegments]))
                    # print('ipqs:', np.array([np.round(isoperimetric_quotient(s), 3) for s in candidate_compact_subsegments]))

                    min_area = min(s.area
                                   for s in candidate_compact_subsegments)
                    min_ipq = min(isoperimetric_quotient(s)
                                  for s in candidate_compact_subsegments)
                    
                    if min_area < min_area_threshold:
                        # The split would be too small. Ignore it and proceed with the previous "level".
                        # print('Split would be too small.')
                        break

                    if min_ipq >= narrowness_threshold:
                        # The split successfully made all components compact. Save it and proceed.
                        compact_subsegments = candidate_compact_subsegments
                        # print('Split successful.')
                        break

                    if num_subsegments == 24:
                        # Too many attempts; something's not working.
                        raise ValueError('Unable to remesh.')
                    
                    if min_ipq <= prev_ipq:
                        # It makes no sense for a finer segmentation to give worse results.
                        # The sampling is evidently too coarse. Try again with smaller distances.
                        # print('Try finer sampling.')
                        raise ValueError('Try finer sampling.')
                    
                    prev_ipq = min_ipq
                    
            small_compact_segments.extend(compact_subsegments)
    
    # Triangulate each polygon segment.
    segments = [shapely.constrained_delaunay_triangles(seg).geoms
                for seg in small_compact_segments]
    
    # Reformat as vertices and indices.
    # N.B.: Ignore the last coordinate (coords[:-1]) because it's a repetition of the first.
    new_vertices_2D = np.concatenate([np.concatenate([np.array(tri.exterior.coords)[:-1]
                                                      for tri in seg])
                                      for seg in segments])
    # Retrieve 3D vertices from 2D.
    new_vertices_3D = project_2D_to_3D(new_vertices_2D, basis)

    new_faces = np.arange(new_vertices_3D.shape[0], dtype=int)
    new_faces = new_faces.reshape((-1, 3))
    # For some reason, the process flips all normals. Put them back.
    new_faces = new_faces[:, ::-1]

    new_ids = np.concatenate([np.array([i for tri in seg])
                              for i, seg in enumerate(segments)])

    return new_vertices_3D, new_faces, new_ids


def save_mesh(output_path,
              vertices, faces,
              patch_ids, patch_materials):
    output_lines = list()
    mtl_output_lines = list()
    
    output_lines.append('mtllib mesh.mtl\n\n')

    for vert in vertices:
        output_lines.append(f'v {vert[0]} {vert[1]} {vert[2]}\n')
    output_lines.append('\n')

    for patch_i in range(np.max(patch_ids)+1):
        output_lines.append(f'usemtl Patch_{patch_i+1}_Mat_{patch_materials[patch_i]}\n')

        patch_tris = np.where(patch_ids == patch_i)[0]
        for tri_i in patch_tris:
            output_lines.append(f'f {faces[tri_i, 0]+1} {faces[tri_i, 1]+1} {faces[tri_i, 2]+1}\n')
        
        output_lines.append('\n')

        mtl_output_lines.append(f'newmtl Patch_{patch_i+1}_Mat_{patch_materials[patch_i]}\n')
        rand_color = np.random.uniform(size=3)
        mtl_output_lines.append(f'Kd {rand_color[0]} {rand_color[1]} {rand_color[2]}')
        mtl_output_lines.append('\n')
    
    with open(output_path,
              mode='w') as file:
        for line in output_lines:
            file.write(line)
    
    with open(output_path.replace('.obj', '.mtl'),
              mode='w') as file:
        for line in mtl_output_lines[:-1]:
            file.write(line)


# https://stackoverflow.com/a/54529366
def plot_loghist(ax, data, bins):
    hist, bins = np.histogram(data, bins=bins)
    logbins = np.logspace(np.log10(bins[0]),
                          np.log10(bins[-1]),
                          len(bins))
    ax.hist(data, bins=logbins)
    ax.set_xscale('log')


if __name__ == '__main__':
    mesh_folder = os.path.join('..', 'BRAS meshes')

    target_sample_distance = 1e-1
    min_area_threshold = 1e-1

    room_names = [
        'CR1_DoorAngle1',
        'CR1_DoorAngle3',
        'CR2',
        'CR1_DoorAngle1_simplified',
        'CR1_DoorAngle3_simplified',
        'CR2_simplified',
        'CR1_DoorAngle1_ubersimplified',
        'CR1_DoorAngle3_ubersimplified',
        'CR2_ubersimplified',
        'CR3',
        # 'CR4',
        ]

    for room_name in room_names:
        naive_name = room_name + '_naive_obj'
        naive_dir = os.path.join(mesh_folder, room_name, naive_name)

        for remeshing_strategy in ['split_area', 'split_area_length',
                                   'uber_split_area', 'uber_split_area_length'
                                   ]:
            new_name = room_name + '_' + remeshing_strategy
            new_dir = os.path.join(mesh_folder, room_name, new_name)
            os.makedirs(new_dir, exist_ok=True)

            if 'uber' in remeshing_strategy:
                max_area_threshold = 2.
            else:
                max_area_threshold = 4.
            
            if 'length' in remeshing_strategy:
                narrowness_threshold = 0.25
            else:
                narrowness_threshold = None

            print('Converting mesh', new_name)
            
            # visualize_mesh(naive_dir)

            verts, faces, patch_ids, patch_materials = load_mesh_as_arrays(naive_dir)

            # polyscope.set_verbosity(0)
            # polyscope.set_use_prefs_file(False)
            # polyscope.set_enable_render_error_checks(False)
            # polyscope.init()
            
            # ps_mesh = polyscope.register_surface_mesh(naive_name,
            #                                           verts,
            #                                           faces)
            # random_colors = np.random.uniform(size=(int(np.max(patch_ids))+1, 3))
            # ps_mesh.add_color_quantity('face_colors', random_colors[patch_ids],
            #                            defined_on='faces', enabled=True)
            # ps_mesh.set_back_face_policy('cull')

            # polyscope.set_up_dir('z_up')
            # polyscope.set_navigation_style('turntable')
            # polyscope.reset_camera_to_home_view()

            # polyscope.show()

            # polyscope.remove_all_structures()

            new_vertices = np.zeros((0, 3))
            new_faces = np.zeros((0, 3), dtype=int)
            new_patch_ids = np.zeros(0, dtype=int)
            new_patch_materials = list()

            for patch_i in tqdm(range(len(patch_materials))):
                patch_tris = np.where(patch_ids == patch_i)

                sample_distance = target_sample_distance
                while sample_distance > 2e-5:
                    # The clustering can fail if the surface sampling is insufficient.
                    try:
                        seg_vertices, seg_faces, seg_ids = segment_patch(verts, faces, patch_tris,
                                                                         min_area_threshold,
                                                                         max_area_threshold,
                                                                         narrowness_threshold,
                                                                         sample_distance)
                    # https://stackoverflow.com/a/13531310
                    except ValueError as e:
                        if 'should be >= n_clusters=' in str(e) or 'it contains a single sample' in str(e) or 'Try finer sampling' in str(e):
                            sample_distance /= np.sqrt(2)
                            if sample_distance <= 2e-5:
                                raise ValueError('Could not find a fine enough sampling.')
                            # print('Trying distance', sample_distance)
                        else:
                            raise
                    else:
                        break

                if len(seg_ids) == 0:
                    # If the patch had zero area, nothing is returned.
                    continue

                new_faces = np.append(new_faces,
                                    seg_faces + new_vertices.shape[0],
                                    axis=0)
                new_vertices = np.append(new_vertices,
                                         seg_vertices,
                                         axis=0)
                if len(new_patch_ids) > 0:
                    new_patch_ids = np.concatenate([new_patch_ids,
                                                    seg_ids + np.max(new_patch_ids) + 1])
                else:
                    new_patch_ids = seg_ids
                for i in range(np.max(seg_ids)+1):
                    new_patch_materials.append(patch_materials[patch_i])
            
            # polyscope.set_verbosity(0)
            # polyscope.set_use_prefs_file(False)
            # polyscope.set_enable_render_error_checks(False)
            # polyscope.init()
            
            # ps_mesh = polyscope.register_surface_mesh(naive_name,
            #                                           new_vertices,
            #                                           new_faces)
            # random_colors = np.random.uniform(size=(int(np.max(new_patch_ids))+1, 3))
            # ps_mesh.add_color_quantity('face_colors', random_colors[new_patch_ids],
            #                         defined_on='faces', enabled=True)
            # ps_mesh.set_back_face_policy('cull')

            # polyscope.set_up_dir('z_up')
            # polyscope.set_navigation_style('turntable')
            # polyscope.reset_camera_to_home_view()

            # polyscope.show()

            # polyscope.remove_all_structures()

            save_mesh(os.path.join(new_dir, 'mesh.obj'),
                      new_vertices, new_faces,
                      new_patch_ids, new_patch_materials)

            shutil.copy(os.path.join(naive_dir, 'materials.csv'),
                        os.path.join(new_dir, 'materials.csv'))

            # visualize_mesh(new_dir)
            
            mesh, patch_materials, _ = load_mesh(new_dir,
                                                 assert_coplanarity=False)
            
            print(new_name, 'has',
                  mesh.size(count_patches=True), 'patches,',
                  mesh.size(count_patches=False), 'triangles.')
            
            for mat, count in zip(*np.unique(patch_materials, return_counts=True)):
                print('\t', count, 'out of', mesh.size(count_patches=True),
                      'patches have material', mat)
            
            # areas = mesh.area
            # perimeters = mesh.perimeter()
            # isoperimetric = 4 * np.pi * areas / perimeters**2
            
            # fig, ax = plt.subplots(2, dpi=200, figsize=(8, 4))

            # plot_loghist(ax[0], areas, bins=100)
            # ax[0].set_title('Triangle areas')
            # ax[0].set_yscale('log')
            # ax[0].grid(True)
        
            # # plot_loghist(ax[1], isoperimetric, bins=100)
            # ax[1].hist(isoperimetric, bins=100)
            # ax[1].set_title('Triangle isoperimetric quotients')
            # ax[1].set_yscale('log')
            # ax[1].grid(True)
            
            # plt.suptitle('Mesh name: ' + new_name)
        
            # plt.tight_layout()
            # plt.show()
