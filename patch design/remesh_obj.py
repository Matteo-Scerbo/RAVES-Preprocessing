import os
import shutil
import numpy as np
import matplotlib.pyplot as plt

import shapely
import shapely.plotting
from sklearn.cluster import KMeans

import polyscope

from raves.src.utils import load_mesh, load_mesh_as_arrays, visualize_mesh


# https://stackoverflow.com/a/26370192
def project_3D_to_2D(points_3D):
    assert len(points_3D.shape) == 2
    assert points_3D.shape[0] >= 3
    assert points_3D.shape[1] == 3

    num_points = points_3D.shape[0]

    local_origin = points_3D[0]

    local_x = points_3D[1] - local_origin
    local_x /= np.linalg.norm(local_x)

    normal = np.cross(local_x,
                      points_3D[2] - local_origin)
    if np.linalg.norm(normal) == 0:
        raise ValueError('All points in the first triangle are colinear.')
    normal /= np.linalg.norm(normal)
    
    local_y = np.cross(normal, local_x)
    local_y /= np.linalg.norm(local_y)

    for p in points_3D:
        z_error = np.dot(p - local_origin, normal)
        if np.abs(z_error) > 1e-2:
            raise ValueError(f'Not all points are coplanar. Z error: {z_error}')

    points_2D = [(np.dot(p - local_origin, local_x),
                  np.dot(p - local_origin, local_y))
                 for p in points_3D]

    return np.array(points_2D), (local_origin, local_x, local_y)


# https://stackoverflow.com/a/26370192
def project_2D_to_3D(points_2D, basis):
    local_origin, local_x, local_y = basis

    points_3D = [local_origin + p[0]*local_x + p[1]*local_y
                 for p in points_2D]

    return np.array(points_3D)


def segment_patch(all_vertices, all_faces, patch_triangle_idxs,
                  area_threshold, narrowness_threshold,
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

    # TODO: Somehow detect and correct thin gaps (mismatched edges).

    # The polygon will be split if its area is too large...
    area = polygon_2D.area
    # ... or if it is very long and narrow.
    # "Narrowness" is evaluated based on the ratio between the polygon's area
    #  and the area of its minimum bounding circle.
    radius = shapely.minimum_bounding_radius(polygon_2D)
    narrowness = 2 * np.pi * radius**2 / area

    # shapely.plotting.plot_polygon(polygon_2D)
    # plt.title(str(narrowness))
    # plt.show()

    num_segments = 1
    if area_threshold is not None:
        num_segments = max(int(np.ceil(area / area_threshold)), num_segments)
    if narrowness_threshold is not None:
        num_segments = max(int(np.ceil(narrowness / narrowness_threshold)), num_segments)

    if num_segments > 1:
        # Consider the polygon's maximum extent w.r.t. the 2D axes, and prepare a tight grid of sample points.
        X = np.arange(np.min(coords_2D[:, 0]),
                      np.max(coords_2D[:, 0]),
                      sample_distance)
        Y = np.arange(np.min(coords_2D[:, 1]),
                      np.max(coords_2D[:, 1]),
                      sample_distance)

        # Filter the sample points which fall within the polygon.
        samples = list()
        for x in X:
            for y in Y:
                p = (x, y)
                if polygon_2D.contains(shapely.Point(p)):
                    samples.append(p)

        # Run K-means clustering to locate the Voronoi polygon centers.
        segment_centers = KMeans(n_clusters=num_segments).fit(samples).cluster_centers_

        # Find edges of Voronoi polygons.
        segment_edges = shapely.voronoi_polygons(shapely.MultiPoint(segment_centers),
                                                 extend_to=polygon_2D)

        # "Cut out" individual polygon segments based on the Voronoi edges.
        segments = [shapely.intersection(edges, polygon_2D)
                    for edges in segment_edges.geoms]
    else:
        segments = [polygon_2D]

    # Drop miniscule segments.
    # TODO: Merge them into adjacent segment if possible.
    segments = [seg for seg in segments
                if seg.area > 1e-4]
    if len(segments) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)

    # Triangulate each polygon segment.
    segments = [shapely.constrained_delaunay_triangles(seg).geoms
                for seg in segments]
    
    # Drop miniscule triangles.
    # TODO: Find better triangulation if possible.
    segments = [[tri for tri in seg
                 if tri.area > 1e-4]
                for seg in segments]
    segments = [seg for seg in segments
                if len(seg) != 0]
    if len(segments) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)

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
        output_lines.append(f'v {vert[0]:.3f} {vert[1]:.3f} {vert[2]:.3f}\n')
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

    area_threshold = 4.
    target_sample_distance = 5e-2

    room_names = ['CR1_DoorAngle1',
                  'CR1_DoorAngle3',
                  'CR2',
                  'CR3',
                  'CR4',
                  ]

    for room_name in room_names:
        naive_name = room_name + '_naive_obj'
        naive_dir = os.path.join(mesh_folder, room_name, naive_name)

        for remeshing_strategy in ['split_area', 'split_area_length']:
            new_name = room_name + '_' + remeshing_strategy
            new_dir = os.path.join(mesh_folder, room_name, new_name)
            os.makedirs(new_dir, exist_ok=True)

            if remeshing_strategy == 'split_area_length':
                narrowness_threshold = 5.
            else:
                narrowness_threshold = None

            print('Converting mesh', new_name)
            
            # visualize_mesh(naive_dir)

            verts, faces, patch_ids, patch_materials = load_mesh_as_arrays(naive_dir)

            new_vertices = np.zeros((0, 3))
            new_faces = np.zeros((0, 3), dtype=int)
            new_patch_ids = np.zeros(0, dtype=int)
            new_patch_materials = list()

            for patch_i in range(len(patch_materials)):
                patch_tris = np.where(patch_ids == patch_i)

                sample_distance = target_sample_distance
                while sample_distance > 1e-4:
                    # The clustering can fail if the surface sampling is insufficient.
                    try:
                        seg_vertices, seg_faces, seg_ids = segment_patch(verts, faces, patch_tris,
                                                                         area_threshold,
                                                                         narrowness_threshold,
                                                                         sample_distance)
                    # https://stackoverflow.com/a/13531310
                    except ValueError as e:
                        if 'should be >= n_clusters=' in str(e) or 'it contains a single sample' in str(e):
                            sample_distance /= 2
                            if sample_distance <= 1e-4:
                                raise ValueError('Could not find a fine enough sampling.')
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
