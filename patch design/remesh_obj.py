import os
import numpy as np
import matplotlib.pyplot as plt

import shapely
import shapely.plotting

from sklearn.cluster import KMeans

import pymeshlab
import polyscope

from raves.src.utils import TriangleMesh, load_mesh_as_arrays, visualize_mesh, merge_small_patches


# https://stackoverflow.com/a/26370192
def project_3D_to_2D(points_3D):
    assert len(points_3D.shape) == 2
    assert points_3D.shape[0] >= 3
    assert points_3D.shape[1] == 3

    num_points = points_3D.shape[0]

    local_origin = points_3D[0]

    local_x = points_3D[1] - local_origin
    local_x /= np.linalg.norm(local_x)

    found_normal = False
    for i in range(2, num_points):
        normal = np.cross(local_x,
                          points_3D[i] - local_origin)
        if np.linalg.norm(normal) > 1e-3:
            found_normal = True
            break
    if not found_normal:
        raise ValueError('All points are colinear.')

    local_y = np.cross(normal, local_x)
    local_y /= np.linalg.norm(local_y)

    for p in points_3D:
        z_error = np.dot(p - local_origin, normal)
        if np.linalg.norm(z_error) > 1e-3:
            raise ValueError('Not all points are coplanar.')

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
                  area_threshold,
                  sample_distance=5e-2):
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

    area = polygon_2D.area
    if area > area_threshold:
        num_segments = int(np.ceil(area / area_threshold))

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

    # Triangulate each polygon segment.
    segments = [shapely.constrained_delaunay_triangles(seg).geoms
                for seg in segments]
    
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

    room_names = ['CR1_DoorAngle1',
                  # 'CR1_DoorAngle3',
                  # 'CR2',
                  # 'CR3',
                  # 'CR4',
                  ]

    for room_name in room_names:
        naive_name = room_name + '_naive_obj'
        naive_dir = os.path.join(mesh_folder, room_name, naive_name)

        print('Converting mesh', naive_name)
        
        # visualize_mesh(naive_dir)

        verts, faces, patch_ids, patch_materials = load_mesh_as_arrays(naive_dir)

        new_vertices = np.zeros((0, 3))
        new_faces = np.zeros((0, 3), dtype=int)
        new_patch_ids = np.zeros(0, dtype=int)
        new_patch_materials = list()

        meshset = pymeshlab.MeshSet()
        for patch_i in range(len(patch_materials)):
            patch_tris = np.where(patch_ids == patch_i)

            seg_vertices, seg_faces, seg_ids = segment_patch(verts, faces, patch_tris, area_threshold)

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
        
        polyscope.set_verbosity(0)
        polyscope.set_use_prefs_file(False)
        polyscope.set_enable_render_error_checks(False)
        polyscope.init()
        
        ps_mesh = polyscope.register_surface_mesh(naive_name,
                                                  new_vertices,
                                                  new_faces)
        random_colors = np.random.uniform(size=(int(np.max(new_patch_ids))+1, 3))
        ps_mesh.add_color_quantity('face_colors', random_colors[new_patch_ids],
                                   defined_on='faces', enabled=True)
        ps_mesh.set_back_face_policy('cull')

        polyscope.set_up_dir('z_up')
        polyscope.set_navigation_style('turntable')
        polyscope.reset_camera_to_home_view()

        polyscope.show()

        polyscope.remove_all_structures()

        continue

        """
        meshset = pymeshlab.MeshSet()

        for patch_i in range(num_patches):
            patch_tris = np.where(patch_ids == patch_i)

            meshset.add_mesh(pymeshlab.Mesh(verts,
                                            faces[patch_tris]),
                             naive_name + str(patch_i))
            
            area = meshset.get_geometric_measures()['surface_area']
            
            if area > 30:
                print(area)

                meshset.meshing_isotropic_explicit_remeshing(targetlen=pymeshlab.PercentageValue(100))

                meshset.show_polyscope()
            
            meshset.clear()
            polyscope.remove_all_structures()
        """

        """
        print(naive_name, 'has',
              mesh.size(count_patches=True), 'patches,',
              mesh.size(count_patches=False), 'triangles.')
        
        for mat, count in zip(*np.unique(patch_materials, return_counts=True)):
            print('\t', count, 'out of', mesh.size(count_patches=True),
                'patches have material', mat)
        
        areas = mesh.area
        perimeters = mesh.perimeter()
        isoperimetric = 4 * np.pi * areas / perimeters**2
        
        fig, ax = plt.subplots(2, dpi=200, figsize=(8, 4))

        plot_loghist(ax[0], areas, bins=100)
        ax[0].set_title('Triangle areas')
        ax[0].set_yscale('log')
        ax[0].grid(True)
    
        # plot_loghist(ax[1], isoperimetric, bins=100)
        ax[1].hist(isoperimetric, bins=100)
        ax[1].set_title('Triangle isoperimetric quotients')
        ax[1].set_yscale('log')
        ax[1].grid(True)
        
        plt.suptitle('Mesh name: ' + naive_name)
    
        plt.tight_layout()
        plt.show()
        """
