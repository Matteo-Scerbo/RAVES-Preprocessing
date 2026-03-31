import os
import numpy as np
import matplotlib.pyplot as plt

import pymeshlab
import polyscope

from raves.src.utils import TriangleMesh, load_mesh_as_arrays, visualize_mesh, merge_small_patches


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
    thoroughness = 1.

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
        new_patch_materials = list()

        meshset = pymeshlab.MeshSet()
        for patch_i in range(len(patch_materials)):
            patch_tris = np.where(patch_ids == patch_i)

            meshset.add_mesh(pymeshlab.Mesh(verts,
                                            faces[patch_tris]),
                             naive_name + str(patch_i))

            # Only remesh if above area_threshold (to avoid needlessly increasing the number of triangles).
            area = meshset.get_geometric_measures()['surface_area']
            if area >= area_threshold:
                meshset.meshing_isotropic_explicit_remeshing(targetlen=pymeshlab.PercentageValue(3))
            else:
                # Even with small area, the patch might benefit from subdivision if it is long and narrow.
                meshset.compute_selection_by_edge_length(threshold=np.sqrt(area_threshold))
                meshset.meshing_surface_subdivision_midpoint(iterations=1, selected=True)
            
                # meshing_surface_subdivision_midpoint(iterations=1, threshold=pymeshlab.PercentageValue(3))

            meshset.meshing_remove_null_faces()

            new_faces = np.append(new_faces, meshset.current_mesh().face_matrix() + new_vertices.shape[0], axis=0)
            new_vertices = np.append(new_vertices, meshset.current_mesh().vertex_matrix(), axis=0)
            for i in range(meshset.current_mesh().face_matrix().shape[0]):
                new_patch_materials.append(patch_materials[patch_i])
            
            meshset.clear()
        
        new_patch_ids = np.arange(new_faces.shape[0])
        
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

        mesh = TriangleMesh(new_vertices,
                            new_faces,
                            new_patch_ids)

        merge_small_patches(new_vertices, new_faces,
                            mesh, new_patch_materials,
                            area_threshold, thoroughness)
        # This was changed in-place: retrieve the new values to avoid mix-ups
        new_new_patch_ids = mesh.patch_ids

        ps_mesh = polyscope.register_surface_mesh(naive_name + '_new',
                                                  new_vertices,
                                                  new_faces)
        random_colors = np.random.uniform(size=(int(np.max(new_new_patch_ids))+1, 3))
        ps_mesh.add_color_quantity('face_colors', random_colors[new_new_patch_ids],
                                   defined_on='faces', enabled=True)
        ps_mesh.set_back_face_policy('cull')

        polyscope.set_up_dir('z_up')
        polyscope.set_navigation_style('turntable')
        polyscope.reset_camera_to_home_view()

        polyscope.show()

        polyscope.remove_all_structures()

        # TODO: Re-triangulate patches to minimize number of triangles.

        # new_new_vertices = np.zeros((0, 3))
        # new_new_faces = np.zeros((0, 3), dtype=int)
        # new_new_patch_materials = list()

        # for patch_i in range(len(new_patch_materials)):
        #     patch_tris = np.where(patch_ids == patch_i)

        #     meshset.add_mesh(pymeshlab.Mesh(new_vertices, new_faces[patch_tris]),
        #                      naive_name + 'new' + str(patch_i))
            
        #     meshset.meshing_isotropic_explicit_remeshing(targetlen=pymeshlab.PercentageValue(100))

        #     new_new_faces = np.append(new_new_faces, meshset.current_mesh().face_matrix() + new_new_vertices.shape[0], axis=0)
        #     new_new_vertices = np.append(new_new_vertices, meshset.current_mesh().vertex_matrix(), axis=0)
        #     for i in range(meshset.current_mesh().face_matrix().shape[0]):
        #         new_new_patch_materials.append(new_patch_materials[patch_i])
            
        #     meshset.clear()
        
        # meshset.add_mesh(pymeshlab.Mesh(new_new_vertices, new_new_faces), naive_name + '_new_new')
        # meshset.show_polyscope()

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
