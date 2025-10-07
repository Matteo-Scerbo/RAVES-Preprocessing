import os
import sys
import numpy as np
from scipy.sparse import lil_array

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from utils import validate_inputs, load_mesh, RayBundle


def main(folder_path: str,
         points_per_square_meter: int = 10,
         rays_per_hemisphere: int = 100) -> None:
    if os.path.isdir(folder_path):
        if validate_inputs(folder_path):
            print('Running `compute_ART` in the environment "' + folder_path.split('/')[-1] + '"')

            mesh, patch_materials = load_mesh(folder_path + '/mesh.obj')

            num_patches = len(patch_materials)

            # dict of lists: indices of the triangles forming each patch.
            patch_triangles = dict()
            # Patch areas, as sum of triangle areas.
            patch_areas = np.zeros(num_patches)
            for triangle_index, triangle_patch_ID in enumerate(mesh.ID):
                if triangle_patch_ID not in patch_triangles.keys():
                    # This is the first triangle found for this patch. Create the list with one element.
                    patch_triangles[triangle_patch_ID] = [triangle_index]
                else:
                    # This is not the first triangle found for this patch. Add element to the existing list.
                    patch_triangles[triangle_patch_ID].append(triangle_index)

                patch_areas[triangle_patch_ID] += mesh.area[triangle_index]

            if num_patches > 1000:
                print('Warning: the mesh contains a very large number of patches (' + str(num_patches) + ')')
            if np.any(patch_areas < 0.1):
                print('Warning: the mesh contains very small patches (smallest area: ' + str(np.min(patch_areas)) + ')')

            # Initialize `path_lengths`, `diffuse_kernel`, and `specular_kernel` (sparse arrays).
            num_paths = num_patches ** 2
            path_lengths = lil_array((num_paths, 1))
            path_etendues = lil_array((num_paths, 1))
            specular_kernel = lil_array((num_paths, num_paths))
            # `diffuse_kernel` is created after the loop, based on `path_etendues`

            def path_idx(i: int, j: int) -> int:
                return i + (j * num_patches)

            # For debugging: plot the surface sample points, adding one triangle at a time.
            """
            all_plots = list()
            for patch_idx in range(num_patches):
                for triangle_idx in patch_triangles[patch_idx]:
                    # Uniformly sample the triangle's surface.
                    sample_points = mesh.sample_triangle(triangle_idx, points_per_square_meter)

                    all_plots.append(dict())
                    all_plots[-1]['Sample points'] = sample_points
                    all_plots[-1]['Triangle normal'] = mesh.n[triangle_idx]
                    all_plots[-1]['Triangle vertex'] = mesh.v1[triangle_idx]
                    all_plots[-1]['Triangle edge 1'] = mesh.edge1[triangle_idx]
                    all_plots[-1]['Triangle edge 2'] = mesh.edge2[triangle_idx]

                    fig = plt.figure(figsize=(4, 4), dpi=200)
                    ax = fig.add_subplot(111, projection='3d')

                    mpl_colors = mpl.colormaps['tab10'].colors
                    for plot in all_plots:
                        clr_i = 0
    
                        X, Y, Z = zip(*plot['Sample points'])
                        ax.scatter(X, Y, Z, color=mpl_colors[clr_i], alpha=0.5, label='Sample points')
                        clr_i += 1
    
                        plt.quiver(plot['Triangle vertex'][0], plot['Triangle vertex'][1], plot['Triangle vertex'][2],
                                   plot['Triangle edge 1'][0], plot['Triangle edge 1'][1], plot['Triangle edge 1'][2],
                                   color=mpl_colors[clr_i], alpha=0.5, label='Triangle edge 1')
                        clr_i += 1
                        plt.quiver(plot['Triangle vertex'][0], plot['Triangle vertex'][1], plot['Triangle vertex'][2],
                                   plot['Triangle edge 2'][0], plot['Triangle edge 2'][1], plot['Triangle edge 2'][2],
                                   color=mpl_colors[clr_i], alpha=0.5, label='Triangle edge 2')
                        clr_i += 1
                        plt.quiver(plot['Triangle vertex'][0], plot['Triangle vertex'][1], plot['Triangle vertex'][2],
                                   plot['Triangle normal'][0], plot['Triangle normal'][1], plot['Triangle normal'][2],
                                   length=1, normalize=True, color=mpl_colors[clr_i], alpha=0.5, label='Triangle normal')
                        clr_i += 1

                    # aspect ratio is 1:1:1 in data space
                    # ax.set_box_aspect((np.ptp(sample_points[:, 0]),
                    #                    np.ptp(sample_points[:, 1]),
                    #                    np.ptp(sample_points[:, 2])))
                    # ax.set(xlim=(np.min(sample_points[:, 0]), np.max(sample_points[:, 0])),
                    #        ylim=(np.min(sample_points[:, 1]), np.max(sample_points[:, 1])),
                    #        zlim=(np.min(sample_points[:, 2]), np.max(sample_points[:, 2])),
                    #        xlabel='x [m]', ylabel='y [m]', zlabel='z [m]')

                    plt.tight_layout()
                    plt.legend()
                    plt.show()
            """

            for patch_idx in range(num_patches):
                # All triangles in each patch are coplanar. Take the plane normal from the first triangle.
                patch_normal = mesh.n[patch_triangles[patch_idx][0]]

                # Prepare a pencil of rays uniformly sampling the hemisphere.
                # This pencil's origin will be moved to different sample points, to avoid re-instantiating the class.
                hemispherePencil = RayBundle.sample_sphere(rays_per_hemisphere,
                                                           hemisphere_only=True,
                                                           north_pole=patch_normal)

                for triangle_idx in patch_triangles[patch_idx]:
                    # Uniformly sample the triangle's surface.
                    sample_points = mesh.sample_triangle(triangle_idx, points_per_square_meter)

                    for sample_point in mesh.sample_triangle(triangle_idx, points_per_square_meter):
                        hemispherePencil.moveOrigins(sample_point)
                        hemispherePencil.traceAll(mesh)
                        
                        # TODO: Accumulate as in existing code
                        #  path_lengths
                        #  path_etendues
                        #  specular_kernel

            # TODO: Normalize as in existing code
            #  path_lengths
            #  path_etendues
            #  specular_kernel
            # TODO: Generate diffuse_kernel from path_etendues and patch_areas

            # TODO: Assess numerical quality by comparing path etendue symmetricity

            # TODO: Write
            #           ART_diffuse_kernel.mtx
            #           ART_specular_kernel.mtx
            #           path_indexing.mtx
            #           path_lengths.csv

            # TODO: Read materials.csv
            # TODO: Write ART_octave_band_1.mtx, ART_octave_band_2.mtx, etc. (for each frequency band)
    else:
        print('Not a valid folder path:\n\t' + folder_path)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print('No arguments provided. Please provide a valid folder path as argument.')
