import os
import sys
import time
import numpy as np
from tqdm import tqdm
from scipy.sparse import lil_array, diags

import matplotlib as mpl
import matplotlib.pyplot as plt

from utils import validate_inputs, load_mesh, RayBundle


def main(folder_path: str,
         points_per_square_meter: int = 10,
         rays_per_hemisphere: int = 100,
         detect_open_surface: bool = True) -> None:
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

            # Initialize `path_lengths`, `path_etendues`, `diffuse_kernel`, and `specular_kernel` (sparse arrays).
            # The `path_etendues` will be used to assess the integration accuracy.
            num_paths = num_patches ** 2
            path_lengths = lil_array((num_paths, 1))
            path_etendues = lil_array((num_paths, 1))
            diffuse_kernel = lil_array((num_paths, num_paths))
            specular_kernel = lil_array((num_paths, num_paths))

            if detect_open_surface:
                miss_percentages = np.zeros(num_patches)

            def path_index(i: int, j: int) -> int:
                return i + (j * num_patches)

            # For debugging: plot the surface sample points, adding one triangle at a time.
            """
            all_plots = list()
            for i in range(num_patches):
                for triangle_idx in patch_triangles[i]:
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

            times = {'Build ray pencils ': 0.,
                     'Sample surface    ': 0.,
                     'Move ray pencils  ': 0.,
                     'Trace ray pencils ': 0.,
                     'Bundle rays       ': 0.,
                     'Sum contributions ': 0.,
                     'Normalize and log ': 0.,
                     'Assess accuracy   ': 0.}

            # TODO: Envelop each patch computation in a function
            # TODO: Envelop each surface point computation in a function
            # TODO: Add parallelization of tasks
            for i in tqdm(range(num_patches), desc='ART surface integral (# patches)'):
                # All triangles in each patch are coplanar. Take the plane normal from the first triangle.
                patch_normal = mesh.n[patch_triangles[i][0]]

                start = time.time()

                # Prepare a pencil of rays uniformly sampling the hemisphere.
                # This pencil's origin will be moved to different sample points, to avoid re-instantiating the class.
                hemisphere_pencil = RayBundle.sample_sphere(rays_per_hemisphere, hemisphere_only=True, north_pole=patch_normal)
                # We need to keep track of the surface sample points used to integrate this patch.
                num_points = 0

                # Prepare a pencil formed by the specular reflection of `hemisphere_pencil` across the surface normal.
                # This pencil will be moved and traced in conjunction with `hemisphere_pencil` to obtain the specular reflection kernel.
                hemisphere_directions = hemisphere_pencil.getDirections()
                hemisphere_cosines = np.einsum('ij,j->i', hemisphere_directions, patch_normal)
                specular_directions = 2 * hemisphere_cosines[:, np.newaxis] * patch_normal[np.newaxis] - hemisphere_directions
                specular_pencil = RayBundle.from_shared_origin(origin=np.zeros(3), directions=specular_directions)

                # These accumulators will be built up at each surface sample point, and combined after the loop to form the patch contributions.
                # Refer to "ART_theory.md" for more info on this process.
                accumulated_num_hits = np.zeros(num_patches)
                accumulated_distances = np.zeros(num_patches)
                accumulated_cosines = np.zeros(num_patches)
                accumulated_specular_kernel = np.zeros((num_patches, num_patches))

                if detect_open_surface:
                    accumulated_num_misses = 0

                end = time.time()
                times['Build ray pencils '] += end - start

                for triangle_idx in patch_triangles[i]:
                    start = time.time()

                    # Uniformly sample the triangle's surface.
                    sample_points = mesh.sample_triangle(triangle_idx, points_per_square_meter)
                    # We need to keep track of the surface sample points used to integrate this patch.
                    num_points += sample_points.shape[0]

                    end = time.time()
                    times['Sample surface    '] += end - start

                    for sample_point in sample_points:
                        start = time.time()

                        hemisphere_pencil.moveOrigins(sample_point)
                        specular_pencil.moveOrigins(sample_point)

                        end = time.time()
                        times['Move ray pencils  '] += end - start

                        start = time.time()

                        hemisphere_pencil.traceAll(mesh)
                        specular_pencil.traceAll(mesh)

                        end = time.time()
                        times['Trace ray pencils '] += end - start

                        start = time.time()

                        hemisphere_patch_ids, _ = hemisphere_pencil.getIndices(copy=False)
                        specular_patch_ids, _ = specular_pencil.getIndices(copy=False)
                        hemisphere_distances, _ = hemisphere_pencil.getDistances(copy=False)
                        specular_distances, _ = specular_pencil.getDistances(copy=False)
                        # hemisphere_cosines, _ = hemisphere_pencil.getCosines(copy=False)
                        # specular_cosines, _ = specular_pencil.getCosines(copy=False)

                        hemisphere_hits_per_patch = np.zeros((num_patches, rays_per_hemisphere), dtype=bool)
                        specular_hits_per_patch = np.zeros((num_patches, rays_per_hemisphere), dtype=bool)
                        num_hemisphere_hits_per_patch = np.zeros(num_patches)
                        num_specular_hits_per_patch = np.zeros(num_patches)
                        for j in range(num_patches):
                            hemisphere_hits_per_patch[j] = (hemisphere_patch_ids == j)
                            specular_hits_per_patch[j] = (specular_patch_ids == j)
                            num_hemisphere_hits_per_patch[j] = np.count_nonzero(hemisphere_patch_ids == j)
                            num_specular_hits_per_patch[j] = np.count_nonzero(specular_patch_ids == j)

                        if detect_open_surface:
                            # An index of -1 indicates that the ray has no valid intersections.
                            accumulated_num_misses += np.count_nonzero(hemisphere_patch_ids == -1)
                            accumulated_num_misses += np.count_nonzero(specular_patch_ids == -1)

                        end = time.time()
                        times['Bundle rays       '] += end - start

                        start = time.time()

                        for j in range(num_patches):
                            # Combine the two bundles to ensure symmetry.
                            # Each ray appears once as "main" and once as specular; both count as hits.
                            accumulated_num_hits[j] += num_hemisphere_hits_per_patch[j]
                            accumulated_num_hits[j] += num_specular_hits_per_patch[j]

                            accumulated_distances[j] += np.sum(hemisphere_distances[hemisphere_hits_per_patch[j]])
                            accumulated_distances[j] += np.sum(specular_distances[specular_hits_per_patch[j]])

                            # The departure cosine of each ray is the same as the departure cosine of its specular ray.
                            accumulated_cosines[j] += np.sum(hemisphere_cosines[hemisphere_hits_per_patch[j]])
                            accumulated_cosines[j] += np.sum(hemisphere_cosines[specular_hits_per_patch[j]])

                            for h in range(num_patches):
                                accumulated_specular_kernel[h, j] += 2 * np.count_nonzero(hemisphere_hits_per_patch[j] & specular_hits_per_patch[h])
                                # The multiplication by 2 makes this equivalent to:
                                # accumulated_specular_kernel[h, j] += np.count_nonzero(hemisphere_hits_per_patch[j] & specular_hits_per_patch[h])
                                # accumulated_specular_kernel[h, j] += np.count_nonzero(hemisphere_hits_per_patch[h] & specular_hits_per_patch[j])

                        end = time.time()
                        times['Sum contributions '] += end - start

                start = time.time()

                # Normalize accumulators and add to global trackers.
                for j in range(num_patches):
                    if accumulated_num_hits[j] == 0:
                        # No visibility between any point in j and any point in i.
                        continue

                    ij = path_index(i, j)

                    path_lengths[ij] = accumulated_distances[j] / accumulated_num_hits[j]

                    # Etendue is equal to form factor times surface area (times pi, but we don't need that for the check we'll make).
                    path_etendues[ij] = patch_areas[i] * accumulated_cosines[j] / (rays_per_hemisphere * num_points)

                    for h in range(num_patches):
                        if accumulated_num_hits[h] == 0:
                            # No visibility between any point in h and any point in i.
                            continue

                        hi = path_index(h, i)

                        # Note: in theory, the diffuse kernel integral involves a multiplication by 2.
                        # In practice, we do not need it because each ray is counted once as "main" and once as specular.
                        diffuse_kernel[hi, ij] = accumulated_cosines[j] / (rays_per_hemisphere * num_points)
                        specular_kernel[hi, ij] = accumulated_specular_kernel[h, j] / accumulated_num_hits[h]

                if detect_open_surface:
                    # Note: again, we multiply by 50 instead of 100 because we accumulated both ray pencils.
                    miss_percentages[i] = 50 * accumulated_num_misses / (rays_per_hemisphere * num_points)

                end = time.time()
                times['Normalize and log '] += end - start

            start = time.time()

            if detect_open_surface:
                print('\nPercentage of invalid rays from each patch:')
                print('\t Maximum (patch {}): {:.2f}%'.format(np.argmax(miss_percentages)+1, np.max(miss_percentages)))
                print('\t Average: {:.2f}%'.format(np.max(miss_percentages)))
                print('\t Median: {:.2f}%'.format(np.max(miss_percentages)))
                print('A high percentage of missed rays indicates the surface is not closed.')
                print('If the average is above 50%, check the orientation of normal vectors.')

            # This is used in some upcoming assertions.
            reverse_path_indexing = np.zeros(num_paths, dtype=int)
            for i in range(num_patches):
                for j in range(num_patches):
                    reverse_path_indexing[path_index(i, j)] = path_index(j, i)

            # These should theoretically be identical, but may not be due to the discretized integration.
            # Nevertheless, they should be close enough.
            path_visibility = (path_lengths.toarray().squeeze() != 0)
            reverse_path_visibility = path_visibility[reverse_path_indexing]

            num_mismatches = np.count_nonzero(path_visibility & ~reverse_path_visibility)
            print('\n' + str(num_mismatches) + ' pairs of patches have mismatched visibility (one sees the other, but not vice versa).')
            print('This makes up {:.2f}% of all possible propagation paths.'.format(num_mismatches / num_paths))
            print('If this seems too high, consider increasing `points_per_square_meter` and/or `rays_per_hemisphere`.')
            print('The mismatched pairs will be dropped (i.e., we assume there is no visibility).')
            path_visibility = path_visibility & reverse_path_visibility

            # Set elements without visibility to 0, in case of mismatches.
            diffuse_kernel[~path_visibility] = 0
            diffuse_kernel[:, ~path_visibility] = 0
            specular_kernel[~path_visibility] = 0
            specular_kernel[:, ~path_visibility] = 0

            # Evaluate the row sums of both kernels. All rows should sum to 1; any divergence is an artefact of numerical integration.
            # As such, we can use these to assess the accuracy of the integration.
            diffuse_row_sums = diffuse_kernel.sum(axis=1)
            specular_row_sums = specular_kernel.sum(axis=1)

            diffuse_row_sums_RMSE = np.sqrt(np.mean(np.abs(diffuse_row_sums[path_visibility] - 1.) ** 2))
            specular_row_sums_RMSE = np.sqrt(np.mean(np.abs(specular_row_sums[path_visibility] - 1.) ** 2))

            print('\nThe kernel rows sum to 1 with an RMSE of {:.2e} for the diffuse kernel and {:.2e} for the specular kernel.'.format(diffuse_row_sums_RMSE, specular_row_sums_RMSE))
            print('If either of these seems too high, consider increasing `points_per_square_meter` and/or `rays_per_hemisphere`.')
            print('The row sums will now be forcibly normalized.')

            # Apply the normalization safely w.r.t. zero rows.
            diffuse_row_normalization = np.divide(1., diffuse_row_sums,
                                                  out=np.zeros(num_paths),
                                                  where=(diffuse_row_sums != 0))
            diffuse_kernel = lil_array(diags(diffuse_row_normalization) @ diffuse_kernel)
            specular_row_normalization = np.divide(1., specular_row_sums,
                                                   out=np.zeros(num_paths),
                                                   where=(specular_row_sums != 0))
            specular_kernel = lil_array(diags(specular_row_normalization) @ specular_kernel)
            specular_kernel = lil_array(specular_kernel @ diags(specular_row_normalization))

            # For debugging: plot the row sums.
            """
            fig, ax = plt.subplots(dpi=200, figsize=(8, 6))
            plt.plot(diffuse_row_sums[path_visibility], label='diffuse (RMSE {:.2e})'.format(diffuse_row_sums_RMSE))
            plt.plot(specular_row_sums[path_visibility], label='specular (RMSE {:.2e})'.format(specular_row_sums_RMSE))
            plt.tight_layout()
            plt.legend()
            plt.show()
            """

            # Assess numerical precision by comparing diffuse kernel symmetricity.
            etendue_RMSE = np.sqrt(np.mean(np.abs(path_etendues - path_etendues[reverse_path_indexing]) ** 2))
            print('\nThe etendues between patch pairs (i, j) have an RMSE of {:.2e} with respect to their counterparts (j, i).'.format(etendue_RMSE))
            print('The propagation path etendues should be symmetric, i.e., the RMSE should be low.')
            print('If it seems too high, consider increasing `points_per_square_meter` and/or `rays_per_hemisphere`.')

            # For debugging: plot the etendues.
            """
            fig, ax = plt.subplots(dpi=200, figsize=(8, 6))
            plt.plot(path_etendues.toarray()[path_visibility])
            plt.plot(path_etendues.toarray()[reverse_path_indexing][path_visibility])
            plt.tight_layout()
            plt.show()
            """

            end = time.time()
            times['Assess accuracy   '] += end - start

            print('\nTime elapsed for different tasks (seconds):')
            for k, i in times.items():
                print('\t', k, i)

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
