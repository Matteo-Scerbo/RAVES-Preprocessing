import os
import numpy as np
from tqdm import tqdm
import multiprocessing
from scipy.sparse import lil_array, csr_array, diags
from scipy.io import mmread, mmwrite
from typing import List, Tuple

from .utils import RayBundle, TriangleMesh, load_all_inputs, air_absorption_in_band, sound_speed


# https://stackoverflow.com/a/21130146
def integrate_patch(args: Tuple[TriangleMesh, int, int, List[int], int, float]
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """
    Integrate ART kernel contributions for one patch.

    This traces two ray bundles from uniformly sampled surface points on all
    triangles that form the patch: (1) a hemisphere bundle aligned with the
    patch normal and (2) its specularly reflected counterpart. For each
    sample point, rays are traced, hits are tallied per target patch, and
    accumulators are updated for distances, departure cosines, and specular
    pair counts. The index of the source patch is returned so that results
    from multiprocessing can be re-associated (return order isn't guaranteed).

    Parameters
    ----------
    args : tuple
        Expected layout:
        (mesh, num_patches, this_patch, patch_triangles,
         rays_per_hemisphere, points_per_square_meter)

        - mesh : TriangleMesh
          Mesh defining the environment geometry.
        - num_patches : int
          Total number of patches in the mesh.
        - this_patch : int
          Index of the source patch to integrate.
        - patch_triangles : sequence of int
          Triangle indices that belong to this source patch.
        - rays_per_hemisphere : int
          Number of rays shot per surface sample point.
        - points_per_square_meter : float
          Surface sampling density used to generate sample points.

    Returns
    -------
    cum_distances
        1D array of size num_patches with summed hit distances from this
        source patch to each target patch.
    cum_cosines
        1D array of size num_patches with summed departure cosines for rays
        originating on this source patch and hitting each target patch.
    cum_num_hits
        1D array of size num_patches with the number of ray hits per target
        patch, combining hemisphere and specular rays from all sample points.
    cum_specular_kernel
        2D array of shape (num_patches, num_patches) with counts of specular
        ray pairings: entry (h, j) counts pairs where a hemisphere ray from
        this source hits j and its specular counterpart hits h.
    num_points
        Total number of surface sample points used on this source patch.
    this_patch
        The source patch index, returned to reconnect unordered results.
    """
    mesh, num_patches, this_patch, patch_triangles, rays_per_hemisphere, points_per_square_meter = args

    # All triangles in each patch are coplanar. Take the plane normal from the first triangle.
    patch_normal = mesh.n[patch_triangles[0]]

    # Prepare a pencil of rays uniformly sampling the hemisphere.
    # This pencil's origin will be moved to different sample points, to avoid re-instantiating the class.
    hemisphere_pencil = RayBundle.sample_sphere(rays_per_hemisphere, hemisphere_only=True, north_pole=patch_normal)
    # We need to keep track of the surface sample points used to integrate this patch.
    num_points = 0

    # Prepare a pencil formed by the specular reflection of `hemisphere_pencil` across the surface normal.
    # This pencil will be moved and traced in conjunction with `hemisphere_pencil` to obtain the specular reflection kernel.
    hemisphere_directions = hemisphere_pencil.get_directions()
    hemisphere_cosines = np.einsum('ij,j->i', hemisphere_directions, patch_normal)
    specular_directions = 2 * hemisphere_cosines[:, np.newaxis] * patch_normal[np.newaxis] - hemisphere_directions
    specular_pencil = RayBundle.from_shared_origin(origin=np.zeros(3), directions=specular_directions)

    # These accumulators will be built up at each surface sample point, and combined after the loop to form the patch contributions.
    # Refer to "ART_theory.md" for more info on this process.
    cum_num_hits = np.zeros(num_patches)
    cum_distances = np.zeros(num_patches)
    cum_cosines = np.zeros(num_patches)
    cum_specular_kernel = np.zeros((num_patches, num_patches))

    for triangle_idx in patch_triangles:
        # Uniformly sample the triangle's surface.
        sample_points = mesh.sample_triangle(triangle_idx, points_per_square_meter)
        # We need to keep track of the surface sample points used to integrate this patch.
        num_points += sample_points.shape[0]

        for sample_point in sample_points:
            hemisphere_pencil.move_origins(sample_point)
            specular_pencil.move_origins(sample_point)

            hemisphere_pencil.trace_all(mesh)
            specular_pencil.trace_all(mesh)

            hemisphere_patch_ids, _ = hemisphere_pencil.get_indices(copy=False)
            specular_patch_ids, _ = specular_pencil.get_indices(copy=False)
            hemisphere_distances, _ = hemisphere_pencil.get_distances(copy=False)
            specular_distances, _ = specular_pencil.get_distances(copy=False)
            # hemisphere_cosines, _ = hemisphere_pencil.get_cosines(copy=False)
            # specular_cosines, _ = specular_pencil.get_cosines(copy=False)

            hemisphere_hits_per_patch = np.zeros((num_patches, rays_per_hemisphere), dtype=bool)
            specular_hits_per_patch = np.zeros((num_patches, rays_per_hemisphere), dtype=bool)
            num_hemisphere_hits_per_patch = np.zeros(num_patches)
            num_specular_hits_per_patch = np.zeros(num_patches)
            for j in range(num_patches):
                hemisphere_hits_per_patch[j] = (hemisphere_patch_ids == j)
                specular_hits_per_patch[j] = (specular_patch_ids == j)
                num_hemisphere_hits_per_patch[j] = np.count_nonzero(hemisphere_patch_ids == j)
                num_specular_hits_per_patch[j] = np.count_nonzero(specular_patch_ids == j)

            for j in range(num_patches):
                # Combine the two bundles to ensure symmetry.
                # Each ray appears once as "main" and once as specular; both count as hits.
                cum_num_hits[j] += num_hemisphere_hits_per_patch[j]
                cum_num_hits[j] += num_specular_hits_per_patch[j]

                cum_distances[j] += np.sum(hemisphere_distances[hemisphere_hits_per_patch[j]])
                cum_distances[j] += np.sum(specular_distances[specular_hits_per_patch[j]])

                # The departure cosine of each ray is the same as the departure cosine of its specular ray.
                cum_cosines[j] += np.sum(hemisphere_cosines[hemisphere_hits_per_patch[j]])
                cum_cosines[j] += np.sum(hemisphere_cosines[specular_hits_per_patch[j]])

                for h in range(num_patches):
                    cum_specular_kernel[h, j] += 2 * np.count_nonzero(hemisphere_hits_per_patch[j] & specular_hits_per_patch[h])
                    # The multiplication by 2 makes this equivalent to:
                    # cum_specular_kernel[h, j] += np.count_nonzero(hemisphere_hits_per_patch[j] & specular_hits_per_patch[h])
                    # cum_specular_kernel[h, j] += np.count_nonzero(hemisphere_hits_per_patch[h] & specular_hits_per_patch[j])

    return cum_distances, cum_cosines, cum_num_hits, cum_specular_kernel, num_points, this_patch


def assess_ART_on_grid(folder_path: str,
                       points_per_square_meter: List[float], rays_per_hemisphere: List[int],
                       area_threshold: float = 0., thoroughness: float = 0.,
                       compute_missing: bool = True, save_kernels: bool = True,
                       multiprocess_pool_size: int = 1
                       ) -> str:
    """
    Assess ART accuracy over a grid of parameters and plot summaries.

    For each combination of surface sampling density and rays per hemisphere,
    this loads the symmetric absolute percentage error (SAPE) of propagation
    path etendues from CSV files created by assess_ART runs. If a CSV is
    missing and compute_missing is True, the ART computation is performed
    via assess_ART to generate it. Heatmaps and summaries are shown.

    Parameters
    ----------
    folder_path
        Path to the environment folder.
    points_per_square_meter
        List of surface sampling densities to evaluate.
    rays_per_hemisphere
        List of ray counts per sample point to evaluate.
    area_threshold
        Optional patch-area simplification threshold passed to assess_ART.
    thoroughness
        Optional remeshing effort parameter passed to assess_ART.
    compute_missing
        If True, run assess_ART for grid points that do not have a saved CSV.
    save_kernels
        Passed to assess_ART to control whether kernels are saved.
    multiprocess_pool_size
        Number of worker processes to use when assess_ART is invoked.

    Returns
    -------
    folder_path
        The (possibly updated) environment folder path.
    """

    weights = np.zeros((len(points_per_square_meter), len(rays_per_hemisphere)))
    medians = np.zeros((len(points_per_square_meter), len(rays_per_hemisphere)))

    for p_i, ppsm in enumerate(points_per_square_meter):
        for r_i, rays in enumerate(rays_per_hemisphere):
            file_name = 'etendue_SAPE_{:.0f}pnts_{:d}rays.csv'.format(int(ppsm), rays)

            weights[p_i, r_i] = ppsm * rays

            if not os.path.isfile(os.path.join(folder_path, file_name)):
                if compute_missing:
                    folder_path = assess_ART(folder_path=folder_path,
                                             points_per_square_meter=ppsm, rays_per_hemisphere=rays,
                                             area_threshold=area_threshold, thoroughness=thoroughness,
                                             save_kernels=save_kernels, multiprocess_pool_size=multiprocess_pool_size)
                else:
                    continue

            etendue_sape = np.loadtxt(os.path.join(folder_path, file_name), delimiter=',')
            median_sape = np.median(etendue_sape)
            medians[p_i, r_i] = median_sape

    # https://stackoverflow.com/q/71119762
    # https://matplotlib.org/stable/users/explain/axes/arranging_axes.html
    import matplotlib.pyplot as plt

    fig = plt.figure(layout="constrained")
    subfigs = fig.subfigures(1, 2)

    # weighted_medians = np.log10(np.multiply(medians, weights,
    #                                         where=(medians != 0),
    #                                         out=np.ones_like(medians)))
    weighted_medians = medians * np.log10(weights)

    # Do not show missing entries in the plots.
    medians = np.ma.masked_where(medians == 0, medians)
    weighted_medians = np.ma.masked_where(medians == 0, weighted_medians)

    for sub_i, sub_data in enumerate([medians, weighted_medians]):
        axs = subfigs[sub_i].subplots(2, 2, sharex="col", sharey="row",
                                      gridspec_kw=dict(height_ratios=[2, sub_data.shape[0]],
                                                       width_ratios=[sub_data.shape[1], 2]))
        subfigs[sub_i].delaxes(axs[0, 1])

        axs[1, 0].imshow(sub_data, aspect="auto", origin="lower")
        axs[1, 0].set_xticks(range(sub_data.shape[1]), rays_per_hemisphere)
        axs[1, 0].set_xlabel('Rays per hemisphere')
        if sub_i == 0:
            axs[1, 0].set_yticks(range(sub_data.shape[0]), points_per_square_meter)
            axs[1, 0].set_ylabel('Points per square meter')
        else:
            axs[1, 0].set_yticks(range(sub_data.shape[0]), [])

        for i in range(sub_data.shape[0]):
            for j in range(sub_data.shape[1]):
                axs[1, 0].text(j, i, np.round(sub_data[i, j], 2),
                               ha="center", va="center", color="w")

        axs[0, 0].plot(range(sub_data.shape[1]), sub_data.mean(axis=0))
        axs[0, 0].set_ylabel('Averaged over points')
        axs[0, 0].set_ylim(0, None)
        axs[0, 0].grid()

        axs[1, 1].plot(sub_data.mean(axis=1), range(sub_data.shape[0]))
        axs[1, 1].set_xlim(0, None)
        axs[1, 1].set_xlabel('Averaged over rays')
        axs[1, 1].grid()

    plt.suptitle('Etendue SAPE over number of rays and samples per square meter.'
                 '\nThe left figure shows the median.'
                 '\nThe right figure shows (median $\\cdot$ $\\log_{10}$ traced_rays_per_square_meter).'
                 '\nIn other words, the right figure is weighted by the processing runtime.')

    plt.show()

    return folder_path


def assess_ART(folder_path: str,
               points_per_square_meter: float, rays_per_hemisphere: int,
               area_threshold: float = 0., thoroughness: float = 0.,
               save_kernels: bool = True,
               multiprocess_pool_size: int = 1
               ) -> str:
    """
    Compute core ART kernels and assess numerical integration accuracy.
    A CSV with the symmetric absolute percentage error (SAPE) of propagation
    path etendues is saved for later analysis.

    Parameters
    ----------
    folder_path
        Path to the environment folder.
    points_per_square_meter
        Surface sampling density used to integrate each patch.
    rays_per_hemisphere
        Number of rays traced from each surface sample point.
    area_threshold
        Optional patch-area simplification threshold applied when loading inputs.
    thoroughness
        Optional remeshing effort parameter applied when loading inputs.
    save_kernels
        If True, save the core ART kernels resulting from the integration.
    multiprocess_pool_size
        Number of worker processes to use (1 disables multiprocessing).

    Returns
    -------
    folder_path
        The (possibly updated) environment folder path.
    """

    multiprocess_pool_size = min(multiprocess_pool_size, os.cpu_count() - 1)
    num_processes = max(1, multiprocess_pool_size)
    if num_processes == 1:
        print('Will use a single process.')
    else:
        print(os.cpu_count(), 'cores available. Pool will use', num_processes, 'SUB-processes.')

    mesh, patch_materials, material_coefficients, folder_path = load_all_inputs(folder_path, area_threshold, thoroughness)

    num_patches = len(patch_materials)

    # dict of lists: indices of the triangles forming each patch.
    patch_triangles = dict()
    # Patch areas, as sum of triangle areas.
    patch_areas = np.zeros(num_patches)
    for triangle_index, triangle_patch_id in enumerate(mesh.patch_ids):
        if triangle_patch_id not in patch_triangles.keys():
            # This is the first triangle found for this patch. Create the list with one element.
            patch_triangles[triangle_patch_id] = [triangle_index]
        else:
            # This is not the first triangle found for this patch. Add element to the existing list.
            patch_triangles[triangle_patch_id].append(triangle_index)

        patch_areas[triangle_patch_id] += mesh.area[triangle_index]

    # This is defined here in order to bake `num_patches` into it.
    def path_index(i: int, j: int) -> int:
        return i + (j * num_patches)

    # Initialize `path_lengths`, `path_etendues`, `diffuse_kernel`, and `specular_kernel`.
    # The path etendues are used to assess the integration accuracy, and are also needed to scale MoD-ART eigenvectors.
    num_paths = num_patches ** 2
    path_lengths = np.zeros(num_paths)
    path_etendues = np.zeros(num_paths)
    diffuse_kernel = lil_array((num_paths, num_paths))
    specular_kernel = lil_array((num_paths, num_paths))
    path_indexing = lil_array((num_patches, num_patches), dtype=int)

    # Parallelize integration across patches
    if multiprocess_pool_size == 1:
        for i in tqdm(range(num_patches), desc='ART surface integral (# patches)'):
            # These accumulators will be built up at each surface sample point, and combined after the loop to form the patch contributions.
            # Refer to "ART_theory.md" for more info on this process.
            returned_tuple = integrate_patch((mesh, num_patches, i, patch_triangles[i],
                                              rays_per_hemisphere, points_per_square_meter))
            cum_distances, cum_cosines, cum_num_hits, cum_specular_kernel, num_points, i = returned_tuple

            # Normalize accumulators and add to global trackers.
            for j in range(num_patches):
                if cum_num_hits[j] == 0:
                    # No visibility between any point in j and any point in i.
                    continue

                ij = path_index(i, j)

                path_lengths[ij] = cum_distances[j] / cum_num_hits[j]

                # Etendue is equal to form factor times surface area times pi.
                path_etendues[ij] = np.pi * patch_areas[i] * cum_cosines[j] / (rays_per_hemisphere * num_points)

                for h in range(num_patches):
                    if cum_num_hits[h] == 0:
                        # No visibility between any point in h and any point in i.
                        continue

                    hi = path_index(h, i)

                    # Note: in theory, the diffuse kernel integral involves a multiplication by 2.
                    # In practice, we do not need it because each ray is counted once as "main" and once as specular.
                    diffuse_kernel[hi, ij] = cum_cosines[j] / (rays_per_hemisphere * num_points)
                    specular_kernel[hi, ij] = cum_specular_kernel[h, j] / cum_num_hits[h]
    else:
        task_list = list()
        for i in range(num_patches):
            task = (mesh, num_patches, i, patch_triangles[i], rays_per_hemisphere, points_per_square_meter)
            task_list.append(task)

        with multiprocessing.Pool(multiprocess_pool_size) as pool:
            patch_contributions = pool.imap_unordered(integrate_patch, task_list)

            # https://stackoverflow.com/a/41921948
            # https://stackoverflow.com/a/72514814
            with tqdm(total=num_patches, desc='ART surface integral (# patches)',
                      miniters=min(int(num_patches / 10), multiprocess_pool_size * 10), maxinterval=600) as progress_bar:
                for returned_tuple in patch_contributions:
                    cum_distances, cum_cosines, cum_num_hits, cum_specular_kernel, num_points, i = returned_tuple

                    # Normalize accumulators and add to global trackers.
                    for j in range(num_patches):
                        if cum_num_hits[j] == 0:
                            # No visibility between any point in j and any point in i.
                            continue

                        ij = path_index(i, j)

                        path_lengths[ij] = cum_distances[j] / cum_num_hits[j]

                        # Etendue is equal to form factor times surface area times pi.
                        path_etendues[ij] = np.pi * patch_areas[i] * cum_cosines[j] / (rays_per_hemisphere * num_points)

                        for h in range(num_patches):
                            if cum_num_hits[h] == 0:
                                # No visibility between any point in h and any point in i.
                                continue

                            hi = path_index(h, i)

                            # Note: in theory, the diffuse kernel integral involves a multiplication by 2.
                            # In practice, we do not need it because each ray is counted once as "main" and once as specular.
                            diffuse_kernel[hi, ij] = cum_cosines[j] / (rays_per_hemisphere * num_points)
                            specular_kernel[hi, ij] = cum_specular_kernel[h, j] / cum_num_hits[h]

                    # Advance the progress bar.
                    progress_bar.update()

    # These should theoretically be identical, but may not be due to the discretized integration.
    # Nevertheless, they should be close enough.
    path_visibility = (path_lengths != 0)
    reverse_path_visibility = np.zeros_like(path_visibility)
    for i in range(num_patches):
        for j in range(num_patches):
            reverse_path_visibility[path_index(i, j)] = path_visibility[path_index(j, i)]
    num_mismatches = np.count_nonzero(path_visibility & ~reverse_path_visibility)
    if num_mismatches != 0:
        path_visibility = path_visibility & reverse_path_visibility
        path_etendues[~path_visibility] = 0.

    # Assess numerical precision by comparing etendue symmetricity.
    reverse_path_etendues = np.zeros_like(path_etendues)
    for i in range(num_patches):
        for j in range(num_patches):
            reverse_path_etendues[path_index(i, j)] = path_etendues[path_index(j, i)]
    # Symmetric absolute percentage error. Note: etendues are guaranteed non-negative.
    mean_etendues = (path_etendues + reverse_path_etendues) / 2
    etendue_sape = 100 * np.divide(np.abs(path_etendues - reverse_path_etendues),
                                   mean_etendues,
                                   out=np.zeros_like(mean_etendues),
                                   where=(mean_etendues != 0))

    # Average the path lengths as well, to aid accuracy.
    reverse_path_lengths = np.zeros_like(path_lengths)
    for i in range(num_patches):
        for j in range(num_patches):
            reverse_path_lengths[path_index(i, j)] = path_lengths[path_index(j, i)]
    mean_lengths = (path_lengths + reverse_path_lengths) / 2

    # Drop all non-visible paths from the ART model.
    num_valid_paths = np.count_nonzero(path_visibility)
    etendue_sape = etendue_sape[path_visibility]
    path_lengths = mean_lengths[path_visibility]
    mean_etendues = mean_etendues[path_visibility]
    diffuse_kernel = lil_array(diffuse_kernel[path_visibility][:, path_visibility])
    specular_kernel = lil_array(specular_kernel[path_visibility][:, path_visibility])

    np.savetxt(os.path.join(folder_path, 'etendue_SAPE_{:.0f}pnts_{:d}rays.csv'.format(points_per_square_meter, rays_per_hemisphere)),
               etendue_sape, fmt='%.18f', delimiter=', ')
    print('Etendue SAPE with {:.0f} points/m2, {:d} rays:'.format(points_per_square_meter, rays_per_hemisphere))
    print('\t Median: {:.2f}%'.format(np.median(etendue_sape)))
    print('\t Average: {:.2f}%'.format(np.mean(etendue_sape)))
    print('\t Valid paths: {}'.format(num_valid_paths))

    # Evaluate the row sums of both kernels. All rows should sum to 1; any divergence is an artefact of numerical integration.
    # As such, we can use these to assess the accuracy of the integration.
    diffuse_row_sums = diffuse_kernel.sum(axis=1)
    specular_row_sums = specular_kernel.sum(axis=1)

    # Apply the normalization safely w.r.t. zero rows.
    # Also, switch to Compressed Sparse Row (CSR) format to make later operations more efficient.
    diffuse_row_normalization = np.divide(1., diffuse_row_sums,
                                          out=np.zeros(num_valid_paths),
                                          where=(diffuse_row_sums != 0))
    diffuse_kernel = csr_array(diags(diffuse_row_normalization) @ diffuse_kernel)
    specular_row_normalization = np.divide(1., specular_row_sums,
                                           out=np.zeros(num_valid_paths),
                                           where=(specular_row_sums != 0))
    specular_kernel = csr_array(diags(specular_row_normalization) @ specular_kernel)

    # Prepare the path indexing matrix. Note that:
    #   the indices in this matrix refer to the reduced list, after having removed paths with no visibility.
    #   the indices in this matrix start from 1 and go up to num_visible_paths.
    #   0 elements in this matrix denote invalid paths.
    # This will be used at runtime to relate a pair of patch indices to a propagation path index.
    num_registered_paths = 0
    for i in range(num_patches):
        for j in range(num_patches):
            if path_visibility[path_index(i, j)]:
                num_registered_paths += 1
                path_indexing[i, j] = num_registered_paths
    assert num_registered_paths == num_valid_paths
    # We'll need this to be in Compressed Sparse Row (CSR) format.
    path_indexing = csr_array(path_indexing)

    if save_kernels:
        # Write the core ART parameters.
        mmwrite(os.path.join(folder_path, 'ART_kernel_diffuse.mtx'),
                diffuse_kernel, field='real', symmetry='general',
                comment='Diffuse (Lambertian) component of the acoustic radiance transfer reflection kernel. ' +
                        'Generated using {:.0f} points per square meter and {:d} rays per hemisphere. '.format(points_per_square_meter, rays_per_hemisphere) +
                        'Propagation path etendues have a symmetric mean absolute percentage error (SMAPE) of {:.2f}.'.format(np.mean(etendue_sape)))
        mmwrite(os.path.join(folder_path, 'ART_kernel_specular.mtx'),
                specular_kernel, field='real', symmetry='general',
                comment='Specular component of the acoustic radiance transfer reflection kernel. ' +
                        'Generated using {:.0f} points per square meter and {:d} rays per hemisphere. '.format(points_per_square_meter, rays_per_hemisphere) +
                        'Propagation path etendues have a symmetric mean absolute percentage error (SMAPE) of {:.2f}.'.format(np.mean(etendue_sape)))
        mmwrite(os.path.join(folder_path, 'path_indexing.mtx'),
                path_indexing, field='integer', symmetry='general',
                comment='Relates each pair of surface patch indices to the index of a propagation path. ' +
                        'Zero elements denote invalid paths; patch and path indices both start from 1.')
        np.savetxt(os.path.join(folder_path, 'path_lengths.csv'), path_lengths, fmt='%.18f', delimiter=', ')
        np.savetxt(os.path.join(folder_path, 'path_etendues.csv'), mean_etendues, fmt='%.18f', delimiter=', ')

    return folder_path


def compute_ART(folder_path: str,
                overwrite: bool = False,
                area_threshold: float = 0., thoroughness: float = 0.,
                points_per_square_meter: float = 30., rays_per_hemisphere: int = 1000,
                multiprocess_pool_size: int = 1,
                humidity: float = 50., temperature: float = 20., pressure: float = 100.
                ) -> str:
    """
    Build ART kernels and per-frequency-band reflection matrices for an environment.

    Depending on the presence of prior outputs and the overwrite flag, either
    reuse existing core ART files (diffuse/specular kernels, path indexing,
    lengths, and etendues) or compute them by integrating over patch surfaces.
    Path delays are computed from lengths using the speed of sound. For each
    frequency band in the loaded material data, a complete reflection kernel
    is assembled by weighting the diffuse and specular components by scattering,
    applying surface absorption, and applying air absorption along paths. All
    outputs are written to the given folder.

    Parameters
    ----------
    folder_path
        Path to the environment folder.
    overwrite
        If True, compute core ART files even if they already exist.
    area_threshold
        Optional patch-area simplification threshold applied when loading inputs.
    thoroughness
        Optional remeshing effort parameter applied when loading inputs.
    points_per_square_meter
        Surface sampling density used to integrate each patch.
    rays_per_hemisphere
        Number of rays traced from each surface sample point.
    multiprocess_pool_size
        Number of worker processes to use (1 disables multiprocessing).
    humidity
        Relative humidity (%) used for air absorption.
    temperature
        Air temperature (deg C) used for air absorption and sound speed.
    pressure
        Atmospheric pressure (kPa) used for air absorption.

    Returns
    -------
    folder_path
        The (possibly updated) environment folder path.

    Notes
    -----
    If `area_threshold > 0`, this may write a simplified mesh to a different folder.
    """

    if (type(folder_path) != str
            or type(overwrite) != bool
            or type(points_per_square_meter) != float
            or type(rays_per_hemisphere) != int
            or type(humidity) != float
            or type(temperature) != float
            or type(pressure) != float):
        raise ValueError('Please respect the type hints.')

    if not os.path.isdir(folder_path):
        raise ValueError('Not a valid folder path:\n\t' + folder_path)

    mesh, patch_materials, material_coefficients, folder_path = load_all_inputs(folder_path, area_threshold, thoroughness)

    num_patches = len(patch_materials)

    print('Running `compute_ART` in the environment "' + os.path.split(folder_path)[-1] + '"')

    multiprocess_pool_size = min(multiprocess_pool_size, os.cpu_count() - 1)
    num_processes = max(1, multiprocess_pool_size)
    if num_processes == 1:
        print('Will use a single process.')
    else:
        print(os.cpu_count(), 'cores available. Pool will use', num_processes, 'SUB-processes.')

    # dict of lists: indices of the triangles forming each patch.
    patch_triangles = dict()
    # Patch areas, as sum of triangle areas.
    patch_areas = np.zeros(num_patches)
    for triangle_index, triangle_patch_id in enumerate(mesh.patch_ids):
        if triangle_patch_id not in patch_triangles.keys():
            # This is the first triangle found for this patch. Create the list with one element.
            patch_triangles[triangle_patch_id] = [triangle_index]
        else:
            # This is not the first triangle found for this patch. Add element to the existing list.
            patch_triangles[triangle_patch_id].append(triangle_index)

        patch_areas[triangle_patch_id] += mesh.area[triangle_index]

    if num_patches > 1000:
        print('Warning: the mesh contains a very large number of patches (' + str(num_patches) + ')')
    if np.any(patch_areas < 0.1):
        print('Warning: the mesh contains very small patches (smallest area: ' + str(np.min(patch_areas)) + ')')

    # This is defined here in order to bake `num_patches` into it.
    def path_index(i: int, j: int) -> int:
        return i + (j * num_patches)

    # For debugging: plot the surface sample points, adding one triangle at a time.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    
    all_plots = list()
    for i in range(num_patches):
        for triangle_idx in patch_triangles[i]:
            # Uniformly sample the triangle's surface.
            sample_points = mesh.sample_triangle(triangle_idx, points_per_square_meter)

            all_plots.append(dict())
            all_plots[-1]['Sample points'] = sample_points
            all_plots[-1]['Triangle normal'] = mesh.n[triangle_idx]
            all_plots[-1]['Triangle vertex'] = mesh.v_1[triangle_idx]
            all_plots[-1]['Triangle edge 1'] = mesh.edge_1[triangle_idx]
            all_plots[-1]['Triangle edge 2'] = mesh.edge_2[triangle_idx]

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

    if (not overwrite
            and os.path.isfile(os.path.join(folder_path, 'ART_kernel_diffuse.mtx'))
            and os.path.isfile(os.path.join(folder_path, 'ART_kernel_specular.mtx'))
            and os.path.isfile(os.path.join(folder_path, 'path_indexing.mtx'))
            and os.path.isfile(os.path.join(folder_path, 'path_lengths.csv'))
            and os.path.isfile(os.path.join(folder_path, 'path_etendues.csv'))):
        print('\nCore ART files already exist. They will be read and re-used.')
        print('Current material data will be read and used to make new frequency-band kernels.')
        print('If you want to overwrite the existing core files, pass the argument `--overwrite` to the script.')

        path_lengths = np.loadtxt(os.path.join(folder_path, 'path_lengths.csv'), delimiter=',')
        path_etendues = np.loadtxt(os.path.join(folder_path, 'path_etendues.csv'), delimiter=',')
        diffuse_kernel = mmread(os.path.join(folder_path, 'ART_kernel_diffuse.mtx'), spmatrix=True).tocsr()
        specular_kernel = mmread(os.path.join(folder_path, 'ART_kernel_specular.mtx'), spmatrix=True).tocsr()
        path_indexing = mmread(os.path.join(folder_path, 'path_indexing.mtx'), spmatrix=True).tocsr()

        num_valid_paths = len(path_lengths)
    else:
        # Initialize `path_lengths`, `path_etendues`, `diffuse_kernel`, and `specular_kernel`.
        # The path etendues are used to assess the integration accuracy, and are also needed to scale MoD-ART eigenvectors.
        num_paths = num_patches ** 2
        path_lengths = np.zeros(num_paths)
        path_etendues = np.zeros(num_paths)
        diffuse_kernel = lil_array((num_paths, num_paths))
        specular_kernel = lil_array((num_paths, num_paths))
        path_indexing = lil_array((num_patches, num_patches), dtype=int)

        # Parallelize integration across patches
        if multiprocess_pool_size == 1:
            for i in tqdm(range(num_patches), desc='ART surface integral (# patches)'):
                # These accumulators will be built up at each surface sample point, and combined after the loop to form the patch contributions.
                # Refer to "ART_theory.md" for more info on this process.
                returned_tuple = integrate_patch((mesh, num_patches, i, patch_triangles[i],
                                                  rays_per_hemisphere, points_per_square_meter))
                cum_distances, cum_cosines, cum_num_hits, cum_specular_kernel, num_points, i = returned_tuple

                # Normalize accumulators and add to global trackers.
                for j in range(num_patches):
                    if cum_num_hits[j] == 0:
                        # No visibility between any point in j and any point in i.
                        continue

                    ij = path_index(i, j)

                    path_lengths[ij] = cum_distances[j] / cum_num_hits[j]

                    # Etendue is equal to form factor times surface area times pi.
                    path_etendues[ij] = np.pi * patch_areas[i] * cum_cosines[j] / (rays_per_hemisphere * num_points)

                    for h in range(num_patches):
                        if cum_num_hits[h] == 0:
                            # No visibility between any point in h and any point in i.
                            continue

                        hi = path_index(h, i)

                        # Note: in theory, the diffuse kernel integral involves a multiplication by 2.
                        # In practice, we do not need it because each ray is counted once as "main" and once as specular.
                        diffuse_kernel[hi, ij] = cum_cosines[j] / (rays_per_hemisphere * num_points)
                        specular_kernel[hi, ij] = cum_specular_kernel[h, j] / cum_num_hits[h]
        else:
            task_list = list()
            for i in range(num_patches):
                task = (mesh, num_patches, i, patch_triangles[i], rays_per_hemisphere, points_per_square_meter)
                task_list.append(task)

            with multiprocessing.Pool(multiprocess_pool_size) as pool:
                patch_contributions = pool.imap_unordered(integrate_patch, task_list)

                # https://stackoverflow.com/a/41921948
                # https://stackoverflow.com/a/72514814
                with tqdm(total=num_patches, desc='ART surface integral (# patches)',
                          miniters=min(int(num_patches/10), multiprocess_pool_size*10), maxinterval=600) as progress_bar:
                    for returned_tuple in patch_contributions:
                        cum_distances, cum_cosines, cum_num_hits, cum_specular_kernel, num_points, i = returned_tuple

                        # Normalize accumulators and add to global trackers.
                        for j in range(num_patches):
                            if cum_num_hits[j] == 0:
                                # No visibility between any point in j and any point in i.
                                continue

                            ij = path_index(i, j)

                            path_lengths[ij] = cum_distances[j] / cum_num_hits[j]

                            # Etendue is equal to form factor times surface area times pi.
                            path_etendues[ij] = np.pi * patch_areas[i] * cum_cosines[j] / (rays_per_hemisphere * num_points)

                            for h in range(num_patches):
                                if cum_num_hits[h] == 0:
                                    # No visibility between any point in h and any point in i.
                                    continue

                                hi = path_index(h, i)

                                # Note: in theory, the diffuse kernel integral involves a multiplication by 2.
                                # In practice, we do not need it because each ray is counted once as "main" and once as specular.
                                diffuse_kernel[hi, ij] = cum_cosines[j] / (rays_per_hemisphere * num_points)
                                specular_kernel[hi, ij] = cum_specular_kernel[h, j] / cum_num_hits[h]

                        # Advance the progress bar.
                        progress_bar.update()

        # These should theoretically be identical, but may not be due to the discretized integration.
        # Nevertheless, they should be close enough.
        path_visibility = (path_lengths != 0)
        reverse_path_visibility = np.zeros_like(path_visibility)
        for i in range(num_patches):
            for j in range(num_patches):
                reverse_path_visibility[path_index(i, j)] = path_visibility[path_index(j, i)]
        num_mismatches = np.count_nonzero(path_visibility & ~reverse_path_visibility)
        if num_mismatches != 0:
            print('\n' + str(num_mismatches) + ' pairs of patches have mismatched visibility (one sees the other, but not vice versa).')
            print('This makes up {:.2f}% of all possible propagation paths ({} in total).'.format(num_mismatches / num_paths, num_paths))
            print('If this seems a bit too high, consider increasing `points_per_square_meter` and/or `rays_per_hemisphere`.')
            print('If it seems way too high, check the environment geometry.')
            print('The mismatched pairs will be dropped (i.e., we assume these paths have no visibility).')
            path_visibility = path_visibility & reverse_path_visibility
            # Delete etendues where visibility is not mutual.
            # Where visibility isn't mutual, the etendue is 0 from one side and tiny but nonzero from the other, which skews the upcoming assessment.
            path_etendues[~path_visibility] = 0.

        # Assess numerical precision by comparing etendue symmetricity.
        reverse_path_etendues = np.zeros_like(path_etendues)
        for i in range(num_patches):
            for j in range(num_patches):
                reverse_path_etendues[path_index(i, j)] = path_etendues[path_index(j, i)]
        # Symmetric absolute percentage error. Note: etendues are guaranteed non-negative.
        mean_etendues = (path_etendues + reverse_path_etendues) / 2
        etendue_sape = 100 * np.divide(np.abs(path_etendues - reverse_path_etendues),
                                       mean_etendues,
                                       out=np.zeros_like(mean_etendues),
                                       where=(mean_etendues != 0))
        print('\nSymmetric absolute percentage errors (SAPE) of propagation path etendues:')
        print('\t Maximum: {:.2f}%'.format(np.max(etendue_sape)))
        print('\t Average: {:.2f}%'.format(np.mean(etendue_sape)))
        print('\t Median: {:.2f}%'.format(np.median(etendue_sape)))
        print('The propagation path etendues should be symmetric, i.e., the SAPEs should be low.')
        print('If they seem too high, consider increasing `points_per_square_meter` and/or `rays_per_hemisphere`.')
        print('N.B.: The etendue values are based on the diffuse kernel before it is normalized.')
        print('      If the diffuse kernel row sums are significantly different from 1, the upcoming normalization may skew this assessment.')
        # For debugging: plot the etendues.
        """
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(dpi=200, figsize=(8, 6))
        plt.plot(path_etendues[path_visibility])
        plt.plot(reverse_path_etendues[path_visibility])
        plt.tight_layout()
        plt.show()
        """

        # Average the path lengths as well, to aid accuracy.
        reverse_path_lengths = np.zeros_like(path_lengths)
        for i in range(num_patches):
            for j in range(num_patches):
                reverse_path_lengths[path_index(i, j)] = path_lengths[path_index(j, i)]
        mean_lengths = (path_lengths + reverse_path_lengths) / 2

        # Drop all non-visible paths from the ART model.
        num_valid_paths = np.count_nonzero(path_visibility)
        path_lengths = mean_lengths[path_visibility]
        mean_etendues = mean_etendues[path_visibility]
        diffuse_kernel = lil_array(diffuse_kernel[path_visibility][:, path_visibility])
        specular_kernel = lil_array(specular_kernel[path_visibility][:, path_visibility])

        # Evaluate the row sums of both kernels. All rows should sum to 1; any divergence is an artefact of numerical integration.
        # As such, we can use these to assess the accuracy of the integration.
        diffuse_row_sums = diffuse_kernel.sum(axis=1)
        specular_row_sums = specular_kernel.sum(axis=1)

        # Note: the specular kernel may have 0-sum rows even after removing paths without visibility.
        diffuse_row_sums_rmse = np.sqrt(np.mean(np.abs(diffuse_row_sums - 1.) ** 2))
        specular_row_sums_rmse = np.sqrt(np.mean(np.abs(specular_row_sums[specular_row_sums != 0] - 1.) ** 2))

        print('\nThe kernel rows sum to 1 with a root mean squared error (RMSE) of',
              '{:.2e} for the diffuse kernel and {:.2e} for the specular kernel.'.format(diffuse_row_sums_rmse, specular_row_sums_rmse))
        print('If either of these seems too high, consider increasing `points_per_square_meter` and/or `rays_per_hemisphere`.')
        print('The row sums will now be forcibly normalized.')

        # Apply the normalization safely w.r.t. zero rows.
        # Also, switch to Compressed Sparse Row (CSR) format to make later operations more efficient.
        diffuse_row_normalization = np.divide(1., diffuse_row_sums,
                                              out=np.zeros(num_valid_paths),
                                              where=(diffuse_row_sums != 0))
        diffuse_kernel = csr_array(diags(diffuse_row_normalization) @ diffuse_kernel)
        specular_row_normalization = np.divide(1., specular_row_sums,
                                               out=np.zeros(num_valid_paths),
                                               where=(specular_row_sums != 0))
        specular_kernel = csr_array(diags(specular_row_normalization) @ specular_kernel)

        # For debugging: plot the row sums after normalization.
        """
        diffuse_row_sums = diffuse_kernel.sum(axis=1)
        specular_row_sums = specular_kernel.sum(axis=1)
        diffuse_row_sums_rmse = np.sqrt(np.mean(np.abs(diffuse_row_sums - 1.) ** 2))
        specular_row_sums_rmse = np.sqrt(np.mean(np.abs(specular_row_sums[specular_row_sums != 0] - 1.) ** 2))
    
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(dpi=200, figsize=(8, 6))
        plt.plot(diffuse_row_sums, label='diffuse (RMSE {:.2e})'.format(diffuse_row_sums_rmse))
        plt.plot(specular_row_sums, label='specular (RMSE {:.2e})'.format(specular_row_sums_rmse))
        plt.tight_layout()
        plt.legend()
        plt.show()
        """

        # Prepare the path indexing matrix. Note that:
        #   the indices in this matrix refer to the reduced list, after having removed paths with no visibility.
        #   the indices in this matrix start from 1 and go up to num_visible_paths.
        #   0 elements in this matrix denote invalid paths.
        # This will be used at runtime to relate a pair of patch indices to a propagation path index.
        num_registered_paths = 0
        for i in range(num_patches):
            for j in range(num_patches):
                if path_visibility[path_index(i, j)]:
                    num_registered_paths += 1
                    path_indexing[i, j] = num_registered_paths
        assert num_registered_paths == num_valid_paths
        # We'll need this to be in Compressed Sparse Row (CSR) format.
        path_indexing = csr_array(path_indexing)

        # Write the core ART parameters.
        mmwrite(os.path.join(folder_path, 'ART_kernel_diffuse.mtx'),
                diffuse_kernel, field='real', symmetry='general',
                comment='Diffuse (Lambertian) component of the acoustic radiance transfer reflection kernel. ' +
                'Generated using {:.0f} points per square meter and {:d} rays per hemisphere. '.format(points_per_square_meter, rays_per_hemisphere) +
                'Propagation path etendues have a symmetric mean absolute percentage error (SMAPE) of {:.2f}.'.format(np.mean(etendue_sape)))
        mmwrite(os.path.join(folder_path, 'ART_kernel_specular.mtx'),
                specular_kernel, field='real', symmetry='general',
                comment='Specular component of the acoustic radiance transfer reflection kernel. ' +
                'Generated using {:.0f} points per square meter and {:d} rays per hemisphere. '.format(points_per_square_meter, rays_per_hemisphere) +
                'Propagation path etendues have a symmetric mean absolute percentage error (SMAPE) of {:.2f}.'.format(np.mean(etendue_sape)))
        mmwrite(os.path.join(folder_path, 'path_indexing.mtx'),
                path_indexing, field='integer', symmetry='general',
                comment='Relates each pair of surface patch indices to the index of a propagation path. ' +
                'Zero elements denote invalid paths; patch and path indices both start from 1.')
        np.savetxt(os.path.join(folder_path, 'path_lengths.csv'), path_lengths, fmt='%.18f', delimiter=', ')
        np.savetxt(os.path.join(folder_path, 'path_etendues.csv'), mean_etendues, fmt='%.18f', delimiter=', ')

    # Propagation delays in seconds, based on the path lengths in meters.
    # N.B.: These are prepared and saved separately in case the air parameters have been modified.
    path_delays = path_lengths / sound_speed(humidity, temperature, pressure)
    np.savetxt(os.path.join(folder_path, 'path_delays.csv'), path_delays, fmt='%.18f', delimiter=', ')

    # Construct the full ART reflection kernel for each frequency band.
    for band_idx, center_frequency in enumerate(material_coefficients['Frequencies']):
        # This will be the final reflection kernel for this frequency band:
        #   weighted sum of diffuse and specular kernels,
        #   scaled by wall absorption and air absorption.
        reflection_kernel = lil_array((num_valid_paths, num_valid_paths))

        for i, patch_mat in enumerate(patch_materials):
            # Retrieve the coefficients of patch i for this frequency band.
            patch_i_absorption = material_coefficients[patch_mat][0, band_idx]
            patch_i_scattering = material_coefficients[patch_mat][1, band_idx]

            # Locate all propagation paths which originate at patch i. See docs of `csr_array`.
            all_outgoing_paths_from_i = path_indexing.data[path_indexing.indptr[i]:path_indexing.indptr[i+1]]
            # N.B. The path indices are 1-based; we need them to be 0-based here.
            all_outgoing_paths_from_i -= 1

            # Weighted sum of diffuse and specular kernels.
            reflection_kernel[:, all_outgoing_paths_from_i] =\
                patch_i_scattering * diffuse_kernel[:, all_outgoing_paths_from_i]\
                + (1 - patch_i_scattering) * specular_kernel[:, all_outgoing_paths_from_i]

            # Add surface material energy losses.
            reflection_kernel[:, all_outgoing_paths_from_i] *= 1 - patch_i_absorption

        # Add air absorption energy losses (based on path lengths).
        air_absorption_pressure_gains = np.array([
            air_absorption_in_band(fc=center_frequency, fd=np.sqrt(2),  # Using full octave bands, the half-band factor is sqrt(2).
                                   distance=propagation_distance,
                                   humidity=humidity, temperature=temperature, pressure=pressure)
            for propagation_distance in path_lengths
        ])
        # Power level is the square of the pressure amplitude level.
        air_absorption_energy_gains = air_absorption_pressure_gains**2
        # Scale each column by the relative gain.
        reflection_kernel = reflection_kernel @ diags(air_absorption_energy_gains)
        # TODO: Air absorption, to be totally correct, should not be baked into the reflection kernel.
        #       Making it part of the matrix means that it's applied one too many times when MoD-ART is performed.
        #       In the future, the air_absorption_energy_gains will be saved to a separate file and applied alongside delays.

        # Write complete reflection kernel to ART_kernel_band_<band_idx>.mtx, where band_idx starts from 1.
        mmwrite(os.path.join(folder_path, 'ART_kernel_band_{}.mtx'.format(band_idx+1)),
                reflection_kernel, field='real', symmetry='general',
                comment='Complete acoustic radiance transfer reflection kernel, '
                'w.r.t. frequency band #{} (center freq. {:.2f}Hz). '.format(band_idx+1, center_frequency) +
                'Includes energy losses due to surface materials and air absorption over propagation paths.')

    print('\n')

    return folder_path
