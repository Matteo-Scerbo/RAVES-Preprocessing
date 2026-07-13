import os
import time
import numpy as np
from scipy.sparse import lil_array, csr_array, diags
from scipy.io import mmread, mmwrite
from collections import defaultdict

from raves.src.utils import load_all_inputs, air_absorption_in_band, sound_speed


def compute_ART(folder_path: str,
                humidity: float = 50., temperature: float = 20., pressure: float = 100.,
                assert_coplanarity: bool = True
                ) -> str:
    if not os.path.isdir(folder_path):
        raise ValueError('Not a valid folder path:\n\t' + folder_path)

    print('Running `compute_ART` in the environment "' + os.path.split(folder_path)[-1] + '"')

    runtimes = defaultdict(float)

    start_time = time.time()

    _, patch_materials, material_coefficients, _ = load_all_inputs(folder_path, assert_coplanarity=assert_coplanarity)

    runtimes['read mesh files'] = time.time() - start_time

    start_time = time.time()

    path_lengths = np.loadtxt(os.path.join(folder_path, 'path_lengths.csv'), delimiter=',')
    diffuse_kernel = mmread(os.path.join(folder_path, 'ART_kernel_diffuse.mtx'), spmatrix=True).tocsr()
    specular_kernel = mmread(os.path.join(folder_path, 'ART_kernel_specular.mtx'), spmatrix=True).tocsr()
    path_indexing = mmread(os.path.join(folder_path, 'path_indexing.mtx'), spmatrix=True).tocsr()

    runtimes['read ART files'] = time.time() - start_time

    start_time = time.time()

    num_valid_paths = len(path_lengths)
    # Propagation delays in seconds, based on the path lengths in meters.
    # N.B.: These are prepared and saved separately in case the air parameters have been modified.
    path_delays = path_lengths / sound_speed(humidity, temperature, pressure)
    np.savetxt(os.path.join(folder_path, 'path_delays.csv'), path_delays, fmt='%.18f', delimiter=', ')

    runtimes['write delays file'] = time.time() - start_time

    # Construct the full ART reflection kernel for each frequency band.
    for band_idx, center_frequency in enumerate(material_coefficients['Frequencies']):
        start_time = time.time()

        # This will be the final reflection kernel for this frequency band:
        #   weighted sum of diffuse and specular kernels,
        #   scaled by wall absorption and air absorption.
        reflection_kernel = lil_array((num_valid_paths, num_valid_paths))

        runtimes['init lil_array'] = time.time() - start_time

        for i, patch_mat in enumerate(patch_materials):
            start_time = time.time()

            # Retrieve the coefficients of patch i for this frequency band.
            patch_i_absorption = material_coefficients[patch_mat][0, band_idx]
            patch_i_scattering = material_coefficients[patch_mat][1, band_idx]

            runtimes['retrieve material coeffs'] = time.time() - start_time

            start_time = time.time()

            # Locate all propagation paths which originate at patch i. See docs of `csr_array`.
            all_outgoing_paths_from_i = path_indexing.data[path_indexing.indptr[i]:path_indexing.indptr[i+1]]

            runtimes['retrieve path idxs'] = time.time() - start_time

            start_time = time.time()

            # N.B. The path indices are 1-based; we need them to be 0-based here.
            all_outgoing_paths_from_i -= 1

            runtimes['adapt path idxs'] = time.time() - start_time

            start_time = time.time()

            coeff_d = patch_i_scattering * (1 - patch_i_absorption)
            coeff_s = (1 - patch_i_scattering) * (1 - patch_i_absorption)

            # Weighted sum of diffuse and specular kernels.
            reflection_kernel[:, all_outgoing_paths_from_i] =\
                coeff_d * diffuse_kernel[:, all_outgoing_paths_from_i]\
                + coeff_s * specular_kernel[:, all_outgoing_paths_from_i]

            runtimes['construct the matrices'] = time.time() - start_time

        start_time = time.time()

        # Add air absorption energy losses (based on path lengths).
        # Note: Using full octave bands, the half-band factor is sqrt(2).
        air_pressure_scaling = air_absorption_in_band(fc=center_frequency, fd=np.sqrt(2),
                                                      distance=path_lengths,
                                                      humidity=humidity,
                                                      temperature=temperature,
                                                      pressure=pressure,
                                                      energy_domain=True)
        
        runtimes['assess air absorption'] = time.time() - start_time

        start_time = time.time()

        # Scale each column by the relative gain.
        reflection_kernel = reflection_kernel @ diags(air_pressure_scaling)
        # TODO: Air absorption, to be totally correct, should not be baked into the reflection kernel.
        #       Making it part of the matrix means that it's applied one too many times when MoD-ART is performed.
        #       In the future, the air_absorption_energy_gains will be saved to a separate file and applied alongside delays.

        runtimes['apply air absorption'] = time.time() - start_time

        start_time = time.time()

        # Write complete reflection kernel to ART_kernel_band_<band_idx>.mtx, where band_idx starts from 1.
        mmwrite(os.path.join(folder_path, 'ART_kernel_band_{}.mtx'.format(band_idx+1)),
                reflection_kernel, field='real', symmetry='general',
                comment='Complete acoustic radiance transfer reflection kernel, '
                'w.r.t. frequency band #{} (center freq. {:.2f}Hz). '.format(band_idx+1, center_frequency) +
                'Includes energy losses due to surface materials and air absorption over propagation paths.')

        runtimes['write matrix files'] = time.time() - start_time

    for k, t in runtimes.items():
        print(f'Took {t:.2g} seconds to {k}.')
    print()


if __name__ == '__main__':
    air_parameters = {# 'CR1_DoorAngle1': (18.2, 47.6),
                      # 'CR1_DoorAngle3': (18.2, 47.6),
                      # 'CR2': (19.5, 41.7),
                      'CR3': (22.4, 40.9),
                      # 'CR4': (20.9, 37.5),
                      }

    mesh_folder = os.path.join('BRAS meshes')

    for room_name, params in air_parameters.items():
        for remeshing_strategy in ['naive_obj', 'naive_trng']:
            env_name = room_name + '_' + remeshing_strategy
            env_folder = os.path.join(mesh_folder, room_name, env_name)

            compute_ART(env_folder, assert_coplanarity=False,
                        temperature=params[0], humidity=params[1])
        