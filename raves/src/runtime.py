import os
import warnings
import numpy as np
import multiprocessing
from scipy.sparse import csr_array, lil_array
from scipy.io import mmread, mmwrite
from typing import Tuple, Dict, List

from .utils import RayBundle, TriangleMesh, load_all_inputs, load_frequencies, sound_speed


def energy_contributions(mesh: TriangleMesh,
                         path_indexing: csr_array,
                         position: np.ndarray,
                         overwrite: bool,
                         echogram_sample_rate: float,
                         num_rays: int = 1000,
                         output_file_path: str = None,
                         humidity: float = 50., temperature: float = 20., pressure: float = 100.
                         ) -> csr_array:
    """
    Trace rays from one position and gather them into propagation paths.

    This traces a ray bundle from the given position, over a full sphere of
    directions. For each traced ray, the closest surface patches in the front
    and back directions are found and used to determine the index of the
    propagation path that the ray falls into. Distance to the patch in front
    is also used to determine the propagation delay of the ray hit.
    All of this is accumulated into a (sparse) 2D array of dimensions
    (number of paths X samples in time), where each element states how many
    rays fell into each propagation path at each moment in time. The array is
    normalized by the total number of rays, such that it sums to 1.

    Parameters
    ----------
    mesh: TriangleMesh
        Mesh defining the environment geometry.
    path_indexing: scipy.sparse.csr_array
        Sparse 2D array relating each pair of surface patch indices to the
        index of a propagation path.
    position: numpy.ndarray
        3D coordinates of a single point in space (origin of the rays).
    overwrite: bool
        If True, perform the ray-tracing even if results already exist.
    echogram_sample_rate: float
        Sample rate used to discretize the propagation delays.
    num_rays: int, default: 1000
        Number of directions uniformly sampled on the sphere.
    output_file_path: str, default: None
        If not None, tracing results are saved to this file (should have `.mtx`
        extension). If `overwrite` is False and the file already exists,
        results are loaded without performing any tracing.
    humidity : float, default: 50.0
        Relative humidity (%) used for speed-of-sound computation.
    temperature : float, default: 20.0
        Air temperature (deg C) used for speed-of-sound computation.
    pressure : float, default: 100.0
        Atmospheric pressure (kPa) used for speed-of-sound computation.

    Returns
    -------
    contributions: scipy.sparse.csr_array
        A csr_array of shape (N, M), where N is the number of propagation paths
        and M is the time delay (in samples) of the latest contribution.
    
    Notes
    -----
    The results are dependent on the selected sampe rate, which needs to be
    compensated when building echograms. Consider specifying the sample rate
    in the name of the file or directory.
    
    The output array, being in CSR format, allows easy access to individual
    "filters" in the operator. See `operator_value_at_z()` for an example.
    
    Air absorption is not accounted for. The air parameters are used only to
    compute the speed of sound.
    """
    if position.shape != (3,):
        raise ValueError('The position must be a 1D array of length 3.')
    
    if output_file_path is None:
        print('\tNo output folder specified.\n\tComputing ray-tracing...')
        load_existing = False
    elif not os.path.isfile(output_file_path):
        print('\tOutput folder specified. File:\n\t\t', output_file_path)
        print('\tFile does not exist, it will be created. Computing ray-tracing...')
        load_existing = False
    elif overwrite:
        print('\tOutput folder specified. File:\n\t\t', output_file_path)
        print('\tFile exists, but overwrite flag enabled. Re-computing ray-tracing...')
        load_existing = False
    else:
        print('\tOutput folder specified. File:\n\t\t', output_file_path)
        print('\tLoading existing ray-tracing results...')
        load_existing = True
    
    # IMPORTANT: path_indexing is 1-indexed to benefit from sparsity.
    num_paths = path_indexing.max()
    
    if load_existing:
        operator = csr_array(mmread(output_file_path, spmatrix=True))
    else:
        # Ray-tracing from given position.
        ray_pencil = RayBundle.sample_sphere(num_rays)
        ray_pencil.move_origins(position)
        ray_pencil.trace_all(mesh)

        # We need to know both the index of the surface patch in front of each
        #   ray, and the index of the surface patch "behind" each ray.
        front_patch_ids, back_patch_ids = ray_pencil.get_indices(copy=False)
        # We only care about the distance to the "front" hit.
        hit_distances, _ = ray_pencil.get_distances(copy=False)
        
        # If any ray did not find a valid hit, its distance is NaN.
        valid_hits = np.isfinite(hit_distances)
        
        # Convert the distances (in meters) to delays (in number of samples).
        c = sound_speed(humidity, temperature, pressure)
        hit_delays = np.zeros_like(hit_distances, dtype=int)
        hit_delays[valid_hits] = (hit_distances[valid_hits]
                                  * echogram_sample_rate
                                  / c)
        
        # Create sparse array for TD-ART "filters". The length in samples is
        #   based on the largest valid delay of any ray.
        # Note: this is constructed in LIL format for speed, then converted.
        operator = lil_array((num_paths, np.max(hit_delays[valid_hits]) + 1))
        
        # Populate the array based on the path and delay of each (valid) ray.
        for ray_idx, ray_valid in enumerate(valid_hits):
            if not ray_valid:
                continue
            
            ray_path_idx = path_indexing[front_patch_ids[ray_idx],
                                         back_patch_ids[ray_idx]]
            # IMPORTANT: path_indexing is 1-indexed to benefit from sparsity.
            ray_path_idx -= 1
            
            operator[ray_path_idx, hit_delays[ray_idx]] += 1
        
        # Normalize all gathered amounts by the number of (valid) rays.
        operator /= np.count_nonzero(valid_hits)
        # Convert to CSR format for easier handling down the road.
        operator = csr_array(operator)
        
        if output_file_path is not None:
            # Save sparse array to output_file_path (if provided).
            mmwrite(output_file_path, operator,
                    field='real', symmetry='general',
                    comment='Ray-tracing results from ' +
                            'position {}. '.format(np.round(position, 3)) +
                            'Echogram bins use ' +
                            'sample rate {:.0f}.'.format(echogram_sample_rate))
    
    return operator


def operator_value_at_z(operator: csr_array,
                        z: float, fs: float
                        ) -> np.ndarray:
    """
    Given a sparse representation of an FIR filter, evaluate its Z-transform at
    a given value of z.

    This traces a ray bundle from the given position, over a full sphere of
    directions. For each traced ray, the closest surface patches in the front
    and back directions are found and used to determine the index of the
    propagation path that the ray falls into. Distance to the patch in front
    is also used to determine the propagation delay of the ray hit.
    All of this is accumulated into a (sparse) 2D array of dimensions
    (number of paths X samples in time), where each element states how many
    rays fell into each propagation path at each moment in time. The array is
    normalized by the total number of rays, such that it sums to 1.

    Parameters
    ----------
    operator: scipy.sparse.csr_array
        A 2D sparse array of shape (N, M), where N is the number of channels
        in the FIR filter and M is the duration of the longest channel response
        in samples.
    z: float
        Value at which the Z-transform is evaluated. Note: this is defined
        w.r.t. a period in seconds, not samples (see notes).
    fs: float
        Sample rate which defines the filter.

    Returns
    -------
    result: numpy.ndarray
        An array of shape (N,), where N is the first dimension of the input.
        The second dimension of the input, being time, is collapsed.
    
    Notes
    -----
    Instead of expecting a z value which is normalized w.r.t. the sample rate,
    this function expects a z value normalized w.r.t. seconds and applies the
    sample rate to the delays, for reasons of numerical stability.
    """
    if len(operator.shape) != 2:
        raise ValueError('The operator must be a 2D array.')
    
    num_paths = operator.shape[0]
    result = np.zeros(num_paths)
    
    for path_idx in range(num_paths):
        # For an explanation of indptr, see the docs of csr_array or
        #   https://stackoverflow.com/a/52299730
        start = operator.indptr[path_idx]
        end = operator.indptr[path_idx+1]
        
        # These indices and values correspond to the time samples and
        #   energy amounts of individual contributions to path_idx.
        contrib_delays = operator.indices[start:end]
        contrib_amounts = operator.data[start:end]
        
        # The delays need to be expressed in seconds.
        contrib_delays = contrib_delays / fs
        
        # This is the crux of this function: computing the Z-transform of each
        #   filter at a specified value of z. This collapses the "time sample"
        #   dimension of the sparse array.
        # See "ART_theory.md" for details, specifically the end of section
        #   "ART injection and detection operators".
        for amount, delay in zip(contrib_amounts, contrib_delays):
            # Avoid a warning in case of underflow.
            with warnings.catch_warnings():
                contrib = amount * (z ** -delay)
            # In case of underflow (decay too fast and delay too long), void the contribution.
            if not np.isfinite(contrib):
                contrib = 0
            result[path_idx] += contrib
    
    return result


def run_in_band(folder_path: str, band_idx: int,
                num_samples: int,
                integer_delays: np.ndarray,
                injectors_list: List[np.ndarray],
                detectors_list: List[np.ndarray]
                ) -> np.ndarray:
    """
    Build echograms in a single frequency band using TD-ART.

    Parameters
    ----------
    folder_path: str
        Path to the environment folder.
    band_idx: int
        Zero-based index of the frequency band to run.
    num_samples: int
        Number of time samples in the computed echograms (see notes on sample rate).
    integer_delays: np.ndarray
        Length of each propagation path, in samples (see notes on sample rate).
    injectors_list: list of np.ndarray
        List of energy injection operators, one element for each source.
    detectors_list: list of np.ndarray
        List of energy detection operators, one element for each listener.

    Returns
    -------
    echograms: numpy.ndarray
        An array of shape (S, L, T) where:
        - S is the number of sources (even if only one is given);
        - L is the number of listeners (even if only one is given);
        - T is the duration of the echograms in number of samples.
    
    Notes
    -----
    The echogram sample rate is implicitly defined by the integer_delays, injectors, and detectors.
    Make sure the same rate is used for all components!
    """
    num_paths = len(integer_delays)
    num_sources = len(injectors_list)
    num_listeners = len(detectors_list)

    band_echograms = np.zeros((num_sources, num_listeners, num_samples))

    # ...load the corresponding kernel.
    kernel = csr_array(mmread(os.path.join(folder_path, 'ART_kernel_band_{}.mtx'.format(band_idx+1)), spmatrix=True))

    # print('\tFrequency band {}...'.format(band_idx+1))
    
    # For each source...
    for source_idx in range(num_sources):
        # ...select the appropriate injection operators.
        injectors = injectors_list[source_idx]
        
        # Prepare an array to hold the radiance of each propagation path.
        # N.B.: This is the memory-intensive bottleneck of TD-ART.
        radiance_per_path = np.zeros((num_paths, num_samples))
        
        # Populate the radiance array with the initial radiance (0th order).
        for path_idx in range(num_paths):
            # For an explanation of indptr, see the docs of csr_array or
            #   https://stackoverflow.com/a/52299730
            start = injectors.indptr[path_idx]
            end = injectors.indptr[path_idx+1]
            
            # These indices and values correspond to the time samples and
            #   energy amounts of individual contributions to path_idx.
            contrib_delays = injectors.indices[start:end]
            contrib_amounts = injectors.data[start:end]
            
            for amount, delay in zip(contrib_amounts, contrib_delays):
                if delay < num_samples:
                    radiance_per_path[path_idx, delay] += amount

        # `radiance_per_path` currently holds the energy contributions AT
        #   the surface patches, i.e., about to be reflected.
        # They need to be reflected once by applying the scattering matrix.
        radiance_per_path = kernel @ radiance_per_path

        # Main recursive loop of TD-ART. Recursively propagate radiance for
        #   each time step.
        for time_sample in range(num_samples):
            # Assemble a vector holding the output of each propagation path
            #   at the current time step. This is the radiance currently
            #   reaching each surface patch.
            state_vector = np.zeros(num_paths)
            for path_idx in range(num_paths):
                if integer_delays[path_idx] <= time_sample:
                    state_vector[path_idx] = radiance_per_path[path_idx, time_sample - integer_delays[path_idx]]
            # Reflect the propagated radiance, turning it into the radiance
            #   currently departing from each surface patch.
            # Add it back onto the state array.
            radiance_per_path[:, time_sample] += kernel @ state_vector
        
        # The recursive propagation is done. Radiance can now be detected,
        #   separately, by each listener. For each listener...
        for listener_idx in range(num_listeners):
            # ...select the appropriate detection operators.
            detectors = detectors_list[listener_idx]
            
            # Detect the radiance gathered onto each propagation path,
            #   scaling and delaying it as dictated by the detectors.
            for path_idx in range(num_paths):
                start = detectors.indptr[path_idx]
                end = detectors.indptr[path_idx+1]
                contrib_delays = detectors.indices[start:end]
                contrib_amounts = detectors.data[start:end]
                
                for amount, delay in zip(contrib_amounts, contrib_delays):
                    if delay < num_samples:
                        band_echograms[source_idx, listener_idx, delay:] += amount * radiance_per_path[path_idx, :num_samples-delay]

    return band_echograms


# https://stackoverflow.com/a/21130146
def energy_contributions_wrapper(args: Tuple[TriangleMesh, csr_array, np.ndarray, bool, float, int, str, float, float, float]
                                 ) -> csr_array:
    return energy_contributions(*args)
def run_in_band_wrapper(args: Tuple[str, int, int, np.ndarray, List[np.ndarray], List[np.ndarray]]
                        ) -> np.ndarray:
    return run_in_band(*args)


def run_ART(folder_path: str,
            source_positions: np.ndarray, listener_positions: np.ndarray,
            overwrite_sources: bool = False, overwrite_listeners: bool = False,
            echogram_sample_rate: float = 5e3,
            echogram_duration: float = 1.,
            num_rays: int = 1000,
            output_folder_path: str = None,
            multiprocess_pool_size: int = 4,
            humidity: float = 50., temperature: float = 20., pressure: float = 100.,
            assert_coplanarity: bool = True, assert_min_delays: bool = True
            ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build echograms using TD-ART.

    Depending on the presence of prior outputs and the overwrite flag, either
    reuse existing ray-tracing results or compute them.
    The ray-tracing results are used to generate operators which inject and
    detect energy in the propagation paths. The TD-ART model is then run.

    Parameters
    ----------
    folder_path: str
        Path to the environment folder.
    source_positions: numpy.ndarray
        Position(s) of the sound source(s). May have shape (N, 3) or just (3,).
    listener_positions: numpy.ndarray
        Position(s) of the listener(s). May have shape (N, 3) or just (3,).
    overwrite_sources: bool, default: False
        If True, compute ray-tracing from sources even if data already exists.
    overwrite_listeners: bool, default: False
        If True, compute ray-tracing from listeners even if data already exists.
    echogram_sample_rate: float, default: 5e3
        Sample rate in Hz used to quantize propagation delays.
    echogram_duration: float, default: 1.0
        Duration of the produced echograms, in seconds.
    num_rays: int, default: 1000
        Number of rays traced from each position.
    output_folder_path: str, default: None
        Path to the folder where ray-tracing results are saved (if provided).
        Specifying this is recommended to avoid repeated operations.
    multiprocess_pool_size
        Number of worker processes to use (1 disables multiprocessing).
    humidity : float, default: 50.0
        Relative humidity (%) used for speed-of-sound computation.
    temperature : float, default: 20.0
        Air temperature (deg C) used for speed-of-sound computation.
    pressure : float, default: 100.0
        Atmospheric pressure (kPa) used for speed-of-sound computation.
    assert_coplanarity : bool, default True
        If False, lift the restriction for all triangles in each patch to be coplanar.
    assert_min_delays : bool, default True
        If False, lift the restriction for all delay lines to be at least one sample.
        Delay lengths of 0 samples are artificially extended to be 1 sample long.

    Returns
    -------
    echograms: numpy.ndarray
        An array of shape (S, L, B, T) where:
        - S is the number of sources (even if only one is given);
        - L is the number of listeners (even if only one is given);
        - B is the number of frequency bands;
        - T is the duration of the echograms in number of samples.
    frequencies: numpy.ndarray
        The center frequency of each band.
    
    Notes
    -----
    Air absorption is not accounted for. The air parameters are used only to
    compute the speed of sound.
    """
    if not os.path.isdir(folder_path):
        raise ValueError('Not a valid folder path:\n\t' + folder_path)
    if output_folder_path is not None and not os.path.isdir(output_folder_path):
        raise ValueError('Not a valid folder path:\n\t' + output_folder_path)
    
    # If only one source/listener position was provided,
    #   add a dimension (of size 1) for consistency.
    size_message = 'The source and listener position arguments must either have size (3) or (N, 3) where N is the number of positions.'
    if len(source_positions.shape) == 1:
        if source_positions.shape[0] != 3:
            raise ValueError(size_message)
        source_positions = source_positions[None]
        num_sources = 1
    elif len(source_positions.shape) == 2:
        if source_positions.shape[1] != 3:
            raise ValueError(size_message)
        num_sources = source_positions.shape[0]
    else:
        raise ValueError(size_message)
    if len(listener_positions.shape) == 1:
        if listener_positions.shape[0] != 3:
            raise ValueError(size_message)
        listener_positions = listener_positions[None]
        num_listeners = 1
    elif len(listener_positions.shape) == 2:
        if listener_positions.shape[1] != 3:
            raise ValueError(size_message)
        num_listeners = listener_positions.shape[0]
    else:
        raise ValueError(size_message)

    # Load 3D mesh for ray-tracing.
    mesh, _, _, _ = load_all_inputs(folder_path, assert_coplanarity=assert_coplanarity)

    # Load frequency band centers.
    frequencies = load_frequencies(folder_path)
    num_bands = len(frequencies)
    
    # Read `path_lengths.csv` and `path_etendues.csv`.
    path_delays = np.loadtxt(os.path.join(folder_path, 'path_delays.csv'), delimiter=',')
    path_etendues = np.loadtxt(os.path.join(folder_path, 'path_etendues.csv'), delimiter=',')

    # Prepare integer propagation delays, warn if too short.
    integer_delays = (echogram_sample_rate * path_delays).astype(int)
    min_valid_rate = 1. / np.min(path_delays)
    min_recommended_rate = 10. / np.min(path_delays)
    if np.min(integer_delays) < 1:
        if assert_min_delays:
            raise ValueError('The echogram sample rate {:.0f} is too low for this environment. '.format(np.floor(echogram_sample_rate)) +
                            'It needs to be at least {:.0f} in order for all integer delays to be at least 1 sample. '.format(np.ceil(min_valid_rate)) +
                            'A value above {:.0f} is recommended. '.format(np.ceil(min_recommended_rate)))
        else:
            integer_delays = np.clip(integer_delays, 1, None)
            warnings.warn('The echogram sample rate {:.0f} is very low for this environment. '.format(np.floor(echogram_sample_rate)) +
                        'Consider increasing it to avoid excessive rounding of propagation delays. ' +
                        'A value above {:.0f} is recommended. '.format(np.ceil(min_recommended_rate)))
    elif np.min(integer_delays) < 10:
        warnings.warn('The echogram sample rate {:.0f} is very low for this environment. '.format(np.floor(echogram_sample_rate)) +
                      'Consider increasing it to avoid excessive rounding of propagation delays. ' +
                      'A value above {:.0f} is recommended. '.format(np.ceil(min_recommended_rate)))

    print('Running `run_ART` in the environment "' + os.path.split(folder_path)[-1] + '"')

    multiprocess_pool_size = min(multiprocess_pool_size, os.cpu_count() - 1)
    num_processes = max(1, multiprocess_pool_size)
    if num_processes == 1:
        print('Will use a single process.')
    else:
        print(os.cpu_count(), 'cores available. Pool will use', num_processes, 'SUB-processes.')

    # Load propagation path indexing (relates patch indices to path indices).
    path_indexing = csr_array(mmread(os.path.join(folder_path, 'path_indexing.mtx'), spmatrix=True))
    # IMPORTANT: path_indexing is 1-indexed to benefit from sparsity.
    num_paths = path_indexing.max()
    
    # Evaluate the path "visibility" from each source position.
    injectors_list = list()

    # Parallelize ray-tracing across sources.
    if multiprocess_pool_size == 1 or num_sources == 1:
        for source_idx in range(num_sources):
            print('Processing source', source_idx+1)

            if output_folder_path is not None:
                file_name = 'S{}_operator_{:.0f}Hz.mtx'.format(source_idx+1, echogram_sample_rate)
                operator_file_path = os.path.join(output_folder_path, file_name)
            else:
                operator_file_path = None
            
            # The 2D array returned by energy_contributions is a distribution of energy
            #   over the propagation paths, over time. The entire array sums to 1;
            #   it specifies how the point source's unit-energy pulse at time 0
            #   gets distributed among the propagation paths.
            injectors = energy_contributions(mesh, path_indexing,
                                             source_positions[source_idx],
                                             overwrite_sources,
                                             echogram_sample_rate,
                                             num_rays,
                                             operator_file_path,
                                             humidity, temperature, pressure)
        
            # Injectors need to be rescaled to convert units.
            # Refer to "ART_theory.md" for more info on this process.
            injectors *= 4*np.pi
            
            injectors_list.append(injectors)
    else:
        task_list = list()
        for source_idx in range(num_sources):
            if output_folder_path is not None:
                file_name = 'S{}_operator_{:.0f}Hz.mtx'.format(source_idx+1, echogram_sample_rate)
                operator_file_path = os.path.join(output_folder_path, file_name)
            else:
                operator_file_path = None
            
            task = (mesh, path_indexing,
                    source_positions[source_idx],
                    overwrite_sources,
                    echogram_sample_rate,
                    num_rays,
                    operator_file_path,
                    humidity, temperature, pressure)
            task_list.append(task)

        with multiprocessing.Pool(multiprocess_pool_size) as pool:
            # The 2D array returned by energy_contributions is a distribution of energy
            #   over the propagation paths, over time. The entire array sums to 1;
            #   it specifies how the point source's unit-energy pulse at time 0
            #   gets distributed among the propagation paths.
            process_pool_results = pool.imap(energy_contributions_wrapper, task_list)

            for injectors in process_pool_results:
                # Injectors need to be rescaled to convert units.
                # Refer to "ART_theory.md" for more info on this process.
                injectors *= 4*np.pi
                
                # Note: results of pool.imap() preserve the indexing order.
                injectors_list.append(injectors)

    # Evaluate the path "visibility" from each listener position.
    detectors_list = list()

    # Parallelize ray-tracing across listeners.
    if multiprocess_pool_size == 1 or num_listeners == 1:
        for listener_idx in range(num_listeners):
            print('Processing listener', listener_idx+1)

            if output_folder_path is not None:
                file_name = 'L{}_operator_{:.0f}Hz.mtx'.format(listener_idx+1, echogram_sample_rate)
                operator_file_path = os.path.join(output_folder_path, file_name)
            else:
                operator_file_path = None
            
            # The 2D array returned by energy_contributions is a distribution of energy
            #   over the propagation paths, over time. The entire array sums to 1;
            #   it specifies how energy reaching the point listener from different
            #   paths gets delayed and weighted.
            detectors = energy_contributions(mesh, path_indexing,
                                            listener_positions[listener_idx],
                                            overwrite_listeners,
                                            echogram_sample_rate,
                                            num_rays,
                                            operator_file_path,
                                            humidity, temperature, pressure)
            
            # Detectors need to be rescaled to convert units.
            # Refer to "ART_theory.md" for more info on this process.
            detectors = csr_array(detectors.multiply(4*np.pi / path_etendues[:, None]))
            
            detectors_list.append(detectors)
    else:
        task_list = list()
        for listener_idx in range(num_listeners):
            if output_folder_path is not None:
                file_name = 'L{}_operator_{:.0f}Hz.mtx'.format(listener_idx+1, echogram_sample_rate)
                operator_file_path = os.path.join(output_folder_path, file_name)
            else:
                operator_file_path = None
            
            task = (mesh, path_indexing,
                    listener_positions[listener_idx],
                    overwrite_listeners,
                    echogram_sample_rate,
                    num_rays,
                    operator_file_path,
                    humidity, temperature, pressure)
            task_list.append(task)

        with multiprocessing.Pool(multiprocess_pool_size) as pool:
            # The 2D array returned by energy_contributions is a distribution of energy
            #   over the propagation paths, over time. The entire array sums to 1;
            #   it specifies how energy reaching the point listener from different
            #   paths gets delayed and weighted.
            process_pool_results = pool.imap(energy_contributions_wrapper, task_list)

            for detectors in process_pool_results:
                # Detectors need to be rescaled to convert units.
                # Refer to "ART_theory.md" for more info on this process.
                detectors = csr_array(detectors.multiply(4*np.pi / path_etendues[:, None]))
        
                # Note: results of pool.imap() preserve the indexing order.
                detectors_list.append(detectors)

    print('All components ready. Assembling echograms.')

    # Prepare the output array.
    time_axis = np.arange(0., echogram_duration, 1/echogram_sample_rate)
    echograms = np.zeros((num_sources, num_listeners, num_bands, len(time_axis)))
    
    # Parallelize echogram building across frequency bands.
    # TODO: Choose parallelism based on max(num_sources, num_listeners, num_bands).
    if multiprocess_pool_size == 1 or num_bands == 1:
        # For each frequency band...
        for band_idx in range(num_bands):
            band_echograms = run_in_band(folder_path, band_idx,
                                         len(time_axis), integer_delays,
                                         injectors_list, detectors_list)
            
            echograms[:, :, band_idx, :] = band_echograms
    else:
        task_list = list()
        for band_idx in range(num_bands):
            task = (folder_path, band_idx,
                    len(time_axis), integer_delays,
                    injectors_list, detectors_list)
            task_list.append(task)

        with multiprocessing.Pool(multiprocess_pool_size) as pool:
            process_pool_results = pool.imap(run_in_band_wrapper, task_list)

            # Note: results of pool.imap() preserve the indexing order.
            for band_idx, band_echograms in enumerate(process_pool_results):
                echograms[:, :, band_idx, :] = band_echograms
    
    print('Adding line-of-sight components where unobstructed.')
    
    # Cast rays from each listener to each source, to determine if the
    #   line-of-sight is obstructed.
    los_ray_origins = np.repeat(listener_positions, num_sources, axis=0)
    los_ray_targets = np.tile(source_positions, (num_listeners, 1))
    los_ray_directions = los_ray_targets - los_ray_origins
    los_rays = RayBundle.from_origins_and_directions(los_ray_origins,
                                                     los_ray_directions)
    los_rays.trace_all(mesh)
    
    # Evaluate the propagation delays of unobstructed line-of-sight components.
    free_distances = np.linalg.norm(los_ray_directions, axis=-1)
    free_distances = free_distances.reshape(num_listeners, num_sources).T
    ray_distances, _ = los_rays.get_distances(copy=False)
    ray_distances = ray_distances.reshape(num_listeners, num_sources).T
    # The line-of-sight is unobstructed if the mesh hit is behind the listener.
    los_visibility = (ray_distances > free_distances)
    
    c = sound_speed(humidity, temperature, pressure)
    los_delays = (free_distances * echogram_sample_rate / c).astype(int)

    # Add line-of-sight component (in all bands) where unobstructed.
    for s in range(num_sources):
        for l in range(num_listeners):
            # Regardless of whether the line-of-sight is obstructed or not,
            #   there can be no energy earlier than the line-of-sight
            #   propagation delay. Use that fact to truncate any early excess.
            echograms[s, l, :, :los_delays[s, l]] = 0.
            
            if los_visibility[s, l]:
                # On top of the inverse-square-distance term, we need to
                #   divide by a "4 pi" term, due to the units of measure being
                #   used by convention (we assume a unit-power point source).
                contribution = 1 / (4 * np.pi * free_distances[s, l]**2)
                
                echograms[s, l, :, los_delays[s, l]] += contribution
    
    # Up to this point, the echograms have used the histrogram-like convention
    #   "The value of each echogram sample is the amount of energy which falls
    #    in that time bin."
    # Note that, with this convention, the echogram values are a function of
    #   the echogram sample rate: larger bins lead to more energy in each one.
    # In order to compare the TD-ART echograms with the MoD-ART ones, we want
    #   to use a sample-rate-agnostic "energy per second" convention. In other
    #   words, we want the output to be a sampling of the continuous-time
    #   acoustic intensity response (square of the continuous-time RIR).
    echograms *= echogram_sample_rate
    
    return echograms, frequencies


# TODO: Take a T60 threshold (max slopes per band) as argument.
# TODO: Add parallelization here, like in TD-ART.
def run_MoDART(folder_path: str,
               source_positions: np.ndarray, listener_positions: np.ndarray,
               overwrite_sources: bool = False, overwrite_listeners: bool = False,
               avoid_saving_residues: bool = True,
               echogram_sample_rate: float = 5e3,
               echogram_duration: float = 1.,
               num_rays: int = 1000,
               output_folder_path: str = None,
               humidity: float = 50., temperature: float = 20., pressure: float = 100.,
               assert_coplanarity: bool = True
               ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Build echograms using MoD-ART.

    Depending on the presence of prior outputs and the overwrite flag, either
    reuse existing ray-tracing results or compute them.
    The ray-tracing results are used to generate operators which inject and
    detect energy in the propagation paths. These are converted into residues,
    and the echograms are built as sums of weighted slopes.
    Modal parameters, including the residue components, are also returned.

    Parameters
    ----------
    folder_path: str
        Path to the environment folder.
    source_positions: numpy.ndarray
        Position(s) of the sound source(s). May have shape (N, 3) or just (3,).
    listener_positions: numpy.ndarray
        Position(s) of the listener(s). May have shape (N, 3) or just (3,).
    overwrite_sources: bool, default: False
        If True, compute ray-tracing from sources even if data already exists.
    overwrite_listeners: bool, default: False
        If True, compute ray-tracing from listeners even if data already exists.
    avoid_saving_residues: bool, default: True
        If True, always compute resiudes on the fly
        (they're usually more memory-expensive than compute-expensive).
    echogram_sample_rate: float, default: 5e3
        Sample rate in Hz used to quantize propagation delays.
    echogram_duration: float, default: 1.0
        Duration of the produced echograms, in seconds.
    num_rays: int, default: 1000
        Number of rays traced from each position.
    output_folder_path: str, default: None
        Path to the folder where ray-tracing results are saved (if provided).
        Specifying this is recommended to avoid repeated operations.
    humidity : float, default: 50.0
        Relative humidity (%) used for speed-of-sound computation.
    temperature : float, default: 20.0
        Air temperature (deg C) used for speed-of-sound computation.
    pressure : float, default: 100.0
        Atmospheric pressure (kPa) used for speed-of-sound computation.
    assert_coplanarity : bool, default True
        If False, lift the restriction for all triangles in each patch to be coplanar.

    Returns
    -------
    echograms
        An array of shape (S, L, B, T) where:
        - S is the number of sources (even if only one is given);
        - L is the number of listeners (even if only one is given);
        - B is the number of frequency bands;
        - T is the duration of the echograms in number of samples.
    frequencies: numpy.ndarray
        The center frequency of each band.
    MoDART_data: dict
        A dictionary containing the following NumPy arrays:
        - 'Band idx', shape (M,): integer index of each mode's frequency band.
        - 'T60', shape (M,): T60 of each mode.
        - 'V_hat', shape (M, N): right eigenvector of each mode.
        - 'W_hat', shape (M, N): left eigenvector of each mode.
        - 'Source residues', shape (S, M): source residues components.
        - 'Listener residues', shape (L, M): Listener residues components.
    
    Notes
    -----
    Air absorption is not accounted for. The air parameters are used only to
    compute the speed of sound.
    """
    if not os.path.isdir(folder_path):
        raise ValueError('Not a valid folder path:\n\t' + folder_path)
    if output_folder_path is not None and not os.path.isdir(output_folder_path):
        raise ValueError('Not a valid folder path:\n\t' + output_folder_path)
    
    # If only one source/listener position was provided,
    #   add a dimension (of size 1) for consistency.
    size_message = 'The source and listener position arguments must either have size (3) or (N, 3) where N is the number of positions.'
    if len(source_positions.shape) == 1:
        if source_positions.shape[0] != 3:
            raise ValueError(size_message)
        source_positions = source_positions[None]
        num_sources = 1
    elif len(source_positions.shape) == 2:
        if source_positions.shape[1] != 3:
            raise ValueError(size_message)
        num_sources = source_positions.shape[0]
    else:
        raise ValueError(size_message)
    if len(listener_positions.shape) == 1:
        if listener_positions.shape[0] != 3:
            raise ValueError(size_message)
        listener_positions = listener_positions[None]
        num_listeners = 1
    elif len(listener_positions.shape) == 2:
        if listener_positions.shape[1] != 3:
            raise ValueError(size_message)
        num_listeners = listener_positions.shape[0]
    else:
        raise ValueError(size_message)

    # Load 3D mesh for ray-tracing.
    mesh, _, _, _ = load_all_inputs(folder_path, assert_coplanarity=assert_coplanarity)

    # Load frequency band centers.
    frequencies = load_frequencies(folder_path)
    num_bands = len(frequencies)
    
    print('Running `run_MoDART` in the environment "' + os.path.split(folder_path)[-1] + '"')

    # Load propagation path indexing (relates patch indices to path indices).
    path_indexing = csr_array(mmread(os.path.join(folder_path, 'path_indexing.mtx'), spmatrix=True))
    # IMPORTANT: path_indexing is 1-indexed to benefit from sparsity.
    num_paths = path_indexing.max()
    
    # Load MoD-ART data (band index, T60, and eigenvectors of each mode).
    mode_band_idxs = np.zeros(0, dtype=int)
    mode_T60s = np.zeros(0)
    mode_V_hats = np.zeros((0, num_paths))
    mode_W_hats = np.zeros((0, num_paths))
    
    with open(os.path.join(folder_path, 'MoD-ART.csv'), 'r') as file:
        file_iterator = iter(file)
        for line1 in file_iterator:
            line2 = next(file_iterator)
            line3 = next(file_iterator)
    
            band_idx, mode_t60 = line1.split(',')
            band_idx = int(band_idx.strip())
            mode_t60 = float(mode_t60.strip())
    
            V_hat = np.fromstring(line2, sep=',')
            W_hat = np.fromstring(line3, sep=',')
            
            mode_band_idxs = np.append(mode_band_idxs, band_idx)
            mode_T60s = np.append(mode_T60s, mode_t60)
            mode_V_hats = np.append(mode_V_hats, V_hat[None], axis=0)
            mode_W_hats = np.append(mode_W_hats, W_hat[None], axis=0)
    
    # IMPORTANT: mode_band_idxs is 1-indexed in the file.
    # Change it for intuitiveness.
    mode_band_idxs -= 1
    
    MoDART_data = {'Band idx': mode_band_idxs,
                   'T60': mode_T60s,
                   'V_hat': mode_V_hats,
                   'W_hat': mode_W_hats}
    
    num_modes = len(mode_T60s)
    
    # Convert the T60 values to "energy decay per second", as needed later.
    mode_decays = 10 ** (-6 / mode_T60s)
    
    # Create residue arrays of the required shapes.
    source_residues = np.zeros((num_sources, num_modes))
    listener_residues = np.zeros((num_listeners, num_modes))

    # Evaluate the source residue components at each position, for each mode.
    for source_idx in range(num_sources):
        print('Processing source', source_idx+1)

        if output_folder_path is not None:
            file_name = 'S{}_operator_{:.0f}Hz.mtx'.format(source_idx+1, echogram_sample_rate)
            operator_file_path = os.path.join(output_folder_path, file_name)
            file_name = 'S{}_residue_{:.0f}Hz.mtx'.format(source_idx+1, echogram_sample_rate)
            residue_file_path = os.path.join(output_folder_path, file_name)
        else:
            operator_file_path = None
            residue_file_path = None
        
        if avoid_saving_residues:
            residue_file_path = None
        
        # The 2D array returned by this function is a distribution of energy
        #   over the propagation paths, over time. The entire array sums to 1;
        #   it specifies how the point source's unit-energy pulse at time 0
        #   gets distributed among the propagation paths.
        injectors = energy_contributions(mesh, path_indexing,
                                         source_positions[source_idx],
                                         overwrite_sources,
                                         echogram_sample_rate,
                                         num_rays,
                                         operator_file_path,
                                         humidity, temperature, pressure)
        
        # If a path is provided and data already exists, load the residues.
        # Otherwise, compute them and save the results.
        if (residue_file_path is None
                or not os.path.isfile(residue_file_path)
                or overwrite_sources):
            print('\tComputing residues...')
            for mode_idx in range(num_modes):
                # For the residues, we need to compute the Z-transform of each
                #   filter setting z at the pole value. See "ART_theory.md" for
                #   details, specifically the end of section
                #   "ART injection and detection operators".
                contributions = operator_value_at_z(injectors,
                                                    mode_decays[mode_idx],
                                                    echogram_sample_rate)
                
                # The processed energy contributions are combined with the
                #   LEFT eigenvector, which already includes the appropriate
                #   normalization terms. Again, see "ART_theory.md".
                source_residues[source_idx, mode_idx] = np.dot(mode_W_hats[mode_idx],
                                                               contributions)
            
            if residue_file_path is not None:
                np.savetxt(residue_file_path, source_residues, fmt='%.18f', delimiter=', ')
        else:
            print('\tLoading existing residues... File:\n\t\t', residue_file_path)
            source_residues = np.loadtxt(residue_file_path, delimiter=',')
        
    # Evaluate the listener residue components at each position, for each mode.
    for listener_idx in range(num_listeners):
        print('Processing listener', listener_idx+1)
        
        if output_folder_path is not None:
            file_name = 'L{}_operator_{:.0f}Hz.mtx'.format(listener_idx+1, echogram_sample_rate)
            operator_file_path = os.path.join(output_folder_path, file_name)
            file_name = 'L{}_residue_{:.0f}Hz.mtx'.format(listener_idx+1, echogram_sample_rate)
            residue_file_path = os.path.join(output_folder_path, file_name)
        else:
            operator_file_path = None
            residue_file_path = None
        
        if avoid_saving_residues:
            residue_file_path = None
        
        # The 2D array returned by this function is a distribution of energy
        #   over the propagation paths, over time. The entire array sums to 1;
        #   it specifies how energy reaching the point listener from different
        #   paths gets delayed and weighted.
        detectors = energy_contributions(mesh, path_indexing,
                                         listener_positions[listener_idx],
                                         overwrite_listeners,
                                         echogram_sample_rate,
                                         num_rays,
                                         operator_file_path,
                                         humidity, temperature, pressure)
        
        # If a path is provided and data already exists, load the residues.
        # Otherwise, compute them and save the results.
        if (residue_file_path is None
                or not os.path.isfile(residue_file_path)
                or overwrite_listeners):
            print('\tComputing residues...')
            for mode_idx in range(num_modes):
                # For the residues, we need to compute the Z-transform of each
                #   filter setting z at the pole value. See "ART_theory.md" for
                #   details, specifically the end of section
                #   "ART injection and detection operators".
                contributions = operator_value_at_z(detectors,
                                                    mode_decays[mode_idx],
                                                    echogram_sample_rate)
                
                # The processed energy contributions are combined with the
                #   RIGHT eigenvector, which already includes the appropriate
                #   normalization terms. Again, see "ART_theory.md".
                listener_residues[listener_idx, mode_idx] = np.dot(mode_V_hats[mode_idx],
                                                                   contributions)
            
            if residue_file_path is not None:
                np.savetxt(residue_file_path, listener_residues, fmt='%.18f', delimiter=', ')
        else:
            print('\tLoading existing residues... File:\n\t\t', residue_file_path)
            listener_residues = np.loadtxt(residue_file_path, delimiter=',')
    
    # Add the residues to the returned modal data.
    MoDART_data['Source residues'] = source_residues
    MoDART_data['Listener residues'] = listener_residues
    
    print('All residues ready. Assembling echograms.')
    
    # Construct echograms as sums of weighted slope terms.
    time_axis = np.arange(0., echogram_duration, 1/echogram_sample_rate)
    echograms = np.zeros((num_sources, num_listeners, num_bands, len(time_axis)))
    
    for mode_idx in range(num_modes):
        slope = mode_decays[mode_idx] ** time_axis

        residue_matrix = np.outer(listener_residues[:, mode_idx],
                                  source_residues[:, mode_idx]).T
        
        slope = residue_matrix[:, :, None] * slope[None, None]

        echograms[:, :, mode_band_idxs[mode_idx], :] += slope

    # Remove any negative values due to mode truncation.
    echograms = np.clip(echograms, 0, None)

    print('Adding line-of-sight components where unobstructed.')
    
    # Cast rays from each listener to each source, to determine if the
    #   line-of-sight is obstructed.
    los_ray_origins = np.repeat(listener_positions, num_sources, axis=0)
    los_ray_targets = np.tile(source_positions, (num_listeners, 1))
    los_ray_directions = los_ray_targets - los_ray_origins
    los_rays = RayBundle.from_origins_and_directions(los_ray_origins,
                                                     los_ray_directions)
    los_rays.trace_all(mesh)
    
    # Evaluate the propagation delays of unobstructed line-of-sight components.
    free_distances = np.linalg.norm(los_ray_directions, axis=-1)
    free_distances = free_distances.reshape(num_listeners, num_sources).T
    ray_distances, _ = los_rays.get_distances(copy=False)
    ray_distances = ray_distances.reshape(num_listeners, num_sources).T
    # The line-of-sight is unobstructed if the mesh hit is behind the listener.
    los_visibility = (ray_distances > free_distances)
    
    c = sound_speed(humidity, temperature, pressure)
    los_delays = (free_distances * echogram_sample_rate / c).astype(int)

    # Add line-of-sight component (in all bands) where unobstructed.
    for s in range(num_sources):
        for l in range(num_listeners):
            # Regardless of whether the line-of-sight is obstructed or not,
            #   there can be no energy earlier than the line-of-sight
            #   propagation delay. Use that fact to truncate any early excess.
            echograms[s, l, :, :los_delays[s, l]] = 0.
            
            if los_visibility[s, l]:
                # On top of the inverse-square-distance term, we need to
                #   divide by a "4 pi" term, due to the units of measure being
                #   used by convention (we assume a unit-power point source).
                contribution = 1 / (4 * np.pi * free_distances[s, l]**2)
                
                # In order to match the rest of the echogram, we need to use a
                #   sample-rate-agnostic "energy per second" convention for
                #   these values. As such, they need to be adjusted (see notes
                #   in `run_ART()`.
                contribution *= echogram_sample_rate
                
                echograms[s, l, :, los_delays[s, l]] += contribution
    
    return echograms, frequencies, MoDART_data
    