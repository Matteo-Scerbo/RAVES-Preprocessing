import os
import warnings
import numpy as np
from scipy.io import mmread
from typing import Dict

from .utils import eig_to_T60, load_frequencies, build_ssm, real_positive_search


def plot_T60(folder_path: str,
             all_pole_T60s: Dict[int, np.ndarray] = None,
             max_slopes_per_band: int = 10,
             echogram_sample_rate: float = 5e3,
             show_figure: bool = False
             ) -> None:
    """
    Plot the set of energy modes located in each frequency band.

    Parameters
    ----------
    folder_path : str
        Path to the environment folder. If it contains `materials.csv`,
        the frequency band centers will be used for the X axis of the plots.
    all_pole_T60s : dict, default: None
        Dictonary containing the T60 values located in each frequency band,
        keyed by band index. If None, the values are loaded from MoD-ART.csv.
    max_slopes_per_band : int, default: 10
        Maximum number of modes reported per band in MoD-ART.csv.
    echogram_sample_rate : float, default: 5e3
        Sample rate in Hz used to quantize propagation delays.
    show_figure : bool, default: False
        If False, suppress the matplotlib figure from actually displaying
        (i.e., save only the file).

    Returns
    -------
    None
        Plots are saved to:
        - MoD-ART (rate {echogram_sample_rate}) T60 values, lin scale.png
        - MoD-ART (rate {echogram_sample_rate}) T60 values, log scale.png
    
    Notes
    -----
    If the values are loaded from an existing file, the echogram sample rate
    and the selection of values in 'MoD-ART' VS 'MoD-ART extra' are not known.
    The image files are saved to:
        - MoD-ART T60 values, lin scale.png
        - MoD-ART T60 values, log scale.png
    """
    if not os.path.isdir(folder_path):
        raise ValueError('Not a valid folder path:\n\t' + folder_path)

    if all_pole_T60s is None:
        # The sample rate used to prepare these values can't be loaded.
        echogram_sample_rate = None

        all_pole_T60s = dict()

        with open(os.path.join(folder_path, 'MoD-ART extra.csv'), 'r') as file:
            file_iterator = iter(file)
            for line1 in file_iterator:
                line2 = next(file_iterator)
                line3 = next(file_iterator)
        
                band_idx, mode_t60 = line1.split(',')
                band_idx = int(band_idx.strip())
                mode_t60 = float(mode_t60.strip())

                if band_idx not in all_pole_T60s.keys():
                    all_pole_T60s[band_idx] = list()
                
                all_pole_T60s[band_idx].append(mode_t60)
        
        all_pole_T60s = {band_idx: np.array(poles)
                         for band_idx, poles in all_pole_T60s.items()}
    
    try:
        import matplotlib.ticker as ticker
        import matplotlib.pyplot as plt
    except ImportError as e:
        print('Failed to import matplotlib package. Aborting T60 plots. Encountered error:\n', e)
        return

    try:
        frequencies = load_frequencies(folder_path)
        frequencies_load_succeded = True
    except Exception as e:
        print('Failed to import frequency band centers. The T60 plots will only specify band indices. Encountered error:\n', e)
        frequencies = np.range(1, len(all_pole_T60s)+1)
        frequencies_load_succeded = False

    print('Plotting results.')

    fig, ax = plt.subplots(dpi=200, figsize=(8, 8))

    for band_idx, T60s in all_pole_T60s.items():
        num_selected = min(len(T60s), max_slopes_per_band)

        if echogram_sample_rate is not None:
            plt.scatter(np.full(num_selected, frequencies[band_idx-1]),
                        T60s[:num_selected], marker='o',
                        facecolors='none', edgecolors='black')
        
        plt.scatter(np.full(len(T60s), frequencies[band_idx-1]),
                    T60s, marker='+')

    if echogram_sample_rate is not None:
        plt.title('The modes circled in black are reported in `MoD-ART.csv`.\nAll modes are reported in `MoD-ART extra.csv`.')

    if frequencies_load_succeded:
        plt.xlabel('Frequency band center [Hz]')
        plt.xscale('log')
        plt.xticks(frequencies)
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.xaxis.set_tick_params(which='minor', bottom=False)
    else:
        plt.xlabel('Frequency band index')
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.ylabel('T60 [s]')
    plt.grid(True, axis='y')
    plt.ylim(0, None)

    if echogram_sample_rate is None:
        plt.savefig(os.path.join(folder_path, 'MoD-ART T60 values, lin scale.png'))
    else:
        plt.savefig(os.path.join(folder_path, 'MoD-ART (rate {:.0f}) T60 values, lin scale.png'.format(echogram_sample_rate)))

    plt.yscale('log')
    plt.autoscale(axis='y')

    ax.yaxis.set_major_locator(ticker.LogLocator(subs=np.arange(0.1, 1, 0.1)))
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.yaxis.set_minor_locator(ticker.LogLocator(subs=np.arange(0.01, 1, 0.01)))
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())

    if echogram_sample_rate is None:
        plt.savefig(os.path.join(folder_path, 'MoD-ART T60 values, log scale.png'))
    else:
        plt.savefig(os.path.join(folder_path, 'MoD-ART (rate {:.0f}) T60 values, log scale.png'.format(echogram_sample_rate)))

    if show_figure:
        plt.show()
    else:
        plt.close()


def compute_MoDART(folder_path: str,
                   T60_threshold: float = 1e-1, max_slopes_per_band: int = 10,
                   echogram_sample_rate: float = 5e3, skip_T60_plots: bool = False,
                   raise_on_failure: bool = False
                   ) -> None:
    """
    Perform modal decomposition of acoustic radiance transfer for all frequency bands.

    This function reads path delays and etendues, assembles the state
    transition matrix per frequency band from the ART kernel and integer
    delays, searches for real positive poles above the given threshold,
    and writes mode eigen-pairs to CSV files. Optionally, it generates
    scatter plots of T60 values per band.

    Parameters
    ----------
    folder_path : str
        Path to the environment folder. Must contain:
        - path_delays.csv
        - path_etendues.csv
        - ART_kernel_band_1.mtx, ART_kernel_band_2.mtx, ...
    T60_threshold : float, default: 1e-1
        Minimum T60 (seconds) used to derive the eigenvalue threshold for pole search.
        The search will continue until a mode below this threshold is found.
    max_slopes_per_band : int, default: 10
        Maximum number of modes reported per band in `MoD-ART.csv`.
        Additional modes found will be reported in `MoD-ART extra.csv`.
    echogram_sample_rate : float, default: 5e3
        Sample rate in Hz used to quantize propagation delays.
    skip_T60_plots : bool, default: False
        If True, do not generate T60 scatter plots.
    raise_on_failure : bool, default False
        Raise a RuntimeError if no eigenvalues are found in a band.

    Returns
    -------
    None
        Results are written to:
        - MoD-ART.csv
        - MoD-ART extra.csv
        - Optional PNG plots of T60 values (linear and log scale).

    Notes
    -----
    - Integer propagation delays are computed as floor(echogram_sample_rate * delay).
      If the minimum integer delay is below 3, the state transition matrix cannot be
      constructed; values below 10 trigger a warning.
    - Kernels are processed in order: ART_kernel_band_1.mtx, ART_kernel_band_2.mtx, ...
      Modes are appended to the CSVs one frequency band at a time.
    - Modes in each band are sorted by decreasing T60. Eigenvectors are scaled as
      discussed in `ART_theory.md`.
    """
    if not os.path.isdir(folder_path):
        raise ValueError('Not a valid folder path:\n\t' + folder_path)

    print('Running `compute_MoDART` in the environment "' + os.path.split(folder_path)[-1] + '"')

    # Read `path_lengths.csv` and `path_etendues.csv`.
    path_delays = np.loadtxt(os.path.join(folder_path, 'path_delays.csv'), delimiter=',')
    path_etendues = np.loadtxt(os.path.join(folder_path, 'path_etendues.csv'), delimiter=',')

    # Prepare integer propagation delays.
    integer_delays = (echogram_sample_rate * path_delays).astype(int)
    min_valid_rate = 3. / np.min(path_delays)
    min_recommended_rate = 10. / np.min(path_delays)
    if np.min(integer_delays) < 3:
        raise ValueError('The echogram sample rate {:.0f} is too low for this environment. '.format(np.floor(echogram_sample_rate)) +
                         'It needs to be at least {:.0f} in order for all integer delays to be sufficient. '.format(np.ceil(min_valid_rate)) +
                         'A value above {:.0f} is recommended. '.format(np.ceil(min_recommended_rate)))
    elif np.min(integer_delays) < 10:
        warnings.warn('The echogram sample rate {:.0f} is very low for this environment. '.format(np.floor(echogram_sample_rate)) +
                      'Consider increasing it to avoid excessive rounding of propagation delays. ' +
                      'A value above {:.0f} is recommended. '.format(np.ceil(min_recommended_rate)))

    # Create `MoD-ART.csv` and `MoD-ART extra.csv` (if they exist, their contents are emptied).
    open(os.path.join(folder_path, 'MoD-ART.csv'), mode='w')
    open(os.path.join(folder_path, 'MoD-ART extra.csv'), mode='w')

    # Save all found poles in a dictionary, for plotting.
    all_pole_T60s = dict()

    # Decompose all kernels matching `ART_kernel_band_<band_idx>.mtx`. For each frequency band, results are appended to `MoD-ART.csv`.
    band_idx = 0
    while True:
        band_idx += 1
        if not os.path.isfile(os.path.join(folder_path, 'ART_kernel_band_{}.mtx'.format(band_idx))):
            if band_idx == 1:
                raise ValueError('Unable to run MoD-ART. ART kernel must be prepared for at least one frequency band (i.e., `ART_kernel_band_1.mtx` needs to exist).')
            else:
                break

        print('\nAnalyzing frequency band #{}.'.format(band_idx))

        # Load the kernel for this frequency band.
        kernel = mmread(os.path.join(folder_path, 'ART_kernel_band_{}.mtx'.format(band_idx)), spmatrix=True)

        print('\tGenerating full state transition matrix.')

        # Assemble the state transition matrix (extremely sparse).
        state_transition_matrix = build_ssm(kernel, integer_delays)

        # Perform modal decomposition, keeping only real positive eigenvalues.
        # N.B. These are the STATE-SPACE eigenvectors; their size is the system order.
        poles, right_vecs, left_vecs = \
            real_positive_search(ssm=state_transition_matrix,
                                 T60_thresh=T60_threshold,
                                 sample_rate=echogram_sample_rate,
                                 num_thresh=None,
                                 raise_on_failure=raise_on_failure)

        print('\tRearranging and scaling results.')

        # Rearrange the modes by decreasing T60.
        poles_order = np.argsort(np.abs(poles))[::-1]
        poles = poles[poles_order]
        right_vecs = right_vecs[:, poles_order]
        left_vecs = left_vecs[:, poles_order]

        # All following operations prepare the eigenvectors for RAVES. Refer to `ART_theory.md` for details.

        # Take the relevant slices (last sample of each delay line).
        N = kernel.shape[0]
        M = state_transition_matrix.shape[0]
        V = right_vecs[slice(M - 2 * N, M - N)]
        W = left_vecs[slice(M - 2 * N, M - N)]

        # Recall that V, W of length N are slices of the full state-space vectors of size M.
        # Given the structure of the s.s.m. used above, both V and W refer to the last sample of each delay line.
        # In the ART format used in RAVES:
        #   - energy is injected at the surface patches, as if it had just been propagated (about to be reflected)
        #   - energy is detected at the surface patches, as if it had just been reflected (about to be propagated)
        # As such, the "injection" eigenvector (W_hat) should refer directly to the last sample of each propagation line,
        # while the "detection" eigenvector (V_hat) should refer to the last sample of each propagation line AND apply scattering.
        # N.B.: If the "detection" eigenvector (V_hat) referred to the first sample of each line, it would differ by a one-sample delay.
        #       The true "next input" would be the "future first sample of each line", which is not explicitly a part of the state space.
        V_hat = kernel @ V
        W_hat = W

        # Prefer pairs of mostly positive vectors rather than pairs of mostly negative vectors (kind of inconsequential).
        V_signs = np.sign(np.mean(V_hat, axis=0))
        V_hat *= V_signs[None]
        W_hat *= V_signs[None]

        # Scale by the path etendues to "translate" quantities between power and radiance.
        # The signals circulating in the loop are power, and must be translated to radiance
        # in order to use solid angles as detectors. This means dividing by the path etendue (P = G * L).
        V_hat /= path_etendues[:, None]

        # The injectors and detectors, being solid angles, should both sum to 4 pi.
        # Given the way we perform ray-tracing in practice, they sum to 1 instead.
        # Apply the "4 pi" factor here, to save some multiplications at runtime.
        V_hat *= 4 * np.pi
        W_hat *= 4 * np.pi

        # Append results to `MoD-ART.csv` and `MoD-ART extra.csv` (the former limits the number of modes per band, the latter does not).
        with open(os.path.join(folder_path, 'MoD-ART.csv'), mode='a') as file:
            for p in range(min(len(poles), max_slopes_per_band)):
                file.write(str(band_idx) + ', ' + str(eig_to_T60(poles[p], echogram_sample_rate)) + '\n')
                file.write(', '.join([str(v) for v in V_hat[:, p]]) + '\n')
                file.write(', '.join([str(w) for w in W_hat[:, p]]) + '\n')
        with open(os.path.join(folder_path, 'MoD-ART extra.csv'), mode='a') as file:
            for p in range(len(poles)):
                file.write(str(band_idx) + ', ' + str(eig_to_T60(poles[p], echogram_sample_rate)) + '\n')
                file.write(', '.join([str(v) for v in V_hat[:, p]]) + '\n')
                file.write(', '.join([str(w) for w in W_hat[:, p]]) + '\n')

        all_pole_T60s[band_idx] = [eig_to_T60(p, echogram_sample_rate) for p in poles]

    if not skip_T60_plots:
        plot_T60(folder_path,
                 all_pole_T60s,
                 max_slopes_per_band,
                 echogram_sample_rate)

    print('\n')
