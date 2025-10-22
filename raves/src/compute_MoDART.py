import os
import warnings
import numpy as np
from scipy.io import mmread

from .utils import build_ssm, real_positive_search


def eig_to_T60(eigenvalue: float, fs: float) -> float:
    # TODO: Fill out documentation properly.
    """

    Args:
        eigenvalue:
        fs:

    Returns:

    """
    if np.abs(eigenvalue) >= 1:
        return np.inf
    elif np.abs(eigenvalue) == 0:
        return 0.
    else:
        return -6 / (np.log10(np.abs(eigenvalue)) * fs)


def T60_to_eig(T60: float, fs: float) -> float:
    # TODO: Fill out documentation properly.
    """

    Args:
        T60:
        fs:

    Returns:

    """
    if T60 == 0:
        return 0.
    elif np.isfinite(T60):
        return 10**(-6 / (T60 * fs))
    else:
        return 1.


def compute_MoDART(folder_path: str,
                   T60_threshold: float = 1e-1, max_slopes_per_band: int = 10,
                   echogram_sample_rate: float = 1e3, skip_T60_plots: bool = False
                   ) -> None:
    # TODO: Fill out documentation properly.
    """

    Args:
        folder_path:
        echogram_sample_rate:
        T60_threshold:
        max_slopes_per_band:
        skip_T60_plots:

    Returns:

    """
    if (type(folder_path) != str
            or type(T60_threshold) != float
            or type(max_slopes_per_band) != int
            or type(echogram_sample_rate) != float
            or type(skip_T60_plots) != bool):
        raise ValueError('Please respect the type hints.')

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

    # Create `MoD-ART.csv` (if it exists, its contents are emptied).
    open(os.path.join(folder_path, 'MoD-ART.csv'), mode='w')

    # Save all found poles in a dictionary, for plotting.
    all_pole_T60s = dict()

    # Decompose all kernels matching `ART_kernel_<band_idx>.mtx`. For each frequency band, results are appended to `MoD-ART.csv`.
    band_idx = 0
    while True:
        band_idx += 1
        if not os.path.isfile(os.path.join(folder_path, 'ART_kernel_{}.mtx'.format(band_idx))):
            if band_idx == 1:
                raise ValueError('Unable run MoD-ART. ART kernel must be prepared for at least one frequency band (i.e., `ART_kernel_1.mtx` needs to exist).')
            else:
                break

        print('\nAnalyzing frequency band #{}.'.format(band_idx))

        # Load the kernel for this frequency band.
        kernel = mmread(os.path.join(folder_path, 'ART_kernel_{}.mtx'.format(band_idx)), spmatrix=True)

        print('\tGenerating full state transition matrix.')

        # Assemble the state transition matrix (extremely sparse).
        state_transition_matrix = build_ssm(kernel, integer_delays)

        # Perform modal decomposition, keeping only real positive eigenvalues.
        poles, right_vecs, left_vecs = real_positive_search(state_transition_matrix,
                                                            T60_to_eig(T60_threshold, echogram_sample_rate),
                                                            max_slopes_per_band)

        print('\tRearranging and scaling results.')

        # Rearrange the modes by decreasing T60.
        poles_order = np.argsort(np.abs(poles))[::-1]
        poles = poles[poles_order]
        right_vecs = right_vecs[:, poles_order]
        left_vecs = left_vecs[:, poles_order]

        # Take the relevant slices (last sample of each delay line).
        N = kernel.shape[0]
        M = state_transition_matrix.shape[0]
        V = right_vecs[slice(M - 2 * N, M - N)]
        W = left_vecs[slice(M - 2 * N, M - N)]

        # Bake all necessary scaling factors directly into the eigenvectors for RAVES.

        # Recall that V, W of length N are slices of the full state-space vectors of size M.
        # Given the structure of the s.s.m. used above, both V and W now refer to the last sample of each delay line.
        # In the ART format used in RAVES:
        #   - energy is injected at the surface patches, as if it had just been propagated (about to be reflected)
        #   - energy is detected at the surface patches, as if it had just been reflected (about to be propagated)
        # As such, the "injection" eigenvector (W_hat) should refer directly to the last sample of each propagation line,
        # while the "detection" eigenvector (V_hat) should refer to the last sample of each propagation line AND apply scattering.
        # N.B.: if the "detection" eigenvector (V_hat) referred to the first sample of each line, it would differ by a one-sample delay.
        V_hat = kernel @ V
        W_hat = W.copy()

        # Prefer pairs of mostly positive vectors rather than pairs of mostly negative vectors.
        V_signs = np.sign(np.mean(V_hat, axis=0))
        V_hat *= V_signs[np.newaxis]
        W_hat *= V_signs[np.newaxis]

        # Scale by the path etendues to "translate" quantities between power and radiance.
        # The signals circulating in the loop are power, and must be translated to radiance
        # in order to use solid angles as detectors. This means dividing by the path etendue (P = G * L)
        V_hat /= path_etendues[:, np.newaxis]

        # The injectors and detectors, being solid angles, should both sum to 4 pi.
        # Given the way we perform ray-tracing in practice, they sum to 1 instead.
        # Apply the "4 pi" factor here, to save some multiplications at runtime.
        V_hat *= 4 * np.pi
        W_hat *= 4 * np.pi

        # Append results to `MoD-ART.csv`.
        with open(os.path.join(folder_path, 'MoD-ART.csv'), mode='a') as file:
            for p in range(len(poles)):
                file.write(str(band_idx) + ', ' + str(eig_to_T60(poles[p], echogram_sample_rate)) + '\n')
                file.write(', '.join([str(v) for v in V_hat[:, p]]) + '\n')
                file.write(', '.join([str(w) for w in W_hat[:, p]]) + '\n')

        all_pole_T60s[band_idx] = [eig_to_T60(p, echogram_sample_rate) for p in poles]

    if not skip_T60_plots:
        import matplotlib as mtpl
        import matplotlib.pyplot as plt

        print('\tPlotting results.')

        fig, ax = plt.subplots(dpi=200, figsize=(8, 8))

        for band_idx, T60s in all_pole_T60s.items():
            plt.scatter(np.full_like(T60s, band_idx), T60s, marker='+')

        plt.xlabel('Frequency band index')

        plt.ylabel('T60 [s]')
        plt.grid(True, axis='y')
        plt.ylim(0, None)

        plt.savefig(os.path.join(folder_path, 'MoD-ART (rate {:.0f}) T60 values, lin scale.png'.format(echogram_sample_rate)))

        plt.yscale('log')
        plt.ylim(T60_threshold, None)
        ax.yaxis.set_major_locator(mtpl.ticker.LogLocator(subs=np.arange(0.1, 1, 0.1)))
        ax.yaxis.set_major_formatter(mtpl.ticker.ScalarFormatter())
        ax.yaxis.set_minor_locator(mtpl.ticker.LogLocator(subs=np.arange(0.01, 1, 0.01)))
        ax.yaxis.set_minor_formatter(mtpl.ticker.NullFormatter())

        plt.savefig(os.path.join(folder_path, 'MoD-ART (rate {:.0f}) T60 values, log scale.png'.format(echogram_sample_rate)))

        plt.show()
        plt.close()

    print('\n')
