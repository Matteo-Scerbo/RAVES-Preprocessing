import os
import itertools
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches

from scipy.io.wavfile import read
from collections import defaultdict

from raves.src.utils import load_frequencies, air_impedance

if __name__ == '__main__':
    mesh_folder = os.path.join('..', 'BRAS meshes')
    mesh_strategies = ['naive_trng', 'naive_obj', 'split_area', 'split_area_length']

    audio_sample_rate = 44.1e3
    echo_sample_rate = 1e4
    # Choose a low sample rate for plotting and evaluating energy results.
    # In order to avoid having to do any resampling, choose a common divider of
    #  the recording and ART sample rates.
    downsampled_rate = np.gcd(int(audio_sample_rate),
                              int(echo_sample_rate))
    # If the downsampled rate is still too high, choose a lower common divider.
    while downsampled_rate > 1e3:
        success = False
        for prime in [2, 3, 5, 7, 11]:
            if downsampled_rate % prime == 0:
                downsampled_rate /= prime
                success = True
                break
        if not success:
            break
    audio_stride = int(audio_sample_rate / downsampled_rate)
    echo_stride = int(echo_sample_rate / downsampled_rate)
    audio_nyquist = audio_sample_rate / 2

    plotted_band_idx = 5
    plotted_time_range = 2.5
    backwards_integration = False
    normalize_total_energy = True
    show_genelecs = False

    full_room_names = {'CR1_DoorAngle1': 'CR1 coupled rooms (laboratory and reverberation chamber)',
                    'CR1_DoorAngle3': 'CR1 coupled rooms (laboratory and reverberation chamber)',
                    'CR2': 'CR2 small room (seminar room)',
                    'CR3': 'CR3 medium room (chamber music hall)',
                    'CR4': 'CR4 large room (auditorium)',
                    }
    source_positions = {'CR1_DoorAngle1': {'LS1': [1.5, -2.225, 1.239],
                                        'LS2': [-1.77, -2.28, 1.189],
                                        },
                        'CR1_DoorAngle3': {'LS1': [1.5, -2.225, 1.239],
                                        'LS2': [-1.77, -2.28, 1.189],
                                        },
                        'CR2': {'LS1': [0.931, -2.547, 1.23],
                                'LS2': [0.119, 2.88, 1.23],
                                },
                        'CR3': {'LS1': [-2.02, 2.0, 2.76],
                                'LS2': [-3.32, 2.0, 2.76],
                                'LS3': [4.812, 2.358, 1.76], # Only Genelec
                                },
                        'CR4': {'LS1': [-2.8, -4.5, 1.79],
                                'LS2': [0.0, 4.5, 1.79],
                                },
                        }
    listener_positions = {'CR1_DoorAngle1': {'MP3': [-1.205, 0.68, 1.235],
                                            'MP4': [-4.35, 0.695, 1.235],
                                            },
                        'CR1_DoorAngle3': {'MP3': [-1.205, 0.68, 1.235],
                                            'MP4': [-4.35, 0.695, 1.235],
                                            },
                        'CR2': {'MP1': [-0.993, -1.426, 1.23],
                                'MP2': [0.439, -0.147, 1.23],
                                'MP3': [1.361, -0.603, 1.23],
                                'MP4': [-1.11, -0.256, 1.23],
                                'MP5': [-0.998, -1.409, 1.23],
                                },
                        'CR3': {'MP1': [7.84, 0.0, 1.23],
                                'MP2': [2.165, 3.441, 1.23],
                                'MP3': [9.227, 2.366, 1.23],
                                'MP4': [5.86, -2.359, 1.23],
                                'MP5': [12.726, -3.24, 1.23],
                                },
                        'CR4': {'MP1': [8.5, 0.0, 1.09],
                                'MP2': [3.33, -7.95, 0.57],
                                'MP3': [9.33, -6.96, 1.18],
                                'MP4': [5.91, 6.34, 0.83],
                                'MP5': [11.83, 8.43, 1.43],
                                },
                        }
    air_parameters = {'CR1_DoorAngle1': (18.2, 47.6),
                      'CR1_DoorAngle3': (18.2, 47.6),
                      'CR2': (19.5, 41.7),
                      'CR3': (22.4, 40.9),
                      'CR4': (20.9, 37.5),
                      }
    if show_genelecs:
        source_types = ['Dodecahedron',
                        'Genelec8020c_LSorientation-negativeX',
                        'Genelec8020c_LSorientation-negativeY',
                        'Genelec8020c_LSorientation-positiveX',
                        'Genelec8020c_LSorientation-positiveY',
                        'Genelec8020c_LSorientation-01',
                        'Genelec8020c_LSorientation-02',
                        'Genelec8020c_LSorientation-03',
                        'Genelec8020c_LSorientation-04'
                        ]
    else:
        source_types = ['Dodecahedron',]

    # Consider the frequency band centers provided alongside the input data.
    band_centers = load_frequencies(mesh_folder, 'materials_oct_bands.csv')
    num_bands = len(band_centers)
    # Factor for octave-band boundaries.
    band_bound = np.sqrt(2)
    # Ensure that all frequencies support band-pass filtering.
    if np.any(band_centers >= audio_nyquist):
        print('Warning: the audio sample rate is too low for some frequency bands.')
        # Select only acceptable bands.
        band_centers = band_centers[band_centers < audio_nyquist]
        # Update the number of rendered bands.
        num_bands = len(band_centers)

    echos_per_room = defaultdict(dict)
    
    for short_name, full_name in full_room_names.items():
        ref_echo_subfolder = os.path.join(mesh_folder, short_name, 'Reference echograms')

        if '_' in short_name:
            rir_prefix = short_name.replace('_', '_RIR_')
        else:
            rir_prefix = short_name + '_RIR'
        
        for src in source_positions[short_name].keys():
            for lst in listener_positions[short_name].keys():
                for src_typ in source_types:
                    rir_name = '_'.join([rir_prefix, src, lst, src_typ])
                    ref_echo_path = os.path.join(ref_echo_subfolder, rir_name + '.wav')
                    
                    try:
                        fs, ref_data = read(ref_echo_path)
                    except FileNotFoundError:
                        # print(f'Missing reference file:\n\t{rir_name}\nFull path:\n\t{ref_echo_path}\n')
                        continue
                    assert fs == audio_sample_rate, (fs, audio_sample_rate)
                    echogram = ref_data.T
            
                    if backwards_integration:
                        # Reverse integration
                        echogram = np.cumsum(echogram[:, ::-1], axis=-1)[:, ::-1]
                        # Downsampling the EDC is easy, just skip values
                        echogram = echogram[:, ::audio_stride]
                        # dB scale
                        echogram = 10 * np.log10(echogram)
                        # Normalize by total energy
                        if normalize_total_energy:
                            echogram -= echogram[:, 0, None]
                    else:
                        # Short-time average (and downsampling)
                        num_windows = echogram.shape[-1] // audio_stride
                        remainder = echogram.shape[-1] % audio_stride
                        if remainder != 0:
                            echogram = echogram[:, :-remainder]
                        # https://stackoverflow.com/a/71800940
                        echogram = np.array(np.split(echogram, num_windows, axis=-1)
                                            ).mean(axis=-1).T
                        
                        # Normalize by total energy
                        if normalize_total_energy:
                            echogram /= np.sum(echogram, axis=-1)[:, None]
                        # dB scale
                        echogram = 10 * np.log10(echogram)
                    
                    echos_per_room[short_name][(src, lst, f'reference ({src_typ})')] = echogram

                for mesh_strat in mesh_strategies:
                    env_name = short_name + '_' + mesh_strat
                    env_folder = os.path.join(mesh_folder, short_name, env_name)
                    sim_echo_subfolder = os.path.join(env_folder, 'Echograms')
                    sim_echo_path = os.path.join(sim_echo_subfolder, src + lst + '.wav')
                    
                    try:
                        fs, sim_data = read(sim_echo_path)
                    except FileNotFoundError:
                        # print(f'Missing simulation file:\n\t{rir_name}\nFull path:\n\t{sim_echo_path}\n')
                        continue
                    assert fs == echo_sample_rate, (fs, echo_sample_rate)
                    echogram = sim_data.T

                    # TODO: Figure out sensible normalization.
                    echogram /= 4 * np.pi
                    echogram /= echo_sample_rate
                    # The echograms are calibrated such that, when a unit-energy signal is band-passed
                    #  to the relative octave band and then modulated by the echogram's square root, it
                    #  has the correct energy. The band-passing removes energy according to the bandwidth,
                    #  which we need to compensate for.
                    band_widths = (band_centers * band_bound) - (band_centers / band_bound)
                    echogram *= band_widths[:, None]

                    if backwards_integration:
                        # Reverse integration
                        echogram = np.cumsum(echogram[:, ::-1], axis=-1)[:, ::-1]
                        # Downsampling the EDC is easy, just skip values
                        echogram = echogram[:, ::echo_stride]
                        # dB scale
                        echogram = 10 * np.log10(echogram)
                        # Normalize by total energy
                        if normalize_total_energy:
                            echogram -= echogram[:, 0, None]
                    else:
                        # Short-time average (and downsampling)
                        num_windows = echogram.shape[-1] // echo_stride
                        remainder = echogram.shape[-1] % echo_stride
                        if remainder != 0:
                            echogram = echogram[:, :-remainder]
                        # https://stackoverflow.com/a/71800940
                        echogram = np.array(np.split(echogram, num_windows, axis=-1)
                                            ).mean(axis=-1).T
                        
                        # Normalize by total energy
                        if normalize_total_energy:
                            echogram /= np.sum(echogram, axis=-1)[:, None]
                        # dB scale
                        echogram = 10 * np.log10(echogram)
                    
                    echos_per_room[short_name][(src, lst, mesh_strat)] = echogram

                # Trim all echograms in the room to the length of the shortest one.
                min_length_in_room = min([e.shape[-1] for e in echos_per_room[short_name].values()])
                echos_per_room[short_name] = {k: e[:, :min_length_in_room]
                                              for k, e in echos_per_room[short_name].items()}

    for short_name, echos_dict in echos_per_room.items():
        # Count the number of src/lst configuration, i.e., reference echograms.
        sl_configs = [(s, l) for s, l, k in echos_dict
                      if k == 'reference (Dodecahedron)']
        num_sl_configs = len(sl_configs)
        if num_sl_configs == 0:
            print(f'No reference echograms were found for room {short_name}.')
            continue
        # Count the number of echograms that were loaded, other than the reference ones.
        num_sim_echos = len([k for s, l, k in echos_dict
                             if 'reference' not in k])
        if num_sim_echos == 0:
            print(f'No ART echograms were found for room {short_name}.')
            continue

        fig, ax = plt.subplots(dpi=200, figsize=(9, 6))

        for (src, lst, key), echogram in echos_dict.items():
            time_axis = np.arange(echogram.shape[-1]) / downsampled_rate

            if 'reference' not in key:
                plt.plot(time_axis, echogram[plotted_band_idx],
                         label=key)
            elif 'Dodecahedron' in key:
                plt.plot(time_axis, echogram[plotted_band_idx],
                         label='reference', ls=':')
            else:  # Genelecs: match color of Dodecahedron, don't add new labels
                plt.plot(time_axis, echogram[plotted_band_idx],
                         color=plt.gca().lines[-1].get_color(),
                         ls=':')

        plt.xlim(0, plotted_time_range)
        if backwards_integration and normalize_total_energy:
            plt.ylim(-60, 0)
        
        # https://stackoverflow.com/a/10101532
        def flip(items, ncol):
            return itertools.chain(*[items[i::ncol]
                                     for i in range(ncol)])
        handles, labels = ax.get_legend_handles_labels()
        plt.legend(flip(handles, 3),
                   flip(labels, 3),
                   ncol=3)

        plt.title(f'Room {short_name}; {band_centers[plotted_band_idx]}Hz octave band.')
        plt.tight_layout()
        plt.show()

        # Plot the energy differences.

        # Keep track of error data for the next figure (violin plots).
        violin_data = defaultdict(dict)

        # Diverging colormap to differentiate positive and negative values.
        contour_levels = np.linspace(-10, 10, 21)
        cmap = plt.get_cmap('RdBu', len(contour_levels) + 1)
        # Set "bad" values (i.e., reference is below noise floor) to black.
        cmap.set_bad('black', 1.)
        norm = mpl.colors.BoundaryNorm(contour_levels, ncolors=cmap.N, extend='both')

        def tick_label_func(val, pos=None):
            if val >= num_bands:
                return 'error'
            elif band_centers[int(val)] < 1e3:
                return f'{int(band_centers[int(val)])}'
            else:
                return f'{int(band_centers[int(val)] / 1e3)}k'

        fig, axes = plt.subplots(num_sl_configs, len(mesh_strategies),
                                 figsize=(4*len(mesh_strategies), 3*num_sl_configs),
                                 squeeze=False, constrained_layout=True)

        cs = None
        for i, (src, lst) in enumerate(sl_configs):
            reference = echos_dict[(src, lst, 'reference (Dodecahedron)')]
            
            X, Y = np.meshgrid(np.arange(reference.shape[-1]) / downsampled_rate,
                               np.arange(len(band_centers)))

            for j, mesh_strat in enumerate(mesh_strategies):
                if (src, lst, mesh_strat) in echos_dict:
                    error = echos_dict[(src, lst, mesh_strat)] - reference

                    # Keep track of error data for the next figure (violin plots).
                    violin_data[(src, lst)][mesh_strat] = error
                    
                    cs = axes[i, j].pcolormesh(X, Y, error, norm=norm, cmap=cmap)
                    
                    axes[i, j].yaxis.set_major_formatter(ticker.FuncFormatter(tick_label_func))
                else:
                    axes[i, j].text(0.5, 0.5, 'MISSING DATA',
                                    ha='center', va='center')
                
                axes[i, j].set_title(f'{src} {lst} {mesh_strat}')
                if i == num_sl_configs-1:
                    axes[i, j].set_xlabel('Time [s]')
                else:
                    axes[i, j].set_xlabel('')
                if j == 0:
                    axes[i, j].set_ylabel('Octave band center [Hz]')
                else:
                    axes[i, j].set_ylabel('')

        cbar = fig.colorbar(cs, ax=axes, format='{x:.0f}dB')
        if backwards_integration:
            cbar.ax.set_ylabel('Backward-integrated energy diff (ART - truth)',
                               rotation=270, labelpad=15)
        else:
            cbar.ax.set_ylabel('Short-time-average energy diff (ART - truth)',
                               rotation=270, labelpad=15)

        plt.suptitle(f'Room {short_name}')
        plt.show()

        # Plot the energy differences statistics.

        num_sources = len(source_positions[short_name])
        num_listeners = len(listener_positions[short_name])
        fig, axes = plt.subplots(num_listeners, num_sources,
                                 figsize=(4*num_sources, 3*num_listeners),
                                 constrained_layout=True)

        group_centers = np.arange(num_bands)
        # https://stackoverflow.com/a/11603806
        margin = 0.25
        width = (1 - 2*margin) / len(mesh_strategies)

        # https://stackoverflow.com/a/58324984
        def add_violin_label(violin, label):
            color = violin["bodies"][0].get_facecolor().flatten()
            violin_labels.append((mpatches.Patch(color=color), label))

        for i, lst in enumerate(listener_positions[short_name]):
            for j, src in enumerate(source_positions[short_name]):
                # Reset legend labels.
                violin_labels = list()

                # Reference horizontal line at 0.
                line = axes[i, j].hlines(0, -1, num_bands+1,
                                         color='black', ls='--',
                                         linewidth=1)
                
                for k, (mesh_strat, error_data) in enumerate(violin_data[(src, lst)].items()):
                    # The NaN entries must be filtered out for each octave band.
                    error_data = [e[np.isfinite(e)]
                                  for e in error_data]

                    positions = group_centers - 0.5 + margin + (k+0.5)*width
                    violin = axes[i, j].violinplot(error_data, positions=positions,
                                                   widths=width,
                                                   showextrema=False,
                                                   showmeans=False,
                                                   showmedians=True)

                    add_violin_label(violin, mesh_strat)
                
                axes[i, j].legend(*zip(*violin_labels))

                axes[i, j].set_title(f'{src} {lst}')

                axes[i, j].set_ylim(-15, 15)
                axes[i, j].set_xlim(-0.5, num_bands-0.5)
                axes[i, j].xaxis.set_major_formatter(ticker.FuncFormatter(tick_label_func))

                if i == num_listeners-1:
                    axes[i, j].set_xlabel('Octave band center [Hz]')
                else:
                    axes[i, j].set_xlabel('')
                if j == 0:
                    axes[i, j].set_ylabel('Energy diff (ART - truth) in dB')
                else:
                    axes[i, j].set_ylabel('')

        if backwards_integration:
            plt.suptitle(f'{short_name} - backward-integrated energy diff')
        else:
            plt.suptitle(f'{short_name} - short-time-average energy diff')
        plt.show()

        # break
