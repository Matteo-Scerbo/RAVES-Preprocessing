import os
import itertools
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from scipy.io.wavfile import read
from scipy.signal import butter, sosfilt
from collections import defaultdict

from raves.src.utils import load_frequencies, air_impedance

if __name__ == '__main__':
    mesh_folder = os.path.join('..', 'BRAS meshes')
    response_folder = os.path.join('..', '..', '..', 'BRAS', '1 Scene descriptions')
    mesh_strategies = ['naive_trng', 'naive_obj']

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

    plotted_band_idx = 2
    plotted_time_range = 2.5
    backwards_integration = False
    normalize_total_energy = False
    simulate_noise_floor = True

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
    noise_floors = {'CR1_DoorAngle1': [-71, -77, -78, -79, -81, -79, -73, -70],
                    'CR1_DoorAngle3': [-77, -76, -77, -78, -80, -79, -74, -69],
                    'CR2': [-81, -76, -75, -72, -72, -73, -69, -64],
                    'CR3': [-93, -97, -98, -96, -91, -87, -81, -76],
                    'CR4': [-86, -87, -88, -88, -87, -85, -80, -75]}

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

    # https://stackoverflow.com/a/5029958
    rirs_per_room = defaultdict(lambda: defaultdict(list))
    echos_per_room = defaultdict(dict)
    
    for short_name, full_name in full_room_names.items():
        rir_folder = os.path.join(response_folder, full_name, 'RIRs', 'wav')
        if '_' in short_name:
            rir_prefix = short_name.replace('_', '_RIR_')
        else:
            rir_prefix = short_name + '_RIR'
        
        for src in source_positions[short_name].keys():
            for lst in listener_positions[short_name].keys():
                for src_typ in source_types:
                    rir_name = '_'.join([rir_prefix, src, lst, src_typ])
                    rir_path = os.path.join(rir_folder, rir_name) + '.wav'
                    
                    try:
                        fs, rir_data = read(rir_path)
                    except FileNotFoundError:
                        # print(f'Missing RIR file:\n\t{rir_name}\nFull path:\n\t{rir_path}\n')
                        continue
                    assert fs == audio_sample_rate, (fs, audio_sample_rate)
    
                    rirs_per_room[short_name][(src, lst)].append(rir_data)
    
                for mesh_strat in mesh_strategies:
                    env_name = short_name + '_' + mesh_strat
                    env_folder = os.path.join(mesh_folder, short_name, env_name)
                    echo_subfolder = os.path.join(env_folder, 'Echograms')
                    echo_path = os.path.join(echo_subfolder, src + lst + '.wav')
                    
                    try:
                        fs, echo_data = read(echo_path)
                    except FileNotFoundError:
                        print(f'Missing RIR file:\n\t{echo_path}\n')
                        continue
                    if fs != echo_sample_rate:
                        continue
                    assert fs == echo_sample_rate, (fs, echo_sample_rate)
    
                    echos_per_room[short_name][(src, lst, mesh_strat)] = echo_data.T
    
        # num_rirs = sum([len(d) for k, d in rirs_per_room[short_name].items()])
        # print(f'Found {num_rirs} recordings in total in room {short_name}.')

        # num_echos = sum([len(d) for k, d in echos_per_room[short_name].items()])
        # print(f'Found {num_echos} ART echograms in total in room {short_name}.')
    
    specgrams_per_room = defaultdict(dict)

    for short_name, rirs_dict in rirs_per_room.items():
        num_echos = sum([len(d) for k, d in echos_per_room[short_name].items()])
        # print(f'Found {num_echos} ART echograms in total in room {short_name}.')
        if num_echos == 0:
            continue

        # The noise floors were assessed from the RIRs (squared pressure),
        #  we need to convert them to sound intensity like the RIRs (see below).
        for b in range(num_bands):
            impedance_factor = air_impedance(temperature=air_parameters[short_name][0],
                                             humidity=air_parameters[short_name][1])
            noise_floors[short_name][b] += 10 * np.log10(impedance_factor)

        fig, ax = plt.subplots(dpi=200, figsize=(9, 6))

        for (src, lst), rirs in rirs_dict.items():
            # print(f'Found {len(rirs)} recordings in room {short_name} '
            #       f'with source {src} and listener {lst}.')

            # List the spectrograms of all loudspeaker types,
            #  in order to average them together.
            specgram_list = list()

            first = True
            for rir in rirs:
                # Prepare an array for the band-pass filtered response.
                banded_rir = np.zeros((num_bands, len(rir)))

                for b in range(num_bands):
                    if band_centers[b] * band_bound >= audio_nyquist:
                        upper_lim = audio_nyquist * 0.999
                    else:
                        upper_lim = band_centers[b] * band_bound
                    lower_lim = band_centers[b] / band_bound
                    
                    # Prepare the suitable band-pass filter...
                    sos = butter(6, (lower_lim, upper_lim),
                                 btype='bandpass', output='sos',
                                 fs=audio_sample_rate)
                    # ...and apply it to the room impulse response.
                    banded_rir[b] = sosfilt(sos, rir)
                
                banded_energy = banded_rir**2
                # Normalize to energy-per-second, matching our echogram convention.
                banded_energy *= audio_sample_rate
                # As reported in the BRAS documentation, the RIRs' units are [Pa], pressure.
                # Our echograms are defined as [W/m2], sound intensity.
                # Sound intensity equals squared pressure divided by the characteristic impedance of air.
                # TODO: In theory, this should be a division, not a multiplication.
                #       But they match this way... why?
                banded_energy *= air_impedance(temperature=air_parameters[short_name][0],
                                               humidity=air_parameters[short_name][1])

                if backwards_integration:
                    # Reverse integration
                    banded_energy = np.cumsum(banded_energy[:, ::-1], axis=-1)[:, ::-1]
                    # Decimate (speeds up rendering)
                    banded_energy = banded_energy[:, ::audio_stride]
                    # dB scale
                    banded_energy = 10 * np.log10(banded_energy)
                    # Normalize by total energy
                    if normalize_total_energy:
                        banded_energy -= banded_energy[:, 0, None]
                else:
                    num_windows = banded_energy.shape[-1] // audio_stride
                    # https://stackoverflow.com/a/71800940
                    banded_energy = np.array(np.array_split(banded_energy,
                                                            num_windows,
                                                            axis=-1)
                                             ).mean(axis=-1).T
                    # Normalize by total energy
                    if normalize_total_energy:
                        banded_energy /= np.sum(banded_energy, axis=-1)[:, None]
                    # dB scale
                    banded_energy = 10 * np.log10(banded_energy)
                
                specgram_list.append(banded_energy)
                
                time_axis = np.arange(banded_energy.shape[-1]) / downsampled_rate

                if first:
                    plt.plot(time_axis,
                             banded_energy[plotted_band_idx],
                             label = src + ' ' + lst,
                             ls=':')
                    first = False
                else:
                    plt.plot(time_axis,
                             banded_energy[plotted_band_idx],
                             color = plt.gca().lines[-1].get_color(),
                             ls=':')

            # Average over all loudspeaker types.
            min_reference_len = min([spec.shape[-1]
                                     for spec in specgram_list])
            min_reference_len = min(min_reference_len, int(plotted_time_range * downsampled_rate))
            mean_specgram = np.mean([spec[:, :min_reference_len]
                                     for spec in specgram_list],
                                     axis=0)
            specgrams_per_room[short_name][(src, lst, 'reference')] = mean_specgram

            for mesh_strat in mesh_strategies:
                if short_name in echos_per_room:
                    if (src, lst, mesh_strat) in echos_per_room[short_name]:
                        echogram = echos_per_room[short_name][(src, lst, mesh_strat)]

                        # Extend the duration of the echogram,
                        #  providing more room for the backwards integration.
                        echogram = np.pad(echogram,
                                          ((0, 0),  # No padding over first dimension (freqs)
                                           (0, int(2*echo_sample_rate))))
                        # Add a noise floor matching the recordings,
                        #  especially important for the backwards integration comparison.
                        if simulate_noise_floor:
                            for b in range(num_bands):
                                floor_db = noise_floors[short_name][b]
                                echogram[b] += 10 ** (floor_db / 10)
                
                        if backwards_integration:
                            # Reverse integration
                            echogram = np.cumsum(echogram[:, ::-1], axis=-1)[:, ::-1]
                            # Decimate (speeds up rendering)
                            echogram = echogram[:, ::echo_stride]
                            # dB scale
                            echogram = 10 * np.log10(echogram)
                            # Normalize by total energy
                            if normalize_total_energy:
                                echogram -= echogram[:, 0, None]
                        else:
                            num_windows = echogram.shape[-1] // echo_stride
                            # https://stackoverflow.com/a/71800940
                            echogram = np.array(np.array_split(echogram,
                                                               num_windows,
                                                               axis=-1)
                                                ).mean(axis=-1).T
                            # Normalize by total energy
                            if normalize_total_energy:
                                echogram /= np.sum(echogram, axis=-1)[:, None]
                            # dB scale
                            echogram = 10 * np.log10(echogram)
                        
                        specgrams_per_room[short_name][(src, lst, mesh_strat)] = echogram[:, :min_reference_len]

                        time_axis = np.arange(echogram.shape[-1]) / downsampled_rate

                        plt.plot(time_axis,
                                 echogram[plotted_band_idx],
                                 label = f'{src} {lst} {mesh_strat}')
    
        plt.xlim(0, plotted_time_range)
        if backwards_integration:
            plt.ylim(-60, 0)
        else:
            floor_db = noise_floors[short_name][plotted_band_idx]

            plt.ylim(floor_db-10, None)

            plt.hlines(floor_db, xmin=0, xmax=5,
                       color='black', ls='--', linewidth=1,
                       label='Noise floor')
        
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

        # Diverging colormap to differentiate positive and negative values.
        contour_levels = np.linspace(-10, 10, 21)
        cmap = plt.get_cmap('RdBu', len(contour_levels) + 1)
        norm = mpl.colors.BoundaryNorm(contour_levels, ncolors=cmap.N, extend='both')

        def tick_label_func(val, pos=None):
            if val >= num_bands:
                return 'error'
            elif band_centers[int(val)] < 1e3:
                return f'{int(band_centers[int(val)])}'
            else:
                return f'{int(band_centers[int(val)] / 1e3)}k'

        fig, axes = plt.subplots(len(rirs_dict), len(mesh_strategies),
                                 figsize=(4*len(mesh_strategies), 3*len(rirs_dict)),
                                 squeeze=False, constrained_layout=True)

        cs = None
        for i, (src, lst) in enumerate(rirs_dict.keys()):
            reference = specgrams_per_room[short_name][(src, lst, 'reference')]
                                                        
            X, Y = np.meshgrid(np.arange(reference.shape[-1]) / downsampled_rate,
                               np.arange(len(band_centers)))

            for j, mesh_strat in enumerate(mesh_strategies):
                if (src, lst, mesh_strat) in specgrams_per_room[short_name]:
                    error = specgrams_per_room[short_name][(src, lst, mesh_strat)] - reference
                    
                    cs = axes[i, j].pcolormesh(X, Y, error, norm=norm, cmap=cmap)
                    
                    axes[i, j].yaxis.set_major_formatter(ticker.FuncFormatter(tick_label_func))
                else:
                    axes[i, j].text(0.5, 0.5, 'MISSING DATA',
                                    ha='center', va='center')
                
                axes[i, j].set_title(f'{src} {lst} {mesh_strat}')
                if i == len(rirs_dict)-1:
                    axes[i, j].set_xlabel('Time [s]')
                else:
                    axes[i, j].set_xlabel('')
                if j == 0:
                    axes[i, j].set_ylabel('Octave band center [Hz]')
                else:
                    axes[i, j].set_ylabel('')

        cbar = fig.colorbar(cs, ax=axes, format='{x:.0f}dB')
        if backwards_integration:
            cbar.ax.set_ylabel('Difference (simulation - reference) of backward-integrated energy',
                               rotation=270, labelpad=15)
        else:
            cbar.ax.set_ylabel('Difference (simulation - reference) of short-time-average energy',
                               rotation=270, labelpad=15)

        plt.suptitle(f'Room {short_name}')
        plt.show()
