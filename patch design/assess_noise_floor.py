import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read, write
from scipy.stats import linregress
from scipy.signal import butter, sosfiltfilt, fftconvolve
from scipy.signal.windows import get_window

from raves.src.utils import load_frequencies

if __name__ == '__main__':
    mesh_folder = os.path.join('..', 'BRAS meshes')
    response_folder = os.path.join('..', '..', '..', 'BRAS', '1 Scene descriptions')

    audio_sample_rate = 44.1e3
    audio_nyquist = audio_sample_rate / 2

    # Broadest: 'cosine', 'lanczos', 'tukey'
    longest_window = get_window('tukey', int(1.2 * audio_sample_rate))
    longest_window /= np.sum(longest_window)
    long_window = get_window('tukey', int(0.8 * audio_sample_rate))
    long_window /= np.sum(long_window)
    short_window = get_window('tukey', int(0.4 * audio_sample_rate))
    short_window /= np.sum(short_window)

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

    for short_name, full_name in full_room_names.items():
        print(short_name)

        rir_folder = os.path.join(response_folder, full_name, 'RIRs', 'wav')
        if '_' in short_name:
            rir_prefix = short_name.replace('_', '_RIR_')
        else:
            rir_prefix = short_name + '_RIR'
        
        env_folder = os.path.join(mesh_folder, short_name)
        echo_subfolder = os.path.join(env_folder, 'Reference echograms')
        os.makedirs(echo_subfolder, exist_ok=True)
        
        for src in source_positions[short_name].keys():
            for lst in listener_positions[short_name].keys():
                for src_typ in source_types:
                    rir_name = '_'.join([rir_prefix, src, lst, src_typ])
                    rir_path = os.path.join(rir_folder, rir_name + '.wav')
                    echo_path = os.path.join(echo_subfolder, rir_name + '.wav')

                    try:
                        fs, rir_data = read(rir_path)
                    except FileNotFoundError:
                        # print(f'Missing RIR file:\n\t{rir_name}\nFull path:\n\t{rir_path}\n')
                        continue
                    assert fs == audio_sample_rate, (fs, audio_sample_rate)
        
                    print('\t', src, lst, src_typ)
                
                    # Prepare an array for the band-pass-filtered response energy.
                    echogram = np.zeros((num_bands, len(rir_data)))

                    for b in range(num_bands):
                        if band_centers[b] * band_bound >= audio_nyquist:
                            upper_lim = audio_nyquist * 0.999
                        else:
                            upper_lim = band_centers[b] * band_bound
                        lower_lim = band_centers[b] / band_bound
                        
                        # Prepare the suitable band-pass filter...
                        sos = butter(3, (lower_lim, upper_lim),
                                     btype='bandpass', output='sos',
                                     fs=audio_sample_rate)
                        # ...and apply it to the room impulse response.
                        # N.B.: using ´sosfiltfilt´ means the filter is applied both forward and backward,
                        #  therefore negating any introduced delay. This way, the onset time of the noise floor
                        #  is evaluated correctly.
                        bandpassed_rir = sosfiltfilt(sos, rir_data)
                            
                        band_energy = bandpassed_rir**2
                        # Normalize to energy-per-second, matching our echogram convention.
                        band_energy *= audio_sample_rate

                        if band_centers[b] > 2e3:
                            smooth_energy = fftconvolve(band_energy, short_window, mode='same')
                            # Trim the windowing artefact at the end of the response,
                            #  but keep the one at the start to preserve time indexing.
                            smooth_energy = smooth_energy[:-int(len(short_window) / 2)]
                        elif band_centers[b] > 5e2:
                            smooth_energy = fftconvolve(band_energy, long_window, mode='same')
                            # Trim the windowing artefact at the end of the response,
                            #  but keep the one at the start to preserve time indexing.
                            smooth_energy = smooth_energy[:-int(len(long_window) / 2)]
                        else:
                            smooth_energy = fftconvolve(band_energy, longest_window, mode='same')
                            # Trim the windowing artefact at the end of the response,
                            #  but keep the one at the start to preserve time indexing.
                            smooth_energy = smooth_energy[:-int(len(longest_window) / 2)]

                        # dB scale. Add an offset to avoid true zeros.
                        smooth_energy_dB = 10 * np.log10(smooth_energy + 1e-20)

                        # Extend the array, repeating the end value, to facilitate the linear regression.
                        smooth_energy_dB = np.pad(smooth_energy_dB,
                                                  (0, int(3 * audio_sample_rate)),
                                                  mode='constant',
                                                  constant_values=np.min(smooth_energy_dB))

                        time_axis = np.arange(len(smooth_energy_dB)) / audio_sample_rate
                        
                        start_of_noise = len(smooth_energy_dB)
                        noise_floor = -np.inf
                        best_slope = -np.inf

                        # A slope shallower than 2dB per second is considered flat.
                        # Except for the reverberation chamber with closed door.
                        if 'CR1' in short_name:
                            if band_centers[b] <= 4e3:
                                shallow_threshold = -0.5
                            else:
                                shallow_threshold = -1
                        else:
                            shallow_threshold = -2
                        
                        for cursor in range(np.argmax(smooth_energy_dB),
                                            len(smooth_energy_dB),
                                            int(5e-3 * audio_sample_rate)):
                            regression = linregress(time_axis[cursor:],
                                                    smooth_energy_dB[cursor:])
                            
                            if regression.slope > best_slope:
                                noise_level = regression.intercept + regression.slope * time_axis[cursor]
                                # noise_level = smooth_energy_dB[cursor]

                                if noise_level > np.min(smooth_energy_dB) + 30:
                                    # Only accept noise floor values no more than 30dB above the global minimum.
                                    continue

                                if np.any(smooth_energy_dB <= noise_level):
                                    best_slope = regression.slope
                                    noise_floor = noise_level
                                    start_of_noise = np.min(np.flatnonzero(smooth_energy_dB <= noise_level))
                            
                            if regression.slope > shallow_threshold:
                                break

                        if (short_name not in ['CR3', 'CR4']) and not ('CR1' in short_name and src == 'LS2'):
                            # Some of the recordings drop to the noise floor abruptly, before the decay ends.
                            # These frequency-dependent artefacts may have been caused by a truncation
                            #  which was carried out before the sine-sweep deconvolution.
                            # The unnatural energy roll-off into the noise floor prevents a fair comparison
                            #  with simulation results, so we trim responses before it happens.

                            # Consider only increases in slope which occur more than some delay after the onset.
                            if band_centers[b] < 4e2:
                                range_start = np.inf
                            elif band_centers[b] < 4e3:
                                range_start = int(1.0 * audio_sample_rate)
                            else:
                                range_start = int(0.5 * audio_sample_rate)
                            
                            if range_start < start_of_noise:
                                decay_slope = np.gradient(smooth_energy_dB)

                                slope_estimate = np.median(decay_slope[range_start:start_of_noise])

                                steeper_than_estimate = (decay_slope <= slope_estimate - (2 / audio_sample_rate))
                                steeper_than_estimate &= (time_axis > range_start / audio_sample_rate)
                                # Consider only increases in slope which occur within 30dB of the noise floor.
                                steeper_than_estimate &= (smooth_energy_dB <= noise_floor + 30)
                                # Consider only increases in slope which last for more than 300ms.
                                for i in range(len(steeper_than_estimate) - int(0.3 * audio_sample_rate)):
                                    steeper_than_estimate[i] = np.all(steeper_than_estimate[i:i+int(0.3 * audio_sample_rate)])

                                if np.any(steeper_than_estimate):
                                    start_of_noise = np.min(np.flatnonzero(steeper_than_estimate))

                        # plt.plot(time_axis[range_start:int(3.8*audio_sample_rate)],
                        #          decay_slope[range_start:int(3.8*audio_sample_rate)],
                        #          label=f'{band_centers[b]}Hz')
                        
                        # plt.scatter(start_of_noise / audio_sample_rate,
                        #             decay_slope[start_of_noise-1],
                        #             marker='v', label=f'{(start_of_noise / audio_sample_rate):.2f}s',
                        #             color=plt.gca().lines[-1].get_color())
                    
                        if start_of_noise == len(smooth_energy_dB):
                            # No noise floor was detected. The entire duration of the echogram is preserved.
                            echogram[b] = band_energy
                        else:
                            # A noise floor was detected. The echogram is truncated before the floor is reached.
                            echogram[b, :start_of_noise-1] = band_energy[:start_of_noise-1]

                        plt.plot(time_axis,
                                 smooth_energy_dB,
                                 label=f'{band_centers[b]}Hz')
                        
                        plt.scatter(start_of_noise / audio_sample_rate,
                                    smooth_energy_dB[start_of_noise-1],
                                    marker='v', label=f'{(start_of_noise / audio_sample_rate):.2f}s',
                                    color=plt.gca().lines[-1].get_color())
                        
                    # Truncate to the longest nonzero value.
                    max_valid_sample = np.max(np.nonzero(echogram)[-1])
                    echogram = echogram[:, :max_valid_sample]

                    plt.xlim(0, max_valid_sample / audio_sample_rate)
                    plt.ylim(-90, None)
                    plt.title(short_name)
                    plt.tight_layout()
                    plt.legend(ncol=2)
                    plt.show()

                    # Save the "masked" echogram.
                    write(echo_path, int(audio_sample_rate), echogram.T)
