import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read, write
from scipy.signal import butter, sosfiltfilt
from scipy.signal.windows import get_window

from raves.src.utils import load_frequencies

if __name__ == '__main__':
    mesh_folder = os.path.join('..', 'BRAS meshes')
    response_folder = os.path.join('..', '..', '..', 'BRAS', '1 Scene descriptions')

    audio_sample_rate = 44.1e3
    audio_nyquist = audio_sample_rate / 2
    window_length = int(0.1 * audio_sample_rate)
    # The responses are trimmed when they reach these many dB above the detected noise floor.
    cutoff_before_floor = 12

    window = get_window('hann', window_length, fftbins=False)
    window /= np.sum(window)

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

                        # Short-time average.
                        smooth_energy = np.apply_along_axis(lambda m: np.convolve(m, window),
                                                            arr=band_energy, axis=-1)
                        
                        # Start by assuming there is no detectable noise floor.
                        noise_floor = None
                        
                        # Consider possible starting times for the examined range, in 50ms increments.
                        for start_of_range in range(0, len(smooth_energy), int(0.05*audio_sample_rate)):
                            examined_range_duration = len(smooth_energy) - start_of_range
                            if examined_range_duration / audio_sample_rate < 0.05:
                                # A range of less than 50ms is considered insufficient to detect a noise floor.
                                break

                            # Consider the energy values from 'start_of_range' onwards.
                            late_values = smooth_energy[start_of_range:]
                            # Raise true-zero values to the lowest nonzero value, to avoid errors.
                            late_values = late_values.clip(np.min(late_values[late_values != 0]))
                            # dB scale.
                            late_values = 10 * np.log10(late_values)

                            # In some responses, the energy appears to rise towards the end of the response.
                            # This is assumed to be an artefact and is removed.
                            # If energy is higher in the later half of the examined range than in the earlier half,
                            #  the average energy in the range is taken to be the noise floor.
                            range_midpoint = start_of_range + int(examined_range_duration / 2)
                            if np.sum(smooth_energy[range_midpoint:]) > np.sum(smooth_energy[start_of_range:range_midpoint]):
                                noise_floor = int(np.mean(late_values))
                                break

                            # Make a histogram of the energy values, with bins which are 1 dB wide.
                            hist, bin_edges = np.histogram(late_values, bins=150,
                                                           range=(-150.5, -0.5))
                            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

                            # Note: hist[hist_peak] is the most common energy value in the examined range.
                            hist_peak = np.argmax(hist)
                            
                            # Count the duration of time during which the energy is within +-5dB of the most common value.
                            within_10dB_of_peak = range(hist_peak-6, hist_peak+5)
                            noise_duration = np.sum(hist[within_10dB_of_peak])
                            # Compare that with the total duration of time of the examined range.
                            noise_percentage = 100 * noise_duration / examined_range_duration

                            if noise_percentage > 95:
                                # If this condition is satisfied, it means that the response energy
                                #  stays within a 10dB range for more than 95% of the examined range.
                                # In other words, the energy is "flat" for a prolonged period.
                                # The middle of the "flat" energy range is taken to be the noise floor.
                                noise_floor = int(bin_centers[hist_peak])
                                break

                        if noise_floor is None:
                            # No noise floor was detected. The entire duration of the echogram is preserved.
                            echogram[b] = band_energy
                        else:
                            # A noise floor was detected. The echogram is truncated before the floor is reached.

                            # Find the indices of all samples below the noise floor.
                            low_energy_indices = np.flatnonzero(smooth_energy <= 10**((noise_floor + cutoff_before_floor)/10))
                            # Drop the indices earlier than the peak energy (response onset).
                            low_energy_indices = low_energy_indices[low_energy_indices > np.argmax(smooth_energy)]
                            # Find the earliest sample below the noise floor.
                            max_valid_sample = np.min(low_energy_indices)

                            echogram[b, :max_valid_sample] = band_energy[:max_valid_sample]
                            smooth_energy[max_valid_sample:] = 0

                        # plt.plot(np.arange(len(smooth_energy)) / audio_sample_rate,
                        #          smooth_energy,
                        #          label=(f'{band_centers[b]}Hz -> {noise_floor}dB'
                        #                 if noise_floor is not None else
                        #                 f'{band_centers[b]}Hz -> None'))
                    
                    # Truncate to the longest nonzero value.
                    max_valid_sample = np.max(np.nonzero(echogram)[-1])
                    echogram = echogram[:, :max_valid_sample]

                    # plt.yscale('log')
                    # plt.xlim(0, max_valid_sample / audio_sample_rate)
                    # plt.ylim(10**(-9), None)
                    # plt.title(short_name)
                    # plt.tight_layout()
                    # plt.legend()
                    # plt.show()

                    # Save the "masked" echogram.
                    write(echo_path, int(audio_sample_rate), echogram.T)
