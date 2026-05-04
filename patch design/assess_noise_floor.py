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
    long_window = get_window('tukey', int(0.5 * audio_sample_rate))
    long_window /= np.sum(long_window)
    short_window = get_window('tukey', int(0.2 * audio_sample_rate))
    short_window /= np.sum(short_window)

    full_room_names = {'CR1_DoorAngle1': 'CR1 coupled rooms (laboratory and reverberation chamber)',
                       'CR1_DoorAngle3': 'CR1 coupled rooms (laboratory and reverberation chamber)',
                       'CR2': 'CR2 small room (seminar room)',
                       'CR3': 'CR3 medium room (chamber music hall)',
                       'CR4': 'CR4 large room (auditorium)',
                       }
    source_positions = {'CR1_DoorAngle1': {# 'LS1': [1.5, -2.225, 1.239],
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

                        if band_centers[b] > 2e3:
                            smooth_energy = fftconvolve(band_energy, short_window, mode='same')
                            # Trim the windowing artefact at the end of the response,
                            #  but keep the one at the start to preserve time indexing.
                            smooth_energy = smooth_energy[:-int(len(short_window) / 2)]
                        else:
                            smooth_energy = fftconvolve(band_energy, long_window, mode='same')
                            # Trim the windowing artefact at the end of the response,
                            #  but keep the one at the start to preserve time indexing.
                            smooth_energy = smooth_energy[:-int(len(long_window) / 2)]
                        
                        # dB scale. Add an offset to avoid true zeros.
                        smooth_energy_dB = 10 * np.log10(smooth_energy + 1e-20)

                        time_axis = np.arange(len(smooth_energy))

                        noise_floor = None
                        start_of_noise = None
                        best_slope = -np.inf
                        
                        for cursor in range(np.argmax(smooth_energy),
                                            len(smooth_energy),
                                            int(5e-3 * audio_sample_rate)):
                            regression = linregress(time_axis[cursor:],
                                                    smooth_energy_dB[cursor:])
                            
                            if regression.slope > best_slope:
                                best_slope = regression.slope
                                noise_floor = regression.intercept + regression.slope * cursor
                                # noise_floor = smooth_energy_dB[cursor]

                                # start_of_noise = cursor
                                start_of_noise = np.min(np.flatnonzero(smooth_energy_dB <= noise_floor))
                            
                            if regression.slope > -3e-5:
                                break

                            """
                            linear_approx = regression.intercept + regression.slope * time_axis
                            error = smooth_energy_dB - linear_approx

                            zero_crossings = np.flatnonzero(np.diff(np.sign(error)) != 0)
                            # upward_zero_crossings = np.flatnonzero(np.diff(np.sign(error)) > 0)
                            # downward_zero_crossings = np.flatnonzero(np.diff(np.sign(error)) < 0)

                            # plt.plot(time_axis / audio_sample_rate,
                            #          linear_approx)
                            # plt.plot(time_axis[upward_zero_crossings] / audio_sample_rate,
                            #          linear_approx[upward_zero_crossings],
                            #          marker='^', ls=':',
                            #          color=plt.gca().lines[-1].get_color())
                            # plt.plot(time_axis[downward_zero_crossings] / audio_sample_rate,
                            #          linear_approx[downward_zero_crossings],
                            #          marker='v', ls=':',
                            #          color=plt.gca().lines[-1].get_color())
                            
                            if len(zero_crossings) == 0:
                                break

                            potential_onset = zero_crossings[-1]

                            if potential_onset - start_of_decay < 0.1 * audio_sample_rate:
                                # The response would be too short, something went wrong.
                                break

                            noise_floor_onset = potential_onset
                            noise_floor = linear_approx[noise_floor_onset]

                            smooth_energy_dB = smooth_energy_dB[:noise_floor_onset]
                            time_axis = time_axis[:noise_floor_onset]
                            """
                        
                        if noise_floor is None:
                            # No noise floor was detected. The entire duration of the echogram is preserved.
                            echogram[b] = band_energy
                        else:
                            # A noise floor was detected. The echogram is truncated before the floor is reached.
                            echogram[b, :start_of_noise-1] = band_energy[:start_of_noise-1]

                        plt.plot(time_axis / audio_sample_rate,
                                 smooth_energy_dB,
                                 label=f'{band_centers[b]}Hz')
                        
                        plt.scatter(start_of_noise / audio_sample_rate,
                                    smooth_energy_dB[start_of_noise-1],
                                    marker='v', label=f'{(start_of_noise / audio_sample_rate):.2f}s',
                                    color=plt.gca().lines[-1].get_color())
                        
                        # print(f'Chosen cutoff for {band_centers[b]:.0f}Hz: {(noise_floor_onset / audio_sample_rate):.2f}s.')

                    # Truncate to the longest nonzero value.
                    max_valid_sample = np.max(np.nonzero(echogram)[-1])
                    echogram = echogram[:, :max_valid_sample]

                    # plt.xlim(0, max_valid_sample / audio_sample_rate)
                    plt.xlim(0, 4.5)
                    plt.ylim(-90, None)
                    plt.title(short_name)
                    plt.tight_layout()
                    plt.legend(ncol=2)
                    plt.show()

                    break

                    # Save the "masked" echogram.
                    # write(echo_path, int(audio_sample_rate), echogram.T)

                break
            break
        break
