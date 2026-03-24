import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
from scipy.signal import butter, sosfilt
from collections import defaultdict

from raves.src.utils import load_frequencies

mesh_folder = os.path.join('..', 'BRAS meshes')
response_folder = os.path.join('..', '..', '..', 'BRAS', '1 Scene descriptions')

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

plotted_band_idx = 4
backwards_integration = False

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

if __name__ == '__main__':
    # Consider the frequency band centers provided alongside the input data.
    band_centers = load_frequencies(mesh_folder, 'materials_oct_bands.csv')
    num_bands = len(band_centers)
    # Factor for octave-band boundaries.
    band_bound = np.sqrt(2)
    # Ensure that all frequencies support band-pass filtering.
    if np.any(band_centers * band_bound >= audio_sample_rate / 2):
        print('Warning: the audio sample rate is too low for some frequency bands.')
        # Select only acceptable bands.
        band_centers = band_centers[band_centers * band_bound < audio_sample_rate / 2]
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
    
                for remeshing_strategy in ['naive_trng', 'naive_obj']:
                    env_name = short_name + '_' + remeshing_strategy
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
    
                    echos_per_room[short_name][(src, lst, remeshing_strategy)] = echo_data.T
    
        # num_rirs = sum([len(d) for k, d in rirs_per_room[short_name].items()])
        # print(f'Found {num_rirs} recordings in total in room {short_name}.')

        # num_echos = sum([len(d) for k, d in echos_per_room[short_name].items()])
        # print(f'Found {num_echos} ART echograms in total in room {short_name}.')

    for short_name, rirs_dict in rirs_per_room.items():
        fig, ax = plt.subplots(dpi=200, figsize=(9, 6))
    
        for (src, lst), rirs in rirs_dict.items():
            print(f'Found {len(rirs)} recordings in room {short_name} '
                  f'with source {src} and listener {lst}.')
            
            first = True
            for rir in rirs:
                # Prepare an array for the band-pass filtered response.
                banded_rir = np.zeros((num_bands, len(rir)))

                for b in range(num_bands):
                    # Prepare the suitable band-pass filter...
                    sos = butter(6, (band_centers[b] / band_bound,
                                     band_centers[b] * band_bound),
                                 btype='bandpass', output='sos',
                                 fs=audio_sample_rate)
                    # ...and apply it to the room impulse response.
                    banded_rir[b] = sosfilt(sos, rir)
                
                banded_energy = banded_rir**2
                # Normalize to energy-per-second, matching our echogram convention.
                banded_energy *= audio_sample_rate
                # Our convention also normalizes by 4 pi.
                banded_energy *= 4 * np.pi

                if backwards_integration:
                    # Reverse integration
                    banded_energy = np.cumsum(banded_energy[:, ::-1], axis=-1)[:, ::-1]
                    # Decimate (speeds up rendering)
                    banded_energy = banded_energy[:, ::audio_stride]
                    # dB scale
                    banded_energy = 10 * np.log10(banded_energy)
                    # Normalize by total energy
                    banded_energy -= banded_energy[:, 0, None]
                else:
                    num_windows = banded_energy.shape[-1] // audio_stride
                    # https://stackoverflow.com/a/71800940
                    banded_energy = np.array(np.array_split(banded_energy,
                                                            num_windows,
                                                            axis=-1)
                                             ).sum(axis=-1).T
                    # dB scale
                    banded_energy = 10 * np.log10(banded_energy)

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
            
            for remeshing_strategy in ['naive_trng', 'naive_obj']:
                if short_name in echos_per_room:
                    if (src, lst, remeshing_strategy) in echos_per_room[short_name]:
                        echogram = echos_per_room[short_name][(src, lst, remeshing_strategy)]
                
                        if backwards_integration:
                            # Reverse integration
                            echogram = np.cumsum(echogram[:, ::-1], axis=-1)[:, ::-1]
                            # Decimate (speeds up rendering)
                            echogram = echogram[:, ::echo_stride]
                            # dB scale
                            echogram = 10 * np.log10(echogram)
                            # Normalize by total energy
                            echogram -= echogram[:, 0, None]
                        else:
                            num_windows = echogram.shape[-1] // echo_stride
                            # https://stackoverflow.com/a/71800940
                            echogram = np.array(np.array_split(echogram,
                                                                    num_windows,
                                                                    axis=-1)
                                                     ).sum(axis=-1).T
                            # dB scale
                            echogram = 10 * np.log10(echogram)

                        time_axis = np.arange(echogram.shape[-1]) / downsampled_rate

                        plt.plot(time_axis,
                                 echogram[plotted_band_idx],
                                 label = f'{src} {lst} {remeshing_strategy}')
                    
        plt.xlim(0, 2.5)
        # plt.ylim(-60, 0)
        plt.ylim(-120, None)
        plt.title(f'Mesh name {short_name}, with source {src} and listener {lst}; '
                  f'results for {band_centers[plotted_band_idx]} octave band.')
    
        plt.tight_layout()
        plt.legend()
        plt.show()
