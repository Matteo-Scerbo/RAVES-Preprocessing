import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
from collections import defaultdict

from raves import compute_ART, run_ART

mesh_folder = os.path.join('..', 'BRAS meshes')
response_folder = os.path.join('..', '..', '..', 'BRAS', '1 Scene descriptions')

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

# https://stackoverflow.com/a/5029958
rirs_per_room = defaultdict(lambda: defaultdict(list))

for short_name, full_name in full_room_names.items():
    rir_folder = os.path.join(response_folder, full_name, 'RIRs', 'wav')
    if '_' in short_name:
        rir_prefix = short_name.replace('_', '_RIR_')
    else:
        rir_prefix = short_name + '_RIR'
    
    for src, src_pos in source_positions[short_name].items():
        for lst, lst_pos in listener_positions[short_name].items():
            for src_typ in source_types:
                rir_name = '_'.join([rir_prefix, src, lst, src_typ])
                rir_path = os.path.join(rir_folder, rir_name) + '.wav'
                
                try:
                    fs, rir_data = read(rir_path)
                except FileNotFoundError:
                    # print(f'Missing RIR file:\n\t{rir_name}\nFull path:\n\t{rir_path}\n')
                    continue

                rirs_per_room[short_name][(src, lst)].append(rir_data)

for short_name, rirs_dict in rirs_per_room.items():
    fig, ax = plt.subplots(dpi=200, figsize=(8, 4))

    for (src, lst), rirs in rirs_dict.items():
        print('Found', len(rirs), 'recordings in room', short_name,
              'with source', src, 'and listener', lst)
        
        first = True
        for rir in rirs:
            energy = rir**2
            # Reverse integration
            edc = np.cumsum(energy[::-1])[::-1]
            # dB scale
            edc = 10 * np.log10(edc)
            # Normalize by total energy
            edc -= edc[0]
            # Decimate (speeds up rendering)
            edc = edc[::100]
            
            if first:
                plt.plot(edc, label = src + ' ' + lst)
                first = False
            else:
                plt.plot(edc, color = plt.gca().lines[-1].get_color())
    
    plt.ylim(-60, 0)
    plt.title('Mesh name: ' + short_name)

    plt.tight_layout()
    plt.legend()
    plt.show()
