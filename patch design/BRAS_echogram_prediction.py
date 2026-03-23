import os
import numpy as np
from scipy.io.wavfile import write

from raves import compute_ART, run_ART

# Duration of the echograms to be displayed, in seconds.
shown_duration = 2.5
# All frequency band plots are saved to files; this one is also displayed.
shown_band = 3
# Sample rate used for the echograms. Mostly relevant to avoid rounding
#   errors in the propagation delays.
echogram_sample_rate = 1e4

mesh_folder = os.path.join('..', 'BRAS meshes')

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

if __name__ == '__main__':
    for room_name in source_positions.keys():
        for remeshing_strategy in ['naive_trng', 'naive_obj']:
            env_name = room_name + '_' + remeshing_strategy
            env_folder = os.path.join(mesh_folder, room_name, env_name)
            
            # Results of the echogram comparison will be saved to this subfolder.
            echograms_subfolder = os.path.join(env_folder, 'Echograms')
            os.makedirs(echograms_subfolder, exist_ok=True)
            
            print('\nPrecomputing environment', env_name, '...\n')
            
            compute_ART(env_folder, assert_coplanarity=False,
                        # points_per_square_meter=10.0,
                        # rays_per_hemisphere=1000,
                        multiprocess_pool_size=16,
                        temperature=air_parameters[room_name][0],
                        humidity=air_parameters[room_name][1])
            
            print('\nRunning environment', env_name, '...\n')
            
            # Need to put positions into arrays, ensuring the correct order.
            # https://stackoverflow.com/a/36634885
            sorted_source_keys = sorted(source_positions[room_name].keys())
            sources_array = np.array([source_positions[room_name][key]
                                      for key in sorted_source_keys])
            sorted_listener_keys = sorted(listener_positions[room_name].keys())
            listeners_array = np.array([listener_positions[room_name][key]
                                        for key in sorted_listener_keys])
                
            echograms, freqs = run_ART(env_folder, sources_array, listeners_array,
                                       echogram_sample_rate=echogram_sample_rate,
                                       echogram_duration=shown_duration,
                                       output_folder_path=echograms_subfolder,
                                       assert_coplanarity=False,
                                       assert_min_delays=False)
            
            for src_idx, src in enumerate(sorted_source_keys):
                for lst_idx, lst in enumerate(sorted_listener_keys):
                    write(os.path.join(echograms_subfolder, src + lst + '.wav'),
                          echogram_sample_rate, echograms[src_idx, lst_idx])
