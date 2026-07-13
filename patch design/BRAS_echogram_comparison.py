import os
import csv
import itertools
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches

from cycler import cycler
from okabeito import lightblue, yellow, orange, green, purple, red, blue, black

default_cycler = (cycler(color=[lightblue, yellow, orange, green, black, purple, red, blue]))
plt.rc('axes', prop_cycle=default_cycler)

from scipy.io.wavfile import read
from scipy.signal import fftconvolve
from scipy.signal.windows import get_window

from collections import defaultdict

from raves.src.utils import load_frequencies


# https://gist.github.com/vishalkuo/f4aec300cf6252ed28d3
def remove_outliers(x, outlier_const):
    a = np.array(x).flatten()
    if outlier_const <= 0:
        return a
    
    upper_quartile = np.percentile(a, 75)
    lower_quartile = np.percentile(a, 25)
    iqr = (upper_quartile - lower_quartile) * outlier_const
    quartileSet = (lower_quartile - iqr, upper_quartile + iqr)
    
    return a[np.where((a >= quartileSet[0]) & (a <= quartileSet[1]))]


# https://stackoverflow.com/a/58324984
def add_violin_label(violin, label, label_list):
    color = violin["bodies"][0].get_facecolor().flatten()
    label_list.append((mpatches.Patch(color=color), label))


if __name__ == '__main__':
    mesh_folder = os.path.join('..', 'BRAS meshes')
    output_folder = os.path.join('.', 'data_for_figures')

    shown_plots = [# 'Echogram',
                   'Full echogram',
                   # 'Spectrogram error',
                   # 'Single spectrogram error',
                   # 'Violin plot',
                   # 'Single violin plot',
                   # 'Freq-wise violin plot'
                   ]

    audio_sample_rate = 44100
    echo_sample_rate = 8820
    # Choose a low sample rate for plotting and evaluating energy results.
    # In order to avoid having to do any resampling, choose a common divider of
    #  the recording and ART sample rates.
    downsampled_rate = 2205
    # Possible divisors:
    #  [8820, 4410, 2940, 2205, 1764, 1470, 1260,  980,
    #    882,  735,  630,  588,  490,  441,  420,  315,
    #    294,  252,  245,  210,  196,  180,  147,  140,
    #    126,  105,   98,   90,   84,   70,   63,   60,
    #     49,   45,   42,   36,   35,   30,   28,   21,
    #     20,   18,   15,   14,   12,   10,    9,    7,
    #      6,    5,    4,    3,    2,    1]
    audio_stride = int(audio_sample_rate / downsampled_rate)
    echo_stride = int(echo_sample_rate / downsampled_rate)
    audio_nyquist = audio_sample_rate / 2

    # Length of the smoothing window applied to the energy envelopes after downsampling.
    smoothing_window_len = 100e-3
    # Broadest: 'cosine', 'lanczos', 'tukey'
    smoothing_window = get_window('tukey', int(smoothing_window_len * downsampled_rate))
    smoothing_window /= np.sum(smoothing_window)

    plotted_src_lst_band = ('LS2', 'MP4', 3)
    plotted_time_range = 'max'
    plotted_band_range = (0, 7)
    backwards_integration = False
    forwards_integration = False
    # Responses are normalized to have unit mean energy between 0 and ´normalization_period´.
    # Set it to 0 to disable normalization. Set it to np.inf to normalize the total energy.
    normalization_period = 0
    # Normalize each frequency band separately, or all together.
    band_wise_norm = False
    outlier_constant = 0.0

    dode_2k_compensation = 4.0

    reference_name = 'Dodecahedron'
    # reference_name = 'Genelec 1'

    full_room_names = {# 'CR1_DoorAngle1': 'CR1 coupled rooms (laboratory and reverberation chamber)',
                       # 'CR1_DoorAngle1_simplified': 'CR1 coupled rooms (laboratory and reverberation chamber)',
                       # 'CR1_DoorAngle1_ubersimplified': 'CR1 coupled rooms (laboratory and reverberation chamber)',
                       'CR1_DoorAngle3': 'CR1 coupled rooms (laboratory and reverberation chamber)',
                    #    'CR1_DoorAngle3_simplified': 'CR1 coupled rooms (laboratory and reverberation chamber)',
                    #    'CR1_DoorAngle3_ubersimplified': 'CR1 coupled rooms (laboratory and reverberation chamber)',
                    #    'CR2': 'CR2 small room (seminar room)',
                    #    'CR2_simplified': 'CR2 small room (seminar room)',
                    #    'CR2_ubersimplified': 'CR2 small room (seminar room)',
                    #    'CR3': 'CR3 medium room (chamber music hall)',
                       # 'CR4': 'CR4 large room (auditorium)',
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
                                # 'LS3': [4.812, 2.358, 1.76], # Only Genelec
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
    shown_durations = {'CR1_DoorAngle1': 4.0,
                       'CR1_DoorAngle3': 4.0,
                       'CR2': 2.5,
                       'CR3': 2.0,
                       'CR4': 2.5,
                       }

    loudspeaker_aliases = {'Genelec8020c_LSorientation-negativeX': 'Genelec 1',
                           'Genelec8020c_LSorientation-negativeY': 'Genelec 2',
                           'Genelec8020c_LSorientation-positiveX': 'Genelec 3',
                           'Genelec8020c_LSorientation-positiveY': 'Genelec 4',
                           'Genelec8020c_LSorientation-01': 'Genelec 1',
                           'Genelec8020c_LSorientation-02': 'Genelec 2',
                           'Genelec8020c_LSorientation-03': 'Genelec 3',
                           'Genelec8020c_LSorientation-04': 'Genelec 4',
                           'Dodecahedron': 'Dodecahedron'
                           }
    strategy_aliases = {'naive_obj': 'Largest possible',
                        'naive_trng': 'Bad triangulation',
                        'split_area': r'Target $4\text{m}^2$',
                        'split_area_length': r'Target $4\text{m}^2$, compact',
                        'uber_split_area': r'Target $2\text{m}^2$',
                        'uber_split_area_length': r'Target $2\text{m}^2$, compact'
                        }
    room_aliases = {k: k.replace(
                        '_DoorAngle1',
                        ', closed'
                        ).replace(
                            '_DoorAngle3',
                            ', open'
                            ).replace(
                                '_simplified',
                                '\n(simplified)'
                                ).replace(
                                    '_ubersimplified',
                                    '\n(ultra-simplified)')
                    for k in full_room_names.keys()}

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
    # Bandwidths used for some normalization aspects.
    band_widths = (band_centers * band_bound) - (band_centers / band_bound)

    def tick_label_func(val, pos=None):
        if val >= num_bands:
            return 'error'
        elif band_centers[int(val)] < 1e3:
            return f'{int(band_centers[int(val)])}'
        else:
            return f'{int(band_centers[int(val)] / 1e3)}k'

    echos_per_room = defaultdict(dict)
    full_echos_per_room = defaultdict(dict)
    
    for short_name, full_name in full_room_names.items():
        print('Loading echograms,', short_name)
        
        base_name = short_name.replace('_simplified', '')
        base_name = base_name.replace('_ubersimplified', '')

        ref_echo_subfolder = os.path.join(mesh_folder, base_name, 'Reference echograms')

        if '_' in base_name:
            rir_prefix = base_name.replace('_', '_RIR_')
        else:
            rir_prefix = base_name + '_RIR'
        
        for src in source_positions[base_name].keys():
            for lst in listener_positions[base_name].keys():
                for ls_long, ls_short in loudspeaker_aliases.items():
                    rir_name = '_'.join([rir_prefix, src, lst, ls_long])
                    ref_echo_path = os.path.join(ref_echo_subfolder, rir_name + '.wav')
                    full_echo_path = os.path.join(ref_echo_subfolder, rir_name + '_unmasked.wav')
                    
                    try:
                        fs, ref_data = read(ref_echo_path)
                    except FileNotFoundError:
                        # print(f'Missing reference file:\n\t{rir_name}\nFull path:\n\t{ref_echo_path}\n')
                        if rir_name == 'CR2_RIR_LS1_MP5_Genelec8020c_LSorientation-positiveY':
                            # This RIR is missing from the dataset. Replace it to make plots fit correctly.
                            ref_data = np.ones_like(echogram.T)
                        else:
                            continue
                    assert fs == audio_sample_rate, (fs, audio_sample_rate)
                    echogram = ref_data.copy().T

                    # The recording setup was calibrated based on the sound pressure at 1kHz,
                    #  in front of the loudspeaker. There is a problem with this.
                    # The dodecahedron measurements have a "dip" around 2kHz, possibly due to
                    #  phase cancellation in the crossover between the mid and high speakers.
                    if ls_short == 'Dodecahedron':
                        echogram[4] *= dode_2k_compensation

                    # The simulations are calibrated such that, when a unit-energy signal is band-passed
                    #  to the relative octave band and modulated by the (simulated) echogram's root, it
                    #  has the correct energy. The band-passing removes energy according to the bandwidth,
                    #  which we need to compensate for.
                    echogram /= band_widths[:, None]
                    echogram *= audio_sample_rate

                    # Trim the duration to a multiple of the downsampling stride, to match the simulations.
                    # This is necessary for a fair comparison with backwards integration.
                    for b in range(num_bands):
                        ref_duration = np.max(np.flatnonzero(echogram[b]))
                        ref_duration = (ref_duration // audio_stride) * audio_stride
                        echogram[b, ref_duration+1:] = 0
                    
                    # Normalize by early or total energy.
                    if np.isinf(normalization_period):
                        if band_wise_norm:
                            echogram /= np.sum(echogram, axis=-1)[:, None]
                        else:
                            band_energies = np.sum(echogram, axis=-1)
                            band_energies /= band_widths / np.sum(band_widths)
                            echogram /= np.average(band_energies)
                    elif normalization_period > 0:
                        if band_wise_norm:
                            echogram /= np.sum(echogram[:int(fs*normalization_period)],
                                               axis=-1)[:, None]
                        else:
                            band_energies = np.sum(echogram[:int(fs*normalization_period)],
                                                   axis=-1)
                            band_energies /= band_widths / np.sum(band_widths)
                            echogram /= np.average(band_energies)
        
                    if backwards_integration:
                        # Reverse integration
                        echogram = np.cumsum(echogram[:, ::-1], axis=-1)[:, ::-1]
                        # Downsampling the EDC is easy, just skip values
                        echogram = echogram[:, ::audio_stride]
                    elif forwards_integration:
                        # Regular integration
                        echogram = np.cumsum(echogram, axis=-1)
                        # Downsampling the accumulated energy is easy, just skip values
                        echogram = echogram[:, ::audio_stride]
                    else:
                        # Short-time average (and downsampling)
                        num_windows = echogram.shape[-1] // audio_stride
                        remainder = echogram.shape[-1] % audio_stride
                        if remainder != 0:
                            echogram = echogram[:, :-remainder]
                        # https://stackoverflow.com/a/71800940
                        echogram = np.array(np.split(echogram, num_windows, axis=-1)
                                            ).sum(axis=-1).T

                        # Finally, apply smoothing.
                        for b in range(num_bands):
                            # The strict zero elements need to be preserved through the smoothing,
                            #  because they define the mise floor "masking" of the references,
                            #  i.e., which samples are suitable for comparison.
                            max_valid_sample = np.max(np.nonzero(echogram[b])[-1])

                            echogram[b] = fftconvolve(echogram[b], smoothing_window,
                                                      mode='same')
                            
                            # N.B.: This gets rid of the artifacts at both ends of the response,
                            #  which means the early reflections are NOT part of the comparison.
                            half_window_len = int(len(smoothing_window) // 2)
                            echogram[b, :half_window_len] = 0
                            echogram[b, max_valid_sample-half_window_len:] = 0

                    max_duration = int(np.floor(shown_durations[base_name] * downsampled_rate))
                    if echogram.shape[-1] >= max_duration:
                        echogram = echogram[:, :max_duration]
                    elif echogram.shape[-1] < max_duration:
                        echogram = np.pad(echogram,
                                          ((0, 0), (0, max_duration - echogram.shape[-1])),
                                          mode=('edge' if forwards_integration else 'constant'))
                    
                    # dB scale
                    echogram = 10 * np.log10(echogram)
                
                    echos_per_room[short_name][(src, lst, ls_short)] = echogram

                    # Load the unmasked echogram as well, for a figure in the paper.

                    try:
                        fs, full_data = read(full_echo_path)
                    except FileNotFoundError:
                        # print(f'Missing reference file:\n\t{rir_name}\nFull path:\n\t{ref_echo_path}\n')
                        if rir_name == 'CR2_RIR_LS1_MP5_Genelec8020c_LSorientation-positiveY':
                            # This RIR is missing from the dataset. Replace it to make plots fit correctly.
                            full_data = np.ones_like(echogram.T)
                        else:
                            continue
                    assert fs == audio_sample_rate, (fs, audio_sample_rate)
                    echogram = full_data.copy().T

                    # The recording setup was calibrated based on the sound pressure at 1kHz,
                    #  in front of the loudspeaker. There is a problem with this.
                    # The dodecahedron measurements have a "dip" around 2kHz, possibly due to
                    #  phase cancellation in the crossover between the mid and high speakers.
                    if ls_short == 'Dodecahedron':
                        echogram[4] *= dode_2k_compensation

                    # The simulations are calibrated such that, when a unit-energy signal is band-passed
                    #  to the relative octave band and modulated by the (simulated) echogram's root, it
                    #  has the correct energy. The band-passing removes energy according to the bandwidth,
                    #  which we need to compensate for.
                    echogram /= band_widths[:, None]
                    echogram *= audio_sample_rate

                    # Normalize by early or total energy.
                    if np.isinf(normalization_period):
                        if band_wise_norm:
                            echogram /= np.sum(echogram, axis=-1)[:, None]
                        else:
                            band_energies = np.sum(echogram, axis=-1)
                            band_energies /= band_widths / np.sum(band_widths)
                            echogram /= np.average(band_energies)
                    elif normalization_period > 0:
                        if band_wise_norm:
                            echogram /= np.sum(echogram[:int(fs*normalization_period)],
                                               axis=-1)[:, None]
                        else:
                            band_energies = np.sum(echogram[:int(fs*normalization_period)],
                                                   axis=-1)
                            band_energies /= band_widths / np.sum(band_widths)
                            echogram /= np.average(band_energies)
        
                    if backwards_integration:
                        # Reverse integration
                        echogram = np.cumsum(echogram[:, ::-1], axis=-1)[:, ::-1]
                        # Downsampling the EDC is easy, just skip values
                        echogram = echogram[:, ::audio_stride]
                    elif forwards_integration:
                        # Regular integration
                        echogram = np.cumsum(echogram, axis=-1)
                        # Downsampling the accumulated energy is easy, just skip values
                        echogram = echogram[:, ::audio_stride]
                    else:
                        # Short-time average (and downsampling)
                        num_windows = echogram.shape[-1] // audio_stride
                        remainder = echogram.shape[-1] % audio_stride
                        if remainder != 0:
                            echogram = echogram[:, :-remainder]
                        # https://stackoverflow.com/a/71800940
                        echogram = np.array(np.split(echogram, num_windows, axis=-1)
                                            ).sum(axis=-1).T

                        # Finally, apply smoothing.
                        for b in range(num_bands):
                            # The strict zero elements need to be preserved through the smoothing,
                            #  because they define the mise floor "masking" of the references,
                            #  i.e., which samples are suitable for comparison.
                            max_valid_sample = np.max(np.nonzero(echogram[b])[-1])

                            echogram[b] = fftconvolve(echogram[b], smoothing_window,
                                                      mode='same')
                            
                            # N.B.: The windowing artifacts are NOT removed for the full responses.

                    max_duration = int(np.floor(shown_durations[base_name] * downsampled_rate))
                    if echogram.shape[-1] >= max_duration:
                        echogram = echogram[:, :max_duration]
                    elif echogram.shape[-1] < max_duration:
                        echogram = np.pad(echogram,
                                          ((0, 0), (0, max_duration - echogram.shape[-1])),
                                          mode=('edge' if forwards_integration else 'constant'))
                    
                    # dB scale
                    echogram = 10 * np.log10(echogram)
                
                    full_echos_per_room[short_name][(src, lst, ls_short)] = echogram

                for mesh_strat in strategy_aliases.keys():
                    if 'CR3' in short_name and mesh_strat not in ['naive_obj', 'naive_trng', 'split_area']:
                        continue

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
                    echogram = sim_data.copy().T

                    # Trim the duration in each band to the length of the reference.
                    # This is necessary for a fair comparison with backwards integration.
                    reference = echos_per_room[short_name][(src, lst, reference_name)]
                    for b in range(num_bands):
                        # Note: the reference was converted to dB, the zero values are now NaN.
                        ref_duration = np.max(np.flatnonzero(np.isfinite(reference[b])))
                        ref_duration *= echo_stride
                        echogram[b, ref_duration+1:] = 0
                    
                    # Normalize by early or total energy.
                    if np.isinf(normalization_period):
                        if band_wise_norm:
                            echogram /= np.sum(echogram, axis=-1)[:, None]
                        else:
                            band_energies = np.sum(echogram, axis=-1)
                            band_energies /= band_widths / np.sum(band_widths)
                            echogram /= np.average(band_energies)
                    elif normalization_period > 0:
                        if band_wise_norm:
                            echogram /= np.sum(echogram[:int(fs*normalization_period)],
                                               axis=-1)[:, None]
                        else:
                            band_energies = np.sum(echogram[:int(fs*normalization_period)],
                                                   axis=-1)
                            band_energies /= band_widths / np.sum(band_widths)
                            echogram /= np.average(band_energies)
        
                    if backwards_integration:
                        # Reverse integration
                        echogram = np.cumsum(echogram[:, ::-1], axis=-1)[:, ::-1]
                        # Downsampling the EDC is easy, just skip values
                        echogram = echogram[:, ::echo_stride]
                    elif forwards_integration:
                        # Regular integration
                        echogram = np.cumsum(echogram, axis=-1)
                        # Downsampling the accumulated energy is easy, just skip values
                        echogram = echogram[:, ::audio_stride]
                    else:
                        # Short-time average (and downsampling)
                        num_windows = echogram.shape[-1] // echo_stride
                        remainder = echogram.shape[-1] % echo_stride
                        if remainder != 0:
                            echogram = echogram[:, :-remainder]
                        # https://stackoverflow.com/a/71800940
                        echogram = np.array(np.split(echogram, num_windows, axis=-1)
                                            ).sum(axis=-1).T
                    
                        # Finally, apply smoothing.
                        for b in range(num_bands):
                            # The strict zero elements need to be preserved through the smoothing,
                            #  because they define the mise floor "masking" of the references,
                            #  i.e., which samples are suitable for comparison.
                            max_valid_sample = np.max(np.nonzero(echogram[b])[-1])

                            echogram[b] = fftconvolve(echogram[b], smoothing_window,
                                                      mode='same')
                            
                            # N.B.: This gets rid of the artifacts at both ends of the response,
                            #  which means the early reflections are NOT part of the comparison.
                            half_window_len = int(len(smoothing_window) // 2)
                            echogram[b, :half_window_len] = 0
                            echogram[b, max_valid_sample-half_window_len:] = 0

                    max_duration = int(np.floor(shown_durations[base_name] * downsampled_rate))
                    if echogram.shape[-1] >= max_duration:
                        echogram = echogram[:, :max_duration]
                    elif echogram.shape[-1] < max_duration:
                        echogram = np.pad(echogram,
                                          ((0, 0), (0, max_duration - echogram.shape[-1])),
                                          mode=('edge' if forwards_integration else 'constant'))
                    
                    # dB scale
                    echogram = 10 * np.log10(echogram)
                
                    echos_per_room[short_name][(src, lst, mesh_strat)] = echogram

                    # Load the unmasked echogram as well, for a figure in the paper.

                    echogram = sim_data.copy().T

                    # Normalize by early or total energy.
                    if np.isinf(normalization_period):
                        if band_wise_norm:
                            echogram /= np.sum(echogram, axis=-1)[:, None]
                        else:
                            band_energies = np.sum(echogram, axis=-1)
                            band_energies /= band_widths / np.sum(band_widths)
                            echogram /= np.average(band_energies)
                    elif normalization_period > 0:
                        if band_wise_norm:
                            echogram /= np.sum(echogram[:int(fs*normalization_period)],
                                               axis=-1)[:, None]
                        else:
                            band_energies = np.sum(echogram[:int(fs*normalization_period)],
                                                   axis=-1)
                            band_energies /= band_widths / np.sum(band_widths)
                            echogram /= np.average(band_energies)
        
                    if backwards_integration:
                        # Reverse integration
                        echogram = np.cumsum(echogram[:, ::-1], axis=-1)[:, ::-1]
                        # Downsampling the EDC is easy, just skip values
                        echogram = echogram[:, ::echo_stride]
                    elif forwards_integration:
                        # Regular integration
                        echogram = np.cumsum(echogram, axis=-1)
                        # Downsampling the accumulated energy is easy, just skip values
                        echogram = echogram[:, ::audio_stride]
                    else:
                        # Short-time average (and downsampling)
                        num_windows = echogram.shape[-1] // echo_stride
                        remainder = echogram.shape[-1] % echo_stride
                        if remainder != 0:
                            echogram = echogram[:, :-remainder]
                        # https://stackoverflow.com/a/71800940
                        echogram = np.array(np.split(echogram, num_windows, axis=-1)
                                            ).sum(axis=-1).T
                    
                        # Finally, apply smoothing.
                        for b in range(num_bands):
                            # The strict zero elements need to be preserved through the smoothing,
                            #  because they define the mise floor "masking" of the references,
                            #  i.e., which samples are suitable for comparison.
                            max_valid_sample = np.max(np.nonzero(echogram[b])[-1])

                            echogram[b] = fftconvolve(echogram[b], smoothing_window,
                                                      mode='same')
                    
                    # dB scale
                    echogram = 10 * np.log10(echogram)
                
                    full_echos_per_room[short_name][(src, lst, mesh_strat)] = echogram

    all_violin_data = dict()

    for short_name, echos_dict in echos_per_room.items():
        print('Preparing plots,', short_name)
        
        base_name = short_name.replace('_simplified', '')
        base_name = base_name.replace('_ubersimplified', '')

        # Count the number of src/lst configuration, i.e., reference echograms.
        num_sl_configs = len([(s, l) for s, l, k in echos_dict
                              if k == reference_name])
        if num_sl_configs == 0:
            print(f'No reference echograms were found for room {short_name}.')
            continue
        # Count the number of echograms that were loaded, other than the reference ones.
        num_sim_echos = len([k for s, l, k in echos_dict
                             if 'Dodecahedron' not in k and 'Genelec' not in k])
        if num_sim_echos == 0:
            print(f'No ART echograms were found for room {short_name}.')
            continue

        num_sources = len(source_positions[base_name])
        num_listeners = len(listener_positions[base_name])

        if 'Echogram' in shown_plots:
            fig, ax = plt.subplots(dpi=100, figsize=(2.0*4, 2.0*3))

            bottom = np.inf
            for (src, lst, key), echogram in echos_dict.items():
                if src != plotted_src_lst_band[0]:
                    continue
                if lst != plotted_src_lst_band[1]:
                    continue

                time_axis = np.arange(echogram.shape[-1]) / downsampled_rate

                plt.plot(time_axis, echogram[plotted_src_lst_band[2]],
                         ls=('-' if 'Genelec' in key or 'Dodecahedron' in key else '--'),
                         label=(key
                                if 'Genelec' in key or 'Dodecahedron' in key else
                                strategy_aliases[key]))
                
                nonzero_idxs = np.isfinite(echogram[plotted_src_lst_band[2]])
                bottom = min(bottom, np.min(echogram[plotted_src_lst_band[2]][nonzero_idxs]))

            plt.xlim(0, (shown_durations[base_name]
                         if plotted_time_range == 'max'
                         else plotted_time_range))
            if backwards_integration and np.isinf(normalization_period):
                plt.ylim(-60, 0)
            elif not forwards_integration:
                plt.ylim(bottom, None)
            
            plt.xlabel('Time [s]')
            plt.ylabel('Energy [dB]')

            # https://stackoverflow.com/a/77328370
            plt.legend(ncol=2, handleheight=2)

            plt.title(f'Room {room_aliases[short_name]}; {band_centers[plotted_src_lst_band[2]]}Hz octave band.')
            plt.tight_layout()
            plt.show()
        
        if 'Full echogram' in shown_plots and short_name == 'CR1_DoorAngle3':
            fig, ax = plt.subplots(dpi=100, figsize=(2.0*4, 2.0*3))

            bottom = np.inf
            for (src, lst, key), echogram in full_echos_per_room[short_name].items():
                if src != plotted_src_lst_band[0]:
                    continue
                if lst != plotted_src_lst_band[1]:
                    continue

                time_axis = np.arange(echogram.shape[-1]) / downsampled_rate

                plt.plot(time_axis, echogram[plotted_src_lst_band[2]],
                         ls=('-' if 'Genelec' in key or 'Dodecahedron' in key else '--'),
                         label=(key
                                if 'Genelec' in key or 'Dodecahedron' in key else
                                strategy_aliases[key]))
                
                with open(os.path.join(output_folder, f'{tick_label_func(plotted_src_lst_band[2])}Hz_{short_name}_{src}_{lst}_{key.replace(' ', '_')}-echogram.dat'),
                          mode='w') as file:
                    for time_idx in range(echogram.shape[-1]):
                        if np.isfinite(echogram[plotted_src_lst_band[2], time_idx]):
                            file.write(f'{time_axis[time_idx]} {echogram[plotted_src_lst_band[2], time_idx]}\n')

                nonzero_idxs = np.isfinite(echogram[plotted_src_lst_band[2]])
                bottom = min(bottom, np.min(echogram[plotted_src_lst_band[2]][nonzero_idxs]))

            plt.xlim(0, (shown_durations[base_name]
                         if plotted_time_range == 'max'
                         else plotted_time_range))
            if backwards_integration and np.isinf(normalization_period):
                plt.ylim(-60, 0)
            elif not forwards_integration:
                plt.ylim(bottom, None)
            
            plt.xlabel('Time [s]')
            plt.ylabel('Energy [dB]')

            # https://stackoverflow.com/a/77328370
            plt.legend(ncol=2, handleheight=2)

            plt.title(f'Room {room_aliases[short_name]}; {band_centers[plotted_src_lst_band[2]]}Hz octave band.')
            plt.tight_layout()
            plt.show()
        
        # Plot the energy differences.

        spectrogram_errors = defaultdict(dict)
        for (src, lst, strat), echo in echos_dict.items():
            if strat == reference_name:
                continue

            reference = echos_dict[(src, lst, reference_name)]
            error = echo - reference
            spectrogram_errors[(src, lst)][strat] = error
        
        num_comparisons = max([len(d) for d in spectrogram_errors.values()])

        # Diverging colormap to differentiate positive and negative values.
        contour_levels = np.linspace(-10, 10, 21)
        cmap = plt.get_cmap('RdBu', len(contour_levels)+1)
        # Set "bad" values (i.e., reference is below noise floor) to black.
        cmap.set_bad('black', 1.)
        norm = mpl.colors.BoundaryNorm(contour_levels, ncolors=cmap.N, extend='both')

        if 'Spectrogram error' in shown_plots:
            fig, axes = plt.subplots(num_sl_configs, num_comparisons,
                                     figsize=(4*num_comparisons, 3*num_sl_configs),
                                     squeeze=False, constrained_layout=True)

            cs = None
            for i, ((src, lst), spec_dict) in enumerate(spectrogram_errors.items()):
                X, Y = np.meshgrid(np.arange(reference.shape[-1]) / downsampled_rate,
                                   np.arange(len(band_centers)))

                for j, (strat, error) in enumerate(spec_dict.items()):
                    cs = axes[i, j].pcolormesh(X, Y, error, norm=norm, cmap=cmap)

                    axes[i, j].yaxis.set_major_formatter(ticker.FuncFormatter(tick_label_func))

                    axes[i, j].set_title(f'{src} {lst} {strat}')
                    axes[i, j].set_xlim(0, (shown_durations[base_name]
                         if plotted_time_range == 'max'
                         else plotted_time_range))
                    # axes[i, j].set_ylim(plotted_band_range[0]-0.5, plotted_band_range[1]+0.5)
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
                cbar.ax.set_ylabel('Backward-integrated energy difference',
                                   rotation=270, labelpad=15)
            elif forwards_integration:
                cbar.ax.set_ylabel('Forward-integrated energy difference',
                                   rotation=270, labelpad=15)
            else:
                cbar.ax.set_ylabel('Short-time-average energy difference',
                                   rotation=270, labelpad=15)

            plt.suptitle(f'Room {room_aliases[short_name]}')
            plt.show()

        if 'Single spectrogram error' in shown_plots:
            fig, ax = plt.subplots(dpi=100, figsize=(2.0*4, 2.0*3))

            src, lst = plotted_src_lst_band[0], plotted_src_lst_band[1]
            spec_dict = spectrogram_errors[(src, lst)]

            strat = 'split_area'
            error = spec_dict[strat]

            X, Y = np.meshgrid(np.arange(reference.shape[-1]) / downsampled_rate,
                               np.arange(len(band_centers)))

            cs = ax.pcolormesh(X, Y, error, norm=norm, cmap=cmap)

            ax.yaxis.set_major_formatter(ticker.FuncFormatter(tick_label_func))

            ax.set_title(f'{src}, {lst}, {(strat
                                           if 'Genelec' in strat or 'Dodecahedron' in mesh_strat else
                                           strategy_aliases[strat])}')
            ax.set_xlim(0, (shown_durations[base_name]
                         if plotted_time_range == 'max'
                         else plotted_time_range))
            # ax.set_ylim(plotted_band_range[0]-0.5, plotted_band_range[1]+0.5)
            ax.set_xlabel('Time [s]')
            ax.set_ylabel('Octave band center [Hz]')
    
            cbar = fig.colorbar(cs, ax=ax, format='{x:.0f}dB')
            if backwards_integration:
                cbar.ax.set_ylabel('Backward-integrated energy difference',
                                   rotation=270, labelpad=15)
            elif forwards_integration:
                cbar.ax.set_ylabel('Forward-integrated energy difference',
                                   rotation=270, labelpad=15)
            else:
                cbar.ax.set_ylabel('Short-time-average energy difference',
                                   rotation=270, labelpad=15)

            plt.suptitle(f'Room {room_aliases[short_name]}')
            plt.show()

        # Plot the energy differences statistics.

        if 'Violin plot' in shown_plots:
            fig, axes = plt.subplots(num_listeners, num_sources,
                                     figsize=(4*num_sources, 3*num_listeners),
                                     constrained_layout=True)

            group_centers = np.arange(num_bands)
            # https://stackoverflow.com/a/11603806
            group_margin = 0.2
            mid_margin = 0.02
            width = (1 - 2*group_margin) / num_comparisons

            for i, lst in enumerate(listener_positions[base_name]):
                for j, src in enumerate(source_positions[base_name]):
                    # Reset legend labels.
                    violin_labels = list()

                    # Reference horizontal line at 0.
                    line = axes[i, j].hlines(0, -1, num_bands+1,
                                            color='black', ls='--',
                                            linewidth=1)
                    
                    for k, (mesh_strat, error_data) in enumerate(spectrogram_errors[(src, lst)].items()):
                        # The NaN entries must be filtered out for each octave band.
                        error_data = [e[np.isfinite(e)]
                                      for e in error_data]

                        positions = group_centers - 0.5 + group_margin + (k+0.5)*width
                        violin = axes[i, j].violinplot([remove_outliers(x, outlier_constant)
                                                        for x in error_data],
                                                       positions=positions,
                                                       widths=width-mid_margin,
                                                       side='both',
                                                       # quantiles=[[0.05, 0.5, 0.95]]*num_bands,
                                                       showextrema=False,
                                                       showmeans=True,
                                                       showmedians=False)

                        for pc in violin['bodies']:
                            pc.set_edgecolor(pc.get_facecolor())
                            pc.set_alpha(0.5)

                        add_violin_label(violin, (mesh_strat
                                                  if 'Genelec' in mesh_strat or 'Dodecahedron' in mesh_strat else
                                                  strategy_aliases[mesh_strat]),
                                         violin_labels)
                    
                    axes[i, j].set_title(f'{src} {lst}')

                    if backwards_integration:
                        axes[i, j].set_ylim(-15, 15)
                    else:
                        axes[i, j].set_ylim(-15, 15)
                    axes[i, j].set_xlim(plotted_band_range[0]-0.5, plotted_band_range[1]+0.5)
                    axes[i, j].xaxis.set_major_formatter(ticker.FuncFormatter(tick_label_func))

                    axes[i, j].grid(axis='y')

                    if i == num_listeners-1:
                        axes[i, j].set_xlabel('Octave band center [Hz]')
                    else:
                        axes[i, j].set_xlabel('')
                    if j == 0:
                        axes[i, j].set_ylabel('Energy difference [dB]')
                    else:
                        axes[i, j].set_ylabel('')

                    if (i, j) == (0, num_sources-1):
                        axes[i, j].legend(*zip(*violin_labels), ncol=2,
                                          handleheight=2)

            if backwards_integration:
                plt.suptitle(f'{room_aliases[short_name]} - backward-integrated energy diff')
            elif forwards_integration:
                plt.suptitle(f'{room_aliases[short_name]} - forward-integrated energy diff')
            else:
                plt.suptitle(f'{room_aliases[short_name]} - short-time-average energy diff')
            plt.show()

        combined_data = dict()

        for i, lst in enumerate(listener_positions[base_name]):
            for j, src in enumerate(source_positions[base_name]):
                for mesh_strat, error_data in spectrogram_errors[(src, lst)].items():
                    combined_data[mesh_strat] = defaultdict(list)
                    for band_idx, band_error in enumerate(error_data):
                        # The NaN entries must be filtered out for each octave band.
                        finite_error = band_error[np.isfinite(band_error)]
                        finite_error = list(finite_error)
                        combined_data[mesh_strat][band_idx].extend(finite_error)

        if 'Single violin plot' in shown_plots:
            fig, ax = plt.subplots(dpi=100, figsize=(0.75*16, 0.75*9))

            group_centers = np.arange(num_bands)
            # https://stackoverflow.com/a/11603806
            group_margin = 0.1
            mid_margin = 0.02
            width = (1 - 2*group_margin) / num_comparisons

            # Reset legend labels.
            violin_labels = list()

            # Reference horizontal line at 0.
            line = plt.hlines(0, -1, num_bands+1,
                              color='black', ls='--',
                              linewidth=1)
            
            for k, (mesh_strat, error_data) in enumerate(combined_data.items()):
                positions = group_centers - 0.5 + group_margin + (k+0.5)*width
                violin = ax.violinplot([remove_outliers(x, outlier_constant)
                                        for x in error_data.values()],
                                       positions=positions,
                                       widths=width-mid_margin,
                                       side='both',
                                       points=1000,
                                       bw_method=0.2,
                                       # quantiles=[[0.05, 0.5, 0.95]]*num_bands,
                                       showextrema=False,
                                       showmeans=True,
                                       showmedians=False)

                for pc in violin['bodies']:
                    pc.set_edgecolor(pc.get_facecolor())
                    pc.set_alpha(0.5)

                add_violin_label(violin, (mesh_strat
                                          if 'Genelec' in mesh_strat or 'Dodecahedron' in mesh_strat else
                                          strategy_aliases[mesh_strat]),
                                 violin_labels)
            
            if backwards_integration:
                plt.ylim(-15, 15)
            else:
                plt.ylim(-30, 30)
            plt.xlim(plotted_band_range[0]-0.5, plotted_band_range[1]+0.5)

            ax.xaxis.set_major_formatter(ticker.FuncFormatter(tick_label_func))

            plt.xlabel('Octave band center [Hz]')
            plt.ylabel('Energy difference [dB]')

            plt.grid(axis='y')

            plt.legend(*zip(*violin_labels), ncol=2,
                       handleheight=2)

            if backwards_integration:
                plt.suptitle(f'{room_aliases[short_name]} - backward-integrated energy diff')
            elif forwards_integration:
                plt.suptitle(f'{room_aliases[short_name]} - forward-integrated energy diff')
            else:
                plt.suptitle(f'{room_aliases[short_name]} - short-time-average energy diff')
            plt.show()

        all_violin_data[short_name] = combined_data
    
    if 'Freq-wise violin plot' in shown_plots:
        num_rooms = len(all_violin_data)

        comparison_keys = list()
        for violin_data in all_violin_data.values():
            if len(violin_data) > len(comparison_keys):
                comparison_keys = list(violin_data.keys())
        num_comparisons = len(comparison_keys)

        group_centers = np.arange(num_rooms)
        # https://stackoverflow.com/a/11603806
        group_margin = 0.1
        mid_margin = 0.02
        width = (1 - 2*group_margin) / num_comparisons

        for band_idx in range(num_bands):
            fig, ax = plt.subplots(dpi=100, figsize=(0.75*16, 0.75*9))

            # Reset legend labels.
            violin_labels = list()

            # Reference horizontal line at 0.
            line = plt.hlines(0, -1, num_bands+1,
                              color='black', ls='--',
                              linewidth=1)
            
            for k, comparison_key in enumerate(comparison_keys):
                positions = group_centers - 0.5 + group_margin + (k+0.5)*width
                
                violin_keys = list(room_aliases.keys())
                violin_data = [(all_violin_data[short_name][comparison_key][band_idx]
                                if comparison_key in all_violin_data[short_name]
                                else np.zeros(1))
                               for short_name in violin_keys]

                violin = ax.violinplot([remove_outliers(x, outlier_constant)
                                        for x in violin_data],
                                        positions=positions,
                                        widths=width-mid_margin,
                                        side='both',
                                        points=100,
                                        bw_method=0.2,
                                        # quantiles=[[0.05, 0.5, 0.95]]*num_bands,
                                        showextrema=False,
                                        showmeans=True,
                                        showmedians=False)

                for i, body in enumerate(violin['bodies']):
                    if 'Genelec' in comparison_key and 'simplified' in violin_keys[i]:
                        continue

                    violin_contour = body.get_paths()
                    assert len(body.get_paths()) == 1
                    violin_contour = violin_contour[0].vertices.copy()

                    violin_contour[:, 0] -= positions[i]

                    positive = (violin_contour[:, 0] > 0)
                    pos_violin_contour = violin_contour[positive]
                    sorting = np.argsort(pos_violin_contour[:, 1])
                    pos_violin_contour = pos_violin_contour[sorting][::-1]

                    with open(os.path.join(output_folder, f'{tick_label_func(band_idx)}Hz_{comparison_key.replace(' ','_')}_{violin_keys[i]}-pos.dat'),
                              mode='w') as file:
                        for x, y in pos_violin_contour:
                            file.write(f'{x} {y}\n')

                    negative = (violin_contour[:, 0] <= 0)
                    neg_violin_contour = violin_contour[negative]
                    sorting = np.argsort(neg_violin_contour[:, 1])
                    neg_violin_contour = neg_violin_contour[sorting]

                    with open(os.path.join(output_folder, f'{tick_label_func(band_idx)}Hz_{comparison_key.replace(' ','_')}_{violin_keys[i]}-neg.dat'),
                              mode='w') as file:
                        for x, y in neg_violin_contour:
                            file.write(f'{x} {y}\n')

                for pc in violin['bodies']:
                    pc.set_edgecolor(pc.get_facecolor())
                    pc.set_alpha(0.5)

                add_violin_label(violin, (comparison_key
                                          if 'Genelec' in comparison_key or 'Dodecahedron' in comparison_key else
                                          strategy_aliases[comparison_key]),
                                 violin_labels)
            
            if backwards_integration:
                plt.ylim(-15, 15)
            else:
                plt.ylim(-20, 20)
            plt.xlim(-0.5, num_rooms-0.5)

            ax.set_xticks(group_centers, room_aliases.values(),
                          rotation=30, ha='right')

            plt.xlabel('Room')
            plt.ylabel('Energy difference [dB]')

            plt.grid(axis='y')

            plt.legend(*zip(*violin_labels), ncol=2,
                       handleheight=2)

            if backwards_integration:
                plt.suptitle(f'{tick_label_func(band_idx)}Hz band - backward-integrated energy diff')
            elif forwards_integration:
                plt.suptitle(f'{tick_label_func(band_idx)}Hz band - forward-integrated energy diff')
            else:
                plt.suptitle(f'{tick_label_func(band_idx)}Hz band - short-time-average energy diff')
            plt.show()
