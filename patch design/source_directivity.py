import os
import re
import csv
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


if __name__ == '__main__':
    root_folder = os.path.join('..', '..', '..', 'BRAS', '2 Source and receiver descriptions')
    ls_type_paths = {'Genelec': os.path.join('Genelec 8020c', 'Genelec8020_1x1_64442_MPS_front_pole.csv'),
                     'Dodecahedron (high)': os.path.join('ITA dodecahedron', 'Dode_High_1x1_64442_MPS_front_pole.csv'),
                     'Dodecahedron (mid)': os.path.join('ITA dodecahedron', 'Dode_Mid_1x1_64442_MPS_front_pole.csv'),
                     'Dodecahedron (low)': os.path.join('ITA dodecahedron', 'Dode_Low_MPS_front_pole_omnidirectional.csv'),
                     }

    shown_freqs = [125., 250., 500., 1000., 2000., 4000., 8000., 16000.]

    plot_individual_energy = False
    plot_complex_response = False

    energy_per_ls = dict()

    for ls_type, ls_path in ls_type_paths.items():
        file_path = os.path.join(root_folder, ls_path)
        with open(file_path, mode='r', newline='') as csvfile:
            reader = csv.reader(csvfile, skipinitialspace=True)

            header = next(reader)
            frequencies = np.array([float(f)
                                    for f in header[1:]
                                    if f!= ''])

            responses = np.zeros((360, 181, len(frequencies)),
                                 dtype=complex)

            for row in reader:
                match = re.match(r'P(\d+)T(\d+)', row[0])
                if match is None:
                    if row[0] == 'f in Hz':
                        if np.allclose(frequencies, np.array([float(f)
                                                              for f in row[1:]
                                                              if f!= ''])):
                            continue
                        else:
                            print('Mismatched frequencies', row)
                            break
                    else:
                        print('Bad format:', row[0])
                        break

                phi = int(match.group(1))
                theta = int(match.group(2))

                for f_i in range(len(frequencies)):
                    v = row[f_i + 1]
                    v = v.replace(' ', '')
                    v = v.replace('i', 'j')
                    responses[phi, theta, f_i] = complex(v)
        
        if 'omnidirectional' in ls_path:
            # responses[:, :, :] = np.broadcast_to(responses[0, 0, None, None],
            #                                      responses.shape)
            responses[:, :, :] = responses[0, 0, None, None, :]

        # There are some NaN and infinite values in the Genelec responses.
        valid_samples = np.isfinite(responses)
        # for ax in range(valid_samples.ndim):
        #     num_valid = np.count_nonzero(np.all(valid_samples, axis=ax))
        #     max_valid = int(valid_samples.size / valid_samples.shape[ax])
        #     print(ls_type, f'Along axis {ax}:',
        #           f'{(100 * num_valid / max_valid):.2f}% valid',
        #           f'({num_valid} / {max_valid})')
        responses[~valid_samples] = 0

        energy_responses = np.abs(responses) ** 2
        # Since the angular sampling of both phi and theta is uniform,
        #  the samples need to be normalized by the (co)sine of theta before integration.
        energy_responses *= np.sin(np.linspace(0, np.pi, 181))[None, :, None]
        # Take the mean rather than the sum, to normalize for the number of measurements.
        total_energy = energy_responses.mean(axis=(0, 1))

        energy_per_ls[ls_type] = total_energy

        if plot_individual_energy:
            fig, axes = plt.subplots(squeeze=False, constrained_layout=True)
            
            plt.plot(frequencies, total_energy)
            
            plt.xscale('log')
            plt.yscale('log')

            low_lim = np.min(total_energy[frequencies > 50])
            low_lim = 10 ** (np.floor(np.log10(low_lim) * 10 - 1) / 10)
            high_lim = np.max(total_energy[frequencies > 50])
            high_lim = 10 ** (np.ceil(np.log10(high_lim) * 10 + 1) / 10)
            plt.ylim(low_lim, high_lim)
                
            plt.title(f'Total radiated energy of loudspeaker: {ls_type}.')
            plt.show()
        
        if plot_complex_response:
            # Sequential colormap for amplitude values.
            max_amp_dB = 20 * np.log10(np.max(np.abs(responses[np.isfinite(responses)])))
            max_amp_dB = np.ceil(max_amp_dB)
            amp_contour_levels = np.linspace(max_amp_dB-30, max_amp_dB, 11)
            amp_cmap = plt.get_cmap('magma', len(amp_contour_levels))
            # Set "bad" values (0, inf, nan) to neon green.
            amp_cmap.set_bad('lime', 1.)
            amp_norm = mpl.colors.BoundaryNorm(amp_contour_levels, ncolors=amp_cmap.N)

            # Circular colormap for phase values.
            phase_contour_levels = np.linspace(-np.pi, np.pi, 180)
            phase_cmap = plt.get_cmap('twilight', len(phase_contour_levels) + 1)
            phase_norm = mpl.colors.BoundaryNorm(phase_contour_levels, ncolors=phase_cmap.N)

            fig, axes = plt.subplots(len(shown_freqs), 2, figsize=(6, 3*len(shown_freqs)),
                                    subplot_kw={'projection': 'polar'},
                                    squeeze=False, constrained_layout=True)
            
            plot_i = 0
            amp_cs = None
            phase_cs = None
            for f_i, freq in enumerate(frequencies):
                if freq not in shown_freqs:
                    continue

                axes[plot_i, 0].set_rmax(180)
                axes[plot_i, 1].set_rmax(180)
                axes[plot_i, 0].set_rgrids([45, 90, 135], [])
                axes[plot_i, 1].set_rgrids([45, 90, 135], [])
                axes[plot_i, 0].set_theta_zero_location('N')
                axes[plot_i, 1].set_theta_zero_location('N')
                axes[plot_i, 0].grid(ls=':')
                axes[plot_i, 1].grid(ls=':')

                amp_cs = axes[plot_i, 0].pcolormesh(np.linspace(0, 2*np.pi, 360),
                                                    np.linspace(0, 180, 181),
                                                    20 * np.log10(np.abs(responses[:, :, f_i])).T,
                                                    norm=amp_norm, cmap=amp_cmap)
                axes[plot_i, 0].set_title(f'Amplitude ({freq:.0f}Hz)')

                phase_cs = axes[plot_i, 1].pcolormesh(np.linspace(0, 2*np.pi, 360),
                                                    np.linspace(0, 180, 181),
                                                    np.angle(responses[:, :, f_i]).T,
                                                    norm=phase_norm, cmap=phase_cmap)
                axes[plot_i, 1].set_title(f'Phase ({freq:.0f}Hz)')

                plot_i += 1

            amp_cbar = fig.colorbar(amp_cs, ax=axes[-1, 0],
                                    format='{x:.0f}dB',
                                    orientation='horizontal',
                                    location='bottom')
            phase_cbar = fig.colorbar(phase_cs, ax=axes[-1, 1],
                                    orientation='horizontal',
                                    location='bottom')

            phase_cbar.ax.set_xticks(np.linspace(-np.pi, np.pi, 5),
                                    [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'],
                                    minor=False)
            phase_cbar.ax.set_xticks([], [], minor=True)
                
            plt.title(f'Directional response of loudspeaker: {ls_type}.')
            plt.show()
    
    dode_energy_sum = np.sum([e for k, e in energy_per_ls.items()
                              if 'Dodecahedron' in k],
                             axis=0)

    fig, axes = plt.subplots(squeeze=False, constrained_layout=True)
    
    for ls_type, energy in energy_per_ls.items():
        plt.plot(frequencies, energy,
                 label=ls_type)
    
    plt.plot(frequencies, dode_energy_sum,
             label='Dodecahedron (sum)')
    
    plt.xscale('log')
    plt.yscale('log')
        
    plt.title('Total radiated energy of loudspeaker types.')
    plt.legend()
    plt.show()
