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

    plot_complex_response = False
    phased_dodecahedron_sum = True

    responses_per_ls = dict()

    frequencies = None
    for ls_type, ls_path in ls_type_paths.items():
        file_path = os.path.join(root_folder, ls_path)
        with open(file_path, mode='r', newline='') as csvfile:
            reader = csv.reader(csvfile, skipinitialspace=True)

            header = next(reader)
            if frequencies is None:
                frequencies = np.array([float(f)
                                        for f in header[1:]
                                        if f!= ''])
            else:
                assert np.allclose(frequencies,
                                   np.array([float(f)
                                             for f in header[1:]
                                             if f!= '']))

            responses = np.zeros((360, 181, len(frequencies)),
                                 dtype=complex)

            for row in reader:
                match = re.match(r'P(\d+)T(\d+)', row[0])
                if match is None:
                    if row[0] == 'f in Hz':
                        assert np.allclose(frequencies, np.array([float(f)
                                                                  for f in row[1:]
                                                                  if f!= '']))
                        continue
                    else:
                        raise ValueError(f'Bad format: {row[0]}')

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

        responses_per_ls[ls_type] = responses

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
    
    if phased_dodecahedron_sum:
        # Consider the full dodecahedron response as a complex pressure sum.
        responses_per_ls['Dodecahedron (sum)'] = np.sum([r for k, r in responses_per_ls.items()
                                                        if 'Dodecahedron' in k],
                                                        axis=0)
    else:
        # Consider the full dodecahedron response as a pressure amplitude sum.
        responses_per_ls['Dodecahedron (sum)'] = np.sum([np.abs(r) for k, r in responses_per_ls.items()
                                                        if 'Dodecahedron' in k],
                                                        axis=0)
    
    # "The output chain was calibrated to a free field sound pressure of 80dB at 1kHz
    #   and a distance of 2m in front of the loudspeaker, i.e., Φ=Θ=0◦."
    # What does this mean for the dodecahedron?
    # If each component is calibrated at 1kHz separately, they will be all over the place.
    # If all components are calibrated together from a single measurement, the microphone
    #  will not be 2m from each component.
    # Here, for simplicity, the [0, 0, 1kHz] pressure amplitude is normalized to 1.
    # We'll worry about the pressure at 2m later.
    # All dodecahedron components are normalized w.r.t. their (complex pressure) sum.
    ref_i = int(np.flatnonzero(frequencies == 1e3)[0])
    gene_reference_pressure = responses_per_ls['Genelec'][0, 0, ref_i]
    dode_reference_pressure = responses_per_ls['Dodecahedron (sum)'][0, 0, ref_i]
    responses_per_ls = {k: (r / np.abs(dode_reference_pressure)
                            if 'Dodecahedron' in k
                            else r / np.abs(gene_reference_pressure))
                        for k, r in responses_per_ls.items()}

    energy_per_ls = dict()

    for ls_type, responses in responses_per_ls.items():
        energy_responses = np.abs(responses) ** 2
        # Since the angular sampling of both phi and theta is uniform,
        #  the samples need to be normalized by the (co)sine of theta before integration.
        energy_responses *= np.sin(np.linspace(0, np.pi, 181))[None, :, None]
        # Take the mean rather than the sum, to normalize for the number of measurements.
        total_energy = energy_responses.mean(axis=(0, 1))

        energy_per_ls[ls_type] = total_energy

    fig, axes = plt.subplots(squeeze=False, constrained_layout=True)
    
    for ls_type, energy in energy_per_ls.items():
        plt.plot(frequencies, energy,
                 label=ls_type)
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(4e1, 2e4)
    plt.ylim(7e-2, 2e0)
        
    plt.title('Total radiated energy of loudspeaker types.')
    plt.legend()
    plt.show()
