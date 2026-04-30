import os
import re
import csv
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


if __name__ == '__main__':
    root_folder = os.path.join('..', '..', '..', 'BRAS', '2 Source and receiver descriptions')
    ls_type_paths = {'Genelec': os.path.join('Genelec 8020c', 'Genelec8020_1x1_64442_MPS_front_pole.csv'),
                     'Dodecahedron (high)': os.path.join('ITA dodecahedron', 'Dode_High_1x1_64442_MPS_front_pole.csv'),
                     'Dodecahedron (mid)': os.path.join('ITA dodecahedron', 'Dode_Mid_1x1_64442_MPS_front_pole.csv'),
                     'Dodecahedron (low)': os.path.join('ITA dodecahedron', 'Dode_Low_MPS_front_pole_omnidirectional.csv'),
                     }
    
    output_folder = os.path.join('..', 'BRAS meshes', 'Source_normalization')

    octave_bands = [125., 250., 500., 1000., 2000., 4000., 8000., 16000.]

    plot_complex_response = True
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
    
    # "The output chain was calibrated to a free field sound pressure of 80dB
    #   at 1kHz and a distance of 2m in front of the loudspeaker, i.e., Φ=Θ=0◦.
    #   Consequently, the RIR unit is Pascal."
    # What does this mean for the dodecahedron?
    # Components cannot be calibrated at 1kHz separately, it's only in one of their flat ranges.
    # If all components are calibrated together from a single measurement, the microphone
    #  will not be 2m from each component. Phase efects between the components will
    #  complicate the overlap regions.
    # Here, we calibrate the amplitude of the complex pressure sum of all components,
    #  under the assumption that all components are at 2m from the microphone.
    # For now, let's calibrate the directional responses to have unit pressure amplitude
    #  at [0, 0, 1kHz]. We will take the "80dB at 2m" calibration into account later.
    ref_i = int(np.flatnonzero(frequencies == 1e3)[0])
    gene_reference_pressure = responses_per_ls['Genelec'][0, 0, ref_i]
    # All dodecahedron components are normalized w.r.t. their (complex pressure) sum.
    dode_reference_pressure = responses_per_ls['Dodecahedron (sum)'][0, 0, ref_i]

    responses_per_ls = {k: (r / np.abs(dode_reference_pressure)
                            if 'Dodecahedron' in k
                            else r / np.abs(gene_reference_pressure))
                        for k, r in responses_per_ls.items()}

    if plot_complex_response:
        # Sequential colormap for amplitude values.
        amp_contour_levels = np.linspace(-24, 6, 11)
        amp_cmap = plt.get_cmap('magma', len(amp_contour_levels)-1)
        amp_norm = mpl.colors.BoundaryNorm(amp_contour_levels, ncolors=amp_cmap.N)

        # Circular colormap for phase values.
        phase_contour_levels = np.linspace(-np.pi, np.pi, 17)
        phase_cmap = plt.get_cmap('twilight', len(phase_contour_levels)-1)
        phase_norm = mpl.colors.BoundaryNorm(phase_contour_levels, ncolors=phase_cmap.N)

        for ls_type, responses in responses_per_ls.items():
            # Nudge true-zero values to avoid warnings from the logarithms.
            clipped_responses = responses.copy()
            clipped_responses[clipped_responses == 0] = 1e-30

            fig, axes = plt.subplots(len(octave_bands), 2, figsize=(8, 4*len(octave_bands)),
                                     subplot_kw={'projection': 'polar'},
                                     squeeze=False, constrained_layout=True)
            
            plot_i = 0
            amp_cs = None
            phase_cs = None
            for f_i, freq in enumerate(frequencies):
                if freq not in octave_bands:
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
                                                    20 * np.log10(np.abs(clipped_responses[:, :, f_i])).T,
                                                    norm=amp_norm, cmap=amp_cmap)
                axes[plot_i, 0].set_title(f'Amplitude ({freq:.0f}Hz)')

                phase_cs = axes[plot_i, 1].pcolormesh(np.linspace(0, 2*np.pi, 360),
                                                      np.linspace(0, 180, 181),
                                                      np.angle(clipped_responses[:, :, f_i]).T,
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
                
            plt.suptitle(f'Directional response of loudspeaker: {ls_type}.')
            plt.show()
    
    energy_per_ls = dict()

    for ls_type, responses in responses_per_ls.items():
        energy_responses = np.abs(responses) ** 2
        # Since the angular sampling of both phi and theta is uniform,
        #  the samples need to be normalized by the (co)sine of theta before integration.
        energy_responses *= np.sin(np.linspace(0, np.pi, 181))[None, :, None]
        # Take the mean rather than the sum, to normalize for the number of measurements.
        total_energy = energy_responses.mean(axis=(0, 1))

        energy_per_ls[ls_type] = total_energy

    fig, ax = plt.subplots(dpi=200, figsize=(9, 6))

    for ls_type, energy in energy_per_ls.items():
        plt.plot(frequencies, 10 * np.log10(energy),
                 label=ls_type,
                 marker=('+' if 'sum' in ls_type else None),
                 linestyle=('--' if 'sum' in ls_type else '-'))
    
    plt.xscale('log')
    plt.xlim(4e1, 2e4)
    plt.xlabel('Frequency [Hz]')

    plt.ylim(-18, 6)
    plt.ylabel('Total radiated energy [dB]')

    handles, labels = ax.get_legend_handles_labels()
    handles.append(handles.pop(0))
    labels.append(labels.pop(0))
    plt.legend(handles, labels, ncol=2)
    
    plt.title('Total radiated energy of loudspeaker types.')
    plt.show()

    def tick_label_func(val, pos=None):
        if val >= len(octave_bands):
            return 'error'
        elif octave_bands[int(val)] < 1e3:
            return f'{int(octave_bands[int(val)])}'
        else:
            return f'{int(octave_bands[int(val)] / 1e3)}k'

    band_energy_per_ls = dict()
    # third_octave_indices = (frequencies >= octave_bands[0] / np.sqrt(2)) & (frequencies <= octave_bands[0] * np.sqrt(2))
    for ls_type, energy in energy_per_ls.items():
        band_energy_per_ls[ls_type] = np.zeros(len(octave_bands))

        for oct_i, oct_f in enumerate(octave_bands):
            thirds = (frequencies >= oct_f / np.sqrt(2)) & (frequencies <= oct_f * np.sqrt(2))
            assert np.count_nonzero(thirds) == 3
            band_energy_per_ls[ls_type][oct_i] = np.mean(energy[thirds])

    group_centers = np.arange(len(octave_bands))
    # https://stackoverflow.com/a/11603806
    margin = 0.2
    width = (1 - 2*margin) / 2

    fig, ax = plt.subplots(dpi=200, figsize=(9, 6))
    
    for k, ls_type in enumerate(['Dodecahedron (sum)', 'Genelec']):
        energy = band_energy_per_ls[ls_type]
        positions = group_centers - 0.5 + margin + (k+0.5)*width
        plt.bar(positions, 10 * np.log10(energy),
                width, label=ls_type)

    plt.xlim(-0.5, len(octave_bands)-0.5)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(tick_label_func))
    plt.xlabel('Octave band center [Hz]')
    
    plt.ylim(-12, 0)
    plt.ylabel('Total radiated energy [dB]')

    plt.legend()
    plt.grid(axis='y')
    
    plt.title('Total radiated energy of loudspeaker types.')
    plt.show()

    # A pressure of 80dB (10^4) at 2m means a pressure of ~86dB (2*10^4) at 1m.
    # We need to calibrate the room impulse responses with respect to that.
    # The factors we are about to save will be used to normalize the recordings'
    #  energy in each octave band, so we need to divide them by the reference.
    # TODO: In theory, the normalization should be reference_level^2,
    #        but this works without the square. Why?
    reference_level = 2e4
    band_energy_per_ls = {k: e / reference_level
                          for k, e in band_energy_per_ls.items()}

    with open(os.path.join(output_folder, 'Genelec.csv'), mode='w') as new_file:
        writer = csv.writer(new_file)
        writer.writerow(band_energy_per_ls['Genelec'])
    with open(os.path.join(output_folder, 'Dodecahedron.csv'), mode='w') as new_file:
        writer = csv.writer(new_file)
        writer.writerow(band_energy_per_ls['Dodecahedron (sum)'])
