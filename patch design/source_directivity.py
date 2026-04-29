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

    # Circular colormap for phase values.
    phase_contour_levels = np.linspace(-np.pi, np.pi, 180)
    phase_cmap = plt.get_cmap('twilight', len(phase_contour_levels) + 1)
    phase_norm = mpl.colors.BoundaryNorm(phase_contour_levels, ncolors=phase_cmap.N)

    for ls_type, ls_path in ls_type_paths.items():
        if 'omnidirectional' in ls_path:
            continue

        file_path = os.path.join(root_folder, ls_path)
        with open(file_path, mode='r', newline='') as csvfile:
            reader = csv.reader(csvfile, skipinitialspace=True)

            header = next(reader)
            frequencies = np.array([float(f) for f in header[1:-1]])

            responses = np.zeros((360, 181, len(frequencies)),
                                dtype=complex)

            for row in reader:
                match = re.match(r'P(\d+)T(\d+)', row[0])
                if match is None:
                    print('Bad angle:', row[0])
                    break

                phi = int(match.group(1))
                theta = int(match.group(2))

                for f_i in range(len(frequencies)):
                    v = row[f_i + 1]
                    v = v.replace(' ', '')
                    v = v.replace('i', 'j')
                    responses[phi, theta, f_i] = complex(v)

            # Sequential colormap for amplitude values.
            max_amp_dB = 10 * np.log10(np.max(np.abs(responses[np.isfinite(responses)])))
            max_amp_dB = np.ceil(max_amp_dB)
            amp_contour_levels = np.linspace(max_amp_dB-20, max_amp_dB, 11)
            amp_cmap = plt.get_cmap('magma', len(amp_contour_levels))
            # Set "bad" values (0, inf, nan) to neon green.
            amp_cmap.set_bad('lime', 1.)
            amp_norm = mpl.colors.BoundaryNorm(amp_contour_levels, ncolors=amp_cmap.N)

            fig, axes = plt.subplots(len(shown_freqs), 2, figsize=(6, 3*len(shown_freqs)),
                                    subplot_kw={'projection': 'polar'},
                                    # squeeze=False, constrained_layout=True,
                                    layout='constrained'
                                    )
            
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
                                                    10 * np.log10(np.abs(responses[:, :, f_i])).T,
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
                
            plt.title(f'Loudspeaker: {ls_type}.')
            plt.show()
