import os
import csv
import itertools
import numpy as np
from scipy.io import mmread
import shapely
import shapely.plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches

from cycler import cycler
from okabeito import lightblue, yellow, orange, green, purple, red, blue, black

default_cycler = (cycler(color=[lightblue, yellow, orange, green, black, purple, red, blue]))
plt.rc('axes', prop_cycle=default_cycler)

from collections import defaultdict

from raves.src.utils import load_mesh, load_mesh_as_arrays, visualize_mesh


# https://stackoverflow.com/a/26370192
def project_3D_to_2D(points_3D):
    assert len(points_3D.shape) == 2
    assert points_3D.shape[0] >= 3
    assert points_3D.shape[1] == 3

    num_points = points_3D.shape[0]

    local_origin = points_3D[0]

    local_x = points_3D[1] - local_origin
    local_x /= np.linalg.norm(local_x)

    normal = np.cross(local_x,
                      points_3D[2] - local_origin)
    if np.linalg.norm(normal) == 0:
        raise ValueError('All points in the first triangle are colinear.')
    normal /= np.linalg.norm(normal)
    
    local_y = np.cross(normal, local_x)
    local_y /= np.linalg.norm(local_y)

    for p in points_3D:
        z_error = np.dot(p - local_origin, normal)
        if np.abs(z_error) > 1e-2:
            # raise ValueError(f'Not all points are coplanar. Z error: {z_error}')
            # print(f'Not all points are coplanar. Z error: {z_error}')
            continue

    points_2D = [(np.dot(p - local_origin, local_x),
                  np.dot(p - local_origin, local_y))
                 for p in points_3D]

    return np.array(points_2D), (local_origin, local_x, local_y)


def assess_patch(all_vertices, all_faces, patch_triangle_idxs):
    # Select only the vertices which form the patch.
    points_3D = all_vertices[all_faces[patch_triangle_idxs]]

    # Translate to 2D.
    coords_3D = points_3D.reshape((-1, 3))
    # https://stackoverflow.com/a/13531310
    try:
        coords_2D, basis = project_3D_to_2D(coords_3D)
    except ValueError as e:
        if str(e) != 'All points are colinear.':
            raise
        else:
            # If the patch had zero area, return nothing.
            return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
    points_2D = coords_2D.reshape((points_3D.shape[0], 3, 2))

    # Create shapely triangles.
    triangles_2D = [shapely.Polygon(tri)
                    for tri in points_2D]

    # Merge shapely triangles into polygon (may have holes).
    polygon_2D = shapely.union_all(triangles_2D)

    # The polygon will be split if its area is too large...
    area = polygon_2D.area
    # ... or if it is very long and narrow.
    # "Narrowness" is evaluated through the isoperimetric quotient,
    #  defined as the ratio of the polygon's area and the area of
    #  the circle having the same perimeter as the polygon.
    perimeter = polygon_2D.length
    ipq = 4 * np.pi * area / perimeter**2

    # shapely.plotting.plot_polygon(polygon_2D)
    # plt.title(str(ipq))
    # plt.show()

    return area, ipq


# https://stackoverflow.com/a/54529366
def plot_loghist(ax, data, bins):
    hist, bins = np.histogram(data, bins=bins)
    logbins = np.logspace(np.log10(bins[0]),
                          np.log10(bins[-1]),
                          len(bins))
    ax.hist(data, bins=logbins)
    ax.set_xscale('log')


# https://stackoverflow.com/a/58324984
def add_violin_label(violin, label, label_list):
    color = violin["bodies"][0].get_facecolor().flatten()
    label_list.append((mpatches.Patch(color=color), label))


def recolor_mesh(folder_path):
    file_path = os.path.join(folder_path, 'mesh.mtl')

    output_lines = list()
    
    with open(file_path, mode='r') as file:
        for line in file:
            if line.startswith('Kd '):
                rand_color = np.random.uniform(size=3)
                output_lines.append(f'Kd {rand_color[0]} {rand_color[1]} {rand_color[2]}')
            else:
                output_lines.append(line)

    with open(file_path, mode='w') as file:
        for line in output_lines:
            file.write(line)


if __name__ == '__main__':
    mesh_folder = os.path.join('..', 'BRAS meshes')
    output_folder = os.path.join('.', 'data_for_figures')

    full_room_names = {'CR1_DoorAngle1': 'CR1 coupled rooms (laboratory and reverberation chamber)',
                       'CR1_DoorAngle1_simplified': 'CR1 coupled rooms (laboratory and reverberation chamber)',
                       'CR1_DoorAngle1_ubersimplified': 'CR1 coupled rooms (laboratory and reverberation chamber)',
                       'CR1_DoorAngle3': 'CR1 coupled rooms (laboratory and reverberation chamber)',
                       'CR1_DoorAngle3_simplified': 'CR1 coupled rooms (laboratory and reverberation chamber)',
                       'CR1_DoorAngle3_ubersimplified': 'CR1 coupled rooms (laboratory and reverberation chamber)',
                       'CR2': 'CR2 small room (seminar room)',
                       'CR2_simplified': 'CR2 small room (seminar room)',
                       'CR2_ubersimplified': 'CR2 small room (seminar room)',
                       'CR3': 'CR3 medium room (chamber music hall)',
                    #    'CR4': 'CR4 large room (auditorium)',
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

    area_data = defaultdict(dict)
    ipq_data = defaultdict(dict)

    for short_name, full_name in full_room_names.items():
        base_name = short_name.replace('_simplified', '')
        base_name = base_name.replace('_ubersimplified', '')

        for mesh_strat, strat_alias, in strategy_aliases.items():
            combined_name = short_name + '_' + mesh_strat
            combined_dir = os.path.join(mesh_folder, short_name, combined_name)
            if not os.path.isdir(combined_dir):
                area_data[mesh_strat][short_name] = np.full(2, np.nan)
                ipq_data[mesh_strat][short_name] = np.full(2, np.nan)
                continue
            if not os.path.isfile(os.path.join(combined_dir, 'mesh.obj')):
                area_data[mesh_strat][short_name] = np.full(2, np.nan)
                ipq_data[mesh_strat][short_name] = np.full(2, np.nan)
                continue

            verts, faces, patch_ids, patch_materials = load_mesh_as_arrays(combined_dir)

            num_patches = len(patch_materials)
            patch_areas = np.zeros(num_patches)
            patch_IPQs = np.zeros(num_patches)

            for patch_i in range(len(patch_materials)):
                patch_tris = np.where(patch_ids == patch_i)
                area, ipq = assess_patch(verts, faces, patch_tris)
                patch_areas[patch_i] = area
                patch_IPQs[patch_i] = ipq
            
            # kernel = mmread(os.path.join(combined_dir, 'ART_kernel_band_1.mtx'), spmatrix=True)
            # num_paths = kernel.shape[0]
            # num_nonzero = kernel.nnz

            # print(combined_name, 'has', num_patches, 'patches,', num_paths, 'paths, and', num_nonzero, 'nonzeroes.')
            # print('Min-max area', np.min(patch_areas), np.max(patch_areas))
            # print('Min-max IPQ', np.min(patch_IPQs), np.max(patch_IPQs))

            # if 'split_area' in mesh_strat and np.max(patch_areas) > 4:
            #     print('Areas too large:', combined_name, mesh_strat, np.max(patch_areas))
            # if 'uber_split_area' in mesh_strat and np.max(patch_areas) > 2:
            #     print('Areas too large:', combined_name, mesh_strat, np.max(patch_areas))
            # if '_length' in mesh_strat and np.max(patch_IPQs) > 5:
            #     print('IPQs too large:', combined_name, mesh_strat, np.max(patch_IPQs))

            area_data[mesh_strat][short_name] = patch_areas
            ipq_data[mesh_strat][short_name] = patch_IPQs

            """
            fig, ax = plt.subplots(2, dpi=200, figsize=(8, 4))

            plot_loghist(ax[0], areas, bins=100)
            ax[0].set_title('Triangle areas')
            ax[0].set_yscale('log')
            ax[0].grid(True)

            # plot_loghist(ax[1], isoperimetric, bins=100)
            ax[1].hist(isoperimetric, bins=100)
            ax[1].set_title('Triangle isoperimetric quotients')
            ax[1].set_yscale('log')
            ax[1].grid(True)
            
            plt.suptitle('Mesh name: ' + combined_name)

            plt.tight_layout()
            plt.show()
            
            fig, ax = plt.subplots(dpi=200, figsize=(8, 4))

            plot_loghist(ax, patch_areas, bins=100)
            ax.set_title('Triangle areas')
            ax.set_yscale('log')
            ax.grid(True)

            plt.suptitle('Mesh name: ' + combined_name)

            plt.tight_layout()
            plt.show()
            """

            # print(combined_name)
            # image = visualize_mesh(combined_dir, interactive_window=False)

            # if image is not None:
            #     plt.figure()
            #     plt.imshow(image)
            #     plt.axis('off')
            #     plt.show()
    
    num_rooms = len(full_room_names)
    num_strats = len(strategy_aliases)
    
    group_centers = np.arange(num_rooms)
    # https://stackoverflow.com/a/11603806
    group_margin = 0.2
    mid_margin = 0.02
    width = (1 - 2*group_margin) / num_strats

    # Reset legend labels.
    legend_elements = list()

    fig, ax = plt.subplots(dpi=150, figsize=(2.0*4, 2.0*3))

    for k, (mesh_strat, areas_per_room) in enumerate(area_data.items()):
        positions = group_centers - 0.5 + group_margin + (k+0.5)*width
        violin_keys = list(areas_per_room.keys())
        violin = ax.violinplot([np.log10(areas_per_room[k])
                                for k in violin_keys],
                               positions=positions,
                               widths=width - mid_margin,
                               side='both',
                               points=100,
                               bw_method=0.2,
                               # quantiles=[[0.05, 0.5, 0.95]]*num_rooms,
                               showextrema=False,
                               showmeans=False,
                               showmedians=False)
        
        # Save the violin contour coordinates.
        # This is way more efficient than computing the violins in Tikz.

        assert len(violin['bodies']) == len(areas_per_room)

        for i, body in enumerate(violin['bodies']):
            violin_contour = body.get_paths()
            assert len(body.get_paths()) == 1
            violin_contour = violin_contour[0].vertices.copy()

            violin_contour[:, 0] -= positions[i]

            positive = (violin_contour[:, 0] > 0)
            pos_violin_contour = violin_contour[positive]
            sorting = np.argsort(pos_violin_contour[:, 1])
            pos_violin_contour = pos_violin_contour[sorting][::-1]

            with open(os.path.join(output_folder, 'patch_areas_' + mesh_strat + '_' + violin_keys[i] + '-pos.dat'),
                      mode='w') as file:
                for x, y in pos_violin_contour:
                    file.write(f'{x} {y}\n')

            negative = (violin_contour[:, 0] <= 0)
            neg_violin_contour = violin_contour[negative]
            sorting = np.argsort(neg_violin_contour[:, 1])
            neg_violin_contour = neg_violin_contour[sorting]

            with open(os.path.join(output_folder, 'patch_areas_' + mesh_strat + '_' + violin_keys[i] + '-neg.dat'),
                      mode='w') as file:
                for x, y in neg_violin_contour:
                    file.write(f'{x} {y}\n')

        for pc in violin['bodies']:
            pc.set_edgecolor(pc.get_facecolor())
            pc.set_alpha(0.5)

        add_violin_label(violin, strategy_aliases[mesh_strat], legend_elements)
        
    plt.xlim(-0.5, num_rooms-0.5)
    plt.xticks(group_centers, room_aliases.values(),
               rotation=30, ha='right', rotation_mode='anchor')

    plt.ylabel(r'Area [$\text{m}^2$]')
    # plt.yscale('log')
    # plt.ylim(1e-1, None)
    # https://stackoverflow.com/a/60132262
    y_min, y_max = ax.get_ylim()
    y_min = -2
    y_max = np.ceil(y_max)
    tick_range = np.arange(y_min, y_max+1)
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("$10^{{{x:.0f}}}$"))
    ax.yaxis.set_ticks(tick_range)
    ax.yaxis.set_ticks([np.log10(x)
                        for p in tick_range
                        for x in np.linspace(10**p, 10**(p+1), 10)],
                       minor=True)
    plt.ylim(y_min, y_max)

    plt.legend(*zip(*legend_elements), ncol=3,
               loc='upper center', handleheight=2)

    plt.title('Surface patch areas')
    plt.show()

    # Reset legend labels.
    legend_elements = list()

    fig, ax = plt.subplots(dpi=150, figsize=(2.0*4, 2.0*3))

    for k, (mesh_strat, ipqs_per_room) in enumerate(ipq_data.items()):
        positions = group_centers - 0.5 + group_margin + (k+0.5)*width
        violin_keys = list(ipqs_per_room.keys())
        violin = ax.violinplot([ipqs_per_room[k]
                                for k in violin_keys],
                               positions=positions,
                               widths=width - mid_margin,
                               side='both',
                               points=100,
                               bw_method=0.2,
                               # quantiles=[[0.05, 0.5, 0.95]]*num_rooms,
                               showextrema=False,
                               showmeans=False,
                               showmedians=False)
        
        # Save the violin contour coordinates.
        # This is way more efficient than computing the violins in Tikz.

        assert len(violin['bodies']) == len(ipqs_per_room)

        for i, body in enumerate(violin['bodies']):
            violin_contour = body.get_paths()
            assert len(body.get_paths()) == 1
            violin_contour = violin_contour[0].vertices.copy()

            violin_contour[:, 0] -= positions[i]

            positive = (violin_contour[:, 0] > 0)
            pos_violin_contour = violin_contour[positive]
            sorting = np.argsort(pos_violin_contour[:, 1])
            pos_violin_contour = pos_violin_contour[sorting][::-1]

            with open(os.path.join(output_folder, 'patch_IPQs_' + mesh_strat + '_' + violin_keys[i] + '-pos.dat'),
                      mode='w') as file:
                for x, y in pos_violin_contour:
                    file.write(f'{x} {y}\n')

            negative = (violin_contour[:, 0] <= 0)
            neg_violin_contour = violin_contour[negative]
            sorting = np.argsort(neg_violin_contour[:, 1])
            neg_violin_contour = neg_violin_contour[sorting]

            with open(os.path.join(output_folder, 'patch_IPQs_' + mesh_strat + '_' + violin_keys[i] + '-neg.dat'),
                      mode='w') as file:
                for x, y in neg_violin_contour:
                    file.write(f'{x} {y}\n')

        for pc in violin['bodies']:
            pc.set_edgecolor(pc.get_facecolor())
            pc.set_alpha(0.5)

        add_violin_label(violin, strategy_aliases[mesh_strat], legend_elements)
    
    plt.hlines(1, -1, num_rooms+1,
               color='black', ls='--',
               linewidth=1)
    # legend_elements.append(Line2D([0], [0], color='black', ls='--',
    #                               label='Optimal IPQ (circle)'))
    plt.annotate('Optimal IPQ (circle)', ((num_rooms-1)/2, 1),
                 ha='center', va='top')

    plt.xlim(-0.5, num_rooms-0.5)
    plt.xticks(group_centers, room_aliases.values(),
               rotation=30, ha='right', rotation_mode='anchor')

    plt.ylabel('Isoperimetric quotient')
    # plt.yscale('log')

    plt.legend(*zip(*legend_elements), ncol=3,
               loc='lower center', handleheight=2)

    plt.title('Surface patch compactness')
    plt.show()
