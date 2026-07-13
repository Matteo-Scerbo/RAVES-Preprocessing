"""
The website ImageToStl.com was used to convert the SketchUp files provided by BRAS into OBJ/MTL meshes.
This script reformats the resulting OBJ/MTL meshes to match the requirements of the MoD-ART package.
Outputs of this script are the "largest patches possible" and "naive triangulation" segmentations.
"""
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt

from raves.src.utils import load_mesh, visualize_mesh

material_names = {'CR1_DoorAngle1': {'mat0': 'tablesEquipment',
                                     'mat1': 'paintedConcrete',
                                     'mat2': 'concrete',
                                     'mat3': 'concrete',
                                     'mat4': 'absorber',
                                     'mat5': 'unknown',
                                     },
                  'CR1_DoorAngle3': {'mat0': 'paintedConcrete',
                                     'mat1': 'concrete',
                                     'mat2': 'tablesEquipment',
                                     'mat3': 'absorber',
                                     'mat4': 'unknown',
                                     'mat5': 'unknown',
                                     },
                  'CR1_simplified': {'mat0': 'paintedConcrete',
                                     'mat1': 'paintedConcrete',
                                     'mat2': 'concrete',
                                     'mat3': 'tablesEquipment',
                                     'mat4': 'absorber',
                                     'mat5': 'paintedConcrete',
                                     },
                  'CR2': {'mat0': 'concrete',
                          'mat1': 'windows',
                          'mat2': 'ceiling',
                          'mat3': 'plaster',
                          'mat4': 'floor',
                          'mat5': 'unknown',
                          },
                  'CR2_simplified': {'mat0': 'concrete',
                                     'mat1': 'windows',
                                     'mat2': 'plaster',
                                     'mat3': 'floor',
                                     'mat4': 'ceiling',
                                     'mat5': 'unknown',
                                     },
                  'CR2_ubersimplified': {'mat0': 'concrete',
                                         'mat1': 'windows',
                                         'mat2': 'floor',
                                         'mat3': 'plaster',
                                         'mat4': 'ceiling',
                                         'mat5': 'unknown',
                                         },
                  'CR3': {'mat0': 'plaster',
                          'mat1': 'stagePanels',
                          'mat2': 'structuredPlaster',
                          'mat3': 'floor',
                          'mat4': 'ceiling',
                          'mat5': 'windows',
                          'mat6': 'seating',
                          'mat7': 'unknown',
                          },
                  'CR4': {'mat0': 'concrete',
                          'mat1': 'linoleum',
                          'mat2': 'woodPanels',
                          'mat3': 'parquet',
                          'mat4': 'seating',
                          'mat5': 'whitePanels',
                          'mat6': 'brickwall',
                          'mat7': 'windows',
                          'mat8': 'unknown',
                          },
                  }

inches_per_meter = 39.3700787402

def reformat_mesh(input_path: str, output_path: str,
                  strategy: str,
                  area_threshold: float = 0,
                  verbose: bool = True):
    """
    The website ImageToStl.com can convert SketchUp files into OBJ/MTL meshes.
    This function is meant to take the resulting files and re-format them to
    match the input format of MoD-ART. Triangles with area below the threshold
    are removed. All vertex coordinates are converted from inches to meters.
    
    SketchUp items are grouped into objects, like this:
        o obj1
        v 19.88647461 -1.65458953 86.22047424
        v 22.6277256 -1.45809424 1.96850395
        v 19.88647461 -1.65458953 1.96850395
        v 22.83167267 -1.44347501 86.22047424
        v 22.83167267 -1.44347501 1.96850395
        usemtl mat0
        f 1 2 3
        f 2 4 5
        f 4 2 1
        # Vertices: 5, normals: 0, texture coordinates: 0, faces: 3
    Surface patches for ART are defined based on such objects.
    They may not be coplanar.
    
    With strategy 'naive_obj', the object above would be translated into
        # obj1
        v 0.505 -0.042 2.19
        v 0.575 -0.037 0.05
        v 0.505 -0.042 0.05
        v 0.58 -0.037 2.19
        v 0.58 -0.037 0.05
        usemtl Patch_1_Mat_CR1_tablesEquipment
        f 1 2 3
        f 2 4 5
        f 4 2 1
    With strategy 'naive_trng', the object above would be translated into
        # obj1
        v 0.505 -0.042 2.19
        v 0.575 -0.037 0.05
        v 0.505 -0.042 0.05
        v 0.58 -0.037 2.19
        v 0.58 -0.037 0.05
        usemtl Patch_1_Mat_CR1_tablesEquipment
        f 1 2 3
        usemtl Patch_2_Mat_CR1_tablesEquipment
        f 2 4 5
        usemtl Patch_3_Mat_CR1_tablesEquipment
        f 4 2 1
    """
    
    assert strategy in ['naive_obj', 'naive_trng'], \
           'The remeshing strategy must be one of "naive_obj", "naive_trng".'

    mtl_file_name = None
    old_obj = 0
    old_obj_mat = ''

    output_lines = list()
    all_patch_names = list()

    # This flag prevents the creation of patches which do not contain any faces.
    current_patch_is_empty = True
    
    # Keep track of vertices; this is only used to check for "0 area" faces.
    vertex_list = list()
    # For debugging...
    areas = list()
    
    output_lines.append('mtllib mesh.mtl\n')
    
    with open(input_path,
              mode='r') as file:
        for line in file:
            # Separate the line into words.
            split_line = line.split()
    
            if len(split_line) == 0:
                # Ignore empty lines.
                continue
    
            if split_line[0] == 'mtllib':
                if mtl_file_name is None:
                    mtl_file_name = split_line[1]
                elif verbose:
                    print('\tMore than one material library specified!')
            
            elif split_line[0] == 'o':
                old_obj += 1
                
                output_lines.append('\n')
                output_lines.append(f'# {split_line[1]}\n')
                
            elif split_line[0] == 'usemtl':
                old_obj_mat = split_line[1]

                if strategy == 'naive_obj':
                    # Sometimes, by removing faces with small areas, all faces of an object are removed.
                    # When this happens, the empty object must be deleted to avoid skipping patch IDs.
                    if current_patch_is_empty and len(all_patch_names) > 0:
                        dropped_patch_name = all_patch_names[-1]
                        all_patch_names = all_patch_names[:-1]
                        output_lines.remove(f'usemtl {dropped_patch_name}\n')
                    
                    base_name = mtl_file_name[:-4]
                    if 'simplified' in base_name and 'CR1' in base_name:
                        base_name = 'CR1_simplified'

                    mat_name = mtl_file_name[:3] + '_' + material_names[base_name][old_obj_mat]
                    patch_name = f'Patch_{len(all_patch_names)+1}_Mat_{mat_name}'
                    all_patch_names.append(patch_name)
                    output_lines.append(f'usemtl {patch_name}\n')
                    current_patch_is_empty = True

            elif split_line[0] == 'v':
                if len(split_line) == 5:
                    if verbose:
                        print('\t`w` coordinates are ignored.')
                    split_line = split_line[:-1]

                if len(split_line) != 4:
                    raise ValueError('All vertex coordinates must have three dimensions.'
                                     + ' Bad line:\n\t' + line)

                # Convert from inches to meters (blame SketchUp).
                coords = [float(c) / inches_per_meter
                          for c in split_line[1:]]
                # Round to fewer digits. Otherwise, some of the following operations
                #  on the floats will not match what actually gets written in the files.
                coords = [np.round(c, 8) for c in coords]
                converted_line = 'v ' + ' '.join([str(c) for c in coords]) + '\n'
                converted_line = f'v {coords[0]:.10f} {coords[1]:.10f} {coords[2]:.10f}\n'

                vertex_list.append(coords)
                output_lines.append(converted_line)
                
            elif split_line[0] == 'f':
                # Check the face's area and ignore if zero.
                triangle = np.array([vertex_list[int(c)-1]
                                     for c in split_line[1:]])
                area = np.linalg.norm(np.cross(triangle[1] - triangle[0],
                                               triangle[2] - triangle[0]))
                if area <= area_threshold:
                    if verbose:
                        print(f'\tIgnoring triangle with area {area}.')
                    continue
                areas.append(area)
                
                if strategy == 'naive_trng':
                    base_name = mtl_file_name[:-4]
                    if 'simplified' in base_name and 'CR1' in base_name:
                        base_name = 'CR1_simplified'
                    
                    mat_name = mtl_file_name[:3] + '_' + material_names[base_name][old_obj_mat]
                    patch_name = f'Patch_{len(all_patch_names)+1}_Mat_{mat_name}'
                    all_patch_names.append(patch_name)
                    output_lines.append(f'usemtl {patch_name}\n')

                output_lines.append(line)
                current_patch_is_empty = False
    
    with open(output_path,
              mode='w') as file:
        for line in output_lines:
            file.write(line)
    
    if mtl_file_name is not None:
        old_obj_mat = ''
        output_lines = list()
        
        with open(os.path.join(os.path.dirname(input_path),
                               mtl_file_name),
                  mode='r') as file:
            file_iterator = iter(file)
            for line in file_iterator:
                # Separate the line into words.
                split_line = line.split()
        
                if len(split_line) == 0:
                    # Ignore empty lines.
                    continue
        
                if split_line[0] == 'newmtl':
                    # Beginning of a material definition.
                    old_obj_mat = split_line[1]
                    base_name = mtl_file_name[:-4]
                    if 'simplified' in base_name and 'CR1' in base_name:
                        base_name = 'CR1_simplified'
                    
                    mat_name = mtl_file_name[:3] + '_' + material_names[base_name][old_obj_mat]
                    
                    # Store all visual parameters of the old definition.
                    old_parameters = list()
                    next_line = next(file_iterator)
                    while next_line != '\n':
                        old_parameters.append(next_line)
                        next_line = next(file_iterator)
                    
                    for patch_name in all_patch_names:
                        if mat_name in patch_name:
                            output_lines.append(f'newmtl {patch_name}\n')
                            for param in old_parameters:
                                output_lines.append(param)
                            output_lines.append('\n')
        
        with open(output_path.replace('.obj', '.mtl'),
                  mode='w') as file:
            for line in output_lines[:-1]:
                file.write(line)


# https://stackoverflow.com/a/54529366
def plot_loghist(ax, data, bins):
    hist, bins = np.histogram(data, bins=bins)
    logbins = np.logspace(np.log10(bins[0]),
                          np.log10(bins[-1]),
                          len(bins))
    ax.hist(data, bins=logbins)
    ax.set_xscale('log')


root_folder = os.path.join('..', 'BRAS meshes')
# materials_file = 'materials_all_bands.csv'
materials_file = 'materials_oct_bands.csv'

for root, dirs, files in os.walk(root_folder):
    for old_file in files:
        if old_file == 'mesh.obj':
            # Assume it's the output of a previous reformatting.
            continue

        if old_file[-4:] == '.obj':
            for remeshing_strategy in ['naive_obj', 'naive_trng']:
                new_name = old_file[:-4] + '_' + remeshing_strategy
                new_dir = os.path.join(root, new_name)
                os.makedirs(new_dir, exist_ok=True)

                print('Preparing mesh', new_dir)
                
                reformat_mesh(os.path.join(root, old_file),
                              os.path.join(new_dir, 'mesh.obj'),
                              remeshing_strategy)

                shutil.copy(os.path.join(root_folder, materials_file),
                            os.path.join(new_dir, 'materials.csv'))
                
                visualize_mesh(new_dir)
                
                mesh, patch_materials, _ = load_mesh(new_dir,
                                                     assert_coplanarity=False)
                
                print(new_name, 'has',
                      mesh.size(count_patches=True), 'patches,',
                      mesh.size(count_patches=False), 'triangles.')
                
                for mat, count in zip(*np.unique(patch_materials, return_counts=True)):
                    print('\t', count, 'out of', mesh.size(count_patches=True),
                          'patches have material', mat)
