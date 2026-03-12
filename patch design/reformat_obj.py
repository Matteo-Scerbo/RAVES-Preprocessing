import os
import warnings
import numpy as np
import matplotlib.pyplot as plt

from raves.src.utils import load_mesh, visualize_mesh


def reformat_mesh(input_path: str, output_path: str,
                  strategy: str,
                  area_threshold: float = 1e-9):
    """
    The website ImageToStl.com can convert SketchUp files into OBJ/MTL meshes.
    This function is meant to take the resulting files and re-format them to
    match the input format of MoD-ART. Triangles with 0 area are removed.
    
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
        v 19.88647461 -1.65458953 86.22047424
        v 22.6277256 -1.45809424 1.96850395
        v 19.88647461 -1.65458953 1.96850395
        v 22.83167267 -1.44347501 86.22047424
        v 22.83167267 -1.44347501 1.96850395
        usemtl Patch_1_Mat_mat0
        f 1 2 3
        f 2 4 5
        f 4 2 1
    With strategy 'naive_trng', the object above would be translated into
        # obj1
        v 19.88647461 -1.65458953 86.22047424
        v 22.6277256 -1.45809424 1.96850395
        v 19.88647461 -1.65458953 1.96850395
        v 22.83167267 -1.44347501 86.22047424
        v 22.83167267 -1.44347501 1.96850395
        usemtl Patch_1_Mat_mat0
        f 1 2 3
        usemtl Patch_2_Mat_mat0
        f 2 4 5
        usemtl Patch_3_Mat_mat0
        f 4 2 1
    """
    
    assert strategy in ['naive_obj', 'naive_trng'], \
           'The remeshing strategy must be one of "naive_obj", "naive_trng".'

    mtl_file_name = None
    old_obj = 0
    old_mat = ''
    
    output_lines = list()
    patch_names = list()
    
    # Keep track of vertices; this is only used to check for "0 area" faces.
    vertex_list = list()
    
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
                else:
                    warnings.warn('More than one material library specified!')
            
            elif split_line[0] == 'o':
                old_obj += 1
                
                output_lines.append('\n')
                output_lines.append(f'# {split_line[1]}\n')
                
            elif split_line[0] == 'usemtl':
                old_mat = split_line[1]
                
                if strategy == 'naive_obj':
                    new_mat = f'Patch_{len(patch_names)+1}_Mat_{old_mat}'
                    patch_names.append(new_mat)
                    output_lines.append(f'usemtl {new_mat}\n')
    
            elif split_line[0] == 'v':
                if len(split_line) == 5:
                    warnings.warn('`w` coordinates are ignored.')
                    split_line = split_line[:-1]

                if len(split_line) != 4:
                    raise ValueError('All vertex coordinates must have three dimensions.'
                                     + ' Bad line:\n\t' + line)

                vertex_list.append([float(c) for c in split_line[1:]])
                
                output_lines.append(line)
                
            elif split_line[0] == 'f':
                # Check the face's area and ignore if zero.
                triangle = np.array([vertex_list[int(c)-1]
                                     for c in split_line[1:]])
                area = np.linalg.norm(np.cross(triangle[1] - triangle[0],
                                               triangle[2] - triangle[0]))
                if area < area_threshold:
                    warnings.warn(f'Ignoring triangle with area {area}.')
                    continue
                
                if strategy == 'naive_trng':
                    new_mat = f'Patch_{len(patch_names)+1}_Mat_{old_mat}'
                    patch_names.append(new_mat)
                    output_lines.append(f'usemtl {new_mat}\n')

                output_lines.append(line)
    
    with open(output_path,
              mode='w') as file:
        for line in output_lines:
            file.write(line)
    
    if mtl_file_name is not None:
        old_mat = ''
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
                    old_mat = split_line[1]
                    
                    # Store all visual parameters of the old definition.
                    old_parameters = list()
                    next_line = next(file_iterator)
                    while next_line != '\n':
                        old_parameters.append(next_line)
                        next_line = next(file_iterator)
                
                    for new_mat in patch_names:
                        if old_mat in new_mat:
                            output_lines.append(f'newmtl {new_mat}\n')
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
                
                reformat_mesh(os.path.join(root, old_file),
                              os.path.join(new_dir, 'mesh.obj'),
                              remeshing_strategy)
    
                mesh, patch_materials, _ = load_mesh(new_dir,
                                                     assert_coplanarity=False)
                
                print(new_name, 'has',
                      mesh.size(count_patches=True), 'patches,',
                      mesh.size(count_patches=False), 'triangles.')
                
                areas = mesh.area
                perimeters = mesh.perimeter()
                isoperimetric = 4 * np.pi * areas / perimeters**2
                
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
                
                plt.suptitle('Mesh name: ' + new_name)
            
                plt.tight_layout()
                plt.show()
                
                break
                
                # visualize_mesh(new_dir)
