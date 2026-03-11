"""
The website ImageToStl.com can convert SketchUp files into OBJ/MTL meshes.
This code is meant to take the resulting files and re-format them to match the
input format of MoD-ART. SketchUp items are grouped into objects, like this:
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
The object above would be translated into
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
"""

import os

from raves.src.utils import load_mesh, visualize_mesh

root_folder = os.path.join('..', '..', 'BRAS', 'OBJ files')

for root, dirs, files in os.walk(root_folder):
    for old_file in files:
        if old_file == 'mesh.obj':
            # Assume it's the output of a previous reformatting.
            continue
        
        if old_file[-4:] == '.obj':
            mtl_file_name = None
            old_obj = 0
            old_mat = ''
            
            output_lines = list()
            new_mat_names = list()
            
            output_lines.append('mtllib mesh.mtl\n')
            
            with open(os.path.join(root, old_file), mode='r') as file:
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
                            print('More than one material library specified!')
                    
                    elif split_line[0] == 'o':
                        old_obj += 1
                        
                        output_lines.append('\n')
                        output_lines.append(f'# {split_line[1]}\n')
                        
                    elif split_line[0] == 'usemtl':
                        old_mat = split_line[1]
                        
                        new_mat_names.append(f'Patch_{old_obj}_Mat_{old_mat}')
                        
                        output_lines.append(f'usemtl {new_mat_names[-1]}\n')
            
                    elif split_line[0] == 'v':
                        output_lines.append(line)
                        
                    elif split_line[0] == 'f':
                        output_lines.append(line)
            
            with open(os.path.join(root, 'mesh.obj'), mode='w') as file:
                for line in output_lines:
                    file.write(line)
            
            if mtl_file_name is not None:
                old_mat = ''
                output_lines = list()
    
                with open(os.path.join(root, mtl_file_name), mode='r') as file:
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
                        
                            for new_mat in new_mat_names:
                                if old_mat in new_mat:
                                    output_lines.append(f'newmtl {new_mat}\n')
                                    for param in old_parameters:
                                        output_lines.append(param)
                                    output_lines.append('\n')
                
                with open(os.path.join(root, 'mesh.mtl'), mode='w') as file:
                    for line in output_lines[:-1]:
                        file.write(line)
              
            mesh, patch_materials, _ = load_mesh(root, assert_coplanarity=False)
        
            visualize_mesh(root)
            