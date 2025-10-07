import os
import sys

from utils import validate_inputs, load_mesh


def main(folder_path) -> None:
    if os.path.isdir(folder_path):
        if validate_inputs(folder_path):
            print('Running `update_ART_materials` in the environment "' + folder_path.split('/')[-1] + '"')

            mesh, patch_materials = load_mesh(folder_path + '/mesh.obj')

            # TODO: Read `ART_diffuse_kernel.mtx`, `ART_specular_kernel.mtx`, `path_indexing.mtx`
            # TODO: Check that the number of patches in the kernels matches the loaded mesh
            # TODO: Check that path_indexing makes sense (how?)

            # TODO: Prepare and write `ART_octave_band_1.mtx`, `ART_octave_band_2.mtx`, etc. (for each frequency band)
    else:
        print('Not a valid folder path:\n\t' + folder_path)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print('No arguments provided. Please provide a valid folder path as argument.')
