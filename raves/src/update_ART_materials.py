import os
import sys

from utils import validate_inputs, load_mesh


def main(path) -> None:
    if os.path.isdir(path):
        if validate_inputs(path):
            print('Running `update_ART_materials` in the environment "' + path.split('/')[-1] + '"')

            mesh, patch_materials = load_mesh(path + '/mesh.obj')

            # TODO: Read `ART_kernel_diffuse.mtx`, `ART_kernel_specular.mtx`, `path_indexing.mtx`
            # TODO: Check that the number of patches in the kernels matches the loaded mesh
            # TODO: Check that path_indexing makes sense (how?)

            # TODO: Prepare and write `ART_octave_band_1.mtx`, `ART_octave_band_2.mtx`, etc. (for each frequency band)
    else:
        print('Not a valid folder path:\n\t' + path)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print('No arguments provided. Please provide a valid folder path as argument.')
