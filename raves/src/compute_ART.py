import os
import sys
import numpy as np

from utils import validate_inputs, load_mesh, RayBundle


def main(path) -> None:
    if os.path.isdir(path):
        if validate_inputs(path):
            print('Running `compute_ART` in the environment "' + path.split('/')[-1] + '"')

            mesh, patch_materials = load_mesh(path + '/mesh.obj')

            if mesh.size() > 1000:
                print('Warning: the mesh contains a very large number of patches (' + str(mesh.size()) + ')')
            if np.any(mesh.area < 0.1):
                print('Warning: the mesh contains very small patches (smallest area: ' + str(np.min(mesh.area)) + ')')

            # TODO: Initialize `path_lengths`, `kernel_diffuse`, and `kernel_specular` (sparse arrays).
            # TODO: Get patch areas as sum of triangle areas.
            # TODO: Perform surface integrals. For each patch:
            #       Uniformly sample the patch surface.
            #       Prepare a hemisphere pencil, to be moved around with RayBundle.moveOrigins(sample_point).
            #       For each sample point, trace rays in visible hemisphere; accumulate and normalize as in existing code.

            # TODO: Write
            #           ART_kernel_diffuse.mtx
            #           ART_kernel_specular.mtx
            #           path_indexing.mtx
            #           path_lengths.csv

            # TODO: Read materials.csv
            # TODO: Write ART_octave_band_1.mtx, ART_octave_band_2.mtx, etc. (for each frequency band)
    else:
        print('Not a valid folder path:\n\t' + path)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print('No arguments provided. Please provide a valid folder path as argument.')
