import os
import sys

def main(path) -> None:
    if os.path.isdir(path):
        print('I am `compute_ART` and I will process the environment named ' + path.split('/')[-1])
        # TODO: Read mesh.obj file (validate mesh.mtl format?)
        # TODO: Read materials.csv
        # TODO: Create a Model instance (is a class really necessary? avoid if possible)
        # TODO: Make some sanity checks (patch size and number, ...)
        # TODO: Perform the ART pre-computation
        # TODO: Save recursion parameters:
        #           ART_kernel_diffuse.mtx
        #           ART_kernel_specular.mtx
        #           ART_octave_band_1.mtx, ART_octave_band_2.mtx, ...
        #           path_indexing.mtx
        #           path_lengths.csv
    else:
        print('Not a valid folder path:\n\t' + path)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print('No arguments provided. The first argument should be a valid folder path.')
