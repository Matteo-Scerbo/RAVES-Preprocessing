import os
import sys

from utils import validate_inputs


def main(path) -> None:
    if os.path.isdir(path):
        if validate_inputs(path):
            print('Running `compute_MoDART` in the environment "' + path.split('/')[-1] + '"')

            # TODO: Read `ART_kernel_diffuse.mtx`, `ART_kernel_specular.mtx`, `path_lengths.csv`

            # TODO: Perform real-valued decomposition as in existing code (for each frequency band)

            # TODO: Rescale eigenvectors; note that etendue can be inferred from patch areas and `kernel_diffuse`

            # TODO: Write `MoD-ART.csv`
    else:
        print('Not a valid folder path:\n\t' + path)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print('No arguments provided. Please provide a valid folder path as argument.')
