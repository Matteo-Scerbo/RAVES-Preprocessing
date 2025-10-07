import os
import sys

from utils import validate_inputs


def main(folder_path) -> None:
    if os.path.isdir(folder_path):
        if validate_inputs(folder_path):
            print('Running `compute_MoDART` in the environment "' + folder_path.split('/')[-1] + '"')

            # TODO: Read `ART_diffuse_kernel.mtx`, `ART_specular_kernel.mtx`, `path_lengths.csv`

            # TODO: Perform real-valued decomposition as in existing code (for each frequency band)

            # TODO: Rescale eigenvectors; note that etendue can be inferred from patch areas and `diffuse_kernel`

            # TODO: Write `MoD-ART.csv`
    else:
        print('Not a valid folder path:\n\t' + folder_path)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print('No arguments provided. Please provide a valid folder path as argument.')
