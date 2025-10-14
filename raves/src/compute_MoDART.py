import os
import sys


def main(folder_path: str) -> None:
    # TODO: Fill out documentation properly.
    """

    Args:
        folder_path:

    Returns:

    """
    if not os.path.isdir(folder_path):
        raise ValueError('Not a valid folder path:\n\t' + folder_path)

    print('Running `compute_MoDART` in the environment "' + folder_path.split('/')[-1] + '"')

    # TODO: Read `ART_diffuse_kernel.mtx`, `ART_specular_kernel.mtx`, `path_lengths.csv`

    # TODO: Perform real-valued decomposition as in existing code (for each frequency band)

    # TODO: Rescale eigenvectors; note that etendue can be inferred from patch areas and `diffuse_kernel`

    # TODO: Write `MoD-ART.csv`


if __name__ == "__main__":
    # TODO: Accept optional arguments and pass them on.
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print('No arguments provided. Please provide a valid folder path as argument.')
