# Lets you run: python -m raves "C:/your/environment/folder/path"
import os
import sys
from .src import compute_ART
from .src import compute_MoDART


def main(folder_path: str) -> None:
    # TODO: Accept optional arguments for compute_ART and compute_MoDART and pass them on.
    if not os.path.isdir(folder_path):
        raise ValueError('Not a valid folder path:\n\t' + folder_path)

    compute_ART(folder_path)
    compute_MoDART(folder_path)


if __name__ == "__main__":
    # TODO: Accept optional arguments for compute_ART and compute_MoDART and pass them on.
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print('No arguments provided. Please provide a valid folder path as argument.')
