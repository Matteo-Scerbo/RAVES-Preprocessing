# Lets you run: python -m raves "C:/your/environment/folder/path"
import os
import sys
from .src import compute_ART
from .src import compute_MoDART


# Project-wide TODOs
# TODO: Revise all comments and documentation.
# TODO: Consistent use of camel case and underscores.
# TODO: Add `run_ART.py`
# TODO: Add complex-valued decomposition in `compute_MoDART.py`
# TODO: In the theory notes, explain the difference between propagating power vs radiance. We do power here.
# TODO: In the theory notes, explain why etendues are baked into the eigenvectors.


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
