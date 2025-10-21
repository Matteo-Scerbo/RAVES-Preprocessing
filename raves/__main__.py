# Lets you run: python -m raves "C:/your/environment/folder/path"
import os
import sys

import numpy as np

from .src.utils import visualize_mesh
from .src import compute_ART, assess_ART_on_grid, compute_MoDART


# Project-wide TODOs
# TODO: Write path length in seconds; do not take humidity as MoD-ART argument.
# TODO: Double-check the accuracy of `<start patch idx> <end patch idx> <propagation path idx>`.
# TODO: Double-check the accuracy of left / right order throughout.
# TODO: Revise all comments and documentation.
# TODO: Consistent use of camel case and underscores.
# TODO: Add `run_ART.py`
# TODO: Add complex-valued decomposition in `compute_MoDART.py`


def main(folder_path: str) -> None:
    # TODO: Accept optional arguments for compute_ART and compute_MoDART and pass them on.
    if not os.path.isdir(folder_path):
        raise ValueError('Not a valid folder path:\n\t' + folder_path)

    # visualize_mesh(folder_path)

    # assess_ART_on_grid(folder_path, compute_missing=False,
    #                    area_threshold=20., thoroughness=0.1,
    #                    points_per_square_meter=[10., 20., 30., 40., 50.],
    #                    rays_per_hemisphere=[1000],
    #                    pool_size=4)

    folder_path = compute_ART(folder_path, pool_size=4,
                              points_per_square_meter=40.,
                              rays_per_hemisphere=1000)
    compute_MoDART(folder_path, echogram_sample_rate=1e4,
                   max_slopes_per_band=20, t60_threshold=0.2)


if __name__ == "__main__":
    # TODO: Accept optional arguments for compute_ART and compute_MoDART and pass them on.
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print('No arguments provided. Please provide a valid folder path as argument.')
