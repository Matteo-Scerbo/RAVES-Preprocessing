import os
from .src import compute_ART, compute_MoDART


# Project-wide TODOs
# TODO: Revise all comments and documentation.
# TODO: Consistent use of camel case and underscores.
# TODO: Add `run_ART.py`
# TODO: Add complex-valued decomposition in `compute_MoDART.py`


def raves(folder_path: str,
          overwrite: bool = False,
          skip_ART: bool = False, skip_MoDART: bool = False,
          area_threshold: float = 0., thoroughness: float = 0.,
          points_per_square_meter: float = 30., rays_per_hemisphere: int = 1000,
          multiprocess_pool_size: int = 1,
          humidity: float = 50., temperature: float = 20., pressure: float = 100.,
          T60_threshold: float = 1e-1, max_slopes_per_band: int = 10,
          echogram_sample_rate: float = 1e3, skip_T60_plots: bool = False
          ) -> None:
    # TODO: Fill out documentation properly.
    """

    """
    if not os.path.isdir(folder_path):
        raise ValueError('Not a valid folder path:\n\t' + folder_path)

    if not skip_ART:
        # N.B. The folder path is returned by compute_ART and passed on to compute_MoDART,
        # because if `area_threshold > 0` the new mesh might be saved to a different folder.
        folder_path = compute_ART(folder_path=folder_path, overwrite=overwrite,
                                  area_threshold=area_threshold, thoroughness=thoroughness,
                                  points_per_square_meter=points_per_square_meter,
                                  rays_per_hemisphere=rays_per_hemisphere,
                                  multiprocess_pool_size=multiprocess_pool_size,
                                  humidity=humidity, temperature=temperature, pressure=pressure)

    if not skip_MoDART:
        compute_MoDART(folder_path,
                       T60_threshold=T60_threshold,
                       max_slopes_per_band=max_slopes_per_band,
                       echogram_sample_rate=echogram_sample_rate,
                       skip_T60_plots=skip_T60_plots)
