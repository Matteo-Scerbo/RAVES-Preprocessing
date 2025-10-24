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
    """
    Run the full RAVES pipeline in a given environment folder.
    This builds the ART model and immediately performs MoD-ART.

    Parameters
    ----------
    folder_path : str
        Path to the environment folder.
    overwrite : bool, default: False
        If True, any existing ART kernels are re-computed and overwritten.
        Otherwise, existing geometrical data is re-used, whereas material
        properties (and air absorption parameters) are read again and replaced.
    skip_ART : bool, default: False
        If True, skip the ART pre-processing stage and run only MoD-ART.
    skip_MoDART : bool, default: False
        If True, run only ART and skip MoD-ART.
    area_threshold : float, default: 0.0
        Minimum patch area (square meters) used to optionally simplify the mesh.
        Values > 0 may cause a simplified mesh to be written to a new folder.
        The path returned by `compute_ART` is then propagated to `compute_MoDART`.
    thoroughness : float, default: 0.0
        Effort/speed trade-off factor for remeshing (higher is more thorough).
    points_per_square_meter : float, default: 30.0
        Surface sampling density used during ART.
    rays_per_hemisphere : int, default: 1000
        Number of rays cast from each sample point during ART.
    multiprocess_pool_size : int, default: 1
        Number of worker processes to use. Use 1 to disable multiprocessing.
    humidity : float, default: 50.0
        Ambient relative humidity (%).
    temperature : float, default: 20.0
        Ambient temperature (°C).
    pressure : float, default: 100.0
        Ambient pressure (kPa).
    T60_threshold : float, default: 1e-1
        Threshold used during decomposition; the processing of each band
        halts if an energy mode is found to have T60 below this value.
    max_slopes_per_band : int, default: 10
        Threshold used during decomposition; the processing of each band
        halts if more than this many energy modes have been found.
    echogram_sample_rate : float, default: 1e3
        Sample rate (Hz) in the decomposed ART model.
        NOT TO BE CONFUSED WITH AN AUDIO RATE.
    skip_T60_plots : bool, default: False
        If True, suppress generation of T60 plots by `compute_MoDART`.

    Returns
    -------
    None
        Outputs are written to `folder_path` (or a new path if a simplified mesh
        is produced) following the specifications outlined in `README.md`.

    Notes
    -----
    If `skip_ART` is False and `area_threshold > 0`, `compute_ART` may write
    a simplified mesh to a different folder and will return that new path;
    this function forwards that path to `compute_MoDART` to keep outputs consistent.

    Examples
    --------
    >>> from raves import raves
    >>> raves("/path/to/project", multiprocess_pool_size=4, echogram_sample_rate=1e4)
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
