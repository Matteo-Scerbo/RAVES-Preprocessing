import argparse
from .api import raves


def main(argv=None):
    """
    Parser to run RAVES from a command line; see notes in `__main__.py`.
    """
    parser = argparse.ArgumentParser(
        prog='RAVES pre-processing',
        description='One-shot RAVES pipeline: prepares ART model of a given environment and runs MoD-ART on it.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('folder_path', type=str,
                        help='Path to your environment folder.')

    parser.add_argument('--overwrite', action='store_true', default=argparse.SUPPRESS,
                        help='Re-compute and overwrite existing ART kernels (they are re-used by default).')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--skip_ART', action='store_true', default=argparse.SUPPRESS,
                       help='Skip the ART step and only perform MoD-ART, assuming all necessary files are already present.')
    group.add_argument('--skip_MoDART', action='store_true', default=argparse.SUPPRESS,
                       help='Prepare the ART model and stop, without performing MoD-ART.')

    parser.add_argument('--skip_T60_plots', action='store_true', default=argparse.SUPPRESS,
                        help='Avoid saving illustrations of the MoD-ART modes.')

    parser.add_argument('-ppsm', '--points_per_square_meter', type=float, default=30.,
                        help='Number of surface sample points per square meter for the ART numerical integration.')
    parser.add_argument('-rays', '--rays_per_hemisphere', type=int, default=1000,
                        help='Number of rays traced from each surface sample point for the ART numerical integration.')

    parser.add_argument('-pool', '--multiprocess_pool_size', type=int, default=1,
                        help='Number of parallel processes allowed for the ART numerical integration.')

    parser.add_argument('-T60', '--T60_threshold', type=float, default=0.1,
                        help='The MoD-ART eigenvalue search stops after finding at least one mode whose reverberation time '
                             'is below this threshold (in seconds), for each each frequency band.')
    parser.add_argument('-slopes', '--max_slopes_per_band', type=int, default=10,
                        help='The MoD-ART eigenvalue search stops after finding at least this many modes, for each each frequency band.')
    parser.add_argument('-f_e', '--echogram_sample_rate', type=float, default=1e3,
                        help='The sample rate used to discretize propagation path delays for MoD-ART. NOT RELATED TO AUDIO SAMPLE RATE.')

    # N.B. The repeated % sign is important, it's an escape character for the parser
    parser.add_argument('-humi', '--humidity', type=float, default=50.,
                        help='Air humidity (%%) used to compute frequency-dependent energy losses.')
    parser.add_argument('-temp', '--temperature', type=float, default=20.,
                        help='Air temperature (°C) used to compute frequency-dependent energy losses as well as the speed of sound.')
    parser.add_argument('-pres', '--pressure', type=float, default=100.,
                        help='Atmospheric pressure (kPa) used to compute frequency-dependent energy losses.')

    parser.add_argument('-area', '--area_threshold', type=float, default=0.,
                        help='If greater than zero, surface patches are merged until all areas are above this threshold (in square meters) if possible. '
                             'The new mesh is saved to a different folder.')
    parser.add_argument('-thrns', '--thoroughness', type=float, default=0.,
                        help='Used during re-meshing, if area_threshold is also provided.')

    raves(**vars(parser.parse_args(argv)))
