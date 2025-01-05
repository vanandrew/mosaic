"""A workflow to unwrap multi-echo phase data."""

import argparse
import json
import logging
from functools import partial
from pathlib import Path

import nibabel as nib
import numpy as np

from warpkit.distortion import unwrap_phase_data
from warpkit.scripts import epilog
from warpkit.utilities import setup_logging


def main():
    """Build parser object and run workflow."""

    def _path_exists(path, parser):
        """Ensure a given path exists."""
        if path is None or not Path(path).exists():
            raise parser.error(f"Path does not exist: <{path}>.")
        return Path(path).absolute()

    def _is_file(path, parser):
        """Ensure a given path exists and it is a file."""
        path = _path_exists(path, parser)
        if not path.is_file():
            raise parser.error(f"Path should point to a file (or symlink of file): <{path}>.")
        return path

    parser = argparse.ArgumentParser(
        description="Unwrap multi-echo phase data",
        epilog=f"{epilog} 12/09/2022",
    )

    IsFile = partial(_is_file, parser=parser)

    parser.add_argument(
        "--magnitude",
        nargs="+",
        required=True,
        metavar="FILE",
        type=IsFile,
        help="Magnitude data",
    )
    parser.add_argument(
        "--phase",
        nargs="+",
        required=True,
        metavar="FILE",
        type=IsFile,
        help="Phase data",
    )
    parser.add_argument(
        "--metadata",
        nargs="+",
        required=True,
        metavar="FILE",
        type=IsFile,
        help=(
            "JSON sidecar for each echo. "
            "Three fields are required: EchoTime, TotalReadoutTime, and PhaseEncodingDirection."
        ),
    )
    parser.add_argument(
        "--out_prefix",
        help="Prefix to output field maps and displacment maps.",
    )
    parser.add_argument(
        "-f",
        "--noiseframes",
        type=int,
        default=0,
        help=(
            "Number of noise frames at the end of the run. "
            "Noise frames will be removed before unwrapping is performed."
        ),
    )
    parser.add_argument(
        "-n",
        "--n_cpus",
        type=int,
        default=4,
        help="Number of CPUs to use.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode",
    )
    parser.add_argument(
        "--wrap_limit",
        action="store_true",
        default=False,
        help="Turns off some heuristics for phase unwrapping",
    )

    # parse arguments
    args = parser.parse_args()

    # setup logging
    setup_logging()

    # log arguments
    logging.info(f"unwrap_phases: {args}")
    kwargs = vars(args)
    unwrap_phases(**kwargs)


def unwrap_phases(
    *,
    magnitude,
    phase,
    metadata,
    out_prefix,
    noiseframes,
    n_cpus,
    debug,
    wrap_limit,
):
    """Unwrap multi-echo phase data.

    Parameters
    ----------
    magnitude : list of str
        List of magnitude data files.
    phase : list of str
        List of phase data files.
    metadata : list of str
        List of JSON sidecar files for each echo.
    out_prefix : str
        Prefix to output field maps and displacment maps.
    noiseframes : int
        Number of noise frames at the end of the run.
        Noise frames will be removed before unwrapping is performed.
    n_cpus : int
        Number of CPUs to use.
    debug : bool
        Debug mode.
    wrap_limit : bool
        Turns off some heuristics for phase unwrapping.
    """
    # load magnitude and phase data
    magnitude_imgs = [nib.load(m) for m in magnitude]
    phase_imgs = [nib.load(p) for p in phase]

    # if noiseframes specified, remove them
    if noiseframes > 0:
        logging.info(f"Removing {noiseframes} noise frames from the end of the run...")
        magnitude_imgs = [m.slicer[..., : -noiseframes] for m in magnitude_imgs]
        phase_imgs = [p.slicer[..., : -noiseframes] for p in phase_imgs]

    # check if data is 4D or 3D
    if phase_imgs[0].ndim == 3:
        # convert data to 4D
        phase_imgs = [
            nib.Nifti1Image(p.get_fdata()[..., np.newaxis], p.affine, p.header) for p in phase_imgs
        ]
        magnitude_imgs = [
            nib.Nifti1Image(m.get_fdata()[..., np.newaxis], m.affine, m.header) for m in magnitude_imgs
        ]
    else:
        raise ValueError("Data must be 3D or 4D.")

    # get metadata
    echo_times = []
    total_readout_time = None
    phase_encoding_direction = None
    for i_run, json_file in enumerate(metadata):
        with open(json_file, "r") as fobj:
            metadata_dict = json.load(fobj)
            echo_times.append(metadata_dict["EchoTime"] * 1000)  # convert TE from s to ms

        if i_run == 0:
            total_readout_time = metadata_dict.get("TotalReadoutTime")
            phase_encoding_direction = metadata_dict.get("PhaseEncodingDirection")

    if total_readout_time is None:
        raise ValueError("Could not find 'TotalReadoutTime' field in metadata.")

    if phase_encoding_direction is None:
        raise ValueError("Could not find 'PhaseEncodingDirection' field in metadata.")

    # Sort the echo times and data by echo time
    echo_times, magnitude_imgs, phase_imgs = zip(*sorted(zip(echo_times, magnitude_imgs, phase_imgs)))

    # now run MEDIC's phase-unwrapping method
    unwrapped_phases = unwrap_phase_data(
        phase=phase_imgs,
        mag=magnitude_imgs,
        TEs=echo_times,
        total_readout_time=total_readout_time,
        phase_encoding_direction=phase_encoding_direction,
        out_prefix=out_prefix,
        n_cpus=n_cpus,
        debug=debug,
        wrap_limit=wrap_limit,
    )

    # save the fmaps and dmaps to file
    logging.info("Saving field maps and displacement maps to file...")
    for i_echo, unwrapped_phase in enumerate(unwrapped_phases):
        unwrapped_phase.to_filename(f"{out_prefix}_echo-{i_echo + 1}_phase.nii.gz")
    logging.info("Done.")
