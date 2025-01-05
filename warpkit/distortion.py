from typing import List, Optional, Tuple, Union

import nibabel as nib
import numpy as np

from warpkit.unwrap import unwrap_phase_data, compute_field_maps
from warpkit.utilities import (
    check_affines,
    displacement_maps_to_field_maps,
    field_maps_to_displacement_maps,
    invert_displacement_maps,
)


def medic(
    phase: List[nib.Nifti1Image],
    mag: List[nib.Nifti1Image],
    TEs: Union[List[float], Tuple[float]],
    total_readout_time: float,
    phase_encoding_direction: str,
    frames: Optional[List[int]] = None,
    border_size: int = 5,
    border_filt: Tuple[int, int] = (1, 5),
    svd_filt: int = 10,
    n_cpus: int = 4,
    debug: bool = False,
    wrap_limit: bool = False,
) -> Tuple[nib.Nifti1Image, nib.Nifti1Image, nib.Nifti1Image]:
    """This runs Multi-Echo DIstortion Correction (MEDIC) on a set of phase and magnitude images.

    Computes field maps from unwrapped phase images, converts to displacement maps, and inverts to
    get a set of correction maps for a Multi-Echo EPI sequence. Underneath the hood, this uses
    the ROMEO algorithm to unwrap the phase images. To learn more, see the following paper:

    Dymerska, B., Eckstein, K., Bachrata, B., Siow, B., Trattnig, S., Shmueli, K., Robinson, S.D., 2020.
    Phase Unwrapping with a Rapid Opensource Minimum Spanning TreE AlgOrithm (ROMEO).
    Magnetic Resonance in Medicine. https://doi.org/10.1002/mrm.28563

    Parameters
    ----------
    phase : List[nib.Nifti1Image]
        Phases to unwrap
    mag : List[nib.Nifti1Image]
        Magnitudes associated with each phase
    TEs : Union[List[float], Tuple[float]]
        Echo times associated with each phase (in milliseconds)
    total_readout_time : float
        Total readout time (in seconds)
    phase_encoding_direction : str
        Phase encoding direction (can be i, j, k, i-, j-, k-) or (x, y, z, x-, y-, z-)
    frames : int, optional
        Only process these frame indices, by default None (which means all frames)
    border_size : int, optional
        Size of border in automask, by default 5
    border_filt : Tuple[int, int], optional
        Number of SVD components for each step of border filtering, by default (1, 5)
    svd_filt : int, optional
        Number of SVD components to use for filtering of field maps, by default 10
    n_cpus : int, optional
        Number of CPUs to use, by default 4
    debug : bool, optional
        Whether to save intermediate files, by default False

    Returns
    -------
    nib.Nifti1Image
        Field maps in Hz (distorted space)
    nib.Nifti1Image
        Displacement maps (distorted -> undistorted) in mm
    nib.Nifti1Image
        Field maps in Hz (undistorted space)
    """
    check_affines(phase, mag)

    # unwrap phase and compute field maps
    try:
        unwrapped, new_masks = unwrap_phase_data(
            phase,
            mag,
            TEs,
            border_size=border_size,
            svd_filt=svd_filt,
            frames=frames,
            n_cpus=n_cpus,
            debug=debug,
            wrap_limit=wrap_limit,
        )
        field_maps_native = compute_field_maps(
            unwrapped=unwrapped,
            new_masks=new_masks,
            img=phase[0],
            mag=mag,
            TEs=TEs,
            border_filt=border_filt,
            svd_filt=svd_filt,
            frames=frames,
            n_cpus=n_cpus,
        )
    except IndexError as e:
        raise IndexError(
            "An IndexError was encountered while unwrapping phase images. "
            "This tends to happen if you have noise frames left in your data. "
            "You should remove these frames and try again. "
            "Though, if you already have removed the noise frames from your data, "
            "then I'm out of ideas... Sorry! "
            "But, I'm just a error message that was written to anticipate a common "
            "error after all, not a psychic! "
            ":("
        ) from e

    # convert to displacement maps (these are in distorted space)
    inv_displacement_maps = field_maps_to_displacement_maps(
        field_maps_native, total_readout_time, phase_encoding_direction
    )

    # invert displacement maps (these are in undistorted space)
    displacement_maps = invert_displacement_maps(inv_displacement_maps, phase_encoding_direction)

    # convert correction maps back to undistorted space field map
    field_maps = displacement_maps_to_field_maps(
        displacement_maps, total_readout_time, phase_encoding_direction, flip_sign=True
    )

    # check if we need to flip sign of field maps (this is done by comparing sign of correlation between
    # native and undistorted field maps)
    if np.corrcoef(field_maps.dataobj[..., 0].ravel(), field_maps_native.dataobj[..., 0].ravel())[0, 1] < 0:
        field_map_data = field_maps.get_fdata() * -1
        field_maps = nib.Nifti1Image(field_map_data, field_maps.affine, field_maps.header)

    # return correction maps
    return field_maps_native, displacement_maps, field_maps
