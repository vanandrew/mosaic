import logging
import sys
from pathlib import Path
from typing import Any, Tuple, cast

import nibabel as nib
import numpy as np
import numpy.typing as npt
from scipy import signal
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    binary_fill_holes,
    generate_binary_structure,
)
from skimage.filters import threshold_otsu  # type: ignore
from skimage.measure import label, regionprops
from transforms3d.affines import decompose44

from . import (
    compute_hausdorff_distance as compute_hausdorff_distance_cpp,  # type: ignore
)
from . import (
    compute_jacobian_determinant as compute_jacobian_determinant_cpp,  # type: ignore
)
from . import invert_displacement_field as invert_displacement_field_cpp  # type: ignore
from . import invert_displacement_map as invert_displacement_map_cpp  # type: ignore
from . import resample as resample_cpp  # type: ignore

# map axis names to axis codes
AXIS_MAP = {"x": 0, "y": 1, "z": 2, "x-": 0, "y-": 1, "z-": 2, "i": 0, "j": 1, "k": 2, "i-": 0, "j-": 1, "k-": 2}


# warp_itk_flips
WARP_ITK_FLIPS = {
    "itk": np.array([1, 1, 1]),
    "fsl": np.array([-1, 1, 1]),
    "ants": np.array([-1, -1, 1]),
    "afni": np.array([-1, -1, 1]),
}


def setup_logging(log_file: str = None) -> None:
    """Sets up logging output.

    Parameters
    ----------
    log_file: str
        Setup path to log file.
    """
    # create handlers list
    handlers = list()

    # create file write handler if log file specified
    if log_file:
        # get log file path
        log_file_path = Path(log_file).resolve()

        # create path to log if needed
        log_file_path.parent.mkdir(parents=True, exist_ok=True)

        # append to handlers
        handlers.append(logging.FileHandler(str(log_file_path), mode="w"))  # will overwrite logs if they exist at path

    # add stdout streaming to handlers
    handlers.append(logging.StreamHandler(sys.stdout))

    # setup log output config
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s", handlers=handlers)


def normalize(data: npt.NDArray) -> npt.NDArray:
    """Normalize data to [0, 1]

    Parameters
    ----------
    data : npt.NDArray
        data to be normalized

    Returns
    -------
    npt.NDArray
        normalized data
    """
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def corr2_coeff(A: npt.NDArray, B: npt.NDArray) -> npt.NDArray:
    """Efficiently calculates correlation coefficient between the columns of two 2D arrays

    Parameters
    ----------
    A : npt.NDArray
        1st array to correlate
    B : npt.NDArray
        2nd array to correlate

    Returns
    -------
    npt.NDArray
        array of correlation coefficients
    """
    # Transpose A and B
    A = A.T
    B = B.T

    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(axis=1, keepdims=True)
    B_mB = B - B.mean(axis=1, keepdims=True)

    # Sum of squares across rows
    ssA = (A_mA**2).sum(axis=1, keepdims=True)
    ssB = (B_mB**2).sum(axis=1, keepdims=True).T

    # Finally get corr coeff
    return np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA, ssB))


def rescale_phase(data: npt.NDArray[Any], min: int = -4096, max: int = 4096) -> npt.NDArray[Any]:
    """Rescale phase data to [-pi, pi]

    Rescales data to [-pi, pi] using the specified min and max inputs.

    Parameters
    ----------
    data : npt.NDArray[Any]
        phase data to be rescaled
    min : int, optional
        min value that should be mapped to -pi, by default -4096
    max : int, optional
        max value that should be mapped to pi, by default 4096

    Returns
    -------
    npt.NDArray[Any]
        rescaled phase data
    """
    return (data - min) / (max - min) * 2 * np.pi - np.pi


def get_largest_connected_component(mask_data: npt.NDArray[np.bool_]) -> npt.NDArray[np.bool_]:
    """Get the largest connected component of a mask

    Parameters
    ----------
    mask_data : npt.NDArray[np.bool_]
        Mask to get the largest connected component of

    Returns
    -------
    npt.NDArray[np.bool_]
        Mask with only the largest connected component
    """
    # get the largest connected component
    labelled_mask = label(mask_data)
    props = regionprops(labelled_mask)
    sorted_props = sorted(props, key=lambda x: x.area, reverse=True)
    mask_data = labelled_mask == sorted_props[0].label

    # return the mask
    return mask_data


def create_brain_mask(mag_shortest: npt.NDArray[np.float32], extra_dilation: int = 0) -> npt.NDArray[np.bool_]:
    """Create a quick brain mask for a single frame.

    Parameters
    ----------
    mag_shortest : npt.NDArray[np.float32]
        Magnitude data with the shortest echo time
    extra_dilation : int
        Number of extra dilations (or erosions if negative) to perform, by default 0

    Returns
    -------
    npt.NDArray[np.bool_]
        Mask of voxels to use for unwrapping
    """
    # create structuring element
    strel = generate_binary_structure(3, 2)

    # get the otsu threshold
    threshold = threshold_otsu(mag_shortest)
    mask_data = mag_shortest > threshold
    mask_data = cast(npt.NDArray[np.float32], binary_fill_holes(mask_data, strel))

    # erode mask
    mask_data = cast(npt.NDArray[np.bool_], binary_erosion(mask_data, structure=strel, iterations=2, border_value=1))

    # get largest connected component
    mask_data = get_largest_connected_component(mask_data)

    # dilate the mask
    mask_data = binary_dilation(mask_data, structure=strel, iterations=2)

    # extra dilation to get areas on the edge of the brain
    if extra_dilation > 0:
        mask_data = binary_dilation(mask_data, structure=strel, iterations=extra_dilation)
    # if negative, erode instead
    if extra_dilation < 0:
        mask_data = binary_erosion(mask_data, structure=strel, iterations=abs(extra_dilation))

    # since we can't have a completely empty mask, set all zeros to ones
    # if the mask is all empty
    if np.all(np.isclose(mask_data, 0)):
        mask_data = np.ones(mask_data.shape)

    # return the mask
    return mask_data.astype(np.bool_)


def field_maps_to_displacement_maps(
    field_maps: nib.Nifti1Image, total_readout_time: float, phase_encoding_direction: str
) -> nib.Nifti1Image:
    """Convert field maps (Hz) to displacement maps (mm)

    Parameters
    ----------
    field_maps : nib.Nifti1Image
        Field map data in Hz
    total_readout_time : float
        Total readout time (in seconds)
    phase_encoding_direction : str
        Phase encoding direction

    Returns
    -------
    nib.Nifti1Image
        Displacment maps in mm
    """
    # get number of phase encoding lines
    axis_code = AXIS_MAP[phase_encoding_direction]

    # get voxel size
    voxel_size = field_maps.header.get_zooms()[axis_code]  # type: ignore

    # if there is a negative sign in the phase encoding direction
    # we need to flip the direction of the displacment map
    if "-" in phase_encoding_direction:
        voxel_size *= -1

    # for itk form, we need to flip x/y axis
    if axis_code == 0 or axis_code == 1:
        voxel_size *= -1

    # convert field maps to displacement maps
    data = field_maps.get_fdata()

    new_data = data * total_readout_time * voxel_size

    return nib.Nifti1Image(new_data, field_maps.affine, field_maps.header)


def displacement_maps_to_field_maps(
    displacement_maps: nib.Nifti1Image,
    total_readout_time: float,
    phase_encoding_direction: str,
    flip_sign: bool = False,
) -> nib.Nifti1Image:
    """Convert displacement maps (mm) to field maps (Hz)

    Parameters
    ----------
    displacement_maps : nib.Nifti1Image
        Displacement map data in mm
    total_readout_time : float
        Total readout time (in seconds)
    phase_encoding_direction : str
        Phase encoding direction
    flip_sign : bool, optional
        Flips the sign of the field maps. This is needed depending on how the reference frame was defined.
        By default False

    Returns
    -------
    nib.Nifti1Image
        Field maps in Hz
    """
    # get number of phase encoding lines
    axis_code = AXIS_MAP[phase_encoding_direction]

    # get voxel size
    voxel_size = displacement_maps.header.get_zooms()[axis_code]  # type: ignore

    # if there is a negative sign in the phase encoding direction
    # we need to flip the direction of the displacment map
    if "-" in phase_encoding_direction:
        voxel_size *= -1

    # for itk form, we need to flip x/y axis
    if axis_code == 0 or axis_code == 1:
        voxel_size *= -1

    # convert displacement maps to field maps
    data = displacement_maps.get_fdata()
    new_data = data / (total_readout_time * voxel_size)
    if flip_sign:
        new_data *= -1

    return nib.Nifti1Image(new_data, displacement_maps.affine, displacement_maps.header)


def displacement_map_to_field(
    displacement_map: nib.Nifti1Image,
    axis: str = "y",
    format: str = "itk",
    frame: int = 0,
) -> nib.Nifti1Image:
    """Convert a displacement map to a displacement field

    Parameters
    ----------
    displacement_map : nib.Nifti1Image
        Displacement map data in mm
    axis : str, optional
        Axis displacement maps are along, by default "y"
    format : str, optional
        Format of the displacement field, by default "itk"
    frame : int, optional
        If the displacement map has multiple frames, this specifies which frame to convert, by default 0

    Returns
    -------
    nib.Nifti1Image
        Displacement field in mm
    """
    # get the axis displacement map should insert
    axis_code = AXIS_MAP[axis]
    displacement_map = cast(nib.Nifti1Image, displacement_map)

    # check if the displacement map has multiple frames
    if len(displacement_map.shape) == 4:
        data = np.asarray(displacement_map.dataobj[..., frame])
    else:  # otherwise just get the data
        data = displacement_map.get_fdata()

    # get affine and header info
    affine = displacement_map.affine
    header = cast(nib.Nifti1Header, displacement_map.header)

    # create a new array to hold the displacement field
    new_data = np.zeros((*data.shape, 3), dtype=np.float32)

    # insert data at axis
    new_data[..., axis_code] = data  # type: ignore

    # set the header to the correct intent code
    # this is needed for ANTs, but has no effect on FSL/AFNI
    header.set_intent("vector")

    # form the image
    warp = nib.Nifti1Image(new_data, affine, header)

    # convert to whatever format is needed and return
    return convert_warp(warp, in_type="itk", out_type=format)


def get_x_orient_transform(img: nib.Nifti1Image, x: str) -> Tuple[np.ndarray, np.ndarray]:
    """Get the transform to x orientation and back

    Parameters
    ----------
    img : nib.Nifti1Image
        Image to convert to X orientation
    x : str
        X orientation to convert to

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        X transform and inverse x transform
    """
    img_orientation = nib.orientations.io_orientation(img.affine)
    x_orientation = nib.orientations.axcodes2ornt(x)
    to_canonical = nib.orientations.ornt_transform(img_orientation, x_orientation)
    from_canonical = nib.orientations.ornt_transform(x_orientation, img_orientation)
    return to_canonical, from_canonical


def get_ras_orient_transform(img: nib.Nifti1Image) -> Tuple[np.ndarray, np.ndarray]:
    """Get the transform to RAS orientation and back

    RAS orientation ensures no zooms are negative, which is necessary for
    passing orientation information into ITK. Since ITK internally uses
    LPS orientation, transforms passed into ITK may need to be flipped in x/y
    based on the software package. To prevent user confusion, this handled via
    the WARP_ITK_FLIPS global variable. Users should not need to worry about
    the conversions as long as the `convert_warp` function is used to convert
    the transform to the itk form.

    Parameters
    ----------
    img : nib.Nifti1Image
        Image to convert to RAS orientation

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        RAS transform and inverse RAS transform
    """
    return get_x_orient_transform(img, "RAS")


def invert_displacement_maps(
    displacement_maps: nib.Nifti1Image, axis: str = "y", verbose: bool = False
) -> nib.Nifti1Image:
    """Invert displacement maps

    Parameters
    ----------
    displacement_maps : nib.Nifti1Image
        Displacement map data in mm
    axis : str, optional
        Axis displacement maps are along, by default "y"
    verbose : bool, optional
        Print debugging information, by default False

    Returns
    -------
    nib.Nifti1Image
        Inverted displacement maps in mm
    """
    # get the axis that this displacement map is along
    axis_code = AXIS_MAP[axis]

    # we convert the data to RAS, for passing into ITK
    # but then return to original coordinate system after
    to_canonical, from_canonical = get_ras_orient_transform(displacement_maps)

    # get to RAS orientation
    displacement_maps_ras = displacement_maps.as_reoriented(to_canonical)

    # get data
    data = displacement_maps_ras.dataobj

    # split affine into components
    translations, rotations, zooms, _ = decompose44(displacement_maps_ras.affine)

    # invert maps
    new_data = np.zeros(data.shape, dtype=np.float32)
    logging.info("Inverting displacement maps...")
    for i_vol in range(data.shape[-1]):
        logging.info(f"Processing frame: {i_vol}")
        # pad array with edge values so edge effects of inverse are avoided
        mod_data = np.pad(data[..., i_vol], pad_width=1)

        new_data[..., i_vol] = invert_displacement_map_cpp(
            mod_data,
            translations,
            rotations,
            zooms,
            axis=axis_code,
            verbose=verbose,
        )[1 : data.shape[0] + 1, 1 : data.shape[1] + 1, 1 : data.shape[2] + 1]

    # make new image in original orientation
    inv_displacement_maps = nib.Nifti1Image(
        new_data, displacement_maps_ras.affine, displacement_maps_ras.header
    ).as_reoriented(from_canonical)

    # return inverted maps
    return cast(nib.Nifti1Image, inv_displacement_maps)


def invert_displacement_field(displacement_field: nib.Nifti1Image, verbose: bool = False) -> nib.Nifti1Image:
    """Invert displacement field

    Parameters
    ----------
    displacement_field : nib.Nifti1Image
        Displacement field data in mm
    verbose : bool, optional
        Print debugging information, by default False

    Returns
    -------
    nib.Nifti1Image
        Inverted displacement field in mm
    """
    # we convert the data to RAS, for passing into ITK
    # but then return to original coordinate system after
    to_canonical, from_canonical = get_ras_orient_transform(displacement_field)

    # get to RAS orientation
    displacement_field_ras = displacement_field.as_reoriented(to_canonical)

    # get data
    data = displacement_field_ras.get_fdata()

    # split affine into components
    translations, rotations, zooms, _ = decompose44(displacement_field_ras.affine)

    # pad array with edge values so edge effects of inverse are avoided
    mod_data = np.pad(data, pad_width=1)

    # invert displacement field
    new_data = invert_displacement_field_cpp(mod_data, translations, rotations, zooms, verbose=verbose)[
        1 : data.shape[0] + 1, 1 : data.shape[1] + 1, 1 : data.shape[2] + 1
    ]

    # make new image
    inv_displacement_field = nib.Nifti1Image(
        new_data, displacement_field_ras.affine, displacement_field_ras.header
    ).as_reoriented(from_canonical)

    # return inverted maps
    return cast(nib.Nifti1Image, inv_displacement_field)


def resample_image(
    reference_image: nib.Nifti1Image, input_image: nib.Nifti1Image, transform: nib.Nifti1Image
) -> nib.Nifti1Image:
    """Resample an image in a reference image space with a given transform.

    Parameters
    ----------
    reference_image : nib.Nifti1Image
        Reference image grid to use.
    input_image : nib.Nifti1Image
        Input image to resample.
    transform : nib.Nifti1Image
        Transform to apply to input image (must be itk format).

    Returns
    -------
    nib.Nifti1Image
        Transformed input image in reference space.
    """
    # get all data in RAS orientation
    ref_to_canonical, ref_from_canonical = get_ras_orient_transform(reference_image)
    in_to_canonical, _ = get_ras_orient_transform(input_image)
    tr_to_canonical, _ = get_ras_orient_transform(transform)

    # get to RAS orientation
    reference_image_ras = reference_image.as_reoriented(ref_to_canonical)
    input_image_ras = input_image.as_reoriented(in_to_canonical)
    transform_ras = transform.as_reoriented(tr_to_canonical)

    # get data
    ref_shape = reference_image_ras.shape
    input_data = input_image_ras.get_fdata()
    transform_data = transform_ras.get_fdata()

    # if the transform data is 5D do a squeeze
    if transform_data.ndim == 5:
        transform_data = transform_data.squeeze()

    # check the last dimension for size 3
    if transform_data.shape[-1] != 3:
        raise ValueError("Transform data must have size 3 in last axis")

    # split affine into components
    ref_origin, ref_rotations, ref_zooms, _ = decompose44(reference_image_ras.affine)
    in_origin, in_rotations, in_zooms, _ = decompose44(input_image_ras.affine)
    tr_origin, tr_rotations, tr_zooms, _ = decompose44(transform_ras.affine)

    # call resample function
    resampled_data = resample_cpp(
        input_data,
        in_origin,
        in_rotations,
        in_zooms,
        ref_shape,
        ref_origin,
        ref_rotations,
        ref_zooms,
        transform_data,
        tr_origin,
        tr_rotations,
        tr_zooms,
    )

    # make new image
    resampled_image = nib.Nifti1Image(
        resampled_data, reference_image_ras.affine, reference_image_ras.header
    ).as_reoriented(ref_from_canonical)

    # return resampled image
    return cast(nib.Nifti1Image, resampled_image)


def convert_warp(in_warp: nib.Nifti1Image, in_type: str, out_type: str) -> nib.Nifti1Image:
    """Converts warp from one type to another.

    Parameters
    ----------
    in_warp : nib.Nifti1Image
        Input warp to convert.
    in_type : str
        Type of the input warp. Can be "fsl", "ants", "itk", or "afni"
    out_type : str
        Type of warp to output. Can be "fsl", "ants", "itk", or "afni"

    Returns
    -------
    nib.Nifti1Image
        Output warp in desired format.
    """
    # check data shape
    if len(in_warp.shape) != 4:
        if len(in_warp.shape) != 5:
            raise ValueError("Input warp must be 4D or 5D.")
        else:
            if in_warp.shape[3] == 1 and in_warp.shape[-1]:
                raise ValueError("Input warp must have singleton dimension in 4th axis and size in last axis.")
    else:
        # check last axis size
        if in_warp.shape[-1] != 3:
            raise ValueError("Warp must have size 3 in last axis.")

    # get input in RAS orientation
    to_canonical, from_canonical = get_ras_orient_transform(in_warp)

    # get to RAS orientation
    in_warp_ras = in_warp.as_reoriented(to_canonical)

    # get data in itk form (without extra dim)
    warp_data = in_warp_ras.get_fdata().squeeze()

    # convert input to itk form
    if in_type in WARP_ITK_FLIPS:
        flip_array = WARP_ITK_FLIPS[in_type]
        warp_data[..., 0] = warp_data[..., 0] * flip_array[0]
        warp_data[..., 1] = warp_data[..., 1] * flip_array[1]
        warp_data[..., 2] = warp_data[..., 2] * flip_array[2]
    else:
        raise ValueError(f"Input type {in_type} not recognized")

    # now convert the data into output type
    if out_type in WARP_ITK_FLIPS:
        flip_array = WARP_ITK_FLIPS[out_type]
        warp_data[..., 0] = warp_data[..., 0] * flip_array[0]
        warp_data[..., 1] = warp_data[..., 1] * flip_array[1]
        warp_data[..., 2] = warp_data[..., 2] * flip_array[2]
    else:
        raise ValueError(f"Output type {out_type} not recognized")

    # for afni and ants types we need to add a 4th dimension
    if out_type in ["afni", "ants"]:
        warp_data = warp_data[..., np.newaxis, :]

    # create nifti image
    out_warp = nib.Nifti1Image(warp_data, in_warp_ras.affine, in_warp_ras.header).as_reoriented(from_canonical)

    # add the vector intent code to the header
    cast(nib.Nifti1Header, out_warp.header).set_intent("vector", (), "")

    # return the warp
    return cast(nib.Nifti1Image, out_warp)


def build_low_pass_filter(
    TR_in_sec: float,
    critical_freq: float = 0.01,
    filter_order: int = 6,
) -> Tuple[npt.NDArray, npt.NDArray]:
    """Function for calculating filter parameters for low pass filter

    Parameters
    ----------
    TR_in_sec : float
        TR in seconds
    critical_freq : float, optional
        Critical frequency for low pass filter, by default 0.01
    filter_order : int, optional
        Filter order, by default 6

    Returns
    -------
    Tuple[npt.NDArray, npt.NDArray]
        Filter coefficients for IIR filter
    """

    fs = 1.0 / TR_in_sec  # Sampling frequency (Hz)
    fn = fs / 2.0  # Nyquist frequency (Hz)
    w0_cutoff = critical_freq / fn  # Normalized cutoff frequency
    b, a = signal.iirfilter(filter_order, w0_cutoff, btype="lowpass", output="ba", ftype="butter")
    return b, a


def compute_hausdorff_distance(image1: nib.Nifti1Image, image2: nib.Nifti1Image) -> float:
    """Compute the Hausdorff distance between two images

    Parameters
    ----------
    image1 : nib.Nifti1Image
        First image
    image2 : nib.Nifti1Image
        Second image

    Returns
    -------
    float
        Hausdorff distance
    """
    # get all data in RAS orientation
    to_canonical, from_canonical = get_ras_orient_transform(image1)

    # get to RAS orientation
    image1_ras = image1.as_reoriented(to_canonical)
    image2_ras = image2.as_reoriented(to_canonical)

    # get data
    image1_data = image1_ras.get_fdata()
    image2_data = image2_ras.get_fdata()

    # split affine into components
    image1_origin, image1_rotations, image1_zooms, _ = decompose44(image1_ras.affine)
    image2_origin, image2_rotations, image2_zooms, _ = decompose44(image2_ras.affine)

    # compute hausdorff distance
    hausdorff_distance = compute_hausdorff_distance_cpp(
        image1_data,
        image1_origin,
        image1_rotations,
        image1_zooms,
        image2_data,
        image2_origin,
        image2_rotations,
        image2_zooms,
    )

    # return hausdorff distance
    return hausdorff_distance


def compute_jacobian_determinant(displacement_field: nib.Nifti1Image) -> nib.Nifti1Image:
    """Compute the Jacobian determinant of a displacement field

    Parameters
    ----------
    displacement_field : nib.Nifti1Image
        Displacement field data in mm

    Returns
    -------
    nib.Nifti1Image
        Jacobian determinant of the displacement field
    """
    # get data in RAS orientation
    to_canonical, from_canonical = get_ras_orient_transform(displacement_field)

    # get to RAS orientation
    displacement_field_ras = displacement_field.as_reoriented(to_canonical)

    # get data
    data = displacement_field_ras.get_fdata()

    # split affine into components
    origin, rotations, zooms, _ = decompose44(displacement_field_ras.affine)

    # compute jacobian determinant
    jacobian_determinant = compute_jacobian_determinant_cpp(data, origin, rotations, zooms)

    # make new image
    jacobian_determinant_image = nib.Nifti1Image(
        jacobian_determinant, displacement_field_ras.affine, displacement_field_ras.header
    ).as_reoriented(from_canonical)

    # return jacobian determinant
    return cast(nib.Nifti1Image, jacobian_determinant_image)


def check_affines(phase, mag):
    """Make sure affines/shapes are all correct."""
    for p1, m1 in zip(phase, mag):
        for p2, m2 in zip(phase, mag):
            if not (
                np.allclose(p1.affine, p2.affine, rtol=1e-3, atol=1e-3)
                and np.allclose(m1.affine, m2.affine, rtol=1e-3, atol=1e-3)
                and p1.shape == p2.shape
                and m1.shape == m2.shape
            ):
                print(p1.affine, p2.affine)
                print(p1.affine - p2.affine)
                print(p1.shape, p2.shape)
                print(m1.affine, m2.affine)
                print(m1.affine - m2.affine)
                print(m1.shape, m2.shape)
                raise ValueError("Affines and shapes must match")
