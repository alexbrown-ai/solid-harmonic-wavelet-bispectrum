import argparse
import glob
import math as m
from typing import Dict, Tuple, List

import cv2
import h5py
import numpy as np
import pandas as pd
from astropy.io import fits

NUM_PROJECTIONS = 3


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Process TNG simulation FITS files into HDF5 format"
    )
    parser.add_argument("--f", "-files", required=True,
                        help="Glob pattern for input FITS files")
    parser.add_argument("--o", "-output", required=True,
                        help="Output HDF5 file path (without extension)")
    parser.add_argument("--l", "-lookback", required=True,
                        help="Path to lookback times Excel file")
    parser.add_argument("--d", "-dimensions", required=True, type=int,
                        help="Output image dimensions (creates square MxM images)")
    parser.add_argument("--r", "-radiusmultiplier", required=True, type=int,
                        help="Half stellar mass radius multiplier for cropping")
    parser.add_argument("--n", "-normalised", action='store_true',
                        help="Apply normalization to images")
    return parser.parse_args()


def cosmocal(z, h0=70, wm=0.3, wv=0.7, verbose=False, dictreturn=True):
    """
    Calculates cosmological parameters
    @type z: number
    @param z: redshift
    @type h0: number
    @param h0: Hubble Constant (Defaults to 70)
    @type wm: number
    @param wm: Matter Density (Defaults to 0.3)
    @type wv: number
    @param wv: Lambda Density (Defaults to 0.7)
    @type verbose: True/False
    @param verbose: True will print out information, False will return this information (Default False)
    @type dictreturn: True/False
    @param dictreturn: True will return dictionary, False will return list (Default True)
    @rtype: list/Dictionary/PrintOut Depending on Input
    @return: Years since the Big Bang, Comoving Radial Distance, Scale, Distance Module
    @note: This is Ned Wrights Cosmology Calculator
    (Input and output mildly modified) U{http://www.astro.ucla.edu/~wright/intro.html}
    """
    c = 299792.458  # velocity of light in km/sec
    tyr = 977.8  # coefficent for converting 1/H into Gyr
    h = h0 / 100.
    wr = 4.165E-5 / (h * h)  # includes 3 massless neutrino species, T0 = 2.72528
    wk = 1 - wm - wr - wv
    az = 1.0 / (1 + 1.0 * z)
    age = 0.
    n = 1000  # number of points in integrals
    for i in range(n):
        a = az * (i + 0.5) / n
        adot = m.sqrt(wk + (wm / a) + (wr / (a * a)) + (wv * a * a))
        age += 1. / adot
    zage = az * age / n
    zage_gyr = (tyr / h0) * zage
    dtt = 0.0
    dcmr = 0.0
    # do integral over a=1/(1+z) from az to 1 in n steps, midpoint rule
    for i in range(n):
        a = az + (1 - az) * (i + 0.5) / n
        adot = m.sqrt(wk + (wm / a) + (wr / (a * a)) + (wv * a * a))
        dtt += 1. / adot
        dcmr += 1. / (a * adot)

    dtt = (1. - az) * dtt / n
    dcmr = (1. - az) * dcmr / n
    age = dtt + zage
    age_gyr = age * (tyr / h0)
    dtt_gyr = (tyr / h0) * dtt
    dcmr_gyr = (tyr / h0) * dcmr
    dcmr_mpc = (c / h0) * dcmr
    # tangential comoving distance
    x = m.sqrt(abs(wk)) * dcmr
    if x > 0.1:
        if wk > 0:
            ratio = 0.5 * (m.exp(x) - m.exp(-x)) / x
        else:
            ratio = m.sin(x) / x
    else:
        y = x * x
        if wk < 0:
            y = -y
        ratio = 1. + y / 6. + y * y / 120.
    dcmt = ratio * dcmr
    da = az * dcmt
    da_mpc = (c / h0) * da
    kpc_da = da_mpc / 206.264806
    da_gyr = (tyr / h0) * da
    dl = da / (az * az)
    dl_mpc = (c / h0) * dl
    dl_gyr = (tyr / h0) * dl
    # comoving volume computation
    x = m.sqrt(abs(wk)) * dcmr
    if x > 0.1:
        if wk > 0:
            ratio = (0.125 * (m.exp(2. * x) - m.exp(-2. * x)) - x / 2.) / (x * x * x / 3.)
        else:
            ratio = (x / 2. - m.sin(2. * x) / 4.) / (x * x * x / 3.)
    else:
        y = x * x
        if wk < 0:
            y = -y
        ratio = 1. + y / 5. + (2. / 105.) * y * y
    vcm = ratio * dcmr * dcmr * dcmr / 3.
    v_gpc = 4. * m.pi * ((0.001 * c / h0) ** 3) * vcm
    dm_factor = 4 * np.pi * (dl_mpc * 3.08568025e24) ** 2
    if verbose == 1:
        print('For H_o = ' + '%1.1f' % h0 + ', Omega_M = ' + '%1.2f' % wm + ', Omega_vac = ', )
        print('%1.2f' % wv + ', z = ' + '%1.3f' % z)
        print('It is now ' + '%1.1f' % age_gyr + ' Gyr since the Big Bang.')
        print('The age at redshift z was ' + '%1.1f' % zage_gyr + ' Gyr.')
        print('The light travel time was ' + '%1.1f' % dtt_gyr + ' Gyr.')
        print('The comoving radial distance, which goes into Hubbles law, is', )
        print('%1.1f' % dcmr_mpc + ' Mpc or ' + '%1.1f' % dcmr_gyr + ' Gly.')
        print('The comoving volume within redshift z is ' + '%1.1f' % v_gpc + ' Gpc^3.')
        print('The angular size distance D_A is ' + '%1.1f' % da_mpc + ' Mpc or', )
        print('%1.1f' % da_gyr + ' Gly.')
        print('This gives a scale of ' + '%.2f' % kpc_da + ' kpc/".')
        print('The luminosity distance D_L is ' + '%1.1f' % dl_mpc + ' Mpc or ' + '%1.1f' % dl_gyr + ' Gly.')
        print('The distance modulus, m-M, is ' + '%1.2f' % (5 * np.log10(dl_mpc * 1e6) - 5))
    if not dictreturn:
        return ([zage_gyr, dcmr_mpc, kpc_da, (5 * np.log10(dl_mpc * 1e6) - 5)])
    else:
        return (
            {'ageAtZ': zage_gyr, 'ComovingDistanceMpc': dcmr_mpc, 'Scale': kpc_da,
             'DM': (5 * np.log10(dl_mpc * 1e6) - 5),
             'D_a': da_mpc, 'dl_mpc': dl_mpc, 'dm_factor': dm_factor, 'comvol': v_gpc})


def get_bounding_box_length(header: fits.Header,
                            multiple_of_r_half: int = 32) -> float:
    """
    Calculate the bounding box length for cropping based on half-mass radius.

    :param header: FITS file header containing galaxy properties
    :param multiple_of_r_half: Multiplier for the half-mass radius
    :return: Bounding box length in pixels
    """
    z = header['redshift']
    r_half = header['stellar_half_mass_radius'] * (1 + z) * 0.7  # kpc
    scale = cosmocal(z)['Scale']  # kpc/arcsec
    r_arcsec = r_half / scale

    imsize_arcsec = header['imsize']
    num_pixels = int(header["npixels"].split(",")[0])
    arcsec_per_pixel = imsize_arcsec / num_pixels

    return r_arcsec * arcsec_per_pixel * multiple_of_r_half


def arbitrary_crop(original_image: np.ndarray, crop_box_length: int) -> np.ndarray:
    """
    Crop a square region from the center of an image.

    :param original_image: Input image array
    :param crop_box_length: Side length of the square crop box
    :return: Cropped image
    """
    center = original_image.shape
    x = int(center[1] / 2 - crop_box_length / 2)
    y = int(center[0] / 2 - crop_box_length / 2)
    return original_image[y:y + crop_box_length, x:x + crop_box_length]


def resize(original_image: np.ndarray,
           output_shape: Tuple[int, int] = (64, 64)) -> np.ndarray:
    """
    Resize image using Lanczos interpolation.

    :param original_image: Input image
    :param output_shape: Target dimensions (width, height)
    :return: Resized image
    """
    return cv2.resize(original_image, output_shape, interpolation=cv2.INTER_LANCZOS4)


def clean(img: np.ndarray) -> np.ndarray:
    """
    Apply logarithmic transformation and handle invalid values.

    :param img: Input image array
    :return: Cleaned image with log transformation applied
    """
    img = np.log10(img)
    return np.nan_to_num(img, nan=0.0, posinf=np.max(img[img < np.inf]), neginf=0)


def log_normalise(img: np.ndarray) -> np.ndarray:
    """
    Normalize image to [0, 1] range.

    :param img: Input image array
    :return: Normalized image
    """
    min_x, max_x = np.min(img), np.max(img)
    if max_x > min_x:
        return (img - min_x) / (max_x - min_x)
    else:
        return img / img


def count_valid_files(file_paths: List[str]) -> int:
    """
    Count the number of valid FITS files with non-blank projections.

    :param file_paths: List of file paths to validate
    :return: Number of valid files
    """
    num_valid = 0
    for path in file_paths:
        try:
            with fits.open(path) as file:
                valid_projections = 0
                for i in range(1, NUM_PROJECTIONS + 1):
                    if not file[i].data.any():
                        break
                    image = file[i].data[0]
                    if image.std() < 1e-9:
                        break
                    valid_projections += 1

                if valid_projections == NUM_PROJECTIONS:
                    num_valid += 1
        except Exception:
            continue
    return num_valid


def process_image(image: np.ndarray, header: fits.Header,
                  half_stellar_mass_multiplier: int,
                  target_size: Tuple[int, int],
                  normalise: bool) -> Tuple[np.ndarray, int]:
    """
    Process a single image through the full pipeline.

    :param image: Input image array
    :param header: FITS header for calculating crop size
    :param half_stellar_mass_multiplier: Multiplier for half-mass radius
    :param target_size: Target output dimensions (M, N)
    :param normalise: Whether to apply normalization
    :return: Tuple of (processed image, bounding box length used)
    """
    bounding_box_length = get_bounding_box_length(
        header,
        multiple_of_r_half=half_stellar_mass_multiplier
    )
    bounding_box_length = int(np.rint(bounding_box_length))
    bounding_box_length = min(bounding_box_length, image.shape[0])

    cropped_img = arbitrary_crop(image, bounding_box_length)
    cleaned_img = clean(cropped_img)
    resized_img = resize(cleaned_img, target_size)

    if normalise:
        processed_img = log_normalise(resized_img)
    else:
        processed_img = resized_img

    return processed_img, bounding_box_length


def load_lookback_times(lookback_path: str) -> Dict[int, float]:
    """
    Load lookback times from Excel file.

    :param lookback_path: Path to lookback times Excel file
    :return: Dictionary mapping snapshot to lookback time
    """
    lookback_df = pd.read_excel(lookback_path)
    return lookback_df.set_index("snapshot")["lookback_time_planck15"].to_dict()


def create_hdf5_datasets(hdf5_file: h5py.File, num_files: int,
                         M: int, N: int) -> Dict[str, h5py.Dataset]:
    """
    Create HDF5 datasets for storing processed data.

    :param hdf5_file: Open HDF5 file handle
    :param num_files: Number of samples to store
    :param M: Image height
    :param N: Image width
    :return: Dictionary of created datasets
    """
    datasets = {
        'X': hdf5_file.create_dataset('X', shape=(num_files, NUM_PROJECTIONS, M, N),
                                      dtype=np.float32, chunks=(4, NUM_PROJECTIONS, M, N)),
        'Y_ratio': hdf5_file.create_dataset('Y_ratio', shape=(num_files,),
                                            dtype=np.float32, chunks=True),
        'Y_time': hdf5_file.create_dataset('Y_time', shape=(num_files,),
                                           dtype=np.float32, chunks=True),
        'Y_stellar_mass': hdf5_file.create_dataset('Y_stellar_mass', shape=(num_files,),
                                                   dtype=np.float32, chunks=True),
        'arcsec_per_pixel': hdf5_file.create_dataset('arcsec_per_pixel', shape=(num_files,),
                                                     dtype=np.float32, chunks=True),
        'bounding_box_length': hdf5_file.create_dataset('bounding_box_length', shape=(num_files,),
                                                        dtype=np.float32, chunks=True)
    }
    return datasets


def process_fits_file(file_path: str, header_only: bool = False):
    """
    Process a single FITS file and extract images and metadata.

    :param file_path: Path to FITS file
    :param header_only: If True, only validate without processing
    :return: Tuple of (images list, header, bounding_box_length) or None if invalid
    """
    try:
        with fits.open(file_path) as file:
            header = file[0].header

            if header_only:
                return None

            images = []
            n_pixel_original = int(header["npixels"].split(",")[0])

            for file_idx in range(1, NUM_PROJECTIONS + 1):
                if not file[file_idx].data.any():
                    return None

                image = file[file_idx].data[0]
                if image.std() < 1e-9:
                    return None

                images.append((image, n_pixel_original))

            return images, header
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None


def main():
    args = parse_arguments()

    save_path = args.o
    tng_files_pattern = args.f
    lookback_path = args.l
    normalise = args.n
    M = args.d
    N = M
    half_stellar_mass_multiplier = args.r

    # Generate output path
    save_path = f"{save_path}_{M}x{N}_halfstellar{half_stellar_mass_multiplier}.hdf5"

    # Get file list
    tng_files = glob.glob(tng_files_pattern)
    num_files = len(tng_files)
    print(f"Verifying {num_files} samples")

    # Count valid files
    num_valid_files = count_valid_files(tng_files)
    print(f"Processing {num_valid_files} valid samples")

    # Load lookback times
    snapshot_lookback = load_lookback_times(lookback_path)

    # Process files and save to HDF5
    with h5py.File(save_path, 'w') as hf:
        datasets = create_hdf5_datasets(hf, num_valid_files, M, N)

        valid_idx = 0
        for path in tng_files:
            result = process_fits_file(path)
            if result is None:
                continue

            images_data, header = result
            n_pixel_original = int(header["npixels"].split(",")[0])
            arcsec_per_pixel_original = header['imsize'] / n_pixel_original

            processed_images = []
            bounding_box_length = None

            for image, _ in images_data:
                processed_img, bbox_len = process_image(
                    image, header, half_stellar_mass_multiplier, (M, N), normalise
                )
                processed_images.append(processed_img)
                bounding_box_length = bbox_len

            # Store results
            datasets['X'][valid_idx] = np.array(processed_images)
            datasets['Y_ratio'][valid_idx] = header["ratio_last"]
            datasets['Y_stellar_mass'][valid_idx] = header["stellar_mass"]

            img_snapshot = header["snapshot"]
            merger_snapshot = header["snap_last"]
            time_since_last = abs(
                snapshot_lookback[img_snapshot] - snapshot_lookback[merger_snapshot]
            )
            datasets['Y_time'][valid_idx] = time_since_last
            datasets['bounding_box_length'][valid_idx] = bounding_box_length

            final_arcsec_per_pixel = arcsec_per_pixel_original * bounding_box_length / M
            datasets['arcsec_per_pixel'][valid_idx] = final_arcsec_per_pixel

            valid_idx += 1
            if valid_idx % 100 == 0:
                hf.flush()

        print("Applying log transformations...")
        Y_stellar_mass_data = datasets['Y_stellar_mass'][:]
        Y_ratio_data = datasets['Y_ratio'][:]

        datasets['Y_stellar_mass'][:] = np.log10(Y_stellar_mass_data)
        datasets['Y_ratio'][:] = np.log10(Y_ratio_data)

    print(f"\nData saved to {save_path}")
    print(f"X: {(num_valid_files, NUM_PROJECTIONS, M, N)}")
    print(f"Stellar mass: {(num_valid_files,)}")
    print(f"Time since last merger: {(num_valid_files,)}")
    print(f"Mass ratio: {(num_valid_files,)}")

    # Verify saved data
    print("\nVerifying saved data...")
    with h5py.File(save_path, 'r') as hf:
        print("Datasets in file:", list(hf.keys()))
        for key in hf.keys():
            print(f"{key}: shape={hf[key].shape}, dtype={hf[key].dtype}")


if __name__ == "__main__":
    main()
