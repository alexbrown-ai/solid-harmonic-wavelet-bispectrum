import os

import h5py
import numpy as np


def JyMagAB(flux):
    """
    Source: Dr Carolin Villforth

    Calculates AB Magnitudes from Janksy
    @type flux: number or numpy array
    @param flux: Input Flux in Janksy
    @return: AB Magnitude
    """
    mag = 2.5 * (23 - np.log10(flux)) - 48.6
    return (mag)


def MagABJy(mag):
    """
    Source: Dr Carolin Villforth

    Calculates Jansky Flux from AB Magnitudes
    @type mag: number or numpy array
    @param mag: In put AB magnitude
    @return: Flux in Jansky
    """
    flux = 10 ** (23 - (mag + 48.6) / 2.5)
    return (flux)


# SKY FIRST
def ObsRealism(image, arcsec_per_pixel, exposure_time, return_flux):
    """
    Apply HST observation realism to an image.
    
    Adapted from Mathilda Avirett-Mackenzie's HST realism script.
    This version is usable on preprocessed (cropped, resized, normalised) HST h5 datasets
    which were generated using the h5_pre_process.py script.

    It is more efficient to apply noise to already pre-processed datasets, as opposed to
    including in the full pipeline which would incur large overheads for I/O and repeatedly
    resizing large images for each noise-level dataset.

    :param image: Input image (assumes log scale, will be converted)
    :param arcsec_per_pixel: Angular scale in arcseconds per pixel
    :param exposure_time: Exposure time in seconds
    :param return_flux: If True, return flux; if False, return AB magnitude
        
    :return: Processed image with observational realism applied
        
    Note:
    * Assumes data has been cleaned AND not normalised prior to use (nan values clipped etc).
    * This script applies the necessary normalisation.
    """
    # properties of the HST filters ~ 'wfc3_ir_f160w'
    image = 10 ** image  # if image on a log scale beforehand

    sky_level = 25  # mag/arcsec2
    image_jy = MagABJy(image)
    sky_jy = MagABJy(sky_level)
    image_jy += sky_jy
    image = JyMagAB(image_jy)

    # print(image.min(), image.max())
    hst_gain = 2.5  # [e/DN]
    hst_lambda_eff = 15278.47  # [A]
    hst_photflam = 1.9429e-20  # erg cm**-2 A**-1 e**-1

    speed_of_light = 2.99792458e8  # speed of light [m/s]

    # convert from surface brightness to flux
    img_flux = 10 ** (-0.4 * (image + 48.6))  # [erg/(cm**2 s Hz)/arcsec**2]
    img_flux *= (speed_of_light * 1e10 / hst_lambda_eff ** 2)
    img_flux *= arcsec_per_pixel ** 2  # [erg/(cm**2 s A)]

    img_counts = img_flux / hst_photflam * exposure_time
    img_counts = np.clip(img_counts, a_min=0, a_max=None)
    img_counts = np.nan_to_num(img_counts)

    # Calculate electron counts
    electron_counts = img_counts * hst_gain
    electron_counts = np.clip(electron_counts, 0, 1e8)

    img_counts = np.random.poisson(lam=electron_counts) / hst_gain
    # convert back to flux
    img_flux = img_counts * hst_photflam / exposure_time

    # convert back to surface brightness
    img_flux /= arcsec_per_pixel ** 2
    # convert back to AB magnitude
    img_flux *= hst_lambda_eff ** 2 / (speed_of_light * 1e10)
    mask = img_flux <= 0
    if np.any(mask):
        min_positive = np.min(img_flux[~mask]) if np.any(~mask) else 1e-40
        img_flux[mask] = min_positive * 1e-6

    if return_flux:
        return img_flux
    return JyMagAB(img_flux)


def normalise(img):
    min_x, max_x = np.min(img), np.max(img)
    if max_x > min_x:
        return (img - min_x) / (max_x - min_x)
    else:
        return np.ones_like(img) if max_x > 0 else img


def apply_obs_realism_streaming(input_path, output_prefix, exposure_time, chunk_size=128, log_scale=True, return_flux=True):
    """
    :param input_path: Path to input HDF5 file
    :param output_prefix: Prefix for output file name
    :param exposure_time: Exposure time in seconds
    :param chunk_size: Number of samples to process at once
    :param log_scale: If True, input is on log scale
    :param return_flux: If True, return flux; if False, return AB magnitude
    :return: HDF5 dataset handle for processed data
    """
    output_path = (f"../datasets/tng_simulated_as_hst/{output_prefix}-{exposure_time}seconds"
                   f"_{'log' if log_scale else 'linear'}"
                   f"_{'flux' if return_flux else 'ABmag'}"
                   f"_HST_128x128_halfstellar32.hdf5")
    if os.path.exists(output_path):
        with h5py.File(output_path, 'r') as f:
            if 'X' in f:
                print("Existing dataset.")
                return f

    with h5py.File(input_path, 'r') as input_file:
        X = input_file["X"]
        arcsec_per_pixel = input_file["arcsec_per_pixel"][:]
        num_samples, num_projections, M, N = X.shape

        with h5py.File(output_path, 'w') as output_file:
            X_realistic = output_file.create_dataset(
                'X', shape=(num_samples, num_projections, M, N),
                dtype=np.float32, chunks=(chunk_size, num_projections, M, N)
            )

            for key in ['Y_ratio', 'Y_time', 'Y_stellar_mass', 'redshift', 'snapshot', 'arcsec_per_pixel']:
                output_file.create_dataset(key, data=input_file[key][:])

            for i in range(0, num_samples, chunk_size):
                end_idx = min(i + chunk_size, num_samples)

                X_chunk = X[i:end_idx]
                arcsec_chunk = arcsec_per_pixel[i:end_idx]

                for sample_idx in range(X_chunk.shape[0]):
                    for proj_idx in range(num_projections):
                        im = ObsRealism(
                            X_chunk[sample_idx, proj_idx],
                            arcsec_chunk[sample_idx],
                            exposure_time,
                            return_flux
                        )
                        if log_scale:
                            im = np.log10(im)
                        X_chunk[sample_idx, proj_idx] = normalise(im)

                X_realistic[i:end_idx] = X_chunk

                if (i + 1) % 2500 == 0:
                    print(f"Processed samples {i + 1}-{end_idx} / {num_samples}")

    return h5py.File(output_path, 'r')['X']
