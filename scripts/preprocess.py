import os

import cv2
import numpy as np
from astropy.io import fits

from scripts import sky


def remove_sky(image):
    _image = image
    sky_mask = sky.sigma_clip(_image)
    sky_med = np.nanmedian(_image[sky_mask])
    return np.clip(_image, sky_med, np.nanpercentile(_image, 99))


def normalise_image(img):
    img = np.nan_to_num(img, nan=0.0, posinf=np.max(img[img < np.inf]), neginf=0)
    img = np.log10(img + np.abs(np.min(img)) + 0.1)
    min_x, max_x = np.min(img), np.max(img)
    if max_x > min_x:
        return (img - min_x) / (max_x - min_x)
    else:
        return img / img


def prep(original_image, output_shape=(64, 64)):
    return cv2.resize(original_image, output_shape, interpolation=cv2.INTER_LANCZOS4)


def preprocess(input_path, save_path, M=64, N=64):
    tng_files = input_path

    if os.path.exists(save_path):
        print("Loading saved data...")
        data = np.load(save_path)
        X = data['X']
        Y_ratio = data['Y_ratio']
        Y_time = data['Y_time']
        Y_is_merger = data['Y_is_merger']
        Y_stellar_mass = data['Y_stellar_mass']
    else:
        print("Processing data...")

        X, Y_ratio, Y_time, Y_is_merger, Y_stellar_mass = [], [], [], [], []

        for path in tng_files:
            file = fits.open(path)
            header = file[0].header

            ratio_last = header["ratio_last"]
            ratio_biggest = header["ratio_biggest"]
            time_last = header["dt_last"]
            time_biggest = header["dt_biggest"]

            not_merger = not (ratio_biggest > 0.01 and time_last <= 2)
            is_merger = ratio_biggest > 0.25 and time_biggest < 0.5
            if not (not_merger or is_merger):
                print(f"ratio: {ratio_biggest:.4f}, dt_last: {time_last:.2f}, dt_biggest: {time_biggest:.2f}")
                continue

            for file_idx in range(1, 4):

                channels = []
                for ch in range(1, 4):
                    img = file[file_idx].data[ch]
                    channels.append((normalise_image(prep(remove_sky(img), output_shape=(M, N)))))

                Y_ratio.append(ratio_biggest)
                Y_time.append(time_last)
                Y_is_merger.append(1 if is_merger else 0)
                Y_stellar_mass.append(header["stellar_mass"])

                X.append(np.array(channels))

        Y_ratio, Y_time, Y_is_merger, Y_stellar_mass = np.array(Y_ratio), np.array(Y_time), np.array(
            Y_is_merger), np.array(Y_stellar_mass)
        Y_ratio = np.log10(Y_ratio)

        X = np.array(X)
        np.savez(save_path, X=X, Y_ratio=Y_ratio, Y_time=Y_time, Y_is_merger=Y_is_merger, Y_stellar_mass=Y_stellar_mass)
        print(f"Data saved to {save_path}")

    return X, Y_ratio, Y_time, Y_is_merger, Y_stellar_mass
