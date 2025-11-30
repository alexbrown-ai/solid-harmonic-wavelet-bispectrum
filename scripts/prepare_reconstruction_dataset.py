"""Prepare HDF5 dataset for image reconstruction experiments."""
import os

import cv2
import h5py
import numpy as np
import torch
from tqdm import tqdm

from wavelets import SolidHarmonicModel


def create_hdf5_dataset(dataset_path, write_path, dim=128, batch_size=32, scattering_config=None):
    """
    Create HDF5 dataset with images and pre-computed scattering coefficients for reconstruction tasks.

    :param dataset_path: Path to directory containing image files (.jpg)
    :param write_path: Path to save HDF5 file
    :param dim: Image dimension (assumes square images)
    :param batch_size: Batch size for processing
    :param scattering_config: Configuration dict for scattering transform
    """
    if os.path.exists(write_path):
        print(f"Dataset already exists at {write_path}")
        return

    if scattering_config is None:
        scattering_config = {
            "L": 15,
            "J": 4,
            "sigma": 1.5,
            "lp_norms": [0.5, 1, 2, 4],
            "bispectrum": True,
            "bicoherence": True,
            "reduce": True
        }

    scattering = SolidHarmonicModel(
        M=dim, N=dim,
        scattering_config=scattering_config,
        higher_order_config=scattering_config.copy()
    )

    image_files = [f for f in os.listdir(dataset_path) if f.lower().endswith('.jpg')]
    print(f"Processing {len(image_files)} images")

    sample_input = torch.zeros((1, dim, dim))
    scattered_shape = scattering.scatter(sample_input).shape[1:]
    print(f"Scattering output shape: {scattered_shape}")

    with h5py.File(write_path, 'w') as hf:
        images_ds = hf.create_dataset(
            'images',
            shape=(len(image_files), dim, dim),
            dtype=np.float32,
            chunks=(batch_size, dim, dim),
            shuffle=True
        )
        scattered_ds = hf.create_dataset(
            'scattering_coef',
            shape=(len(image_files), *scattered_shape),
            dtype=np.float32,
            chunks=(batch_size, *scattered_shape),
            shuffle=True
        )

        for start in tqdm(range(0, len(image_files), batch_size), desc="Processing batches"):
            batch_files = image_files[start:start + batch_size]
            batch_imgs = []

            for filename in batch_files:
                img_path = os.path.join(dataset_path, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"Warning: Could not read {img_path}, skipping")
                    continue
                img = cv2.resize(img, (dim, dim), interpolation=cv2.INTER_LANCZOS4)
                img = img.astype(np.float32) / 255.0
                batch_imgs.append(img)

            if not batch_imgs:
                continue

            batch_imgs = np.stack(batch_imgs, axis=0)
            end_idx = start + len(batch_imgs)
            images_ds[start:end_idx] = batch_imgs

            img_tensor = torch.from_numpy(batch_imgs)
            scattered = scattering.scatter(img_tensor).cpu().numpy()
            scattered_ds[start:end_idx] = scattered

    print(f"Dataset saved to {write_path}")
    print(f"Final dataset contains {len(image_files)} samples")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare HDF5 dataset for reconstruction")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to directory with images or NPZ file")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to save HDF5 file")
    parser.add_argument("--dim", type=int, default=128, help="Image dimension")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--from_npz", action="store_true",
                        help="Load from NPZ file instead of image directory")
    parser.add_argument("--L", type=int, default=15, help="Maximum angular frequency")
    parser.add_argument("--J", type=int, default=4, help="Maximum scale")
    parser.add_argument("--sigma", type=float, default=1.5, help="Base scale parameter")

    args = parser.parse_args()

    scattering_config = {
        "L": args.L,
        "J": args.J,
        "sigma": args.sigma,
        "lp_norms": [0.5, 1, 2, 4],
        "bispectrum": True,
        "bicoherence": True,
        "reduce": True
    }

    create_hdf5_dataset(
        args.dataset_path, args.output_path,
        dim=args.dim, batch_size=args.batch_size,
        scattering_config=scattering_config
    )
