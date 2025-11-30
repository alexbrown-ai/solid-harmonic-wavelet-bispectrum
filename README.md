# Solid Harmonic Wavelet Bispectrum

Welcome to the official repository for the **Solid Harmonic Wavelet Bispectrum**.

## Related Publications

**Solid Harmonic Wavelet Bispectrum for Image Analysis**  
*Wiley Advanced Science* (October 2025)  
DOI: [10.1002/advs.202517383](https://doi.org/10.1002/advs.202517383)

## Installation

This package uses Poetry for dependency management. To install:

```bash
poetry install
```

Or install with pip:

```bash
pip install -e .
```

Or install from requirements.txt:

```bash
pip install -r requirements.txt
```

## Dependencies

- Python >= 3.11, < 3.15
- numpy >= 2.3.5
- scipy >= 1.16.3
- torch >= 2.9.1
- kymatio >= 0.3.0
- h5py, astropy, pandas, opencv-python, scikit-learn

## Datasets

Note datasets are too big to be hosted using git lfs.

* 'tng_as_sdss' - refers to the original dataset as used in galaxy classification: https://arxiv.org/abs/2401.09632
* Euclid refers to images sampled from real world survey
* 'tng_simulated_as_hst' - preprocessed, unnormalised TNG samples mocked as HST

Datasets can be shared upon request.

## Package Structure

- `wavelets/`: Solid harmonic wavelet implementations
  - `solid_harmonic_wavelets.py`: Core wavelet functions
- `scattering/`: Scattering transform implementations
  - `scattering_torch.py`: PyTorch-based scattering and bispectrum computation
  - `coefficients.py`: Utilities for computing scattering coefficients
  - `higher_order_spectra.py`: Bispectrum and bicoherence computation
- `models/`: Model implementations
  - `reconstruction/`: Autoencoder models for image reconstruction
    - `autoencoder.py`: Scattering-based autoencoder architectures
  - `classical/`: Classical CNN models
  - `hybrid/`: Hybrid scattering + CNN models
- `scripts/`: Data processing and analysis scripts
  - `fit_models.py`: Model fitting utilities
  - `train_autoencoder.py`: Training script for reconstruction autoencoders
  - `prepare_reconstruction_dataset.py`: Dataset preparation for reconstruction experiments
  - `obs_realism_hst.py`: HST observation realism simulation
  - `sky.py`: Sky background and image processing utilities
  - `h5_pre_process.py`: HDF5 data preprocessing

## License

MIT License

## Citation

If you use this code in your research, please cite:

TODO: update the citation below accordingly

```bibtex
@article{brown2025solidharmonicwaveletbispectrum,
  title={Solid Harmonic Wavelet Bispectrum for Image Analysis},
  author={Brown, Alex and Avirett-Mackenzie, Mathilda and Villforth, Carolin and Exarchakis, Georgios},
  journal={Wiley Advanced Science},
  year={2025},
  doi={10.1002/advs.202517383}
}
```
