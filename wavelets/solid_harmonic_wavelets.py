import numpy as np
from scipy.special import gamma


def solid_harmonic_2d(M, N, sigma, l, fourier=True):
    grid = np.fft.ifftshift(
        np.mgrid[
        -M // 2: -M // 2 + M,
        -N // 2: -N // 2 + N
        ].astype("float32"),
        axes=(1, 2)
    )

    _sigma = sigma

    if fourier:
        grid[0] *= 2 * np.pi / M
        grid[1] *= 2 * np.pi / N
        _sigma = 1. / sigma

    r = np.sqrt((grid ** 2).sum(0))
    phi = np.arctan2(grid[1], grid[0])

    if fourier:
        norm = np.sqrt(4 * np.pi * (sigma ** (2 * (l + 1))) / gamma(l + 1))
    else:
        norm = 1. / np.sqrt(np.pi * (sigma ** (2 * (l + 1))) * gamma(l + 1))

    gaussian = np.exp(-0.5 * (r ** 2) / _sigma ** 2).astype('complex64')

    radial = r ** l
    angular = np.exp(1j * l * phi).astype('complex64')

    if l == 0:
        wavelet = norm * gaussian
    else:
        wavelet = norm * gaussian * radial * angular

    if not fourier:
        wavelet = np.fft.fftshift(wavelet)

    return wavelet


def solid_harmonic_filter_bank(M, N, J, L, sigma_0, fourier=True):
    filters = []
    for l in range(L + 1):
        filters_l = np.zeros((J + 1, M, N), dtype="complex128")
        for j in range(J + 1):
            sigma = sigma_0 * 2 ** j
            filters_l[j, ...] = solid_harmonic_2d(M, N, sigma, l, fourier=fourier)
        filters.append(filters_l)
    return filters
