"""SHWB/SHWBic"""
import torch
from kymatio.scattering2d.backend.torch_backend import TorchBackend2D
from kymatio.scattering3d.backend.torch_backend import TorchBackend3D


def get_wavelet_responses(x, filters, L, J):
    calculated_coefficients = {}
    signal_shape = x.shape[-2:]
    x = x.reshape((-1,) + signal_shape + (1,))

    U_0_c = TorchBackend2D.rfft(x)
    for l in range(L + 1):
        for j1 in range(J + 1):
            f = filters[l][j1]
            calculated_coefficients[(l, j1)] = TorchBackend2D.cdgmm(U_0_c.to(f.dtype), f)

    return calculated_coefficients


def bispectral_coefficients(x, filters, L, J, powers, bicoherence=True):
    batch_shape = x.shape[0]
    calculated_coefficients = get_wavelet_responses(x, filters, L, J)

    bispectral_coef = []
    for j in range(J + 1):
        for l1 in range(L + 1):
            # l1 < l2 due to symmetry - redundant information otherwise
            for l2 in range(l1 + 1, L + 1):
                l3 = l1 + l2
                if l3 <= L:
                    l1_coef = calculated_coefficients[(l1, j)]
                    l2_coef = calculated_coefficients[(l2, j)]
                    l3_coef = calculated_coefficients[(l3, j)]
                    if not bicoherence:
                        norm = 2 ** j
                        l1_coef /= norm
                        l2_coef /= norm
                        l3_coef /= norm

                    bispectrum = torch.abs(TorchBackend2D.cdgmm(
                        TorchBackend2D.cdgmm(l1_coef, l2_coef),
                        torch.conj(l3_coef)
                    ))

                    if bicoherence:
                        l1_norm = TorchBackend3D.compute_integrals(torch.abs(l1_coef), [2])
                        l2_norm = TorchBackend3D.compute_integrals(torch.abs(l2_coef), [2])
                        l3_norm = TorchBackend3D.compute_integrals(torch.abs(l3_coef), [2])

                        product_norm = torch.sqrt(l1_norm * l2_norm * l3_norm) + 1e-10

                        product_norm_view = product_norm.view(-1, 1, 1, 1)
                        bispectrum /= product_norm_view

                    integrals = compute_integrals(bispectrum, powers)
                    bispectral_coef.append(integrals)

    bispectral_coef = torch.concat(bispectral_coef, dim=1)
    bispectral_coef = bispectral_coef.reshape(batch_shape, -1)

    return bispectral_coef


def compute_integrals(input_array, integral_powers):
    integrals = torch.zeros((input_array.shape[0], len(integral_powers)),
            device=input_array.device)
    for i_q, q in enumerate(integral_powers):
        integrals[:, i_q] = ((input_array ** q).view(
            input_array.shape[0], -1).sum(1)) ** (1 / q)  # Norms

    return integrals

