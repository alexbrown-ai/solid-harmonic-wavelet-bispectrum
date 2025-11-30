"""SHWS, SHWSB, SHWSBic"""

import torch
from kymatio.scattering2d.backend.torch_backend import TorchBackend2D
from kymatio.scattering3d.backend.torch_backend import TorchBackend3D


def scattering(x, filters, L, J, max_order, powers, bispectrum=False, bicoherence=True):
    """
    Compute SHWS, SHWSB or SHWSBic

    :param x: Input tensor of shape (batch, height, width) or (batch, channels, height, width)
    :param filters: solid harmonic wavelet filters
    :param L: Maximum angular frequency
    :param J: Maximum scale
    :param max_order: Maximum scattering order (1 or 2)
    :param powers: List of Lp norm powers to compute
    :param bispectrum: If True, compute bispectrum; if False, compute standard scattering
    :param bicoherence: If True, normalise bispectrum by bicoherence

    :return: Scattering coefficients
    """
    if bispectrum:
        return scattering_bispectrum(x, filters, L, J, max_order, powers, bicoherence=bicoherence)

    return scattering2d(x, filters, L, J, max_order, powers)


def scattering2d(x, filters, L, J, max_order, powers):
    batch_shape = x.shape[:-2]
    signal_shape = x.shape[-2:]
    x = x.reshape((-1,) + signal_shape + (1,))

    s_order_1, s_order_2 = [], []
    U_0_c = TorchBackend2D.rfft(x)
    for l in range(L + 1):
        s_order_1_l, s_order_2_l = [], []
        for j1 in range(J + 1):
            f = filters[l][j1]
            U_1_c = TorchBackend2D.ifft(TorchBackend2D.cdgmm(U_0_c, f.to(U_0_c.dtype)))
            U_1_m = TorchBackend2D.modulus(U_1_c)
            S_1_l = compute_integrals(U_1_m, powers)
            s_order_1_l.append(S_1_l)

            if max_order > 1:
                U_1_c = TorchBackend2D.rfft(U_1_m)

                for j2 in range(j1 + 1, J + 1):
                    f2 = filters[l][j2]
                    U_2_c = TorchBackend2D.ifft(TorchBackend2D.cdgmm(U_1_c, f2.to(U_1_c.dtype)))
                    U_2_m = TorchBackend2D.modulus(U_2_c)
                    S_2_l = compute_integrals(U_2_m, powers)
                    s_order_2_l.append(S_2_l)

        s_order_1.append(s_order_1_l)
        if max_order == 2:
            s_order_2.append(s_order_2_l)

    S = s_order_1 if max_order == 1 else [x + y for x, y in zip(s_order_1, s_order_2)]
    S = torch.stack([x for y in zip(*S) for x in y], dim=1)
    S = S.reshape((S.shape[0], S.shape[1] // (L + 1), (L + 1)) + S.shape[2:])
    scattering_shape = S.shape[1:]
    S = S.reshape(batch_shape + scattering_shape)
    return S.reshape(S.shape[0], -1)



def scattering_bispectrum(x, filters, L, J, max_order, powers, chunk_size=64, bicoherence=True, reduce=True):
    batch_size = x.shape[0]
    num_chunks = (batch_size + chunk_size - 1) // chunk_size

    results = []
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, batch_size)
        chunk = x[start_idx:end_idx]

        chunk_result = scattering_bispectrum_chunk(chunk, filters, L, J, max_order, powers, bicoherence)
        results.append(chunk_result)

    return torch.cat(results, dim=0)


def scattering_bispectrum_chunk(x, filters, L, J, max_order, powers, bicoherence):
    batch_shape = x.shape[:-2]
    signal_shape = x.shape[-2:]
    x = x.reshape((-1,) + signal_shape + (1,))

    s_order_1 = [[None for _ in range(J + 1)] for _ in range(L + 1)]
    s_order_2 = [[[None for _ in range(J + 1)] for _ in range(J + 1)] for _ in range(L + 1)]

    U_0_c = TorchBackend2D.rfft(x)

    for l in range(L + 1):
        for j1 in range(J + 1):
            f = filters[l][j1]
            U_1_c = TorchBackend2D.cdgmm(U_0_c, f.to(U_0_c.dtype))
            s_order_1[l][j1] = U_1_c

            if max_order > 1:
                for j2 in range(j1 + 1, J + 1):
                    f2 = filters[l][j2]
                    U_1_m = TorchBackend2D.rfft(TorchBackend2D.modulus(TorchBackend2D.ifft(U_1_c)))
                    s_order_2[l][j1][j2] = TorchBackend2D.cdgmm(U_1_m, f2.to(U_1_m.dtype))

    bispectral = []

    for l1 in range(L + 1):
        for l2 in range(l1 + 1, L + 1):
            l3 = l1 + l2
            if l3 <= L:
                for j1 in range(J + 1):
                    l1_j1 = s_order_1[l1][j1]
                    l2_j1 = s_order_1[l2][j1]
                    l3_j1 = s_order_1[l3][j1]
                    bispectral.append(compute_bispectrum(l1_j1, l2_j1, l3_j1, powers, bicoherence))


                    for j2 in range(j1 + 1, J + 1):
                        l1_j2 = s_order_2[l1][j1][j2]
                        l2_j2 = s_order_2[l2][j1][j2]
                        l3_j2 = s_order_2[l3][j1][j2]
                        bispectral.append(compute_bispectrum(l1_j2, l2_j2, l3_j2, powers, bicoherence))

    B = torch.cat(bispectral, dim=1)
    return B.reshape(batch_shape[0], -1)


def compute_bispectrum(l1, l2, l3, powers, bicoherence=True):
    bispectrum = torch.abs(TorchBackend2D.cdgmm(
        TorchBackend2D.cdgmm(l1, l2),
        torch.conj(l3)
    ))

    if bicoherence:
        l1_norm = TorchBackend3D.compute_integrals(torch.abs(l1), [2])
        l2_norm = TorchBackend3D.compute_integrals(torch.abs(l2), [2])
        l3_norm = TorchBackend3D.compute_integrals(torch.abs(l3), [2])

        product_norm = torch.sqrt(l1_norm * l2_norm * l3_norm) + 1e-10
        bispectrum /= product_norm.view(-1, 1, 1, 1)

    return compute_integrals(bispectrum, powers)



def compute_integrals(input_array, integral_powers):
    integrals = torch.zeros((input_array.shape[0], len(integral_powers)), device=input_array.device)
    for i_q, q in enumerate(integral_powers):
        integrals[:, i_q] = ((input_array ** q).view(
            input_array.shape[0], -1).sum(1)) ** (1 / q)  # Norms

    return integrals
