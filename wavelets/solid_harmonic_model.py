import torch

from scattering import scattering_torch as scattering
from wavelets import solid_harmonic_wavelets as wvlts
from scattering import higher_order_spectra as hos


def _init_config(config, bispectrum_default=None):
    if config is None:
        return {
            'sigma': None, 'J': None, 'L': None, 'lp_norms': None,
            'bispectrum': bispectrum_default, 'bicoherence': True
        }

    return {
        'sigma': config['sigma'],
        'J': config['J'],
        'L': config['L'],
        'lp_norms': config['lp_norms'],
        'bispectrum': config.get('bispectrum', bispectrum_default),
        'bicoherence': config.get('bicoherence', True)
    }


class SolidHarmonicModel:
    def __init__(self, M, N, scattering_config=None, higher_order_config=None):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.M = M
        self.N = N

        self.scattering = _init_config(scattering_config, bispectrum_default=False)
        self.higher_order = _init_config(higher_order_config)

        self.scattering_filters = self._build_filters(self.scattering)
        self.higher_order_filters = self._build_filters(self.higher_order)

    def _build_filters(self, config):
        if config['J'] is None:
            return None

        filters = wvlts.solid_harmonic_filter_bank(
            M=self.M, N=self.N, J=config['J'], L=config['L'],
            sigma_0=config['sigma'], fourier=True
        )

        return [self._numpy_to_torch_complex(f) for f in filters]

    def _numpy_to_torch_complex(self, array):
        tensor = torch.zeros(array.shape + (2,), device=self.device)
        tensor[..., 0] = torch.from_numpy(array.real)
        tensor[..., 1] = torch.from_numpy(array.imag)
        return tensor

    def scatter(self, input_data, max_order=2):
        if self.scattering['J'] is None:
            raise ValueError("Not initialized for scattering")

        input_data = input_data.to(self.device).contiguous()
        return scattering.scattering(
            input_data, self.scattering_filters, self.scattering['L'],
            self.scattering['J'], max_order, self.scattering['lp_norms'],
            self.scattering['bispectrum'], self.scattering['bicoherence']
        )

    def bispectrum(self, input_data):
        if self.higher_order['J'] is None:
            raise ValueError("Not initialized for higher-order spectra")

        input_data = input_data.to(self.device).contiguous()
        return hos.bispectral_coefficients(
            input_data, self.higher_order_filters, self.higher_order['L'],
            self.higher_order['J'], self.higher_order['lp_norms'],
            self.higher_order['bicoherence']
        )