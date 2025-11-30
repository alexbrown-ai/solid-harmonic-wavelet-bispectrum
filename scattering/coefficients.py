"""Utilities for calculating scattering coefficients."""

import os

import numpy as np
import torch
from kymatio.torch import Scattering2D

from wavelets.solid_harmonic_model import SolidHarmonicModel

strategies = {
    "scattering": lambda batch, scat: scat.scatter(batch),
    "bispectrum": lambda batch, scat: scat.bispectrum(batch),
}


def calculate_coefficients(X, y, M, N, scattering_config, higher_order_config, output_name, strategy, verbose=False):
    """
    Calculate scattering coefficients using solid harmonic wavelets.
    
    :param X: Input images array of shape (n_samples, height, width) or (n_samples, channels, height, width)
    :param y: Target labels/values
    :param M: Image height
    :param N: Image width
    :param scattering_config: Configuration dictionary for scattering transform
    :param higher_order_config: Configuration dictionary for higher-order statistics
    :param output_name: Name for cached coefficients file
    :param strategy: Higher-order statistics type (e.g., "scattering" ¬ shws, shwsb, "bispectrum" ¬ shwb, shwbic)
    :param verbose: If True, print progress messages
        
    Returns:
        Tuple of (scattering_coefficients, y)
    """
    lp_norms = scattering_config["lp_norms"]
    path = f"{output_name}/{M}x{N}_scattering_{str(scattering_config.values())}_higher_order_{str(higher_order_config.values())}_{strategy}"
    full_path = f"./cached-coefficients/{path}.npz"
    os.makedirs(os.path.dirname(full_path), exist_ok=True)

    if os.path.exists(full_path):
        data = np.load(full_path)
        scattering_coef, y = data["X"], data["Y"]
        if verbose:
            print("Loading coef")
    else:
        if verbose:
            print("Calculating coef")

        batch_size = 32
        n_samples = len(X)
        n_batches = int(np.ceil(n_samples / batch_size))

        order_0, orders_1_2 = [], []
        scattering = SolidHarmonicModel(
            M=M, N=N, 
            scattering_config=scattering_config,
            higher_order_config=higher_order_config
        )

        for i in range(n_batches):
            start = i * batch_size
            end = min(start + batch_size, n_samples)

            batch = torch.from_numpy(X[start:end])
            if verbose:
                print(f"Processing batch {i + 1}/{n_batches}")

            zeroth_order = compute_integrals(batch, lp_norms)
            order_0.append(zeroth_order.cpu().numpy())

            strategy_func = strategies.get(strategy)
            if strategy_func is None:
                raise ValueError(
                    f"Unknown strategy: {strategy}. "
                    f"Available options: {list(strategies.keys())}"
                )
            first_second_orders = strategy_func(batch, scattering)
            orders_1_2.append(first_second_orders.cpu().numpy())

        order_0 = np.concatenate(order_0, axis=0)
        orders_1_2 = np.concatenate(orders_1_2, axis=0)

        scattering_coef = np.concatenate([order_0, orders_1_2], axis=1)
        np.savez(full_path, X=scattering_coef, Y=y)

    return scattering_coef, y


def calculate_coefficients_morlet(X, M, N, L, J, output_name, verbose=False):
    path = f"L_{L}_J_{J}_dim_{M}x{N}_{output_name}"
    full_path = f"./morlet-cached-coefficients/{path}.npy"
    os.makedirs(os.path.dirname(full_path), exist_ok=True)

    if os.path.exists(full_path):
        results = np.load(full_path)
        if verbose: print("Loading coef")
    else:
        if verbose: print("Calculating coef")

        batch_size = 32
        n_samples = len(X)
        n_batches = int(np.ceil(n_samples / batch_size))

        results = []
        scattering = Scattering2D(J=J, L=L, shape=(M, N), max_order=2)

        for i in range(n_batches):
            start = i * batch_size
            end = min(start + batch_size, n_samples)
            batch = torch.from_numpy(X[start:end]).type(torch.float32).contiguous()
            res = scattering(batch).cpu().numpy()
            results.append(res)

        results = np.concatenate(results, axis=0)
        np.save(full_path, results)

    return results


def compute_integrals(input_array, integral_powers):
    integrals = torch.zeros((input_array.shape[0], len(integral_powers)), device=input_array.device)
    for i_q, q in enumerate(integral_powers):
        integrals[:, i_q] = ((input_array ** q).view(
            input_array.shape[0], -1).sum(1)) ** (1 / q)  # Norms

    return integrals
