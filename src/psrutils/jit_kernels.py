import numpy as np
from numba import njit, prange

import psrutils as pu

__all__ = ["dft_kernel", "idft_kernel"]


@njit(parallel=True, fastmath=True)
def dft_kernel(
    input_arr: np.ndarray,
    output_arr: np.ndarray,
    phi_arr: np.ndarray,
    lambda_arr: np.ndarray,
    lambda_shift: float,
    norm_const: float,
) -> np.ndarray:
    nphi = len(phi_arr)
    lambda_shifted = lambda_arr - lambda_shift
    for ii in prange(nphi):
        output_arr[ii] = np.sum(input_arr * np.exp(-2.0 * pu.IMAG * phi_arr[ii] * lambda_shifted))
    output_arr *= norm_const
    return output_arr


@njit(parallel=True, fastmath=True)
def idft_kernel(
    input_arr: np.ndarray,
    output_arr: np.ndarray,
    phi_arr: np.ndarray,
    lambda_arr: np.ndarray,
    lambda_shift: float,
    norm_const: float,
) -> np.ndarray:
    nlam = len(lambda_arr)
    lambda_shifted = lambda_arr - lambda_shift
    for ii in prange(nlam):
        output_arr[ii] = np.sum(input_arr * np.exp(2.0 * pu.IMAG * phi_arr * lambda_shifted[ii]))
    output_arr *= norm_const
    return output_arr
