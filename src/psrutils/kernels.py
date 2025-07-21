import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

__all__ = ["dft_kernel", "idft_kernel"]


# TODO: Make docstring
@njit(parallel=True, fastmath=True)
def dft_kernel(
    input_arr: NDArray[np.floating],
    output_arr: NDArray[np.floating],
    phi_arr: NDArray[np.floating],
    lambda_arr: NDArray[np.floating],
    lambda_shift: float,
    norm_const: float,
) -> NDArray[np.floating]:
    nphi = len(phi_arr)
    lambda_shifted = lambda_arr - lambda_shift
    for ii in prange(nphi):
        output_arr[ii] = np.sum(
            input_arr * np.exp(-2.0 * 1j * phi_arr[ii] * lambda_shifted)
        )
    output_arr *= norm_const
    return output_arr


# TODO: Make docstring
@njit(parallel=True, fastmath=True)
def idft_kernel(
    input_arr: NDArray[np.floating],
    output_arr: NDArray[np.floating],
    phi_arr: NDArray[np.floating],
    lambda_arr: NDArray[np.floating],
    lambda_shift: float,
    norm_const: float,
) -> NDArray[np.floating]:
    nlam = len(lambda_arr)
    lambda_shifted = lambda_arr - lambda_shift
    for ii in prange(nlam):
        output_arr[ii] = np.sum(
            input_arr * np.exp(2.0 * 1j * phi_arr * lambda_shifted[ii])
        )
    output_arr *= norm_const
    return output_arr
