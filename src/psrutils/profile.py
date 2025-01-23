import logging

import numpy as np

import psrutils

__all__ = ["get_offpulse_region"]


def get_offpulse_region(
    data: np.ndarray, windowsize: int | None = None, logger: logging.Logger | None = None
) -> np.ndarray:
    """Determine the off-pulse window by minimising the integral over a range.
    i.e., because noise should integrate towards zero, finding the region that
    minimises the area mean it is representative of the noise level.

    Method taken from PyPulse (Lam, 2017. https://ascl.net/1706.011).

    Parameters
    ----------
    data : `np.ndarray`
        The original pulse profile.
    windowsize : `np.ndarray`, optional
        Window width (in bins) defining the trial regions to integrate. Default: `None`
    logger : `logging.Logger`, optional
        A logger to use. Default: `None`.

    Returns
    -------
    offpulse_win : `np.ndarray`
        A list of bins corresponding to the off-pulse region.
    """
    if logger is None:
        logger = psrutils.get_logger()

    nbins = len(data)

    if windowsize is None:
        logger.debug("No off-pulse window size set, assuming 1/8 of profile.")
        windowsize = nbins // 8

    integral = np.zeros_like(data)
    for i in range(nbins):
        win = np.arange(i - windowsize // 2, i + windowsize // 2) % nbins
        integral[i] = np.trapz(data[win])

    minidx = np.argmin(integral)
    offpulse_win = np.arange(minidx - windowsize // 2, minidx + windowsize // 2) % nbins

    return offpulse_win
