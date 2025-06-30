import logging

import matplotlib.pyplot as plt
import numpy as np
import pkg_resources
from scipy.integrate import trapezoid
from scipy.optimize import curve_fit
from scipy.special import erf
from uncertainties import correlated_values

import psrutils

__all__ = [
    "centre_offset_degrees",
    "get_offpulse_region",
    "get_onpulse_region",
    "get_bias_corrected_pol_profile",
    "lookup_sigma_pa",
    "compute_sigma_pa_table",
    "pa_dist",
    "get_delta_vi",
]


def centre_offset_degrees(phase_bins: np.ndarray) -> np.ndarray:
    return phase_bins * 360 - 180


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
    windowsize : `int`, optional
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


def get_onpulse_region(
    data: np.ndarray, windowsize: int | None = None, logger: logging.Logger | None = None
) -> np.ndarray:
    """Determine the on-pulse window by maximising the integral over a range.

    Parameters
    ----------
    data : `np.ndarray`
        The original pulse profile.
    windowsize : `int`, optional
        Window width (in bins) defining the trial regions to integrate. Default: `None`
    logger : `logging.Logger`, optional
        A logger to use. Default: `None`.

    Returns
    -------
    onpulse_win : `np.ndarray`
        A list of bins corresponding to the on-pulse region.
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

    maxidx = np.argmax(integral)
    onpulse_win = np.arange(maxidx - windowsize // 2, maxidx + windowsize // 2) % nbins

    return onpulse_win


def get_bias_corrected_pol_profile(
    cube: psrutils.StokesCube,
    windowsize: int | None = None,
    logger: logging.Logger | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Get the full polarisation profile and correct for bias in the linear
    polarisation degree and angle. If no polarisation information is found in
    the archive, then `None` will be returned for all outputs except for the
    Stokes I profile.

    Parameters
    ----------
    cube : `psrutils.StokesCube`
        A StokesCube object.
    windowsize : `int`, optional
        Window width (in bins) defining the trial regions to integrate. Default: `None`
    logger : `logging.Logger`, optional
        A logger to use. Default: `None`.

    Returns
    -------
    iquv_profile : `np.ndarray`
        A 4xN array containing Stokes I/Q/U/V for N bins.
    l_true : `np.ndarray | None`
        A 1xN array containing the debiased Stokes L for N bins.
    pa : `np.ndarray | None`
        A 2xN array containing the position angle value and uncertainty.
    p0_l : `np.ndarray | None`
        A 1xN array containing the debiased linear polarisation measure L/sigma_I.
    p0_v : `np.ndarray | None`
        A 1xN array containing the circular polarisation measure V/sigma_I.
    sigma_i : `float | None`
        The standard deviation of the offpulse noise in the Stokes I profile.
    """
    if logger is None:
        logger = psrutils.get_logger()

    iquv_profile = cube.pol_profile

    # Default if the archive does not contain polarisation information
    if iquv_profile.shape[0] == 1:
        return iquv_profile, None, None, None, None, None

    # Get the indices of the offpulse bins
    offpulse_win = psrutils.get_offpulse_region(
        iquv_profile[0], windowsize=windowsize, logger=logger
    )

    # Create a profile mask to get the offpulse
    offpulse_mask = np.full(iquv_profile.shape[1], False)
    for bin_idx in offpulse_win:
        offpulse_mask[bin_idx] = True

    # Offpulse Stokes I noise
    sigma_i = np.std(iquv_profile[0, offpulse_mask])

    # Measured Stokes L
    l_meas = np.sqrt(iquv_profile[1] ** 2 + iquv_profile[2] ** 2)

    # Bias-corrected Stokes L
    l_true = np.where(
        np.abs(l_meas) > np.abs(sigma_i),
        np.sqrt(np.abs(l_meas**2 - sigma_i**2)),
        -np.sqrt(np.abs(l_meas**2 - sigma_i**2)),
    )

    # Bias-corrected polarisation measure
    p0_l = l_true / sigma_i
    p0_v = iquv_profile[3] / sigma_i

    # Position angle of linear polarisation
    # arctan2 is defined on [-pi, pi] radians
    pa = np.empty(shape=(2, iquv_profile.shape[1]), dtype=np.float64)
    pa_unc_dist = lookup_sigma_pa(p0_l)
    pa[0, :] = 0.5 * np.arctan2(iquv_profile[2], iquv_profile[1])
    pa[1, :] = np.where(
        p0_l > 10,
        0.5 / p0_l,
        pa_unc_dist,
    )
    pa = np.rad2deg(pa)

    return iquv_profile, l_true, pa, p0_l, p0_v, sigma_i


def lookup_sigma_pa(p0_meas: np.ndarray) -> np.ndarray:
    """Get the analytical uncertainty in a position angle measurement given
    the polarisation measure p0.

    Parameters
    ----------
    p0_meas : `np.ndarray`
        An array of p0 values, where p0 = L_true / sigma_I, L_true is the
        debiased intensity of linear polarisation and sigma_I is the offpulse
        noise in the Stokes I profile.

    Returns
    -------
    sigma_meas : `np.ndarray`
        The uncertainties in the position angles in radians.
    """
    sigma_pa_table = np.load(pkg_resources.resource_filename("psrutils", "data/sigma_pa_table.npy"))
    sigma_meas = np.interp(p0_meas, sigma_pa_table[0], sigma_pa_table[1])
    return sigma_meas


def compute_sigma_pa_table(
    pa_true: float = 0.0,
    p0_min: float = 0.0,
    p0_max: float = 10.0,
    p0_step: float = 0.01,
    sigma_pa_num: int = 1000,
    savename: str = "sigma_pa_table",
    make_plot: bool = False,
    logger: logging.Logger | None = None,
) -> None:
    """Compute the analytical distribution of position angles for an array
    of polarisation measures between p0_min and p0_max in steps of p0_step.
    Then find the integration limits that contain 68.26% of the distribution
    by looping through sigma_pa_num steps and integrating between the PA
    limits +/-(pi/2)/sigma_pa_num.

    For further details see Naghizadeh-Khouei and Clarke (1993):
    https://ui.adsabs.harvard.edu/abs/1993A%26A...274..968N/abstract

    Parameters
    ----------
    pa_true : `float`, optional
        The centre of the PA distribution. Default: 0.0.
    p0_min : `float`, optional
        The minimum p0 to evaluate at. Default: 0.0.
    p0_max : `float`, optional
        The maximum p0 to evaluate at. Default: 10.0.
    p0_step : `float`, optional
        The p0 step size. Default: 0.01.
    sigma_pa_num : `int`, optional
        The number of steps to use to find sigma_pa. Default: 1000.
    savename : `str`, optional
        The name of the output table (without extension). Default: "sigma_pa_table".
    make_plot : `bool`, optional
        Make a plot of the results. Default: `False`.
    logger : `logging.Logger`, optional
        A logger to use. Default: None.
    """
    if logger is None:
        logger = psrutils.get_logger()

    p0 = np.arange(p0_min, p0_max, p0_step, dtype=np.float64)
    sigma_pa_table = np.empty(shape=(2, sigma_pa_num), dtype=np.float64)
    sigma_pa_range = np.linspace(0, np.pi / 2, sigma_pa_num, dtype=np.float64)
    for ii, ip0 in enumerate(p0):
        integral_table = np.empty_like(sigma_pa_range)
        for jj, isigma in enumerate(sigma_pa_range):
            # Choose a range of PAs to integrate over in the range [-pi, pi)
            pa_range = np.linspace(pa_true - isigma, pa_true + isigma, 1000, dtype=np.float64)
            pa_range = (pa_range + np.pi) % (2 * np.pi) - np.pi

            # Numerically integrate over PA range using the trapezoid rule
            integral_table[jj] = trapezoid(
                pa_dist(pa_range, np.float64(pa_true), ip0),
                pa_range,
            )
        # Linearly interpolate to find 1-sigma
        sigma_pa_table[0, ii] = ip0
        sigma_pa_table[1, ii] = np.interp(0.6826, integral_table, sigma_pa_range)

    if make_plot:
        fig, ax = plt.subplots(dpi=300, tight_layout=True)
        ax.errorbar(sigma_pa_table[0], sigma_pa_table[1], fmt="k-", ms=1)
        logger.info(f"Saving file: {savename}.png")
        fig.savefig(f"{savename}.png")
        plt.close()

    logger.info(f"Saving file: {savename}.npy")
    np.save(savename, sigma_pa_table)


def pa_dist(
    pa: np.ndarray[np.float64], pa_true: np.float64, p0: np.float64
) -> np.ndarray[np.float64]:
    """Calculate the position angle distribution for low signal-to-noise ratio
    measurements. See from Naghizadeh-Khouei and Clarke (1993).

    Parameters
    ----------
    pa : `np.ndarray[np.float64]`
        An array of position angles to evaluate the distribution at.
    pa_true : `np.float64`
        The centre of the distribution.
    p0 : `np.float64`
        The polarisation measure.

    Returns
    -------
    G_pa : `np.ndarray[np.float64]`
        The distribution evaluated at pa.
    """
    k = np.pi ** (-0.5)
    eta = p0 * 2 ** (-0.5) * np.cos(2 * (pa - pa_true))
    G_pa = k * (k + eta * np.exp(eta**2) * (1 + erf(eta))) * np.exp(-0.5 * p0**2)
    return G_pa


def get_delta_vi(
    cube: psrutils.StokesCube,
    onpulse_win: np.ndarray | None = None,
    logger: logging.Logger | None = None,
) -> np.ndarray:
    """Calculate the change in Stokes V/I over the observing bandwidth.

    Parameters
    ----------
    cube : `psrutils.StokesCube`
        A StokesCube object.
    onpulse_win : `np.ndarray`, optional
        A list of bins corresponding to the on-pulse region. Default: `None`.
    logger : `logging.Logger`, optional
        A logger to use. Default: `None`.

    Returns
    -------
    delta_vi : `np.ndarray`
        The change in Stokes V/I calculated per bin.
    """
    if logger is None:
        logger = psrutils.get_logger()

    freqs = cube.freqs / 1e6  # MHz
    spectra = cube.subbands  # -> (pol, freq, phase)
    vi_spectra = spectra[3] / spectra[0]
    delta_vi = np.full((2, cube.num_bin), np.nan, dtype=np.float64)

    for bin_idx in range(vi_spectra.shape[1]):
        if bin_idx not in onpulse_win:
            continue
        vi_spectrum = vi_spectra[:, bin_idx]
        ql, qh = np.quantile(
            vi_spectrum[~np.isnan(vi_spectrum) & np.isfinite(vi_spectrum)], q=(0.05, 0.95)
        )
        mask = (vi_spectrum > ql) & (vi_spectrum < qh)
        try:
            par, cov = curve_fit(
                lambda f, c, m: f * m + c,
                freqs[mask],
                vi_spectrum[mask],
                p0=(np.mean(vi_spectrum), 0.0),
            )
        except RuntimeError:
            logger.debug(f"Bin {bin_idx}: Stokes V model fit could not converge.")
            continue

        (c_best, m_best) = correlated_values(par, cov)

        # Filter out outliers
        if np.abs(m_best.n) < 1e-4 or np.abs(m_best.n) > 1:
            continue
        if m_best.s < 1e-4 or m_best.s > 1:
            continue

        model = freqs * m_best + c_best
        delta_vi_bin = model[-1] - model[0]

        delta_vi[0, bin_idx] = delta_vi_bin.n
        delta_vi[1, bin_idx] = delta_vi_bin.s

    return delta_vi
