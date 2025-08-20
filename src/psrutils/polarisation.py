########################################################
# Licensed under the Academic Free License version 3.0 #
########################################################

import importlib.resources
import logging
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import trapezoid
from scipy.optimize import curve_fit
from scipy.special import erf
from uncertainties import correlated_values

from .cube import StokesCube
from .profile import SplineProfile

__all__ = [
    "PolProfile",
    "get_bias_corrected_pol_profile",
    "lookup_sigma_pa",
    "compute_sigma_pa_table",
    "pa_dist",
    "get_delta_vi",
]

logger = logging.getLogger(__name__)


class PolProfile(NamedTuple):
    """A named tuple containing a full-polarisation pulse profile."""

    iquv: NDArray[np.float64]
    """(4, Nbin) array containing Stokes I/Q/U/V."""
    l_true: NDArray[np.float64]
    """(1, Nbin) array containing the Stokes L."""
    pa: NDArray[np.float64]
    """(2, Nbin) array containing the position angle value and uncertainty."""
    p0_l: NDArray[np.float64]
    """(1, Nbin) array containing the linear polarisation measure L/sigma_I."""
    p0_v: NDArray[np.float64]
    """(1, Nbin) array containing the circular polarisation measure V/sigma_I."""
    sigma_i: float
    """The standard deviation of the offpulse noise in the Stokes I profile."""


def get_bias_corrected_pol_profile(
    iquv: NDArray[np.float64], spline: bool = False
) -> PolProfile:
    """Get the full-polarisation pulse profile and correct for bias in the
    linear polarisation degree and angle.

    Parameters
    ----------
    iquv : NDArray[float64]
        The polarisation profile, with dimensions (pol, phase).
    spline : bool, default: False
        Use the spline fitting method to estimate the noise in the profile when
        debiasing the linear polarisation. Otherwise, use the minimum integrated
        flux density method.

    Returns
    -------
    PolProfile
        A named tuple containing the full-polarisation pulse profile.
    """
    if iquv.shape[0] == 1:
        raise ValueError("Profile does not contain polarisation information.")

    # Estimate the offpulse noise
    profile = SplineProfile(iquv[0])
    if spline:
        profile.gridsearch_onpulse_regions()
        sigma_i = float(profile.noise_est)
    else:
        sigma_i = float(profile.get_simple_noise_stats()[1])

    # Measured Stokes L
    l_meas: NDArray[np.float64] = np.sqrt(iquv[1] ** 2 + iquv[2] ** 2)

    # Bias-corrected Stokes L
    l_true: NDArray[np.float64] = np.where(
        np.abs(l_meas) > np.abs(sigma_i),
        np.sqrt(np.abs(l_meas**2 - sigma_i**2)),
        -np.sqrt(np.abs(l_meas**2 - sigma_i**2)),
    )

    # Bias-corrected polarisation measures
    p0_l: NDArray[np.float64] = l_true / sigma_i
    p0_v: NDArray[np.float64] = iquv[3] / sigma_i

    # Position angle of linear polarisation
    # arctan2 is defined on [-pi, pi] radians
    pa = np.empty(shape=(2, iquv.shape[1]), dtype=np.float64)
    pa_unc_dist = lookup_sigma_pa(p0_l)
    pa[0, :] = 0.5 * np.arctan2(iquv[2], iquv[1])
    pa[1, :] = np.where(
        p0_l > 10,
        0.5 / p0_l,
        pa_unc_dist,
    )
    pa = np.rad2deg(pa)

    return PolProfile(iquv, l_true, pa, p0_l, p0_v, sigma_i)


def lookup_sigma_pa(p0_meas: NDArray[np.floating]) -> NDArray[np.float64]:
    """Get the analytical uncertainties for an array of position angle
    measurement given an array of the polarisation measures.

    Parameters
    ----------
    p0_meas : NDArray[floating]
        An array of polarisation measures (i.e. L/sigma_I).

    Returns
    -------
    sigma_meas : NDArray[float64]
        The uncertainties in the position angles in radians.
    """
    ref = importlib.resources.files("psrutils") / "data/sigma_pa_table.npy"
    with importlib.resources.as_file(ref) as path:
        sigma_pa_table: NDArray[np.float64] = np.load(path)
    sigma_meas: NDArray[np.float64] = np.interp(
        p0_meas, sigma_pa_table[0], sigma_pa_table[1]
    )
    return sigma_meas


def compute_sigma_pa_table(
    pa_true: float = 0.0,
    p0_min: float = 0.0,
    p0_max: float = 10.0,
    p0_step: float = 0.01,
    sigma_pa_num: int = 1000,
    savename: str = "sigma_pa_table",
    make_plot: bool = False,
) -> None:
    """Compute the analytical distribution of position angles for an array
    of polarisation measures between `p0_min` and `p0_max` in steps of
    `p0_step`. Then find the integration limits that contain 68.26% of the
    distribution by looping through `sigma_pa_num` steps and integrating
    between the position angle limits `+/-(pi/2)/sigma_pa_num`.

    Parameters
    ----------
    pa_true : float, default: 0.0
        The centre of the position angle distribution.
    p0_min : float, default: 0.0
        The minimum p0 to evaluate at.
    p0_max : float, default: 10.0
        The maximum p0 to evaluate at.
    p0_step : float, default: 0.01
        The p0 step size.
    sigma_pa_num : int, default: 1000
        The number of steps to use to find sigma_pa.
    savename : str, default: "sigma_pa_table"
        The name of the output file, excluding the file extension.
    make_plot : bool, default: False
        Make a plot of the results.

    References
    ----------
    For further details see Naghizadeh-Khouei and Clarke (1993):
    https://ui.adsabs.harvard.edu/abs/1993A%26A...274..968N/abstract
    """
    p0 = np.arange(p0_min, p0_max, p0_step, dtype=np.float64)
    sigma_pa_table = np.empty(shape=(2, sigma_pa_num), dtype=np.float64)
    sigma_pa_range = np.linspace(0, np.pi / 2, sigma_pa_num, dtype=np.float64)
    for ii, ip0 in enumerate(p0):
        integral_table = np.empty_like(sigma_pa_range)
        for jj, isigma in enumerate(sigma_pa_range):
            # Choose a range of PAs to integrate over in the range [-pi, pi)
            pa_range = np.linspace(
                pa_true - isigma, pa_true + isigma, 1000, dtype=np.float64
            )
            pa_range = ((pa_range + np.pi) % (2 * np.pi) - np.pi).astype(np.float64)

            # Numerically integrate over PA range using the trapezoid rule
            integral_table[jj] = trapezoid(pa_dist(pa_range, pa_true, ip0), pa_range)
        # Linearly interpolate to find 1-sigma
        sigma_pa_table[0, ii] = ip0
        sigma_pa_table[1, ii] = np.interp(0.6826, integral_table, sigma_pa_range)

    if make_plot:
        fig, ax = plt.subplots(tight_layout=True)
        ax.errorbar(sigma_pa_table[0], sigma_pa_table[1], fmt="k-", ms=1)
        logger.info(f"Saving file: {savename}.png")
        fig.savefig(f"{savename}.png")
        plt.close()

    logger.info(f"Saving file: {savename}.npy")
    np.save(savename, sigma_pa_table)


def pa_dist(pa: NDArray[np.float64], pa_true: float, p0: float) -> NDArray[np.float64]:
    """Calculate the position angle distribution for low signal-to-noise
    ratio measurements.

    Parameters
    ----------
    pa : NDArray[float64]
        The position angles to evaluate the distribution at.
    pa_true : float
        The centre of the distribution.
    p0 : float
        The polarisation measure.

    Returns
    -------
    G_pa : NDArray[float64]
        The distribution evaluated at the position angles in `pa`.

    References
    ----------
    For further details see Naghizadeh-Khouei and Clarke (1993):
    https://ui.adsabs.harvard.edu/abs/1993A%26A...274..968N/abstract
    """
    k: float = np.pi ** (-0.5)
    eta: NDArray[np.float64] = p0 * 2 ** (-0.5) * np.cos(2 * (pa - pa_true))
    G_pa: NDArray[np.float64] = (
        k * (k + eta * np.exp(eta**2) * (1 + erf(eta))) * np.exp(-0.5 * p0**2)
    )
    return G_pa


def get_delta_vi(
    cube: StokesCube, onpulse_bins: NDArray[np.integer] | None = None
) -> NDArray[np.float64]:
    """Calculate the change in Stokes V/I over the observing bandwidth.

    Parameters
    ----------
    cube : StokesCube
        A StokesCube containing a full-polarisation data archive.
    onpulse_bins : NDArray[integer] or None, default: None
        A list of bin indices corresponding to the on-pulse region. If None
        then use the whole profile.

    Returns
    -------
    delta_vi : NDArray[float64]
        The change in Stokes V/I calculated per bin.
    """
    freqs = (cube.freqs / 1e6).astype(np.float64)  # MHz
    spectra = cube.subbands.astype(np.float64)  # -> (pol, freq, phase)
    vi_spectra: NDArray[np.float64] = spectra[3] / spectra[0]
    delta_vi = np.full((2, cube.num_bin), np.nan, dtype=np.float64)

    if onpulse_bins is None:
        # Use the whole profile
        onpulse_bins = np.arange(cube.num_bin)

    for bin_idx in range(vi_spectra.shape[1]):
        if bin_idx not in onpulse_bins:
            continue
        vi_spectrum = vi_spectra[:, bin_idx]
        ql, qh = np.quantile(
            vi_spectrum[~np.isnan(vi_spectrum) & np.isfinite(vi_spectrum)],
            q=(0.05, 0.95),
        )
        mask: NDArray[np.bool_] = (vi_spectrum > ql) & (vi_spectrum < qh)
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
