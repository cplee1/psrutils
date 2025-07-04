import importlib.resources
import itertools
import logging

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import trapezoid
from scipy.interpolate import BSpline, PPoly, splrep
from scipy.optimize import curve_fit
from scipy.special import erf
from uncertainties import correlated_values

import psrutils

__all__ = [
    "centre_offset_degrees",
    "find_optimimum_pulse_window",
    "get_offpulse_noise",
    "find_onpulse_regions",
    "get_bias_corrected_pol_profile",
    "lookup_sigma_pa",
    "compute_sigma_pa_table",
    "pa_dist",
    "get_delta_vi",
]


def centre_offset_degrees(phase_bins: np.ndarray) -> np.ndarray:
    return phase_bins * 360 - 180


def find_optimimum_pulse_window(
    profile: np.ndarray,
    windowsize: int | None = None,
    maximise: bool = False,
    logger: logging.Logger | None = None,
) -> np.ndarray:
    """Find the pulse window which minimises/maximises the integrated flux density
    within it. Noise should integrate towards zero, so minimising the integral will
    find an offpulse window. Conversely, maximising the integral will find an
    onpulse window. The window size should be tweaked depending on the pulsar.

    Method taken from PyPulse (Lam, 2017. https://ascl.net/1706.011).

    Parameters
    ----------
    profile : `np.ndarray`
        The original pulse profile.
    windowsize : `int`, optional
        Window width (in bins) defining the trial regions to integrate. If `None`,
        then will use 1/8 of the profile. Default: `None`
    maximise : `bool`, optional
        If `True`, will maximise the integral; otherwise, will minimise. Default: `False`.
    logger : `logging.Logger`, optional
        A logger to use. Default: `None`.

    Returns
    -------
    window_bins : `np.ndarray`
        A list of bins corresponding to the located region.
    """
    if logger is None:
        logger = psrutils.get_logger()

    nbins = len(profile)

    if windowsize is None:
        logger.debug("No off-pulse window size set, assuming 1/8 of profile.")
        windowsize = nbins // 8

    integral = np.zeros_like(profile)
    for i in range(nbins):
        win = np.arange(i - windowsize // 2, i + windowsize // 2) % nbins
        integral[i] = np.trapz(profile[win])

    if maximise:
        maxidx = np.argmax(integral)
        return np.arange(maxidx - windowsize // 2, maxidx + windowsize // 2) % nbins
    else:
        minidx = np.argmin(integral)
        return np.arange(minidx - windowsize // 2, minidx + windowsize // 2) % nbins


def get_offpulse_noise(profile: np.ndarray, logger: logging.Logger | None = None):
    if logger is None:
        logger = psrutils.get_logger()

    offpulse_win = psrutils.find_optimimum_pulse_window(profile, logger=logger)

    # Create a profile mask to get the offpulse
    offpulse_mask = np.full(profile.size, False)
    for bin_idx in offpulse_win:
        offpulse_mask[bin_idx] = True

    # Standard deviation of the noise in the offpulse window
    return np.std(profile[offpulse_mask])


def find_onpulse_regions(
    profile: np.ndarray,
    sigma_cutoff: float = 2.0,
    plotname: str = None,
    logger: logging.Logger | None = None,
) -> dict:
    """Find under- and over-estimates of the onpulse regions from a PPoly representation
    of a smoothed profile.

    Parameters
    ----------
    profile : `np.ndarray`
        The total intensity profile.
    sigma_cutoff : `float`, optional
        The number of standard deviations above which a signal will be considered real.
        Default: 2.0.
    plotname : `str`, optional
        If provided, will save a plot of the smoothed profile and its derivatives with
        this plot name. The file extension should be excluded.
    logger : `logging.Logger`, optional
        A logger to use. Default: `None`.

    Returns
    -------
    results: `dict`
        A dictionary containing the following:

        "bins" : The bins at which the profile was evaluated at.
        "sm_prof" : The evaluated smoothed profile.
        "d1_sm_prof" : The first derivative of sm_prof.
        "d2_sm_prof" : The second derivative of sm_prof.
        "underest_onpulse" : A list of pairs of underestimated onpulse bin indices.
        "overest_onpulse" : A list of pairs of overestimated onpulse bin indices.
    """
    if logger is None:
        logger = psrutils.get_logger()

    # Normalise profile
    profile /= np.max(profile)

    # Get offpulse noise using sliding window method
    noise_est = psrutils.get_offpulse_noise(profile, logger=logger)

    # Bins
    nbin = profile.size
    bins = np.arange(nbin)

    # Fit a degree-k smoothing spline in the B-spline basis with periodic boundary
    # conditions
    tck = splrep(bins, profile, k=5, s=nbin * noise_est**2, per=True)
    splrep_bspl = BSpline(*tck, extrapolate="periodic")

    # Convert to a piecewise degree-k polynomial representation, which has a better
    # root finding algorithm
    splrep_ppoly = PPoly.from_spline(splrep_bspl, extrapolate="periodic")

    # Find the onpulse regions
    opr_result = _find_onpulse_regions_from_ppoly(
        bins, splrep_ppoly, noise_est, sigma_cutoff=sigma_cutoff, logger=logger
    )

    if plotname:
        _plot_onpulse_regions(
            opr_result,
            profile,
            plot_underestimate=False,
            savename=plotname,
            logger=logger,
        )

    return opr_result


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
    offpulse_win = psrutils.find_optimimum_pulse_window(
        iquv_profile[0], windowsize=windowsize, maximise=False, logger=logger
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
    ref = importlib.resources.files("psrutils") / "data/sigma_pa_table.npy"
    with importlib.resources.as_file(ref) as path:
        sigma_pa_table = np.load(path)
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


def _fill_onpulse_regions(
    onpulse_pairs: list,
    sm_prof: np.ndarray,
    noise_est: float,
    sigma_cutoff: float,
) -> list:
    """Attempts to fill small gaps in the onpulse pairs provided these gaps exceed
    a set sigma threshold (i.e. they resemble real signal.)

    Parameters
    ----------
    onpulse_pairs : `list`
        A list of pairs of bin indices that mark the onpulse regions of the profile.
    sm_prof : `np.ndarray`
        The smoothed profile.
    noise_est : `float`
        The standard deviation of the offpulse noise.
    sigma_cutoff : `float`
        The number of standard deviations above which a signal will be considered real.

    Return:
    -------
    filled_onpulse_pairs : `list`
        A list of pairs of bin indices that represent the ranges of the filled onpulse regions.
    """
    # Nothing to do if the pairs are of length 1
    if len(onpulse_pairs) <= 1:
        return onpulse_pairs

    loop_pairs = onpulse_pairs.copy()
    # Create two cycling generators to avoid out of bounds errors for the last pair
    on_pulse_ammendments = []
    current_pair_gen = itertools.cycle(loop_pairs)
    next_pair_gen = itertools.cycle(loop_pairs)
    next(next_pair_gen)
    for _ in range(len(loop_pairs)):
        # Next pair from the generator
        current_pair = next(current_pair_gen)
        next_pair = next(next_pair_gen)

        # Figure out offpulse region
        pair = [current_pair[1], next_pair[0]]
        if pair[0] < pair[1]:
            offpulse_region = sm_prof[pair[0] : pair[1]]
        else:
            offpulse_region = np.concatenate([sm_prof[pair[0] :], sm_prof[: pair[1]]])

        # See if this looks like signal
        if min(offpulse_region) >= sigma_cutoff * noise_est:
            # Add to ammendments that we need to make
            on_pulse_ammendments.append([current_pair[1], next_pair[0]])

    # Apply the ammendments to the on-pulse region
    filled_onpulse_pairs = loop_pairs.copy()
    for ammendment_pair in on_pulse_ammendments:
        # The beginning point of the on-pulse to be ammended
        begin = ammendment_pair[0]
        end = ammendment_pair[1]
        # Find where the beginning meets up with the endpoints
        current_begs = np.array([i[0] for i in filled_onpulse_pairs])
        current_ends = np.array([i[1] for i in filled_onpulse_pairs])
        # Find the indexes of the start and end pairs
        idx_begin = np.where(current_ends == begin)[0][0]
        idx_end = np.where(current_begs == end)[0][0]
        # Modify the on-pulse pair
        start_of_begin = filled_onpulse_pairs[idx_begin][0]
        end_of_end = filled_onpulse_pairs[idx_end][1]
        filled_onpulse_pairs[idx_begin] = [start_of_begin, end_of_end]
        # Remove the other pair as it's been absorbed
        del filled_onpulse_pairs[idx_end]

    return filled_onpulse_pairs


def _plot_onpulse_regions(
    opr_result: dict,
    real_prof: np.ndarray,
    plot_underestimate: bool = True,
    plot_overestimate: bool = True,
    lw: float = 1.2,
    savename: str = "onpulse",
    logger: logging.Logger | None = None,
) -> None:
    """Plot the smoothed profile and its derivatives, as well as the onpulse regions.

    Parameters
    ----------
    opr_result: dict
        A results dictionary from `_find_onpulse_regions_from_ppoly()`.
    real_prof : np.ndarray
        The real profile, with the same number of bins as the opr_result.
    plot_underestimate : `bool`
        Whether to plot the onpulse underestimate. Default: `True`.
    plot_overestimate : `bool`
        Whether to plot the onpulse overestimate. Default: `True`.
    lw : `float`, optional
        The linewidth used throughout the plot. Default: 1.1.
    savename : `str`, optional
        The name of the plot file excluding the extension. Default: 'onpulse'.
    logger : `logging.Logger`, optional
        A logger to use. Default: `None`.
    """
    bins = opr_result["bins"]
    sm_prof = opr_result["sm_prof"]
    d1_sm_prof = opr_result["d1_sm_prof"]
    d2_sm_prof = opr_result["d2_sm_prof"]
    underest_onpulse = opr_result["underest_onpulse"]
    overest_onpulse = opr_result["overest_onpulse"]

    # Figure setup
    fig, axes = plt.subplots(nrows=3, figsize=(8, 7), sharex=True, dpi=300, tight_layout=True)
    xrange = (bins[0], bins[-1])

    # Profile
    axes[0].plot(bins, real_prof, color="k", linewidth=lw, alpha=0.2, label="Real Profile")
    axes[0].plot(bins, sm_prof, color="k", linewidth=lw * 0.8, label="Smoothed Profile")
    yrange = axes[0].get_ylim()

    # Onpulse estimates
    underest_args = dict(color="tab:blue", edgecolor=None, alpha=0.3, zorder=0)
    overest_args = dict(color="tab:blue", edgecolor=None, alpha=0.2, zorder=0)
    for pairlist, args, flag, label in zip(
        [underest_onpulse, overest_onpulse],
        [underest_args, overest_args],
        [plot_underestimate, plot_overestimate],
        ["Underestimate", "Overestimate"],
        strict=True,
    ):
        if flag:
            for idx, op_pair in enumerate(pairlist):
                used_label = None
                if idx == 0:
                    used_label = label
                if op_pair[0] < op_pair[-1]:
                    axes[0].fill_betweenx(yrange, op_pair[0], op_pair[-1], label=used_label, **args)
                else:
                    axes[0].fill_betweenx(yrange, op_pair[0], xrange[-1], label=used_label, **args)
                    axes[0].fill_betweenx(yrange, xrange[0], op_pair[-1], **args)

    # Derivatives
    axes[1].plot(bins, d1_sm_prof, color="k", linewidth=lw * 0.8, label="1st Derivative")
    axes[2].plot(bins, d2_sm_prof, color="k", linewidth=lw * 0.8, label="2nd Derivative")

    # Finishing touches
    axes[0].set_ylim(yrange)
    axes[2].set_xlabel("Bin Index")
    for ax in axes:
        ax.axhline(0, linestyle=":", color="k", linewidth=lw * 0.8, alpha=0.3, zorder=1)
        ax.set_xlim(xrange)
        ax.legend(loc="upper left")

    logger.info(f"Saving plot file: {savename}.png")
    fig.savefig(savename + ".png")


def _find_onpulse_regions_from_ppoly(
    bins: np.ndarray,
    ppoly: PPoly,
    noise_est: float,
    sigma_cutoff: float = 2.0,
    logger: logging.Logger | None = None,
) -> dict:
    """Find under- and over-estimates of the onpulse regions from a PPoly representation
    of a smoothed profile.

    Parameters
    ----------
    bins : `np.ndarray`
        An array of bin indices.
    ppoly : `PPoly`
        A degree >=3 PPoly object describing the smoothed profile.
    noise_est : `float`
        The standard deviation of the offpulse noise.
    sigma_cutoff : `float`, optional
        The number of standard deviations above which a signal will be considered real.
        Default: 2.0.
    logger : `logging.Logger`, optional
        A logger to use. Default: `None`.

    Returns
    -------
    results: `dict`
        A dictionary containing the following:

        "bins" : The bins at which the profile was evaluated at.
        "sm_prof" : The evaluated smoothed profile.
        "d1_sm_prof" : The first derivative of sm_prof.
        "d2_sm_prof" : The second derivative of sm_prof.
        "underest_onpulse" : A list of pairs of underestimated onpulse bin indices.
        "overest_onpulse" : A list of pairs of overestimated onpulse bin indices.
    """
    if logger is None:
        logger = psrutils.get_logger()

    # Evaluate derivatives and roots
    d1_ppoly = ppoly.derivative(nu=1)
    d1_roots = [round(rt) for rt in d1_ppoly.roots()]
    d2_ppoly = ppoly.derivative(nu=2)
    d2_roots = [round(rt) for rt in d2_ppoly.roots()]

    # Interpolate smoothed profiles
    sm_prof = ppoly(bins)
    d1_sm_prof = d1_ppoly(bins)
    d2_sm_prof = d2_ppoly(bins)

    # Filter out extrapolated roots
    d1_roots_inbounds = []
    for root in d1_roots:
        if root < bins[0] or root > bins[-1]:
            continue
        d1_roots_inbounds.append(root)
    d2_roots_inbounds = []
    for root in d2_roots:
        if root < bins[0] or root > bins[-1]:
            continue
        d2_roots_inbounds.append(root)

    # Catagorise minima and maxima
    minima = []
    true_maxima, false_maxima = [], []
    for root in d1_roots_inbounds:
        if d2_sm_prof[root] > 0:  # Minimum
            minima.append(root)
        else:  # Maximum
            if sm_prof[root] > noise_est * sigma_cutoff:
                true_maxima.append(root)
            else:
                false_maxima.append(root)

    if not true_maxima:
        logger.warning(f"No profile maxima found >{sigma_cutoff} sigma.")
        return

    underest_onpulse = []
    overest_onpulse = []
    for d1_root in true_maxima:
        # Underestimated onpulse
        # Figure out which two inflections (d2 roots) flank the maximum (d1 root)
        d2_roots = sorted(np.append(d2_roots, d1_root))
        mloc = d2_roots.index(d1_root)
        d2_roots.remove(d1_root)
        try:
            underestimate = [round(d2_roots[mloc - 1]), round(d2_roots[mloc])]
        except IndexError:
            # The next inflection is on the other side of the profile
            if mloc == 0:  # Left side
                underestimate = [round(d2_roots[-1]), round(d2_roots[mloc])]
            else:  # Right side
                underestimate = [round(d2_roots[mloc - 1]), round(d2_roots[0])]
        underest_onpulse.append(underestimate)

        # Overestimated onpulse
        # Figure out which two false minima (d1 roots) flank the maximum (d1 root)
        minima = sorted(np.append(minima, d1_root))
        mloc = minima.index(d1_root)
        minima.remove(d1_root)
        try:
            overestimate = [round(minima[mloc - 1]), round(minima[mloc])]
        except IndexError:
            # The next inflection is on the other side of the profile
            if mloc == 0:  # Left side
                overestimate = [round(minima[-1]), round(minima[mloc])]
            else:  # Right side
                overestimate = [round(minima[mloc - 1]), round(minima[0])]

        # If the bounds are touching
        if overest_onpulse and overestimate[0] == overest_onpulse[-1][1]:
            overest_onpulse[-1][1] = overestimate[1]

        # If the last element wraps around to the first
        elif overest_onpulse and overestimate[1] == overest_onpulse[0][0]:
            overest_onpulse[0][0] = overestimate[0]
        else:
            overest_onpulse.append(overestimate)

    # Remove any duplicate items because somehow this happens sometimes
    underest_onpulse.sort()
    underest_onpulse = list(k for k, _ in itertools.groupby(underest_onpulse))
    overest_onpulse.sort()
    overest_onpulse = list(k for k, _ in itertools.groupby(overest_onpulse))

    # Fill the on-pulse regions
    underest_onpulse = _fill_onpulse_regions(underest_onpulse, sm_prof, noise_est, sigma_cutoff)
    overest_onpulse = _fill_onpulse_regions(overest_onpulse, sm_prof, noise_est, sigma_cutoff)

    return dict(
        bins=bins,
        sm_prof=sm_prof,
        d1_sm_prof=d1_sm_prof,
        d2_sm_prof=d2_sm_prof,
        underest_onpulse=underest_onpulse,
        overest_onpulse=overest_onpulse,
    )
