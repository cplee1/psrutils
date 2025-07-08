import itertools
import logging

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import BSpline, PPoly, splrep

import psrutils

__all__ = [
    "centre_offset_degrees",
    "find_optimal_pulse_window",
    "est_offpulse_noise",
    "find_onpulse_regions",
    "get_profile_mask_from_pairs",
    "get_offpulse_from_onpulse",
]


def centre_offset_degrees(phase_bins: np.ndarray) -> np.ndarray:
    return phase_bins * 360 - 180


def find_optimal_pulse_window(
    profile: np.ndarray,
    windowsize: int | None = None,
    maximise: bool = False,
    logger: logging.Logger | None = None,
) -> np.ndarray:
    """Find the pulse window which minimises/maximises the integrated flux density within it. Noise
    should integrate towards zero, so minimising the integral will find an offpulse window.
    Conversely, maximising the integral will find an onpulse window. The window size should be
    tweaked depending on the pulsar.

    Method taken from PyPulse (Lam, 2017. https://ascl.net/1706.011).

    Parameters
    ----------
    profile : `np.ndarray`
        The pulse profile.
    windowsize : `int`, optional
        Window width (in bins) defining the trial regions to integrate. If `None`, then will use 1/8
        of the profile. Default: `None`
    maximise : `bool`, optional
        If `True`, will maximise the integral; otherwise, will minimise. Default: `False`.
    logger : `logging.Logger`, optional
        A logger to use. Default: `None`.

    Returns
    -------
    left_idx : `int`
        The bin index of the leading edge of the window.
    right_idx : `int`
        The bin index of the trailing edge of the window.
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
        opt_idx = np.argmax(integral)
    else:
        opt_idx = np.argmin(integral)

    left_idx = (opt_idx - windowsize // 2) % nbins
    right_idx = (opt_idx + windowsize // 2) % nbins
    return left_idx, right_idx


def est_offpulse_noise(
    profile: np.ndarray, windowsize: int | None = None, logger: logging.Logger | None = None
) -> float:
    """Estimate the standard deviation of the offpulse noise using the flux integration method.

    Parameters
    ----------
    profile : `np.ndarray`
        The pulse profile.
    windowsize : `int`, optional
        Window width (in bins) defining the trial regions to integrate. If `None`, then will use 1/8
        of the profile. Default: `None`
    logger : `logging.Logger`, optional
        A logger to use. Default: `None`.

    Returns
    -------
    noise_est : `float`
        The standard deviation of the offpulse noise.
    """
    if logger is None:
        logger = psrutils.get_logger()

    op_l, op_r = psrutils.find_optimal_pulse_window(profile, windowsize=windowsize, logger=logger)

    bins = np.arange(profile.size)

    if op_r > op_l:
        mask = np.where(np.logical_and(bins > op_l, bins < op_r), True, False)
    else:
        mask = np.where(np.logical_or(bins > op_l, bins < op_r), True, False)

    # Standard deviation of the noise in the offpulse window
    return np.std(profile[mask])


def find_onpulse_regions(
    profile: np.ndarray,
    ntrials: int = 10,
    tol: float = 0.1,
    sigma_cutoff: float = 2.0,
    plotname: str = None,
    logger: logging.Logger | None = None,
) -> tuple[list, list, list, float, dict]:
    """Estimate the on- and off-pulse region(s) of a pulse profile.

    The method is described as follows:

    (1) Make an initial estimate of the offpulse noise by finding the pulse window which minimises
        the integrated signal within it.

    (2) Fit a periodic smoothing spline with the smoothness parameter set to `nbin*noise_est**2`.

    (3) Find the maxima of the profile which exceed `noise_est*sigma_cutoff`. If no maxima exceed
        the cutoff, then return `None` for each output.

    (4) Find the flanking inflections and minima around the maxima. These define the under- and
        over-estimates of the onpulse regions. If the flux between two onpulse regions exceeds the
        sigma cutoff, then those regions are considered a single region.

    (5) Calculate the standard deviation of the offpulse noise, where the offpulse is all bins
        not included in the onpulse overestimate.

    (6) Repeat (2)-(5) until `ntrials` have been reached or the fractional difference between
        subsequent noise estimates is below `tol`.

    Parameters
    ----------
    profile : `np.ndarray`
        The pulse profile.
    ntrials : `int`, optional
        The maximum number of times to iterate the noise estimate. Default: 10.
    tol : `float`, optional
        The minimum fractional difference between subsequent noise estimates before exiting the
        loop. Default: 0.1.
    sigma_cutoff : `float`, optional
        The number of standard deviations above which a peak will be considered real. Default: 2.0.
    plotname : `str`, optional
        If provided, will save a plot of the smoothed profile, its derivatives, and the onpulse
        estimates. The file extension should be **excluded**.
    logger : `logging.Logger`, optional
        A logger to use. Default: `None`.

    Returns
    -------
    underest_onpulse_pairs : `list` | `None`
        A list of pairs of bin indices defining the underestimated onpulse region.
        May miss some onpulse but will not include any offpulse.
    overest_onpulse_pairs : `list` | `None`
        A list of pairs of bin indices defining the overestimated onpulse region.
        May include some offpulse but will not miss any onpulse.
    offpulse_pairs : `list` | `None`
        A list of pairs of bin indices defining the offpulse. These pairs are the
        opposite of the 'overest_onpulse_pairs'.
    new_noise_est : `float` | `None`
        The standard deviation of the offpulse noise.
    plot_dict : `dict` | `None`
        A dictionary containing the data needed to plot the profile, the spline, its derivatives,
        and the onpulse estimates.
    """
    if logger is None:
        logger = psrutils.get_logger()

    nbin = profile.size
    bins = np.arange(nbin)

    old_noise_est = None

    logger.debug("Bootstrapping to estimate on/off-pulse and noise")
    for trial in range(ntrials):
        if old_noise_est is None:
            # Get offpulse noise using sliding window method
            old_noise_est = psrutils.est_offpulse_noise(profile, logger=logger)

        # Fit a degree-k smoothing spline in the B-spline basis with periodic boundary
        # conditions
        tck = splrep(bins, profile, k=5, s=nbin * old_noise_est**2, per=True)
        splrep_bspl = BSpline(*tck, extrapolate="periodic")

        # Convert to a piecewise degree-k polynomial representation, which has a better
        # root finding algorithm
        splrep_ppoly = PPoly.from_spline(splrep_bspl, extrapolate="periodic")

        # Find the onpulse regions
        underest_onpulse_pairs, overest_onpulse_pairs, plot_dict = _find_onpulse_regions_from_ppoly(
            bins, splrep_ppoly, old_noise_est, sigma_cutoff=sigma_cutoff, logger=logger
        )

        if overest_onpulse_pairs is None:
            # Either the onpulse is nothing or the whole profile
            offpulse_pairs = None
            new_noise_est = None
            break

        # If the onpulse detection was successful, then calculate the offpulse noise
        offpulse_pairs = psrutils.get_offpulse_from_onpulse(overest_onpulse_pairs)
        _, offpulse = psrutils.get_profile_mask_from_pairs(profile, offpulse_pairs)
        new_noise_est = np.nanstd(offpulse)

        # Check whether the tolerance level has been reached
        if old_noise_est is not None:
            tollvl = abs((old_noise_est - new_noise_est) / new_noise_est)
            logger.debug(f"Iteration {trial}: {new_noise_est=} {tollvl=}")
            tolcol = tollvl <= tol
        else:
            tolcol = False

        if tolcol:
            logger.debug(f"Took {trial + 1} trials to reach tolerance")
            break

        if trial + 1 == ntrials:
            logger.debug("Reached max number of trials")
            break

        old_noise_est = new_noise_est

    # If a plotname is provided, save a plot showing the smoothed profile, its
    # derivatives, and the onpulse estimates
    if plot_dict is not None and plotname:
        _plot_onpulse_regions(
            plot_dict,
            profile,
            new_noise_est,
            savename=plotname,
            logger=logger,
        )

    return underest_onpulse_pairs, overest_onpulse_pairs, offpulse_pairs, new_noise_est, plot_dict


def get_profile_mask_from_pairs(profile: np.ndarray, bin_pairs: list) -> np.ndarray:
    """Return a profile with the regions defined by the bin_pairs filled with NaNs.

    Parameters
    ----------
    profile : `np.ndarray`
        The profile to mask.
    bin_pairs : `list`
        A list of pairs of bin indices defining the profile regions.

    Returns
    -------
    mask : `np.ndarray`
        A profile mask that is True between the bin pairs.
    masked_profile : `np.ndarray`
        The profile with the regions between the bin pairs filled with Nans.
    """
    bins = np.arange(profile.size)
    mask = np.full(profile.size, False)
    for pair in bin_pairs:
        if pair[0] < pair[1]:
            mask = np.logical_or(mask, np.logical_and(bins > pair[0], bins < pair[1]))
        else:
            mask = np.logical_or(mask, bins > pair[0])
            mask = np.logical_or(mask, bins < pair[1])
    return mask, np.where(mask, profile, np.nan)


def get_offpulse_from_onpulse(onpulse_pairs: list) -> list:
    """Given a list of pairs of bin indices defining the onpulse regions, find the corresponding
    list of pairs of bin indices defining the offpulse regions.

    Parameters
    ----------
    onpulse_pairs : `list`
        A list of pairs of bin indices defining the onpulse.

    Returns
    -------
    offpulse_pairs : `list`
        A list of pairs of bin indices defining the offpulse.
    """
    offpulse_pairs = []

    # Make cycling generators to deal with wrap-around
    current_pair_gen = itertools.cycle(onpulse_pairs)
    next_pair_gen = itertools.cycle(onpulse_pairs)
    next(next_pair_gen)

    # Figure out the off-pulse profile
    for _ in range(len(onpulse_pairs)):
        current_pair = next(current_pair_gen)
        next_pair = next(next_pair_gen)
        offpulse_pairs.append([current_pair[1], next_pair[0]])

    return offpulse_pairs


def _find_onpulse_regions_from_ppoly(
    bins: np.ndarray,
    ppoly: PPoly,
    noise_est: float,
    sigma_cutoff: float = 2.0,
    logger: logging.Logger | None = None,
) -> tuple[list, list, dict]:
    """Find under- and over-estimates of the onpulse regions from a PPoly representation of a
    smoothed profile. If true maxima cannot be found (i.e. no maxima exceed the cutoff), then `None`
    will be returned for each of the outputs.

    Parameters
    ----------
    bins : `np.ndarray`
        An array of bin indices.
    ppoly : `PPoly`
        A degree >=3 PPoly object describing the smoothed profile.
    noise_est : `float`
        The standard deviation of the offpulse noise.
    sigma_cutoff : `float`, optional
        The number of standard deviations above which a peak will be considered real. Default: 2.0.
    logger : `logging.Logger`, optional
        A logger to use. Default: `None`.

    Returns
    -------
    underest_onpulse : `list` | `None`
        A list of pairs of underestimated onpulse bin indices.
    overest_onpulse : `list` | `None`
        A list of pairs of overestimated onpulse bin indices.
    plot_dict: `dict` | `None`
        A dictionary containing the data needed to plot the profile, the spline, its derivatives,
        and the onpulse estimates.
    """
    if logger is None:
        logger = psrutils.get_logger()

    # Evaluate derivatives and roots
    d1_ppoly = ppoly.derivative(nu=1)
    d1_roots = [round(rt) for rt in d1_ppoly.roots()]
    d2_ppoly = ppoly.derivative(nu=2)
    d2_roots = [round(rt) for rt in d2_ppoly.roots()]

    # Remove duplicates due to rounding
    d1_roots = list(set(d1_roots))
    d2_roots = list(set(d2_roots))

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
    true_maxima = []
    false_maxima = []
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
        return None, None, None

    underest_onpulse = []
    overest_onpulse = []
    for d1_root in true_maxima:
        # Underestimated onpulse
        # Figure out which two inflections (d2 roots) flank the maximum (d1 root)
        d2_roots = sorted(np.append(d2_roots, d1_root))
        mloc = d2_roots.index(d1_root)
        d2_roots.remove(d1_root)
        try:
            underestimate = [d2_roots[mloc - 1], d2_roots[mloc]]
        except IndexError:
            # The next inflection is on the other side of the profile
            if mloc == 0:  # Left side
                underestimate = [d2_roots[-1], d2_roots[mloc]]
            else:  # Right side
                underestimate = [d2_roots[mloc - 1], d2_roots[0]]
        underest_onpulse.append(underestimate)

        # Overestimated onpulse
        # Figure out which two false minima (d1 roots) flank the maximum (d1 root)
        minima = sorted(np.append(minima, d1_root))
        mloc = minima.index(d1_root)
        minima.remove(d1_root)
        try:
            overestimate = [minima[mloc - 1], minima[mloc]]
        except IndexError:
            # The next inflection is on the other side of the profile
            if mloc == 0:  # Left side
                overestimate = [minima[-1], minima[mloc]]
            else:  # Right side
                overestimate = [minima[mloc - 1], minima[0]]
        overest_onpulse.append(overestimate)

    # Make sure the regions are non-overlapping and non-contiguous
    underest_onpulse = _merge_and_sort_pairs(underest_onpulse, bins.size, logger=logger)
    overest_onpulse = _merge_and_sort_pairs(overest_onpulse, bins.size, logger=logger)

    # Fill the onpulse regions if there is signal between them
    if underest_onpulse is not None:
        underest_onpulse = _fill_onpulse_regions(underest_onpulse, sm_prof, noise_est, sigma_cutoff)
    if overest_onpulse is not None:
        overest_onpulse = _fill_onpulse_regions(overest_onpulse, sm_prof, noise_est, sigma_cutoff)

    # Store all the information required to make a plot
    plot_dict = dict(
        bins=bins,
        sm_prof=sm_prof,
        d1_sm_prof=d1_sm_prof,
        d2_sm_prof=d2_sm_prof,
        underest_onpulse=underest_onpulse,
        overest_onpulse=overest_onpulse,
    )

    return underest_onpulse, overest_onpulse, plot_dict


def _merge_and_sort_pairs(pairs: list, nbin: int, logger: logging.Logger | None = None) -> list:
    """From a list of bin index pairs, generate a new list of bin index pairs with no overlap and
    no continguous regions.

    Parameters
    ----------
    pairs : `list`
        A list of bin index pairs.
    nbin : `int`
        The total number of bins in the profile.
    logger : `logging.Logger`, optional
        A logger to use. Default: `None`.

    Returns
    -------
    new_pairs : `list` | `None`
        A list of merged and sorted bin index pairs.
    """
    if logger is None:
        logger = psrutils.get_logger()

    bins = np.arange(nbin)
    mask = np.full(nbin, False)

    for pair in pairs:
        if pair[0] == pair[1]:
            continue
        elif pair[0] < pair[1]:
            submask = np.logical_and(bins >= pair[0], bins <= pair[1])
        else:
            submask = np.logical_or(bins >= pair[-1], bins <= pair[0])
        mask = np.logical_or(mask, submask)

    if np.logical_and.reduce(np.logical_not(mask)):
        logger.debug("Merge found no bins contained in a valid region")
        return None

    if np.logical_and.reduce(mask):
        logger.debug("Merge found all bins contained in a valid region")
        return None

    new_pairs = []
    left = 0
    consec = 0
    for ibin in bins:
        if not mask[ibin]:
            if consec > 0:
                new_pairs.append([left, ibin - 1])
                consec = 0
            continue
        if consec == 0:
            left = ibin
        consec += 1

    if new_pairs[-1][1] == new_pairs[0][0]:
        new_pairs = new_pairs[1:-2].append([new_pairs[-1][0], new_pairs[0][1]])

    return new_pairs


def _fill_onpulse_regions(
    onpulse_pairs: list,
    sm_prof: np.ndarray,
    noise_est: float,
    sigma_cutoff: float,
) -> list:
    """Attempts to fill small gaps in the onpulse pairs provided these gaps exceed a set sigma
    threshold (i.e. they resemble real signal.)

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
    plot_dict: dict,
    real_prof: np.ndarray,
    noise_est: float,
    plot_underestimate: bool = True,
    plot_overestimate: bool = True,
    lw: float = 1.2,
    savename: str = "onpulse",
    logger: logging.Logger | None = None,
) -> None:
    """Plot the smoothed profile and its derivatives, as well as the onpulse regions.

    Parameters
    ----------
    plot_dict: dict
        A plot dictionary from `_find_onpulse_regions_from_ppoly()`.
    real_prof : np.ndarray
        The real profile, with the same number of bins as the plot_dict.
    noise_est : `float`
        An estimate of the standard deviation of the offpulse noise.
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
    bins = plot_dict["bins"]
    sm_prof = plot_dict["sm_prof"]
    d1_sm_prof = plot_dict["d1_sm_prof"]
    d2_sm_prof = plot_dict["d2_sm_prof"]
    underest_onpulse = plot_dict["underest_onpulse"]
    overest_onpulse = plot_dict["overest_onpulse"]

    # Figure setup
    fig, axes = plt.subplots(nrows=3, figsize=(8, 7), sharex=True, dpi=300, tight_layout=True)
    xrange = (bins[0], bins[-1])

    # Profile
    axes[0].plot(bins, real_prof, color="k", linewidth=lw, alpha=0.2, label="Real Profile")
    axes[0].plot(bins, sm_prof, color="k", linewidth=lw * 0.8, label="Smoothed Profile")
    yrange = axes[0].get_ylim()

    xpos = (xrange[1] - xrange[0]) * 0.95 + xrange[0]
    ypos = (yrange[-1] - yrange[0]) * 0.8 + yrange[0]
    axes[0].errorbar(xpos, ypos, yerr=noise_est, color="k", marker="none", capsize=3, elinewidth=lw)

    # Onpulse estimates
    if overest_onpulse is not None:
        underest_args = dict(color="tab:blue", alpha=0.2, hatch="///", zorder=0.1)
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
                        axes[0].fill_betweenx(
                            yrange, op_pair[0], op_pair[-1], label=used_label, **args
                        )
                    else:
                        axes[0].fill_betweenx(
                            yrange, op_pair[0], xrange[-1], label=used_label, **args
                        )
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
