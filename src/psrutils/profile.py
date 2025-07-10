import itertools
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.interpolate import BSpline, PPoly, splrep

from .logger import get_logger

logger = get_logger(__name__)


__all__ = ["Profile", "get_profile_mask_from_pairs", "get_offpulse_from_onpulse"]


class Profile(object):
    """A class for storing and analysing a pulse profile"""

    def __init__(self, profile: npt.ArrayLike) -> None:
        """Create a Profile instance from an array.

        Parameters
        ----------
        profile : array_like
            A 1-dimensional array containing the pulse profile.
        """
        profile = np.array(profile, dtype=float)

        if profile.ndim != 1:
            raise ValueError("profile must be a 1-dimensional array")

        self._prof = profile
        self._nbin = profile.size
        self._bins = np.arange(profile.size, dtype=int)

    @property
    def profile(self) -> npt.NDArray[np.float_]:
        """The real pulse profile."""
        return self._prof

    @property
    def nbin(self) -> int:
        """The number of phase bins in the profile."""
        return self._nbin

    @property
    def bins(self) -> npt.NDArray[np.int_]:
        """An array of bin indices."""
        return self._bins

    @property
    def ppoly(self) -> PPoly:
        """The spline fit to the pulse profile as a PPoly object."""
        if not self._ppoly:
            raise AttributeError("The spline fit has not been computed")
        return self._ppoly

    @property
    def noise_est(self) -> np.float_:
        """An estimate of the standard deviation of the offpulse noise."""
        if not self._noise_est:
            self.get_simple_noise_est()
        return self._noise_est

    @property
    def underest_onpulse_pairs(self) -> list[tuple[np.int_, np.int_]]:
        """A list of pairs of bin indices defining the underestimated onpulse region(s)."""
        if not self._underest_onpulse_pairs:
            raise AttributeError("Onpulse has not been computed")
        return self._underest_onpulse_pairs

    @property
    def overest_onpulse_pairs(self) -> list[tuple[np.int_, np.int_]]:
        """A list of pairs of bin indices defining the overestimated onpulse region(s)."""
        if not self._overest_onpulse_pairs:
            raise AttributeError("Onpulse has not been computed")
        return self._overest_onpulse_pairs

    @property
    def offpulse_pairs(self) -> list[tuple[np.int_, np.int_]]:
        """A list of pairs of bin indices defining the offpulse region(s)."""
        if not self._offpulse_pairs:
            raise AttributeError("Offpulse has not been computed")
        return self._offpulse_pairs

    @property
    def underest_onpulse_bins(self) -> npt.NDArray[np.int_]:
        """An array of bins in the underestimated onpulse region(s)."""
        if not self._underest_onpulse_pairs:
            raise AttributeError("Onpulse has not been computed")
        if self._underest_onpulse_pairs is None:
            return None
        mask = get_profile_mask_from_pairs(self._nbin, self._underest_onpulse_pairs)
        return self._bins[mask]

    @property
    def overest_onpulse_bins(self) -> npt.NDArray[np.int_]:
        """An array of bins in the overestimated onpulse region(s)."""
        if not self._overest_onpulse_pairs:
            raise AttributeError("Onpulse has not been computed")
        if self._overest_onpulse_pairs is None:
            return None
        mask = get_profile_mask_from_pairs(self._nbin, self._overest_onpulse_pairs)
        return self._bins[mask]

    @property
    def offpulse_bins(self) -> npt.NDArray[np.int_]:
        """An array of bins in the offpulse region(s)."""
        if not self._offpulse_pairs:
            raise AttributeError("Offpulse has not been computed")
        if self._offpulse_pairs is None:
            return None
        mask = get_profile_mask_from_pairs(self._nbin, self._offpulse_pairs)
        return self._bins[mask]

    def get_opt_pulse_window(
        self, windowsize: int | None = None, maximise: bool = False
    ) -> tuple[int, int]:
        """Find the pulse window which minimises/maximises the integrated flux density within it.
        Noise should integrate towards zero, so minimising the integral will find an offpulse
        window. Conversely, maximising the integral will find an onpulse window. The window size
        should be tweaked depending on the pulsar.

        Method taken from PyPulse (Lam, 2017. https://ascl.net/1706.011).

        Parameters
        ----------
        windowsize : `int`, optional
            Window width (in bins) defining the trial regions to integrate. If `None`, then will use
            1/8 of the profile. Default: `None`.
        maximise : `bool`, optional
            If `True`, will maximise the integral; otherwise, will minimise. Default: `False`.

        Returns
        -------
        left_idx : `int`
            The bin index of the leading edge of the window.
        right_idx : `int`
            The bin index of the trailing edge of the window.
        """
        if windowsize is None:
            windowsize = self._nbin // 8

        integral = np.zeros_like(self._prof)
        for i in range(self._nbin):
            win = np.arange(i - windowsize // 2, i + windowsize // 2) % self._nbin
            integral[i] = np.trapz(self._prof[win])

        if maximise:
            opt_idx = np.argmax(integral)
        else:
            opt_idx = np.argmin(integral)

        left_idx = (opt_idx - windowsize // 2) % self._nbin
        right_idx = (opt_idx + windowsize // 2) % self._nbin

        return left_idx, right_idx

    def get_simple_noise_est(self) -> np.float_:
        """Get an estimate of the standard deviation of the offpulse noise using the sliding window
        method.

        Returns
        -------
        noise_est : `float`
            The standard deviation of the profile within the minimised sliding window.
        """
        # Find the window with the minimum integrated flux within it
        op_l, op_r = self.get_opt_pulse_window()

        # Mask out the onpulse
        if op_r > op_l:
            mask = np.where(np.logical_and(self._bins > op_l, self._bins < op_r), True, False)
        else:
            mask = np.where(np.logical_or(self._bins > op_l, self._bins < op_r), True, False)
        offpulse = self._prof[mask]

        return np.std(offpulse)

    def fit_spline(self, sigma: float, k: int = 5) -> None:
        """Fit a periodic smoothing spline with a smoothness equal to `nbin*sigma**2`, where `sigma`
        is an estimate of the standard deviation of the noise in the profile.

        Parameters
        ----------
        sigma : `float`
            The standard deviation of the noise in the profile.
        k : `int`
            The degree of the spline fit. See `scipy.interpolate.splrep()` for details.
        """
        # Fit a degree-k smoothing spline in the B-spline basis with periodic boundary conditions
        tck = splrep(self._bins, self._prof, k=k, s=self._nbin * sigma**2, per=True)
        bspl = BSpline(*tck, extrapolate="periodic")

        # Convert to a piecewise degree-k polynomial representation
        ppoly = PPoly.from_spline(bspl, extrapolate="periodic")

        self._ppoly = ppoly

    def find_onpulse_regions(
        self, noise_est: float | None = None, sigma_cutoff: float = 2.0
    ) -> None:
        """Find under- and over-estimates of the onpulse region(s).

        Parameters
        ----------
        noise_est : `float`, optional
            An estimate of the standard deviation of the offpulse noise that will be used to
            determine if a pulse peak exceeds the sigma cutoff.
        sigma_cutoff : `float`, optional
            The number of standard deviations above which a peak will be considered real.
            Default: 2.0.
        """
        if noise_est is None:
            if not self._noise_est:
                noise_est = self.get_simple_noise_est()
            else:
                noise_est = self._noise_est

        # Evaluate derivatives and roots
        d1_ppoly = self._ppoly.derivative(nu=1)
        d1_roots = [round(rt) for rt in d1_ppoly.roots()]
        d2_ppoly = self._ppoly.derivative(nu=2)
        d2_roots = [round(rt) for rt in d2_ppoly.roots()]

        # Remove duplicates due to rounding
        d1_roots = list(set(d1_roots))
        d2_roots = list(set(d2_roots))

        # Interpolate smoothed profiles
        spl_prof = self._ppoly(self._bins)
        d2_spl_prof = d2_ppoly(self._bins)

        # Filter out extrapolated roots
        d1_roots_inbounds = []
        for root in d1_roots:
            if root < self._bins[0] or root > self._bins[-1]:
                continue
            d1_roots_inbounds.append(root)
        d2_roots_inbounds = []
        for root in d2_roots:
            if root < self._bins[0] or root > self._bins[-1]:
                continue
            d2_roots_inbounds.append(root)

        # Catagorise minima and maxima
        minima = []
        true_maxima = []
        false_maxima = []
        for root in d1_roots_inbounds:
            if d2_spl_prof[root] > 0:  # Minimum
                minima.append(root)
            else:  # Maximum
                if spl_prof[root] > noise_est * sigma_cutoff:
                    true_maxima.append(root)
                else:
                    false_maxima.append(root)

        if not true_maxima:
            logger.warning(f"No profile maxima found >{sigma_cutoff} sigma.")
            return None, None, None

        underest_onpulse_pairs = []
        overest_onpulse_pairs = []
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
            underest_onpulse_pairs.append(underestimate)

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
            overest_onpulse_pairs.append(overestimate)

        # Make sure the regions are non-overlapping and non-contiguous
        underest_onpulse_pairs = _merge_and_sort_pairs(underest_onpulse_pairs, self._nbin)
        overest_onpulse_pairs = _merge_and_sort_pairs(overest_onpulse_pairs, self._nbin)

        # Fill the onpulse regions if there is signal between them
        if underest_onpulse_pairs is not None:
            underest_onpulse_pairs = _fill_onpulse_regions(
                underest_onpulse_pairs, spl_prof, noise_est, sigma_cutoff
            )
        if overest_onpulse_pairs is not None:
            overest_onpulse_pairs = _fill_onpulse_regions(
                overest_onpulse_pairs, spl_prof, noise_est, sigma_cutoff
            )

        self._spl_prof = spl_prof
        self._sigma_cutoff = sigma_cutoff
        self._underest_onpulse_pairs = underest_onpulse_pairs
        self._overest_onpulse_pairs = overest_onpulse_pairs

    def bootstrap_onpulse_regions(
        self, ntrials: int = 10, tol: float = 0.1, sigma_cutoff: float = 2.0
    ) -> None:
        """Bootstrap the onpulse finding method by fitting a new spline with the estimated offpulse
        noise on each iteration.

        The algorithm follows these steps:

        (1) Make an initial estimate of the offpulse noise by finding the pulse window which
            minimises the integrated signal within it.

        (2) Fit a periodic smoothing spline with the smoothness parameter set to
            `nbin*noise_est**2`.

        (3) Find the maxima of the profile which exceed `noise_est*sigma_cutoff`. If no maxima
            exceed the cutoff, then return `None` for each output.

        (4) Find the flanking inflections and minima around the maxima. These define the under- and
            over-estimates of the onpulse regions. If the flux between two onpulse regions exceeds
            the sigma cutoff, then those regions are considered a single region.

        (5) Calculate the standard deviation of the offpulse noise, where the offpulse is all bins
            not included in the onpulse overestimate.

        (6) Repeat (2)-(5) until `ntrials` have been reached or the fractional difference between
            subsequent noise estimates is below `tol`.

        Parameters
        ----------
        ntrials : `int`, optional
            The maximum number of times to iterate the noise estimate. Default 10.
        tol : `float`, optional
            The minimum fractional difference between subsequent noise estimates before exiting the
            loop. Default 0.1.
        sigma_cutoff : `float`, optional
            The number of standard deviations above which a peak will be considered real.
            Default 2.0.
        """
        old_noise_est = None

        logger.debug("Bootstrapping to estimate on/off-pulse and noise")
        for trial in range(ntrials):
            if old_noise_est is None:
                # Get offpulse noise using sliding window method
                old_noise_est = self.get_simple_noise_est()

            self.fit_spline(old_noise_est)
            self.find_onpulse_regions(old_noise_est, sigma_cutoff=sigma_cutoff)

            if self._overest_onpulse_pairs is None:
                # Either the onpulse is non-existant or the whole profile is onpulse
                offpulse_pairs = None
                new_noise_est = None
                break

            # If the onpulse detection was successful, then calculate the offpulse noise
            offpulse_pairs = get_offpulse_from_onpulse(self._overest_onpulse_pairs)
            mask = get_profile_mask_from_pairs(self._nbin, offpulse_pairs)
            new_noise_est = np.nanstd(self._prof[mask])

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

        self._offpulse_pairs = offpulse_pairs
        self._noise_est = new_noise_est

    def plot_onpulse_regions(
        self,
        plot_underestimate: bool = True,
        plot_overestimate: bool = True,
        lw: float = 1.2,
        savename: str = "onpulse",
    ) -> None:
        # Figure setup
        fig, axes = plt.subplots(nrows=3, figsize=(8, 7), sharex=True, dpi=300, tight_layout=True)
        xrange = (self._bins[0], self._bins[-1])

        # Profile
        axes[0].plot(
            self._bins, self._prof, color="k", linewidth=lw, alpha=0.2, label="Real Profile"
        )
        axes[0].plot(
            self._bins,
            self._ppoly(self._bins),
            color="k",
            linewidth=lw * 0.8,
            label="Smoothed Profile",
        )
        yrange = axes[0].get_ylim()

        xpos = (xrange[1] - xrange[0]) * 0.95 + xrange[0]
        ypos = (yrange[-1] - yrange[0]) * 0.8 + yrange[0]
        axes[0].errorbar(
            xpos, ypos, yerr=self._noise_est, color="k", marker="none", capsize=3, elinewidth=lw
        )

        # Onpulse estimates
        if self._overest_onpulse_pairs is not None:
            underest_args = dict(color="tab:blue", alpha=0.2, hatch="///", zorder=0.1)
            overest_args = dict(color="tab:blue", edgecolor=None, alpha=0.2, zorder=0)
            for pairlist, args, flag, label in zip(
                [self._underest_onpulse_pairs, self._overest_onpulse_pairs],
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
        d1_ppoly = self._ppoly.derivative(nu=1)
        d2_ppoly = self._ppoly.derivative(nu=2)
        axes[1].plot(
            self._bins, d1_ppoly(self._bins), color="k", linewidth=lw * 0.8, label="1st Derivative"
        )
        axes[2].plot(
            self._bins, d2_ppoly(self._bins), color="k", linewidth=lw * 0.8, label="2nd Derivative"
        )

        # Finishing touches
        axes[0].set_ylim(yrange)
        axes[2].set_xlabel("Bin Index")
        for ax in axes:
            ax.axhline(0, linestyle=":", color="k", linewidth=lw * 0.8, alpha=0.3, zorder=1)
            ax.set_xlim(xrange)
            ax.legend(loc="upper left")

        logger.info(f"Saving plot file: {savename}.png")
        fig.savefig(savename + ".png")


def get_profile_mask_from_pairs(
    nbin: int, bin_pairs: list[tuple[np.int_, np.int_]]
) -> npt.NDArray[np.bool_]:
    """Return a profile with the regions defined by the bin_pairs filled with NaNs.

    Parameters
    ----------
    nbin : `int`
        The number of phase bins in the profile.
    bin_pairs : `list`
        A list of pairs of bin indices defining the profile region(s).

    Returns
    -------
    mask : `np.ndarray`
        A profile mask that is True between the bin pairs.
    masked_profile : `np.ndarray`
        The profile with the regions between the bin pairs filled with Nans.
    """
    bins = np.arange(nbin)
    mask = np.full(nbin, False)

    for pair in bin_pairs:
        if pair[0] < pair[1]:
            mask = np.logical_or(mask, np.logical_and(bins > pair[0], bins < pair[1]))
        else:
            mask = np.logical_or(mask, bins > pair[0])
            mask = np.logical_or(mask, bins < pair[1])

    return mask


def get_offpulse_from_onpulse(
    onpulse_pairs: list[tuple[np.int_, np.int_]],
) -> list[tuple[np.int_, np.int_]]:
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


def _merge_and_sort_pairs(
    pairs: list[tuple[np.int_, np.int_]], nbin: int
) -> list[tuple[np.int_, np.int_]]:
    """From a list of bin index pairs, generate a new list of bin index pairs with no overlap and
    no continguous regions.

    Parameters
    ----------
    pairs : `list`
        A list of bin index pairs.
    nbin : `int`
        The total number of bins in the profile.

    Returns
    -------
    new_pairs : `list` | `None`
        A list of merged and sorted bin index pairs.
    """
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
    sm_prof: npt.NDArray[np.floating[Any]],
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
