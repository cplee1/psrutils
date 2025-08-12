########################################################
# Licensed under the Academic Free License version 3.0 #
########################################################

import bisect
import itertools
import logging
from typing import Iterable

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from astropy.stats import histogram
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import BSpline, PPoly, splrep
from scipy.stats import shapiro
from statsmodels.sandbox.stats.runs import runstest_1samp
from statsmodels.stats.diagnostic import acorr_ljungbox

__all__ = ["SplineProfile", "get_profile_mask_from_pairs", "get_offpulse_from_onpulse"]

logger = logging.getLogger(__name__)


# TODO: Format docstrings
class SplineProfile(object):
    """A class for storing and analysing a pulse profile"""

    def __init__(self, profile: ArrayLike) -> None:
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
        self._bins = np.arange(self._nbin, dtype=int)
        self._llim = self._bins[0]
        self._rlim = self._bins[0] + self._nbin
        self._lims = (self._llim, self._rlim)
        self._noise_est: np.float_ | None = None

    @property
    def profile(self) -> NDArray[np.float_]:
        """The real pulse profile."""
        return self._prof

    @property
    def nbin(self) -> int:
        """The number of phase bins in the profile."""
        return self._nbin

    @property
    def bins(self) -> NDArray[np.int_]:
        """An array of bin indices."""
        return self._bins

    @property
    def phases(self) -> NDArray[np.float_]:
        """An array of bin phases."""
        return self._bins / (self._nbin - 1)

    @property
    def bspl(self) -> BSpline:
        """The spline fit to the pulse profile as a BSPpline object."""
        if not hasattr(self, "_bspl") or self._bspl is None:
            raise AttributeError("The spline fit has not been computed")
        return self._bspl

    @property
    def ppoly(self) -> PPoly:
        """The spline fit to the pulse profile as a PPoly object."""
        if not hasattr(self, "_ppoly") or self._ppoly is None:
            raise AttributeError("The spline fit has not been computed")
        return self._ppoly

    @property
    def residuals(self) -> NDArray[np.float_]:
        """The real profile subtract the smoothed profile."""
        if not hasattr(self, "_bspl") or self._bspl is None:
            raise AttributeError("The spline fit has not been computed")
        return self._prof - self._bspl(self._bins)

    @property
    def noise_est(self) -> np.float_:
        """An estimate of the standard deviation of the offpulse noise."""
        if hasattr(self, "_noise_est") and self._noise_est is not None:
            return self._noise_est
        return np.std(self.residuals)

    @property
    def baseline_est(self) -> np.float_:
        """An estimate of the mean of the offpulse noise"""
        mask = get_profile_mask_from_pairs(self._nbin, self.offpulse_pairs)
        return np.mean(self._prof[mask])

    @property
    def underest_onpulse_pairs(self) -> list[tuple[np.int_, np.int_]]:
        """A list of pairs of bin indices defining the underestimated
        onpulse region(s)."""
        if (
            not hasattr(self, "_underest_onpulse_pairs")
            or self._underest_onpulse_pairs is None
        ):
            raise AttributeError("Onpulse has not been computed")
        return self._underest_onpulse_pairs

    @property
    def overest_onpulse_pairs(self) -> list[tuple[np.int_, np.int_]]:
        """A list of pairs of bin indices defining the overestimated
        onpulse region(s)."""
        if (
            not hasattr(self, "_overest_onpulse_pairs")
            or self._overest_onpulse_pairs is None
        ):
            raise AttributeError("Onpulse has not been computed")
        return self._overest_onpulse_pairs

    @property
    def offpulse_pairs(self) -> list[tuple[np.int_, np.int_]]:
        """A list of pairs of bin indices defining the offpulse
        region(s)."""
        if not hasattr(self, "_offpulse_pairs") or self._offpulse_pairs is None:
            raise AttributeError("Offpulse has not been computed")
        return self._offpulse_pairs

    @property
    def underest_onpulse_bins(self) -> NDArray[np.int_]:
        """An array of bins in the underestimated onpulse region(s)."""
        if (
            not hasattr(self, "_underest_onpulse_pairs")
            or self._underest_onpulse_pairs is None
        ):
            raise AttributeError("Onpulse has not been computed")
        mask = get_profile_mask_from_pairs(self._nbin, self._underest_onpulse_pairs)
        return self._bins[mask]

    @property
    def overest_onpulse_bins(self) -> NDArray[np.int_]:
        """An array of bins in the overestimated onpulse region(s)."""
        if (
            not hasattr(self, "_overest_onpulse_pairs")
            or self._overest_onpulse_pairs is None
        ):
            raise AttributeError("Onpulse has not been computed")
        mask = get_profile_mask_from_pairs(self._nbin, self._overest_onpulse_pairs)
        return self._bins[mask]

    @property
    def offpulse_bins(self) -> NDArray[np.int_]:
        """An array of bins in the offpulse region(s)."""
        if not hasattr(self, "_offpulse_pairs") or self._offpulse_pairs is None:
            raise AttributeError("Offpulse has not been computed")
        mask = get_profile_mask_from_pairs(self._nbin, self._offpulse_pairs)
        return self._bins[mask]

    def get_opt_pulse_window(
        self, windowsize: int | None = None, maximise: bool = False
    ) -> tuple[int, int]:
        """Find the pulse window which minimises/maximises the integrated
        flux density within it. Noise should integrate towards zero, so
        minimising the integral will find an offpulse window. Conversely,
        maximising the integral will find an onpulse window. The window
        size should be tweaked depending on the pulsar.

        Method taken from PyPulse (Lam, 2017. https://ascl.net/1706.011).

        Parameters
        ----------
        windowsize : `int`, optional
            Window width (in bins) defining the trial regions to integrate.
            If `None`, then will use 1/8 of the profile. Default: `None`.
        maximise : `bool`, optional
            If `True`, will maximise the integral; otherwise, will
            minimise. Default: `False`.

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

    # TODO: Make docstring
    def get_simple_noise_stats(self) -> tuple[np.float_, np.float_]:
        # Find the window with the minimum integrated flux within it
        op_l, op_r = self.get_opt_pulse_window()

        # Mask out the onpulse
        if op_r > op_l:
            mask = np.where(
                np.logical_and(self._bins > op_l, self._bins < op_r), True, False
            )
        else:
            mask = np.where(
                np.logical_or(self._bins > op_l, self._bins < op_r), True, False
            )
        offpulse = self._prof[mask]

        return np.mean(offpulse), np.std(offpulse)

    def fit_spline(self, sigma: float, k: int = 5) -> None:
        """Fit a periodic smoothing spline with a smoothness equal to
        `nbin*sigma**2`, where `sigma` is an estimate of the standard
        deviation of the noise in the profile.

        Parameters
        ----------
        sigma : `float`
            The standard deviation of the noise in the profile.
        k : `int`
            The degree of the spline fit. See `scipy.interpolate.splrep()`
            for details.
        """
        # Fit a degree-k smoothing spline in the B-spline basis with
        # periodic boundary conditions
        tck = splrep(self._bins, self._prof, k=k, s=self._nbin * sigma**2, per=True)
        bspl = BSpline(*tck, extrapolate="periodic")
        d1_bspl = bspl.derivative(nu=1)
        d2_bspl = bspl.derivative(nu=2)

        # Convert to a piecewise degree-k polynomial representation
        ppoly = PPoly.from_spline(bspl, extrapolate="periodic")
        d1_ppoly = PPoly.from_spline(d1_bspl, extrapolate="periodic")
        d2_ppoly = PPoly.from_spline(d2_bspl, extrapolate="periodic")

        self._bspl = bspl
        self._d1_bspl = d1_bspl
        self._d2_bspl = d2_bspl
        self._ppoly = ppoly
        self._d1_ppoly = d1_ppoly
        self._d2_ppoly = d2_ppoly

    def get_onpulse_regions(
        self, noise_est: float | None = None, sigma_cutoff: float = 2.0
    ) -> tuple[
        list[tuple[np.float_, np.float_]] | None,
        list[tuple[np.float_, np.float_]] | None,
        list[np.float_] | None,
    ]:
        """Find under- and over-estimates of the onpulse region(s) usin
        the following method:

        (1) Fit a smoothing spline with the smoothness parameter set to
            `nbin*noise_est**2`.

        (2) Find the maxima of the profile that exceed
            `noise_est*sigma_cutoff`.

        (3) Find the flanking inflections and minima around the maxima.
            These define the under- and over-estimates of the onpulse
            regions, respectively. If the flux between two adjacent onpulse
            regions exceeds `noise_est*sigma_cutoff`, then considered them
            a single region.

        Parameters
        ----------
        noise_est : `float`, optional
            An estimate of the standard deviation of the offpulse noise
            that will be used to determine if a pulse peak exceeds the
            sigma cutoff. If `None` is given, then the noise will be
            estimated using the `get_simple_noise_stats()` method.
            Default: `None`.
        sigma_cutoff : `float`, optional
            The number of standard deviations above which a peak will be
            considered real. Default: 2.0.

        Returns
        -------
        underest_onpulse_pairs : `list[tuple[np.float_, np.float_]] | None`
            A list of intervals defining the underestimated onpulse
            region(s) in bin units.
        overest_onpulse_pairs : `list[tuple[np.float_, np.float_]] | None`
            A list of intervals defining the overestimated onpulse
            region(s) in bin units.
        true_maxima : `list[np.float_] | None`
            A list of profile maxima above the sigma threshold in bin
            units.
        minima : `list[np.float_] | None`
            A list of all profile minima in bin units.
        """
        if noise_est is None:
            noise_est = self.get_simple_noise_stats()[1]

        # This will update the non-public spline attributes
        self.fit_spline(noise_est)

        # Evaluate the roots of the spline derivatives
        try:
            d1_roots = self._d1_ppoly.roots()
            d2_roots = self._d2_ppoly.roots()
        except ValueError:
            logger.error("Roots of spline could not be found.")
            return None, None, None, None

        # Remove out-of-bounds roots
        d1_roots = _get_inbounds_roots(d1_roots, self._lims)
        d2_roots = _get_inbounds_roots(d2_roots, self._lims)

        # Catagorise minima and maxima
        true_maxima = []
        minima = []
        for root in d1_roots:
            if self._d2_bspl(root) > 0:  # Minimum
                minima.append(root)
            else:  # Maximum
                if self._bspl(root) > noise_est * sigma_cutoff:
                    # We only care about true maxima
                    true_maxima.append(root)

        if not true_maxima:
            logger.error(f"No profile maxima found above {sigma_cutoff} sigma.")
            return None, None, None, None

        # The underestimate is defined by the flanking inflections
        underest_onpulse_pairs = _get_flanking_roots(true_maxima, d2_roots)

        # The overestimate is defined by the flanking minima
        overest_onpulse_pairs = _get_flanking_roots(true_maxima, minima)

        # Fill the onpulse regions if there is signal between them
        underest_onpulse_pairs = _fill_onpulse_regions(
            underest_onpulse_pairs, self._bspl, noise_est, 1.0, self._lims
        )
        overest_onpulse_pairs = _fill_onpulse_regions(
            overest_onpulse_pairs, self._bspl, noise_est, 1.0, self._lims
        )

        return underest_onpulse_pairs, overest_onpulse_pairs, true_maxima, minima

    def gridsearch_onpulse_regions(
        self, ntrials: int = 1000, logspan: float = 5.0, sigma_cutoff: float = 2.0
    ) -> None:
        """Perform a gridsearch over the smoothness parameter to find the spline
        fit which results in the whitest residuals.

        Parameters
        ----------
        ntrials : int, default: 1000
            The number of trials to evaluate over the search range.
        logspan : float, default: 5.0
            Search in the range [sigma/logspan, sigma*logspan], where sigma is
            an estimate of the noise in the profile.
        sigma_cutoff : float, default: 2.0
            The number of standard deviations above which a peak will be
            considered real.
        """
        # Get an initial noise estimate using the sliding window method
        noise_est = self.get_simple_noise_stats()[1]

        # Define log-spaced trials in the specified search range
        noise_est_trials = np.logspace(
            np.log10(noise_est / logspan), np.log10(noise_est * logspan), ntrials
        )

        p_values = np.empty(ntrials)
        for ii in range(ntrials):
            self.fit_spline(noise_est_trials[ii])
            # Runs test for non-random runs of signs
            p_runs = float(runstest_1samp(self.residuals)[1])
            # Ljung-Box test for autocorrelations
            p_lb = float(acorr_ljungbox(self.residuals, lags=[20])["lb_pvalue"])
            # Calculate a "whiteness score" which is the log of the combined
            # p-values, but note that since the tests are not independent, this
            # does not represent a true p-value
            p_values[ii] = np.log10(p_runs) + np.log10(p_lb)

        # Maximise the whiteness score
        idx_best = np.argmax(p_values)
        noise_est_best = noise_est_trials[idx_best]

        underest_onpp, overest_onpp, maxima, minima = self.get_onpulse_regions(
            noise_est_best, sigma_cutoff=sigma_cutoff
        )
        if overest_onpp is None or underest_onpp is None:
            offpp = None
        else:
            offpp = get_offpulse_from_onpulse(overest_onpp)

        self._noise_est_trials = noise_est_trials
        self._p_values = p_values
        self._offpulse_pairs = offpp
        self._overest_onpulse_pairs = overest_onpp
        self._underest_onpulse_pairs = underest_onpp
        self._maxima = maxima
        self._minima = minima
        self._noise_est = noise_est_best

    def bootstrap_onpulse_regions(
        self, ntrials: int = 10, tol: float = 0.05, sigma_cutoff: float = 2.0
    ) -> None:
        """Bootstrap the onpulse-finding method by fitting a spline with
        the new estimated offpulse noise on each iteration. The
        bootstrapping is repeated until either the maximum number of trials
        has been reaches or the fractional difference between subsequent
        noise estimates falls below a specified tolerance.

        See the `get_onpulse_regions()` method for further details.

        Parameters
        ----------
        ntrials : `int`, optional
            The maximum number of times to iterate the noise estimate.
            Default 10.
        tol : `float`, optional
            The minimum fractional difference between subsequent noise
            estimates before exiting the loop. Default 0.1.
        sigma_cutoff : `float`, optional
            The number of standard deviations above which a peak will be
            considered real. Default 2.0.
        """
        logger.debug("Bootstrapping to estimate onpulse...")

        old_noise_est = self.get_simple_noise_stats()[1]
        logger.debug(f"Initial noise estimate: {old_noise_est}")

        underest_onpp, overest_onpp, maxima, minima = self.get_onpulse_regions(
            old_noise_est, sigma_cutoff=sigma_cutoff
        )
        if overest_onpp is None or underest_onpp is None:
            raise RuntimeError("Bootstrapping failed on initial run")

        old_offpp = get_offpulse_from_onpulse(overest_onpp)
        old_overest_onpp = overest_onpp
        old_underest_onpp = underest_onpp
        old_maxima = maxima
        old_minima = minima
        old_tollvl = None

        for trial in range(ntrials):
            # Calculate the offpulse noise
            mask = get_profile_mask_from_pairs(self._nbin, old_offpp)

            if np.all(np.logical_not(mask)):
                logger.warning("Offpulse and onpulse region cannot be separated")
                break

            # Perform a normal test on the offpulse samples and the spline
            # residuals, and estimate the noise from whichever one is more
            # normally distributed
            if shapiro(self.residuals)[1] > shapiro(self._prof[mask])[1]:
                logger.debug("Using residuals to estimate noise")
                new_noise_est = np.std(self.residuals)
            else:
                logger.debug("Using offpulse to estimate noise")
                new_noise_est = np.std(self._prof[mask])

            if np.isclose(new_noise_est, 0.0):
                logger.warning("New noise estimate is zero")
                break

            # Check whether the tolerance level has been reached
            new_tollvl = abs((old_noise_est - new_noise_est) / new_noise_est)
            logger.debug(f"Trial {trial + 1}: {new_noise_est=} {new_tollvl=}")

            if old_tollvl is not None and new_tollvl > old_tollvl:
                logger.debug(f"Hit tolerance minimum on trial {trial}")
                break

            if new_tollvl <= tol:
                logger.debug(f"Reached tolerance on trial {trial + 1}")
                break

            if trial + 1 == ntrials:
                logger.debug(f"Reached max number of trials ({ntrials})")
                break

            # Perform the analysis
            underest_onpp, overest_onpp, maxima, minima = self.get_onpulse_regions(
                new_noise_est, sigma_cutoff=sigma_cutoff
            )
            if overest_onpp is None or underest_onpp is None:
                logger.warning(f"Bootstrapping failed on trial {trial}")
                # Roll back the spline attributes to the previous fit
                self.fit_spline(old_noise_est)
                break

            old_offpp = get_offpulse_from_onpulse(overest_onpp)
            old_overest_onpp = overest_onpp
            old_underest_onpp = underest_onpp
            old_maxima = maxima
            old_minima = minima
            old_noise_est = new_noise_est
            old_tollvl = new_tollvl

        self._offpulse_pairs = old_offpp
        self._overest_onpulse_pairs = old_overest_onpp
        self._underest_onpulse_pairs = old_underest_onpp
        self._maxima = old_maxima
        self._minima = old_minima
        self._noise_est = old_noise_est

    def measure_pulse_widths(
        self, peak_fracs: list | None = None, sigma_cutoff: int = 2.0
    ) -> None:
        """Measure the pulse width(s) at a given fraction of the peak flux
        density.

        Parameters
        ----------
        peak_fracs : `list`, optional
            The peak fraction to find the width(s) at (i.e. 0.1 for W10).
            Default: [0.5, 0.1].
        sigma_cutoff : `int`, optional
            The number of standard deviations above which a peak will be
            considered real. Default 2.0.
        """
        assert self._ppoly
        assert self._d1_ppoly
        assert self._maxima
        assert self._minima
        assert self._noise_est

        if peak_fracs is None:
            peak_fracs = [0.5, 0.1]

        peak_flux = np.max(self._ppoly(np.array(self._maxima)))
        peak_snr = peak_flux / self._noise_est

        widths_dict = {}
        for peak_frac in peak_fracs:
            if peak_snr < 1 / peak_frac:
                logger.debug(f"Skipping W{peak_frac * 100:.0f} - not enough S/N")
                continue
            logger.debug(f"Measuring W{peak_frac * 100:.0f} for S/N={peak_snr:.1f}")

            roots = self._ppoly.solve(peak_frac * peak_flux)

            # Remove out-of-bounds roots
            roots = _get_inbounds_roots(roots, self._lims)

            # There must be an even number of roots
            try:
                assert len(roots) % 2 == 0, f"{len(roots)=} is not divisible by 2"
            except AssertionError as e:
                logger.error(f"Could not measure pulse widths: {e}")
                return

            # Match each leading edge root with its corresponding trailing
            # edge root
            root_pairs = []
            current_pair = []
            for ii in range(len(roots)):
                if ii == 0 and self._d1_ppoly(roots[ii]) < 0.0:
                    # The first root is a trailing edge
                    continue
                else:
                    current_pair.append(roots[ii])

                if ii + 1 == len(roots) and len(current_pair) == 1:
                    # If the first root was a trailing edge
                    current_pair.append(roots[0])

                if len(current_pair) == 2:
                    root_pairs.append(tuple(current_pair))
                    current_pair = []

            # Sanity check
            assert len(root_pairs) == len(roots) / 2, "root pairing unsuccessful"

            # Fill the pairs if there is signal between them
            root_pairs = _fill_onpulse_regions(
                root_pairs, self._bspl, self._noise_est, 1.0, self._lims
            )

            widths = []
            for root_pair in root_pairs:
                if root_pair[0] < root_pair[1]:
                    width = root_pair[1] - root_pair[0]
                else:
                    width = root_pair[1] - self._llim + self._rlim - root_pair[0]
                widths.append((root_pair, width))

            widths_dict[f"W{peak_frac * 100:.0f}"] = (
                peak_frac,
                peak_frac * peak_flux,
                widths,
            )

        self._widths = widths_dict

    def plot_diagnostics(
        self,
        plot_underestimate: bool = True,
        plot_overestimate: bool = True,
        plot_width: bool = False,
        sourcename: str | None = None,
        savename: str = "profile_diagnostics",
    ) -> None:
        """Create a plot showing various diagnostics to verify that the
        spline fit is reasonable.

        Parameters
        ----------
        plot_underestimate : `bool`, optional
            Show the onpulse underestimate(s). Default: `True`.
        plot_overestimate : `bool`, optional
            Show the onpulse overestimate(s). Default: `True`.
        sourcename : `str`, optional
            The name of the source to add to the plot.
        savename : `str`, optional
            The name of the output plot, excluding extension.
            Default: "profile_diagnostics".
        """
        # Create figure and axes
        fig = plt.figure(layout="constrained", figsize=(10, 7))

        gs = gridspec.GridSpec(
            4, 2, width_ratios=[2, 1], height_ratios=[1, 3, 3, 3], figure=fig
        )

        ax_text = fig.add_subplot(gs[0, :])
        axLT = fig.add_subplot(gs[1, 0])
        axLM = fig.add_subplot(gs[2, 0])
        axLB = fig.add_subplot(gs[3, 0])
        axRT = fig.add_subplot(gs[1, 1])
        axRM = fig.add_subplot(gs[2, 1])
        axRB = fig.add_subplot(gs[3, 1])

        left_axes = [axLT, axLM, axLB]
        right_axes = [axRT, axRM, axRB]
        all_axes = left_axes + right_axes

        xrange = (self._bins[0], self._bins[-1])
        lw = 0.9

        # Profile
        axLT.plot(self._bins, self._prof, color="k", linewidth=lw, alpha=0.2)
        axLT.plot(self._bins, self._bspl(self._bins), color="k", linewidth=lw)
        yrangeLT = axLT.get_ylim()

        xpos = (xrange[1] - xrange[0]) * 0.92 + xrange[0]
        ypos = (yrangeLT[-1] - yrangeLT[0]) * 0.8 + yrangeLT[0]
        axLT.errorbar(
            xpos,
            ypos,
            yerr=np.std(self.residuals),
            color="k",
            marker="none",
            capsize=3,
            elinewidth=lw,
        )

        # Residuals
        axLM.plot(
            self._bins, self._prof - self._bspl(self._bins), color="k", linewidth=lw
        )
        yrangeLM = axLM.get_ylim()

        # Derivatives
        axLB.plot(
            self._bins, self._d1_bspl(self._bins), color="k", linewidth=lw, label="1st"
        )
        axLB.plot(
            self._bins,
            self._d2_bspl(self._bins),
            color="tab:red",
            linewidth=lw,
            label="2nd",
        )
        yrangeLB = axLB.get_ylim()

        # Onpulse estimates
        if (
            hasattr(self, "_overest_onpulse_pairs")
            and self._overest_onpulse_pairs is not None
        ):
            underest_args = dict(color="tab:blue", alpha=0.2, hatch="///", zorder=0.1)
            overest_args = dict(color="tab:blue", edgecolor=None, alpha=0.2, zorder=0)
            for pairlist, args, flag in zip(
                [self._underest_onpulse_pairs, self._overest_onpulse_pairs],
                [underest_args, overest_args],
                [plot_underestimate, plot_overestimate],
                strict=True,
            ):
                if not flag:
                    continue

                for op_pair in pairlist:
                    for ax, yrange in zip(
                        left_axes, [yrangeLT, yrangeLM, yrangeLB], strict=True
                    ):
                        if op_pair[0] < op_pair[-1]:
                            ax.fill_betweenx(yrange, op_pair[0], op_pair[-1], **args)
                        else:
                            ax.fill_betweenx(yrange, op_pair[0], xrange[-1], **args)
                            ax.fill_betweenx(yrange, xrange[0], op_pair[-1], **args)
                        ax.set_ylim(yrange)

        # Pulse widths
        w_info_lines = []
        if hasattr(self, "_widths") and self._widths is not None:
            w_kwargs = dict(
                color="tab:red",
                marker=".",
                markersize=5,
                linestyle=":",
                linewidth=lw,
                alpha=1.0,
            )
            for key in self._widths.keys():
                peak_frac, peak_frac_flux, peak_pairs = self._widths[key]
                w_info_line = [f"W{peak_frac * 100:.0f}="]
                for ii, ((wl, wr), width) in enumerate(peak_pairs):
                    if ii > 0:
                        w_info_line.append(", ")
                    w_info_line.append(f"{width:.1f} ({width / self._nbin * 100:.1f}%)")
                    if plot_width:
                        if wl < wr:
                            axLT.errorbar(
                                x=[wl, wr], y=[peak_frac_flux] * 2, **w_kwargs
                            )
                        else:
                            axLT.errorbar(
                                x=[xrange[0], wr], y=[peak_frac_flux] * 2, **w_kwargs
                            )
                            axLT.errorbar(
                                x=[wl, xrange[1]], y=[peak_frac_flux] * 2, **w_kwargs
                            )
                w_info_lines.append("".join(w_info_line))

        # Histograms
        hist_kwargs = dict(bins="knuth", density=False)
        stairs_kwargs = dict(lw=lw, fill=True)
        vline_kwargs = dict(linestyle="--", linewidth=1.4, color="tab:red")

        # Residuals
        res = self.residuals
        p = shapiro(res)[1]
        mu = np.mean(res)
        std = np.std(res)
        hb = histogram(res, **hist_kwargs)
        axRM.stairs(*hb, color="tab:blue", **stairs_kwargs)
        axRM.axvline(mu, **vline_kwargs)
        axRM.text(
            0.05,
            0.93,
            f"$p_\\mathrm{{SW}}=${p:.3f}\n$\\mu=${mu:.2E}\n$\\sigma=${std:.2E}",
            verticalalignment="top",
            horizontalalignment="left",
            transform=axRM.transAxes,
            fontsize=9,
        )

        # Offpulse samples
        noff = 0
        if hasattr(self, "_offpulse_pairs") and self._offpulse_pairs is not None:
            mask = get_profile_mask_from_pairs(self._nbin, self._offpulse_pairs)
            if not np.all(np.logical_not(mask)):
                noff = len(self._prof[mask])
                if noff > 3:
                    p = shapiro(self._prof[mask])[1]
                    mu = np.mean(self._prof[mask])
                    std = np.std(self._prof[mask])
                    hb = histogram(self._prof[mask], **hist_kwargs)
                    axRT.stairs(*hb, color="tab:blue", **stairs_kwargs)
                    axRT.axvline(mu, **vline_kwargs)
                    axRT.text(
                        0.05,
                        0.93,
                        f"$p_\\mathrm{{SW}}=${p:.3f}\n$\\mu=${mu:.2E}\n$\\sigma=${std:.2E}",
                        verticalalignment="top",
                        horizontalalignment="left",
                        transform=axRT.transAxes,
                        fontsize=9,
                    )

                    # Make sure the histograms have the same x-limits
                    xlims_rm = axRM.get_xlim()
                    xlims_rt = axRT.get_xlim()
                    xlims_hist = [
                        min(xlims_rt[0], xlims_rm[0]),
                        max(xlims_rt[1], xlims_rm[1]),
                    ]
                    axRT.set_xlim(xlims_hist)
                    axRM.set_xlim(xlims_hist)

        # p-values
        if hasattr(self, "_p_values") and self._p_values is not None:
            axRB.set_title("Whiteness Score", fontsize=12)
            axRB.plot(self._noise_est_trials, self._p_values, linewidth=lw, color="k")
            axRB.set_xlim([self._noise_est_trials[0], self._noise_est_trials[-1]])
            axRB.set_ylim([-100, 0])
            axRB.set_xscale("log")
            axRB.set_xlabel("Std. of Noise ($\sigma$)")
        else:
            axRB.set_visible(False)

        # Info
        if self._noise_est is not None:
            snr = f"{np.max(self._prof) / self._noise_est:.1f}"
        else:
            snr = "unknown"

        ax_text.set_xticks([])
        ax_text.set_yticks([])
        ax_text.spines[["left", "right", "top", "bottom"]].set_visible(False)
        fit_info_lines = [
            sourcename,
            f"S/N={snr}",
            f"Nbin={self._nbin}",
            f"Nbin(on)={self._nbin - noff} "
            + f"({(self._nbin - noff) / self._nbin * 100:.1f}%)",
            f"Nbin(off)={noff} ({noff / self._nbin * 100:.1f}%)",
        ]
        ax_text.text(
            0.0,
            0.9,
            "    ".join(fit_info_lines),
            verticalalignment="top",
            horizontalalignment="left",
            transform=ax_text.transAxes,
            fontsize=12,
        )
        ax_text.text(
            0.0,
            0.4,
            "    ".join(w_info_lines),
            verticalalignment="top",
            horizontalalignment="left",
            transform=ax_text.transAxes,
            fontsize=12,
        )

        # Finishing touches
        axLB.legend(fontsize=9)

        axLT.set_title("Profile", fontsize=12)
        axLM.set_title("Residuals", fontsize=12)
        axLB.set_title("Profile Derivatives", fontsize=12)
        axRT.set_title("Offpulse Sample Distribution", fontsize=12)
        axRM.set_title("Residual Distribution", fontsize=12)

        for ax in left_axes:
            ax.axhline(0, linestyle=":", color="k", linewidth=lw, alpha=0.3, zorder=1)
            ax.set_xlim(xrange)

        for ax in all_axes:
            ax.tick_params(which="both", right=True, top=True)
            ax.minorticks_on()
            ax.tick_params(axis="both", which="both", direction="in")
            ax.tick_params(axis="both", which="major", length=4)
            ax.tick_params(axis="both", which="minor", length=2)

        axLB.set_xlabel("Bin Index")

        logger.info(f"Saving plot file: {savename}.png")
        fig.savefig(savename + ".png")
        plt.close()


# TODO: Format docstring
def _get_inbounds_roots(
    roots: Iterable[np.float_], bounds: tuple[np.float_, np.float_]
) -> NDArray[np.float_]:
    """Return the roots within the specified bounds.

    Parameters
    ----------
    roots : `Iterable[np.float_]`
        A list of roots.
    bounds : `tuple[np.float_, np.float_]`
        The bounds to enforce.

    Returns
    -------
    inbounds_roots : `NDArray[np.float_]`
        An array of roots within the bounds.
    """
    inbounds_roots = []
    for root in roots:
        if root >= bounds[0] and root < bounds[1]:
            inbounds_roots.append(root)
    return np.array(inbounds_roots)


# TODO: Format docstring
def _get_flanking_roots(
    locs: Iterable[np.float_], roots: Iterable[np.float_]
) -> list[tuple[np.float_, np.float_]]:
    """Return the pairs of flanking roots around each location.

    Parameters
    ----------
    locs : `Iterable[np.float_]`
        A list of profile locations.
    roots : `Iterable[np.float_]`
        A list of profile roots.

    Returns
    -------
    pairs : `list[tuple[np.float_, np.float_]]`
        A list of pairs of roots around the specified locations.
    """
    pairs = []
    raw_pairs = []
    for loc in locs:
        # Figure out which two roots flank the maximum
        ridx = bisect.bisect_left(roots, loc)
        lower_idx = (ridx - 1) % len(roots)
        upper_idx = ridx % len(roots)
        pair = [roots[lower_idx], roots[upper_idx]]

        # Remove duplicates
        if pair in raw_pairs:
            continue
        raw_pairs.append(pair.copy())

        # Merge adjacent intervals
        if pairs and pair[0] == pairs[-1][1]:
            # If the bounds are touching
            pairs[-1][1] = pair[1]
        elif pairs and pair[1] == pairs[0][0]:
            # If the last element wraps around to the first
            pairs[0][0] = pair[0]
        else:
            pairs.append(pair)
    return [tuple(pair) for pair in pairs]


# TODO: Format docstring
def _get_contiguous_regions(
    pairs: Iterable[tuple[np.float_, np.float_]], minima: Iterable[np.float_]
) -> list[tuple[np.float_, np.float_]]:
    """Merge together regions which are connected by a common minimum.

    Parameters
    ----------
    pairs : `Iterable[tuple[np.float_, np.float_]]`
        A list of pairs defining the inverval of each region.
    minima : `Iterable[np.float_]`
        A list of minima to use to connect adjacent regions.

    Returns
    -------
    new_pairs : `list[tuple[np.float_, np.float_]]`
        A list of pairs defining the interval of each new region.
    """
    flk_pairs = []
    for pair in pairs:
        flk_roots = _get_flanking_roots(pair, minima)
        flk_pairs.append((flk_roots[0][0], flk_roots[-1][1]))

    new_pairs = []
    new_flk_pairs = []
    for pair, flk_pair in zip(pairs, flk_pairs, strict=True):
        if new_pairs and flk_pair[0] == new_flk_pairs[-1][1]:
            # If the flanking minima are touching
            new_pairs[-1][1] = pair[1]
            new_flk_pairs[-1][1] = flk_pair[1]
        elif new_pairs and flk_pair[1] == new_flk_pairs[0][0]:
            # If the last element wraps around to the first
            new_pairs[0][0] = pair[0]
            new_flk_pairs[0][0] = flk_pair[0]
        else:
            new_pairs.append(list(pair))
            new_flk_pairs.append(list(flk_pair))
    return [tuple(pair) for pair in new_pairs]


# TODO: Format docstring
def _fill_onpulse_regions(
    onpulse_pairs: list[tuple[np.float_, np.float_]],
    bspl: BSpline,
    noise_est: float,
    sigma_cutoff: float,
    interval: tuple[np.float_, np.float_],
) -> list[tuple[np.float_, np.float_]]:
    """Attempts to fill small gaps in the onpulse pairs provided these gaps
    exceed a set sigma threshold (i.e. they resemble real signal.)

    Parameters
    ----------
    onpulse_pairs : `list[tuple[np.float_, np.float_]]`
        A list of pairs of bin indices that mark the onpulse regions of the
        profile.
    bspl : `BSpline`
        The smoothed profile as a BSpline object.
    noise_est : `float`
        The standard deviation of the offpulse noise.
    sigma_cutoff : `float`
        The number of standard deviations above which a signal will be
        considered real.
    interval : `tuple[np.float_, np.float_]`
        The full range of bins that the pairs are a subset of.

    Return:
    -------
    filled_onpulse_pairs : `list[tuple[np.float_, np.float_]]`
        A list of pairs of bin indices that represent the ranges of the
        filled onpulse regions.
    """
    # Nothing to do if the pairs are of length 1
    if len(onpulse_pairs) <= 1:
        return onpulse_pairs

    loop_pairs = onpulse_pairs.copy()
    # Create two cycling generators to avoid out of bounds errors for the
    # last pair
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
            offpulse_region = bspl(np.linspace(pair[0], pair[1], 1000))
        else:
            offpulse_region = np.concatenate(
                [
                    bspl(np.linspace(pair[0], interval[1], 1000)),
                    bspl(np.linspace(interval[0], pair[1], 1000)),
                ]
            )

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

    return [tuple(pair) for pair in filled_onpulse_pairs]


# TODO: Format docstring
def get_profile_mask_from_pairs(
    nbin: int, bin_pairs: Iterable[tuple[np.int_, np.int_]]
) -> NDArray[np.bool_]:
    """Return a profile with the regions defined by the bin_pairs filled
    with NaNs.

    Parameters
    ----------
    nbin : `int`
        The number of phase bins in the profile.
    bin_pairs : `Iterable[tuple[np.int_, np.int_]]`
        A list of pairs of bin indices defining the profile region(s).

    Returns
    -------
    mask : `NDArray[np.bool_]`
        A profile mask that is True between the bin pairs.
    """
    assert bin_pairs is not None

    bins = np.arange(nbin, dtype=float)
    mask = np.full(nbin, False)

    if len(bin_pairs) == 1 and bin_pairs[0][0] == bin_pairs[0][1]:
        return mask

    for pair in bin_pairs:
        if pair[0] == pair[1]:
            continue
        elif pair[0] < pair[1]:
            mask = np.logical_or(mask, np.logical_and(bins > pair[0], bins < pair[1]))
        else:
            mask = np.logical_or(mask, bins > pair[0])
            mask = np.logical_or(mask, bins < pair[1])
    return mask


# TODO: Format docstring
def get_offpulse_from_onpulse(
    onpulse_pairs: list[tuple[np.int_, np.int_]],
) -> list[tuple[np.int_, np.int_]]:
    """Given a list of pairs of bin indices defining the onpulse regions,
    find the corresponding list of pairs of bin indices defining the
    offpulse regions.

    Parameters
    ----------
    onpulse_pairs : `list[tuple[np.int_, np.int_]]`
        A list of pairs of bin indices defining the onpulse.

    Returns
    -------
    offpulse_pairs : `list[tuple[np.int_, np.int_]]`
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
