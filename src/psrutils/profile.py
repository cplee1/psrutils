########################################################
# Licensed under the Academic Free License version 3.0 #
########################################################

import bisect
import itertools
import logging
from typing import Iterable

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from astropy.stats import histogram
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import BSpline, PPoly, splrep
from statsmodels.sandbox.stats.runs import runstest_1samp
from statsmodels.stats.diagnostic import acorr_ljungbox

__all__ = ["SplineProfile", "get_profile_mask_from_pairs", "invert_bin_pairs"]

logger = logging.getLogger(__name__)


class SplineProfile(object):
    """A class for storing and analysing a pulse profile"""

    def __init__(self, profile: ArrayLike) -> None:
        """Create a Profile instance from an array.

        Parameters
        ----------
        profile : array_like
            A 1-dimensional array containing the pulse profile.
        """
        profile = np.array(profile, dtype=np.float64)

        if profile.ndim != 1:
            raise ValueError("profile must be a 1-dimensional array")

        self._profile = profile
        self._spline_fitted: bool = False
        self._get_onpulse_attempted: bool = False

    @property
    def profile(self) -> NDArray[np.float64]:
        """The pulse profile."""
        return self._profile

    @property
    def nbin(self) -> int:
        """The number of phase bins in the profile."""
        return self.profile.size

    @property
    def bins(self) -> NDArray[np.int64]:
        """An array of bin indices."""
        return np.arange(self.nbin, dtype=np.int64)

    @property
    def phases(self) -> NDArray[np.float64]:
        """An array of bin phases."""
        return self.bins / (self.nbin - 1)

    @property
    def bspl(self) -> BSpline:
        """The spline fit to the pulse profile as a BSPpline object."""
        if not hasattr(self, "_bspl") or self._bspl is None:
            raise AttributeError("Spline has not been fitted.")
        return self._bspl

    @property
    def d1_bspl(self) -> BSpline:
        """The first derivative of the spline fit to the pulse profile as
        a BSPpline object."""
        if not hasattr(self, "_d1_bspl") or self._d1_bspl is None:
            raise AttributeError("Spline has not been fitted.")
        return self._d1_bspl

    @property
    def d2_bspl(self) -> BSpline:
        """The second derivative of the spline fit to the pulse profile as
        a BSPpline object."""
        if not hasattr(self, "_d2_bspl") or self._d2_bspl is None:
            raise AttributeError("Spline has not been fitted.")
        return self._d2_bspl

    @property
    def ppoly(self) -> PPoly:
        """The spline fit to the pulse profile as a PPoly object."""
        if not hasattr(self, "_ppoly") or self._ppoly is None:
            raise AttributeError("Spline has not been fitted.")
        return self._ppoly

    @property
    def d1_ppoly(self) -> BSpline:
        """The first derivative of the spline fit to the pulse profile as
        a PPoly object."""
        if not hasattr(self, "_d1_ppoly") or self._d1_ppoly is None:
            raise AttributeError("Spline has not been fitted.")
        return self._d1_ppoly

    @property
    def d2_ppoly(self) -> BSpline:
        """The second derivative of the spline fit to the pulse profile as
        a PPoly object."""
        if not hasattr(self, "_d2_ppoly") or self._d2_ppoly is None:
            raise AttributeError("Spline has not been fitted.")
        return self._d2_ppoly

    @property
    def residuals(self) -> NDArray[np.float64]:
        """The difference between the profile and the spline."""
        return self.profile - self.bspl(self.bins)

    @property
    def offpulse_pairs(self) -> list[tuple[np.int_, np.int_]]:
        """A list of pairs of bins defining the offpulse."""
        if not self._get_onpulse_attempted:
            raise AttributeError("Offpulse has not been computed.")
        elif self._offpulse_pairs is None:
            raise AttributeError("Offpulse could not be identified.")
        return self._offpulse_pairs

    @property
    def underest_onpulse_pairs(self) -> list[tuple[np.int_, np.int_]]:
        """A list of pairs of bins defining the underestimated onpulse."""
        if not self._get_onpulse_attempted:
            raise AttributeError("Onpulse has not been computed.")
        elif self._underest_onpulse_pairs is None:
            raise AttributeError("Onpulse could not be identified.")
        return self._underest_onpulse_pairs

    @property
    def overest_onpulse_pairs(self) -> list[tuple[np.int_, np.int_]]:
        """A list of pairs of bins defining the overestimated onpulse."""
        if not self._get_onpulse_attempted:
            raise AttributeError("Onpulse has not been computed.")
        elif self._overest_onpulse_pairs is None:
            raise AttributeError("Onpulse could not be identified.")
        return self._overest_onpulse_pairs

    @property
    def offpulse_mask(self) -> NDArray[np.bool_]:
        """A profile mask that is True for offpulse bins."""
        return get_profile_mask_from_pairs(self.nbin, self.offpulse_pairs)

    @property
    def underest_onpulse_mask(self) -> NDArray[np.bool_]:
        """A profile mask that is True for underestimated onpulse bins."""
        return get_profile_mask_from_pairs(self.nbin, self.underest_onpulse_pairs)

    @property
    def overest_onpulse_mask(self) -> NDArray[np.bool_]:
        """A profile mask that is True for overestimated onpulse bins."""
        return get_profile_mask_from_pairs(self.nbin, self.overest_onpulse_pairs)

    @property
    def baseline_est(self) -> None:
        """An estimation of the profile baseline."""
        if not self._get_onpulse_attempted or self._offpulse_pairs is None:
            return self.offpulse_window_stats()[0]
        else:
            offpulse = self.profile[self.offpulse_mask]
            if len(offpulse) < 10:
                return self.offpulse_window_stats()[0]
            else:
                return np.mean(offpulse)

    @property
    def noise_est(self) -> None:
        """An estimation of the noise in the profile."""
        return np.std(self.residuals)

    @property
    def debase_profile(self) -> None:
        """The baseline-subtracted pulse profile."""
        return self.profile - self.baseline_est

    @property
    def debase_spline_profile(self) -> None:
        """The baseline-subtracted smoothed pulse profile."""
        return self.bspl(self.bins) - self.baseline_est

    @property
    def width_eq(self) -> float:
        """The equivalent pulse width in bins."""
        return np.sum(self.debase_profile) / np.max(self.debase_profile)

    def sliding_window(
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
        windowsize : int, default: nbin//8
            Window width (in bins) defining the trial regions to integrate.
        maximise : bool, default: False
            Whether to maximise the integral rather than minimise.

        Returns
        -------
        left_idx, right_idx : int
            The indices of the leading and trailing edges of the window.
        """
        if windowsize is None:
            windowsize = self.nbin // 8

        integral = np.zeros(self.nbin)
        for i in range(self.nbin):
            win = np.arange(i - windowsize // 2, i + windowsize // 2) % self.nbin
            integral[i] = np.trapz(self.profile[win])

        if maximise:
            opt_idx = np.argmax(integral)
        else:
            opt_idx = np.argmin(integral)

        left_idx = (opt_idx - windowsize // 2) % self.nbin
        right_idx = (opt_idx + windowsize // 2) % self.nbin

        return int(left_idx), int(right_idx)

    def offpulse_window_stats(
        self, windowsize: int | None = None
    ) -> tuple[float, float]:
        """Get the mean and standard deviation of the offpulse window
        located using the sliding window method.

        Parameters
        ----------
        windowsize : int, default: nbin//8
            Window width (in bins) defining the trial regions to integrate.

        Returns
        -------
        offpulse_mean : float
            The mean of the samples in the offpulse window.
        offpulse_std : float
            The standard deviation of the samples in the offpulse window.
        """
        # Find the window with the minimum integrated flux within it
        op_l, op_r = self.sliding_window(windowsize)

        # Mask out the onpulse
        if op_r > op_l:
            mask = np.where(
                np.logical_and(self.bins > op_l, self.bins < op_r), True, False
            )
        else:
            mask = np.where(
                np.logical_or(self.bins > op_l, self.bins < op_r), True, False
            )
        offpulse = self.profile[mask]

        return float(np.mean(offpulse)), float(np.std(offpulse))

    def fit_spline(self, sigma: float, k: int = 5) -> None:
        """Fit a periodic smoothing spline with a smoothness equal to
        `nbin*sigma**2`, where `sigma` is the estimated standard deviation
        of the noise in the profile.

        Parameters
        ----------
        sigma : float
            The standard deviation of the noise in the profile.
        k : int, default: 5
            The degree of the spline fit. See `scipy.interpolate.splrep()`
            for details.
        """
        # Fit a degree-k smoothing spline in the B-spline basis with
        # periodic boundary conditions
        tck = splrep(self.bins, self.profile, k=k, s=self.nbin * sigma**2, per=True)
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
        self._spline_fitted = True

    def fit_spline_gridsearch(self, ntrials: int = 1000, logspan: float = 5.0) -> None:
        """Perform a gridsearch over the smoothness parameter to find the spline
        fit which results in the whitest residuals.

        Parameters
        ----------
        ntrials : int, default: 1000
            The number of trials to evaluate over the search range.
        logspan : float, default: 5.0
            Search in the range [sigma/logspan, sigma*logspan], where sigma is
            an estimate of the noise in the profile.
        """
        # Get an initial noise estimate using the sliding window method
        noise_est = self.offpulse_window_stats()[1]

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
            # Calculate a "whiteness" score which is the log of the combined
            # p-values, but note that since the tests are not independent, this
            # does not represent a true p-value
            p_values[ii] = np.log10(p_runs) + np.log10(p_lb)

        # Maximise the whiteness score
        idx_best = np.argmax(p_values)
        noise_est_best = noise_est_trials[idx_best]

        # Re-run the best spline fit
        self.fit_spline(noise_est_best)

        self._noise_est_trials = noise_est_trials
        self._p_values = p_values

    def get_onpulse(self, sigma_cutoff: float = 2.0) -> None:
        if not self._spline_fitted:
            raise AttributeError("Spline has not been fitted.")

        std_noise = self.noise_est

        # Evaluate the roots of the spline derivatives
        try:
            d1_roots = self.d1_ppoly.roots()
            d2_roots = self.d2_ppoly.roots()
        except ValueError:
            logger.error("Roots of spline could not be found.")
            return None, None, None, None

        # Remove out-of-bounds roots
        d1_roots = _get_inbounds_roots(d1_roots, [self.bins[0], self.bins[-1]])
        d2_roots = _get_inbounds_roots(d2_roots, [self.bins[0], self.bins[-1]])

        # Catagorise minima and maxima
        true_maxima = []
        minima = []
        for root in d1_roots:
            if self.d2_bspl(root) > 0:  # Minimum
                minima.append(root)
            else:  # Maximum
                if self.bspl(root) > std_noise * sigma_cutoff:
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
            underest_onpulse_pairs,
            self.bspl,
            std_noise,
            1.0,
            [self.bins[0], self.bins[-1]],
        )
        overest_onpulse_pairs = _fill_onpulse_regions(
            overest_onpulse_pairs,
            self.bspl,
            std_noise,
            1.0,
            [self.bins[0], self.bins[-1]],
        )

        if underest_onpulse_pairs is None or overest_onpulse_pairs is None:
            offpulse_pairs = None
        else:
            offpulse_pairs = invert_bin_pairs(overest_onpulse_pairs)

        self._underest_onpulse_pairs = underest_onpulse_pairs
        self._overest_onpulse_pairs = overest_onpulse_pairs
        self._offpulse_pairs = offpulse_pairs
        self._maxima = true_maxima
        self._minima = minima
        self._get_onpulse_attempted = True

    def measure_pulse_widths(self, peak_fracs: list[float] | None = None) -> None:
        """Measure the pulse width(s) in bins at a given fraction of the
        peak flux density.

        Parameters
        ----------
        peak_fracs : list[float], default: [0.5, 0.1]
            The peak fraction to find the width(s) at (i.e. 0.1 for W10).
        """
        try:
            assert self._spline_fitted, "Spline has not been fitted."
            assert self._maxima, "No profile maxima found."
            assert self._minima, "No profile minima found."
        except AssertionError as e:
            logger.error(f"Cannot measure pulse widths: {e}")
            return

        if peak_fracs is None:
            peak_fracs = [0.5, 0.1]

        peak_flux = np.max(self.ppoly(np.array(self._maxima)))
        peak_snr = peak_flux / self.noise_est

        widths_dict = {}
        for peak_frac in peak_fracs:
            if peak_snr < 1 / peak_frac:
                logger.debug(f"Skipping W{peak_frac * 100:.0f} - not enough S/N")
                continue
            logger.debug(f"Measuring W{peak_frac * 100:.0f} for S/N={peak_snr:.1f}")

            roots = self.ppoly.solve(peak_frac * peak_flux)

            # Remove out-of-bounds roots
            roots = _get_inbounds_roots(roots, [self.bins[0], self.bins[-1]])

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
                if ii == 0 and self.d1_ppoly(roots[ii]) < 0.0:
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
                root_pairs,
                self.bspl,
                self.noise_est,
                1.0,
                [self.bins[0], self.bins[-1]],
            )

            widths = []
            for root_pair in root_pairs:
                if root_pair[0] < root_pair[1]:
                    width = root_pair[1] - root_pair[0]
                else:
                    width = root_pair[1] - self.bins[0] + self.bins[-1] - root_pair[0]
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
        lw: float = 0.9,
        sourcename: str | None = None,
        savename: str = "profile_diagnostics",
        save_pdf: bool = False,
    ) -> None:
        """Create a plot showing various diagnostics to verify that the
        spline fit is reasonable.

        Parameters
        ----------
        plot_underestimate : bool, default: True
            Show the onpulse underestimate(s).
        plot_overestimate : bool, default: True
            Show the onpulse overestimate(s).
        plot_width : bool, default: True
            Show the width estimate(s).
        lw : float, default: 0.9
            The linewidth to use in all subplots.
        sourcename : str, default: None
            The name of the source to add to the plot.
        savename : str, default: "profile_diagnostics"
            The name of the output plot, excluding extension.
        save_pdf : bool, default: False
            Save the plot as a pdf?
        """
        # Create figure and axes
        fig = plt.figure(layout="tight", figsize=(6, 8))
        gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[9, 3])

        gs_prof = gs[0, :].subgridspec(3, 1, wspace=0)
        axs_prof = gs_prof.subplots()
        ax_white = fig.add_subplot(gs[1, 0])
        ax_hists = fig.add_subplot(gs[1, 1])

        xrange = (0, 1)
        xvalues = self.phases
        bins_to_x = 1 / self.nbin

        # Profile
        axs_prof[0].set_ylabel("Profile")
        axs_prof[0].plot(
            xvalues, self.debase_profile, color="k", linewidth=lw, alpha=0.2
        )
        axs_prof[0].plot(xvalues, self.debase_spline_profile, color="k", linewidth=lw)

        # Indicate the standard deviation of the noise
        yrangeProf0 = axs_prof[0].get_ylim()
        xpos = (xrange[1] - xrange[0]) * 0.92 + xrange[0]
        ypos = (yrangeProf0[-1] - yrangeProf0[0]) * 0.8 + yrangeProf0[0]
        axs_prof[0].errorbar(
            xpos,
            ypos,
            yerr=self.noise_est,
            color="k",
            marker="none",
            capsize=3,
            elinewidth=lw,
        )

        # Residuals
        axs_prof[1].set_ylabel("Residuals")
        axs_prof[1].plot(xvalues, self.residuals, color="k", linewidth=lw)
        yrangeProf1 = axs_prof[1].get_ylim()

        # Cumulative profiles
        axs_prof[2].set_ylabel("Cumulative Sum")
        axs_prof[2].plot(
            xvalues,
            np.cumsum(self.debase_profile),
            color="k",
            linewidth=lw,
            label="Profile",
        )
        axs_prof[2].plot(
            xvalues,
            np.cumsum(self.debase_spline_profile),
            color="tab:green",
            linewidth=lw,
            label="Spline",
        )
        axs_prof[2].plot(
            xvalues,
            np.cumsum(self.residuals),
            color="tab:red",
            linewidth=lw,
            label="Residuals",
        )
        yrangeProf2 = axs_prof[2].get_ylim()
        axs_prof[2].legend(loc="upper left", fontsize=7, frameon=False)

        # Onpulse estimates
        if (
            hasattr(self, "_overest_onpulse_pairs")
            and self._overest_onpulse_pairs is not None
        ):
            underest_args = dict(color="tab:blue", alpha=0.2, hatch="///", zorder=0.1)
            overest_args = dict(color="tab:blue", edgecolor=None, alpha=0.2, zorder=0)
            for pairlist, args, flag in zip(
                [self.underest_onpulse_pairs, self.overest_onpulse_pairs],
                [underest_args, overest_args],
                [plot_underestimate, plot_overestimate],
                strict=True,
            ):
                if not flag:
                    continue

                for op_pair in pairlist:
                    oplp = op_pair[0] * bins_to_x
                    oprp = op_pair[1] * bins_to_x
                    for ax, yrange in zip(
                        axs_prof, [yrangeProf0, yrangeProf1, yrangeProf2], strict=True
                    ):
                        if op_pair[0] < op_pair[-1]:
                            ax.fill_betweenx(yrange, oplp, oprp, **args)
                        else:
                            ax.fill_betweenx(yrange, oplp, xrange[-1], **args)
                            ax.fill_betweenx(yrange, xrange[0], oprp, **args)
                        ax.set_ylim(yrange)

        # Pulse widths
        w_info_lines = []
        if hasattr(self, "_widths") and self._widths is not None:
            w_kwargs = dict(
                color="k",
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
                    w_info_line.append(f"{width:.1f} ({width / self.nbin * 100:.1f}%)")
                    if plot_width:
                        wlp = wl * bins_to_x
                        wrp = wr * bins_to_x
                        if wl < wr:
                            axs_prof[0].errorbar(
                                x=[wlp, wrp], y=[peak_frac_flux] * 2, **w_kwargs
                            )
                        else:
                            axs_prof[0].errorbar(
                                x=[xrange[0], wrp], y=[peak_frac_flux] * 2, **w_kwargs
                            )
                            axs_prof[0].errorbar(
                                x=[wlp, xrange[1]], y=[peak_frac_flux] * 2, **w_kwargs
                            )
                w_info_lines.append("".join(w_info_line))

        # Histograms
        ax_hists.set_ylabel("Number of samples")
        ax_hists.set_xlabel("Sample value")
        hist_kwargs = dict(bins="knuth", density=False)
        stairs_kwargs = dict(lw=lw, fill=False)
        vline_kwargs = dict(linestyle=":", linewidth=lw)

        # Offpulse samples histogram
        noff = 0
        if hasattr(self, "_offpulse_pairs") and self._offpulse_pairs is not None:
            mask = get_profile_mask_from_pairs(self.nbin, self.offpulse_pairs)
            if not np.all(np.logical_not(mask)):
                noff = len(self.debase_profile[mask])
                if noff > 3:
                    mu = np.mean(self.debase_profile[mask])
                    hb = histogram(self.debase_profile[mask], **hist_kwargs)
                    ax_hists.stairs(*hb, color="k", label="Offpulse", **stairs_kwargs)
                    ax_hists.axvline(mu, color="k", **vline_kwargs)

        # Residuals histogram
        mu = np.mean(self.residuals)
        hb = histogram(self.residuals, **hist_kwargs)
        ax_hists.stairs(*hb, color="tab:red", label="Residuals", **stairs_kwargs)
        ax_hists.axvline(mu, color="tab:red", **vline_kwargs)
        ax_hists.legend(loc="upper left", fontsize=7, frameon=False)

        # p-values
        if hasattr(self, "_p_values") and self._p_values is not None:
            noise_est_init = self.offpulse_window_stats()[1]
            noise_est_trials = self._noise_est_trials / noise_est_init
            ax_white.plot(noise_est_trials, self._p_values, linewidth=lw, color="k")
            ax_white.set_xlim([noise_est_trials[0], noise_est_trials[-1]])
            ax_white.set_ylim([-100, 0])
            ax_white.set_xscale("log")
            ax_white.set_xticks([0.2, 1, 5])
            ax_white.set_xticklabels(["0.2", "1", "5"])
            ax_white.set_ylabel("Whiteness Score")
            ax_white.set_xlabel(
                "Noise Trial, $\\hat{{\\sigma}}_i/\\hat{{\\sigma}}_\\mathrm{{est}}$"
            )
        else:
            ax_white.set_visible(False)

        # Final touches
        axs_prof[0].set_title(
            sourcename + "   " + f"$N_\\mathrm{{b}}={self.nbin}$", pad=12
        )
        axs_prof[2].set_xlabel("Pulse Phase")

        for ax in list(axs_prof) + [ax_white, ax_hists]:
            ax.tick_params(which="both", right=True, top=True)
            ax.minorticks_on()
            ax.tick_params(axis="both", which="both", direction="in")
            ax.tick_params(axis="both", which="major", length=4)
            ax.tick_params(axis="both", which="minor", length=2)

        for ax in axs_prof:
            ax.axhline(0, linestyle=":", color="k", linewidth=lw, alpha=0.3, zorder=1)
            ax.set_xlim(xrange)
            ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.1))
            ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.01))

        fig.align_ylabels()

        logger.info(f"Saving plot file: {savename}.png")
        fig.savefig(savename + ".png")

        if save_pdf:
            logger.info(f"Saving plot file: {savename}.pdf")
            fig.savefig(savename + ".pdf")

        plt.close()

    def plot_pubfig(
        self,
        title: str | None = None,
        lw: float = 0.8,
        savename: str = "pubfig",
        save_pdf: bool = False,
    ) -> None:
        """Create a publication-quality plot showing the profile and spline.

        Parameters
        ----------
        title : `str`, optional
            The plot title.
        lw : float, default: 0.8
            The linewidth to use in all subplots.
        savename : `str`, optional
            The name of the output plot, excluding extension.
            Default: "pubfig".
        save_pdf : bool, default: Fale
            Save the plot as a pdf?
        """
        # Create figure and axes
        fig = plt.figure(layout="tight", figsize=(6, 4.5))

        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.0, figure=fig)

        axT = fig.add_subplot(gs[0])
        axB = fig.add_subplot(gs[1])
        axTi = axT.inset_axes([0.66, 0.56, 0.29, 0.35])

        xrange = [self.phases[0], self.phases[-1]]
        bins_interp = np.linspace(self.bins[0], self.bins[-1], 5000)
        phases_interp = bins_interp / (self.nbin - 1)

        # Profile
        axT.plot(
            phases_interp,
            (self.bspl(bins_interp) - self.baseline_est) / self.noise_est,
            color="tab:red",
            linewidth=1.8,
            alpha=1,
            label="Spline",
        )
        axT.plot(
            self.phases,
            self.debase_profile / self.noise_est,
            color="k",
            linestyle="-",
            linewidth=lw,
            marker="o",
            markersize=2,
            label="Data",
        )
        axT.axvline()
        axT.set_xticklabels([])
        axT.legend(loc="upper left")

        # Whiteness
        if hasattr(self, "_p_values") and self._p_values is not None:
            noise_est_init = self.offpulse_window_stats()[1]
            noise_est_trials = self._noise_est_trials / noise_est_init
            axTi.set_xlabel(
                "Noise Trial, $\\hat{{\\sigma}}_i/\\hat{{\\sigma}}_\\mathrm{{est}}$",
                fontsize=9,
            )
            axTi.set_ylabel("Score", fontsize=9)
            axTi.plot(noise_est_trials, self._p_values, linewidth=lw, color="k")
            axTi.set_xlim([noise_est_trials[0], noise_est_trials[-1]])
            axTi.set_xscale("log")
            axTi.set_xticks([0.2, 1, 5])
            axTi.set_xticklabels(["0.2", "1", "5"])
            axTi.set_ylim([-50, 0])
            axTi.set_yticks([])
            axTi.minorticks_on()
            axTi.tick_params(axis="x", which="major", length=4, labelsize=9)
            axTi.tick_params(axis="x", which="minor", length=2)
        else:
            axTi.set_visible(False)

        # Residuals
        axB.plot(
            self.phases,
            self.residuals / self.noise_est,
            color="k",
            linestyle="-",
            linewidth=lw,
            marker="o",
            markersize=2,
        )
        axB.axhline(0, linestyle=":", color="k", linewidth=lw)
        axB.set_ylim([-5, 5])

        # Ticks
        for ax in [axT, axB]:
            ax.set_xlim(xrange)
            ax.tick_params(which="both", right=True, top=True)
            ax.minorticks_on()
            ax.tick_params(axis="both", which="both", direction="in")
            ax.tick_params(axis="both", which="major", length=4)
            ax.tick_params(axis="both", which="minor", length=2)
            ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.1))
            ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.01))

        axB.set_yticks([-3, 0, 3])
        axB.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(1))

        # Labels
        axB.set_xlabel("Pulse Phase")
        axB.set_ylabel("Resid. [$\\hat{{\\sigma}}$]")
        axT.set_ylabel("Profile [$\\hat{{\\sigma}}$]")
        fig.align_ylabels()
        if title is not None:
            axT.set_title(title)

        logger.info(f"Saving plot file: {savename}.png")
        fig.savefig(savename + ".png")

        if save_pdf:
            logger.info(f"Saving plot file: {savename}.pdf")
            fig.savefig(savename + ".pdf")

        plt.close()


# ~~~ FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def _get_inbounds_roots(
    roots: Iterable[np.floating], bounds: tuple[np.floating, np.floating]
) -> NDArray[np.float64]:
    """Return the roots within the specified bounds.

    Parameters
    ----------
    roots : Iterable[floating]
        A list of roots.
    bounds : tuple[floating, floating]
        The bounds to enforce.

    Returns
    -------
    inbounds_roots : NDArray[float64]
        An array of roots within the bounds.
    """
    inbounds_roots = []
    for root in roots:
        if root >= bounds[0] and root < bounds[1]:
            inbounds_roots.append(root)
    return np.array(inbounds_roots, dtype=np.float64)


def _get_flanking_roots(
    locs: Iterable[np.floating], roots: Iterable[np.floating]
) -> list[tuple[float, float]]:
    """Return the pairs of flanking roots around each location.

    Parameters
    ----------
    locs : Iterable[floating]
        A list of profile locations.
    roots : Iterable[floating]
        A list of profile roots.

    Returns
    -------
    pairs : list[tuple[float, float]]
        A list of pairs of roots around the specified locations.
    """
    pairs = []
    raw_pairs = []
    for loc in locs:
        # Figure out which two roots flank the maximum
        ridx = bisect.bisect_left(roots, loc)
        lower_idx = (ridx - 1) % len(roots)
        upper_idx = ridx % len(roots)
        pair = [float(roots[lower_idx]), float(roots[upper_idx])]

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


def _fill_onpulse_regions(
    onpulse_pairs: list[tuple[float, float]],
    bspl: BSpline,
    noise_est: float,
    sigma_cutoff: float,
    interval: tuple[float, float],
) -> list[tuple[float, float]]:
    """Attempts to fill small gaps in the onpulse pairs provided these gaps
    exceed a set sigma threshold (i.e. they resemble real signal.)

    Parameters
    ----------
    onpulse_pairs : list[tuple[float, float]]
        A list of pairs of bin indices that mark the onpulse regions of the
        profile.
    bspl : BSpline
        The smoothed profile as a BSpline object.
    noise_est : float
        The standard deviation of the offpulse noise.
    sigma_cutoff : float
        The number of standard deviations above which a signal will be
        considered real.
    interval : tuple[float, float]
        The full range of bins that the pairs are a subset of.

    Return:
    -------
    filled_onpulse_pairs : list[tuple[float, float]]
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


def get_profile_mask_from_pairs(
    nbin: int, bin_pairs: Iterable[tuple[int, int]]
) -> NDArray[np.bool_]:
    """Return a profile with the regions defined by the bin_pairs filled
    with NaNs.

    Parameters
    ----------
    nbin : int
        The number of phase bins in the profile.
    bin_pairs : Iterable[tuple[int, int]]
        A list of pairs of bin indices defining the profile region(s).

    Returns
    -------
    mask : NDArray[np.bool_]
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


def invert_bin_pairs(
    in_pairs: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Given a list of pairs of array indices defining intervals in an
    array with periodic boundary conditions, find the complementary
    intervals in the array. For example: find the offpulse given the
    onpulse of a pulse profile.

    Parameters
    ----------
    in_pairs : list[tuple[int, int]]
        A list of pairs of array indices.

    Returns
    -------
    out_pairs : list[tuple[int, int]]
        A list of pairs of array indices defining the complementary
        array intervals.
    """
    out_pairs = []

    # Make cycling generators to deal with wrap-around
    current_pair_gen = itertools.cycle(in_pairs)
    next_pair_gen = itertools.cycle(in_pairs)
    next(next_pair_gen)

    for _ in range(len(in_pairs)):
        current_pair = next(current_pair_gen)
        next_pair = next(next_pair_gen)
        out_pairs.append([current_pair[1], next_pair[0]])

    return out_pairs
