import logging
from typing import Tuple, Union

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import psrchive
from scipy.optimize import curve_fit

import psrutils as pu

__all__ = ["StokesCube"]


class StokesCube(object):
    """Wrapper for a PSRCHIVE archive, stored in the Stokes basis."""

    def __init__(
        self,
        archive: psrchive.Archive,
        clone: bool = False,
        tscrunch: int = None,
        fscrunch: int = None,
        bscrunch: int = None,
        rotate_phase: float = None,
    ):
        """Create a StokesCube instance from a PSRCHIVE archive.

        Parameters
        ----------
        archive : `psrchive.Archive`
            An Archive object.
        clone : `bool`, optional
            If True, clone the input object. Otherwise store a reference to 'archive'.
            Default: False.
        tscrunch : `int`, optional
            Scrunch in time to this number of sub-integrations.
        fscrunch : `int`, optional
            Scrunch in frequency to this number of channels.
        bscrunch : `int`, optional
            Scrunch in phase to this number of bins.
        rotate_phase : `float`, optional
            Rotate in phase by this amount. Default: None.
        """
        if type(archive) is not psrchive.Archive:
            raise ValueError("archive must be a psrchive.Archive")

        if clone:
            self._archive = archive.clone()
        else:
            self._archive = archive

        # Ensure the archive is in the Stokes basis
        if self._archive.get_state() != "Stokes":
            self._archive.convert_state("Stokes")

        # Ensure the archive is dedispersed
        if not self._archive.get_dedispersed():
            self._archive.dedisperse()

        # Must remove the baseline before downsampling
        self._archive.remove_baseline()

        # Downsample
        if type(tscrunch) is int:
            self._archive.tscrunch_to_nsub(tscrunch)
        if type(fscrunch) is int:
            self._archive.fscrunch_to_nchan(fscrunch)
        if type(bscrunch) is int:
            self._archive.bscrunch_to_nbin(bscrunch)

        # Rotate
        if type(rotate_phase) is float:
            self._archive.rotate_phase(rotate_phase)

        # Store dimensions for later use
        self.nsubint = self._archive.get_nsubint()
        self.nchan = self._archive.get_nchan()
        self.nbin = self._archive.get_nbin()

        # Flag defaults
        self._rmsynth_done = False
        self._rmclean_done = False

    @classmethod
    def from_psrchive(
        cls,
        archive: Union[str, psrchive.Archive],
        tscrunch: int = None,
        fscrunch: int = None,
        bscrunch: int = None,
        rotate_phase: float = None,
    ):
        """Create a StokesCube from a PSRCHIVE archive object.

        Parameters
        ----------
        archive : `str` or `psrchive.Archive`
            Path to an archive file, or an Archive object, to load.
        tscrunch : `int`, optional
            Scrunch in time to this number of sub-integrations. Default: None.
        fscrunch : `int`, optional
            Scrunch in frequency to this number of channels. Default: None.
        bscrunch : `int`, optional
            Scrunch in phase to this number of bins. Default: None.
        rotate_phase : `float`, optional
            Rotate in phase by this amount. Default: None.

        Returns
        -------
        cube : StokesCube
            A StokesCube object.
        """
        if type(archive) is str:
            archive = psrchive.Archive.load(archive)

        return cls(archive, False, tscrunch, fscrunch, bscrunch, rotate_phase)

    def plot_profile(
        self, pol: int = 0, savename: str = "profile.png"
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Create a plot of integrated flux density vs phase for a specified polarisation.

        Parameters
        ----------
        pol : `int`, optional
            The polarisation index (0=I, 1=Q, 2=U, 3=V). Default: 0.
        savename : `str`, optional
            The name of the plot file. Default: 'profile.png'.

        Returns
        -------
        fig : `matplotlib.figure.Figure`
            A Figure object.
        ax : `matplotlib.axes.Axes`
            An Axes object.
        """
        if pol not in [0, 1, 2, 3]:
            raise ValueError("pol must be an integer between 0 and 3 inclusive")

        tmp_archive = self._archive.clone()
        tmp_archive.tscrunch()
        tmp_archive.fscrunch()
        profile = tmp_archive.get_Profile(0, pol, 0)
        amps = profile.get_amps()

        fig, ax = plt.subplots(dpi=300, tight_layout=True)

        edges = np.arange(len(amps) + 1) / len(amps)
        ax.stairs(amps, edges, color="k")
        ax.set_xlim([edges[0], edges[-1]])
        ax.set_xlabel("Pulse Phase [rot]")
        ax.set_ylabel("Flux Density [arb.]")

        fig.savefig(savename)
        return fig, ax

    def plot_freq_phase(
        self, pol: int = 0, savename: str = "freq_phase.png"
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Create a plot of frequency vs phase for a specified polarisation.

        Parameters
        ----------
        pol : `int`, optional
            The polarisation index (0=I, 1=Q, 2=U, 3=V). Default: 0.
        savename : `str`, optional
            The name of the plot file. Default: 'freq_phase.png'.

        Returns
        -------
        fig : `matplotlib.figure.Figure`
            A Figure object.
        ax : `matplotlib.axes.Axes`
            An Axes object.
        """
        if pol not in [0, 1, 2, 3]:
            raise ValueError("pol must be an integer between 0 and 3 inclusive")

        tmp_archive = self._archive.clone()
        tmp_archive.tscrunch()
        data = tmp_archive.get_data()

        freq_lo = tmp_archive.get_centre_frequency() - tmp_archive.get_bandwidth() / 2.0
        freq_hi = tmp_archive.get_centre_frequency() + tmp_archive.get_bandwidth() / 2.0

        fig, ax = plt.subplots(dpi=300, tight_layout=True)

        ax.imshow(
            data[0, pol, :, :],
            extent=(0, 1, freq_lo, freq_hi),
            aspect="auto",
            cmap="cubehelix_r",
            interpolation="none",
        )
        ax.set_xlabel("Pulse Phase [rot]")
        ax.set_ylabel("Frequency [MHz]")

        fig.savefig(savename)
        return fig, ax

    def plot_time_phase(
        self, pol: int = 0, savename: str = "time_phase.png"
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Create a plot of time vs phase for a specified polarisation.

        Parameters
        ----------
        pol : `int`, optional
            The polarisation index (0=I, 1=Q, 2=U, 3=V). Default: 0.
        savename : `str`, optional
            The name of the plot file. Default: 'folded_spectrum.png'.

        Returns
        -------
        fig : `matplotlib.figure.Figure`
            A Figure object.
        ax : `matplotlib.axes.Axes`
            An Axes object.
        """
        if pol not in [0, 1, 2, 3]:
            raise ValueError("pol must be an integer between 0 and 3 inclusive")

        tmp_archive = self._archive.clone()
        tmp_archive.fscrunch()
        data = tmp_archive.get_data()

        t_start = tmp_archive.get_first_Integration().get_start_time()
        t_end = tmp_archive.get_last_Integration().get_end_time()
        t_span = t_end - t_start

        fig, ax = plt.subplots(dpi=300, tight_layout=True)

        ax.imshow(
            data[:, pol, 0, :],
            extent=(0, 1, 0, t_span.in_seconds()),
            aspect="auto",
            cmap="cubehelix_r",
            interpolation="none",
        )
        ax.set_xlabel("Pulse Phase [rot]")
        ax.set_ylabel("Time [s]")

        fig.savefig(savename)
        return fig, ax

    def rm_synthesis(
        self,
        phi: np.ndarray,
        norm_mod: bool = False,
        norm_vals: bool = False,
        logger: logging.Logger = None,
    ) -> None:
        """Perform RM-synthesis for each phase bin.

        Parameters
        ----------
        phi : `np.ndarray`
            An array of Faraday depths (in rad/m^2) to compute.
        norm_mod : `bool`, optional
            Normalise by the best-fit linear model to Stokes I. Default: False.
        norm_vals : `bool`, optional
            Normalise by each Stokes I sample. Default: False.
        logger : `logging.Logger`, optional
            A logger to use. Default: None.
        """
        if logger is None:
            logger = pu.get_logger()

        # Compute squared wavelengths (l2)
        freqs_Hz = self._archive.get_frequencies() * 1e6
        l2 = (pu.C0 / freqs_Hz) ** 2
        l2_0 = np.mean(l2)

        # Compute some statistics
        span_l2 = np.max(l2) - np.min(l2)
        rmsf_fwhm = 3.8 / span_l2
        max_scale = np.pi / np.min(l2)
        min_dl2 = np.min(np.abs(np.diff(l2)))
        rm_max = np.sqrt(3.0) / min_dl2

        # Compute the Faraday depths to evaluate the RMSF at
        dphi = np.min(np.diff(phi))
        nrmsf = 2 * len(phi) + 1
        rmsf_half_span = float(0.5 * nrmsf) * dphi
        rmsf_phi = np.linspace(-1.0 * rmsf_half_span, rmsf_half_span, nrmsf)

        # Initialise arrays to store the RMSF and FDF for all bins
        rmsf = np.zeros(shape=(self.nbin, len(rmsf_phi)), dtype=np.complex128)
        fdf = np.zeros(shape=(self.nbin, len(phi)), dtype=np.complex128)

        # Extract the spectral data from the archive
        tmp_archive = self._archive.clone()
        tmp_archive.tscrunch()
        data = tmp_archive.get_data()[0, :, :, :]  # -> dim=(pol,freq,bin)

        # Perform the computations for each bin
        for bin in range(self.nbin):
            logger.debug(f"RM-synthesis: bin={bin}")
            S = data[:, :, bin]  # -> dim=(pol,freq)
            W = np.where(S[0] == 0.0, 0.0, 1.0)
            K = 1.0 / np.sum(W)
            P = S[1] + pu.IMAG * S[2]
            if norm_mod:
                model = _fit_linear_model(freqs_Hz, S[0], logger=logger)
                P /= model
            if norm_vals:
                P /= S[0]

            # Run the kernels
            rmsf[bin] = pu.dft_kernel(W, rmsf[bin], rmsf_phi, l2, l2_0, K)
            fdf[bin] = pu.dft_kernel(P * W, fdf[bin], phi, l2, l2_0, K)

        # Store statistics and metadata
        self._span_l2 = span_l2
        self._rmsf_fwhm = rmsf_fwhm
        self._max_scale = max_scale
        self._rm_max = rm_max
        self._nphi = len(phi)
        self._dphi = dphi
        self._rmsf_fwhm_pix = int(rmsf_fwhm / dphi + 0.5)
        # Store data
        self._phi = phi
        self._rmsf = rmsf
        self._fdf = fdf
        # Flag as done
        self._rmsynth_done = True

    def rm_clean(
        self,
        niter: int = 2000,
        gain: float = 0.1,
        cutoff: float = 3.0,
        logger: logging.Logger = None,
    ) -> None:
        """Run the RM-CLEAN algorithm for each phase bin.

        Parameters
        ----------
        niter : `int`, optional
            Maximum number of RM-CLEAN iterations. Default: 2000.
        gain : `float`, optional
            RM-CLEAN gain. Default: 0.1.
        cutoff : `float`, optional
            RM-CLEAN component cutoff in sigma. Default: 3.0.
        logger : `logging.Logger`, optional
            A logger to use. Default: None.
        """
        if logger is None:
            logger = pu.get_logger()

        if not self._rmsynth_done:
            logger.error("RM-Synthesis has not been performed.")
            return

        # Initialise arrays to store RM-CLEANed data
        self._cln_fdf = np.empty_like(self._fdf)
        self._cln_model = np.empty_like(self._fdf)
        self._cln_comps = np.empty_like(self._fdf)
        self._cln_resid = np.empty_like(self._fdf)

        # Run RM-CLEAN for each bin
        for bin in range(self._fdf.shape[0]):
            fdf = self._fdf[bin]
            rmsf = self._rmsf[bin]

            noise = np.std(np.real(fdf))
            zerolev = np.median(np.abs(fdf))
            cleanlim = cutoff * noise + zerolev
            logger.debug("CLEAN will proceed down to %f" % (cleanlim))
            num = 0
            res = fdf.copy()
            modcomp = np.zeros_like(res, dtype=np.complex128)
            resp = np.abs(res)
            mr = range(len(resp))
            while np.max(resp[mr]) > cleanlim and num < niter:
                maxloc = np.where(resp[mr] == np.max(resp[mr]))[0] + mr[0]
                if num == 0:
                    logger.debug("First component found at %f rad/m2" % (self._phi[maxloc]))
                num += 1
                if num % 10 ** int(np.log10(num)) == 0:
                    logger.debug("Iteration %d: max residual = %f" % (num, np.max(resp)))
                srmsf = np.roll(rmsf, maxloc - self._nphi)
                modcomp[maxloc] += res[maxloc] * gain
                subtr = res[maxloc] * gain * srmsf[: self._nphi]
                res -= subtr
                resp = np.abs(res)
            logger.debug("Convolving clean components...")
            if 10 * self._rmsf_fwhm_pix > len(self._phi):
                kernel = np.exp(
                    -((self._phi - np.mean(self._phi)) ** 2)
                    / (2.0 * (self._rmsf_fwhm / 2.355) ** 2)
                )
            else:
                kernel = np.exp(
                    -(np.arange(-self._rmsf_fwhm * 5.0, self._rmsf_fwhm * 5.0, self._dphi) ** 2)
                    / (2.0 * (self._rmsf_fwhm / 2.355) ** 2)
                )
            cln_model = np.convolve(modcomp, kernel, mode="same")
            logger.debug("Restoring convolved clean components...")

            # Store data for bin
            self._cln_model[bin, :] = cln_model
            self._cln_fdf[bin, :] = cln_model + res
            self._cln_comps[bin, :] = modcomp
            self._cln_resid[bin, :] = res

        # Flag as done
        self._rmclean_done = True

    def get_rm_per_bin(self) -> np.ndarray:
        """Measure the RM for each phase bin.

        Returns
        -------
        rm_results : `np.ndarray`
            An array with shape (nbin, 3) containing, for each bin, the measured
            RM, its statistical uncertainty, and the SNR in the FDF. For more
            details see `psrutils.core._meaure_rm()`.
        """
        if self._rmclean_done:
            fdf = self._cln_fdf
            res = self._cln_resid
        else:
            fdf = self._fdf
            res = self._fdf
        rm_results = np.empty(shape=(self.nbin, 3), dtype=float)
        for bin in range(self.nbin):
            fdf_bin = np.abs(fdf[bin])
            res_bin = np.real(res[bin])
            rm_results[bin, :] = _measure_rm(self._phi, fdf_bin, res_bin, self._rmsf_fwhm)
        return rm_results

    def get_rm_profile(self) -> Tuple[float, float, float]:
        """Measure the RM for the profile by summing together the FDF power spectra
        for each phase bin.

        Returns
        -------
        fdf_peak_rm : `float`
            The Faraday depth at which the FDF peaks (in rad/m^2).
        fdf_peak_rm_err : `float`
            The statistical uncertainty in the Faraday depth (in rad/m^2).
        fdf_snr : `float
            The signal-to-noise in the FDF.
        """
        if self._rmclean_done:
            fdf = self._cln_fdf
            res = self._cln_resid
        else:
            fdf = self._fdf
            res = self._fdf
        fdf_profile = np.abs(fdf).mean(0)
        res_profile = np.real(res).mean(0)
        return _measure_rm(self._phi, fdf_profile, res_profile, self._rmsf_fwhm)

    def plot_fdf(
        self,
        plot_stairs: bool = False,
        plot_peaks: bool = False,
        phase_range: Tuple[float, float] = None,
        phi_range: Tuple[float, float] = None,
        savename: str = "fdf.png",
        logger: logging.Logger = None,
    ) -> plt.Figure:
        """Plot the 1-D and 2-D FDF as a function of phase.

        Parameters
        ----------
        plot_stairs : bool, optional
            Plot the profile bins as stairs. Default: False.
        plot_peaks : bool, optional
            Plot the measure RM and error bars. Default: False.
        phase_range : Tuple[float, float], optional
            The phase range in rotations. Default: [0, 1].
        phi_range : Tuple[float, float], optional
            The Faraday depth range in rad/m^2. Default: computed range.
        savename : `str`, optional
            The name of the plot file. Default: 'fdf.png'.
        logger : logging.Logger, optional
            A logger to use. Default: None.

        Returns
        -------
        fig : `matplotlib.figure.Figure`
            A Figure object.
        """
        if logger is None:
            logger = pu.get_logger()

        if not self._rmsynth_done:
            logger.error("RM-Synthesis has not been performed.")
            return

        if self._rmclean_done:
            fdf_2D = np.abs(self._cln_fdf)
        else:
            fdf_2D = np.abs(self._fdf)
        fdf_1D = fdf_2D.mean(0)

        rm, rm_err, _ = self.get_rm_profile()
        tmp_archive = self._archive.clone()
        tmp_archive.tscrunch()
        tmp_archive.set_rotation_measure(rm)
        tmp_archive.fscrunch()
        I_profile = tmp_archive.get_Profile(0, 0, 0)
        Q_profile = tmp_archive.get_Profile(0, 1, 0)
        U_profile = tmp_archive.get_Profile(0, 2, 0)
        V_profile = tmp_archive.get_Profile(0, 3, 0)
        I_amps = I_profile.get_amps()
        Q_amps = Q_profile.get_amps()
        U_amps = U_profile.get_amps()
        V_amps = V_profile.get_amps()
        L_amps = np.sqrt(Q_amps**2 + U_amps**2)

        # Define Figure and Axes
        fig = plt.figure(figsize=(7, 6), tight_layout=True, dpi=300)
        gs = gridspec.GridSpec(
            ncols=2,
            nrows=2,
            figure=fig,
            height_ratios=(1, 3),
            width_ratios=(3, 1),
            hspace=0,
            wspace=0,
        )
        ax_prof = fig.add_subplot(gs[0, 0])
        ax_2dfdf = fig.add_subplot(gs[1, 0])
        ax_1dfdf = fig.add_subplot(gs[1, 1])

        # Default plot limits
        if phase_range is None:
            phase_range = [0, 1]
        if phi_range is None:
            phi_range = [self._phi[0], self._phi[-1]]

        # Plot profile
        if plot_stairs:
            bin_edges = np.arange(len(I_amps) + 1) / len(I_amps)
            ax_prof.stairs(I_amps, bin_edges, color="k", zorder=10)
            ax_prof.stairs(L_amps, bin_edges, color="tab:red", zorder=9)
            ax_prof.stairs(V_amps, bin_edges, color="tab:blue", zorder=8)
        else:
            bin_centres = np.arange(len(I_amps)) / (len(I_amps) - 1)
            ax_prof.plot(bin_centres, I_amps, linewidth=1, color="k", zorder=10)
            ax_prof.plot(bin_centres, L_amps, linewidth=1, color="tab:red", zorder=9)
            ax_prof.plot(bin_centres, V_amps, linewidth=1, color="tab:blue", zorder=8)
        ax_prof.set_xlim(phase_range)
        ax_prof.set_yticks([])
        ax_prof.set_xticks([])
        ax_prof.text(
            0.03,
            0.90,
            f"{self._archive.get_source()}",
            horizontalalignment="left",
            verticalalignment="top",
            transform=ax_prof.transAxes,
        )

        # Plot 1D FDF
        ax_1dfdf.plot(fdf_1D, self._phi, color="k", linewidth=1)
        xlims = ax_1dfdf.get_xlim()
        # Plot RM=0 + uncertainty region
        y1 = [0 - self._rmsf_fwhm / 2.0] * 2
        y2 = [0 + self._rmsf_fwhm / 2.0] * 2
        ax_1dfdf.fill_between(xlims, y1, y2, color="k", alpha=0.3, zorder=0)
        ax_1dfdf.axhline(y=0, linestyle="--", color="k", linewidth=1, zorder=1)
        # Plot RM_profile + uncertainty region
        y1 = [rm - rm_err] * 2
        y2 = [rm + rm_err] * 2
        ax_1dfdf.fill_between(xlims, y1, y2, color="tab:red", alpha=0.6, zorder=0)
        # ax_1dfdf.axhline(y=rm, linestyle="-", color="tab:red", zorder=10)
        ax_1dfdf.set_ylim(phi_range)
        ax_1dfdf.set_xlim(xlims)
        ax_1dfdf.set_yticks([])
        ax_1dfdf.set_xticks([])

        # Plot 2D FDF
        ax_2dfdf.imshow(
            np.flipud(fdf_2D.transpose()),
            extent=(0, 1, self._phi[0], self._phi[-1]),
            aspect="auto",
            cmap="cubehelix_r",
            interpolation="none",
        )
        if plot_peaks:
            # Plot RM measurements
            rm_bins = self.get_rm_per_bin()
            bin_centres = np.arange(rm_bins.shape[0]) / (rm_bins.shape[0] - 1)
            mask = np.where(rm_bins[:, 2] > 5, True, False)
            # mask = np.full(rm_bins.shape[0], True)
            ax_2dfdf.errorbar(
                x=bin_centres[mask],
                y=rm_bins[:, 0][mask],
                yerr=np.abs(rm_bins[:, 1])[mask],
                color="k",
                marker="none",
                linestyle="none",
                elinewidth=0.7,
                capthick=0.7,
                capsize=1,
            )
        ax_2dfdf.set_xlabel("Pulse Phase [rot]")
        ax_2dfdf.set_ylabel("$\phi$ [$\mathrm{rad}\,\mathrm{m}^2$]")
        ax_2dfdf.set_xlim(phase_range)
        ax_2dfdf.set_ylim(phi_range)
        ax_2dfdf.minorticks_on()

        logger.info(f"Saving plot file: {savename}")
        fig.savefig(savename)
        return fig


def _fit_linear_model(
    freqs: np.ndarray, amps: np.ndarray, logger: logging.Logger = None
) -> np.ndarray:
    """Fit a linear model to spectral data.

    Parameters
    ----------
    freqs : `np.ndarray`
        A list of frequencies.
    amps : `np.ndarray`
        A list of amplitudes corresponding to each frequency.
    logger: `logging.Logger`, optional
        A logger to use. Default: None.

    Returns
    -------
    model : `np.ndarray`
        The fitted linear model evaluated at the given frequencies.
    """
    if logger is None:
        logger = pu.get_logger()

    try:
        par, _ = curve_fit(
            lambda f, s, a: s * (f / np.mean(f)) ** a,
            freqs,
            amps,
            p0=(np.mean(amps), 0.0),
        )
        model = par[0] * (freqs / np.mean(freqs)) ** par[1]
        logger.debug(f"Spectral index: {par[1]:.3f}")
    except RuntimeError:
        logger.debug("Model fit could not converge - falling back to flat model.")
        model = np.ones_like(freqs) * np.mean(amps)
    return model


def _measure_rm(
    phi: np.ndarray, fdf_pow: np.ndarray, res: np.ndarray, rmsf_fwhm: float
) -> Tuple[float, float, float]:
    """Measure the RM and its statistical uncertainty from a FDF.

    Parameters
    ----------
    phi : `np.ndarray`
        An array of Faraday depths (in rad/m^2) at which the FDF is computed.
    fdf_pow : `np.ndarray`
        The Faraday dispersion function power spectrum.
    res : `np.ndarray`
        A real-valued array to use to compute the noise level. For example, the
        real component of the RM-CLEAN residuals.
    rmsf_fwhm : `float`
        The full width at half maximum of the RMSF.

    Returns
    -------
    fdf_peak_rm : `float`
        The Faraday depth at which the FDF peaks (in rad/m^2).
    fdf_peak_rm_err : `float`
        The statistical uncertainty in the Faraday depth (in rad/m^2).
    fdf_snr : `float
        The signal-to-noise in the FDF.
    """
    x0 = np.argmax(fdf_pow)
    if x0 == 0 or x0 == len(phi) - 1:
        # Peak is an edge bin
        fdf_peak_rm = phi[x0]
        fdf_peak = np.max(fdf_pow)
        fdf_snr = np.nan
        fdf_peak_rm_err = np.nan
    else:
        y = fdf_pow[x0 - 1 : x0 + 2]
        x = phi[x0 - 1 : x0 + 2]
        poly = np.polyfit(x, y, 2)
        fdf_peak_rm = -1.0 * poly[1] / (2.0 * poly[0])
        fdf_peak = np.polyval(poly, fdf_peak_rm)
        fdf_snr = fdf_peak / np.std(res)
        fdf_peak_rm_err = rmsf_fwhm / (2.355 * fdf_snr)
    return fdf_peak_rm, fdf_peak_rm_err, fdf_snr
