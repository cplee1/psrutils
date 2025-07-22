import logging
from typing import Tuple

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
from astropy.visualization import hist, quantity_support, time_support
from numpy.typing import NDArray
from psrqpy import QueryATNF
from scipy.optimize import curve_fit
from spinifex import get_rm
from tqdm import trange

import psrutils

__all__ = ["rm_synthesis", "rm_clean", "get_rm_iono"]

logger = logging.getLogger(__name__)


# TODO: Format docstring
def _fit_linear_model(freqs: NDArray, amps: NDArray) -> NDArray:
    """Fit a linear model to spectral data.

    Parameters
    ----------
    freqs : `NDArray`
        A list of frequencies.
    amps : `NDArray`
        A list of amplitudes corresponding to each frequency.

    Returns
    -------
    model : `NDArray`
        The fitted linear model evaluated at the given frequencies.
    """
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


# TODO: Format docstring
def _normalise_spectrum(
    P: NDArray, S: NDArray, freqs: NDArray, norm: str | None = None
) -> NDArray:
    """Normalise the complex linear polarisation by Stokes I.

    Parameters
    ----------
    P : `NDArray`
        The complex linear polarisation.
    S : `NDArray`
        The Stokes I values for each frequency.
    freqs : `NDArray`
        The frequencies of each channel.
    norm : `str`, optional
        norm : `str`, optional
        Spectral model subtraction method.
            'mod' - fit a linear model to Stokes I
            'val' - use per-channel Stokes I values
        Default: `None`.

    Returns
    -------
    p : `NDArray`
        The normalised complex linear polarisation.
    model : `NDArray`
        The model used, evalued at each frequency.
    """
    if norm is None:
        model = np.float64(1.0)
    elif norm == "mod":
        # TODO: Check that 'freqs' is in the correct units here
        model = _fit_linear_model(freqs, S)
    elif norm == "val":
        model = S
    else:
        raise ValueError("Invalid normalisation method specified.")
    return P / model, model


# TODO: Format docstring
def _measure_rm(phi: NDArray, fdf_amp: NDArray) -> Tuple[float, float]:
    """Measure the RM and its statistical uncertainty from a FDF.

    If both 'res' and 'rmsf_fwhm' are provided, will compute analytic uncertainty
    based on the S/N in the FDF.

    Parameters
    ----------
    phi : `NDArray`
        An array of Faraday depths (in rad/m^2) at which the FDF is computed.
    fdf_amp : `NDArray`
        The amplitude of the Faraday dispersion function.

    Returns
    -------
    fdf_peak_rm : `float`
        The Faraday depth at which the FDF peaks (in rad/m^2).
    fdf_peak_amp : `float
        The amplitude at the peak of the FDF.
    """
    x0 = np.argmax(fdf_amp)
    if x0 == 0 or x0 == len(phi) - 1:
        # Peak is an edge bin
        fdf_peak_rm = phi[x0]
        fdf_peak_amp = np.max(fdf_amp)
    else:
        y = fdf_amp[x0 - 1 : x0 + 2]
        x = phi[x0 - 1 : x0 + 2]
        poly = np.polyfit(x, y, 2)
        fdf_peak_rm = -1.0 * poly[1] / (2.0 * poly[0])
        fdf_peak_amp = np.polyval(poly, fdf_peak_rm)
    return fdf_peak_rm, fdf_peak_amp


# TODO: Format docstring
def _measure_rm_unc_analytic(
    fdf_peak_amp: float, rmsf_fwhm: float, noise: NDArray
) -> float:
    """Compute the analytic uncertainty on the RM.

    Parameters
    ----------
    fdf_peak_amp : `float`
        The amplitude at the peak of the FDF.
    rmsf_fwhm : `float`
        The FWHM of the RM spread function.
    noise : `NDArray`
        An array of amplitudes to use to compute the noise level.

    Returns
    -------
    fdf_peak_rm_un : `float`
        The analytic uncertainty in the RM measured from the peak of the FDF.
    """
    fdf_snr = fdf_peak_amp / np.std(noise)
    return rmsf_fwhm / (2.355 * fdf_snr)


# TODO: Format docstring
def rm_synthesis(
    cube: psrutils.StokesCube,
    phi: NDArray,
    norm: str | None = None,
    meas_rm_prof: bool = False,
    meas_rm_scat: bool = False,
    bootstrap_nsamp: int | None = None,
    onpulse_bins: NDArray | None = None,
    offpulse_bins: NDArray | None = None,
) -> None:
    """Perform RM-synthesis for each phase bin.

    Parameters
    ----------
    cube : `psrutils.StokesCube`
        A StokesCube object.
    phi : `NDArray`
        An array of Faraday depths (in rad/m^2) to compute.
    norm : `str`, optional
        Spectral model subtraction method.
            'mod' - fit a linear model to Stokes I
            'val' - use per-channel Stokes I values
        Default: `None`.
    meas_rm_prof : `bool`, optional
        Measure RM_prof. Default: `False`.
    meas_rm_scat : `bool`, optional
        Measure RM_scat. Default: `False`.
    boostrap_nsamp : `int`, optional
        Number of bootstrap iterations. Default: `None`.
    onpulse_bins : `NDArray`
        A list of bins corresponding to the onpulse region.
        Default: `None`.
    offpulse_bins : `NDArray`, optional
        The bin indices of the offpulse window. Default: `None`.
    """
    # Compute squared wavelengths and reference squared wavelength
    l2 = cube.lambda_sq
    l2_0 = np.mean(l2)

    # Compute minimum sample spacing
    dl2 = np.min(np.abs(np.diff(l2)))
    dphi = np.min(np.diff(phi))

    # Compute RM limit
    max_rm = np.sqrt(3.0) / dl2

    # Compute max scale in the FDF
    max_scale = np.pi / np.min(l2)

    # Compute total span in lambda^2
    span_l2 = np.max(l2) - np.min(l2)

    # Compute the FWHM of the RMSF
    rmsf_fwhm = 3.8 / span_l2

    # Store for later use
    rm_stats = dict(
        dl2=dl2,
        dphi=dphi,
        max_rm=max_rm,
        max_scale=max_scale,
        span_l2=span_l2,
        rmsf_fwhm=rmsf_fwhm,
    )

    # Compute the Faraday depths to evaluate the RMSF at
    nrmsf = 2 * len(phi) + 1
    rmsf_half_span = float(0.5 * nrmsf) * dphi
    rmsf_phi = np.linspace(-1.0 * rmsf_half_span, rmsf_half_span, nrmsf)

    # Initialise arrays to store the RMSF and FDF for all phase bins
    rmsf = np.empty(shape=(cube.num_bin, len(rmsf_phi)), dtype=np.complex128)
    fdf = np.empty(shape=(cube.num_bin, len(phi)), dtype=np.complex128)

    # Extract the spectral data from the archive
    data = cube.subbands  # -> (pol,freq,phase)

    # Generate a mask for the offpulse
    if offpulse_bins is None:
        # Use the whole profile
        offpulse_mask = np.full(cube.num_bin, True)
    else:
        offpulse_mask = np.full(cube.num_bin, False)
        for bin_idx in offpulse_bins:
            offpulse_mask[bin_idx] = True

    if onpulse_bins is None:
        # Use the whole profile
        onpulse_bins = np.arange(cube.num_bin)

    # Initialise arrays
    tmp_fdf = np.empty(len(phi), dtype=np.complex128)
    tmp_prof_fdf = np.zeros(len(phi), dtype=np.float64)
    if type(bootstrap_nsamp) is int:
        masked_data = data[:, :, offpulse_mask]

        # Compute the median standard deviation in the bin spectra
        q_std = np.median(masked_data[1].std(0))
        logger.debug(f"std(Q) ~ {q_std}")
        u_std = np.median(masked_data[2].std(0))
        logger.debug(f"std(U) ~ {u_std}")

        rm_phi_samples = np.empty(
            shape=(cube.num_bin, bootstrap_nsamp), dtype=np.float64
        )
        rm_prof_samples = np.empty(bootstrap_nsamp, dtype=np.float64)
        rm_scat_samples = np.empty(bootstrap_nsamp, dtype=np.float64)
    else:
        rm_phi_samples = np.empty(shape=cube.num_bin, dtype=np.float64)

    logger.info("Computing RM(phi)...")
    for bin in trange(cube.num_bin):
        S = data[:, :, bin]  # -> dim=(pol,freq)
        W = np.where(S[0] == 0.0, 0.0, 1.0)  # Uniform weights
        K = 1.0 / np.sum(W)

        # Perform RM synthesis on the real observed data (for plotting)
        P, model = _normalise_spectrum(S[1] + 1j * S[2], S[0], cube.freqs, norm)
        rmsf[bin] = psrutils.dft_kernel(W, rmsf[bin], rmsf_phi, l2, l2_0, K)
        fdf[bin] = psrutils.dft_kernel(P * W, fdf[bin], phi, l2, l2_0, K)

        if type(bootstrap_nsamp) is int:
            # Bootstrap the RM_phi
            for iter in range(bootstrap_nsamp):
                Q_rvs = st.norm.rvs(S[1], q_std).astype(np.float64)
                U_rvs = st.norm.rvs(S[2], u_std).astype(np.float64)
                P = (Q_rvs + 1j * U_rvs) / model
                tmp_fdf = psrutils.dft_kernel(P * W, tmp_fdf, phi, l2, l2_0, K)
                rm_phi_samples[bin, iter], _ = _measure_rm(phi, np.abs(tmp_fdf))
        else:
            rm_phi_samples[bin], _ = _measure_rm(phi, np.abs(fdf[bin]))

    if meas_rm_prof:
        logger.info("Computing RM_prof...")
        if type(bootstrap_nsamp) is int:
            # Bootstrap RM_prof
            for iter in trange(bootstrap_nsamp):
                tmp_prof_fdf = np.zeros(len(phi), dtype=np.float64)
                for bin in onpulse_bins:
                    S = data[:, :, bin]  # -> dim=(pol,freq)
                    W = np.where(S[0] == 0.0, 0.0, 1.0)  # Uniform weights
                    K = 1.0 / np.sum(W)
                    Q_rvs = st.norm.rvs(S[1], q_std).astype(np.float64)
                    U_rvs = st.norm.rvs(S[2], u_std).astype(np.float64)
                    P, model = _normalise_spectrum(
                        Q_rvs + 1j * U_rvs, S[0], cube.freqs, norm
                    )
                    tmp_fdf = psrutils.dft_kernel(P * W, tmp_fdf, phi, l2, l2_0, K)
                    tmp_prof_fdf += np.abs(tmp_fdf)
                tmp_prof_fdf /= len(onpulse_bins)
                rm_prof_samples[iter], _ = _measure_rm(phi, tmp_prof_fdf)
        else:
            rm_prof_samples, _ = _measure_rm(phi, np.abs(fdf).mean(0))
    else:
        rm_prof_samples = None

    if meas_rm_scat:
        logger.info("Computing RM_scat...")
        S = cube.mean_subbands  # -> dim=(pol,freq)
        W = np.where(S[0] == 0.0, 0.0, 1.0)  # Uniform weights
        K = 1.0 / np.sum(W)
        if type(bootstrap_nsamp) is int:
            # Bootstrap RM_scat
            for iter in trange(bootstrap_nsamp):
                Q_rvs = st.norm.rvs(S[1], q_std).astype(np.float64)
                U_rvs = st.norm.rvs(S[2], u_std).astype(np.float64)
                P, model = _normalise_spectrum(
                    Q_rvs + 1j * U_rvs, S[0], cube.freqs, norm
                )
                tmp_fdf = psrutils.dft_kernel(P * W, tmp_fdf, phi, l2, l2_0, K)
                rm_scat_samples[iter], _ = _measure_rm(phi, np.abs(tmp_fdf))
        else:
            P, model = _normalise_spectrum(S[1] + 1j * S[2], S[0], cube.freqs, norm)
            tmp_fdf = psrutils.dft_kernel(P * W, tmp_fdf, phi, l2, l2_0, K)
            rm_scat_samples, _ = _measure_rm(phi, np.abs(tmp_fdf))
    else:
        rm_scat_samples = None

    return (
        fdf,
        rmsf,
        rmsf_phi,
        rm_phi_samples,
        rm_prof_samples,
        rm_scat_samples,
        rm_stats,
    )


# TODO: Format docstring
def _rm_clean_1d(
    phi: NDArray,
    fdf: NDArray,
    rmsf: NDArray,
    rmsf_fwhm: float,
    niter: int = 2000,
    gain: float = 0.1,
    cutoff: float = 3.0,
) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    """Deconvolve the RMSF from the FDF using the RM-CLEAN algorithm.

    Parameters
    ----------
    phi : `NDArray`
        The Faraday depths at which the FDF is computed.
    fdf : `NDArray`
        The Faraday dispersion function.
    rmsf : `NDArray`
        The RM spread function.
    rmsf_fwhm : `float`
        The full width at half maximum of the RMSF.
    niter : `int`, optional
        Maximum number of RM-CLEAN iterations. Default: 2000.
    gain : `float`, optional
        RM-CLEAN gain. Default: 0.1.
    cutoff : `float`, optional
        RM-CLEAN component cutoff in sigma. Default: 3.0.
    """
    nphi = len(phi)
    dphi = np.min(np.diff(phi))
    rmsf_fwhm_pix = int(rmsf_fwhm / dphi + 0.5)

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
            logger.debug("First component found at %f rad/m2" % (phi[maxloc]))
        num += 1
        if num % 10 ** int(np.log10(num)) == 0:
            logger.debug("Iteration %d: max residual = %f" % (num, np.max(resp)))
        srmsf = np.roll(rmsf, maxloc - nphi)
        modcomp[maxloc] += res[maxloc] * gain
        subtr = res[maxloc] * gain * srmsf[:nphi]
        res -= subtr
        resp = np.abs(res)
    logger.debug("Convolving clean components...")
    if 10 * rmsf_fwhm_pix > len(phi):
        kernel = np.exp(-((phi - np.mean(phi)) ** 2) / (2.0 * (rmsf_fwhm / 2.355) ** 2))
    else:
        kernel = np.exp(
            -(np.arange(-rmsf_fwhm * 5.0, rmsf_fwhm * 5.0, dphi) ** 2)
            / (2.0 * (rmsf_fwhm / 2.355) ** 2)
        )
    cln_model = np.convolve(modcomp, kernel, mode="same")
    logger.debug("Restoring convolved clean components...")
    cln_fdf = cln_model + res

    return cln_fdf, cln_model, modcomp, res


# TODO: Format docstring
def rm_clean(
    phi: NDArray,
    fdf: NDArray,
    rmsf: NDArray,
    rmsf_fwhm: float,
    niter: int = 2000,
    gain: float = 0.1,
    cutoff: float = 3.0,
) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    """Deconvolve the RMSF from the FDF using the RM-CLEAN algorithm.

    Parameters
    ----------
    phi : `NDArray`
        The Faraday depths at which the FDF is computed.
    fdf : `NDArray`
        The Faraday dispersion function.
    rmsf : `NDArray`
        The RM spread function.
    rmsf_fwhm : `float`
        The full width at half maximum of the RMSF.
    niter : `int`, optional
        Maximum number of RM-CLEAN iterations. Default: 2000.
    gain : `float`, optional
        RM-CLEAN gain. Default: 0.1.
    cutoff : `float`, optional
        RM-CLEAN component cutoff in sigma. Default: 3.0.
    """
    cln_fdf = np.empty_like(fdf)
    cln_model = np.empty_like(fdf)
    cln_comps = np.empty_like(fdf)
    cln_res = np.empty_like(fdf)

    if fdf.ndim == 1:
        rmcln_result = _rm_clean_1d(phi, fdf, rmsf, rmsf_fwhm, niter, gain, cutoff)
        cln_fdf[:], cln_model[:], cln_comps[:], cln_res[:] = rmcln_result
    elif fdf.ndim == 2:
        for bin in range(fdf.shape[0]):
            rmcln_result = _rm_clean_1d(
                phi, fdf[bin], rmsf[bin], rmsf_fwhm, niter, gain, cutoff
            )
            cln_fdf[bin, :], cln_model[bin, :], cln_comps[bin, :], cln_res[bin, :] = (
                rmcln_result
            )
    else:
        raise ValueError("The FDF must have dimensions of (phi) or (phase, phi).")

    return cln_fdf, cln_model, cln_comps, cln_res


# TODO: Format docstring
def get_rm_iono(
    cube: psrutils.StokesCube,
    bootstrap_nsamp: int | None = None,
    prefix: str = "jpl",
    server: str = "cddis",
    savename: str | None = None,
) -> tuple[np.float_, np.float_]:
    """Get the mean ionospheric RM during an observation.

    Parameters
    ----------
    cube : `psrutils.StokesCube`
        A StokesCube object.
    bootstrap_nsamp : `int | None`, optional
        The number of bootstrap iteration to use to find the mean
        ionospheric RM. If `None` is provided, then no bootstrapping will
        be performed. Default: `None`.
    prefix : `str`, optional
        The analysis centre prefix. Default: "jpl".
    server : `str`, optional
        Server to download from. Default: "cddis".
    savename : `str | None`, optional
        If provided, will save a plot with this name. Default: `None`.

    Returns
    -------
    rm : `np.float_`
        The mean ionospheric RM during the observation.
    rm_err : `np.float_`
        The standard deviation of the mean ionospheric RM during the
        observation.
    """
    mwa_loc = EarthLocation(
        lat=-26.703319 * u.deg, lon=116.67081 * u.deg, height=377.827 * u.m
    )
    times = Time(cube.start_mjd, format="mjd") + np.linspace(0, cube.int_time, 10) * u.s
    query = QueryATNF(psrs=cube.source, params=["RAJD", "DECJD"])
    psr = query.get_pulsar(cube.source)
    source = SkyCoord(ra=psr["RAJD"][0] * u.deg, dec=psr["DECJD"][0] * u.deg)

    rm = get_rm.get_rm_from_skycoord(
        loc=mwa_loc,
        times=times,
        source=source,
        prefix=prefix,
        server=server,
    )

    if bootstrap_nsamp:
        logger.debug("Bootstrapping RM_iono...")
        rm_samples = np.zeros(bootstrap_nsamp, dtype=float)
        for iter in range(bootstrap_nsamp):
            rm_samples[iter] = np.mean(st.norm.rvs(rm.rm, rm.rm_error))
        rm_val = np.mean(rm_samples)
        rm_err = np.std(rm_samples)
    else:
        rm_val = (np.mean(rm.rm),)
        rm_err = np.mean(rm.rm_error)

    if savename:
        with time_support(), quantity_support():
            if bootstrap_nsamp:
                fig, (ax_rm, ax_hist) = plt.subplots(
                    ncols=2,
                    figsize=(8, 5),
                    tight_layout=True,
                    sharey=True,
                    width_ratios=(3, 1),
                )
                hist(
                    rm_samples,
                    bins="knuth",
                    ax=ax_hist,
                    density=True,
                    orientation="horizontal",
                )
                ax_hist.set_xlabel("Probability Density")
                psrutils.format_ticks(ax_hist)
            else:
                fig, ax_rm = plt.subplots(figsize=(7, 5), tight_layout=True)

            ax_rm.errorbar(rm.times.datetime, rm.rm, rm.rm_error, fmt="ko")
            ax_rm.axhline(rm_val, linestyle="--", color="k", linewidth=1, alpha=0.3)
            ax_rm.set_ylabel(
                "$\mathrm{RM}_\mathrm{iono}$ [$\mathrm{rad}\,\mathrm{m}^{-2}$]"
            )
            ax_rm.set_xlabel("UTC Date")
            psrutils.format_ticks(ax_rm)
            ax_rm.set_xticklabels(ax_rm.get_xticklabels(), rotation=30)
            logger.info(f"Saving plot file: {savename}.png")
            fig.savefig(savename + ".png")
            plt.close()

    return rm_val, rm_err
