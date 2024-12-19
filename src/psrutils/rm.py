import logging
from typing import Tuple

import numpy as np
import scipy.stats as st
from scipy.optimize import curve_fit
from tqdm import trange

import psrutils
from psrutils import IMAG

__all__ = ["rm_synthesis", "rm_clean"]


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
        logger = psrutils.get_logger()

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


def _normalise_spectrum(
    P: np.ndarray, S: np.ndarray, freqs: np.ndarray, norm: str = None, logger: logging.Logger = None
) -> np.ndarray:
    """Normalise the complex linear polarisation by Stokes I.

    Parameters
    ----------
    P : `np.ndarray`
        The complex linear polarisation.
    S : `np.ndarray`
        The Stokes I values for each frequency.
    freqs : `np.ndarray`
        The frequencies of each channel.
    norm : `str`, optional
        norm : `str`, optional
        Spectral model subtraction method.
            'mod' - fit a linear model to Stokes I
            'val' - use per-channel Stokes I values
        Default: None.
    logger : `logging.Logger`, optional
        A logger to use. Default: None.

    Returns
    -------
    p : `np.ndarray`
        The normalised complex linear polarisation.
    model : `np.ndarray`
        The model used, evalued at each frequency.
    """
    if logger is None:
        logger = psrutils.get_logger()

    if norm is None:
        model = np.float64(1.0)
    elif norm == "mod":
        # TODO: Check that 'freqs' is in the correct units here
        model = _fit_linear_model(freqs, S, logger=logger)
    elif norm == "val":
        model = S
    else:
        raise ValueError("Invalid normalisation method specified.")
    return P / model, model


def _measure_rm(phi: np.ndarray, fdf_amp: np.ndarray) -> Tuple[float, float]:
    """Measure the RM and its statistical uncertainty from a FDF.

    If both 'res' and 'rmsf_fwhm' are provided, will compute analytic uncertainty
    based on the S/N in the FDF.

    Parameters
    ----------
    phi : `np.ndarray`
        An array of Faraday depths (in rad/m^2) at which the FDF is computed.
    fdf_amp : `np.ndarray`
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


def _measure_rm_unc_analytic(fdf_peak_amp: float, rmsf_fwhm: float, noise: np.ndarray) -> float:
    """Compute the analytic uncertainty on the RM.

    Parameters
    ----------
    fdf_peak_amp : `float`
        The amplitude at the peak of the FDF.
    rmsf_fwhm : `float`
        The FWHM of the RM spread function.
    noise : `np.ndarray`
        An array of amplitudes to use to compute the noise level.

    Returns
    -------
    fdf_peak_rm_un : `float`
        The analytic uncertainty in the RM measured from the peak of the FDF.
    """
    fdf_snr = fdf_peak_amp / np.std(noise)
    return rmsf_fwhm / (2.355 * fdf_snr)


def rm_synthesis(
    cube: psrutils.StokesCube,
    phi: np.ndarray,
    norm: str = None,
    bootstrap_nsamp: int = None,
    logger: logging.Logger = None,
) -> None:
    """Perform RM-synthesis for each phase bin.

    Parameters
    ----------
    cube : `psrutils.StokesCube`
        A StokesCube object.
    phi : `np.ndarray`
        An array of Faraday depths (in rad/m^2) to compute.
    norm : `str`, optional
        Spectral model subtraction method.
            'mod' - fit a linear model to Stokes I
            'val' - use per-channel Stokes I values
        Default: None.
    boostrap_nsamp : `int`, optional
        Number of bootstrap iterations. Default: None.
    logger : `logging.Logger`, optional
        A logger to use. Default: None.
    """
    if logger is None:
        logger = psrutils.get_logger()

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
        dl2=dl2, dphi=dphi, max_rm=max_rm, max_scale=max_scale, span_l2=span_l2, rmsf_fwhm=rmsf_fwhm
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

    if type(bootstrap_nsamp) is int:
        # Compute the median standard deviation in the bin spectra
        q_std = np.median(data[1].std(0))
        logger.debug(f"std(Q) ~ {q_std}")
        u_std = np.median(data[2].std(0))
        logger.debug(f"std(U) ~ {u_std}")
        # Initialise an array for bootstrapping
        tmp_fdf = np.empty(len(phi), dtype=np.complex128)
        rm_samples = np.empty(shape=(cube.num_bin, bootstrap_nsamp), dtype=np.float64)
        rm_prof_samples = np.empty(bootstrap_nsamp, dtype=np.float64)

    logger.info("Computing RM per bin...")
    for bin in trange(cube.num_bin):
        S = data[:, :, bin]  # -> dim=(pol,freq)
        W = np.where(S[0] == 0.0, 0.0, 1.0)  # Uniform weights
        K = 1.0 / np.sum(W)

        # Perform RM synthesis on the real observed data
        P, model = _normalise_spectrum(S[1] + IMAG * S[2], S[0], cube.freqs, norm, logger)
        rmsf[bin] = psrutils.dft_kernel(W, rmsf[bin], rmsf_phi, l2, l2_0, K)
        fdf[bin] = psrutils.dft_kernel(P * W, fdf[bin], phi, l2, l2_0, K)

        if type(bootstrap_nsamp) is int:
            # Bootstrap bin spectrum
            for iter in range(bootstrap_nsamp):
                Q_rvs = st.norm.rvs(S[1], q_std).astype(np.float64)
                U_rvs = st.norm.rvs(S[2], u_std).astype(np.float64)
                P = (Q_rvs + IMAG * U_rvs) / model
                tmp_fdf = psrutils.dft_kernel(P * W, tmp_fdf, phi, l2, l2_0, K)
                rm_samples[bin, iter], _ = _measure_rm(phi, np.abs(tmp_fdf))
        else:
            rm_samples = None

    if type(bootstrap_nsamp) is int:
        # Bootstrap summed profile spectrum
        logger.info("Computing RM of profile...")
        for iter in trange(bootstrap_nsamp):
            tmp_prof_fdf = np.zeros(len(phi), dtype=np.float64)
            for bin in range(cube.num_bin):
                S = data[:, :, bin]  # -> dim=(pol,freq)
                W = np.where(S[0] == 0.0, 0.0, 1.0)  # Uniform weights
                K = 1.0 / np.sum(W)
                Q_rvs = st.norm.rvs(S[1], q_std).astype(np.float64)
                U_rvs = st.norm.rvs(S[2], u_std).astype(np.float64)
                P, model = _normalise_spectrum(Q_rvs + IMAG * U_rvs, S[0], cube.freqs, norm, logger)
                tmp_fdf = psrutils.dft_kernel(P * W, tmp_fdf, phi, l2, l2_0, K)
                tmp_prof_fdf += np.abs(tmp_fdf)
            tmp_prof_fdf /= cube.num_bin
            rm_prof_samples[iter], _ = _measure_rm(phi, tmp_prof_fdf)
    else:
        rm_prof_samples = None

    return fdf, rmsf, rm_samples, rm_prof_samples, rm_stats


def _rm_clean_1d(
    phi: np.ndarray,
    fdf: np.ndarray,
    rmsf: np.ndarray,
    rmsf_fwhm: float,
    niter: int = 2000,
    gain: float = 0.1,
    cutoff: float = 3.0,
    logger: logging.Logger = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Deconvolve the RMSF from the FDF using the RM-CLEAN algorithm.

    Parameters
    ----------
    phi : `np.ndarray`
        The Faraday depths at which the FDF is computed.
    fdf : `np.ndarray`
        The Faraday dispersion function.
    rmsf : `np.ndarray`
        The RM spread function.
    rmsf_fwhm : `float`
        The full width at half maximum of the RMSF.
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
        logger = psrutils.get_logger()

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


def rm_clean(
    phi: np.ndarray,
    fdf: np.ndarray,
    rmsf: np.ndarray,
    rmsf_fwhm: float,
    niter: int = 2000,
    gain: float = 0.1,
    cutoff: float = 3.0,
    logger: logging.Logger = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Deconvolve the RMSF from the FDF using the RM-CLEAN algorithm.

    Parameters
    ----------
    phi : `np.ndarray`
        The Faraday depths at which the FDF is computed.
    fdf : `np.ndarray`
        The Faraday dispersion function.
    rmsf : `np.ndarray`
        The RM spread function.
    rmsf_fwhm : `float`
        The full width at half maximum of the RMSF.
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
        logger = psrutils.get_logger()

    cln_fdf = np.empty_like(fdf)
    cln_model = np.empty_like(fdf)
    cln_comps = np.empty_like(fdf)
    cln_res = np.empty_like(fdf)

    if fdf.ndim == 1:
        rmcln_result = _rm_clean_1d(phi, fdf, rmsf, rmsf_fwhm, niter, gain, cutoff, logger)
        cln_fdf[:], cln_model[:], cln_comps[:], cln_res[:] = rmcln_result
    elif fdf.ndim == 2:
        for bin in range(fdf.shape[0]):
            rmcln_result = _rm_clean_1d(
                phi, fdf[bin], rmsf[bin], rmsf_fwhm, niter, gain, cutoff, logger
            )
            cln_fdf[bin, :], cln_model[bin, :], cln_comps[bin, :], cln_res[bin, :] = rmcln_result
    else:
        raise ValueError("The FDF must have dimensions of (phi) or (phase, phi).")

    return cln_fdf, cln_model, cln_comps, cln_res
