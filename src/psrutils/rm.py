########################################################
# Licensed under the Academic Free License version 3.0 #
########################################################

import logging
from typing import Tuple

import numpy as np
import scipy.stats as st
from numpy.typing import NDArray
from tqdm import trange

from .cube import StokesCube
from .kernels import dft_kernel
from .plotting import centre_offset_degrees, plot_qu_spectra

__all__ = ["rm_synthesis", "rm_clean"]

logger = logging.getLogger(__name__)


# TODO: Format docstring
def _form_linear(Q: NDArray, U: NDArray, subtract_mean_qu: bool = False) -> NDArray:
    """Form the complex linear polarisation vector by first subtracting the
    baseline from Q and U.

    Parameters
    ----------
    Q, U : `NDArray`
        The components of linear polarisation.
    subtract_mean_qu : `bool`
        Subtract the mean from Q and U.

    Returns
    -------
    P : `NDArray`
        The complex linear polarisation.
    """
    if subtract_mean_qu:
        Q -= np.mean(Q)
        U -= np.mean(U)

    P = Q + 1j * U

    return P


# TODO: Format docstring
def _measure_rm(
    phi: NDArray, fdf_amp: NDArray, zp_hwhm: float | None
) -> Tuple[float, float]:
    """Measure the RM and its statistical uncertainty from a FDF.

    Parameters
    ----------
    phi : `NDArray`
        An array of Faraday depths (in rad/m^2) at which the FDF is computed.
    fdf_amp : `NDArray`
        The amplitude of the Faraday dispersion function.
    zp_hwhm : `float`, optional
        If not None, will mask the samples near zero, using this value as the
        half width at half maximum of the zero peak.

    Returns
    -------
    fdf_peak_rm : `float`
        The Faraday depth at which the FDF peaks (in rad/m^2).
    fdf_peak_amp : `float
        The amplitude at the peak of the FDF.
    """
    edge_case = False
    if isinstance(zp_hwhm, float):
        masked_fdf_amp = np.where(np.abs(phi) < zp_hwhm, np.nan, fdf_amp)
        x0 = np.nanargmax(masked_fdf_amp)
        if x0 - 1 >= 0 and np.isnan(masked_fdf_amp[x0 - 1]):
            edge_case = True
        if x0 + 1 <= len(phi) - 1 and np.isnan(masked_fdf_amp[x0 + 1]):
            edge_case = True
    else:
        x0 = np.argmax(fdf_amp)

    if x0 == 0 or x0 == len(phi) - 1 or edge_case:
        fdf_peak_rm = np.nan
        fdf_peak_amp = np.nan
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
    cube: StokesCube,
    phi: NDArray,
    meas_rm_prof: bool = False,
    meas_rm_scat: bool = False,
    bootstrap_nsamp: int | None = None,
    onpulse_bins: NDArray | None = None,
    offpulse_bins: NDArray | None = None,
    mask_zero_peak: str = None,
    subtract_mean_qu: bool = False,
    plot_qu: bool = False,
) -> None:
    """Perform RM-synthesis for each phase bin.

    Parameters
    ----------
    cube : `StokesCube`
        A StokesCube object.
    phi : `NDArray`
        An array of Faraday depths (in rad/m^2) to compute.
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
    mask_zero_peak : `str`, optional
        If "fwhm", ignore samples with a Faraday less than the FWHM of the RMSF.
        If "hwhm", ignore samples with a Faraday less than the HWHM of the RMSF.
    subtract_mean_qu : `bool`, optional
        Subtract the mean from Q and U. Default: False.
    plot_qu : `bool`, optional
        Make plots of Q and U as a function of frequency. Default: False.
    """
    if mask_zero_peak is not None and mask_zero_peak not in ["fwhm", "hwhm"]:
        raise ValueError(f"Invalid choice of mask_zero_peak: {mask_zero_peak}")

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

    # Only define this variable if you want to mask the zero-peak
    if mask_zero_peak == "fwhm":
        logger.info("Masking samples < FWHM of RMSF")
        zp_hwhm = rmsf_fwhm
    elif mask_zero_peak == "hwhm":
        logger.info("Masking samples < HWHM of RMSF")
        zp_hwhm = rmsf_fwhm / 2
    else:
        zp_hwhm = None

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
    if offpulse_bins is None or onpulse_bins.size == 0:
        # If no offpulse is given or there is no onpulse, use the whole profile
        offpulse_mask = np.full(cube.num_bin, True)
    else:
        offpulse_mask = np.full(cube.num_bin, False)
        for bin_idx in offpulse_bins:
            offpulse_mask[bin_idx] = True

    # Generate a mask for the onpulse
    if onpulse_bins is None:
        # If no onpulse is given, use the whole profile
        onpulse_mask = np.full(cube.num_bin, True)

        # We need this array to be defined so we can loop through it later
        onpulse_bins = np.arange(cube.num_bin)
    else:
        onpulse_mask = np.full(cube.num_bin, False)
        for bin_idx in onpulse_bins:
            onpulse_mask[bin_idx] = True

    # Compute the median standard deviation in the offpulse bin QU spectra
    masked_data = data[:, :, offpulse_mask]
    q_std = np.median(masked_data[1].std(0))
    logger.debug(f"std(Q) ~ {q_std}")
    u_std = np.median(masked_data[2].std(0))
    logger.debug(f"std(U) ~ {u_std}")
    l_std = np.sqrt(q_std**2 + u_std**2)
    logger.debug(f"std(L) ~ {l_std}")

    # Initialise arrays
    tmp_fdf = np.empty(len(phi), dtype=np.complex128)
    tmp_prof_fdf = np.zeros(len(phi), dtype=np.float64)
    if type(bootstrap_nsamp) is int:
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
        P = _form_linear(S[1], S[2], subtract_mean_qu)
        if plot_qu:
            if bin in onpulse_bins:
                plot_qu_spectra(
                    S[1],
                    S[2],
                    cube.freqs,
                    norm_fact=l_std,
                    title="Longitude "
                    + f"${centre_offset_degrees(bin / (cube.num_bin - 1)):.1f}$",
                    savename=f"qu_spectra_bin_{bin:04d}",
                )
        rmsf[bin] = dft_kernel(W, rmsf[bin], rmsf_phi, l2, l2_0, K)
        fdf[bin] = dft_kernel(P * W, fdf[bin], phi, l2, l2_0, K)

        if type(bootstrap_nsamp) is int:
            # Bootstrap the RM_phi
            for iter in range(bootstrap_nsamp):
                Q_rvs = st.norm.rvs(S[1], q_std).astype(np.float64)
                U_rvs = st.norm.rvs(S[2], u_std).astype(np.float64)
                P = _form_linear(Q_rvs, U_rvs, subtract_mean_qu)
                tmp_fdf = dft_kernel(P * W, tmp_fdf, phi, l2, l2_0, K)
                rm_phi_samples[bin, iter], _ = _measure_rm(
                    phi, np.abs(tmp_fdf), zp_hwhm
                )
        else:
            rm_phi_samples[bin], _ = _measure_rm(phi, np.abs(fdf[bin]), zp_hwhm)

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
                    P = _form_linear(Q_rvs, U_rvs, subtract_mean_qu)
                    tmp_fdf = dft_kernel(P * W, tmp_fdf, phi, l2, l2_0, K)
                    tmp_prof_fdf += np.abs(tmp_fdf)
                tmp_prof_fdf /= len(onpulse_bins)
                rm_prof_samples[iter], _ = _measure_rm(phi, tmp_prof_fdf, zp_hwhm)
        else:
            rm_prof_samples, _ = _measure_rm(phi, np.abs(fdf).mean(0), zp_hwhm)
    else:
        rm_prof_samples = None

    if meas_rm_scat:
        logger.info("Computing RM_scat...")
        # Only sum the onpulse bins
        S = cube.subbands[:, :, onpulse_mask].mean(2)  # -> dim=(pol,freq)
        W = np.where(S[0] == 0.0, 0.0, 1.0)  # Uniform weights
        K = 1.0 / np.sum(W)
        if type(bootstrap_nsamp) is int:
            # Bootstrap RM_scat
            for iter in trange(bootstrap_nsamp):
                Q_rvs = st.norm.rvs(S[1], q_std).astype(np.float64)
                U_rvs = st.norm.rvs(S[2], u_std).astype(np.float64)
                P = _form_linear(Q_rvs, U_rvs, subtract_mean_qu)
                tmp_fdf = dft_kernel(P * W, tmp_fdf, phi, l2, l2_0, K)
                rm_scat_samples[iter], _ = _measure_rm(phi, np.abs(tmp_fdf), zp_hwhm)
        else:
            P = _form_linear(S[1], S[2], subtract_mean_qu)
            if plot_qu:
                plot_qu_spectra(S[1], S[2], cube.freqs, savename="qu_spectra_phase_avg")
            tmp_fdf = dft_kernel(P * W, tmp_fdf, phi, l2, l2_0, K)
            rm_scat_samples, _ = _measure_rm(phi, np.abs(tmp_fdf), zp_hwhm)
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
