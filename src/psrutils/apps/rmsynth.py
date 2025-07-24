########################################################
# Licensed under the Academic Free License version 3.0 #
########################################################

import logging
import warnings
from typing import Any

import click
import numpy as np
import rtoml
from requests.exceptions import HTTPError

import psrutils

logger = logging.getLogger(__name__)


@click.command()
@click.argument("archive", nargs=1, type=click.Path(exists=True))
@click.help_option("-h", "--help")
@click.version_option(psrutils.__version__, "-V", "--version")
@click.option(
    "-L",
    "log_level",
    type=click.Choice(psrutils.log_levels.keys(), case_sensitive=False),
    default="info",
    show_default=True,
    help="The logger verbosity level.",
)
@click.option("-f", "fscr", type=int, help="Fscrunch to this number of channels.")
@click.option("-b", "bscr", type=int, help="Bscrunch to this number of phase bins.")
@click.option("-r", "rotate", type=float, help="Rotate phase by this amount.")
@click.option("-c", "centre", is_flag=True, help="Centre the pulse.")
@click.option("--rmlim", type=float, default=100.0, show_default=True, help="RM limit.")
@click.option(
    "--rmres", type=float, default=0.1, show_default=True, help="RM resolution."
)
@click.option(
    "-n",
    "nsamp",
    type=int,
    help="The number of bootstrap samples. By default, will not bootstrap.",
)
@click.option(
    "--p0_cutoff",
    type=float,
    default=3.0,
    show_default=True,
    help="Mask below this L/sigma_I value.",
)
@click.option("--phi_plotlim", type=float, nargs=2, help="Plot limits in rad/m^2.")
@click.option("--phase_plotlim", type=float, nargs=2, help="Plot limits in rotations.")
@click.option(
    "--discard",
    type=float,
    nargs=2,
    help="Discard RM samples outside this range in rad/m^2.",
)
@click.option(
    "--clean_cutoff",
    type=float,
    default=3.0,
    show_default=True,
    help="RM-CLEAN component S/N cutoff.",
)
@click.option("--meas_rm_prof", is_flag=True, help="Measure RM_prof.")
@click.option("--meas_rm_scat", is_flag=True, help="Measure RM_scat.")
@click.option("--meas_widths", is_flag=True, help="Measure the pulse width(s).")
@click.option("--get_rm_iono", is_flag=True, help="Get the ionospheric RM.")
@click.option("--no_clean", is_flag=True, help="Do not run RM-CLEAN on the FDF.")
@click.option("--boxplot", is_flag=True, help="Plot RM_phi as boxplots.")
@click.option("--peaks", is_flag=True, help="Plot RM measurements.")
@click.option(
    "--plot_clean_comps", is_flag=True, help="Do not run RM-CLEAN on the FDF."
)
@click.option("--plot_onpulse", is_flag=True, help="Shade the on-pulse region.")
@click.option("--plot_pa", is_flag=True, help="Plot the position angle.")
@click.option(
    "--plot_pol_prof", is_flag=True, help="Plot the full polarisation profile."
)
@click.option("--save_pdf", is_flag=True, help="Save plots in PDF format.")
@click.option(
    "--save_phase_resolved", is_flag=True, help="Save phase-resolved measurements."
)
@click.option("-d", "dark_mode", is_flag=True, help="Use a dark background.")
def main(
    archive: str,
    log_level: str,
    fscr: int,
    bscr: int,
    rotate: float,
    centre: bool,
    rmlim: float,
    rmres: float,
    nsamp: int,
    p0_cutoff: float,
    phi_plotlim: tuple[float, float],
    phase_plotlim: tuple[float, float],
    discard: tuple[float, float],
    clean_cutoff: float,
    meas_rm_prof: bool,
    meas_rm_scat: bool,
    meas_widths: bool,
    get_rm_iono: bool,
    no_clean: bool,
    boxplot: bool,
    peaks: bool,
    plot_clean_comps: bool,
    plot_onpulse: bool,
    plot_pa: bool,
    plot_pol_prof: bool,
    save_pdf: bool,
    save_phase_resolved: bool,
    dark_mode: bool,
) -> None:
    psrutils.setup_logger("psrutils", log_level)

    # Initialise a dictionary to store the various results
    results: dict[str, Any] = {}

    logger.info(f"Loading archive: {archive}")
    cube = psrutils.StokesCube.from_psrchive(archive, False, 1, fscr, bscr, rotate)
    logger.info(f"Number of bins: {cube.num_bin}")

    results["Source"] = cube.source
    results["Nbin"] = cube.num_bin

    if centre:
        logger.info("Rotating the peak to the centre of the profile")
        max_idx = np.argmax(cube.profile)
        phase_rot = (max_idx - cube.num_bin // 2) / cube.num_bin
        cube.rotate_phase(phase_rot)
        results["Phase_rotation"] = phase_rot

    # Note that we are normalising the profile here so that the numbers are
    # sensible, but this needs to be accounted for when accessing the
    # measured values later (e.g. noise_est)
    peak_flux = np.max(cube.profile)
    profile = psrutils.SplineProfile(cube.profile / peak_flux)
    profile.bootstrap_onpulse_regions()

    if meas_widths:
        peak_fracs = [0.5, 0.1]
        profile.measure_pulse_widths(peak_fracs=peak_fracs)
        for peak_frac in peak_fracs:
            w_param = f"W{peak_frac * 100:.0f}"
            if w_param in profile._widths.keys():
                results[w_param] = [width for _, width in profile._widths[w_param][2]]

    profile.plot_diagnostics(
        savename=f"{cube.source}_profile_diagnostics",
        plot_overestimate=True,
        plot_underestimate=False,
        plot_width=meas_widths,
    )

    logger.info("Running RM-Synthesis...")
    results["RM_search_limit"] = rmlim
    results["RM_search_resolution"] = rmres
    phi = np.arange(-1.0 * rmlim, rmlim + rmres, rmres)
    rmsyn_result = psrutils.rm_synthesis(
        cube,
        phi,
        meas_rm_prof=meas_rm_prof,
        meas_rm_scat=meas_rm_scat,
        bootstrap_nsamp=nsamp,
        onpulse_bins=profile.overest_onpulse_bins,
        offpulse_bins=profile.offpulse_bins,
    )
    fdf, rmsf, _, rm_phi_samples, rm_prof_samples, rm_scat_samples, rm_stats = (
        rmsyn_result
    )
    logger.info(f"RM-synthesis statistics: {rm_stats}")
    results["RM_stats"] = rm_stats

    if type(nsamp) is int:
        if discard is not None:
            rm_phi_samples_valid = np.where(
                (rm_phi_samples > discard[0]) & (rm_phi_samples < discard[1]),
                rm_phi_samples,
                np.nan,
            )
        else:
            rm_phi_samples_valid = rm_phi_samples
        with warnings.catch_warnings():
            # For bins with no valid samples (when using --discard),
            # suppress the "Mean of empty slice" warning
            warnings.simplefilter("ignore", category=RuntimeWarning)
            rm_phi_meas = np.nanmean(rm_phi_samples_valid, axis=1)
            rm_phi_unc = np.nanstd(rm_phi_samples_valid, axis=1)
        rm_phi_qty = (rm_phi_meas, rm_phi_unc)
        peak_mask = np.where(rm_phi_unc < rm_stats["rmsf_fwhm"] / 2, True, False)
        if save_phase_resolved:
            results["RM_phi"] = rm_phi_meas
            results["RM_phi_unc"] = rm_phi_unc

        if boxplot:
            psrutils.plotting.plot_rm_vs_phi(
                rm_phi_samples,
                savename=f"{cube.source}_rm_phi_boxplot",
                save_pdf=save_pdf,
            )

        if meas_rm_prof:
            if discard is not None:
                rm_prof_samples_valid = rm_prof_samples[
                    (rm_prof_samples > discard[0]) & (rm_prof_samples < discard[1])
                ]
                plot_valid_samples = rm_prof_samples_valid
            else:
                rm_prof_samples_valid = rm_prof_samples
                plot_valid_samples = None
            rm_prof_meas = np.mean(rm_prof_samples_valid)
            rm_prof_unc = np.std(rm_prof_samples_valid)
            rm_prof_qty = (rm_prof_meas, rm_prof_unc)
            results["RM_prof"] = rm_prof_meas
            results["RM_prof_unc"] = rm_prof_unc

            psrutils.plotting.plot_rm_hist(
                rm_prof_samples,
                valid_samples=plot_valid_samples,
                range=discard,
                title=cube.source.replace("-", "$-$"),
                savename=f"{cube.source}_rm_prof_hist",
                save_pdf=save_pdf,
            )
        else:
            rm_prof_qty = None

        if meas_rm_scat:
            rm_scat_meas = np.mean(rm_scat_samples)
            rm_scat_unc = np.std(rm_scat_samples)
            results["RM_scat"] = rm_scat_meas
            results["RM_scat_unc"] = rm_scat_unc

            psrutils.plotting.plot_rm_hist(
                rm_scat_samples,
                title=cube.source,
                savename=f"{cube.source}_rm_scat_hist",
                save_pdf=save_pdf,
            )
    else:
        rm_phi_qty = (rm_phi_samples, None)
        if meas_rm_prof:
            rm_prof_qty = (rm_prof_samples, None)
            results["RM_prof"] = rm_prof_samples
        else:
            rm_prof_qty = None
        peak_mask = None

    cln_comps = None
    if no_clean:
        cln_fdf = fdf
    else:
        logger.info("Running RM-CLEAN...")
        rmcln_result = psrutils.rm_clean(
            phi, fdf, rmsf, rm_stats["rmsf_fwhm"], gain=0.5, cutoff=clean_cutoff
        )
        cln_fdf = rmcln_result[0]
        if plot_clean_comps:
            cln_comps = rmcln_result[2]

    meas_delta_vi = True
    if meas_delta_vi:
        delta_vi = psrutils.get_delta_vi(
            cube, onpulse_bins=profile.overest_onpulse_bins
        )
        if save_phase_resolved:
            results["delta_V_I"] = delta_vi

    # If there is no offpulse, then default to using the whole profile
    onpp = profile.overest_onpulse_pairs
    if len(onpp) == 1 and onpp[0][0] == onpp[0][1]:
        onpp = None

    psrutils.plotting.plot_2d_fdf(
        cube,
        np.abs(cln_fdf),
        phi,
        rmsf_fwhm=rm_stats["rmsf_fwhm"],
        rm_phi_qty=rm_phi_qty,
        rm_prof_qty=rm_prof_qty,
        onpulse_pairs=onpp,
        rm_mask=peak_mask,
        cln_comps=cln_comps,
        plot_peaks=peaks,
        plot_onpulse=plot_onpulse,
        plot_pa=plot_pa,
        phase_range=phase_plotlim,
        phi_range=phi_plotlim,
        p0_cutoff=p0_cutoff,
        bin_func=psrutils.centre_offset_degrees,
        savename=f"{cube.source}_fdf",
        save_pdf=save_pdf,
        dark_mode=dark_mode,
    )

    if plot_pol_prof:
        psrutils.plotting.plot_pol_profile(
            cube,
            rmsf_fwhm=rm_stats["rmsf_fwhm"],
            rm_phi_qty=rm_phi_qty,
            rm_prof_qty=rm_prof_qty,
            rm_mask=peak_mask,
            delta_vi=delta_vi,
            phase_range=phase_plotlim,
            p0_cutoff=p0_cutoff,
            savename=f"{cube.source}_pol_profile",
            save_pdf=save_pdf,
            save_data=True,
        )

    if get_rm_iono:
        psrutils.setup_logger("spinifex", log_level)
        try:
            rm_iono, rm_iono_err = psrutils.get_rm_iono(
                cube, bootstrap_nsamp=int(1e4), savename=f"{cube.source}_rm_iono"
            )
            results["RM_iono"] = rm_iono
            results["RM_iono_unc"] = rm_iono_err

            if meas_rm_prof:
                rm_obs, rm_obs_err = rm_prof_qty
                rm_ism = rm_obs - rm_iono
                if rm_obs_err is not None:
                    rm_ism_err = np.sqrt(rm_obs_err**2 + rm_iono_err**2)
                else:
                    rm_ism_err = rm_iono_err
                results["RM_prof_ISM"] = rm_ism
                results["RM_prof_ISM_unc"] = rm_ism_err
        except HTTPError as e:
            logger.error(e)

    logger.info(f"Saving results: {cube.source}_rmsynth_results.toml")
    with open(f"{cube.source}_rmsynth_results.toml", "w") as f:
        rtoml.dump(psrutils.pythonise(results), f)
