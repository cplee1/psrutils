########################################################
# Licensed under the Academic Free License version 3.0 #
########################################################

import logging

import click
import numpy as np

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
@click.option("-b", "bscr", type=int, help="Bscrunch to this number of phase bins.")
@click.option("-r", "rotate", type=float, help="Rotate phase by this amount.")
@click.option("-c", "centre", is_flag=True, help="Centre the pulse.")
@click.option(
    "--plot_diagnostics", is_flag=True, help="Plot profile fitting diagnostics."
)
@click.option("--save_pdf", is_flag=True, help="Save plots in PDF format.")
def main(
    archive: str,
    log_level: str,
    bscr: int,
    rotate: float,
    centre: bool,
    plot_diagnostics: bool,
    save_pdf: bool,
) -> None:
    psrutils.setup_logger("psrutils", log_level)

    logger.info(f"Loading archive: {archive}")
    cube = psrutils.StokesCube.from_psrchive(archive, False, 1, 1, bscr, rotate)
    logger.info(f"Number of bins: {cube.num_bin}")

    if centre:
        logger.info("Rotating the peak to the centre of the profile")
        max_idx = np.argmax(cube.profile)
        phase_rot = (max_idx - cube.num_bin // 2) / cube.num_bin
        cube.rotate_phase(phase_rot)

    # Note that we are normalising the profile here so that the numbers are
    # sensible, but this needs to be accounted for when accessing the
    # measured values later (e.g. noise_est)
    peak_flux = np.max(cube.profile)
    profile = psrutils.SplineProfile(cube.profile / peak_flux)
    profile.gridsearch_onpulse_regions()

    if plot_diagnostics:
        profile.plot_diagnostics(
            plot_overestimate=True,
            plot_underestimate=False,
            plot_width=False,
            sourcename=cube.source,
            savename=f"{cube.source}_profile_diagnostics",
            save_pdf=save_pdf,
        )

    # W_eq = sum(profile) / max(profile)
    # Because the profile is normalised, the peak flux is unity, and the
    # integrated flux becomes the equivalent width
    w_eq = np.sum(profile.profile[profile.overest_onpulse_bins])

    # Eq 7.1 on pg 167 of Lorimer and Kramer (2012)
    profile_corr = profile.profile - profile.baseline_est
    snr = np.sum(profile_corr[profile.overest_onpulse_bins]) / (
        profile.noise_est * np.sqrt(w_eq)
    )

    # The S/N of the peak bin in the profile
    snr_peak = np.max(profile.profile) / profile.noise_est

    # The phase-averaged S/N
    snr_mean = np.mean(profile.profile) / profile.noise_est

    logger.info(f"Peak S/N = {snr_peak:.3f}")
    logger.info(f"Phase-averaged S/N = {snr_mean:.3f}")
    logger.info(f"Profile S/N (Eq 7.1 Handbook) = {snr:.3f}")
