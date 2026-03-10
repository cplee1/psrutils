########################################################
# Licensed under the Academic Free License version 3.0 #
########################################################

import logging
from typing import Any

import click
import numpy as np
import rtoml

from psrutils import __version__
from psrutils.cube import StokesCube
from psrutils.logger import log_levels, setup_logger
from psrutils.misc import pythonise
from psrutils.profile import SplineProfile

logger = logging.getLogger(__name__)


@click.command()
@click.argument("archive", nargs=1, type=click.Path(exists=True))
@click.help_option("-h", "--help")
@click.version_option(__version__, "-V", "--version")
@click.option(
    "-L",
    "log_level",
    type=click.Choice(log_levels.keys(), case_sensitive=False),
    default="info",
    show_default=True,
    help="The logger verbosity level.",
)
@click.option("-b", "bscr", type=int, help="Bscrunch to this number of phase bins.")
@click.option("-r", "rotate", type=float, help="Rotate phase by this amount.")
@click.option("-c", "centre", is_flag=True, help="Centre the pulse.")
@click.option("--meas_widths", is_flag=True, help="Measure the pulse width(s).")
@click.option(
    "--plot_diagnostics", is_flag=True, help="Plot profile fitting diagnostics."
)
@click.option("--save_pdf", is_flag=True, help="Save plots in PDF format.")
@click.option("-o", "--outfile", type=str, help="The prefix of the output file names.")
def main(
    archive: str,
    log_level: str,
    bscr: int,
    rotate: float,
    centre: bool,
    meas_widths: bool,
    plot_diagnostics: bool,
    save_pdf: bool,
    outfile: str,
) -> None:
    setup_logger("psrutils", log_level)

    # Initialise a dictionary to store the various results
    results: dict[str, Any] = {}

    logger.info(f"Loading archive: {archive}")
    cube = StokesCube.from_psrchive(archive, False, 1, 1, bscr, rotate)
    logger.info(f"Number of bins: {cube.num_bin}")
    srcname_raw = cube.source
    srcname_ltx = cube.source.replace("-", "$-$")
    if outfile is None:
        outfile = srcname_raw

    if centre:
        logger.info("Rotating the peak to the centre of the profile")
        max_idx = np.argmax(cube.profile)
        phase_rot = (max_idx - cube.num_bin // 2) / cube.num_bin
        cube.rotate_phase(phase_rot)

    # Note that we are normalising the profile here so that the numbers are
    # sensible, but this needs to be accounted for when accessing the
    # measured values later (e.g. noise_est)
    peak_flux = np.max(cube.profile)
    profile = SplineProfile(cube.profile / peak_flux)
    profile.gridsearch_onpulse_regions()

    if meas_widths:
        peak_fracs = [0.5, 0.1]
        profile.measure_pulse_widths(peak_fracs=peak_fracs)
        if hasattr(profile, "_widths"):
            for peak_frac in peak_fracs:
                w_param = f"W{peak_frac * 100:.0f}"
                if w_param in profile._widths.keys():
                    results[w_param] = [
                        width for _, width in profile._widths[w_param][2]
                    ]
                    results[w_param + "_roots"] = [
                        root_pair for root_pair, _ in profile._widths[w_param][2]
                    ]

    if plot_diagnostics:
        profile.plot_diagnostics(
            plot_overestimate=True,
            plot_underestimate=False,
            plot_width=meas_widths,
            sourcename=srcname_ltx,
            savename=f"{outfile}_profile_diagnostics",
            save_pdf=save_pdf,
        )

    results["Offpulse_std"] = np.std(profile.profile[profile.offpulse_bins])
    results["Residual_std"] = np.std(profile.residuals)
    results["W_eq"] = profile.w_eq

    # Eq 7.1 on pg 167 of Lorimer and Kramer (2012)
    # For pure Gaussian noise, the S/N is invariant under up/down-sampling
    results["SNR"] = np.sum(profile.profile) / (
        np.std(profile.residuals) * np.sqrt(profile.w_eq)
    )

    if len(results) > 0:
        logger.info(f"Saving results: {outfile}_prof_results.toml")
        with open(f"{outfile}_prof_results.toml", "w") as f:
            rtoml.dump(pythonise(results), f)
