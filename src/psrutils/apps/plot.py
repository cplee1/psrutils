########################################################
# Licensed under the Academic Free License version 3.0 #
########################################################

from typing import Tuple

import click
import numpy as np

import psrutils


@click.command()
@click.argument("archive", nargs=1, type=click.Path(exists=True))
@click.help_option("-h", "--help")
@click.version_option(psrutils.__version__, "-V", "--version")
@click.option(
    "-L",
    "log_level",
    type=click.Choice(["DEBUG", "INFO", "ERROR"], case_sensitive=False),
    default="INFO",
    help="The logger verbosity level.",
)
@click.option("-Y", "plot_tvsp", is_flag=True, help="Plot a time vs phase.")
@click.option("-G", "plot_fvsp", is_flag=True, help="Plot a frequency vs phase.")
@click.option("-D", "plot_prof", is_flag=True, help="Plot a pulse profile.")
@click.option(
    "-S", "plot_pol_prof", is_flag=True, help="Plot a full-Stokes pulse profile."
)
@click.option(
    "-P",
    "plot_pol_frac",
    is_flag=True,
    help="Plot the fractional degree of polarisation.",
)
@click.option("-C", "centre", is_flag=True, help="Centre the pulse profile in phase.")
@click.option(
    "-N",
    "normalise",
    is_flag=True,
    help="Normalise the pulse profile by the peak intensity.",
)
@click.option("--longitude", is_flag=True, help="Plot pulse longitude in degrees.")
@click.option(
    "-t", "tscr", type=int, help="Tscrunch to this number of sub-integrations."
)
@click.option("-f", "fscr", type=int, help="Fscrunch to this number of channels.")
@click.option("-b", "bscr", type=int, help="Bscrunch to this number of phase bins.")
@click.option("-r", "rotate", type=float, help="Rotate phase by this amount.")
@click.option("--phase_plotlim", type=float, nargs=2, help="Plot limits in rotations.")
def main(
    archive: str,
    log_level: str,
    plot_tvsp: bool,
    plot_fvsp: bool,
    plot_prof: bool,
    plot_pol_prof: bool,
    plot_pol_frac: bool,
    centre: bool,
    normalise: bool,
    longitude: bool,
    tscr: int,
    fscr: int,
    bscr: int,
    rotate: float,
    phase_plotlim: Tuple[float, float],
) -> None:
    psrutils.setup_logger("psrutils", log_level)

    cube = psrutils.StokesCube.from_psrchive(
        archive,
        clone=False,
        tscrunch=tscr,
        fscrunch=fscr,
        bscrunch=bscr,
        rotate_phase=rotate,
    )

    if centre:
        rot_phase = (np.argmax(cube.profile) - cube.num_bin // 2) / cube.num_bin
        cube.rotate_phase(rot_phase)

    if longitude:
        bin_func = psrutils.plotting.centre_offset_degrees
    else:
        bin_func = None

    if plot_tvsp:
        psrutils.plotting.plot_time_phase(cube)
    if plot_fvsp:
        psrutils.plotting.plot_freq_phase(cube)
    if plot_prof:
        psrutils.plotting.plot_profile(cube, normalise=normalise)
    if plot_pol_prof:
        psrutils.plotting.plot_pol_profile(
            cube,
            normalise=normalise,
            bin_func=bin_func,
            phase_range=phase_plotlim,
            plot_pol_frac=plot_pol_frac,
        )
