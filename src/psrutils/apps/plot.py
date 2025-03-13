from typing import Tuple

import click

import psrutils


@click.command()
@click.argument("archive", nargs=1, type=click.Path(exists=True))
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
@click.option("-S", "plot_pol_prof", is_flag=True, help="Plot a full-Stokes pulse profile.")
@click.option("-t", "tscr", type=int, help="Tscrunch to this number of sub-integrations.")
@click.option("-f", "fscr", type=int, help="Fscrunch to this number of channels.")
@click.option("-b", "bscr", type=int, help="Bscrunch to this number of phase bins.")
@click.option("--phase_plotlim", type=float, nargs=2, help="Plot limits in rotations.")
def main(
    archive: str,
    log_level: str,
    plot_tvsp: bool,
    plot_fvsp: bool,
    plot_prof: bool,
    plot_pol_prof: bool,
    tscr: int,
    fscr: int,
    bscr: int,
    phase_plotlim: Tuple[float, float],
) -> None:
    log_level_dict = psrutils.get_log_levels()
    logger = psrutils.get_logger(log_level=log_level_dict[log_level])

    cube = psrutils.StokesCube.from_psrchive(archive, tscr, fscr, bscr)

    if plot_tvsp:
        click.echo("Plotting time vs phase")
        psrutils.plotting.plot_time_phase(cube, logger=logger)
    if plot_fvsp:
        click.echo("Plotting frequency vs phase")
        psrutils.plotting.plot_freq_phase(cube, logger=logger)
    if plot_prof:
        click.echo("Plotting profile")
        psrutils.plotting.plot_profile(cube, logger=logger)
    if plot_pol_prof:
        click.echo("Plotting polarisation profile")
        psrutils.plotting.plot_pol_profile(cube, phase_range=phase_plotlim, logger=logger)
