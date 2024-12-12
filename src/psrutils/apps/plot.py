import click

import psrutils


@click.command()
@click.argument("archive", nargs=1, type=click.Path(exists=True))
@click.option("-Y", "plot_tvsp", is_flag=True, help="Plot a time vs phase.")
@click.option("-G", "plot_fvsp", is_flag=True, help="Plot a frequency vs phase.")
@click.option("-D", "plot_prof", is_flag=True, help="Plot a pulse profile.")
@click.option("-t", "tscr", type=int, help="Tscrunch to this number of sub-integrations.")
@click.option("-f", "fscr", type=int, help="Fscrunch to this number of channels.")
@click.option("-b", "bscr", type=int, help="Bscrunch to this number of phase bins.")
def main(
    archive: str,
    plot_tvsp: bool,
    plot_fvsp: bool,
    plot_prof: bool,
    tscr: int,
    fscr: int,
    bscr: int,
) -> None:
    cube = psrutils.StokesCube.from_psrchive(archive, tscr, fscr, bscr)

    if plot_tvsp:
        click.echo("Plotting time vs phase")
        cube.plot_time_phase()
    if plot_fvsp:
        click.echo("Plotting frequency vs phase")
        cube.plot_freq_phase()
    if plot_prof:
        click.echo("Plotting profile")
        cube.plot_profile()
