import logging
from typing import Tuple

import click
import numpy as np

import psrutils as pu


@click.command()
@click.argument("archive", nargs=1, type=click.Path(exists=True))
@click.option("-f", "fscr", type=int, help="Fscrunch to this number of channels.")
@click.option("-b", "bscr", type=int, help="Bscrunch to this number of phase bins.")
@click.option("-r", "rotate", type=float, help="Rotate phase by this amount.")
@click.option("--rmlim", type=float, default=100.0, help="RM limit.")
@click.option("--rmres", type=float, default=0.01, help="RM resolution.")
@click.option("--phi_plotlim", type=float, nargs=2, help="Plot limits in rad/m^2.")
@click.option("--phase_plotlim", type=float, nargs=2, help="Plot limits in rotations.")
@click.option("--stairs", is_flag=True, help="Plot profile bins as stairs.")
@click.option("--peaks", is_flag=True, help="Plot RM measurements.")
def main(
    archive: str,
    fscr: int,
    bscr: int,
    rotate: float,
    rmlim: float,
    rmres: float,
    phi_plotlim: Tuple[float, float],
    phase_plotlim: Tuple[float, float],
    stairs: bool,
    peaks: bool,
) -> None:
    logger = pu.get_logger(log_level=logging.INFO)

    logger.info(f"Loading archive: {archive}")
    cube = pu.StokesCube.from_psrchive(archive, 1, fscr, bscr, rotate)

    phi = np.arange(-1.0 * rmlim, rmlim + rmres, rmres)

    logger.info("Running RM-Synthesis")
    cube.rm_synthesis(phi, logger=logger)

    logger.info("Running RM-CLEAN")
    cube.rm_clean(gain=0.5, logger=logger)

    logger.info("Plotting")
    cube.plot_fdf(
        plot_stairs=stairs,
        plot_peaks=peaks,
        phase_range=phase_plotlim,
        phi_range=phi_plotlim,
        logger=logger,
    )
