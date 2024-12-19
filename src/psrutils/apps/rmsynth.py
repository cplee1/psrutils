import logging
from typing import Tuple

import click
import numpy as np

import psrutils


@click.command()
@click.argument("archive", nargs=1, type=click.Path(exists=True))
@click.option("-f", "fscr", type=int, help="Fscrunch to this number of channels.")
@click.option("-b", "bscr", type=int, help="Bscrunch to this number of phase bins.")
@click.option("-r", "rotate", type=float, help="Rotate phase by this amount.")
@click.option("--rmlim", type=float, default=100.0, help="RM limit.")
@click.option("--rmres", type=float, default=0.01, help="RM resolution.")
@click.option("-n", "nsamp", type=int, help="The number of bootstrap samples.")
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
    nsamp: int,
    phi_plotlim: Tuple[float, float],
    phase_plotlim: Tuple[float, float],
    stairs: bool,
    peaks: bool,
) -> None:
    if nsamp is None:
        raise NotImplementedError("Current implementation only supports bootstrapping.")

    logger = psrutils.get_logger(log_level=logging.INFO)

    logger.info(f"Loading archive: {archive}")
    cube = psrutils.StokesCube.from_psrchive(archive, False, 1, fscr, bscr, rotate)

    phi = np.arange(-1.0 * rmlim, rmlim + rmres, rmres)

    logger.info("Running RM-Synthesis")
    rmsyn_result = psrutils.rm_synthesis(cube, phi, bootstrap_nsamp=nsamp, logger=logger)
    fdf, rmsf, rm_samples, rm_prof_samples, rm_stats = rmsyn_result
    rm_phi_meas = np.mean(rm_samples, axis=1)
    rm_phi_unc = np.std(rm_samples, axis=1)
    rm_prof_meas = np.mean(rm_prof_samples)
    rm_prof_unc = np.std(rm_prof_samples)

    psrutils.plotting.plot_rm_hist(rm_prof_samples, logger=logger)

    logger.info("Running RM-CLEAN")
    rmcln_result = psrutils.rm_clean(phi, fdf, rmsf, rm_stats["rmsf_fwhm"], gain=0.5, logger=logger)
    cln_fdf = rmcln_result[0]
    signal = np.max(np.abs(cln_fdf), axis=1)
    noise = np.std(np.real(rmcln_result[3]))
    snr = signal / noise
    mask = np.where(snr > 6, True, False)

    logger.info("Plotting")
    psrutils.plotting.plot_2d_fdf(
        cube,
        np.abs(cln_fdf),
        phi,
        rm_stats["rmsf_fwhm"],
        (rm_phi_meas, rm_phi_unc),
        (rm_prof_meas, rm_prof_unc),
        mask=mask,
        plot_stairs=stairs,
        plot_peaks=peaks,
        phase_range=phase_plotlim,
        phi_range=phi_plotlim,
        logger=logger,
    )
