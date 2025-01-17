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
@click.option(
    "--discard", type=float, nargs=2, help="Discard RM samples outside this range in rad/m^2."
)
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
    discard: Tuple[float, float],
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
    fdf, rmsf, rm_samples, rm_prof_samples, rm_scat_samples, rm_stats = rmsyn_result
    logger.info(rm_stats)

    if discard is not None:
        rm_prof_samples_valid = rm_prof_samples[
            (rm_prof_samples > discard[0]) & (rm_prof_samples < discard[1])
        ]
    else:
        rm_prof_samples_valid = None

    rm_phi_meas = np.mean(rm_samples, axis=1)
    rm_phi_unc = np.std(rm_samples, axis=1)
    rm_phi_mean = np.average(rm_phi_meas, weights=rm_phi_unc)
    rm_scat_meas = np.mean(rm_scat_samples)
    rm_scat_unc = np.std(rm_scat_samples)

    if discard is not None:
        rm_prof_meas = np.mean(rm_prof_samples_valid)
        rm_prof_unc = np.std(rm_prof_samples_valid)
    else:
        rm_prof_meas = np.mean(rm_prof_samples)
        rm_prof_unc = np.std(rm_prof_samples)

    with open(f"{cube.source}.csv", "w") as f:
        f.write(
            f"{cube.source},{rm_phi_mean:.4f},{rm_prof_meas:.4f},{rm_prof_unc:.4f},{rm_scat_meas:.4f},{rm_scat_unc:.4f}\n"
        )

    with open(f"{cube.source}_rm_phi.csv", "w") as f:
        for meas, unc in zip(rm_phi_meas, rm_phi_unc):
            f.write(f"{meas:.4f},{unc:.4f}\n")

    psrutils.plotting.plot_rm_hist(
        rm_prof_samples,
        valid_samples=rm_prof_samples_valid,
        range=discard,
        title=cube.source,
        savename=f"{cube.source}_rm_prof_hist",
        save_pdf=True,
        logger=logger,
    )
    psrutils.plotting.plot_rm_hist(
        rm_scat_samples,
        title=cube.source,
        savename=f"{cube.source}_rm_scat_hist",
        logger=logger,
    )
    psrutils.plotting.plot_rm_vs_phi(
        rm_samples,
        savename=f"{cube.source}_rm_phi_boxplot",
        logger=logger,
    )

    logger.info("Running RM-CLEAN")
    rmcln_result = psrutils.rm_clean(phi, fdf, rmsf, rm_stats["rmsf_fwhm"], gain=0.5, logger=logger)
    cln_fdf = rmcln_result[0]

    logger.info("Plotting")
    psrutils.plotting.plot_2d_fdf(
        cube,
        np.abs(cln_fdf),
        phi,
        rmsf_fwhm=rm_stats["rmsf_fwhm"],
        rm_phi_meas=(rm_phi_meas, rm_phi_unc),
        rm_prof_meas=(rm_prof_meas, rm_prof_unc),
        mask=np.where(rm_phi_unc < 1, True, False),
        plot_stairs=stairs,
        plot_peaks=peaks,
        phase_range=phase_plotlim,
        phi_range=phi_plotlim,
        savename=f"{cube.source}_fdf",
        save_pdf=True,
        logger=logger,
    )
