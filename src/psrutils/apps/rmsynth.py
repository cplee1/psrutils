import warnings

import click
import numpy as np

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
@click.option("-f", "fscr", type=int, help="Fscrunch to this number of channels.")
@click.option("-b", "bscr", type=int, help="Bscrunch to this number of phase bins.")
@click.option("-r", "rotate", type=float, help="Rotate phase by this amount.")
@click.option("-c", "centre", is_flag=True, help="Centre the pulse.")
@click.option("--w_off", "offpulse_ws", type=int, help="The size of the offpulse window in bins.")
@click.option("--w_on", "onpulse_ws", type=int, help="The size of the onpulse window in bins.")
@click.option("--rmlim", type=float, default=100.0, help="RM limit.")
@click.option("--rmres", type=float, default=0.01, help="RM resolution.")
@click.option("-n", "nsamp", type=int, help="The number of bootstrap samples.")
@click.option("--p0_cutoff", type=float, default=3.0, help="Mask below this L/sigma_I value.")
@click.option("--phi_plotlim", type=float, nargs=2, help="Plot limits in rad/m^2.")
@click.option("--phase_plotlim", type=float, nargs=2, help="Plot limits in rotations.")
@click.option(
    "--discard", type=float, nargs=2, help="Discard RM samples outside this range in rad/m^2."
)
@click.option("--clean_cutoff", type=float, default=3.0, help="RM-CLEAN component S/N cutoff.")
@click.option("--meas_rm_prof", is_flag=True, help="Measure RM_prof.")
@click.option("--meas_rm_scat", is_flag=True, help="Measure RM_scat.")
@click.option("--no_clean", is_flag=True, help="Do not run RM-CLEAN on the FDF.")
@click.option("--boxplot", is_flag=True, help="Plot RM_phi as boxplots.")
@click.option("--peaks", is_flag=True, help="Plot RM measurements.")
@click.option("--plot_clean_comps", is_flag=True, help="Do not run RM-CLEAN on the FDF.")
@click.option("--plot_onpulse", is_flag=True, help="Shade the on-pulse region.")
@click.option("--plot_pa", is_flag=True, help="Plot the position angle.")
@click.option("--plot_pol_prof", is_flag=True, help="Plot the full polarisation profile.")
@click.option("--save_pdf", is_flag=True, help="Save as a PDF.")
@click.option("-d", "dark_mode", is_flag=True, help="Use a dark background.")
def main(
    archive: str,
    log_level: str,
    fscr: int,
    bscr: int,
    rotate: float,
    centre: bool,
    offpulse_ws: int,
    onpulse_ws: int,
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
    no_clean: bool,
    boxplot: bool,
    peaks: bool,
    plot_clean_comps: bool,
    plot_onpulse: bool,
    plot_pa: bool,
    plot_pol_prof: bool,
    save_pdf: bool,
    dark_mode: bool,
) -> None:
    log_level_dict = psrutils.get_log_levels()
    logger = psrutils.get_logger(log_level=log_level_dict[log_level])

    logger.info(f"Loading archive: {archive}")
    cube = psrutils.StokesCube.from_psrchive(archive, False, 1, fscr, bscr, rotate)
    logger.info(f"Number of bins: {cube.num_bin}")

    if centre:
        logger.info("Rotating the peak to the centre of the profile")
        max_idx = np.argmax(cube.profile)
        cube.rotate_phase((max_idx - cube.num_bin // 2) / cube.num_bin)

    # Get off/on-pulse windows, assuming offpulse is 1/8 of profile
    offpulse_win = psrutils.get_offpulse_region(cube.profile, windowsize=offpulse_ws, logger=logger)
    logger.debug(f"Offpulse bin indices: {offpulse_win}")
    logger.info(f"Offpulse window size: {offpulse_win.size}")
    onpulse_win = psrutils.get_onpulse_region(cube.profile, windowsize=onpulse_ws, logger=logger)
    logger.debug(f"Onpulse bin indices: {onpulse_win}")
    logger.info(f"Onpulse window size: {onpulse_win.size}")
    psrutils.plot_profile(
        cube,
        offpulse_win=offpulse_win,
        onpulse_win=onpulse_win,
        savename=f"{cube.source}_profile",
        save_pdf=save_pdf,
        logger=logger,
    )

    logger.info("Running RM-Synthesis")
    phi = np.arange(-1.0 * rmlim, rmlim + rmres, rmres)
    rmsyn_result = psrutils.rm_synthesis(
        cube,
        phi,
        onpulse_win=onpulse_win,
        meas_rm_prof=meas_rm_prof,
        meas_rm_scat=meas_rm_scat,
        bootstrap_nsamp=nsamp,
        offpulse_win=offpulse_win,
        logger=logger,
    )
    fdf, rmsf, _, rm_phi_samples, rm_prof_samples, rm_scat_samples, rm_stats = rmsyn_result
    logger.info(rm_stats)

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
            # For bins with no valid samples (when using --discard), suppress
            # the "Mean of empty slice" warning
            warnings.simplefilter("ignore", category=RuntimeWarning)
            rm_phi_meas = np.nanmean(rm_phi_samples_valid, axis=1)
            rm_phi_unc = np.nanstd(rm_phi_samples_valid, axis=1)
        rm_phi_qty = (rm_phi_meas, rm_phi_unc)
        peak_mask = np.where(rm_phi_unc < rm_stats["rmsf_fwhm"] / 2, True, False)

        with open(f"{cube.source}_rm_phi.csv", "w") as f:
            for meas, unc in zip(rm_phi_meas, rm_phi_unc, strict=False):
                f.write(f"{meas:.4f},{unc:.4f}\n")

        if boxplot:
            psrutils.plotting.plot_rm_vs_phi(
                rm_phi_samples,
                savename=f"{cube.source}_rm_phi_boxplot",
                save_pdf=save_pdf,
                logger=logger,
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

            logger.info(f"Saving CSV file: {cube.source}_rm_prof.csv")
            with open(f"{cube.source}_rm_prof.csv", "w") as f:
                f.write(f"{cube.source},{rm_prof_meas:.4f},{rm_prof_unc:.4f}\n")

            psrutils.plotting.plot_rm_hist(
                rm_prof_samples,
                valid_samples=plot_valid_samples,
                range=discard,
                title=cube.source.replace("-", "$-$"),
                savename=f"{cube.source}_rm_prof_hist",
                save_pdf=save_pdf,
                logger=logger,
            )
        else:
            rm_prof_qty = None

        if meas_rm_scat:
            rm_scat_meas = np.mean(rm_scat_samples)
            rm_scat_unc = np.std(rm_scat_samples)

            logger.info(f"Saving CSV file: {cube.source}_rm_scat.csv")
            with open(f"{cube.source}_rm_scat.csv", "w") as f:
                f.write(f"{cube.source},{rm_scat_meas:.4f},{rm_scat_unc:.4f}\n")

            psrutils.plotting.plot_rm_hist(
                rm_scat_samples,
                title=cube.source,
                savename=f"{cube.source}_rm_scat_hist",
                save_pdf=save_pdf,
                logger=logger,
            )
    else:
        rm_phi_qty = (rm_phi_samples, None)
        if meas_rm_prof:
            rm_prof_qty = (rm_prof_samples, None)

            logger.info(f"Saving CSV file: {cube.source}_rm_prof.csv")
            with open(f"{cube.source}_rm_prof.csv", "w") as f:
                f.write(f"{cube.source},{rm_prof_samples:.4f},none\n")
        else:
            rm_prof_qty = None
        peak_mask = None

    cln_comps = None
    if no_clean:
        cln_fdf = fdf
    else:
        logger.info("Running RM-CLEAN")
        rmcln_result = psrutils.rm_clean(
            phi, fdf, rmsf, rm_stats["rmsf_fwhm"], gain=0.5, cutoff=clean_cutoff, logger=logger
        )
        cln_fdf = rmcln_result[0]
        if plot_clean_comps:
            cln_comps = rmcln_result[2]

    meas_delta_vi = True
    if meas_delta_vi:
        delta_vi = psrutils.get_delta_vi(cube, onpulse_win=onpulse_win, logger=logger)

    logger.info("Plotting")
    psrutils.plotting.plot_2d_fdf(
        cube,
        np.abs(cln_fdf),
        phi,
        rmsf_fwhm=rm_stats["rmsf_fwhm"],
        rm_phi_qty=rm_phi_qty,
        rm_prof_qty=rm_prof_qty,
        onpulse_win=onpulse_win,
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
        logger=logger,
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
            logger=logger,
        )
