########################################################
# Licensed under the Academic Free License version 3.0 #
########################################################

import logging
from typing import Any

import click
import mwalib
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from mwa_vcs_fluxcal import __version__
from mwa_vcs_fluxcal.interface import simulate_sefd
from mwa_vcs_fluxcal.utils import qty_dict_to_toml
from scipy import integrate

from psrutils.cube import StokesCube
from psrutils.logger import log_levels, setup_logger

logger = logging.getLogger(__name__)


@click.command()
@click.help_option("-h", "--help")
@click.version_option(__version__, "-V", "--version")
@click.option(
    "-L",
    "log_level",
    type=click.Choice(log_levels.keys(), case_sensitive=False),
    default="INFO",
    show_default=True,
    help="The logger verbosity level.",
)
@click.option(
    "-m",
    "--metafits",
    "metafits",
    type=click.Path(exists=True),
    help="An MWA metafits file.",
)
@click.option(
    "-a",
    "--archive",
    "archive",
    type=click.Path(exists=True),
    help="An archive file to use to compute the pulse profile and get the "
    + "start/end times of the data.",
)
@click.option(
    "-b",
    "--bscrunch",
    "bscrunch",
    type=int,
    help="Bscrunch to this number of phase bins.",
)
@click.option("-c", "--centre", "centre", is_flag=True, help="Centre the pulse.")
@click.option(
    "--fine_res",
    type=float,
    default=2.0,
    show_default=True,
    help="The resolution of the integral, in arcmin.",
)
@click.option(
    "--coarse_res",
    type=float,
    default=30.0,
    show_default=True,
    help="The resolution of the primary beam map, in arcmin. Must be an integer "
    + "multiple of --fine_res.",
)
@click.option(
    "--min_pbp",
    type=click.FloatRange(0.0, 1.0),
    default=0.001,
    show_default=True,
    help="Only integrate above this primary beam power.",
)
@click.option(
    "--nfreq",
    type=int,
    default=1,
    show_default=True,
    help="The number of frequency steps to simulate.",
)
@click.option(
    "--ntime",
    type=int,
    default=1,
    show_default=True,
    help="The number of time steps to simulate.",
)
@click.option(
    "--max_pix_per_job",
    type=int,
    default=10**5,
    show_default=True,
    help="The maximum number of sky area pixels to compute per job.",
)
@click.option(
    "--fc",
    type=click.FloatRange(1.0, 10.0),
    default=1.43,
    show_default=True,
    help="The beamforming coherency factor.",
)
@click.option(
    "--eta",
    type=click.FloatRange(0.0, 1.0),
    default=0.98,
    show_default=True,
    help="The radiation efficiency of the array.",
)
@click.option(
    "--bw_flagged",
    type=click.FloatRange(0.0, 1.0),
    default=0.0,
    show_default=True,
    help="The fraction of the bandwidth flagged.",
)
@click.option(
    "--time_flagged",
    type=click.FloatRange(0.0, 1.0),
    default=0.0,
    show_default=True,
    help="The fraction of the integration time flagged.",
)
@click.option(
    "-os",
    "--offset_start",
    "offset_start",
    type=float,
    help="Data start offset in seconds from scheduled observation start time.",
)
@click.option(
    "-oe",
    "--offset_end",
    "offset_end",
    type=float,
    help="Data end offset in seconds from scheduled observation start time.",
)
@click.option("--plot_trec", is_flag=True, help="Plot the receiver temperature.")
@click.option("--plot_pb", is_flag=True, help="Plot the primary beam in Alt/Az.")
@click.option("--plot_tab", is_flag=True, help="Plot the tied-array beam in Alt/Az.")
@click.option("--plot_tsky", is_flag=True, help="Plot sky temperature in Alt/Az.")
@click.option(
    "--plot_integrals", is_flag=True, help="Plot the integral quantities in Alt/Az."
)
@click.option(
    "--plot_3d", is_flag=True, help="Plot the results in 3D (time,freq,data)."
)
@click.option(
    "--plot_diagnostics", is_flag=True, help="Plot profile fitting diagnostics."
)
@click.option("-o", "--outfile", type=str, help="The prefix of the output file names.")
def main(
    log_level: str,
    metafits: str,
    archive: str,
    bscrunch: int,
    centre: bool,
    fine_res: float,
    coarse_res: float,
    min_pbp: float,
    nfreq: int,
    ntime: int,
    max_pix_per_job: int,
    fc: float,
    eta: float,
    bw_flagged: float,
    time_flagged: float,
    offset_start: float,
    offset_end: float,
    plot_trec: bool,
    plot_pb: bool,
    plot_tab: bool,
    plot_tsky: bool,
    plot_integrals: bool,
    plot_3d: bool,
    plot_diagnostics: bool,
    outfile: str,
) -> None:
    setup_logger("psrutils", log_level)
    setup_logger("mwa_vcs_fluxcal", log_level)

    logger.info(f"Loading archive: {archive}")
    cube = StokesCube.from_psrchive(archive, tscrunch=1, fscrunch=1, bscrunch=bscrunch)
    logger.info(f"Number of bins: {cube.num_bin}")
    srcname_raw = cube.source
    srcname_ltx = cube.source.replace("-", "$-$")
    if outfile is None:
        outfile = srcname_raw

    phase_rot = 0.0
    if centre:
        logger.info("Rotating the peak to the centre of the profile")
        max_idx = np.argmax(cube.profile)
        phase_rot = (max_idx - cube.num_bin // 2) / cube.num_bin
        cube.rotate_phase(phase_rot)

    # Get the profile as a SplineProfile object to use for analysis
    profile = cube.spline_profile

    # Find the onpulse using the spline method
    profile.fit_spline_gridsearch()
    profile.get_onpulse()
    if plot_diagnostics:
        profile.plot_diagnostics(
            plot_underestimate=False,
            plot_overestimate=True,
            sourcename=srcname_ltx,
            savename=f"{outfile}_profile_diagnostics",
        )

    # Normalise the profile so that the noise has a standard deviation of unity
    snr_profile = profile.debase_profile / profile.noise_est

    # Get pulsar coordinates
    ra_hms, dec_dms = cube.archive.get_coordinates().getHMSDMS().split(" ")
    target_coords = SkyCoord(ra_hms, dec_dms, frame="icrs", unit=("hourangle", "deg"))
    logger.info(f"Target RA/Dec = {target_coords.to_string(style='hmsdms')}")

    # Get start and end times
    context = mwalib.MetafitsContext(metafits)
    obs_start = Time(context.sched_start_mjd, format="mjd")
    # obs_end = Time(context.sched_end_mjd, format="mjd")
    start_time = Time(cube.start_mjd, format="mjd")
    end_time = Time(cube.end_mjd, format="mjd")

    if offset_start is None:
        start_time_offset = start_time - obs_start
    else:
        start_time_offset = offset_start * u.s
    if offset_end is None:
        end_time_offset = end_time - obs_start
    else:
        end_time_offset = offset_end * u.s

    results = simulate_sefd(
        metafits,
        target_coords,
        start_time_offset.to(u.s).value,
        end_time_offset.to(u.s).value,
        fine_res,
        coarse_res,
        min_pbp,
        nfreq,
        ntime,
        max_pix_per_job,
        fc,
        eta,
        plot_trec,
        plot_pb,
        plot_tab,
        plot_tsky,
        plot_integrals,
        plot_3d,
        outfile,
    )

    # Radiometer equation
    int_time = end_time - start_time
    bw = cube.bandwidth * u.MHz
    int_time_per_bin = int_time * (1 - time_flagged) / cube.num_bin
    bw_valid = bw * (1 - bw_flagged)
    radiometer_noise = results["SEFD_mean"] / np.sqrt(
        2 * bw_valid.to(1 / u.s) * int_time_per_bin
    )
    flux_scale = radiometer_noise / profile.noise_est

    # Flux density calculations
    flux_density_profile = snr_profile * radiometer_noise
    S_peak = np.max(flux_density_profile)
    S_mean = integrate.trapezoid(flux_density_profile) / cube.num_bin

    # Statistical uncertainty due to random noise in the profile
    u_stat_S_peak = radiometer_noise
    u_stat_S_mean = radiometer_noise / np.sqrt(cube.num_bin)

    # Systematic uncertainty due to assumptions in calibration
    if abs(target_coords.galactic.l.deg) < 10:
        u_sys_rel = 0.4
    else:
        u_sys_rel = 0.3
    u_sys_S_peak = u_sys_rel * S_peak
    u_sys_S_mean = u_sys_rel * S_mean

    # Total uncertainty
    u_S_peak = np.sqrt(u_stat_S_peak**2 + u_sys_S_peak**2)
    u_S_mean = np.sqrt(u_stat_S_mean**2 + u_sys_S_mean**2)

    # Log results
    logger.info(f"SEFD = {results['SEFD_mean'].to(u.Jy).value:.2f} Jy")
    logger.info(
        f"Peak flux density = {S_peak.to(u.mJy).value:.2f} "
        + f"+/- {u_stat_S_peak.to(u.mJy).value:.2f} (stat) "
        + f"+/- {u_sys_S_peak.to(u.mJy).value:.2f} (sys) mJy"
    )
    logger.info(
        f"Mean flux density = {S_mean.to(u.mJy).value:.2f} "
        + f"+/- {u_stat_S_mean.to(u.mJy).value:.2f} (stat) "
        + f"+/- {u_sys_S_mean.to(u.mJy).value:.2f} (sys) mJy"
    )

    # Add to end of the dictionary
    results["Noise_std"] = radiometer_noise.to(u.mJy)
    results["Flux_scale"] = flux_scale.to(u.mJy)
    results["S_peak"] = S_peak.to(u.mJy)
    results["u_stat_S_peak"] = u_stat_S_peak.to(u.mJy)
    results["u_sys_S_peak"] = u_sys_S_peak.to(u.mJy)
    results["u_S_peak"] = u_S_peak.to(u.mJy)
    results["S_mean"] = S_mean.to(u.mJy)
    results["u_stat_S_mean"] = u_stat_S_mean.to(u.mJy)
    results["u_sys_S_mean"] = u_sys_S_mean.to(u.mJy)
    results["u_S_mean"] = u_S_mean.to(u.mJy)

    # Add to beginning of the dictionary
    pre_dict = dict(
        Source=srcname_raw,
        Nbin=cube.num_bin,
        Phase_rotation=phase_rot,
        Time_frac_flagged=time_flagged,
        BW_frac_flagged=bw_flagged,
    )
    results = dict(pre_dict, **results)

    qty_dict_to_toml(results, f"{outfile}_fluxcal_results.toml")


if __name__ == "__main__":
    main()
