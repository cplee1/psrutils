########################################################
# Licensed under the Academic Free License version 3.0 #
########################################################

import logging

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
from astropy.visualization import hist, quantity_support, time_support
from psrqpy import QueryATNF

from .cube import StokesCube
from .plotting import format_ticks

try:
    from spinifex import get_rm
except ImportError as e:
    MSG = "spinifex is not installed. To get ionospheric RM, install psrutils[iono]."
    raise ImportError(MSG) from e


__all__ = ["get_rm_iono"]

logger = logging.getLogger(__name__)

TELESCOPE_LOCS = {
    "MWA": EarthLocation(
        lat=-26.703319 * u.deg, lon=116.67081 * u.deg, height=377.827 * u.m
    ),
    "CHIME": EarthLocation(
        lat=49.3208 * u.deg, lon=-119.6236 * u.deg, height=545.0 * u.m
    ),
}


# TODO: Format docstring
def get_rm_iono(
    cube: StokesCube,
    bootstrap_nsamp: int | None = None,
    prefix: str = "jpl",
    server: str = "cddis",
    location: str = "mwa",
    savename: str | None = None,
) -> tuple[np.float_, np.float_]:
    """Get the mean ionospheric RM during an observation.

    Parameters
    ----------
    cube : `psrutils.StokesCube`
        A StokesCube object.
    bootstrap_nsamp : `int | None`, optional
        The number of bootstrap iteration to use to find the mean
        ionospheric RM. If `None` is provided, then no bootstrapping will
        be performed. Default: `None`.
    prefix : `str`, optional
        The analysis centre prefix. Default: "jpl".
    server : `str`, optional
        Server to download from. Default: "cddis".
    location : `str`, optional
        Earth location of the observation ["mwa", "chime"]. Default: "mwa".
    savename : `str | None`, optional
        If provided, will save a plot with this name. Default: `None`.

    Returns
    -------
    rm : `np.float_`
        The mean ionospheric RM during the observation.
    rm_err : `np.float_`
        The standard deviation of the mean ionospheric RM during the
        observation.
    """
    if location.upper() in TELESCOPE_LOCS.keys():
        logger.info(f"Using location: '{location.upper()}'")
        telescope_loc = TELESCOPE_LOCS[location.upper()]
    else:
        raise ValueError(f"Invalid location specified: '{location}'")

    times = Time(cube.start_mjd, format="mjd") + np.linspace(0, cube.int_time, 10) * u.s
    query = QueryATNF(psrs=cube.source, params=["RAJD", "DECJD"])
    psr = query.get_pulsar(cube.source)
    source = SkyCoord(ra=psr["RAJD"][0] * u.deg, dec=psr["DECJD"][0] * u.deg)

    rm = get_rm.get_rm_from_skycoord(
        loc=telescope_loc,
        times=times,
        source=source,
        prefix=prefix,
        server=server,
    )

    if bootstrap_nsamp:
        logger.debug("Bootstrapping RM_iono...")
        rm_samples = np.zeros(bootstrap_nsamp, dtype=float)
        for iter in range(bootstrap_nsamp):
            rm_samples[iter] = np.mean(st.norm.rvs(rm.rm, rm.rm_error))
        rm_val = np.mean(rm_samples)
        rm_err = np.std(rm_samples)
    else:
        rm_val = (np.mean(rm.rm),)
        rm_err = np.mean(rm.rm_error)

    if savename:
        with time_support(), quantity_support():
            if bootstrap_nsamp:
                fig, (ax_rm, ax_hist) = plt.subplots(
                    ncols=2,
                    figsize=(8, 5),
                    tight_layout=True,
                    sharey=True,
                    width_ratios=(3, 1),
                )
                hist(
                    rm_samples,
                    bins="knuth",
                    ax=ax_hist,
                    density=True,
                    orientation="horizontal",
                )
                ax_hist.set_xlabel("Probability Density")
                format_ticks(ax_hist)
            else:
                fig, ax_rm = plt.subplots(figsize=(7, 5), tight_layout=True)

            ax_rm.errorbar(rm.times.datetime, rm.rm, rm.rm_error, fmt="ko")
            ax_rm.axhline(rm_val, linestyle="--", color="k", linewidth=1, alpha=0.3)
            ax_rm.set_ylabel(
                "$\mathrm{RM}_\mathrm{iono}$ [$\mathrm{rad}\,\mathrm{m}^{-2}$]"
            )
            ax_rm.set_xlabel("UTC Date")
            format_ticks(ax_rm)
            ax_rm.set_xticklabels(ax_rm.get_xticklabels(), rotation=30)
            logger.info(f"Saving plot file: {savename}.png")
            fig.savefig(savename + ".png")
            plt.close()

    return rm_val, rm_err
