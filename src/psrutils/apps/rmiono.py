########################################################
# Licensed under the Academic Free License version 3.0 #
########################################################

import logging
from typing import Any

import click
import rtoml
from psrqpy import QueryATNF
from requests.exceptions import HTTPError

import psrutils

logger = logging.getLogger(__name__)


@click.command()
@click.argument("archive", nargs=1, type=click.Path(exists=True))
@click.help_option("-h", "--help")
@click.version_option(psrutils.__version__, "-V", "--version")
@click.option(
    "-L",
    "log_level",
    type=click.Choice(psrutils.log_levels.keys(), case_sensitive=False),
    default="info",
    show_default=True,
    help="The logger verbosity level.",
)
@click.option(
    "-t",
    "telescope",
    type=click.Choice(["mwa", "chime"], case_sensitive=False),
    default="mwa",
    show_default=True,
    help="The telescope used to collect the observation.",
)
def main(archive: str, log_level: str, telescope: str) -> None:
    psrutils.setup_logger("psrutils", log_level)

    # Initialise a dictionary to store the various results
    results: dict[str, Any] = {}

    # We only need the archive for the metadata, so scrunch
    logger.info(f"Loading archive: {archive}")
    cube = psrutils.StokesCube.from_psrchive(archive, False, 1, 1, None, None)

    # Query catalogue to get J-names
    cat_table = QueryATNF(version="2.7.0", params=["PSRB", "PSRJ"]).table
    psrb = list(cat_table["PSRB"])
    psrj = list(cat_table["PSRJ"])
    if cube.source.startswith("B"):
        try:
            jname = psrj[psrb.index(cube.source)]
        except ValueError:
            jname = cube.source
    else:
        jname = cube.source

    psrutils.setup_logger("spinifex", log_level)

    try:
        rm_iono, rm_iono_err = psrutils.iono.get_rm_iono(
            cube,
            bootstrap_nsamp=int(1e4),
            location=telescope,
            savename=f"{jname}_rm_iono",
        )
    except HTTPError as e:
        logger.error(e)

    logger.info(f"RM_iono = {rm_iono:.3f}+/-{rm_iono_err:.3f} rad/m2")
    results["RM_iono"] = rm_iono
    results["RM_iono_unc"] = rm_iono_err

    logger.info(f"Saving results: {jname}_rmiono_results.toml")
    with open(f"{jname}_rmiono_results.toml", "w") as f:
        rtoml.dump(psrutils.pythonise(results), f)
