########################################################
# Licensed under the Academic Free License version 3.0 #
########################################################

import logging
from typing import Callable

import cmasher as cmr
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
from astropy.visualization import hist
from matplotlib.axes import Axes
from numpy.typing import NDArray

from .cube import StokesCube
from .misc import jname_to_name
from .polarisation import PolProfile, get_bias_corrected_pol_profile
from .profile import get_profile_mask_from_pairs

__all__ = [
    "centre_offset_degrees",
    "format_ticks",
    "plot_profile",
    "plot_pol_profile",
    "plot_freq_phase",
    "plot_time_phase",
    "plot_2d_fdf",
    "plot_rm_hist",
    "plot_rm_vs_phi",
]

logger = logging.getLogger(__name__)


def centre_offset_degrees(phase_bins: NDArray[np.floating]) -> NDArray[np.floating]:
    """Convert an array of bins from rotations to degrees with a 180 degree
    phase shift.

    Parameters
    ----------
    phase_bins : NDArray[floating]
        An array of phase bins in units of rotations.

    Returns
    -------
    NDArray[floating]
        An array of phase bins in units of degrees, with a 180 degree
        phase shift from the input array.
    """
    return phase_bins * 360 - 180


def format_ticks(ax: Axes) -> None:
    """Format the x and y ticks so that they point inwards and show both
    major and minor ticks on all four spines.

    Parameters
    ----------
    ax : Axes
        The Axes to update.
    """
    ax.minorticks_on()
    ax.tick_params(which="both", right=True, top=True)
    ax.tick_params(axis="both", which="both", direction="in")
    ax.tick_params(axis="both", which="major", length=4)
    ax.tick_params(axis="both", which="minor", length=2)


# TODO: Make docstring
def _add_profile_to_axes(
    cube: StokesCube,
    ax: Axes | None = None,
    normalise: bool = True,
    voffset: float = 0.0,
    bin_func: Callable[[NDArray[np.floating]], NDArray[np.floating]] | None = None,
    **plot_kwargs,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    # Get an array of bins in the desired units
    bins = np.linspace(0, 1, cube.num_bin, dtype=np.float64)
    if isinstance(bin_func, Callable):
        bins = bin_func(bins)

    # Normalise by the peak of the profile
    norm_const: float = 1.0
    if normalise:
        norm_const = float(np.max(cube.profile))
    norm_prof = (cube.profile / norm_const).astype(np.float64)

    ax.plot(bins, norm_prof + voffset, **plot_kwargs)

    return bins, norm_prof


# TODO: Make docstring
def _add_pol_profile_to_axes(
    cube: StokesCube,
    ax_prof: Axes | None = None,
    ax_pa: Axes | None = None,
    ax_frac: Axes | None = None,
    normalise: bool = True,
    plot_v: bool = True,
    p0_cutoff: float | None = 3.0,
    voffset: float = 0.0,
    bin_func: Callable[[NDArray[np.floating]], NDArray[np.floating]] | None = None,
    I_colour: str = "k",
    L_colour: str = "tab:red",
    V_colour: str = "tab:blue",
    PA_colour: str = "k",
    L_frac_colour: str = "tab:red",
    V_frac_colour: str = "tab:blue",
    **plot_kwargs,
) -> tuple[NDArray[np.float64], PolProfile]:
    # Get an array of bins in the desired units
    bins = np.linspace(0, 1, cube.num_bin, dtype=np.float64)
    if isinstance(bin_func, Callable):
        bins = bin_func(bins)

    # Normalise by the peak of the profile
    norm_const: float = 1.0
    if normalise:
        norm_const = float(np.max(cube.profile))

    pol_profile = get_bias_corrected_pol_profile(
        cube.pol_profile.astype(np.float64), spline=False
    )

    if isinstance(ax_prof, Axes):
        ax_prof.plot(
            bins,
            pol_profile.iquv[0] / norm_const + voffset,
            linestyle="-",
            color=I_colour,
            zorder=8,
            **plot_kwargs,
        )
        ax_prof.plot(
            bins,
            pol_profile.l_true / norm_const + voffset,
            linestyle="--",
            color=L_colour,
            zorder=10,
            **plot_kwargs,
        )
        if plot_v:
            ax_prof.plot(
                bins,
                pol_profile.iquv[3] / norm_const + voffset,
                linestyle=":",
                color=V_colour,
                zorder=9,
                **plot_kwargs,
            )

    if isinstance(ax_pa, Axes):
        pa_mask = pol_profile.p0_l > p0_cutoff
        if "linewidth" in plot_kwargs.keys():
            lw = plot_kwargs["linewidth"]
        elif "lw" in plot_kwargs.keys():
            lw = plot_kwargs["lw"]
        else:
            lw = None
        for offset in [0, -180, 180]:
            ax_pa.errorbar(
                x=bins[pa_mask],
                y=pol_profile.pa[0, pa_mask] + offset,
                yerr=pol_profile.pa[1, pa_mask],
                color=PA_colour,
                ecolor=PA_colour,
                marker="none",
                ms=1,
                linestyle="none",
                elinewidth=lw,
                capthick=lw,
                capsize=0,
                **plot_kwargs,
            )

    if isinstance(ax_frac, Axes):
        l_frac_mask = np.logical_and(
            np.where(pol_profile.l_frac[1] / pol_profile.l_frac[0] < 1.0, True, False),
            np.where(pol_profile.l_frac[0] > 0.0, True, False),
            np.where(pol_profile.l_frac[0] < 1.0, True, False),
        )
        ax_frac.errorbar(
            x=bins[l_frac_mask],
            y=pol_profile.l_frac[0, l_frac_mask],
            yerr=pol_profile.l_frac[1, l_frac_mask],
            color=L_frac_colour,
            ecolor=L_frac_colour,
            marker=".",
            ms=1,
            linestyle="none",
            elinewidth=lw,
            capthick=lw,
            capsize=0,
            **plot_kwargs,
        )
        if plot_v:
            v_frac_mask = np.logical_and(
                np.where(
                    pol_profile.v_frac[1] / pol_profile.v_frac[0] < 1.0, True, False
                ),
                np.where(pol_profile.v_frac[0] > 0.0, True, False),
                np.where(pol_profile.v_frac[0] < 1.0, True, False),
            )
            ax_frac.errorbar(
                x=bins[v_frac_mask],
                y=pol_profile.v_frac[0, v_frac_mask],
                yerr=pol_profile.v_frac[1, v_frac_mask],
                color=V_frac_colour,
                ecolor=V_frac_colour,
                marker=".",
                ms=1,
                linestyle="none",
                elinewidth=lw,
                capthick=lw,
                capsize=0,
                **plot_kwargs,
            )

    return bins, pol_profile


def plot_profile(
    cube: StokesCube,
    pol: int = 0,
    normalise: bool = False,
    savename: str = "profile",
    save_pdf: bool = False,
) -> None:
    """Create a plot of integrated flux density vs phase for a specified
    polarisation.

    Parameters
    ----------
    cube : StokesCube
        A StokesCube object.
    pol : int, default: 0
        The polarisation index (0=I, 1=Q, 2=U, 3=V).
    savename : str, default: "profile"
        The name of the plot file excluding the extension.
    save_pdf : bool, default: Fale
        Save the plot as a pdf?
    """
    if pol not in [0, 1, 2, 3]:
        raise ValueError("pol must be an integer between 0 and 3 inclusive")

    fig, ax = plt.subplots(tight_layout=True)

    bins, _ = _add_profile_to_axes(
        cube, ax, normalise=normalise, color="k", linewidth=1, label="Profile"
    )
    ax.axhline(0, linestyle=":", linewidth=1.2, color="k", alpha=0.3)
    ax.set_xlim([bins[0], bins[-1]])
    ax.set_xlabel("Pulse Phase")
    if normalise:
        ax.set_ylabel("Flux Density")
    else:
        ax.set_ylabel("Normalised Intensity")

    logger.info(f"Saving plot file: {savename}.png")
    fig.savefig(savename + ".png")

    if save_pdf:
        logger.info(f"Saving plot file: {savename}.pdf")
        fig.savefig(savename + ".pdf")

    plt.close()


def plot_pol_profile(
    cube: StokesCube,
    normalise: bool = False,
    bin_func: Callable[[NDArray[np.floating]], NDArray[np.floating]] | None = None,
    rmsf_fwhm: float | None = None,
    rm_phi_qty: tuple[NDArray, NDArray] | None = None,
    rm_prof_qty: tuple[float, float] | None = None,
    rm_mask: NDArray | None = None,
    delta_vi: NDArray | None = None,
    phase_range: tuple[float, float] | None = None,
    p0_cutoff: float | None = 3.0,
    plot_v: bool = True,
    plot_pol_frac: bool = False,
    savename: str = "pol_profile",
    save_pdf: bool = False,
    save_data: bool = False,
) -> None:
    """Create a plot of integrated flux density vs phase for a specified
    polarisation.

    Parameters
    ----------
    cube : StokesCube
        A StokesCube object.
    normalise : bool, default: False
        Normalise the pulse profile by the peak intensity.
    bin_func : Callable, default: None
        A function to transform the bin units from phase into something else.
    rmsf_fwhm : float, default: None
        The FWHM of the RM spread function.
    rm_phi_qty : tuple[NDArray, NDArray], default: None
        The RM measurements and uncertainties for each phase bin.
    rm_prof_qty : tuple[float, float], default: None
        The RM measurement and uncertainty for the profile.
    rm_mask : NDArray, default: None
        An array of booleans to act as a mask for the measured RM values.
    delta_vi : NDArray, default: None
        The change in Stokes V/I over the bandwidth per bin.
    phase_range : tuple[float, float], default: [0, 1]
        The phase range in rotations.
    p0_cutoff : float, default: 3.0
        Mask all RM and PA measurements below this polarisation measure. If `None` is
        specified then no mask will be applied.
    plot_v : bool, default: True
        Plot Stokes V.
    plot_pol_frac : bool, default: False
        Plot the fractional polarisation profiles.
    savename : str, default: "pol_profile"
        The name of the plot file excluding the extension.
    save_pdf : bool, default: False
        Save the plot as a pdf?
    save_data : bool, default: False
        Save the plot data?
    """
    valid_rm = False
    if rm_prof_qty is not None:
        if rm_phi_qty[1] is not None and rm_prof_qty[1] < rmsf_fwhm / 2:
            # If an uncertainty is provided, it must be <HWHM of RMSF
            valid_rm = True
        elif rm_phi_qty[1] is None:
            # If no uncertainty is provided, assume it is fine
            valid_rm = True

    if valid_rm:
        cube.defaraday(rm_prof_qty[0])

    plot_rm = False
    if rm_phi_qty is not None:
        plot_rm = True

    # Define Figure and Axes
    if plot_rm and delta_vi is not None:
        if plot_pol_frac:
            logger.error("Cannot plot RM, V/I, and polarisation fractions at once.")
        fig = plt.figure(figsize=(6, 6.5), layout="tight")
        gs = gridspec.GridSpec(
            ncols=1, nrows=4, figure=fig, height_ratios=(1, 1, 1, 3), hspace=0
        )
        ax_frac = None
        ax_rm = fig.add_subplot(gs[0])
        ax_dv = fig.add_subplot(gs[1])
        ax_pa = fig.add_subplot(gs[2])
        ax_prof = fig.add_subplot(gs[3])
    elif plot_rm:
        if plot_pol_frac:
            logger.error("Cannot plot RM and polarisation fractions at once.")
        fig = plt.figure(figsize=(6, 5.6), layout="tight")
        gs = gridspec.GridSpec(
            ncols=1, nrows=3, figure=fig, height_ratios=(1, 1, 3), hspace=0
        )
        ax_frac = None
        ax_rm = fig.add_subplot(gs[0])
        ax_dv = None
        ax_pa = fig.add_subplot(gs[1])
        ax_prof = fig.add_subplot(gs[2])
    elif plot_pol_frac:
        fig = plt.figure(figsize=(6, 6), layout="tight")
        gs = gridspec.GridSpec(
            ncols=1, nrows=3, figure=fig, height_ratios=(1, 1, 2), hspace=0
        )
        ax_frac = fig.add_subplot(gs[0])
        ax_rm = None
        ax_dv = None
        ax_pa = fig.add_subplot(gs[1])
        ax_prof = fig.add_subplot(gs[2])
    else:
        fig = plt.figure(figsize=(5, 4), layout="tight")
        gs = gridspec.GridSpec(
            ncols=1, nrows=2, figure=fig, height_ratios=(1, 2), hspace=0
        )
        ax_frac = None
        ax_rm = None
        ax_dv = None
        ax_pa = fig.add_subplot(gs[0])
        ax_prof = fig.add_subplot(gs[1])

    # Styles
    lw = 0.7
    line_col = "k"

    # Phase range to plot
    if phase_range is None:
        phase_range = [0, 1]
    if bin_func is not None:
        phase_range = [bin_func(phase_range[0]), bin_func(phase_range[1])]

    # Add flux and PA
    bins, pol_profile = _add_pol_profile_to_axes(
        cube,
        ax_prof=ax_prof,
        ax_pa=ax_pa,
        ax_frac=ax_frac,
        normalise=normalise,
        plot_v=plot_v,
        p0_cutoff=p0_cutoff,
        bin_func=bin_func,
        linewidth=lw,
    )
    # ax_prof.axhline(y=0, color="k", lw=lw, linestyle="--")

    lw = 0.7
    caplw = 0.7
    scatter_params = dict(
        color="k",
        marker="none",
        ms=1,
        linestyle="none",
        elinewidth=caplw,
        capthick=caplw,
        capsize=0,
    )

    # Add RM
    if plot_rm:
        if rm_mask is None:
            rm_mask = np.full(rm_phi_qty[0].shape[0], True)

        full_rm_mask = rm_mask & (pol_profile.p0_l > p0_cutoff)

        if rm_phi_qty[1] is not None:
            rm_phi_unc = np.abs(rm_phi_qty[1])[full_rm_mask]
        else:
            rm_phi_unc = None
            scatter_params["marker"] = "o"

        ax_rm.errorbar(
            x=bins[full_rm_mask],
            y=rm_phi_qty[0][full_rm_mask],
            yerr=rm_phi_unc,
            **scatter_params,
        )

        rm_lims = ax_rm.get_ylim()

        # Plot RM_profile + uncertainty region
        if valid_rm:
            if rm_prof_qty[1] is not None:
                y1 = [rm_prof_qty[0] - rm_prof_qty[1]] * 2
                y2 = [rm_prof_qty[0] + rm_prof_qty[1]] * 2
                ax_rm.fill_between(
                    phase_range, y1, y2, color="tab:red", alpha=0.5, zorder=0
                )
            else:
                ax_rm.axhline(
                    y=rm_prof_qty[0],
                    linestyle="--",
                    color="tab:red",
                    linewidth=lw,
                    zorder=1,
                )

            # Plot RM=0 + uncertainty region
            if rmsf_fwhm is not None:
                y1 = [0 - rmsf_fwhm / 2.0] * 2
                y2 = [0 + rmsf_fwhm / 2.0] * 2
                ax_rm.fill_between(
                    phase_range, y1, y2, color=line_col, alpha=0.2, zorder=0
                )
                ax_rm.axhline(
                    y=0, linestyle=":", color=line_col, linewidth=lw, zorder=1
                )

        ax_rm.set_ylim(rm_lims)

    # Plot Delta(V/I)
    if ax_dv is not None:
        ax_dv.errorbar(
            x=bins,
            y=delta_vi[0],
            yerr=delta_vi[1],
            **scatter_params,
        )
        ax_dv.axhline(y=0, linestyle=":", color=line_col, linewidth=lw, zorder=1)

    # Add text to profile to axis
    if plot_rm:
        text_shift = 0.01
    else:
        text_shift = 0.0
    ax_prof.text(
        0.025 + text_shift,
        0.95,
        f"{jname_to_name(cube.source).replace('-', '$-$')}",
        horizontalalignment="left",
        verticalalignment="top",
        transform=ax_prof.transAxes,
    )
    ax_prof.text(
        0.975 - text_shift,
        0.95,
        f"{cube.ctr_freq:.0f} MHz",
        horizontalalignment="right",
        verticalalignment="top",
        transform=ax_prof.transAxes,
    )

    # X limits
    for iax in [ax_frac, ax_rm, ax_dv, ax_pa, ax_prof]:
        if iax is not None:
            iax.set_xlim(phase_range)

    # Y limits
    ax_pa.set_ylim([-120, 120])
    if ax_frac is not None:
        ax_frac.set_ylim([0.0, 1.0])

    # Ticks
    if ax_frac is not None:
        ax_frac.set_xticklabels([])
    if plot_rm:
        ax_rm.set_xticklabels([])
    if ax_dv is not None:
        ax_dv.set_xticklabels([])
    ax_pa.set_xticklabels([])
    ax_pa.set_yticks([-90, 0, 90])
    ax_pa.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(15))
    for iax in [ax_frac, ax_rm, ax_dv, ax_pa, ax_prof]:
        if iax is not None:
            iax.tick_params(which="both", right=True, top=True)
            iax.minorticks_on()
            iax.tick_params(axis="both", which="both", direction="in")
            iax.tick_params(axis="both", which="major", length=4)
            iax.tick_params(axis="both", which="minor", length=2)

    # Labels
    if ax_frac is not None:
        ax_frac.set_ylabel("Frac. Pol.")
    if plot_rm:
        ax_rm.set_ylabel("RM\n[$\mathrm{rad}\,\mathrm{m}^{-2}$]")
    if ax_dv is not None:
        ax_dv.set_ylabel("$\Delta(V/I)$")
    if bin_func is centre_offset_degrees:
        ax_prof.set_xlabel("Pulse Longitude [deg]")
    else:
        ax_prof.set_xlabel("Pulse Phase")
    if normalise:
        ax_prof.set_ylabel("Normalised Intensity")
    else:
        ax_prof.set_ylabel("Flux Density")
    ax_pa.set_ylabel("PPA [deg]")

    fig.align_ylabels()

    logger.info(f"Saving plot file: {savename}.png")
    fig.savefig(savename + ".png")

    if save_pdf:
        logger.info(f"Saving plot file: {savename}.pdf")
        fig.savefig(savename + ".pdf")

    plt.close()

    if save_data:
        header1 = f"source: {cube.source}"
        header2 = f"rmsf_fwhm: {rmsf_fwhm}"
        if rm_prof_qty is None:
            header3 = "rm_prof: nan +/- nan"
        elif rm_prof_qty[1] is None:
            header3 = f"rm_prof: {rm_prof_qty[0]}"
        elif rm_prof_qty[1] < rmsf_fwhm:
            header3 = f"rm_prof: {rm_prof_qty[0]} +- {rm_prof_qty[1]}"
        else:
            header3 = "rm_prof: nan +- nan"
        header4 = f"sigma_i: {pol_profile.sigma_i}"
        header5 = "bin,I,Q,U,V,L,P0_L,P0_V,PA,PA_err,RM,RM_err,dvi,dvi_err"
        header = "\n".join([header1, header2, header3, header4, header5])
        prof_array = np.empty(shape=(14, cube.num_bin), dtype=np.float64)
        prof_array[0, :] = bins
        prof_array[1:5, :] = pol_profile.iquv
        prof_array[5, :] = pol_profile.l_true
        prof_array[6, :] = pol_profile.p0_l
        prof_array[7, :] = pol_profile.p0_v
        prof_array[8, :] = pol_profile.pa[0]
        prof_array[9, :] = pol_profile.pa[1]
        if plot_rm:
            prof_array[10, :] = np.where(rm_mask, rm_phi_qty[0], None)
            if rm_phi_qty[1] is not None:
                prof_array[11, :] = np.where(rm_mask, rm_phi_qty[1], None)
        if delta_vi is not None:
            prof_array[12, :] = delta_vi[0]
            prof_array[13, :] = delta_vi[1]
        logger.info(f"Saving data file: {savename}_data.csv")
        np.savetxt(f"{savename}_data.csv", prof_array.T, delimiter=",", header=header)


# TODO: Format docstring
def plot_freq_phase(
    cube: StokesCube,
    pol: int = 0,
    savename: str = "freq_phase",
    save_pdf: bool = False,
) -> None:
    """Create a plot of frequency vs phase for a specified polarisation.

    Parameters
    ----------
    cube : `StokesCube`
        A StokesCube object.
    pol : `int`, optional
        The polarisation index (0=I, 1=Q, 2=U, 3=V). Default: 0.
    savename : `str`, optional
        The name of the plot file excluding the extension.
        Default: 'freq_phase'.
    save_pdf : `bool`, optional
        Save the plot as a pdf? Default: `False`.
    """
    if pol not in [0, 1, 2, 3]:
        raise ValueError("pol must be an integer between 0 and 3 inclusive")

    fig, ax = plt.subplots(tight_layout=True)

    ax.imshow(
        cube.subbands[pol],
        extent=(0, 1, cube.min_freq, cube.max_freq),
        aspect="auto",
        cmap="cubehelix_r",
        interpolation="none",
    )
    ax.set_xlabel("Pulse Phase")
    ax.set_ylabel("Frequency [MHz]")

    logger.info(f"Saving plot file: {savename}.png")
    fig.savefig(savename + ".png")

    if save_pdf:
        logger.info(f"Saving plot file: {savename}.pdf")
        fig.savefig(savename + ".pdf")

    plt.close()


# TODO: Format docstring
def plot_time_phase(
    cube: StokesCube,
    pol: int = 0,
    savename: str = "time_phase",
    save_pdf: bool = False,
) -> None:
    """Create a plot of time vs phase for a specified polarisation.

    Parameters
    ----------
    cube : `StokesCube`
        A StokesCube object.
    pol : `int`, optional
        The polarisation index (0=I, 1=Q, 2=U, 3=V). Default: 0.
    savename : `str`, optional
        The name of the plot file excluding the extension.
        Default: 'time_phase'.
    save_pdf : `bool`, optional
        Save the plot as a pdf? Default: `False`.
    """
    if pol not in [0, 1, 2, 3]:
        raise ValueError("pol must be an integer between 0 and 3 inclusive")

    fig, ax = plt.subplots(tight_layout=True)

    ax.imshow(
        cube.subints[:, pol, :],
        extent=(0, 1, 0, cube.int_time),
        aspect="auto",
        cmap="cubehelix_r",
        interpolation="none",
    )
    ax.set_xlabel("Pulse Phase")
    ax.set_ylabel("Time [s]")

    logger.info(f"Saving plot file: {savename}.png")
    fig.savefig(savename + ".png")

    if save_pdf:
        logger.info(f"Saving plot file: {savename}.pdf")
        fig.savefig(savename + ".pdf")

    plt.close()


# TODO: Format docstring
def plot_2d_fdf(
    cube: StokesCube,
    fdf_amp_2D: NDArray,
    phi: NDArray,
    rmsf_fwhm: float,
    rm_phi_qty: tuple | None = None,
    rm_prof_qty: tuple | None = None,
    onpulse_pairs: list | None = None,
    rm_mask: NDArray | None = None,
    cln_comps: NDArray | None = None,
    plot_peaks: bool = False,
    plot_onpulse: bool = False,
    plot_pa: bool = False,
    phase_range: tuple[float, float] | None = None,
    phi_range: tuple[float, float] | None = None,
    p0_cutoff: float | None = 3.0,
    bin_func: Callable[[NDArray], NDArray] | None = None,
    savename: str = "fdf",
    save_pdf: bool = False,
    dark_mode: bool = False,
) -> None:
    """Plot the 1-D and 2-D FDF as a function of phase.

    Parameters
    ----------
    cube : `StokesCube`
        A StokesCube object.
    fdf_amp_2D: `NDArray`
        The amplitude of the FDF, with dimensions (phase, phi).
    phi : `NDArray`
        The Faraday depths (in rad/m^2) which the FDF is computed at.
    rmsf_fwhm : `float`
        The FWHM of the RM spread function.
    rm_phi_qty : `tuple[NDArray, NDArray]`, optional
        The RM measurements and uncertainties for each phase bin.
        Default: `None`.
    rm_prof_qty : `tuple[float, float]`, optional
        The RM measurement and uncertainty for the profile.
        Default: `None`.
    onpulse_pairs : `list`, optional
        An list of bin index pairs defining the onpulse region(s). If
        `None`, will use the
        full phase range. Default: `None`.
    rm_mask : `NDArray`, optional
        An array of booleans to act as a mask for the measured RM values.
        Default: `None`.
    cln_comps : `NDArray`, optional
        RM-CLEAN components to plot. Default: `None`.
    plot_peaks : `bool`, optional
        Plot the measure RM and error bars. Default: `False`.
    plot_onpulse : `bool`, optional
        Shade the on-pulse region.
    plot_pa : `bool`, optional
        Plot the position angle as a function of phase. Default: `False`.
    phase_range : `tuple[float, float]`, optional
        The phase range in rotations. Default: [0, 1].
    phi_range : `tuple[float, float]`, optional
        The Faraday depth range in rad/m^2. Default: full range.
    p0_cutoff : `float`, optional
        Mask all RM and PA measurements below this polarisation measure.
        If `None` is specified then no mask will be applied. Default: 3.0.
    bin_func : `Callable`, optional
        A function that maps the phase bins from [0,1] to anything.
        Default: `None`.
    savename : `str`, optional
        The name of the plot file excluding the extension. Default: 'fdf'.
    save_pdf : `bool`, optional
        Save the plot as a pdf? Default: `False`.
    dark_mode : `bool`, optional
        Use a black background and white lines. Default: `False`.
    """
    if rm_prof_qty is not None:
        logger.debug(f"Correcting Faraday rotation at RM={rm_prof_qty[0]}")
        cube.defaraday(rm_prof_qty[0])

    if onpulse_pairs is None:
        fdf_amp_1Dy = fdf_amp_2D.mean(0)
    else:
        onpulse_mask = get_profile_mask_from_pairs(cube.num_bin, onpulse_pairs)
        fdf_amp_1Dy = fdf_amp_2D[onpulse_mask].mean(0)

    # Styles
    lw = 0.6
    if dark_mode:
        plt.style.use("dark_background")
        line_col = "w"
        cmap_name = "arctic"
    else:
        line_col = "k"
        cmap_name = "arctic_r"

    # Define Figure and Axes
    if plot_pa:
        fig = plt.figure(figsize=(6, 6.5), layout="tight")
        gs = gridspec.GridSpec(
            ncols=2,
            nrows=3,
            figure=fig,
            height_ratios=(1, 2.5, 1),
            width_ratios=(3, 1),
            hspace=0,
            wspace=0,
        )
        ax_pa = fig.add_subplot(gs[2, 0])
    else:
        fig = plt.figure(figsize=(5.4, 4.95), layout="tight")
        gs = gridspec.GridSpec(
            ncols=2,
            nrows=2,
            figure=fig,
            height_ratios=(1.1, 2.5),
            width_ratios=(2.5, 1),
            hspace=0,
            wspace=0,
        )
        ax_pa = None
    ax_prof = fig.add_subplot(gs[0, 0])
    ax_fdf_1dy = fig.add_subplot(gs[1, 1])
    ax_fdf_2d = fig.add_subplot(gs[1, 0])

    # Plot pulse profile and PA
    bins, pol_profile = _add_pol_profile_to_axes(
        cube,
        ax_prof=ax_prof,
        ax_pa=ax_pa,
        normalise=False,
        p0_cutoff=p0_cutoff,
        bin_func=bin_func,
        lw=lw,
    )
    ax_prof.text(
        0.03,
        0.91,
        f"{jname_to_name(cube.source).replace('-', '$-$')}",
        horizontalalignment="left",
        verticalalignment="top",
        transform=ax_prof.transAxes,
    )
    if onpulse_pairs is not None and plot_onpulse:
        ylims = ax_prof.get_ylim()
        fill_args = dict(color="tab:blue", alpha=0.3, zorder=0)
        for op_pair in onpulse_pairs:
            op_l = op_pair[0] / (cube.num_bin - 1)
            op_r = op_pair[1] / (cube.num_bin - 1)
            if bin_func is not None:
                op_l = bin_func(op_l)
                op_r = bin_func(op_r)
            if op_l < op_r:
                ax_prof.fill_betweenx(ylims, op_l, op_r, **fill_args)
            else:
                ax_prof.fill_betweenx(ylims, op_l, bins[-1], **fill_args)
                ax_prof.fill_betweenx(ylims, bins[0], op_r, **fill_args)
            ax_prof.set_ylim(ylims)

    # Plot 1D Y FDF
    ax_fdf_1dy.plot(fdf_amp_1Dy, phi, color=line_col, linewidth=lw)
    xlims = ax_fdf_1dy.get_xlim()
    # Plot RM=0 + uncertainty region
    y1 = [0 - rmsf_fwhm / 2.0] * 2
    y2 = [0 + rmsf_fwhm / 2.0] * 2
    ax_fdf_1dy.fill_between(xlims, y1, y2, color=line_col, alpha=0.2, zorder=0)
    ax_fdf_1dy.axhline(y=0, linestyle="--", color=line_col, linewidth=lw, zorder=1)
    # Plot RM_profile + uncertainty region
    if rm_prof_qty is not None:
        if rm_prof_qty[1] is not None:
            y1 = [rm_prof_qty[0] - rm_prof_qty[1]] * 2
            y2 = [rm_prof_qty[0] + rm_prof_qty[1]] * 2
            ax_fdf_1dy.fill_between(xlims, y1, y2, color="tab:red", alpha=0.5, zorder=0)
            ax_fdf_1dy.axhline(
                y=rm_prof_qty[0],
                linestyle="--",
                color="tab:red",
                linewidth=lw,
                zorder=1,
            )
        else:
            ax_fdf_1dy.axhline(
                y=rm_prof_qty[0],
                linestyle="--",
                color="tab:red",
                linewidth=lw,
                zorder=1,
            )

    # Plot 2D FDF
    cmap = plt.get_cmap(f"cmr.{cmap_name}")
    ax_fdf_2d.imshow(
        np.transpose(fdf_amp_2D).astype(float),
        origin="lower",
        extent=(bins[0], bins[-1], phi[0], phi[-1]),
        aspect="auto",
        cmap=cmap,
        interpolation="none",
    )
    if plot_peaks and rm_phi_qty is not None:
        # Plot RM measurements
        if rm_mask is None:
            rm_mask = np.full(rm_phi_qty[0].shape[0], True)

        full_rm_mask = rm_mask & (pol_profile.p0_l > p0_cutoff)

        if rm_phi_qty[1] is not None:
            rm_phi_unc = np.abs(rm_phi_qty[1])[full_rm_mask]
            marker = "none"
        else:
            rm_phi_unc = None
            marker = "o"

        ax_fdf_2d.errorbar(
            x=bins[full_rm_mask],
            y=rm_phi_qty[0][full_rm_mask],
            yerr=rm_phi_unc,
            color="tab:red",
            marker=marker,
            ms=1,
            linestyle="none",
            elinewidth=lw,
            capthick=lw,
            capsize=0,
        )
    if cln_comps is not None:
        for bin_idx in range(bins.size):
            cln_comps_bin = np.abs(cln_comps[bin_idx])
            fd_bin_idx = np.arange(cln_comps_bin.size, dtype=int)
            fd_bin_idx_nonzero = fd_bin_idx[cln_comps_bin > 0.0]
            ax_fdf_2d.errorbar(
                x=[bins[bin_idx]] * len(fd_bin_idx_nonzero),
                y=phi[fd_bin_idx_nonzero],
                color="tab:red",
                marker="o",
                mec="none",
                ms=1.2,
                linestyle="none",
            )

    # Phase limits
    if phase_range is None:
        phase_range = (0, 1)

    # Convert phase units
    if bin_func is not None:
        phase_range = (bin_func(phase_range[0]), bin_func(phase_range[1]))

    for iax in [ax_fdf_2d, ax_pa, ax_prof]:
        if iax is not None:
            iax.set_xlim(phase_range)

    # Faraday depth limits
    if phi_range is None:
        phi_range = [phi[0], phi[-1]]
    for iax in [ax_fdf_1dy, ax_fdf_2d]:
        iax.set_ylim(phi_range)

    # Other limits
    ax_fdf_1dy.set_xlim(xlims)
    if plot_pa:
        ax_pa.set_ylim([-120, 120])

    # Tick locations
    ax_fdf_1dy.set_xticks([])
    ax_prof.set_yticks([])
    if plot_pa:
        ax_pa.set_yticks([-90, 0, 90])
        ax_pa.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(15))

    # Tick labels
    ax_fdf_1dy.set_yticklabels([])
    ax_prof.set_xticklabels([])
    if plot_pa:
        ax_fdf_2d.set_xticklabels([])

    # Tick configuration
    ax_fdf_2d.tick_params(which="both", right=True, top=True)
    if plot_pa:
        ax_pa.tick_params(which="both", right=True, top=True)
    for iax in [ax_prof, ax_fdf_1dy, ax_fdf_2d, ax_pa]:
        if iax is not None:
            iax.minorticks_on()
            iax.tick_params(axis="both", which="both", direction="in")
            iax.tick_params(axis="both", which="major", length=4)
            iax.tick_params(axis="both", which="minor", length=2)

    # Labels
    if bin_func is not None:
        # Assume degrees
        xlab = "Pulse Longitude [deg]"
    else:
        xlab = "Pulse Phase"
    if plot_pa:
        ax_pa.set_xlabel(xlab)
        ax_pa.set_ylabel("PPA [deg]")
        fig.align_ylabels([ax_fdf_2d, ax_pa])
    else:
        ax_fdf_2d.set_xlabel(xlab)
    ax_fdf_2d.set_ylabel("Faraday Depth, $\phi$ [$\mathrm{rad}\,\mathrm{m}^{-2}$]")

    logger.info(f"Saving plot file: {savename}.png")
    fig.savefig(savename + ".png")

    if save_pdf:
        logger.info(f"Saving plot file: {savename}.pdf")
        fig.savefig(savename + ".pdf")

    plt.close()


# TODO: Format docstring
def plot_rm_hist(
    samples: NDArray,
    valid_samples: NDArray | None = None,
    range: tuple[float, float] | None = None,
    title: str | None = None,
    savename: str = "rm_hist",
    save_pdf: bool = False,
) -> None:
    """Plot a histogram of RM samples. If 'valid_samples' are provided,
    then plot them in an inset. If 'range' is also provided, then indicate
    this range on the primary plot using dashed lines.

    Parameters
    ----------
    samples : `NDArray`
        The RM samples to generate a histogram for.
    valid_samples : `NDArray`, optional
        A subset of the RM samples to generate a histogram for.
        Default: `None`.
    range : `tuple[float, float]`, optional
        A range to indicate on the primary plot. Default: `None`.
    title : `str`, optional
        A title to add to the figure. Default: `None`.
    savename : `str`, optional
        The name of the plot file excluding the extension.
        Default: 'rm_hist'.
    save_pdf : `bool`, optional
        Save the plot as a pdf? Default: `False`.
    """
    fig, ax = plt.subplots(figsize=(4.5, 4), tight_layout=True)

    samples = samples[~np.isnan(samples)]
    hist(samples, bins="knuth", ax=ax, histtype="stepfilled", density=True)
    main_ax = ax
    main_samples = samples

    if valid_samples is not None:
        valid_samples = valid_samples[~np.isnan(valid_samples)]
        ax_ins = ax.inset_axes([0.1, 0.6, 0.3, 0.3])
        main_ax = ax_ins
        main_samples = valid_samples
        hist(
            valid_samples, bins="knuth", ax=ax_ins, histtype="stepfilled", density=True
        )
        ax_ins.minorticks_on()

        if range is not None:
            ax.axvline(range[0], linestyle="--", linewidth=1, c="tab:red")
            ax.axvline(range[1], linestyle="--", linewidth=1, c="tab:red")

    # Plot best-fit Gaussian distribution
    xlims = main_ax.get_xlim()
    gauss_x = np.linspace(*xlims, 1000)
    main_ax.plot(
        gauss_x,
        st.norm(np.mean(main_samples), np.std(main_samples)).pdf(gauss_x),
        color="k",
        lw=1,
        zorder=1,
    )
    main_ax.set_xlim(xlims)

    ax.minorticks_on()
    ax.set_ylabel("Probability Density")
    ax.set_xlabel("$\mathrm{RM}_\mathrm{profile}$ [$\mathrm{rad}\,\mathrm{m}^{-2}$]")
    if title is not None:
        ax.set_title(title)

    logger.info(f"Saving plot file: {savename}.png")
    fig.savefig(savename + ".png")

    if save_pdf:
        logger.info(f"Saving plot file: {savename}.pdf")
        fig.savefig(savename + ".pdf")

    plt.close()


# TODO: Format docstring
def plot_rm_vs_phi(
    rm_phi_samples: NDArray, savename: str = "rm_phi", save_pdf: bool = False
) -> None:
    """Plot boxplots showing the distribution of RM samples for each phase
    bin.

    Parameters
    ----------
    rm_phi_samples : `NDArray`
        A 2-D array used to make a boxplot.
    savename : `str`, optional
        The name of the plot file excluding the extension.
        Default: 'rm_phi'.
    save_pdf : `bool`, optional
        Save the plot as a pdf? Default: `False`.
    """
    fig, ax = plt.subplots(tight_layout=True)
    ax.boxplot(rm_phi_samples.T, showfliers=False)

    xlims = ax.get_xlim()
    ax.set_xticks(np.linspace(xlims[0], xlims[1], 5))
    ax.set_xticklabels(np.arange(5) / 4)

    ax.set_ylabel("RM [$\mathrm{rad}\,\mathrm{m}^{-2}$]")
    ax.set_xlabel("Pulse Phase")

    logger.info(f"Saving plot file: {savename}.png")
    fig.savefig(savename + ".png")

    if save_pdf:
        logger.info(f"Saving plot file: {savename}.pdf")
        fig.savefig(savename + ".pdf")

    plt.close()
