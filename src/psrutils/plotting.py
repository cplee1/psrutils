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

import psrutils

__all__ = [
    "add_profile_to_axes",
    "plot_profile",
    "plot_pol_profile",
    "plot_freq_phase",
    "plot_time_phase",
    "plot_2d_fdf",
    "plot_rm_hist",
    "plot_rm_vs_phi",
]


def add_profile_to_axes(
    cube: psrutils.StokesCube,
    ax_pa: Axes | None = None,
    ax_prof: Axes | None = None,
    normalise_flux: bool = True,
    plot_pol: bool = True,
    p0_cutoff: float | None = 3.0,
    voffset: float = 0.0,
    lw: float = 0.55,
    I_colour: str = "k",
    L_colour: str = "tab:red",
    V_colour: str = "tab:blue",
    PA_colour: str = "k",
    alpha: float = 1.0,
    bin_func: Callable[[np.ndarray], np.ndarray] | None = None,
    label: str = None,
    logger: logging.Logger | None = None,
) -> tuple[np.ndarray, tuple]:
    if plot_pol:
        profile_data = psrutils.get_bias_corrected_pol_profile(cube, logger=logger)
        iquv_prof, l_prof, pa, p0_l, p0_v, sigma_i = profile_data

        if l_prof is None:
            plot_pol = False
    else:
        iquv_prof = cube.pol_profile
        l_prof, pa, p0_l, p0_v, sigma_i = None, None, None, None, None

    bins = np.linspace(0, 1, cube.num_bin)

    # Transforms the bin coordinates from [0,1] -> anything
    if bin_func is not None:
        bins = bin_func(bins)

    if normalise_flux:
        peak_flux = np.max(iquv_prof[0])
        iquv_prof /= peak_flux
        if plot_pol:
            l_prof /= peak_flux
            sigma_i /= peak_flux

    if ax_prof is not None:
        ax_prof.plot(
            bins,
            iquv_prof[0] + voffset,
            linewidth=lw,
            linestyle="-",
            color=I_colour,
            alpha=alpha,
            zorder=8,
        )
        if plot_pol:
            ax_prof.plot(
                bins,
                l_prof + voffset,
                linewidth=lw,
                linestyle="--",
                color=L_colour,
                alpha=alpha,
                zorder=10,
            )
            ax_prof.plot(
                bins,
                iquv_prof[3] + voffset,
                linewidth=lw,
                linestyle=":",
                color=V_colour,
                alpha=alpha,
                zorder=9,
            )

    if ax_pa is not None and plot_pol:
        pa_mask = p0_l > p0_cutoff
        for offset in [0, -180, 180]:
            if offset == 0:
                ilabel = label
            else:
                ilabel = None
            ax_pa.errorbar(
                x=bins[pa_mask],
                y=pa[0, pa_mask] + offset,
                yerr=pa[1, pa_mask],
                ecolor=PA_colour,
                marker="none",
                ms=1,
                linestyle="none",
                elinewidth=lw,
                capthick=lw,
                capsize=0,
                label=ilabel,
            )
    return bins, (iquv_prof, l_prof, pa, p0_l, p0_v, sigma_i)


def plot_profile(
    cube: psrutils.StokesCube,
    pol: int = 0,
    offpulse_win: np.ndarray | None = None,
    onpulse_win: np.ndarray | None = None,
    savename: str = "profile",
    save_pdf: bool = False,
    logger: logging.Logger | None = None,
) -> None:
    """Create a plot of integrated flux density vs phase for a specified polarisation.

    Parameters
    ----------
    cube : `psrutils.StokesCube`
        A StokesCube object.
    pol : `int`, optional
        The polarisation index (0=I, 1=Q, 2=U, 3=V). Default: 0.
    offpulse_win : `np.ndarray`, optional
        The bin indices of the offpulse window. Default: `None`.
    onpulse_win : `np.ndarray`, optional
        The bin indices of the onpulse window. Default: `None`.
    savename : `str`, optional
        The name of the plot file excluding the extension. Default: 'profile'.
    save_pdf : `bool`, optional
        Save the plot as a pdf? Default: `False`.
    logger : logging.Logger, optional
        A logger to use. Default: `None`.
    """
    if logger is None:
        logger = psrutils.get_logger()

    if pol not in [0, 1, 2, 3]:
        raise ValueError("pol must be an integer between 0 and 3 inclusive")

    fig, ax = plt.subplots(dpi=300, tight_layout=True)

    bins = np.arange(cube.num_bin) / cube.num_bin
    ax.plot(bins, cube.profile, color="k", linewidth=0.8)
    xlims = [bins[0], bins[-1]]
    ylims = ax.get_ylim()

    if offpulse_win is not None:
        offpulse_win = offpulse_win.astype(float) / (cube.num_bin - 1)
        if offpulse_win[0] < offpulse_win[-1]:
            ax.fill_betweenx(
                ylims, offpulse_win[0], offpulse_win[-1], color="tab:red", alpha=0.4, zorder=0
            )
        else:
            ax.fill_betweenx(
                ylims, offpulse_win[0], xlims[-1], color="tab:red", alpha=0.4, zorder=0
            )
            ax.fill_betweenx(
                ylims, xlims[0], offpulse_win[-1], color="tab:red", alpha=0.4, zorder=0
            )

    if onpulse_win is not None:
        onpulse_win = onpulse_win.astype(float) / (cube.num_bin - 1)
        if onpulse_win[0] < onpulse_win[-1]:
            ax.fill_betweenx(
                ylims, onpulse_win[0], onpulse_win[-1], color="tab:blue", alpha=0.4, zorder=0
            )
        else:
            ax.fill_betweenx(
                ylims, onpulse_win[0], xlims[-1], color="tab:blue", alpha=0.4, zorder=0
            )
            ax.fill_betweenx(
                ylims, xlims[0], onpulse_win[-1], color="tab:blue", alpha=0.4, zorder=0
            )

    # Plot the noise floor
    ax.axhline(0, linestyle="--", linewidth=0.8, color="k")

    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.set_xlabel("Pulse Phase")
    ax.set_ylabel("Flux Density [arb. units]")

    logger.info(f"Saving plot file: {savename}.png")
    fig.savefig(savename + ".png")

    if save_pdf:
        logger.info(f"Saving plot file: {savename}.pdf")
        fig.savefig(savename + ".pdf")

    plt.close()


def plot_pol_profile(
    cube: psrutils.StokesCube,
    rmsf_fwhm: float | None = None,
    rm_phi_qty: tuple[np.ndarray, np.ndarray] | None = None,
    rm_prof_qty: tuple[float, float] | None = None,
    rm_mask: np.ndarray | None = None,
    delta_vi: np.ndarray | None = None,
    phase_range: tuple[float, float] | None = None,
    p0_cutoff: float | None = 3.0,
    savename: str = "pol_profile",
    save_pdf: bool = False,
    save_data: bool = False,
    logger: logging.Logger | None = None,
) -> None:
    """Create a plot of integrated flux density vs phase for a specified polarisation.

    Parameters
    ----------
    cube : `psrutils.StokesCube`
        A StokesCube object.
    rmsf_fwhm : `float`, optional
        The FWHM of the RM spread function. Default: `None`.
    rm_phi_qty : `tuple[np.ndarray, np.ndarray]`, optional
        The RM measurements and uncertainties for each phase bin. Default: `None`.
    rm_prof_qty : `tuple[float, float]`, optional
        The RM measurement and uncertainty for the profile. Default: `None`.
    rm_mask : `np.ndarray`, optional
        An array of booleans to act as a mask for the measured RM values.
        Default: `None`.
    delta_vi : `np.ndarray`, optional
        The change in Stokes V/I over the bandwidth per bin. Default: `None`.
    phase_range : `tuple[float, float]`, optional
        The phase range in rotations. Default: [0, 1].
    p0_cutoff : `float`, optional
        Mask all RM and PA measurements below this polarisation measure. If `None` is
        specified then no mask will be applied. Default: 3.0.
    savename : `str`, optional
        The name of the plot file excluding the extension. Default: 'pol_profile'.
    save_pdf : `bool`, optional
        Save the plot as a pdf? Default: `False`.
    save_data : `bool`, optional
        Save the plot data? Default: `False`.
    logger : logging.Logger, optional
        A logger to use. Default: `None`.
    """
    if logger is None:
        logger = psrutils.get_logger()

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
        fig = plt.figure(figsize=(6, 6.5), layout="tight", dpi=300)
        gs = gridspec.GridSpec(ncols=1, nrows=4, figure=fig, height_ratios=(1, 1, 1, 3), hspace=0)
        ax_rm = fig.add_subplot(gs[0])
        ax_dv = fig.add_subplot(gs[1])
        ax_pa = fig.add_subplot(gs[2])
        ax_prof = fig.add_subplot(gs[3])
    elif plot_rm:
        fig = plt.figure(figsize=(6, 5.6), layout="tight", dpi=300)
        gs = gridspec.GridSpec(ncols=1, nrows=3, figure=fig, height_ratios=(1, 1, 3), hspace=0)
        ax_rm = fig.add_subplot(gs[0])
        ax_dv = None
        ax_pa = fig.add_subplot(gs[1])
        ax_prof = fig.add_subplot(gs[2])
    else:
        fig = plt.figure(figsize=(5, 4), layout="tight", dpi=300)
        gs = gridspec.GridSpec(ncols=1, nrows=2, figure=fig, height_ratios=(1, 2), hspace=0)
        ax_rm = None
        ax_dv = None
        ax_pa = fig.add_subplot(gs[0])
        ax_prof = fig.add_subplot(gs[1])

    # Styles
    lw = 0.6
    line_col = "k"

    # Phase range to plot
    if phase_range is None:
        phase_range = [0, 1]

    # Add flux and PA
    bins, profile_data = psrutils.add_profile_to_axes(
        cube, ax_pa=ax_pa, ax_prof=ax_prof, p0_cutoff=p0_cutoff, lw=lw, logger=logger
    )
    iquv_prof, l_prof, pa_prof, p0_l, p0_v, sigma_i = profile_data

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

        full_rm_mask = rm_mask & (p0_l > p0_cutoff)

        if rm_phi_qty[1] is not None:
            rm_phi_unc = np.abs(rm_phi_qty[1])[full_rm_mask]
        else:
            rm_phi_unc = None
            scatter_params["marker"] = "o"

        ax_rm.errorbar(
            x=bins[full_rm_mask], y=rm_phi_qty[0][full_rm_mask], yerr=rm_phi_unc, **scatter_params
        )

        rm_lims = ax_rm.get_ylim()

        # Plot RM_profile + uncertainty region
        if valid_rm:
            if rm_prof_qty[1] is not None:
                y1 = [rm_prof_qty[0] - rm_prof_qty[1]] * 2
                y2 = [rm_prof_qty[0] + rm_prof_qty[1]] * 2
                ax_rm.fill_between(phase_range, y1, y2, color="tab:red", alpha=0.5, zorder=0)
            else:
                ax_rm.axhline(
                    y=rm_prof_qty[0], linestyle="--", color="tab:red", linewidth=lw, zorder=1
                )

            # Plot RM=0 + uncertainty region
            if rmsf_fwhm is not None:
                y1 = [0 - rmsf_fwhm / 2.0] * 2
                y2 = [0 + rmsf_fwhm / 2.0] * 2
                ax_rm.fill_between(phase_range, y1, y2, color=line_col, alpha=0.2, zorder=0)
                ax_rm.axhline(y=0, linestyle=":", color=line_col, linewidth=lw, zorder=1)

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
        f"{cube.source}",
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
    for iax in [ax_rm, ax_dv, ax_pa, ax_prof]:
        if iax is not None:
            iax.set_xlim(phase_range)

    # Y limits
    ax_pa.set_ylim([-120, 120])

    # Ticks
    if plot_rm:
        ax_rm.set_xticklabels([])
    if ax_dv is not None:
        ax_dv.set_xticklabels([])
    ax_pa.set_xticklabels([])
    ax_pa.set_yticks([-90, 0, 90])
    ax_pa.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(15))
    for iax in [ax_rm, ax_dv, ax_pa, ax_prof]:
        if iax is not None:
            iax.tick_params(which="both", right=True, top=True)
            iax.minorticks_on()
            iax.tick_params(axis="both", which="both", direction="in")
            iax.tick_params(axis="both", which="major", length=4)
            iax.tick_params(axis="both", which="minor", length=2)

    # Labels
    if plot_rm:
        ax_rm.set_ylabel("RM\n[$\mathrm{rad}\,\mathrm{m}^{-2}$]")
    if ax_dv is not None:
        ax_dv.set_ylabel("$\Delta(V/I)$")
    ax_prof.set_xlabel("Pulse Phase")
    ax_prof.set_ylabel("Flux Density\n[arbitrary units]")
    ax_pa.set_ylabel("P.A.\n[deg]")

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
        if rm_prof_qty is not None and rm_prof_qty[1] < rmsf_fwhm:
            header3 = f"rm_prof: {rm_prof_qty[0]} +- {rm_prof_qty[1]}"
        else:
            header3 = "rm_prof: nan +- nan"
        header4 = f"sigma_i: {sigma_i}"
        header5 = "bin,I,Q,U,V,L,P0_L,P0_V,PA,PA_err,RM,RM_err,dvi,dvi_err"
        header = "\n".join([header1, header2, header3, header4, header5])
        prof_array = np.empty(shape=(14, cube.num_bin), dtype=np.float64)
        prof_array[0, :] = bins
        prof_array[1:5, :] = iquv_prof
        prof_array[5, :] = l_prof
        prof_array[6, :] = p0_l
        prof_array[7, :] = p0_v
        prof_array[8, :] = pa_prof[0]
        prof_array[9, :] = pa_prof[1]
        if plot_rm:
            prof_array[10, :] = np.where(rm_mask, rm_phi_qty[0], None)
            if rm_phi_qty[1] is not None:
                prof_array[11, :] = np.where(rm_mask, rm_phi_qty[1], None)
        if delta_vi is not None:
            prof_array[12, :] = delta_vi[0]
            prof_array[13, :] = delta_vi[1]
        logger.info(f"Saving data file: {savename}_data.csv")
        np.savetxt(f"{savename}_data.csv", prof_array.T, delimiter=",", header=header)


def plot_freq_phase(
    cube: psrutils.StokesCube,
    pol: int = 0,
    savename: str = "freq_phase",
    save_pdf: bool = False,
    logger: logging.Logger | None = None,
) -> None:
    """Create a plot of frequency vs phase for a specified polarisation.

    Parameters
    ----------
    cube : `psrutils.StokesCube`
        A StokesCube object.
    pol : `int`, optional
        The polarisation index (0=I, 1=Q, 2=U, 3=V). Default: 0.
    savename : `str`, optional
        The name of the plot file excluding the extension. Default: 'freq_phase'.
    save_pdf : `bool`, optional
        Save the plot as a pdf? Default: `False`.
    logger : logging.Logger, optional
        A logger to use. Default: `None`.
    """
    if logger is None:
        logger = psrutils.get_logger()

    if pol not in [0, 1, 2, 3]:
        raise ValueError("pol must be an integer between 0 and 3 inclusive")

    fig, ax = plt.subplots(dpi=300, tight_layout=True)

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


def plot_time_phase(
    cube: psrutils.StokesCube,
    pol: int = 0,
    savename: str = "time_phase",
    save_pdf: bool = False,
    logger: logging.Logger | None = None,
) -> None:
    """Create a plot of time vs phase for a specified polarisation.

    Parameters
    ----------
    cube : `psrutils.StokesCube`
        A StokesCube object.
    pol : `int`, optional
        The polarisation index (0=I, 1=Q, 2=U, 3=V). Default: 0.
    savename : `str`, optional
        The name of the plot file excluding the extension. Default: 'time_phase'.
    save_pdf : `bool`, optional
        Save the plot as a pdf? Default: `False`.
    logger : logging.Logger, optional
        A logger to use. Default: `None`.
    """
    if logger is None:
        logger = psrutils.get_logger()

    if pol not in [0, 1, 2, 3]:
        raise ValueError("pol must be an integer between 0 and 3 inclusive")

    fig, ax = plt.subplots(dpi=300, tight_layout=True)

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


def plot_2d_fdf(
    cube: psrutils.StokesCube,
    fdf_amp_2D: np.ndarray,
    phi: np.ndarray,
    rmsf_fwhm: float,
    rm_phi_qty: tuple | None = None,
    rm_prof_qty: tuple | None = None,
    onpulse_win: np.ndarray | None = None,
    rm_mask: np.ndarray | None = None,
    cln_comps: np.ndarray | None = None,
    plot_peaks: bool = False,
    plot_onpulse: bool = False,
    plot_pa: bool = False,
    phase_range: tuple[float, float] | None = None,
    phi_range: tuple[float, float] | None = None,
    p0_cutoff: float | None = 3.0,
    bin_func: Callable[[np.ndarray], np.ndarray] | None = None,
    savename: str = "fdf",
    save_pdf: bool = False,
    dark_mode: bool = False,
    logger: logging.Logger | None = None,
) -> None:
    """Plot the 1-D and 2-D FDF as a function of phase.

    Parameters
    ----------
    cube : `psrutils.StokesCube`
        A StokesCube object.
    fdf_amp_2D: `np.ndarray`
        The amplitude of the FDF, with dimensions (phase, phi).
    phi : `np.ndarray`
        The Faraday depths (in rad/m^2) which the FDF is computed at.
    rmsf_fwhm : `float`
        The FWHM of the RM spread function.
    rm_phi_qty : `tuple[np.ndarray, np.ndarray]`, optional
        The RM measurements and uncertainties for each phase bin. Default: `None`.
    rm_prof_qty : `tuple[float, float]`, optional
        The RM measurement and uncertainty for the profile. Default: `None`.
    onpulse_win : `np.ndarray`, optional
        An array of bin indices defining the onpulse window. If `None`, will use the
        full phase range. Default: `None`.
    rm_mask : `np.ndarray`, optional
        An array of booleans to act as a mask for the measured RM values.
        Default: `None`.
    cln_comps : `np.ndarray`, optional
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
        Mask all RM and PA measurements below this polarisation measure. If `None` is
        specified then no mask will be applied. Default: 3.0.
    bin_func : `Callable`, optional
        A function that maps the phase bins from [0,1] to anything. Default: `None`.
    savename : `str`, optional
        The name of the plot file excluding the extension. Default: 'fdf'.
    save_pdf : `bool`, optional
        Save the plot as a pdf? Default: `False`.
    dark_mode : `bool`, optional
        Use a black background and white lines. Default: `False`.
    logger : logging.Logger, optional
        A logger to use. Default: `None`.
    """
    if logger is None:
        logger = psrutils.get_logger()

    if rm_prof_qty is not None:
        cube.defaraday(rm_prof_qty[0])

    # Will by default use 1/8 of the profile to find offpulse noise
    iquv_prof, l_prof, pa_prof, p0_l, _, _ = psrutils.get_bias_corrected_pol_profile(
        cube, logger=logger
    )

    bins = np.linspace(0, 1, cube.num_bin)

    # Transforms the bin coordinates from [0,1] -> anything
    if bin_func is not None:
        bins = bin_func(bins)

    if onpulse_win is None:
        fdf_amp_1Dy = fdf_amp_2D.mean(0)
    else:
        fdf_amp_1Dy = fdf_amp_2D[onpulse_win].mean(0)

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
        fig = plt.figure(figsize=(6, 6.5), layout="tight", dpi=300)
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
        fig = plt.figure(figsize=(5.4, 4.95), layout="tight", dpi=300)
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

    # Plot profile
    ax_prof.plot(bins, iquv_prof[0], linewidth=lw, linestyle="-", color=line_col, zorder=8)
    ax_prof.plot(bins, iquv_prof[3], linewidth=lw, linestyle=":", color="tab:blue", zorder=9)
    ax_prof.plot(bins, l_prof, linewidth=lw, linestyle="--", color="tab:red", zorder=10)
    ax_prof.text(
        0.03,
        0.91,
        f"{cube.source.replace('-', '$-$')}",
        horizontalalignment="left",
        verticalalignment="top",
        transform=ax_prof.transAxes,
    )
    # ax_prof.text(
    #     0.97,
    #     0.91,
    #     f"{cube.ctr_freq:.0f} MHz",
    #     horizontalalignment="right",
    #     verticalalignment="top",
    #     transform=ax_prof.transAxes,
    # )
    if onpulse_win is not None and plot_onpulse:
        ylims = ax_prof.get_ylim()
        onpulse_win = onpulse_win.astype(float) / (cube.num_bin - 1)
        if bin_func is not None:
            onpulse_win = bin_func(onpulse_win)
        if onpulse_win[0] < onpulse_win[-1]:
            ax_prof.fill_betweenx(
                ylims, onpulse_win[0], onpulse_win[-1], color="tab:blue", alpha=0.3, zorder=0
            )
        else:
            ax_prof.fill_betweenx(
                ylims, onpulse_win[0], bins[-1], color="tab:blue", alpha=0.2, zorder=0
            )
            ax_prof.fill_betweenx(
                ylims, bins[0], onpulse_win[-1], color="tab:blue", alpha=0.2, zorder=0
            )
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
                y=rm_prof_qty[0], linestyle="--", color="tab:red", linewidth=lw, zorder=1
            )
        else:
            ax_fdf_1dy.axhline(
                y=rm_prof_qty[0], linestyle="--", color="tab:red", linewidth=lw, zorder=1
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

        full_rm_mask = rm_mask & (p0_l > p0_cutoff)

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

    # Position angle vs phase
    if plot_pa:
        pa_mask = p0_l > p0_cutoff
        for offset in [0, -180, 180]:
            ax_pa.errorbar(
                x=bins[pa_mask],
                y=pa_prof[0, pa_mask] + offset,
                yerr=pa_prof[1, pa_mask],
                color="k",
                marker="none",
                ms=1,
                linestyle="none",
                elinewidth=lw,
                capthick=lw,
                capsize=0,
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
        ax_pa.set_ylabel("P.A. [deg]")
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


def plot_rm_hist(
    samples: np.ndarray,
    valid_samples: np.ndarray | None = None,
    range: tuple[float, float] | None = None,
    title: str | None = None,
    savename: str = "rm_hist",
    save_pdf: bool = False,
    logger: logging.Logger | None = None,
) -> None:
    """Plot a histogram of RM samples. If 'valid_samples' are provided, then
    plot them in an inset. If 'range' is also provided, then indicate this range
    on the primary plot using dashed lines.

    Parameters
    ----------
    samples : `np.ndarray`
        The RM samples to generate a histogram for.
    valid_samples : `np.ndarray`, optional
        A subset of the RM samples to generate a histogram for. Default: `None`.
    range : `tuple[float, float]`, optional
        A range to indicate on the primary plot. Default: `None`.
    title : `str`, optional
        A title to add to the figure. Default: `None`.
    savename : `str`, optional
        The name of the plot file excluding the extension. Default: 'rm_hist'.
    save_pdf : `bool`, optional
        Save the plot as a pdf? Default: `False`.
    logger : logging.Logger, optional
        A logger to use. Default: `None`.
    """
    if logger is None:
        logger = psrutils.get_logger()

    fig, ax = plt.subplots(figsize=(4.5, 4), dpi=300, tight_layout=True)
    hist(samples, bins="knuth", ax=ax, histtype="stepfilled", density=True)
    main_ax = ax
    main_samples = samples

    if valid_samples is not None:
        ax_ins = ax.inset_axes([0.1, 0.6, 0.3, 0.3])
        main_ax = ax_ins
        main_samples = valid_samples
        hist(valid_samples, bins="knuth", ax=ax_ins, histtype="stepfilled", density=True)
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


def plot_rm_vs_phi(
    rm_phi_samples: np.ndarray,
    savename: str = "rm_phi",
    save_pdf: bool = False,
    logger: logging.Logger | None = None,
) -> None:
    """Plot boxplots showing the distribution of RM samples for each phase bin.

    Parameters
    ----------
    rm_phi_samples : `np.ndarray`
        A 2-D array used to make a boxplot.
    savename : `str`, optional
        The name of the plot file excluding the extension. Default: 'rm_phi'.
    save_pdf : `bool`, optional
        Save the plot as a pdf? Default: `False`.
    logger : logging.Logger, optional
        A logger to use. Default: `None`.
    """
    if logger is None:
        logger = psrutils.get_logger()

    fig, ax = plt.subplots(dpi=300, tight_layout=True)
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
