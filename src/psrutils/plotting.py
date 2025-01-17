import logging
from typing import Tuple

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
from astropy.visualization import hist

import psrutils

__all__ = [
    "plot_profile",
    "plot_freq_phase",
    "plot_time_phase",
    "plot_2d_fdf",
    "plot_rm_hist",
    "plot_rm_vs_phi",
]


plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "cm"
plt.rcParams["font.size"] = 12


def plot_profile(
    cube: psrutils.StokesCube,
    pol: int = 0,
    savename: str = "profile",
    save_pdf: bool = False,
    logger: logging.Logger = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Create a plot of integrated flux density vs phase for a specified polarisation.

    Parameters
    ----------
    cube : `psrutils.StokesCube`
        A StokesCube object.
    pol : `int`, optional
        The polarisation index (0=I, 1=Q, 2=U, 3=V). Default: 0.
    savename : `str`, optional
        The name of the plot file excluding the extension. Default: 'profile'.
    save_pdf : `bool`, optional
        Save the plot as a pdf? Default: False.
    logger : logging.Logger, optional
        A logger to use. Default: None.

    Returns
    -------
    fig : `matplotlib.figure.Figure`
        A Figure object.
    ax : `matplotlib.axes.Axes`
        An Axes object.
    """
    if logger is None:
        logger = psrutils.get_logger()

    if pol not in [0, 1, 2, 3]:
        raise ValueError("pol must be an integer between 0 and 3 inclusive")

    fig, ax = plt.subplots(dpi=300, tight_layout=True)

    edges = np.arange(cube.num_bin + 1) / cube.num_bin
    ax.stairs(cube.profile, edges, color="k")
    ax.set_xlim([edges[0], edges[-1]])
    ax.set_xlabel("Pulse Phase [rot]")
    ax.set_ylabel("Flux Density [arb.]")

    logger.info(f"Saving plot file: {savename}.png")
    fig.savefig(savename + ".png")

    if save_pdf:
        logger.info(f"Saving plot file: {savename}.pdf")
        fig.savefig(savename + ".pdf")

    return fig, ax


def plot_freq_phase(
    cube: psrutils.StokesCube,
    pol: int = 0,
    savename: str = "freq_phase",
    save_pdf: bool = False,
    logger: logging.Logger = None,
) -> Tuple[plt.Figure, plt.Axes]:
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
        Save the plot as a pdf? Default: False.
    logger : logging.Logger, optional
        A logger to use. Default: None.

    Returns
    -------
    fig : `matplotlib.figure.Figure`
        A Figure object.
    ax : `matplotlib.axes.Axes`
        An Axes object.
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
    ax.set_xlabel("Pulse Phase [rot]")
    ax.set_ylabel("Frequency [MHz]")

    logger.info(f"Saving plot file: {savename}.png")
    fig.savefig(savename + ".png")

    if save_pdf:
        logger.info(f"Saving plot file: {savename}.pdf")
        fig.savefig(savename + ".pdf")

    return fig, ax


def plot_time_phase(
    cube: psrutils.StokesCube,
    pol: int = 0,
    savename: str = "time_phase",
    save_pdf: bool = False,
    logger: logging.Logger = None,
) -> Tuple[plt.Figure, plt.Axes]:
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
        Save the plot as a pdf? Default: False.
    logger : logging.Logger, optional
        A logger to use. Default: None.

    Returns
    -------
    fig : `matplotlib.figure.Figure`
        A Figure object.
    ax : `matplotlib.axes.Axes`
        An Axes object.
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
    ax.set_xlabel("Pulse Phase [rot]")
    ax.set_ylabel("Time [s]")

    logger.info(f"Saving plot file: {savename}.png")
    fig.savefig(savename + ".png")

    if save_pdf:
        logger.info(f"Saving plot file: {savename}.pdf")
        fig.savefig(savename + ".pdf")

    return fig, ax


def plot_2d_fdf(
    cube: psrutils.StokesCube,
    fdf_amp_2D: np.ndarray,
    phi: np.ndarray,
    rmsf_fwhm: float,
    rm_phi_meas: tuple,
    rm_prof_meas: tuple,
    mask: np.ndarray = None,
    plot_stairs: bool = False,
    plot_peaks: bool = False,
    phase_range: Tuple[float, float] = None,
    phi_range: Tuple[float, float] = None,
    savename: str = "fdf",
    save_pdf: bool = False,
    logger: logging.Logger = None,
) -> plt.Figure:
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
    rm_phi_meas : `Tuple[np.ndarray, np.ndarray]`
        The RM measurements and uncertainties for each phase bin.
    rm_prof_meas : `Tuple[float, float]`
        The RM measurement and uncertainty for the profile.
    mask : `np.ndarray`, optional
        An array of booleans to act as a mask for the measured RM values.
    plot_stairs : `bool`, optional
        Plot the profile bins as stairs. Default: False.
    plot_peaks : `bool`, optional
        Plot the measure RM and error bars. Default: False.
    phase_range : `Tuple[float, float]`, optional
        The phase range in rotations. Default: [0, 1].
    phi_range : `Tuple[float, float]`, optional
        The Faraday depth range in rad/m^2. Default: computed range.
    savename : `str`, optional
        The name of the plot file excluding the extension. Default: 'fdf'.
    save_pdf : `bool`, optional
        Save the plot as a pdf? Default: False.
    logger : logging.Logger, optional
        A logger to use. Default: None.

    Returns
    -------
    fig : `matplotlib.figure.Figure`
        A Figure object.
    """
    if logger is None:
        logger = psrutils.get_logger()

    tmp_archive = cube.archive_clone
    tmp_archive.set_rotation_measure(rm_prof_meas[0])
    tmp_archive.fscrunch()
    tmp_archive.tscrunch()
    iquv_profile = tmp_archive.get_data()[0, :, 0, :]
    l_profile = np.sqrt(iquv_profile[1] ** 2 + iquv_profile[2] ** 2)

    fdf_amp_1D = fdf_amp_2D.mean(0)

    # Define Figure and Axes
    fig = plt.figure(figsize=(5.5, 5), tight_layout=True, dpi=300)
    gs = gridspec.GridSpec(
        ncols=2,
        nrows=2,
        figure=fig,
        height_ratios=(1, 3),
        width_ratios=(3.5, 1),
        hspace=0,
        wspace=0,
    )
    ax_prof = fig.add_subplot(gs[0, 0])
    ax_2dfdf = fig.add_subplot(gs[1, 0])
    ax_1dfdf = fig.add_subplot(gs[1, 1])

    # Default plot limits
    if phase_range is None:
        phase_range = [0, 1]
    if phi_range is None:
        phi_range = [phi[0], phi[-1]]

    # Plot profile
    if plot_stairs:
        bin_edges = np.arange(cube.num_bin + 1) / cube.num_bin
        ax_prof.stairs(iquv_profile[0], bin_edges, color="k", zorder=10)
        ax_prof.stairs(l_profile, bin_edges, color="tab:red", zorder=9)
        ax_prof.stairs(iquv_profile[3], bin_edges, color="tab:blue", zorder=8)
    else:
        bin_centres = np.arange(cube.num_bin) / (cube.num_bin - 1)
        ax_prof.plot(bin_centres, iquv_profile[0], linewidth=1, color="k", zorder=10)
        ax_prof.plot(bin_centres, l_profile, linewidth=1, color="tab:red", zorder=9)
        ax_prof.plot(bin_centres, iquv_profile[3], linewidth=1, color="tab:blue", zorder=8)
    ax_prof.set_xlim(phase_range)
    ax_prof.set_yticks([])
    ax_prof.set_xticks([])
    ax_prof.text(
        0.028,
        0.89,
        f"{tmp_archive.get_source()}",
        horizontalalignment="left",
        verticalalignment="top",
        transform=ax_prof.transAxes,
    )

    # Plot 1D FDF
    ax_1dfdf.plot(fdf_amp_1D, phi, color="k", linewidth=1)
    xlims = ax_1dfdf.get_xlim()
    # Plot RM=0 + uncertainty region
    y1 = [0 - rmsf_fwhm / 2.0] * 2
    y2 = [0 + rmsf_fwhm / 2.0] * 2
    ax_1dfdf.fill_between(xlims, y1, y2, color="k", alpha=0.3, zorder=0)
    ax_1dfdf.axhline(y=0, linestyle="--", color="k", linewidth=1, zorder=1)
    # Plot RM_profile + uncertainty region
    y1 = [rm_prof_meas[0] - rm_prof_meas[1]] * 2
    y2 = [rm_prof_meas[0] + rm_prof_meas[1]] * 2
    ax_1dfdf.fill_between(xlims, y1, y2, color="tab:red", alpha=0.6, zorder=0)
    # ax_1dfdf.axhline(y=rm, linestyle="-", color="tab:red", zorder=10)
    ax_1dfdf.set_ylim(phi_range)
    ax_1dfdf.set_xlim(xlims)
    ax_1dfdf.set_yticks([])
    ax_1dfdf.set_xticks([])

    # Plot 2D FDF
    ax_2dfdf.imshow(
        np.flipud(fdf_amp_2D.transpose()).astype(float),
        extent=(0, 1, phi[0], phi[-1]),
        aspect="auto",
        cmap="cubehelix_r",
        interpolation="none",
    )
    if plot_peaks:
        # Plot RM measurements
        rm_bin = rm_phi_meas[0]
        rm_bin_unc = rm_phi_meas[1]
        bin_centres = np.arange(rm_bin.shape[0]) / (rm_bin.shape[0] - 1)
        if mask is None:
            mask = np.full(rm_bin.shape[0], True)
        ax_2dfdf.errorbar(
            x=bin_centres[mask],
            y=rm_bin[mask],
            yerr=np.abs(rm_bin_unc)[mask],
            color="r",
            marker="none",
            linestyle="none",
            elinewidth=0.6,
            capthick=0.6,
            capsize=0,
        )
    ax_2dfdf.set_xlabel("Pulse Phase [rot]")
    ax_2dfdf.set_ylabel("$\phi$ [$\mathrm{rad}\,\mathrm{m}^{-2}$]")
    ax_2dfdf.set_xlim(phase_range)
    ax_2dfdf.set_ylim(phi_range)
    ax_2dfdf.minorticks_on()

    logger.info(f"Saving plot file: {savename}.png")
    fig.savefig(savename + ".png")

    if save_pdf:
        logger.info(f"Saving plot file: {savename}.pdf")
        fig.savefig(savename + ".pdf")

    return fig


def plot_rm_hist(
    samples: np.ndarray,
    valid_samples: np.ndarray = None,
    range: Tuple[float, float] = None,
    title: str = None,
    savename: str = "rm_hist",
    save_pdf: bool = False,
    logger: logging.Logger = None,
):
    """Plot a histogram of RM samples. If 'valid_samples' are provided, then
    plot them in an inset. If 'range' is also provided, then indicate this range
    on the primary plot using dashed lines.

    Parameters
    ----------
    samples : `np.ndarray`
        The RM samples to generate a histogram for.
    valid_samples : `np.ndarray`, optional
        A subset of the RM samples to generate a histogram for. Default: None.
    range : `Tuple[float, float]`, optional
        A range to indicate on the primary plot. Default: None.
    title : `str`, optional
        A title to add to the figure. Default: None.
    savename : `str`, optional
        The name of the plot file excluding the extension. Default: 'rm_hist'.
    save_pdf : `bool`, optional
        Save the plot as a pdf? Default: False.
    logger : logging.Logger, optional
        A logger to use. Default: None.

    Returns
    -------
    fig : `matplotlib.figure.Figure`
        A Figure object.
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

    return fig


def plot_rm_vs_phi(
    rm_phi_samples: np.ndarray,
    savename: str = "rm_phi",
    save_pdf: bool = False,
    logger: logging.Logger = None,
):
    """Plot boxplots showing the distribution of RM samples for each phase bin.

    Parameters
    ----------
    rm_phi_samples : `np.ndarray`
        A 2-D array used to make a boxplot.
    savename : `str`, optional
        The name of the plot file excluding the extension. Default: 'rm_phi'.
    save_pdf : `bool`, optional
        Save the plot as a pdf? Default: False.
    logger : logging.Logger, optional
        A logger to use. Default: None.

    Returns
    -------
    fig : `matplotlib.figure.Figure`
        A Figure object.
    """
    if logger is None:
        logger = psrutils.get_logger()

    fig, ax = plt.subplots(dpi=300, tight_layout=True)
    ax.boxplot(rm_phi_samples.T, showfliers=False)

    xlims = ax.get_xlim()
    ax.set_xticks(np.linspace(xlims[0], xlims[1], 5))
    ax.set_xticklabels(np.arange(5) / 4)

    ax.set_ylabel("RM [$\mathrm{rad}\,\mathrm{m}^{-2}$]")
    ax.set_xlabel("Pulse Phase [rot]")

    logger.info(f"Saving plot file: {savename}.png")
    fig.savefig(savename + ".png")

    if save_pdf:
        logger.info(f"Saving plot file: {savename}.pdf")
        fig.savefig(savename + ".pdf")

    return fig
