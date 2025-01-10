import logging
from typing import Tuple

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
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
    cube: psrutils.StokesCube, pol: int = 0, savename: str = "profile.png"
) -> Tuple[plt.Figure, plt.Axes]:
    """Create a plot of integrated flux density vs phase for a specified polarisation.

    Parameters
    ----------
    cube : `psrutils.StokesCube`
        A StokesCube object.
    pol : `int`, optional
        The polarisation index (0=I, 1=Q, 2=U, 3=V). Default: 0.
    savename : `str`, optional
        The name of the plot file. Default: 'profile.png'.

    Returns
    -------
    fig : `matplotlib.figure.Figure`
        A Figure object.
    ax : `matplotlib.axes.Axes`
        An Axes object.
    """
    if pol not in [0, 1, 2, 3]:
        raise ValueError("pol must be an integer between 0 and 3 inclusive")

    fig, ax = plt.subplots(dpi=300, tight_layout=True)

    edges = np.arange(cube.num_bin + 1) / cube.num_bin
    ax.stairs(cube.profile, edges, color="k")
    ax.set_xlim([edges[0], edges[-1]])
    ax.set_xlabel("Pulse Phase [rot]")
    ax.set_ylabel("Flux Density [arb.]")

    fig.savefig(savename)
    return fig, ax


def plot_freq_phase(
    cube: psrutils.StokesCube, pol: int = 0, savename: str = "freq_phase.png"
) -> Tuple[plt.Figure, plt.Axes]:
    """Create a plot of frequency vs phase for a specified polarisation.

    Parameters
    ----------
    cube : `psrutils.StokesCube`
        A StokesCube object.
    pol : `int`, optional
        The polarisation index (0=I, 1=Q, 2=U, 3=V). Default: 0.
    savename : `str`, optional
        The name of the plot file. Default: 'freq_phase.png'.

    Returns
    -------
    fig : `matplotlib.figure.Figure`
        A Figure object.
    ax : `matplotlib.axes.Axes`
        An Axes object.
    """
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

    fig.savefig(savename)
    return fig, ax


def plot_time_phase(
    cube: psrutils.StokesCube, pol: int = 0, savename: str = "time_phase.png"
) -> Tuple[plt.Figure, plt.Axes]:
    """Create a plot of time vs phase for a specified polarisation.

    Parameters
    ----------
    cube : `psrutils.StokesCube`
        A StokesCube object.
    pol : `int`, optional
        The polarisation index (0=I, 1=Q, 2=U, 3=V). Default: 0.
    savename : `str`, optional
        The name of the plot file. Default: 'folded_spectrum.png'.

    Returns
    -------
    fig : `matplotlib.figure.Figure`
        A Figure object.
    ax : `matplotlib.axes.Axes`
        An Axes object.
    """
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

    fig.savefig(savename)
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
    savename: str = "fdf.png",
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
        The name of the plot file. Default: 'fdf.png'.
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
    fig = plt.figure(figsize=(7, 6), tight_layout=True, dpi=300)
    gs = gridspec.GridSpec(
        ncols=2,
        nrows=2,
        figure=fig,
        height_ratios=(1, 3),
        width_ratios=(3, 1),
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
        0.03,
        0.90,
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
            elinewidth=0.7,
            capthick=0.7,
            capsize=1,
        )
    ax_2dfdf.set_xlabel("Pulse Phase [rot]")
    ax_2dfdf.set_ylabel("$\phi$ [$\mathrm{rad}\,\mathrm{m}^{-2}$]")
    ax_2dfdf.set_xlim(phase_range)
    ax_2dfdf.set_ylim(phi_range)
    ax_2dfdf.minorticks_on()

    logger.info(f"Saving plot file: {savename}")
    fig.savefig(savename)
    return fig


def plot_rm_hist(
    vals: np.ndarray,
    savename: str = "rm_hist.png",
    logger: logging.Logger = None,
):
    if logger is None:
        logger = psrutils.get_logger()

    fig, ax = plt.subplots(dpi=300, tight_layout=True)
    hist(vals, bins="knuth", ax=ax, histtype="stepfilled", density=True)
    ax.set_ylabel("Probability Density")
    ax.set_xlabel("RM [$\mathrm{rad}\,\mathrm{m}^{-2}$]")

    logger.info(f"Saving plot file: {savename}")
    fig.savefig(savename)
    return fig, ax


def plot_rm_vs_phi(
    rm_phi_samples: np.ndarray, savename: str = "rm_phi.png", logger: logging.Logger = None
):
    if logger is None:
        logger = psrutils.get_logger()

    fig, ax = plt.subplots(dpi=300, tight_layout=True)
    ax.boxplot(rm_phi_samples.T, showfliers=False)

    xlims = ax.get_xlim()
    ax.set_xticks(np.linspace(xlims[0], xlims[1], 5))
    ax.set_xticklabels(np.arange(5) / 4)

    ax.set_ylabel("RM [$\mathrm{rad}\,\mathrm{m}^{-2}$]")
    ax.set_xlabel("Pulse Phase [rot]")

    logger.info(f"Saving plot file: {savename}")
    fig.savefig(savename)
    return fig, ax
