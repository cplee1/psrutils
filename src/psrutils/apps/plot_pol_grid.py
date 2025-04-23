import logging

import click
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import psrqpy
from uncertainties import ufloat

import psrutils


def make_figure(
    csvfiles: tuple,
    ncols: int,
    nrows: int,
    colsize: float,
    rowsize: float,
    p0_cutoff: float,
    savename: str = "pol_plot_grid",
    logger: logging.Logger | None = None,
) -> None:
    if logger is None:
        logger = psrutils.get_logger()

    query = psrqpy.QueryATNF()
    print(f"Using PSRCAT v{query.get_version}")
    psrs = query.get_pulsars()

    fig = plt.figure(figsize=(ncols * colsize, nrows * rowsize), dpi=300, tight_layout=True)

    gs0 = fig.add_gridspec(nrows, ncols)

    lw = 0.7
    caplw = 0.7
    scatter_params = dict(
        color="k", marker="none", linestyle="none", elinewidth=caplw, capthick=caplw, capsize=0.6
    )
    line_params = dict(linewidth=lw)

    for row in range(nrows):
        for col in range(ncols):
            idx = row * ncols + col
            if idx >= len(csvfiles):
                continue

            csvfile = csvfiles[idx]
            data = np.loadtxt(csvfile, unpack=True, dtype=np.float64, delimiter=",")

            with open(csvfile, "r") as f:
                header1 = f.readline().rstrip()
                header2 = f.readline().rstrip()
                header3 = f.readline().rstrip()
                source_name = header1.split(" ")[2]
                rmsf_fwhm = float(header2.split(" ")[2])
                rm_prof_val = float(header3.split(" ")[2])
                rm_prof_unc = float(header3.split(" ")[4])
                if not np.isnan(rm_prof_val) and not np.isnan(rm_prof_unc):
                    rm_prof = ufloat(rm_prof_val, rm_prof_unc)
                else:
                    rm_prof = None

            # Phase to degrees
            data[0, :] = data[0] * 360 - 180

            phase_range = (0, 1)
            match source_name:
                case "J0030+0451":
                    data[1:] = np.roll(data, 48, axis=1)[1:]
                    phase_range = (0.2, 0.8)
                case "J0437-4715":
                    phase_range = (0.18, 0.82)
                case "J0737-3039A":
                    data[1:] = np.roll(data, 32, axis=1)[1:]
                    phase_range = (0.1, 0.9)
                case "J1022+1001":
                    phase_range = (0.3, 0.7)
                case "J1300+1240":
                    phase_range = (0.2, 0.8)
                case "J1400-1431":
                    phase_range = (0.35, 0.65)
                case "J1455-3330":
                    phase_range = (0.3, 0.7)
                case "J2051-0827":
                    phase_range = (0.3, 0.7)
                case "J2145-0750":
                    phase_range = (0.32, 0.68)
                case "J2222-0137":
                    phase_range = (0.42, 0.58)
                case "J2241-5236":
                    phase_range = (0.4, 0.6)
                case "J2256-1024":
                    phase_range = (0.3, 0.7)
                case _:
                    pass
            phase_range = (phase_range[0] * 360 - 180, phase_range[1] * 360 - 180)

            gs_tmp = gs0[row, col].subgridspec(3, 1, hspace=0, height_ratios=(1, 1, 3))
            ax_rm = fig.add_subplot(gs_tmp[0])
            ax_pa = fig.add_subplot(gs_tmp[1])
            ax_pr = fig.add_subplot(gs_tmp[2])

            mask = data[6] > p0_cutoff

            ax_rm.errorbar(data[0][mask], data[9][mask], yerr=data[10][mask], **scatter_params)
            for offset in [0, -180, 180, -360, 360]:
                ax_pa.errorbar(
                    data[0][mask], data[7][mask] + offset, yerr=data[8][mask], **scatter_params
                )
            ax_pr.plot(data[0], data[1], color="k", linestyle="-", zorder=1, **line_params)
            ax_pr.plot(data[0], data[4], color="tab:blue", linestyle=":", zorder=2, **line_params)
            ax_pr.plot(data[0], data[5], color="tab:red", linestyle="--", zorder=3, **line_params)

            if rm_prof is not None:
                # Plot profile-averaged RM with uncertainty band
                y1 = [rm_prof.n - rm_prof.s] * 2
                y2 = [rm_prof.n + rm_prof.s] * 2
                ax_rm.fill_between(phase_range, y1, y2, color="tab:red", alpha=0.4, zorder=0)
                ax_rm.axhline(y=rm_prof.n, linestyle="--", color="tab:red", linewidth=lw, zorder=1)

                # Get y limits with 10% margin
                ax_rm.margins(0.15, 0.15)
                rm_lim = ax_rm.get_ylim()

                # Plot instrumental error region
                y1 = [0 - rmsf_fwhm / 2.0] * 2
                y2 = [0 + rmsf_fwhm / 2.0] * 2
                ax_rm.fill_between(phase_range, y1, y2, color="k", alpha=0.15, zorder=0)
                ax_rm.axhline(y=0, linestyle="--", color="k", linewidth=lw, zorder=1)

                # ax_rm.set_ylim([rm_prof.n - rmsf_fwhm / 2, rm_prof.n + rmsf_fwhm / 2])
                ax_rm.set_ylim(rm_lim)
            else:
                ax_rm.set_yticks([])

            ax_pr.text(
                0.04,
                0.91,
                f"{psrs[source_name]['Name'].replace('-', '$-$')}",
                horizontalalignment="left",
                verticalalignment="top",
                transform=ax_pr.transAxes,
            )
            # ax_pr.text(
            #     0.96,
            #     0.91,
            #     "$N_\mathrm{{b}}={:.0f}$".format(data.shape[1]),
            #     horizontalalignment="right",
            #     verticalalignment="top",
            #     transform=ax_pr.transAxes,
            # )

            for ax in [ax_rm, ax_pa, ax_pr]:
                ax.set_xlim(phase_range)
                ax.tick_params(which="both", right=True, top=True)
                ax.minorticks_on()
                ax.tick_params(axis="both", which="both", direction="in")
                ax.tick_params(axis="both", which="major", length=4)
                ax.tick_params(axis="both", which="minor", length=2)

            ax_pa.set_ylim([-120, 120])
            ax_pa.set_yticks([-90, 0, 90])
            ax_pa.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(15))

            ax_pr.margins(0.1, 0.1)
            ax_pr.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.5))
            ax_pr.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.1))

            ax_rm.set_xticklabels([])
            ax_pa.set_xticklabels([])

            if col % ncols == 0:
                ax_rm.set_ylabel("RM\n[$\mathrm{rad}\,\mathrm{m}^{-2}$]")
                ax_pa.set_ylabel("P.A.\n[deg]")
                ax_pr.set_ylabel("Intensity\n[arbitrary units]")
                fig.align_ylabels([ax_rm, ax_pa, ax_pr])

            if row == nrows - 1:
                ax_pr.set_xlabel("Pulse Longitude [deg]")

    logger.info(f"Saving plot file: {savename}.png")
    fig.savefig(savename + ".png")
    fig.savefig(savename + ".pdf")
    plt.close()


@click.command()
@click.argument("csvfiles", nargs=-1, type=click.Path(exists=True))
@click.option("-c", "ncols", type=int, help="Number of columns.")
@click.option("-r", "nrows", type=int, help="Number of rows.")
@click.option("-cs", "colsize", type=float, default=3.5, help="Size of each column.")
@click.option("-rs", "rowsize", type=float, default=3.2, help="Size of each row.")
@click.option("--p0_cutoff", type=float, default=3.0, help="Mask below this L/sigma_I value.")
def main(
    csvfiles: tuple, ncols: int, nrows: int, colsize: float, rowsize: float, p0_cutoff: float
) -> None:
    logger = psrutils.get_logger()

    plt.rcParams["mathtext.fontset"] = "dejavuserif"
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = "cm"
    plt.rcParams["font.size"] = 12

    # Filter out bad detections
    best_csvfiles = []
    for csvfile in csvfiles:
        data = np.loadtxt(csvfile, unpack=True, dtype=np.float64, delimiter=",")
        mask = (data[6] > p0_cutoff) & ~np.isnan(data[9])
        if len(data[9][mask]) > 0:
            best_csvfiles.append(csvfile)
    csvfiles = best_csvfiles

    nsubplots = ncols * nrows
    nfigures = int(np.ceil(len(csvfiles) / nsubplots))
    for ifig in range(nfigures):
        start_idx = ifig * nsubplots
        end_idx = (ifig + 1) * nsubplots
        if end_idx > len(csvfiles):
            end_idx = len(csvfiles)
        make_figure(
            csvfiles[start_idx:end_idx],
            ncols,
            nrows,
            colsize,
            rowsize,
            p0_cutoff,
            savename=f"pol_plot_grid_{ifig}",
            logger=logger,
        )
