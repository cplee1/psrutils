import click
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import psrqpy

import psrutils


@click.command()
@click.argument("spec_files", nargs=-1, type=click.Path(exists=True))
@click.option("-c", "ncols", type=int, help="Number of columns.")
@click.option("-r", "nrows", type=int, help="Number of rows.")
@click.option("-cs", "colsize", type=float, default=4, help="Size of each column.")
@click.option("-rs", "rowsize", type=float, default=4.5, help="Size of each row.")
@click.option("-z", "zoom", type=float, nargs=2, help="Zoom into this longitude range in degrees.")
@click.option("-l", "plot_ctrline", is_flag=True, help="Plot a centre line.")
def main(
    spec_files: tuple,
    ncols: int,
    nrows: int,
    colsize: float,
    rowsize: float,
    zoom: tuple[float, float],
    plot_ctrline: bool,
) -> None:
    if ncols * nrows < len(spec_files):
        raise ValueError("To few subplots for the provided archives.")

    if zoom is None:
        zoom = (-180, 180)
    elif zoom[1] < zoom[0]:
        zoom = (zoom[1], zoom[0])

    logger = psrutils.get_logger()

    fig = plt.figure(figsize=(ncols * colsize, nrows * rowsize), dpi=300, tight_layout=True)

    gs0 = fig.add_gridspec(nrows, ncols)

    query = psrqpy.QueryATNF()
    # logger.info(f"Using PSRCAT v{query.get_version}")
    psrs = query.get_pulsars()

    lw = 0.7
    text_margin = 0.05
    p0_cutoff = 3

    for row in range(nrows):
        for col in range(ncols):
            ii = row * ncols + col
            if ii >= len(spec_files):
                continue

            spec = np.loadtxt(spec_files[ii], dtype=str, delimiter=",")
            if spec.ndim == 1:
                spec = spec.reshape(1, -1)

            zoom_psr = zoom
            extra_shift = 0.0
            with open(spec_files[ii], "r") as f:
                end_header = False
                while not end_header:
                    line = f.readline().rstrip().split(" ")
                    if len(line) == 1:
                        end_header = True
                    else:
                        match line[1]:
                            case "ZOOM":
                                zoom_psr = (float(line[2]), float(line[3]))
                                if zoom_psr[1] < zoom_psr[0]:
                                    zoom_psr = (zoom_psr[1], zoom_psr[0])
                            case "SHIFT":
                                extra_shift = float(line[2])
                            case _:
                                end_header = True

            cube_list = []
            telescope_list = []
            for archive, shift, bscr, telescope in spec:
                if bscr != "":
                    bscr = int(bscr)
                else:
                    bscr = None
                if shift != "":
                    shift = float(shift) + extra_shift
                else:
                    shift = extra_shift
                tmp_cube = psrutils.StokesCube.from_psrchive(
                    str(archive),
                    clone=False,
                    tscrunch=1,
                    fscrunch=1,
                    bscrunch=bscr,
                    rotate_phase=shift,
                )
                cube_list.append(tmp_cube)
                telescope_list.append(telescope)

            gs_box = gs0[row, col].subgridspec(5, 1, hspace=0, height_ratios=(2, 1, 2, 2, 1))
            ax_pa = fig.add_subplot(gs_box[0])
            ax_dp = fig.add_subplot(gs_box[1])
            ax_pr1 = fig.add_subplot(gs_box[2])
            ax_pr2 = fig.add_subplot(gs_box[3])
            ax_ds = fig.add_subplot(gs_box[4])
            ax_pr = [ax_pr1, ax_pr2]

            prof_data = {}
            for jj in range(2):
                cube = cube_list[jj]
                tele = telescope_list[jj]
                if cube.num_pol != 4:
                    continue

                if jj == 0:
                    PA_colour = ("k",)
                else:
                    PA_colour = "tab:red"

                bins, (iquv_prof, l_prof, pa, p0_l, _, _) = psrutils.add_profile_to_axes(
                    cube,
                    ax_pa=ax_pa,
                    ax_prof=ax_pr[jj],
                    bin_func=psrutils.centre_offset_degrees,
                    p0_cutoff=p0_cutoff,
                    PA_colour=PA_colour,
                    plot_pol=True,
                    lw=lw,
                    label=tele,
                    logger=logger,
                )
                prof_data[jj] = dict(
                    bins=bins, iquv_prof=iquv_prof, l_prof=l_prof, pa=pa, p0_l=p0_l
                )
                ax_pa.legend(fontsize=9, loc="upper left")

                minfreq = f"{cube.min_freq:.0f}"
                maxfreq = f"{cube.max_freq:.0f}"
                if minfreq == maxfreq:
                    label = f"{minfreq} MHz"
                else:
                    label = f"{minfreq}\u2013{maxfreq} MHz"
                ax_pr[jj].text(
                    zoom_psr[0] + text_margin * (zoom_psr[1] - zoom_psr[0]),
                    0.5,
                    f"{tele}\n{label}",
                    fontsize=9,
                    horizontalalignment="left",
                    verticalalignment="center",
                )

                if plot_ctrline:
                    ax_pr[jj].axvline(0.5, linestyle="-", color="k", linewidth=lw, alpha=0.4)

            psrname = cube_list[0]._archive.get_source()
            ax_pa.set_title(f"{psrs[psrname].Name.replace('-', '$-$')}")

            assert len(prof_data[0]["bins"]) == len(prof_data[1]["bins"])

            pa_mask = (prof_data[0]["p0_l"] > p0_cutoff) & (prof_data[1]["p0_l"] > p0_cutoff)
            dpa = prof_data[0]["pa"][0, pa_mask] - prof_data[1]["pa"][0, pa_mask]
            dpa = np.where(dpa > 0.0, dpa, dpa + 180.0)
            dpa_unc = np.sqrt(
                prof_data[0]["pa"][1, pa_mask] ** 2 + prof_data[1]["pa"][1, pa_mask] ** 2
            )
            di = prof_data[0]["iquv_prof"][0] - prof_data[1]["iquv_prof"][0]
            dv = prof_data[0]["iquv_prof"][3] - prof_data[1]["iquv_prof"][3]
            dl = prof_data[0]["l_prof"] - prof_data[1]["l_prof"]

            # logger.info(f"{np.std(dpa/dpa_unc)=}")
            # logger.info(f"{np.std(di)=}")
            # logger.info(f"{np.std(dv)=}")
            # logger.info(f"{np.std(dl)=}")

            print(f"{shift},{np.std(di)}")

            ax_dp.errorbar(
                x=prof_data[0]["bins"][pa_mask],
                y=dpa - np.mean(dpa),
                yerr=dpa_unc,
                ecolor="k",
                marker="none",
                ms=1,
                linestyle="none",
                elinewidth=lw,
                capthick=lw,
                capsize=0,
            )

            ax_ds.plot(
                prof_data[0]["bins"],
                di,
                linewidth=lw,
                linestyle="-",
                color="k",
                zorder=1,
            )

            ax_ds.plot(
                prof_data[0]["bins"],
                dv,
                linewidth=lw,
                linestyle=":",
                color="tab:blue",
                zorder=2,
            )

            ax_ds.plot(
                prof_data[0]["bins"],
                dl,
                linewidth=lw,
                linestyle="--",
                color="tab:red",
                zorder=3,
            )

            phase_range = zoom_psr
            for ax in [ax_pa, ax_dp, ax_pr1, ax_pr2, ax_ds]:
                ax.set_xlim(phase_range)
                ax.tick_params(which="both", right=True, top=True)
                ax.minorticks_on()
                ax.tick_params(axis="both", which="both", direction="in")
                ax.tick_params(axis="both", which="major", length=4)
                ax.tick_params(axis="both", which="minor", length=2)

            # ax_pa.set_ylim([-120, 120])
            # ax_pa.set_yticks([-90, 0, 90])
            ax_pa.set_ylim([-30, 210])
            ax_pa.set_yticks([0, 90, 180])
            ax_pa.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(15))
            ax_pa.set_xticklabels([])
            ax_pa.set_ylabel("P.A.\n[deg]")

            for iax_pr in ax_pr:
                iax_pr.set_ylim([-0.3, 1.2])
                iax_pr.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.5))
                iax_pr.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.1))
                iax_pr.set_xticklabels([])
                iax_pr.set_ylabel("Intensity\n[arb. units]")

            # ax_dp.margins(-0.3, 0.3)
            ax_dp.set_ylim([-40, 40])
            ax_dp.set_xticklabels([])
            ax_dp.set_ylabel("$\Delta$P.A.\n[deg]")

            ax_ds.set_ylim([-0.4, 0.4])
            ax_ds.set_yticks([-0.2, 0.2])
            ax_ds.set_ylabel("$\Delta$Intensity\n[arb. units]")
            ax_ds.set_xlabel("Pulse Longitude [deg]")

            if col != 0:
                for ax in [ax_pa, ax_dp, ax_pr1, ax_pr2, ax_ds]:
                    ax.set_ylabel(None)
                    ax.set_yticklabels([])

            fig.align_ylabels([ax_pa, ax_dp, ax_pr1, ax_pr2, ax_ds])
    fig.savefig("compare.png")
    fig.savefig("compare.pdf")
    plt.close()
