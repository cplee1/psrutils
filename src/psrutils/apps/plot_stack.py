import click
import matplotlib.pyplot as plt
import numpy as np
import psrqpy

import psrutils


@click.command()
@click.argument("spec_files", nargs=-1, type=click.Path(exists=True))
@click.help_option("-h", "--help")
@click.version_option(psrutils.__version__, "-V", "--version")
@click.option("-c", "ncols", type=int, help="Number of columns.")
@click.option("-r", "nrows", type=int, help="Number of rows.")
@click.option("-cs", "colsize", type=float, default=2.6, help="Size of each column.")
@click.option("-rs", "rowsize", type=float, default=4.3, help="Size of each row.")
@click.option("-s", "spacing", type=float, default=1.2, help="Vertical spacing.")
@click.option("-z", "zoom", type=float, nargs=2, help="Zoom into this longitude range in degrees.")
@click.option("-l", "plot_ctrline", is_flag=True, help="Plot a centre line.")
@click.option("-p", "plot_pol", is_flag=True, help="Plot polarisation profiles.")
def main(
    spec_files: tuple,
    ncols: int,
    nrows: int,
    colsize: float,
    rowsize: float,
    spacing: float,
    zoom: tuple[float, float],
    plot_ctrline: bool,
    plot_pol: bool,
) -> None:
    psrutils.setup_logger("psrutils")

    if ncols * nrows < len(spec_files):
        raise ValueError("To few subplots for the provided archives.")

    if zoom is None:
        zoom = (-180, 180)
    elif zoom[1] < zoom[0]:
        zoom = (zoom[1], zoom[0])

    fig = plt.figure(figsize=(ncols * colsize, nrows * rowsize), dpi=300, tight_layout=True)

    gs0 = fig.add_gridspec(nrows, ncols)

    query = psrqpy.QueryATNF()
    print(f"Using PSRCAT v{query.get_version}")
    psrs = query.get_pulsars()

    lw = 0.8
    text_margin = 0.0

    for ii in range(ncols * nrows):
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

        col = ii % ncols
        row = ii // ncols

        ax = fig.add_subplot(gs0[row, col])

        for jj, (cube, tele) in enumerate(zip(cube_list, telescope_list, strict=False)):
            if not plot_pol and tele == "MWA":
                I_colour = "tab:red"
            else:
                I_colour = "k"

            if cube.num_pol == 4:
                psrutils.add_profile_to_axes(
                    cube,
                    ax_prof=ax,
                    voffset=jj * spacing,
                    bin_func=psrutils.centre_offset_degrees,
                    plot_pol=plot_pol,
                    lw=lw,
                    I_colour=I_colour,
                )
            else:
                psrutils.add_profile_to_axes(
                    cube,
                    ax_prof=ax,
                    voffset=jj * spacing,
                    bin_func=psrutils.centre_offset_degrees,
                    plot_pol=plot_pol,
                    lw=lw,
                    I_colour=I_colour,
                )
            minfreq = f"{cube.min_freq:.0f}"
            maxfreq = f"{cube.max_freq:.0f}"
            if minfreq == maxfreq:
                label = f"{minfreq} MHz"
            else:
                label = f"{minfreq}\u2013{maxfreq} MHz"
            ax.text(
                zoom_psr[0] + text_margin * (zoom_psr[1] - zoom_psr[0]),
                (jj + 0.4) * spacing,
                f"{tele}\n{label}",
                fontsize=8,
                horizontalalignment="left",
                verticalalignment="center",
                color=I_colour,
            )

            ax.set_yticks([])
            ax.set_xlim(zoom_psr)
            ax.minorticks_on()
            ax.tick_params(axis="both", which="major", length=4)
            ax.tick_params(axis="both", which="minor", length=2)

            ax.set_xlabel("Pulse Longitude [deg]")

        if plot_ctrline:
            ax.axvline(0.5, linestyle="-", color="k", linewidth=lw, alpha=0.4)

        psrname = cube_list[0]._archive.get_source()
        ax.text(
            text_margin,
            0.99,
            f"{psrs[psrname].Name.replace('-', '$-$')}",
            horizontalalignment="left",
            verticalalignment="top",
            transform=ax.transAxes,
        )

        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["top"].set_visible(False)

    fig.savefig("stack.png")
    fig.savefig("stack.pdf")
    plt.close()
