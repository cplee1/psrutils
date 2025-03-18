import click
import matplotlib.pyplot as plt
import numpy as np
import psrqpy

import psrutils


@click.command()
@click.argument("spec_files", nargs=-1, type=click.Path(exists=True))
@click.option("-c", "ncols", type=int, help="Number of columns.")
@click.option("-r", "nrows", type=int, help="Number of rows.")
@click.option("-s", "spacing", type=float, default=1.2, help="Vertical spacing.")
@click.option("-l", "plot_ctrline", is_flag=True, help="Plot a centre line.")
def main(spec_files: tuple, ncols: int, nrows: int, spacing: float, plot_ctrline: bool) -> None:
    if ncols * nrows < len(spec_files):
        raise ValueError("To few subplots for the provided archives.")

    plt.rcParams["mathtext.fontset"] = "dejavuserif"
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = "cm"
    plt.rcParams["font.size"] = 12

    fig, axes = plt.subplots(
        figsize=(0.5 + ncols * 3, 0.5 + nrows * 4),
        ncols=ncols,
        nrows=nrows,
        dpi=300,
        tight_layout=True,
    )

    query = psrqpy.QueryATNF()
    print(f"Using PSRCAT v{query.get_version}")
    psrs = query.get_pulsars()

    lw = 0.45

    for ii, spec_file in enumerate(spec_files):
        spec = np.loadtxt(spec_file, dtype=str, delimiter=",")

        cube_list = []
        telescope_list = []
        for archive, shift, bscr, telescope in spec:
            if bscr != "":
                bscr = int(bscr)
            else:
                bscr = None
            tmp_cube = psrutils.StokesCube.from_psrchive(
                str(archive),
                clone=False,
                tscrunch=1,
                fscrunch=1,
                bscrunch=bscr,
                rotate_phase=float(shift),
            )
            cube_list.append(tmp_cube)
            telescope_list.append(telescope)

        col = ii % ncols
        row = ii // ncols

        if ncols * nrows == 1:
            ax = axes
        elif ncols == 1:
            ax = axes[row]
        elif nrows == 1:
            ax = axes[col]
        else:
            ax = axes[row, col]

        for jj, (cube, tele) in enumerate(zip(cube_list, telescope_list)):
            if cube.num_pol == 4:
                psrutils.add_profile_to_axes(
                    cube, ax_prof=ax, voffset=jj * spacing, plot_pol=True, lw=lw
                )
            else:
                psrutils.add_profile_to_axes(
                    cube, ax_prof=ax, voffset=jj * spacing, plot_pol=False, lw=lw
                )
            minfreq = f"{cube.min_freq:.0f}"
            maxfreq = f"{cube.max_freq:.0f}"
            if minfreq == maxfreq:
                label = f"{minfreq} MHz"
            else:
                label = f"{minfreq}-{maxfreq} MHz"
            ax.text(
                0.05,
                (jj + 0.4) * spacing,
                f"{tele}\n{label}",
                fontsize=8,
                horizontalalignment="left",
                verticalalignment="center",
            )
            ax.set_yticklabels([])
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
            ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
            ax.set_xlim([0, 1])
            ax.set_yticks([])

        if plot_ctrline:
            ax.axvline(0.5, linestyle="-", color="k", linewidth=lw, alpha=0.4)

        psrname = cube_list[0]._archive.get_source()
        ax.set_title(
            # f"{psrs[psrname].Name}\nP={psrs[psrname].P0*1e3:.3f}  DM={psrs[psrname].DM:.2f}",
            f"{psrs[psrname].Name}",
            fontsize=14,
        )

    # fig.supxlabel("Pulse Phase")
    # fig.supylabel("Flux Density [arb. units]")
    fig.savefig("stack.png")
    fig.savefig("stack.pdf")
    plt.close()
