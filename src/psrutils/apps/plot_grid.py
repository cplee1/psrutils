import click
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import psrqpy

import psrutils


@click.command()
@click.argument("archives", nargs=-1, type=click.Path(exists=True))
@click.option("-b", "bscr", type=int, help="Bscrunch to this number of phase bins.")
@click.option("-c", "ncols", type=int, help="Number of columns.")
@click.option("-r", "nrows", type=int, help="Number of rows.")
def main(archives: tuple, bscr: int, ncols: int, nrows: int) -> None:
    if ncols * nrows < len(archives):
        raise ValueError("To few subplots for the provided archives.")

    data_list = []
    for archive in archives:
        tmp_data = psrutils.StokesCube.from_psrchive(archive, False, 1, 1, bscr)
        data_list.append(tmp_data)

    plt.rcParams["mathtext.fontset"] = "dejavuserif"
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = "cm"
    plt.rcParams["font.size"] = 14

    fig, axes = plt.subplots(
        figsize=(ncols * 2.7, nrows * 2.5), ncols=ncols, nrows=nrows, dpi=300, tight_layout=True
    )

    query = psrqpy.QueryATNF()
    print(f"Using PSRCAT v{query.get_version}")
    psrs = query.get_pulsars()

    for row in range(nrows):
        for col in range(ncols):
            ax = axes[row, col]

            ax.set_yticklabels([])
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
            ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])

            ax.minorticks_on()
            ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.05))
            ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
            ax.tick_params(axis="both", which="major", length=4)
            ax.tick_params(axis="both", which="minor", length=2)

            sp_idx = col + row * ncols
            if sp_idx >= len(data_list):
                continue
            data = data_list[sp_idx]

            prof = data.profile
            bins = np.arange(data.num_bin) / (data.num_bin - 1)
            ax.plot(bins, prof, color="k", linewidth=0.6)
            ax.set_xlim([bins[0], bins[-1]])
            psrname = data._archive.get_source()
            ax.set_title(
                f"{psrs[psrname].Name}\nP={psrs[psrname].P0 * 1e3:.3f}  DM={psrs[psrname].DM:.2f}",
                fontsize=14,
            )

    # fig.supxlabel("Pulse Phase [rot]")
    # fig.supylabel("Flux Density [arb. units]")
    fig.savefig("grid.png")
    fig.savefig("grid.pdf")
    plt.close()
