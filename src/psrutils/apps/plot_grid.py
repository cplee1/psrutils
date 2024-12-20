import click
import matplotlib.pyplot as plt
import numpy as np

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

    fig, axes = plt.subplots(
        figsize=(ncols * 2, nrows * 2), ncols=ncols, nrows=nrows, dpi=300, tight_layout=True
    )

    for row in range(nrows):
        for col in range(ncols):
            data = data_list[col + row * ncols]
            ax = axes[row, col]
            data._archive.centre_max_bin()
            prof = data.profile
            bins = np.arange(data.num_bin) / (data.num_bin - 1)
            ax.plot(bins, prof, color="k")
            ax.set_yticklabels([])
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
            ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
            ax.set_xlim([bins[0], bins[-1]])
            ax.set_title(f"{data._archive.get_source()}")

    fig.supxlabel("Pulse Phase [rot]")
    fig.supylabel("Flux Density [arb]")
    fig.savefig("grid.png")
    plt.close()
