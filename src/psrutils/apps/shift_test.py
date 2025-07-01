import click
import matplotlib.pyplot as plt
import numpy as np

import psrutils


@click.command()
@click.argument("spec_file", nargs=1, type=click.Path(exists=True))
@click.help_option("-h", "--help")
@click.version_option(psrutils.__version__, "-V", "--version")
def main(spec_file: str) -> None:
    spec = np.loadtxt(spec_file, dtype=str, delimiter=",")
    if spec.ndim == 1:
        spec = spec.reshape(1, -1)

    zoom_psr = (-180, 180)
    extra_shift = 0.0
    with open(spec_file, "r") as f:
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
    for archive, shift, bscr, _ in spec:
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

    dshifts = np.arange(-0.02, 0.02, 0.00025)
    test = np.empty_like(dshifts)

    prof_ref = cube_list[0].profile
    for ii in range(dshifts.size):
        tmp_archive = cube_list[1].archive_clone
        tmp_archive.pscrunch()
        tmp_archive.rotate_phase(dshifts[ii])
        prof_dph = tmp_archive.get_data()[0, 0, 0, :]

        di = prof_ref - prof_dph
        test[ii] = np.std(di)

    best_shift = dshifts[np.argmin(test)]
    print(best_shift)

    plt.plot(dshifts, test)
    plt.ylabel("SD($\Delta$Intensity)")
    plt.xlabel("Phase Shift")
    plt.gca().axvline(best_shift, color="k", linestyle=":")

    source_name = cube_list[0].source

    plt.title(source_name)
    plt.savefig(f"{source_name}_shifts_test.png", dpi=200)
    plt.close()
