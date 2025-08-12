########################################################
# Licensed under the Academic Free License version 3.0 #
########################################################

import bisect
import importlib.resources

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from statsmodels.sandbox.stats.runs import runstest_1samp
from statsmodels.stats.diagnostic import acorr_ljungbox


def simulate_joint_null(
    nsamp: int, nbootstrap: int = 1_000_000, savename: str = "null_dist.png"
) -> None:
    nlags = 20
    cmap = plt.get_cmap("cmr.arctic_r")

    T_lb_sim = np.empty(nbootstrap, dtype=float)
    T_runs_sim = np.empty(nbootstrap, dtype=float)
    for ii in range(nbootstrap):
        sim_samps = np.random.normal(0, 1, size=nsamp)
        T_lb_sim[ii] = float(acorr_ljungbox(sim_samps, lags=[nlags])["lb_stat"])
        T_runs_sim[ii] = float(runstest_1samp(sim_samps)[0])

    # Compute 2D hist
    nbins = 100
    bin_edges = [
        np.linspace(-5, 5, nbins + 1),
        np.linspace(0, 60, nbins + 1),
    ]
    H, x, y = np.histogram2d(T_runs_sim, T_lb_sim, bins=bin_edges, density=True)
    H = H.T

    # Get bin centre arrays
    x1, y1 = 0.5 * (x[1:] + x[:-1]), 0.5 * (y[1:] + y[:-1])

    # Get bin centre grids
    X1, Y1 = np.meshgrid(x1, y1, indexing="xy")

    # Get a (npix, 2) array of pixel coordinates
    pixels = np.vstack((X1.ravel(), Y1.ravel())).T

    # Smooth the hist
    H = gaussian_filter(H, 1.5)

    # Get contours
    qcs = plt.contour(X1, Y1, H, levels=1000, colors="k", alpha=0.5)
    c_levels = qcs.levels
    c_paths = qcs.get_paths()

    # Assign each pixel a contour level
    lvl_grid = np.full(shape=H.shape, fill_value=0.0, dtype=float)
    for ii, (c_level, c_path) in enumerate(zip(c_levels, c_paths, strict=True)):
        if ii == 0 or ii == len(c_levels) - 1:
            continue
        mask = c_path.contains_points(pixels).reshape(H.shape)
        lvl_grid[mask] = c_level

    # Calculate the p-value for each contour level
    total_mass = np.sum(H)
    p_levels = []
    for c_level in c_levels:
        cdf = np.sum(H[lvl_grid >= c_level])
        p_levels.append((total_mass - cdf) / total_mass)

    # Map the p-values to the grid
    p_grid = np.full(shape=H.shape, fill_value=0.0, dtype=float)
    for ii, (c_level, p_level) in enumerate(zip(c_levels, p_levels, strict=True)):
        if ii == 0 or ii == len(c_levels) - 1:
            continue
        p_grid[lvl_grid == c_level] = p_level

    np.save("lb_runs_joint_null_p_grid.npy", p_grid)
    np.save("lb_runs_joint_null_T_runs.npy", x1)
    np.save("lb_runs_joint_null_T_lb.npy", y1)

    # Make the plot
    plt.clf()
    fig = plt.figure(dpi=300, tight_layout=True)
    ax = fig.add_subplot()
    cs = ax.pcolormesh(
        X1, Y1, p_grid, rasterized=True, shading="auto", cmap=cmap, vmin=0.0, vmax=1.0
    )
    cbar = fig.colorbar(cs)
    cbar.ax.set_ylabel("$p$-value")
    ax.set_xlabel("$T_\\mathrm{runs}$")
    ax.set_ylabel("$T_\\mathrm{LB}$")
    fig.savefig(savename)
    plt.close()


def lookup_joint_null(T_runs_obs: float, T_lb_obs: float) -> np.float64:
    ref = importlib.resources.files("psrutils") / "data/lb_runs_joint_null_p_grid.npy"
    with importlib.resources.as_file(ref) as path:
        p_grid = np.load(path)

    ref = importlib.resources.files("psrutils") / "data/lb_runs_joint_null_T_runs.npy"
    with importlib.resources.as_file(ref) as path:
        T_runs_arr = np.load(path)

    ref = importlib.resources.files("psrutils") / "data/lb_runs_joint_null_T_lb.npy"
    with importlib.resources.as_file(ref) as path:
        T_lb_arr = np.load(path)

    if (
        T_runs_obs < T_runs_arr[0]
        or T_runs_obs > T_runs_arr[-1]
        or T_lb_obs < T_lb_arr[0]
        or T_lb_obs > T_lb_obs > T_lb_arr[-1]
    ):
        return 0.0

    xidx = bisect.bisect_left(T_runs_arr, T_runs_obs)
    yidx = bisect.bisect_left(T_lb_arr, T_lb_obs)

    return p_grid[xidx - 1, yidx - 1]
