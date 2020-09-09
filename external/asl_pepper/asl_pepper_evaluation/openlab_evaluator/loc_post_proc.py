from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from CMap2D import CMap2D
from map2d import gridshow

LOGFILE = "./results/results_loc_live_eval_corridor_koze_kids.txt"
if len(sys.argv) == 2:
    LOGFILE = sys.argv[1]
MAPNAME = "openlab_office_j"
MAPFOLDER = os.path.expanduser("~/maps")

map_ = CMap2D(MAPFOLDER, MAPNAME)

print("loading {}".format(LOGFILE))
csv = np.loadtxt(LOGFILE, dtype="string", delimiter=", ")

values = csv[:, 1:]
values[values == "None"] = "nan"
values = values.astype(float)

start_times = values[:, 0]
durations = values[:, 1]
starts = values[:, [2, 3, 4]]
true_loc = values[:, [5, 6, 7]]
est_loc = values[:, [8, 9, 10]]

loc_error = np.linalg.norm(true_loc[:, :2] - est_loc[:, :2], axis=-1)

true_loc_ij = map_.xy_to_floatij(np.ascontiguousarray(true_loc[:, :2]))
est_loc_ij = map_.xy_to_floatij(np.ascontiguousarray(est_loc[:, :2]))
iscorrect = loc_error < 1
isfalse = loc_error >= 1
isnan = np.isnan(loc_error)
n_total = len(start_times)
n_nan = np.sum(isnan)
n_estimates = n_total - n_nan

print("No localization: {}".format(n_nan))
print("Mean loc error: {}".format(np.mean(loc_error[np.logical_not(isnan)])))
plt.figure()
gridshow(map_.occupancy(), zorder=-1)
for i in range(len(start_times)):
    start = true_loc_ij[i]
    end = est_loc_ij[i]
    color = "r" if isfalse[i] else "g"
    plt.plot(
        np.array([start[0], end[0]]),
        np.array([start[1], end[1]]),
        "--" + color,
        linewidth=1,
        zorder=0,
    )
plt.scatter(
    true_loc_ij[:, 0][iscorrect],
    true_loc_ij[:, 1][iscorrect],
    facecolors="green",
    edgecolors="k",
    label="true position (e < 1m)",
    zorder=2,
)
plt.scatter(
    true_loc_ij[:, 0][isfalse],
    true_loc_ij[:, 1][isfalse],
    facecolors="red",
    edgecolors="k",
    label="true position (e >= 1m)",
    zorder=2,
)
plt.scatter(
    est_loc_ij[:, 0][iscorrect],
    est_loc_ij[:, 1][iscorrect],
    marker="o",
    facecolors=(0, 0, 0, 0),
    edgecolors="green",
    label="estimate (e < 1m)",
    zorder=1,
)
plt.scatter(
    est_loc_ij[:, 0][isfalse],
    est_loc_ij[:, 1][isfalse],
    marker="o",
    facecolors=(0, 0, 0, 0),
    edgecolors="red",
    label="estimate (e >= 1m)",
    zorder=1,
)
plt.title("Localization Cold Starts (20s deadline)")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.legend()
ax = plt.gca()
ticks = (ax.get_xticks() * map_.resolution()).astype(int)
ax.set_xticklabels(ticks)
ticks = (ax.get_yticks() * map_.resolution()).astype(int)
ax.set_yticklabels(ticks)
plt.figure()
plt.hist(loc_error[np.logical_not(np.isnan(loc_error))], bins=100, color="black")
plt.title(
    "Error Distribution for Localization Estimates ({} cold starts, {} estimates, {} no estimate found)".format(
        n_total, n_estimates, n_nan
    )
)
plt.xlabel("localization error [m]")
plt.ylabel("count")
plt.show()
