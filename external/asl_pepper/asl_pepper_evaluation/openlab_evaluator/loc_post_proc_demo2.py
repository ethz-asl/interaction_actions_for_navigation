import matplotlib.pyplot as plt
import numpy as np
import os
from CMap2D import CMap2D
from map2d import gridshow

MAPNAME = "asl_office_j"
MAPFOLDER = os.path.expanduser("~/maps")

map_ = CMap2D(MAPFOLDER, MAPNAME)

csv = np.loadtxt("./results_loc_live_eval_demo2.txt", dtype="string", delimiter=', ')

values = csv[:,1:]
values[values == 'None'] = 'nan'
values = values.astype(float)

start_times = values[:, 0]
durations = values[:, 1]
starts = values[:, [2,3,4]]
true_loc = values[:, [5, 6, 7]]
est_loc = values[:, [8, 9, 10]]


est_loc_ij = map_.xy_to_floatij(np.ascontiguousarray(est_loc[:,:2]))
n_total = len(start_times)
n_nan = np.sum(np.isnan(est_loc[:,0]))
n_estimates = n_total - n_nan

plt.figure()
gridshow(map_.occupancy(), zorder=-1)
plt.scatter(est_loc_ij[:, 0], est_loc_ij[:, 1], facecolors=(0,0,0,0), edgecolors='green', label='estimates', zorder=2)
plt.title("Localization Cold Starts (20s deadline)")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.legend()
plt.show()

