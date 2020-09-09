from __future__ import print_function
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from pyIA.arguments import parse_sim_args
from pyIA.simulator import Environment
import pyIA.agents as agents
import pyIA.state_estimation as state_estimation
import pyIA.paths as paths
import pyIA.ia_planning as ia_planning

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

mapnames = ["asl", "unity_scene_map", "asl_office_j"]

# map should be given as arg

if __name__ == "__main__":
    args = parse_sim_args()
    print("------------------------------------------------------------")
    plt.figure()
    for mapname in mapnames:
        print(mapname)
        args.map_name = mapname
        env = Environment(args, silent=True)
        contours = env.worldstate.map.as_closed_obst_vertices()
        env.worldstate.map.plot_contours(contours)
        plt.axis('off')
    plt.gca().set_xticklabels([])
    plt.gca().set_yticklabels([])
    plt.show()
