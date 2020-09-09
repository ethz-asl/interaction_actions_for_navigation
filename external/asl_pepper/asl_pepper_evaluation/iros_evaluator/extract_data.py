from __future__ import print_function
import numpy as np
import numpy.ma as ma
import os
import rosbag
import argparse

# Arguments
parser = argparse.ArgumentParser(description='ROS node for clustering 2d lidar')
parser.add_argument('root_dir')
parser.add_argument('--full', action='store_true', help='Run full evaluation (slow)',)
parser.add_argument('--early-stop', action='store_true',)
args = parser.parse_args()

PLANNERS = ["IA", "Move Base", "Intend", "Say", "Nudge", "CADRL", "navrep"]
MAPNAMES = ["asl", "unity_scene_map", "asl_office_j"]
N_SCENES = 6
N_TRIALS = 10
distances_to_goal_10sample = {
    "irosasl1": 7.999999821186066,
    "irosasl2": 22.389168497502457,
    "irosasl3": 22.853390559734407,
    "irosasl4": 7.999999821186066,
    "irosasl5": 17.599999606609344,
    "irosasl6": 22.853390559734407,
    "irosasl_office_j1": 8.099405499368514,
    "irosasl_office_j2": 42.62186984664014,
    "irosasl_office_j3": 12.418566887427874,
    "irosasl_office_j4": 9.023260564876859,
    "irosasl_office_j5": 24.242836110813236,
    "irosasl_office_j6": 42.184684099701585,
    "irosunity_scene_map1": 9.241890349538409,
    "irosunity_scene_map2": 26.734565422571528,
    "irosunity_scene_map3": 17.529466682086248,
    "irosunity_scene_map4": 16.838010448160635,
    "irosunity_scene_map5": 17.181325690611718,
    "irosunity_scene_map6": 23.189255112287523,
}
distances_to_goal_10sample_array = np.nan * np.ones((len(MAPNAMES), N_SCENES))
for l, mapname in enumerate(MAPNAMES):
    i_map = l + 1
    for i_scene in range(1, N_SCENES+1):
        distances_to_goal_10sample_array[l, i_scene-1] = \
            distances_to_goal_10sample["iros" + mapname + str(i_scene)]


ROOT_DIR = args.root_dir
AUTOEVAL_DIR = os.path.join(ROOT_DIR, "autoeval")
try:
    instances = os.listdir(AUTOEVAL_DIR)
except OSError as e:
    print(e)
    exit(1)

# evaluation settings
FULL_EVAL = args.full
COMFORT_DISTANCES = [0.1, 0.2, 0.3, 0.45, 0.5, 0.6, 0.75, 1., 1.2, 1.5, np.inf]  # last one is a sanity check
EVAL_DISTANCES = [('d_i', 3), ('d_p', 8)]  # indices of comfort distances we care about


# variables
ttgs = np.nan * np.ones((len(instances), len(PLANNERS), len(MAPNAMES), N_SCENES, N_TRIALS))
ngestures = np.zeros_like(ttgs)
avgdmins = np.nan * np.ones_like(ttgs)
tbelowcd = np.nan * np.ones(
    (len(instances), len(PLANNERS), len(MAPNAMES), N_SCENES, N_TRIALS, len(COMFORT_DISTANCES))
)
strangenesses = np.zeros_like(ttgs)

for n, instance in enumerate(instances):
    INSTANCE_DIR = os.path.join(AUTOEVAL_DIR, instance)
    print()
    print("----------------------")
    print("--", instance, "--")
    print("----------------------")
    print()
    try:
        with open(os.path.join(INSTANCE_DIR, "info.txt"), 'r') as f:
            print(f.read())
    except IOError:
        pass
    for k, plannername in enumerate(PLANNERS):
        i_planner = k + 1
        print("----------------", "PLANNER: ", plannername, "----------------------")
        for l, mapname in enumerate(MAPNAMES):
            i_map = l + 1
            print("MAP: ", mapname)
            for i_scene in range(1, N_SCENES+1):
                for i_trial in range(1, N_TRIALS+1):
                    try:
                        path = os.path.join(
                            INSTANCE_DIR,
                            "planner_{}".format(i_planner),
                            "map_{}".format(i_map),
                            "scene_{}".format(i_scene),
                            "trial_{}.bag".format(i_trial),
                        )
                        bag = rosbag.Bag(path)
                    except IOError:
                        continue
                    except KeyboardInterrupt:
                        print("KeyboardInterrupt, exiting.")
                        exit(0)
                    goal_reached_time = None
                    strangeness = ""
                    # goal reached time
                    for topic, msg, t in bag.read_messages(topics=['/goal_reached', ]):
                        goal_reached_time = msg.stamp.to_sec()
                        break
                    # gestures used
                    n_gestures_used = 0
                    for topic, msg, t in bag.read_messages(topics=['/gestures', ]):
                        if "You" in msg.data:
                            n_gestures_used += 1
                    ngestures[n, k, l, i_scene-1, i_trial-1] = n_gestures_used
                    if FULL_EVAL and goal_reached_time is not None:
                        # agent_positions, average min distance
                        mindsqrs = []
                        last_t = None
                        tbelowcd[n, k, l, i_scene-1, i_trial-1, :] = 0.
                        for topic, msg, t in bag.read_messages(topics=['/agent_goals', ]):
                            if last_t is None:
                                last_t = t.to_sec()
                            dt = t.to_sec() - last_t
                            last_t = t.to_sec()
                            if dt > 1. or dt < 0:
                                print("dt is weird:", dt, last_t, t.to_sec())
                                continue
                            mindsqr = np.inf
                            rob_x = msg.markers[1].pose.position.x  # first marker is goals
                            rob_y = msg.markers[1].pose.position.y
                            # yeah, got caught with this one. markers are actually
                            # 0: goals, 1: robot, 2: arrow, 3: agent, 4: arrow, 5: agent, etc..
                            for marker in msg.markers[3::2]:
                                dx = rob_x - marker.pose.position.x
                                dy = rob_y - marker.pose.position.y
                                dsqr = dx**2 + dy**2
                                if dsqr < mindsqr:
                                    mindsqr = dsqr
                            mindsqrs.append(mindsqr)
                            for i_dist, dist in enumerate(COMFORT_DISTANCES):
                                if np.sqrt(mindsqr) < dist:
                                    tbelowcd[n, k, l, i_scene-1, i_trial-1, i_dist] += dt
                        avgdmin = np.sqrt(np.mean(mindsqrs))
                        avgdmins[n, k, l, i_scene-1, i_trial-1] = avgdmin
                    if goal_reached_time is not None:
                        # reached goal
                        ttgs[n, k, l, i_scene-1, i_trial-1] = goal_reached_time
                    else:
                        # Didn't reach goal
                        ttgs[n, k, l, i_scene-1, i_trial-1] = np.inf
                        # determine last clock and strangeness
                        last_clock = None
                        is_strange = False
                        for topic, msg, t in bag.read_messages(topics=['/clock', ]):
                            last_clock = msg.clock.to_sec()
                            # should be either 300 or 600 (+- 1s)
                            is_strange = (abs(last_clock - 300.) > 1) and (abs(last_clock - 600.) > 1.)
                        strangeness = "(" + str(last_clock) + ")"
                        if is_strange:
                            strangeness = strangeness + " STRANGE!"
                            strangenesses[n, k, l, i_scene-1, i_trial-1] = 1
                    print("  scene_{}/trial_{}           ".format(i_scene, i_trial),
                          goal_reached_time, strangeness)

print()
print("--------------------------------------------------------------------")
print("--                         Gestures                               --")
print("--------------------------------------------------------------------")
print()

for k, plannername in enumerate(PLANNERS):
    planner_n_gestures = np.sum(ngestures[:, k, :, :, :])
    print(plannername, "n_gestures", planner_n_gestures)

if FULL_EVAL:
    print()
    print("--------------------------------------------------------------------")
    print("--                       Min Distances                            --")
    print("--------------------------------------------------------------------")
    print()
    mavgdmins_mask = np.isnan(avgdmins)
    mavgdmins = ma.MaskedArray(avgdmins, mavgdmins_mask)
    tbelowcd_mask = np.isnan(tbelowcd)
    mtbelowcd = ma.MaskedArray(tbelowcd, tbelowcd_mask)
    for k, plannername in enumerate(PLANNERS):
        planner_avg_avgdmin = np.mean(mavgdmins[:, k, :, :, :])
        print(plannername, "mean average_min_distance", planner_avg_avgdmin)
    print("t below dist")
    print("{| c", end="")
    for _ in range((N_SCENES+1) * len(EVAL_DISTANCES)):
        print("| c", end="")
    print("|}")
    print("\hline")
    N_COL = len(EVAL_DISTANCES)
    print(' & \multicolumn{' + str(N_COL) + '}{|c|}{Overall} ', end="")
    for i_scene in range(1, N_SCENES+1):
        print(' & \multicolumn{' + str(N_COL) + '}{|c|}{Scene ' + str(i_scene) + '} ', end="")
    print('\\'+'\\')
    print("\hline")
    print('           ', end="")
    for i_scene in range(1, N_SCENES+1 + 1):  # add 1 for "overall" multicol
        for symbol, i_dist in EVAL_DISTANCES:
            print("& $t_{" + symbol + "}$", end="")
    print('\\'+'\\')
    print("\hline")
    for k, plannername in enumerate(PLANNERS):
        print("{:<10} ".format(plannername), end="")
        for symbol, i_dist in EVAL_DISTANCES:
            planner_avgptbd = np.mean(
                mtbelowcd[:, k, :, :, :, i_dist] / mtbelowcd[:, k, :, :, :, -1])
            print("& {:.2f} ".format(planner_avgptbd), end="")
        for i_scene in range(1, N_SCENES+1):
            for symbol, i_dist in EVAL_DISTANCES:
                planner_scene_avgptbd = np.mean(
                    mtbelowcd[:, k, :, i_scene-1, :, i_dist] / mtbelowcd[:, k, :, i_scene-1, :, -1])
                print("& {:.2f} ".format(planner_scene_avgptbd), end="")
        print('\\'+'\\')
    print(', '.join(["${}$ = {}".format(symbol, COMFORT_DISTANCES[i_dist])
                     for symbol, i_dist in EVAL_DISTANCES]))

print()
print("--------------------------------------------------------------------")
print("--                     Average Time-to-goal                       --")
print("--------------------------------------------------------------------")
print()

mttgs_mask = np.logical_or(np.isnan(ttgs), ttgs == np.inf)
is_successes = np.logical_not(mttgs_mask)
is_trials = np.logical_not(np.isnan(ttgs))
mttgs = ma.MaskedArray(ttgs, mttgs_mask)

for k, plannername in enumerate(PLANNERS):
    planner_avgttg = np.mean(mttgs[:, k, :, :, :])
    n_ = np.sum(is_trials[:, k, :, :, :])
    planner_avgsr = "--"
    if n_ != 0:
        planner_avgsr = str(np.sum(is_successes[:, k, :, :, :]) * 1. / n_)
    print(plannername, "ttg", planner_avgttg, "sr", planner_avgsr, "n", n_)
    for l, mapname in enumerate(MAPNAMES):
        planner_map_avgttg = np.mean(mttgs[:, k, l, :, :])
        n_ = np.sum(is_trials[:, k, l, :, :])
        planner_map_avgsr = "--"
        if n_ != 0:
            planner_map_avgsr = str(np.sum(is_successes[:, k, l, :, :]) * 1. / n_)
        print(" ", mapname, "ttg", planner_map_avgttg, "sr", planner_map_avgsr, "n", n_)
        for i_scene in range(1, N_SCENES+1):
            planner_map_scene_avgttg = np.mean(mttgs[:, k, l, i_scene-1, :])
            n_trials = np.sum(is_trials[:, k, l, i_scene-1, :])
            planner_map_scene_avgsr = "--"
            if n_trials != 0:
                planner_map_scene_avgsr = np.sum(is_successes[:, k, l, i_scene-1, :] * 1. / n_trials)
            print("    scene_{} ttg".format(i_scene), planner_map_scene_avgttg,
                  "sr", planner_map_scene_avgsr, "n", n_trials)

print()
print("LATEX ---------------------------------------------------------------")
print()

# ttg
print("TTGs")
print("\hline")
print("           & Overall    ", end="")
for i_scene in range(1, N_SCENES+1):
    print("& Scene {}    ".format(i_scene), end="")
print('\\'+'\\')
print("\hline")
for k, plannername in enumerate(PLANNERS):
    planner_avgttg = np.mean(mttgs[:, k, :, :, :])
    print("{:<10} & {:>10.2f} ".format(plannername, planner_avgttg), end="")
    for i_scene in range(1, N_SCENES+1):
        planner_scene_avgttg = np.mean(mttgs[:, k, :, i_scene-1, :])
        print("& {:>10.2f} ".format(planner_scene_avgttg), end="")
    print('\\'+'\\')


# ttg
print("TTGs with STDDev")
print("\hline")
print("           & Overall    ", end="")
for i_scene in range(1, N_SCENES+1):
    print("& Scene {}    ".format(i_scene), end="")
print('\\'+'\\')
print("\hline")
for k, plannername in enumerate(PLANNERS):
    planner_avgttg = np.mean(mttgs[:, k, :, :, :])
    planner_stddevttg = np.std(mttgs[:, k, :, :, :])
    print("{:<10} & {:>10.2f} +- {:>10.2f} ".format(plannername, planner_avgttg, planner_stddevttg), end="")
    for i_scene in range(1, N_SCENES+1):
        planner_scene_avgttg = np.mean(mttgs[:, k, :, i_scene-1, :])
        planner_scene_stddevttg = np.std(mttgs[:, k, :, i_scene-1, :])
        print("& {:>10.2f} +- {:>10.2f} ".format(planner_scene_avgttg, planner_scene_stddevttg), end="")
    print('\\'+'\\')

print()
print("Success rates")
print("\hline")
print("           & Overall    ", end="")
for i_scene in range(1, N_SCENES+1):
    print("& Scene {}    ".format(i_scene), end="")
print('\\'+'\\')
print("\hline")
for k, plannername in enumerate(PLANNERS):
    n_ = np.sum(is_trials[:, k, :, :, :])
    planner_avgsr = "--"
    if n_ != 0:
        planner_avgsr = "{:.2f}".format(np.sum(is_successes[:, k, :, :, :]) * 1. / n_)
    print("{:<10} & {:>10} ".format(plannername, planner_avgsr), end="")
    for i_scene in range(1, N_SCENES+1):
        n_trials = np.sum(is_trials[:, k, :, i_scene-1, :])
        planner_scene_avgsr = "--"
        if n_trials != 0:
            planner_scene_avgsr = "{:.2f}".format(np.sum(
                is_successes[:, k, :, i_scene-1, :] * 1. / n_trials))
        print("& {:>10} ".format(planner_scene_avgsr), end="")
    print('\\'+'\\')
print()

# PTG
dists = distances_to_goal_10sample_array[None, None, :, :, None] * np.logical_not(
    mttgs_mask
)

print("AVG PTG")
print("\hline")
print("           & Overall    ", end="")
for i_scene in range(1, N_SCENES+1):
    print("& Scene {}    ".format(i_scene), end="")
print('\\'+'\\')
print("\hline")
for k, plannername in enumerate(PLANNERS):
    # weird but it's the 'simplest' way to get weighted average
    planner_avgptg = np.sum(
        dists[:, k, :, :, :] /
        mttgs[:, k, :, :, :] * mttgs[:, k, :, :, :]
    ) / np.sum(mttgs[:, k, :, :, :])
    print("{:<10} & {:>10.2f} ".format(plannername, planner_avgptg), end="")
    for i_scene in range(1, N_SCENES+1):
        planner_scene_avgptg = np.sum(
            dists[:, k, :, i_scene-1, :] /
            mttgs[:, k, :, i_scene-1, :] * mttgs[:, k, :, i_scene-1, :]
        ) / np.sum(mttgs[:, k, :, i_scene-1, :])
        print("& {:>10.2f} ".format(planner_scene_avgptg), end="")
    print('\\'+'\\')

print()
k = 0  # IA
for l, mapname in enumerate(MAPNAMES):
    print("per scene avg ptg ({}):".format(mapname))
    ptg = np.sum(
        dists[:, k, l, :, :] /
        mttgs[:, k, l, :, :] * mttgs[:, k, l, :, :]
    ) / np.sum(mttgs[:, k, l, :, :])
    print("Overall:", ptg)
    for i_scene in range(1, N_SCENES+1):
        ptg = np.sum(
            dists[:, k, l, i_scene-1, :] /
            mttgs[:, k, l, i_scene-1, :] * mttgs[:, k, l, i_scene-1, :]
        ) / np.sum(mttgs[:, k, l, i_scene-1, :])
        print("Scene " + str(i_scene) + ": {:>10.2f} m/s".format(ptg))



# strangeness
if np.any(strangenesses):
    print("Strangeness detected!")
else:
    print("No strangeness detected.")
