from matplotlib import pyplot as plt
import numpy as np
import os
import sys
import rosbag
from pose2d import Pose2D, apply_tf

# parameters
TRANSFORM_TO_FIX = False
MORE_PLOTS = False

# get some constants
HOME = os.path.expanduser("~/")
ROSBAG_DIR = "rosbags/openlab_rosbags/" 
BAGNAMES = ["manip_corner", "corridor_koze_kids", "koze", "koze_circuit"]
BAG_SUFFIX = ".bag"
if len(sys.argv) >= 2:
    HOME = ""
    ROSBAG_DIR = "" 
    BAGNAMES = sys.argv[1:]
    BAG_SUFFIX = ""
    

TF_BAG = False
if TRANSFORM_TO_FIX:
    try:
        import tf_bag
        TF_BAG = True
        FIXED_FRAME = "gmap"
        print("Transform to reference frame enabled")
    except ImportError:
        print("Transform to reference frame disabled")


if __name__ == "__main__":

    logdata = {
        'Run #': [],
        'Run Time': [],
        'Not-stopped Time': [],
        'Average vel (odometry, filtered)': [],
        'Average vel (odometry)': [],
        'Start time': [],
        'Rosbag': [],
        'Goal Reached': [],
    }
    run_number = 1

    plt.ion()

    for bagname in BAGNAMES:
        ROSBAG = HOME + ROSBAG_DIR + bagname + BAG_SUFFIX
        bag = rosbag.Bag(ROSBAG)

        print(bagname)
        print("------------------------------------------------------------------------")

        # CMD VEL
        cmd_ts = None
        if True:
            cmd_vel_msgs = []
            cmd_vel_ts = []
            for topic, msg, t in bag.read_messages(topics=[
                    '/cmd_vel',
            ]):
                cmd_vel_msgs.append(msg)
                cmd_vel_ts.append(t)

            cmd_ts = np.zeros((len(cmd_vel_msgs),))
            cmd_xs = np.zeros((len(cmd_vel_msgs),))
            cmd_ys = np.zeros((len(cmd_vel_msgs),))
            for i, t in enumerate(cmd_vel_ts):
                cmd_ts[i] = t.to_sec()
            for i, msg in enumerate(cmd_vel_msgs):
                cmd_xs[i] = msg.linear.x
                cmd_ys[i] = msg.linear.y

        # Goals
        g_ts = None
        if True:
            goal_msgs = []
            goal_ts = []
            for topic, msg, t in bag.read_messages(topics=[
                    '/move_base_simple/goal',
            ]):
                goal_msgs.append(msg)
                goal_ts.append(t)
            g_ts = np.zeros((len(goal_ts),))
            g_xs = np.zeros((len(goal_msgs),))
            g_ys = np.zeros((len(goal_msgs),))
            for i, msg in enumerate(goal_msgs):
                g_ts[i] = msg.header.stamp.to_sec()
                g_xs[i] = msg.pose.position.x
                g_ys[i] = msg.pose.position.y
                if TF_BAG:
                    bag_transformer = tf_bag.BagTfTransformer(bag)
                    p2_msg_in_ref = Pose2D(bag_transformer.lookupTransform(
                        FIXED_FRAME, msg.header.frame_id, msg.header.stamp))
                    g_xs[i], g_ys[i] = apply_tf(np.array([[g_xs[i], g_ys[i]]]), p2_msg_in_ref)[0]

            print(".")

        # Position
        if True:
            odom_msgs = []
            odom_ts = []
            for topic, msg, t in bag.read_messages(topics=[
                    '/pepper_robot/odom',
            ]):
                odom_msgs.append(msg)
                odom_ts.append(t)

            ts = np.zeros((len(odom_ts),))
            xs = np.zeros((len(odom_msgs),))
            ys = np.zeros((len(odom_msgs),))
            us = np.zeros((len(odom_msgs),))
            vs = np.zeros((len(odom_msgs),))
            for i, msg in enumerate(odom_msgs):
                ts[i] = msg.header.stamp.to_sec()
                xs[i] = msg.pose.pose.position.x
                ys[i] = msg.pose.pose.position.y
                us[i] = msg.twist.twist.linear.x
                vs[i] = msg.twist.twist.linear.y
                if TF_BAG:
                    bag_transformer = tf_bag.BagTfTransformer(bag)
                    p2_msg_in_ref = Pose2D(bag_transformer.lookupTransform(
                        FIXED_FRAME, msg.header.frame_id, msg.header.stamp))
                    xs[i], ys[i] = apply_tf(np.array([[xs[i], ys[i]]]), p2_msg_in_ref)[0]
                    print(i)

            # norm of velocity vector
            vnorm = np.sqrt(us**2 + vs**2)
            # detect stopped times (no cmd vel for >5s)
            isnotstopped = np.where(vnorm != 0)
            stop_durations = np.diff(ts[isnotstopped])
            # outliers
            long_stop_durations = stop_durations[stop_durations > 5]
            long_stop_times = ts[isnotstopped][:-1][stop_durations > 5]
            total_time = ts[-1] - ts[0]
            not_stopped_time = total_time - np.sum(long_stop_durations)
            print("Total Time: {} [s]".format(total_time))
            print("Not-stopped Time: {} [s]".format(not_stopped_time))
            # time downsample filter
            fxs = xs[::100]
            fys = ys[::100]
            fts = ts[::100]
            dfxs = np.diff(fxs)
            dfys = np.diff(fys)
            dfxys = np.vstack([dfxs, dfys]).T
            dfns = np.linalg.norm(dfxys, axis=-1)
            avgvel_filtered = np.sum(dfns) / not_stopped_time
            print("Average vel (odometry, filtered): {} [m/s]".format(avgvel_filtered))
            avgvel = np.mean(vnorm)
            print("Average vel (odometry): {} [m/s]".format(avgvel))

            # distance to goal
            if g_ts is not None:
                dtgs = np.zeros((len(goal_msgs),))
                set_times = np.concatenate([[-np.inf], g_ts])
                set_goalx = np.concatenate([[np.nan], g_xs])
                set_goaly = np.concatenate([[np.nan], g_ys])
                elapsed_since_goalmsg = ts[:, None] - set_times[None, :]
                elapsed_since_goalmsg[elapsed_since_goalmsg < 0] = np.inf
                current_goal_indice = np.argmin(elapsed_since_goalmsg, axis=-1)
                last_gx = set_goalx[current_goal_indice]
                last_gy = set_goaly[current_goal_indice]
                dtgs = np.sqrt((xs - last_gx) ** 2 + (ys - last_gy) ** 2)

            # segment bag into runs
            if g_ts is not None:
                run_start_indices = []
                last_idx = None
                for i, idx in enumerate(current_goal_indice):
                    if idx != last_idx:
                        run_start_indices.append(i)
                    last_idx = idx
                run_end_indices = [None for _ in run_start_indices]
                for i in range(len(run_start_indices)):
                    if i == len(run_start_indices)-1:
                        run_end_indices[i] = len(ts)
                    else:
                        run_end_indices[i] = run_start_indices[i+1]
                # print run info
                for start, end in zip(run_start_indices, run_end_indices):
                    run_ts = ts[start:end]
                    run_xs = xs[start:end]
                    run_ys = ys[start:end]
                    run_us = us[start:end]
                    run_vs = vs[start:end]
                    # norm of velocity vector
                    run_vnorm = np.sqrt(run_us**2 + run_vs**2)
                    # detect stopped times (no cmd vel for >5s)
                    run_isnotstopped = np.where(run_vnorm != 0)
                    run_stop_durations = np.diff(run_ts[run_isnotstopped])
                    # outliers
                    run_long_stop_durations = run_stop_durations[run_stop_durations > 5]
                    run_long_stop_times = run_ts[run_isnotstopped][:-1][run_stop_durations > 5]
                    run_total_time = run_ts[-1] - run_ts[0]
                    run_not_stopped_time = run_total_time - np.sum(run_long_stop_durations)
                    # ignore spammed messages runs
                    if run_total_time < 3.:
                        continue
                    print("  Run #: {}:".format(run_number))
                    print("    Run Time: {} [s]".format(run_total_time))
                    print("    Not-stopped Time: {} [s]".format(run_not_stopped_time))
                    # time downsample filter
                    run_fxs = run_xs[::100]
                    run_fys = run_ys[::100]
                    run_fts = run_ts[::100]
                    run_dfxs = np.diff(run_fxs)
                    run_dfys = np.diff(run_fys)
                    run_dfxys = np.vstack([run_dfxs, run_dfys]).T
                    run_dfns = np.linalg.norm(run_dfxys, axis=-1)
                    run_avgvel_filtered = np.sum(run_dfns) / run_not_stopped_time
                    print("    Average vel (odometry, filtered): {} [m/s]".format(run_avgvel_filtered))
                    run_avgvel = np.mean(run_vnorm)
                    print("    Average vel (odometry): {} [m/s]".format(run_avgvel))
                    logdata['Run #'].append(run_number)
                    logdata['Run Time'].append(run_total_time)
                    logdata['Not-stopped Time'].append(run_not_stopped_time)
                    logdata['Average vel (odometry, filtered)'].append(run_avgvel_filtered)
                    logdata['Average vel (odometry)'].append(run_avgvel)
                    logdata['Start time'].append(ts[start])
                    logdata['Rosbag'].append(bagname)
                    logdata['Goal Reached'].append(True)
                    run_number += 1

            # --- PLOTTING -----------
            if MORE_PLOTS:
                if cmd_ts is not None:
                    # cmd vel chart
                    plt.figure()
                    plt.scatter(cmd_ts, cmd_xs, marker='+')
                    plt.scatter(cmd_ts, cmd_ys, marker='+')
                    plt.scatter(cmd_ts, np.zeros_like(cmd_xs), marker='+')
                    plt.show()

                # stoppage times
                plt.figure()
                plt.plot(ts[isnotstopped][:-1], np.diff(ts[isnotstopped]))
                plt.scatter(long_stop_times, long_stop_durations, color='r')
                plt.title("Time spent stopped (cmd_vel = 0)")

                # top down view
                plt.figure()
                plt.scatter(fxs, fys, marker='o', edgecolors='k', facecolors=(0, 0, 0, 0))
                plt.scatter(xs, ys, marker=',', s=1, facecolors='k', edgecolors=(0, 0, 0, 0))
                plt.title("Top down view")

            # time plot of vel
            plt.figure()
            plt.plot(ts, us)
            plt.plot(ts, vs)
            if g_ts is not None:
                for t in g_ts:
                    plt.axvline(t, color='k')
            plt.plot(ts, dtgs)
            plt.title("Velocity")

            plt.show()
            plt.pause(2.)

    import pandas as pd
    logdataframe = pd.DataFrame(data=logdata)
    logdataframe.to_csv("/tmp/avgvel_eval_log.csv")
