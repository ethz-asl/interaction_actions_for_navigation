from __future__ import print_function
from matplotlib import pyplot as plt
import numpy as np
import sys
import rosbag
from tf_bag import BagTfTransformer
from pose2d import Pose2D
from visualization_msgs.msg import Marker
import pyIA.actions as actions

# parameters
bagpaths = sys.argv[1:]


plt.ion()

finished_tasks = []
for bagpath in bagpaths:

    bag = rosbag.Bag(bagpath)

    bag_transformer = BagTfTransformer(bag)

    print(bagpath)
    print("------------------------------------------------------------------------")

    # Gestures
    n_gestures = 0
    for topic, msg, t in bag.read_messages(topics=['/gestures', ]):
        if msg.data == "animations/Stand/Gestures/You_2":
            n_gestures += 1

    n_say = 0
    for topic, msg, t in bag.read_messages(topics=['/speech', ]):
        if "xcuse me" in msg.data:
            n_say += 1

    currenttask = None
    currenttask_succeeded = False
    for topic, msg, t in bag.read_messages(topics=['/ros_ia_node/debug/task_being_executed', ]):
        # extract task or reject invalid
        if not msg.markers:
            print("no markers!")
            continue
        task_marker = msg.markers[0]
        try:
            rob_in_msg = Pose2D(bag_transformer.lookupTransform(
                task_marker.header.frame_id, "base_footprint", t))
        except RuntimeError:
            continue
        # add robot position to real path
        if currenttask is not None:
            currenttask["realpath"].append(rob_in_msg)
        taskaction = None
        if task_marker.color.g == 1.:
            taskaction = "Intend"
        elif task_marker.color.b == 1.:
            taskaction = "Say"
        elif task_marker.color.r == 1.:
            taskaction = "Nudge"
        if not task_marker.points:
            print("empty taskpath!")
            continue
        taskpath = np.array([[p.x, p.y] for p in task_marker.points])
        if taskaction is None and task_marker.action != Marker.DELETEALL:
            print("can't parse task from marker")
            continue
        # check success of current task
        if currenttask is not None and not currenttask_succeeded:
            if np.linalg.norm(rob_in_msg[:2] - currenttask["path"][-1]) < 0.5:
                currenttask_succeeded = True
        # check if task has changed
        if currenttask is not None:
            if taskaction == currenttask["action"]:
                if len(taskpath) == len(currenttask["path"]):
                    if np.allclose(taskpath, currenttask["path"]):
                        continue
        else:
            if taskaction is None:
                continue
        # save old task
        if currenttask is not None:
            currenttask["success"] = currenttask_succeeded
            if currenttask_succeeded:
                print("task succeeded")
            else:
                print("task failed")
            finished_tasks.append(currenttask)
        if taskaction is not None:
            # start new task
            print("new task:", taskaction)
            currenttask = {"path": taskpath, "action": taskaction, "realpath": []}
            currenttask_succeeded = False


actions_taken = set([t["action"] for t in finished_tasks])
log = [[t["action"], t["success"]] for t in finished_tasks]

for name in actions_taken:
    action_successes = [t["success"] for t in finished_tasks if t["action"] == name]
    n_total = len(action_successes)
    n_succ = np.sum(action_successes)
    sr = n_succ * 1. / n_total
    print(name, n_total, sr)


print("according to gestures and speech topics")
print("nudge gestures:", n_gestures)
print("say speeches:", n_say)

plt.figure()
for task in finished_tasks:
    path = np.array(task["realpath"])
    action_str = task["action"]
    if action_str == "Intend":
        action = actions.Intend()
    if action_str == "Say":
        action = actions.Say()
    if action_str == "Nudge":
        action = actions.Nudge()
    plt.plot(path[:, 0], path[:, 1], color=action.color())
plt.axis('equal')
plt.show()
