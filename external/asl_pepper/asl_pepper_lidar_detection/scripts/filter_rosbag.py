from __future__ import print_function
import rosbag
import numpy as np
from matplotlib import pyplot as plt
import os
from dr_spaam.detector import Detector
from copy import deepcopy
from collections import deque

PLOTTING = True
SCRUB_HISTORY = 10
cmap = plt.get_cmap("viridis")
rosbag_path = os.path.expanduser("~/rosbags/openlab_rosbags/corridor_koze_kids.bag")
ckpt = os.path.expanduser("~/Code/DR-SPAAM-Detector/dr_spaam/ckpts/dr_spaam_e40.pth")
detector = Detector(ckpt, original_drow=False, gpu=True, stride=1)

# set angular grid (this is only required once)
ang_inc = 0.00581718236208  # np.radians(1./3.)  # angular increment of the scanner
num_pts = 811  # 1080  # number of points in a scan
angles = np.arange(num_pts) * ang_inc - np.pi / 4
detector.set_laser_spec(ang_inc, num_pts)

cls_thresh = 0.4  # confidence threshold
scrub_radius = 0.4  # how far from the detection to scrub points
scrub_memory = 10  # number of scans to scrub for each detections (10 == ~1s)

# open rosbag
bag = rosbag.Bag(rosbag_path)

if PLOTTING:
    plt.ion()
    f, (ax1, ax2) = plt.subplots(2, 1)
    f2, (ax3, ax4) = plt.subplots(2, 1)

past_scrubmasks = deque(maxlen=scrub_memory)

t0 = None
# inference
with rosbag.Bag("scrubbed.bag", "w") as outbag:
    for topic, msg, t in bag.read_messages():
        # This also replaces tf timestamps under the assumption
        # that all transforms in the message share the same timestamp
        if topic in [
            "/sick_laser_front/scan",
            "/sick_laser_rear/scan",
        ]:
            scan = np.array(msg.ranges)  # scan is a 1D numpy array with positive values
            dets_xy, dets_cls, instance_mask = detector(scan)  # get detection

            xx = -scan * np.cos(angles)
            yy = scan * np.sin(angles)

            if PLOTTING:
                ax1.cla()
                ax1.scatter(xx, yy, color="red", s=1)
                ax1.scatter(dets_xy[:, 0], dets_xy[:, 1], c=cmap(dets_cls[:, 0]))
                ax1.axis("equal")
                ordered = sorted(dets_cls[:, 0])[::-1]
                ax2.cla()
                ax2.axhline(cls_thresh, color="k", zorder=-1000)
                ax2.bar(range(len(dets_cls)), ordered, color=cmap(ordered))

            cls_mask = dets_cls > cls_thresh
            pers_xy = dets_xy[cls_mask[:, 0], :]
            pers_cls = dets_cls[cls_mask]

            if PLOTTING:
                for (x, y), p in zip(pers_xy, pers_cls):
                    c = plt.Circle((x, y), scrub_radius, color=cmap(p), fill=False)
                    ax1.add_artist(c)
                plt.pause(0.01)

            deltas = np.sqrt(
                (xx[None, :] - pers_xy[:, 0][:, None]) ** 2
                + (yy[None, :] - pers_xy[:, 1][:, None]) ** 2
            )
            scrubmask = np.any(deltas < scrub_radius, axis=0)

            if False:
                ax3.cla()
                ax3.scatter(xx, yy, color="red", s=1)
                ax3.axis("equal")
                ax4.cla()
                ax4.scatter(xx[scrubmask], yy[scrubmask], color="grey", alpha=0.3, s=1)
                ax4.scatter(xx[np.logical_not(scrubmask)], yy[np.logical_not(scrubmask)], color="red", s=1)
                ax4.axis("equal")

            newscan = scan.copy()
            newscan[scrubmask] = np.inf
            # also scrub using N previous detections
            for oldmask in past_scrubmasks:
                newscan[oldmask] = np.inf
            past_scrubmasks.append(scrubmask)

            newmsg = deepcopy(msg)
            newmsg.ranges = tuple(newscan)

            outbag.write(topic, newmsg, t)  # replace original with scrubbed
            outbag.write("/nonscrubbed" + topic, msg, t)  # save original
        else:
            outbag.write(topic, msg, t)

        if t0 is None:
            t0 = t
        print("{:.1f}".format(t.to_sec() - t0.to_sec()), end='\r')
