import rospy
import tf
from sensor_msgs.msg import Image, CameraInfo
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from matplotlib import pyplot as plt
from IPython.display import clear_output
from subprocess import Popen
import sys
import os
home = os.path.expanduser("~")
script_dir = os.path.dirname(os.path.realpath(__file__))


class Listener(object):
    def __init__(self, max_frames=10):
        self.fixed_frame = 'SurroundingFrontLaser_frame'
        self.depth_topic = "/pepper_robot/camera/depth/image_raw"
        self.ir_topic = "/pepper_robot/camera/ir/image_raw"
        self.front_topic = "/pepper_robot/camera/front/image_raw"
        self.depth_info_topic = "/pepper_robot/camera/depth/camera_info"
        self.frames = []
        self.tfs = []
        self.cam_info = None
        self.MAX_FRAMES = max_frames

        rospy.init_node('listener', anonymous=False)
        #self.image_sub = rospy.Subscriber(self.depth_topic,
        #                                  Image, self._depth_image_callback)
        #self.cam_info_sub = rospy.Subscriber(self.depth_info_topic,
        #                                    CameraInfo, self._cam_info_callback)
        #self.tf_listener = tf.TransformListener()
        self.ir_sub = rospy.Subscriber(self.ir_topic, Image, self._ir_callback)
        self.front_sub = rospy.Subscriber(self.front_topic, Image, self._front_callback)

    def spin(self):
        # spin() simply keeps python from exiting until this node is stopped
        rospy.spin()

    def _cam_info_callback(self, data):
        if self.cam_info is None:
            self.cam_info = data

    def _depth_image_callback(self, data):
        rospy.loginfo("Frame: %s " +  rospy.get_caller_id() + "I heard %s", len(self.frames), data.width)
        clear_output(wait=True)
        try:
            tf_fixed_to_image = self.tf_listener.lookupTransform(self.fixed_frame, data.header.frame_id, rospy.Time(0))
            self.tfs.append(tf_fixed_to_image)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            print("tf not found")
        self.frames.append(data)
        if len(self.frames) > self.MAX_FRAMES:
          rospy.signal_shutdown("Message received")

    def _ir_callback(self, data):
        self.latest_ir = data

    def _front_callback(self, data):
        self.latest_front = data

def write_images(listener):
    cvbridge = CvBridge()
    ir_directory = os.path.join(script_dir, "../calibration_data/cam0/")
    front_directory = os.path.join(script_dir, "../calibration_data/cam1/")
    if not os.path.exists(ir_directory):
      os.makedirs(ir_directory)
    if not os.path.exists(front_directory):
      os.makedirs(front_directory)
    while True:
        if raw_input("Press enter to save frames, write exit to quit: ") == "exit":
            rospy.signal_shutdown("Quit")
            raise KeyboardInterrupt
        cv_ir = cvbridge.imgmsg_to_cv2(listener.latest_ir, "").astype('uint8')
        cv_front = cvbridge.imgmsg_to_cv2(listener.latest_front, "")
        timestr = "{}".format(listener.latest_front.header.stamp.to_nsec())
        ir_filename = ir_directory+timestr+".png"
        front_filename = front_directory+timestr+".png"
        cv2.imwrite(ir_filename, cv_ir)
        print("  saved ir image as {}".format(ir_filename))
        cv2.imwrite(front_filename, cv_front)
        print("  saved front image as {}".format(front_filename))

if __name__ == '__main__':
    from threading import Thread
    l = Listener()
    t = Thread(target=write_images, args=(l,))
    try:
      t.start()
      l.spin()
    except KeyboardInterrupt:
      print("Ended")
    t.join()
