import numpy as np
import rospy

class Evaluator(object):
    def __init__(self):
        from geometry_msgs.msg import Twist
        rospy.Subscriber("/cmd_vel", Twist, self.cmd_vel_cb, queue_size=1)
        self.cmd_vel_msgs = []

    def cmd_vel_cb(self, msg):
        self.cmd_vel_msgs.append(msg)

if __name__=="__main__":
    Evaluator()
