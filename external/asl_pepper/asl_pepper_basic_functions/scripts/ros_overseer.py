# -*- encoding: UTF-8 -*-

import time
import argparse
from naoqi import ALProxy

def main(robotIP, PORT=9559, rosIP):
    motionProxy  = ALProxy("ALMotion", robotIP, PORT)
    postureProxy = ALProxy("ALRobotPosture", robotIP, PORT)
    basicAwarenessProxy = ALProxy("ALBasicAwareness", robotIP, PORT)

    # Initialize robot for autonomous navigation
    motionProxy.setExternalCollisionProtectionEnabled("All", 0)
    basicAwarenessProxy.pauseAwareness()

    # Check that connection stands
    # ping rosIP
    time.sleep(3)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pepper-ip", type=str, default="127.0.0.1",
                        help="Robot ip address")
    parser.add_argument("--port", type=int, default=9559,
                        help="Robot port number")
    parser.add_argument("--ros-ip", type=str, default="10.42.0.168",
                        help="External ros computer ip adress")

    args = parser.parse_args()
    main(args.pepper_ip, args.port, args.ros_ip)
