#!/usr/bin/env python
from __future__ import print_function
from functools import partial
import numpy as np
import os
import threading
from timeit import default_timer as timer
from CMap2D import CSimAgent, CMap2D
from CMap2D import fast_2f_norm, fast_3f_clip
import pose2d
from collections import deque

from math import floor, cos, sin, sqrt

class Raytracer(object):
    def __init__(self, map2d, GPU=False, silent=True):
        import range_libc
        self.MAX_DIST_IJ = map2d.occupancy().shape[0] * 2
        self.map2d = map2d
        if not silent: print("Creating PyOmap")
        self.pyomap = range_libc.PyOMap(
                np.ascontiguousarray(map2d.occupancy().T >= map2d.thresh_occupied()))
        if not silent: print("Creating PyRayMarching")
        self.rmnogpu = range_libc.PyRayMarching(self.pyomap, self.MAX_DIST_IJ)
        if GPU:
            rmgpu = range_libc.PyRayMarchingGPU(self.pyomap, self.MAX_DIST_IJ)
            self.rm = rmgpu
        else:
            self.rm = self.rmnogpu


    def raymarch(self, angles, origin_ij, ranges):
        xythetas = np.zeros((len(ranges), 3))
        xythetas[:, 0] = origin_ij[0]
        xythetas[:, 1] = origin_ij[1]
        xythetas[:, 2] = angles
        xythetas = xythetas.astype(np.float32)
        self.rm.calc_range_many(xythetas, ranges)
        ranges *= self.map2d.resolution()

    def get_dist(self):
        min_distances = np.ascontiguousarray(np.zeros_like(self.map2d.occupancy(), dtype=np.float32))
        self.rmnogpu.get_dist(min_distances)
        return min_distances

class Virtual2DPepper(object):
    def __init__(self, map2d, raytracer=None, aggressive_locomotion=True, debug=False, no_inertia=False):
        # Initialize variables
        self.lock = threading.Lock() # for avoiding race conditions
        # Initialize constants
        if aggressive_locomotion:
            self.set_move_config_aggressive()
        else:
            self.set_move_config_default()
        self.reset()
        self.map2d = map2d # occupancy grid
        self.kCmdVelTopic = "/cmd_vel"
        self.kBaseLinkFrame = "base_link"
        self.kBaseFootprintFrame = "base_footprint"
        self.kOdomFrame = "odom"
        self.kSimFrame = "sim_map"
        self.kLidarFrontTopic = "/sick_laser_front"
        self.kLidarRearTopic = "/sick_laser_rear"
        self.kLidarMergedTopic = "/merged_lidar"
        self.kOdomTopic = "/pepper_robot/odom"
        self.kTrackedPersonsTopic = "/rwth_tracker/tracked_persons"
        self.kNodeName = "pepper_2d_simulator"
        self.set_lidar_scan_parameters()
        self.set_simulation_frequency_parameters()
        self.set_odom_parameters()
        self.pub_lidar_front = None
        self.pub_lidar_rear = None
        self.pub_cloud_rear = None
        self.pub_odom = None
        self.tf_br = None
        self.DEBUG = debug
        self.PUBLISH_REAR_POINTCLOUD = False
        self.NO_INERTIA = no_inertia
        if raytracer is None:
            raytracer = Raytracer(map2d)
        self.raytracer = raytracer

    def reset(self):
        with self.lock:
            # x axis points "forward" from the robot, y points "left"
            self.acc = np.zeros((3,)) # x y theta
            self.target_vel = np.zeros((3,)) # x y theta
            self.vel = np.zeros((3,)) # x y theta
            self.pos = np.zeros((3,)) # x y theta, stores own pose2d in map frame, i.e m_T_self
            self.odom_in_map_frame = np.zeros((3,))
            # convenience variables
            self.vel_in_base_frame = np.zeros((3,))
            self.distance_travelled_in_base_frame = np.zeros((3,))

    def set_simulation_frequency_parameters(self):
        self.kLidarPeriod = 0.05 # Publish simulated lidar at 10hz
        self.kOdomPeriod = 0.05 # Publish simulated odom at 10hz
        self.kCmdVelDelay = 0.05
        self.kTransformPublishDelay = 0.00 # Same as in gmapping

    def rosify_simulation_frequency_parameters(self):
        self.kLidarPeriod = self.rospy.Duration(self.kLidarPeriod) # Publish simulated lidar at 10hz
        self.kOdomPeriod = self.rospy.Duration(self.kOdomPeriod) # Publish simulated odom at 10hz
        self.kCmdVelDelay = self.rospy.Duration(self.kCmdVelDelay)
        self.kTransformPublishDelay = self.rospy.Duration(self.kTransformPublishDelay) # Same as in gmapping

    def set_lidar_scan_parameters(self):
        self.kLidarAngleIncrement = 0.00581718236208
        self.kLidarMinAngle = -2.35619449615
        self.kLidarMaxAngle = 2.35572338104
        self.kLidarMergedMinAngle = 0
        self.kLidarMergedMaxAngle = 6.27543783188 + self.kLidarAngleIncrement
        self.kTFBaseLinkToLidarFront = np.array([0.245, 0, 0]) # x y theta
        self.kTFBaseLinkToLidarRear = np.array([-0.245, 0, np.pi])
        self.kTFBaseLinkToLidarMerged = np.array([0., 0., 0.]) # virtual merged lidar at base_footprint
        self.kLidarScanTime = 0.0666666701436 # empirical
        self.kLidarMinRange = 0.05 # based on published values
        self.kLidarMaxRange = 100. # meters, based on published values
        self.kLidarFrontFrame = "sick_laser_front"
        self.kLidarRearFrame = "sick_laser_rear"
        self.kLidarMergedFrame = self.kBaseFootprintFrame

    def set_odom_parameters(self):
        ii_cxy = 0.1   # translational covariance for same-axis [m^2]
        ii_cth = 0.001 # rotational covariance for same-axis [rad^2]
        self.kOdomCovariance = [
                ii_cxy,      0,      0,      0,      0,      0,
                     0, ii_cxy,      0,      0,      0,      0,
                     0,      0,      0,      0,      0,      0,
                     0,      0,      0,      0,      0,      0,
                     0,      0,      0,      0,      0,      0,
                     0,      0,      0,      0,      0, ii_cth,
                     ]

    def set_move_config_default(self):
        self.kMaxVel = np.array([0.35, 0.35, 1.]) # x y theta
        self.kPrefVel = np.array([0.2, 0.2, 0.75]) # x y theta
        self.kMaxAcc  = np.array([0.3, 0.3, 0.75]) # x y theta
        self.kMaxJerk = np.array([1., 1., 2.]) # x y theta

    def set_move_config_aggressive(self):
        self.kMaxVel = np.array([0.55, 0.55, 2.]) # x y theta
        self.kPrefVel = np.array([0.35, 0.35, 1.]) # x y theta
        self.kMaxAcc  = np.array([0.55, 0.55, 3.]) # x y theta
        self.kMaxJerk = np.array([5., 5., 50.]) # x y theta

    def cmd_vel_callback(self, msg):
        if self.DEBUG:
            print("Moving toward: {}, {}, {}".format(msg.linear.x, msg.linear.y, msg.angular.z))
        self.rospy.sleep(self.kCmdVelDelay) # simulate naoqi delay
        self.set_cmd_vel([msg.linear.x, msg.linear.y, msg.angular.z])

    def set_cmd_vel(self, cmd_vel):
        """
        cmd_vel should be of shape (3,), in SI units
        """
        with self.lock:
            self.target_vel = fast_3f_clip(np.array(cmd_vel),
                                      -self.kMaxVel,
                                      self.kMaxVel)

    def odom_callback(self, event=None, dt=None, publish_tf=True):
        from nav_msgs.msg import Odometry
        from geometry_msgs.msg import Twist, Quaternion
        if dt is None:
            dt=self.kOdomPeriod.to_sec()
        self.odom_update(dt=dt)
        if publish_tf:
            # Publish odom tf
            with self.lock:
                map_in_odom_frame = pose2d.inverse_pose2d(self.odom_in_map_frame)
                pos_in_odom_frame = pose2d.apply_tf_to_pose(self.pos, map_in_odom_frame)
                vel_in_base_frame = np.copy(self.vel_in_base_frame)
            now = self.rospy.Time.now() + self.kTransformPublishDelay
            # sim_ to base link
            self.tf_br.sendTransform(
                (0, 0, 0),
                self.tf.transformations.quaternion_from_euler(0, 0, 0),
                now,
                self.kOdomFrame,
                self.kSimFrame,
            )
            # odom to base link
            self.tf_br.sendTransform(
                (pos_in_odom_frame[0], pos_in_odom_frame[1], 0),
                self.tf.transformations.quaternion_from_euler(0, 0, pos_in_odom_frame[2]),
                now,
                self.kBaseLinkFrame,
                self.kOdomFrame,
            )
            # also base_footprint
            self.tf_br.sendTransform(
                (0, 0, 0),
                self.tf.transformations.quaternion_from_euler(0, 0, 0),
                now,
                self.kBaseFootprintFrame,
                self.kBaseLinkFrame,
            )
            # lidar front
            self.tf_br.sendTransform(
                (self.kTFBaseLinkToLidarFront[0], self.kTFBaseLinkToLidarFront[1], 0),
                self.tf.transformations.quaternion_from_euler(0, 0, self.kTFBaseLinkToLidarFront[2]),
                now,
                self.kLidarFrontFrame,
                self.kBaseLinkFrame,
            )
            # lidar rear
            self.tf_br.sendTransform(
                (self.kTFBaseLinkToLidarRear[0], self.kTFBaseLinkToLidarRear[1], 0),
                self.tf.transformations.quaternion_from_euler(0, 0, self.kTFBaseLinkToLidarRear[2]),
                now,
                self.kLidarRearFrame,
                self.kBaseLinkFrame,
            )
            # odom on odom topic?
            odom_msg = Odometry()
            odom_msg.header.stamp.secs = now.secs
            odom_msg.header.stamp.nsecs = now.nsecs
            odom_msg.header.frame_id = self.kOdomFrame
            odom_msg.child_frame_id = self.kBaseFootprintFrame
            odom_msg.pose.pose.position.x = pos_in_odom_frame[0]
            odom_msg.pose.pose.position.y = pos_in_odom_frame[1]
            odom_msg.pose.pose.position.z = 0
            odom_msg.pose.pose.orientation = Quaternion(
                    *self.tf.transformations.quaternion_from_euler(0, 0, pos_in_odom_frame[2]))
            odom_msg.pose.covariance = self.kOdomCovariance
            odom_msg.twist.twist.linear.x = vel_in_base_frame[0]
            odom_msg.twist.twist.linear.y = vel_in_base_frame[1]
            odom_msg.twist.twist.linear.z = 0
            odom_msg.twist.twist.angular.x = 0
            odom_msg.twist.twist.angular.y = 0
            odom_msg.twist.twist.angular.z = vel_in_base_frame[2]
            odom_msg.twist.covariance = self.kOdomCovariance
            self.pub_odom.publish(odom_msg)
            # TODO odom velocities?

    def odom_update(self, dt=None):
        clip_flag = False
        with self.lock:
            if dt is None:
                dt = self.kOdomPeriod
            if self.NO_INERTIA:  # used for more 'interpretable' data in RL training
                target_vel_in_map_frame = pose2d.apply_tf_to_vel(self.target_vel, self.pos)
                # update velocity
                self.vel = target_vel_in_map_frame
                # update position
                self.pos += self.vel * dt
                # store for convenience
                self.vel_in_base_frame = self.target_vel * 1.
                self.distance_travelled_in_base_frame += self.vel_in_base_frame * dt
            else:  # default odom update
                # update position
                self.pos += self.vel * dt
                # update velocity
                self.vel += self.acc * dt
                # store for convenience
                self.distance_travelled_in_base_frame += self.vel_in_base_frame * dt
                self.vel_in_base_frame = pose2d.apply_tf_to_vel(self.vel, -self.pos)
                # update accelerations within jerk and acceleration constraints
                target_vel_in_map_frame = pose2d.apply_tf_to_vel(self.target_vel, self.pos)
                target_acc = (target_vel_in_map_frame - self.vel) / dt
                self.acc += fast_3f_clip(
                    target_acc - self.acc,
                    -self.kMaxJerk * dt,
                    self.kMaxJerk * dt,
                )
                self.acc = fast_3f_clip(
                    self.acc,
                    -self.kMaxAcc,
                    self.kMaxAcc,
                )
            if not self.map2d.is_inside_ij(
                        self.map2d.xy_to_floatij([self.pos[:2]], clip_if_outside=False)
                    )[0]:
#                 print("WARNING: robot is clipping against the simulation boundaries.")
                new_xy = self.map2d.ij_to_xy(
                        self.map2d.xy_to_ij([self.pos[:2]], clip_if_outside=True)[0])
                self.pos[0] = new_xy[0]
                self.pos[1] = new_xy[1]
                clip_flag = True
            if self.DEBUG:
                print("acc: {}\nvel: {}\nt_vel: {}\npos: {}".format(
                    self.acc, self.vel, self.target_vel, self.pos))
        return clip_flag

    def publish_scan(self, ranges, which_lidar):
        from sensor_msgs.msg import LaserScan, PointCloud2, PointField
        now = self.rospy.Time.now()

        if which_lidar not in ["front", "rear", "merged"]:
            raise NotImplementedError("which_lidar must be 'front', 'rear' or 'merged'.")

        if which_lidar != "merged":
            if which_lidar == "front":
                lidar_publisher = self.pub_lidar_front
                lidar_frame = self.kLidarFrontFrame
            elif which_lidar == "rear":
                lidar_publisher = self.pub_lidar_rear
                lidar_frame = self.kLidarRearFrame
            # publish LaserScan
            scan_msg = LaserScan()
            scan_msg.header.stamp.secs = now.secs
            scan_msg.header.stamp.nsecs = now.nsecs
            scan_msg.header.frame_id = lidar_frame
            scan_msg.angle_min = self.kLidarMinAngle
            scan_msg.angle_max = self.kLidarMaxAngle
            scan_msg.angle_increment = self.kLidarAngleIncrement
            scan_msg.scan_time = self.kLidarScanTime
            scan_msg.time_increment = self.kLidarScanTime / len(ranges)
            scan_msg.range_min = self.kLidarMinRange
            scan_msg.range_max = self.kLidarMaxRange
            scan_msg.ranges = list(ranges)
            scan_msg.intensities = list(np.ones(ranges.shape) * 10000.) # fake intensities
            lidar_publisher.publish(scan_msg)

        PUBLISH_CLOUD_IN_MAP_FRAME = True
        if PUBLISH_CLOUD_IN_MAP_FRAME:
            MIN_ANGLE = self.kLidarMinAngle
            MAX_ANGLE = self.kLidarMaxAngle
            if which_lidar == "front":
                lidarTF = self.kTFBaseLinkToLidarFront
                lidar_topic = self.kLidarFrontTopic
            elif which_lidar == "rear":
                lidarTF = self.kTFBaseLinkToLidarRear
                lidar_topic = self.kLidarRearTopic
            elif which_lidar == "merged":
                lidarTF = self.kTFBaseLinkToLidarMerged
                lidar_topic = self.kLidarMergedTopic
                MIN_ANGLE = self.kLidarMergedMinAngle
                MAX_ANGLE = self.kLidarMergedMaxAngle
            angles = np.arange(MIN_ANGLE, 
                               MAX_ANGLE,
                               self.kLidarAngleIncrement)
            points = ranges[:,None] * np.stack([np.cos(angles), 
                                                np.sin(angles),
                                                np.zeros_like(angles)+ 0.01], axis=-1)
            points_in_mapframe = np.copy(points)
            points_in_mapframe[:,:2] = pose2d.apply_tf(
                    pose2d.apply_tf(points[:, :2], lidarTF),
                    self.pos
                    )
            msg = PointCloud2()
            msg.header.stamp.secs = now.secs
            msg.header.stamp.nsecs = now.nsecs
            msg.header.frame_id = self.kOdomFrame
            msg.height = 1
            msg.width = len(points_in_mapframe)
            msg.fields = [
                PointField('x', 0, PointField.FLOAT32, 1),
                PointField('y', 4, PointField.FLOAT32, 1),
                PointField('z', 8, PointField.FLOAT32, 1)]
            msg.is_bigendian = False
            msg.point_step = 12
            msg.row_step = 12*points_in_mapframe.shape[0]
            msg.is_dense = int(np.isfinite(points_in_mapframe).all())
            msg.data = np.asarray(points_in_mapframe, np.float32).tostring()
            self.pub_fix = self.rospy.Publisher(lidar_topic + "/cloud_in_map_frame", PointCloud2, queue_size=1)
            self.pub_fix.publish(msg)

    def lidar_callback(self, which_lidar, event=None, other_agents=[]):
        from sensor_msgs.msg import LaserScan, PointCloud2, PointField
        now = self.rospy.Time.now()

        if which_lidar == "front":
            lidar_publisher = self.pub_lidar_front
            lidar_frame = self.kLidarFrontFrame
        elif which_lidar == "rear":
            lidar_publisher = self.pub_lidar_rear
            lidar_frame = self.kLidarRearFrame
        else:
            raise NotImplementedError("which_lidar must be 'front' or 'rear'.")

        # raytrace lidar
        ranges = self.lidar_update(which_lidar, other_agents)

        # publish LaserScan
        scan_msg = LaserScan()
        scan_msg.header.stamp.secs = now.secs
        scan_msg.header.stamp.nsecs = now.nsecs
        scan_msg.header.frame_id = lidar_frame
        scan_msg.angle_min = self.kLidarMinAngle
        scan_msg.angle_max = self.kLidarMaxAngle
        scan_msg.angle_increment = self.kLidarAngleIncrement
        scan_msg.scan_time = self.kLidarScanTime
        scan_msg.time_increment = self.kLidarScanTime / len(ranges)
        scan_msg.range_min = self.kLidarMinRange
        scan_msg.range_max = self.kLidarMaxRange
        scan_msg.ranges = list(ranges)
        scan_msg.intensities = list(np.ones(ranges.shape) * 10000.) # fake intensities
        lidar_publisher.publish(scan_msg)

        if self.PUBLISH_REAR_POINTCLOUD and which_lidar == "rear":
            angles = np.arange(self.kLidarMinAngle, 
                               self.kLidarMaxAngle,
                               self.kLidarAngleIncrement)
            points = ranges[:,None] * np.stack([np.cos(angles), 
                                                np.sin(angles),
                                                np.zeros_like(angles)], axis=-1)
            msg = PointCloud2()
            msg.header.stamp.secs = now.secs
            msg.header.stamp.nsecs = now.nsecs
            msg.header.frame_id = lidar_frame
            msg.height = 1
            msg.width = len(points)
            msg.fields = [
                PointField('x', 0, PointField.FLOAT32, 1),
                PointField('y', 4, PointField.FLOAT32, 1),
                PointField('z', 8, PointField.FLOAT32, 1)]
            msg.is_bigendian = False
            msg.point_step = 12
            msg.row_step = 12*points.shape[0]
            msg.is_dense = int(np.isfinite(points).all())
            msg.data = np.asarray(points, np.float32).tostring()
            self.pub_cloud_rear.publish(msg)

        return ranges



    def lidar_update(self, which_lidar, other_agents):
        if self.DEBUG:
            start_time = timer()

        if which_lidar == "front":
            lidar_tf = self.kTFBaseLinkToLidarFront
        elif which_lidar == "rear":
            lidar_tf = self.kTFBaseLinkToLidarRear
        else:
            raise NotImplementedError("which_lidar must be 'front' or 'rear'.")

        # raytrace lidar
        ranges = self.raytrace(lidar_tf, other_agents)

        if self.DEBUG:
            end_time = timer()
            print("{}: Lidar update took {}s".format(self.rospy.get_rostime(),
                                                     end_time - start_time))
        return ranges

    def get_lidar_update_ijangles(self, which_lidar, n_angles=None):
        MIN_ANGLE = self.kLidarMinAngle
        MAX_ANGLE = self.kLidarMaxAngle
        if which_lidar == "front":
            lidar_tf = self.kTFBaseLinkToLidarFront
        elif which_lidar == "rear":
            lidar_tf = self.kTFBaseLinkToLidarRear
        elif which_lidar == "merged":
            lidar_tf = self.kTFBaseLinkToLidarMerged
            MIN_ANGLE = self.kLidarMergedMinAngle
            MAX_ANGLE = self.kLidarMergedMaxAngle
        else:
            raise NotImplementedError("which_lidar must be 'front', 'rear' or 'merged'.")
        with self.lock:
            lidar_in_base_link_frame = np.array([
                lidar_tf[0] * np.cos(self.pos[2]) - lidar_tf[1] * np.sin(self.pos[2]),
                lidar_tf[0] * np.sin(self.pos[2]) + lidar_tf[1] * np.cos(self.pos[2]),
                lidar_tf[2]
            ])
            lidar_pos = self.pos + lidar_in_base_link_frame
        if n_angles is None:
            angles = np.arange(MIN_ANGLE, MAX_ANGLE, self.kLidarAngleIncrement) + lidar_pos[2]
        else:
            angles = np.linspace(MIN_ANGLE, MAX_ANGLE-self.kLidarAngleIncrement, n_angles) + lidar_pos[2]
        lidar_pos_ij = self.map2d.xy_to_floatij([lidar_pos[:2]])[0]
        return lidar_pos_ij, angles


    def run(self):
        import rospy
        import tf
        self.rospy = rospy
        self.tf = tf
        from nav_msgs.msg import Odometry
        from sensor_msgs.msg import LaserScan, PointCloud2, PointField
        from geometry_msgs.msg import Twist, Quaternion
        """ Spins the simulator as a real-time ros node """
        self.rosify_simulation_frequency_parameters()
        # Publishers
        self.pub_lidar_front = self.rospy.Publisher(self.kLidarFrontTopic + "/scan", LaserScan, queue_size=1)
        self.pub_lidar_rear = self.rospy.Publisher(self.kLidarRearTopic + "/scan", LaserScan, queue_size=1)
        self.pub_cloud_rear = self.rospy.Publisher(self.kLidarRearTopic + "/cloud", PointCloud2, queue_size=1)
        self.pub_odom = self.rospy.Publisher(self.kOdomTopic, Odometry, queue_size=1)
        self.pub_trackedpersons = "not_initialized"
        self.tf_br = self.tf.TransformBroadcaster()
        # Initialize ros
        self.rospy.init_node(self.kNodeName)
        self.rospy.Subscriber(self.kCmdVelTopic, Twist, self.cmd_vel_callback, queue_size=1)
        self.rospy.Timer(self.kOdomPeriod, self.odom_callback)
        self.rospy.Timer(self.kLidarPeriod, partial(self.lidar_callback, "front"))
        self.rospy.Timer(self.kLidarPeriod, partial(self.lidar_callback, "rear"))
        self.rospy.Timer(rospy.Duration(1.), self.publish_trackedpersons_callback)
        try:
            self.rospy.spin()
        except KeyboardInterrupt:
            print("Keyboard interrupt - Shutting down")
        self.rospy.signal_shutdown("Keyboard Interrupt")

    def publish_trackedpersons_callback(self, event=None):
        if self.pub_trackedpersons == "not_initialized":
            try:
                from frame_msgs.msg import TrackedPersons, TrackedPerson
                self.pub_trackedpersons = self.rospy.Publisher(self.kTrackedPersonsTopic, TrackedPersons, queue_size=1)
            except ImportError:
                self.rospy.logwarn("Failed to import frame_msgs, disabling trackedpersons_callback")
                self.pub_trackedpersons = None
        elif self.pub_trackedpersons is None:
            return
        else:
            from frame_msgs.msg import TrackedPersons, TrackedPerson
            tp_msg = TrackedPersons()
            tp_msg.header.frame_id = self.kOdomFrame
            tp_msg.header.stamp = self.rospy.Time.now()
            self.pub_trackedpersons.publish(tp_msg)

    def raytrace(self, lidar_tf, other_agents, debug_out=None):
#         print("raytrace...")
        with self.lock:
            lidar_in_base_link_frame = np.array([
                lidar_tf[0] * np.cos(self.pos[2]) - lidar_tf[1] * np.sin(self.pos[2]),
                lidar_tf[0] * np.sin(self.pos[2]) + lidar_tf[1] * np.cos(self.pos[2]),
                lidar_tf[2]
            ])
            lidar_pos = self.pos + lidar_in_base_link_frame
        angles = np.arange(self.kLidarMinAngle, self.kLidarMaxAngle, self.kLidarAngleIncrement) + lidar_pos[2]
        if self.DEBUG:
            start_time = timer()
            # stretch the ray to guarantee every radius hits exactly one cell
            raystretch = 1. / np.maximum(np.abs(np.cos(angles)), np.abs(np.sin(angles)))
            ray_r = np.arange(self.kLidarMaxRange / self.map2d.resolution()) * self.map2d.resolution()
            ray_x = ray_r[None,:] * (np.cos(angles) * raystretch)[:,None] + lidar_pos[0]
            ray_y = ray_r[None,:] * (np.sin(angles) * raystretch)[:,None] + lidar_pos[1]
            ray_i, ray_j = self.map2d.xy_to_floatij([[ray_x, ray_y]])[0]
            hits = self.map2d.occupancy()[(ray_i, ray_j)] # n_rays, n_samples
            # Find the first nonzero hit distance in each ray
            first_nonzero = (hits > self.map2d.thresh_occupied()).argmax(axis=-1)
            ranges = ray_r[first_nonzero] * raystretch
            end_time = timer()
            print("Raytrace took {}s".format(end_time - start_time))
            if debug_out is not None:
                # a += b (iadd) mutates the list, whereas a = a + b does not
                debug_out += [angles, ray_r, ray_i, ray_j, ray_x, ray_y, hits, ranges]
        else:
            ranges = np.zeros_like(angles)
            lidar_pos_ij = self.map2d.xy_to_floatij([lidar_pos[:2]])[0]
#             print("in...")
#             threadsperblock = 32
#             blockspergrid = (angles.size + (threadsperblock - 1)) // threadsperblock
#             cuda_raytrace[blockspergrid, threadsperblock](angles, lidar_pos_ij, self.map2d.render_agents(other_agents),
#                     self.map2d.thresh_occupied(), self.map2d.resolution(), self.map2d.origin, ranges)
#             tic =timer()
            ranges = ranges.astype(np.float32)
            self.raytracer.raymarch(angles, lidar_pos_ij, ranges)
#             toc = timer()
#             print("libcraytrace: {}s".format(toc-tic))
#             tic =timer()
#             compiled_raymarch(angles, lidar_pos_ij, self.map2d.occupancy(), self.esdf_ij,
#                     self.map2d.thresh_occupied(), self.map2d.resolution(), self.map2d.origin, ranges)
#             compiled_raytrace(angles, lidar_pos_ij, self.map2d.occupancy(),
#                     self.map2d.thresh_occupied(), self.map2d.resolution(), self.map2d.origin, ranges)
#             toc = timer()
#             print("raytrace: {}s".format(toc-tic))
#             tic = timer()
            if other_agents:
                self.map2d.render_agents_in_lidar(ranges, angles, other_agents, lidar_pos_ij)
#             toc = timer()
#             print("render: {}s".format(toc-tic))
#             print("out")
#         print("done")
        return ranges

def populate_PepperRLEnv_args(parser, exclude_IA_args=False):
    # used when RLEnv is combined with IAenv, which already determines those arguments
    if not exclude_IA_args:
        # map
        parser.add_argument('--map-folder', type=str, default=os.path.expanduser("~/maps"),)
        parser.add_argument('--map-name', type=str, default="empty",)
        # affects goals and spawns
        determ = parser.add_mutually_exclusive_group()
        determ.add_argument('--deterministic', action='store_true', help='if enabled agents and goals spawn point is fixed')
        determ.add_argument('--circular-scenario', action='store_true', help='if enabled runs a determinstic circle scenario (compatible with big_empty map)')
        # affects run mode
        parser.add_argument('--dt', type=float, default=0.2, help='simulation time increment')
        parser.add_argument('--n-agents', type=int, default=16)
        parser.add_argument('--mode', type=str, default='RESET_ALL',
                choices=['GHOSTS', 'RESET_ALL', 'BOUNCE'],
                help=""" In GHOSTS agents do not see each other and do not crash into each other.
                In RESET_ALL, all agents are reset after any collision.
                In BOUNCE, agents bounce back after any collision and do not get reset.""")
        parser.add_argument('--bounce-reset-vote', action='store_true', help='if enabled, when a majority of agents are in collision, reset all agents',)
        parser.add_argument('--no-ros', action='store_true', help='disable outputs to ros topics')
    # reward function
    parser.add_argument('--reward-arrival', type=float, default=100.)
    parser.add_argument('--reward-progress', type=float, default=5.)
    parser.add_argument('--reward-collision', type=float, default=-15)
    parser.add_argument('--reward-velocity', type=float, default=-0.1)
    parser.add_argument('--reward-standstill', type=float, default=-0.01)
    # affects inputs outputs
    parser.add_argument('--add-relative-obstacles', action='store_true', help='adds relative position of other agents to the observation space')
    parser.add_argument('--continuous', action='store_true', help='models actions as continuous')
    parser.add_argument('--unmerged-scans', action='store_true', help='use front/rear scans instead of a single merged lidar scan')
    parser.add_argument('--merged-scan-size', type=int, default=1080)
    # performance
    parser.add_argument( '--gpu-raytrace', action='store_true', help='raytrace using the GPU',)

def check_PepperRLEnv_args(args):
    # Detect nonsensical configurations
    if args.deterministic and args.mode != "GHOSTS" and args.n_agents > 1:
        raise ValueError("Deterministic non-ghost agents would crash into each other at init. Disallowed.")

def parse_PepperRLEnv_args(args=None, ignore_unknown=True):
    import argparse
    parser = argparse.ArgumentParser()
    # environment args
    populate_PepperRLEnv_args(parser)

    if ignore_unknown:
        parsed_args, unknown_args = parser.parse_known_args(args=args)
    else:
        parsed_args = parser.parse_args(args=args)
        unknown_args = []

    check_PepperRLEnv_args(parsed_args)

    return parsed_args, unknown_args


class PepperRLEnv(object):
    def __init__(self, args=None, map_=None, tsdf_=None, silent=False):
        if args is None:
            raise NotImplementedError
        if map_ is not None:
            map2d = map_
        else:
            map2d = CMap2D(args.map_folder, args.map_name, silent=silent)
        self.args = args
        self.ROS = not args.no_ros
        self.kRobotRadius = 0.3
        N_AGENTS = args.n_agents
        self.MODE = args.mode
        self.DIRECT_AGENT_OBS = args.add_relative_obstacles
        self.BOUNCE_RESET_VOTE = args.bounce_reset_vote
        self.DT = args.dt
        self.DETERMINISTIC = args.deterministic
        self.DETERMINISTIC_GOAL = lambda i : np.array([0., 5.5, 0])
        self.CIRCULAR_SCENARIO = args.circular_scenario
        self.DISCRETE = not args.continuous
        GPU_RAYTRACE = args.gpu_raytrace
        self.GOAL_REACHED_PADDING = 0.2 # if 0, the goal must be inside the agent circumference.
        self.MERGED_SCAN = not args.unmerged_scans
        self.kMaxSampleGoalDist = np.inf
        self.kMinSampleGoalDist = 3.  # sample goal at least 3m away
        # reward parameters
        self.kRArrival = args.reward_arrival
        self.kRCollision = args.reward_collision
        self.kRProgress = args.reward_progress
        self.kRVelocity = args.reward_velocity
        self.kRStandstill = args.reward_standstill
        # Map2D
        self.map2d = map2d
        self.map2d_as_obstacle_vertices = None
        if not silent: print("Creating Raytracer")
        self.raytracer = Raytracer(map2d, GPU=GPU_RAYTRACE, silent=silent)
        if not silent: print("Creating virtual Peppers")
        self.virtual_peppers = [Virtual2DPepper(map2d, raytracer=self.raytracer) for i in range(N_AGENTS)]
        self.vp_radii = [self.kRobotRadius for i in range(N_AGENTS)]
        if tsdf_ is None:
            if not silent: print("calculating TSDF")
            tic = timer()
            self.tsdf = map2d.as_tsdf(2*max(self.vp_radii))
            toc = timer()
            if not silent: print("TSDF calculation: {}s".format(toc-tic))
        else:
            self.tsdf = tsdf_
        self.coarse_map2d = coarse = map2d.as_coarse_map2d().as_coarse_map2d().as_coarse_map2d()
        coarse_origin_ij = coarse.xy_to_ij([[0, 0]])[0]
        # TODO: check that this is not occupied
#         if self.coarse_map2d.occupancy()[coarse_origin_ij[0], coarse_origin_ij[1]] > self.coarse_map2d.thresh_free:
#             raise ValueError("Map origin is occupied!")
        tsdf = (coarse.as_tsdf(self.kRobotRadius+0.2) < (self.kRobotRadius+0.2)).astype(np.uint8)
        self.reachable_space = coarse.dijkstra(coarse_origin_ij, tsdf) < 10000.
        self.reachable_xy = coarse.ij_to_xy(np.array(np.where(self.reachable_space)).T) # shape(n_points, 2)
        self.kObsBufferSize = 4
        self._reset_agents()
        if self.ROS:
            self._ros_init()
        # spaces
        from gym.spaces.box import Box
        from gym.spaces.tuple import Tuple
        self.kStateSize = 5
        if self.MERGED_SCAN:
            self.kMergedScanSize = args.merged_scan_size
            lidar_obs_space = Box(low=0, high=100, shape=(self.kObsBufferSize,self.kMergedScanSize,1), dtype=np.float32)
        else:
            lidar_obs_space = Box(low=0, high=100, shape=(self.kObsBufferSize,811,2), dtype=np.float32)
        state_obs_space = Box(low=-100, high=100, shape=(self.kStateSize,), dtype=np.float32)
        self.observation_space = Tuple([lidar_obs_space, state_obs_space])
        if self.DIRECT_AGENT_OBS:
            self.MAX_N_RELATIVE_OBSTACLES = args.max_n_relative_obstacles
            rel_obst_size = N_AGENTS-1
            if self.MODE=="GHOSTS":
                rel_obst_size = 0
            rel_obst_space = Box(low=-100, high=100, shape=(self.kObsBufferSize, rel_obst_size, 2, 3), dtype=np.float32)
            self.observation_space = Tuple([lidar_obs_space, state_obs_space, rel_obst_space])

        if self.DISCRETE:
            self.action_values = np.array([-1, -0.8, -0.5, -0.2, 0, 0.2, 0.5, 0.8, 1])
            self.action_space = Box(low=0., high=1., shape=(3,len(self.action_values)), dtype=np.float32) # 3 dimensional [x_vel, y_vel, theta_vel]
        else:
            self.action_space = Box(low=-1., high=1., shape=(3,), dtype=np.float32)
        self.metadata = {}
        self.viewer = None

    def _set_agents_pos(self, agents_pos):
        for i, pos in enumerate(agents_pos):
            if pos is not None:
                vp = self.virtual_peppers[i]
                vp.reset()
                vp.pos = pos

    def _set_agents_goals(self, agents_goals):
        for i, goal in enumerate(agents_goals):
            if goal is not None:
                self.agent_goals[i] = goal

    def _reset_agents(self, agents_mask=None):
        """ agents mask is of size (n_agents,), and is true for every agent to be reset """
        if agents_mask is None:
            agents_mask = [True for _ in range(len(self.virtual_peppers))]
        # Reset agent goals and buffers
        # in case this is the first reset, assign arrays
        if np.all(agents_mask):
            self.agent_goals = np.zeros((len(self.virtual_peppers), 3))
            self.scan_buffer = [deque(maxlen=self.kObsBufferSize) for _ in range(len(self.virtual_peppers))]
            self.pos_buffer  = [deque(maxlen=self.kObsBufferSize) for _ in range(len(self.virtual_peppers))]
            self.obst_buffer  = [deque(maxlen=self.kObsBufferSize) for _ in range(len(self.virtual_peppers))]
            self.dist_travelled_buffer  = [deque(maxlen=self.kObsBufferSize) for _ in range(len(self.virtual_peppers))]
            self.total_reward = np.zeros((len(self.virtual_peppers)))
            self.episode_step = np.zeros((len(self.virtual_peppers)), dtype=int)
            self.episode_damage = np.zeros((len(self.virtual_peppers)))
        # respawn agents
        self._respawn_agents(agents_mask)
        # sample goal
        for i, reset_this_agent in enumerate(agents_mask):
            if reset_this_agent:
                self.agent_goals[i] = self.sample_goal(i)
        # reset buffers and sample goal
        for i, reset_this_agent in enumerate(agents_mask):
            if reset_this_agent:
                self.scan_buffer[i] = deque(maxlen=self.kObsBufferSize)
                self.pos_buffer[i]  = deque(maxlen=self.kObsBufferSize)
                self.obst_buffer[i]  = deque(maxlen=self.kObsBufferSize)
        # reset episode reward info
        self.total_reward[agents_mask] = 0
        self.episode_step[agents_mask] = 0
        self.episode_damage[agents_mask] = 0
    def _respawn_agents(self, agents_mask=None):
        """ agents mask is of size (n_agents,), and is true for every agent to be reset """
        if agents_mask is None:
            agents_mask = [True for _ in range(len(self.virtual_peppers))]
        # Sample agent spawn
        if self.DETERMINISTIC:
            for i, reset_this_agent in enumerate(agents_mask):
                if reset_this_agent:
                    xy = np.array([0., 0.])
                    th = 0
                    pos =  np.array([xy[0], xy[1], th])
                    vp = self.virtual_peppers[i]
                    vp.reset()
                    vp.pos = pos
        elif self.CIRCULAR_SCENARIO:
            # agents on a circle
    #         r = 4 + 0.2 * len(self.virtual_peppers)
    #         ths = np.linspace(0, 2*np.pi, 1+len(self.virtual_peppers))[1:]
    #         agents_start_pos = [[r*np.cos(th), r*np.sin(th), th] for th in ths]
            R = 6
            th_inc = 2*np.pi / len(self.virtual_peppers)
            for i, reset_this_agent in enumerate(agents_mask):
                if reset_this_agent:
                    th = th_inc * i
                    pos =  np.array([R*np.cos(th), R*np.sin(th), -th])
                    vp = self.virtual_peppers[i]
                    vp.reset()
                    vp.pos = pos
        else:
            # agents at random positions
            remaining_xy = np.copy(self.reachable_xy)
            # remove already spawned agents neighborhood from remaining spawn possibilities
            for i, reset_this_agent in enumerate(agents_mask):
                if not reset_this_agent:
                    spawn_radius = 2 * self.vp_radii[i]
                    pos = self.virtual_peppers[i].pos
                    xy = pos[:2]
                    remaining_xy = remaining_xy[np.linalg.norm(remaining_xy - xy, axis=-1) > spawn_radius, :]
            # spawn desired agents
            for i, reset_this_agent in enumerate(agents_mask):
                if reset_this_agent:
                    if remaining_xy.shape[0] == 0:
                        import matplotlib.pyplot as plt
                        from map2d import gridshow
                        gridshow(self.reachable_space)
                        plt.show()
                        raise ValueError("Failed to spawn {} agents".format(len(self.virtual_peppers)))
                    spawn_radius = 2 * self.vp_radii[i]
                    if self.DETERMINISTIC:
                        xy = np.array([0., 0.])
                        th = 0
                    else:
                        randidx = np.random.randint(len(remaining_xy))
                        xy = remaining_xy[randidx]
                        th = np.random.rand() * 2 * np.pi
                        # remove new agent neighborhood from remaining spawn possibilities
                        remaining_xy = remaining_xy[np.linalg.norm(remaining_xy - xy, axis=-1) > spawn_radius, :]
                    pos =  np.array([xy[0], xy[1], th])
                    vp = self.virtual_peppers[i]
                    vp.reset()
                    vp.pos = pos

    def reset(self, agents_mask=None, ONLY_FOR_AGENT_0=False):
        self._reset_agents(agents_mask=agents_mask)
        ob, _, _, _  = self.step(None, ONLY_FOR_AGENT_0=ONLY_FOR_AGENT_0)
        return ob

    def _ros_init(self):
        import rospy
        import tf
        self.rospy = rospy
        self.tf = tf
        from nav_msgs.msg import Odometry
        from sensor_msgs.msg import LaserScan, PointCloud2, PointField
        from geometry_msgs.msg import Twist, Quaternion
        tf_br = tf.TransformBroadcaster()
        for i, vp in enumerate(self.virtual_peppers):
            vp.tf = tf
            vp.rospy = rospy
            vp.rosify_simulation_frequency_parameters()
            suffix = "_{}".format(i) if i != 0 else ""
            vp.kLidarFrontFrame = "sick_laser_front{}".format(suffix)
            vp.kLidarRearFrame = "sick_laser_rear{}".format(suffix)
            vp.kCmdVelTopic = "/cmd_vel{}".format(suffix)
            vp.kBaseLinkFrame = "base_link{}".format(suffix)
            vp.kBaseFootprintFrame = "base_footprint{}".format(suffix)
            vp.kOdomFrame = "odom{}".format(suffix)
            vp.kGoalFrame = "goal_agent{}".format(suffix)
            vp.kLidarFrontTopic = "/sick_laser_front{}".format(suffix)
            vp.kLidarRearTopic = "/sick_laser_rear{}".format(suffix)
            vp.kLidarMergedTopic = "/merged_lidar{}".format(suffix)
            vp.kOdomTopic = "/pepper_robot/odom{}".format(suffix)
            # Publishers
            vp.pub_lidar_front = rospy.Publisher(vp.kLidarFrontTopic + "/scan", LaserScan, queue_size=10)
            vp.pub_lidar_rear = rospy.Publisher(vp.kLidarRearTopic + "/scan", LaserScan, queue_size=10)
            vp.pub_cloud_rear = rospy.Publisher(vp.kLidarRearTopic + "/cloud", PointCloud2, queue_size=10)
            vp.pub_odom = rospy.Publisher(vp.kOdomTopic, Odometry, queue_size=10)
            vp.tf_br = tf_br
        # disable signals to allow python to catch SIGINT
        rospy.init_node("pepper_rl_env", disable_signals=True)
        print("ROS publisher / subscribers initialized")

    def _ros_shutdown(self, reason="Shutdown requested"):
        self.rospy.signal_shutdown(reason)

    def sample_goal(self, i):
        """ finds a random reachable goal for agent i """
        if self.DETERMINISTIC:
            goal = self.DETERMINISTIC_GOAL(i)
        elif self.CIRCULAR_SCENARIO:
            R = 6
            th_inc = 2*np.pi / len(self.virtual_peppers)
            th = th_inc * i
            goal = np.array([-R*np.cos(th), -R*np.sin(th), -th])
        else:
            remaining_xy = np.copy(self.reachable_xy)
            # only sample goals 3-5m away from agent
            xy = self.virtual_peppers[i].pos[:2]
            distance = np.linalg.norm(remaining_xy - xy, axis=-1)
            is_valid_goal = np.logical_and(distance < self.kMaxSampleGoalDist,
                                           distance > self.kMinSampleGoalDist)
            remaining_xy = remaining_xy[is_valid_goal, :]
            randidx = np.random.randint(len(remaining_xy))
            goal_xy = remaining_xy[randidx]
            goal_th = np.random.rand() * 2 * np.pi
            goal = np.array([goal_xy[0], goal_xy[1], goal_th])
        return goal

    def publish_trackedpersons(self, now, collided):
        # ground truth for visualization
        from spencer_tracking_msgs.msg import TrackedPersons, TrackedPerson
        pub = self.rospy.Publisher("/tracked_persons", TrackedPersons, queue_size=1)
        tp_msg = TrackedPersons()
        tp_msg.header.frame_id = self.virtual_peppers[0].kSimFrame
        tp_msg.header.stamp = now
        for i, vp  in enumerate(self.virtual_peppers):
            if i == 0: # skip robot
                continue
            tp = TrackedPerson()
            tp.track_id = i
            tp.is_occluded = False
            tp.is_matched = not collided[i]
            tp.detection_id = i
            tp.pose.pose.position.x = vp.pos[0]
            tp.pose.pose.position.y = vp.pos[1]
            from geometry_msgs.msg import Quaternion
            tp.pose.pose.orientation = Quaternion(
                *self.tf.transformations.quaternion_from_euler(0, 0, vp.pos[2]))
            tp.twist.twist.linear.x = vp.vel[0]
            tp.twist.twist.linear.y = vp.vel[1]
            tp.twist.twist.angular.z = vp.vel[2]
            tp_msg.tracks.append(tp)
        pub.publish(tp_msg)
        # ground truth for unity
        try:
            from crowdbotsim.msg import TwistArrayStamped
            from geometry_msgs.msg import Twist
            pub = self.rospy.Publisher("/crowd", TwistArrayStamped, queue_size=1)
            cwd_msg = TwistArrayStamped()
            cwd_msg.header.frame_id = self.virtual_peppers[0].kSimFrame
            cwd_msg.header.stamp = now
            for i, vp in enumerate(self.virtual_peppers):
                goal = self.agent_goals[i]
                a = Twist()
                a.linear.x = vp.pos[0]
                a.linear.y = vp.pos[1]
                if i == 0:  # robot
                    heading = vp.pos[2]
                else:
                    heading = np.arctan2(vp.vel[1], vp.vel[0])
                    if self.are_goals_reached()[i]:
                        heading = np.arctan2(goal[1] - vp.pos[1], goal[0] - vp.pos[0])
                new_angle_for_unity = 90. - 360. / 2. / np.pi * heading
                if np.linalg.norm(vp.vel[:2]) > 0.1:
                    new_angle_for_unity = 1000.
                a.linear.z = new_angle_for_unity 
                cwd_msg.twist.append(a)
            pub.publish(cwd_msg)
        except:
            pass
        # simulate trackedpersons from realsense
        # assumes frame_soft frame to be odom!
        from frame_msgs.msg import TrackedPersons, TrackedPerson
        pub = self.rospy.Publisher("/rwth_tracker/tracked_persons", TrackedPersons, queue_size=1)
        robot = self.virtual_peppers[0]
        tp_msg = TrackedPersons()
        tp_msg.header.frame_id = robot.kOdomFrame
        tp_msg.header.stamp = now
        p2_sim_in_robot = pose2d.inverse_pose2d(robot.pos)
        p2_sim_in_odom = pose2d.inverse_pose2d(robot.odom_in_map_frame)
        for i, vp in enumerate(self.virtual_peppers):
            if i == 0:  # skip robot
                continue
            pose_in_sim = vp.pos
            pose_in_robot = pose2d.apply_tf_to_pose(pose_in_sim, p2_sim_in_robot)
            pose_in_odom = pose2d.apply_tf_to_pose(pose_in_sim, p2_sim_in_odom)
            vel_in_odom = pose2d.apply_tf_to_vel(vp.vel, p2_sim_in_odom)
            # check FOV
            is_in_front = pose_in_robot[0] > 0
            is_in_fov = abs(pose_in_robot[0] / (pose_in_robot[1] + 0.0000001)) > 1  # D435: 45deg half FOV
            if not is_in_front or not is_in_fov:
                continue
            tp = TrackedPerson()
            tp.track_id = i
            tp.is_occluded = False
            tp.is_matched = not collided[i]
            tp.detection_id = i
            tp.pose.pose.position.x = pose_in_odom[0]
            tp.pose.pose.position.y = pose_in_odom[1]
            from geometry_msgs.msg import Quaternion
            tp.pose.pose.orientation = Quaternion(
                *self.tf.transformations.quaternion_from_euler(0, 0, vp.pos[2]))
            tp.twist.twist.linear.x = vel_in_odom[0]
            tp.twist.twist.linear.y = vel_in_odom[1]
            tp.twist.twist.angular.z = vel_in_odom[2]
            tp_msg.tracks.append(tp)
        pub.publish(tp_msg)



    def broadcast_goals(self, collided):
        """ broadcast goal visualizations in ROS """
        now = self.rospy.Time.now()
        if False:  # deprecated, takes too long
            for vp, goal in zip(self.virtual_peppers, self.agent_goals):
                vp.tf_br.sendTransform(
                        (goal[0], goal[1], 0),
                        self.tf.transformations.quaternion_from_euler(0, 0, goal[2]),
                        now,
                        vp.kGoalFrame,
                        vp.kOdomFrame,
                    )
        from visualization_msgs.msg import Marker, MarkerArray
        from geometry_msgs.msg import Point
        from std_msgs.msg import ColorRGBA
        pub = self.rospy.Publisher("/agent_goals", MarkerArray, queue_size=1)
        ma = MarkerArray()
        mk = Marker()
        mk.header.frame_id = self.virtual_peppers[0].kOdomFrame
        mk.ns = "all"
        mk.id = 0
        mk.type = 5
        mk.action = 0
        mk.scale.x = 0.02
        mk.color.g = 1
        mk.color.a = 1
        mk.frame_locked = True
        for vp, goal in zip(self.virtual_peppers, self.agent_goals):
            # vary color from red to green depending on dist to goal
            dist = np.linalg.norm(vp.pos[:2] - goal[:2])
            normdist = np.clip( dist / 10., 0, 1 )
            cl = ColorRGBA()
            cl.r = 1 - max(0.5 - normdist, 0)
            cl.g = 1 - max(normdist - 0.5, 0)
            # create points
            pt = Point()
            pt.x = vp.pos[0]
            pt.y = vp.pos[1]
            mk.points.append(pt)
            mk.colors.append(cl)
            pt = Point()
            pt.x = goal[0]
            pt.y = goal[1]
            mk.points.append(pt)
            mk.colors.append(cl)
        ma.markers.append(mk)
        # create cylinders for robots
        for i, vp  in enumerate(self.virtual_peppers):
            mk = Marker()
            mk.header.frame_id = self.virtual_peppers[0].kOdomFrame
            mk.ns = "all"
            mk.id = i + 1
            mk.type = 3
            mk.action = 0
            mk.scale.x = self.vp_radii[i] * 2.
            mk.scale.y = self.vp_radii[i] * 2.
            mk.scale.z = 0.02
            mk.color.r = 1
            mk.color.g = 1
            mk.color.b = 1
            if collided[i]:
                mk.color.r = 1
                mk.color.g = 0.7
                mk.color.b = 0
            if np.linalg.norm(self.agent_goals[i][:2] - vp.pos[:2]) <= (self.vp_radii[i] + self.GOAL_REACHED_PADDING):
                mk.color.r = 0
                mk.color.g = 1
                mk.color.b = 0
            mk.color.a = 1
            mk.frame_locked = True
            mk.pose.position.x = vp.pos[0]
            mk.pose.position.y = vp.pos[1]
            mk.pose.orientation.z = vp.pos[2]
            ma.markers.append(mk)
            mk = Marker()
            mk.header.frame_id = self.virtual_peppers[0].kOdomFrame
            mk.ns = "all"
            mk.id = i + 1 + len(self.virtual_peppers)
            mk.type = 0
            mk.action = 0
            mk.scale.x = 0.02
            mk.scale.y = 0.02
            mk.color.r = mk.color.g = mk.color.b = 0.5
            mk.color.a = 1
            mk.frame_locked = True
            pt = Point()
            pt.x = vp.pos[0]
            pt.y = vp.pos[1]
            pt.z = 0.02
            mk.points.append(pt)
            pt = Point()
            pt.x = vp.pos[0] + self.vp_radii[i] * np.cos(vp.pos[2])
            pt.y = vp.pos[1] + self.vp_radii[i] * np.sin(vp.pos[2])
            pt.z = 0.02
            mk.points.append(pt)
            ma.markers.append(mk)
        pub.publish(ma)
        # create cmd_vel arrow for robots
        pub = self.rospy.Publisher("/agent_cmd_vels", MarkerArray, queue_size=1)
        ma = MarkerArray()
        for i, vp  in enumerate(self.virtual_peppers):
            mk = Marker()
            mk.header.frame_id = vp.kBaseFootprintFrame
            mk.ns = "arrows"
            mk.id = i 
            mk.type = 0
            mk.action = 0
            mk.scale.x = 0.02
            mk.scale.y = 0.02
            mk.color.b = 1
            mk.color.a = 1
            mk.frame_locked = True
            pt = Point()
            pt.x = 0
            pt.y = 0
            pt.z = 0.03
            mk.points.append(pt)
            pt = Point()
            pt.x = 0 + vp.target_vel[0]
            pt.y = 0 + vp.target_vel[1]
            pt.z = 0.03
            mk.points.append(pt)
            ma.markers.append(mk)
            mk = Marker()
            mk.header.frame_id = vp.kBaseFootprintFrame
            mk.ns = "arrows"
            mk.id = i + len(self.virtual_peppers)
            mk.type = 0
            mk.action = 0
            mk.scale.x = 0.02
            mk.scale.y = 0.02
            mk.color.g = 0.5
            mk.color.b = 1
            mk.color.a = 1
            mk.frame_locked = True
            pt = Point()
            pt.x = 0
            pt.y = 0
            pt.z = 0.03
            mk.points.append(pt)
            pt = Point()
            pt.x = 0
            pt.y = 0 + vp.target_vel[2]
            pt.z = 0.03
            mk.points.append(pt)
            ma.markers.append(mk)
        pub.publish(ma)
        # publish tracked persons for mesh visuals
        self.publish_trackedpersons(now, collided)


    def ros_spin(self):
        raise NotImplementedError # this function is deprecated
        self.rospy.Timer(self.rospy.Duration(0.001), partial(self.step, self.get_linear_controller_action()))
        try:
            self.rospy.spin()
        except KeyboardInterrupt:
            print("Keyboard interrupt - Shutting down")
        self.rospy.signal_shutdown("Keyboard Interrupt")


    def step(self, actions,
             BYPASS_SI_UNITS_CONVERSION=False,
             DISABLE_A2A_COLLISION=None,
             ONLY_FOR_AGENT_0=False,
             ):
        """
        takes agent actions, runs a single physics update for all agents and computes
        observed state and reward for each agent.

        input:
            actions: None or ndarray of shape (n_agents, n_actions) containing
                     continuous values or discrete categories if DISCRETE.
                     actions are expected to be in unitless units (0-1) as given by a policy.
                     if None, no physics update or movement will be made, only observations.
        """
#         print("Begin step")
        MODE = self.MODE
        MERGED_SCAN = self.MERGED_SCAN
        TIMES = False
        if TIMES:
            tic = timer()
        should_reset = [False for i in self.virtual_peppers]
        if actions is not None:
            # transform actions into command velocities
            assert len(actions) == len(self.virtual_peppers)
            if np.any(np.isnan(actions)):
                raise ValueError("NaN values detected in agent actions: {}".format(actions))
            # turn one-hot actions into continuous values
            if self.DISCRETE:
                actions = self.categorical_to_continuous(actions)
            # transform unitless action into SI unit cmd_vel values
            if not BYPASS_SI_UNITS_CONVERSION:
                actions_as_cmd_vel = self.normalized_to_cmd_vel(actions)
            else:
                actions_as_cmd_vel = np.array(actions)
            # Move all agents
            for i, (vp, cmd_vel) in enumerate(zip(self.virtual_peppers, actions_as_cmd_vel)):
                vp.set_cmd_vel(cmd_vel)
                # update odometry
                if self.ROS:
                    publish_tf = True
                    if ONLY_FOR_AGENT_0 and i != 0:
                        publish_tf = False
                    vp.odom_callback(dt=self.DT, publish_tf=publish_tf)
                else:
                    is_clipping = vp.odom_update(dt=self.DT)
                    if not ONLY_FOR_AGENT_0 or i == 0:
                        if is_clipping:
                            should_reset[i] = True
        if TIMES:
            toc = timer()
            print("Moving agents: {}s".format(abs(tic-toc)))
            tic = timer()
        # Simulate lidars for all agents --------------
        # other agents
        all_agents = []
        if MODE == "GHOSTS":
            pass
        elif MODE == "BOUNCE" or MODE == "RESET_ALL":
            for i, vp in enumerate(self.virtual_peppers):
                # populate agents list
                agent = CSimAgent(vp.pos.astype(np.float32),
                                  vp.distance_travelled_in_base_frame.astype(np.float32),
                                  vp.vel.astype(np.float32))
                all_agents.append(agent)
        # get initial state for each ray of each agent
        n_scans = 1 if ONLY_FOR_AGENT_0 else len(self.virtual_peppers) # number of scans to render
        if MERGED_SCAN:
            ijthetas = np.zeros((n_scans, self.kMergedScanSize, 1, 3), dtype=np.float32)
            for i in range(n_scans):
                vp = self.virtual_peppers[i]
                merged_lidar_pos_ij, merged_angles = vp.get_lidar_update_ijangles("merged", self.kMergedScanSize)
                ijthetas[i, :, 0, :2] = merged_lidar_pos_ij
                ijthetas[i, :, 0, 2] = merged_angles
            # raytrace in single batch
            flat_ijthetas = ijthetas.reshape((-1, 3)) # flatten all but last dim
            ranges = np.zeros((flat_ijthetas.shape[0]), dtype=np.float32)
            self.raytracer.rm.calc_range_many(flat_ijthetas, ranges)
            ranges *= self.map2d.resolution()
            scans = ranges.reshape((n_scans, self.kMergedScanSize, 1)) # shape (n_agents, n_points, n_lidars_per_agent)
            self.map2d.render_agents_in_many_lidars(scans, ijthetas, all_agents, ONLY_FOR_AGENT_0)
            if self.ROS:
                for i in range(n_scans):
                    vp = self.virtual_peppers[i]
                    vp.publish_scan(scans[i, :, 0], "merged")
        else: 
            ijthetas = np.zeros((n_scans, 811, 2, 3), dtype=np.float32)
            for i in range(n_scans):
                vp = self.virtual_peppers[i]
                front_lidar_pos_ij, front_angles = vp.get_lidar_update_ijangles("front")
                rear_lidar_pos_ij, rear_angles = vp.get_lidar_update_ijangles("rear")
                ijthetas[i, :, 0, :2] = front_lidar_pos_ij
                ijthetas[i, :, 1, :2] = rear_lidar_pos_ij
                ijthetas[i, :, 0, 2] = front_angles
                ijthetas[i, :, 1, 2] = rear_angles
            # raytrace in single batch
            flat_ijthetas = ijthetas.reshape((-1, 3)) # flatten all but last dim
            ranges = np.zeros((flat_ijthetas.shape[0]), dtype=np.float32)
            self.raytracer.rm.calc_range_many(flat_ijthetas, ranges)
            ranges *= self.map2d.resolution()
            scans = ranges.reshape((n_scans, 811, 2)) # shape (n_agents, n_points, n_lidars_per_agent)
            self.map2d.render_agents_in_many_lidars(scans, ijthetas, all_agents, ONLY_FOR_AGENT_0)
            if self.ROS:
                for i in range(n_scans):
                    vp = self.virtual_peppers[i]
                    vp.publish_scan(scans[i, :, 0], "front")
                    vp.publish_scan(scans[i, :, 1], "rear")
        if TIMES:
            toc = timer()
            print("Simulating lidar: {}s".format(abs(tic-toc)))
            tic = timer()
        # Get relative states
        if self.DIRECT_AGENT_OBS:
            rel_obsts = np.zeros( # almost final size except for buffer dimension
                (len(self.virtual_peppers), self.MAX_N_RELATIVE_OBSTACLES, 2, 3))
            rel_obsts[:,:,0,:3] = 100. # default pos is far away
            rel_obsts[:,:,1,:3] = 0. # default vel is 0
            if MODE == "GHOSTS":
                pass
            else:
                if len(self.virtual_peppers) > self.MAX_N_RELATIVE_OBSTACLES:
                    raise ValueError("model max obstacles is smaller than amount observed.")
                for i, vp in enumerate(self.virtual_peppers):
                    for j, o_vp in enumerate(self.virtual_peppers):
                        if i == j:
                            continue
                        rel_obsts[i, j, 0, :3] = \
                            pose2d.apply_tf_to_pose(o_vp.pos, pose2d.inverse_pose2d(vp.pos))
                        rel_obsts[i, j, 1, :3] = \
                            pose2d.apply_tf_to_vel(o_vp.vel, pose2d.inverse_pose2d(vp.pos))
        # Update buffers and observation -----------------------------
        # if this is the first iteration fill history buffers with present value
        for vp, buffer_ in zip(self.virtual_peppers, self.pos_buffer):
            if len(buffer_) < self.kObsBufferSize:
                buffer_.extend([vp.pos*1.]*self.kObsBufferSize)
        for scan, buffer_ in zip(scans, self.scan_buffer): # scans might be shorter than scan_buffer
            if len(buffer_) < self.kObsBufferSize:
                buffer_.extend([scan*1.]*self.kObsBufferSize)
        if self.DIRECT_AGENT_OBS:
            for rel_obst, buffer_ in zip(rel_obsts, self.obst_buffer):
                if len(buffer_) < self.kObsBufferSize:
                    buffer_.extend([rel_obst*1.]*self.kObsBufferSize)
        if MODE == "BOUNCE":
            for vp, buffer_ in zip(self.virtual_peppers, self.dist_travelled_buffer):
                if len(buffer_) < self.kObsBufferSize:
                    buffer_.extend([vp.distance_travelled_in_base_frame*1.]*self.kObsBufferSize)
        # don't update buffers if no action step (observation-only step), as it should
        # not advance time
        if actions is not None:
            for vp, buffer_ in zip(self.virtual_peppers, self.pos_buffer):
                    buffer_.append(vp.pos*1.)
            for scan, buffer_ in zip(scans, self.scan_buffer): # scans might be shorter than scan_buffer
                    buffer_.append(scan*1.)
            if self.DIRECT_AGENT_OBS:
                for rel_obst, buffer_ in zip(rel_obsts, self.obst_buffer):
                        buffer_.append(rel_obst*1.)
            if MODE == "BOUNCE":
                for vp, buffer_ in zip(self.virtual_peppers, self.dist_travelled_buffer):
                        buffer_.append(vp.distance_travelled_in_base_frame*1.)
            self.episode_step[:] += 1
        # lidar observation
        lidar_obs = np.array([list(buf) for buf in self.scan_buffer]) # shape (n_agents, scan_hist_len, points_per_scan, n_lidars_per_agent )
        # get observation for goal position and agent velocity
        state_obs = [np.zeros(5) for i in self.virtual_peppers]
        for i, (vp, goal) in enumerate(zip( self.virtual_peppers, self.agent_goals )):
            s_g = pose2d.apply_tf(goal[:2], pose2d.inverse_pose2d(vp.pos)) # goal xy in agent frame
            s_w = vp.vel_in_base_frame
            state_obs[i] = np.hstack([s_g, s_w])
        state_obs = np.array(state_obs) # shape (n_agents, 5 [grx, gry, vx, vy, vtheta])
        # relative state obs (other agents) (fixed size)
        if self.DIRECT_AGENT_OBS:
            assert self.observation_space[2].shape[2] == 2
            assert self.observation_space[2].shape[3] == 3
            relobst_obs = np.zeros(# (n_agents, scan_hist_len, n_visible, 2 [pos, vel], 3 [rx, ry, rtheta])
                (len(self.virtual_peppers), self.kObsBufferSize, self.MAX_N_RELATIVE_OBSTACLES, 2, 3))
            relobst_obs[:,:,:,0,:3] = 100. # default pos is far away
            relobst_obs[:,:,:,1,:3] = 0. # default vel is 0
            for i, buf in enumerate(self.obst_buffer):
                for k, obs in enumerate(buf):
                    relobst_obs[i,k,:,:,:] = obs[:,:,:] 
        # all observatios
        obs = [lidar_obs, state_obs]
        if self.DIRECT_AGENT_OBS:
            obs = [lidar_obs, state_obs, relobst_obs]
        # Calculate rewards ----------------------------
        rews = [0 for i in self.virtual_peppers]
        coll = [False for _ in self.virtual_peppers]
        is_goal_reached = [False for _ in self.virtual_peppers]
        for i, (vp, goal, pbuf) in enumerate(zip(
                self.virtual_peppers, self.agent_goals, self.pos_buffer)):
            # collision reward (i.e. penalty)
            collision_reward = 0
            #   collision between agents
            if MODE == "GHOSTS":
                pass
            elif MODE == "BOUNCE" or MODE == "RESET_ALL":
                for j, o_vp in enumerate(self.virtual_peppers):
                    if i == j:
                        continue
                    if i > j:  # reciprocical, no need to collide twice
                        continue
                    if DISABLE_A2A_COLLISION is not None:
                        if DISABLE_A2A_COLLISION[i, j]:
                            continue
                    o_vpl1, o_vpl2 = all_agents[j].get_legs_pose2d_in_map()
                    o_vplr = all_agents[j].leg_radius
                    vpr = self.vp_radii[i]
                    # collision between agent and two other legs
                    if fast_2f_norm(vp.pos[:2] - o_vpl1[:2]) < ( vpr + o_vplr ) or \
                       fast_2f_norm(vp.pos[:2] - o_vpl2[:2]) < ( vpr + o_vplr ):
#                     if fast_2f_norm(vp.pos[:2] - o_vp.pos[:2]) < ( self.vp_radii[i] + self.vp_radii[j] ):
    #                     print("Collision between agents {} and {}".format(i, j))
                        collision_reward = self.kRCollision
                        self.episode_damage[i] += fast_2f_norm(vp.vel[:2])
                        self.episode_damage[j] += fast_2f_norm(o_vp.vel[:2])
                        coll[i] = True
                        coll[j] = True
                        if MODE == "RESET_ALL":
                            should_reset[i] = True
                        if MODE == "BOUNCE":
                            vp.pos = 1. * self.pos_buffer[i][-2] # bounce back agent to previous pos
                            o_vp.pos = 1. * self.pos_buffer[j][-2] # bounce back other agent to previous pos
                            self.pos_buffer[i][-1] = 1. * vp.pos
                            self.pos_buffer[j][-1] = 1. * o_vp.pos
                            vp.vel *= 0
                            vp.vel_in_base_frame *= 0
                            vp.distance_travelled_in_base_frame = 1. * self.dist_travelled_buffer[i][-2]
                            self.dist_travelled_buffer[i][-1] = 1. * vp.distance_travelled_in_base_frame
                            o_vp.vel *= 0
                            o_vp.vel_in_base_frame *= 0
                            o_vp.distance_travelled_in_base_frame = 1. * self.dist_travelled_buffer[j][-2]
                            self.dist_travelled_buffer[j][-1] = 1. * o_vp.distance_travelled_in_base_frame
                        break
            #   collision with static obstacle
            if self.tsdf[tuple(vp.map2d.xy_to_ij([vp.pos[:2]])[0])] < self.vp_radii[i]:
#                 print("Collision between agent {} and obstacle".format(i))
                coll[i] = True
                self.episode_damage[i] += fast_2f_norm(vp.vel[:2])
                if MODE == "BOUNCE":
                    vp.pos = 1. * self.pos_buffer[i][-2] # bounce back agent to previous pos
                    self.pos_buffer[i][-1] = 1. * vp.pos
                    vp.vel *= 0
                    vp.vel_in_base_frame *= 0
                    vp.distance_travelled_in_base_frame = 1. * self.dist_travelled_buffer[i][-2]
                    self.dist_travelled_buffer[i][-1] = 1. * vp.distance_travelled_in_base_frame
                if MODE == "GHOSTS" or MODE == "RESET_ALL":
                    should_reset[i] = True
                collision_reward = self.kRCollision
            # goal reward
            goal_reward = self.kRProgress*(
                    fast_2f_norm(pbuf[-2][:2] - goal[:2]) -
                    fast_2f_norm(pbuf[-1][:2] - goal[:2])
                    )
            # check if goal is reached
            episodic_reward = goal_reward / 1000.
            if fast_2f_norm(vp.pos[:2] - goal[:2]) < (self.vp_radii[i] + self.GOAL_REACHED_PADDING):
                is_goal_reached[i] = True
                goal[:] = self.sample_goal(i)
                goal_reward = self.kRArrival
                if MODE == "GHOSTS" or MODE == "RESET_ALL":
                    should_reset[i] = True
                # episodic style lump-sum reward
                damage_discount = np.power(0.9, self.episode_damage[i] * 10.)  # 0.1 damage 9/10ths score
                time_discount = np.power(0.9, self.episode_step[i] * self.DT / 60.) # every minute score 9/10ths
                episodic_reward = 100. * damage_discount * time_discount
                self.episode_damage[i] = 0
                self.episode_step[i] = 0
#                 self.total_reward[i] = 0
            # velocity reward
            velocity_reward = 0
            # use actual velocity or cmd_vel ? (cmd_vel punishes earlier)
            witnessed_velocity = np.zeros_like(vp.vel_in_base_frame)
            if True:
                if actions is not None:
                    witnessed_velocity = actions_as_cmd_vel[i]
            else: # deprecated
                witnessed_velocity = vp.vel_in_base_frame
            #   Don't turn too fast ( zero until 10% max vel, then proportional )
            if np.abs(witnessed_velocity[2]) >= np.abs(0.1 * vp.kMaxVel[2]):
                velocity_reward += self.kRVelocity * np.abs(witnessed_velocity[2])
            #   Dont go backwards
            if witnessed_velocity[0] < 0:
                velocity_reward += self.kRVelocity * np.abs(witnessed_velocity[0])
            #   Don't go sideways ( 50% of max velocity )
            if np.abs(witnessed_velocity[1]) > (0.5 * vp.kMaxVel[0]):
                velocity_reward += self.kRVelocity  * np.abs(witnessed_velocity[1]) 
            #   Don't go too fast ( 80% of max velocity )
            if fast_2f_norm(witnessed_velocity[:2]) > (0.8 * vp.kMaxVel[0]):
                velocity_reward += self.kRVelocity * fast_2f_norm(witnessed_velocity[:2])
            # standstill reward
            standstill_reward = self.kRStandstill
#             reward = goal_reward + collision_reward + velocity_reward + standstill_reward
            reward = episodic_reward
#             print(goal_reward, collision_reward, velocity_reward, reward)
            rews[i] = reward
        self.total_reward += rews
        # Calculate termination
        if MODE == "GHOSTS" and self.BOUNCE_RESET_VOTE and not ONLY_FOR_AGENT_0:
            if np.sum(coll) >= (len(coll) * 0.6):
                should_reset = np.ones_like(should_reset, dtype=bool)
        if self.ROS:
            self.broadcast_goals(coll)
        # TODO: currently only pass observations for the first agent to the rl framework
        rew = np.array(rews)
        new = np.array(should_reset)
        if MODE == "RESET_ALL" and not ONLY_FOR_AGENT_0:
            if np.any(new):
                new[:] = True
        # return observations, rewards, done, whatever
        if TIMES:
            toc = timer()
            print("Rew: {}s".format(abs(tic-toc)))
        infos = [{"goal_reached": is_goal_reached[i]} for i in range(len(self.virtual_peppers))]
        # Skip all but 0 if unnecessary
        if ONLY_FOR_AGENT_0:
            obs = tuple([np.array(ob[0]) for ob in obs])
            rew = rew[0]
            new = new[0]
            infos = infos[0]
        return obs, rew, new, infos

    def get_linear_controller_action(self):
        actions = []
        for vp, goal in zip(self.virtual_peppers, self.agent_goals):
            vx, vth = linear_controller(vp.pos, goal)
            actions.append(np.array([vx, 0, vth]))
        if self.DISCRETE:
            actions = self.continuous_to_categorical(np.array(actions))
        return actions

    def get_rvo_action(self):
        import rvo2
        DT = self.DT
        #RVOSimulator(timeStep, neighborDist, maxNeighbors, timeHorizon, timeHorizonObst, radius, maxSpeed, velocity = [0,0]);
        sim = rvo2.PyRVOSimulator(DT, 1.5, 5, 1.5, 2, 0.4, 2)

        # agents
        agents = []
        goals = self.agent_goals[:,:2]
        positions = np.array([vp.pos for vp in self.virtual_peppers])
        for vp, p, r in zip(self.virtual_peppers, positions, self.vp_radii):
            #addAgent(posxy, neighborDist, maxNeighbors, timeHorizon, timeHorizonObst, radius, maxSpeed, velocity = [0,0]);
            agents.append(sim.addAgent(
                tuple(p[:2]), radius=r, velocity=tuple(vp.vel[:2]), maxSpeed=vp.kMaxVel)
            )
        for i in range(len(agents)):
            a = agents[i]
            p = self.virtual_peppers[i].pos[:2]
            g = goals[i]
            d = [g[i] - p[i] for i in range(2)]
            norm = np.sqrt(d[0]**2 + d[1]**2)
            if abs(norm) < 0.1:
                d = (0, 0)
            pref_vel = (d[0]/norm, d[1]/norm)
            sim.setAgentPrefVelocity(a, pref_vel)

        # Obstacle(list of vertices), anticlockwise (clockwise for bounding obstacle)
        if self.map2d_as_obstacle_vertices is None:
              self.map2d_as_obstacle_vertices = self.map2d.as_closed_obst_vertices()
        for verts in self.map2d_as_obstacle_vertices:
            o1 = sim.addObstacle(list(verts))
        sim.processObstacles()

        # simulate
        sim.doStep()

        # return
        final_pos = np.array([list(sim.getAgentPosition(a)) + [0] for a in agents])
        final_vels = ( final_pos - positions ) / DT
#         final_vels = np.array([list(sim.getAgentVelocity(a)) + [0] for a in agents])
        return self.cmd_vel_to_normalized(final_vels)


    def normalized_to_cmd_vel(self, actions):
        actions_as_cmd_vel = np.zeros_like(actions)
        # clip
        norm_actions = np.clip(actions, -1, 1)
        for i, (vp, norm_action) in enumerate(zip(self.virtual_peppers, norm_actions)):
            # make sure the xy norm is not larger than max vel otherwise diagonal is faster
            xy_norm = fast_2f_norm(norm_action[:2])
            if xy_norm > 1:
                norm_action[0] = norm_action[0] / xy_norm
                norm_action[1] = norm_action[1] / xy_norm
            # Actions are unitless, transform to SI units
            actions_as_cmd_vel[i] = norm_action * vp.kMaxVel
        return actions_as_cmd_vel

    def cmd_vel_to_normalized(self, actions):
        """ (0.4, 0.2, 0.1) m/s (or rad/s) -> (~0.9, ~0.4, ~0.2) unitless """
        actions_normalized = np.zeros_like(actions)
        for i, (vp, action) in enumerate(zip(self.virtual_peppers, actions)):
            # transform to unitless
            actions_normalized[i] = action / vp.kMaxVel
        return actions_normalized

    # only useful if discrete actions
    def continuous_to_onehot(self, actions):
        """ (0.9, 0.4, 0.2) unitless -> ([0,0,0,0,0,1], [0,0,0,0,1,0], [0,0,1,0,0,0]) """
        if not self.DISCRETE:
            raise NotImplementedError("shouldn't invoke this function in continuous model")
        actionindex = np.argmin(np.abs(actions[:, :, None] - self.action_values[None, None, :]), axis=-1)
        ohactions = np.eye(self.action_space.shape[1])[actionindex]
        return ohactions

    def continuous_to_categorical(self, actions):
        """ (0.9, 0.4, 0.2) unitless -> (5, 4, 2) """
        if not self.DISCRETE:
            raise NotImplementedError("shouldn't invoke this function in continuous model")
        actionindex = np.argmin(np.abs(actions[:, :, None] - self.action_values[None, None, :]), axis=-1)
        return actionindex


    # only useful if discrete actions
    def onehot_to_continuous(self, actions):
        if not self.DISCRETE:
            raise NotImplementedError("shouldn't invoke this function in continuous model")
        actionindex = np.argmax(actions, axis=-1)
        return self.action_values[actionindex]

    def categorical_to_continuous(self, actions):
        if not self.DISCRETE:
            raise NotImplementedError("shouldn't invoke this function in continuous model")
        return self.action_values[actions]

    def are_goals_reached(self):
        result = np.zeros((len(self.virtual_peppers),))
        for i, (vp, goal) in enumerate(zip(self.virtual_peppers, self.agent_goals)):
          result[i] = fast_2f_norm(vp.pos[:2] - goal[:2]) < (self.vp_radii[i] + self.GOAL_REACHED_PADDING )
        return result

    def n_agents(self):
        return len(self.virtual_peppers)

    def render(self, mode='human', close=False, RENDER_LIDAR=True, lidar_scan_override=None, goal_override=None):
        if close:
            if self.viewer is not None:
                self.viewer.close()
            return
        renderer_map2d = self.map2d
        if np.any(np.array(renderer_map2d.occupancy().shape) > 512):
            renderer_map2d = self.coarse_map2d
        array = renderer_map2d.occupancy()[:,:,None] * np.array([1.,1.,1.]) # RGB channels
        ijs = renderer_map2d.xy_to_ij([vp.pos[:2] for vp in self.virtual_peppers], clip_if_outside=True)
        for n, ij in enumerate(ijs):
            i, j = ij
            if renderer_map2d.is_inside_ij(np.array([ij]).astype(np.float32))[0]:
                array[i, j, 0] = 1. # Red
            angle = self.virtual_peppers[n].pos[2]
            ioffset = int(np.around(np.cos(angle)))
            joffset = int(np.around(np.sin(angle)))
            if renderer_map2d.is_inside_ij(np.array([[i+ioffset, j+joffset]]).astype(np.float32))[0]:
                array[i+ioffset, j+joffset, :] = 0.3 # gray
        if mode == 'rgb_array':
            return array
        elif mode == 'human':
            # Window and viewport size
            WINDOW_W = renderer_map2d.occupancy().shape[0]
            WINDOW_H = renderer_map2d.occupancy().shape[1]
            VP_W = WINDOW_W
            VP_H = WINDOW_H
            from gym.envs.classic_control import rendering
            import pyglet
            from pyglet import gl
            # Create viewer
            if self.viewer is None:
                self.viewer = rendering.Viewer(WINDOW_W, WINDOW_H)
                self.score_label = pyglet.text.Label('0000', font_size=12,
                    x=20, y=WINDOW_H*2.5/40.00, anchor_x='left', anchor_y='center',
                    color=(255,255,255,255))
#                 self.transform = rendering.Transform()
                self.currently_rendering_iteration = 0
                self.rendering_obstacle_vertices = renderer_map2d.as_closed_obst_vertices_ij()
                self.image_lock = threading.Lock()
            # Render in pyglet
            def make_circle(c, r, res=10):
                thetas = np.linspace(0, 2*np.pi, res+1)[:-1]
                verts = np.zeros((res, 2))
                verts[:,0] = c[0] + r * np.cos(thetas)
                verts[:,1] = c[1] + r * np.sin(thetas)
                return verts
            with self.image_lock:
                self.currently_rendering_iteration += 1
                self.viewer.draw_circle(r=10, color=(0.3,0.3,0.3))
                win = self.viewer.window
                win.switch_to()
                win.dispatch_events()
                win.clear()
                gl.glViewport(0, 0, VP_W, VP_H)
                # colors
                bgcolor = np.array([0.4, 0.8, 0.4])
                obstcolor = np.array([0.3, 0.3, 0.3])
                goallinecolor = 0.9 * bgcolor
                goalcolor = np.array([1., 1., 0.3])
                nosecolor = np.array([0.3, 0.3, 0.3])
                lidarcolor = np.array([1., 0., 0.])
                agentcolor = np.array([0., 1., 1.])
                # Green background
                gl.glBegin(gl.GL_QUADS)
                gl.glColor4f(bgcolor[0], bgcolor[1], bgcolor[2], 1.0)
                gl.glVertex3f(0, VP_H, 0)
                gl.glVertex3f(VP_W, VP_H, 0)
                gl.glVertex3f(VP_W, 0, 0)
                gl.glVertex3f(0, 0, 0)
                gl.glEnd()
                # Map closed obstacles ---
                for poly in self.rendering_obstacle_vertices:
                    gl.glBegin(gl.GL_LINE_LOOP)
                    gl.glColor4f(obstcolor[0], obstcolor[1], obstcolor[2], 1)
                    for vert in poly:
                        gl.glVertex3f(vert[0], vert[1], 0)
                    gl.glEnd()
                # All dynamic objects ---
                ijs = renderer_map2d.xy_to_ij([vp.pos[:2] for vp in self.virtual_peppers], clip_if_outside=True)
                ij_radii = np.array(self.vp_radii) / renderer_map2d.resolution()
                ij_goals = renderer_map2d.xy_to_ij(self.agent_goals[:,:2], clip_if_outside=True)
                # LIDAR
                if RENDER_LIDAR:
                    for n, (ij, r, gij) in enumerate(zip(ijs, ij_radii, ij_goals)):
                        i, j = ij
                        igoal, jgoal = gij
                        angle = self.virtual_peppers[n].pos[2]
                        # LIDAR rays
                        if n == 0:
                            scan = lidar_scan_override
                            if scan is None:
                                scan = self.scan_buffer[n][-1][...,0]  # latest scan
                            _, lidar_angles = self.virtual_peppers[n].get_lidar_update_ijangles(
                                "merged", self.kMergedScanSize
                            )
                            i_ray_ends = i + scan / renderer_map2d.resolution() * np.cos(lidar_angles)
                            j_ray_ends = j + scan / renderer_map2d.resolution() * np.sin(lidar_angles)
                            is_in_fov = np.cos(lidar_angles - angle) >= 0.78
                            for ray_idx in range(len(scan)):
                                end_i = i_ray_ends[ray_idx]
                                end_j = j_ray_ends[ray_idx]
                                gl.glBegin(gl.GL_LINE_LOOP)
                                if is_in_fov[ray_idx]:
                                    gl.glColor4f(1., 1., 0., 0.1)
                                else:
                                    gl.glColor4f(lidarcolor[0], lidarcolor[1], lidarcolor[2], 0.1)
                                gl.glVertex3f(i, j, 0)
                                gl.glVertex3f(end_i, end_j, 0)
                                gl.glEnd()
                # Agent body
                for n, (ij, r, gij) in enumerate(zip(ijs, ij_radii, ij_goals)):
                    i, j = ij
                    igoal, jgoal = gij
                    angle = self.virtual_peppers[n].pos[2]
                    # Agent as Circle
                    poly = make_circle((i, j), r)
                    gl.glBegin(gl.GL_POLYGON)
                    if n == 0:
                        color = np.array([1., 1., 1.])
                    else:
                        color = agentcolor
                    gl.glColor4f(color[0], color[1], color[2], 1)
                    for vert in poly:
                        gl.glVertex3f(vert[0], vert[1], 0)
                    gl.glEnd()
                    # Direction triangle
                    inose = i + r * np.cos(angle)
                    jnose = j + r * np.sin(angle)
                    iright = i + 0.3 * r * -np.sin(angle)
                    jright = j + 0.3 * r * np.cos(angle)
                    ileft = i - 0.3 * r * -np.sin(angle)
                    jleft = j - 0.3 * r * np.cos(angle)
                    gl.glBegin(gl.GL_TRIANGLES)
                    gl.glColor4f(nosecolor[0], nosecolor[1], nosecolor[2], 1)
                    gl.glVertex3f(inose, jnose, 0)
                    gl.glVertex3f(iright, jright, 0)
                    gl.glVertex3f(ileft, jleft, 0)
                    gl.glEnd()
                # Goals
                for n, (ij, r, gij) in enumerate(zip(ijs, ij_radii, ij_goals)):
                    i, j = ij
                    igoal, jgoal = gij
                    if n == 0 and goal_override is not None:
                        igoal, jgoal = goal_override
                    angle = self.virtual_peppers[n].pos[2]
                    # Goal line
                    gl.glBegin(gl.GL_LINE_LOOP)
                    gl.glColor4f(goallinecolor[0], goallinecolor[1], goallinecolor[2], 1)
                    gl.glVertex3f(i, j, 0)
                    gl.glVertex3f(igoal, jgoal, 0)
                    gl.glEnd()
                    # Goal markers
                    gl.glBegin(gl.GL_TRIANGLES)
                    gl.glColor4f(goalcolor[0], goalcolor[1], goalcolor[2], 1)
                    triangle = make_circle((igoal, jgoal), r/3., res=3)
                    for vert in triangle:
                        gl.glVertex3f(vert[0], vert[1], 0)
                    gl.glEnd()
                # Text
                self.score_label.text = "S {} D {:.1f} R {:.1f}".format(self.episode_step[0], self.episode_damage[0], self.total_reward[0])
                self.score_label.draw()
                win.flip()
                return self.viewer.isopen

    def close(self):
        self.render(close=True)

    def _get_viewer(self):
        return self.viewer


def linear_controller(pos, goal):
    Krho = 0.5
    Kalpha = 1.5
    Kbeta = -0.6
    constantSpeed = 1.

    # current robot position and orientation
    x = pos[0]
    y = pos[1]
    theta = pos[2]

    # goal position and orientation
    xg = goal[0]
    yg = goal[1]
    thetag = goal[2]

    # compute control quantities
    rho = np.sqrt((xg-x)**2+(yg-y)**2)  # pythagoras theorem, sqrt(dx^2 + dy^2)
    lbda = np.arctan2(yg-y, xg-x)     # angle of the vector pointing from the robot to the goal in the inertial frame
    alpha = lbda - theta         # angle of the vector pointing from the robot to the goal in the robot frame
    alpha = normalizeAngle(alpha)
    beta = thetag-lbda
    beta = normalizeAngle(beta)

    # compute vu, omega
    vu = Krho * rho # [m/s]
    omega = Kalpha * alpha + Kbeta * beta # [rad/s]

    if True: # Constant speed enabled
        absVel = abs(vu)
        if absVel>1e-6:
            vu = vu/absVel*constantSpeed
            omega = omega/absVel*constantSpeed

    # if goal is reached
    if rho < 0.2:
        vu = 0
        omega = -normalizeAngle(theta - thetag)

    return vu, omega


def normalizeAngle(angle):
    return ( (angle+np.pi) % (2*np.pi) ) - np.pi 
