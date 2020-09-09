#!/usr/bin/env python
from functools import partial
from math import log
import numpy as np
import os
import rospy
import tf
from tf2_ros import TransformException
import threading
from timeit import default_timer as timer
import time
import warnings

from matplotlib import pyplot as plt

from geometry_msgs.msg import Twist, Point, PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Trigger, TriggerResponse
from visualization_msgs.msg import Marker

from branch_and_bound import BranchAndBound
from map2d import Map2D, LocalMap2D
import pose2d

from map_matcher.srv import *

from pyniel.numpy_tools.indexing import filter_if_out_of_bounds
from pyniel.numpy_tools.logic import absminimum

REMOVE_BEFORE_FLIGHT = 1

class Node(object):
    def __init__(self, id_, parent, v, x, cost=None):
        self.id = id_
        self.parent = parent
        self.v = v
        self.x = x
        self.cost = cost
        self.travel_cost = 0

class CostElements(object):
    """ A class to be filled with data used in calculating cost """
    def __init__(self):
        return

class MotionPlanner(object):
    def __init__(self, map2d, aggressive_locomotion=True, debug=False, start_disabled=True):
        # Initialize variables
        # x axis points "forward" from the robot, y points "left"
        self.lock = threading.Lock() # for avoiding race conditions
        self.global_map = map2d # occupancy grid
        self.kCmdVelTopic = "/cmd_vel"
        self.kLidarFrontTopic = "/sick_laser_front/filtered"
        self.kLidarRearTopic = "/sick_laser_rear/filtered"
        self.kOdomTopic = "/pepper_robot/odom"
        self.kSlamMapTopic = "/gmap"
        self.kNavGoalTopic = "/move_base_simple/goal"
        self.kLocalTSDFTopic = "debug/LocalTSDF"
        self.kLocalDijkstraTopic = "debug/LocalDijkstra"
        self.kLocalOccupancyTopic = "debug/LocalOccupancy"
        self.kGlobalMatchTopic = "debug/GlobalMatch"
        self.kGlobalPathVizTopic = "debug/GlobalPath"
        self.kGlobalPathTopic = "planner/GlobalPath"
        self.kLocalPathTopic = "debug/LocalPath"
        self.kDebugMarkerTopic = "debug/DebugMarker"
        self.kDebugMarker2Topic = "debug/DebugMarker2"
        self.kDebugMarker3Topic = "debug/DebugMarker3"
        self.kGlobalGoalTopic = "debug/GlobalGoal"
        self.kLocalWaypointTopic = "debug/LocalWaypoint"
        self.kBaseLinkFrame = "base_link"
        self.kBaseFootprintFrame = "base_footprint"
        self.kOdomFrame = "odom"
        self.kGlobalMapFrame = "reference_map"
        self.kLocalMapFrame = "base_footprint"
        self.kSlamMapFrame = "gmap"
        self.kGlobalMotionPlanningPeriod = rospy.Duration(0.1) # [s]
        self.kCoherenceCheckPeriod = rospy.Duration(1.) # [s]
        self.kPublishLocalizationPeriod = rospy.Duration(0.05) # [s]
        self.set_motion_planning_parameters()
        self.DEBUG = debug
        self.pos = np.array([0,0,0]) # x[m] y[m] theta[rad]
        self.cmd_vel_uv = []
        self.cmd_vel_dt = []
        self.cmd_vel_w = []
        self.cmd_vel_queue_expiry = None
        self.kCmdVelQueueDefaultTimeToExpiry = rospy.Duration(3.)
        self.STOP = start_disabled
        self.goal_is_reached = False

        self.initialize_path_planning()

    def set_motion_planning_parameters(self):
        self.kMinCmdVelPeriod = 0.1 # [s]
        self.kPepperWidth = 0.60 # [m]
        self.kPepperComfortZoneRadius = 0.7 # [m]
        self.set_move_config_aggressive()

        # Deprecated
#         self.kDT = 0.4 # [s]
#         self.kDTinv = 1. / self.kDT
#         self.kDynamicWindowMaxSamples = 10000
#         self.kAccPenalty = 0.00001
#         self.kJerkPenalty = 0.000001
        # ^^^^^^^

        self.global_dijkstra_lowres = None
        self.global_path_xy = None
        self.kDijkstraTSDFPenalty = 10.
        self.kDijkstraUnknownTerrainPenalty = 1.
        self.kLocalAreaLimits = np.array([[-3., 5.], # x min, x max
                                          [-3., 3.]]) # y min, y max
        self.kNLocalMapObservations = 20
        self.sensor_model={"p_hit": 0.75, "p_miss": 0.25}

    def set_move_config_default(self):                  # m/s2, m/s2, rad/s2
        self.kPepperMaxVel = np.array([0.35, 0.35, 1.]) # x y theta
        self.kPepperMaxAcc  = np.array([0.3, 0.3, 0.75]) # x y theta
        self.kPepperMaxJerk = np.array([1., 1., 2.]) # x y theta

    def set_move_config_aggressive(self):
        self.kPepperMaxVel = np.array([0.55, 0.55, 2.]) # x y theta
        self.kPepperMaxAcc  = np.array([0.55, 0.55, 3.]) # x y theta
        self.kPepperMaxJerk = np.array([5., 5., 50.]) # x y theta

    def stop_autonomous_motion_service_call(self, req):
        with self.lock:
            if not self.STOP:
                print("Surrendering robot control")
            self.STOP = True
        return TriggerResponse(True, "")

    def resume_autonomous_motion_service_call(self, req):
        with self.lock:
            if self.STOP:
                print("Assuming robot control")
            self.STOP = False
        return TriggerResponse(True, "")


    def initialize_path_planning(self):
        print("Loading global map.")
        self.global_map_8ds = self.global_map.as_coarse_map2d().as_coarse_map2d().as_coarse_map2d()
        self.global_sdf = self.global_map_8ds.as_sdf()
        self.goal_xy = None # In the global map frame
        self.local_map = LocalMap2D(self.kLocalAreaLimits, self.global_map.resolution(),
                sensor_model=self.sensor_model,
                max_observations=self.kNLocalMapObservations)
        self.bnb = BranchAndBound(self.global_map_8ds, rot_downsampling=2.)
        self.bnb_theta_prior = 0
        self.bnb_last_stamp = None
        print("Global map loaded.")

    def numpy_to_occupancy_grid_msg(self, arr, ref_map2d, frame_id, time=None):
        if not len(arr.shape) == 2:
                raise TypeError('Array must be 2D')
        arr = arr.T * 100.
        if not arr.dtype == np.int8:
            arr = arr.astype(np.int8)
        if time is None:
            time = rospy.Time.now()
        grid = OccupancyGrid()
        grid.header.frame_id = frame_id
        grid.header.stamp.secs = time.secs
        grid.header.stamp.nsecs = time.nsecs
        grid.data = arr.ravel()
        grid.info.resolution = ref_map2d.resolution()
        grid.info.height = arr.shape[0]
        grid.info.width = arr.shape[1]
        grid.info.origin.position.x = ref_map2d.origin[0]
        grid.info.origin.position.y = ref_map2d.origin[1]
        return grid 

    def global_goal_callback(self, msg): # x y is in the global map frame
        if self.bnb_last_stamp is not None:
            # get nav goal in global frame
            try:
                time = rospy.Time.now()
                tf_info = [self.kGlobalMapFrame, msg.header.frame_id, time]
                self.tf_listener.waitForTransform(*(tf_info + [self.tf_timeout]))
                tf_ = self.tf_listener.lookupTransform(*tf_info)
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException,
                    TransformException) as e:
                print("[{}.{}] tf for time {}.{} not found: {}".format(
                    rospy.Time.now().secs, rospy.Time.now().nsecs, time.secs, time.nsecs, e))
                return
            goal_pose2d = self.tf_to_pose2d(tf_)
        else:
            print("Failed to set goal without localization.")
            return
        goal_xy = pose2d.apply_tf(np.array([msg.pose.position.x, msg.pose.position.y]), goal_pose2d)
        goal_ij = self.global_map_8ds.xy_to_ij(goal_xy[:2], clip_if_outside=False)
        # dijkstra on the low resolution map with SDF extra costs
        # sdf_extra_costs encodes the penalty for moving between nodes in the region between
        # kPepperComfortZoneRadius and kPepperRadius.
        # Djikstra uses these extra costs as always-positive edge penalties to
        # favor moving away from obstacles while ensuring a monotonic gradient to the goal.
        print("Computing global Dijkstra.")
        sdf_extra_costs = self.dijkstra_extra_costs_from_sdf(self.global_sdf)
        self.global_dijkstra_lowres = self.global_map_8ds.dijkstra(
                goal_ij,
                mask=self.global_sdf < 0,
                extra_costs=sdf_extra_costs)
        print("Global Dijkstra computed.")
        self.goal_xy = goal_xy
        # Goal as marker
        marker = self.goal_as_marker(goal_xy)
        self.goal_pub.publish(marker)
        self.goal_is_reached = False
        print("New global goal: {}[meters], {}[ij]".format(goal_xy, goal_ij))

    def goal_as_marker(self, goal_xy,
            time = None,
            frame = None,
            namespace = 'global_goal',
            marker_type = 2, # SPHERE
            resolution = None,
            color = [0., 1., 0., 1.], # rgba
            ):
        if time is None:
            time = rospy.Time.now()
        if frame is None:
            frame = self.kGlobalMapFrame
        if resolution is None:
            resolution = self.global_map_8ds.resolution()
        marker = Marker()
        marker.header.stamp.secs = time.secs
        marker.header.stamp.nsecs = time.nsecs
        marker.header.frame_id = frame
        marker.ns = namespace
        marker.id = 0
        marker.type = marker_type
        marker.action = 0
        s = resolution
        marker.pose.position.x = goal_xy[0]
        marker.pose.position.y = goal_xy[1]
        marker.pose.position.z = 1.
        marker.scale.x = s
        marker.scale.y = s
        marker.scale.z = s
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = color[3]
        return marker

    # TODO: @njit
    def compute_path(self, costmap, first, enable_probabilistic_jumps=False):
        # trace path
        # Initialize edge costs
        r = np.roll(costmap, -1, axis=0) - costmap
        l = np.roll(costmap,  1, axis=0) - costmap
        u = np.roll(costmap, -1, axis=1) - costmap
        d = np.roll(costmap,  1, axis=1) - costmap
        edge_costs = np.stack([r, l, u, d], axis=-1)
        # Neighbor offsets
        offsets = np.array([
                            [ 1, 0],
                            [-1, 0],
                            [0,  1],
                            [0, -1]])
        # Init
        path = []
        jump_log = []
        # Path in global lowres map ij frame
        path.append(first)
        while True:
            current = path[-1]
            current_idx = tuple(current.astype(int))
            choices = edge_costs[current_idx]
            cheapest = np.argsort(choices)
            best_cost = choices[cheapest[0]]
            second_best_cost = choices[cheapest[1]]
            selected_offset = offsets[cheapest[0]]
            has_jumped = False
            if best_cost >= 0:
#                 print("local minima")
                jump_log.append(has_jumped)
                break
            if enable_probabilistic_jumps:
                if second_best_cost < 0:
                    # probabilistic jump
                    rand = np.random.random()
                    jump_chance = (second_best_cost 
                                             / (best_cost + second_best_cost))
                    if rand <= jump_chance:
                        selected_offset = offsets[cheapest[1]]
                        has_jumped = True
            next_ = current + selected_offset
            path.append(next_)
            jump_log.append(has_jumped)
        return np.array(path), np.array(jump_log)


    def lidar_callback(self, msg, which_lidar):
        if which_lidar == "front":
            None
        elif which_lidar == "rear":
            None
        else:
            raise NotImplementedError("which_lidar must be 'front' or 'rear'.")
        # Get Lidar pose in odom frame
        try:
            time = rospy.Time(msg.header.stamp.secs, msg.header.stamp.nsecs)
            tf_info = [self.kOdomFrame, msg.header.frame_id, time]
            self.tf_listener.waitForTransform(*(tf_info + [self.tf_timeout]))
            tf_ = self.tf_listener.lookupTransform(*tf_info)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException,
                TransformException) as e:
            print("[{}.{}] tf for time {}.{} not found: {}".format(
                rospy.Time.now().secs, rospy.Time.now().nsecs, time.secs, time.nsecs, e))
            print("Skipping message.")
            return
        lidar_pose2d = self.tf_to_pose2d(tf_)
        # Get base pose in odom frame
        try:
            time = rospy.Time(msg.header.stamp.secs, msg.header.stamp.nsecs)
            tf_info = [self.kOdomFrame, self.kLocalMapFrame, time]
            self.tf_listener.waitForTransform(*(tf_info + [self.tf_timeout]))
            tf_ = self.tf_listener.lookupTransform(*tf_info)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException,
                TransformException) as e:
            print("[{}.{}] tf for time {}.{} not found: {}".format(
                rospy.Time.now().secs, rospy.Time.now().nsecs, time.secs, time.nsecs, e))
            print("Skipping message.")
            return
        base_pose2d = self.tf_to_pose2d(tf_)
        with self.lock:
            self.local_map.add_observation(msg, lidar_pose2d, base_pose2d)

    def tf_to_pose2d(self, tf_):
        return np.array([
            tf_[0][0], # x
            tf_[0][1], # y
            tf.transformations.euler_from_quaternion(tf_[1])[2], # theta 
            ])


    def local_motion_planning_callback(self, msg):
        # Generate local occupancy
        with self.lock:
            local_map_frozen = self.local_map.copy()
        local_occupancy = local_map_frozen.generate()
        if local_occupancy is None:
            print("Not enough scans for local_map")
            return
        # Publish latest scan
        latest_scan = local_map_frozen.observations[local_map_frozen.ci_(0)]
        x_entropy_error, latest_hits = local_map_frozen.cross_entropy_error(
                latest_scan,
                np.array([0,0,0]), # TODO for now the local map frame corresponds to laser front
                local_occupancy.occupancy())
        self.local_occ_pub.publish(self.numpy_to_occupancy_grid_msg(local_occupancy.occupancy(), local_map_frozen, self.kLocalMapFrame, latest_scan.header.stamp))
        # Get local TSDF
        local_tsdf = local_occupancy.as_tsdf(self.kPepperComfortZoneRadius)
        self.local_tsdf_pub.publish(self.numpy_to_occupancy_grid_msg(local_tsdf, local_map_frozen, self.kLocalMapFrame, latest_scan.header.stamp))
        # Local motion planning
        if self.global_dijkstra_lowres is None:
            return
        if self.global_path_xy is None:
            return
        # get tf between local and global
        time = latest_scan.header.stamp
        try:
            tf_info = [self.kLocalMapFrame, self.kGlobalMapFrame, time] 
            self.tf_listener.waitForTransform(*(tf_info + [self.tf_timeout]))
            tf_ = self.tf_listener.lookupTransform(*tf_info)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException,
                TransformException) as e:
            print("[{}.{}] tf for time {}.{} not found: {}".format(
                rospy.Time.now().secs, rospy.Time.now().nsecs, time.secs, time.nsecs, e))
            return
        pose2d_local_global = self.tf_to_pose2d(tf_)
        # get tf between odom and global
        try:
            tf_info = [self.kGlobalMapFrame, msg.header.frame_id, time] 
            self.tf_listener.waitForTransform(*(tf_info + [self.tf_timeout]))
            tf_ = self.tf_listener.lookupTransform(*tf_info)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException,
                TransformException) as e:
            print("[{}.{}] tf for time {}.{} not found: {}".format(
                rospy.Time.now().secs, rospy.Time.now().nsecs, time.secs, time.nsecs, e))
            return
        pose2d_global_odom = self.tf_to_pose2d(tf_)
        # get tf between global and base_link
        try:
            tf_info = [self.kGlobalMapFrame, msg.child_frame_id, time] 
            self.tf_listener.waitForTransform(*(tf_info + [self.tf_timeout]))
            tf_ = self.tf_listener.lookupTransform(*tf_info)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException,
                TransformException) as e:
            print("[{}.{}] tf for time {}.{} not found: {}".format(
                rospy.Time.now().secs, rospy.Time.now().nsecs, time.secs, time.nsecs, e))
            return
        pose2d_global_baselink = self.tf_to_pose2d(tf_)
        # Odometry pos/vel in global frame
        v_odom = np.array([msg.twist.twist.linear.x, # current
                           msg.twist.twist.linear.y,
                           msg.twist.twist.angular.z])
        v_glob = pose2d.apply_tf_to_vel(v_odom, pose2d_global_baselink)
        q = msg.pose.pose.orientation
        x_odom = np.array([msg.pose.pose.position.x,  # save or compute? memory vs execution time
                           msg.pose.pose.position.y,
                           tf.transformations.euler_from_quaternion(np.array([q.x, q.y, q.z, q.w]))[2]])
        x_glob = pose2d.apply_tf_to_pose(x_odom, pose2d_global_odom)
        # Find local goal from global path
        global_path_in_local_xy = pose2d.apply_tf(
                self.global_path_xy,
                pose2d_local_global,
                )
        global_goal_in_local_xy = pose2d.apply_tf(self.goal_xy, pose2d_local_global)
        if np.linalg.norm( global_goal_in_local_xy ) < self.kPepperComfortZoneRadius:
            if not self.goal_is_reached:
              print("goal reached")
            with self.lock:
                self.goal_is_reached = True
                self.cmd_vel_dt = [0]
                self.cmd_vel_uv = [np.array([0,0])]
                self.cmd_vel_w = [0]
                self.cmd_vel_queue_expiry = None
            return
        global_path_in_local_ij = local_occupancy.xy_to_ij(
            global_path_in_local_xy,
            clip_if_outside=False,
            )
        in_mask = local_occupancy.is_inside_ij(global_path_in_local_ij) # filter points not in local
        tsdf_mask = local_tsdf[tuple(global_path_in_local_ij[in_mask].T)] > self.kPepperWidth / 2 # filter points based on tsdf
        in_robot_mask = np.linalg.norm(global_path_in_local_xy[in_mask], axis=-1) < self.kPepperWidth / 2.
        valid_mask = np.logical_or(tsdf_mask, in_robot_mask) # ignore points inside robot
        # alternate lookup, starting from robot
#         last_good_point_index = list(np.array(np.where(valid_mask == False))[0] - 1)
#         last_good_point_index.append(-1)
#         local_goal_ij = global_path_in_local_ij[in_mask][last_good_point_index[0]]
        valid_global_path_in_local_ij = global_path_in_local_ij[in_mask][valid_mask]
        if len(valid_global_path_in_local_ij) == 0:
            # TODO
            print("no path to global waypoint found")
            self.clear_rviz_path_markers()
            self.clear_cmd_vel_queue()
            return
        local_goal_ij = valid_global_path_in_local_ij[-1]
        marker = self.goal_as_marker(local_occupancy.ij_to_xy(local_goal_ij),
            time = rospy.Time.now(),
            frame = self.kLocalMapFrame,
            namespace = 'waypoint_goal',
            marker_type = 1, # Cube
            resolution = self.global_map_8ds.resolution() / 2,
            color = [1,0,0,1],
            )
        self.local_waypoint_pub.publish(marker)
        # Local dijkstra
        sdf_extra_costs = self.dijkstra_extra_costs_from_sdf(local_tsdf)
        local_dijkstra = local_occupancy.dijkstra(
                local_goal_ij,
                mask=np.logical_and(local_tsdf < self.kPepperWidth / 2, local_tsdf >= 0),
                extra_costs=sdf_extra_costs,
                )
        self.local_dijkstra_pub.publish(self.numpy_to_occupancy_grid_msg(local_dijkstra, local_map_frozen, self.kLocalMapFrame, latest_scan.header.stamp))
        path, _ = self.compute_path(
                costmap=local_dijkstra,
                first=local_occupancy.xy_to_ij([0,0]),
                )
        if len(path) <= 1:
            # TODO
            print("local path not found")
            self.clear_rviz_path_markers()
            self.clear_cmd_vel_queue()
            return
        path_downsample_factor = int(len(path) / 30)
        path_downsample_factor = 1 if path_downsample_factor == 0 else path_downsample_factor
        path = path[::path_downsample_factor]
        path = np.array(path).astype(float)
        marker = self.path_as_marker(
                local_occupancy.ij_to_xy(path),
                self.kLocalMapFrame,
                local_occupancy.resolution(),
                'local_path',
                rospy.Time.now(), # - rospy.Duration(0.1),
                )
        self.local_path_pub.publish(marker)
        # Path smoothing
        start_angle = np.arctan2(v_odom[1], v_odom[0])
        #print("start_angle: " , start_angle)
        marker = self.path_as_marker(
                [np.array([0,0]), v_odom[:2]],
                self.kLocalMapFrame,
                local_occupancy.resolution(),
                'start_angle',
                rospy.Time.now(), # - rospy.Duration(0.1),
                color=[1,0,0,1],
                )
        self.debug_marker_2_pub.publish(marker)
        end_angle = np.arctan2(*(path[-1] - path[-2])[::-1])

        from path_smoothing import path_smoothing, curvature_
        smooth_path, immobile = path_smoothing(path, local_tsdf, start_angle, end_angle)
        if len(smooth_path) <= 1:
            # TODO
            print("failed to smoothe path")
            return
        smooth_path_xy = local_occupancy.ij_to_xy(smooth_path)
        curvature = curvature_(smooth_path_xy, start_angle, end_angle)
        vmax = np.sqrt(self.kPepperMaxAcc[0] / np.abs(curvature))
        #print("Max velocity along local path: {}m/s".format(np.min(vmax)))
        marker = self.path_as_marker(
                smooth_path_xy,
                self.kLocalMapFrame,
                local_occupancy.resolution(),
                'smooth_path',
                rospy.Time.now(), # - rospy.Duration(0.1),
                color=[0,0,1,1],
                )
        self.debug_marker_pub.publish(marker)

        # to Cmd Vel
        kTargetVel = 0.25 # [m/s] # Set to conservative value ! 0.1
        target_vel = min(kTargetVel, np.min(vmax))
        target_w = np.pi / 2 / 20 # [rad/s] 90 deg in 10 seconds
        angle_to_goal = np.arctan2(*(smooth_path_xy[-1] - smooth_path_xy[0])[::-1])
        deltaw = (angle_to_goal) % (2 * np.pi)
        if deltaw > np.pi:
            deltaw -= 2 * np.pi # deltaw is positive for left turn, negative for right turn
        # minimum threshold on the angle. This prevents oscillation when
        # imperfect transforms lead to overshooting of the rotational
        # correction
        if abs(deltaw) < (np.pi / 4 / 10):
            deltaw = 0
        #print(deltaw)
        sign = np.sign(deltaw)
        if sign == 0: sign = 1
        target_w = target_w * sign
        min_dx = target_vel * self.kMinCmdVelPeriod # smallest distance between two cmd_vel updates
        xy_latest = smooth_path_xy[0]
        rot_latest = 0
        cmd_vel_dt = [] # intervals between cmd_vels
        cmd_vel_uv = [] # cmd_vel xy in current robot frame
        cmd_vel_w = [] # cmd_vel xy in current robot frame
        for xy in smooth_path_xy[1:]:
            dx = xy - xy_latest
            dx = pose2d.rotate(dx, -rot_latest)
            dx_norm = np.linalg.norm( dx )
            if dx_norm >= min_dx:
                # Compute incremental xy velocities
                xy_latest = xy
                dt =  dx_norm / target_vel 
                # compute twists
                    # idea: slowly turn at constant rate to face local goal.
                remaining_deltaw = deltaw - rot_latest
                remaining_deltat_rot = abs(remaining_deltaw / target_w)
                w = 0
                if remaining_deltat_rot > dt:
                    w = target_w
                    rot_latest += target_w * dt

                # append
                cmd_vel_dt.append( dt )
                cmd_vel_uv.append( dx / dt )
                cmd_vel_w.append( w )


        with self.lock:
            self.cmd_vel_dt = cmd_vel_dt
            self.cmd_vel_uv = cmd_vel_uv
            self.cmd_vel_w = cmd_vel_w
            self.cmd_vel_queue_expiry = rospy.Time.now() + self.kCmdVelQueueDefaultTimeToExpiry



    def global_motion_planning_callback(self, event=None):
        if self.goal_xy is None:
            return
        if self.bnb_last_stamp is not None:
            # get robot tf (using localmapframe as robot position) in global frame
            try:
                time = rospy.Time.now()
                tf_info = [self.kGlobalMapFrame, self.kLocalMapFrame, time] 
                self.tf_listener.waitForTransform(*(tf_info + [self.tf_timeout]))
                tf_ = self.tf_listener.lookupTransform(*tf_info)
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException,
                    TransformException) as e:
                print("[{}.{}] tf for time {}.{} not found: {}".format(
                    rospy.Time.now().secs, rospy.Time.now().nsecs, time.secs, time.nsecs, e))
                return
            robot_pose2d_in_global_map = self.tf_to_pose2d(tf_)
            # Get the path
            path, _ = self.compute_path(
                    costmap=self.global_dijkstra_lowres,
                    first=self.global_map_8ds.xy_to_ij(robot_pose2d_in_global_map[:2]),
                    )
            path_xy = self.global_map_8ds.ij_to_xy(path)
            # Visualize the path
            marker = self.path_as_marker(
                    path_xy,
                    self.kGlobalMapFrame,
                    self.global_map_8ds.resolution(),
                    'global_path')
            self.path_viz_pub.publish(marker)
            path_msg = Path()
            path_msg.header.stamp = rospy.Time.now()
            path_msg.header.frame_id = self.kGlobalMapFrame
            for i in range(len(path_xy)):
                xy = path_xy[i]
                ij = path[i]
                dijkstra_dist = self.global_dijkstra_lowres[ij[0], ij[1]]
                ps = PoseStamped()
                ps.pose.position.x = xy[0]
                ps.pose.position.y = xy[1]
                ps.pose.position.z = dijkstra_dist
                path_msg.poses.append(ps)
            self.path_pub.publish(path_msg)
            self.global_path_xy = path_xy

    def path_as_marker(self, path_xy, frame, scale, namespace, time=None, color=None):
        marker = Marker()
        time = rospy.Time.now() if time is None else time
        marker.header.stamp.secs = time.secs
        marker.header.stamp.nsecs = time.nsecs
        marker.header.frame_id = frame
        marker.ns = namespace
        marker.id = 0
        marker.type = 4 # LINE_STRIP
        marker.action = 0
        s = scale
        marker.scale.x = s
        marker.scale.y = s
        marker.scale.z = s
        if color is None:
            marker.color.g = 1.
            marker.color.a = 0.5
        else:
            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            marker.color.a = color[3]
        marker.points = [Point(xy[0], xy[1], 1) for xy in path_xy]
        return marker
    def clear_rviz_path_markers(self):
        marker = self.path_as_marker(
                [],
                self.kLocalMapFrame,
                0.01,
                'smooth_path',
                rospy.Time.now(),
                color=[0,0,1,1],
                )
        self.debug_marker_pub.publish(marker)
        marker = self.path_as_marker(
                [],
                self.kLocalMapFrame,
                0.01,
                'local_path',
                rospy.Time.now(),
                )
        self.local_path_pub.publish(marker)

    def clear_cmd_vel_queue(self):
        with self.lock:
            self.cmd_vel_dt = [0]
            self.cmd_vel_uv = [np.array([0,0])]
            self.cmd_vel_w = [0]
            self.cmd_vel_queue_expiry = None



    def dijkstra_extra_costs_from_sdf(self, sdf):
        # in meters
        return  ( 
                # penalize steps in zones close to obstacles, proportional to obstacle closeness
                self.kDijkstraTSDFPenalty * (self.kPepperComfortZoneRadius - np.abs(sdf)) *
                (np.abs(sdf) < self.kPepperComfortZoneRadius) +
                # penalize steps in unknown territory
                ( sdf < 0 ) * self.kDijkstraUnknownTerrainPenalty
                )

    def coherence_check_callback(self, event=None):
        # TODO: check diff between global_map and slam_map -> slow dynamic objects
        # TODO: check diff between local_map and slam_map -> fast dynamic objects
        return

    def map_match_localization_callback(self, msg):
        # TODO: maybe align global and slam maps? -> get localization
        # branch and bound?
        # slam map as list of hits
        # create graph
        slam_map = Map2D()
        slam_map.from_msg(msg)
        res_ratio = int(np.log2(self.global_map_8ds.resolution() / slam_map.resolution()))
        for n in range(res_ratio):
            slam_map = slam_map.as_coarse_map2d()
        if abs(slam_map.resolution() - self.global_map_8ds.resolution()) > (slam_map.resolution() * 0.01):
            raise ValueError(
                "Failed to match map resolution ({}) with reference map resolution ({})".format(
                    slam_map.resolution(), self.global_map_8ds.resolution())
                )
        # MapMatcher
        try:
            # reset matcher, a little overhead but safer
            setref_srv = rospy.ServiceProxy('/map_matcher_server/set_reference_map', SetReferenceMap)
            setref_srv(self.numpy_to_occupancy_grid_msg(self.global_map_8ds.occupancy(),
                self.global_map_8ds, self.kGlobalMapFrame))
            # run matcher
            match_srv = rospy.ServiceProxy('/map_matcher_server/match_to_reference', MatchToReference)
            resp = match_srv(
                    self.numpy_to_occupancy_grid_msg(slam_map.occupancy(), slam_map, msg.header.frame_id),
                    0.5,
                    1,
                    0,
                    self.bnb_theta_prior != 0,
                    0,
                    0, 0, 0, 0,
                    self.bnb_theta_prior, 2*np.pi)
        except rospy.ServiceException as e:
            print(e)
            return
        if resp.found_valid_match:
            theta = resp.theta
            score = resp.score
            pose = np.array([resp.i_source_origin_in_reference, resp.j_source_origin_in_reference])
        else:
            theta = None
        if theta is None:
            print("No solution found.")
            return
        self.bnb_theta_prior = theta
#         print("Branch and bound solution found. score: {}, pose: {}, theta: {}".format(score,pose,theta))
        hits = slam_map.as_occupied_points_ij()
        hits = self.bnb.rotate_points_around_map_center(hits, theta, slam_map)
        hits += pose
        debug_match = self.global_map_8ds.occupancy() * 1.
        valid_hits = filter_if_out_of_bounds(hits, debug_match)
        debug_match[tuple(valid_hits.T)] -= 2
        # find Tf between slam and global map
        q = msg.info.origin.orientation
        slam_th = tf.transformations.euler_from_quaternion(np.array([q.x, q.y, q.z, q.w]))[2]
        assert slam_th == 0
        # origin of the slam map in bnb map frame
        o_ij = self.bnb.rotate_points_around_map_center(
                slam_map.xy_to_ij(np.array([[0.,0.]]), clip_if_outside=False), theta, slam_map)
        o_ij += np.array(pose)
        o_th = slam_th + theta
        o_xy = self.global_map_8ds.ij_to_xy(o_ij)[0]
        # inverse of transform
        o_xy_inv = pose2d.inverse_pose2d(np.array([o_xy[0], o_xy[1], o_th]))
        stamp = rospy.Time.now()
        with self.lock:
            self.bnb_last_tf = [
                    (o_xy_inv[0], o_xy_inv[1], 0),
                    tf.transformations.quaternion_from_euler(0,0,o_xy_inv[2]),
                    stamp,
                    self.kGlobalMapFrame,
                    msg.header.frame_id,
                    ]
            self.tf_broadcaster.sendTransform(*self.bnb_last_tf)
            self.bnb_last_stamp  = stamp
        self.global_match_pub.publish(self.numpy_to_occupancy_grid_msg(debug_match, self.global_map_8ds, self.kGlobalMapFrame))
#         print("Published global match")
        return

    def publish_localization_callback(self, event=None):
        if self.bnb_last_stamp is not None:
            with self.lock:
                tf = self.bnb_last_tf[:]
                tf[2] = rospy.Time.now()
                self.tf_broadcaster.sendTransform(*tf)

    def publish_cmd_vel_callback(self, event=None):
        # Check whether cmd_vel_queue has valid entries.
        with self.lock:
            if not self.STOP:
                if self.cmd_vel_queue_expiry:
                    now = rospy.Time.now()
                    if self.cmd_vel_queue_expiry < now:
                        rospy.logwarn("cmd_vel_queue has expired. This could indicate that the motion planner has crashed.")
                if self.cmd_vel_uv:
                    xy = self.cmd_vel_uv.pop(0)
                    dt = self.cmd_vel_dt.pop(0)
                    w  = self.cmd_vel_w.pop(0)
                    # send cmd_vel
                    cmd_vel = Twist()
                    cmd_vel.linear.x = xy[0]
                    cmd_vel.linear.y = xy[1]
                    cmd_vel.angular.z = w
                    self.cmd_vel_pub.publish(cmd_vel)
                    # wait until next command
                    if self.kMinCmdVelPeriod > dt:
                        time.sleep( self.kMinCmdVelPeriod - dt )
            else:
                # Robot might move while control is surrendered. Clear cmd_vel_queue
                self.cmd_vel_dt = [0]
                self.cmd_vel_uv = [np.array([0,0])]
                self.cmd_vel_w = [0]
                self.cmd_vel_queue_expiry = None
#                 print("Did not publish cmd_vel (control not assumed)")

    def shutdown_hook(self):
        with self.lock:
            self.cmd_vel_pub.publish(Twist())


    def run(self):
        # Initialize ros
        rospy.init_node("pepper_motion_planner")
        self.tf_listener = tf.TransformListener()
        self.tf_broadcaster = tf.TransformBroadcaster()
        self.tf_timeout = rospy.Duration(1.0) # [s]
        # Publishers
        self.local_tsdf_pub = rospy.Publisher(self.kLocalTSDFTopic, OccupancyGrid, queue_size=1)
        self.local_dijkstra_pub = rospy.Publisher(self.kLocalDijkstraTopic, OccupancyGrid, queue_size=1)
        self.local_occ_pub = rospy.Publisher(self.kLocalOccupancyTopic, OccupancyGrid, queue_size=1)
        self.global_match_pub = rospy.Publisher(self.kGlobalMatchTopic, OccupancyGrid, queue_size=1)
        self.goal_pub = rospy.Publisher(self.kGlobalGoalTopic, Marker, queue_size=1)
        self.local_waypoint_pub = rospy.Publisher(self.kLocalWaypointTopic, Marker, queue_size=1)
        self.path_viz_pub = rospy.Publisher(self.kGlobalPathVizTopic, Marker, queue_size=1)
        self.path_pub = rospy.Publisher(self.kGlobalPathTopic, Path, queue_size=1)
        self.local_path_pub = rospy.Publisher(self.kLocalPathTopic, Marker, queue_size=1)
        self.debug_marker_pub = rospy.Publisher(self.kDebugMarkerTopic, Marker, queue_size=1)
        self.debug_marker_2_pub = rospy.Publisher(self.kDebugMarker2Topic, Marker, queue_size=1)
        self.debug_marker_3_pub = rospy.Publisher(self.kDebugMarker3Topic, Marker, queue_size=1)
        self.cmd_vel_pub = rospy.Publisher(self.kCmdVelTopic, Twist, queue_size=1)
        # Services
        rospy.Service('stop_autonomous_motion', Trigger, 
                self.stop_autonomous_motion_service_call)
        rospy.Service('resume_autonomous_motion', Trigger, 
                self.resume_autonomous_motion_service_call)
        rospy.Subscriber(self.kLidarFrontTopic, LaserScan, partial(self.lidar_callback, which_lidar="front"), queue_size=1)
        rospy.Subscriber(self.kLidarRearTopic, LaserScan, partial(self.lidar_callback,  which_lidar="rear"), queue_size=1)
        rospy.Subscriber(self.kSlamMapTopic, OccupancyGrid, self.map_match_localization_callback, queue_size=1)
        rospy.Subscriber(self.kNavGoalTopic, PoseStamped, self.global_goal_callback, queue_size=1)
        rospy.Subscriber(self.kOdomTopic, Odometry, self.local_motion_planning_callback, queue_size=1)
        rospy.Timer(self.kGlobalMotionPlanningPeriod, self.global_motion_planning_callback)
        rospy.Timer(self.kCoherenceCheckPeriod, self.coherence_check_callback)
        rospy.Timer(self.kPublishLocalizationPeriod, self.publish_localization_callback)
        rospy.Timer(rospy.Duration(self.kMinCmdVelPeriod), self.publish_cmd_vel_callback)
        rospy.on_shutdown(self.shutdown_hook)
        try:
            rospy.spin()
        except KeyboardInterrupt:
            print("Keyboard interrupt - Shutting down")
        rospy.signal_shutdown("Keyboard Interrupt")



if __name__ == "__main__":
    pass
