from __future__ import print_function
from geometry_msgs.msg import Point, PoseStamped
import json
import os
import pickle
from map2d_ros_tools import ReferenceMapAndLocalizationManager
from matplotlib import pyplot as plt
import numpy as np
from pose2d import Pose2D, apply_tf, inverse_pose2d
import rospy
from std_srvs.srv import Trigger, TriggerResponse
from std_msgs.msg import String
import tf
from frame_msgs.msg import TrackedPersons
from geometry_msgs.msg import Twist
from tf2_ros import TransformException
import threading
from timeit import default_timer as timer
import traceback
from visualization_msgs.msg import Marker, MarkerArray
import datetime
from nav_msgs.msg import OccupancyGrid

from pyIA import ia_planning
import pyIA.actions as actions
from pyIA import state_estimation
from cia import apply_value_operations_to_state, CTaskState, kStateFeatures

VISUALIZE_STATE = False
PUBLISH_STATE_RVIZ = True

GOAL_INFLATION_RADIUS = 0.2
TASK_ALOTTED_TIME_INFLATION = 1.3
SLIDING_WPT = True

IA_SKILLS = [
    actions.Intend(),
    actions.Say(),
    actions.Nudge(),
]

RUN_NUMBER = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
LOG_DIR = "/tmp/ros_ia_node_{}/".format(RUN_NUMBER)
PLANNING_LOG_DIR = os.path.join(LOG_DIR, 'planning/')
try:
    os.makedirs(PLANNING_LOG_DIR)
except: # noqa
    pass

def shutdown_if_fail(function):
    def wrapper_for_routine(*args, **kwargs):
        try:
            function(*args, **kwargs)
        except: # noqa
            traceback.print_exc()
            rospy.signal_shutdown("Exception caught in routine thread.")
            raise
    return wrapper_for_routine

# STATE ESTIMATOR ----------
class IAStateEstimationNodelet(object):
    def __init__(self, parent):
        # variables
        self.fixed_state = None
        self.latest_state_estimate = None
        self.latest_state_estimate_time = None
        self.latest_state_estimate_goal_xy_in_refmap = None
        self.latest_state_estimate_is_stale_flag = False
        self.lock = threading.Lock()  # for avoiding race conditions
        # publishers
        self.state_publishers = {}
        for name in kStateFeatures.keys():
            self.state_publishers[name] = rospy.Publisher(
                "/ros_ia_node/debug/state_" + name, OccupancyGrid, queue_size=1)
            unc = "_uncertainty"
            self.state_publishers[name+unc] = rospy.Publisher(
                "/ros_ia_node/debug/state_" + name + unc, OccupancyGrid, queue_size=1)

    def tracked_persons_update(self, msg, pose2d_tracksframe_in_refmap, map_8ds,
                               robot_pos_in_refmap, robot_radius, robot_goal_xy_in_refmap,
                               ):
        # change state based on detections
#         print("Computing new state estimate")
#         tic = timer()
        time_elapsed = None
        if self.latest_state_estimate_time is not None:
            time_elapsed = (msg.header.stamp - self.latest_state_estimate_time).to_sec()
        latest_state_estimate, fixed_state = \
            state_estimation.state_estimate_from_tracked_persons(
                msg, pose2d_tracksframe_in_refmap,
                robot_pos_in_refmap, robot_radius, robot_goal_xy_in_refmap,
                map_8ds,
                state_memory=self.latest_state_estimate,
                time_elapsed=time_elapsed,
                reuse_state_cache=self.fixed_state,
            )
#         toc = timer()
#         print("Computed. {:.1f}s".format(toc-tic))
        with self.lock:
            self.latest_state_estimate = latest_state_estimate
            self.latest_state_estimate_time = msg.header.stamp
            self.fixed_state = fixed_state
            self.latest_state_estimate_goal_xy_in_refmap = robot_goal_xy_in_refmap
            self.latest_state_estimate_is_stale_flag = False

    def update_routine(self, event=None):
        # in the absence of detections slowly reverts state back to initial
        raise NotImplementedError

    def get_state_estimate(self):
        with self.lock:
            if self.latest_state_estimate_is_stale_flag:
                return None, None, None, None
            latest_state_estimate = self.latest_state_estimate
            fixed_state = self.fixed_state
            latest_state_estimate_goal_xy_in_refmap = self.latest_state_estimate_goal_xy_in_refmap
            stamp = self.latest_state_estimate_time
        return latest_state_estimate, fixed_state, latest_state_estimate_goal_xy_in_refmap, stamp

    def update_state_based_on_task_outcome(self, outcome_node, update_stencil):
        # apply state update to memory
        apply_value_operations_to_state(
            outcome_node.task_state_update,
            self.latest_state_estimate,
            update_stencil
        )

    def publish_state_rviz(self, refmap_frame, hide_uncertain=True):
        if not PUBLISH_STATE_RVIZ:
            return
        state = self.latest_state_estimate
        fixed_state = self.fixed_state
        for feature_name in kStateFeatures.keys():
            val_grid = 1. * state.grid_features_values()[kStateFeatures[feature_name], :]
            unc_grid = 1. * state.grid_features_uncertainties()[kStateFeatures[feature_name], :]
            if hide_uncertain:
                val_grid[np.greater_equal(
                    state.grid_features_uncertainties()[kStateFeatures[feature_name], :],
                    1.)] = np.nan
            val_grid[fixed_state.map.occupancy() > fixed_state.map.thresh_occupied()] = np.nan
            unc_grid[fixed_state.map.occupancy() > fixed_state.map.thresh_occupied()] = np.nan
            val_msg = numpy_to_occupancy_grid_msg(val_grid, fixed_state.map, refmap_frame)
            unc_msg = numpy_to_occupancy_grid_msg(unc_grid, fixed_state.map, refmap_frame)
            unc = "_uncertainty"
            self.state_publishers[feature_name].publish(val_msg)
            self.state_publishers[feature_name + unc].publish(unc_msg)

# PLANNER ------------------


class IAPlanningNode(object):
    def __init__(self, args):
        self.args = args
        # consts
        self.kNavGoalTopic = "/move_base_simple/goal"
        self.kRobotFrame = "base_footprint"
        self.kMaxObstacleVel_ms = 10.  # [m/s]
        self.kTrackedPersonsTopic = "/tracked_persons"
        # IA Planning tools
        self.state_estimator_node = IAStateEstimationNodelet(self)
        self.ia_skills = IA_SKILLS
        if self.args.only_intend:
            self.ia_skills = [actions.OnlyIntend()]
        if self.args.only_say:
            self.ia_skills = [actions.OnlySay()]
        if self.args.only_nudge:
            self.ia_skills = [actions.OnlyNudge()]
        # vars
        self.newest_plan = None
        self.lock = threading.Lock()  # for avoiding race conditions
        self.goal_xy_in_refmap = None
        self.goal_ij_in_refmap = None
        self.latest_goal_msg = None
        self.replan_due_to_new_goal_flag = [False]
        self.planning_interrupt_flag = [False]
        self.STOP = True
        if args.no_stop:
            self.STOP = False
        # ROS
        rospy.init_node('ia_planner', anonymous=True)
        # tf
        self.tf_listener = tf.TransformListener()
        self.tf_br = tf.TransformBroadcaster()
        self.tf_timeout = rospy.Duration(1.)
        self.cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        # Localization Manager
        self.kRobotRadius = rospy.get_param("/robot_radius", 0.3)
        mapname = rospy.get_param("~reference_map_name", "map")
        mapframe = rospy.get_param("~reference_map_frame", "reference_map")
        mapfolder = rospy.get_param("~reference_map_folder", "~/maps")
        map_downsampling_passes = rospy.get_param("~reference_map_downsampling_passes", 3)

        def refmap_update_callback(self_):
            self_.map_8ds = self_.map_
            for _ in range(map_downsampling_passes):
                self_.map_8ds = self_.map_8ds.as_coarse_map2d()
            self_.map_8ds_sdf = self_.map_8ds.as_sdf()
        self.refmap_manager = ReferenceMapAndLocalizationManager(
            mapfolder, mapname, mapframe, self.kRobotFrame,
            refmap_update_callback=refmap_update_callback,
        )
        self.plan_executor_node = PlanExecutorNodelet(self.kRobotRadius, self.refmap_manager,
                                                      export_logs=self.args.export_logs)
        self.plan_executor_node.STOP = self.STOP
        self.plan_executor_node.replan_flag = self.replan_due_to_new_goal_flag  # hacky
        # callback
        rospy.Subscriber(self.kNavGoalTopic, PoseStamped, self.global_goal_callback, queue_size=1)
        rospy.Subscriber(self.kTrackedPersonsTopic, TrackedPersons, self.tracked_persons_callback,
                         queue_size=1)
        # Timers
        rospy.Timer(rospy.Duration(0.1), self.planning_routine)
        rospy.Timer(rospy.Duration(0.1), self.set_goal_routine)
        rospy.Timer(rospy.Duration(0.1), self.replan_detector_routine)
        rospy.Timer(rospy.Duration(0.1), self.plan_executor_routine)
        rospy.Timer(rospy.Duration(0.1), self.plan_executor_debug)
        rospy.Timer(rospy.Duration(0.1), self.plan_executor_node.currenttask_planner_update_routine)
        rospy.Timer(rospy.Duration(0.1), self.plan_executor_node.gestures_control_routine)
        rospy.Timer(rospy.Duration(0.1), self.turn_to_face_goal_callback)
        # Services
        rospy.Service('stop_autonomous_motion', Trigger,
                      self.stop_autonomous_motion_service_call)
        rospy.Service('resume_autonomous_motion', Trigger,
                      self.resume_autonomous_motion_service_call)
        # let's go.
        try:
            self.spin_loop()
        except KeyboardInterrupt:
            rospy.loginfo("Keyboard interrupt - shutting down.")
            rospy.signal_shutdown('KeyboardInterrupt')
        if False:
            rospy.loginfo(self.plan_executor_node.executed_tasks_log)
        if False:
            with open('/tmp/ros_ia_node.log', 'w') as f:
                json.dumps(self.plan_executor_node.executed_tasks_log, f)

    def turn_to_face_goal_callback(self, event=None):
        with self.lock:
            if self.plan_executor_node.task_being_executed is not None:
                return
            if self.refmap_manager.tf_frame_in_refmap is None:
                return
            if self.goal_xy_in_refmap is None:
                return
            p2_refmap_in_robot = inverse_pose2d(Pose2D(self.refmap_manager.tf_frame_in_refmap))
            goal_xy_in_robot = apply_tf(self.goal_xy_in_refmap, p2_refmap_in_robot)
        rospy.loginfo_throttle(5., "turning to face goal")
        gx, gy = goal_xy_in_robot
        angle_to_goal = np.arctan2(gy, gx)  # [-pi, pi]
        w = 0
        if np.abs(angle_to_goal) > ((np.pi / 4) / 10):  # deadzone
            w = 0.3 * np.sign(angle_to_goal)
        if not self.STOP:
            cmd_vel_msg = Twist()
            cmd_vel_msg.angular.z = w
            self.cmd_vel_pub.publish(cmd_vel_msg)

    def spin_loop(self):
        if VISUALIZE_STATE:
            plt.ion()
            fig, (ax1, ax2) = plt.subplots(2)
        while not rospy.is_shutdown():
            if VISUALIZE_STATE:
                state_e, fixed_state, state_goal_xy, _ = self.state_estimator_node.get_state_estimate()
                if state_e is not None:
                    ax1.cla()
                    plt.sca(ax1)
                    ia_planning.visualize_state_feature(state_e, fixed_state, "crowdedness")
                    ax2.cla()
                    plt.sca(ax2)
                    ia_planning.visualize_state_feature(state_e, fixed_state, "crowdedness",
                                                        hide_uncertain=False, uncertainties=True)
                plt.show()
                plt.pause(0.1)
            rospy.sleep(0.01)
        rospy.logwarn("Signal shutdown received. terminating")

    def stop_autonomous_motion_service_call(self, req):
        with self.lock:
            if not self.STOP:
                print("Stopping all sub-controller activities")
                with self.plan_executor_node.lock:
                    self.plan_executor_node.switch_to_new_plan_requested = True
                    self.plan_executor_node.switch_to_plan = None
                    self.newest_plan = None
                    self.plan_executor_node.STOP = True
                    while not self.plan_executor_node.kill_currenttask_planner():
                        pass
                    # in case the turn towards goal was active
                    cmd_vel_msg = Twist()
                    self.cmd_vel_pub.publish(cmd_vel_msg)
                self.STOP = True
        return TriggerResponse(True, "")

    def resume_autonomous_motion_service_call(self, req):
        with self.lock:
            if self.STOP:
                print("Restarting planning and sub-controller activities")
                self.planning_interrupt_flag[0] = True
                with self.plan_executor_node.lock:
                    self.plan_executor_node.STOP = False
                self.STOP = False
        return TriggerResponse(True, "")

    @shutdown_if_fail
    def tracked_persons_callback(self, msg):
        if self.refmap_manager.tf_frame_in_refmap is None:
            # localization not available yet
            return
        if self.goal_xy_in_refmap is None:
            # goal not available yet
            return
        pose2d_tracksframe_in_refmap = None
        if msg.header.frame_id != self.refmap_manager.kRefMapFrame:
            # get tf
            try:
                tf_frame_in_refmap = self.tf_listener.lookupTransform(
                    self.refmap_manager.kRefMapFrame, msg.header.frame_id, rospy.Time(0))
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
                rospy.logwarn_throttle(10., e)
                raise ValueError(e)
            # set for future use
            pose2d_tracksframe_in_refmap = Pose2D(tf_frame_in_refmap)
        robot_pos_in_refmap = Pose2D(self.refmap_manager.tf_frame_in_refmap)
        robot_radius = self.kRobotRadius
        robot_goal_xy_in_refmap = self.goal_xy_in_refmap
        # state update
        self.state_estimator_node.tracked_persons_update(
            msg, pose2d_tracksframe_in_refmap,
            self.refmap_manager.map_8ds,
            robot_pos_in_refmap, robot_radius, robot_goal_xy_in_refmap,
        )
        self.state_estimator_node.publish_state_rviz(self.refmap_manager.kRefMapFrame)

    @shutdown_if_fail
    def global_goal_callback(self, msg):  # x y is in the global map frame
        rospy.loginfo("set_goal message received")
        self.latest_goal_msg = msg

    @shutdown_if_fail
    def set_goal_routine(self, event=None):
        if self.latest_goal_msg is None:
            return
        if self.refmap_manager.tf_frame_in_refmap is None:
            rospy.logwarn_throttle(
                1., "IA: Reference map transform ({} -> {}) not available yet.".format(
                    self.refmap_manager.kRefMapFrame, self.refmap_manager.kFrame,
                )
            )
            return
        # goal might not be in reference map frame. find goal_xy in refmap frame
        try:
            time = rospy.Time.now()
            tf_info = [self.refmap_manager.kRefMapFrame, self.latest_goal_msg.header.frame_id, time]
            self.tf_listener.waitForTransform(*(tf_info + [self.tf_timeout]))
            tf_msg_in_refmap = self.tf_listener.lookupTransform(*tf_info)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException,
                TransformException) as e:
            rospy.logwarn("[{}.{}] tf to refmap frame for time {}.{} not found: {}".format(
                rospy.Time.now().secs, rospy.Time.now().nsecs, time.secs, time.nsecs, e))
            return
        pose2d_msg_in_refmap = Pose2D(tf_msg_in_refmap)
        goal_xy = apply_tf(np.array([self.latest_goal_msg.pose.position.x,
                                     self.latest_goal_msg.pose.position.y]), pose2d_msg_in_refmap)
        # goal ij
        goal_ij = self.refmap_manager.map_8ds.xy_to_ij([goal_xy[:2]], clip_if_outside=False)[0]
        if not self.refmap_manager.map_8ds.is_inside_ij(np.array([goal_ij], dtype=np.float32)):
            rospy.logwarn("Goal (i,j : {}, {}) is outside of reference map (size: {}, {})".format(
                goal_ij[0], goal_ij[1], *self.refmap_manager.map_8ds.occupancy().shape
            ))
            return
        if self.goal_xy_in_refmap is not None:
            if not np.allclose(self.goal_ij_in_refmap, goal_ij):
                self.replan_due_to_new_goal_flag[0] = True
                rospy.loginfo("Global goal has changed.")
        self.goal_xy_in_refmap = goal_xy
        self.goal_ij_in_refmap = goal_ij
        # Goal as marker
        marker = self.goal_as_marker(goal_xy)
        goal_pub = rospy.Publisher("/ia_planner/goal", Marker, queue_size=1)
        goal_pub.publish(marker)
        self.goal_is_reached = False
        rospy.loginfo("New global goal: {}[meters], {}[ij]".format(goal_xy, goal_ij))
        self.latest_goal_msg = None

    @shutdown_if_fail
    def planning_routine(self, event=None):
        # necessary inputs
        with self.lock:
            planning_goal = self.goal_ij_in_refmap
            planning_goal_xy = self.goal_xy_in_refmap
            if planning_goal is None:
                rospy.logwarn_throttle(
                    5., "IA: Goal not set yet. Waiting for global goal message.")
                return
        # receive info from state estimator
        state_e, fixed_state, state_goal_xy, se_stamp = self.state_estimator_node.get_state_estimate()
        if state_e is None:
            rospy.logwarn_throttle(
                5., "IA: State estimate not available yet. Waiting for state estimate \
                (are tracked persons being published?).")
            return
        if not np.allclose(state_goal_xy, planning_goal_xy):
            rospy.logwarn_throttle(
                5., "IA: State estimate goal not equal to planning goal. Waiting for updated state estimate.")
            return

        # run planner, MCTS, get plan
        tic = timer()
        possible_actions = self.ia_skills
        messages = []
        optimistic_sequence = []
        try:
            rospy.loginfo_once("Planning...")
            byproducts = {}
            plan_stochastic_tree, optimistic_sequence, firstfailure_sequence = \
                ia_planning.plan(state_e, fixed_state, possible_actions,
                                 BYPRODUCTS=byproducts, n_trials=self.args.n_trials,
                                 INTERRUPT=self.planning_interrupt_flag, DEBUG_IF_FAIL=False)
        except ValueError as e:
            if e.message in ["Non terminal node has no branches",
                             "No estimates found for root nodes"]:
                optimistic_sequence = []
                messages = e.message
            elif e.message == "low >= high":
                traceback.print_exc()
                optimistic_sequence = []
                messages = e.message
            else:
                raise e
        toc = timer()
        stamp = rospy.Time.now()
        if self.planning_interrupt_flag[0]:
            rospy.loginfo("Planning interrupted after {:.1f}s".format(toc - tic))
            self.planning_interrupt_flag[0] = False
            return
        if toc - tic > 5.:
            rospy.loginfo("Planning took {:.1f}s ({} trials) for goal {}, {}".format(
                toc - tic, self.args.n_trials, planning_goal, stamp))

        # write to planning log
        if self.args.export_logs:
            tic = timer()
            planning_log = {
                "state_e_stamp": se_stamp,
                "planning_stamp": stamp,
                "state_e": state_e.serialize(),
                "fixed_state": fixed_state.serialize(),
                "possible_actions": possible_actions,
                "optimistic_sequence": optimistic_sequence,
                "messages": messages,
            }
#                 "plan_stochastic_tree": plan_stochastic_tree,
#                 "firstfailure_sequence": firstfailure_sequence,
#                 "byproducts": byproducts,
            stamp_str = "{:010d}_{:09d}.log".format(stamp.secs, stamp.nsecs)
            log_path = os.path.join(PLANNING_LOG_DIR, stamp_str)
            with open(log_path, 'wb') as f:
                pickle.dump(planning_log, f)
            toc = timer()

        # split long tasks into subtasks

        # Checks, store plan
        with self.lock:
            if not optimistic_sequence:
                rospy.logwarn_throttle(5., "No solution found for plan.")
                return
            if self.STOP:
                rospy.loginfo_throttle(5., "Stopped, discarding new plan")
                return
            if self.goal_ij_in_refmap is None or planning_goal is None:
                rospy.loginfo("No global goal, discarding new plan.")
                return
            if not np.allclose(self.goal_ij_in_refmap, planning_goal):
                rospy.loginfo("Global goal changed during planning, discarding new plan.")
                return
            else:
                self.newest_plan = {
                    "stamp": stamp,
                    "plan_stochastic_tree": plan_stochastic_tree,
                    "optimistic_sequence": optimistic_sequence,
                    "firstfailure_sequence": firstfailure_sequence,
                    "initial_state": state_e
                }

    @shutdown_if_fail
    def replan_detector_routine(self, event=None):
        # init, set plan as soon as available
        if self.plan_executor_node.plan_being_executed is None:
            if self.newest_plan is None:
                return  # no plan is found yet, do nothing and wait
            self.plan_executor_node.switch_to_new_plan_requested = True
            self.plan_executor_node.switch_to_plan = self.newest_plan
            return
        with self.lock:
            pass
#             newest_plan = self.newest_plan
#             plan_being_executed = self.plan_executor_node.plan_being_executed
        # has the global goal changed? Trigger replan
        if self.replan_due_to_new_goal_flag[0]:
            with self.lock:
                self.replan_due_to_new_goal_flag[0] = False
                self.planning_interrupt_flag[0] = True
                self.state_estimator_node.latest_state_estimate_is_stale_flag = True
                self.newest_plan = None
            with self.plan_executor_node.lock:
                self.plan_executor_node.switch_to_new_plan_requested = True
                self.plan_executor_node.switch_to_plan = None
            return
        # Have we reached the goal? Stop planning.
        if self.refmap_manager.tf_frame_in_refmap is not None and self.goal_xy_in_refmap is not None:
            robot_xy_in_refmap = Pose2D(self.refmap_manager.tf_frame_in_refmap)[:2]
            robot_goal_xy_in_refmap = self.goal_xy_in_refmap
            if np.linalg.norm(robot_xy_in_refmap - robot_goal_xy_in_refmap) < (
                    self.kRobotRadius + GOAL_INFLATION_RADIUS):
                if self.plan_executor_node.plan_being_executed is not None:
                    rospy.loginfo("Goal reached, exiting current plan")
                    with self.plan_executor_node.lock:
                        self.plan_executor_node.switch_to_new_plan_requested = True
                        self.plan_executor_node.switch_to_plan = None
                        self.newest_plan = None
                    with self.lock:
                        self.goal_xy_in_refmap = None
                        self.goal_ij_in_refmap = None
            # have we detected task success? Notify plan executor
            task_being_executed = self.plan_executor_node.task_being_executed
            if task_being_executed is not None:
                tbe_taskinfo = task_being_executed["taskinfo"]
                tbe_path = tbe_taskinfo["taskfullpath"]
                tbe_target_pos = tbe_path[tbe_taskinfo["tasktargetpathidx"]]
                if np.linalg.norm(robot_xy_in_refmap - tbe_target_pos) < (
                        self.kRobotRadius + GOAL_INFLATION_RADIUS):
                    with self.plan_executor_node.lock:
                        self.plan_executor_node.task_succeeded_flag = True

        # is the current plan still valid? if not trigger replan
        if False:  # TODO
            with self.plan_executor_node.lock:
                self.plan_executor_node.switch_to_new_plan_requested = True
                self.plan_executor_node.switch_to_plan = self.newest_plan
            return
        # is the newest plan very different from the current plan? trigger replan
        # is the newest plan a much better best cost? a much worse worst cost?
        if False:  # TODO
            with self.plan_executor_node.lock:
                self.plan_executor_node.switch_to_new_plan_requested = True
                self.plan_executor_node.switch_to_plan = self.newest_plan
            return
        # is the state i'm in similar to the state predicted by my plan?
        if False:  # TODO
            with self.plan_executor_node.lock:
                self.plan_executor_node.switch_to_new_plan_requested = True
                self.plan_executor_node.switch_to_plan = self.newest_plan
            return

    @shutdown_if_fail
    def plan_executor_routine(self, event=None):
        self.plan_executor_node.routine(self.state_estimator_node)

    @shutdown_if_fail
    def plan_executor_debug(self, event=None):
        self.plan_executor_node.debug_routine(self.state_estimator_node)

    def goal_as_marker(self, goal_xy,
                       time=None,
                       frame=None,
                       namespace='global_goal',
                       marker_type=2,  # SPHERE
                       resolution=None,
                       color=[0., 1., 0., 1.],  # rgba
                       ):
        if time is None:
            time = rospy.Time.now()
        if frame is None:
            frame = self.refmap_manager.kRefMapFrame
        if resolution is None:
            resolution = self.refmap_manager.map_8ds.resolution()
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

# EXECUTOR -------------------------------


class PlanExecutorNodelet(object):
    def __init__(self, robot_radius, refmap_manager, export_logs=False):
        # needed external state (read only)
        self.refmap_manager = refmap_manager
        self.kRobotRadius = robot_radius
        self.STOP = True
        # Constants
        self.export_logs = export_logs
        self.kWaypointRadius = 0.3
        self.kWaypointDist_m = {"Intend": 3, "Say": 3, "Nudge": 1.5}
        # self state
        self.plan_being_executed = None
        self.consecutive_failures_in_current_plan = 0
        self.task_succeeded_flag = False
        self.progress_in_plan = 0  # node id of current node in plan
        self.progress_in_plan_seq = 0  # index of current task in sequence
        self.switch_to_new_plan_requested = False
        self.switch_to_plan = None
        self.task_being_executed = None
        self.executed_tasks_log = []
        self.current_waypoint_index = 0
        self.lock = threading.Lock()  # for avoiding race conditions
        self.gesture_to_execute = None
        self.plan_execution_log = []  # [(event, associated plan), ...]
        # publishers
        self.responsive_wpt_pub = rospy.Publisher("/responsive/waypoint", Marker, queue_size=1)
        self.rvo_planner_wpt_pub = rospy.Publisher("/rvo_planner/waypoint", Marker, queue_size=1)
        self.speech_pub = rospy.Publisher("/speech", String, queue_size=1)
        self.gestures_pub = rospy.Publisher("/gestures", String, queue_size=1)
        self.taskevent_pub = rospy.Publisher("/ros_ia_node/debug/task_events", String, queue_size=1)
        # services
        self.enable_rvo_planner_serv = rospy.ServiceProxy("/rvo_planner/resume_autonomous_motion", Trigger)
        self.enable_responsive_serv = rospy.ServiceProxy("/responsive/resume_autonomous_motion", Trigger)
        self.disable_rvo_planner_serv = rospy.ServiceProxy("/rvo_planner/stop_autonomous_motion", Trigger)
        self.disable_responsive_serv = rospy.ServiceProxy("/responsive/stop_autonomous_motion", Trigger)
        self.enable_gestures_serv = rospy.ServiceProxy("enable_gestures", Trigger)
        self.disable_gestures_serv = rospy.ServiceProxy("disable_gestures", Trigger)

    def routine(self, state_estimator_node):
        # if no plan exists yet, sleep
        if self.plan_being_executed is None and not self.switch_to_new_plan_requested:
            return
        # check if switch to new plan is requested
        if self.switch_to_new_plan_requested:
            rospy.loginfo("Switching to new plan")
            self.finish_and_log_task(state_estimator_node, success=0, stopped_early=True)
            self.start_plan(self.switch_to_plan)
            self.switch_to_new_plan_requested = False
            return
        # No task is running, pick next task in plan
        if self.task_being_executed is None:
            rospy.loginfo("No task is running, pick next task in plan")
            rospy.loginfo(plan_as_string(self.plan_being_executed))
            self.start_next_task(state_estimator_node)
            return
        # Task is running, check status
        nowstamp = rospy.Time.now()
#         state = state_estimator_node.get_state_estimate()
        task_end_condition_reached = False
        # is allowed time elapsed?
        if nowstamp > self.task_being_executed["time_limit"]:
            rospy.loginfo("Task timed out.")
            task_end_condition_reached = True
            assume_succeeded = False
        # have we moved by certain amount?
        if False:  # TODO
            task_end_condition_reached = True
            assume_succeeded = False
        # has the task succeeded? call from planner
        if self.task_succeeded_flag:  # TODO
            task_end_condition_reached = True
            assume_succeeded = True
        # Exit task
        if task_end_condition_reached:
            current_node = self.plan_being_executed["plan_stochastic_tree"].nodes[
                self.progress_in_plan]
            existing_branches = [branch for branch in current_node.childbranchbundles[0].branches
                                 if branch.target in self.plan_being_executed["plan_stochastic_tree"].nodes]
            success_nodes = [self.plan_being_executed["plan_stochastic_tree"].nodes[branch.target]
                             for branch in existing_branches
                             if branch.meta["success"] == 1]
            failure_nodes = [self.plan_being_executed["plan_stochastic_tree"].nodes[branch.target]
                             for branch in existing_branches
                             if branch.meta["success"] < 1]
            if not success_nodes:
                rospy.logwarn("No success nodes found")
                rospy.logwarn(current_node.id)
                rospy.logwarn(current_node.childbranchbundles[0].branches)
                rospy.logwarn([branch.target for branch in current_node.childbranchbundles[0].branches])
                rospy.logwarn(self.plan_being_executed["plan_stochastic_tree"].nodes)
                ia_planning.visualize_mc_estimate_tree(self.plan_being_executed["plan_stochastic_tree"])
                plt.show()
            if not failure_nodes:
                rospy.logwarn("No failure nodes found")
            # pick outcome based on success report from planner
            # TODO pick outcome state in plan tree which most matches measured state?
            if assume_succeeded:
                closest_outcome_node = success_nodes[0]
                if closest_outcome_node.is_terminal:
                    rospy.logwarn("Plan success, target reached.")
            else:
                closest_outcome_node = failure_nodes[0]
                self.consecutive_failures_in_current_plan += 1
            # TODO update state based on outcome?
            # - catch 22, pick outcome based on state, state updated based on outcome?
            # generate stencils
            with self.lock:
                current_task = self.task_being_executed
                taskinfo = current_task["taskinfo"]
                current_task_path = taskinfo["taskfullpath"]
                current_waypoint_index = self.current_waypoint_index
                map_ = self.refmap_manager.map_8ds
            path_xy_so_far = current_task_path[:current_waypoint_index]
            path_ij_so_far = map_.xy_to_ij(path_xy_so_far)
            stencil_map = map_.empty_like()
            for i, j in path_ij_so_far:
                stencil_map._occupancy[i, j] = 1.
            kStencilRadius = 1.
            update_stencil = stencil_map.as_sdf() < kStencilRadius
            # apply update
            state_estimator_node.update_state_based_on_task_outcome(
                closest_outcome_node, update_stencil.astype(np.float32))
            state_estimator_node.publish_state_rviz(self.refmap_manager.kRefMapFrame)
            self.finish_and_log_task(
                state_estimator_node, success=assume_succeeded, stopped_early=False)
            self.progress_in_plan = closest_outcome_node.id
        else:
            # do nothing, keep executing plan
            return

    def start_plan(self, plan):
        # TODO (debug) publish state_0 before plan enacted
        with self.lock:
            self.plan_being_executed = plan
            self.consecutive_failures_in_current_plan = 0
            if plan is None:
                self.progress_in_plan = None
                rospy.loginfo("PLAN: None")
            else:
                self.progress_in_plan = plan["plan_stochastic_tree"].root_nodes[0]
#                 rospy.loginfo(plan_as_string(plan))
            if self.export_logs:
                plan_stamp = None
                if plan is not None:
                    plan_stamp = plan["stamp"]
                new_log_entry = {
                    "event": "started_new_plan",
                    "plan_stamp": plan_stamp,
                    "switch_stamp": rospy.Time.now(),
                }
                self.plan_execution_log.append(new_log_entry)
                log_path = os.path.join(LOG_DIR, "plan_execution.log")
                with open(log_path, 'wb') as f:
                    pickle.dump(self.plan_execution_log, f)

    def start_next_task(self, state_estimator_node):
        with self.lock:
            state_before_task = state_estimator_node.get_state_estimate()
            startstamp = rospy.Time.now()
            current_node = self.plan_being_executed["plan_stochastic_tree"].nodes[
                self.progress_in_plan]
            resolution = state_estimator_node.fixed_state.map.resolution()
            if self.consecutive_failures_in_current_plan > 0:
                rospy.logwarn("Failure detected: require replan")
                self.switch_to_new_plan_requested = True
                self.switch_to_plan = None
                self.replan_flag[0] = True
                return False  # starting task failed. try again on next loop iteration
            if not current_node.childbranchbundles:
                rospy.logwarn("Could not start task: Current node {} has no children".format(current_node))
                self.switch_to_new_plan_requested = True
                self.switch_to_plan = None
                self.replan_flag[0] = True
                return False  # starting task failed. try again on next loop iteration
            taskinfo = current_node.childbranchbundles[0].meta
            # get allotted time
            resultidx = taskinfo["tasktargetpathidx"]
            action = taskinfo["taskaction"]
            taskinfo["taskstartidx"] = current_node.startidx
            task_state = CTaskState(current_node.path_state, current_node.startidx, resultidx, resolution)
            predicted_duration = action.predict_duration(task_state, params=None)
            # TODO ensure taskinfo is suitable to task
            rospy.loginfo("New task predicted time: {:.2f}s, cost: {}".format(
                predicted_duration, '-'))
            if not self.start_currenttask_planner(taskinfo):
                rospy.logwarn("Could not start task!")
                rospy.logwarn(taskinfo)
                return False  # starting task failed. try again on next loop iteration
            self.task_being_executed = {
                "taskinfo": taskinfo,
                "state_before_task": state_before_task,
                "start_stamp": startstamp,
                "predicted_duration": predicted_duration,
                "time_limit": startstamp + rospy.Duration(predicted_duration * TASK_ALOTTED_TIME_INFLATION),
            }
            self.task_succeeded_flag = False
            self.taskevent_pub.publish(String(
                "Task started: " + action.typestr() + ", path length: " + str(task_state.path_length())
            ))

    def start_currenttask_planner(self, taskinfo):
        new_task_action = taskinfo["taskaction"]
        new_task_path = taskinfo["taskfullpath"]
        new_task_target_pos = new_task_path[taskinfo["tasktargetpathidx"]]
        # TODO:
        # actually call planner node service
        rospy.loginfo("Calling planner for task: {} {}".format(
            new_task_action.typestr(), new_task_target_pos))
        # WIP ---------- Call enable service
        # enable / disable planner
        if new_task_action.typestr() in ["Intend", "Say"]:
            self.enable_rvo_planner_serv()
        elif new_task_action.typestr() in ["Crawl", "Nudge"]:
            self.enable_responsive_serv()
        else:
            rospy.logerr("Unknown task action {}".format(new_task_action.typestr()))
        # enable / disable gestures
        if new_task_action.typestr() in ["Nudge"]:
            self.enable_gestures_serv()
        else:
            self.disable_gestures_serv()
        # enable / disable speech
        if new_task_action.typestr() in ["Say"]:
            self.speech_pub.publish(String("Excuse me, I'm coming through!"))
            self.gesture_to_execute = String("animations/Stand/Gestures/Give_5")
        return True

    @shutdown_if_fail
    def gestures_control_routine(self, event=None):
        kGestureCooldownTime = 3.
        with self.lock:
            gesture_to_execute = self.gesture_to_execute
        if gesture_to_execute is not None:
            with self.lock:
                if not self.STOP:
                    self.gestures_pub.publish(gesture_to_execute)
            rospy.sleep(kGestureCooldownTime)
            with self.lock:
                if not self.STOP:
                    self.gestures_pub.publish(String("animations/Stand/Gestures/Desperate_4"))
                self.gesture_to_execute = None

    @shutdown_if_fail
    def currenttask_planner_update_routine(self, event=None):
        with self.lock:
            if self.refmap_manager.tf_frame_in_refmap is None:
                return
            if self.task_being_executed is None:
                return
            if self.STOP:
                return
            current_task = self.task_being_executed
            taskinfo = current_task["taskinfo"]
            current_task_action = taskinfo["taskaction"]
            startidx = taskinfo["taskstartidx"]
            resultidx = taskinfo["tasktargetpathidx"]
            current_task_path = taskinfo["taskfullpath"][startidx:resultidx]
#             current_task_target_pos = current_task_path[taskinfo["tasktargetpathidx"]]
            # get next waypoint
            robot_xy_in_refmap = Pose2D(self.refmap_manager.tf_frame_in_refmap)[:2]
            current_waypoint_in_refmap = np.array(current_task_path[self.current_waypoint_index])
            # move waypoint forward by Xm if reached
            waypoint_dist = self.kWaypointDist_m[current_task_action.typestr()]
            if SLIDING_WPT:
                for index in range(self.current_waypoint_index, len(current_task_path)):
                    new_pos = np.array(current_task_path[index])
                    if np.linalg.norm(new_pos - robot_xy_in_refmap) >= waypoint_dist:
                        break
                    self.current_waypoint_index = index
                    current_waypoint_in_refmap = new_pos
            else:
                if np.linalg.norm(robot_xy_in_refmap - current_waypoint_in_refmap) < (
                        self.kRobotRadius + self.kWaypointRadius) or self.current_waypoint_index == 0:
                    for index in range(self.current_waypoint_index, len(current_task_path)):
                        new_pos = np.array(current_task_path[index])
                        if np.linalg.norm(new_pos - current_waypoint_in_refmap) >= waypoint_dist:
                            break
                        result_index = index
                        result_waypoint_in_refmap = new_pos
                    self.current_waypoint_index = result_index
                    current_waypoint_in_refmap = result_waypoint_in_refmap
            # send waypoint
            from visualization_msgs.msg import Marker
            mk_msg = Marker()
            mk_msg.header.frame_id = self.refmap_manager.kRefMapFrame
            mk_msg.header.stamp = rospy.Time.now()
            mk_msg.type = Marker.CUBE
            mk_msg.color.a = 1
            mk_msg.color.r = 1
            mk_msg.color.g = 1
            mk_msg.scale.x = self.kWaypointRadius
            mk_msg.scale.y = self.kWaypointRadius
            mk_msg.scale.z = self.kWaypointRadius
            mk_msg.pose.position.x = current_waypoint_in_refmap[0]
            mk_msg.pose.position.y = current_waypoint_in_refmap[1]
            mk_msg.text = current_task_action.typestr()
            # some planners also get a path
            for xy in current_task_path:
                p = Point()
                p.x = xy[0]
                p.y = xy[1]
                mk_msg.points.append(p)
            # publish on relevant topics
            if current_task_action.typestr() in ["Intend", "Say"]:
                pub = self.rvo_planner_wpt_pub
                pub.publish(mk_msg)
            elif current_task_action.typestr() in ["Crawl", "Nudge"]:
                pub = self.responsive_wpt_pub
                pub.publish(mk_msg)
            else:
                rospy.logerr("Unknown task action {}".format(current_task_action.typestr()))

    def finish_and_log_task(self, state_estimator_node, success, stopped_early=False):
        with self.lock:
            # Blocks until it finishes
            if self.task_being_executed is None:
                return
            # finish current action
            stopped_stamp = rospy.Time.now()
            start_stamp = self.task_being_executed["start_stamp"]
            predicted_duration = self.task_being_executed["predicted_duration"]
            task_duration = (stopped_stamp - start_stamp).to_sec()
            rospy.loginfo("Task execution time: {:.2f}s, pred: {:.2f}s, {:.2f}".format(
                task_duration, predicted_duration, task_duration / predicted_duration
            ))
            while not self.kill_currenttask_planner():
                pass
            # disable child skill node
            # log final state
            state_after_task = state_estimator_node.get_state_estimate()
            task = self.task_being_executed
            task["state_after_task"] = state_after_task
            task["stopped_early"] = stopped_early
            task["stopped_stamp"] = stopped_stamp
            task["succeeded"] = success
            self.executed_tasks_log.append(task)
            self.task_being_executed = None
            self.current_waypoint_index = 0
            success_str = "(success)" if success else "(failure)"
            success_str = "(stopped)" if stopped_early else success_str
            msg_str = "Finished task {}".format(success_str) 
            rospy.loginfo(msg_str)
            self.taskevent_pub.publish(String(msg_str))

    def kill_currenttask_planner(self):
        self.disable_responsive_serv()
        self.disable_rvo_planner_serv()
        return True

    def debug_routine(self, state_estimator_node):
        with self.lock:
            plan_being_executed = self.plan_being_executed
            task_being_executed = self.task_being_executed
        # publish plan being executed
        plan_pub = rospy.Publisher("/ros_ia_node/debug/plan_being_executed", MarkerArray, queue_size=1)
        ma = MarkerArray()
        if plan_being_executed is None:
            damk = Marker()
            damk.action = Marker.DELETEALL
            ma.markers.append(damk)
        else:
            pass  # TODO
#             tree = plan_being_executed["plan_stochastic_tree"]
        plan_pub.publish(ma)
        # publish task being executed
        task_pub = rospy.Publisher("/ros_ia_node/debug/task_being_executed", MarkerArray, queue_size=1)
        ma = MarkerArray()
        if task_being_executed is None:
            damk = Marker()
            damk.action = Marker.DELETEALL
            ma.markers.append(damk)
        else:
            try:
                startidx = task_being_executed["taskinfo"]["taskstartidx"]
                resultidx = task_being_executed["taskinfo"]["tasktargetpathidx"]
                path = task_being_executed["taskinfo"]["taskfullpath"][startidx:resultidx]
                color = task_being_executed["taskinfo"]["taskaction"].color()
            except ValueError:
                rospy.logerr(task_being_executed)
                raise ValueError
            mk = Marker()
            mk.header.frame_id = self.refmap_manager.kRefMapFrame
            mk.type = Marker.LINE_STRIP
            mk.color.r = color[0]
            mk.color.g = color[1]
            mk.color.b = color[2]
            mk.color.a = 1.
            mk.scale.x = 0.1
            for xy in path:
                p = Point()
                p.x = xy[0]
                p.y = xy[1]
                mk.points.append(p)
            ma.markers.append(mk)
        task_pub.publish(ma)


def plan_as_string(plan):
    string = "PLAN:\n"
    if plan is None:
        return string + 'None\n'
    for key in plan:
        string += "    " + str(key)
        val = plan[key]
        if key in ["firstfailure_sequence", "optimistic_sequence"]:
            valstr = "\n"
            for (vtask, vest) in val:
                valstr += "        "
                if "taskaction" in vtask:
                    valstr += " - action: " + vtask["taskaction"].typestr()
                if "prob" in vest:
                    valstr += " - prob: " + str(vest["prob"])
                valstr += "\n"
        else:
            valstr = str(val) + "\n"
        string += ": " + valstr
    return string

def numpy_to_occupancy_grid_msg(arr, ref_map2d, frame_id, time=None):
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
