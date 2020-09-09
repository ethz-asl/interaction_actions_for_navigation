import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import multiprocessing
import threading
from pepper_2d_simulator import PepperRLEnv
from pepper_2d_simulator import populate_PepperRLEnv_args, check_PepperRLEnv_args
from pepper_2d_simulator import linear_controller
from pyniel.pyplot_tools.realtime import plot_closeall_button
from pyniel.python_tools.timetools import WalltimeRate
try:
    import rospy
    from std_msgs.msg import Header
    from geometry_msgs.msg import PoseStamped
    from costmap_converter.msg import ObstacleArrayMsg
    from rosgraph_msgs.msg import Clock
    ROS_FOUND = True
except:# ModuleNotFoundError, ImportError
    print("Ros was not found, disabled.")
    ROS_FOUND = False
import time
from timeit import default_timer as timer
import pose2d

from pyIA.agents import Innerstate
import pyIA.actions as actions
from pyIA.arguments import populate_sim_args
from pyIA.simulator import Environment
from pyIA import state_estimation
from pyIA import ia_planning
from cia import unserialize_cstate

SIM_AGENTS_POSSIBLE_ACTIONS = [actions.Intend()]
ROBOT_POSSIBLE_ACTIONS = [actions.Intend(), actions.Say(), actions.Crawl()]

STATELESS_NAIVE_PLAN = True # experimental, should not change behavior and speed things up as state is ignored in naive path planning

MAX_TASK_ALOTTED_TIME = 10.
NAIVE_AGENT_LIVE_WAYPOINT_DISTANCE = 1.  # meters
PUBLISH_GOAL_EVERY = 0.
kCmdVelTopic = "/cmd_vel"
kNavGoalTopic = "/move_base_simple/goal"
kObstaclesTopic = "/sim_obstacles"
kFixedFrame = "sim_map"


def parse_iaenv_args(args=None):
    import argparse
    ## Arguments
    parser = argparse.ArgumentParser(description='Test node for the pepper RL path planning algorithm.')
    populate_iaenv_args(parser)
    ARGS, unknown_args = parser.parse_known_args(args=args)

    # deal with unknown arguments
    # ROS appends some weird args, ignore those, but not the rest
    if unknown_args:
        non_ros_unknown_args = rospy.myargv(unknown_args)
        if non_ros_unknown_args:
            print("unknown arguments:")
            print(non_ros_unknown_args)
            parser.parse_args(args=["--help"])
            raise ValueError

    return ARGS

def populate_iaenv_args(parser):
    parser.add_argument(
            '--realtime',
            action='store_true',
            help='Have the simulator run as close to real-time as possible',
    )
    # Input source
    sourcegroup = parser.add_mutually_exclusive_group()
    # default input is through step(actions) function
    sourcegroup.add_argument(
            '--autopilot',
            action='store_true',
            help='drive robot internally, rather than listening to cmd_vel',
    )
    sourcegroup.add_argument(
            '--cmd_vel',
            action='store_true',
            help='drive robot through cmd_vel topic',
    )
    # Output to ros
    parser.add_argument('--force-publish-ros', action='store_true', 
            help="""By default the training script enables the no-ros argument in the
            environment. force-publish-ros restores the environment's default ros behavior.""")
    parser.add_argument('--no-ros', action='store_true', help='disable ros',)
    parser.add_argument('--no-goal', action='store_true', help='dont publish goals for agent 0',)
    parser.add_argument('--naive-plan', action='store_true', help='Bypass ia planning for agents.',)
    parser.add_argument('--plan-async', action='store_true', help='Bypass ia planning for agents.',)
    parser.add_argument('--no-stop', action='store_true', help='hot start (DANGER)',)
    parser.add_argument('--no-pass-through', action='store_true', help='If enabled, agent 0 is unable to pass through other agents',)
    parser.add_argument('--shutdown-on-success', action='store_true', help='stop simulation if goal is reached',)
    parser.add_argument('--max-runtime', type=float, default=np.inf, help='max simulation runtime',)
    parser.add_argument('--delay-start', action='store_true', help='delay start',)
    parser.add_argument('--pre-step-once', action='store_true', help="""
                        move-base refuses to register goal until it has received data.
                        this is a hack to get it running.""",)
    parser.add_argument('--verbose', action='store_true', help='print more information',)
    parser.add_argument('--dt', type=float, default=0.2, help='simulation time increment')
    # add IA args
    populate_sim_args(parser)
    # add RL args
    populate_PepperRLEnv_args(parser, exclude_IA_args=True)

def check_iaenv_args(args):
    # Override disabled pepperrlenv arguments
    args.bounce_reset_vote = False
    args.circular_scenario = False
    args.n_agents = "unassigned"
    args.mode = "BOUNCE" # agents see each other
    args.deterministic = True

    # check incompatibilities
    if not args.no_ros and not args.continuous:
        pass
#         print(args)
#         raise ValueError("In ros mode only continuous control is supported")

PPO_model_type_letter = "E"
def parse_training_args(ignore_unknown=True, parser=None, env_populate_args_func=None, env_name="PepperRLSim"):
    import argparse
    if parser is None:
        parser = argparse.ArgumentParser()
    # Training args
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument( '--resume-latest', action='store_true',
            help='loads latest compatible saved model and continues training from there.',)
    resume.add_argument('--resume-from', type=str, default='',
            help='loads a specific checkpoint',)
    parser.add_argument('--ignore-compatibility', action='store_true', help='suppresses errors when incompatible models are loaded')
    parser.add_argument('--checkpoint-root-dir', type=str, default='~/PPO2/models', help='root directory for storing models of all runs')
    parser.add_argument('--summary-root-dir', type=str, default='~/PPO2/summaries', help='root directory for storing logs of all runs')
    parser.add_argument('--dry-run', action='store_true', help='overrides save directories, redirects to /tmp directory')
    trparser = parser.add_argument_group('Training', 'parameters for the model training.')
#     trparser.add_argument('--environment', type=str, default='PepperRLSim')
    trparser.add_argument('--total-timesteps', type=int, default=10000000)
    trparser.add_argument('--max-n-relative-obstacles', type=int, default=10)
    trparser.add_argument('--policy', type=str, default='MlpPolicy')
    trparser.add_argument('--model', type=str, default='PPO2')
    trparser.add_argument('--a-learning-rate', '--lr-a', type=float, default=0.00001)
    trparser.add_argument('--c-learning-rate', '--lr-c', type=float, default=0.00002)
    trparser.add_argument('--entropy-coefficient', '--ent', type=float, default=0.01)
    trparser.add_argument('--cliprange', type=float, default=0.2)
    trparser.add_argument('--batch-size', type=int, default=2048) # n steps in a batch
    trparser.add_argument('--minibatch-size', type=int, default=512)
    trparser.add_argument('--max-episode-length', type=int, default=1024)
    trparser.add_argument('--optimization-iterations', type=int, default=4)
    trparser.add_argument('--gamma', type=float, default= 0.99)
    trparser.add_argument('--lmbda', type=float, default= 0.95)
    trparser.add_argument('--random-seed', type=int, default=np.random.randint(1000000))
    trparser.add_argument('--clip-gradients', type=float, default=0.5) # max grad norm
    trparser.add_argument( '--progress-to-file', action='store_true', help='also log progress in a file. useful for cluster execution.',)
    # Environment args
    if env_populate_args_func is not None:
        envparser = parser.add_argument_group('PepperRLEnv', 'these arguments are passed to the PepperRLEnv object used for training the model')
        env_populate_args_func(envparser)

    if ignore_unknown:
        args, unknown_args = parser.parse_known_args()
    else:
        args = parser.parse_args()
        unknown_args = []

    args.environment = env_name

    # Paths --------------------------------------------
    # expanduser in directory paths
    args.checkpoint_root_dir = os.path.expanduser(args.checkpoint_root_dir)
    args.summary_root_dir = os.path.expanduser(args.summary_root_dir)
    if args.dry_run:
        args.checkpoint_root_dir = '/tmp/PPO2/models'
        args.summary_root_dir = '/tmp/PPO2/summaries'

    # Create full paths for this run
    from datetime import datetime
    date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    discrete_type_letter = 'c' if args.continuous else 'd'
    relobs_type_letter = '_rel{}'.format(args.max_n_relative_obstacles) if args.add_relative_obstacles else ''
    env_policy_folder =  args.environment + '_' + args.model + '_' + args.policy + '_' + PPO_model_type_letter + discrete_type_letter + relobs_type_letter
    args.run_folder = 'run_' + date_time  + '_' + "{}".format(args.random_seed)
    args.run_name = env_policy_folder + '_' + args.run_folder
    args.summary_path = args.summary_root_dir + '/' + args.run_name
    args.checkpoint_path = args.checkpoint_root_dir + '/' + args.run_name

    # if resuming, change checkpoint path to previous one.
    # however, keep logging path the same
    # resume from latest
    if args.resume_latest:
        # find compatible models
        from os import listdir
        models = listdir(args.checkpoint_root_dir)
        compatible_models = sorted([model_dir for model_dir in models if env_policy_folder in model_dir])
        if args.ignore_compatibility:
            compatible_models = sorted(models)
        args.latest_model_folder = None
        if compatible_models:
            args.latest_model_folder = compatible_models[-1]
        # change to latest model
        if args.latest_model_folder is None:
            print(env_policy_folder)
            raise ValueError("No model to resume from. Found models: {} of which compatible models: {}".format(
                models, compatible_models))
        args.resume_from = args.checkpoint_root_dir + '/' + args.latest_model_folder
    if args.resume_from != '':
        print("Resuming model {}".format(args.resume_from))
        if env_policy_folder not in args.resume_from:
            print("WARNING: detected potential incompatibility between model to load and current model")
            print(env_policy_folder)
            print(args.resume_from)
        args.checkpoint_path = args.resume_from
    # ---------------------------------------------------


    return args, unknown_args


class IARLEnv(object):
    """ This one is tricky as it is used both in RL and non-RL pipelines
    It uses a rlenv to simulate lidars and odometry, and some IA tools to simulate agent behaviors 
    """
    def __init__(self, args, silent=False):
        # Load scenario
        self.args = args
        self.iaenv = Environment(args, silent=silent)
        self.iaenv.populate(silent=silent)
        self.n_sim_agents = len(self.iaenv.worldstate.get_agents())
        self.args.n_agents = self.n_sim_agents # tell rlenv to create an agent for every ia sim agent

        # variables
        self.exit = False  # flag to shutdown all threads if a single thread fails
        self.fixed_state = [None] # a container in order to be shared between envs
        self.sim_start_walltime = None
        self.lock = threading.Lock()
        self.is_in_gesture_episode = False
        self.is_in_speech_episode = False

        # Action and observation space

        # ROS or not?
        self.current_sim_time = 0.
        if args.no_ros:
            pass
        else:
            self.latest_cmd_vel = np.array([0,0,0])
            # Publishers
            self.goalpub = rospy.Publisher(kNavGoalTopic, PoseStamped, queue_size=1)
            self.obstpub = rospy.Publisher(kObstaclesTopic, ObstacleArrayMsg, queue_size=1)
            self.goalreachedpub = rospy.Publisher("/goal_reached", Header, queue_size=1)
            self.clockpub = rospy.Publisher("/clock", Clock, queue_size=1)

        # Sim Env
        self.rlenv = PepperRLEnv(
            args=args,
#             map_=self.iaenv.worldstate.map,  # don't downsample map. MAPS WILL DIFFER
            silent=silent,
        )

        # initialize agent velocities
        for i in range(len(self.rlenv.virtual_peppers)):
            if i == 0:
                continue
            vp = self.rlenv.virtual_peppers[i]
            vp.kMaxVel = np.array([0.8, 0.8, 1.]) # x y theta
            vp.kPrefVel = np.array([0.6, 0.6, 0.75]) # x y theta
            vp.kMaxAcc  = np.array([0.5, 0.5, 0.75]) # x y theta
            vp.kMaxJerk = np.array([1., 1., 2.]) # x y theta

        # Initialize agents according to scenario
        self.xystates0 = self._xystates()
        goalstates0 = np.array([inn.goal for inn in self.iaenv.worldstate.get_innerstates()])
        deltas = goalstates0 - self.xystates0
        thstates = np.arctan2(deltas[:,1], deltas[:,0])
        self.agents_pos0 = np.zeros((self.n_sim_agents, 3))
        self.agents_goals0 = np.zeros((self.n_sim_agents, 3))
        self.agents_pos0[:,:2] = self.xystates0 * 1.
        self.agents_pos0[:,2] = thstates * 1.
        self.agents_goals0[:,:2] = goalstates0 * 1.
        self.agents_goals0[:,2] = thstates * 1.
        # set radii in rlenv
        for i in range(self.n_sim_agents):
            self.rlenv.vp_radii[i] = self.iaenv.worldstate.get_agents()[i].radius

        # Reset environment
        self.reset()

        # spaces
        self.action_space = self.rlenv.action_space
        self.observation_space = self.rlenv.observation_space

        if self.args.realtime:
            self.realtime_rate = WalltimeRate(1. / self.args.dt)

        # wait 3 seconds for everyone to subscribe
        if self.args.delay_start:
            time.sleep(3.)

        # start ROS routines
        if not args.no_ros:
            # Subscribers
            from geometry_msgs.msg import Twist
            if self.args.cmd_vel:
                rospy.Subscriber(kCmdVelTopic, Twist, self.set_cmd_vel_callback, queue_size=1)
            from std_msgs.msg import String
            rospy.Subscriber("/gestures", String, self.gestures_callback, queue_size=1)
            rospy.Subscriber("/speech", String, self.speech_callback, queue_size=1)
            if PUBLISH_GOAL_EVERY != 0:
                rospy.Timer(rospy.Duration(PUBLISH_GOAL_EVERY), self.publish_goal_callback)
            else:
                self.publish_goal_callback()

        # dirty hack for move base
        if self.args.pre_step_once:
            # step environment
            robot_action = np.array([0, 0, 0])
            for i in range(100):
                ob, rew, new, _ = self.step(robot_action, ONLY_FOR_AGENT_0=True)
            self.reset()
            self.publish_goal_callback()

    # Timeekeeping
    def reset_time(self):
        self.current_sim_time = 0.
        self.sim_start_walltime = None
    def increment_time(self):
        self.current_sim_time += self.args.dt
        if self.args.no_ros:
            pass
        else:
            cmsg = Clock()
            cmsg.clock = self.get_sim_time()
            self.clockpub.publish(cmsg)
        # store the moment the simulation was started
        if self.sim_start_walltime is None:
            self.sim_start_walltime = self.get_walltime()
    def get_walltime(self):
        if self.args.no_ros:
            return time.time()
        else:
            return rospy.Time.from_sec(time.time())
    def get_walltime_since_sim_start_sec(self):
        if self.sim_start_walltime is None:
            return 0
        if self.args.no_ros:
            return self.get_walltime() - self.sim_start_walltime
        else:
            return (self.get_walltime() - self.sim_start_walltime).to_sec()
    def get_sim_time(self):
        if self.args.no_ros:
            return self.current_sim_time
        else:
            return rospy.Time.from_sec(self.current_sim_time)
    def get_sim_time_sec(self):
        return self.current_sim_time
    def get_max_task_alotted_time(self):
        if self.args.no_ros:
            return MAX_TASK_ALOTTED_TIME
        else:
            return rospy.Duration(MAX_TASK_ALOTTED_TIME)
    def sleep_for_simtime(self, duration):
        EXPECTED_SIM_SPEED_FACTOR = 10.  # if the sim is much faster, needs to be increased
        start = self.get_sim_time_sec()
        while self.get_sim_time_sec() < (start + duration):
            time.sleep(duration / EXPECTED_SIM_SPEED_FACTOR)

    # ROS publishing methods
    def publish_goal(self, goal):
        if self.args.no_ros or self.args.no_goal:
            return
        else:
            from tf.transformations import quaternion_from_euler
            from geometry_msgs.msg import Quaternion
            from geometry_msgs.msg import PoseStamped
            print("Publishing goal")
            posemsg = PoseStamped()
            posemsg.header.frame_id = kFixedFrame
            posemsg.pose.position.x = goal[0]
            posemsg.pose.position.y = goal[1]
            quaternion = quaternion_from_euler(0, 0, goal[2])
            posemsg.pose.orientation = Quaternion(*quaternion)
            self.goalpub.publish(posemsg)
    def publish_obstacle_msg(self):
        if self.args.no_ros:
            return
        else:
            from tf.transformations import quaternion_from_euler
            from geometry_msgs.msg import Point32, Quaternion
            from costmap_converter.msg import ObstacleArrayMsg, ObstacleMsg
            obstacles_msg = ObstacleArrayMsg()
            obstacles_msg.header.stamp =  self.get_sim_time()
            obstacles_msg.header.frame_id = kFixedFrame
            for i in range(1, self.n_sim_agents):
                # Add point obstacle
                obst = ObstacleMsg()
                obst.id = i
                obst.polygon.points = [Point32()]
                obst.polygon.points[0].x = self.rlenv.virtual_peppers[i].pos[0]
                obst.polygon.points[0].y = self.rlenv.virtual_peppers[i].pos[1]
                obst.polygon.points[0].z = 0

                obst.radius = self.rlenv.vp_radii[i]

                yaw = self.rlenv.virtual_peppers[i].pos[2]
                q = quaternion_from_euler(0,0,yaw)
                obst.orientation = Quaternion(*q)

                obst.velocities.twist.linear.x = self.rlenv.virtual_peppers[i].vel[0]
                obst.velocities.twist.linear.y = self.rlenv.virtual_peppers[i].vel[1]
                obst.velocities.twist.linear.z = 0
                obst.velocities.twist.angular.x = 0
                obst.velocities.twist.angular.y = 0
                obst.velocities.twist.angular.z = self.rlenv.virtual_peppers[i].vel[2]
                obstacles_msg.obstacles.append(obst)
            self.obstpub.publish(obstacles_msg)
            return

    # callbacks
    def set_cmd_vel_callback(self, msg):
        if self.args.no_ros:
            raise NotImplementedError
        elif self.args.cmd_vel:
            self.latest_cmd_vel = np.array([msg.linear.x, msg.linear.y, msg.angular.z])
            return
        else:
            return
    def publish_goal_callback(self, event=None):
        self.publish_goal(self.rlenv.agent_goals[0])
    def gestures_callback(self, msg):
        gesture = msg.data
        if gesture == "animations/Stand/Gestures/Desperate_4":
            print("starting gesture episode")
            self.is_in_gesture_episode = True
            self.sleep_for_simtime(5.)
            self.is_in_gesture_episode = False
            print("ending gesture episode")
    def speech_callback(self, msg):
        speech = msg.data
        if "xcuse me" in speech:
            print("starting speech episode")
            self.is_in_speech_episode = True
            self.sleep_for_simtime(5.)
            self.is_in_speech_episode = False
            print("ending speech episode")
    def task_update_routine(self):
        import multiprocessing
        data_queue = multiprocessing.Queue()
        is_done_queue = multiprocessing.Queue()
        rate = WalltimeRate(10)
        # get fixed_state once and for all
        if True:  # replace this with for loop if separate fixed_state for each agent
            agent_index = 0
            _, self.fixed_state[0] = state_estimation.state_estimate_from_sim_worldstate(
                agent_index, self.iaenv.worldstate)
            # otherwise subprocesses keep recomputing
            _ = self.fixed_state[0].derived_state.map_sdf(self.fixed_state[0])
        while True:
            tic = timer()
            t = multiprocessing.Process(target=self.update_agent_tasks, args=(data_queue,is_done_queue))
            t.start()
            toc = timer()
            if False:
                print("Process creation: {}ms".format(1000 * (toc-tic)))
            # check whether process is finished
            is_done_queue.get()
            # extract output data from process and apply changes
            while not data_queue.empty():
                update = data_queue.get()
                is_update = update[1]
                if is_update:
                    (agent_index, _, time_now, state_e, task) = update
                    self.state_estimate_times[agent_index] = time_now
                    self.state_estimates[agent_index] = unserialize_cstate(state_e)
                    self.current_tasks[agent_index] = task
                    self.task_start_times[agent_index] = self.get_sim_time()
            #
            t.join(timeout=1.)
            rate.sleep()
            if self.exit:
                break


    # Other Methods
    def get_agent_radius(self, agent_index):
        return self.iaenv.worldstate.get_agents()[agent_index].radius
    def _xystates(self):
        return np.array(self.iaenv.worldstate.get_xystates())
    def _permissivities(self):
        return [s.permissivity for s in self.iaenv.worldstate.get_innerstates()]
    def reset_loop(self):
        for i in range(1, self.n_sim_agents):
            self.task_start_times[i] = None
            self.current_tasks[i] = None
        self.collision_episodes[:,:] = 0
        # reset data in the two simulators
        for i in range(1, self.n_sim_agents):
            self.iaenv.worldstate.get_xystates()[i] = self.xystates0[i] * 1.
        new_agents_pos = list(self.agents_pos0 * 1.)
        new_agents_pos[0] = None
        self.rlenv._set_agents_pos(new_agents_pos)
    def rewind_robot(self):
        """ Basically, set robot goal to start position, set starting position to previous goal"""
        self.state_estimate_times[0] = None
        self.state_estimates[0] = None
        self.task_start_times[0] = None
        self.current_tasks[0] = None
        self.collision_episodes[0,:] = 0
        self.collision_episodes[:,0] = 0
        # reset data in the two simulators
        self.iaenv.worldstate.get_innerstates()[0].goal = self.next_robot_goal[:2] * 1.
        new_agents_goals = [None for _ in range(self.n_sim_agents)]
        new_agents_goals[0] = self.next_robot_goal * 1.
        self.rlenv._set_agents_goals(new_agents_goals)
        self.publish_goal(new_agents_goals[0])
        # swap next with current
        temp = self.next_robot_goal * 1.
        self.next_robot_goal = self.current_agent_goals[0] * 1.
        self.current_agent_goals[0] = temp * 1.

    def update_agent_task(self, agent_index):
        current_task = self.current_tasks[agent_index]
        last_se_time = self.state_estimate_times[agent_index]
        time_now = self.get_sim_time_sec()
        last_state = self.state_estimates[agent_index]
        worldstate = self.iaenv.worldstate
        fixed_state = self.fixed_state[0]
        are_goals_reached = self.rlenv.are_goals_reached()
        args = self.args
        # ---------------------------
        if current_task is not None:
            return (agent_index, False)
        # Get state estimate
        time_elapsed = None
        if last_se_time is not None:
            time_elapsed = time_now - last_se_time
        if args.naive_plan and fixed_state is not None and agent_index != 0 and STATELESS_NAIVE_PLAN:
            # create fake state estimate (ignore agents and knowledge) for speedup
            pos_xy = worldstate.get_xystates()[agent_index]  # + Noise?
            pos_ij = worldstate.map.xy_to_floatij([pos_xy])[0]
            from cia import CState
            state_e = CState(
                radius=worldstate.get_agents()[agent_index].radius,
                pos=np.array(pos_xy, dtype=np.float32),
                pos_ij=np.array(pos_ij, dtype=np.float32),
                mapwidth=worldstate.map.occupancy().shape[0],
                mapheight=worldstate.map.occupancy().shape[1],
                goal=np.array(worldstate.get_innerstates()[agent_index].goal, dtype=np.float32),
            )
        else:
            state_e, fixed_state = state_estimation.state_estimate_from_sim_worldstate(
                agent_index, worldstate,
                state_memory=last_state,
                time_elapsed=time_elapsed,
                reuse_state_cache=fixed_state)
        # if agent is at goal, switch to loiter mode
        goalstate = np.array(worldstate.get_innerstates()[agent_index].goal)
        if are_goals_reached[agent_index]:

            if self.args.verbose:
                print("{:.1f} | {}: Agent {} | goal reached.".format(
                    self.get_walltime_since_sim_start_sec(), self.get_sim_time_sec(), agent_index))
            task = {
                'tasktargetpathidx': 0,
                'taskfullpath': np.array([goalstate]),
                'taskaction': actions.Loiter(),  # action should go here but we want things serializable
            }
        else:
            # get task using ia planning
            possible_actions = SIM_AGENTS_POSSIBLE_ACTIONS
            if agent_index == 0:
                possible_actions = ROBOT_POSSIBLE_ACTIONS
            tic = timer()
            if args.naive_plan and agent_index != 0:
                from pyIA import paths
                agent_ij = state_e.get_pos_ij()
                goal_ij = fixed_state.map.xy_to_ij([state_e.get_goal()])[0]
                dijkstra = None
                if self.dijkstra_caches[agent_index] is not None:
                    if np.allclose(goal_ij, self.dijkstra_caches[agent_index]["goal_ij"]):
                        dijkstra = self.dijkstra_caches[agent_index]["field"]
                if dijkstra is None:
                    fixed_state.derived_state.suppress_warnings()
                    dijkstra = fixed_state.derived_state.dijkstra_from_goal(state_e, fixed_state, paths.NaivePath())
                    self.dijkstra_caches[agent_index] = {"goal_ij": goal_ij, "field": dijkstra}
                from CMap2D import path_from_dijkstra_field
                path_ij, _ = path_from_dijkstra_field(dijkstra, agent_ij, connectedness=8)
                path_xy = fixed_state.map.ij_to_xy(path_ij)
                paths_xy = [np.array(path_xy, dtype=np.float32)]
#                 possible_path_variants = [paths.NaivePath()]
#                 paths_xy, path_variants = ia_planning.path_options(
#                     state_e, fixed_state, possible_path_variants=possible_path_variants)
                taskpath = paths_xy[0]
                task = {
                    'taskfullpath': taskpath,
                    'tasktargetpathidx': len(taskpath)-1,
                    'taskaction': actions.Intend(),
                }
                optimistic_sequence = [[task, None]]
            else:
                byproducts = {}
                pruned_stochastic_tree, optimistic_sequence, firstfailure_sequence = \
                    ia_planning.plan(
                        state_e, fixed_state, possible_actions,
                        n_trials=args.n_trials, max_depth=args.max_depth,
                        BYPRODUCTS=byproducts,
                    )
            planning_time = timer()-tic
            # DEBUG
            if args.debug:
                if agent_index == 0:
                    ia_planning.visualize_planning_process(
                        state_e, fixed_state, possible_actions,
                        pruned_stochastic_tree, optimistic_sequence, firstfailure_sequence, byproducts,
                        env=self.iaenv,
                    )
                    plt.show()
            # Assign task
            task = optimistic_sequence[0][0]
            task['tasktargetpathidx'] = min(10, task['tasktargetpathidx']) # limit task length in space
            if self.args.verbose:
                print("{:.1f} | {}: Agent {} | planning: {:.1f}s | new task: {}".format(
                    self.get_walltime_since_sim_start_sec(), self.get_sim_time_sec(), agent_index,
                    planning_time, task['taskaction'].typestr()))
        # direct update (does nothing if this is running in subprocess)
        self.state_estimate_times[agent_index] = time_now
        self.state_estimates[agent_index] = state_e
        self.fixed_state[0] = fixed_state
        self.current_tasks[agent_index] = task
        self.task_start_times[agent_index] = self.get_sim_time()
        # return values are unused unless multiprocessing is enabled
        return (agent_index, True, time_now, state_e.serialize(), task)
    def update_agent_tasks(self, queue=None, is_done_queue=None):
        for i in range(1, self.n_sim_agents):
            result = self.update_agent_task(i)
            if queue is not None:
                queue.put(result)
        if is_done_queue is not None:
            is_done_queue.put("done")
            queue.close()
            is_done_queue.close()

    def reset_task(self, agent_index):
        self.current_tasks[agent_index] = None
        self.task_start_times[agent_index] = None
    def execute_current_task(self, agent_index):
        # TODO run a task-specific controller instead
        task = self.current_tasks[agent_index]
        task_start = self.task_start_times[agent_index]
        if task is None or task_start is None:
            return np.array([0,0,0])
        # get action from current task
        N = int(NAIVE_AGENT_LIVE_WAYPOINT_DISTANCE / self.iaenv.worldstate.map.resolution())
        waypoint_idx = min(task['tasktargetpathidx'], N)
        waypoint = task['taskfullpath'][waypoint_idx]
        vp_pos = self.rlenv.virtual_peppers[agent_index].pos
        waypoint_in_vp_frame = pose2d.apply_tf(waypoint, pose2d.inverse_pose2d(vp_pos))
        theta = np.arctan2(waypoint_in_vp_frame[1], waypoint_in_vp_frame[0])
        delta = np.array([waypoint_in_vp_frame[0], waypoint_in_vp_frame[1], theta])
        action = delta
        # apply actions to iaenv, such as SAY
#         TODO
        # end task
        task_elapsed = self.get_sim_time() - task_start
        if task_elapsed > self.get_max_task_alotted_time():
            self.reset_task(agent_index)
            if self.args.verbose:
                print("{:.1f} | {}: Agent {} | task timed-out".format(
                    self.get_walltime_since_sim_start_sec(), self.get_sim_time_sec(), agent_index))
            # reset blocked collision episodes for agent (new chance to be allowed through)
            for j in range(self.n_sim_agents):
                if agent_index == j:
                    continue
                if self.collision_episodes[agent_index, j] == 1:
                    self.collision_episodes[agent_index, j] = 0
                    self.collision_episodes[j, agent_index] = 0
        # we should stop, but I kind of like the idle movement. just slow down.
        if isinstance(task['taskaction'], actions.Loiter):
            action = action * 0.8
        else:
            # check if task succeeded
            if np.linalg.norm(vp_pos[:2] - waypoint) < self.get_agent_radius(agent_index):
                self.reset_task(agent_index)
                action = np.array([0,0,0])
                if self.args.verbose:
                    print("Agent {} | task succeeded".format(agent_index))
        return action
    def execute_current_tasks(self, robot_action):
        actions = np.zeros((self.n_sim_agents, 3)) # cmd_vel in SI units
        for i in range(1, self.n_sim_agents):
            actions[i, :] = self.execute_current_task(i)
        # convert actions
        if not self.args.cmd_vel:
            actions = self.rlenv.cmd_vel_to_normalized(actions)
        if not self.args.continuous:
            actions = self.rlenv.continuous_to_categorical(actions)
        actions[0, ...] = robot_action
        return actions

    def collision_update(self):
        """ checks if collision episodes are started or ended. A collision episode starts when
        agents get closer than a fixed threshold, and end when they are no longer inside this
        threshold.

        Pass-through probability is sampled once at the start of each episode,
        or when either agent's task is updated. Once sampled, it remains constant for the whole
        episode.
        """
        for i in range(self.n_sim_agents):
            for j in range(self.n_sim_agents):
                if i >= j: # avoid checking twice
                    continue
                xi = self._xystates()[i]
                ri = self.get_agent_radius(i)
                pi = self._permissivities()[i]
                xj = self._xystates()[j]
                rj = self.get_agent_radius(j)
                pj = self._permissivities()[j]
                if i == 0:  # robot
                    if self.is_in_gesture_episode:
                        # model pass-through behavior
                        self.collision_episodes[i,j] = 2
                        self.collision_episodes[j,i] = 2
                        continue
                    if self.is_in_speech_episode:
                        # model average positive reaction to speech
                        pi = np.sqrt(pi)
                        pj = np.sqrt(pj)
                if np.linalg.norm(xj - xi) < (2*rj + 2*ri):
                    if self.collision_episodes[i,j] == 0:
                        # enter collision episode
                        pmin = np.clip(min(pi, pj), 0, 1)
                        # sample collision
                        sample = np.random.random()
                        if sample < pmin:
                            # pass through
                            self.collision_episodes[i,j] = 2
                            self.collision_episodes[j,i] = 2
                        else:
                            # collision
                            self.collision_episodes[i,j] = 1
                            self.collision_episodes[j,i] = 1
#                         print(self._permissivities())
#                         print("episode | {} {} | pmin {} | sample {}".format(i, j, pmin, sample))
                else:
                    # exit collision episode
                    self.collision_episodes[i,j] = 0
                    self.collision_episodes[j,i] = 0
        if self.args.no_pass_through:
            self.collision_episodes[0,:] = 0
            self.collision_episodes[:,0] = 0
        disabled_collisions = self.collision_episodes == 2
        return disabled_collisions

    # RL Interface
    def reset(self, agents_mask=None, ONLY_FOR_AGENT_0=False):
        _ = self.rlenv.reset(agents_mask=agents_mask)


        # reset to scene definition
        self.current_agent_goals = self.agents_goals0 * 1.
        self.next_robot_goal = self.agents_pos0[0] * 1. # set future goal to start position
        self.rlenv._set_agents_pos(self.agents_pos0 * 1.)
        self.rlenv._set_agents_goals(self.current_agent_goals * 1.)
        self.rlenv.DETERMINISTIC_GOAL = lambda i : self.agents_goals0[i] * 1.
        # Simulator state
        self.reset_time()
        self.state_estimate_times = [None for _ in range(self.n_sim_agents)]
        self.state_estimates = [None for _ in range(self.n_sim_agents)]
        self.dijkstra_caches = [None for _ in range(self.n_sim_agents)]
        self.task_start_times = [None for _ in range(self.n_sim_agents)]
        self.current_tasks = [None for _ in range(self.n_sim_agents)]
        self.collision_episodes = np.zeros((self.n_sim_agents, self.n_sim_agents)) # 0 none 1 collision 2 passthrough

        ob, rew, new, _ =  self.rlenv.step(None, ONLY_FOR_AGENT_0=ONLY_FOR_AGENT_0)
        # ob is [lidar_obs, robot_obs, (relobst_obs)]

        if ONLY_FOR_AGENT_0:
            pass
        else:
            ob = [obs[:1] for obs in ob]
        return ob
    def n_agents(self):
        """ returns the number of steerable agents in the environment """
        return 1
    def step(self, robot_action,
             ONLY_FOR_AGENT_0=False,
             ):
        """ robot_action is in the units expected by PepperRLEnv (see observation_space) """
        must_break = False
        if not self.args.plan_async:
            self.update_agent_tasks()
        # drive agents
        if robot_action is None:
            actions = None # time does not advance
        else:
            actions = self.execute_current_tasks(robot_action)
        # collisions
        disabled_collisions = self.collision_update()
        # step
        ob, rew, new, info = self.rlenv.step(actions,
             BYPASS_SI_UNITS_CONVERSION=self.args.cmd_vel,
             DISABLE_A2A_COLLISION=disabled_collisions,
             ONLY_FOR_AGENT_0=ONLY_FOR_AGENT_0)
        are_goals_reached = self.rlenv.are_goals_reached()
        self.publish_obstacle_msg()
        if robot_action is not None:
            self.increment_time()
        # update iaenv worldstate
        for i in range(self.n_sim_agents):
            self.iaenv.worldstate.get_xystates()[i] = self.rlenv.virtual_peppers[i].pos[:2]
        # all agents (robot excluded) have reached their goal
        if np.all(are_goals_reached[1:]):
            self.reset_loop()
        if are_goals_reached[0] == 1:
            if self.args.verbose:
                print("Agent 0 : goal reached.")
            if not self.args.no_ros:
                msg = Header()
                msg.stamp = self.get_sim_time()
                msg.frame_id = "goal 0 is reached"
                self.goalreachedpub.publish(msg,)
                time.sleep(0.1)
            if self.args.shutdown_on_success:
                self.shutdown()
            else:
                self.rewind_robot()
        if self.current_sim_time >= self.args.max_runtime:
            self.shutdown()
        if ONLY_FOR_AGENT_0:
            pass
        else:
            ob = [obs[:1] for obs in ob]
            rew = rew[:1]
            new = new[:1]
        return ob, rew, new, info

    # Other Interface
    def run(self):
        """ typically when run in ROS, this starts the simulator update routine """
        try:
            if self.args.plan_async:
                import threading
                threading.Thread(target=self.task_update_routine).start()
            while True:
                if self.exit:
                    break
                # Get robot action
                if self.args.autopilot:
                    # drive robot internally
                    self.update_agent_task(0)
                    robot_action = self.execute_current_task(0)
                elif self.args.cmd_vel:
                    robot_action = self.latest_cmd_vel * 1.
                else:
                    raise NotImplementedError("Robot control mode not specified (e.g. --cmd-vel or --autopilot)")
                # step environment
                ob, rew, new, _ = self.step(robot_action, ONLY_FOR_AGENT_0=True)
                if new:
                    break
                if self.args.realtime:
                    wall_t = self.get_walltime_since_sim_start_sec()
                    sim_t = self.get_sim_time_sec()
                    if sim_t < (wall_t - 1):
                        rospy.logwarn_throttle(5., "IARLEnv missed realtime rate.")
                    elif sim_t > (wall_t + 0.1):
                        time.sleep(sim_t - (wall_t + 0.09))
        except KeyboardInterrupt:
            if self.args.plan_async:
                self.exit = True
            if not self.args.no_ros:
                self.rlenv._ros_shutdown("SIGINT")

    def shutdown(self):
        if not self.args.no_ros:
            rospy.signal_shutdown("OK")
        self.exit = True

    def render(self, *args, **kwargs):
        self.rlenv.render(*args, **kwargs)

    def seed(self, seed):
        pass
        # TODO

    def close(self):
        self.shutdown()
        self.rlenv.close()

    def _get_viewer(self):
        return self.rlenv.viewer


class MultiIARLEnv(object):
    def __init__(self, args, verbose=False):
        if args.force_publish_ros:
            raise NotImplementedError
        # initialize several environments
        self.num_envs = 8
        self.IARLEnvs = []
        self.scenarios = [
            "aslquiet",
            "aslgates",
            "aslgroups",
            "aslguards",
        ]
        self.global_fixed_state = [None]
        for i in range(self.num_envs):
            args.scenario = self.scenarios[i % len(self.scenarios)]
            self.IARLEnvs.append(IARLEnv(args, verbose=verbose))
            self.IARLEnvs[-1].fixed_state = self.global_fixed_state
        # action space
        self.action_space = self.IARLEnvs[0].action_space
        self.observation_space = self.IARLEnvs[0].observation_space
        self.metadata = {}
    # RL Interface
    def reset(self, agents_mask=None):
        if agents_mask is None:
            agents_mask = [True for _ in self.IARLEnvs]
        for env, mask in zip(self.IARLEnvs, agents_mask):
            if mask:
                env.reset()
        obs, _, _, _ = self.step(None)
        return obs
    def n_agents(self):
        """ returns the number of steerable agents in the environment """
        return self.num_envs
    def step(self, robot_action):
        """ robot_action is in the units expected by PepperRLEnv (see observation_space) """
        rews = []
        news = []
        for i, env in enumerate(self.IARLEnvs):
            if robot_action is None:
                action = None
            else:
                action = robot_action[i:i+1]
            ob, rew, new, _ = env.step(action)
            # ob is [lidar_obs, state_obs, (relobst_obs)] (2-3, n_sim_agents, ...)
            first_agent_ob = [obs[:1] for obs in ob] # (2-3, 1, ...)
            if i == 0:
                merged_obs = [[] for obs in first_agent_ob]
            for k, obs in enumerate(first_agent_ob):
                merged_obs[k].extend(obs) # (2-3, n_env, ...)
            rews.extend(rew)
            news.extend(new)
        # as ndarrays
        merged_obs[0] = np.array(merged_obs[0]) # lidar obs
        merged_obs[1] = np.array(merged_obs[1]) # state obs
        # can not convert obstacles obs to ndarray, as n obst. varies for each env
        rews = np.array(rews)
        news = np.array(news)
        return merged_obs, rews, news, {}
    def close(self):
        pass
    def render(self):
        pass

