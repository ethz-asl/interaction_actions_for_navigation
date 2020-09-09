from __future__ import print_function
import numpy as np
from pyniel.numpy_tools import indexing
import warnings

from cia import CState, FixedState
from cia import kStateFeatures

# State estimation
PERMISSIVITY_RADIUS = 1.
PERCEPTIVITY_RADIUS = 1.
CROWDEDNESS_RADIUS = 1.

MEMORY_FORGET_RATE = 1.2  # stddev multiplicator, /sec
BIG_SIGMASQ = 1000.  # stddev larger than this value is treated like infinite for numerical purp.
TINY_SIGMASQ = 0.00001  # stddev smaller than this is ... 0 for numerical purp.
CHECK_NUMERICS = True

# initial P(S)
CROWDEDNESS_UNINFORMED_PRIOR = 0.1
PERMISSIVITY_UNINFORMED_PRIOR = 0.99
PERCEPTIVITY_UNINFORMED_PRIOR = 0.99
CWD_UNCERTAINTY_UNINFORMED_PRIOR = BIG_SIGMASQ
PRM_UNCERTAINTY_UNINFORMED_PRIOR = BIG_SIGMASQ
PER_UNCERTAINTY_UNINFORMED_PRIOR = BIG_SIGMASQ
# P(Z|S) parameters mu, sigma^2 for empty observations
NOHUMAN_CROWDEDNESS_SENSOR_PR = 0.
NOHUMAN_PERMISSIVITY_SENSOR_PR = 0.99
NOHUMAN_PERCEPTIVITY_SENSOR_PR = 0.99
# TODO: issue with occluded humans (for now boosted uncertainties on absence of detection)
NOHUMAN_CWD_UNCERTAINTY_SENSOR_PR = 2.
NOHUMAN_PRM_UNCERTAINTY_SENSOR_PR = 2.
NOHUMAN_PER_UNCERTAINTY_SENSOR_PR = 2.
# P(Z|S) parameters mu, sigma^2 for human detections
HUMAN_CROWDEDNESS_SENSOR_PR = 1.
HUMAN_PERMISSIVITY_SENSOR_PR = 0.5
HUMAN_PERCEPTIVITY_SENSOR_PR = 0.8
# few false positives with YOLO (some smearing due to RGB-D)
HUMAN_CWD_UNCERTAINTY_SENSOR_PR = 0.05
HUMAN_PRM_UNCERTAINTY_SENSOR_PR = 2.  # observing a person tells me little about their PRM
HUMAN_PER_UNCERTAINTY_SENSOR_PR = 0.1

# sensor properties
HALF_FOV = np.pi / 4.  # 45 degrees ( total 90 deg realsense fov )

# CORE INTERFACE -----------
def true_state_from_sim_worldstate(agent_index, worldstate):
    raise NotImplementedError

def state_estimate_from_sim_worldstate(agent_index, worldstate,
                                       state_memory=None, time_elapsed=None,
                                       reuse_state_cache=None):
    pos_xy = worldstate.get_xystates()[agent_index]  # + Noise?
    pos_ij = worldstate.map.xy_to_floatij([pos_xy])[0]
    # fixed state
    if reuse_state_cache is not None:
        fixed_state = reuse_state_cache
    else:
        fixed_state = FixedState(worldstate.map)
    # prior
    state = CState(
        radius=worldstate.get_agents()[agent_index].radius,
        pos=np.array(pos_xy, dtype=np.float32),
        pos_ij=np.array(pos_ij, dtype=np.float32),
        mapwidth=worldstate.map.occupancy().shape[0],
        mapheight=worldstate.map.occupancy().shape[1],
        goal=np.array(worldstate.get_innerstates()[agent_index].goal, dtype=np.float32),
    )
    if state_memory is not None:
        merge_state_memory_into_state(state, state_memory, time_elapsed)
    else:
        uninformed_prior_into_state(state, fixed_state)
    # apply sensor update to prior
    grid_feature_estimate_from_sim_worldstate(
        agent_index, worldstate, state)
    return state, fixed_state

def state_estimate_from_tracked_persons(tracked_persons, pose2d_tracksframe_in_refmap,
                                        robot_pos, robot_radius, goal_xy, map_,
                                        state_memory=None, time_elapsed=None,
                                        reuse_state_cache=None):
    """
    pos_xy is robot position in map
    """
    pos_xy = robot_pos[:2]
    robot_heading = robot_pos[2]
    pos_ij = map_.xy_to_floatij([pos_xy])[0]
    # fixed state
    if reuse_state_cache is not None:
        fixed_state = reuse_state_cache
    else:
        fixed_state = FixedState(map_)
    # prior
    state = CState(
        radius=robot_radius,
        pos=np.array(pos_xy, dtype=np.float32),
        pos_ij=np.array(pos_ij, dtype=np.float32),
        mapwidth=map_.occupancy().shape[0],
        mapheight=map_.occupancy().shape[1],
        goal=np.array(goal_xy, dtype=np.float32),
    )
    if state_memory is not None:
        merge_state_memory_into_state(state, state_memory, time_elapsed)
    else:
        uninformed_prior_into_state(state, fixed_state)
    # apply sensor update to prior
    grid_feature_estimate_from_sensor_data(
        tracked_persons, pose2d_tracksframe_in_refmap, robot_heading, state, fixed_state)
    return state, fixed_state

# TODO Add lidar for free space confirmation

# INTERNAL -----------------
def tracked_persons_from_sim_worldstate(robot_index, worldstate):
    from frame_msgs.msg import TrackedPersons, TrackedPerson
    tracked_persons = TrackedPersons()
    n_agents = len(worldstate.get_agents())
    robot_ij = worldstate.map.xy_to_ij([worldstate.get_xystates()[robot_index]])[0]
    robot_heading = agent_heading_from_sim_worldstate(robot_index, worldstate)
    # visibility_map  # TODO: use lidar free space instead!
    fov = [robot_heading - HALF_FOV, robot_heading + HALF_FOV]
    visibility = worldstate.map.visibility_map_ij(robot_ij, fov=fov)
    for k in range(n_agents):
        if k == robot_index:
            continue
        axy = worldstate.get_xystates()[k]
        auv = worldstate.get_uvstates()[k]
        ai, aj = worldstate.map.xy_to_ij([axy])[0]
        agent_heading = agent_heading_from_sim_worldstate(k, worldstate)
        if visibility[ai, aj] < 0:
            continue
        tp = TrackedPerson()
        from tf.transformations import quaternion_from_euler
        quaternion = quaternion_from_euler(0, 0, agent_heading)
        tp.pose.pose.orientation.x = quaternion[0]
        tp.pose.pose.orientation.y = quaternion[1]
        tp.pose.pose.orientation.z = quaternion[2]
        tp.pose.pose.orientation.w = quaternion[3]
        tp.pose.pose.position.x = axy[0]
        tp.pose.pose.position.y = axy[1]
        tp.twist.twist.linear.x = auv[0]
        tp.twist.twist.linear.y = auv[1]
        tracked_persons.tracks.append(tp)
    pose2d_tracksframe_in_refmap = np.array([0, 0, 0])
    return tracked_persons, pose2d_tracksframe_in_refmap

def agent_heading_from_sim_worldstate(agent_index, worldstate):
    uv = worldstate.get_uvstates()[agent_index]
    if np.linalg.norm(uv) == 0:
        pos_xy = worldstate.get_xystates()[agent_index]
        goal_xy = worldstate.get_innerstates()[agent_index].goal
        heading = np.arctan2(goal_xy[1] - pos_xy[1], goal_xy[0] - pos_xy[0])
    else:
        heading = np.arctan2(uv[1], uv[0])
    return heading

def grid_feature_estimate_from_sim_worldstate(agent_index, worldstate, state):
    tracked_persons, pose2d_tracksframe_in_refmap = \
        tracked_persons_from_sim_worldstate(agent_index, worldstate)
    agent_heading = agent_heading_from_sim_worldstate(agent_index, worldstate)
    fixed_state = FixedState(worldstate.map)
    grid_feature_estimate_from_sensor_data(tracked_persons, pose2d_tracksframe_in_refmap,
                                           agent_heading, state, fixed_state)

def grid_feature_estimate_from_sensor_data(tracked_persons, pose2d_tracksframe_in_refmap,
                                           robot_heading, state, fixed_state):
    # same for all features
    n_agents = len(tracked_persons.tracks)
    # visibility_map  # TODO: use lidar free space instead!
    map_ = fixed_state.map
    ij = indexing.as_idx_array(map_.occupancy(), axis='all')
    ii = ij[..., 0]
    jj = ij[..., 1]
    fov = [robot_heading - HALF_FOV, robot_heading + HALF_FOV]
    visibility = fixed_state.map.visibility_map_ij(state.get_pos_ij(), fov=fov)
    for feature_name in kStateFeatures:
        feature_index = kStateFeatures[feature_name]
        # start by assuming no humans detected, then add detections one by one.
        # TODO: increase nohuman uncertainty with distance?
        if feature_name == "crowdedness":
            feature_map = NOHUMAN_CROWDEDNESS_SENSOR_PR * np.zeros_like(ii, dtype=np.float32)
            feature_uncertainty_map = NOHUMAN_CWD_UNCERTAINTY_SENSOR_PR * np.ones_like(ii, dtype=np.float32)
        elif feature_name == "permissivity":
            feature_map = NOHUMAN_PERMISSIVITY_SENSOR_PR * np.ones_like(ii, dtype=np.float32)
            feature_uncertainty_map = NOHUMAN_PRM_UNCERTAINTY_SENSOR_PR * np.ones_like(ii, dtype=np.float32)
        elif feature_name == "perceptivity":
            feature_map = NOHUMAN_PERCEPTIVITY_SENSOR_PR * np.ones_like(ii, dtype=np.float32)
            feature_uncertainty_map = NOHUMAN_PER_UNCERTAINTY_SENSOR_PR * np.ones_like(ii, dtype=np.float32)
        for k in range(n_agents):
            tp = tracked_persons.tracks[k]
            from tf.transformations import euler_from_quaternion
            import pose2d
            quaternion = (
                tp.pose.pose.orientation.x,
                tp.pose.pose.orientation.y,
                tp.pose.pose.orientation.z,
                tp.pose.pose.orientation.w,
            )
            tp_pos_in_tracksframe = np.array([
                tp.pose.pose.position.x,
                tp.pose.pose.position.y,
                euler_from_quaternion(quaternion)[2],
            ])
            tp_pos_in_refmap = pose2d.apply_tf_to_pose(
                tp_pos_in_tracksframe, pose2d_tracksframe_in_refmap)
            ai, aj = map_.xy_to_ij([tp_pos_in_refmap[:2]])[0]
            # compute P(Z|S)
            if feature_name == "crowdedness":
                density_radius = CROWDEDNESS_RADIUS
            elif feature_name == "permissivity":
                density_radius = PERMISSIVITY_RADIUS
            elif feature_name == "perceptivity":
                density_radius = PERCEPTIVITY_RADIUS
            else:
                raise NotImplementedError("state estimate for {} not implemented.".format(
                    feature_name))
            sqr_density_radius_ij = (density_radius / map_.resolution())**2
            sqr_distance = ((ii - ai)**2 + (jj - aj)**2)
            mask = sqr_distance < sqr_density_radius_ij
            if feature_name == "crowdedness":
#                 influence_radius = 1.  # bigger means "larger" people
#                 feature_map[mask] += np.clip(1. / (0.0001 + sqr_distance[mask] * influence_radius),
#                                              0., 1.)
                feature_map[mask] += HUMAN_CROWDEDNESS_SENSOR_PR  # additive
                feature_uncertainty_map[mask] = HUMAN_CWD_UNCERTAINTY_SENSOR_PR
            elif feature_name == "permissivity":
                feature_map[mask] = HUMAN_PERMISSIVITY_SENSOR_PR
                feature_uncertainty_map[mask] = HUMAN_PRM_UNCERTAINTY_SENSOR_PR
            elif feature_name == "perceptivity":
                feature_map[mask] = HUMAN_PERCEPTIVITY_SENSOR_PR
                feature_uncertainty_map[mask] = HUMAN_PER_UNCERTAINTY_SENSOR_PR
        # what you can't see...
        feature_uncertainty_map[visibility < 0] = np.inf  # no sensor update outside of sensor fov
        # apply observation normal pdf using bayesian posterior  P(S|Z) ~= P(S) * P(Z|S)
        # product of two gaussian pdfs
        mu1 = np.array(state.grid_features_values()[feature_index, :, :])
        mu2 = feature_map
        sigmasq1 = np.array(state.grid_features_uncertainties()[feature_index, :, :])
        sigmasq2 = feature_uncertainty_map
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', r'invalid value encountered')
            warnings.filterwarnings('ignore', r'divide by zero')
            new_mu = (mu1 * sigmasq2 + mu2 * sigmasq1) / (sigmasq2 + sigmasq1)
            new_sigmasq = 1. / (1. / sigmasq1 + 1. / sigmasq2)
        # check numerics
        if CHECK_NUMERICS:
            is_both_small = np.logical_and(sigmasq1 < TINY_SIGMASQ, sigmasq2 < TINY_SIGMASQ)
            is_1_big = sigmasq1 > BIG_SIGMASQ
            is_2_big = sigmasq2 > BIG_SIGMASQ
            is_both_big = np.logical_and(sigmasq1 > BIG_SIGMASQ, sigmasq2 > BIG_SIGMASQ)
            new_mu[is_both_small] = ((mu1 + mu2) / 2.)[is_both_small]
            new_mu[is_1_big] = mu2[is_1_big]
            new_mu[is_2_big] = mu1[is_2_big]
            new_mu[is_both_big] = ((mu1 + mu2) / 2.)[is_both_big]
            new_sigmasq[is_both_small] = TINY_SIGMASQ
            new_sigmasq[is_1_big] = sigmasq2[is_1_big]
            new_sigmasq[is_2_big] = sigmasq1[is_2_big]
            new_mu[is_both_big] = BIG_SIGMASQ
            new_sigmasq[np.isnan(new_sigmasq)] = np.inf  # this shouldn't happen... so it will
        # assign
        state.grid_features_values()[feature_index, :, :] = new_mu
        state.grid_features_uncertainties()[feature_index, :, :] = new_sigmasq

def merge_state_memory_into_state(state, state_memory, time_elapsed):
    """ should be split into two functions:
    1. copies memory prior into state
    2. adds forgetfulness update to memory prior
    """
    for feature_name in kStateFeatures:
        feature = kStateFeatures[feature_name]
        # copy prior into state
        state.grid_features_values()[feature] = \
            1. * state_memory.grid_features_values()[feature]
        state.grid_features_uncertainties()[feature] = \
            1. * state_memory.grid_features_uncertainties()[feature]
        # knowledge that state features are transient -> increase uncertainty over time
        forgetfulness = np.power(MEMORY_FORGET_RATE, time_elapsed)
        state.grid_features_uncertainties()[feature] = \
            forgetfulness * state.grid_features_uncertainties()[feature]
        if np.any(np.isnan(state.grid_features_uncertainties()[feature])):
            print("IA SE: nan values in memory update")
            state.grid_features_uncertainties()[feature][
                np.isnan(state.grid_features_uncertainties()[feature])
            ] = np.inf
            state.grid_features_uncertainties()[feature][
                state.grid_features_uncertainties()[feature] > BIG_SIGMASQ
            ] = BIG_SIGMASQ

def uninformed_prior_into_state(state, fixed_state):
    """ The absolutely uninformed prior.
    How you expect the state to be in a place you've never seen """
    map_ = fixed_state.map
    occ = map_.occupancy()
    for feature_name in kStateFeatures:
        feature_index = kStateFeatures[feature_name]
        # feature_map
        if feature_name == "crowdedness":
            feature_map = CROWDEDNESS_UNINFORMED_PRIOR * np.ones_like(occ, dtype=np.float32)
            feature_uncertainty_map = CWD_UNCERTAINTY_UNINFORMED_PRIOR * np.ones_like(occ, dtype=np.float32)
        elif feature_name == "permissivity":
            feature_map = PERMISSIVITY_UNINFORMED_PRIOR * np.ones_like(occ, dtype=np.float32)
            feature_uncertainty_map = PRM_UNCERTAINTY_UNINFORMED_PRIOR * np.ones_like(occ, dtype=np.float32)
        elif feature_name == "perceptivity":
            feature_map = PERCEPTIVITY_UNINFORMED_PRIOR * np.ones_like(occ, dtype=np.float32)
            feature_uncertainty_map = PER_UNCERTAINTY_UNINFORMED_PRIOR * np.ones_like(occ, dtype=np.float32)
        state.grid_features_values()[feature_index, :, :] = feature_map
        state.grid_features_uncertainties()[feature_index, :, :] = feature_uncertainty_map
