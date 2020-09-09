import copy
import numpy as np

# DEBUG ONLY REMOVE
import matplotlib.pyplot as plt
from pyniel.pyplot_tools.colormaps import Greys_c10
from map2d import gridshow
# ------

from cia import CTaskStateUpdate
from cia import kStateFeatures as kSF
from cia import segments_from_bool1d
from cia import sultans_wife

from cia import INTEND_predict_success
from cia import SAY_predict_success
from cia import NUDGE_predict_success


UNCERTAINTY_THRESH = 1.

PROB_DISTANCE_REF = 10.

def example_segment_constraint(segment, path_state):
    """ function which returns 1 if segment obeys constraints, 0 otherwise """
    return 1

def no_segment_constraint(segment, path_state):
    """ function which returns 1 for any segment """
    return 1

def minlength_segment_constraint(segment, path_state):
    """ function which returns 1 if segment is over a certain length, 0 otherwise """
    minlength_m = 1.
    start_xy = path_state.path_xy[segment[0]]
    end_xy = path_state.path_xy[segment[1]]
    segment_length = np.linalg.norm(end_xy - start_xy)
    if segment_length < minlength_m:
        return 0
    return 1

class Outcome(object):
    def __init__(self):
        self.success = None # the anticipated progress
        self.task_state_update = None # some measure of the outcome's impact on state
        self.probability = None
        self.cost = None

class Action(object):
    """ Prototype for action classes """
    def type(self):
        return type(self)

    def typestr(self):
        return str(self.type()).replace("<class 'pyIA.actions.", '').replace("'>", '')

    def letterstr(self):
        return self.typestr()[0]

    def color(self):
        raise NotImplementedError

    # INTERFACE ----
    def predict_success(self, task_state, params):
        raise NotImplementedError("method not implemented for {}".format(self.type()))

    def predict_duration(self, task_state, params):
        raise NotImplementedError("method not implemented for {}".format(self.type()))

    def predict_cost(self, task_state, params):
        return self.predict_duration(task_state, params)

    def sample_outcome(self, task_state, params):
        raise NotImplementedError("method not implemented for {}".format(self.type()))
        return outcome

    def predict_outcome(self, task_state, params):
        raise NotImplementedError("method not implemented for {}".format(self.type()))
        return outcomes

    def check_preconditions_on_path(self, path_state):
        raise NotImplementedError("method not implemented for {}".format(self.type()))
        return proto_segments, segment_constraints

## actions


class Intend(Action):
    """ Local planner follows planned path-to-goal as long as not obstructed, while showing intent.
    TODO: split into two planners, one which knows avoidance will occur and one which doesn't.

    Success depends on the presence of other agents and whether they are perceptive, and permissive """
    def __init__(self):
        pass

    def color(self):
        return np.array([0., 1., 0.5, 1.])  # green

    def predict_success(self, task_state, params):
        """ success prob as a function of task_state P(Success|S) """
        return INTEND_predict_success(task_state)

    def predict_duration(self, task_state, params):
        # assume average velocity of 0.3m/s
        path_length = task_state.path_length()
        duration = path_length / 0.3
        noise = np.random.rand() * 0.2 + 0.9
        return duration * noise

    def predict_outcome(self, task_state, params):
        outcomes = []
        success_prob = self.predict_success(task_state, params)

        # success
        outcome = Outcome()
        outcome.success = 1
        outcome.probability = success_prob
        # cost
        outcome.cost = self.predict_cost(task_state, params)
        # state update
        outcome.task_state_update = CTaskStateUpdate(
            task_state,
            True, # is_pos_update
            task_state.get_subpath_xy()[-1],
            task_state.get_subpath_ij()[-1],
            "INTEND",
            "S",
        )
        outcomes.append(outcome)
        # failure
        outcome = Outcome()
        outcome.success = 0
        outcome.probability = 1-success_prob
        # cost
        outcome.cost = self.predict_cost(task_state, params)
        # if permissivity was high, decrease it
        outcome.task_state_update = CTaskStateUpdate(
            task_state,
            False, # is_pos_update
            None,
            None,
            "INTEND",
            "F",
        )
        outcomes.append(outcome)
        return outcomes

    def check_preconditions_on_path(self, path_state):
        """
        Returns
            proto_segments: list of tuples (start_index, end_index) of segment in path
            segment_constraints: a function with args (segment, PathState),
                and which returns 1 if the segment is within the action constraints, 0 otherwise

        Notes:
            both segment[0] and segment[1] are valid points for action a in the path

            proto_segments are expected to be resized downstream to fit within a task sequence.
            For this reason segment_constraints is returned along with the segments,
            so that the downstream can check whether the resized segments are still valid.
        """
        # custom precondition function
        is_point_valid = \
            np.logical_and.reduce([
                np.logical_or(
                    path_state.path_features_values()[kSF["crowdedness"], :] <= 2.,
                    path_state.path_features_uncertainties()[kSF["crowdedness"], :] >= UNCERTAINTY_THRESH,
                ),
                np.logical_or(
                    path_state.path_features_values()[kSF["permissivity"], :] > 0.5,
                    path_state.path_features_uncertainties()[kSF["permissivity"], :] >= UNCERTAINTY_THRESH,
                ),
            ]).astype(np.uint8)
        # transform bool along path is_point_valid to action segments
        proto_segments = segments_from_bool1d(is_point_valid)
        # segment constraints
        segment_constraints = no_segment_constraint
        return proto_segments, segment_constraints


class Say(Action):
    """ Robot announces plan to move to location,
    Local planner then follows planned path-to-goal as long as not obstructed, while showing intent.

    Increases perceptivity
    Success depends on the presence of other agents and whether they are perceptive, and permissive """

    def __init__(self):
        pass

    def color(self):
        return np.array([0., 0., 1., 1.])  # blue

    def predict_success(self, task_state, params):
        """ success prob as a function of task_state """
        return SAY_predict_success(task_state)

    def predict_duration(self, task_state, params):
        return 1.5 * Intend().predict_duration(task_state, params)

    def predict_outcome(self, task_state, params):
        outcomes = []
        success_prob = self.predict_success(task_state, params)

        # success
        outcome = Outcome()
        outcome.success = 1
        outcome.probability = success_prob
        # cost
        outcome.cost = self.predict_cost(task_state, params)
        outcome.task_state_update = CTaskStateUpdate(
            task_state,
            True, # is_pos_update
            task_state.get_subpath_xy()[-1],
            task_state.get_subpath_ij()[-1],
            "SAY",
            "S",
        )
        outcomes.append(outcome)
        # failure
        outcome = Outcome()
        outcome.success = 0
        outcome.probability = 1-success_prob
        # cost
        outcome.cost = self.predict_cost(task_state, params)
        outcome.task_state_update = CTaskStateUpdate(
            task_state,
            False, # is_pos_update
            None,
            None,
            "SAY",
            "F",
        )
        outcomes.append(outcome)
        return outcomes

    def check_preconditions_on_path(self, path_state):
        """
        Returns
            proto_segments: list of tuples (start_index, end_index) of segment in path
            segment_constraints: a function with args (segment, PathState),
                and which returns 1 if the segment is within the action constraints, 0 otherwise

        Notes:
            both segment[0] and segment[1] are valid points for action a in the path

            proto_segments are expected to be resized downstream to fit within a task sequence.
            For this reason segment_constraints is returned along with the segments,
            so that the downstream can check whether the resized segments are still valid.
        """
        # custom precondition function
        is_point_valid = \
            np.logical_and.reduce([
                path_state.path_features_uncertainties()[kSF["crowdedness"], :] < UNCERTAINTY_THRESH,
                path_state.path_features_values()[kSF["crowdedness"], :] >= 0.5,
                path_state.path_features_values()[kSF["perceptivity"], :] > 0.05,
                path_state.path_features_values()[kSF["perceptivity"], :] < 0.95,
            ]).astype(np.uint8)
        # transform bool along path is_point_valid to action segments
        proto_segments = segments_from_bool1d(is_point_valid)
        # segment constraints
        segment_constraints = no_segment_constraint
        return proto_segments, segment_constraints


class Crawl(Action):
    """ Robot attempts to slowly move past semi-permissive

    # TODO adapt to flow!
    higher cost but ideal for flow scenarios """

    def __init__(self):
        pass

    def color(self):
        return np.array([1., 1., 0., 1.])  # yellow

    def predict_success(self, task_state, params):
        """ success prob as a function of task_state """
        subpath_p =  np.array( task_state.task_features_values()[kSF["permissivity"],:] )
        # multiply by perceptivity factor [0-1] : agents which haven't seen robot can't let it pass
        perceptivity_factor = np.clip(task_state.task_features_values()[kSF["perceptivity"],:], 0., 1.)
        # crowdedness is not a problem
        crowdedness_factor = 1.
        # take minimum of devalued permissivity
        success_prob = np.clip(np.min(subpath_p * crowdedness_factor * perceptivity_factor), 0.05, 0.95)
        return np.power(success_prob, task_state.path_length()/PROB_DISTANCE_REF)

    def predict_duration(self, task_state, params):
        return 2 * Intend().predict_duration(task_state, params)

    def predict_outcome(self, task_state, params):
        """ if crawling failed, the likelihood of a non-permissive agent is high """
        outcomes = []
        success_prob = self.predict_success(task_state, params)

        # success
        outcome = Outcome()
        outcome.success = 1
        outcome.probability = success_prob
        # cost
        outcome.cost = self.predict_cost(task_state, params)
        outcome.task_state_update = CTaskStateUpdate(
            task_state,
            True, # is_pos_update
            task_state.get_subpath_xy()[-1],
            task_state.get_subpath_ij()[-1],
            "CRAWL",
            "S",
        )
        outcomes.append(outcome)
        # failure
        outcome = Outcome()
        outcome.success = 0
        outcome.probability = 1-success_prob
        # cost
        outcome.cost = self.predict_cost(task_state, params)
        outcome.task_state_update = CTaskStateUpdate(
            task_state,
            False, # is_pos_update
            None,
            None,
            "CRAWL",
            "F",
        )
        outcomes.append(outcome)
        return outcomes

    def check_preconditions_on_path(self, path_state):
        """
        Returns
            proto_segments: list of tuples (start_index, end_index) of segment in path
            segment_constraints: a function with args (segment, PathState),
                and which returns 1 if the segment is within the action constraints, 0 otherwise

        Notes:
            both segment[0] and segment[1] are valid points for action a in the path

            proto_segments are expected to be resized downstream to fit within a task sequence.
            For this reason segment_constraints is returned along with the segments,
            so that the downstream can check whether the resized segments are still valid.
        """
        # custom precondition function
        is_point_valid = \
            np.logical_and(
                path_state.path_features_uncertainties()[kSF["permissivity"], :] < UNCERTAINTY_THRESH,
                path_state.path_features_values()[kSF["crowdedness"], :] >= 1.
            ).astype(np.uint8)
        # transform bool along path is_point_valid to action segments
        proto_segments = segments_from_bool1d(is_point_valid)
        # segment constraints
#         segment_constraints = minlength_segment_constraint
        segment_constraints = no_segment_constraint
        return proto_segments, segment_constraints


class Nudge(Action):
    """ Robot attempts to slowly move past voluntarily obstructing agents

    high cost, low progress but may resolve otherwise unsolvable situations """

    def __init__(self):
        pass

    def color(self):
        return np.array([1., 0., 0., 1.])  # red

    def predict_success(self, task_state, params):
        """ success prob as a function of task_state """
        return NUDGE_predict_success(task_state)

    def predict_duration(self, task_state, params):
        # assume average velocity of 0.1 m/s
        return 3 * Intend().predict_duration(task_state, params)

    def predict_outcome(self, task_state, params):
        outcomes = []
        success_prob = self.predict_success(task_state, params)

        # success
        outcome = Outcome()
        outcome.success = 1
        outcome.probability = success_prob
        # cost
        outcome.cost = self.predict_cost(task_state, params)
        outcome.task_state_update = CTaskStateUpdate(
            task_state,
            True, # is_pos_update
            task_state.get_subpath_xy()[-1],
            task_state.get_subpath_ij()[-1],
            "NUDGE",
            "S",
        )
        outcomes.append(outcome)
        # failure
        outcome = Outcome()
        outcome.success = 0
        outcome.probability = 1-success_prob
        # cost
        outcome.cost = self.predict_cost(task_state, params)
        outcome.task_state_update = CTaskStateUpdate(
            task_state,
            False, # is_pos_update
            None,
            None,
            "NUDGE",
            "F",
        )
        outcomes.append(outcome)
        return outcomes

    def check_preconditions_on_path(self, path_state):
        """
        Returns
            proto_segments: list of tuples (start_index, end_index) of segment in path
            segment_constraints: a function with args (segment, PathState),
                and which returns 1 if the segment is within the action constraints, 0 otherwise

        Notes:
            both segment[0] and segment[1] are valid points for action a in the path

            proto_segments are expected to be resized downstream to fit within a task sequence.
            For this reason segment_constraints is returned along with the segments,
            so that the downstream can check whether the resized segments are still valid.
        """
        # custom precondition function
        is_point_valid = \
            np.logical_or.reduce([
                np.logical_and(
                    path_state.path_features_uncertainties()[kSF["crowdedness"], :] < UNCERTAINTY_THRESH,
                    path_state.path_features_values()[kSF["crowdedness"], :] >= 0.5,
                ),
                np.logical_and(
                    path_state.path_features_uncertainties()[kSF["permissivity"], :] < UNCERTAINTY_THRESH,
                    path_state.path_features_values()[kSF["permissivity"], :] <= 0.5,
                ),
            ]).astype(np.uint8)
        # transform bool along path is_point_valid to action segments
        proto_segments = segments_from_bool1d(is_point_valid)
        # segment constraints
#         segment_constraints = minlength_segment_constraint
        segment_constraints = no_segment_constraint
        return proto_segments, segment_constraints


class Look(Action):
    """ Robot looks around in order to increase knowledge and certainty in state estimate """
    def __init__(self):
        pass

class Loiter(Action):
    """ Agent stays at a point, while avoiding physical contact """
    def __init__(self):
        pass

# Hacked actions to emulate single-action planning

class OnlyIntend(Action):
    """ Local planner follows planned path-to-goal as long as not obstructed, while showing intent.
    TODO: split into two planners, one which knows avoidance will occur and one which doesn't.

    Success depends on the presence of other agents and whether they are perceptive, and permissive """
    def __init__(self):
        pass

    def typestr(self):
        return "Intend"

    def color(self):
        return np.array([0., 1., 0.5, 1.])  # green

    def predict_success(self, task_state, params):
        """ success prob as a function of task_state P(Success|S) """
        return 99.9

    def predict_duration(self, task_state, params):
        # assume average velocity of 0.3m/s
        path_length = task_state.path_length()
        duration = path_length / 0.3
        noise = np.random.rand() * 0.2 + 0.9
        return duration * noise

    def predict_outcome(self, task_state, params):
        outcomes = []
        success_prob = self.predict_success(task_state, params)

        # success
        outcome = Outcome()
        outcome.success = 1
        outcome.probability = success_prob
        # cost
        outcome.cost = self.predict_cost(task_state, params)
        # state update
        outcome.task_state_update = CTaskStateUpdate(
            task_state,
            True, # is_pos_update
            task_state.get_subpath_xy()[-1],
            task_state.get_subpath_ij()[-1],
            "INTEND",
            "S",
        )
        outcomes.append(outcome)
        # failure
        outcome = Outcome()
        outcome.success = 0
        outcome.probability = 1-success_prob
        # cost
        outcome.cost = self.predict_cost(task_state, params)
        # if permissivity was high, decrease it
        outcome.task_state_update = CTaskStateUpdate(
            task_state,
            False, # is_pos_update
            None,
            None,
            "INTEND",
            "F",
        )
        outcomes.append(outcome)
        return outcomes

    def check_preconditions_on_path(self, path_state):
        """
        Returns
            proto_segments: list of tuples (start_index, end_index) of segment in path
            segment_constraints: a function with args (segment, PathState),
                and which returns 1 if the segment is within the action constraints, 0 otherwise

        Notes:
            both segment[0] and segment[1] are valid points for action a in the path

            proto_segments are expected to be resized downstream to fit within a task sequence.
            For this reason segment_constraints is returned along with the segments,
            so that the downstream can check whether the resized segments are still valid.
        """
        # custom precondition function
        is_point_valid = \
            np.ones_like(path_state.path_features_values()[kSF["crowdedness"], :], dtype=np.uint8)
        # transform bool along path is_point_valid to action segments
        proto_segments = segments_from_bool1d(is_point_valid)
        # segment constraints
        segment_constraints = no_segment_constraint
        return proto_segments, segment_constraints


class OnlySay(Action):
    """ Robot announces plan to move to location,
    Local planner then follows planned path-to-goal as long as not obstructed, while showing intent.

    Increases perceptivity
    Success depends on the presence of other agents and whether they are perceptive, and permissive """

    def __init__(self):
        pass

    def typestr(self):
        return "Say"

    def color(self):
        return np.array([0., 0., 1., 1.])  # blue

    def predict_success(self, task_state, params):
        """ success prob as a function of task_state """
        return 99.9

    def predict_duration(self, task_state, params):
        return 1.5 * Intend().predict_duration(task_state, params)

    def predict_outcome(self, task_state, params):
        outcomes = []
        success_prob = self.predict_success(task_state, params)

        # success
        outcome = Outcome()
        outcome.success = 1
        outcome.probability = success_prob
        # cost
        outcome.cost = self.predict_cost(task_state, params)
        outcome.task_state_update = CTaskStateUpdate(
            task_state,
            True, # is_pos_update
            task_state.get_subpath_xy()[-1],
            task_state.get_subpath_ij()[-1],
            "SAY",
            "S",
        )
        outcomes.append(outcome)
        # failure
        outcome = Outcome()
        outcome.success = 0
        outcome.probability = 1-success_prob
        # cost
        outcome.cost = self.predict_cost(task_state, params)
        outcome.task_state_update = CTaskStateUpdate(
            task_state,
            False, # is_pos_update
            None,
            None,
            "SAY",
            "F",
        )
        outcomes.append(outcome)
        return outcomes

    def check_preconditions_on_path(self, path_state):
        """
        Returns
            proto_segments: list of tuples (start_index, end_index) of segment in path
            segment_constraints: a function with args (segment, PathState),
                and which returns 1 if the segment is within the action constraints, 0 otherwise

        Notes:
            both segment[0] and segment[1] are valid points for action a in the path

            proto_segments are expected to be resized downstream to fit within a task sequence.
            For this reason segment_constraints is returned along with the segments,
            so that the downstream can check whether the resized segments are still valid.
        """
        # custom precondition function
        is_point_valid = \
            np.ones_like(path_state.path_features_values()[kSF["crowdedness"], :], dtype=np.uint8)
        # transform bool along path is_point_valid to action segments
        proto_segments = segments_from_bool1d(is_point_valid)
        # segment constraints
        segment_constraints = no_segment_constraint
        return proto_segments, segment_constraints


class OnlyNudge(Action):
    """ Robot attempts to slowly move past voluntarily obstructing agents

    high cost, low progress but may resolve otherwise unsolvable situations """

    def __init__(self):
        pass

    def typestr(self):
        return "Nudge"

    def color(self):
        return np.array([1., 0., 0., 1.])  # red

    def predict_success(self, task_state, params):
        """ success prob as a function of task_state """
        return 99.9

    def predict_duration(self, task_state, params):
        # assume average velocity of 0.1 m/s
        return 3 * Intend().predict_duration(task_state, params)

    def predict_outcome(self, task_state, params):
        outcomes = []
        success_prob = self.predict_success(task_state, params)

        # success
        outcome = Outcome()
        outcome.success = 1
        outcome.probability = success_prob
        # cost
        outcome.cost = self.predict_cost(task_state, params)
        outcome.task_state_update = CTaskStateUpdate(
            task_state,
            True, # is_pos_update
            task_state.get_subpath_xy()[-1],
            task_state.get_subpath_ij()[-1],
            "NUDGE",
            "S",
        )
        outcomes.append(outcome)
        # failure
        outcome = Outcome()
        outcome.success = 0
        outcome.probability = 1-success_prob
        # cost
        outcome.cost = self.predict_cost(task_state, params)
        outcome.task_state_update = CTaskStateUpdate(
            task_state,
            False, # is_pos_update
            None,
            None,
            "NUDGE",
            "F",
        )
        outcomes.append(outcome)
        return outcomes

    def check_preconditions_on_path(self, path_state):
        """
        Returns
            proto_segments: list of tuples (start_index, end_index) of segment in path
            segment_constraints: a function with args (segment, PathState),
                and which returns 1 if the segment is within the action constraints, 0 otherwise

        Notes:
            both segment[0] and segment[1] are valid points for action a in the path

            proto_segments are expected to be resized downstream to fit within a task sequence.
            For this reason segment_constraints is returned along with the segments,
            so that the downstream can check whether the resized segments are still valid.
        """
        # custom precondition function
        is_point_valid = \
            np.ones_like(path_state.path_features_values()[kSF["crowdedness"], :], dtype=np.uint8)
        # transform bool along path is_point_valid to action segments
        proto_segments = segments_from_bool1d(is_point_valid)
        # segment constraints
#         segment_constraints = minlength_segment_constraint
        segment_constraints = no_segment_constraint
        return proto_segments, segment_constraints

