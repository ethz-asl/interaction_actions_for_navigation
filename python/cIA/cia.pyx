# distutils: language=c++

from libcpp cimport bool
from libcpp.queue cimport priority_queue as cpp_priority_queue
from libcpp.pair cimport pair as cpp_pair
import numpy as np
import copy
cimport numpy as np
from cython.operator cimport dereference as deref
cimport cython
from math import sqrt
from libc.math cimport cos as ccos
from libc.math cimport sin as csin
from libc.math cimport acos as cacos
from libc.math cimport sqrt as csqrt
from libc.math cimport floor as cfloor

import matplotlib.pyplot as plt
from CMap2D import path_from_dijkstra_field
from CMap2D import CMap2D
from map2d import gridshow

# compile time feature definition
DEF N_STATE_FEATURES = 3
DEF CROWDEDNESS_INDEX = 0
DEF PERMISSIVITY_INDEX = 1
DEF PERCEPTIVITY_INDEX = 2
DEF CROWDEDNESS_NAME = u"crowdedness"
DEF PERMISSIVITY_NAME = u"permissivity"
DEF PERCEPTIVITY_NAME = u"perceptivity"

DEF PROB_DISTANCE_REF = 10.

# feature definition (exported to outside libraries)
kStateFeatures = {
    CROWDEDNESS_NAME: CROWDEDNESS_INDEX,
    PERMISSIVITY_NAME: PERMISSIVITY_INDEX,
    PERCEPTIVITY_NAME: PERCEPTIVITY_INDEX,
} # flow, agency, ...


# Features are represented as normal distributions at every point in space
# Ensemble of state features over N-dimensional space
cdef class CVectorFeatures:
    cdef np.float32_t[:,::1] values
    cdef np.float32_t[:,::1] uncertainties
    def __cinit__(self, int task_path_length):
        self.values = np.zeros((N_STATE_FEATURES, task_path_length), dtype=np.float32)
        self.uncertainties = np.inf * np.ones((N_STATE_FEATURES, task_path_length), dtype=np.float32)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef void copy_from(self, CVectorFeatures source):
        self.values[:,:] = source.values
        self.uncertainties[:,:] = source.uncertainties
    cdef np.float32_t[:,::1] _values(self):
        return self.values
    cdef np.float32_t[:,::1] _uncertainties(self):
        return self.uncertainties
cdef class CGridFeatures:
    cdef np.float32_t[:,:,::1] values
    cdef np.float32_t[:,:,::1] uncertainties
    cdef int grid_width
    cdef int grid_height
    def __cinit__(self, int state_map_width, int state_map_height):
        self.grid_width = state_map_width
        self.grid_height = state_map_height
        self.values = np.zeros(
            (N_STATE_FEATURES, state_map_width, state_map_height), dtype=np.float32)
        self.uncertainties = np.inf * np.ones(
            (N_STATE_FEATURES, state_map_width, state_map_height), dtype=np.float32)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef void copy_from(self, CGridFeatures source):
        self.values[:,:,:] = source.values
        self.uncertainties[:,:,:] = source.uncertainties
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef void fill(self, np.float32_t[:,:,::1] values, np.float32_t[:,:,::1] uncertainties):
        cdef int i
        cdef int j
        cdef int n
        cdef int max_n = N_STATE_FEATURES
        for i in range(self.grid_width):
            for j in range(self.grid_height):
                for n in range(max_n):
                    self.values[n, i, j] = values[n, i, j]
                    self.uncertainties[n, i, j] = uncertainties[n, i, j]
    cdef np.float32_t[:,:,::1] _values(self):
        return self.values
    cdef np.float32_t[:,:,::1] _uncertainties(self):
        return self.uncertainties
    def serialize(self):
        return {
            "values": np.array(self.values),
            "uncertainties": np.array(self.uncertainties),
            "grid_width": int(self.grid_width),
            "grid_height": int(self.grid_height),
        }
def unserialize_cgridfeatures(dict_):
    gf = CGridFeatures(dict_["grid_width"], dict_["grid_height"])
    gf.fill(dict_["values"], dict_["uncertainties"])
    return gf

# Feature diffs stores the operation to be executed on a feature
ctypedef np.float32_t (*Func)(np.float32_t[:] values, np.float32_t[:] uncertainties)
cdef class CFeaturesDiff:
    cdef public bool is_changed[N_STATE_FEATURES]
    cdef Func value_operations[N_STATE_FEATURES]
    cdef Func uncertainty_operations[N_STATE_FEATURES]
    def __cinit__(self):
        cdef int i
        for i in range(N_STATE_FEATURES):
            self.is_changed[i] = False

# The following classes represent the state at various levels of locality
cdef class CPathState:
    """ Reduction of State: 2d elements are now 1d along path"""
    cdef public int pos_along_path
    cdef public np.float32_t[:,::1] path_xy
    cdef public np.float32_t[:,::1] path_ij # redundant
    cdef public CVectorFeatures path_features
    def __cinit__(self):
        pass
    # from path state
    def from_state_and_path(self, state, path_xy, path_ij):
        self.cfrom_state_and_path(state, path_xy, path_ij)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef cfrom_state_and_path(self, CState state, np.float32_t[:,::1] path_xy, np.float32_t[:,::1] path_ij):
        self.pos_along_path = index_along_path_from_pos_ij(state.pos_ij, path_ij)
        self.path_xy = path_xy
        self.path_ij = path_ij
        self.path_features = CVectorFeatures(path_ij.shape[0])
        cdef int n
        cdef int k
        cdef int i
        cdef int j
        for n in range(N_STATE_FEATURES):
            for k in range(path_ij.shape[0]):
                i = int(path_ij[k][0])
                j = int(path_ij[k][1])
                self.path_features._values()[n, k] = state.grid_features._values()[n, i, j]
                self.path_features._uncertainties()[n, k] = state.grid_features._uncertainties()[n, i, j]
    def copy(self):
        return self.ccopy()
    cdef CPathState ccopy(self):
        copy_ = CPathState()
        copy_.pos_along_path = self.pos_along_path
        copy_.path_xy = self.path_xy
        copy_.path_ij = self.path_ij
        copy_.path_features = CVectorFeatures(self.path_ij.shape[0])
        copy_.path_features.copy_from(self.path_features)
        return copy_
    def path_features_values(self):
        return np.array(self.path_features._values())
    def path_features_uncertainties(self):
        return np.array(self.path_features._uncertainties())
    def get_path_len(self):
        return len(self.path_xy)
    def get_path_xy(self):
        return np.array(self.path_xy)
    def get_path_ij(self):
        return np.array(self.path_ij)
def path_state_from_state_and_path(state, path_xy, path_ij):
    path_state = CPathState()
    path_state.from_state_and_path(state, path_xy, path_ij)
    return path_state

cdef class CTaskState:
    """ Reduction of State: 2d elements are now 1d, for single action along a subpath"""
    cdef int startidx_in_fullpath
    cdef np.float32_t[:,::1] subpath_xy
    cdef np.float32_t[:,::1] subpath_ij
    cdef CVectorFeatures task_features
    cdef np.float32_t resolution
    # from path state
    def __cinit__(self, CPathState path_state, int startidx, int endidx, np.float32_t resolution):
        self.cfrom_path_state(path_state, startidx, endidx, resolution)
    cdef cfrom_path_state(self, CPathState path_state, int startidx, int endidx, np.float32_t resolution):
        self.task_features = CVectorFeatures(endidx+1 - startidx)
        self.subpath_xy = path_state.path_xy[startidx:endidx+1,:]
        self.subpath_ij = path_state.path_ij[startidx:endidx+1,:]
        self.resolution = resolution
        self.startidx_in_fullpath = startidx
        # along subpath
        self.task_features._values()[:,:] = path_state.path_features._values()[:,startidx:endidx+1]
        self.task_features._uncertainties()[:,:] = path_state.path_features._uncertainties()[:,startidx:endidx+1]
    def get_subpath_xy(self):
        return np.array(self.subpath_xy)
    def get_subpath_ij(self):
        return np.array(self.subpath_ij)
    def task_features_values(self):
        return self.task_features._values()
    def task_features_uncertainties(self):
        return self.task_features._uncertainties()
    def path_length(self):
        return self.cpath_length()
    cdef np.float32_t cpath_length(self):
        cdef np.float32_t length = len(self.subpath_ij) * self.resolution
        return length
cdef class CTaskStateUpdate:
    """ Diff for a TaskState, obtained from predict_outcome """
    cdef bool is_pos_update
    cdef np.float32_t[:] pos_update
    cdef np.float32_t[:] pos_ij_update
    # task definition
    cdef int startidx_in_fullpath
    cdef np.float32_t[:,::1] subpath_xy
    cdef np.float32_t[:,::1] subpath_ij
    cdef CVectorFeatures task_features
    cdef CFeaturesDiff features_diff
    def __cinit__(
                self, CTaskState task_state,
                bool is_pos_update,
                np.float32_t[:] pos_update,
                np.float32_t[:] pos_ij_update,
                str action_name,
                str success,
            ):
        self.cgenerate_update_properties(
            task_state,
            is_pos_update,
            pos_update,
            pos_ij_update,
            action_name,
            success,
        )
    cdef cgenerate_update_properties(
                self, CTaskState task_state,
                bool is_pos_update,
                np.float32_t[:] pos_update,
                np.float32_t[:] pos_ij_update,
                str action_name,
                str success,
            ):
        self.is_pos_update = is_pos_update
        if is_pos_update:
            self.pos_update = pos_update
            self.pos_ij_update = pos_ij_update
        else:
            self.pos_update = np.zeros((2,), dtype=np.float32)
            self.pos_ij_update = np.zeros((2,), dtype=np.float32)
        # task definition
        self.startidx_in_fullpath = task_state.startidx_in_fullpath
        self.subpath_xy = task_state.subpath_xy
        self.subpath_ij = task_state.subpath_ij
        self.task_features = task_state.task_features
        # diff
        self.features_diff = CFeaturesDiff()
        # Fill in diffs
        if action_name == "INTEND":
            if success == "S":
                self.features_diff.is_changed[PERMISSIVITY_INDEX] = True
                self.features_diff.value_operations[PERMISSIVITY_INDEX] = INTEND_S_PERMISSIVITY_value_operation
                self.features_diff.uncertainty_operations[PERMISSIVITY_INDEX] = INTEND_S_PERMISSIVITY_uncertainty_operation
            else:
                self.features_diff.is_changed[PERMISSIVITY_INDEX] = True
                self.features_diff.value_operations[PERMISSIVITY_INDEX] = INTEND_F_PERMISSIVITY_value_operation
                self.features_diff.uncertainty_operations[PERMISSIVITY_INDEX] = INTEND_F_PERMISSIVITY_uncertainty_operation
                self.features_diff.is_changed[PERCEPTIVITY_INDEX] = True
                self.features_diff.value_operations[PERCEPTIVITY_INDEX] = INTEND_F_PERCEPTIVITY_value_operation
                self.features_diff.uncertainty_operations[PERCEPTIVITY_INDEX] = INTEND_F_PERCEPTIVITY_uncertainty_operation
        if action_name == "SAY":
            if success == "S":
                self.features_diff.is_changed[PERMISSIVITY_INDEX] = True
                self.features_diff.value_operations[PERMISSIVITY_INDEX] = SAY_S_PERMISSIVITY_value_operation
                self.features_diff.uncertainty_operations[PERMISSIVITY_INDEX] = SAY_S_PERMISSIVITY_uncertainty_operation
            else:
                self.features_diff.is_changed[PERMISSIVITY_INDEX] = True
                self.features_diff.value_operations[PERMISSIVITY_INDEX] = SAY_F_PERMISSIVITY_value_operation
                self.features_diff.uncertainty_operations[PERMISSIVITY_INDEX] = SAY_F_PERMISSIVITY_uncertainty_operation
                self.features_diff.is_changed[PERCEPTIVITY_INDEX] = True
                self.features_diff.value_operations[PERCEPTIVITY_INDEX] = SAY_F_PERCEPTIVITY_value_operation
                self.features_diff.uncertainty_operations[PERCEPTIVITY_INDEX] = SAY_F_PERCEPTIVITY_uncertainty_operation
        if action_name == "CRAWL":
            if success == "S":
                pass
            else:
                self.features_diff.is_changed[PERMISSIVITY_INDEX] = True
                self.features_diff.value_operations[PERMISSIVITY_INDEX] = CRAWL_F_PERMISSIVITY_value_operation
                self.features_diff.uncertainty_operations[PERMISSIVITY_INDEX] = CRAWL_F_PERMISSIVITY_uncertainty_operation
        if action_name == "NUDGE":
            if success == "S":
                pass
            else:
                self.features_diff.is_changed[PERMISSIVITY_INDEX] = True
                self.features_diff.value_operations[PERMISSIVITY_INDEX] = NUDGE_F_PERMISSIVITY_value_operation
                self.features_diff.uncertainty_operations[PERMISSIVITY_INDEX] = NUDGE_F_PERMISSIVITY_uncertainty_operation
    def apply_to_state_along_taskpath(self, CState state):
        return self.capply_to_state_along_taskpath(state)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef bool capply_to_state_along_taskpath(self, CState state):
        """ Applies diff along task path to state, inplace. UNUSED (State remains constant) """
        cdef bool is_any_changed = False
        cdef int n
        cdef int k
        cdef int i
        cdef int j
        for n in range(N_STATE_FEATURES):
            if self.features_diff.is_changed[n]:
                is_any_changed = True
                break
        if is_any_changed:
            # TODO currently diff is point_to_point, change? (circular stencils, maybe?)
            for k in range(self.subpath_ij.shape[0]):
                i = int(self.subpath_ij[k, 0])
                j = int(self.subpath_ij[k, 1])
                for n in range(N_STATE_FEATURES):
                    if self.features_diff.is_changed[n]:
                        state.grid_features.values[n, i, j] = self.features_diff.value_operations[n](
                            state.grid_features.values[:, i, j],
                            state.grid_features.uncertainties[:, i, j]
                        )
                        state.grid_features.uncertainties[n, i, j] = self.features_diff.uncertainty_operations[n](
                            state.grid_features.values[:, i, j],
                            state.grid_features.uncertainties[:, i, j]
                        )
        # update the position if it has changed
        if self.is_pos_update:
            is_any_changed = True
            state.pos = self.pos_update
            state.pos_ij = self.pos_ij_update
        return is_any_changed
    def apply_to_path_state(self, CPathState path_state):
        return self.capply_to_path_state(path_state)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef bool capply_to_path_state(self, CPathState path_state):
        """ Applies diff along task path to path state, inplace """
        cdef bool is_any_changed = False
        cdef int n
        cdef int k
        cdef int i
        cdef int j
        cdef int k_start = self.startidx_in_fullpath
        cdef int k_end = self.startidx_in_fullpath + self.subpath_ij.shape[0]
        for n in range(N_STATE_FEATURES):
            if self.features_diff.is_changed[n]:
                is_any_changed = True
                break
        if is_any_changed:
            # TODO currently diff is point_to_point, change? (circular stencils, maybe?)
            for k in range(k_start, k_end):
                for n in range(N_STATE_FEATURES):
                    if self.features_diff.is_changed[n]:
                        path_state.path_features.values[n, k] = \
                            self.features_diff.value_operations[n](
                                path_state.path_features.values[:, k],
                                path_state.path_features.uncertainties[:, k]
                            )
                        path_state.path_features.uncertainties[n, k] = \
                            self.features_diff.uncertainty_operations[n](
                                path_state.path_features.values[:, k],
                                path_state.path_features.uncertainties[:, k]
                            )
        # update the position if it has changed
        if self.is_pos_update:
            is_any_changed = True
            path_state.pos_along_path = k_end-1
        return is_any_changed

cdef class CState:
    """ storage class for convenience 
    Simplified state used by IA from single agent POV over the whole map """
    cdef float radius
    cdef np.float32_t[:] pos
    cdef np.float32_t[:] pos_ij # not rounded. redundant, for performance
    cdef np.float32_t[:] goal
    cdef CGridFeatures grid_features
    def __cinit__(self, radius, pos, pos_ij, mapwidth, mapheight, goal):
        # would like to do away with these
        self.radius = radius
        self.pos = pos
        self.pos_ij = pos_ij
        self.goal = goal
        # mathematical state features
        self.grid_features = CGridFeatures(mapwidth, mapheight)
    def copy(self):
        return self.ccopy()
    cdef CState ccopy(self):
        copy_ = CState(self.radius, self.pos, self.pos_ij,
                       self.grid_features.grid_width, self.grid_features.grid_height, self.goal)
        copy_.grid_features.copy_from(self.grid_features)
        return copy_
    def grid_features_values(self):
        return np.asarray(self.grid_features._values())
    def grid_features_uncertainties(self):
        return np.asarray(self.grid_features._uncertainties())
    def get_pos_ij(self):
        return np.asarray(self.pos_ij)
    def get_pos(self):
        return np.asarray(self.pos)
    def get_goal(self):
        return np.asarray(self.goal)
    def get_radius(self):
        return self.radius
    def serialize(self):
        return {
            "radius": float(self.radius),
            "pos": np.array(self.pos),
            "pos_ij": np.array(self.pos_ij),
            "goal": np.array(self.goal),
            "grid_features": self.grid_features.serialize(),
        }
def unserialize_cstate(dict_):
    gf = unserialize_cgridfeatures(dict_["grid_features"])
    s = CState(
        dict_["radius"], dict_["pos"], dict_["pos_ij"], 
        dict_["grid_features"]["grid_width"], dict_["grid_features"]["grid_height"], dict_["goal"]
    )
    s.grid_features = gf
    return s



class FixedState(object):
    """ Stores the constant parts of the state separately, to reduce computation """
    def __init__(self, map_):
        self.map = map_
        self.derived_state = DerivedState()
    def serialize(self):
        return {
            "map": self.map.serialize(),
            "derived_state": None, # TODO actually serialize
        }
def unserialize_fixedstate(dict_):
    map_ = CMap2D()
    map_.unserialize(dict_["map"])
    fs = FixedState(map_)
    return fs


class DerivedState(object):
    """ convenience storage class for precomputed/cached fields,
    this class is for algorithmic optimization and thus isn't formally in the IA algo"""
    def __init__(self):
        self._map_sdf = None
        self._cache_map = None

        self._suppress_warnings = False

    # CORE INTERFACE -----------
    def copy(self):
        copy_ = DerivedState()
        copy_._map_sdf = self._map_sdf
        copy_._cache_map = self._cache_map
        return copy_

    def map_sdf(self, fixed_state):
        if fixed_state.map is not self._cache_map:
            self._on_set_map(fixed_state)
        return self._map_sdf

    def dijkstra_from_goal(self, state, fixed_state, path_variant):
        goal_ij = fixed_state.map.xy_to_ij([state.get_goal()])[0]
        sdf = self.map_sdf(fixed_state)
        is_traversable = path_variant.is_state_traversable(state, fixed_state)
        mask = np.logical_not(is_traversable).astype(np.uint8)
        extra_costs = (0.2/(0.00001+sdf**2) + 10 * (sdf < state.get_radius())).astype(np.float32)
        speeds = 1./(1.+extra_costs)
        return fixed_state.map.fastmarch(goal_ij, mask=mask, speeds=speeds)

    def path_to_goal(self, state, fixed_state, path_variant):
        agent_ij = state.get_pos_ij()
        dijkstra = self.dijkstra_from_goal(state, fixed_state, path_variant)
        path_ij, _ = path_from_dijkstra_field(dijkstra, agent_ij, connectedness=8)
        path_xy = fixed_state.map.ij_to_xy(path_ij)
        # check if path reaches goal
        goal_ij = fixed_state.map.xy_to_ij([state.get_goal()])[0]
        if not np.allclose(path_ij[-1], goal_ij):
#             print("Path does not reach goal.")
            return None
        return path_xy

    def suppress_warnings(self, set_to=True):
        self._suppress_warnings = set_to


    # INTERNAL --------------
    def _on_set_map(self, fixed_state):
        if not self._suppress_warnings:
            print("DerivedState: Precomputing cached sdf")
        self._map_sdf = fixed_state.map.as_sdf()
        self._cache_map = fixed_state.map


cdef int index_along_path_from_pos_ij(np.float32_t[:] pos_ij, np.float32_t[:,::1] path_ij):
    cdef int n
    cdef np.float32_t i
    cdef np.float32_t j
    for n in range(path_ij.shape[0]):
        i = path_ij[n,0]
        j = path_ij[n,1]
        if cfloor(i) == cfloor(pos_ij[0]) and cfloor(j) == cfloor(pos_ij[1]):
            return n
    raise ValueError("State position {} not in path {}".format(pos_ij, path_ij))

# Utilitites
# ----------------------------------
from libc.stdlib cimport rand, RAND_MAX
def sultans_wife(probs):
    """
    see http://datagenetics.com/blog/november52019/index.html
    assumes probs sum up to 1
    """
    return csultans_wife(probs)
cdef int csultans_wife(np.float32_t[:] probs):
    cdef np.float32_t r
    cdef int i
    cdef np.float32_t remaining_p_space = 1.
    cdef np.float32_t prob
    cdef np.float32_t rescaled_prob # probability rescaled to remaining p space
    for i in range(len(probs)):
        prob = probs[i]
        rescaled_prob = prob / remaining_p_space
        r = np.float32(rand())/np.float32(RAND_MAX) # random integer 0-1
        if r <= rescaled_prob:
            return i
        else:
            remaining_p_space -= prob
    raise ValueError

def segment_endindex_at_pathpos_i(np.int16_t[:,::1] segments, CPathState path_state):
    """ 
    segments              endindices
    [[0, 2], [4, 6]] --> [2,2,2,-1,6,6,6]
    """
    # -1 represents an invalid value, not the end of the path!
    endindices = -1 * np.ones((path_state.path_xy.shape[0],), dtype=np.int16)
    csegment_endindex_at_pathpos_i(segments, endindices, path_state)
    return endindices
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void csegment_endindex_at_pathpos_i(np.int16_t[:,::1] segments, np.int16_t[:] endindices, CPathState path_state):
    cdef np.int16_t n
    cdef np.int16_t i
    for n in range(segments.shape[0]):
        for i in range(segments[n,0], segments[n,1]+1):
            endindices[i] = segments[n,1]

def segments_from_bool1d(bool1d):
    """
    transform list of bools to segments where bool1d is true
    11100011111110 -> 2 segments, {(0,2), (6, 12)} """
    segments = np.zeros((cn_segments_from_bool1d(bool1d), 2), dtype=np.int16)
    csegments_from_bool1d(bool1d, segments)
    return segments
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef int cn_segments_from_bool1d(np.uint8_t[:] bool1d):
    cdef int i
    cdef int n_segments = 0
    cdef np.uint8_t segment_is_open = False
    for i in range(bool1d.shape[0]):
        if not segment_is_open:
            if bool1d[i]:
                segment_is_open = True
            continue
        else:
            if bool1d[i]:
                continue
            else:
                n_segments += 1
                segment_is_open = False
    if segment_is_open:
        n_segments += 1
        segment_is_open = False
    return n_segments
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef csegments_from_bool1d(np.uint8_t[:] bool1d, np.int16_t[:,::1] segments):
    cdef int i
    cdef int segment_index = 0
    cdef np.uint8_t segment_is_open = False
    for i in range(bool1d.shape[0]):
        if not segment_is_open:
            if bool1d[i]:
                # open new segment
                segment_is_open = True
                segments[segment_index, 0] = i
            continue
        else:
            if bool1d[i]:
                continue
            else:
                segments[segment_index, 1] = i
                segment_is_open = False
                segment_index += 1
    if segment_is_open:
        segments[segment_index, 1] = i


def bool1d_from_segments(segments, length=None):
    """
    reverse of segments_from_bool1d operation
    11100011111110 <- 2 segments, {(0,2), (6, 12)} """
    if length is None:
        length = int(0)
        clength_bool1d_from_segments(segments, length)
    bool1d = np.zeros((length), dtype=np.uint8)
    cbool1d_from_segments(segments, bool1d)
    return bool1d

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void clength_bool1d_from_segments(np.int16_t[:,::1] segments, int length):
    cdef np.int16_t end = 0
    cdef int i
    for i in range(segments.shape[0]):
        end = segments[i,1]
        if end > length:
            length = end
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void cbool1d_from_segments(np.int16_t[:,::1] segments, np.uint8_t[:] bool1d):
    cdef int i
    cdef int n
    for n in range(segments.shape[0]):
        for i in range(segments[n,0], segments[n,1]+1):
            bool1d[i] = True

@cython.nonecheck(False)
cdef np.float32_t cclip(np.float32_t val, np.float32_t min_, np.float32_t max_):
    cdef np.float32_t result = val
    if val < min_:
        result = min_
    if val > max_:
        result = max_
    return result

# Below are success probabilities for actions P(Success | S)
# ---------------------------------------------------------------------------------
def INTEND_predict_success(task_state):
    return cINTEND_predict_success(task_state)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef np.float32_t cINTEND_predict_success(CTaskState task_state):
    # probability as a function of state
    cdef int k
    cdef int k_len = len(task_state.subpath_ij)
    cdef np.float32_t k_length = k_len * task_state.resolution
    cdef np.float32_t path_prob
    cdef np.float32_t cumulative_prob = 1.
    cdef np.float32_t permissivity_factor
    cdef np.float32_t perceptivity_factor
    cdef np.float32_t crowdedness_factor
    cdef np.float32_t local_prob
    cdef np.float32_t distance_equivalent_prob
    # sample values from distribution (take max prob i.e. mean)
    cdef np.float32_t[:,::1] values = task_state.task_features_values()
    cdef np.float32_t[:,::1] uncertainties = task_state.task_features_uncertainties()
    for k in range(k_len):
        permissivity_factor = values[PERMISSIVITY_INDEX, k]
        perceptivity_factor = cclip(values[PERCEPTIVITY_INDEX, k], 0., 1.)
        crowdedness_factor = 0.5 / cclip(values[CROWDEDNESS_INDEX, k], 1., 10.)
        # probability of success for single step in path ( if that step is 10m long )
        local_prob = permissivity_factor * perceptivity_factor * crowdedness_factor
        cumulative_prob = min(cumulative_prob, local_prob)  # the right way (geometric mean) is numerically unstable
    path_prob = cclip(cumulative_prob, 0.05, 0.99)
    # convert from reference step size (10m) to actual size
    distance_equivalent_prob = path_prob ** (k_length / PROB_DISTANCE_REF)
    # shortcut (numerically unstable)
#     cumulative_prob = cumulative_prob * local_prob
#     distance_equivalent_prob = cumulative_prob ** (task_state.resolution / PROB_DISTANCE_REF)
    return distance_equivalent_prob

def SAY_predict_success(task_state):
    return cSAY_predict_success(task_state)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef np.float32_t cSAY_predict_success(CTaskState task_state):
    # probability as a function of state
    cdef int k
    cdef int k_len = len(task_state.subpath_ij)
    cdef np.float32_t k_length = k_len * task_state.resolution
    cdef np.float32_t path_prob
    cdef np.float32_t cumulative_prob = 1.
    cdef np.float32_t permissivity_bonus = 1.5
    cdef np.float32_t permissivity_factor
    cdef np.float32_t perceptivity_factor = 1.  # saying action should have maxed PER
    cdef np.float32_t crowdedness_factor
    cdef np.float32_t local_prob
    cdef np.float32_t distance_equivalent_prob
    # sample values from distribution (take max prob i.e. mean)
    cdef np.float32_t[:,::1] values = task_state.task_features_values()
    cdef np.float32_t[:,::1] uncertainties = task_state.task_features_uncertainties()
    for k in range(k_len):
        permissivity_factor = permissivity_bonus * values[PERMISSIVITY_INDEX, k]
        crowdedness_factor = 1. / cclip(values[CROWDEDNESS_INDEX, k], 1., 10.)
        # probability of success for single step in path ( if that step is 10m long )
        local_prob = permissivity_factor * perceptivity_factor * crowdedness_factor
        cumulative_prob = min(cumulative_prob, local_prob)
    path_prob = cclip(cumulative_prob, 0.05, 0.99)
    # convert from reference step size (10m) to actual size
    distance_equivalent_prob = path_prob ** (k_length / PROB_DISTANCE_REF)
    # shortcut (numerically unstable)
#     cumulative_prob = cumulative_prob * local_prob
#     distance_equivalent_prob = cumulative_prob ** (task_state.resolution / PROB_DISTANCE_REF)
    return distance_equivalent_prob

def NUDGE_predict_success(task_state):
    return cNUDGE_predict_success(task_state)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef np.float32_t cNUDGE_predict_success(CTaskState task_state):
    # probability as a function of state
    cdef int k
    cdef int k_len = len(task_state.subpath_ij)
    cdef np.float32_t k_length = k_len * task_state.resolution
    cdef np.float32_t path_prob
    cdef np.float32_t cumulative_prob = 1.
    cdef np.float32_t permissivity_bonus = 5.
    cdef np.float32_t permissivity_factor
    cdef np.float32_t perceptivity_factor = 1.  # perceptivity is not a problem
    cdef np.float32_t crowdedness_factor = 1.  # crowdedness is not a problem
    cdef np.float32_t local_prob
    cdef np.float32_t distance_equivalent_prob
    # sample values from distribution (take max prob i.e. mean)
    cdef np.float32_t[:,::1] values = task_state.task_features_values()
    cdef np.float32_t[:,::1] uncertainties = task_state.task_features_uncertainties()
    for k in range(k_len):
        permissivity_factor = permissivity_bonus * values[PERMISSIVITY_INDEX, k]
        # probability of success for single step in path ( if that step is 10m long )
        local_prob = permissivity_factor * perceptivity_factor * crowdedness_factor
        cumulative_prob = min(cumulative_prob, local_prob)
    path_prob = cclip(cumulative_prob, 0.05, 0.99)
    # convert from reference step size (10m) to actual size
    distance_equivalent_prob = path_prob ** (k_length / PROB_DISTANCE_REF)
    # shortcut (numerically unstable)
#     cumulative_prob = cumulative_prob * local_prob
#     distance_equivalent_prob = cumulative_prob ** (task_state.resolution / PROB_DISTANCE_REF)
    return distance_equivalent_prob

# Below are value operations  P(S' | Outcome) = T_outcome(S, b, Eps)
# ---------------------------------------------------------------------------------
#
# INTEND - SUCCESS - PERMISSIVITY
cdef np.float32_t INTEND_S_PERMISSIVITY_value_operation(np.float32_t[:] values, np.float32_t[:] uncertainties):
    if values[PERMISSIVITY_INDEX] > 0:
        return 1 - (1 - values[PERMISSIVITY_INDEX]) * 0.9 # if permissivity was low, increase it
    else:
        return 0.1
cdef np.float32_t INTEND_S_PERMISSIVITY_uncertainty_operation(np.float32_t[:] values, np.float32_t[:] uncertainties):
    return uncertainties[PERMISSIVITY_INDEX] * 0.1
# INTEND - FAILURE - PERMISSIVITY
cdef np.float32_t INTEND_F_PERMISSIVITY_value_operation(np.float32_t[:] values, np.float32_t[:] uncertainties):
    return values[PERMISSIVITY_INDEX] * 0.1
cdef np.float32_t INTEND_F_PERMISSIVITY_uncertainty_operation(np.float32_t[:] values, np.float32_t[:] uncertainties):
    return uncertainties[PERMISSIVITY_INDEX] * 0.001
# INTEND - FAILURE - PERCEPTIVITY
cdef np.float32_t INTEND_F_PERCEPTIVITY_value_operation(np.float32_t[:] values, np.float32_t[:] uncertainties):
    return values[PERCEPTIVITY_INDEX] * 0.5
cdef np.float32_t INTEND_F_PERCEPTIVITY_uncertainty_operation(np.float32_t[:] values, np.float32_t[:] uncertainties):
    return uncertainties[PERCEPTIVITY_INDEX] * 0.1

# SAY - SUCCESS - PERMISSIVITY
cdef np.float32_t SAY_S_PERMISSIVITY_value_operation(np.float32_t[:] values, np.float32_t[:] uncertainties):
    if values[PERMISSIVITY_INDEX] > 0:
        return 1 - (1 - values[PERMISSIVITY_INDEX]) * 0.9 # increase to 1
    else:
        return 0.1
cdef np.float32_t SAY_S_PERMISSIVITY_uncertainty_operation(np.float32_t[:] values, np.float32_t[:] uncertainties):
    return uncertainties[PERMISSIVITY_INDEX] * 0.1
# SAY - FAILURE - PERMISSIVITY
cdef np.float32_t SAY_F_PERMISSIVITY_value_operation(np.float32_t[:] values, np.float32_t[:] uncertainties):
    return values[PERMISSIVITY_INDEX] * 0.1  # decrease to 0
cdef np.float32_t SAY_F_PERMISSIVITY_uncertainty_operation(np.float32_t[:] values, np.float32_t[:] uncertainties):
    return uncertainties[PERMISSIVITY_INDEX] * 0.001
# SAY - FAILURE - PERCEPTIVITY
cdef np.float32_t SAY_F_PERCEPTIVITY_value_operation(np.float32_t[:] values, np.float32_t[:] uncertainties):
    return 1.
cdef np.float32_t SAY_F_PERCEPTIVITY_uncertainty_operation(np.float32_t[:] values, np.float32_t[:] uncertainties):
    return uncertainties[PERCEPTIVITY_INDEX] * 0.1

# CRAWL - FAILURE - PERMISSIVITY
cdef np.float32_t CRAWL_F_PERMISSIVITY_value_operation(np.float32_t[:] values, np.float32_t[:] uncertainties):
    return values[PERMISSIVITY_INDEX] * 0.25 # decrease to 0
cdef np.float32_t CRAWL_F_PERMISSIVITY_uncertainty_operation(np.float32_t[:] values, np.float32_t[:] uncertainties):
    return uncertainties[PERMISSIVITY_INDEX] * 0.1

# NUDGE - FAILURE - PERMISSIVITY
cdef np.float32_t NUDGE_F_PERMISSIVITY_value_operation(np.float32_t[:] values, np.float32_t[:] uncertainties):
    # maybe not? the nudge planner could simply be bad in high permissivity cases
    return values[PERMISSIVITY_INDEX] * 0.1 # decrease to 0
cdef np.float32_t NUDGE_F_PERMISSIVITY_uncertainty_operation(np.float32_t[:] values, np.float32_t[:] uncertainties):
    return uncertainties[PERMISSIVITY_INDEX] * 0.001


def apply_value_operations_to_state(update, state, update_stencil):
    return capply_value_operations_to_state(update, state, update_stencil)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef bool capply_value_operations_to_state(CTaskStateUpdate update, CState state,
                                          np.float32_t[:,::1] update_stencil):
    """ Applies diff to entire state (full grids), inplace - expensive """
    cdef bool is_any_changed = False
    cdef int n
    cdef int i
    cdef int j
    cdef int len_i = state.grid_features.grid_width
    cdef int len_j = state.grid_features.grid_height
    cdef np.float32_t update_strenth
    cdef np.float32_t updated_value
    cdef np.float32_t updated_uncertainty
    cdef np.float32_t old_value
    cdef np.float32_t old_uncertainty
    for n in range(N_STATE_FEATURES):
        if update.features_diff.is_changed[n]:
            is_any_changed = True
            break
    if is_any_changed:
        for i in range(len_i):
            for j in range(len_j):
                update_strength = update_stencil[i, j]
                for n in range(N_STATE_FEATURES):
                    if update.features_diff.is_changed[n]:
                        old_value = state.grid_features.values[n, i, j]
                        old_uncertainty = state.grid_features.uncertainties[n, i, j]
                        updated_value = update.features_diff.value_operations[n](
                            state.grid_features.values[:, i, j],
                            state.grid_features.uncertainties[:, i, j]
                        )
                        updated_uncertainty = update.features_diff.uncertainty_operations[n](
                            state.grid_features.values[:, i, j],
                            state.grid_features.uncertainties[:, i, j]
                        )
                        # apply
                        state.grid_features.values[n, i, j] = (
                            update_strength * updated_value + (1 - update_strength) * old_value
                        )
                        state.grid_features.uncertainties[n, i, j] = (
                            update_strength * updated_uncertainty + (1 - update_strength) * old_uncertainty
                        )
    # update the position if it has changed
    if update.is_pos_update:
        is_any_changed = True
        state.pos = update.pos_update
        state.pos_ij = update.pos_ij_update
    return is_any_changed
