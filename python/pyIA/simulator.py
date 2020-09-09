from __future__ import print_function
import copy
from CMap2D import CMap2D
from map2d import gridshow
from matplotlib import pyplot as plt
from matplotlib import patches as patches
import matplotlib
import numpy as np
import os
import pickle
import sys
import time

import pyIA.agents as agents
import pyIA.actions as actions
from pyIA.mdptree import Node

X_DIM = 2 # x, y

# scriptdir = os.path.dirname(os.path.realpath(__file__))
# SCENARIOS_FOLDER = os.path.join(scriptdir, "../scenarios")

class Worldstate(object):
    """ storage class for convenience
    Worlstate contains all possible information to define the state of the simulated world """
    def __init__(self):
        # agent state
        self._agentstate = {
            "agents": [], # list of agent prototypes
            "xystates": [], # [n_agents, 2]
            "uvstates": [],
            "innerstates": [],
        }
        # map state
        self.map = CMap2D()
        # others
        self.derived_state = DerivedWorldstate()
        # TODO: adding signals to the worldstate?
        # TODO: add undertakings to inner_state? (multi-step actions)

    def get_agents(self):
        return self._agentstate["agents"]
    def get_xystates(self):
        return self._agentstate["xystates"]
    def get_uvstates(self):
        return self._agentstate["uvstates"]
    def get_innerstates(self):
        return self._agentstate["innerstates"]

    def copy(self):
        """ non-trivial : agents, xy, uv, innerstates are copied but reference to the map and derived state is shared.
        Due to the fact that typically the map does not get modified in use-cases where copies of the worldstate are needed.
        Warning: Changing the map in a copy has side effects!"""
        copy_ = Worldstate()
        for key in self._agentstate:
            copy_._agentstate[key] = copy.deepcopy(self._agentstate[key])
        copy_.map = self.map
        copy_.derived_state = self.derived_state
        return copy_

    def remove_agent(self, agent_index):
        for key in self._agentstate:
            self._agentstate[key].pop(agent_index)

    def save_agents_to_file(self, filepath):
        with open(filepath,"wb") as f:
            pickle.dump(self._agentstate, f)
        print("Agentstate saved to {}".format(filepath))

    def restore_agents_from_file(self, filepath, silent=False):
        with open(filepath,"rb") as f:
            if sys.version_info.major >= 3:
                agentstate = pickle.load(f, encoding='latin1')
            else:
                agentstate = pickle.load(f)
        for key in agentstate:
            self._agentstate[key] = agentstate[key]
        # switch old agent classes for new agent classes
        for i in range(len(self._agentstate["agents"])):
            agent = self._agentstate["agents"][i]
            self._agentstate["agents"][i] = agents.agent_creator(agent.type())
        if not silent:
            print("Agentstate loaded from {}".format(filepath))



class DerivedWorldstate(object):
    """ convenience storage class for precomputed/cached fields,
    this class is for algorithmic optimization and thus isn't formally in the IA algo"""
    def __init__(self):
        self._map_sdf = None
        self._map_tsdf_2 = None
        self._map_as_closed_obst_vertices = None
        self._map_ij_arrays = None
        self._cache_map = None

    # CORE INTERFACE ------------------
    def copy(self):
        """ needed as otherwise, DerivedWorldstate shared between various states where goals differ
        might interfere with each other """
        copy_ = DerivedWorldstate()
        copy_._map_sdf = self._map_sdf
        copy_._map_tsdf_2 = self._map_tsdf_2
        copy_._map_as_closed_obst_vertices = self._map_as_closed_obst_vertices
        copy_._cache_map = self._cache_map
        return copy_

    def map_sdf(self, worldstate):
        if worldstate.map is not self._cache_map:
            self._on_set_map(worldstate)
        return self._map_sdf

    def map_tsdf_2(self, worldstate):
        if worldstate.map is not self._cache_map:
            self._on_set_map(worldstate)
        return self._map_tsdf_2

    def map_as_closed_obst_vertices(self, worldstate):
        if worldstate.map is not self._cache_map:
            self._on_set_map(worldstate)
        return self._map_as_closed_obst_vertices

    def map_ij_arrays(self, worldstate):
        if worldstate.map is not self._cache_map:
            self._on_set_map(worldstate)
        return self._map_ij_arrays

    def agent_precomputed_ctg(self, agent_index, worldstate):
        goal = worldstate.get_innerstates()[agent_index].goal
        goal_ij = worldstate.map.xy_to_ij(np.array([goal]))[0]
#         return worldstate.map.dijkstra(goal_ij, extra_costs=2-self.map_tsdf_2(worldstate), inv_value=-1, connectedness=16)
        return worldstate.map.dijkstra(
            goal_ij, extra_costs=1/(0.0000001 +self.map_sdf(worldstate)), inv_value=-1, connectedness=16)

    def density_map(self, worldstate, density_radius=2.):
        ijagents = [worldstate.map.xy_to_ij([xy])[0] for xy in worldstate.get_xystates()]
        ii, jj = self.map_ij_arrays(worldstate)
        density_map = np.zeros_like(ii)
        sqr_density_radius = (density_radius / worldstate.map.resolution())**2 
        for ai, aj in ijagents:
            density_map[((ii - ai)**2 + (jj - aj)**2) < sqr_density_radius] += 1
        return density_map

    def smooth_density_map(self, worldstate, density_radius=2.):
        map_ = worldstate.map
        ijagents = [worldstate.map.xy_to_ij([xy])[0] for xy in worldstate.get_xystates()]
        ii, jj = self.map_ij_arrays(worldstate)
        density_map = np.zeros_like(ii, dtype=np.float32)
        sqr_density_radius = (density_radius / map_.resolution())**2 
        for ai, aj in ijagents:
            mask = ((ii - ai)**2 + (jj - aj)**2) < sqr_density_radius
            density_map[mask] += 1 - np.sqrt((ii - ai)**2 + (jj - aj)**2)[mask]/ np.sqrt(sqr_density_radius)
        return density_map

    def inv_density_map(self, worldstate, density_radius=2.):
        map_ = worldstate.map
        ijagents = [worldstate.map.xy_to_ij([xy])[0] for xy in worldstate.get_xystates()]
        ii, jj = self.map_ij_arrays(worldstate)
        density_map = np.zeros_like(ii, dtype=np.float32)
        sqr_density_radius = (density_radius / map_.resolution())**2 
        for ai, aj in ijagents:
            mask = ((ii - ai)**2 + (jj - aj)**2) < sqr_density_radius
            density_map[mask] += 1. / (np.sqrt((ii - ai)**2 + (jj - aj)**2)[mask]*4/np.sqrt(sqr_density_radius) + 0.000001)
        return density_map


    def visibility_map_ij(self, agent_index, worldstate):
        ij = worldstate.map.xy_to_ij([worldstate.get_xystates()[agent_index]])[0]
        visibility = worldstate.map.visibility_map_ij(ij)
        return visibility

    # INTERNAL -------------------------
    def _on_set_map(self, worldstate):
        """ Calculates and caches derived state for obstacle map """
        self._cache_map = worldstate.map
        self._map_sdf = worldstate.map.as_sdf()
        self._map_tsdf_2 = worldstate.map.as_tsdf(2)
        self._map_as_closed_obst_vertices = worldstate.map.as_closed_obst_vertices()
        from pyniel.numpy_tools import indexing
        ij = indexing.as_idx_array(worldstate.map.occupancy(), axis='all')
        self._map_ij_arrays = (ij[...,0], ij[...,1])

    def check_is_up_to_date(self):
        """ compares checksum for latest value updates with current checksums """
        raise NotImplementedError


class Environment(object):
    def __init__(self, args, silent=False):
        self.worldstate = Worldstate()
        map_ = CMap2D(args.map_folder, args.map_name, silent=silent)
        for _ in range(args.map_downsampling_passes):
            map_ = map_.as_coarse_map2d()
        self.worldstate.map = map_
        self.args = args

    # CORE INTERFACE -------------------
    def step(self, actions):
        # high level (ia action consequences) or low level (execute planners)
        pass

    def get_worldstate(self):
        return self.worldstate

    def get_sensor_readings(self, agent_index):
        # ???
        pass

    # INTERNAL INTERFACE ---------------
    def fullstate(self):
        relstate = self.relstate(self.worldstate)
        return (self.worldstate, relstate)

    def relstate(self, worldstate): # DEPRECATED
        n_agents = len(worldstate.get_agents())
        xystates = np.array(worldstate.get_xystates())
        deltas = xystates[:, None, :] - xystates[None, :, :]
        drelstates = np.linalg.norm(deltas, axis=-1)
        threlstates = np.arctan2(deltas[1], deltas[0])
        # TODO: rel uv
        return (drelstates, threlstates)

    def policies(self, fullstate):
        """ runs a policy simulation on each agent,
        returns each agents chosen X_desired, and action """
        worldstate, relstate = fullstate
        for i, _ in enumerate(worldstate.get_agents()):
            agent_i = worldstate.get_agents()[i]
            innerstate_i = worldstate.get_innerstates()[i]
            xystate_i = worldstate.get_xystates()[i]
            relstate_i = relstate[0][i]
            # build mdp tree
            node0 = Node(worldstate, [], [], 0, [])
            mdpt = [node0]
            # iterate through potential actions
#             for a in agent_i.actions:
#                 outcome_distribution = a.old_predict_outcome(i, worldstate)
#                 for outcome, prob in outcome_distribution:
#                     pass


        A = []
        Xd = []
        return (A, Xd) # actions, desired X for each agent

    def enact_policies(self, A, Xd, worldstate):
        """ Once policies are chosen, they can be enacted to advance to a new state.
        Two alternatives, execute_planners, and predict_planners. """
        return predict_planners(A, Xd, worldstate)

    def execute_planners(self, A, Xd, worldstate):
        """ Runs a low level simulation of the agents and their action planners interacting,
        outputs resulting positions (Xr), and whether the planner succeeded  """
        return (Xr, success)

    def predict_planners(self, A, Xd, worldstate):
        """ Tries to directly predict positions (Xr) and success of the planners by 
        using simple interaction models of the planners """
        return (Xr, success)

    # SAMPLING -------------------------
    def x_samples(self, agent, innerstate, xystate, relstate, nudge_stddev=0.01):
        import hexgrid
        """ Generate a graph of X samples starting from an agent's X state """
        # neighbor nodes in hexagonal grid
        neighbors = hexgrid.neighbors(coordinate_system="xy")
        neighbors = np.concatenate((neighbors, hexgrid.neighbors(coordinate_system="xy", hex_orientation="pointy")), axis=0)
        nudge = np.random.normal(0, nudge_stddev, size=neighbors.shape)
        x_samples = xystate + neighbors + nudge
        return x_samples

    # VISUALIZATIONS -------------------
    def plot_map(self):
        gridshow(self.worldstate.map.occupancy())

    def plot_map_contours_ij(self, *args, **kwargs):
        if not args:
            args = ('-,',)
        contours = [self.worldstate.map.xy_to_ij(c) for c in self.worldstate.derived_state.map_as_closed_obst_vertices(self.worldstate)]
        for c in contours:
            # add the first vertice at the end to close the plotted contour
            cplus = np.concatenate((c, c[:1, :]), axis=0)
            plt.plot(cplus[:,0], cplus[:,1], *args, **kwargs)

    def plot_map_contours_xy(self, *args, **kwargs):
        if not args:
            args = ('-,',)
        contours = self.worldstate.derived_state.map_as_closed_obst_vertices(self.worldstate)
        for c in contours:
            # add the first vertice at the end to close the plotted contour
            cplus = np.concatenate((c, c[:1, :]), axis=0)
            plt.plot(cplus[:,0], cplus[:,1], *args, **kwargs)

    def plot_agents_xy(self, figure=None):
        if figure is None:
            figure = plt.gcf()
        patches = []
        worldstate = self.worldstate
        for i, _ in enumerate(worldstate.get_agents()):
            patch = self.get_viz_agent_patch(i)
            figure.gca().add_artist(patch)
            patches.append(patch)
        return patches

    def get_viz_agent_patch(self, agent_index):
        worldstate = self.worldstate
        agent = worldstate.get_agents()[agent_index]
        xy = np.array(worldstate.get_xystates()[agent_index])
        c = self.get_viz_agent_color(agent_index)
        r = agent.radius
        patchcreator = self.get_viz_agent_patchcreator(agent)
        patch = patchcreator((xy[0], xy[1]), r, facecolor=c, edgecolor="black")
        return patch

    def get_viz_agent_color(self, agent_index):
        worldstate = self.worldstate
        GrYlRd = matplotlib.colors.LinearSegmentedColormap.from_list("", ["tomato","yellow","springgreen"])
        c = GrYlRd(worldstate.get_innerstates()[agent_index].permissivity)
        return c

    def get_viz_agent_patchcreator(self, agent):
        if agent.type() == agents.Robot:
            def PatchCreator(xy, r, **kwargs):
                return patches.RegularPolygon(xy, 6, r, **kwargs)
        elif agent.type() == agents.ORCAAgent:
            def PatchCreator(xy, r, **kwargs):
                return patches.RegularPolygon(xy, 4, r, **kwargs)
        else:
            def PatchCreator(xy, r, **kwargs):
                return plt.Circle(xy, r, **kwargs)
        return PatchCreator

    def plot_agents_ij(self):
        worldstate = self.worldstate
        for i, a in enumerate(worldstate.get_agents()):
            xy = np.array(worldstate.get_xystates()[i])
            ij = self.worldstate.map.xy_to_ij([xy])[0]
            GrYlRd = matplotlib.colors.LinearSegmentedColormap.from_list("", ["tomato","yellow","springgreen"])
            c = GrYlRd(worldstate.get_innerstates()[i].permissivity)
            r = a.radius / self.worldstate.map.resolution()
            if a.type() == agents.Robot:
                circle = patches.RegularPolygon((ij[0], ij[1]), 6, r, color=c)
                plt.gcf().gca().add_artist(circle)
#                 marker = "^"
#                 plt.scatter(ij[:1], ij[1:], marker=marker, c=[c])
            else:
                circle = plt.Circle((ij[0], ij[1]), r, color=c)
                plt.gcf().gca().add_artist(circle)

    def plot_goals(self):
        ij_goals = self.worldstate.map.xy_to_ij(np.array([innerstate.goal for innerstate in self.worldstate.get_innerstates()]))
        plt.scatter(ij_goals[:,0], ij_goals[:,1], marker="D", color='y')



    def plot_samples(self):
        pass


    # LOGISTICS --------------------------------------- 
    def populate(self, silent=False):
        """ add agents according to some scenario """
        if self.args.scenario == "":
            pass
        else:
            filepath = os.path.join(self.args.scenario_folder, self.args.scenario+".pickle")
            self.worldstate.restore_agents_from_file(filepath, silent=silent)

