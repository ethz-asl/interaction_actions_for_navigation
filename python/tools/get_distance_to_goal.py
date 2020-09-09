from __future__ import print_function
import os
import numpy as np

from pyIA.arguments import parse_sim_args
from pyIA.simulator import Environment
import pyIA.agents as agents
import pyIA.state_estimation as state_estimation
import pyIA.paths as paths

FILTER = "iros"

if __name__ == "__main__":
    args = parse_sim_args()
    scenario_files = sorted(os.listdir(args.scenario_folder))
    print("checking for {} scenarios in {}".format(len(scenario_files), args.scenario_folder))
    print("------------------------------------------------------------")
    for file in scenario_files:
        name, ext = os.path.splitext(file)
        if FILTER not in name:
            continue
        if ext == ".pickle":
            args.scenario = name
            if "asl_office_j" in name:
                args.map_name = "asl_office_j"
            elif "unity_scene_map" in name:
                args.map_name = "unity_scene_map"
            else:
                args.map_name = "asl"
            env = Environment(args, silent=True)
            try:
                env.populate(silent=True)
            except AttributeError as e:
                print("could not populate ", name, "(", e, ")")
                continue
            if not env.worldstate.get_agents():
                print(name, "has no agents!")
                continue
            robot_agent = env.worldstate.get_agents()[0]
            if not isinstance(robot_agent, agents.Robot):
                print(name, "does not have robot as agent 0!")
            agent_index = 0
            state_e, fixed_state = state_estimation.state_estimate_from_sim_worldstate(
                agent_index, env.worldstate
            )
            path_xy = fixed_state.derived_state.path_to_goal(state_e, fixed_state, paths.NaivePath())
            length = 0
            print(name)
            try:
                dxy = np.diff(path_xy, axis=0)
            except:
                import pyIA.ia_planning as ia_planning
                import matplotlib.pyplot as plt
                plt.figure()
                ia_planning.visualize_state_features(state_e, fixed_state, hide_uncertain=False)
                ia_planning.visualize_traversabilities(state_e, fixed_state, [paths.NaivePath()])
                ia_planning.visualize_dijkstra_fields(state_e, fixed_state, [paths.NaivePath()])
                plt.show()
                print(path_xy)
            length = np.sum(np.linalg.norm(dxy, axis=-1))
            print("  manhattan dist: ", length)
            dxy = np.diff(path_xy[::5], axis=0)
            length = np.sum(np.linalg.norm(dxy, axis=-1))
            print("  5-sample dist: ", length)
            dxy = np.diff(path_xy[::10], axis=0)
            length = np.sum(np.linalg.norm(dxy, axis=-1))
            print("  10-sample dist: ", length)
