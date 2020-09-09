from __future__ import print_function
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from pyIA.arguments import parse_sim_args
from pyIA.simulator import Environment
import pyIA.agents as agents
import pyIA.state_estimation as state_estimation
import pyIA.paths as paths
import pyIA.ia_planning as ia_planning

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

FILTER = "rl"

# map should be given as arg

if __name__ == "__main__":
    args = parse_sim_args()
    scenario_files = sorted(os.listdir(args.scenario_folder))
    filtered_scenarios = []
    for file in scenario_files:
        name, ext = os.path.splitext(file)
        if FILTER not in name:
            continue
        if ext != ".pickle":
            continue
        if args.map_name != name[len(FILTER):-1]:
            continue
        filtered_scenarios.append(name)

    print("filtered to {} scenarios in {}".format(len(filtered_scenarios), args.scenario_folder))
    print("------------------------------------------------------------")
    fig, axes = plt.subplots(1, len(filtered_scenarios))
    fig.subplots_adjust(hspace=0.3, wspace=0)
    axes = np.array(axes).flatten()
    fixed_state = None
    for name, ax in zip(filtered_scenarios, axes):
        print(name)
        args.scenario = name
        env = Environment(args, silent=True)
        try:
            env.populate(silent=True)
        except AttributeError as e:
            print("could not populate ", name, "(", e, ")")
            continue
        all_agents = env.worldstate.get_agents()
        if not all_agents:
            print(name, "has no agents!")
            continue
        robot_agent = all_agents[0]
        if not isinstance(robot_agent, agents.Robot):
            print(name, "does not have robot as agent 0!")
        agent_index = 0
        state_e, fixed_state = state_estimation.state_estimate_from_sim_worldstate(
            agent_index, env.worldstate, reuse_state_cache=fixed_state
        )
        plt.sca(ax)
        # agent goals
        xys = env.worldstate.get_xystates()
        inners = env.worldstate.get_innerstates()
        for inner, xy in zip(inners[1:], xys[1:]):
            plt.plot([xy[0], inner.goal[0]], [xy[1], inner.goal[1]], '--', color="Grey")
        # world
        ia_planning.visualize_world(state_e, fixed_state, [], env,
                                    fig=fig, possible_path_variants=[paths.NaivePath()])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        plt.axis('off')
        plt.title("Scene " + name[-1])
    plt.show()
