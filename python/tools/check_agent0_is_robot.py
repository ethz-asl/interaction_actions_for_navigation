from __future__ import print_function
import os

from pyIA.arguments import parse_sim_args
from pyIA.simulator import Environment
import pyIA.agents as agents


if __name__ == "__main__":
    args = parse_sim_args()
    scenario_files = sorted(os.listdir(args.scenario_folder))
    print("checking for {} scenarios in {}".format(len(scenario_files), args.scenario_folder))
    print("------------------------------------------------------------")
    for file in scenario_files:
        name, ext = os.path.splitext(file)
        if ext == ".pickle":
            args.scenario = name
            env = Environment(args, silent=True)
            try:
                env.populate(silent=True)
            except AttributeError as e:
                print("could not populate ", name, "(", e, ")")
                continue
            if not env.worldstate.get_agents():
                print(name, "has no agents!")
                continue
            if not isinstance(env.worldstate.get_agents()[0], agents.Robot):
                print(name, "does not have robot as agent 0!")
                continue
            print(name, len(env.worldstate.get_agents()), "agents, ok.")
