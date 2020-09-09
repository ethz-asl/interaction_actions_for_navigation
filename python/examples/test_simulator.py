from timeit import default_timer as timer
import matplotlib.pyplot as plt
import matplotlib
from pyniel.pyplot_tools.realtime import plotpause, plotshow
from pyniel.pyplot_tools.windows import movefigure
from pyniel.pyplot_tools.colormaps import Greys_c10
from CMap2D import path_from_dijkstra_field

from pyIA.arguments import parse_sim_args
from pyIA.simulator import Environment, Worldstate

args = parse_sim_args()
env = Environment(args)

# Add agents and precompute dijkstras
tic=timer()
env.populate()
toc=timer()
print("Environment populated. ({:.2f}s)".format(toc-tic))

# Visuals
# fig = plt.figure("situation")
# movefigure(fig, (100, 100))
# fig = plt.figure("dijkstra")
# movefigure(fig, (500, 100))
# fig = plt.figure("density")
# movefigure(fig, (100, 500))
try:
    for i in range(100):
        S_w = env.fullstate()
        actions = env.policies(S_w)

        worldstate, relstate = S_w

        plt.ion()
        plt.figure("situation")
        plt.cla()
        # plot map and agents
        env.plot_map()
        env.plot_agents_ij()
        # plot samples
        x_samples = env.x_samples(worldstate.get_agents()[0],
                worldstate.get_innerstates()[0],
                worldstate.get_xystates()[0],
                relstate[0])
        ij_samples = env.worldstate.map.xy_to_ij(x_samples)
    #     plt.scatter(ij_samples[:,0], ij_samples[:,1], marker="o", color='g')
        # plot goals
        env.plot_goals()

        # example of value predicition heuristics for worldstates
        # dijkstra-distance
        plt.figure("dijkstra")
        plt.cla()
        cached_ = env.worldstate.derived_state.agent_precomputed_ctg(0, env.worldstate)
        gridshow(cached_, cmap=Greys_c10)
        path, jumps = path_from_dijkstra_field(cached_, [0,0], connectedness=32)
        plt.plot(path[:,0], path[:,1], ':or', mec='red', linewidth=0, mfc=[0,0,0,0])
        # avg-velocity-map-estimation
        plt.figure("density")
        plt.cla()
        density = env.worldstate.derived_state.smooth_density_map(env.worldstate, 4.)
        gridshow(density, )
        env.plot_map_contours_ij('k-,')
        # ...

        t_horizon=1.
        # run action simulation for agent 1 (should be an ORCA agent)
        inputs = worldstate.get_agents()[2].actions[0].worldstate_to_inputs(worldstate)
    #     worldstate.get_agents()[1].actions[0].planner(inputs, t_horizon=t_horizon, DEBUG=True)

        # Run outcomes and update simulation
        outcomes, prob = worldstate.get_agents()[2].actions[0].old_predict_outcome(1, worldstate, t_horizon)
        new_S_w = outcomes[0]

        plotpause(0.01)
        env.worldstate = new_S_w

        if np.linalg.norm(np.array(new_S_w.get_xystates()[0]) - np.array(new_S_w.get_innerstates()[0].goal)) < 0.05:
            break
except KeyboardInterrupt:
    print("Simulation stopped. (^C again to quit)")
    pass

plotshow()

