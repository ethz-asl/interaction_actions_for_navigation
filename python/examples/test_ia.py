from __future__ import print_function
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import traceback

from pyIA import ia_planning
from pyIA.arguments import parse_sim_args
from pyIA.simulator import Environment
from pyIA import state_estimation

import numpy as np
np.random.seed(1)

args = parse_sim_args()
env = Environment(args)
env.populate()

agent_index = 0
worldstate = env.get_worldstate()
possible_actions = worldstate.get_agents()[agent_index].actions

# ----------------- PROCESSING -------------------------------------------------------------------
tic = timer()
state_e, fixed_state = state_estimation.state_estimate_from_sim_worldstate(
    agent_index, worldstate
)
toc = timer()
print("State Estimation: {:.1f}s".format(toc - tic))

tic = timer()
try:
    pruned_stochastic_tree = ia_planning.StochasticTree()
    optimistic_sequence = []
    firstfailure_sequence = []
    byproducts = {}
    pruned_stochastic_tree, optimistic_sequence, firstfailure_sequence = ia_planning.plan(
        state_e,
        fixed_state,
        possible_actions,
        n_trials=args.n_trials,
        max_depth=args.max_depth,
        BYPRODUCTS=byproducts,
    )
except:
    traceback.print_exc()
toc = timer()
print("Planning: {:.1f}s ({} trials)".format(toc - tic, args.n_trials))
# ---------------- PLOTTING ----------------------------------------------------------------------
if not args.plotless:
    try:
        ia_planning.visualize_planning_process(
            state_e,
            fixed_state,
            possible_actions,
            pruned_stochastic_tree,
            optimistic_sequence,
            firstfailure_sequence,
            byproducts,
            env=env,
        )
        plt.show()
    except:
        traceback.print_exc()
