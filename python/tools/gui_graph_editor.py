import os
from builtins import input
import matplotlib.pyplot as plt
from pyniel.pyplot_tools.graph_creator_gui import GraphCreatorGui

from pyIA.arguments import parse_sim_args
from pyIA.simulator import Environment


args = parse_sim_args()
env = Environment(args)
env.populate()


# plot map
worldstate = env.get_worldstate()
fig = plt.figure("map")
contours = worldstate.map.as_closed_obst_vertices()
worldstate.map.plot_contours(contours, '-k')
plt.axis('equal')

gcg = GraphCreatorGui(figure=fig, debug=args.debug)
# load graph if exists
filedir = os.path.expanduser(os.path.expandvars(args.map_folder))
filepath = os.path.join(filedir, args.map_name + "_graph.pickle")
if os.path.isfile(filepath):
    print("Previous graph for map found. Loading...")
    gcg.graph.restore_from_file(filepath)
gcg.run()

graph = gcg.graph

# change node ids
graph.reassign_ids()

print(str(graph))

yesno = input("Save graph to file? [Y/n]")
if yesno not in ["n", "N", "no", "NO", "No"]:
    if os.path.isfile(filepath):
        overwrite = input("Graph already exists for current map. Overwrite? [y/N]")
        if overwrite not in ["y", "Y", "yes", "YES", "Yes"]:
            i = 1
            while True:
                filepath = os.path.join(filedir, args.map_name + "_graph({}).pickle".format(i))
                if os.path.isfile(filepath):
                    i += 1
                    continue
                else:
                    break
    graph.save_to_file(filepath)
