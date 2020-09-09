from __future__ import print_function
from builtins import input
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
import os

import pyIA.agents as agents
from pyIA.arguments import parse_sim_args
from pyIA.simulator import Environment

AGENT_TYPES = [agents.Robot, agents.Person, agents.ORCAAgent]

class WorldEditorGui(object):
    def __init__(self, env, debug=False):
        self.debug = debug
        # state variables
        self.env = env
        # plotting variables
        self.fig = None
        self.patches = None
        self.deadpatches = None
        self.selected = None
        self.agent_creation_mode = [False, [0, 0], 0]  # is_in_creation_mode, coords, agent_type

    def run(self):
        if self.fig is None:
            self.fig = plt.figure()
        self.env.plot_map_contours_xy('--k')
        self.patches = self.env.plot_agents_xy(figure=self.fig)
        self.gp, = plt.plot([], [], '+--', color="orange")
        # patches for agent_creation_mode
        self.agent_creation_patches = []
        self.main_axes = self.fig.gca()
        for at in AGENT_TYPES:
            tempagent = at()
            patch = self.env.get_viz_agent_patchcreator(tempagent)(
                (0, 0), tempagent.radius * 10, facecolor="white", edgecolor="black")
            patch.set_alpha(0.)
            self.main_axes.add_artist(patch)
            self.agent_creation_patches.append(patch)
        self.deadpatches = []
        plt.axis('equal')
        # buttons
        ax1 = plt.axes([0.75, 0.20, 0.1, 0.075])
        self.bperm = Button(ax1, 'permissivity')
        self.bperm.color = self.bperm.hovercolor
        self.bperm.on_clicked(self.onpermissivity)
        ax2 = plt.axes([0.75, 0.30, 0.1, 0.075])
        self.bperc = Button(ax2, 'perceptivity')
        self.bperc.color = self.bperc.hovercolor
        self.bperc.on_clicked(self.onperceptivity)
        # connect
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        plt.show()

    def onpermissivity(self, event):
        if self.selected is not None:
            if self.debug:
                print("Cycling permissivity")
            self.env.worldstate.get_innerstates()[self.selected].permissivity += 0.25
            if self.env.worldstate.get_innerstates()[self.selected].permissivity > 1:
                self.env.worldstate.get_innerstates()[self.selected].permissivity = 0
            self.update_figure()

    def onperceptivity(self, event):
        if self.selected is not None:
            if self.debug:
                print("Cycling perceptivity")
            self.env.worldstate.get_innerstates()[self.selected].perceptivity += 0.25
            if self.env.worldstate.get_innerstates()[self.selected].perceptivity > 1:
                self.env.worldstate.get_innerstates()[self.selected].perceptivity = 0
            self.update_figure()

    def onclick(self, event):
        if event.inaxes is not self.main_axes:
            return
        is_leftclick = event.button == 1
        is_middleclick = event.button == 2
        is_rightclick = event.button == 3
        ix, iy = event.xdata, event.ydata
        if ix is None or iy is None:
            if self.debug:
                print("invalid click")
                print(event)
            return
        # Find if an existing node is under the click
        clicked = None
        for i, xy in enumerate(self.env.worldstate.get_xystates()):
            radius = self.env.worldstate.get_agents()[i].radius
            if np.linalg.norm(xy - np.array([ix, iy])) <= radius:
                clicked = i
                break

        if self.agent_creation_mode[0]:
            # left to confirm
            if is_leftclick:
                if self.debug:
                    print("Exiting agent creation mode")
                innerstate = agents.Innerstate()
                agent = AGENT_TYPES[self.agent_creation_mode[2] % len(AGENT_TYPES)]()
                self.env.worldstate.get_agents().append(agent)
                self.env.worldstate.get_xystates().append(self.agent_creation_mode[1])
                self.env.worldstate.get_uvstates().append([0., 0.])
                self.env.worldstate.get_innerstates().append(innerstate)
                agent_index = len(self.env.worldstate.get_agents()) - 1
                # patches
                patch = self.env.get_viz_agent_patch(agent_index)
                self.main_axes.add_artist(patch)
                self.patches.append(patch)
                # select agent
                self.selected = agent_index
                self.agent_creation_mode[0] = False
            elif is_middleclick:
                self.agent_creation_mode[2] += 1
            elif is_rightclick:
                self.agent_creation_mode[0] = False
        else:
            if is_leftclick:
                if clicked is not None:
                    if self.selected is not None:
                        if self.selected == clicked:
                            if self.debug:
                                print("Unselecting agent")
                            self.selected = None
                        else:
                            if self.debug:
                                print("Selecting agent")
                            self.selected = clicked
                    else:
                        if self.debug:
                            print("Selecting agent")
                        self.selected = clicked
                else:
                    if self.debug:
                        print("Entering Agent Creation Mode")
                    self.agent_creation_mode[0] = True
                    self.agent_creation_mode[1] = [ix, iy]
                    self.agent_creation_mode[2] = 0
                    self.selected = None
            elif is_rightclick:
                if self.selected is not None:
                    if self.debug:
                        print("Unselecting")
                    self.selected = None
                else:
                    if clicked is not None:
                        if self.debug:
                            print("Removing agent")
                        self.env.worldstate.remove_agent(clicked)
                        deadpatch = self.patches.pop(clicked)
                        deadpatch.remove()
#                         self.deadpatches.append(deadpatch)
            elif is_middleclick:
                if self.selected is not None:
                    if self.debug:
                        print("Setting goal for agent")
                    self.env.worldstate.get_innerstates()[self.selected].goal = [ix, iy]

        if self.debug:
            print(event)
            print("selected:", self.selected)
            print("clicked:", clicked)
            print("agents:", self.env.worldstate.get_agents())
            print("agent_creation_mode:", self.agent_creation_mode)
            print("------------------------------")

        self.update_figure()

        if False:
            self.fig.canvas.mpl_disconnect(self.cid)

    def update_figure(self):
        # visualize
        # show agents
        for i in range(len(self.patches)):
            self.patches[i].center = (
                self.env.worldstate.get_xystates()[i][0], self.env.worldstate.get_xystates()[i][1])
            self.patches[i].set_facecolor(self.env.get_viz_agent_color(i))
#         for i in range(len(self.deadpatches)):
#             self.deadpatches[i].set_alpha(0.)
        # display goal
        if self.selected is not None:
            X = [self.env.worldstate.get_xystates()[self.selected][0],
                 self.env.worldstate.get_innerstates()[self.selected].goal[0]]
            Y = [self.env.worldstate.get_xystates()[self.selected][1],
                 self.env.worldstate.get_innerstates()[self.selected].goal[1]]
        else:
            X = []
            Y = []
        self.gp.set_data(X, Y)
        # visualize agent being created
        if self.agent_creation_mode[0]:
            for i in range(len(AGENT_TYPES)):
                p = self.agent_creation_patches[i]
                if self.agent_creation_mode[2] % len(AGENT_TYPES) == i:
                    p.xy = self.agent_creation_mode[1][0], self.agent_creation_mode[1][1]
                    p.center = p.xy  # in case it is a circle
                    p.set_alpha(1.)
                else:
                    p.set_alpha(0.)
        else:
            for p in self.agent_creation_patches:
                p.set_alpha(0.)
        # visualize selected agent
        if self.selected is not None:
            for i in range(len(self.patches)):
                if i == self.selected:
                    self.patches[i].set_linewidth(2.)
                else:
                    self.patches[i].set_linewidth(1.)
        else:
            for p in self.patches:
                p.set_linewidth(1.)
        # grey out button if not in selection mode
        if self.selected is not None:
            self.bperm.label.set_text(self.env.worldstate.get_innerstates()[self.selected].permissivity)
            self.bperm.color = "oldlace"
        else:
            self.bperm.color = self.bperm.hovercolor

        # figure title
        if self.selected is not None:
            self.main_axes.set_title("""selected mode: middle click to set agent goal, left click to create new agent,
                               left click on agent to toggle selection, right click to unselect""")
        else:
            self.main_axes.set_title(
                "left click to create agent, right click to delete, click agent to select")
        if self.agent_creation_mode[0]:
            self.main_axes.set_title("""agent creation mode: left click to confirm,
                                     middle click to cycle agent type, right click to cancel""")
        # update
        self.fig.canvas.draw()


if __name__ == "__main__":
    args = parse_sim_args()
    env = Environment(args)
    env.populate()

    weg = WorldEditorGui(env, debug=args.debug)
    weg.run()

    yesno = input("Save worldstate to file? [Y/n]")
    if yesno not in ["n", "N", "no", "NO"]:
        name = input("Filename for worldstate: \n>>")
        filepath = os.path.join(args.scenario_folder, name + ".pickle")
        env.worldstate.save_agents_to_file(filepath)
