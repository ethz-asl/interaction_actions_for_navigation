import pyIA.actions as actions

def agent_creator(agent_type):
    """ Creates an agent based on a type string """
#     print("creating agent of type {}".format(str(agent_type)))
    if "Robot" in str(agent_type):
        return Robot()
    if "Person" in str(agent_type):
        return Person()
    if "ORCAAgent" in str(agent_type):
        return ORCAAgent()
    else:
        raise ValueError("agent type {} does not exist".format(str(agent_type)))

## Agents
class Agent(object):
    """ Prototype for agent class """
    def type(self):
        return type(self)


class Robot(Agent):
    def __init__(self):
        self.radius = 0.3
        self.actions = [
                actions.Intend(),
                actions.Say(),
#                 actions.Crawl(),
                actions.Nudge(),
                ]

class Person(Agent):
    def __init__(self):
        self.radius = 0.3
        self.actions = [
#                 actions.Disturb(), # targeted
#                 actions.Intend(),
#                 actions.Loiter(),
                ]

# NIY or Deprecated ------------------------------
class MovableObstacle(Agent):
    def __init__(self):
        self.is_passive = True

class ORCAAgent(Agent):
    """ A cylindrical entity which moves around according to simple ORCA rules """
    def __init__(self):
        self.actions = [
                actions.Intend()
                ]
        self.radius = 0.3
# ------------------------------------------------

## inner states
class Innerstate(object):
    def __init__(self):
        self.patience = 1
        self.permissivity = 0.5
        self.perceptivity = 0.9
        self.goal = [0., 0.] # TODO: several goals with preference?
        self.undertaking = None


