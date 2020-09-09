class Node(object):
    def __init__(self, worldstate, neighbors, edges, rank, values):
        self.worldstate = worldstate
        self.neighbors = neighbors
        self.edges = edges
        self.rank = rank
        self.values = values
