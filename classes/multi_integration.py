import networkx as nx
import matplotlib.pyplot as plt


class MIModule:
    """Implements associative memory between percepts"""

    def __init__(self):
        self.G = nx.Graph()

    def encode(self, *percepts):
        """
        Encode associations (Hebb) between percepts
        :param percepts: list of list of units for each dimension/sensor/modality
        """
        # add weight for each pair
        for n in range(len(percepts)-1):  # for each feature/dimension
            for i in range(len(percepts[n])):  # for each unit in that dimension
                for m in range(n,len(percepts)):  # for all others dimensions
                    for j in range(len(percepts[m])):  # for each unit in others dimensions
                        if self.G.has_edge(percepts[i], percepts[j]):
                            self.G[i][j]['weight'] += 1
                        else:
                            self.G.add_edge(percepts[i], percepts[j], weight=1)

    def draw_graph(self):
        nx.draw(self.G)
        plt.draw()
