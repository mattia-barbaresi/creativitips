import networkx as nx
import matplotlib.pyplot as plt

from multil import LayeredNetworkGraph


class MIModule:
    """Implements associative memory between percepts"""

    def __init__(self):
        self.G = nx.Graph()

    def encode(self, *percepts, weight=1.0, decay=0.05):
        """
        Encode associations (Hebb) between percepts
        :param decay:
        :param weight:
        :param percepts list of units for each dimension/sensor/modality
        """
        added = []
        # add weight for each pair
        for n in range(len(percepts)-1):  # for each feature/dimension
            for i in range(len(percepts[n])):  # for each unit in that dimension
                for m in range(n+1,len(percepts)):  # for all others dimensions
                    for j in range(len(percepts[m])):  # for each unit in others dimensions
                        if self.G.has_edge(percepts[n][i], percepts[m][j]):
                            self.G[percepts[n][i]][percepts[m][j]]['weight'] += weight
                        else:
                            self.G.add_edge(percepts[n][i], percepts[m][j], weight=weight)
                        self.G.nodes[percepts[n][i]]["level"] = n
                        self.G.nodes[percepts[m][j]]["level"] = m
                        added.append((percepts[n][i], percepts[m][j]))
        # forgetting
        # for e in self.G.edges():
        #     if e not in added:
        #         self.G[e[0]][e[1]]["weight"] -= decay
        #         if self.G[e[0]][e[1]]["weight"] <= 0:
        #             self.G.remove_edge(e[0],e[1])
        # self.G.remove_nodes_from(list(nx.isolates(self.G)))

    def draw_graph(self):
        # nx.draw_networkx(self.G)
        colm = [["#ff1500","#0000FF"][node[1]["level"]] for node in self.G.nodes(data=True)]
        # pos = nx.spring_layout(self.G, dim=2, scale=2, k=100, seed=779)
        pos = nx.nx_agraph.graphviz_layout(self.G, prog="dot")
        # pos = nx.multipartite_layout(self.G, subset_key="level")
        nx.draw(self.G, pos=pos, node_color=colm, with_labels=True)
        plt.draw()
        plt.show()

        # # 3d spring layout
        #
        # # Extract node and edge positions from the layout
        # node_xyz = np.array([pos[v] for v in sorted(self.G)])
        # edge_xyz = np.array([(pos[u], pos[v]) for u, v in self.G.edges()])
        #
        # # Create the 3D figure
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection="3d")
        #
        # # Plot the nodes - alpha is scaled by "depth" automatically
        # ax.scatter(*node_xyz.T, s=100, ec="w")
        #
        # # Plot the edges
        # for vizedge in edge_xyz:
        #     ax.plot(*vizedge.T, color="tab:gray")
        #
        # for v in sorted(self.G):
        #     ax.text(pos[v][0],pos[v][1],pos[v][2], v)
        #
        # def _format_axes(ax):
        #     """Visualization options for the 3D axes."""
        #     # Turn gridlines off
        #     ax.grid(False)
        #     # Suppress tick labels
        #     for dim in (ax.xaxis, ax.yaxis, ax.zaxis):
        #         dim.set_ticks([])
        #
        # ax.set_axis_off()
        # _format_axes(ax)
        # fig.tight_layout()
        # plt.show()

    def get_associated(self, node, fun=set):
        return fun(self.G.neighbors(node))

    def draw_graph2(self, g1,g2):
        # initialise figure and plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        node_labels = {nn:str(nn) for nn in self.G.nodes}
        LayeredNetworkGraph([g1.G, g2.G], node_labels=node_labels, ax=ax, layout=nx.spring_layout)
        ax.set_axis_off()
        plt.show()
