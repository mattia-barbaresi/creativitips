import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection


class MultiIntegration:
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
        for n in range(len(percepts) - 1):  # for each feature/dimension
            for i in range(len(percepts[n])):  # for each unit in that dimension
                for m in range(n + 1, len(percepts)):  # for all others dimensions
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
        colm = [["#ff1500", "#0000FF"][node[1]["level"]] for node in self.G.nodes(data=True)]
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

    def draw_graph2(self, g1, g2):
        # initialise figure and plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        node_labels = {nn: str(nn) for nn in self.G.nodes}
        LayeredNetworkGraph([g1.G, g2.G], node_labels=node_labels, ax=ax, layout=nx.spring_layout)
        ax.set_axis_off()
        plt.show()


class LayeredNetworkGraph(object):
    """
    Plot multi-graphs in 3D.
    """

    def __init__(self, graphs, node_labels=None, layout=nx.spring_layout, ax=None):
        """Given an ordered list of graphs [g1, g2, ..., gn] that represent
        different layers in a multi-layer network, plot the network in
        3D with the different layers separated along the z-axis.

        Within a layer, the corresponding graph defines the connectivity.
        Between layers, nodes in subsequent layers are connected if
        they have the same node ID.

        Arguments:
        ----------
        graphs : list of networkx.Graph objects
            List of graphs, one for each layer.

        node_labels : dict node ID : str label or None (default None)
            Dictionary mapping nodes to labels.
            If None is provided, nodes are not labelled.

        layout_func : function handle (default networkx.spring_layout)
            Function used to compute the layout.

        ax : mpl_toolkits.mplot3d.Axes3d instance or None (default None)
            The axis to plot to. If None is given, a new figure and a new axis are created.

        """

        # book-keeping
        self.graphs = graphs
        self.total_layers = len(graphs)

        self.node_labels = node_labels
        self.layout = layout

        if ax:
            self.ax = ax
        else:
            fig = plt.figure()
            self.ax = fig.add_subplot(111, projection='3d')

        # create internal representation of nodes and edges
        self.get_nodes()
        self.get_edges_within_layers()
        self.get_edges_between_layers()

        # compute layout and plot
        self.get_node_positions()
        self.draw()

    def get_nodes(self):
        """Construct an internal representation of nodes with the format (node ID, layer)."""
        self.nodes = []
        for z, g in enumerate(self.graphs):
            self.nodes.extend([(node, z) for node in g.nodes()])

    def get_edges_within_layers(self):
        """Remap edges in the individual layers to the internal representations of the node IDs."""
        self.edges_within_layers = []
        for z, g in enumerate(self.graphs):
            self.edges_within_layers.extend([((source, z), (target, z)) for source, target in g.edges()])

    def get_edges_between_layers(self):
        """Determine edges between layers. Nodes in subsequent layers are
        thought to be connected if they have the same ID."""
        self.edges_between_layers = []
        for z1, g in enumerate(self.graphs[:-1]):
            z2 = z1 + 1
            h = self.graphs[z2]
            shared_nodes = set(g.nodes()) & set(h.nodes())
            self.edges_between_layers.extend([((node, z1), (node, z2)) for node in shared_nodes])

    def get_node_positions(self, *args, **kwargs):
        """Get the node positions in the layered layout."""
        # What we would like to do, is apply the layout function to a combined, layered network.
        # However, networkx layout functions are not implemented for the multi-dimensional case.
        # Futhermore, even if there was such a layout function, there probably would be no straightforward way to
        # specify the planarity requirement for nodes within a layer.
        # Therefor, we compute the layout for the full network in 2D, and then apply the
        # positions to the nodes in all planes.
        # For a force-directed layout, this will approximately do the right thing.
        # TODO: implement FR in 3D with layer constraints.

        composition = self.graphs[0]
        for h in self.graphs[1:]:
            composition = nx.compose(composition, h)

        pos = self.layout(composition, *args, **kwargs)

        self.node_positions = dict()
        for z, g in enumerate(self.graphs):
            self.node_positions.update({(node, z): (*pos[node], z) for node in g.nodes()})

    def draw_nodes(self, nodes, *args, **kwargs):
        x, y, z = zip(*[self.node_positions[node] for node in nodes])
        self.ax.scatter(x, y, z, *args, **kwargs)

    def draw_edges(self, edges, *args, **kwargs):
        segments = [(self.node_positions[source], self.node_positions[target]) for source, target in edges]
        line_collection = Line3DCollection(segments, *args, **kwargs)
        self.ax.add_collection3d(line_collection)

    def get_extent(self, pad=0.1):
        xyz = np.array(list(self.node_positions.values()))
        xmin, ymin, _ = np.min(xyz, axis=0)
        xmax, ymax, _ = np.max(xyz, axis=0)
        dx = xmax - xmin
        dy = ymax - ymin
        return (xmin - pad * dx, xmax + pad * dx), \
               (ymin - pad * dy, ymax + pad * dy)

    def draw_plane(self, z, *args, **kwargs):
        (xmin, xmax), (ymin, ymax) = self.get_extent(pad=0.1)
        u = np.linspace(xmin, xmax, 10)
        v = np.linspace(ymin, ymax, 10)
        U, V = np.meshgrid(u, v)
        W = z * np.ones_like(U)
        self.ax.plot_surface(U, V, W, *args, **kwargs)

    def draw_node_labels(self, node_labels, *args, **kwargs):
        for node, z in self.nodes:
            if node in node_labels:
                self.ax.text(*self.node_positions[(node, z)], node_labels[node], *args, **kwargs)

    def draw(self):

        self.draw_edges(self.edges_within_layers, color='k', alpha=0.3, linestyle='-', zorder=2)
        self.draw_edges(self.edges_between_layers, color='k', alpha=0.3, linestyle='--', zorder=2)

        for z in range(self.total_layers):
            self.draw_plane(z, alpha=0.2, zorder=1)
            self.draw_nodes([node for node in self.nodes if node[1] == z], s=300, zorder=3)

        if self.node_labels:
            self.draw_node_labels(self.node_labels,
                                  horizontalalignment='center',
                                  verticalalignment='center',
                                  zorder=100)


if __name__ == '__main__':
    # define graphs
    n = 5
    g = nx.erdos_renyi_graph(4 * n, p=0.1)
    h = nx.erdos_renyi_graph(3 * n, p=0.2)
    i = nx.erdos_renyi_graph(2 * n, p=0.4)

    node_labels = {nn: str(nn) for nn in range(4 * n)}

    # initialise figure and plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    LayeredNetworkGraph([g, h, i], node_labels=node_labels, ax=ax, layout=nx.spring_layout)
    ax.set_axis_off()
    plt.show()
