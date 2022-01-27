# form class calculated on graph, based on nodes similarity
# notes:
#   k-component structure of a graph G: number of connected component of a graph:
#   - the best graph is perhaps composed by one single component!!

import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms import community


class GraphModule:
    def __init__(self, tps, be=None, thresh=0.0):
        self.G = nx.DiGraph()
        added = set()
        rows, cols = tps.norm_mem.shape
        for i in range(rows):
            if be:
                li = be.base_decode(tps.le_rows.inverse_transform([i])[0])
            else:
                li = tps.le_rows.inverse_transform([i])[0]
            if li not in added:
                self.G.add_node(li, label="{} ({:.3f})".format(li, tps.state_entropies[li]))
                added.add(li)
            for j in range(cols):
                if tps.norm_mem[i][j] > thresh:
                    if be:
                        lj = be.base_decode(tps.le_cols.inverse_transform([j])[0])
                    else:
                        lj = tps.le_cols.inverse_transform([j])[0]
                    if tps.norm_mem[i][j] == 1.0:
                        self.G.add_edge(li, lj, weight=tps.norm_mem[i][j], label="{:.3f}".format(tps.norm_mem[i][j]), penwidth="2", color="red")
                    else:
                        self.G.add_edge(li, lj, weight=tps.norm_mem[i][j], label="{:.3f}".format(tps.norm_mem[i][j]))

    def draw_graph(self, filename):
        nx.draw(self.G)
        # plt.draw()
        nx.nx_pydot.write_dot(self.G, filename)
        # plt.show()

    def print_values(self):
        print("k_components: ", nx.algorithms.k_components(self.G.to_undirected()))
        print("maximal_independent_set: ", nx.algorithms.maximal_independent_set(self.G.to_undirected()))
        # print("traveling_salesman_problem: ", nx.algorithms.approximation.traveling_salesman_problem(self.G))
        # print("topological: ",list(nx.topological_sort(self.G)))
        # print("common_neighbors (kof,mer): ",list(nx.common_neighbors(self.G.to_undirected(),"kof","mer")))
        # print("flow_hierarchy: ",nx.flow_hierarchy(self.G))
        # print("max clique: ", list(nx.algorithms.approximation.clique.max_clique(self.G.to_undirected())))

    def form_classes(self):
        print("topological_generations: ",[sorted(generation) for generation in nx.topological_generations(self.G)])

    def get_communities(self):
        communities_generator = community.girvan_newman(self.G)
        top_level_communities = next(communities_generator)
        next_level_communities = next(communities_generator)
        print("communities:", sorted(map(sorted, next_level_communities)))

