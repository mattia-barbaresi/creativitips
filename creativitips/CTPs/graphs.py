import os
from graphviz import Digraph
from matplotlib import pyplot as plt
import networkx as nx
from networkx.algorithms import community

import utils
import const


class TPsGraph:
    """Store a Digraph G to be generalized, to discover form classes.
     Calling #generalize() results in the DiGraph GG"""
    def __init__(self, tps=None, graph=None, thresh=0.0):
        self.p_index = 0
        self.fc = dict()
        self.G = nx.DiGraph()
        self.GG = nx.DiGraph()
        # create graph
        if tps:
            added = set()
            rows, cols = tps.norm_mem.shape
            for i in range(rows):
                li = tps.le_rows.inverse_transform([i])[0]
                if li not in added:
                    # self.G.add_node(li, label="{} ({:.3f})".format(li, tps.state_entropies[li]))
                    self.G.add_node(li, label="{}".format(li))
                    added.add(li)
                for j in range(cols):
                    if tps.norm_mem[i][j] > thresh:
                        lj = tps.le_cols.inverse_transform([j])[0]
                        if tps.norm_mem[i][j] == 1.0:
                            self.G.add_edge(li, lj, weight=tps.norm_mem[i][j],
                                            label="{:.3f}".format(tps.norm_mem[i][j]), penwidth="2", color="red")
                        else:
                            self.G.add_edge(li, lj, weight=tps.norm_mem[i][j], label="{:.3f}".format(tps.norm_mem[i][j]))
        elif graph:
            self.G = graph.copy()

    def draw_g_graph(self, filename):
        nx.draw_networkx(self.GG)
        # nx.draw_networkx(self.GG, pos=nx.nx_pydot.pydot_layout(self.GG, prog="dot"))
        plt.draw()
        # nx.nx_pydot.write_dot(self.GG, filename + ".dot")
        plt.savefig(filename + ".pdf")
        # plt.show()

    def cf_by_sim_rank(self, tsh=0.5):
        # sim class using ingoing
        inw = nx.algorithms.similarity.simrank_similarity(self.G, importance_factor=0.9)
        # sim class using outgoing
        otw = nx.algorithms.similarity.simrank_similarity(self.G.reverse(), importance_factor=0.9)
        cl_form = set()
        for k, v in inw.items():
            tt1 = set(k2 for k2, v2 in v.items() if v2 > tsh)
            tt2 = set(k2 for k2, v2 in otw[k].items() if v2 > tsh)
            # tt = tuple((set(tt1).intersection(tt2)).difference(("START","END")))
            tt = tuple(sorted(tt1.intersection(tt2)))
            if len(tt) > 0:
                cl_form.add(tt)
        return sorted(cl_form)

    def generalize(self, dir_name, gens, render=False):
        inverse_d = dict()
        for i, x in enumerate(self.cf_by_sim_rank()):
            self.fc[x] = i
            inverse_d[i] = x
        paths_to_add = []
        for path in gens:
            paths_to_add.append([(self.get_class_from_node(nd), nd) for nd in path])
        for pta in paths_to_add:
            for i in range(len(pta) - 1):
                if self.GG.has_edge(pta[i][0], pta[i+1][0]):
                    # self.GG.edges[pta[i][0], pta[i+1][0]]["weight"] += self.G[pta[i][1]][pta[i+1][1]]["weight"]
                    self.GG.edges[pta[i][0], pta[i+1][0]]["weight"] += 1.0
                else:
                    self.GG.add_edge(pta[i][0], pta[i+1][0], weight=1.0)
                self.GG.nodes[pta[i][0]]["label"] = "P" + str(pta[i][0])
                self.GG.nodes[pta[i][0]]["words"] = const.GRAPH_SEP.join([x for x in inverse_d[pta[i][0]]])
                self.GG.nodes[pta[i+1][0]]["label"] = "P" + str(pta[i+1][0])
                self.GG.nodes[pta[i+1][0]]["words"] = const.GRAPH_SEP.join([x for x in inverse_d[pta[i+1][0]]])
        # normalize edges
        for n in self.GG.nodes:
            tot = sum([ew[2] for ew in self.GG.edges(n,data="weight")])
            for u,v,ww in self.GG.edges(n,data="weight"):
                self.GG[u][v]["weight"] = ww/tot

        plot_gra_from_nx(self.GG, filename=dir_name + "ggraph", render=render)

    def generate_sequences(self, rand_gen, n_seq=20, min_len=100):
        res = []
        sn = self.get_class_from_node("START")
        if sn != -1:
            for _ in range(n_seq):
                seq = []
                next = ""
                # start nodes
                sn = self.get_class_from_node("START")
                itr = 0
                while next != "END" and sn != -1 and itr <= min_len:
                    # get successors list and values
                    keys = list(self.GG.successors(sn))
                    if len(keys) == 0:
                        # no successors
                        break
                    values = [self.GG[sn][x]["weight"] for x in keys]
                    next_c = keys[utils.mc_choice(rand_gen, values)]
                    next = rand_gen.choice(self.GG.nodes[next_c]["words"].split(const.GRAPH_SEP))
                    if next != "END":
                        seq.append(next)
                    itr += 1
                    sn = next_c
                res.append(" ".join(seq))
        else:
            print("No START found!")
        return res

    def get_class_from_node(self, node_name):
        for cl, l in self.fc.items():
            if node_name in cl:
                return l
        return -1

    def get_topological_generations(self):
        cls = nx.topological_generations(self.G)
        print("topological_generations: ",[sorted(gen) for gen in cls])

    def get_communities(self):
        communities_generator = community.girvan_newman(self.G)
        # top_level_communities = next(communities_generator)
        next_level_communities = next(communities_generator)
        print("communities:", sorted(map(sorted, next_level_communities)))

    def print_values(self):
        print("k_components: ", nx.algorithms.k_components(self.G.to_undirected()))
        # print("maximal_independent_set: ", nx.algorithms.maximal_independent_set(self.G.to_undirected()))
        # print("dominating_set: ", nx.algorithms.dominating.dominating_set(self.G))
        # print("flow_hierarchy: ", nx.algorithms.hierarchy.flow_hierarchy(self.G))
        # d_graph, d_nodes = nx.algorithms.summarization.dedensify(self.G,3)
        # nx.nx_pydot.write_dot(d_graph, "dedensified.dot")
        # print("betweenness_centrality: ", nx.algorithms.centrality.betweenness_centrality(self.G))
        # print("traveling_salesman_problem: ", nx.algorithms.approximation.traveling_salesman_problem(self.G))
        # print("topological: ",list(nx.topological_sort(self.G)))
        # print("common_neighbors (kof,mer): ",list(nx.common_neighbors(self.G.to_undirected(),"kof","mer")))
        # print("flow_hierarchy: ",nx.flow_hierarchy(self.G))
        # print("max clique: ", list(nx.algorithms.approximation.clique.max_clique(self.G.to_undirected())))


def plot_gra_from_nx(graph, filename="", render=False):
    print("Plotting ggraph..")
    gra = Digraph()  # comment='Normalized TPS'
    for li in graph.nodes():
        gra.node(str(li), label=graph.nodes[li]["words"])
    for x,y,attr in graph.edges(data=True):
        p = attr["weight"]
        gra.edge(str(x), str(y), label="{:.3f}".format(p), p=str(p), u="-1", v="0", penwidth=str(3 * p))
    # print(gra.source)

    if render:
        gra.render(filename, view=False, engine="dot", format="pdf")
        os.rename(filename, filename + '.dot')
    else:
        gra.save(filename + '.dot')
    return gra
