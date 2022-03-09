import utils
from tps import TPSModule
from pparser import ParserModule
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms import community
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.extmath import randomized_svd


class ComputeModule:
    """Implements computation module for compute sequences in iteration"""

    def __init__(self, rng, order=2, weight=1.0, interference=0.001, forgetting=0.05,
                 memory_thres=1.0, unit_len=None, method="BRENT"):

        if not unit_len:
            unit_len = [2,3]
        self.rng = rng
        self.pars = ParserModule()
        self.method = method
        self.tps_1 = TPSModule(order)  # memory for TPs between symbols
        self.tps_units = TPSModule(1)  # memory for TPs between units
        self.weight = weight
        self.order = order
        self.t_mem = memory_thres
        self.ulens = unit_len
        self.interf = interference
        self.fogs = forgetting
        self.actions = []
        self.state_entropies = {}
        self.old_p = []
        self.old_p_units = []

    def compute(self, s, first_in_seq, last_in_seq=False):
        if first_in_seq:
            self.old_p = ["START"]
            self.old_p_units = ["START"]
        if last_in_seq:
            self.tps_1.encode(self.old_p + ["END"])
            self.tps_units.encode(self.old_p_units + ["END"])
            return ["END"]
        # print(" ------------------------------------------------------ ")
        # read percept as an array of units
        # active elements in mem shape perception
        active_mem = dict((k, v) for k, v in self.pars.mem.items() if v >= self.t_mem)
        # interference could be applied for those units activated but not used (reinforced)!
        # active_mem = dict((k, v) for k, v in pars.mem.items() if v >= 0.5)
        units, action = utils.read_percept(self.rng, active_mem, s, old_seq=self.old_p, ulens=self.ulens,
                                           tps=self.tps_1, method=self.method)

        self.actions.extend(action)
        p = " ".join(units)

        # encode units
        self.tps_1.encode(self.old_p + p.strip().split(" "))
        self.tps_units.encode(self.old_p_units + units)
        # self.encode_patterns(self.old_p_units + units)
        # save past for tps
        self.old_p = p.strip().split(" ")[-self.order:]
        # save past for tps units
        self.old_p_units = units[-1:]

        # add entire percept
        if len(p.strip().split(" ")) <= max(self.ulens):
            # p is a unit, a primitive
            if p in self.pars.mem:
                self.pars.mem[p] += self.weight / 2
            else:
                self.pars.mem[p] = self.weight
        else:
            self.pars.add_weight(p, comps=units, weight=self.weight)

        # forgetting and interference
        self.pars.forget_interf(self.rng, p, comps=units, forget=self.fogs, interfer=self.interf, ulens=self.ulens)
        # tps_units.forget(units, forget=fogs)
        self.tps_units.forget(units, forget=self.fogs)
        # self.tps_units.forget(units, forget=0.02)
        # print("mem: ", tps_units.mem)
        return p, units

    def generalize(self):
        graph = GraphModule(self.tps_units)
        cl_dict = {x:i for i,x in enumerate(graph.sim_rank())}
        found_paths = set()


class GraphModule:
    def __init__(self, tps, thresh=0.0):
        self.p_index = 0
        self.fc = None
        self.G = nx.DiGraph()

        # create graph
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

    def draw_graph(self, filename):
        nx.draw(self.G)
        # plt.draw()
        nx.nx_pydot.write_dot(self.G, filename)
        # plt.show()

    def sim_rank(self,tsh=0.5):
        inw = nx.algorithms.similarity.simrank_similarity(self.G)
        otw = nx.algorithms.similarity.simrank_similarity(self.G.reverse())
        cl_form = set()
        for k, v in inw.items():
            tt1 = tuple(k2 for k2, v2 in v.items() if v2 > tsh)
            tt2 = tuple(k2 for k2, v2 in otw[k].items() if v2 > tsh)
            tt = tuple(set(tt1 + tt2))
            if len(tt) > 1:
                cl_form.add(tt)
        return cl_form

    def generalize(self, sim_classes):

        for cls in sim_classes:
            self.G.add_node("P" + self.p_index)
            for node in self.G.nodes(data=True):
                if node in cls:
                    self.G.remove_node(node)

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

    def form_classes(self):
        self.fc = nx.topological_generations(self.G)
        print("topological_generations: ",[sorted(gen) for gen in self.fc])

    def get_communities(self):
        communities_generator = community.girvan_newman(self.G)
        # top_level_communities = next(communities_generator)
        next_level_communities = next(communities_generator)
        print("communities:", sorted(map(sorted, next_level_communities)))


class EmbedModule:
    def __init__(self, mtx):
        self.vh = None
        self.u = None  # Each row of U matrix is a 3-dimensional vector representation of word
        self.s = None
        self.trans = None
        self.mtx = mtx
        self.squared = None

    def compute(self,lex):
        # re-norm (Hellinger) distance
        self.squared = np.sqrt(self.mtx)
        # singular value decomposition
        svd = TruncatedSVD(n_components=3, random_state=0)
        self.trans = svd.fit_transform(self.squared)
        self.plot3D(self.trans,lex)
        # u, s, vt = randomized_svd(self.squared, n_components=3, random_state=0)
        # self.plot3D(u,lex)
        # self.u, self.s, self.vh = np.linalg.svd(self.squared, full_matrices=True)

    @staticmethod
    def plot3D(mtx,le):
        fig = plt.figure(figsize=(10, 10))
        fig.subplots_adjust(top=1.1, bottom=-.1)
        ax = fig.add_subplot(111, projection='3d')

        x,y,z = mtx.T
        for _ in range(len(z)):
            ax.scatter(x[_], y[_], z[_], cmap='viridis', linewidth=0.5)
            xr = np.random.uniform(0, 0.01) * -np.random.randint(0, 2)
            yr = np.random.uniform(0, 0.01) * -np.random.randint(0, 2)
            zr = np.random.uniform(0, 0.01) * -np.random.randint(0, 2)
            ax.text(x[_]+xr, y[_]+yr, z[_]+zr, le.inverse_transform([_])[0])
        plt.show()

    @staticmethod
    def plot2D(mtx, le):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)

        x, y = mtx.T
        for _ in range(len(x)):
            ax.scatter(x[_], y[_],cmap='viridis', linewidth=0.5)
            xr = np.random.uniform(0, 0.01)*-np.random.randint(0,2)
            yr = np.random.uniform(0, 0.01)*-np.random.randint(0,2)
            ax.text(x[_]+xr, y[_]+yr, le.inverse_transform([_])[0])
            print(le.inverse_transform([_])[0])
        plt.show()
