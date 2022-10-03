from creativitips import utils
from creativitips.CTPs.graphs import TPsGraph
from creativitips.CTPs.tps import TPS
from creativitips.CTPs.pparser import Parser
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.extmath import randomized_svd


class Computation:
    """Implements computation module for compute sequences in iteration"""

    def __init__(self, rng, order=2, weight=1.0, interference=0.001, forgetting=0.05,
                 mem_thres=1.0, unit_len=None, method="BRENT"):

        if not unit_len:
            unit_len = [2, 3]
        self.rng = rng
        self.pars = Parser(unit_len)
        self.method = method
        self.tps_1 = TPS(order)  # memory for TPs between symbols
        self.tps_units = TPS(1)  # memory for TPs between units
        self.graph = TPsGraph()  # memory for tree representation
        self.weight = weight
        self.order = order
        self.t_mem = mem_thres
        self.interf = interference
        self.fogs = forgetting
        self.actions = []
        self.state_entropies = {}
        self.old_p = []
        self.old_p_units = []

    def compute(self, s, first_in_seq):
        if first_in_seq:
            self.old_p = ["START"]
            self.old_p_units = ["START"]
        # print(" ------------------------------------------------------ ")
        # read percept as an array of units
        # active elements in mem shape perception
        active_mem = dict((k, v["weight"]) for k, v in self.pars.mem.items() if v["weight"] >= self.t_mem)
        # certain tps
        # tpc = self.tps_1.get_certain_units()
        # print("tpc1: ", tpc)
        # next nodes from last unit
        higher_mem = []
        # if self.old_p_units[-1] in self.tps_units.mem.keys():
        #     higher_mem = list(u for u in self.tps_units.mem[self.old_p_units[-1]])

        # interference could be applied for those units activated but not used (reinforced)!
        # active_mem = dict((k, v) for k, v in pars.mem.items() if v["weight"] >= 0.5)
        units, action = utils.read_percept(self.rng, active_mem, s, old_seq=self.old_p,
                                           tps=self.tps_1, method=self.method, ulens=self.pars.ulens)
        self.actions.extend(action)
        # chunking
        p = " ".join(units)

        # encode units
        self.pars.encode(p, units, weight=self.weight)
        self.tps_1.encode(self.old_p + p.strip().split(" "))
        # self.tps_2.encode(self.old_p + p.strip().split(" "))
        # self.tps_3.encode(self.old_p + p.strip().split(" "))
        self.tps_units.encode(self.old_p_units + units)
        # self.encode_patterns(self.old_p_units + units)

        # forgetting and interference for parser
        self.pars.forget_interf(self.rng, p, comps=units, interfer=self.interf)

        # forgetting and interference for tps
        if "NFNI" not in self.method:
            # forgetting
            if "WF" in self.method:
                self.tps_units.forget(self.old_p_units + units)
            # interference
            if "WI" in self.method:
                self.tps_units.interfere(self.old_p_units + units, interf=self.interf)
            elif "LI" in self.method:
                self.tps_units.interfere(self.old_p_units + units, interf=self.interf/10)

            self.tps_units.cleaning()

        # generalization step
        # self.graph.encode(units)

        # save past for tps and tps units
        self.old_p = p.strip().split(" ")[-self.order:]
        self.old_p_units = units[-1:]

        return p, units

    def compute_last(self):
        self.tps_1.encode(self.old_p + ["END"])
        self.tps_units.encode(self.old_p_units + ["END"])
        return ["END"]

    def generalize(self, out_dir, gens):
        # self.tps_units.normalize()
        self.graph = TPsGraph(self.tps_units)
        self.graph.generalize(out_dir, gens)


class Embedding:
    def __init__(self, mtx):
        self.vh = None
        self.u = None  # Each row of U matrix is a 3-dimensional vector representation of word
        self.s = None
        self.trans = None
        self.mtx = mtx
        self.squared = None

    def compute(self, lex):
        # re-norm (Hellinger) distance
        self.squared = np.sqrt(self.mtx)
        # singular value decomposition
        svd = TruncatedSVD(n_components=3, random_state=0)
        self.trans = svd.fit_transform(self.squared)
        self.plot3D(self.trans, lex)
        # u, s, vt = randomized_svd(self.squared, n_components=3, random_state=0)
        # self.plot3D(u,lex)
        # self.u, self.s, self.vh = np.linalg.svd(self.squared, full_matrices=True)

    @staticmethod
    def plot3D(mtx, le):
        fig = plt.figure(figsize=(10, 10))
        fig.subplots_adjust(top=1.1, bottom=-.1)
        ax = fig.add_subplot(111, projection='3d')

        x, y, z = mtx.T
        for _ in range(len(z)):
            ax.scatter(x[_], y[_], z[_], cmap='viridis', linewidth=0.5)
            xr = np.random.uniform(0, 0.01) * -np.random.randint(0, 2)
            yr = np.random.uniform(0, 0.01) * -np.random.randint(0, 2)
            zr = np.random.uniform(0, 0.01) * -np.random.randint(0, 2)
            ax.text(x[_] + xr, y[_] + yr, z[_] + zr, le.inverse_transform([_])[0])
        plt.show()

    @staticmethod
    def plot2D(mtx, le):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)

        x, y = mtx.T
        for _ in range(len(x)):
            ax.scatter(x[_], y[_], cmap='viridis', linewidth=0.5)
            xr = np.random.uniform(0, 0.01) * -np.random.randint(0, 2)
            yr = np.random.uniform(0, 0.01) * -np.random.randint(0, 2)
            ax.text(x[_] + xr, y[_] + yr, le.inverse_transform([_])[0])
            print(le.inverse_transform([_])[0])
        plt.show()
