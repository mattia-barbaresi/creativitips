from plotly.matplotlylib.mplexporter.utils import iter_all_children

from creativitips.CTPs.tps import TPS
from creativitips.CTPs.pparser import Parser
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.extmath import randomized_svd


class Computation:
    """Implements computation module for compute sequences in iteration"""

    def __init__(self, rng, order=2, weight=1.0, interference=0.001, mem_thres=1.0, unit_len=None, method="BRENT"):

        if not unit_len:
            unit_len = [2, 3]
        self.rng = rng
        self.pars = Parser(unit_len)
        self.method = method
        self.tps_1 = TPS(order)  # memory for TPs between symbols
        self.tps_units = TPS(1)  # memory for TPs between units
        self.weight = weight
        self.order = order
        self.t_mem = mem_thres
        self.interf = interference
        self.actions = []
        self.state_entropies = {}
        self.old_p = []
        self.old_p_units = []
        self.shallow_parsing = []

    def read_percept(self, sequence):
        # certain tps
        # tpc = self.tps_1.get_certain_units()
        # print("tpc1: ", tpc)

        # next nodes from last unit
        # interference could be applied for those units activated but not used (reinforced)!
        higher_mem = []
        # if self.old_p_units[-1] in self.tps_units.mem.keys():
        #     higher_mem = list(u for u in self.tps_units.mem[self.old_p_units[-1]])

        active_mem = dict((k, v["weight"]) for k, v in self.pars.mem.items() if v["weight"] >= self.t_mem)
        """Return next percept in sequence as an ordered array of units in mem or components (bigrams)"""
        ulens = self.pars.ulens
        if ulens is None:
            ulens = [2]  # default like parser (bigrams)
        old_seq = self.old_p
        res = []
        # number of units embedded in next percepts
        i = self.rng.integers(low=1, high=4)
        s = sequence
        actions = []
        while len(s) > 0 and i > 0:
            action = ""
            unit = []
            # units_list = [(k,v) for k,v in mem.items() if s.startswith(k)]
            units_list = [k for k in active_mem.keys() if (" ".join(s)+" ").startswith(k+" ")]
            h_list = []
            # if higher_list:
            #     h_list = [k for k in higher_list if " ".join(s).startswith(k)]
            # action = ""
            # if len(s) <= max(ulens):
            #     unit = s
            #     action = "end"
            # el
            # if h_list:
            #     unit = (sorted(h_list, key=lambda key: len(key), reverse=True)[0]).strip().split(" ")
            #     # print("mem unit:", unit)
            #     action = "high_mem"
            if units_list:
                # a unit in mem matched
                unit = (sorted(units_list, key=lambda key: len(key), reverse=True)[0]).strip().split(" ")
                print("mem unit:", unit)
                action = "mem"
            elif self.tps_1:
                if "BRENT" in self.method:
                    unit = self.tps_1.get_next_unit_brent(s[:5], past=old_seq)
                elif "CT" in self.method:
                    unit = self.tps_1.get_next_certain_unit(s[:5], past=old_seq)
                elif "MI" in self.method:
                    unit = self.tps_1.get_next_unit_mi(s[:5], past=old_seq)
                elif "BTP" in self.method:
                    unit = self.tps_1.get_next_unit_btps(s[:5], past=old_seq)
                elif "FTPAVG" in self.method:
                    unit = self.tps_1.get_next_unit_ftps_withAVG(s[:5], past=old_seq)
                elif "AVG" in self.method:
                    unit = self.tps_1.get_next_unit_with_AVG(s[:5], past=old_seq)
                else:  # if TPS
                    unit = self.tps_1.get_next_unit_ftps(s[:5], past=old_seq)
                action = "tps"

            # if no unit found, pick at random length
            if not unit:
                # unit = s[:2]  # add Parser basic components (bigram/syllable)..
                # random unit
                unit = s[:self.rng.choice(ulens)]
                print("random unit:", unit)
                action = "rnd"

            # check if last symbol
            sf = s[len(unit):]
            if len(sf) == 1:
                unit += sf
            # if self.tps_1:
            #     self.tps_1.update_avg(self.tps_1.get_ftps_sequence(old_seq + unit))
            res.append(" ".join(unit))
            actions.append(action)
            # print("final unit:", unit)
            s = s[len(unit):]
            # for calculating next unit with tps
            old_seq = unit
            i -= 1

        return res, actions

    def compute(self, sequences):
        for s in sequences:
            # if len(s) < 2:
            #     continue
            self.old_p = ["START"]
            self.old_p_units = ["START"]
            shpar_units = []
            while len(s) > 0:
                # --------------- COMPUTE ---------------

                # compute next percept
                units, action = self.read_percept(s)
                for un in units:
                    shpar_units.append(un)
                self.actions.extend(action)
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
                # update s
                s = s[len(p.strip().split(" ")):]

            sp_seq = " || ".join(shpar_units).strip(".?")
            print("shallow parsed: ", sp_seq)
            self.shallow_parsing.append(sp_seq)
            # compute last
            self.tps_1.encode(self.old_p + ["END"])
            self.tps_units.encode(self.old_p_units + ["END"])

            # --------------- GENERATE ---------------
            # if iteration % 5 == 1:
            #     results[iteration] = dict()
            #     utils.generate_seqs(rng, cm, results[iteration])

    def compute_last(self):
        self.tps_1.encode(self.old_p + ["END"])
        self.tps_units.encode(self.old_p_units + ["END"])
        return ["END"]


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
