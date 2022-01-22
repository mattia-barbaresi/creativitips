import fnmatch
import json
import math
import os
import shutil
import time

import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing
from scipy.special import softmax
import complexity
import form_class as fc
import const
import utils
from Parser import ParserModule


class TPSModule:
    """Implements a transitional probabilities module"""

    def __init__(self, order=1):
        self.mem = dict()
        self.norm_mem = np.array([])
        self.order = order
        self.prev = ""
        self.le_rows = preprocessing.LabelEncoder()
        self.le_cols = preprocessing.LabelEncoder()
        self.state_entropies = {}

    def encode(self, percept, wg=1.0):
        """
        Encode TPs of the given percept, for the given order.
        :param wg: assigned weight for memory entries
        :param percept: could be a (i) string, or (ii) an (ordered, sequential) array of strings.
        In case (ii) TPs are counted between elements of the array (units), instead of between symbols
        """

        if self.order > 0:
            for ii in range(self.order, len(percept)):
                h = "".join(percept[ii - self.order:ii])
                o = "".join(percept[ii:ii + 1])
                if h in self.mem:
                    if o in self.mem[h]:
                        self.mem[h][o] += wg
                    else:
                        self.mem[h][o] = wg
                else:
                    self.mem[h] = {o: wg}
        else:
            print("(TPmodule):Order must be grater than 1.")

    def normalize(self):
        cols = []
        keys = list(self.mem.keys())
        self.le_rows.fit(keys)
        for k in keys:
            cols.extend(list(self.mem[k].keys()))
        self.le_cols.fit(cols)
        self.norm_mem = np.zeros((len(self.le_rows.classes_), len(self.le_cols.classes_)))
        for k, v in self.mem.items():
            for kk, vv in v.items():
                self.norm_mem[self.le_rows.transform([k])[0]][self.le_cols.transform([kk])[0]] = vv
        # self.norm_mem = softmax(self.norm_mem, axis=1)
        self.norm_mem = utils.softmax(self.norm_mem)

    def forget(self, uts, forget):
        for h, vs in self.mem.items():
            for o, v in vs.items():
                if h not in uts and o not in uts:
                    self.mem[h][o] -= forget
        # cleaning
        for h, vs in self.mem.items():
            for key in [k for k, v in vs.items() if v <= 0.0]:
                self.mem[h].pop(key)
        for key in [k for k, v in self.mem.items() if len(v) == 0]:
            self.mem.pop(key)

    def get_units(self, percept, ths=0.5):
        """
        Returns segmented percept using stored TPs.
        :param ths: segmentation threshold
        :param percept could be a string, or an (ordered, sequential) array of strings.
        In latter case TPs are counted between elements of the array(units), instead of between symbols
        """
        self.normalize()
        res = []
        tps_seqs = []
        if self.order > 0:
            for ii in range(self.order, len(percept)):
                h = "".join(percept[ii - self.order:ii])
                o = "".join(percept[ii:ii + 1])
                tps_seqs.append(self.norm_mem[self.le_rows.transform([h])[0]][self.le_cols.transform([o])[0]])
            start = 0
            for ind, tp in enumerate(tps_seqs):
                if tp < ths:  # insert a break
                    res.append(percept[start: self.order + ind])
                    start = self.order + ind
            res.append(percept[start:])
        else:
            print("(TPmodule):Order must be grater than 1.")

        return res

    def get_units_brent(self, percept):
        """
        Returns segmented percept using stored TPs.
        (Brent 1999) a formalization of the original proposal by Saffarn, Newport et al.
        Consider the segment “wxyz”…whenever the statistical value (TPs or O/E) of the transitions under consideration
        is lower than the statistical values of its adjacent neighbors, a boundary is inserted.
        IF TPs (“xy”) < TPs(“wx”) and < TPs(“yz”)  segments between “x” e “y”
        :param percept could be a string, or an (ordered, sequential) array of strings.
        In latter case TPs are counted between elements of the array(units), instead of between symbols
        """
        self.normalize()
        res = []
        tps_seqs = []
        if self.order > 0:
            for ii in range(self.order, len(percept)):
                h = "".join(percept[ii - self.order:ii])
                o = "".join(percept[ii:ii + 1])
                tps_seqs.append(self.norm_mem[self.le_rows.transform([h])[0]][self.le_cols.transform([o])[0]])
            start = 0
            _i = 1
            tps_seqs = [1.0] + tps_seqs  # consider the first as high transitions(for segmenting first positions too)
            while _i < len(tps_seqs) - 1:
                if tps_seqs[_i - 1] > tps_seqs[_i] < tps_seqs[_i + 1]:  # insert a break
                    res.append(percept[start: self.order + _i - 1])
                    start = self.order + _i - 1
                _i += 1
            # for _i in range(1,len(tps_seqs)-1):
            #     if tps_seqs[_i-1] < tps_seqs[_i] < tps_seqs[_i+1]:  # insert a break
            #         res.append(percept[start:self.order + _i])
            #         start = self.order + _i
            res.append(percept[start:])
        else:
            print("(TPmodule):Order must be grater than 1.")

        return res

    def get_probs(self, percept):
        """
        Returns TPs between symbols in percepts.
        """
        res = []
        tps_seqs = []
        if self.order > 0:
            for ii in range(self.order, len(percept)):
                h = "".join(percept[ii - self.order:ii])
                o = "".join(percept[ii:ii + 1])
                tps_seqs.append(self.norm_mem[self.le_rows.transform([h])[0]][self.le_cols.transform([o])[0]])
            return tps_seqs
        else:
            print("(TPmodule):Order must be grater than 1.")

        return res

    def get_certain_units(self):
        ures = []
        for k in self.mem.keys():
            if len(self.mem[k]) == 1:
                ures.append(k + next(iter(self.mem[k])))
        return ures

    def get_next_unit(self, percept):
        tps_seqs = []
        if self.order > 0:
            if self.order < len(percept):
                for ii in range(self.order, len(percept)):
                    h = "".join(percept[ii - self.order:ii])
                    o = "".join(percept[ii:ii + 1])
                    if h in self.mem and o in self.mem[h]:
                        tps_seqs.append(self.mem[h][o])
                    else:
                        tps_seqs.append(0)  # no encoded transition
                print("tps_seqs: ", tps_seqs)
                for ii in range(len(tps_seqs) - 1):
                    if tps_seqs[ii] < tps_seqs[ii + 1]:  # insert a break
                        print("------ unit: ", percept[:self.order + ii])
                        return percept[:self.order + ii]
            else:
                print("(TPmodule):Order grater than percetp length. Percept is too short.")
        else:
            print("(TPmodule):Order must be grater than 1.")

        return ""

    def generate_new_sequences(self, n_seq=10, min_len=20):
        # row classes minus cols classes returns the nodes that have no inward edges
        init_set = list(set(self.le_rows.classes_) - set(self.le_cols.classes_))
        print("tps.le_rows.classes_:", self.le_rows.classes_)
        print("tps.le_cols.classes_:", self.le_cols.classes_)
        print("init_set:", init_set)
        res = []
        if not init_set:
            print("Empty init set. No generation occurred.")
            return res

        for _ in range(n_seq):
            # choose rnd starting point
            seq = np.random.choice(init_set)
            _s = seq
            for _ in range(min_len):
                if _s not in self.le_rows.classes_:
                    break
                i = self.le_rows.transform([_s])[0]
                _s = self.le_cols.inverse_transform([utils.mc_choice(self.norm_mem[i])])[0]
                seq += _s
            res.append(seq)
        return res

    def get_next_unit_brent(self, percept):
        tps_seqs = []
        if self.order > 0:
            if self.order < len(percept):
                for ii in range(self.order, len(percept)):
                    h = "".join(percept[ii - self.order:ii])
                    o = "".join(percept[ii:ii + 1])
                    if h in self.mem and o in self.mem[h]:
                        tps_seqs.append(self.mem[h][o])
                    else:
                        tps_seqs.append(0)  # no encoded transition
                tps_seqs = [float('inf')] + tps_seqs  # add an init high trans (for segmenting first positions too)
                for ii in range(1, len(tps_seqs) - 1):
                    if tps_seqs[ii - 1] > tps_seqs[ii] < tps_seqs[ii + 1]:  # insert a break
                        print("------ unit: ", percept[:self.order + ii - 1])
                        return percept[:self.order + ii - 1]
            else:
                print("(TPmodule):Order grater than percetp length. Percept is too short.")
        else:
            print("(TPmodule):Order must be grater than 1.")

        return ""

    def compute_states_entropy(self, be):
        self.state_entropies = {}
        rows, cols = self.norm_mem.shape
        for _r in range(rows):
            _s = 0
            for _el in self.norm_mem[_r]:
                if _el > 0.0:
                    _s -= _el * math.log(_el, 2)
            if be:
                self.state_entropies[be.base_decode(self.le_rows.inverse_transform([_r])[0])] = _s
            else:
                self.state_entropies[self.le_rows.inverse_transform([_r])[0]] = _s
        return self.state_entropies


if __name__ == "__main__":
    np.set_printoptions(linewidth=np.inf)
    np.random.seed(const.RND_SEED)
    file_names = ["all_irish-notes_and_durations"]
    base_encoder = None

    for fn in file_names:
        # init
        pars = ParserModule()
        tps_units = TPSModule(1)  # memory for TPs inter
        tps_1 = TPSModule(const.TPS_ORDER)  # memory for TPs intra

        out_dir = const.OUT_DIR + "{}_{}/".format(fn, time.strftime("%Y%m%d-%H%M%S"))
        os.makedirs(out_dir,exist_ok=True)
        shutil.copy2("const.py",out_dir+"/pars.txt")
        # --------------- INPUT ---------------
        if fn == "saffran":
            # load Saffran input
            sequences = utils.generate_Saffran_sequence()
        elif fn == "all_irish-notes_and_durations":
            base_encoder = utils.Encoder()
            sequences = utils.load_irish_n_d("data/all_irish-notes_and_durations-abc.txt", base_encoder)
        elif fn == "bicinia":
            base_encoder = utils.Encoder()
            sequences = utils.load_bicinia_single("data/bicinia/", base_encoder, seq_n=1)
        else:
            with open("data/{}.txt".format(fn), "r") as fp:
                sequences = [line.strip() for line in fp]

        # read percepts using parser function
        actions = []
        for s in sequences:
            old_p = ""
            old_p_units = []
            while len(s) > 0:
                print(" ------------------------------------------------------ ")
                # read percept as an array of units
                # active elements in mem shape perception
                active_mem = dict((k, v) for k, v in pars.mem.items() if v >= const.MEM_THRES)
                # active_mem = dict((k, v) for k, v in pars.mem.items() if v >= 0.5)
                units, action = utils.read_percept(active_mem, s, ulens=const.ULENS, tps=tps_1)
                actions.append(action)
                p = "".join(units)
                tps_1.encode(old_p + p)
                # save past for tps
                old_p = p[-const.TPS_ORDER:]
                # print("units: ", units, " -> ", p)
                # add entire percept
                if len(p) <= max(const.ULENS):
                    # p is a unit, a primitive
                    if p in pars.mem:
                        pars.mem[p] += const.WEIGHT / 2
                    else:
                        pars.mem[p] = const.WEIGHT
                else:
                    tps_units.encode(old_p_units + units)
                    # save past for tps units
                    old_p_units = units[-1:]
                    pars.add_weight(p, comps=units, weight=const.WEIGHT)
                # forgetting and interference
                pars.forget_interf(p, comps=units, forget=const.FORGETTING, interfer=const.INTERFERENCE, ulens=const.ULENS)
                tps_units.forget(units, forget=const.FORGETTING)
                s = s[len(p):]

        # dc = fc.distributional_context(fc_seqs, 3)
        # # print("---- dc ---- ")
        # # pp.pprint(dc)
        # classes = fc.form_classes(dc)
        # class_patt = fc.classes_patterns(classes, fc_seqs)

        # normilizes memories
        tps_1.normalize()
        tps_units.normalize()
        
        # calculate states uncertainty
        tps_1.compute_states_entropy(be=base_encoder)
        tps_units.compute_states_entropy(be=base_encoder)
        # generate sample sequences
        decoded = []
        gens = tps_units.generate_new_sequences(min_len=100)
        print("gens: ", gens)
        # save all
        with open(out_dir + "generated.json", "w") as of:
            if base_encoder:
                for gg in gens:
                    decoded.append(base_encoder.base_decode(gg))
                json.dump(decoded, of)
                print("decoded: ", decoded)
            else:
                json.dump(gens, of)

        # save all
        with open(out_dir + "action.json", "w") as of:
            json.dump(actions,of)
        utils.plot_actions(actions,path=out_dir)

        # print(tps_units.mem)
        # utils.plot_gra(tps_units.mem)
        utils.plot_gra_from_normalized(tps_units, filename=out_dir + "tps_units", be=base_encoder)
        utils.plot_gra_from_normalized(tps_1, filename=out_dir + "tps_1", be=base_encoder)

        # plot memeory chunks
        # for "bicinia" and "all_irish_notes_and_durations" use base_decode
        if fn == "bicinia" or fn == "all_irish-notes_and_durations":
            ord_mem = dict(sorted([(base_encoder.base_decode(x),y) for x,y in pars.mem.items()], key=lambda it: it[1], reverse=True))
        else:
            ord_mem = dict(sorted([(x, y) for x, y in pars.mem.items()], key=lambda item: item[1], reverse=True))
        utils.plot_mem(ord_mem, out_dir + "words_plot.png", save_fig=True, show_fig=True)

