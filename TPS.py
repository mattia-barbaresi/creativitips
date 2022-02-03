from datetime import datetime
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
from sympy import sequence
import complexity
import form_class as fc
import const
from Graph import GraphModule
import utils
from Parser import ParserModule
from word_emmbedding import EmbedModule


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
            # for ind, tp in enumerate(tps_seqs):
            #     if tp < ths:  # insert a break
            #         res.append(percept[start: self.order + ind])
            #         start = self.order + ind
            #
            for ind, tp in enumerate(tps_seqs[:-1]):
                if tps_seqs[ind] < tps_seqs[ind + 1]:  # insert a break
                    res.append(percept[start: self.order + ind])
                    start = self.order + ind

            res.append(percept[start:])
        else:
            print("(TPmodule):Order must be grater than 1.")

        return res

    def get_units_brent(self, percept):
        """
        Returns segmented percept using stored TPs. (trough-based segmentation strategy)
        (Brent 1999) a formalization of the original proposal by Saffarn, Newport et al.
        Consider the segment “wxyz”…whenever the statistical value (TPs or O/E) of the transitions under consideration
        is lower than the statistical values of its adjacent neighbors, a boundary is inserted.
        IF TPs(“wx”) > TPs(“xy”) < TPs(“yz”)  segments between “x” e “y”
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
        res = []
        for k in self.mem.keys():
            if len(self.mem[k]) == 1:
                res.append(k + next(iter(self.mem[k])))
        return res

    def get_next_unit(self, percept, past=""):
        # if order = 1 no past required
        if self.order > 1:
            past = past[-(self.order-1):]
        else:
            past = ""
        pp_seq = past + percept
        if self.order > 0:
            if self.order < len(pp_seq):
                tps_seqs = self.get_tps_sequence(pp_seq)
                # print("percept: ", percept)
                # print("tps_seqs: ", tps_seqs)
                # tps_seqs += [float('-inf'),float('+inf')]
                for ii in range(len(tps_seqs)-1):
                    if tps_seqs[ii] < tps_seqs[ii + 1]:  # insert a break
                        # print("tps unit: ", percept[:(self.order - len(past)) + ii])
                        return percept[:(self.order - len(past)) + ii]
            else:
                print("(TPmodule):Order grater than percept length. Percept is too short.")
        else:
            print("(TPmodule):Order must be grater than 1.")

        return ""

    def get_next_unit_brent(self, percept, past=""):
        """
        Returns segmented percept using stored TPs. (trough-based segmentation strategy)
        (Brent 1999) a formalization of the original proposal by Saffarn, Newport et al.
        Consider the segment “wxyz”…whenever the statistical value (TPs or O/E) of the transitions under consideration
        is lower than the statistical values of its adjacent neighbors, a boundary is inserted.
        IF TPs(“wx”) > TPs(“xy”) < TPs(“yz”)  segments between “x” e “y”.
        Paper: https://link.springer.com/content/pdf/10.1023/A:1007541817488.pdf
        """
        # if order = 1 no past required
        if self.order > 1:
            past = past[-(self.order-1):]
        else:
            past = ""
        pp_seq = past + percept
        if self.order > 0:
            if self.order < len(pp_seq):
                tps_seqs = self.get_tps_sequence(pp_seq)
                # print("percept: ", percept)
                # print("tps_seqs: ", tps_seqs)
                tps_seqs = [float('inf')] + tps_seqs  # add an init high trans (for segmenting first positions too)
                for ii in range(1, len(tps_seqs) - 1):
                    if tps_seqs[ii - 1] > tps_seqs[ii] < tps_seqs[ii + 1]:  # insert a break
                        # print("tps unit: ", percept[:(self.order - len(past)) + ii - 1])
                        return percept[:(self.order - len(past)) + ii - 1]
            else:
                print("(TPmodule):Order grater than percept length. Percept is too short.")
        else:
            print("(TPmodule):Order must be grater than 1.")

        return ""

    def get_tps_sequence(self, seq):
        tps_seqs = []
        for ii in range(self.order, len(seq)):
            h = "".join(seq[ii - self.order:ii])
            o = "".join(seq[ii:ii + 1])
            if h in self.mem and o in self.mem[h]:
                # v = self.mem[h][o]
                v = self.mem[h][o] / np.sum(list(self.mem[h].values()))
            else:
                v = 0  # no encoded transition
            tps_seqs.append(v)  # TPS
            # print("{}-{} = {}".format(h, o, v))
        return tps_seqs

    def generate_new_sequences(self, rand_gen, n_seq=20, min_len=20, initials=None, be=None):
        res = []
        if not initials:
            # row classes minus cols classes returns the nodes that have no inward edges
            init_set = sorted(set(self.le_rows.classes_) - set(self.le_cols.classes_))
        else:
            init_set = sorted(initials.intersection(set(self.le_rows.classes_)))

        if not init_set:
            print("Empty init set. No generation occurred.")
            return res, []

        for _ in range(n_seq):
            # choose rnd starting point
            seq = rand_gen.choice(init_set)
            _s = seq
            for _ in range(min_len):
                if _s not in self.le_rows.classes_:
                    break
                i = self.le_rows.transform([_s])[0]
                _s = self.le_cols.inverse_transform([utils.mc_choice(rand_gen, self.norm_mem[i])])[0]
                seq += _s
            res.append(seq)

        if be:
            decs = []
            init_set = [be.base_decode(x) for x in init_set]
            for gg in res:
                decs.append(be.base_decode(gg))
            res = decs

        return res, init_set

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
    rng = np.random.default_rng(const.RND_SEED)

    # file_names = \
    #     ["input", "input2", "saffran", "thompson_newport", "reber", "all_songs_in_G", "all_irish-notes_and_durations"]

    file_names = ["input"]

    # root_dir = const.OUT_DIR + "TPS_{}_({}_{}_{})_{}/".format(
    #     const.TPS_ORDER, const.MEM_THRES, const.FORGETTING, const.INTERFERENCE,
    #     time.strftime("%Y%m%d-%H%M%S"))
    # # root_dir = const.OUT_DIR + "RND_(2-3)_({}_{}_{})_{}/".
    # # format(const.MEM_THRES,const.FORGETTING, const.INTERFERENCE, time.strftime("%Y%m%d-%H%M%S"))

    # base_encoder = None
    # os.makedirs(root_dir, exist_ok=True)
    # shutil.copy2("const.py", root_dir + "pars.txt")

    # maintaining INTERFERENCES/FORGETS separation by a factor of 10
    interferences = [0.005]
    forgets = [0.05]
    thresholds_mem = [0.9]
    tps_orders = [2]
    method = "TPS"
    for order in tps_orders:
        for interf in interferences:
            for fogs in forgets:
                for t_mem in thresholds_mem:
                    # init
                    root_dir = const.OUT_DIR + "{}_{}_({}_{}_{})_{}/"\
                        .format(method, order, t_mem, fogs, interf, time.strftime("%Y%m%d-%H%M%S"))
                    # root_dir = const.OUT_DIR + "RND_(2-3)_({}_{}_{})_{}/".\
                    #     format(t_mem, fogs, interf, time.strftime("%Y%m%d-%H%M%S"))

                    base_encoder = None
                    os.makedirs(root_dir, exist_ok=True)
                    with open(root_dir + "pars.txt", "w") as of:
                        json.dump({
                            "rnd": const.RND_SEED,
                            "w": const.WEIGHT,
                            "interference": interf,
                            "forgetting": fogs,
                            "mem thresh": t_mem,
                            "lens": const.ULENS,
                            # "tps_order": order,
                        }, of)

                    for fn in file_names:
                        print("processing {} series ...".format(fn))
                        # init
                        pars = ParserModule()
                        tps_units = TPSModule(1)  # memory for TPs inter
                        tps_1 = TPSModule(order)  # memory for TPs intra
                        out_dir = root_dir + "{}/".format(fn)
                        os.makedirs(out_dir, exist_ok=True)
                        results = dict()
                        # --------------- INPUT ---------------
                        if fn == "saffran":
                            # load Saffran input
                            sequences = utils.generate_Saffran_sequence(rng)
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
                        initial_set = set()
                        start_time = datetime.now()
                        for iteration,s in enumerate(sequences):
                            old_p = ""
                            old_p_units = []
                            first_in_seq = True
                            while len(s) > 0:
                                # print(" ------------------------------------------------------ ")
                                # read percept as an array of units
                                # active elements in mem shape perception
                                active_mem = dict((k, v) for k, v in pars.mem.items() if v >= t_mem)
                                # active_mem = dict((k, v) for k, v in pars.mem.items() if v >= 0.5)
                                units, action = utils.read_percept(rng, active_mem, s, old_seq=old_p, ulens=const.ULENS, tps=tps_1, method=method)
                                # add initial nodes of sequences for generation
                                if first_in_seq:
                                    initial_set.add(units[0])
                                    first_in_seq = False
                                actions.append(action)
                                p = "".join(units)
                                tps_1.encode(old_p + p)
                                # save past for tps
                                old_p = p[-order:]
                                # print("units: ", units, " -> ", p)
                                tps_units.encode(old_p_units + units)

                                # add entire percept
                                if len(p) <= max(const.ULENS):
                                    # p is a unit, a primitive
                                    if p in pars.mem:
                                        pars.mem[p] += const.WEIGHT / 2
                                    else:
                                        pars.mem[p] = const.WEIGHT
                                else:
                                    pars.add_weight(p, comps=units, weight=const.WEIGHT)

                                # save past for tps units
                                old_p_units = units[-1:]
                                # forgetting and interference
                                pars.forget_interf(rng, p, comps=units, forget=fogs, interfer=interf, ulens=const.ULENS)
                                tps_units.forget(units, forget=fogs)
                                # print("mem: ", tps_units.mem)
                                s = s[len(p):]

                            # generation
                            if iteration % 10 == 1:
                                tps_units.normalize()
                                results[iteration] = dict()
                                results[iteration]["generated"], results[iteration]["initials"] = tps_units.generate_new_sequences(rng, min_len=100, be=base_encoder, initials=initial_set)
                                if fn == "bicinia" or fn == "all_irish-notes_and_durations":
                                    iter_mem = dict(sorted([(base_encoder.base_decode(x), y) for x, y in pars.mem.items()], key=lambda it: it[1], reverse=True))
                                else:
                                    iter_mem = dict(sorted([(x, y) for x, y in pars.mem.items()], key=lambda item: item[1], reverse=True))
                                results[iteration]["mem"] = iter_mem

                        # dc = fc.distributional_context(fc_seqs, 3)
                        # # print("---- dc ---- ")
                        # # pp.pprint(dc)
                        # classes = fc.form_classes(dc)
                        # class_patt = fc.classes_patterns(classes, fc_seqs)

                        results["processing time"] = str((datetime.now() - start_time).total_seconds())
                        # normilizes memories
                        # tps_1.normalize()
                        tps_units.normalize()

                        # embedding chunks
                        # wem = EmbedModule(tps_units.norm_mem)
                        # wem.compute(tps_units.le_rows)

                        # calculate states uncertainty
                        # tps_1.compute_states_entropy(be=base_encoder)
                        tps_units.compute_states_entropy(be=base_encoder)

                        # generate sample sequences
                        print("initials: ", sorted(initial_set))
                        # gens, init = tps_units.generate_new_sequences(min_len=100, be=base_encoder, initials=initial_set)
                        gens, init = tps_units.generate_new_sequences(rng, min_len=100, be=base_encoder, initials=initial_set)
                        print("init set: ", init)
                        print("gens: ", gens)

                        # save results
                        with open(out_dir + "results.json", "w") as of:
                            json.dump(results, of)
                        # save generated
                        with open(out_dir + "generated.json", "w") as of:
                            json.dump(gens, of)
                        # save actions
                        with open(out_dir + "action.json", "w") as of:
                            json.dump(actions,of)
                        utils.plot_actions(actions,path=out_dir, show_fig=False)
                        # print(tps_units.mem)
                        # utils.plot_gra(tps_units.mem)
                        print("plotting tps units...")
                        utils.plot_gra_from_normalized(tps_units, filename=out_dir + "tps_units", be=base_encoder)
                        # print("plotting tps all...")
                        # utils.plot_gra_from_normalized(tps_1, filename=out_dir + "tps_symbols", be=base_encoder)
                        print("plotting memory...")
                        # plot memeory chunks
                        # for "bicinia" and "all_irish_notes_and_durations" use base_decode
                        if fn == "bicinia" or fn == "all_irish-notes_and_durations":
                            ord_mem = dict(sorted([(base_encoder.base_decode(x),y) for x,y in pars.mem.items()], key=lambda it: it[1], reverse=True))
                        else:
                            ord_mem = dict(sorted([(x, y) for x, y in pars.mem.items()], key=lambda item: item[1], reverse=True))
                        utils.plot_mem(ord_mem, out_dir + "words_plot.png", save_fig=True, show_fig=False)

                        graph = GraphModule(tps_units, be=base_encoder)
                        graph.form_classes()
                        # graph.draw_graph(out_dir + "nxGraph.dot")
                        graph.print_values()
                        # graph.get_communities()

