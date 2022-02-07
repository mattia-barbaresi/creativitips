import math
import numpy as np
from sklearn import preprocessing
import utils


class TPSModule:
    """Implements a transitional probabilities module"""

    def __init__(self, ordr=1):
        self.mem = dict()
        self.norm_mem = np.array([])
        self.order = ordr
        self.le_rows = preprocessing.LabelEncoder()
        self.le_cols = preprocessing.LabelEncoder()
        self.state_entropies = {}

    def encode(self, percept, wg=1.0):
        """
        Encode TPs of the given percept, for the given order
        :param wg assigned weight for memory entries
        :param percept could be a (i) string, or (ii) an (ordered, sequential) array of strings.
        In case (ii) TPs are counted between elements of the array (units), instead of between symbols
        """

        if self.order > 0:
            for ii in range(self.order, len(percept)):
                h = " ".join(percept[ii - self.order:ii])
                o = " ".join(percept[ii:ii + 1])
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
        Returns segmented percept using stored TPs
        :param ths segmentation threshold
        :param percept could be a string, or an (ordered, sequential) array of strings.
        In latter case TPs are counted between elements of the array(units), instead of between symbols
        """
        self.normalize()
        res = []
        percept = percept.strip().split(" ")
        if self.order > 0:
            tps_seqs = self.get_probs_normalized(percept)
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
        percept = percept.strip().split(" ")
        if self.order > 0:
            tps_seqs = self.get_probs_normalized(percept)
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

    def get_probs_normalized(self, percept):
        """
        Returns TPs between symbols in percepts.
        """
        res = []
        tps_seqs = []
        if self.order > 0:
            for ii in range(self.order, len(percept)):
                h = " ".join(percept[ii - self.order:ii])
                o = " ".join(percept[ii:ii + 1])
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

    def get_next_unit(self, percept, past=None):
        # if order = 1 no past required
        if self.order > 1 and past:
            past = past[-(self.order-1):]
        else:
            past = []
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

        return []

    def get_next_unit_brent(self, percept, past=None):
        """
        Returns segmented percept using stored TPs. (trough-based segmentation strategy)
        (Brent 1999) a formalization of the original proposal by Saffarn, Newport et al.
        Consider the segment “wxyz”…whenever the statistical value (TPs or O/E) of the transitions under consideration
        is lower than the statistical values of its adjacent neighbors, a boundary is inserted.
        IF TPs(“wx”) > TPs(“xy”) < TPs(“yz”)  segments between “x” e “y”.
        Paper: https://link.springer.com/content/pdf/10.1023/A:1007541817488.pdf
        """
        # if order = 1 no past required
        if self.order > 1 and past:
            past = past[-(self.order-1):]
        else:
            past = []
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

        return []

    def get_tps_sequence(self, seq):
        tps_seqs = []
        for ii in range(self.order, len(seq)):
            h = " ".join(seq[ii - self.order:ii])
            o = " ".join(seq[ii:ii + 1])
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
                seq += " " + _s
            res.append(seq)

        if be:
            decs = []
            init_set = [be.base_decode(x) for x in init_set]
            for gg in res:
                decs.append(be.base_decode(gg))
            res = decs

        return res, init_set

    def compute_states_entropy(self, be=None):
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