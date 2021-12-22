import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing
from scipy.special import softmax
import const
import utils
from Parser import Parser


class TPS:
    """Implements a transitional probabilities module"""

    def __init__(self, order=1):
        self.mem = dict()
        self.norm_mem = []
        self.order = order
        self.prev = ""
        self.le_rows = preprocessing.LabelEncoder()
        self.le_cols = preprocessing.LabelEncoder()

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
        self.norm_mem = softmax(self.norm_mem, axis=1)
        # self.norm_mem = utils.softmax(self.norm_mem)

    def get_units(self, percept, ths=0.5):
        """
        Returns segmented percept using stored TPs.
        :param ths: segmentation threshold
        :param percept could be a string, or an (ordered, sequential) array of strings.
        In latter case TPs are counted between elements of the array(units), instead of between symbols
        """
        res = []
        tps_seqs = []
        if self.order > 0:
            for ii in range(self.order, len(percept)):
                h = "".join(percept[ii - self.order:ii])
                o = "".join(percept[ii:ii + 1])
                tps_seqs.append(self.norm_mem[self.le_rows.transform(h)][self.le_rows.transform(o)])
            start = 0
            for ind, tp in enumerate(tps_seqs):
                if tp < ths:  # insert a break
                    res.append(percept[start:ind])
                    start = ind
            res.append(percept[start:])
        else:
            print("(TPmodule):Order must be grater than 1.")

        return res

    def get_units_brent(self, percept, ths=0.5):
        """
        Returns segmented percept using stored TPs.
        (Brent 1999) a formalization of the original proposal by Saffarn, Newport et al.
        Consider the segment “wxyz”…whenever the statistical value (TPs or O\E) of the transitions under consideration
        is lower than the statistical values of its adjacent neighbors, a boundary is inserted.
        IF TPs (“xy”) < TPs(“wx”) and < TPs(“yz”)  segments between “x” e “y”
        :param ths: segmentation threshold
        :param percept could be a string, or an (ordered, sequential) array of strings.
        In latter case TPs are counted between elements of the array(units), instead of between symbols
        """
        res = []
        tps_seqs = []
        if self.order > 0:
            for ii in range(self.order, len(percept)):
                h = "".join(percept[ii - self.order:ii])
                o = "".join(percept[ii:ii + 1])
                tps_seqs.append(self.norm_mem[self.le_rows.transform(h)][self.le_rows.transform(o)])
            start = 0
            for ind, tp in enumerate(tps_seqs):
                if tp < ths:  # insert a break
                    res.append(percept[start:ind])
                    start = ind
            res.append(percept[start:])
        else:
            print("(TPmodule):Order must be grater than 1.")

        return res


if __name__ == "__main__":
    np.random.seed(19)
    pars = Parser()
    w = const.WEIGHT
    f = const.FORGETTING
    i = const.INTERFERENCE
    tps1 = TPS(1)  # memory for TPs

    # input
    # with open("data/input.txt", "r") as fp:
    #     sequences = [line.rstrip() for line in fp]
    sequences = utils.generate_Saffran_sequence()

    # read percepts using parser function
    for s in sequences:
        while len(s) > 0:
            # read percept as an array of units
            units = utils.read_percept(dict((k, v) for k, v in pars.mem.items() if v >= 1.0), s)
            p = "".join(units)

            print("units: ", units, " -> ", p)
            # add entire percept
            if len(p) <= 2:
                # p is a unit, a primitive
                if p in pars.mem:
                    pars.mem[p] += w / 2
                else:
                    pars.mem[p] = w
            else:
                tps1.encode(units)
                pars.add_weight(p, comps=units, weight=w)
            # forgetting and interference
            pars.forget_interf(p, comps=units, forget=f, interfer=i)
            s = s[len(p):]

    tps1.normalize()
    print(tps1.mem)
    # utils.plot_gra(tps1.mem)
    utils.plot_gra_from_m(tps1.norm_mem, ler=tps1.le_rows, lec=tps1.le_cols)
    ord_mem = dict(sorted([(x, y) for x, y in pars.mem.items()], key=lambda item: item[1], reverse=True))
    # for bicinia use base_decode
    # ord_mem = dict(sorted([(utils.base_decode(x), y) for x, y in pars.mem.items()], key=lambda item: item[1], reverse=True))
    plt.bar(range(len(ord_mem)), list(ord_mem.values()), align='center')
    plt.gcf().autofmt_xdate()
    plt.xticks(range(len(ord_mem)), list(ord_mem.keys()))
    plt.show()
