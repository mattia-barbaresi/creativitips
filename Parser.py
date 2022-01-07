import fnmatch
import os
from matplotlib import pyplot as plt
from sklearn import preprocessing
import const
import utils
import re
import numpy as np


class ParserModule:
    """Class for PARSER"""
    def __init__(self, memory=None):
        if memory is None:
            memory = dict()
        self.mem = memory

    def get_units(self, pct, thr=1.0):
        # list of units filtered with thr and then ordered by length
        mems = sorted([x for x, y in self.mem.items() if y >= thr], key=lambda item: len(item), reverse=True)
        us = self._get_unit(mems, pct)
        # orders units
        us = sorted(us, key=lambda x: pct.find(x))
        return us

    def _get_unit(self, mem_items, pp):
        res = []
        if len(pp) <= 2:
            res.append(pp)
        else:
            for s in mem_items:  # for each piece in mem
                if s in pp:  # if the piece is in percept
                    for pc in pp.split(s):
                        if len(pc) <= 2:
                            if len(pc) > 0:
                                res.append(pc)
                        else:
                            for u in self._get_unit(mem_items, pc):
                                res.append(u)
                    res.append(s)
                    break
            # if no units were found
            # if res = [] -> no chunks for pp
            # if res = [c] -> either c is pp or c is a sub of pp
            if len("".join(res)) != len(pp):
                # create new components (bi-grams)
                if len(res) == 0:
                    # splits bi-grams
                    lst = [pp[_i:_i + 2] for _i in range(0, len(pp), 2)]
                    res = res + lst
                else:
                    for c in re.split("|".join(res), pp):
                        if c:
                            # splits bi-grams
                            lst = [c[_i:_i + 2] for _i in range(0, len(c), 2)]
                            res = res + lst
        return res

    def read_percept(self, sequence):
        """Return next percept in sequence as an ordered array of units in mem or components (bigrams)"""
        res = []
        # number of units embedded in next percepts
        i = np.random.randint(low=1, high=4)
        s = sequence
        while len(s) > 0 and i != 0:
            units_list = [k for k,v in self.mem.items() if v > 0.9 and len(k) > 1 and s.startswith(k)]
            if units_list:
                # a unit in mem matched
                unit = sorted(units_list, key=lambda item: len(item), reverse=True)[0]
                print("unit shape perception:", unit)
            else:
                return res
            res.append(unit)
            s = s[len(unit):]
            i -= 1
        return res

    def add_weight(self, pct, comps=None, weight=1.0):
        # add weight to units
        for u in comps:
            if u in self.mem:
                self.mem[u] += weight/2
                # print("u: ", u, weight/2)
            else:
                self.mem[u] = weight
                # print("u: ", u, weight)

        if pct in self.mem:
            self.mem[pct] += weight/2
        else:
            self.mem[pct] = weight
        # print("pct: ", pct, weight)

    def forget_interf(self, pct, comps=None, forget=0.05, interfer=0.005, ulens=[2]):
        # forgetting
        self.mem.update((k, v - forget) for k, v in self.mem.items() if k != pct)

        # interference
        for s in comps:
            # decompose in units
            if len(s) > max(ulens):
                _i = 0
                uts = []
                while _i <= len(s):
                    ul = np.random.choice(ulens)
                    uts.append(s[_i:_i + ul])
                    _i += ul
            else:
                uts = [s]
            # uts = [s[_i:_i + 2] for _i in range(0, len(s), 2)]
            for s2 in uts:
                for k, v in self.mem.items():
                    if s2 in k and k != pct and k not in comps:
                        self.mem[k] -= interfer
                        # print("k: ", k, -interfer)
        # cleaning
        for key in [k for k, v in self.mem.items() if v <= 0.0]:
            self.mem.pop(key)


if __name__ == "__main__":
    np.random.seed(19)
    pars = Parser()
    w = const.WEIGHT
    f = const.FORGETTING
    i = const.INTERFERENCE

    # load input
    # with open("data/input.txt", "r") as fp:
    #     sequences = [line.rstrip() for line in fp]

    # load bicinia
    # sequences = utils.load_bicinia_single("data/bicinia/")

    sequences = utils.generate_Saffran_sequence()

    for s in sequences:
        while len(s) > 0:
            # read percept as an array of units
            units = utils.read_percept(dict((k,v) for k,v in pars.mem.items() if v >= 1.0), s)
            p = "".join(units)
            print("units: ", units, " -> ", p)
            # add entire percept
            if len(p) <= 2:
                # p is a unit, a primitive
                if p in pars.mem:
                    pars.mem[p] += w/2
                else:
                    pars.mem[p] = w
            else:
                pars.add_weight(p, comps=units, weight=w)
            # forgetting and interference
            pars.forget_interf(p, comps=units, forget=f, interfer=i)
            s = s[len(p):]
    ord_mem = dict(sorted([(x, y) for x, y in pars.mem.items()], key=lambda item: item[1], reverse=True))
    # for bicinia use base_decode
    # ord_mem = dict(sorted([(utils.base_decode(x),y) for x,y in pars.mem.items()], key=lambda it: it[1], reverse=True))
    plt.rcParams["figure.figsize"] = (15, 7)
    utils.plot_mem(ord_mem, save_fig=False)

