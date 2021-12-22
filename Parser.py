import fnmatch
import os

from matplotlib import pyplot as plt
from sklearn import preprocessing

import utils
import re
import numpy as np


class Parser:
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

    def add_weight(self, pct, comps=None, weight=1.0):
        # add weight to units
        for u in comps:
            if u in self.mem:
                self.mem[u] += weight/2
            else:
                self.mem[u] = weight
            print("u: ", u, 0.5)

        if pct in self.mem:
            self.mem[pct] += weight
        else:
            self.mem[pct] = weight
        print("pct: ", pct, weight)

    def forget_interf(self, pct, comps=None, forget=0.05, interfer=0.005):
        # forgetting
        self.mem.update((k, v - forget) for k, v in self.mem.items() if k != pct)

        # interference
        for s in comps:
            # decompose in units
            for s2 in [s[_i:_i + 2] for _i in range(0, len(s), 2)]:
                for k, v in self.mem.items():
                    if s2 in k and k != pct and k not in comps:
                        self.mem[k] -= interfer
                        print("k: ", k, -interfer)
        # cleaning
        for key in [k for k, v in self.mem.items() if v <= 0.0]:
            self.mem.pop(key)


if __name__ == "__main__":
    np.random.seed(19)
    pars = Parser()
    w = 1.0
    f = 0.05
    i = 0.005
    with open("data/input.txt", "r") as fp:
        sequences = [line.rstrip() for line in fp]
    # load bicinia
    seq1 = []
    seq2 = []
    ss= set()
    for file in os.listdir("data/bicinia"):
        if fnmatch.fnmatch(file, "*.mid.txt"):
            with open("data/bicinia/" + file, "r") as fp:
                lines = fp.readlines()
                a = lines[0].strip().split(" ")
                seq1.append(a)
                ss.update(a)
                # lines[1] is empty
                b = lines[2].strip().split(" ")
                seq2.append(b)
    utils.base_fit(ss)
    sequences = ["".join([utils.base_encode(y) for y in x]) for x in seq1]
    # sequences = utils.generate_Saffran_sequence()

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
    ord_mem = dict(sorted([(utils.base_decode(x), y) for x, y in pars.mem.items()], key=lambda item: item[1], reverse=True))
    plt.bar(range(len(ord_mem)), list(ord_mem.values()), align='center')
    plt.gcf().autofmt_xdate()
    plt.xticks(range(len(ord_mem)), list(ord_mem.keys()))
    plt.show()

