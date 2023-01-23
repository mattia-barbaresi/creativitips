import math
import re

import const
import utils


class Parser:
    """A class implementing PARSER: A Model for Word Segmentation
    Pierre Perruchet and Annie Vinter, 1998"""

    def __init__(self, ulen=None):
        if ulen is None:
            ulen = [2]
        self.mem = dict()
        self.ulens = ulen
        # for exponential forgetting
        self.time = 0

    def init_syllables(self, sequences, weight):
        syllables = set()
        for sq in sequences:
            syllables.update([" ".join(sq[_i:_i + 2]) for _i in range(0, len(sq), 2)])
        for sl in syllables:
            self.mem[sl] = weight

    def read_percept(self, rng, sequence, threshold=1.0):
        """Return next percept in sequence as an ordered array of units in mem or components (bigrams)"""
        res = []
        # number of units embedded in next percepts
        i = rng.integers(low=1, high=4)
        s = sequence
        while len(s) > 0 and i != 0:
            units_list = [k for k,v in self.mem.items() if v["weight"] >= threshold and len(k) > 1 and " ".join(s).startswith(k)]
            if units_list:
                # a unit in mem matched
                unit = (sorted(units_list, key=lambda item: len(item), reverse=True)[0]).strip().split(" ")
                print("unit shape perception:", unit)
            else:
                unit = s[:rng.choice(self.ulens)]

            # check if last symbol
            sf = s[len(unit):]
            if len(sf) == 1:
                unit += sf
                print("added last:", unit)
            res.append(" ".join(unit))
            # print("final unit:", unit)
            s = s[len(unit):]
            i -= 1

        return res

    def add_weight(self, pct, comps=None, weight=1.0):
        # add weight to units
        for u in comps:
            if u in self.mem:
                self.mem[u]["weight"] += weight/2
            else:
                self.mem[u] = dict()
                self.mem[u]["weight"] = weight
                self.mem[u]["t"] = self.time
        # add weight to  entire percept (chunking)
        if len(comps) > 1:
            if pct in self.mem:
                self.mem[pct]["weight"] += weight
            else:
                self.mem[pct] = dict()
                self.mem[pct]["weight"] = weight
                self.mem[pct]["t"] = self.time

    def encode(self, p, units, weight=1.0):
        self.time += 1
        # add entire percept
        if len(p.strip().split(" ")) <= max(self.ulens):
            # p is a unit, a primitive
            if p in self.mem:
                self.mem[p]["weight"] += weight / 2
                # N.B. forgetting
                self.mem[p]["t"] = self.time
            else:
                self.mem[p] = dict()
                self.mem[p]["weight"] = weight
                self.mem[p]["t"] = self.time
        else:
            self.add_weight(p, comps=units, weight=weight)

    # calculate exponential forgetting
    def calculateExp(self, init_time, s=const.STM_DECAY_RATE):
        # r = e ^ (-t / s)
        # s = memory stability
        # return 0.05
        return math.exp(- (self.time - init_time) / s) / const.PARSER_MEM_C

    def forget_interf(self, rng, pct, comps=None, interfer=0.005):
        # forgetting
        for k, v in self.mem.items():
            if k != pct:
                self.mem[k]["weight"] -= self.calculateExp(v["t"])

        # interference
        uts = []
        for s in comps:
            # decompose in units
            if len(s.split(" ")) > max(self.ulens):
                ss = s.split(" ")
                _i = 0
                while _i < len(ss):
                    ul = rng.choice(self.ulens)
                    uts.append(" ".join(ss[_i:_i + ul]))
                    _i += ul
            else:
                uts.append(s)
        for s2 in list(uts):
            for k in self.mem.keys():
                if k != pct and k not in comps:
                    if s2 in k:
                        self.mem[k]["weight"] -= interfer
        # cleaning
        for key in [k for k,v in self.mem.items() if v["weight"] <= 0.0]:
            self.mem.pop(key)
