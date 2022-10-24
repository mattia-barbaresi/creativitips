import re


class Parser:
    """Class for PARSER: A Model for Word Segmentation
    Pierre Perruchet and Annie Vinter, 1998"""

    def __init__(self, ulen=None, memory=None):
        if memory is None:
            memory = dict()
        if ulen is None:
            ulen = [2]
        self.mem = memory
        self.ulens = ulen

    # def get_units(self, pct, thr=1.0):
    #     # list of units filtered with thr and then ordered by length
    #     mems = sorted([x for x, y in self.mem.items() if y >= thr], key=lambda item: len(item), reverse=True)
    #     us = self._get_unit_recursive(mems, pct)
    #     # orders units
    #     us = sorted(us, key=lambda x: pct.find(x))
    #     return us
    #
    # def _get_unit_recursive(self, mem_items, pp):
    #     res = []
    #     if len(pp) <= 2:
    #         res.append(pp)
    #     else:
    #         for s in mem_items:  # for each piece in mem
    #             if s in pp:  # if the piece is in percept
    #                 for pc in pp.split(s):
    #                     if len(pc) <= 2:
    #                         if len(pc) > 0:
    #                             res.append(pc)
    #                     else:
    #                         for u in self._get_unit_recursive(mem_items, pc):
    #                             res.append(u)
    #                 res.append(s)
    #                 break
    #         # if no units were found
    #         # if res = [] -> no chunks for pp
    #         # if res = [c] -> either c is pp or c is a sub of pp
    #         if len("".join(res)) != len(pp):
    #             # create new components (bi-grams)
    #             if len(res) == 0:
    #                 # splits bi-grams
    #                 lst = [pp[_i:_i + 2] for _i in range(0, len(pp), 2)]
    #                 res = res + lst
    #             else:
    #                 for c in re.split("|".join(res), pp):
    #                     if c:
    #                         # splits bi-grams
    #                         lst = [c[_i:_i + 2] for _i in range(0, len(c), 2)]
    #                         res = res + lst
    #     return res

    def read_percept(self, rng, sequence, threshold=1.0):
        """Return next percept in sequence as an ordered array of units in mem or components (bigrams)"""
        res = []
        # number of units embedded in next percepts
        i = rng.integers(low=1, high=4)
        s = sequence
        while len(s) > 0 and i != 0:
            units_list = [k for k,v in self.mem.items() if v > threshold and len(k) > 1 and " ".join(s).startswith(k)]
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
                self.mem[u] += weight/2
            else:
                self.mem[u] = weight
        # add weight to percept
        if pct in self.mem:
            self.mem[pct] += weight
        else:
            self.mem[pct] = weight

    def forget_interf(self, rng, pct, comps=None, forget=0.05, interfer=0.005):
        # forgetting
        self.mem.update((k, v - forget) for k, v in self.mem.items() if k != pct)

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
                        self.mem[k] -= interfer
        # cleaning
        for key in [k for k,v in self.mem.items() if v <= 0.0]:
            self.mem.pop(key)

    def encode(self, p, units, weight=1.0):
        # add entire percept
        if len(p.strip().split(" ")) <= max(self.ulens):
            # p is a unit, a primitive
            if p in self.mem:
                self.mem[p] += weight / 2
            else:
                self.mem[p] = weight
        else:
            self.add_weight(p, comps=units, weight=weight)
