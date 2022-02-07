import re


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

    def read_percept(self, rng, sequence):
        """Return next percept in sequence as an ordered array of units in mem or components (bigrams)"""
        res = []
        # number of units embedded in next percepts
        i = rng.randint(low=1, high=4)
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

    def forget_interf(self, rng, pct, comps=None, forget=0.05, interfer=0.005, ulens=None):
        if not ulens:
            ulens = [2]
        # forgetting
        self.mem.update((k, v - forget) for k, v in self.mem.items() if k != pct)

        # interference
        for s in comps:
            # decompose in units
            if len(s) > max(ulens):
                _i = 0
                uts = []
                while _i <= len(s):
                    ul = rng.choice(ulens)
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
