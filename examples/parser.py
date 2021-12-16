import re

import matplotlib.pyplot as plt
import numpy as np


def get_units(mm, pct, thr=1.0):
    # list of units filtered with thr and then ordered by length
    mems = sorted([x for x, y in mm.items() if y >= thr], key=lambda item: len(item), reverse=True)
    us = _get_unit(mems, pct)
    # orders units
    us = sorted(us, key=lambda x: pct.find(x))
    return us


def _get_unit(mm, pp):
    res = []
    if len(pp) <= 2:
        res.append(pp)
    else:
        for s in mm:  # for each piece in mem
            if s in pp:  # if the piece is in percept
                for pc in pp.split(s):
                    if len(pc) <= 2:
                        if len(pc) > 0:
                            res.append(pc)
                    else:
                        for u in _get_unit(mm, pc):
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


def add_weight(mm, pct, comps=None, weight=1.0):
    # add weight to units
    for u in comps:
        if u in mm:
            mm[u] += 0.5
        else:
            mm[u] = 0.5
        print("u: ", u, 0.5)

    if pct in mm:
        mm[pct] += weight
    else:
        mm[pct] = weight
    print("pct: ", pct, 1)


def forget_interf(mm, pct, comps=None, forget=0.05, interfer=0.005):
    # forgetting
    mm.update((k, v - forget) for k, v in mm.items() if k != pct)

    # interference
    for s in comps:
        # decompose in units
        for s2 in [s[_i:_i + 2] for _i in range(0, len(s), 2)]:
            for k, v in mm.items():
                if s2 in k and k != pct and k not in comps:
                    mm[k] -= interfer
                    print("k: ", k, -interfer)
    # cleaning
    for key in [k for k, v in mm.items() if v <= 0.0]:
        mm.pop(key)


def read_percepts(seqs):
    words = ["babupu", "bupada", "dutaba", "patubi", "pidabu", "tutibu"]
    ss = ""
    prev = ""
    # for x in range(0, 910):  # strict criterion
    for x in range(0, 450):  # looser criterion
        ww = np.random.choice(words)
        # no repeated words in succession
        while ww == prev:
            ww = np.random.choice(words)
        prev = ww
        ss += ww
    seqs = [ss]
    # p = ["tuti","buduta","batu","tibupa","tu","bi"]  # for testing
    res = []
    for s in seqs:
        while len(s[::2]) > 1:
            # bi-grams at least
            ln = np.random.randint(low=1, high=4) * 2
            print("rand: ", ln / 2)
            res.append(s[:ln])
            s = s[ln:]
        if len(s) > 0:
            res.append(s)
    return res


if __name__ == "__main__":
    # for testing
    mm = {
        "bu": 1,
        "pa": 1,
        "ba": 1,
        "du": 1,
        "ta": 1,
        "tu": 1,
        "ti": 1,
        "da": 1,
        "bi": 1,
        "pu": 1,
        "pi": 1,

    }
    # add_weight(mm, "babupada",comps=["ba", "bupa", "da"], weight=1.0)
    # forget_interf(mm, "babupada", comps=["ba", "bupa", "da"])
    # print("mem:", list(sorted(mm.items(), key=lambda item: item[1], reverse=True)))

    # -------------------------------------------------------------------------------------

    np.random.seed(3)
    mem = dict()
    w = 1.0
    f = 0.05
    i = 0.005
    with open("../data/input.txt", "r") as fp:
        sequences = [line.rstrip() for line in fp]

    for p in read_percepts(sequences):
        print("-----------------------------------")
        print("percept: ", p)
        units = []
        if len(p) <= 2:
            # p is a unit, a primitive
            if p in mem:
                mem[p] += 0.5
            else:
                mem[p] = 1
        else:
            # add p and its components (bi-grams)
            units = get_units(mem, p)
            print("units: ", units)
            add_weight(mem, p, comps=units, weight=w)
        # apply forgetting and interference
        forget_interf(mem, p, comps=units, forget=f, interfer=i)
        print("mem:", list(sorted(mem.items(), key=lambda item: item[1], reverse=True)))
    ord_mem = dict(sorted([(x, y) for x, y in mem.items()], key=lambda item: item[1], reverse=True))
    plt.bar(range(len(ord_mem)), list(ord_mem.values()), align='center')
    plt.gcf().autofmt_xdate()
    plt.xticks(range(len(ord_mem)), list(ord_mem.keys()))
    plt.show()
