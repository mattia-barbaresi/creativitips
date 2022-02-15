import utils
import numpy as np
from pparser import ParserModule

if __name__ == "__main__":
    rng = np.random.default_rng(77)
    threshold = 1.0
    w = 1.0
    f = 0.05
    i = 0.005

    # load input
    # with open("data/input.txt", "r") as fp:
    #     sequences = [list(line.strip()) for line in fp]

    n_iter = 1
    tot_mem = {}

    for bb in range(0, n_iter):
        pars = ParserModule()
        sequences = utils.generate_Saffran_sequence(rng)
        # initialise syllables
        syllables = set()
        for sq in sequences:
            syllables.update([" ".join(sq[_i:_i + 2]) for _i in range(0, len(sq), 2)])
        for sl in syllables:
            pars.mem[sl] = w

        for s in sequences:
            while len(s) > 0:
                # read percept as an array of units
                units = utils.read_percept(rng, dict((k, v) for k, v in pars.mem.items() if v >= threshold), s)[0]
                p = " ".join(units)
                print("units: ", units, " -> ", p)
                # add entire percept
                if len(p.strip().split(" ")) <= 2:
                    # p is a unit, a primitive
                    if p in pars.mem:
                        pars.mem[p] += w / 2
                    else:
                        pars.mem[p] = w
                else:
                    pars.add_weight(p, comps=units, weight=w)
                # forgetting and interference
                pars.forget_interf(rng, p, comps=units, forget=f, interfer=i)
                s = s[len(p.strip().split(" ")):]

        for k, v in pars.mem.items():
            if k in tot_mem:
                tot_mem[k] += v
            else:
                tot_mem[k] = v

    # calculate mean
    for k, v in tot_mem.items():
        tot_mem[k] = v / n_iter
    ord_mem = dict(sorted([(x, y) for x, y in tot_mem.items()], key=lambda item: item[1], reverse=True))
    utils.plot_mem(ord_mem, fig_name="./data/parser_results/parser.png")
