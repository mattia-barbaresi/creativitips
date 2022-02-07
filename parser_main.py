from matplotlib import pyplot as plt
import const
import utils
import numpy as np
from pparser import ParserModule

if __name__ == "__main__":
    rng = np.random.default_rng(const.RND_SEED)
    pars = ParserModule()
    w = const.WEIGHT
    f = const.FORGETTING
    i = const.INTERFERENCE

    # load input
    # with open("data/input.txt", "r") as fp:
    #     sequences = [line.rstrip() for line in fp]

    # load bicinia
    # sequences = utils.load_bicinia_single("data/bicinia/")

    sequences = utils.generate_Saffran_sequence(rng)

    for s in sequences:
        while len(s) > 0:
            # read percept as an array of units
            units = utils.read_percept(rng, dict((k,v) for k,v in pars.mem.items() if v >= const.MEM_THRES), s)[0]
            p = " ".join(units)
            print("units: ", units, " -> ", p)
            # add entire percept
            if len(p.strip().split(" ")) <= 2:
                # p is a unit, a primitive
                if p in pars.mem:
                    pars.mem[p] += w/2
                else:
                    pars.mem[p] = w
            else:
                pars.add_weight(p, comps=units, weight=w)
            # forgetting and interference
            pars.forget_interf(rng, p, comps=units, forget=f, interfer=i)
            s = s[len(p.strip().split(" ")):]
    ord_mem = dict(sorted([(x, y) for x, y in pars.mem.items()], key=lambda item: item[1], reverse=True))
    plt.rcParams["figure.figsize"] = (15, 7)
    utils.plot_mem(ord_mem, save_fig=False)
