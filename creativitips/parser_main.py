import os

import const
import utils
import numpy as np
from pparser import Parser

if __name__ == "__main__":
    out_dir = const.OUT_DIR + "parser_results/"
    rng = np.random.default_rng(33)
    threshold = 1.0
    w = 1.0
    i = 0.005

    # load input
    # with open("data/input.txt", "r") as fp:
    #     sequences = [list(line.strip()) for line in fp]

    # num of repeated experiments
    n_iter = 1
    tot_mem = {}

    for bb in range(0, n_iter):
        pars = Parser()
        sequences = utils.generate_Saffran_sequence(rng)
        sequences = utils.read_sequences(rng, "thompson_newport")
        # initialise syllables
        # pars.init_syllables(sequences, w)

        for s in sequences:
            while len(s) > 0:
                # read percept as an array of units
                # units = utils.read_percept(rng, dict((k, v) for k, v in pars.mem.items() if v["weight"] >= threshold), s)[0]
                units = pars.read_percept(rng, s, threshold)
                p = " ".join(units)
                print("units: ", units, " -> ", p)
                pars.encode(p, units, weight=w)
                # forgetting and interference
                pars.forget_interf(rng, p, comps=units, interfer=i)
                s = s[len(p.strip().split(" ")):]

        for k, v in pars.mem.items():
            if k in tot_mem.keys():
                tot_mem[k] += v["weight"]
            else:
                tot_mem[k] = v["weight"]

    # calculate mean
    for k, v in tot_mem.items():
        tot_mem[k] = v / n_iter
    ord_mem = dict(sorted([(x, y) for x, y in tot_mem.items()], key=lambda item: item[1], reverse=True)[:30])
    os.makedirs(out_dir, exist_ok=True)
    utils.plot_mem(ord_mem, fig_name=out_dir + "parser_exp_forg.png", save_fig=True, show_fig=False)
