import time
from datetime import datetime
import json
import os
import const
import utils
from classes.pparser import ParserModule
from classes.tps import TPSModule
import numpy as np

from computation import ComputeModule
from multi_integration import MIModule

np.set_printoptions(linewidth=np.inf)
rng = np.random.default_rng(const.RND_SEED)

# file_names = \
#     ["input", "input2", "saffran", "thompson_newport", "reber", "all_songs_in_G", "all_irish-notes_and_durations"]

file_names = ["bicinia"]

# maintaining INTERFERENCES/FORGETS separation by a factor of 10
interferences = [0.005]
forgets = [0.05]
thresholds_mem = [0.95]
tps_orders = [1]
method = "BRENT"

for order in tps_orders:
    for interf in interferences:
        for fogs in forgets:
            for t_mem in thresholds_mem:
                # init
                root_dir = const.OUT_DIR + "{}_{}_({}_{}_{})_{}/" \
                    .format(method, order, t_mem, fogs, interf, time.strftime("%Y%m%d-%H%M%S"))
                # root_dir = const.OUT_DIR + "RND_(2-3)_({}_{}_{})_{}/".\
                #     format(t_mem, fogs, interf, time.strftime("%Y%m%d-%H%M%S"))

                os.makedirs(root_dir, exist_ok=True)
                with open(root_dir + "pars.txt", "w") as of:
                    json.dump({
                        "rnd": const.RND_SEED,
                        "w": const.WEIGHT,
                        "interference": interf,
                        "forgetting": fogs,
                        "mem thresh": t_mem,
                        "lens": const.ULENS,
                        # "tps_order": order,
                    }, of)

                for fn in file_names:
                    print("processing {} series ...".format(fn))
                    # init
                    pars = ParserModule()
                    tps_units = TPSModule(1)  # memory for TPs inter
                    tps_1 = TPSModule(order)  # memory for TPs intra
                    out_dir = root_dir + "{}/".format(fn)
                    os.makedirs(out_dir, exist_ok=True)
                    results = dict()
                    # --------------- INPUT ---------------
                    if fn == "saffran":
                        # load/generate Saffran input
                        sequences = utils.generate_Saffran_sequence(rng)
                    elif fn == "all_irish-notes_and_durations":
                        # read
                        # sequences = utils.load_irish_n_d_repeated("data/all_irish-notes_and_durations-abc.txt")
                        sequences = utils.load_irish_n_d("data/all_irish-notes_and_durations-abc.txt")
                    elif fn == "bicinia":
                        sequences = utils.load_bicinia_full("data/bicinia/")
                    else:
                        with open("data/{}.txt".format(fn), "r") as fp:
                            # split lines char by char
                            sequences = [list(line.strip()) for line in fp]

                    # read percepts using parser function
                    start_time = datetime.now()
                    mim = MIModule()
                    cm1 = ComputeModule(rng, order=order, weight=const.WEIGHT, interference=interf, forgetting=fogs,
                                        memory_thres=t_mem, unit_len=const.ULENS, method=method)
                    cm2 = ComputeModule(rng, order=order, weight=const.WEIGHT, interference=interf, forgetting=fogs,
                                        memory_thres=t_mem, unit_len=const.ULENS, method=method)
                    for iter, (s1, s2) in enumerate(zip(sequences[0], sequences[1])):
                        first_in_seq = True
                        while len(s1) > 0 and len(s2) > 0:
                            p1, units1, _ = cm1.compute(s1, first_in_seq)
                            s1 = s1[len(p1.strip().split(" ")):]
                            p2, units2, _ = cm2.compute(s2, first_in_seq)
                            s2 = s2[len(p2.strip().split(" ")):]
                            first_in_seq = False
                            mim.encode(units1,units2)
                    mim.draw_graph()
