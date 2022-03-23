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
#     ["input", "input2", "saffran", "thompson_newport", "reber", "all_songs_in_G",
#     "all_irish-notes_and_durations","cello", "bach_compact"]

file_name = "bach_compact"

# maintaining INTERFERENCES/FORGETS separation by a factor of 10
interferences = [0.005]
forgets = [0.05]
thresholds_mem = [1.0]
tps_orders = [1]
method = "BRENT"

for order in tps_orders:
    for interf in interferences:
        for fogs in forgets:
            for t_mem in thresholds_mem:
                # init
                root_dir = const.OUT_DIR + "{}_({}_{}_{})_{}/" \
                    .format(order, t_mem, fogs, interf, time.strftime("%Y%m%d-%H%M%S"))
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

                print("processing {} series ...".format(file_name))
                # init
                pars = ParserModule()
                tps_units = TPSModule(1)  # memory for TPs inter
                tps_1 = TPSModule(order)  # memory for TPs intra
                out_dir = root_dir + "{}/".format(file_name)
                os.makedirs(out_dir, exist_ok=True)
                results = dict()
                # --------------- INPUT ---------------
                sequences = utils.load_bach_separated("data/" + file_name + "/")

                # read percepts using parser function
                start_time = datetime.now()
                mim = MIModule()
                cm1 = ComputeModule(rng, order=order, weight=const.WEIGHT, interference=interf, forgetting=fogs,
                                    mem_thres=t_mem, unit_len=const.ULENS, method=method)
                cm2 = ComputeModule(rng, order=order, weight=const.WEIGHT, interference=interf, forgetting=fogs,
                                    mem_thres=t_mem, unit_len=const.ULENS, method=method)
                for itr, (s1, s2) in enumerate(zip(sequences[0], sequences[1])):
                    first_in_seq = True
                    while len(s1) > 0 and len(s2) > 0:
                        p1, units1 = cm1.compute(s1, first_in_seq)
                        p2, units2 = cm2.compute(s2, first_in_seq)
                        s1 = s1[len(p1.strip().split(" ")):]
                        s2 = s2[len(p2.strip().split(" ")):]
                        first_in_seq = False
                        mim.encode(units1, units2)

                    if itr % 5 == 1:
                        cm1.tps_units.normalize()
                        cm2.tps_units.normalize()
                        results[itr] = dict()
                        results[itr]["generated"] = utils.multi_generation(rng, cm1, cm2, mim)

                with open(out_dir + "multi_results.json","w") as op:
                    json.dump(results, op)
                cm1.tps_units.normalize()
                cm1.tps_units.compute_states_entropy()
                cm2.tps_units.normalize()
                cm2.tps_units.compute_states_entropy()

                print("plotting tps units...")
                utils.plot_gra_from_normalized(cm1.tps_units, filename=out_dir + "tps1")

                print("plotting tps all...")
                utils.plot_gra_from_normalized(cm2.tps_units, filename=out_dir + "tps2")
                print("plotting memory...")
                om1 = dict(sorted([(x, y) for x, y in cm1.pars.mem.items()], key=lambda it: it[1], reverse=True))
                utils.plot_mem(om1, out_dir + "words_plot.png", save_fig=True, show_fig=False)
                om2 = dict(sorted([(x, y) for x, y in cm2.pars.mem.items()], key=lambda it: it[1], reverse=True))
                utils.plot_mem(om2, out_dir + "words_plot.png", save_fig=True, show_fig=False)

                mim.draw_graph()
                # mim.draw_graph2(GraphModule(cm1.tps_units),GraphModule(cm2.tps_units))
