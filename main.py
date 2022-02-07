import time
from datetime import datetime
import json
import os
import const
import utils
from classes.pparser import ParserModule
from classes.tps import TPSModule
import numpy as np

np.set_printoptions(linewidth=np.inf)
rng = np.random.default_rng(const.RND_SEED)

# file_names = \
#     ["input", "input2", "saffran", "thompson_newport", "reber", "all_songs_in_G", "all_irish-notes_and_durations"]

file_names = ["input"]

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
                root_dir = const.OUT_DIR + "{}_{}_({}_{}_{})_{}/"\
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
                        sequences = utils.load_bicinia_single("data/bicinia/", seq_n=2)
                    else:
                        with open("data/{}.txt".format(fn), "r") as fp:
                            # split lines char by char
                            sequences = [list(line.strip()) for line in fp]

                    # read percepts using parser function
                    actions = []
                    initial_set = set()
                    start_time = datetime.now()
                    for iter,(s1,s2) in enumerate(zip(sequences)):
                        old_p = ""
                        while len(s1) > 0:
                            print(" ------------------------------------------------------ ")
                            # read percept as an array of units
                            # active elements in mem shape perception
                            active_mem = dict((k, v) for k, v in pars.mem.items() if v >= const.MEM_THRES)
                            # active_mem = dict((k, v) for k, v in pars.mem.items() if v >= 0.5)
                            units, action = utils.read_percept(rng, active_mem, s1, ulens=units_len, tps=tps_1)
                            actions.append(action)
                            p = "".join(units)
                            tps_1.encode(old_p + p)
                            # save past for tps
                            old_p = p[-tps_order:]
                            print("units: ", units, " -> ", p)
                            # add entire percept
                            if len(p) <= max(units_len):
                                # p is a unit, a primitive
                                if p in pars.mem:
                                    pars.mem[p] += const.WEIGHT / 2
                                else:
                                    pars.mem[p] = const.WEIGHT
                            else:
                                tps_units.encode(units)
                                pars.add_weight(p, comps=units, weight=const.WEIGHT)
                            # forgetting and interference
                            pars.forget_interf(rng, p, comps=units, forget=const.FORGETTING, interfer=const.INTERFERENCE, ulens=units_len)
                            tps_units.forget(units, forget=const.FORGETTING)
                            s = s[len(p):]

                # dc = fc.distributional_context(fc_seqs, 3)
                # # print("---- dc ---- ")
                # # pp.pprint(dc)
                # classes = fc.form_classes(dc)
                # class_patt = fc.classes_patterns(classes, fc_seqs)

                # normilizes memories
                tps_1.normalize()
                tps_units.normalize()

                # generate sample sequences
                decoded = []
                gens = tps_units.generate_new_sequences(rng, min_len=100)
                print("gens: ", gens)


                # save all
                with open(out_dir + "action.json", "w") as of:
                    json.dump(actions,of)
                utils.plot_actions(actions, path=out_dir)

                # print(tps_units.mem)
                # utils.plot_gra(tps_units.mem)
                utils.plot_gra_from_normalized(tps_units.norm_mem, filename=out_dir + "tps_units", be=base_encoder)
                utils.plot_gra_from_normalized(tps_1.norm_mem,  filename=out_dir + "tps_1", be=base_encoder)
                # plot memeory chunks
                # for "bicinia" and "all_irish_notes_and_durations" use base_decode
                o_mem = dict(sorted([(base_encoder.base_decode(x),y) for x,y in pars.mem.items()], key=lambda it: it[1], reverse=True))
                utils.plot_mem(o_mem, out_dir + "words_plot.png", save_fig=True, show_fig=True)
