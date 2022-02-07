from datetime import datetime
import json
import os
import time
import numpy as np
import const
import utils
from classes.computation import ComputeModule, GraphModule, EmbedModule

if __name__ == "__main__":
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
                        # init module for computation
                        cm = ComputeModule(rng, order=order, weight=const.WEIGHT, interference=interf, forgetting=fogs,
                                           memory_thres=t_mem, unit_len=const.ULENS, method=method)
                        for iteration, s in enumerate(sequences):

                            cm.compute(s)

                            # generation
                            if iteration % 10 == 1:
                                cm.tps_units.normalize()
                                results[iteration] = dict()
                                results[iteration]["generated"], results[iteration]["initials"] = \
                                    cm.tps_units.generate_new_sequences(rng, min_len=100, initials=initial_set)
                                im = dict(sorted([(x, y) for x, y in cm.pars.mem.items()],
                                                 key=lambda it: it[1], reverse=True))
                                results[iteration]["mem"] = im

                        # dc = fc.distributional_context(fc_seqs, 3)
                        # # print("---- dc ---- ")
                        # # pp.pprint(dc)
                        # classes = fc.form_classes(dc)
                        # class_patt = fc.classes_patterns(classes, fc_seqs)

                        results["processing time"] = str((datetime.now() - start_time).total_seconds())
                        # normalizes memories
                        cm.tps_1.normalize()
                        cm.tps_units.normalize()

                        # embedding chunks
                        wem = EmbedModule(cm.tps_units.norm_mem)
                        wem.compute(cm.tps_units.le_rows)

                        # calculate states uncertainty
                        cm.tps_1.compute_states_entropy()
                        cm.tps_units.compute_states_entropy()

                        # generate sample sequences
                        print("initials: ", sorted(initial_set))
                        # gens, init = tps_units.generate_new_sequences(min_len=100, initials=initial_set)
                        gens, init = cm.tps_units.generate_new_sequences(rng, min_len=100, initials=initial_set)
                        print("init set: ", init)
                        print("gens: ", gens)

                        print("REGENERATION")
                        for g in gens:
                            print(cm.tps_1.get_units_brent(g))

                        # save results
                        with open(out_dir + "results.json", "w") as of:
                            json.dump(results, of)
                        # save generated
                        with open(out_dir + "generated.json", "w") as of:
                            json.dump(gens, of)
                        # save actions
                        with open(out_dir + "action.json", "w") as of:
                            json.dump(actions,of)
                        utils.plot_actions(actions,path=out_dir, show_fig=False)
                        # print(tps_units.mem)
                        # utils.plot_gra(tps_units.mem)
                        print("plotting tps units...")
                        utils.plot_gra_from_normalized(cm.tps_units, filename=out_dir + "tps_units")
                        # print("plotting tps all...")
                        utils.plot_gra_from_normalized(cm.tps_1, filename=out_dir + "tps_symbols")
                        print("plotting memory...")
                        # plot memory chunks
                        om = dict(sorted([(x, y) for x, y in cm.pars.mem.items()], key=lambda it: it[1], reverse=True))
                        utils.plot_mem(om, out_dir + "words_plot.png", save_fig=True, show_fig=False)

                        # graph = GraphModule(cm.tps_units)
                        # graph.form_classes()
                        # graph.draw_graph(out_dir + "nxGraph.dot")
                        # # graph.print_values()
                        # graph.get_communities()
