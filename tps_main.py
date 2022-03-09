from datetime import datetime
import json
import os
import time
import numpy as np
from sklearn import preprocessing
import const
import utils
from classes.computation import ComputeModule, GraphModule, EmbedModule

if __name__ == "__main__":
    np.set_printoptions(linewidth=np.inf)
    rng = np.random.default_rng(const.RND_SEED)

    # file_names = \
    #     ["input", "input2", "saffran", "thompson_newport", "reber", "all_songs_in_G",
    #     "all_irish-notes_and_durations","cello", "bach_compact"]

    file_names = ["input"]

    # maintaining INTERFERENCES/FORGETS separation by a factor of 10
    interferences = [0.005]
    forgets = [0.05]
    thresholds_mem = [0.9]
    tps_orders = [2]
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
                        out_dir = root_dir + "{}/".format(fn)
                        os.makedirs(out_dir, exist_ok=True)
                        results = dict()

                        # --------------- INPUT ---------------
                        sequences = utils.read_sequences(fn, rng)

                        # read percepts using parser function
                        start_time = datetime.now()

                        # init module for computation
                        cm = ComputeModule(rng, order=order, weight=const.WEIGHT, interference=interf, forgetting=fogs,
                                           memory_thres=t_mem, unit_len=const.ULENS, method=method)

                        for iteration, s in enumerate(sequences):
                            first_in_seq = True
                            while len(s) > 0:
                                # --------------- COMPUTE ---------------
                                # compute next percept
                                p, units = cm.compute(s, first_in_seq)
                                first_in_seq = False
                                # update s
                                s = s[len(p.strip().split(" ")):]
                            cm.compute(["END"], first_in_seq=False, last_in_seq=True)
                            # --------------- GENERATE ---------------
                            # if iteration % 5 == 1:
                            #     cm.tps_units.normalize()
                            #     results[iteration] = dict()
                            #     results[iteration]["generated"] = cm.tps_units.generate_new_sequences(rng,min_len=100)
                            #     im = dict(sorted([(x, y) for x, y in cm.pars.mem.items()],
                            #                      key=lambda it: it[1], reverse=True))
                            #     results[iteration]["mem"] = im

                                # class form on graph
                                # graph = GraphModule(cm.tps_units)
                                # cl_form = {i:x for i,x in enumerate(graph.sim_rank())}
                                # print("cl: ", cl_form)

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
                        # emb_le = preprocessing.LabelEncoder()
                        # emb_le.fit(list(cm.tps_units.le_rows.classes_) + list(cm.tps_units.le_cols.classes_))
                        # last_nodes = set(cm.tps_units.le_cols.classes_) - set(cm.tps_units.le_rows.classes_)
                        # nx, ny = cm.tps_units.norm_mem.shape
                        # emb_matrix = np.zeros((len(emb_le.classes_), len(emb_le.classes_)))
                        # for x in range(nx):
                        #     for y in range(ny):
                        #         emb_matrix[emb_le.transform(cm.tps_units.le_rows.inverse_transform([x]))[0]] \
                        #             [emb_le.transform(cm.tps_units.le_cols.inverse_transform([y]))[0]] = \
                        #             cm.tps_units.norm_mem[x][y]
                        # for lbl in last_nodes:
                        #     emb_matrix[emb_le.transform([lbl])[0]] = np.ones(len(emb_le.classes_)) * 0.5
                        # wem = EmbedModule(emb_matrix)
                        # wem.compute(emb_le)

                        # calculate states uncertainty
                        cm.tps_1.compute_states_entropy()
                        cm.tps_units.compute_states_entropy()

                        # generate sample sequences
                        gens = cm.tps_units.generate_new_sequences(rng, min_len=100)
                        print("gens: ", gens)

                        print("REGENERATION")
                        # for g in gens:
                        #     print(cm.tps_1.get_units_brent(g))

                        # save results
                        with open(out_dir + "results.json", "w") as of:
                            json.dump(results, of)
                        # save generated
                        with open(out_dir + "generated.json", "w") as of:
                            json.dump(gens, of)
                        # save actions
                        with open(out_dir + "action.json", "w") as of:
                            json.dump(cm.actions, of)
                        utils.plot_actions(cm.actions, path=out_dir, show_fig=False)
                        # print(tps_units.mem)
                        # utils.plot_gra(tps_units.mem)
                        print("plotting tps units...")
                        utils.plot_gra_from_normalized(cm.tps_units, filename=out_dir + "tps_units", render=True)
                        print("plotting tps all...")
                        utils.plot_gra_from_normalized(cm.tps_1, filename=out_dir + "tps_symbols", render=True)
                        print("plotting memory...")
                        # plot memory chunks
                        om = dict(sorted([(x, y) for x, y in cm.pars.mem.items()], key=lambda it: it[1], reverse=True))
                        utils.plot_mem(om, out_dir + "words_plot.png", save_fig=True, show_fig=False)

                        graph = GraphModule(cm.tps_units)
                        # graph.form_classes()
                        # graph.draw_graph(out_dir + "nxGraph.dot")
                        cl_form = graph.sim_rank()
                        print("class form: ", cl_form)
                        with open(out_dir + "classes.json", "w") as of:
                            json.dump(list(cl_form), of)
                        # graph.get_communities()
