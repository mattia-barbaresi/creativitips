from datetime import datetime
import json
import os
import numpy as np
from matplotlib import pyplot as plt

import complexity
from creativitips import utils
from creativitips import const
from creativitips.CTPs.computation import Computation
from creativitips.CTPs.graphs import TPsGraph


# NOTES: more iterations over the same input enhance resulting model!
# def elaborate(file_in="", dir_out=""):
#     print("processing {} series ...".format(file_in))
#     # init
#     res = dict()
#     rnd_gen = np.random.default_rng(const.RND_SEED)
#     rout = dir_out + "/tps_results/"
#     os.makedirs(rout, exist_ok=True)
#     with open(file_in, "r") as fp:
#         seqs = [list(line.strip().split(" ")) for line in fp]
#     cm_m = Computation(rnd_gen)
#
#     for iter, s in enumerate(seqs):
#         fis = True
#         while len(s) > 0:
#             # compute next percept
#             p, units = cm_m.compute(s, first_in_seq=fis)
#             fis = False
#             # update s
#             s = s[len(p.strip().split(" ")):]
#         cm_m.compute_last()
#
#     cm_m.tps_units.normalize()
#     res[iter] = dict()
#     res[iter]["generated"] = cm_m.tps_units.generate_new_seqs(rng)
#     im = dict(sorted([(x, y) for x, y in cm_m.pars.mem.items()],
#                      key=lambda it: it[1], reverse=True))
#     res[iter]["mem"] = im
#
#     res["processing time"] = str((datetime.now() - start_time).total_seconds())
#     # calculate states uncertainty
#     cm_m.tps_1.compute_states_entropy()
#     cm_m.tps_units.compute_states_entropy()
#     # generalization
#     cm_m.generalize(rout)
#     # save result
#     with open(rout + "res.json", "w") as of:
#         json.dump(res, of)
#     # save generated
#     with open(rout + "generated.json", "w") as of:
#         json.dump(gens, of)
#

if __name__ == "__main__":

    np.set_printoptions(linewidth=np.inf)
    rng = np.random.default_rng(const.RND_SEED)

    file_names = ["input", "input2", "saffran", "thompson_newport",
                  "Onnis2003_L1_2","Onnis2003_L2_2","Onnis2003_L1_6","Onnis2003_L2_6",
                  "Onnis2003_L1_12","Onnis2003_L2_12","Onnis2003_L1_24","Onnis2003_L2_24",
                  "all_songs_in_G", "all_irish-notes_and_durations-abc", "bach_preludes", "ocarolan", "scottish"]

    # file_names = ["input", "input2", "saffran", "thompson_newport"]

    # maintaining INTERFERENCES/FORGETS separation by a factor of 10
    thresholds_mem = [1.0]
    interferences = [0.005]
    forgets = [0.05]
    tps_orders = [2]
    methods = ["FTP_NFNI_exp"]  # MI, CT or BRENT, FTP

    for tps_method in methods:
        for tps_order in tps_orders:
            for fogt in forgets:
                for interf in interferences:
                    for t_mem in thresholds_mem:
                        # init
                        root_out_dir = const.OUT_DIR + "tps_results_expf/" + \
                                       utils.params_to_string(tps_method, tps_order, fogt, interf, t_mem)
                        os.makedirs(root_out_dir, exist_ok=True)

                        with open(root_out_dir + "params.txt", "w") as of:
                            json.dump({
                                "rnd": const.RND_SEED,
                                "mem thresh": t_mem,
                                "forgetting": fogt,
                                "interference": interf,
                                "weight": const.WEIGHT,
                                "lens": const.ULENS,
                                "tps_order": tps_order,
                            }, of)

                        for fn in file_names:
                            print("processing {} series ...".format(fn))
                            # init
                            fi_dir = root_out_dir + "{}/".format(fn)

                            try:
                                os.makedirs(fi_dir, exist_ok=False)
                            except (Exception,):
                                raise

                            results = dict()
                            # --------------- INPUT ---------------
                            sequences = utils.read_sequences(rng, fn)

                            # read percepts using parser function
                            start_time = datetime.now()

                            # init module for computation
                            cm = Computation(rng, order=tps_order, weight=const.WEIGHT, interference=interf,
                                             forgetting=fogt, mem_thres=t_mem, unit_len=const.ULENS, method=tps_method)

                            for iteration, s in enumerate(sequences):
                                fis = True
                                while len(s) > 0:
                                    # --------------- COMPUTE ---------------
                                    # compute next percept
                                    p, units = cm.compute(s, first_in_seq=fis)
                                    fis = False
                                    # update s
                                    s = s[len(p.strip().split(" ")):]
                                cm.compute_last()

                                # --------------- GENERATE ---------------
                                # if iteration % 5 == 1:
                                #     results[iteration] = dict()
                                #     utils.generate_seqs(rng, cm, results[iteration])

                            # class form on graph
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
                            # wem = Embedding(emb_matrix)
                            # wem.compute(emb_le)

                            # calculate states uncertainty
                            cm.tps_1.compute_states_entropy()
                            cm.tps_units.compute_states_entropy()

                            # generate sample sequences
                            gens = cm.tps_units.generate_new_seqs(rng, min_len=100)
                            print("gens: ", gens)

                            # save results
                            with open(fi_dir + "results.json", "w") as of:
                                json.dump(results, of)
                            # save generated
                            with open(fi_dir + "generated.json", "w") as of:
                                json.dump(gens, of)
                            # save actions
                            with open(fi_dir + "action.json", "w") as of:
                                json.dump(cm.actions, of)
                            utils.plot_actions(cm.actions, path=fi_dir, show_fig=False)
                            # print(tps_units.mem)
                            # utils.plot_gra(tps_units.mem)

                            # save tps
                            with open(fi_dir + "tps_units.json", "w") as of:
                                json.dump(cm.tps_units.mem, of)

                            with open(fi_dir + "tps.json", "w") as of:
                                json.dump(cm.tps_1.mem, of)

                            graph = TPsGraph(cm.tps_units)
                            # graph.form_classes()
                            # graph.draw_graph(out_dir + "nxGraph.dot")
                            cl_form = graph.cf_by_sim_rank()
                            print("class form: ", cl_form)
                            with open(fi_dir + "classes.json", "w") as of:
                                json.dump(list(cl_form), of)
                            # graph.get_communities()

                            # generalization
                            # if input is a single array (as original saffran) the next command loops forever
                            # for the presence of cycles in the graph
                            gen_paths = cm.tps_units.generate_paths(rng, n_paths=20, min_len=50)
                            if gen_paths:
                                print("generalizing ...")
                                cm.generalize(fi_dir, gen_paths)
                            print("generating with generalized graph ...")
                            # generate sample sequences from generalized graph
                            gg_gens = cm.graph.generate_sequences(rng)
                            print("gg_gens: ", gg_gens)
                            # save generated
                            with open(fi_dir + "ggen.json", "w") as of:
                                json.dump(gg_gens, of)
                            # plot tps
                            # utils.plot_tps_sequences(cm, [" ".join(x) for x in sequences[:20]], fi_dir)
                            if gens:
                                comp_res = complexity.calculate_complexities(gens)
                                results["generated complexities"] = comp_res
                                print("---------- complexities:")
                                for it, vl in comp_res.items():
                                    print(it, vl)
                                print("----------")

                            print("plotting memory...")
                            # plot memory chunks
                            om = dict(sorted([(x, y["weight"]) for x, y in cm.pars.mem.items()][:30], key=lambda _i: _i[1],
                                             reverse=True))
                            utils.plot_mem(om, fi_dir + "words_plot.png", save_fig=True, show_fig=False)

                            # setting threshold for plotting
                            plot_thresh = 0.0
                            if fn == "all_songs_in_G" or fn == "all_irish-notes_and_durations-abc"\
                                    or fn == "scottish" or fn == "ocarolan":
                                plot_thresh = 0.099

                            print("plotting tps units...")
                            utils.plot_gra_from_normalized(cm.tps_units, filename=fi_dir + "tps_units",
                                                           thresh=plot_thresh, render=True)
                            # print("plotting tps symbols...")
                            # utils.plot_gra_from_normalized(cm.tps_1, filename=fi_dir + "tps_symbols",
                            #                                thresh=plot_thresh, render=True)

    print("END..")
