from datetime import datetime
import json
import os
import numpy as np
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

    # file_names = ["input", "input2", "input_full", "input2_full", "reber",
    #               "saffran", "thompson_newport","thompson_newport_ABCDEF",
    #               "all_irish-notes_and_durations-abc", "all_songs_in_G",  "bach_preludes", "ocarolan", "scottish"]
    # file_names = ["thompson_newport_ABCDEF"]
    # file_names = ["thompson_newport_ABCDEF_nebrelsot", "thompson_newport_nebrelsot"]
    file_names = ["thompson_newport_train"]

    # maintaining INTERFERENCES/FORGETS separation by a factor of 10
    thresholds_mem = [1.0]
    interferences = [0.0001]
    tps_orders = [2]
    gen_num = 1000  # number og generation for tests
    # method = [met_pars]
    # met: AVG, FTP, MI, CT or BRENT
    # pars: W = with, N=No, F=forgetting, I=interference
    # methods = ["BRENT_WFWI","BRENT_NFWI","FTPAVG_WFWI","FTPAVG_NFWI","AVG_WFWI","AVG_NFWI"]
    methods = ["AVG_NFWI","BRENT_NFWI","FTPAVG_NFWI"]

    for seed in [4,13,77,128,142]:
        for tps_method in methods:
            for tps_order in tps_orders:
                for interf in interferences:
                    for t_mem in thresholds_mem:
                        for rip in [10, 100, 500, 1000, 5000, 10000]:
                            for fn in file_names:
                                rng = np.random.default_rng(seed)
                                root_out_dir = const.OUT_DIR + "convergence_divergence_results_" + str(
                                    seed) + "/" + "tps_results_pars_" + str(
                                    const.PARSER_MEM_C) + "/" + "tps_results_" + str(rip) + "/" + "{}/".format(tps_method)
                                # root_out_dir = const.OUT_DIR + "tps_results_childes_500/{}/".format(tps_method)
                                os.makedirs(root_out_dir, exist_ok=True)

                                # with open(root_out_dir + "params.txt", "w") as of:
                                #     json.dump({
                                #         "rnd": const.RND_SEED,
                                #         "mem thresh": t_mem,
                                #         "interference": interf,
                                #         "weight": const.WEIGHT,
                                #         "lens": const.ULENS,
                                #         "tps_order": tps_order,
                                #         "parser_decay_rate": const.STM_DECAY_RATE,
                                #         "tps_decay_rate": const.LTM_DECAY_RATE,
                                #     }, of)
                                print("processing {} series ...".format(fn))
                                fi_dir = root_out_dir + "{}/".format(fn)
                                os.makedirs(fi_dir, exist_ok=True)

                                results = dict()
                                # input
                                sequences = utils.read_sequences(rng, fn)
                                # read percepts using parser function
                                start_time = datetime.now()
                                # init module for computation
                                cm = Computation(rng, order=tps_order, weight=const.WEIGHT, interference=interf,
                                                 mem_thres=t_mem, unit_len=const.ULENS, method=tps_method)
                                # compute series
                                cm.compute(sequences)

                                # class form on graph
                                # dc = fc.distributional_context(fc_seqs, 3)
                                # # print("---- dc ---- ")
                                # # pp.pprint(dc)
                                # classes = fc.form_classes(dc)
                                # class_patt = fc.classes_patterns(classes, fc_seqs)

                                shallow_parsed = []
                                with open(fi_dir + 'parsed.shpartips', "w", encoding='utf-8') as fp:
                                    for ln in cm.shallow_parsing:
                                        shallow_parsed.append(ln.strip().replace(" ", "").split("||"))
                                        fp.write(ln + "\n")

                                results["processing time"] = str((datetime.now() - start_time).total_seconds())

                                # normalizes memories
                                cm.tps_1.normalize()
                                cm.tps_units.normalize()

                                # calculate states uncertainty
                                cm.tps_1.compute_states_entropy()
                                cm.tps_units.compute_states_entropy()

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

                                # generate sample sequences
                                gens = cm.tps_units.generate_new_seqs(rng, n_seq=gen_num, min_len=100)
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

                                # generalization
                                ggraph = TPsGraph(tps=cm.tps_units)
                                gen_paths = cm.tps_units.generate_paths(rng, n_paths=rip, max_len=50)
                                # gen_paths = cm.tps_units.generate_paths(rng, n_paths=100, max_len=50)
                                if gen_paths:
                                    print("generalizing ...")
                                    # self.tps_units.normalize()
                                    ggraph.generalize(fi_dir, gen_paths, render=False)
                                    # cm.generalize(fi_dir, gen_paths)

                                # generate sample sequences from generalized graph
                                print("generating with generalized graph ...")
                                gg_gens = ggraph.generate_sequences(rng, n_seq=gen_num)
                                # gg_gens = ggraph.generate_creative_sequences(rng)
                                print("gg_gens: ", gg_gens)

                                # save generated
                                with open(fi_dir + "ggen.json", "w") as of:
                                    json.dump(gg_gens, of)

                                # plot tps
                                # utils.plot_tps_sequences(cm, [" ".join(x) for x in sequences[:20]], fi_dir)
                                # if gens:
                                #     comp_res = complexity.calculate_complexities(gens)
                                #     results["generated complexities"] = comp_res
                                #     print("---------- complexities:")
                                #     for it, vl in comp_res.items():
                                #         print(it, vl)
                                #     print("----------")

                                # DIVERGENCE TESTS
                                if fn == "thompson_newport_train":
                                    test_data = []
                                    train_data = []
                                    with open("data/thompson_newport_test.txt", "r") as fpt:
                                        for ln in fpt.readlines():
                                            test_data.append(ln.strip())
                                    with open("data/thompson_newport_train.txt", "r") as fpt:
                                        for ln in fpt.readlines():
                                            train_data.append(ln.strip())
                                    nn_gen = 0
                                    nn_ggen = 0
                                    gen_err = 0
                                    ggen_err = 0
                                    set_g = set()
                                    set_g_err = set()
                                    set_gg = set()
                                    set_gg_err = set()
                                    for xg in gens:
                                        if xg.replace(" ", "") in test_data:
                                            nn_gen += 1
                                            set_g.add(xg.replace(" ", ""))
                                        if xg.replace(" ", "") in train_data:
                                            gen_err += 1
                                            set_g_err.add(xg.replace(" ", ""))
                                    for xgg in gg_gens:
                                        if xgg.replace(" ", "") in test_data:
                                            nn_ggen += 1
                                            set_gg.add(xgg.replace(" ", ""))
                                        if xgg.replace(" ", "") in train_data:
                                            ggen_err += 1
                                            set_gg_err.add(xgg.replace(" ", ""))
                                    with open(fi_dir + "test_gen.json", "w") as fpt:
                                        json.dump({
                                            "gen_hits": nn_gen,
                                            "gen_same": gen_err,
                                            "set_g": len(set_g),
                                            "set_g_same": len(set_g_err),
                                            "ggen_hits": nn_ggen,
                                            "ggen_same": ggen_err,
                                            "set_gg": len(set_gg),
                                            "set_gg_same": len(set_gg_err)}, fpt)

                                # TEST nebrelsot
                                if "nebrelsot" in fn:
                                    nebrelsot_g = 0
                                    nebrelsot_gg = 0
                                    for xg in gens:
                                        if "nebrelsot" in xg.replace(" ", ""):
                                            nebrelsot_g += 1
                                    for xgg in gg_gens:
                                        if "nebrelsot" in xgg.replace(" ", ""):
                                            nebrelsot_gg += 1
                                    with open(fi_dir + "nebrelsot.json", "w") as fpt:
                                        json.dump({
                                            "g_hits": nebrelsot_g,
                                            "gg_hits": nebrelsot_gg}, fpt)

                                # # UNSEGMENTED TESTS
                                # seg_data = []
                                # with open("data/CHILDES_seg_Barbara.txt","r") as fpu:
                                #     for ln in fpu.readlines():
                                #         seg_data.append(ln.strip().split(" "))
                                #
                                # tot_count = 0
                                # hits = 0
                                # miss = 0
                                # seq_count = []
                                # seq_count_rate = []
                                # for i,ln in enumerate(shallow_parsed):
                                #     tot_count += len(seg_data[i])
                                #     ref_set = set(seg_data[i])
                                #     p_set = set(ln)
                                #     miss += len(ref_set - p_set)
                                #     lh = 0
                                #     for w in seg_data[i]:
                                #         if w in ln:
                                #             lh += 1
                                #     hits += lh
                                #     seq_count.append(lh)
                                #     seq_count_rate.append(lh/len(seg_data[i]))
                                # with open(fi_dir + "unsegmented_parsing.json", "w") as fpt:
                                #     json.dump({
                                #         "tot_count":tot_count,
                                #         "hits":hits,
                                #         "miss": miss,
                                #         "hit_rate per sequence": seq_count_rate,
                                #         "hits per sequence": seq_count}, fpt)

                                # plot memory chunks

                                print("plotting memory...")
                                om = dict(sorted([(x, y["weight"]) for x, y in cm.pars.mem.items()
                                                  if y["weight"] >= t_mem][:30], key=lambda _i: _i[1], reverse=True))
                                utils.plot_mem(om, fi_dir + "words_plot.png", save_fig=True, show_fig=False)

                                # setting threshold for plotting
                                plot_thresh = 0.0
                                if fn == "all_songs_in_G" or fn == "all_irish-notes_and_durations-abc" \
                                        or fn == "scottish" or fn == "ocarolan":
                                    plot_thresh = 0.01

                                print("plotting tps units...")
                                utils.plot_gra_from_normalized(cm.tps_units,
                                                               filename=fi_dir + "tps_units",
                                                               thresh=plot_thresh,
                                                               render=False)
                                print("plotting tps symbols...")
                                utils.plot_gra_from_normalized(cm.tps_1,
                                                               filename=fi_dir + "tps_symbols",
                                                               thresh=plot_thresh,
                                                               render=False)

    print("END..")
