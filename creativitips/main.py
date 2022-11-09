import json
import os

import networkx as nx
import const
import utils
import numpy as np
import creativity as ct

np.set_printoptions(linewidth=np.inf)
rng = np.random.default_rng(const.RND_SEED)

# file_names = ["input", "input2", "saffran", "thompson_newport", "reber", "all_songs_in_G",
#               "all_irish-notes_and_durations", "cello", "bach_compact"]

file_names = ["all_irish-notes_and_durations-abc"]

# maintaining INTERFERENCES/FORGETS separation by a factor of 10
interferences = [0.005]
forgets = [0.05]
thresholds_mem = [1.0]
tps_orders = [2]
ltm = [10,20,50]
methods = ["FTP_WFWI","AVG_WFWI","FTP_NFWI","AVG_NFWI"]

for ltmp in ltm:
    for file_name in file_names:
        for method in methods:
            for order in tps_orders:
                for interf in interferences:
                    for forg in forgets:
                        for t_mem in thresholds_mem:
                            # read input model
                            file_dir_in = "tps_results_" + str(const.STM_DECAY_RATE) + "_" + str(ltmp) \
                                          + "/" + utils.params_to_string(method, order, forg, interf, t_mem) + file_name + "/"
                            out_dir = const.OUT_DIR + "creativeGens/" + file_dir_in
                            in_path = const.OUT_DIR + file_dir_in + "tps_units.dot"
                            in_gpath = const.OUT_DIR + file_dir_in + "ggraph.dot"

                            # create graph from input model
                            G = nx.DiGraph(nx.nx_pydot.read_dot(in_path))
                            GG = nx.DiGraph(nx.nx_pydot.read_dot(in_gpath))
                            # with open("data/" + file_name + ".txt", "r") as fp:
                            #     rep = fp.readlines()
                            gens_data = dict()

                            print("generating...  ", file_dir_in)
                            for _i in range(0, 50):
                                # GENERATE
                                # gens = ct.creative_gens(rng, G, n_seq=10, min_len=100)
                                gens, gens_id = ct.creative_ggens(rng, GG, n_seq=50, min_len=50)
                                # EVALUATE
                                # g_evals = ct.evaluate_similarity(gens, rep)  # prova
                                # g_evals = ct.evaluate_online(gens)
                                g_evals = ct.evaluate_interval_function(gens)
                                # UPDATE
                                # G = ct.update(g_evals, G)
                                GG = ct.gupdate(GG, g_evals, gens_id)
                                if _i % 10 == 0:
                                    gens_data[_i] = {}
                                    gens_data[_i]["gens"] = gens
                                    gens_data[_i]["ids"] = gens_id

                            os.makedirs(out_dir, exist_ok=True)
                            ct.plot_nx_creativity(GG, out_dir + "graph")
                            gens_out, goid = ct.creative_ggens(rng, GG, n_seq=10, min_len=100)
                            with open(out_dir + "generated.json", "w") as fp:
                                json.dump(gens_out, fp)
                            with open(out_dir + "gens_data.json", "w") as fp:
                                json.dump(gens_data, fp)

                            print("GG edges: ", GG.edges(data=True))
print("END")
