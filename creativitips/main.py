import json
import os
import networkx as nx
import const
import utils
import numpy as np
import creativity as ct

np.set_printoptions(linewidth=np.inf)

# file_names = ["input", "input2", "saffran", "thompson_newport", "reber", "all_songs_in_G",
#               "all_irish-notes_and_durations", "cello", "bach_compact"]

file_names = ["all_irish-notes_and_durations"]

# maintaining INTERFERENCES/FORGETS separation by a factor of 10
interferences = [0.005]
thresholds_mem = [1.0]
tps_orders = [2]
methods = ["BRENT_WFWI","FTPAVG_WFWI","AVG_WFWI"]
#
# for file_name in file_names:
#     for method in methods:
#         for order in tps_orders:
#             for interf in interferences:
#                 for forg in forgets:
#                     for t_mem in thresholds_mem:
#                         rng = np.random.default_rng(const.RND_SEED)
#                         # read input model
#                         file_dir_in = "tps_results_*"
#                         out_dir = const.OUT_DIR + file_dir_in
#                         in_path = const.OUT_DIR + file_dir_in + "tps_units.dot"
#                         in_gpath = const.OUT_DIR + file_dir_in + "ggraph.dot"
#
#                         # create graph from input model
#                         G = nx.DiGraph(nx.nx_pydot.read_dot(in_path))
#                         GG = nx.DiGraph(nx.nx_pydot.read_dot(in_gpath))
#                         # with open("data/" + file_name + ".txt", "r") as fp:
#                         #     rep = fp.readlines()
#                         gens_data = dict()
#
#                         print("generating...  ", file_dir_in)
#                         for _i in range(0, 50):
#                             # GENERATE
#                             # gens = ct.creative_gens(rng, G, n_seq=10, min_len=100)
#                             gens, gens_id = ct.creative_ggens(rng, GG, n_seq=50, min_len=50)
#                             # EVALUATE
#                             # g_evals = ct.evaluate_similarity(gens, rep)  # prova
#                             # g_evals = ct.evaluate_online(gens)
#                             g_evals = ct.evaluate_interval_function(gens)
#                             # UPDATE
#                             # G = ct.update(g_evals, G)
#                             GG = ct.gupdate(GG, g_evals, gens_id)
#                             if _i % 10 == 0:
#                                 gens_data[_i] = {}
#                                 gens_data[_i]["gens"] = gens
#                                 gens_data[_i]["ids"] = gens_id
#
#                         os.makedirs(out_dir, exist_ok=True)
#                         ct.plot_nx_creativity(GG, out_dir + "graph")
#                         gens_out, goid = ct.creative_ggens(rng, GG, n_seq=10, min_len=100)
#                         with open(out_dir + "generated.json", "w") as fp:
#                             json.dump(gens_out, fp)
#                         with open(out_dir + "gens_data.json", "w") as fp:
#                             json.dump(gens_data, fp)
#
#                         print("GG edges: ", GG.edges(data=True))


rng = np.random.default_rng(const.RND_SEED)

# rep = "merhoxjessottafnav"

# read input model
for subdir, dirs, files in os.walk(const.OUT_DIR + "convergence_divergence_results/"):
    for file in files:
        if 'tps_units.dot' in file:
            in_path = subdir + "/tps_units.dot"
            in_gpath = subdir + "/ggraph.dot"
            # create graph from input model
            G = nx.DiGraph(nx.nx_pydot.read_dot(in_path))
            GG = nx.DiGraph(nx.nx_pydot.read_dot(in_gpath))
            # with open("data/" + file_name + ".txt", "r") as fp:
            #     rep = fp.readlines()
            gens_data = dict()
            gens_data["results"] = dict()

            print("generating...  ", subdir)
            for _i in range(0, 1000):
                # GENERATE
                gens = ct.creative_gens(rng, G, n_seq=100, min_len=100)
                ggens, ggens_id = ct.creative_ggens(rng, GG, n_seq=100, min_len=100)
                # EVALUATE
                # g_evals = ct.evaluate_similarity(gens, rep)
                # gg_evals = ct.evaluate_similarity(ggens, rep)
                # g_evals = ct.evaluate_online(gens)
                g_evals = ct.evaluate_interval_function(gens)
                gg_evals = ct.evaluate_interval_function(ggens)
                # UPDATE
                G = ct.update(g_evals, G)
                GG = ct.gupdate(GG, gg_evals, ggens_id)
                if _i % 10 == 0:
                    gens_data[_i] = {}
                    gens_data[_i]["gens"] = gens
                    gens_data[_i]["ggens"] = ggens
                    gens_data[_i]["gg_ids"] = ggens_id

            output_dir = subdir.replace("convergence_divergence_results","music_result")
            ct.plot_nx_creativity(G, output_dir + "/creative_graph3", gen=False, render=False)
            ct.plot_nx_creativity(GG, output_dir + "/creative_ggraph3", render=False)
            gens_out = ["".join([y.replace(" ","") for y in x]) for x in ct.creative_gens(rng, G, n_seq=1000)]
            ggens_out, ggoid = ct.creative_ggens(rng, GG, n_seq=1000)
            ggens_out = ["".join([y.replace(" ","") for y in x]) for x in ggens_out]

            # gens_data["results"]["gen_hits"] = len([x for x in gens_out if x == rep])
            # gens_data["results"]["ggen_hits"] = len([x for x in ggens_out if x == rep])
            with open(output_dir + "/generatedM3.json", "w") as fp:
                json.dump(gens_out, fp)
            with open(output_dir + "/ggeneratedM3.json", "w") as fp:
                json.dump(ggens_out, fp)
            with open(output_dir + "/gens_dataM3.json", "w") as fp:
                json.dump(gens_data, fp)

            print("GG edges: ", GG.edges(data=True))
print("END")
