import json
import os
import networkx as nx
from matplotlib import pyplot as plt

import const
import numpy as np
import creativity as ct


rep = []
with open("data/thompson_newport_ABCDEF_nebrelsot_only.txt", "r") as fpt:
    for ln in fpt.readlines():
        rep.append(ln.strip())
# rep = "nebrelsot"


# read input model
def plot_trends(data,dir_out,sname):
    # Plotting the Data
    plt.cla()
    plt.clf()
    plt.plot(data["G"]["mean"], label='mean')
    plt.plot(data["G"]["max"], label='max')
    plt.plot(data["G"]["min"], label='min')
    plt.xlabel('iterations')
    plt.ylabel('C')
    plt.title("Creativity values for TPs graph")
    plt.legend()
    plt.savefig(dir_out + '/trends_G' + sname + '.pdf')
    plt.cla()
    plt.clf()
    plt.plot(data["GG"]["mean"], label='mean')
    plt.plot(data["GG"]["max"], label='max')
    plt.plot(data["GG"]["min"], label='min')
    plt.xlabel('iterations')
    plt.ylabel('C')
    plt.title("Creativity values for GG")
    plt.legend()
    plt.savefig(dir_out + '/trends_GG' + sname + '.pdf')


for seed in [4]:
    for subdir, dirs, files in os.walk(const.OUT_DIR + "convergence_divergence_results_" + str(seed) + "/"):
        if "thompson_newport_ABCDEF_nebrelsot" in subdir:
            for file in files:
                if "\\AVG_NFWI" in subdir and "results_pars_20\\" in subdir and "tps_results_1000\\" not in subdir \
                        and "tps_results_10000\\" not in subdir and "tps_results_5000\\" not in subdir:
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
                        gens_data["results"]["gen_hits"] = 0
                        gens_data["results"]["ggen_hits"] = 0
                        gens_data["trends"] = {
                            "G":{"mean":[],"min":[],"max":[]},
                            "GG":{"mean":[],"min":[],"max":[]}
                        }
                        # g_creative = {}
                        # gg_creative = {}
                        g_creative = []
                        gg_creative = []

                        print("generating...  ", subdir)
                        rng = np.random.default_rng(const.RND_SEED)
                        for _i in range(0, 1000):
                            # GENERATE CREATIVELY
                            gens = ct.creative_gens(rng, G, n_seq=100, min_len=100)
                            ggens, ggens_id = ct.creative_ggens(rng, GG, n_seq=100, min_len=100)
                            # EVALUATE
                            g_evals = ct.evaluate_similarity(gens, rep)
                            gg_evals = ct.evaluate_similarity(ggens, rep)
                            # g_evals = ct.evaluate_online(gens)
                            # g_evals = ct.evaluate_interval_function(gens)
                            # gg_evals = ct.evaluate_interval_function(ggens)
                            # UPDATE
                            G = ct.update(g_evals, G)
                            GG = ct.gupdate(GG, gg_evals, ggens_id)
                            # ct.evaluate_creative_g(g_evals, G, _i, g_creative)
                            # ct.evaluate_creative_gg(GG, gg_evals, ggens_id, _i, gg_creative)
                            ct.collect_creative_g_arr(g_evals, G, g_creative,gens_data)
                            ct.collect_creative_gg_arr(GG, gg_evals, ggens_id, gg_creative,gens_data)


                        # output_dir = subdir.replace("convergence_divergence_results","divergence_results")
                        output_dir = subdir.replace("convergence_divergence_results_", "cdr_1_")
                        os.makedirs(output_dir, exist_ok=True)
                        creative_g_sorted = g_creative
                        creative_gg_sorted = gg_creative
                        gens_data["results"]["tot"] = 1000
                        gens_out = []
                        ggens_out = []
                        uf = "3"
                        ct.plot_nx_creativity(G, output_dir + "/creative_graph" + uf, gen=False, render=True)
                        ct.plot_nx_creativity(GG, output_dir + "/creative_ggraph" + uf, render=True)
                        for x in ct.creative_gens(rng, G, n_seq=gens_data["results"]["tot"]):
                            xg = "".join(x).replace(" ","")
                            gens_out.append(x)
                            if xg in rep:
                                gens_data["results"]["gen_hits"] += 1
                        gg_gen,gg_genids = ct.creative_ggens(rng, GG, n_seq=gens_data["results"]["tot"])
                        for x in gg_gen:
                            xgg = "".join(x).replace(" ","")
                            ggens_out.append(xgg)
                            if xgg in rep:
                                gens_data["results"]["ggen_hits"] += 1
                        with open(output_dir + "/creative_g" + uf + ".json", "w") as fp:
                            json.dump(creative_g_sorted, fp)
                        with open(output_dir + "/creative_gg" + uf + ".json", "w") as fp:
                            json.dump(creative_gg_sorted, fp)
                        with open(output_dir + "/gens_dataM" + uf + ".json", "w") as fp:
                            json.dump(gens_data, fp)
                        with open(output_dir + "/generatedM" + uf + ".json", "w") as fp:
                            json.dump(gens_out, fp)
                        with open(output_dir + "/ggeneratedM" + uf + ".json", "w") as fp:
                            json.dump(ggens_out, fp)

                        plot_trends(gens_data["trends"],output_dir, uf)

print("END")
