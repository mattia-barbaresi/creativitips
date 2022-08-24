import networkx as nx
from networkx.drawing.nx_pydot import read_dot
import const
import utils
import numpy as np
import creativity as ct

np.set_printoptions(linewidth=np.inf)
rng = np.random.default_rng(const.RND_SEED)

# file_names = ["input", "input2", "saffran", "thompson_newport", "reber", "all_songs_in_G",
#               "all_irish-notes_and_durations", "cello", "bach_compact"]

file_names = ["mediapipe"]

# maintaining INTERFERENCES/FORGETS separation by a factor of 10
interferences = [0.005]
forgets = [0.05]
thresholds_mem = [1.0]
tps_orders = [2]
methods = ["CT"]

for file_name in file_names:
    for method in methods:
        for order in tps_orders:
            for interf in interferences:
                for forg in forgets:
                    for t_mem in thresholds_mem:
                        # # init
                        # root_dir = const.OUT_DIR + "{}_{}/".format(method, time.strftime("%Y%m%d-%H%M%S"))
                        # os.makedirs(root_dir, exist_ok=True)
                        # with open(root_dir + "params.txt", "w") as of:
                        #     json.dump({
                        #         "rnd": const.RND_SEED,
                        #         "w": const.WEIGHT,
                        #         "interference": interf,
                        #         "forgetting": forg,
                        #         "mem thresh": t_mem,
                        #         "lens": const.ULENS,
                        #         "tps_order": order,
                        #     }, of)
                        #
                        # print("processing {} series ...".format(file_name))

                        # read input model
                        ipath = const.OUT_DIR + "tps_results/" + \
                                utils.params_to_string(method, order, forg, interf, t_mem) \
                                + file_name + "/tps_units.dot"

                        # ============= GRAPH
                        # create graph from input model
                        G = nx.DiGraph(read_dot(ipath))
                        # with open("data/" + file_name + ".txt", "r") as fp:
                        #     rep = fp.readlines()
                        rep = ["DGGABABdBADFEDADFEDDGGABABdBADFEDADFEDDGGABABdBADFEDGGEDGGABABdBADFEDGGEGGBdgfefgfedBcBABcdefgfefgfedBcAFAGGBdGGBdgfefgfedBcBABcdefgfefgfedBcAFAGGBd"]
                        print("generating... ")
                        for _i in range(0, 100):
                            gens = ct.creative_gens(rng, G, n_seq=10, min_len=100)
                            g_evals = ct.evaluate_similarity(gens, rep)  # prova
                            print("generated: ", g_evals)
                            G = ct.update(g_evals, G)
                        utils.plot_nx_creativity(G, "data/out/creativeGens")
                        print("G edges: ", G.edges(data=True))
