import time
from datetime import datetime
import json
import os

import networkx as nx
from networkx.drawing.nx_pydot import read_dot

import const
import utils
from CTPs.pparser import Parser
from CTPs.tps import TPS
import numpy as np

from computation import Computation
from creativity import creative_gens
from multi_integration import MultiIntegration

np.set_printoptions(linewidth=np.inf)
rng = np.random.default_rng(const.RND_SEED)

file_names = ["input", "input2", "saffran", "thompson_newport", "reber", "all_songs_in_G",
              "all_irish-notes_and_durations", "cello", "bach_compact"]

file_names = ["input"]

# maintaining INTERFERENCES/FORGETS separation by a factor of 10
interferences = [0.005]
forgets = [0.05]
thresholds_mem = [1.0]
tps_orders = [1]
methods = ["BRENT"]

# for file_name in file_names:
#     for method in methods:
#         for order in tps_orders:
#             for interf in interferences:
#                 for forg in forgets:
#                     for t_mem in thresholds_mem:
#                         # init
#                         root_dir = const.OUT_DIR + "{}_{}/".format(method, time.strftime("%Y%m%d-%H%M%S"))
#                         os.makedirs(root_dir, exist_ok=True)
#                         with open(root_dir + "params.txt", "w") as of:
#                             json.dump({
#                                 "rnd": const.RND_SEED,
#                                 "w": const.WEIGHT,
#                                 "interference": interf,
#                                 "forgetting": forg,
#                                 "mem thresh": t_mem,
#                                 "lens": const.ULENS,
#                                 "tps_order": order,
#                             }, of)
#
#                         print("processing {} series ...".format(file_name))
#                         # init
#                         pars = Parser()
#                         tps_units = TPS(1)  # memory for TPs inter
#                         tps_1 = TPS(order)  # memory for TPs intra
#                         out_dir = root_dir + "{}/".format(file_name)
#                         os.makedirs(out_dir, exist_ok=True)
#                         results = dict()
#                         # --------------- INPUT ---------------
#                         sequences = utils.load_bach_separated("data/" + file_name + "/")
#
#                         # read percepts using parser function
#                         start_time = datetime.now()
#                         cm1 = Computation(rng, order=order, weight=const.WEIGHT, interference=interf, forgetting=forg,
#                                             mem_thres=t_mem, unit_len=const.ULENS, method=method)
#                         for itr, s1 in enumerate(sequences):
#                             first_in_seq = True
#                             while len(s1) > 0:
#                                 p1, units1 = cm1.compute(s1, first_in_seq)
#                                 s1 = s1[len(p1.strip().split(" ")):]
#                                 first_in_seq = False
#
#                             if itr % 5 == 1:
#                                 cm1.tps_units.normalize()
#                                 results[itr] = dict()
#                                 results[itr]["generated"] = utils.generate(rng, cm1.tps_units)
#
#                         with open(out_dir + "multi_results.json","w") as op:
#                             json.dump(results, op)
#                         cm1.tps_units.normalize()
#                         cm1.tps_units.compute_states_entropy()
#
#                         print("plotting tps units...")
#                         utils.plot_gra_from_normalized(cm1.tps_units, filename=out_dir + "tps1")
#                         print("plotting tps all...")
#                         print("plotting memory...")
#                         om1 = dict(sorted([(x, y) for x, y in cm1.pars.mem.items()], key=lambda it: it[1], reverse=True))
#                         utils.plot_mem(om1, out_dir + "words_plot.png", save_fig=True, show_fig=False)


ipath = "data/out/CT_20220405-163851/input/tps_units.dot"
G = nx.DiGraph(read_dot(ipath))
ggs = creative_gens(rng, G)
print(ggs)
