import json
import os
import string
from matplotlib import pyplot as plt
import const
import utils
from Parser import ParserModule
from TPS import TPSModule
import numpy as np

np.set_printoptions(linewidth=np.inf)
rng = np.random.default_rng(const.RND_SEED)
max_order = 3
units_len = [3]

# with open("data/{}.txt".format("input"), "r") as fp:
#     sequences = [line.strip() for line in fp]
#
# sequences = utils.generate_Saffran_sequence()
#
# symbols = set([x for sl in sequences for x in sl])
# print(symbols)
# parser = ParserModule()
# symbols_dict = {s:0 for s in symbols}
# TPS = list()
# for o in range(max_order):
#     TPS.append(TPSModule(o+1))

# old version
# for seq in sequences:
#     # buffer for sequential encoding
#     buffer = seq[:max_order + 1]
#     # encode initial buffer
#     for _s in buffer:
#         symbols_dict[_s] += 1.0
#     for _tps in TPS:
#         _tps.encode(buffer)
#     # encode the rest of seq
#     for idx in range(max_order + 1, len(seq)):
#         # count last symbol
#         symbols_dict[seq[idx]] += 1.0
#         # encode TP
#         for _i, _tps in enumerate(TPS):
#             _tps.encode(buffer[-(_i+1):] + seq[idx])
#         # update buffer
#         buffer = buffer[1:] + seq[idx]
#
#         units_tps = []
#         for _tps in TPS:
#             units_tps.append(_tps.get_units(buffer))
#         # else units using brent tps
#         units_tps_brent = []
#         for _tps in TPS:
#             units_tps_brent.append(_tps.get_units_brent(buffer))
#
#         # mem
#         # tps_units.encode(units)
#         # pars.add_weight(p, comps=units, weight=w)
#         # # forgetting and interference
#         # pars.forget_interf(p, comps=units, forget=f, interfer=i, ulens=units_len)
#         # tps_units.forget(units, i)
#
#         print("*****************************")
#         print("units_tps: ", units_tps)
#         print("units_tps_brent: ", units_tps_brent)
#         print("*****************************")

# print(symbols_dict)
# print(sum(symbols_dict.values()))
# for tps in TPS:
#     print("---- order: ",tps.order)
#     print(tps.mem)
#
# print("==========================")

# segmentation
# seq = "tutibubupadadutabapatubipidabu"
# # seq = "kofhoxrellumtaf"
# for tps in TPS:
#     print("--- order: ", tps.order)
#     tps.normalize()
#     print("certain units:", tps.get_certain_units())
#     utils.plot_gra_from_normalized(tps.norm_mem, filename="./data/out_main/" + "tps{}_norm".format(tps.order))
#     print(list(seq[tps.order:]))
#     print(np.around(tps.get_probs(seq), decimals=2))
#     print(tps.get_units(seq))
#     print(tps.get_units_brent(seq))
#     print(tps.get_next_unit(seq))

# ===============================================================================================================


file_names = ["bicinia"]
tps_order = 3
base_encoder = utils.Encoder()
sequences = utils.load_bicinia_full("data/bicinia/", base_encoder)

# init
pars = ParserModule()
tps_units = TPSModule(1)  # memory for TPs inter
tps_1 = TPSModule(tps_order)  # memory for TPs intra

out_dir = const.OUT_DIR + "MULTI_bicinia_{}_({})/".format(const.MEM_THRES, "-".join([str(u) for u in units_len]))
os.makedirs(out_dir,exist_ok=True)
# read percepts using parser function
actions = []
for s1,s2 in zip(sequences):
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
if base_encoder:
    for gg in gens:
        decoded.append(base_encoder.base_decode(gg))
    print("decoded: ", decoded)


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
