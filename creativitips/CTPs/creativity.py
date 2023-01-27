"""Module for creativity classes and functions"""
import math
import os

import more_itertools as mit
import numpy as np
import const
from creativitips import utils
from difflib import SequenceMatcher
from graphviz import Digraph
from metrics import interval_functions as intfun, utility
import jellyfish

from thefuzz import fuzz
from thefuzz import process

def calculate_creativity(p, u, v):
    if u < 0.0:
        return 1.0001 - p
    else:
        return (1.0001 - p) * u * (1.0001 - v)


def calculate_creativity2(p, u, v):
    ut = u if u > 0.0 else 1.0
    vt = v if v > 0.0 else 0.0001
    return (1.0001 - p) * ut * (1.0001 - vt)


def calculate_creativity3(p, u, v):
    ut = u if u > 0.0 else 1.0
    return (1.0001 - p) * ut


def calculate_creativityb3(p, u, v):
    if u < 0.0:
        return 1.0001 - p
    else:
        return (1.0001 - p) * u


def normalize_arr(arr):
    nar = np.array(arr)
    sm = nar.sum()
    if sm == 0:
        # flat probs
        return [1.0 / len(nar)] * len(nar)
    else:
        return nar / sm


def creative_gens(rand_gen, kg0, n_seq=10, min_len=30):
    res = []
    init_keys = []
    init_values = []
    if "START" in kg0.nodes:
        # tot = sum([float(x[2]["label"]) for x in kg0.edges(data=True) if x[0] in kg0.successors("START")])
        for x, y, v in kg0.edges("START", data=True):
            init_keys.append(y)
            c = calculate_creativity3(float(v["p"]), float(v["u"]), float(v["v"]))
            init_values.append(c)
    else:
        print("no START found")

    if not init_keys:
        print("Empty init set. No generation occurred.")
        return res

    for _ in range(n_seq):
        seq = []
        _s = init_keys[utils.mc_choice(rand_gen, normalize_arr(init_values))]  # choose rnd starting point (monte carlo)
        seq.append(_s)
        for _ in range(min_len):
            # succs = list(kg0.edges(_s, data=True))
            # _s = succs[utils.mc_choice(rand_gen, succs)]
            succs = []
            succs_values = []
            for x, y, v in kg0.edges(_s, data=True):
                succs.append(y)
                succs_values.append(calculate_creativity3(float(v["p"]), float(v["u"]), float(v["v"])))
            if succs:
                _s = succs[utils.mc_choice(rand_gen, normalize_arr(succs_values))]
                if _s != "END":
                    seq.append(_s)
        res.append(seq)

    return res


def creative_ggens(rand_gen, kg0, n_seq=10, min_len=30):
    res = []
    res_id = []
    init_keys = []
    init_values = []
    init_node = [x[0] for x in kg0.nodes(data="label") if x[1] == "START"]
    if init_node:
        # tot = sum([float(x[2]["label"]) for x in kg0.edges(data=True) if x[0] in kg0.successors("START")])
        for x, y, v in kg0.edges(init_node, data=True):
            init_keys.append(y)
            c = calculate_creativity3(float(v["p"]), float(v["u"]), float(v["v"]))
            init_values.append(c)
    else:
        print("no START found")

    if not init_keys:
        print("Empty init set. No generation occurred.")
        return res, res_id

    for _ in range(n_seq):
        seq = []
        seq_id = []
        _s = init_keys[utils.mc_choice(rand_gen, normalize_arr(init_values))]  # choose rnd starting point (monte carlo)
        seq.append(rand_gen.choice(kg0.nodes[_s]["label"].replace('"', '').split(const.GRAPH_SEP)))
        seq_id.append(_s)
        _iter = 0
        # for _ in range(min_len):
        while kg0.nodes[_s]["label"] != "END" and _iter < min_len:
            # succs = list(kg0.edges(_s, data=True))
            # _s = succs[utils.mc_choice(rand_gen, succs)]
            succs = []
            succs_values = []
            for x, y, v in kg0.edges(_s, data=True):
                succs.append(y)
                succs_values.append(calculate_creativity3(float(v["p"]), float(v["u"]), float(v["v"])))
            if succs:
                _s = succs[utils.mc_choice(rand_gen, normalize_arr(succs_values))]
                # if _s != "END":
                if kg0.nodes[_s]["label"] != "END":
                    seq.append(rand_gen.choice(kg0.nodes[_s]["label"].replace('"', '').split(const.GRAPH_SEP)))
                    seq_id.append(_s)
            _iter += 1

        res.append(seq)
        res_id.append(seq_id)

    return res, res_id


def evaluate_ad_hoc(seqs):
    vals = []
    for seq in seqs:
        sseq = "".join([_.replace(" ", "") for _ in seq])
        val = 0
        if "ddcA" in sseq:
            val += 1
        if "cec" in sseq:
            val += 1
        if "GBGAGBGA" in sseq:
            val += 2
        if "BBAGEED" in sseq:
            val += 2
        if "DF" in sseq:
            val += 1
        val /= 5

        vals.append((seq, val))

    return vals


def evaluate_online(seqs):
    vals = []
    for seq in seqs:
        vals.append((seq, float(input("sequence:" + " ".join(seq) + ". Please judge:"))))
    return vals


def utility_from_sequence(sequence):
    melody = intfun.key2accidentals(sequence, 'G')  # Hp: key of G
    allowed_intervals = [4, 3, -3, -4, 9, -9]  # only thirds and sixths
    ut = utility.utility(melody, allowed_intervals)
    # ut = utility.utility_kl(melody, allowed_intervals)
    print("utility: ", ut)
    return ut


def evaluate_interval_function(seqs):
    vals = []
    for seq in seqs:
        vals.append((seq, utility_from_sequence(" ".join(seq).split(" "))))
    return vals


def evaluate_similarity(seqs, rep):
    vals = []
    for seq in seqs:
        sseq = "".join([_.replace(" ", "") for _ in seq])
        val = rep_similarity(sseq, rep)
        vals.append((seq, val))
    return vals


def evaluate_similarity_nebrelsot(seqs):
    vals = []
    for seq in seqs:
        sseq = "".join([_.replace(" ", "") for _ in seq])
        val = 1 if "nebrelsot" in sseq else 0
        vals.append((seq, val))
    return vals


def rep_similarity(seq, rep):
    val = 0
    # print(seq)
    # print("val1:", (SequenceMatcher(seq, rep).ratio() + fuzz.partial_token_sort_ratio(seq, "nebrelsot")) / 2)
    # for _ in rep:
    #     val += SequenceMatcher(None, seq, _).ratio()
    # return val / len(rep)  # mean similarity
    # return (SequenceMatcher(seq, rep).ratio() + fuzz.partial_token_sort_ratio(seq, "nebrelsot")) / 200
    if seq in rep:
        return 1.0
    else:
        return fuzz.partial_token_sort_ratio(seq, "nebrelsot") / 200


def creative_edge(edge, out_par=""):
    c = (1.0 - float(edge["p"])) * edge["u"] * (1.0 - edge["v"])
    if c > 0.75:
        print("CREATIVE " + out_par + " EDGE: ", edge, c)


def update(g_evals, G):
    for seq, val in g_evals:
        for sn, en in mit.pairwise(["START"] + seq):
            u = float(G[sn][en]["u"]) if "u" in G[sn][en] else 0
            u = 0 if u == -1.0 else u
            # v = float(G[sn][en]["v"]) if "v" in G[sn][en] else 0
            G[sn][en]["u"] = (u + val) / 2  # u average (u mean)
            G[sn][en]["v"] = 1.0 - math.sqrt((u - val) ** 2)  # v average (u variability)
            # creative_edge(G[sn][en], "G")
    return G


def collect_creative_g(g_evals, G, itr, creative_solutions):
    best_seq = " "
    best_c = 0
    for seq, val in g_evals:
        p = 1
        v = 0
        count = 0
        for sn, en in mit.pairwise(seq):
            p *= float(G[sn][en]["p"])
            v += float(G[sn][en]["v"]) if "v" in G[sn][en] else 0
            count += 1
        v = v/count
        ss = "".join(seq).replace(" ","")
        c = (1-p) * val * (1-v)
        if ss not in creative_solutions:
            if c > best_c:
                best_c = c
                best_seq = ss
        else:
            if creative_solutions[ss][0] < c:
                print("WTF (old,new): ", creative_solutions[ss][0], " - ", c)

    creative_solutions[best_seq] = [best_c, itr]


def collect_creative_g_arr(g_evals, G, creative_solutions,gen_data):
    best_seq = " "
    max_c = 0
    min_c = 2 #j ust grater than mac C
    mean_c = 0
    for seq, val in g_evals:
        p = 1
        v = 0
        count = 0
        for sn, en in mit.pairwise(seq):
            p *= float(G[sn][en]["p"])
            v += float(G[sn][en]["v"]) if "v" in G[sn][en] else 0
            count += 1
        v = v/count
        ss = "".join(seq).replace(" ","")
        c = (1-p) * val * (1-v)
        if c > max_c:
            max_c = c
            best_seq = ss
        if c < min_c:
            min_c = c
        mean_c += c
    gen_data["trends"]["G"]["mean"].append(mean_c/len(g_evals))
    gen_data["trends"]["G"]["min"].append(min_c)
    gen_data["trends"]["G"]["max"].append(max_c)

    creative_solutions.append((best_seq,max_c))


def gupdate(GG, g_evals, gensids):
    init_node = [x[0] for x in GG.nodes(data="label") if x[1] == "START"]
    for indx, (seq, val) in enumerate(g_evals):
        path_idxs = init_node + gensids[indx]
        # path_idxs.append(end_node)
        for sn, en in mit.pairwise(path_idxs):  # NOTE: there could be more than one class per symbol!
            u = float(GG[sn][en]["u"])
            u = 0 if u == -1.0 else u
            # v = float(GG[sn][en]["v"]) if "v" in GG[sn][en] else 0
            GG[sn][en]["u"] = (u + val) / 2  # u average (u mean)
            GG[sn][en]["v"] = 1.0 - math.sqrt((u - val) ** 2)  # v average (u variability)
            # creative_edge(GG[sn][en], "GG")
    return GG


def collect_creative_gg(GG, g_evals, gensids, itr, creative_solutions):
    best_seq = " "
    best_c = 0
    init_node = [x[0] for x in GG.nodes(data="label") if x[1] == "START"]
    for indx, (seq, val) in enumerate(g_evals):
        path_idxs = init_node + gensids[indx]
        p = 0
        v = 0
        count = 0
        for sn, en in mit.pairwise(path_idxs):
            p *= float(GG[sn][en]["p"])
            v += float(GG[sn][en]["v"]) if "v" in GG[sn][en] else 0
            count += 1
        v = v / count
        ss = "".join(seq).replace(" ","")
        c = (1 - p) * val * (1 - v)
        if ss not in creative_solutions:
            if c > best_c:
                best_c = c
                best_seq = ss
        else:
            if creative_solutions[ss][0] <= c:
                print("WTF(old,new): ", creative_solutions[ss][0],creative_solutions[ss][0],creative_solutions[ss][0], " - ", c)
    creative_solutions[best_seq] = [best_c, itr]


def collect_creative_gg_arr(GG, g_evals, gensids, creative_solutions, gen_data):
    best_seq = " "
    mean_c = 0
    min_c = 2  # just a number greater than max C
    max_c = 0
    init_node = [x[0] for x in GG.nodes(data="label") if x[1] == "START"]
    for indx, (seq, val) in enumerate(g_evals):
        path_idxs = init_node + gensids[indx]
        count = 0
        p = 0
        v = 0
        for sn, en in mit.pairwise(path_idxs):
            p *= float(GG[sn][en]["p"])
            v += float(GG[sn][en]["v"]) if "v" in GG[sn][en] else 0
            count += 1
        v = v / count
        ss = "".join(seq).replace(" ","")
        c = (1 - p) * val * (1 - v)
        if c > max_c:
            max_c = c
            best_seq = ss
        if c < min_c:
            min_c = c
        mean_c += c
    gen_data["trends"]["GG"]["mean"].append(mean_c/len(g_evals))
    gen_data["trends"]["GG"]["min"].append(min_c)
    gen_data["trends"]["GG"]["max"].append(max_c)
    creative_solutions.append((best_seq,max_c))


def get_class_from_node(self, node_name):
    for cl, l in self.fc.items():
        if node_name in cl:
            return l
    return -1


def plot_nx_creativity(G, filename="tps", gen=True, render=True):
    gra = Digraph()
    for k, v, d in G.edges(data=True):
        # gra.edge(k, v, label="{:.2f}-{:.2f}-{:.2f}".format(float(d["p"]),float(d["u"]),float(d["v"])))
        c = calculate_creativity3(float(d["p"]), float(d["u"]), float(d["v"]))
        if gen:
            gra.edge("\n".join(G.nodes[k]["label"].split(" | ")), "\n".join(G.nodes[v]["label"].split(" | ")),
                     label="{:.4f}\n({:.3f},{:.2f},{:.3f})".format(c, float(d["p"]), float(d["u"]), float(d["v"])),
                     # label="{:.2f}".format(c),
                     penwidth=str(0.1 + abs(3 * c))
                     )
        else:
            end_node = "END" if v == "END" else G.nodes[v]["label"].split("(")[0]
            gra.edge(G.nodes[k]["label"].split("(")[0], end_node,
                     label="{:.4f}\n({:.3f},{:.2f},{:.3f})".format(c, float(d["p"]), float(d["u"]), float(d["v"])),
                     # label="{:.2f}".format(c),
                     penwidth=str(0.1 + abs(3 * c))
                     )
    # print(gra.source)
    if render:
        # gra.render(filename, view=False)
        gra.render(filename, view=False, engine="dot", format="pdf")
        os.rename(filename, filename + '.dot')
    else:
        gra.save(filename + '.dot')
