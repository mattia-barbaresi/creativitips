"""Module for creativity classes and functions"""

import more_itertools as mit
from creativitips import misc
from difflib import SequenceMatcher


def creative_gens(rand_gen, kg0, n_seq=10, min_len=30):
    res = []
    init_keys = []
    init_values = []
    if "START" in kg0.nodes:
        # tot = sum([float(x[2]["label"]) for x in kg0.edges(data=True) if x[0] in kg0.successors("START")])
        for x, y, v in kg0.edges("START", data=True):
            init_keys.append(y)
            c = (1 - float(v["p"])) * float(v["u"]) * (0.67 - float(v["v"]))
            init_values.append(c)
    else:
        print("no START found")

    if not init_keys:
        print("Empty init set. No generation occurred.")
        return res

    for _ in range(n_seq):
        seq = []
        _s = init_keys[misc.mc_choice(rand_gen, init_values)]  # choose rnd starting point (monte carlo)
        seq.append(_s)
        for _ in range(min_len):
            # succs = list(kg0.edges(_s, data=True))
            # _s = succs[utils.mc_choice(rand_gen, succs)]
            succs = []
            succs_values = []
            for x, y, v in kg0.edges(_s, data=True):
                succs.append(y)
                succs_values.append(float(v["p"]))
            if succs:
                _s = succs[misc.mc_choice(rand_gen, succs_values)]
                if _s != "END":
                    seq.append(_s)
        res.append(seq)

    return res


def evaluate(seqs):
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

        vals.append((seq,val))

    return vals


def evaluate_online(seqs):
    vals = []
    for seq in seqs:
        vals.append((seq, float(input("sequence:" + " ".join(seq) + ". Please judge:"))))
    return vals


def evaluate_similarity(seqs, rep):
    vals = []
    for seq in seqs:
        sseq = "".join([_.replace(" ", "") for _ in seq])
        val = rep_similarity(sseq, rep)
        vals.append((seq,val))

    return vals


def rep_similarity(seq, rep):
    val = 0
    for _ in rep:
        val += SequenceMatcher(None, seq, _).ratio()
    return val / len(rep)  # mean similarity


def update(g_evals, G):
    for seq, val in g_evals:
        for sn,en in mit.pairwise(seq):
            u = float(G[sn][en]["u"]) if "u" in G[sn][en] else 0
            v = float(G[sn][en]["v"]) if "v" in G[sn][en] else 0
            G[sn][en]["u"] = (u + val)/2  # u average (u mean)
            G[sn][en]["v"] = (v + abs(u - val))/2  # v average (u variability)
    return G
