"""Module for creativity classes and functions"""

# digraph {
# 	START [label="START (1.549)"]
# 	START -> d [label=0.362 penwidth=1.0851063829787235]
# 	START -> k [label=0.234 penwidth=0.7021276595744681]
# 	START -> m [label=0.404 penwidth=1.2127659574468084]
# 	a [label="a (1.564)"]
# 	a -> f [label=0.383 penwidth=1.148936170212766]
# 	a -> l [label=0.255 penwidth=0.7659574468085106]
# 	a -> z [label=0.362 penwidth=1.0851063829787235]
# 	b [label="b (1.531)"]
# }
import utils


def creative_gens(rand_gen, kg0, n_seq=10, min_len=30):
    res = []
    init_keys = []
    init_values = []
    if "START" in kg0.nodes:
        # tot = sum([float(x[2]["label"]) for x in kg0.edges(data=True) if x[0] in kg0.successors("START")])
        for x, y, v in kg0.edges("START", data=True):
            init_keys.append(y)
            init_values.append(float(v["label"]))
    else:
        print("no START found")

    if not init_keys:
        print("Empty init set. No generation occurred.")
        return res

    for _ in range(n_seq):
        seq = init_keys[utils.mc_choice(rand_gen, init_values)]  # choose rnd starting point (monte carlo)
        _s = seq
        for _ in range(min_len):
            # succs = list(kg0.edges(_s, data=True))
            # _s = succs[utils.mc_choice(rand_gen, succs)]
            succs = []
            succs_values = []
            for x, y, v in kg0.edges(_s, data=True):
                succs.append(y)
                succs_values.append(float(v["label"]))
            if succs:
                _s = succs[utils.mc_choice(rand_gen, succs_values)]
                if _s != "END":
                    seq += " " + _s
        res.append(seq)

    return res
