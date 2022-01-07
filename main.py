import string
from matplotlib import pyplot as plt
import utils
from Parser import ParserModule
from TPS import TPSModule
import numpy as np

np.set_printoptions(linewidth=np.inf)
np.random.seed(77)
max_order = 3

with open("data/{}.txt".format("input"), "r") as fp:
    sequences = [line.strip() for line in fp]

# sequences = utils.generate_Saffran_sequence()

symbols = set([x for sl in sequences for x in sl])
print(symbols)
parser = ParserModule()
symbols_dict = {s:0 for s in symbols}
TPS = list()
for o in range(max_order):
    TPS.append(TPSModule(o+1))


for seq in sequences:
    # buffer for sequential encoding
    buffer = seq[:max_order+1]
    # encode initial buffer
    for _s in buffer:
        symbols_dict[_s] += 1.0
    for _tps in TPS:
        _tps.encode(buffer)
    # encode the rest of seq
    for idx in range(max_order + 1, len(seq)):
        # count last symbol
        symbols_dict[seq[idx]] += 1.0
        # encode transitions
        for _i, _tps in enumerate(TPS):
            _tps.encode(buffer[-(_i+1):] + seq[idx])
        # update buffer
        buffer = buffer[1:] + seq[idx]
        # units from memory, if any
        units = parser.read_percept(buffer)
        units_tps = []
        if len(units) == 0:
            for _tps in TPS:
                units_tps.append(_tps.get_units(buffer))
        # else units using brent tps
        units_tps_brent = []
        if len(units) == 0:
            for _tps in TPS:
                units_tps_brent.append(_tps.get_units_brent(buffer))

        # mem
        # tps_units.encode(units)
        # pars.add_weight(p, comps=units, weight=w)
        # # forgetting and interference
        # pars.forget_interf(p, comps=units, forget=f, interfer=i, ulens=units_len)
        # tps_units.forget(units, i)


        print("*****************************")
        print("units: ", units)
        print("units_tps: ", units_tps)
        print("units_tps_brent: ", units_tps_brent)
        print("*****************************")

print(symbols_dict)
print(sum(symbols_dict.values()))
for tps in TPS:
    print("---- order: ",tps.order)
    print(tps.mem)

print("==========================")

# segmentation
# seq = "tutibubupadadutabapatubipidabu"
seq = "kofhoxrellumtaf"
for tps in TPS:
    print("--- order: ", tps.order)
    tps.normalize()
    print("certain units:", tps.get_certain_units())
    utils.plot_gra_from_m(tps.norm_mem, ler=tps.le_rows, lec=tps.le_cols, filename="./data/out_main/" + "tps{}_norm".format(tps.order))
    print(list(seq[tps.order:]))
    print(np.around(tps.get_probs(seq), decimals=2))


