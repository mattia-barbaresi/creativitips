import fnmatch
import os
import string
from graphviz import Digraph
import matplotlib.pyplot as plt
import numpy as np
# import networkx as nx
# from networkx.drawing.nx_pydot import write_dot
from matplotlib import cm


def params_to_string(tps_method, tps_order, fogt, interf, t_mem):
    return "{}_{}_{}_{}_{}/".format(tps_method, tps_order, fogt, interf, t_mem)


def matrix_from_tps(tps_dict, x_encoding, y_encoding):
    res = np.zeros((len(y_encoding.classes_), len(x_encoding.classes_)))
    for start, ems in tps_dict.items():
        for v, k in ems.items():
            res[y_encoding.transform([start])[0]][x_encoding.transform([v])[0]] = k
    return res


def mc_choice(rng, arr):
    """Return an index using MonteCarlo choice on arr"""
    rnd = rng.uniform()
    sm = arr[0]
    j = 1
    ind = 0
    while sm < rnd and j < len(arr):
        sm += float(arr[j])
        if sm >= rnd:
            ind = j
        j += 1
    return ind


def hebb_gen(rng, sequence, hm):
    """gen with hebb associations"""
    out = []
    for s in sequence:
        idx = mc_choice(rng, hm[s])
        out.append(idx)
    return out


def mtx_from_multi(seq1, seq2, nd1, nd2):
    mtx = np.zeros((nd1, nd2))
    for s1, s2 in zip(seq1, seq2):
        mtx[s1][s2] += 1
    return mtx


def show_counts(hebb_mtx):
    plt.matshow(hebb_mtx, cmap="plasma")
    plt.title('hebb_mtx')
    plt.tight_layout()
    plt.show()


def softmax(x):
    rsum = np.sum(x, axis=1, keepdims=True)  # returns sum of each row and keeps same dims
    f_x = x / rsum
    return f_x


############################################################## INPUTs

def read_input(fn, separator=""):
    sequences = []
    lens = []
    with open(fn, "r") as fp:
        for line in fp:
            if separator == "":
                a = list(line.strip())
            else:
                a = line.strip().split(separator)
            sequences.extend(a)
            lens.append(len(a))
    return sequences, lens


def load_bicinia(dir_name):
    seq1 = []
    seq2 = []
    len1 = []
    len2 = []
    for file in os.listdir(dir_name):
        if fnmatch.fnmatch(file, "*.mid.txt"):
            with open(dir_name + file, "r") as fp:
                lines = fp.readlines()
                a = lines[0].strip().split(" ")
                seq1.extend(a)
                len1.append(len(a))
                # lines[1] is empty
                b = lines[2].strip().split(" ")
                seq2.extend(list(b))
                len2.append(len(b))
    return seq1, seq2, len1, len2


def load_bicinia_single(dir_name, seq_n=1):
    seq = []
    idx = 0  # read first sequence
    if seq_n != 1:
        # read second sequence"
        idx = 2  # line 1 is empty
    for file in os.listdir(dir_name):
        if fnmatch.fnmatch(file, "*.mid.txt"):
            with open(dir_name + file, "r") as fp:
                lines = fp.readlines()
                seq.append(lines[idx].strip().split(" "))
    return seq


def load_cello(dir_name):
    seq = []
    for file in os.listdir(dir_name):
        if fnmatch.fnmatch(file, "*.mid.txt"):
            with open(dir_name + file, "r") as fp:
                seq.append(fp.readline().strip().split(" "))
    return seq


def convert_songs(dir_name, fn):
    seq = []
    for file in os.listdir(dir_name):
        if fnmatch.fnmatch(file, "*cpt.txt"):
            with open(dir_name + file, "r") as fp:
                seq.append(fp.readline().strip())
    with open("data/" + fn + ".txt", "w") as fp:
        for line in seq:
            fp.write(f"{line}\n")
    return seq


def load_bach(dir_name):
    seq = []
    for file in os.listdir(dir_name):
        if fnmatch.fnmatch(file, "*cpt.txt"):
            with open(dir_name + file, "r") as fp:
                seq.append(fp.readline().strip().split(" "))
    return seq


def load_bicinia_full(dir_name):
    seq1 = []
    seq2 = []
    for file in os.listdir(dir_name):
        if fnmatch.fnmatch(file, "*.mid.txt"):
            with open(dir_name + file, "r") as fp:
                lines = fp.readlines()
                a = lines[0].strip().split(" ")
                seq1.append(a)
                # lines[1] is empty
                b = lines[2].strip().split(" ")
                seq2.append(b)
    return seq1, seq2


def read_spaced(filename):
    seq = []
    with open(filename, "r") as fp:
        for line in fp.readlines():
            seq.append(line.strip().split(" "))
    return seq


def read_spaced_repeated(filename):
    seq = []
    with open(filename, "r") as fp:
        for line in fp.readlines():
            seq.append(line.strip().split(" "))
            seq.append(line.strip().split(" "))
            seq.append(line.strip().split(" "))
    return seq


def read_sequences(rng, fn):
    seqs = []
    if fn == "saffran":
        # load/generate Saffran input
        seqs = generate_Saffran_sequence(rng)
    elif fn == "bicinia":
        seqs = load_bicinia_single("data/bicinia/", seq_n=2)
    elif fn == "cello":
        seqs = load_cello("data/" + fn + "/")
    elif fn == "bach_compact":
        seqs = load_bach("data/bach_compact/")
    elif fn == "miller":
        seqs = generate_miller(rng)
    elif fn == "isaac" or fn == "hello" or fn == "mediapipe":
        seqs = read_words("data/" + fn + ".txt")
    elif fn == "all_irish-notes_and_durations-abc" or fn == "bach_preludes" or fn == "ocarolan" or fn == "scottish" or fn == "bach_scot":
        # split lines by space
        seqs = read_spaced("data/" + fn + ".txt")
        # seqs = convert_songs("data/" + fn + "/", fn)
    else:
        with open("data/{}.txt".format(fn), "r") as fp:
            # split lines char by char
            seqs = [list(line.strip()) for line in fp]
    return seqs


def generate_Saffran_sequence_segmented(rng):
    words = ["babupu", "bupada", "dutaba", "patubi", "pidabu", "tutibu"]
    res = []
    prev = ""
    # for x in range(449):  # looser criterion
    for x in range(91):  # strict criterion
        ss = ""
        for y in range(10):
            ww = rng.choice(words)
            # no repeated words in succession
            while ww == prev:
                ww = rng.choice(words)
            prev = ww
            ss += ww
        res.append(list(ss))
    # p = ["tuti","buduta","batu","tibupa","tu","bi"]  # for testing

    return res


def generate_Saffran_sequence_single_array(rng):
    words = ["babupu", "bupada", "dutaba", "patubi", "pidabu", "tutibu"]
    prev = ""
    res = []
    for _ in range(910):  # 910: strict criterion, 449: looser criterion
        ww = rng.choice(words)
        # no repeated words in succession
        while ww == prev:
            ww = rng.choice(words)
        prev = ww
        res += list(ww)
    return [res]


def generate_Saffran_sequence(rng):
    words = ["babupu", "bupada", "dutaba", "patubi", "pidabu", "tutibu"]
    prev = ""
    res = []
    for _ in range(91):  # 910: strict criterion, 449: looser criterion
        seq = []
        for _ in range(10):  # 910: strict criterion, 449: looser criterion
            ww = rng.choice(words)
            # no repeated words in succession
            while ww == prev:
                ww = rng.choice(words)
            prev = ww
            seq += list(ww)
        res.append(seq)
    return res


def generate_Saffran_sequence_exp2(rng):
    words = ["pabiku", "tibudo", "golatu", "daropi"]
    prev = ""
    res = []
    for x in range(910):  # 910: strict criterion, 449: looser criterion
        ww = rng.choice(words)
        # no repeated words in succession
        while ww == prev:
            ww = rng.choice(words)
        prev = ww
        res += list(ww)
    return [res]


def read_words(fn):
    seqs = []
    with open(fn, "r") as fp:
        # split lines word by word
        seqs = [line.strip().split(" ") for line in fp]
    return seqs


def generate_miller(rng, nm="L1"):
    seqs = {
        "L1": ["SSXG", "NNXSG", "SXSXG", "SSXNSG", "SXXXSG", "NNSXNSG", "SXSXNSG", "SXXXSXG", "SXXXXSG"],
        "L2": ["NNSG", "NNSXG", "SXXSG", "NNXSXG", "NNXXSG", "NNXXSXG", "NNXXXSG", "SSXNSXG", "SSXNXSG"],
        "R1": ["GNSX", "NSGXN", "XGSSN", "SXNNGN", "XGSXXS", "GSXXGNS", "NSXXGSG", "SGXGGNN", "XXGNSGG"],
        "R2": ["NXGS", "GNXSG", "SXNGG", "GGSNXG", "NSGNGX", "NGSXXNS", "NGXXGGN", "SXGXGNS", "XGSNGXG"],
    }
    res = []
    for x in range(10):
        setw = seqs[nm]
        for ww in rng.choice(setw, len(setw), replace=False):
            res.append(list(ww))
    return res


def read_percept(rng, mem, sequence, higher_list=None, old_seq=None, ulens=None, tps=None, method=""):
    """Return next percept in sequence as an ordered array of units in mem or components (bigrams)"""
    if ulens is None:
        ulens = [2]  # default like parser (bigrams)
    if not old_seq:
        old_seq = []
    res = []
    # number of units embedded in next percepts
    i = rng.integers(low=1, high=4)
    s = sequence
    actions = []
    while len(s) > 0 and i != 0:
        action = ""
        unit = []
        # units_list = [(k,v) for k,v in mem.items() if s.startswith(k)]
        units_list = [k for k in mem.keys() if " ".join(s).startswith(k)]
        h_list = []
        # if higher_list:
        #     h_list = [k for k in higher_list if " ".join(s).startswith(k)]
        # action = ""
        # if len(s) <= max(ulens):
        #     unit = s
        #     action = "end"
        # el
        if h_list:
            unit = (sorted(h_list, key=lambda key: len(key), reverse=True)[0]).strip().split(" ")
            # print("mem unit:", unit)
            action = "high_mem"
        elif units_list:
            # a unit in mem matched
            unit = (sorted(units_list, key=lambda key: len(key), reverse=True)[0]).strip().split(" ")
            print("mem unit:", unit)
            action = "mem"
        elif tps:
            if method == "BRENT":
                unit = tps.get_next_unit_brent(s[:7], past=old_seq)
            elif method == "CT":
                unit = tps.get_next_certain_unit(s[:7], past=old_seq)
            elif method == "MI":
                unit = tps.get_next_unit_mi(s[:7], past=old_seq)
            else:  # if TPS
                unit = tps.get_next_unit(s[:7], past=old_seq)
                # unit = tps.get_next_unit_with_AVG(s[:7], past=old_seq)
            action = "tps"

        # if no unit found, pick at random length
        if not unit:
            # unit = s[:2]  # add Parser basic components (bigram/syllable)..
            # random unit
            unit = s[:rng.choice(ulens)]
            print("random unit:", unit)
            action = "rnd"

        # check if last symbol
        sf = s[len(unit):]
        if len(sf) == 1:
            unit += sf
            print("added last:", unit)
        tps.update_avg(tps.get_ftps_sequence(old_seq + unit))
        res.append(" ".join(unit))
        actions.append(action)
        # print("final unit:", unit)
        s = s[len(unit):]
        # for calculating next unit with tps
        old_seq = unit
        i -= 1

    return res, actions


class Encoder:
    """Class for encoding input"""

    def __init__(self):
        self.base_list = string.printable
        self.base_dict = {}

    def base_fit(self, ss):
        for i, s in enumerate(ss):
            self.base_dict[s] = self.base_list[i]

    def base_decode(self, istr):
        ret = ""
        rev_d = {v: k for k, v in self.base_dict.items()}
        for c in istr:
            ret += rev_d[c] + " "
        return ret.strip()

    def base_encode(self, sym):
        return self.base_dict[sym]


############################################################## PLOT

def plot_matrix(data, x_labels=None, y_labels=None, fileName="", title="transition matrix", clim=True):
    """Plot"""
    nr, nc = data.shape
    plt.imshow(data, cmap="plasma")
    if clim:
        plt.clim(0, 1.0)
    ax = plt.gca()
    # Major ticks
    ax.set_xticks(np.arange(0, nc, 1))
    ax.set_yticks(np.arange(0, nr, 1))
    # Labels for major ticks
    if y_labels is not None:
        ax.set_yticklabels(y_labels)
    if x_labels is not None:
        ax.set_xticklabels(x_labels)
    # Minor ticks
    ax.set_xticks(np.arange(-.5, nc, 1), minor=True)
    ax.set_yticks(np.arange(-.5, nr, 1), minor=True)
    # Gridlines based on minor ticks
    ax.grid(which='minor', color='w', linestyle='-', linewidth=0.5)
    plt.colorbar()
    plt.xticks(fontsize=6)
    plt.gcf().autofmt_xdate()
    # plt.yticks(fontsize=10)
    ax.set_title(title)
    if fileName:
        plt.savefig("fileName.png", bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


def plot_gra(d, filename="tps"):
    gra = Digraph(comment='TPs')
    added = set()
    for k, v in d.items():
        if k not in added:
            gra.node(k)
            added.add(k)
        for k2, v2 in v.items():
            if v2 > 0.2:
                if k2 not in added:
                    gra.node(k)
                    added.add(k)
                gra.edge(k, k2, label="{:.2f}".format(v2))

    print(gra.source)
    gra.render(filename, view=True)


def plot_nx_creativity(G, filename="tps", ):
    gra = Digraph()
    for k, v, d in G.edges(data=True):
        # gra.edge(k, v, label="{:.2f}-{:.2f}-{:.2f}".format(float(d["p"]),float(d["u"]),float(d["v"])))
        c = (1 - float(d["p"])) * float(d["u"]) * (0.67 - float(d["v"]))
        gra.edge(k, v, label="{:.2f} ({:.2f},{:.2f},{:.2f})".format(c, float(d["p"]), float(d["u"]), float(d["v"])))
    print(gra.source)
    gra.render(filename, view=True)


def plot_gra_from_normalized(tps, filename="", render=False, thresh=0.0):
    gra = Digraph()  # comment='Normalized TPS'
    added = set()
    rows, cols = tps.norm_mem.shape
    for i in range(rows):
        li = tps.le_rows.inverse_transform([i])[0]
        if li not in added:
            gra.node(li, label="{} ({:.3f})".format(li, tps.state_entropies[li]))
            added.add(li)
        for j in range(cols):
            if tps.norm_mem[i][j] > thresh:
                lj = tps.le_cols.inverse_transform([j])[0]
                p = tps.norm_mem[i][j]
                if p == 1.0:
                    gra.edge(li, lj, label="{:.3f}".format(p), p=str(p), u="0", v="0", penwidth=str(3), color="red")
                else:
                    gra.edge(li, lj, label="{:.3f}".format(p), p=str(p), u="0", v="0", penwidth=str(3 * p))
    # print(gra.source)
    if render:
        gra.render(filename, view=False, engine="dot", format="pdf")
        os.rename(filename, filename + '.dot')
    else:
        gra.save(filename + '.dot')


# bar-plot memory content
def plot_mem(mem, fig_name="plt_mem.png", show_fig=True, save_fig=False):
    plt.clf()
    # plt.subplots_adjust(bottom=0.15)
    # plt.rcParams["figure.figsize"] = (18,5)
    plt.bar(range(len(mem)), list(mem.values()), align='center')
    # plt.gcf().autofmt_xdate()
    plt.xticks(range(len(mem)), list(mem.keys()), rotation=90)

    if save_fig:
        plt.savefig(fig_name, bbox_inches='tight')
    if show_fig:
        plt.tight_layout()
        plt.show()


def plot_actions(actions, path="", show_fig=True):
    plt.clf()
    plt.plot(actions, ".", markersize=1)
    if path:
        plt.savefig(path + "actions.pdf", bbox_inches='tight')
    if show_fig:
        plt.tight_layout()
        plt.show()


def plot_actions2(actions, path="", show_fig=True):
    nr = 0
    nt = 0
    nm = 0
    tot = 0
    resm = []
    rest = []
    resr = []
    for a in actions:
        tot += 1
        if a == "mem":
            nm += 1
        elif a == "tps":
            nt += 1
        elif a == "rnd":
            nr += 1
        resm.append(nm / tot)
        rest.append(nt / tot)
        resr.append(nr / tot)
    plt.clf()
    plt.plot(resm, label="memory")
    plt.plot(rest, label="tps")
    plt.plot(resr, label="rnd")
    plt.title('segmentation')
    plt.xlabel('observations')
    plt.ylabel('%')
    plt.legend(loc="upper right")

    if path:
        plt.savefig(path + "actions2.pdf")
    if show_fig:
        plt.tight_layout()
        plt.show()


def plot_tps_sequences(cm, gens, fi_dir=""):
    fig, axs = plt.subplots(3, 1, figsize=(16, 9), gridspec_kw={'height_ratios': [1, 1, 2]})
    axs[0].set_title("ftps")
    axs[1].set_title("mis")
    axs[2].axis('off')
    axs[2].axis("tight")
    ll = 0
    for gg in gens:
        x1 = cm.tps_1.get_ftps_sequence(gg.split(" "))
        x2 = cm.tps_1.get_mis_sequence(gg.split(" "))
        axs[0].plot(x1)
        axs[1].plot(x2)
        axs[2].set_xlim(axs[0].get_xlim())
        axs[2].set_ylim([0, 40])
        for i, x in enumerate(gg.split(" ")[cm.tps_1.order:]):
            axs[2].text(i, ll, '{}'.format(x))
            # axs[1].text(i, ll, '{}'.format(x))
        ll += 2
    # plt.show()
    plt.savefig(fi_dir + "tps_plot.png", bbox_inches='tight')
    plt.close('all')


############################################################## GENERATION

# from transition probabilities, generates (occ) sequences
def generate(rng, tps, n_seq, occ_per_seq=16):
    res = dict()
    for order in tps.keys():
        res[order] = list()
        if int(order) == 0:
            for _ns in range(0, n_seq):
                str_res = ""
                for _ops in range(0, occ_per_seq):
                    idx = mc_choice(rng, list(tps[order].values()))
                    str_res += " " + list(tps[order].keys())[idx]
                res[order].append(str_res.strip(" "))
        else:
            for _ns in range(0, n_seq):
                # first choice
                str_res = rng.choice(list(tps[order].keys()))
                sid = str_res
                # all other occs
                for _ops in range(0, occ_per_seq - order):
                    #  ending symbol, no further nth-order transition
                    # cut first symbol and search for the order-1 transition
                    i = 0
                    while i < order and (sid not in tps[order - i].keys()):
                        sid = " ".join(sid.split(" ")[1:])
                        i += 1
                    if sid:
                        val = tps[order - i][sid]
                        idx = mc_choice(rng, list(val.values()))
                        str_res += " " + list(val.keys())[idx]
                    else:
                        # choose a symbol of the 0-th level
                        idx = mc_choice(rng, list(tps[0].values()))
                        val = list(tps[0].keys())[idx]
                        str_res += " " + val

                    sid = " ".join(str_res.split(" ")[-order:])
                res[order].append(str_res)
    return res


def multi_generation(rng, cm1, cm2, mim):
    init_set = cm1.initial_set
    res = ""
    for _ in range(0, 50):
        gg = cm1.tps_units.generate_new_next(rng, initials=init_set)
        if not gg:
            return res
        res += " " + gg
        init_set = mim.get_associated(str(gg))
        gg2 = cm2.tps_units.generate_new_next(rng, initials=init_set)
        if not gg2:
            return res
        res += " " + gg2
        init_set = mim.get_associated(str(gg2))
    return res


def multi_generation_ass(rng, cm, mim):
    init_set = cm.initial_set
    res = ""
    for _ in range(0, 50):
        gg = cm.tps_units.generate_new_next(rng, initials=init_set)
        tt = mim.get_associated(str(gg), fun=list)
        t = mc_choice(rng, tt)
        res += " " + gg + "-" + t

    return res


def generate_seqs(rng, model, res):
    model.tps_units.normalize()
    res["generated"] = model.tps_units.generate_new_seqs(rng, min_len=100)
    im = dict(sorted([(x, y) for x, y in model.pars.mem.items()], key=lambda it: it[1], reverse=True))
    res["mem"] = im


def convert_french_phono():
    data = []
    with open("data/french_phono.txt", "r") as fp:
        for line in fp:
            data.append(" ".join(line.strip().split()[5:]))

    with open("data/french_phono_converted.txt", "w") as fp:
        for line in data:
            fp.write(line + "\n")
