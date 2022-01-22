import fnmatch
import os
import string
from graphviz import Digraph
import matplotlib.pyplot as plt
import numpy as np
# import networkx as nx
# from networkx.drawing.nx_pydot import write_dot


def matrix_from_tps(tps_dict, x_encoding, y_encoding):
    res = np.zeros((len(y_encoding.classes_), len(x_encoding.classes_)))
    for start, ems in tps_dict.items():
        for v, k in ems.items():
            res[y_encoding.transform([start])[0]][x_encoding.transform([v])[0]] = k
    return res


# return an index using MonteCarlo choice on arr
def mc_choice(arr):
    rnd = np.random.uniform()
    sm = arr[0]
    j = 1
    ind = 0
    while sm < rnd:
        sm += float(arr[j])
        if sm >= rnd:
            ind = j
        j += 1
    return ind


# gen with hebb associations
def hebb_gen(sequence, hm):
    out = []
    for s in sequence:
        idx = mc_choice(hm[s])
        out.append(idx)
    return out


def plot_matrix(data, x_labels=None, y_labels=None, fileName="", title="transition matrix", clim=True):
    nr, nc = data.shape
    plt.imshow(data, cmap="plasma")
    if clim:
        plt.clim(0, 1.0)
    ax = plt.gca()
    # Major ticks
    ax.set_xticks(np.arange(0, nc, 1))
    ax.set_yticks(np.arange(0, nr, 1))
    # Labels for major ticks
    if y_labels > 0:
        ax.set_yticklabels(y_labels)
    if x_labels > 0:
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


def load_bicinia_single(dir_name, be, seq_n=1):
    seq1 = []
    seq2 = []
    ss = set()
    for file in os.listdir(dir_name):
        if fnmatch.fnmatch(file, "*.mid.txt"):
            with open(dir_name + file, "r") as fp:
                lines = fp.readlines()
                a = lines[0].strip().split(" ")
                seq1.append(a)
                # lines[1] is empty
                b = lines[2].strip().split(" ")
                seq2.append(b)
                if seq_n == 1:
                    ss.update(a)
                else:
                    ss.update(b)
    be.base_fit(ss)
    if seq_n == 1:
        return ["".join([be.base_encode(y) for y in x]) for x in seq1]
    else:
        return ["".join([be.base_encode(y) for y in x]) for x in seq2]


def load_bicinia_full(dir_name, be):
    seq1 = []
    seq2 = []
    ss = set()
    for file in os.listdir(dir_name):
        if fnmatch.fnmatch(file, "*.mid.txt"):
            with open(dir_name + file, "r") as fp:
                lines = fp.readlines()
                a = lines[0].strip().split(" ")
                seq1.append(a)
                # lines[1] is empty
                b = lines[2].strip().split(" ")
                seq2.append(b)
                ss.update(a)
                ss.update(b)
    be.base_fit(ss)
    return ["".join([be.base_encode(y) for y in x]) for x in seq1], ["".join([be.base_encode(y) for y in x]) for x in seq2]


def load_irish_n_d(filename, be):
    seq = []
    ss = set()
    with open(filename, "r") as fp:
        for line in fp.readlines():
            a = line.strip().split(" ")
            seq.append(a)
            ss.update(a)
    be.base_fit(ss)
    return ["".join([be.base_encode(y) for y in x]) for x in seq]


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


def generate_Saffran_sequence():
    words = ["babupu", "bupada", "dutaba", "patubi", "pidabu", "tutibu"]
    res = []
    prev = ""
    # for x in range(449):  # looser criterion
    for x in range(91):  # strict criterion
        ss = ""
        for y in range(10):
            ww = np.random.choice(words)
            # no repeated words in succession
            while ww == prev:
                ww = np.random.choice(words)
            prev = ww
            ss += ww
        res.append(ss)
    # p = ["tuti","buduta","batu","tibupa","tu","bi"]  # for testing

    return res


def read_percept(mem, sequence, ulens=None, tps=None):
    """Return next percept in sequence as an ordered array of units in mem or components (bigrams)"""
    if ulens is None:
        # default like parser (bigrams)
        ulens = [2]
    res = []
    # number of units embedded in next percepts
    i = np.random.randint(low=1, high=4)
    s = sequence
    while len(s) > 0 and i != 0:
        # units_list = [(k,v) for k,v in mem.items() if s.startswith(k) and len(k) > 1]
        units_list = [k for k in mem.keys() if s.startswith(k) and len(k) > 1]
        if units_list:
            # a unit in mem matched
            # unit = sorted(units_list, key=lambda item: item[1], reverse=True)[0][0]
            unit = sorted(units_list, key=lambda key: len(key), reverse=True)[0]
            print("unit shape perception:", unit)
            action = "mem"
        else:
            # unit = s[:2]  # add Parser basic components (bigram/syllable)..
            # unit = s[:np.random.choice(ulens)]  # ..or add rnd percept (bigram or trigram..)
            unit = tps.get_next_unit(s[:10])
            # unit = tps.get_next_unit_brent(s[:10])
            print("TPs next unit:", unit)
            action = "tps"
        if unit == "":
            # random unit
            unit = s[:np.random.choice(ulens)]
            print("random unit:", unit)
            action = "rnd"
        # check if last symbol
        sf = s[len(unit):]
        if len(sf) == 1:
            unit += sf
        res.append(unit)
        s = s[len(unit):]
        i -= 1

    return res, action


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


def plot_gra(d):
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
    gra.render('tps', view=True)


def plot_gra_from_normalized(tps, filename="", be=None, thresh=0.0):
    gra = Digraph()  # comment='Normalized TPS'
    added = set()
    rows, cols = tps.norm_mem.shape
    for i in range(rows):
        if be:
            li = be.base_decode(tps.le_rows.inverse_transform([i])[0])
        else:
            li = tps.le_rows.inverse_transform([i])[0]
        if li not in added:
            gra.node(li, label="{} ({:.3f})".format(li, tps.state_entropies[li]))
            added.add(li)
        for j in range(cols):
            if tps.norm_mem[i][j] > thresh:
                if be:
                    lj = be.base_decode(tps.le_cols.inverse_transform([j])[0])
                else:
                    lj = tps.le_cols.inverse_transform([j])[0]
                if tps.norm_mem[i][j] == 1.0:
                    gra.edge(li, lj, label="{:.3f}".format(tps.norm_mem[i][j]), penwidth="2", color="red")
                else:
                    gra.edge(li, lj, label="{:.3f}".format(tps.norm_mem[i][j]))

    # print(gra.source)
    gra.render(filename, view=False, engine="dot", format="pdf")


# bar-plot memory content
def plot_mem(mem, fig_name="plt_mem.png", show_fig=True, save_fig=False):
    plt.clf()
    plt.rcParams["figure.figsize"] = (18,5)
    plt.bar(range(len(mem)), list(mem.values()), align='center')
    # plt.gcf().autofmt_xdate()
    plt.xticks(range(len(mem)), list(mem.keys()), rotation=90)
    if save_fig:
        plt.savefig(fig_name, bbox_inches='tight')
    if show_fig:
        plt.tight_layout()
        plt.show()


def plot_actions(actions, path=""):
    plt.clf()
    plt.plot(actions, ".")
    if path:
        plt.savefig(path + "actions.pdf", bbox_inches='tight')
    plt.tight_layout()
    plt.show()


# from transition probabilities, generates (occ) sequences
def generate(tps, n_seq, occ_per_seq=16):
    res = dict()
    for order in tps.keys():
        res[order] = list()
        if int(order) == 0:
            for _ns in range(0, n_seq):
                str_res = ""
                for _ops in range(0, occ_per_seq):
                    idx = mc_choice(list(tps[order].values()))
                    str_res += " " + list(tps[order].keys())[idx]
                res[order].append(str_res.strip(" "))
        else:
            for _ns in range(0, n_seq):
                # first choice
                str_res = np.random.choice(list(tps[order].keys()))
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
                        idx = mc_choice(list(val.values()))
                        str_res += " " + list(val.keys())[idx]
                    else:
                        # choose a symbol of the 0-th level
                        idx = mc_choice(list(tps[0].values()))
                        val = list(tps[0].keys())[idx]
                        str_res += " " + val

                    sid = " ".join(str_res.split(" ")[-order:])
                res[order].append(str_res)
    return res
