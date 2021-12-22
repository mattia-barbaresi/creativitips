import fnmatch
import os
import string
from graphviz import Digraph
import matplotlib.pyplot as plt
import numpy as np


def matrix_from_tps(tps_dict,x_encoding,y_encoding):
    res = np.zeros((len(y_encoding.classes_),len(x_encoding.classes_)))
    for start, ems in tps_dict.items():
        for v,k in ems.items():
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


def plot_matrix(data, x_labels=[],y_labels=[], fileName="", title="transition matrix", clim=True):
    nr,nc = data.shape
    plt.imshow(data, cmap="plasma")
    if clim:
        plt.clim(0, 1.0)
    ax = plt.gca()
    # Major ticks
    ax.set_xticks(np.arange(0, nc, 1))
    ax.set_yticks(np.arange(0, nr, 1))
    # Labels for major ticks
    if len(y_labels) > 0:
        ax.set_yticklabels(y_labels)
    if len(x_labels) > 0:
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
    return seq1,seq2,len1,len2


def mtx_from_multi(seq1,seq2,nd1,nd2):
    mtx = np.zeros((nd1,nd2))
    for s1,s2 in zip(seq1,seq2):
        mtx[s1][s2] += 1
    return mtx


def show_counts(hebb_mtx):
    plt.matshow(hebb_mtx, cmap="plasma")
    plt.title('hebb_mtx')
    plt.tight_layout()
    plt.show()


def softmax(x):
    sum = np.sum(x, axis=1, keepdims=True)  # returns sum of each row and keeps same dims
    f_x = x / sum
    return f_x


def generate_Saffran_sequence():
    words = ["babupu", "bupada", "dutaba", "patubi", "pidabu", "tutibu"]
    ss = ""
    prev = ""
    # for x in range(0, 910):  # strict criterion
    for x in range(0, 450):  # looser criterion
        ww = np.random.choice(words)
        # no repeated words in succession
        while ww == prev:
            ww = np.random.choice(words)
        prev = ww
        ss += ww
    # p = ["tuti","buduta","batu","tibupa","tu","bi"]  # for testing

    return [ss]


def read_percept(mem, sequence):
    """Return next percept in sequence as an ordered array of units in mem or components (bigrams)"""
    res = []
    # number of units embedded in next percepts
    i = np.random.randint(low=1, high=4)
    s = sequence
    while len(s) > 0 and i != 0:
        units_list = [k for k in mem.keys() if s.startswith(k)]
        if units_list:
            # a unit in mem matched
            unit = sorted(units_list, key=lambda item: len(item), reverse=True)[0]
        else:
            # add thr basic components (bigram)
            unit = s[:2]
        res.append(unit)
        s = s[len(unit):]
        i -= 1
    return res


BASE_LIST = string.ascii_letters + string.digits
BASE_DICT = {}


def base_fit(ss):
    for i,s in enumerate(ss):
        BASE_DICT[s] = BASE_LIST[i]


def base_decode(istr):
    ret = ""
    rev_d = {v: k for k, v in BASE_DICT.items()}
    for c in istr:
        ret += rev_d[c] + " "
    return ret.strip()


def base_encode(sym):
    return BASE_DICT[sym]


def plot_gra(d):
    gra = Digraph(comment='TPs')
    added = set()
    for k,v in d.items():
        if k not in added:
            gra.node(k)
            added.add(k)
        for k2,v2 in v.items():
            if v2 > 0.2:
                if k2 not in added:
                    gra.node(k)
                    added.add(k)
                gra.edge(k,k2,label="{:.2f}".format(v2))

    print(gra.source)
    gra.render('tps', view=True)


def plot_gra_from_m(m, ler, lec):
    gra = Digraph(comment='Normalized TPS > 0.2')
    added = set()
    rows,cols = m.shape
    for i in range(rows):
        li = ler.inverse_transform([i])[0]
        if li not in added:
            gra.node(li)
            added.add(li)
        for j in range(cols):
            if m[i][j] > 0.05:
                lj = lec.inverse_transform([j])[0]
                if lj not in added:
                    gra.node(lj)
                    added.add(lj)
                gra.edge(li,lj,label="{:.2f}".format(m[i][j]))

    print(gra.source)
    gra.render('tps_norm', view=True)
