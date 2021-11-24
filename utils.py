import string
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


def matrix_from_tps(tps_dict,x_encoding,y_encoding):
    res = np.zeros((len(y_encoding.classes_),len(x_encoding.classes_)))
    for start, ems in tps_dict.items():
        for v,k in ems.items():
            res[y_encoding.transform([start])[0]][x_encoding.transform([v])[0]] = k
    return res


def plot_matrix(data, y_labels=string.ascii_lowercase, x_labels=string.ascii_lowercase, save=True, title="transition matrix"):
    fig = plt.figure(figsize=(8,8), dpi=150)
    nr,nc = data.shape
    plt.imshow(data, cmap="plasma")
    ax = plt.gca()
    # Major ticks
    ax.set_xticks(np.arange(0, nc, 1))
    ax.set_yticks(np.arange(0, nr, 1))
    # Labels for major ticks
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)
    # Minor ticks
    ax.set_xticks(np.arange(-.5, nc, 1), minor=True)
    ax.set_yticks(np.arange(-.5, nr, 1), minor=True)
    # Gridlines based on minor ticks
    ax.grid(which='minor', color='w', linestyle='-', linewidth=0.5)
    plt.colorbar()
    # plt.yticks(fontsize=6)
    # plt.yticks(fontsize=10)
    ax.set_title(title)
    if save:
        plt.savefig("tps.png", bbox_inches='tight')
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


def mtx_from_multi(seq1,seq2):
    mtx = np.zeros((max(set(seq1))+1,max(set(seq2))+1))
    for s1,s2 in zip(seq1,seq2):
        mtx[s1][s2] += 1
    return mtx


def show_counts(hebb_mtx):
    plt.matshow(hebb_mtx, cmap="plasma")
    plt.title('hebb_mtx')
    plt.tight_layout()
    plt.show()
