#!/usr/bin/env python
import sys
import math
import numpy as np
from scipy.special import rel_entr
from metrics import interval_functions as intfun


def utility(melody, allowed_intervals):
    # Input: melody (with accidentals) and array of allowed intervals
    mybins = np.arange(-12, 13)
    rhist = np.histogram(allowed_intervals, bins=mybins, density=True)
    intervals = intfun.compute_intervals(melody)
    interval_hist = np.histogram(intervals, bins=mybins, density=True)
    a = np.array(rhist[0])
    b = np.array(interval_hist[0])
    return 1.0 / (math.dist(a, b) + 1)


def utility_kl(melody, allowed_intervals):
    # Input: melody (with accidentals) and array of allowed intervals
    mybins = np.arange(-12, 13)
    rhist = np.histogram(allowed_intervals, bins=mybins, density=True)
    intervals = intfun.compute_intervals(melody)
    interval_hist = np.histogram(intervals, bins=mybins, density=True)
    Q = np.array(rhist[0])
    P = np.array(interval_hist[0])
    kl = sum(rel_entr(Q, P))
    return 1.0 / (kl + 1)


if __name__ == "__main__":
    with open(sys.argv[1], 'r') as f:
        line = f.readline().split()
    # now line contains the list of strings representing a melody in abc notation

    mel = intfun.key2accidentals(line, 'G')  # Hp: key of G
    ints = [4, 3, -3, -4, 9, -9]  # only thirds and sixths
    u = utility(mel, ints)
    ukl = utility_kl(mel, ints)

    print(u)
    print(ukl)
