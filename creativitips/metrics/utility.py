#!/usr/bin/env python

import sys
from metrics import interval_functions as intfun
import numpy as np
import math


#############################################################################################
def utility(melody, allowed_intervals):
    # Input: melody (with accidentals) and array of allowed intervals
    mybins = np.arange(-12, 12)
    rhist = np.histogram(allowed_intervals, bins=mybins, density=True)
    intervals = intfun.compute_intervals(melody)
    interval_hist = np.histogram(intervals, bins=mybins, density=True)
    a = np.array(rhist[0])
    b = np.array(interval_hist[0])
    u = 1.0 / (math.dist(a, b) + 1)
    return u


if __name__ == "__main__":
    with open(sys.argv[1], 'r') as f:
        line = f.readline().split()
    # now line contains the list of strings representing a melody in abc notation

    mel = intfun.key2accidentals(line, 'G')  # Hp: key of G
    ints = [4, 3, -3, -4, 9, -9]  # only thirds and sixths
    u = utility(mel, ints)

    print(u)
