#!/usr/bin/env python

import sys
import interval_functions as intfun
import numpy as np
import math

with open(sys.argv[1], 'r') as f:
    line = f.readline().split()

melody = intfun.key2accidentals(line, 'G')
intervals = intfun.compute_intervals(melody)
nbins = 24
interval_hist = np.histogram(intervals, bins=np.arange(nbins), density=True)

reference_hist = [[1.80121768e-01, 1.32126196e-01, 3.64434253e-01, 1.19079624e-01,
                   7.47212778e-02, 7.78840832e-02, 1.02791176e-03, 1.47861153e-02,
                   1.05953981e-02, 9.56748636e-03, 3.40001581e-03, 3.16280541e-04,
                   9.40934609e-03, 2.37210406e-04, 3.16280541e-04, 4.74420811e-04,
                   6.32561082e-04, 8.69771487e-04, 0.00000000e+00, 0.00000000e+00,
                   0.00000000e+00, 0.00000000e+00, 0.00000000e+00], [np.arange(nbins)]]  # from all_irish

# arbitrary defnd reference histogram
# v = [0] * nbins
# v[11] = 10
# v[12] = 11
# v[14] = 3
# v[17] = 5
# reference_hist = np.histogram(v,bins=np.arange(nbins),density=True)

# print(reference_hist[0])
# print(interval_hist[0])

a = np.array(reference_hist[0])
b = np.array(interval_hist[0])

u = 1.0 / (math.dist(a, b) + 1)

print(u)
