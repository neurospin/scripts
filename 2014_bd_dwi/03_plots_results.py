# -*- coding: utf-8 -*-

"""
Created on Mon august 25 11:55:01 2014

@author: christophe
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pylab as plt


from brainomics import plot_utilities as UP


################
# Input/Output #
################

INPUT_BASE_DIR = "/neurospin/brainomics/2014_bd_dwi"
INPUT_DIR = os.path.join(INPUT_BASE_DIR, "bd_dwi_enettv_csi")
INPUT_RESULTS = os.path.join(INPUT_DIR, "results_CSI.csv")

OUTPUT_DIR = os.path.join(INPUT_DIR, "figures")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


# Parameters

krange = [100, 1000, 10000, 100000, -1]
alphas = [.01, .05, .1, .5, 1.]
ratios = np.array([[1., 0., 1], [0., 1., 1], [.5, .5, 1], [.9, .1, 1],
                       [.1, .9, 1], [.01, .99, 1], [.001, .999, 1]])

# Functions


def find_nearest(a, a0):
    "Element in nd array `a` closest to the scalar value `a0`"
    idx = np.abs(a - a0).argmin()
    return a.flat[idx]


def l1_ratio(l1, l2, tv):
    pos_val = np.array([1., 0., .5, .1, .9, .01, .99, .001, .999])
    if (tv == 1.0):
        return np.nan
    if ((l1 + l2) != 0.0):
        approx_ratio = l1 / (l1 + l2)
        return find_nearest(pos_val, approx_ratio)


# script
results = pd.read_csv(INPUT_RESULTS)
# Compute l1_ratio
results['l1_ratio'] = np.vectorize(l1_ratio)(results['l1'],
                                             results['l2'],
                                             results['tv'])

r1 = results.loc[results.k == 1000]
handles = UP.mapreduce_plot(r1, 'recall_mean', tv_ratio='tv')
for val, handle in handles.items():
    handle.savefig(os.path.join(OUTPUT_DIR, str(val) + '.png'))
plt.show()

## Check some values
#col = r1[['l1_ratio', 'recall_mean']]
#for alpha in alphas:
#    alpha_loc = r1.a == alpha
#    print "alpha:", alpha
#    for tv_ratio in sorted(r1.tv.unique()):
#        tv_ratio_loc = r1.tv == tv_ratio
#        print "tv:", tv_ratio
#        data = col.loc[alpha_loc & tv_ratio_loc]
#        print data.sort('l1_ratio')
#        print data.sort('recall_mean')
#        raw_input()
