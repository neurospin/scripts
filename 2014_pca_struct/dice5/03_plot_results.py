# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 11:29:39 2014

@author: md238665



"""

import os

from itertools import product

import pandas as pd
import numpy as np

import matplotlib.pylab as plt

from brainomics import plot_utilities

################
# Input/Output #
################

INPUT_DIR = "/neurospin/brainomics/2014_pca_struct/dice5/results"
INPUT_RESULTS_FILE = os.path.join(INPUT_DIR, "consolidated_results.csv")

##############
# Parameters #
##############

N_COMP=3
# Global penalty
GLOBAL_PENALTIES = np.array([1e-3, 1e-2, 1e-1, 1])
# Relative penalties
# 0.33 ensures that there is a case with TV = L1 = L2
TVRATIO = np.array([1, 0.5, 0.33, 1e-1, 1e-2, 1e-3, 0])
L1RATIO = np.array([1, 0.5, 1e-1, 1e-2, 1e-3, 0])

PCA_PARAMS = [('pca', 0.0, 0.0, 0.0)]
SPARSE_PCA_PARAMS = list(product(['sparse_pca'],
                                 GLOBAL_PENALTIES,
                                 [0.0],
                                 [1.0]))
STRUCT_PCA_PARAMS = list(product(['struct_pca'],
                                 GLOBAL_PENALTIES,
                                 TVRATIO,
                                 L1RATIO))

SNRS = np.array([0.1, 0.5, 1.0])

MODEL = 'struct_pca'
CURVE_FILE_FORMAT = os.path.join(INPUT_DIR,
                                 'data_100_100_{snr}',
                                 '{metric}_{global_pen}.png')
METRICS = ['recall_mean', 'fscore_mean']

##########
# Script #
##########

# Read data
df = pd.io.parsers.read_csv(INPUT_RESULTS_FILE,
                            index_col=[0, 2, 3, 4]).sort_index()
struct_pca_df = df.xs(MODEL)

# Plot some metrics for struct_pca for each SNR value
snr_groups = struct_pca_df.groupby('snr')
for snr_val, snr_group in snr_groups:
    for metric in METRICS:
        handles = plot_utilities.plot_lines(snr_group,
                                            x_col=1,
                                            y_col=metric,
                                            splitby_col=0,
                                            colorby_col=2)
        for val, handle in handles.items():
            filename = CURVE_FILE_FORMAT.format(metric=metric,
                                                snr=snr_val,
                                                global_pen=val)
            handle.savefig(filename)

#for snr_val, snr_group in snr_groups:
#    print "SNR:", snr_val
#    for global_pen in GLOBAL_PENALTIES:
#        print "global pen:", global_pen
#        for tv_ratio in SRUCTPCA_TVRATIO:
#            print "tv:", tv_ratio
#            data = snr_group.xs((global_pen, tv_ratio), level=[0, 1])
#            data.sort('recall_mean', inplace=True)
#            print data['recall_mean']
#            raw_input()
