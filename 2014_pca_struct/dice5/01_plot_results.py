# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 11:29:39 2014

@author: md238665



"""

import os

import pandas as pd
import numpy as np
import scipy
import matplotlib
import matplotlib.pylab as plt

import dice5_pca

################
# Input/Output #
################

INPUT_DIR = "/neurospin/brainomics/2014_pca_struct/dice5"
INPUT_RESULTS = os.path.join(INPUT_DIR, "consolidated_results.csv")

##############
# Parameters #
##############

SNRS = np.array([0.1, 0.5, 1.0])

##########
# Script #
##########

# Read data
total_df = pd.io.parsers.read_csv(INPUT_RESULTS,
                                  index_col=[0, 1, 2])

# Return indices to columns
total_df_no_ind = total_df.reset_index()
group_model_key = total_df_no_ind.groupby(['model', 'key'])
group_model_key_names = group_model_key.groups

# Plot for each (model, key)
plt.figure()
for group in group_model_key_names:
    g = group_model_key.get_group(group)
    plt.plot(SNRS, g['evr_0'])
plt.legend(group_model_key_names)

plt.figure()
axes = plt.gca()
axes.set_color_cycle(['r', 'g', 'b', 'y', 'c', 'm', 'y', 'k'])
evr = None
for snr in SNRS:
    mask = total_df_no_ind['SNR'] == snr
    val = total_df_no_ind['evr_0'].loc[mask].values
    if evr is None:
        evr = val
    else:
        evr = np.vstack([evr, val])
#b = total_df_no_ind['evr_0'].loc[SNR_1].values
#c = np.vstack([a, b])
n, p = evr.shape
width=0.5/(p+1)
for i in range(p):
    plt.bar(SNRS+i*width, evr[:,i], width=width)

# Plot for each comp
plt.figure()
for group in group_model_key_names:
    g = group_model_key.get_group(group)
    g.index = SNRS
    plt.plot([g['evr_0'][1.0], g['evr_1'][1.0], g['evr_2'][1.0]])
plt.legend(group_model_key_names)

# Plot for each comp
#plt.figure()
#for group in group_model_key_names:
#    g = group_model_key.get_group(group)
#    g.index = INPUT_SNRS
#    plt.plot([g['evr_0'][0.5], g['evr_1'][0.5], g['evr_2'][0.5]])
plt.legend(group_model_key_names)

plt.show()