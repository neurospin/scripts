# -*- coding: utf-8 -*-
"""
Created on Fri Aug  1 13:28:04 2014

@author: md238665

Report some results:
 - ordinary PCA
 - pure l1, l2 and TV
 - l1+TV, l2+TV and l1+l2
 - l1+l2+TV

We use a constant global penalization.

"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

from brainomics import plot_utilities

################
# Input/Output #
################

INPUT_DIR = "/neurospin/brainomics/2014_pca_struct/dice5/results"
INPUT_RESULTS_FILE = os.path.join(INPUT_DIR, "consolidated_results.csv")

OUTPUT_DIR = INPUT_DIR
OUTPUT_RESULTS_FILE = os.path.join(OUTPUT_DIR, "summary.csv")

##############
# Parameters #
##############

SNRS = np.array([0.1, 0.5, 1.0])
N_COMP = 3

# Plot of metrics
EXAMPLE_MODEL = 'struct_pca'
CURVE_FILE_FORMAT = os.path.join(INPUT_DIR,
                                 'data_100_100_{snr}',
                                 '{metric}_{global_pen}.png')
METRICS = ['recall_mean', 'fscore_mean']

# Plot of components
EXAMPLE_GLOBAL_PEN = 1.0
COND = [(('pca', 0.0, 0.0, 0.0), 'Ordinary PCA'),
        (('struct_pca', EXAMPLE_GLOBAL_PEN, 1.0, 0.0), 'Pure TV'),
        (('struct_pca', EXAMPLE_GLOBAL_PEN, 0.0, 1.0), 'Pure l1'),
        (('struct_pca', EXAMPLE_GLOBAL_PEN, 0.0, 0.0), 'Pure l2'),
        (('struct_pca', EXAMPLE_GLOBAL_PEN, 0.5, 1.0), 'l1 + TV'),
        (('struct_pca', EXAMPLE_GLOBAL_PEN, 0.5, 0.0), 'l2 + TV'),
        (('struct_pca', EXAMPLE_GLOBAL_PEN, 0.0, 0.5), 'l1 + l2'),
        (('struct_pca', EXAMPLE_GLOBAL_PEN, 0.33, 0.5), 'l1 + l2 + TV')
       ]
PARAMS = [item[0] for item in COND]
COLS = ['snr', 'correlation_mean', 'frobenius_test']
EXAMPLE_FOLD = 0
COMPONENTS_FILE_FORMAT = os.path.join(INPUT_DIR,
                                      'data_100_100_{snr}',
                                      'results',
                                      '{fold}',
                                      '{key}',
                                      'components.npz')

##########
# Script #
##########

# Open result file (index by model, total_penalization, tv_ratio, l1_ratio)
# We have to explicitly sort the index in order to subsample
df = pd.io.parsers.read_csv(INPUT_RESULTS_FILE,
                            index_col=[0, 2, 3, 4]).sort_index()

#####################
# Create summary df #
#####################

# Subsample it & add a column based on name
summary = df.loc[PARAMS][COLS]
name_serie = pd.Series([item[1] for item in COND], name='Name',
                       index=PARAMS)
summary['name'] = name_serie

# Write in a CSV
summary.to_csv(OUTPUT_RESULTS_FILE)

################
# Plot metrics #
################

# Subsample df
struct_pca_df = df.xs(EXAMPLE_MODEL)

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

## Plot Fronenius distance for a given SNR
#width = 0.8
#ind = np.arange(len(COND))
#for snr in SNRS:
#    plt.figure()
#    ax = plt.gca()
#    plt.xticks(rotation=70)
#    data = summary.loc[summary.snr == snr]
#    plt.bar(ind, data[COLS[2]], width)
#    y_range = [min(data[COLS[2]]), max(data[COLS[2]])]
#    y_lim = plt.ylim()
#    plt.ylim(0.95 * y_range[0], y_lim[1])
#    ax.set_xticks(ind + (width / 2))
#    ax.set_xticklabels(data['name'])
#    plt.title(str(snr))
#
## Plot correlation for a given SNR
#width = 0.8
#ind = np.arange(len(COND))
#for snr in SNRS:
#    plt.figure()
#    ax = plt.gca()
#    plt.xticks(rotation=70)
#    data = summary.loc[summary.snr == snr]
#    plt.bar(ind, data[COLS[1]], width)
#    y_range = [min(data[COLS[1]]), max(data[COLS[1]])]
#    y_lim = plt.ylim()
#    plt.ylim(0.95 * y_range[0], y_lim[1])
#    ax.set_xticks(ind + (width / 2))
#    ax.set_xticklabels(data['name'])
#    plt.title(str(snr))

###################
# Plot components #
###################

# Load components
components = np.zeros((len(SNRS), len(COND), 100*100, N_COMP))
for i, snr in enumerate(SNRS):
    for j, (params, _) in enumerate(COND):
        key = '_'.join([str(param) for param in params])
        filename = COMPONENTS_FILE_FORMAT.format(snr=snr,
                                                 fold=EXAMPLE_FOLD,
                                                 key=key)
        components[i, j, ...] = np.load(filename)['arr_0']
data_min = components.min()
data_max = components.max()
# Plot components
handles = np.zeros((len(SNRS), len(COND)), dtype='object')
for i, snr in enumerate(SNRS):
    for j, (params, name) in enumerate(COND):
        handles[i, j] = fig, axes = plt.subplots(nrows=1,
                                                 ncols=N_COMP,
                                                 figsize=(11.8,3.7))
        f = plt.gcf()
        for l, axe in zip(range(N_COMP), axes.flat):
            data = components[i, j, :, l].reshape(100, 100)
            im = axe.imshow(data, vmin=data_min, vmax=data_max, aspect="auto")
        figtitle = "{name}".format(name=name)
        figname = figtitle.replace(' ', '_')
        plt.suptitle(figtitle)
        #fig.subplots_adjust(right=0.8)
        #cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        #fig.colorbar(im, cax=cbar_ax)
        f.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR,
                                 "data_100_100_{snr}".format(snr=snr),
                                 ".".join([figname, "png"])))
# Create a colobar
# http://matplotlib.org/examples/api/colorbar_only.html
import matplotlib
fig = plt.figure(figsize=(8,1))
ax = fig.add_axes([0.05, 0.30, 0.9, 0.35])
cmap = matplotlib.cm.jet
norm = matplotlib.colors.Normalize(vmin=data_min, vmax=data_max)
cb = matplotlib.colorbar.ColorbarBase(ax, cmap=cmap,
                                      norm=norm,
                                      orientation='horizontal')
fig.savefig(os.path.join(OUTPUT_DIR,
                         "components_colorbar.png"))
