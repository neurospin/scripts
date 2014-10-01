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

This file was copied from scripts/2014_pca_struct/dice5/03_report_results.py.

"""

import os
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import nibabel as nib

import pandas as pd

################
# Input/Output #
################

INPUT_BASE_DIR = "/neurospin/brainomics/2014_pca_struct/Olivetti_faces"
INPUT_DIR = os.path.join(INPUT_BASE_DIR,
                         "results")
INPUT_RESULTS_FILE = os.path.join(INPUT_BASE_DIR, "results.csv")

OUTPUT_DIR = INPUT_BASE_DIR
OUTPUT_RESULTS_FILE = os.path.join(OUTPUT_DIR, "summary.csv")

##############
# Parameters #
##############

GLOBAL_PEN = 1.0
COND = [(('pca', 0.0, 0.0, 0.0), 'Ordinary PCA'),
        (('struct_pca', GLOBAL_PEN, 1.0, 0.0), 'Pure TV'),
        (('struct_pca', GLOBAL_PEN, 0.0, 1.0), 'Pure l1'),
        (('struct_pca', GLOBAL_PEN, 0.0, 0.0), 'Pure l2'),
        (('struct_pca', GLOBAL_PEN, 0.5, 1.0), 'l1 + TV'),
        (('struct_pca', GLOBAL_PEN, 0.5, 0.0), 'l2 + TV'),
        (('struct_pca', GLOBAL_PEN, 0.0, 0.5), 'l1 + l2'),
        (('struct_pca', GLOBAL_PEN, 0.33, 0.5), 'l1 + l2 + TV')
       ]
PARAMS = [item[0] for item in COND]
COLS = ['frobenius_test']
FOLD = 0  # This is the special fold with the whole dataset
N_COMP = 3
COMPONENTS_FILE_FORMAT = os.path.join(INPUT_DIR,
                                      '{fold}',
                                      '{key}',
                                      'components.npz')
IM_SHAPE = (64, 64)

##########
# Script #
##########

# Open result file (index by model, total_penalization, tv_ratio, l1_ratio)
# We have to explicitly sort the index in order to subsample
df = pd.io.parsers.read_csv(INPUT_RESULTS_FILE,
                            index_col=[1, 2, 3, 4]).sort_index()

# Subsample it & add a column based on name
summary = df.loc[PARAMS][COLS]
name_serie = pd.Series([item[1] for item in COND], name='Name',
                       index=PARAMS)
summary['name'] = name_serie

# Write in a CSV
summary.to_csv(OUTPUT_RESULTS_FILE)

## Plot Fronenius distance
#width = 0.8
#ind = np.arange(len(COND))
#plt.figure()
#ax = plt.gca()
#plt.xticks(rotation=70)
#plt.bar(ind, summary[COLS[1]], width)
#y_range = [min(summary[COLS[1]]), max(summary[COLS[1]])]
#y_lim = plt.ylim()
#plt.ylim(0.95 * y_range[0], y_lim[1])
#ax.set_xticks(ind + (width / 2))
#ax.set_xticklabels(summary['name'])
#plt.title('Olivetti faces')

# Load components
components = np.zeros((len(COND), np.prod(IM_SHAPE), N_COMP))
for j, (params, _) in enumerate(COND):
    key = '_'.join([str(param) for param in params])
    filename = COMPONENTS_FILE_FORMAT.format(fold=FOLD,
                                             key=key)
    if os.path.exists(filename):
        components[j, ...] = np.load(filename)['arr_0']
    else:
        print "No components for", COND[j][1]
data_min = components.min()
data_max = components.max()
# Create a symetric colormap
# This assume that max is positive and min is negative
vmax = max([np.abs(data_min), np.abs(data_max)])
vmin = -data_max
norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
cmap = matplotlib.cm.jet

# Plot components
handles = np.zeros((len(COND)), dtype='object')
for j, (params, name) in enumerate(COND):
    handles[j] = fig, axes = plt.subplots(nrows=1,
                                          ncols=N_COMP,
                                          figsize=(11.8, 3.7))
    f = plt.gcf()
    for l, axe in zip(range(N_COMP), axes.flat):
        data = components[j, :, l].reshape(IM_SHAPE)
        im = axe.imshow(data, norm=norm, aspect="auto",
                        cmap=cmap)
    figtitle = "{name}".format(name=name,
                               fold=FOLD)
    figname = figtitle.replace(' ', '_')
    plt.suptitle(figtitle)
#            fig.subplots_adjust(right=0.8)
#            cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
#            fig.colorbar(im, cax=cbar_ax)
    f.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR,
                             ".".join([figname, "png"])))
# Create a colobar
# http://matplotlib.org/examples/api/colorbar_only.html
fig = plt.figure(figsize=(8, 1))
ax = fig.add_axes([0.05, 0.30, 0.9, 0.35])
cb = matplotlib.colorbar.ColorbarBase(ax, cmap=cmap,
                                      norm=norm,
                                      orientation='horizontal')
fig.savefig(os.path.join(OUTPUT_DIR,
                         "colorbar.png"))
