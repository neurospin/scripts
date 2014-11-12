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

This file was copied from scripts/2014_pca_struct/Olivetti_faces/03_report_results.py.

"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

import nibabel as nib

import pandas as pd

from brainomics import plot_utilities

################
# Input/Output #
################

INPUT_BASE_DIR = "/neurospin/brainomics/2014_pca_struct/mescog/mescog_5folds"
INPUT_DIR = os.path.join(INPUT_BASE_DIR,
                         "results")
INPUT_RESULTS_FILE = os.path.join(INPUT_BASE_DIR, "results.csv")

INPUT_MASK = os.path.join(INPUT_BASE_DIR,
                          "mask_bin.nii")

OUTPUT_DIR = INPUT_BASE_DIR
OUTPUT_RESULTS_FILE = os.path.join(OUTPUT_DIR, "summary.csv")

##############
# Parameters #
##############

N_COMP = 3

METRICS = ['correlation_mean',
           'kappa_mean',
           'frobenius_test']

# Plot of metrics
EXAMPLE_MODEL = 'struct_pca'
CURVE_FILE_FORMAT = os.path.join(OUTPUT_DIR,
                                 '{metric}_{global_pen}.png')

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
COLS = METRICS
EXAMPLE_FOLD = 0
INPUT_COMPONENTS_FILE_FORMAT = os.path.join(INPUT_DIR,
                                            '{fold}',
                                            '{key}',
                                            'components.npz')
IM_SHAPE = (182, 218, 182)
OUTPUT_COMPONENTS_FILE_FORMAT = os.path.join(OUTPUT_DIR,
                                             '{name}.nii')

##########
# Script #
##########

# Open mask
babel_mask = nib.load(INPUT_MASK)
bin_mask = babel_mask.get_data() != 0
mask_indices = np.where(bin_mask)
n_voxels_in_mask = np.count_nonzero(bin_mask)

# Open result file (index by model, total_penalization, tv_ratio, l1_ratio)
# We have to explicitly sort the index in order to subsample
df = pd.io.parsers.read_csv(INPUT_RESULTS_FILE,
                            index_col=[1, 2, 3, 4]).sort_index()

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

## Plot Fronenius distance
#width = 0.8
#ind = np.arange(len(COND))
#plt.figure()
#ax = plt.gca()
#plt.xticks(rotation=70)
#plt.bar(ind, summary[COLS[0]], width)
#y_range = [min(summary[COLS[0]]), max(summary[COLS[0]])]
#y_lim = plt.ylim()
#plt.ylim(0.95 * y_range[0], y_lim[1])
#ax.set_xticks(ind + (width / 2))
#ax.set_xticklabels(summary['name'])
#plt.title('Mescog')

# Subsample df
struct_pca_df = df.xs(EXAMPLE_MODEL)

# Plot some metrics for struct_pca
for metric in METRICS:
    handles = plot_utilities.plot_lines(struct_pca_df,
                                        x_col=1,
                                        y_col=metric,
                                        splitby_col=0,
                                        colorby_col=2)
    for val, handle in handles.items():
        filename = CURVE_FILE_FORMAT.format(metric=metric,
                                            global_pen=val)
        handle.savefig(filename)

###################
# Plot components #
###################

## Load components and store them as nifti images
#for j, (params, name) in enumerate(COND):
#    key = '_'.join([str(param) for param in params])
#    filename = INPUT_COMPONENTS_FILE_FORMAT.format(fold=EXAMPLE_FOLD,
#                                                   key=key)
#    if os.path.exists(filename):
#        components = np.load(filename)['arr_0']
#    else:
#        print "No components for", COND[j][1]
#    im_data = np.zeros((IM_SHAPE[0], IM_SHAPE[1], IM_SHAPE[2], N_COMP))
#    for l in range(N_COMP):
#        im_data[mask_indices[0], mask_indices[1], mask_indices[2], l] = components[:, l]
#    im = nib.Nifti1Image(im_data,
#                         affine=babel_mask.get_affine())
#    figname = OUTPUT_COMPONENTS_FILE_FORMAT.format(name=name.replace(' ', '_'))
#    nib.save(im, os.path.join(OUTPUT_DIR, figname))
