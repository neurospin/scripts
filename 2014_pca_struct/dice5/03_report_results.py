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

SNRS = [0.1, 0.2, 0.25, 0.5, 1.0]
N_COMP = 3

METRICS = ['recall_mean',
           'fscore_mean',
           'correlation_mean',
           'kappa_mean',
           'frobenius_test']
METRICS_NAME = ['Mean recall rate',
                'Mean f-score',
                'Mean correlation across folds',
                'Mean $\kappa$ across folds',
                'Mean Frobenius distance on test sample']

TAB_FILE_FORMAT = os.path.join(INPUT_DIR,
                                 '{metric}_summary.tex')

# Plot of metrics
EXAMPLE_MODEL = 'struct_pca'
CURVE_FILE_FORMAT = os.path.join(INPUT_DIR,
                                 'data_100_100_{snr}',
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
COLS = ['snr'] + METRICS
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

# Open result file
df = pd.io.parsers.read_csv(INPUT_RESULTS_FILE)
df_index = df.set_index(['model', 'global_pen', 'tv_ratio', 'l1_ratio'])

#####################
# Create summary df #
#####################

# Subsample it & add a column based on name
summary = df_index.loc[PARAMS][COLS]
name_serie = pd.Series([item[1] for item in COND], name='Name',
                       index=PARAMS)
summary['name'] = name_serie

# Write in a CSV
summary.to_csv(OUTPUT_RESULTS_FILE)

################
# Plot metrics #
################

# Subsample df
struct_pca_df = df[df.model == EXAMPLE_MODEL]

# GroupBy SNR
snr_groups = struct_pca_df.groupby('snr')
# Summary per SNR value (pivot the table to have better display)
for metric, metric_name in zip(METRICS, METRICS_NAME):
    summary = pd.DataFrame(snr_groups[metric].describe()).unstack(1)[0]
    filename = TAB_FILE_FORMAT.format(metric=metric)
    summary.to_latex(open(filename, 'w'))
# Plot some metrics for struct_pca for each SNR value
for snr_val, snr_group in snr_groups:
    for metric, metric_name in zip(METRICS, METRICS_NAME):
        handles = plot_utilities.plot_lines(snr_group,
                                            x_col='tv_ratio',
                                            y_col=metric,
                                            splitby_col='global_pen',
                                            colorby_col='l1_ratio',
                                            use_suptitle=False)
        for val, handle in handles.items():
            # Tune the figure
            ax = handle.get_axes()[0]
            ax.set_xlabel("TV ratio")
            ax.set_ylabel(metric_name)
            l = ax.get_legend()
            l.set_title("$\ell_1$ ratio")
            s = r'$ \alpha $ =' + str(val)
            handle.suptitle(r'$ \alpha $ =' + str(val))
            filename = CURVE_FILE_FORMAT.format(metric=metric,
                                                snr=snr_val,
                                                global_pen=val)
            handle.savefig(filename)

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
