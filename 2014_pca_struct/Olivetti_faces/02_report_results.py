# -*- coding: utf-8 -*-
"""
Created on Fri Aug  1 13:28:04 2014

@author: md238665

Report some results:
 - ordinary PCA
 - pure l1, l2 and TV
 - l1+TV, l2+TV and l1+l2
 - l1+l2+TV
With a constant global penalization.

Also plot some results for all the structured PCA models.

This file was copied from scripts/2014_pca_struct/dice5/03_report_results.py.

"""

import os
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import pandas as pd

from brainomics import plot_utilities

################
# Input/Output #
################

INPUT_BASE_DIR = "/neurospin/brainomics/2014_pca_struct/Olivetti_faces"
INPUT_DIR = os.path.join(INPUT_BASE_DIR,
                         "results")
INPUT_RESULTS_FILE = os.path.join(INPUT_BASE_DIR, "results.csv")

INPUT_DATASET = os.path.join(INPUT_BASE_DIR,
                             "X.npy")
INPUT_TARGET = os.path.join(INPUT_BASE_DIR,
                            "y.npy")
INPUT_COMPONENTS_FILE_FORMAT = os.path.join(INPUT_DIR,
                                            '{fold}',
                                            '{key}',
                                            'components.npz')
INPUT_TEST_PROJ_FILE_FORMAT = os.path.join(INPUT_DIR,
                                           '{fold}',
                                           '{key}',
                                           'X_test_transform.npz')
INPUT_RECONSTRUCTIONS_FILE_FORMAT = os.path.join(INPUT_DIR,
                                                 '{fold}',
                                                 '{key}',
                                                  'X_test_predict.npz')

OUTPUT_DIR = INPUT_BASE_DIR
OUTPUT_RESULTS_FILE = os.path.join(OUTPUT_DIR, "summary.csv")
#
OUTPUT_CURVE_FILE_FORMAT = os.path.join(OUTPUT_DIR,
                                        '{metric}_{global_pen}.png')
# Filename is forged from the figure title
OUTPUT_PLOT_TITLE = "{name}"
OUTPUT_COMPONENT_PLOT_FIGNAME = 'components'
OUTPUT_PROJECTION_PLOT_FIGNAME = 'projections'
OUTPUT_EX_IMAGE_PLOT_FIGNAME = 'example'

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
METRICS = ['frobenius_test']
EXAMPLE_FOLD = 0  # This is the special fold with the whole dataset
N_COMP = 3

IM_SHAPE = (64, 64)

EX_IMAGES = [0, 42, 255]
N_EX_IMAGES = len(EX_IMAGES)

##########
# Script #
##########

# Open result file (index by model, total_penalization, tv_ratio, l1_ratio)
# We have to explicitly sort the index in order to subsample
df = pd.io.parsers.read_csv(INPUT_RESULTS_FILE,
                            index_col=[1, 2, 3, 4]).sort_index()

struct_pca_df = df.xs('struct_pca')

# Extract some cases & add a column based on name
summary = df.loc[PARAMS][METRICS]
name_serie = pd.Series([item[1] for item in COND], name='Name',
                       index=PARAMS)
summary['name'] = name_serie

# Write in a CSV
summary.to_csv(OUTPUT_RESULTS_FILE)

# Load dataset and labels
X = np.load(INPUT_DATASET)
y = np.load(INPUT_TARGET)

n, p = X.shape

# Number of persons
# This assume that y is sorted (bincount works on non-negative int)
[indiv, count] = [np.unique(y), np.bincount(y)]
n_indiv = len(indiv)
print "Found", n_indiv, "persons"

###############################################################################
# Plot metrics                                                                #
###############################################################################

for metric in METRICS:
    handles = plot_utilities.plot_lines(struct_pca_df,
                                        x_col=1,
                                        y_col=metric,
                                        splitby_col=0,
                                        colorby_col=2)
    for val, handle in handles.items():
        filename = OUTPUT_CURVE_FILE_FORMAT.format(metric=metric,
                                                   global_pen=val)
        handle.savefig(filename)

###############################################################################
# Components                                                                  #
###############################################################################

# Load components
components = np.zeros((len(COND), np.prod(IM_SHAPE), N_COMP))
for j, (params, name) in enumerate(COND):
    key = '_'.join([str(param) for param in params])
    filename = INPUT_COMPONENTS_FILE_FORMAT.format(fold=EXAMPLE_FOLD,
                                                   key=key)
    print "Loading components for", name, ":", filename
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
    figtitle = OUTPUT_PLOT_TITLE.format(name=name)
    plt.suptitle(figtitle)
    figname = OUTPUT_COMPONENT_PLOT_FIGNAME + '_' + figtitle.replace(' ', '_')
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
                         "components_colorbar.png"))


###############################################################################
# Projection of individuals on components                                     #
###############################################################################

# Load projections
projections = np.zeros((len(COND), n, N_COMP))
for j, (params, name) in enumerate(COND):
    key = '_'.join([str(param) for param in params])
    filename = INPUT_TEST_PROJ_FILE_FORMAT.format(fold=EXAMPLE_FOLD,
                                                  key=key)
    print "Loading projections for", name, ":", filename
    if os.path.exists(filename):
        projections[j, ...] = np.load(filename)['arr_0']
    else:
        print "No components for", COND[j][1]

# Create point color (one for each person)
all_colors = matplotlib.colors.cnames.keys()
all_colors.remove('white')
point_color = [all_colors[person] for person in y]

# Plot components: we plot on the PC0-PC1 plane and on the PC1-PC2 plane
handles = np.zeros((len(COND)), dtype='object')
for j, (params, name) in enumerate(COND):
    handles[j] = fig, axes = plt.subplots(nrows=1,
                                          ncols=2,
                                          figsize=(11.8, 3.7))
    f = plt.gcf()
    for first_pc, second_pc, axe in zip(range(2), range(1, 3), axes.flat):
        axe.scatter(projections[j, ..., first_pc],
                    projections[j, ..., second_pc],
                    color=point_color)
        axe.set_xlabel('PC {i}'.format(i=first_pc + 1))
        axe.set_ylabel('PC {i}'.format(i=second_pc + 1))
    figtitle = OUTPUT_PLOT_TITLE.format(name=name)
    plt.suptitle(figtitle)
    figname = OUTPUT_PROJECTION_PLOT_FIGNAME + '_' + figtitle.replace(' ', '_')
    f.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR,
                             ".".join([figname, "png"])))

###############################################################################
# Example of reconstruction                                                   #
###############################################################################

# Load reconstructions
reconstructions = np.zeros((len(COND), n, np.prod(IM_SHAPE)))
for j, (params, name) in enumerate(COND):
    key = '_'.join([str(param) for param in params])
    filename = INPUT_RECONSTRUCTIONS_FILE_FORMAT.format(fold=EXAMPLE_FOLD,
                                                        key=key)
    print "Loading reconstructions for", name, ":", filename
    if os.path.exists(filename):
        reconstructions[j, ...] = np.load(filename)['arr_0']
    else:
        print "No reconstruction for", COND[j][1]
data_min = reconstructions.min()
data_max = reconstructions.max()

# Plot components
handles = np.zeros((len(COND)), dtype='object')
for j, (params, name) in enumerate(COND):
    handles[j] = fig, axes = plt.subplots(nrows=1,
                                          ncols=N_EX_IMAGES,
                                          figsize=(11.8, 3.7))
    f = plt.gcf()
    for l, axe in zip(range(N_EX_IMAGES), axes.flat):
        data = reconstructions[j, l, ...].reshape(IM_SHAPE)
        im = axe.imshow(data, norm=norm, aspect="auto",
                        cmap=cmap)
    figtitle = OUTPUT_PLOT_TITLE.format(name=name)
    plt.suptitle(figtitle)
    figname = OUTPUT_EX_IMAGE_PLOT_FIGNAME + '_' + figtitle.replace(' ', '_')
    f.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR,
                             ".".join([figname, "png"])))

# Plot original data
fig, axes = plt.subplots(nrows=1,
                         ncols=N_EX_IMAGES,
                         figsize=(11.8, 3.7))
f = plt.gcf()
for l, axe in zip(range(N_EX_IMAGES), axes.flat):
    data = X[l, ...].reshape(IM_SHAPE)
    im = axe.imshow(data, norm=norm, aspect="auto",
                    cmap=cmap)
figtitle = "Original data"
plt.suptitle(figtitle)
figname = figtitle.replace(' ', '_')
f.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR,
                         ".".join([figname, "png"])))

