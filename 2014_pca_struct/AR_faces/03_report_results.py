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

This file was copied from scripts/2014_pca_struct/AR_faces/02_report_results.py.

"""

import os
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import pandas as pd

from brainomics import plot_utilities, array_utils

################
# Input/Output #
################

BASE_DIR = "/neurospin/brainomics/2014_pca_struct/AR_faces"

INPUT_DB_DIR = os.path.join(BASE_DIR,
                            "raw_data",
                            "cropped_faces")

INPUT_POPULATION = os.path.join(INPUT_DB_DIR,
                                "population.csv")

INPUT_BASE_DIR = os.path.join(BASE_DIR,
                              "pca_tv")
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
OUTPUT_COMPONENT_PLOT_FIGNAME = 'component_{k}_{name}'
OUTPUT_PROJECTION_PLOT_FIGNAME = 'projections_{name}'
OUTPUT_EXTREME_PLOT_FIGNAME = 'extreme_{name}'
OUTPUT_EX_IMAGE_PLOT_FIGNAME = 'example_{name}'

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
METRICS = ['frobenius_test',
           'correlation_0',
           'correlation_1',
           'correlation_2',
           'correlation_mean']
EXAMPLE_FOLD = 0  # This is the special fold with the whole dataset
N_COMP = 3

IM_SHAPE = (38, 27)

EX_IMAGES = [0, 42, 255]
N_EX_IMAGES = len(EX_IMAGES)

##########
# Script #
##########

# Open result file (index by model, total_penalization, tv_ratio, l1_ratio)
# We have to explicitly sort the index in order to subsample
df = pd.io.parsers.read_csv(INPUT_RESULTS_FILE,
                            index_col=[1, 2, 3, 4]).sort_index()

# Extract some cases & add a column based on name
summary = df.loc[PARAMS][METRICS]
name_serie = pd.Series([item[1] for item in COND], name='Name',
                       index=PARAMS)
summary['name'] = name_serie

# Write in a CSV
summary.to_csv(OUTPUT_RESULTS_FILE)

# Load dataset and labels
X = np.load(INPUT_DATASET)
population = pd.io.parsers.read_csv(INPUT_POPULATION)
y = population['id']

n, p = X.shape
data_min = np.min(X)
data_max = np.max(X)

# Number of persons
# This assume that y is sorted (bincount works on non-negative int)
#[indiv, count] = [np.unique(y), np.bincount(y)]
#n_indiv = len(indiv)
#print "Found", n_indiv, "persons"

###############################################################################
# Plot metrics                                                                #
###############################################################################

L1_RATIOS = [0.0, 0.5, 1.0]
l1_ratio_filter = lambda v: v in L1_RATIOS
df_noindex = df.reset_index()
struct_pca_df = df_noindex.loc[(df_noindex.model == 'struct_pca') &
                               (df_noindex.l1_ratio.apply(l1_ratio_filter))]

for metric in METRICS:
    handles = plot_utilities.plot_lines(struct_pca_df,
                                        x_col="tv_ratio",
                                        y_col=metric,
                                        splitby_col="global_pen",
                                        colorby_col="l1_ratio")
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
components_min = components.min()
components_max = components.max()

# Plot components (one figure per component and model)
# We create a colormap for each component (of each model)
# The colormap is symmetric, values below threshold are black
handles = np.zeros((len(COND), N_COMP), dtype='object')
for j, (params, name) in enumerate(COND):
    for k in range(N_COMP):
        handles[j, k] = plt.figure()
        fig = plt.gcf()
        axe = plt.gca()
        data = components[j, :, k].reshape(IM_SHAPE)
        data, t = array_utils.arr_threshold_from_norm2_ratio(data)
        vmax = np.max(np.abs(data))
        vmin = -vmax
        component_norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        x = [0, 0.5-t, 0.5+t, 1.0]
        cdict = {'red': ((x[0], 0.0, 0.0),
                         (x[1], 1.0, 0.0),
                         (x[2], 0.0, 1.0),
                         (x[3], 1.0, 0.0)),
                 'green': ((x[0], 0.0, 0.0),
                           (x[1], 1.0, 0.0),
                           (x[2], 0.0, 1.0),
                           (x[3], 0.0, 0.0)),
                 'blue': ((x[0], 0.0, 1.0),
                          (x[1], 1.0, 0.0),
                          (x[2], 0.0, 1.0),
                          (x[3], 0.0, 0.0))}
        component_cmap = \
            matplotlib.colors.LinearSegmentedColormap('my_colormap',
                                                      cdict,
                                                      256)
        im = axe.imshow(data,
                        aspect="equal",
                        norm=component_norm,
                        cmap=component_cmap,
                        interpolation="none")
        plt.colorbar(mappable=im, ax=axe, use_gridspec=True)
        name = name.replace(' ', '_')
        figname = OUTPUT_COMPONENT_PLOT_FIGNAME.format(name=name,
                                                       k=k)
        fig.savefig(os.path.join(OUTPUT_DIR,
                                 ".".join([figname, "png"])),
                    bbox_inches='tight')

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
    name = name.replace(' ', '_')
    figname = OUTPUT_PROJECTION_PLOT_FIGNAME.format(name=name)
    f.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR,
                             ".".join([figname, "png"])))

# Create a symetric colormap
# This assume that max is positive and min is negative
vmax = max([np.abs(data_min), np.abs(data_max)])
vmin = -vmax
extreme_norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
extreme_cmap = matplotlib.cm.gray

# Find extreme individuals
extreme_individuals_index = np.zeros((len(COND), N_COMP, 2), dtype='int')
min_indiv_index = projections.argmin(axis=1)
extreme_individuals_index[:, :, 0] = min_indiv_index
max_indiv_index = projections.argmax(axis=1)
extreme_individuals_index[:, :, 1] = max_indiv_index

# Plot extreme individuals: 1 figure per condition with N_COMPx2 subplots
handles = np.zeros((len(COND)), dtype='object')
for j, (params, name) in enumerate(COND):
    handles[j] = fig, axes = plt.subplots(nrows=N_COMP,
                                          ncols=2)
    f = plt.gcf()
    for l, (i, axe) in enumerate(zip(extreme_individuals_index[j].flat,
                                     axes.flat)):
        data = X[i, ...].reshape(IM_SHAPE)
        im = axe.imshow(data,
                    aspect="auto",
                    norm=extreme_norm,
                    cmap=extreme_cmap)
    name = name.replace(' ', '_')
    figname = OUTPUT_EXTREME_PLOT_FIGNAME.format(name=name)
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
reconstruction_min = reconstructions.min()
reconstruction_max = reconstructions.max()

# Create a symetric colormap
# This assume that max is positive and min is negative
vmax = max([np.abs(reconstruction_min), np.abs(reconstruction_max),
            np.abs(data_min), np.abs(data_max)])
vmin = -vmax
reconstruction_norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
reconstruction_cmap = matplotlib.cm.gray

# Plot components
handles = np.zeros((len(COND)), dtype='object')
for j, (params, name) in enumerate(COND):
    handles[j] = fig, axes = plt.subplots(nrows=1,
                                          ncols=N_EX_IMAGES,
                                          figsize=(11.8, 3.7))
    f = plt.gcf()
    for l, axe in zip(EX_IMAGES, axes.flat):
        data = reconstructions[j, l, ...].reshape(IM_SHAPE)
        im = axe.imshow(data,
                        aspect="auto",
                        norm=reconstruction_norm,
                        cmap=reconstruction_cmap)
    name = name.replace(' ', '_')
    figname = OUTPUT_EX_IMAGE_PLOT_FIGNAME.format(name=name)
    f.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR,
                             ".".join([figname, "png"])))

# Plot original data
fig, axes = plt.subplots(nrows=1,
                         ncols=N_EX_IMAGES,
                         figsize=(11.8, 3.7))
f = plt.gcf()
for l, axe in zip(EX_IMAGES, axes.flat):
    data = X[l, ...].reshape(IM_SHAPE)
    im = axe.imshow(data,
                    aspect="auto",
                    norm=reconstruction_norm,
                    cmap=reconstruction_cmap)
figtitle = "Original data"
figname = figtitle.replace(' ', '_')
f.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR,
                         ".".join([figname, "png"])))
