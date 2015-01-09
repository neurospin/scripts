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
#import sys

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

OUTPUT_DIR = os.path.join(INPUT_BASE_DIR, "summary")
OUTPUT_RESULTS_FILE = os.path.join(OUTPUT_DIR, "summary.csv")

INPUT_MESCOG_DIR = "/neurospin/mescog/proj_wmh_patterns"

INPUT_POPULATION_FILE = os.path.join(INPUT_MESCOG_DIR,
                                     "population.csv")
OUTPUT_COMPONENTS = os.path.join(OUTPUT_DIR,
                                 "components.csv")
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
COND = [(('pca', 0.0, 0.0, 0.0), 'pca'),
        (('struct_pca', EXAMPLE_GLOBAL_PEN, 1.0, 0.0), 'tv'),
        (('struct_pca', EXAMPLE_GLOBAL_PEN, 0.0, 1.0), 'l1'),
        (('struct_pca', EXAMPLE_GLOBAL_PEN, 0.0, 0.0), 'l2'),
        (('struct_pca', EXAMPLE_GLOBAL_PEN, 0.5, 1.0), 'tvl1'),
        (('struct_pca', EXAMPLE_GLOBAL_PEN, 0.5, 0.0), 'tvl2'),
        (('struct_pca', EXAMPLE_GLOBAL_PEN, 0.0, 0.5), 'l1l2'),
        (('struct_pca', EXAMPLE_GLOBAL_PEN, 0.33, 0.5), 'tvl1l2'),
        ((u'struct_pca', EXAMPLE_GLOBAL_PEN, 0.33, 0.1), 'tvl1l2_smalll1')
       ]
PARAMS = [item[0] for item in COND]
EXAMPLE_FOLD = 0
INPUT_COMPONENTS_FILE_FORMAT = os.path.join(INPUT_DIR,
                                            '{fold}',
                                            '{key}',
                                            'components.npz')
INPUT_PROJECTIONS_FILE_FORMAT = os.path.join(INPUT_DIR,
                                            '{fold}',
                                            '{key}',
                                            'X_train_transform.npz')
#IM_SHAPE = (182, 218, 182)
OUTPUT_COMPONENTS_FILE_FORMAT = os.path.join(OUTPUT_DIR,
                                             '{name}.nii')


################
# Plot results #
################


# Open result file (index by model, total_penalization, tv_ratio, l1_ratio)
# We have to explicitly sort the index in order to subsample
df = pd.io.parsers.read_csv(INPUT_RESULTS_FILE,
                            index_col=[1, 2, 3, 4]).sort_index()

# Subsample it & add a column based on name
summary = df.loc[PARAMS][METRICS]
name_serie = pd.Series([item[1] for item in COND], name='Name',
                       index=PARAMS)
summary['name'] = name_serie

# Write in a CSV
summary.to_csv(OUTPUT_RESULTS_FILE)

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



#####################################################################
# Loading as images                                                 #
# Projection to csv                                                 #
# eventually flip Projection and Loading such that they positivetly #
# correlate with clinical severity                                  #
#####################################################################



# Open mask
mask_ima = nib.load(INPUT_MASK)
mask_arr = mask_ima.get_data() != 0
mask_indices = np.where(mask_arr)
#n_voxels_in_mask = np.count_nonzero(bin_mask)
IM_SHAPE = mask_arr.shape

pop = pd.read_csv(INPUT_POPULATION_FILE)

#clinal_var = ['MMSE', 'MRS', 'TMTB_TIME', 'MDRS_TOTAL']
#pop.loc[:, clinal_var].corr()
#                MMSE       MRS  TMTB_TIME  MDRS_TOTAL
#MMSE        1.000000 -0.655684  -0.858704    0.866949
#MRS        -0.655684  1.000000   0.682066   -0.698889
#TMTB_TIME  -0.858704  0.682066   1.000000   -0.898043
#MDRS_TOTAL  0.866949 -0.698889  -0.898043    1.000000

# Load components and store them as nifti images
for j, (params, name) in enumerate(COND):
    #j, (params, name) = 0, COND[0]
    key = '_'.join([str(param) for param in params])
    components_filename = INPUT_COMPONENTS_FILE_FORMAT.format(fold=EXAMPLE_FOLD,
                                                   key=key)
    projections_filename = INPUT_PROJECTIONS_FILE_FORMAT.format(fold=EXAMPLE_FOLD,
                                                   key=key)             
    components = np.load(components_filename)['arr_0']
    projections = np.load(projections_filename)['arr_0']
    assert projections.shape[1] == components.shape[1]
    # eventually flip Projection and Loading such that they positivetly
    # correlate with clinical severity
    for i in xrange(projections.shape[1]):
        R = np.corrcoef(pop.TMTB_TIME, projections[:, i])[0, 1]
        sign = np.sign(R)
        #sign = np.sign(np.dot(pop.TMTB_TIME, projections[:, i]))
        print name, i, R, sign#, np.corrcoef(pop.TMTB_TIME, projections[:, i])
        projections[:, i] = sign * projections[:, i]
        components[:, i] = sign * components[:, i]
    # Loading as images
    loadings_arr = np.zeros((IM_SHAPE[0], IM_SHAPE[1], IM_SHAPE[2], N_COMP))
    for l in range(N_COMP):
        loadings_arr[mask_indices[0], mask_indices[1], mask_indices[2], l] = components[:, l]
    im = nib.Nifti1Image(loadings_arr,
                         affine=mask_ima.get_affine())
    figname = OUTPUT_COMPONENTS_FILE_FORMAT.format(name=name.replace(' ', '_'))
    nib.save(im, os.path.join(OUTPUT_DIR, figname))
    # projection to csv
    for i in xrange(projections.shape[1]):
        pc_name = 'pc{i}__{name}'.format(name=name, i=i+1)
        pop[pc_name] = projections[:, i]  


pop.to_csv(OUTPUT_COMPONENTS)

cmd = 'rsync -avu %s %s/' % (OUTPUT_DIR, "/home/ed203246/data/mescog/wmh_patterns")
os.system(cmd)
