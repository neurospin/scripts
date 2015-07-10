# -*- coding: utf-8 -*-
"""


@author: md238665


We use a constant global penalization.


"""

import os
#import sys

import numpy as np
import matplotlib.pyplot as plt
import json
import nibabel as nib

import pandas as pd

from brainomics import plot_utilities
import parsimony.utils.check_arrays as check_arrays

################
# Input/Output #
################
#/results/0/1.0_0.007_0.693_0.3_-1.0
INPUT_BASE_DIR = "/neurospin/brainomics/2014_bd_dwi"
INPUT_CV_DIR = os.path.join(INPUT_BASE_DIR, "enettv_bd_dwi_trunc", "results")
INPUT_RESULTS_FILE = os.path.join(INPUT_BASE_DIR, "enettv_bd_dwi_trunc", "results.csv")
INPUT_CONFIG_FILE = os.path.join(INPUT_BASE_DIR, "enettv_bd_dwi_trunc", "config.json")

INPUT_MASK = os.path.join(INPUT_BASE_DIR,"enettv_bd_dwi_trunc",
                          "mask_trunc.nii.gz")


#INPUT_MESCOG_DIR = "/neurospin/mescog/proj_wmh_patterns"

INPUT_POPULATION_FILE = os.path.join(INPUT_BASE_DIR,
                                     "population.csv")
INPUT_DATASET = os.path.join(INPUT_BASE_DIR, "enettv_bd_dwi_trunc", 
                             "X_trunc.npy")

OUTPUT_DIR = os.path.join(INPUT_BASE_DIR, "summary")
OUTPUT_RESULTS_FILE = os.path.join(OUTPUT_DIR, "summary.csv")
INPUT_WEIGHTS_FILE_FORMAT = os.path.join(INPUT_CV_DIR,
                                            '{fold}',
                                            '{key}',
                                            'beta.npz')

OUTPUT_WEIGHTS_FILE_FORMAT = os.path.join(OUTPUT_DIR,
                                             '{name}.nii.gz')
##############
# Parameters #
##############


METRICS = ['recall_0', 'recall_1', 'recall_mean','auc']

# Plot of metrics
CURVE_FILE_FORMAT = os.path.join(OUTPUT_DIR,
                                 '{metric}_{global_pen}.png')

# Plot of weights
COND = [
        ((1.0, 0.006999999999999999, 0.693, 0.30000000000000004, -1.0), 'tvl1l2')
       ]
PARAMS = [item[0] for item in COND]
EXAMPLE_FOLD = 0

##############
# LOAD DATA  #
##############
mask_ima = nib.load(INPUT_MASK)
mask_arr = mask_ima.get_data() != 0
mask_indices = np.where(mask_arr)
#n_voxels_in_mask = np.count_nonzero(bin_mask)
IM_SHAPE = mask_arr.shape
X = np.load(INPUT_DATASET)
pop = pd.read_csv(INPUT_POPULATION_FILE)
config  = json.load(open(INPUT_CONFIG_FILE))


#INPUT_PROJECTIONS_FILE_FORMAT = os.path.join(INPUT_CV_DIR,
#                                            '{fold}',
#                                            '{key}',
 #                                            'X_train_transform.npz')
#IM_SHAPE = (182, 218, 182)

################
# Plot results #
################


# Open result file (index by model, total_penalization, tv_ratio, l1_ratio)
# We have to explicitly sort the index in order to subsample
df = pd.io.parsers.read_csv(INPUT_RESULTS_FILE, index_col=[0]).sort_index()#,
df.index = [eval(s) for s in  df.index]

#                            index_col=[1, 2, 3, 4, 5]).sort_index()
assert df.index.isin(PARAMS).sum() == 1


#df[(df.a == 1.0) & (df.l1 == 0.006999999999999999) & (df.l2 == 0.693) & (df.k == -1)]

# Subsample it & add a column based on name
summary = df.loc[PARAMS][METRICS]
#name_serie = pd.Series([item[1] for item in COND], name='Name',
#                       index=PARAMS)
#summary['name'] = name_serie

# Write in a CSV
summary.to_csv(OUTPUT_RESULTS_FILE)

# Subsample df
#struct_pca_df = df.xs(EXAMPLE_MODEL)

"""
# Plot some metrics for struct_pca
for metric in METRICS:
    metric = 'recall_mean'
    handles = plot_utilities.plot_lines(df,
                                        x_col=3,
                                        y_col=metric,
                                        splitby_col=0,
                                        colorby_col=2)
    for val, handle in handles.items():
        filename = CURVE_FILE_FORMAT.format(metric=metric,
                                            global_pen=val)
        handle.savefig(filename)
"""


#####################################################################
# Weight as images                                                  #
#####################################################################

import scipy
scipy.stats.mstats.mquantiles(weights, np.arange(1, 10) / 10.)
weights.min()
weights.max()
np.sum(weights < weights.min() / 10.)
np.sum(weights > weights.max() / 10.)

# 

#clinal_var = ['MMSE', 'MRS', 'TMTB_TIME', 'MDRS_TOTAL']
#pop.loc[:, clinal_var].corr()
#                MMSE       MRS  TMTB_TIME  MDRS_TOTAL
#MMSE        1.000000 -0.655684  -0.858704    0.866949
#MRS        -0.655684  1.000000   0.682066   -0.698889
#TMTB_TIME  -0.858704  0.682066   1.000000   -0.898043
#MDRS_TOTAL  0.866949 -0.698889  -0.898043    1.000000

# Load weights and store them as nifti images
j, (params, name) = 0, COND[0]
for j, (params, name) in enumerate(COND):
    #j, (params, name) = 0, COND[0]
    key = '_'.join([str(param) for param in params])
    print "process", key
    weights_filename = INPUT_WEIGHTS_FILE_FORMAT.format(fold=EXAMPLE_FOLD,
                                                   key=key)           
    weights = np.load(weights_filename)['arr_0']
    weights = weights[config['penalty_start']:]
    assert mask_arr.sum()  == weights.shape[0]
    weights_3d_arr = np.zeros(shape = mask_arr.shape)
    weights_3d_arr[mask_arr] = weights.ravel()
    weights_3d_ima = nib.Nifti1Image(weights_3d_arr,
                         affine=mask_ima.get_affine())
    weights_3d_ima_filename = OUTPUT_WEIGHTS_FILE_FORMAT.format(name=key)
    print "Save", weights_3d_ima_filename
    nib.save(weights_3d_ima,  weights_3d_ima_filename)



cmd = 'rsync -avu %s %s/' % (OUTPUT_DIR, "/home/ed203246/mega/data/2014_bd_dwi")
os.system(cmd)

"""
python ~/git/scripts/2014_bd_dwi/03_plots_results.py


cd /home/ed203246/mega/data/2014_bd_dwi/summary


~/git/scripts/brainomics/image_clusters_analysis.py -t 0.99 1.0_0.007_0.693_0.3_-1.0.nii.gz



# Threshold image as 0.001407

~/git/scripts/brainomics/image_clusters_analysis.py -t 0.9 tvl1l20001.nii.gz
# Threshold image as 0.001554

#~/git/scripts/brainomics/image_clusters_analysis.py -t 0.9 tvl1l20002.nii.gz
## Threshold image as 0.001876

~/git/scripts/brainomics/image_clusters_analysis.py -t 0.99 tvl1l20002.nii.gz
Threshold image as 0.000675

#~/git/scripts/brainomics/image_clusters_analysis.py -t 0.99 tvl1l20003.nii.gz
#Threshold image as 0.002078

~/git/scripts/brainomics/image_clusters_analysis.py -t 0.99 tvl1l20003.nii.gz
Threshold image as 0.000741

# rendering
~/git/scripts/brainomics/image_clusters_rendering.py pc1 pc2 pc3

cd snapshosts
ls *.png|while read input; do
convert  $input -trim /tmp/toto.png;
convert  /tmp/toto.png -transparent white $input;
done

"""