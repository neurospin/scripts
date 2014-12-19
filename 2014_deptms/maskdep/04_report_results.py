# -*- coding: utf-8 -*-
"""
Created on Thu Dec 11 14:49:39 2014

@author: cp243490
"""

import os
import numpy as np
import pandas as pd
#import matplotlib.pylab as plt
import nibabel as nib
import glob


from brainomics import plot_utilities


################
# Input/Output #
################

INPUT_BASE_DIR = "/neurospin/brainomics/2014_deptms/maskdep"
INPUT_DIR = os.path.join(INPUT_BASE_DIR, "results_enettv")

INPUT_DILATE_DIR = os.path.join(INPUT_DIR, 'dilatation_within-brain_masks')
INPUT_RESULTS = os.path.join(INPUT_DILATE_DIR, "results_more_params.csv")

MASK_FILENAME = os.path.join(INPUT_DILATE_DIR,
                             "mask.nii")
mask_im = nib.load(MASK_FILENAME)
mask_arr = mask_im.get_data() != 0

penalty_start = 3

#######################################################
### Create a file beta.nii to read it with anatomist ##
#######################################################
for key_path in glob.glob(os.path.join(INPUT_DILATE_DIR,
                                       'results_more_params/0/*')):
    print key_path
    k = float(os.path.basename(key_path).split('_')[-1])
    beta_file = np.load(os.path.join(key_path, 'beta.npz'))
    beta = beta_file['arr_0']
    beta_file.close()
    submask_file = np.load(os.path.join(key_path, 'mask.npz'))
    submask = submask_file['arr_0']
    submask_file.close()
    beta_arr = np.zeros(submask.shape)
    beta_arr[submask] = beta[penalty_start:, 0]
    beta_im = nib.Nifti1Image(beta_arr, mask_im.get_affine())
    nib.save(beta_im, os.path.join(key_path, "beta.nii.gz"))

################################
## Compare sets of parameters ##
################################
#OUTPUT_DIR = os.path.join(INPUT_DILATE_DIR, "figures")
#if not os.path.exists(OUTPUT_DIR):
#    os.makedirs(OUTPUT_DIR)

#results = pd.io.parsers.read_csv(INPUT_RESULTS)
#results.l1l2_ratio = np.round(results.l1l2_ratio, 5)
#METRICS = ['recall_mean', 'precision_mean', 'f1_mean']
#k_groups = results.groupby('k_ratio')
#CURVE_FILE_FORMAT = os.path.join(OUTPUT_DIR,
#                                 '{metric}_{k}_{a}.png')
## Plot some metrics for struct_pca for each SNR value
#for k_val, k_group in k_groups:
#    for metric, metric_name in zip(METRICS, METRICS):
#        handles = plot_utilities.plot_lines(k_group,
#                                            x_col='tv',
#                                            y_col=metric,
#                                            splitby_col='a',
#                                            colorby_col='l1l2_ratio',
#                                            use_suptitle=False)
#        for val, handle in handles.items():
#            # Tune the figure
#            ax = handle.get_axes()[0]
#            ax.set_xlabel("TV ratio")
#            ax.set_ylabel(metric_name)
#            l = ax.get_legend()
#            l.set_title("$\ell_1$ ratio")
#            s = r'$ \alpha $ =' + str(val)
#            handle.suptitle(r'$ \alpha $ =' + str(val))
#            filename = CURVE_FILE_FORMAT.format(metric=metric,
#                                                k=k_val,
#                                                a=val)
#            handle.savefig(filename)