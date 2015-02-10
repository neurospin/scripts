# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 11:11:43 2014

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

INPUT_BASE_DIR = "/neurospin/brainomics/2014_deptms/"
INPUT_DIR = os.path.join(INPUT_BASE_DIR, "results_enettv")
MODALITY = "MRI"
ROI = "Roiho-putamen"
INPUT_MOD_DIR = os.path.join(INPUT_DIR, MODALITY + "_" + ROI)
INPUT_RESULTS = os.path.join(INPUT_MOD_DIR, "results_dCV_validation.csv")

# Load mask
MASK_FILENAME = os.path.join(INPUT_MOD_DIR, "mask.nii")

mask_im = nib.load(MASK_FILENAME)
mask_arr = mask_im.get_data() != 0

penalty_start = 3

#######################################################
### Create a file beta.nii to read it with anatomist ##
#######################################################
for key_path in glob.glob(os.path.join(INPUT_MOD_DIR,
                                       'results_dCV_validation/0/*')):
    print key_path
    #k = float(os.path.basename(key_path).split('_')[-1])
#    if (MODALITY == "MRI") or (MODALITY == "PET"):
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
#    elif MODALITY == "MRI+PET":
#        beta_MRI_file = np.load(os.path.join(key_path, 'beta_MRI.npz'))
#        beta_MRI = beta_MRI_file['arr_0']
#        beta_MRI_file.close()
#        submask_MRI_file = np.load(os.path.join(key_path, 'mask_MRI.npz'))
#        submask_MRI = submask_MRI_file['arr_0']
#        submask_MRI_file.close()
#        beta_MRI_arr = np.zeros(submask_MRI.shape)
#        beta_MRI_arr[submask_MRI] = beta_MRI[penalty_start:, 0]
#        beta_MRI_im = nib.Nifti1Image(beta_MRI_arr, mask_im.get_affine())
#        nib.save(beta_MRI_im, os.path.join(key_path, "beta_MRI.nii.gz"))
#
#        beta_PET_file = np.load(os.path.join(key_path, 'beta_PET.npz'))
#        beta_PET = beta_PET_file['arr_0']
#        beta_PET_file.close()
#        submask_PET_file = np.load(os.path.join(key_path, 'mask_PET.npz'))
#        submask_PET = submask_PET_file['arr_0']
#        submask_PET_file.close()
#        beta_PET_arr = np.zeros(submask_PET.shape)
#        beta_PET_arr[submask_PET] = beta_PET[penalty_start:, 0]
#        beta_PET_im = nib.Nifti1Image(beta_PET_arr, mask_im.get_affine())
#        nib.save(beta_PET_im, os.path.join(key_path, "beta_PET.nii.gz"))

################################
## Compare sets of parameters ##
################################
#OUTPUT_DIR = os.path.join(INPUT_MOD_DIR, "figures")
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