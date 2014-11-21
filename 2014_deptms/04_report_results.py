# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 11:11:43 2014

@author: cp243490
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import nibabel as nib
import glob


from brainomics import plot_utilities


################
# Input/Output #
################

INPUT_BASE_DIR = "/neurospin/brainomics/2014_deptms"
INPUT_DIR = os.path.join(INPUT_BASE_DIR, "results_enettv")
MODALITY = "MRI"
ROI = "wb"
INPUT_MOD_DIR = os.path.join(INPUT_DIR, MODALITY + "_" + ROI)
INPUT_RESULTS = os.path.join(INPUT_MOD_DIR, "results.csv")

OUTPUT_DIR = os.path.join(INPUT_MOD_DIR, "figures")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


# Parameters

krange_ratio = [0.1 / 100., 1 / 100., 10 / 100., 50 / 100., -1]
alphas = [.01, .05, .1, .5, 1.]
ratios = np.array([[1., 0., 1], [0., 1., 1], [.5, .5, 1], [.9, .1, 1],
                       [.1, .9, 1], [.01, .99, 1], [.001, .999, 1]])

# Load mask
if not MODALITY == "MRI+PET":
    MASK_FILENAME = os.path.join(INPUT_MOD_DIR,
                                 "mask_" + MODALITY + "_" + ROI + ".nii")
else:
    MASK_FILENAME = os.path.join(INPUT_MOD_DIR, "mask_MRI_" + ROI + ".nii")

mask_im = nib.load(MASK_FILENAME)
mask_arr = mask_im.get_data()
mask_bin = mask_arr != 0

EX_BETA_DIR = os.path.join(INPUT_MOD_DIR, "results/0/0.01_0.0_0.2_0.8_0.5")

penalty_start = 3

#######################################################
### Create a file beta.nii to read it with anatomist ##
#######################################################
#for key_path in glob.glob(os.path.join(INPUT_MOD_DIR, 'results/*/*')):
#    print key_path
#    beta_file = np.load(os.path.join(key_path, 'beta.npz'))
#    beta = beta_file['arr_0']
#    beta_file.close()
#    submask_file = np.load(os.path.join(key_path, 'mask.npz'))
#    submask = submask_file['arr_0']
#    submask_file.close()
#    beta_arr = np.zeros(submask.shape)
#    beta_arr[submask] = beta[penalty_start:, 0]
#    beta_im = nib.Nifti1Image(beta_arr, mask_im.get_affine())
#    nib.save(beta_im, os.path.join(key_path, "beta.nii"))


################################
## Compare sets of parameters ##
################################

results = pd.io.parsers.read_csv(INPUT_RESULTS)
METRICS = ['recall_mean', 'precision_mean', 'f1_mean']
k_groups = results.groupby('k_ratio')

# Plot some metrics for struct_pca for each SNR value
for k_val, k_group in k_groups:
    for metric, metric_name in zip(METRICS, METRICS):
        handles = plot_utilities.plot_lines(k_group,
                                            x_col='tv',
                                            y_col=metric,
                                            splitby_col='a',
                                            colorby_col='l1l2_ratio',
                                            use_suptitle=False)
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
#                                                snr=snr_val,
#                                                global_pen=val)
#            handle.savefig(filename)
