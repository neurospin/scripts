# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 15:40:31 2015

@author: cp243490
"""

import numpy as np
import pandas as pd
import os
import glob
import nibabel as nib
from distutils import dir_util


BASE_PATH = "/neurospin/brainomics/2014_deptms"

MOD = "MRI"

INPUT_ROIS_CSV = os.path.join(BASE_PATH, "base_data", "ROI_labels.csv")

ENETTV_PATH = os.path.join(BASE_PATH, "results_enettv")

penalty_start = 3


def convert_beta(DIR, mask):
    for key_path in glob.glob(os.path.join(DIR, '/0/*')):
        beta_file = np.load(os.path.join(key_path, 'beta.npz'))
        beta = beta_file['arr_0']
        beta_file.close()
        submask_file = np.load(os.path.join(key_path, 'mask.npz'))
        submask = submask_file['arr_0']
        submask_file.close()
        beta_arr = np.zeros(submask.shape)
        beta_arr[submask] = beta[penalty_start:, 0]
        beta_im = nib.Nifti1Image(beta_arr, mask.get_affine())
        nib.save(beta_im, os.path.join(key_path, "beta.nii.gz"))
#############################################################################
## Read ROIs csv
rois = []
df_rois = pd.read_csv(INPUT_ROIS_CSV)
for i, ROI_name_aal in enumerate(df_rois["ROI_name_aal"]):
    cur = df_rois[df_rois.ROI_name_aal == ROI_name_aal]
    roi_name = cur["ROI_name_deptms"].values[0]
    if ((not cur.isnull()["atlas_ho"].values[0])
        and (not cur.isnull()["ROI_name_deptms"].values[0])):
        if ((not roi_name in rois)
          and (roi_name != "Maskdep-sub") and (roi_name != "Maskdep-cort")):
            print "ROI: ", roi_name
            rois.append(roi_name)
rois.append("brain")
print "\n"

## create clusters in predictors map (model selected when the criterion
# accuracy is maximized)
criterion = "a_c_c_u_r_a_c_y"
for i, roi in enumerate(rois):
    ROI_PATH = os.path.join(ENETTV_PATH, MOD + '_' + roi)
    print "ROI", roi
    dir_dCV_validation = os.path.join(ROI_PATH, 'results_dCV_validation')
    if os.path.exists(dir_dCV_validation):
        # Create a files beta.nii from files beta.npz to be read with anatomist
        mask = nib.load(os.path.join(ROI_PATH, 'mask.nii'))
        convert_beta(dir_dCV_validation, mask)
        # generate mesh clusters to visualize predictors map in fold 0
        ANALYSIS_PATH = os.path.join(ROI_PATH, 'analysis_dCV')
        if not os.path.exists(ANALYSIS_PATH):
            os.makedirs(ANALYSIS_PATH)
        dir_util.copy_tree(os.path.join(dir_dCV_validation, '0', criterion),
                           ANALYSIS_PATH)

        """##################################################################
        print "./i2bm/local/Ubuntu-12.04-x86_64/brainvisa/bin/bv_env.sh"
        print "# variables names: atlas, roi, modality
        print "$ATLAS_CORT = '/neurospin/brainomics/2014_deptms/base_data/images/atlases/HarvardOxford-cort-maxprob-thr0-1mm-nn.nii.gz'"
        print "$ATLAS_SUB = '/neurospin/brainomics/2014_deptms/base_data/images/atlases/HarvardOxford-sub-maxprob-thr0-1mm-nn.nii.gz'"
        print "#go to ", ANALYSIS_PATH, "directory"
        print "cd % d", ANALYSIS_PATH
        print "# generate mesh clusters"
        print "# beta-map threshold_norm_ratio = 0.8"
        print "image_clusters_analysis.py beta.nii.gz --atlas_cort $ATLAS_CORT --atlas_sub $ATLAS_SUB -t 0.8"
        print "# Visualise mesh clusters with anatomist
        print "image_clusters_rendering.py beta &"
        print "# save snapshot with window 'snapshot' button"
        """
        
