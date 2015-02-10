# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 10:55:01 2015

@author: cp243490
"""

import os
import numpy as np
import nibabel as nib
import glob

BASE_PATH = "/neurospin/brainomics/2014_deptms"

modality = "MRI"
DATASET_PATH = os.path.join(BASE_PATH,    "datasets")
MODALITY_PATH = os.path.join(DATASET_PATH, modality)
maskdep_file = os.path.join(BASE_PATH, "base_data", "maskdep.nii")

mask_path = glob.glob(os.path.join(MODALITY_PATH, 'mask_MRI_Roiho-*'))
size_roi = []
for i, mask in enumerate(mask_path):
    print i+1, os.path.basename(mask)
    mask_roi_im = nib.load(mask)
    mask_roi = mask_roi_im.get_data() != 0
    mask_roi = np.asarray(mask_roi)
    mask_roi = mask_roi.astype('int')
    size_roi.append(np.sum(mask_roi == 1))
    if i == 0:
        maskdep = np.zeros(mask_roi.shape)
    maskdep[mask_roi == 1] = i + 1
out_im = nib.Nifti1Image(maskdep.astype("int16"),
                                         affine=mask_roi_im.get_affine())
out_im.to_filename(maskdep_file)
brain_mesh_file = "/neurospin/brainomics/neuroimaging_ressources/mesh/MNI152_T1_1mm_Bothhemi.gii"
a_anat_filename = "/usr/share/data/fsl-mni152-templates/MNI152_T1_1mm_brain.nii.gz"                                         
