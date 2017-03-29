#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 17:26:44 2016

@author: ad247405
"""

import os
import numpy as np
import glob
import pandas as pd
import nibabel as nib
import brainomics.image_atlas
import shutil
import mulm
import sklearn
from  scipy import ndimage


BASE_PATH = "/neurospin/brainomics/2016_deptms"
ATLAS_PATH = "/neurospin/brainomics/2016_deptms/atlas"
MASK_PATH = "/neurospin/brainomics/2016_deptms/analysis/VBM/data/mask.nii"
INPUT_CSV = "/neurospin/brainomics/2016_deptms/analysis/VBM/population.csv"
INPUT_ROIS_CSV = "/neurospin/brainomics/2016_deptms/analysis/VBM/ROI_labels.csv"
INPUT_DATA_X = os.path.join(BASE_PATH,"analysis","VBM","data","X.npy")
OUTPUT_ROIS_DATA = os.path.join(BASE_PATH,"analysis","VBM","data","ROIs_data")

sub_image = nib.load(os.path.join(ATLAS_PATH,"HarvardOxford-sub-maxprob-thr0-1.5mm.nii.gz"))
sub_arr = sub_image.get_data()
cort_image = nib.load(os.path.join(ATLAS_PATH,"HarvardOxford-cort-maxprob-thr0-1.5mm.nii.gz"))
cort_arr = cort_image.get_data()

brain_arr = (sub_arr.astype("bool") | cort_arr.astype("bool"))

mask =  nib.load(MASK_PATH).get_data()            

#############################################################################
## Read ROIs csv
atlas = []
dict_rois = {}
df_rois = pd.read_csv(INPUT_ROIS_CSV)
for i, ROI_name_aal in enumerate(df_rois["ROI_name_aal"]):
    cur = df_rois[df_rois.ROI_name_aal == ROI_name_aal]
    label_ho = cur["label_ho"].values[0]
    atlas_ho = cur["atlas_ho"].values[0]
    roi_name = cur["ROI_name_deptms"].values[0]
    if ((not cur.isnull()["atlas_ho"].values[0])
        and (not cur.isnull()["ROI_name_deptms"].values[0])):
        if not roi_name in dict_rois:
            print ("ROI: ", roi_name)
            labels = np.asarray(label_ho.split(), dtype="int")
            dict_rois[roi_name] = [labels]
            dict_rois[roi_name].append(atlas_ho)
            print (dict_rois[roi_name])
            print ("\n")



#############################################################
BASE_PATH = '/neurospin/brainomics/2016_deptms'
INPUT_CSV= "/neurospin/brainomics/2016_deptms/analysis/VBM/population.csv"
OUTPUT = os.path.join(BASE_PATH,"analysis","VBM","data")

# Read pop csv
pop = pd.read_csv(INPUT_CSV)
#############################################################################
# Read images
n = len(pop)
assert n == 34
Z = np.zeros((n, 3)) # Intercept + Age + Gender
Z[:, 0] = 1 # Intercept
y = np.zeros((n, 1)) # DX
images = list()
for i, index in enumerate(pop.index):
    cur = pop[pop.index== index]
    print(cur)
    imagefile_name = cur.path_VBM
    babel_image = nib.load(imagefile_name.as_matrix()[0])
    images.append(babel_image.get_data().ravel())
    Z[i, 1:] = np.asarray(cur[["Age", "Sex"]]).ravel()
    y[i, 0] = cur["Response.num"]

shape = babel_image.get_data().shape

Xtot = np.vstack(images)
mask = (np.min(Xtot, axis=0) > 0.01) & (np.std(Xtot, axis=0) > 1e-6)
mask = mask.reshape(shape)
assert mask.sum() == 311972
########################################################################


for ROI, values in dict_rois.items():
    print ("ROI: ", ROI)
    labels = values[0]
    print ("labels", labels)
    atlas = values[1]
    print ("atlas: ", atlas)
    if (atlas == "sub") or (atlas == "cort"):
        if atlas == "sub":
            mask_atlas_ROI = np.copy(sub_arr)
        elif atlas == "cort":
            mask_atlas_ROI = np.copy(cort_arr)
        if len(labels) == 1:
            mask_atlas_ROI[np.logical_or(np.logical_not(mask),
                                    mask_atlas_ROI != labels[0])] = 0
        mask_atlas_ROI[np.logical_not(mask)] = 0
        mask_ROI = np.zeros(mask_atlas_ROI.shape)
        for lab in labels:
            mask_ROI[mask_atlas_ROI == lab] = 1
        mask_atlas_ROI[np.logical_not(mask_ROI)] = 0
        np.unique(mask_atlas_ROI)
               
        # dilate
        # 3x3 structuring element with connectivity 1 and 2 iterations
        mask_bool_ROI = ndimage.morphology.binary_dilation(
                            mask_atlas_ROI,
                            iterations=2).astype(mask_atlas_ROI.dtype)
        mask_bool_ROI = mask_bool_ROI.astype("bool")
        out_im = nib.Nifti1Image(mask_bool_ROI.astype("int16"),
                                 affine=babel_image.get_affine())
        out_im.to_filename(os.path.join(OUTPUT_ROIS_DATA,
                            "mask_" + ROI + ".nii"))
        im = nib.load(os.path.join(OUTPUT_ROIS_DATA,
                             "mask_" + ROI + ".nii"))
        assert np.all(mask_bool_ROI == im.get_data())
        
        
        # Xcsi for the specific ROIs
        X_ROI = Xtot[:, mask_bool_ROI.ravel()]
        X_ROI = np.hstack([Z, X_ROI])
        X_ROI = np.nan_to_num(X_ROI)
        X_ROI -= X_ROI.mean(axis=0)
        X_ROI /= X_ROI.std(axis=0)
        X_ROI[:, 0] = 1.
        n, p = X_ROI.shape
        np.save(os.path.join(OUTPUT_ROIS_DATA,
                        "X_" + ROI + ".npy"), X_ROI)
        fh = open(os.path.join(OUTPUT_ROIS_DATA,
                        "X_" + ROI +
                                ".npy").replace("npy", "txt"), "w")
        fh.write('Centered and scaled data. ' \
          'Shape = (%i, %i): Intercept + Age + Gender + %i voxels' % \
          (n, p, mask_bool_ROI.sum()))
        fh.close()
        print ('\n')
       

            