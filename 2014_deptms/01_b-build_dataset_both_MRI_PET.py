# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 15:37:20 2014

@author: cp243490

Construction of the datasets for teh modality MRI+PET
Read the data (clinic data, ROI, MRI AND PET images)
Read masks, X, y for each ROI, MRI, PET
Construct the masks, Build and save the datasets for the MODALITY MRI+PET by
concatenating the previous results (MRI and PET results)
for:
    - the whole brain
    - each ROI

INPUTs
- ROIs : /neurospin/brainomics/2014_deptms/ROI_labels.csv
- PET and MRI datasets : /neurospin/brainomics/2014_deptms/*{MRI, PET}/
                        - mask_*.nii
                        (implicit mask and masks describing each ROI)
                        - X_*.npy
                        (Intercept + Age + Gender + MRI or PET images
                        for the whole brain and each ROI)
                        - y.npy (response to the treatment)

OUTPUT
- implicit mask for the whole brain and mask for each ROI :
    /neurospin/brainomics/2014_deptms/*/mask_*.nii
- X for the whole brain and for each ROI associated to a modality :
    /neurospin/brainomics/2014_deptms/MRI+PET/X*.npy
    (Intercept + Age + Gender + concatenation of (MRI and PET) images
    for the whole brain and each ROI)
- y : /neurospin/brainomics/2014_deptms/MRI+PET/y*.npy
    (response to the treatment)
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib

REP_MAP = {"norep": 0, "rep": 1}

MODALITY = "MRI+PET"

print "Modality: ", MODALITY

BASE_PATH = "/neurospin/brainomics/2014_deptms"

INPUT_CSV = os.path.join(BASE_PATH,          "clinic", "deprimPetInfo.csv")
INPUT_ROIS_CSV = os.path.join(BASE_PATH,     "ROI_labels.csv")

input_mri = os.path.join(BASE_PATH,          "MRI")
input_pet = os.path.join(BASE_PATH,          "PET")

OUTPUT_CSI = os.path.join(BASE_PATH,          MODALITY)

if not os.path.exists(OUTPUT_CSI):
    os.makedirs(OUTPUT_CSI)

# Read pop csv
pop = pd.read_csv(INPUT_CSV, sep="\t")
pop['rep_norep.num'] = pop["rep_norep"].map(REP_MAP)

# Read ROIs csv
atlas = []
dict_rois = {}
df_rois = pd.read_csv(INPUT_ROIS_CSV)
for i, ROI_name_aal in enumerate(df_rois["ROI_name_aal"]):
    cur = df_rois[df_rois.ROI_name_aal == ROI_name_aal]
    label_ho = cur["label_ho"].values[0]
    atlas_ho = cur["atlas_ho"].values[0]
    roi_name = cur["ROI_name_deptms"].values[0]
    if ((not cur.isnull()["label_ho"].values[0])
        and (not cur.isnull()["ROI_name_deptms"].values[0])):
        if not roi_name in dict_rois:
            print "ROI_name_deptms", roi_name
            labels = np.asarray(label_ho.split(), dtype="int")
            dict_rois[roi_name] = [labels]
            dict_rois[roi_name].append(atlas_ho)
            print dict_rois[roi_name]
            print "\n"

#############################################################################
# Compute MRI+PET implicit mask and X and y matrix for the whole brain
#############################################################################
# GET MRI AND PET matrices X, y, mask for the whole brain
n = len(pop)
assert n == 34
Z = np.zeros((n, 3))  # Intercept + Age + Gender
Z[:, 0] = 1  # Intercept

X_mri = np.load(os.path.join(BASE_PATH, "MRI", "X_MRI_wb.npy"))
X_pet = np.load(os.path.join(BASE_PATH, "PET", "X_PET_wb.npy"))

Z[:, 1:3] = X_mri[:, 1:3]  # get Intercerpt + Age + Sex data
X_mri = X_mri[:, 3:]  # remove Intercerpt + Age + Sex data
X_pet = X_pet[:, 3:]  # remove Intercerpt + Age + Sex data

y = np.load(os.path.join(BASE_PATH, "MRI", "y.npy"))  # rep_no_rep


mask_ima_mri = nib.load(os.path.join(BASE_PATH, "MRI", "mask_MRI_wb.nii"))
mask_arr_mri = mask_ima_mri.get_data() != 0
mask_ima_pet = nib.load(os.path.join(BASE_PATH, "PET", "mask_PET_wb.nii"))
mask_arr_pet = mask_ima_pet.get_data() != 0

#############################################################################
# X and the masks for MRI + PET modality
X = np.hstack((X_mri, X_pet))


mask = np.vstack((mask_arr_mri, mask_arr_pet))
out_im = nib.Nifti1Image(mask.astype("int16"),
                             affine=mask_ima_mri.get_affine())
out_im.to_filename(os.path.join(OUTPUT_CSI, "mask_" + MODALITY + "_wb.nii"))
babel_mask = nib.load(os.path.join(OUTPUT_CSI, "mask_" + MODALITY + "_wb.nii"))
assert np.all(mask == (babel_mask.get_data() != 0))

#############################################################################
# Xcsi for the whole brain

X = np.hstack([Z, X])
X -= X.mean(axis=0)
X /= X.std(axis=0)
X[:, 0] = 1.
n, p = X.shape
np.save(os.path.join(OUTPUT_CSI, "X_" + MODALITY + "_wb.npy"), X)
fh = open(os.path.join(OUTPUT_CSI, "X_" +
                                   MODALITY +
                                   "_wb.npy").replace("npy", "txt"), "w")
fh.write('Centered and scaled data. '\
            'Shape = (%i, %i): Intercept + Age + Gender + %i voxels' % \
            (n, p, mask.sum()))
fh.close()

np.save(os.path.join(OUTPUT_CSI, "y.npy"), y)


#############################################################################
# Do the same for each ROI
#############################################################################

for ROI, values in dict_rois.iteritems():
    print "ROI: ", ROI
    print "labels", labels

    # GET MRI AND PET matrices X, y, mask for each ROI
    X_mri_ROI = np.load(os.path.join(BASE_PATH,
                                     "MRI", "X_MRI_" + ROI + ".npy"))
    X_pet_ROI = np.load(os.path.join(BASE_PATH,
                                     "PET", "X_PET_" + ROI + ".npy"))
    X_mri_ROI = X_mri_ROI[:, 3:]    # remove Intercerpt + Age + Sex data
    X_pet_ROI = X_pet_ROI[:, 3:]    # remove Intercerpt + Age + Sex data

    mask_ima_mri_roi = nib.load(os.path.join(BASE_PATH, "MRI", "mask_MRI_" +
                                                        ROI + ".nii"))
    mask_arr_mri_roi = mask_ima_mri_roi.get_data() != 0
    mask_ima_pet_roi = nib.load(os.path.join(BASE_PATH, "PET", "mask_PET_" +
                                                         ROI + ".nii"))
    mask_arr_pet_roi = mask_ima_pet_roi.get_data() != 0

    # Create X_ROI and the masks for MRI + PET modality associated to each ROI
    X_ROI = np.hstack((X_mri_ROI, X_pet_ROI))

    mask_ROI = np.vstack((mask_arr_mri_roi, mask_arr_pet_roi))
    out_im = nib.Nifti1Image(mask_ROI.astype("int16"),
                             affine=mask_ima_mri.get_affine())
    out_im.to_filename(os.path.join(OUTPUT_CSI, "mask_" + MODALITY + "_" +
                                                ROI + ".nii"))
    babel_mask = nib.load(os.path.join(OUTPUT_CSI, "mask_" + MODALITY + "_" +
                                                    ROI + ".nii"))
    assert np.all(mask_ROI == (babel_mask.get_data() != 0))

    # Xcsi for the specific ROI
    X_ROI = np.hstack([Z, X_ROI])
    X_ROI -= X_ROI.mean(axis=0)
    X_ROI /= X_ROI.std(axis=0)
    X_ROI[:, 0] = 1.
    n, p = X_ROI.shape
    np.save(os.path.join(OUTPUT_CSI,
                         "X_" + MODALITY + "_" + ROI + ".npy"), X_ROI)
    fh = open(os.path.join(OUTPUT_CSI, "X_" + MODALITY + "_" + ROI +
                                        ".npy").replace("npy", "txt"), "w")
    fh.write('Centered and scaled data.' /
                'Shape = (%i, %i): Intercept + Age + Gender + %i voxels' % \
        (n, p, mask_ROI.sum()))
    fh.close()