# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 14:28:38 2014

@author: cp243490

Build datasets only for maskdep MRI
ROI = "maskdep" (union of all defined ROIs)
MODALITY = MRI

OUTPUTS:
X, y, implicit mask

To build the implicit mask we use 2 ways of dilatation:
    - the first one, as done when considering evrey ROIs and the 3 Modalities
    (script 01_build_dataset_maskdep.py)
    dilatation process: 3x3 structuring element with connectivity 1
    and 2 iterations
    directory: "/dilatated_masks"
    - the second one: same dilattaion process but restricted to the brain
    region (to avoid considering regions that are outside teh brain)
    directory: "/dilatation_within-brain_masks"

"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import ndimage
REP_MAP = {"N": 0, "Y": 1}

BASE_PATH = "/neurospin/brainomics/2014_deptms/"

BASE_DATA_PATH = os.path.join(BASE_PATH,    "base_data")

INPUT_CSV = os.path.join(BASE_DATA_PATH,    "clinic", "deprimPetInfo.csv")
INPUT_ROIS_CSV = os.path.join(BASE_DATA_PATH,    "ROI_labels.csv")

OUTPUT_PATH = os.path.join(BASE_PATH,    "maskdep")
DATASETS_PATH = os.path.join(OUTPUT_PATH,    "datasets")
DILATE_PATH = os.path.join(DATASETS_PATH,    "dilatation_masks")
DILATE_WB_PATH = os.path.join(DATASETS_PATH,    "dilatation_within-brain_masks")
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)
if not os.path.exists(DATASETS_PATH):
    os.makedirs(DATASETS_PATH)
if not os.path.exists(DILATE_PATH):
    os.makedirs(DILATE_PATH)
if not os.path.exists(DILATE_WB_PATH):
    os.makedirs(DILATE_WB_PATH)
#############################################################################
## Read pop csv
pop = pd.read_csv(INPUT_CSV, sep="\t")
pop['Response.num'] = pop["Response"].map(REP_MAP)

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
            labels = np.asarray(label_ho.split(), dtype="int")
            dict_rois[roi_name] = [labels]
            dict_rois[roi_name].append(atlas_ho)

#############################################################################
## Build datasets: implicit mask
#############################################################################

####################
# Read images
n = len(pop)
assert n == 34
Z = np.zeros((n, 3))  # Intercept + Age + Gender
Z[:, 0] = 1  # Intercept
y = np.zeros((n, 1))  # Response
images = list()
fileName = "MRIm_G_file"
image_path = "MRI_images"

for i, MRIm_G_file in enumerate(pop["MRIm_G_file"]):
    cur = pop[pop.MRIm_G_file == MRIm_G_file]
    imagefile_name = os.path.join(BASE_DATA_PATH,
                                   "images",
                                   image_path,
                                   MRIm_G_file)
    babel_image = nib.load(imagefile_name)
    images.append(babel_image.get_data().ravel())
    Z[i, 1:] = np.asarray(cur[["Age", "Sex"]]).ravel()
    y[i, 0] = cur["Response.num"]

shape = babel_image.get_data().shape

#############################################################################
# Compute implicit mask and X and y matrix for the whole brain

# Compute mask
# Implicit Masking involves assuming that a lower than a given
# threshold at some voxel, in any of the images, indicates an unknown
# and is excluded from the analysis.
# maxk from MRI images

# whole brain
Xtot = np.vstack(images)
mask_brain = (np.min(Xtot, axis=0) > 0.01) & (np.std(Xtot, axis=0) > 1e-6)
mask_brain = mask_brain.reshape(shape)
mask_brain_im = nib.Nifti1Image(mask_brain.astype("int16"),
                          affine=babel_image.get_affine())
mask_brain_im.to_filename(os.path.join(DATASETS_PATH, "mask_brain.nii"))
# Xcsi for the whole brain
X_brain = Xtot[:, mask_brain.ravel()]
X_brain = np.hstack([Z, X_brain])
X_brain -= X_brain.mean(axis=0)
X_brain /= X_brain.std(axis=0)
X_brain[:, 0] = 1.
n, p = X_brain.shape
np.save(os.path.join(DATASETS_PATH, "X_brain.npy"), X_brain)
fh = open(os.path.join(DATASETS_PATH, "X_brain.npy").replace("npy", "txt"),
          "w")
fh.write('Centered and scaled data. ' \
         'Shape = (%i, %i): Intercept + Age + Gender + %i voxels' % \
         (n, p, mask_brain.sum()))
fh.close()
np.save(os.path.join(DATASETS_PATH, "y.npy"), y)

##########################################################
# Compute binarized mask with atlas for the specific ROIs

# Resample HarvardOxford atlas (cortical and subcortical) into
# reference space and interpolation

#fsl_cmd = "fsl5.0-applywarp -i %s -r %s -o %s --interp=nn"
#ref = imagefile_name[0]
#input_sub_filename = os.path.join("/usr/share/data",
#                      "harvard-oxford-atlases",
#                      "HarvardOxford",
#                      "HarvardOxford-sub-maxprob-thr0-2mm.nii.gz")
#input_cort_filename = os.path.join("/usr/share/data",
#                      "harvard-oxford-atlases",
#                      "HarvardOxford",
#                      "HarvardOxford-cort-maxprob-thr0-2mm.nii.gz")
#output_sub_filename = os.path.join(BASE_PATH, "images", "atlases",
#                      "HarvardOxford-sub-maxprob-thr0-2mm-nn.nii.gz")
#output_cort_filename = os.path.join(BASE_PATH, "images", "atlases",
#                      "HarvardOxford-cort-maxprob-thr0-2mm-nn.nii.gz")
#os.system(fsl_cmd % (input_sub_filename, ref, output_sub_filename))
#os.system(fsl_cmd % (input_cort_filename, ref, output_cort_filename))

sub_image = nib.load(os.path.join(BASE_DATA_PATH,
                      "images",
                      "atlases",
                      "HarvardOxford-sub-maxprob-thr0-1mm-nn.nii.gz"))
sub_arr = sub_image.get_data()
cort_image = nib.load(os.path.join(BASE_DATA_PATH,
                      "images",
                      "atlases",
                      "HarvardOxford-cort-maxprob-thr0-1mm-nn.nii.gz"))
cort_arr = cort_image.get_data()

# build a bool mask to define the region within the brain
brain_arr = (sub_arr.astype("bool") | cort_arr.astype("bool"))

# Compute the mask associated to each ROI by
# applying the implicit mask to the atlas
# and by identifying the ROI in the atlas (with the labels)
# Then dilate the mask

# Compute implicit masks by applying the implicit mask to the atlas
# and by identifying the ROI in the atlas (with the labels)
# Then dilate the mask

# construct implicit maskdep cortical region (union of all ROIs in the
# cortical region (Harvard Oxford atlas))
values_cort = dict_rois['Maskdep-cort']
labels_cort = values_cort[0]
mask_atlas_cort = np.copy(cort_arr)
mask_atlas_cort[np.logical_not(mask_brain)] = 0
mask_cort = np.zeros(mask_atlas_cort.shape)
for lab in labels_cort:
    mask_cort[mask_atlas_cort == lab] = 1
mask_atlas_cort[np.logical_not(mask_cort)] = 0

# construct implicit maskdep sub-cortical region (union of all ROIs in the
# sub-cortical region (Harvard Oxford atlas))
values_sub = dict_rois['Maskdep-sub']
labels_sub = values_sub[0]
mask_atlas_sub = np.copy(sub_arr)
mask_atlas_sub[np.logical_not(mask_brain)] = 0
mask_sub = np.zeros(mask_atlas_sub.shape)
for lab in labels_sub:
    mask_sub[mask_atlas_sub == lab] = 1
mask_atlas_sub[np.logical_not(mask_sub)] = 0

######################################
## Construct implicit mask for maskdep

# construct implicit maskdep region (union of all ROIs (both in the
# sub-cortical and cortical regions) (Harvard Oxford atlas))
values = dict_rois['Maskdep']
labels = values[0]
mask_dep = (mask_atlas_sub.astype("bool") | mask_atlas_cort.astype("bool"))

#################
## Dilatation

# Build datasets when mask built by dilatation of all mask
# 3x3 structuring element with connectivity 1 and 2 iterations
mask_dilatation = ndimage.morphology.binary_dilation(mask_dep,
                            iterations=2).astype(mask_dep.dtype)
mask_bool_dilatation = mask_dilatation.astype("bool")
out_im = nib.Nifti1Image(mask_bool_dilatation.astype("int16"),
                                 affine=babel_image.get_affine())
out_im.to_filename(os.path.join(DILATE_PATH, "mask_dilatation.nii"))
im = nib.load(os.path.join(DILATE_PATH, "mask_dilatation.nii"))
assert np.all(mask_bool_dilatation == im.get_data())

# Xcsi
X_dilatation = Xtot[:, mask_bool_dilatation.ravel()]
X_dilatation = np.hstack([Z, X_dilatation])
X_dilatation -= X_dilatation.mean(axis=0)
X_dilatation /= X_dilatation.std(axis=0)
X_dilatation[:, 0] = 1.
n, p = X_dilatation.shape
np.save(os.path.join(DILATE_PATH,
                "X_dilatation.npy"), X_dilatation)
fh = open(os.path.join(DILATE_PATH,
                "X_dilatation.npy").replace("npy", "txt"), "w")
fh.write('Centered and scaled data. ' \
    'Shape = (%i, %i): Intercept + Age + Gender + %i voxels' % \
    (n, p, mask_bool_dilatation.sum()))
fh.close()
np.save(os.path.join(DILATE_PATH, "y.npy"), y)

##############################
## Dilatation within the brain

# Build datasets when mask built by dilatation of all mask within the brain
# 3x3 structuring element with connectivity 1 and 2 iterations
# and then restrict the dilated mask to the brain region
mask_dilatation_wb = (mask_bool_dilatation & brain_arr)
mask_bool_dilatation_wb = mask_dilatation_wb.astype("bool")
out_im = nib.Nifti1Image(mask_bool_dilatation_wb.astype("int16"),
                                 affine=babel_image.get_affine())
out_im.to_filename(os.path.join(DILATE_WB_PATH,
                                "mask_dilatation_within-brain.nii"))
im = nib.load(os.path.join(DILATE_WB_PATH, "mask_dilatation_within-brain.nii"))
assert np.all(mask_bool_dilatation_wb == im.get_data())

# Xcsi
X_dilatation_wb = Xtot[:, mask_bool_dilatation_wb.ravel()]
X_dilatation_wb = np.hstack([Z, X_dilatation_wb])
X_dilatation_wb -= X_dilatation_wb.mean(axis=0)
X_dilatation_wb /= X_dilatation_wb.std(axis=0)
X_dilatation_wb[:, 0] = 1.
n, p = X_dilatation_wb.shape
np.save(os.path.join(DILATE_WB_PATH,
                "X_dilatation_within-brain.npy"), X_dilatation_wb)
fh = open(os.path.join(DILATE_WB_PATH,
                "X_dilatation_within-brain.npy").replace("npy", "txt"), "w")
fh.write('Centered and scaled data. ' \
    'Shape = (%i, %i): Intercept + Age + Gender + %i voxels' % \
    (n, p, mask_bool_dilatation_wb.sum()))
fh.close()
np.save(os.path.join(DILATE_WB_PATH, "y.npy"), y)