# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 17:02:08 2014

@author: cp243490
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
import brainomics.image_atlas

#import proj_classif_config
GENDER_MAP = {'Female': 0, 'Male': 1}

BASE_PATH = "/neurospin/brainomics/2014_imagen_fu2_adrs/"


INPUT_CSV = os.path.join(BASE_PATH,      "ADRS_population", "population.csv")
OUTPUT_DATA = os.path.join(BASE_PATH,    "ADRS_datasets")
OUTPUT_ATLAS = os.path.join(OUTPUT_DATA, "atlas")

if not os.path.exists(OUTPUT_DATA):
    os.makedirs(OUTPUT_DATA)
if not os.path.exists(OUTPUT_ATLAS):
    os.makedirs(OUTPUT_ATLAS)

# Read input subjects
# Read pop csv
pop = pd.read_csv(INPUT_CSV)
pop['Gender_num'] = pop["Gender"].map(GENDER_MAP)

#############################################################################
# Read images
n = len(pop)
assert n == 1082
Z = np.zeros((n, 3))  # Intercept + Age + Gender
Z[:, 0] = 1  # Intercept
y = np.zeros((n, 1))  # ADRS
images = list()
for i, mri_path in enumerate(pop["mri_path"]):
    print i
    cur = pop.iloc[i]
    #print cur
    babel_image = nib.load(mri_path)
    images.append(babel_image.get_data().ravel())
    Z[i, 1:] = np.asarray(cur[["Age", "Gender_num"]]).ravel()
    y[i, 0] = cur["adrs"]
shape = babel_image.get_data().shape
np.save(os.path.join(OUTPUT_DATA, "y.npy"), y)

############################################################################
# resample one anat
fsl_cmd = "fsl5.0-applywarp -i %s -r %s -o %s" % \
("/usr/share/data/fsl-mni152-templates/MNI152_T1_1mm_brain.nii.gz",
cur["mri_path"],
os.path.join(OUTPUT_ATLAS,
             "MNI152_T1_1mm_brain.nii.gz"))

os.system(fsl_cmd)

fsl_cmd = "fsl5.0-applywarp -i %s -r %s -o %s --interp=nn" % \
("/usr/share/data/harvard-oxford-atlases/\
    HarvardOxford/HarvardOxford-cort-maxprob-thr0-1mm.nii.gz",
    cur["mri_path"],
    os.path.join(OUTPUT_ATLAS,
             "HarvardOxford-cort-maxprob-thr0-1mm-nn.nii.gz"))

os.system(fsl_cmd)

fsl_cmd = "fsl5.0-applywarp -i %s -r %s -o %s" % \
("/usr/share/data/harvard-oxford-atlases/\
    HarvardOxford/HarvardOxford-sub-maxprob-thr0-1mm.nii.gz",
    cur["mri_path"],
    os.path.join(OUTPUT_ATLAS,
                 "HarvardOxford-sub-maxprob-thr0-1mm-nn.nii.gz"))

os.system(fsl_cmd)

#############################################################################
# Compute implicit mask
# Implicit Masking involves assuming that a lower than a givent threshold
# at some voxel, in any of the images, indicates an unknown and is
# excluded from the analysis.
Xtot = np.vstack(images)
np.save(os.path.join(OUTPUT_DATA, "Xtot.npy"), Xtot)
del images

mask_implicit_arr = (np.min(Xtot, axis=0) > 0.01) & \
                        (np.std(Xtot, axis=0) > 1e-6)
mask_implicit_arr = mask_implicit_arr.reshape(shape)
assert mask_implicit_arr.sum() == 730646
out_im = nib.Nifti1Image(mask_implicit_arr.astype(int),
                             affine=babel_image.get_affine())
out_im.to_filename(os.path.join(OUTPUT_DATA, "mask_implicit.nii.gz"))

#############################################################################
# Compute atlas mask
mask_atlas_ima = brainomics.image_atlas.resample_atlas_harvard_oxford(
    ref=cur["mri_path"],
    output=os.path.join(OUTPUT_DATA, "mask_atlas.nii.gz"))

mask_atlas_arr = mask_atlas_ima.get_data()
assert np.sum(mask_atlas_arr != 0) == 638715
mask_atlas_arr[np.logical_not(mask_implicit_arr)] = 0  # apply implicit mask
# smooth
mask_atlas_arr = brainomics.image_atlas.smooth_labels(mask_atlas_arr,
                                                      size=(3, 3, 3))
assert np.sum(mask_atlas_arr != 0) == 625897
out_im = nib.Nifti1Image(mask_atlas_arr,
                             affine=babel_image.get_affine())
out_im.to_filename(os.path.join(OUTPUT_DATA, "mask_atlas.nii.gz"))
im = nib.load(os.path.join(OUTPUT_DATA, "mask_atlas.nii.gz"))
assert np.all(mask_atlas_arr.astype(int) == im.get_data())


#############################################################################
# Compute mask with atlas but binarized (not group tv)
mask_atlas_binarized_arr = mask_atlas_arr != 0
assert mask_atlas_binarized_arr.sum() == 625897
out_im = nib.Nifti1Image(mask_atlas_binarized_arr.astype("int16"),
                             affine=babel_image.get_affine())
out_im.to_filename(os.path.join(OUTPUT_DATA, "mask_atlas_binarized.nii.gz"))
babel_mask = nib.load(os.path.join(OUTPUT_DATA, "mask_atlas_binarized.nii.gz"))
assert np.all(mask_atlas_binarized_arr == (babel_mask.get_data() != 0))

#############################################################################
# X
X = Xtot[:, mask_atlas_binarized_arr.ravel()]

X = np.hstack([Z, X])
assert X.shape == (242, 285986)
n, p = X.shape
X -= X.mean(axis=0)
X /= X.std(axis=0)
X[:, 0] = 1.
np.save(os.path.join(OUTPUT_DATA, "X.npy"), X)
fh = open(os.path.join(OUTPUT_DATA, "X.npy").replace("npy", "txt"), "w")
fh.write('shape = (%i, %i): Intercept + Age + Gender + %i voxels' % \
    (n, p, mask_atlas_binarized_arr.sum()))
fh.close()