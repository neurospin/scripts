# -*- coding: utf-8 -*-
"""
@author: edouard.Duchesnay@cea.fr

Compute mask, concatenate masked non-smoothed images for all the subjects.
Build X, y, and mask

INPUT:
- subject_list.txt:
- population.csv

OUTPUT:
- mask.nii
- y.npy
- X.npy = intercept + Age + Gender + Voxel
"""

import os
import numpy as np
import glob
import pandas as pd
import nibabel
import brainomics.image_atlas
import shutil

#import proj_classif_config
GENDER_MAP = {'Female': 0, 'Male': 1}

BASE_PATH = "/neurospin/brainomics/2013_adni"
#INPUT_CLINIC_FILENAME = os.path.join(BASE_PATH, "clinic", "adnimerge_baseline.csv")
INPUT_SUBJECTS_LIST_FILENAME = os.path.join(BASE_PATH,
                                   "templates",
                                   "template_FinalQC",
                                   "subject_list.txt")

INPUT_IMAGEFILE_FORMAT = os.path.join(BASE_PATH,
                                   "templates",
                                   "template_FinalQC",
                                   "registered_images",
                                    "mw{PTID}*_Nat_dartel_greyProba.nii")

INPUT_CSV = os.path.join(BASE_PATH,          "MCIc-MCInc", "population.csv")

OUTPUT = os.path.join(BASE_PATH,             "MCIc-MCInc")
OUTPUT_CS = os.path.join(BASE_PATH,          "MCIc-MCInc_cs")
OUTPUT_ATLAS = os.path.join(BASE_PATH,       "MCIc-MCInc_gtvenet")
OUTPUT_CS_ATLAS = os.path.join(BASE_PATH,    "MCIc-MCInc_cs_gtvenet")

os.makedirs(OUTPUT)
os.makedirs(OUTPUT_CS)
os.makedirs(OUTPUT_ATLAS)
os.makedirs(OUTPUT_CS_ATLAS)


#OUTPUT_Xcs_ATLAS = os.path.join(BASE_PATH,  "proj_classif_MCIc-MCInc_cs_gtvenet", "Xcs_atlas.npy")
#OUTPUT_y = os.path.join(BASE_PATH, "proj_classif_MCIc-MCInc", "y.npy")

# Read input subjects
input_subjects = pd.read_table(INPUT_SUBJECTS_LIST_FILENAME, sep=" ",
                               header=None)
input_subjects = [x[:10] for x in input_subjects[1]]

# Read pop csv
pop = pd.read_csv(INPUT_CSV)
pop['PTGENDER.num'] = pop["PTGENDER"].map(GENDER_MAP)

#############################################################################
# Read images
n = len(pop)
Z = np.zeros((n, 3)) # Intercept + Age + Gender
Z[:, 0] = 1 # Intercept
y = np.zeros((n, 1)) # DX
images = list()
for i, PTID in enumerate(pop['PTID']):
    cur = pop[pop.PTID == PTID]
    print cur
    imagefile_pattern = INPUT_IMAGEFILE_FORMAT.format(PTID=PTID)
    imagefile_name = glob.glob(imagefile_pattern)
    if len(imagefile_name) != 1:
        raise ValueError("Found %i files" % len(imagefile_name))
    babel_image = nibabel.load(imagefile_name[0])
    images.append(babel_image.get_data().ravel())
    Z[i, 1:] = np.asarray(cur[["AGE", "PTGENDER.num"]]).ravel()
    y[i, 0] = cur["DX.num"]

shape = babel_image.get_data().shape

#############################################################################
# Compute mask
# Implicit Masking involves assuming that a lower than a givent threshold
# at some voxel, in any of the images, indicates an unknown and is
# excluded from the analysis.
Xtot = np.vstack(images)
mask = (np.min(Xtot, axis=0) > 0.01) & (np.std(Xtot, axis=0) > 1e-6)
mask = mask.reshape(shape)
assert mask.sum() == 320925

#############################################################################
# Compute atlas mask
babel_mask_atlas = brainomics.image_atlas.resample_atlas_harvard_oxford(
    ref=imagefile_name[0],
    output=os.path.join(OUTPUT_ATLAS, "mask.nii"))

mask_atlas = babel_mask_atlas.get_data()
assert np.sum(mask_atlas != 0) == 638715
mask_atlas[np.logical_not(mask)] = 0  # apply implicit mask
# smooth
mask_atlas = brainomics.image_atlas.smooth_labels(mask_atlas, size=(3, 3, 3))
assert np.sum(mask_atlas != 0) == 292971#296306
out_im = nibabel.Nifti1Image(mask_atlas,
                             affine=babel_image.get_affine())
out_im.to_filename(os.path.join(OUTPUT_ATLAS, "mask.nii"))
im = nibabel.load(os.path.join(OUTPUT_ATLAS, "mask.nii"))
assert np.all(mask_atlas == im.get_data())


shutil.copyfile(os.path.join(OUTPUT_ATLAS, "mask.nii"), os.path.join(OUTPUT_CS_ATLAS, "mask.nii"))

#############################################################################
# Compute mask with atlas but binarized (not group tv)
mask_bool = mask_atlas != 0
mask_bool.sum() == 292971#296306
out_im = nibabel.Nifti1Image(mask_bool.astype("int16"),
                             affine=babel_image.get_affine())
out_im.to_filename(os.path.join(OUTPUT, "mask.nii"))
babel_mask = nibabel.load(os.path.join(OUTPUT, "mask.nii"))
assert np.all(mask_bool == (babel_mask.get_data() != 0))

shutil.copyfile(os.path.join(OUTPUT, "mask.nii"), os.path.join(OUTPUT_CS, "mask.nii"))

#############################################################################
# X
X = Xtot[:, mask_bool.ravel()]
X = np.hstack([Z, X])
assert X.shape == (159, 292974)
n, p = X.shape
np.save(os.path.join(OUTPUT, "X.npy"), X)
fh = open(os.path.join(OUTPUT, "X.npy").replace("npy", "txt"), "w")
fh.write('shape = (%i, %i): Intercept + Age + Gender + %i voxels' % \
    (n, p, mask_bool.sum()))
fh.close()

# Xcs
X = Xtot[:, mask_bool.ravel()]
X = np.hstack([Z[:, 1:], X])
assert X.shape == (159, 292973)
X -= X.mean(axis=0)
X /= X.std(axis=0)
n, p = X.shape
np.save(os.path.join(OUTPUT_CS, "X.npy"), X)
fh = open(os.path.join(OUTPUT_CS, "X.npy").replace("npy", "txt"), "w")
fh.write('Centered and scaled data. Shape = (%i, %i): Age + Gender + %i voxels' % \
    (n, p, mask.sum()))
fh.close()

# atlas
X = Xtot[:, (mask_atlas.ravel() != 0)]
X = np.hstack([Z, X])
assert X.shape == (159, 292974)
n, p = X.shape
np.save(os.path.join(OUTPUT_ATLAS, "X.npy"), X)
fh = open(os.path.join(OUTPUT_ATLAS, "X.npy").replace("npy", "txt"), "w")
fh.write('shape = (%i, %i): Intercept + Age + Gender + %i voxels' % \
    (n, p, (mask_atlas.ravel() != 0).sum()))
fh.close()

# atlas cs
X = Xtot[:, (mask_atlas.ravel() != 0)]
X = np.hstack([Z[:, 1:], X])
assert X.shape == (159, 292973)
X -= X.mean(axis=0)
X /= X.std(axis=0)
n, p = X.shape
np.save(os.path.join(OUTPUT_CS_ATLAS, "X.npy"), X)
fh = open(os.path.join(OUTPUT_CS_ATLAS, "X.npy").replace("npy", "txt"), "w")
fh.write('Centered and scaled data. Shape = (%i, %i): Age + Gender + %i voxels' % \
    (n, p, (mask_atlas.ravel() != 0).sum()))
fh.close()

np.save(os.path.join(OUTPUT, "y.npy"), y)
np.save(os.path.join(OUTPUT_CS, "y.npy"), y)
np.save(os.path.join(OUTPUT_ATLAS, "y.npy"), y)
np.save(os.path.join(OUTPUT_CS_ATLAS, "y.npy"), y)

