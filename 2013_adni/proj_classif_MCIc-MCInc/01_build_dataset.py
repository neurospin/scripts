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

INPUT_CSV = os.path.join(BASE_PATH, "proj_classif_MCIc-MCInc", "population.csv")
OUTPUT_MASK = os.path.join(BASE_PATH, "proj_classif_MCIc-MCInc", "mask.nii.gz")
OUTPUT_MASK_ATLAS = os.path.join(BASE_PATH, "proj_classif_MCIc-MCInc", "mask_atlas.nii.gz")
OUTPUT_X = os.path.join(BASE_PATH, "proj_classif_MCIc-MCInc", "X.npy")
OUTPUT_y = os.path.join(BASE_PATH, "proj_classif_MCIc-MCInc", "y.npy")

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
mask = np.min(Xtot, axis=0) > 0.01
mask = mask.reshape(shape)
assert mask.sum() == 320925


# Save mask
out_im = nibabel.Nifti1Image(mask.astype("int16"),
                             affine=babel_image.get_affine())
out_im.to_filename(OUTPUT_MASK)
babel_mask = nibabel.load(OUTPUT_MASK)
assert np.all(mask == (babel_mask.get_data() != 0))

#############################################################################
# Compute atlas mask
brainomics.image_atlas.resample_atlas_harvard_oxford(
    ref=imagefile_name[0],
    output=OUTPUT_MASK_ATLAS)

babel_mask_atlas = nibabel.load(OUTPUT_MASK_ATLAS)
mask_atlas = babel_mask_atlas.get_data()
assert np.sum(mask_atlas != 0) == 618462
mask_atlas[np.logical_not(mask)] = 0  # apply implicit mask
assert np.sum(mask_atlas != 0) == 291676
out_im = nibabel.Nifti1Image(mask_atlas,
                             affine=babel_image.get_affine())
out_im.to_filename(OUTPUT_MASK_ATLAS)
im = nibabel.load(OUTPUT_MASK_ATLAS)
assert np.all(mask_atlas == im.get_data())


#############################################################################
# X
X = Xtot[:, mask]
X = np.hstack([Z, X])
print "X.shape =", X.shape
# X.shape = (202, 314175)
n, p = X.shape

np.save(OUTPUT_X, X)
fh = open(OUTPUT_X.replace("npy", "txt"), "w")
fh.write('shape = (%i, %i): Intercept + Age + Gender + %i voxels' % \
    (n, p, mask.sum()))
fh.close()

np.save(OUTPUT_y, y)
