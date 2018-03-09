# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import nibabel
import brainomics.image_atlas
import mulm
import nilearn
from nilearn import plotting
from mulm import MUOLS


BASE_PATH = "/neurospin/brainomics/2017_memento/analysis/WMH"
INPUT_WMH = "/neurospin/cati/MEMENTO/WMH_registration_MNI_space/DB_Memento_Recalage_MNI_M00_v1"

INPUT_CSV = os.path.join(BASE_PATH,"population.csv")
OUTPUT = os.path.join(BASE_PATH,"data")

# Read pop csv
pop = pd.read_csv(INPUT_CSV)
assert  pop.shape == (1755, 26)
#pop = pop[pop.usubjid != "0020114HEPA"]
#assert  pop.shape == (2174, 26)

#############################################################################
# Read images
n = len(pop)
assert n == 1755

# 0 -->550
images = list()
for i, index in enumerate(pop.index):
    cur = pop[pop.index== index]
    print(cur.usubjid)
    imagefile_name = cur.wmh_path
    babel_image = nibabel.load(imagefile_name.as_matrix()[0])
    images.append(babel_image.get_data().ravel())
Xtot = np.vstack(images)



shape = babel_image.get_data().shape

#############################################################################
# Compute mask
# Implicit Masking involves assuming that a lower than a givent threshold
# at some voxel, in any of the images, indicates an unknown and is
# excluded from the analysis.
Xtot = np.vstack(images)
mask = (np.min(Xtot, axis=0) > 0.01) & (np.std(Xtot, axis=0) > 1e-6)
mask = mask.reshape(shape)
#assert mask.sum() == 345041

#############################################################################
# Compute atlas mask
babel_mask_atlas = brainomics.image_atlas.resample_atlas_harvard_oxford(
    ref=imagefile_name.as_matrix()[0],
    output=os.path.join(OUTPUT, "mask_whole_brain.nii"))

mask_atlas = babel_mask_atlas.get_data()
#assert np.sum(mask_atlas != 0) == 617728
mask_atlas[np.logical_not(mask)] = 0  # apply implicit mask
# smooth
mask_atlas = brainomics.image_atlas.smooth_labels(mask_atlas, size=(3, 3, 3))
#assert np.sum(mask_atlas != 0) == 301440
out_im = nibabel.Nifti1Image(mask_atlas,
                             affine=babel_image.get_affine())
out_im.to_filename(os.path.join(OUTPUT, "mask_whole_brain.nii"))
im = nibabel.load(os.path.join(OUTPUT, "mask_whole_brain.nii"))
#assert np.all(mask_atlas == im.get_data())

#############################################################################
# Compute mask with atlas but binarized (not group tv)
mask_bool = mask_atlas != 0
mask_bool.sum() == 301440
out_im = nibabel.Nifti1Image(mask_bool.astype("int16"),
                             affine=babel_image.get_affine())
out_im.to_filename(os.path.join(OUTPUT, "mask_whole_brain.nii"))
babel_mask = nibabel.load(os.path.join(OUTPUT, "mask_whole_brain.nii"))
assert np.all(mask_bool == (babel_mask.get_data() != 0))


#############################################################################

# Save data X and y
X = Xtot[:, mask_bool.ravel()]


X -= X.mean(axis=0)
X /= X.std(axis=0)
X[:, 0] = 1.
n, p = X.shape
X = np.nan_to_num(X)
np.save(os.path.join(OUTPUT, "X.npy"), X)

###############################################################################
# precompute linearoperator

mask = nibabel.load(os.path.join(OUTPUT, "mask_whole_brain.nii"))

import parsimony.functions.nesterov.tv as nesterov_tv
from parsimony.utils.linalgs import LinearOperatorNesterov

Atv = nesterov_tv.linear_operator_from_mask(mask.get_data(), calc_lambda_max=True)
Atv.save(os.path.join(OUTPUT, "Atv.npz"))
Atv_ = LinearOperatorNesterov(filename=os.path.join(OUTPUT, "Atv.npz"))
assert Atv.get_singular_values(0) == Atv_.get_singular_values(0)
