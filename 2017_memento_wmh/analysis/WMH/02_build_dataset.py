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


BASE_PATH = "/neurospin/brainomics/2017_memento/analysis"
INPUT_WMH = "/neurospin/cati/MEMENTO/WMH_registration_MNI_space/3DT1"

INPUT_CSV = os.path.join(BASE_PATH,"population.csv")
OUTPUT = os.path.join(BASE_PATH,"data")

# Read pop csv
pop = pd.read_csv(INPUT_CSV)
assert  pop.shape == (2175, 26)
pop = pop[pop.usubjid != "0020114HEPA"]
assert  pop.shape == (2174, 26)

#############################################################################
# Read images
n = len(pop)
assert n == 2174

pop["shape"] = 0
pop["transformations"] = 0
# 0 -->550
images = list()
for i, index in enumerate(pop.index):
    cur = pop[pop.index== index]
    print(cur.usubjid)
    imagefile_name = cur.wmh_path
    babel_image = nibabel.load(imagefile_name.as_matrix()[0])
    shape = babel_image.get_data().shape
    transformation = babel_image.get_affine()
    print(shape)
    pop.loc[pop.index== index,"shape"] = str(shape)
    pop.loc[pop.index== index,"transformations"] = str(transformation)
    images.append(babel_image.get_data().ravel())
Xtot1 = np.vstack(images)


pop = pop[["shape","usubjid","wmh_path"]]
pop.to_csv("/neurospin/brainomics/2017_memento/analysis/shape_image.csv", index=False)
#babel_image = nilearn.image.resample_img(babel_image, target_affine=babel_image.get_affine()*2,\
#                                           target_shape=[60,72,60],\
#                                             interpolation='continuous', copy=True, order='F')


# 550 -->1100
images = list()
for i, index in enumerate(pop.index[550:1100]):
    cur = pop[pop.index== index]
    #print(cur)
    imagefile_name = cur.wmh_path
    babel_image = nibabel.load(imagefile_name.as_matrix()[0])
    shape = babel_image.get_data().shape
    print(shape)
Xtot1 = np.vstack(images)

# 1100 -->550
images = list()
for i, index in enumerate(pop.index[0:550]):
    cur = pop[pop.index== index]
    print(cur)
    imagefile_name = cur.wmh_path
    babel_image = nibabel.load(imagefile_name.as_matrix()[0])
    images.append(babel_image.get_data().ravel())
Xtot1 = np.vstack(images)

# 0 -->550
images = list()
for i, index in enumerate(pop.index[0:550]):
    cur = pop[pop.index== index]
    print(cur)
    imagefile_name = cur.wmh_path
    babel_image = nibabel.load(imagefile_name.as_matrix()[0])
    images.append(babel_image.get_data().ravel())
Xtot1 = np.vstack(images)


shape = babel_image.get_data().shape

#############################################################################
# Compute mask
# Implicit Masking involves assuming that a lower than a givent threshold
# at some voxel, in any of the images, indicates an unknown and is
# excluded from the analysis.
Xtot = np.vstack(images)
mask = (np.min(Xtot, axis=0) > 0.01) & (np.std(Xtot, axis=0) > 1e-6)
mask = mask.reshape(shape)
assert mask.sum() == 345041

#############################################################################
# Compute atlas mask
babel_mask_atlas = brainomics.image_atlas.resample_atlas_harvard_oxford(
    ref=imagefile_name.as_matrix()[0],
    output=os.path.join(OUTPUT_DATA_ICAAR, "mask_whole_brain.nii"))

mask_atlas = babel_mask_atlas.get_data()
assert np.sum(mask_atlas != 0) == 617728
mask_atlas[np.logical_not(mask)] = 0  # apply implicit mask
# smooth
mask_atlas = brainomics.image_atlas.smooth_labels(mask_atlas, size=(3, 3, 3))
assert np.sum(mask_atlas != 0) == 301440
out_im = nibabel.Nifti1Image(mask_atlas,
                             affine=babel_image.get_affine())
out_im.to_filename(os.path.join(OUTPUT_DATA_ICAAR, "mask_whole_brain.nii"))
im = nibabel.load(os.path.join(OUTPUT_DATA_ICAAR, "mask_whole_brain.nii"))
assert np.all(mask_atlas == im.get_data())

#############################################################################
# Compute mask with atlas but binarized (not group tv)
mask_bool = mask_atlas != 0
mask_bool.sum() == 301440
out_im = nibabel.Nifti1Image(mask_bool.astype("int16"),
                             affine=babel_image.get_affine())
out_im.to_filename(os.path.join(OUTPUT_DATA_ICAAR, "mask_whole_brain.nii"))
babel_mask = nibabel.load(os.path.join(OUTPUT_DATA_ICAAR, "mask_whole_brain.nii"))
assert np.all(mask_bool == (babel_mask.get_data() != 0))


#############################################################################

# Save data X and y
X = Xtot[:, mask_bool.ravel()]
#Use mean imputation, we could have used median for age
#imput = sklearn.preprocessing.Imputer(strategy = 'median',axis=0)
#Z = imput.fit_transform(Z)
X = np.hstack([Z, X])
assert X.shape == (55, 301443)

#Remove nan lines
X= X[np.logical_not(np.isnan(y)).ravel(),:]
y=y[np.logical_not(np.isnan(y))]
assert X.shape == (41, 301443)


X -= X.mean(axis=0)
X /= X.std(axis=0)
X[:, 0] = 1.
n, p = X.shape
X = np.nan_to_num(X)
np.save(os.path.join(OUTPUT_DATA_ICAAR, "X.npy"), X)
np.save(os.path.join(OUTPUT_DATA_ICAAR, "y.npy"), y)

###############################################################################
# precompute linearoperator
X = np.load(os.path.join(OUTPUT_DATA_ICAAR, "X.npy"))
y = np.load(os.path.join(OUTPUT_DATA_ICAAR, "y.npy"))

mask = nibabel.load(os.path.join(OUTPUT_DATA_ICAAR, "mask_whole_brain.nii"))

import parsimony.functions.nesterov.tv as nesterov_tv
from parsimony.utils.linalgs import LinearOperatorNesterov

Atv = nesterov_tv.linear_operator_from_mask(mask.get_data(), calc_lambda_max=True)
Atv.save(os.path.join(OUTPUT_DATA_ICAAR, "Atv.npz"))
Atv_ = LinearOperatorNesterov(filename=os.path.join(OUTPUT_DATA_ICAAR, "Atv.npz"))
assert Atv.get_singular_values(0) == Atv_.get_singular_values(0)
