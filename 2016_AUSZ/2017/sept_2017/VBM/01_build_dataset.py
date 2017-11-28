# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 09:57:30 2016

@author: ad247405


Compute mask, concatenate masked non-smoothed images for all the subjects.
Build X, y, and mask

INPUT:
- subject_list.txt:
- population.csv

OUTPUT_ICAARZ:
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
import mulm
import sklearn


BASE_PATH = '/neurospin/brainomics/2016_AUSZ/september_2017/results'
INPUT_CSV= os.path.join(BASE_PATH,"VBM","population.csv")
OUTPUT = os.path.join(BASE_PATH,"VBM","data","data_with_intercept")


# Read pop csv
pop = pd.read_csv(INPUT_CSV)
#############################################################################
# Read images
n = len(pop)
assert n == 123
Z = np.zeros((n, 3)) # Intercept + Age + Gender
Z[:, 0] = 1 # Intercept
y = np.zeros((n, 1)) # DX
MASCtot = np.zeros((n, 1)) # MASCtot scores
MASCless = np.zeros((n, 1)) # MASCtot scores
MASCexc = np.zeros((n, 1)) # MASCtot scores

images = list()
for i, index in enumerate(pop.index):
    cur = pop[pop.index== index]
    print(cur)
    imagefile_name = cur.path_VBM
    babel_image = nibabel.load(imagefile_name.as_matrix()[0])
    images.append(babel_image.get_data().ravel())
    Z[i, 1:]  = np.asarray(cur[["Âge_x", "sex.num"]]).ravel()
    y[i, 0] = cur["group.num"]
    MASCtot[i, 0] = cur[" MASCless"]
    MASCless[i, 0] = cur[" MASCless"]
    MASCexc[i, 0] = cur[" MASCexc"]

shape = babel_image.get_data().shape


#############################################################################
# Compute mask
# Implicit Masking involves assuming that a lower than a givent threshold
# at some voxel, in any of the images, indicates an unknown and is
# excluded from the analysis.
Xtot = np.vstack(images)
mask = (np.min(Xtot, axis=0) > 0.01) & (np.std(Xtot, axis=0) > 1e-6)
mask = mask.reshape(shape)
assert mask.sum() == 303240

#############################################################################
# Compute atlas mask
babel_mask_atlas = brainomics.image_atlas.resample_atlas_harvard_oxford(
    ref=imagefile_name.as_matrix()[0],
    output=os.path.join(OUTPUT, "mask.nii"))

mask_atlas = babel_mask_atlas.get_data()
assert np.sum(mask_atlas != 0) == 617728
mask_atlas[np.logical_not(mask)] = 0  # apply implicit mask
# smooth
mask_atlas = brainomics.image_atlas.smooth_labels(mask_atlas, size=(3, 3, 3))
assert np.sum(mask_atlas != 0) ==  261210
out_im = nibabel.Nifti1Image(mask_atlas,
                             affine=babel_image.get_affine())
out_im.to_filename(os.path.join(OUTPUT, "mask.nii"))
im = nibabel.load(os.path.join(OUTPUT, "mask.nii"))
assert np.all(mask_atlas == im.get_data())

#############################################################################
# Compute mask with atlas but binarized (not group tv)
mask_bool = mask_atlas != 0
mask_bool.sum() == 261210
out_im = nibabel.Nifti1Image(mask_bool.astype("int16"),
                             affine=babel_image.get_affine())
out_im.to_filename(os.path.join(OUTPUT, "mask.nii"))
babel_mask = nibabel.load(os.path.join(OUTPUT, "mask.nii"))
assert np.all(mask_bool == (babel_mask.get_data() != 0))


#############################################################################

# Save data X and y
X = Xtot[:, mask_bool.ravel()]
assert X.shape ==  (123, 261210)

X = np.hstack([Z, X])
assert X.shape ==  (123, 261213)


n, p = X.shape
np.save(os.path.join(OUTPUT, "X.npy"), X)
np.save(os.path.join(OUTPUT, "y.npy"), y)
np.save(os.path.join(OUTPUT, "MASCtot.npy"), MASCtot)
np.save(os.path.join(OUTPUT, "MASCless.npy"), MASCless)
np.save(os.path.join(OUTPUT, "MASCexc.npy"), MASCexc)


np.save(os.path.join(OUTPUT, "X_patients_only.npy"), X[(y!=0).ravel(),:])
np.save(os.path.join(OUTPUT, "y_patients_only.npy.npy"), y[(y!=0).ravel()])
np.save(os.path.join(OUTPUT, "MASCtot_patients_only.npy"), MASCtot[(y!=0).ravel()])
np.save(os.path.join(OUTPUT, "MASCless_patients_only.npy"), MASCless[(y!=0).ravel()])
np.save(os.path.join(OUTPUT, "MASCexc_patients_only.npy"),MASCexc[(y!=0).ravel()])

X = X[(y!=0).ravel(),:]
MASCtot = MASCtot[(y!=0).ravel()]
MASCless = MASCless[(y!=0).ravel()]
MASCexc = MASCexc[(y!=0).ravel()]
y = y[(y!=0).ravel()]

np.save(os.path.join(OUTPUT,"data_ASD_SCZ_only", "X.npy"), X[(y!=2).ravel(),:])
np.save(os.path.join(OUTPUT,"data_ASD_SCZ_only", "y.npy"), y[(y!=2).ravel()])
np.save(os.path.join(OUTPUT,"data_ASD_SCZ_only","MASCtot.npy"), MASCtot[(y!=2).ravel()])
np.save(os.path.join(OUTPUT,"data_ASD_SCZ_only","MASCless.npy"),MASCless[(y!=2).ravel()])
np.save(os.path.join(OUTPUT,"data_ASD_SCZ_only","MASCexc.npy"), MASCexc[(y!=2).ravel()])


###############################################################################

###############################################################################
# precompute linearoperator
import parsimony.functions.nesterov.tv as nesterov_tv
from parsimony.utils.linalgs import LinearOperatorNesterov

X = np.load(os.path.join(OUTPUT, "X.npy"))
y = np.load(os.path.join(OUTPUT, "y.npy"))
MASCtot = np.load(os.path.join(OUTPUT, "MASCtot.npy"))

mask = nibabel.load(os.path.join(OUTPUT, "mask.nii"))

Atv = nesterov_tv.linear_operator_from_mask(mask.get_data(), calc_lambda_max=True)
Atv.save(os.path.join(OUTPUT, "Atv.npz"))
Atv_ = LinearOperatorNesterov(filename=os.path.join(OUTPUT, "Atv.npz"))
assert Atv.get_singular_values(0) == Atv_.get_singular_values(0)
assert np.allclose(Atv_.get_singular_values(0), 11.916845597033738)


###############################################################################
#0 : controls
#1: ASD
#2 SCZ-ASD
#3 SCZ

##########################
#ASD vs CONTROLS (1 vs 0)
##########################
#########################
WD = "/neurospin/brainomics/2016_AUSZ/september_2017/results/VBM/data/asd_vs_controls"
X = np.load(os.path.join(OUTPUT, "X.npy"))
y = np.load(os.path.join(OUTPUT, "y.npy"))
X = X[y!= 2,:]
y = y[y!= 2]
X = X[y!= 3,:]
y = y[y!= 3]
np.save(os.path.join(WD,'X.npy'),X)
np.save(os.path.join(WD,'y.npy'),y)
#: 0 = controls and 1 = scz

##########################
#scz_asd_vs_asd (2 vs 1)
##########################
#########################
WD = "/neurospin/brainomics/2016_AUSZ/september_2017/results/VBM/data/scz_asd_vs_asd"
X = np.load(os.path.join(OUTPUT, "X.npy"))
y = np.load(os.path.join(OUTPUT, "y.npy"))
X = X[y!= 0,:]
y = y[y!= 0]
X = X[y!= 3,:]
y = y[y!= 3]
y[y==2]=0
np.save(os.path.join(WD,'X.npy'),X)
np.save(os.path.join(WD,'y.npy'),y)
#: 0 = scz-asd and 1 = asd

##########################
#scz_asd_vs_controls(2 vs 0)
##########################
#########################
WD = "/neurospin/brainomics/2016_AUSZ/september_2017/results/VBM/data/scz_asd_vs_controls"
X = np.load(os.path.join(OUTPUT, "X.npy"))
y = np.load(os.path.join(OUTPUT, "y.npy"))
X = X[y!= 1,:]
y = y[y!= 1]
X = X[y!= 3,:]
y = y[y!= 3]
y[y==2]=1
np.save(os.path.join(WD,'X.npy'),X)
np.save(os.path.join(WD,'y.npy'),y)
#: 0 = controls and 1 = scz-asd

##########################
#scz vs asd(3 vs 1)
##########################
#########################
WD = "/neurospin/brainomics/2016_AUSZ/september_2017/results/VBM/data/scz_vs_asd"
X = np.load(os.path.join(OUTPUT, "X.npy"))
y = np.load(os.path.join(OUTPUT, "y.npy"))
X = X[y!= 0,:]
y = y[y!= 0]
X = X[y!= 2,:]
y = y[y!= 2]
y[y==3]=0
np.save(os.path.join(WD,'X.npy'),X)
np.save(os.path.join(WD,'y.npy'),y)
#: 0 = scz and 1 = asd

##########################
#scz vs controls(3 vs 0)
##########################
#########################
WD = "/neurospin/brainomics/2016_AUSZ/september_2017/results/VBM/data/scz_vs_controls"
X = np.load(os.path.join(OUTPUT, "X.npy"))
y = np.load(os.path.join(OUTPUT, "y.npy"))
X = X[y!= 1,:]
y = y[y!= 1]
X = X[y!= 2,:]
y = y[y!= 2]
y[y==3]=1
np.save(os.path.join(WD,'X.npy'),X)
np.save(os.path.join(WD,'y.npy'),y)
#: 0 = controls and 1 = scz

##########################
#scz vs scz-asd(3 vs 2)
##########################
#########################
WD = "/neurospin/brainomics/2016_AUSZ/september_2017/results/VBM/data/scz_vs_scz-asd"
X = np.load(os.path.join(OUTPUT, "X.npy"))
y = np.load(os.path.join(OUTPUT, "y.npy"))
X = X[y!= 1,:]
y = y[y!= 1]
X = X[y!= 0,:]
y = y[y!= 0]
y[y==3]=1
y[y==2]=0
np.save(os.path.join(WD,'X.npy'),X)
np.save(os.path.join(WD,'y.npy'),y)
#: 0 = scz-asd and 1 = scz


##########################

#0 : controls
#1: ASD
#2 SCZ-ASD
#3 SCZ
#Save such that now labels are :
#0 : controls
#1: SCZ
#2 SCZ-ASD
#3 ASD
WD = "/neurospin/brainomics/2016_AUSZ/september_2017/results/VBM/data/data_for_regression_all"

X = np.load(os.path.join(OUTPUT, "X.npy"))
y = np.load(os.path.join(OUTPUT, "y.npy"))

y[y==1]=99
y[y==3]=1
y[y==99]=3


np.save(os.path.join(WD,'MASCtot.npy'),MASCtot)
np.save(os.path.join(WD,'X.npy'),X)
np.save(os.path.join(WD,'y.npy'),y)
