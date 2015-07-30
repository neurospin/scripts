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

INPUT_CSV = os.path.join(BASE_PATH,          "MCIc-CTL", "population.csv")

OUTPUT = os.path.join(BASE_PATH,             "MCIc-CTL")
OUTPUT_CS = os.path.join(BASE_PATH,          "MCIc-CTL_cs")
OUTPUT_CSI = os.path.join(BASE_PATH,          "MCIc-CTL_csi")
OUTPUT_ATLAS = os.path.join(BASE_PATH,       "MCIc-CTL_gtvenet")
OUTPUT_CS_ATLAS = os.path.join(BASE_PATH,    "MCIc-CTL_cs_gtvenet")
OUTPUT_S = os.path.join(BASE_PATH,          "MCIc-CTL_s") # No Covariate
OUTPUT_S_S = os.path.join(BASE_PATH,          "MCIc-CTL_s_s") # No Covariate + smooth
OUTPUT_CS_S = os.path.join(BASE_PATH,          "MCIc-CTL_cs_s") # Covariate + smooth


os.makedirs(OUTPUT)
os.makedirs(OUTPUT_CS)
os.makedirs(OUTPUT_CSI)
os.makedirs(OUTPUT_ATLAS)
os.makedirs(OUTPUT_CS_ATLAS)
os.makedirs(OUTPUT_S)
os.makedirs(OUTPUT_S_S)

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
assert n == 202
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
assert mask.sum() == 314172

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
assert np.sum(mask_atlas != 0) == 286117
out_im = nibabel.Nifti1Image(mask_atlas,
                             affine=babel_image.get_affine())
out_im.to_filename(os.path.join(OUTPUT_ATLAS, "mask.nii"))
im = nibabel.load(os.path.join(OUTPUT_ATLAS, "mask.nii"))
assert np.all(mask_atlas == im.get_data())


shutil.copyfile(os.path.join(OUTPUT_ATLAS, "mask.nii"), os.path.join(OUTPUT_CS_ATLAS, "mask.nii"))

#############################################################################
# Compute mask with atlas but binarized (not group tv)
mask_bool = mask_atlas != 0
mask_bool.sum() == 286117
out_im = nibabel.Nifti1Image(mask_bool.astype("int16"),
                             affine=babel_image.get_affine())
out_im.to_filename(os.path.join(OUTPUT, "mask.nii"))
babel_mask = nibabel.load(os.path.join(OUTPUT, "mask.nii"))
assert np.all(mask_bool == (babel_mask.get_data() != 0))

shutil.copyfile(os.path.join(OUTPUT, "mask.nii"), os.path.join(OUTPUT_CS, "mask.nii"))
shutil.copyfile(os.path.join(OUTPUT, "mask.nii"), os.path.join(OUTPUT_S, "mask.nii"))
shutil.copyfile(os.path.join(OUTPUT, "mask.nii"), os.path.join(OUTPUT_S_S, "mask.nii"))

#############################################################################
# X
X = Xtot[:, mask_bool.ravel()]
X = np.hstack([Z, X])
assert X.shape == (202, 286120)
n, p = X.shape
np.save(os.path.join(OUTPUT, "X.npy"), X)
fh = open(os.path.join(OUTPUT, "X.npy").replace("npy", "txt"), "w")
fh.write('shape = (%i, %i): Intercept + Age + Gender + %i voxels' % \
    (n, p, mask_bool.sum()))
fh.close()

# Xcs
X = Xtot[:, mask_bool.ravel()]
X = np.hstack([Z[:, 1:], X])
assert X.shape == (202, 286119)
X -= X.mean(axis=0)
X /= X.std(axis=0)
n, p = X.shape
np.save(os.path.join(OUTPUT_CS, "X.npy"), X)
fh = open(os.path.join(OUTPUT_CS, "X.npy").replace("npy", "txt"), "w")
fh.write('Centered and scaled data. Shape = (%i, %i): Age + Gender + %i voxels' % \
    (n, p, mask.sum()))
fh.close()

# Xcsnc
X = Xtot[:, mask_bool.ravel()]
#X = np.hstack([Z[:, 1:], X])
assert X.shape == (202, 286117)
X -= X.mean(axis=0)
X /= X.std(axis=0)
n, p = X.shape
np.save(os.path.join(OUTPUT_S, "X.npy"), X)
fh = open(os.path.join(OUTPUT_S, "X.npy").replace("npy", "txt"), "w")
fh.write('Centered and scaled data. Shape = (%i, %i): %i voxels' % \
    (n, p, mask.sum()))
fh.close()

# Xcsi
X = Xtot[:, mask_bool.ravel()]
X = np.hstack([Z, X])
assert X.shape == (202, 286120)
X -= X.mean(axis=0)
X /= X.std(axis=0)
X[:, 0] = 1.
n, p = X.shape
np.save(os.path.join(OUTPUT_CSI, "X.npy"), X)
fh = open(os.path.join(OUTPUT_CSI, "X.npy").replace("npy", "txt"), "w")
fh.write('Centered and scaled data. Shape = (%i, %i): Intercept + Age + Gender + %i voxels' % \
    (n, p, mask.sum()))
fh.close()

shutil.copyfile(os.path.join(OUTPUT, "mask.nii"), os.path.join(OUTPUT_CSI, "mask.nii"))

# atlas
X = Xtot[:, (mask_atlas.ravel() != 0)]
X = np.hstack([Z, X])
assert X.shape == (202, 286120)
n, p = X.shape
np.save(os.path.join(OUTPUT_ATLAS, "X.npy"), X)
fh = open(os.path.join(OUTPUT_ATLAS, "X.npy").replace("npy", "txt"), "w")
fh.write('shape = (%i, %i): Intercept + Age + Gender + %i voxels' % \
    (n, p, (mask_atlas.ravel() != 0).sum()))
fh.close()

# atlas cs
X = Xtot[:, (mask_atlas.ravel() != 0)]
X = np.hstack([Z[:, 1:], X])
assert X.shape == (202, 286119)
X -= X.mean(axis=0)
X /= X.std(axis=0)
n, p = X.shape
np.save(os.path.join(OUTPUT_CS_ATLAS, "X.npy"), X)
fh = open(os.path.join(OUTPUT_CS_ATLAS, "X.npy").replace("npy", "txt"), "w")
fh.write('Centered and scaled data. Shape = (%i, %i): Age + Gender + %i voxels' % \
    (n, p, (mask_atlas.ravel() != 0).sum()))
fh.close()

###############################################################################
## Spatial smoothing
# smoothing FWHM = 6mm
# Is DARTEL-based voxel-based morphometry affected by width of smoothing kernel and group size? A study using simulated atrophy.
# A smoothing kernel of 6 mm achieved the highest atrophy detection accuracy for groups with 50 participants
# J Ashburner presentation Age Prediction – Model Log Likelihoods
# Full width at half maximum FWHM and sigma:
#FWHM ~ 2.355 * sigma
voxel_size = 1.5
FWHM = 6. / voxel_size
sigma = FWHM / 2.355

import scipy.ndimage as ndimage
# check
ims = ndimage.gaussian_filter(images[0].reshape(shape), sigma=sigma).ravel()
nibabel.Nifti1Image(images[0].reshape(shape), affine=babel_image.get_affine()).to_filename("/tmp/ima.nii.gz")
nibabel.Nifti1Image(ims.reshape(shape), affine=babel_image.get_affine()).to_filename("/tmp/ima_s.nii.gz")

images_s = [ndimage.gaussian_filter(im.reshape(shape), sigma=sigma).ravel() for im in images]

Xtot = np.vstack(images_s)

X = Xtot[:, mask_bool.ravel()]
#X = np.hstack([Z[:, 1:], X])
assert X.shape == (202, 286117)
X -= X.mean(axis=0)
X /= X.std(axis=0)
n, p = X.shape
np.save(os.path.join(OUTPUT_S_S, "X.npy"), X)
fh = open(os.path.join(OUTPUT_S_S, "X.npy").replace("npy", "txt"), "w")
fh.write('Sptially smoothed (FWHM=6mm), centered and scaled data. Shape = (%i, %i): %i voxels' % \
    (n, p, mask.sum()))
fh.close()

# CS_S
shutil.copyfile(os.path.join(OUTPUT, "mask.nii"), os.path.join(OUTPUT_CS_S, "mask.nii"))
im = nibabel.load(os.path.join(OUTPUT_CS_S, "mask.nii"))
mask = im.get_data()

X = np.load(os.path.join(OUTPUT_S_S, "X.npy"))
assert X.shape == (202, 286117)
assert X.shape[1] == mask.sum()
X = np.hstack([Z[:, 1:], X])
assert X.shape == (202, 286119) == (202, mask.sum() + Z.shape[1] - 1)
np.save(os.path.join(OUTPUT_CS_S, "X.npy"), X)
fh = open(os.path.join(OUTPUT_CS_S, "X.npy").replace("npy", "txt"), "w")
fh.write('Sptially smoothed (FWHM=6mm), centered and scaled data. Shape = (%i, %i): Age + Gender +  %i voxels' % \
    (X.shape[0],  mask.sum() + Z.shape[1] - 1, mask.sum()))
fh.close()


np.save(os.path.join(OUTPUT, "y.npy"), y)
np.save(os.path.join(OUTPUT_CS, "y.npy"), y)
np.save(os.path.join(OUTPUT_S, "y.npy"), y)
np.save(os.path.join(OUTPUT_S_S, "y.npy"), y)
np.save(os.path.join(OUTPUT_CS_S, "y.npy"), y)
np.save(os.path.join(OUTPUT_CSI, "y.npy"), y)
np.save(os.path.join(OUTPUT_ATLAS, "y.npy"), y)
np.save(os.path.join(OUTPUT_CS_ATLAS, "y.npy"), y)


#############################################################################
# MULM
mask_ima = nibabel.load(os.path.join(OUTPUT_CSI, "mask.nii"))
mask_arr = mask_ima.get_data() != 0
X = np.load(os.path.join(OUTPUT_CSI, "X.npy"))
y = np.load(os.path.join(OUTPUT_CSI, "y.npy"))
Z = X[:, :3]
Y = X[: , 3:]
assert np.sum(mask_arr) == Y.shape[1]


DesignMat = np.zeros((Z.shape[0], Z.shape[1]+1)) # y, intercept, age, sex
DesignMat[:, 0] = (y.ravel() - y.ravel().mean())  # y
DesignMat[:, 1] = 1  # intercept
DesignMat[:, 2] = Z[:, 1]  # age
DesignMat[:, 3] = Z[:, 2]  # sex

from mulm import MUOLSStatsCoefficients
muols = MUOLSStatsCoefficients()
muols.fit(X=DesignMat, Y=Y)

tvals, pvals, dfs = muols.stats_t_coefficients(X=DesignMat, Y=Y, contrast=[-1, 0, 0, 0], pval=True)
arr = np.zeros(mask_arr.shape); arr[mask_arr] = tvals
out_im = nibabel.Nifti1Image(arr, affine=mask_ima.get_affine())
out_im.to_filename(os.path.join(OUTPUT_CSI, "t_stat_CTL-MCIc.nii.gz"))

thres = .1
m1 = pvals <= thres
m2 = (pvals > thres) & (pvals < (1. - thres))
m3 = pvals >= (1. - thres)
print np.sum(m1), np.sum(m2), np.sum(m3)
arr = np.zeros(mask_arr.shape)
val = np.zeros(pvals.shape, dtype=int)
val[m1] = 1.
val[m2] = 2.
val[m3] = 3.
arr[mask_arr] = val
arr = brainomics.image_atlas.smooth_labels(arr, size=(3, 3, 3))
out_im = nibabel.Nifti1Image(arr, affine=mask_ima.get_affine())
out_im.to_filename(os.path.join(OUTPUT_CSI, "pval-quantile_CTL-MCIc.nii.gz"))

arr = np.zeros(mask_arr.shape); arr[mask_arr] = pvals
out_im = nibabel.Nifti1Image(arr, affine=mask_ima.get_affine())
out_im.to_filename(os.path.join(OUTPUT_CSI, "pval_CTL-MCIc.nii.gz"))

tvals, pvals, dfs = muols.stats_t_coefficients(X=DesignMat, Y=Y, contrast=[1, 0, 0, 0], pval=True)
arr = np.zeros(mask_arr.shape); arr[mask_arr] = tvals
out_im = nibabel.Nifti1Image(arr, affine=mask_ima.get_affine())
out_im.to_filename(os.path.join(OUTPUT_CSI, "t_stat_MCIc-CTL.nii.gz"))

arr = np.zeros(mask_arr.shape); arr[mask_arr] = pvals
out_im = nibabel.Nifti1Image(arr, affine=mask_ima.get_affine())
out_im.to_filename(os.path.join(OUTPUT_CSI, "pval_MCIc-CTL.nii.gz"))

# anatomist /neurospin/brainomics/2013_adni/MCIc-CTL_csi/*.nii.gz
