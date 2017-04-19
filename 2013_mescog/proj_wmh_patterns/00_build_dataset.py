# -*- coding: utf-8 -*-
"""
Create WMH dataset:
 - find subjects for which we have the clinic data and the WMH images
 - read MNI-normalized images
 - due to the large size, we extract only voxels where there is at least a WMH
   (subset of the MNI mask)

INPUT:
 - images: /neurospin/mescog/neuroimaging/original/munich/
 - clinic: /neurospin/mescog/proj_predict_cog_decline/data/dataset_clinic_niglob_20140205_nomissing_BPF-LLV_imputed.csv

OUTPUT_DIR = /neurospin/mescog/proj_wmh_patterns
 - population.csv: clinical status of subjects
 - X.npy: concatenation of masked images
 - mask_atlas.nii: mask of WMH (with regions)
 - mask_bin.nii: binary mask of WMH
 - X_center.npy: centered data
 - means.npy: means used to center

"""
import os
import os.path
import glob

import numpy as np

import sklearn.preprocessing

import pandas as pd
import nibabel as nib
import parsimony.functions.nesterov.tv

import brainomics.image_atlas

INPUT_DIR = "/neurospin/mescog/neuroimaging/original/munich/"
INPUT_CLINIC = "/neurospin/mescog/proj_wmh_patterns/clinic/dataset_clinic_niglob_20140728_nomissing_BPF-LLV.csv"
#RESOURCES_DIR = "/neurospin/mescog/neuroimaging_ressources/"

#INPUT_CSV = "/neurospin/mescog/proj_predict_cog_decline/data/dataset_clinic_niglob_20140205_nomissing_BPF-LLV_imputed.csv"
#INPUT_MASK = os.path.join(RESOURCES_DIR,
#                          "MNI152_T1_1mm_brain_mask.nii.gz")

OUTPUT_DIR = "/neurospin/mescog/proj_wmh_patterns"
#INPUT_CSV = os.path.join(OUTPUT_DIR, "dataset_clinic_niglob_20140205_nomissing_BPF-LLV_imputed.csv")
#INPUT_CLINIC = os.path.join(INPUT_BASE_CLINIC, "dataset_clinic_niglob_20140728.csv")

INPUT_MASK = os.path.join(OUTPUT_DIR,
                          "MNI152_T1_1mm_brain_mask.nii.gz")

if not(os.path.exists(OUTPUT_DIR)):
    os.makedirs(OUTPUT_DIR)

OUTPUT_POP = os.path.join(OUTPUT_DIR, "population.csv")
OUTPUT_IMPLICIT_MASK = os.path.join(OUTPUT_DIR, "mask_implicit.nii.gz")
OUTPUT_ATLAS_MASK = os.path.join(OUTPUT_DIR, "mask_atlas.nii.gz")
OUTPUT_BIN_MASK = os.path.join(OUTPUT_DIR, "mask_bin.nii.gz")
OUTPUT_A_BIN_MASK = os.path.join(OUTPUT_DIR, "mask_bin_A.npz")
OUTPUT_X = os.path.join(OUTPUT_DIR, "X.npy")
OUTPUT_CENTERED_X = os.path.join(OUTPUT_DIR, "X_center.npy")
OUTPUT_MEANS = os.path.join(OUTPUT_DIR, "means.npy")

##############
# Parameters #
##############

IM_SHAPE = (182, 218, 182)

#######################################
# Find subjects and create population #
#######################################

# Open clinic file and extract subjects ID
clinic_data = pd.io.parsers.read_csv(INPUT_CLINIC, index_col=0)
clinic_subjects_id = [int(subject_id[4:]) for subject_id in clinic_data.index]
clinic_data.index = clinic_subjects_id
assert len(clinic_subjects_id) == 319
          
# Find images and extract subjects ID
images_path = glob.glob(os.path.join(INPUT_DIR,
                                     "CAD_norm_M0",
                                     "WMH_norm",
                                     "*M0-WMH_norm.nii.gz"))
assert len(images_path) == 343

# Check images
print("Checking images")
trm = None
for file_path in images_path[:]:
    im = nib.load(file_path)
    if trm is None:
        trm = im.get_affine()
    if not np.all(trm == im.get_affine()):
        print(("{f} has wrong transformation".format(f=file_path)))
        images_path.remove(file_path)
    if im.get_data().shape != IM_SHAPE:
        print(("{f} has wrong dimension".format(f=file_path)))
        images_path.remove(file_path)
    if np.all(im.get_data() == 0):
        print(("{f} is empty".format(f=file_path)))
        images_path.remove(file_path)
print("Found %i correct images" % len(images_path))

assert len(images_path) == 342

images_subject_id = [int(os.path.basename(p)[0:4]) for p in images_path]
images = pd.DataFrame(data=images_path, index=images_subject_id)
images.columns=['IMAGE']

# Create population file (merge of both subjects lists)
pop = pd.merge(clinic_data, images,
               right_index=True, left_index=True)
pop.index.name = 'Subject ID'
pop = pop.sort_index()
#subjects_id = pop.index
print("Found", pop.shape[0], "correct subjects")
assert pop.shape == (301, 25)
pop.to_csv(OUTPUT_POP)


#################################################
# Read images, create masks and extract dataset #
#################################################

# Open MNI mask
babel_mni_mask = nib.load(INPUT_MASK)
mni_mask = babel_mni_mask.get_data() != 0
n_voxels_in_mni_mask = np.count_nonzero(mni_mask)
print("MNI brain mask: {n} voxels".format(n=n_voxels_in_mni_mask))
assert n_voxels_in_mni_mask == 1827243 # ~ 2M of voxels

# Read images and concatenate them
# To save some memory we extract only images in the MNI brain mask
print("Reading images")
n = pop.shape[0]
p = n_voxels_in_mni_mask
X = np.zeros((n, p))

for i, (idx, row) in enumerate(pop.iterrows()):
     print(i, idx)
     X[i] = nib.load(row["IMAGE"]).get_data()[mni_mask]


ref_imag_path = pop.iloc[0, 24]

# Compute the mask of WMH
print("Computation of masks")
flat_implicit_mask = np.any(X != 0, axis=0)
assert flat_implicit_mask.sum() == 1129198
flat_implicit_mask_index = np.where(flat_implicit_mask)[0]
n_features = flat_implicit_mask_index.shape[0]
print("Found {n} voxels with a WMH".format(n=n_features))
implicit_mask = np.zeros(mni_mask.shape, dtype=bool)
implicit_mask[mni_mask] = flat_implicit_mask
implicit_mask_babel = nib.Nifti1Image(implicit_mask.astype(np.uint8),
                                      babel_mni_mask.get_affine(),
                                      header=babel_mni_mask.get_header())
nib.save(implicit_mask_babel, OUTPUT_IMPLICIT_MASK)

# Compute atlas mask
babel_mask_atlas = brainomics.image_atlas.resample_atlas_harvard_oxford(
    ref=ref_imag_path,
    output=OUTPUT_ATLAS_MASK)

mask_atlas = babel_mask_atlas.get_data()
mask_atlas[~implicit_mask] = 0  # apply implicit mask
# Smooth it
mask_atlas = brainomics.image_atlas.smooth_labels(mask_atlas, size=(3, 3, 3))
out_im = nib.Nifti1Image(mask_atlas,
                         affine=babel_mni_mask.get_affine())
out_im.to_filename(OUTPUT_ATLAS_MASK)

# Binarized mask
bin_mask = mask_atlas != 0
assert bin_mask.sum() == 1064455
bin_mask_index = np.where(bin_mask)[0]
n_voxels_in_bin_mask = np.count_nonzero(bin_mask)
print("Extracting {n} voxels".format(n=n_voxels_in_bin_mask))
out_im = nib.Nifti1Image(bin_mask.astype(np.uint8),
                         affine=babel_mni_mask.get_affine())
out_im.to_filename(OUTPUT_BIN_MASK)

############################################
# Save the Linear operator with lambda max #
############################################

A = parsimony.functions.nesterov.tv.linear_operator_from_mask(bin_mask, calc_lambda_max=True)
# Wall time: 3min 1s

A.save(OUTPUT_A_BIN_MASK)

A_ = parsimony.functions.nesterov.tv.LinearOperatorNesterov(filename=OUTPUT_A_BIN_MASK)
assert np.all(np.array([(A[i] - A_[i]).nnz for i in range(len(A))]) == 0)
assert np.allclose(A.get_singular_values(0), 11.9905, rtol=1e-03, atol=1e-03)

#################
# Save the data #
#################

# Extract images with bin_mask
del X
n = pop.shape[0]
p = n_voxels_in_bin_mask
X = np.zeros((n, p))

for i, (idx, row) in enumerate(pop.iterrows()):
     print(i, idx)
     X[i] = nib.load(row["IMAGE"]).get_data()[bin_mask]


np.save(OUTPUT_X, X)

# QC on first image
im_idx = 0
im_data = np.zeros(IM_SHAPE)
im_data[bin_mask] = X[0, :]
out_im = nib.Nifti1Image(im_data,
                         affine=babel_mni_mask.get_affine())
out_im.to_filename(OUTPUT_DIR+"/%d-QC.nii" % pop.index[0])

###############
# Center data #
###############

print("Centering data")
scaler = sklearn.preprocessing.StandardScaler(with_std=False)
X_center = scaler.fit_transform(X)
np.save(OUTPUT_CENTERED_X, X_center)
np.save(OUTPUT_MEANS, scaler.mean_)
