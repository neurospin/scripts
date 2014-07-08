# -*- coding: utf-8 -*-
"""
Create WMH dataset:
 - find subjects for which we have the clinic data and the WMH images
 - read MNI-normalized images
 - due to the large size, we extract only voxels where there is at least a WMH
   (subset of the MNI mask)

INPUT:
 - images: /neurospin/mescog/neuroimaging/original/munich/
 - clinic: /neurospin/mescog/proj_predict_cog_decline/data/dataset_clinic_niglob_20140205_nomissing_BPF-LLV_imputed.csv"

OUTPUT_DIR = /neurospin/mescog/proj_wmh_patterns
 - population.csv: clinical status of subjects
 - X.npy: concatenation of masked images
 - mask_atlas.nii: mask of WMH (with regions)
 - mask_bin.nii: binary mask of WMH
 - X_center.npy: centerd data
 - means.npy: means used to center

"""
import os
import os.path
import glob

import numpy as np

import sklearn.preprocessing

import pandas as pd
import nibabel as nib

import brainomics.image_atlas

INPUT_DIR = "/neurospin/mescog/neuroimaging/original/munich/"
RESOURCES_DIR = "/neurospin/mescog/neuroimaging_ressources/"

INPUT_CSV = "/neurospin/mescog/proj_predict_cog_decline/data/dataset_clinic_niglob_20140205_nomissing_BPF-LLV_imputed.csv"
INPUT_MASK = os.path.join(RESOURCES_DIR,
                          "MNI152_T1_1mm_brain_mask.nii.gz")

OUTPUT_DIR = "/neurospin/mescog/proj_wmh_patterns"
if not(os.path.exists(OUTPUT_DIR)):
    os.makedirs(OUTPUT_DIR)
OUTPUT_CLINIC = os.path.join(OUTPUT_DIR, "population.csv")
OUTPUT_IMPLICIT_MASK = os.path.join(OUTPUT_DIR, "mask_implicit.nii")
OUTPUT_ATLAS_MASK = os.path.join(OUTPUT_DIR, "mask_atlas.nii")
OUTPUT_BIN_MASK = os.path.join(OUTPUT_DIR, "mask_bin.nii")
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
clinic_data = pd.io.parsers.read_csv(INPUT_CSV, index_col=0)
clinic_subjects_id = [int(subject_id[4:]) for subject_id in clinic_data.index]
clinic_data.index = clinic_subjects_id
print "Found", len(clinic_subjects_id), "clinic records"

# Find images and extract subjects ID
images_path = glob.glob(os.path.join(INPUT_DIR,
                                     "CAD_norm_M0",
                                     "WMH_norm",
                                     "*M0-WMH_norm.nii.gz"))
print "Found %i images" % len(images_path)
#Found 343 subjects
# Check images
print "Checking images"
trm = None
for file_path in images_path[:]:
    im = nib.load(file_path)
    if trm is None:
        trm = im.get_affine()
    if not np.all(trm == im.get_affine()):
        print "{f} has wrong transformation".format(f=file_path)
        images_path.remove(file_path)
    if im.get_data().shape != IM_SHAPE:
        print "{f} has wrong dimension".format(f=file_path)
        images_path.remove(file_path)
    if ~np.any(im.get_data()):
        print "{f} is empty".format(f=file_path)
        images_path.remove(file_path)
print "Found %i correct images" % len(images_path)
# Sorting is not necessary since we merge but doesn't hurt
images_subject_id = [int(os.path.basename(p)[0:4]) for p in images_path]
images_subject_id.sort()
images = pd.DataFrame(data=images_path, index=images_subject_id)
images.columns=['IMAGE']

# Merge the two subjects list
subjects_id = np.intersect1d(images_subject_id, clinic_subjects_id)
subjects_id.sort()
#lines_to_keep = np.where(np.in1d(wmh_subjects_id, subjects_id))[0]
print "Found", len(subjects_id), "correct subjects"

# Create population file (merge of both subjects lists)
pop = pd.merge(clinic_data, images,
               right_index=True, left_index=True,
               sort=True)
pop.index.name = 'Subject ID'
pop.to_csv(OUTPUT_CLINIC)

#################################################
# Read images, create masks and extract dataset #
#################################################

# Open MNI mask
babel_mni_mask = nib.load(INPUT_MASK)
mni_mask = babel_mni_mask.get_data() != 0
n_voxels_in_mni_mask = np.count_nonzero(mni_mask)
print "MNI brain mask: {n} voxels".format(n=n_voxels_in_mni_mask)

# Read images and concatenate them
# To save some memory we extract only images in the MNI brain mask
print "Reading images"
n = len(subjects_id)
p = n_voxels_in_mni_mask
X = np.zeros((n, p))
files = []
for i, ID in enumerate(subjects_id):
    file_path = os.path.join(INPUT_DIR,
                             "CAD_norm_M0",
                             "WMH_norm",
                             "{_id}-M0-WMH_norm.nii.gz".format(_id=ID))
    files.append(file_path)
    im = nib.load(file_path)
    X[i] = im.get_data()[mni_mask]

# Compute the mask of WMH
print "Computation of masks"
flat_implicit_mask = np.any(X != 0, axis=0)
flat_implicit_mask_index = np.where(flat_implicit_mask)[0]
n_features = flat_implicit_mask_index.shape[0]
print "Found {n} voxels with a WMH".format(n=n_features)
implicit_mask = np.zeros(mni_mask.shape, dtype=bool)
implicit_mask[mni_mask] = flat_implicit_mask
implicit_mask_babel = nib.Nifti1Image(implicit_mask.astype(np.uint8),
                                      babel_mni_mask.get_affine(),
                                      header=babel_mni_mask.get_header())
nib.save(implicit_mask_babel, OUTPUT_IMPLICIT_MASK)

# Compute atlas mask
babel_mask_atlas = brainomics.image_atlas.resample_atlas_harvard_oxford(
    ref=files[0],
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
bin_mask_index = np.where(bin_mask)[0]
n_voxels_in_bin_mask = np.count_nonzero(bin_mask)
print "Extracting {n} voxels".format(n=n_voxels_in_bin_mask)
out_im = nib.Nifti1Image(bin_mask.astype(np.uint8),
                         affine=babel_mni_mask.get_affine())
out_im.to_filename(OUTPUT_BIN_MASK)

# Extract images with bin_mask
# Here the mask is a 3D mask (not included in the MNI since we have dilated it)
# so we cannot use columns of X to extract data (since they are the MNI)
# Hence we reload the images with bin_mask
del X
n = len(subjects_id)
p = n_voxels_in_bin_mask
X = np.zeros((n, p))
for i, file_path in enumerate(files):
    im = nib.load(file_path)
    X[i] = im.get_data()[bin_mask]
np.save(OUTPUT_X, X)

# QC on first image
im_idx = 0
im_id = subjects_id[im_idx]
im_data = np.zeros(IM_SHAPE)
im_data[bin_mask] = X[im_idx, :]
out_im = nib.Nifti1Image(im_data,
                         affine=babel_mni_mask.get_affine())
out_im.to_filename(OUTPUT_DIR+"/%d-QC.nii" % im_id)

###############
# Center data #
###############

print "Centering data"
scaler = sklearn.preprocessing.StandardScaler(with_std=False)
X_center = scaler.fit_transform(X)
np.save(OUTPUT_CENTERED_X, X_center)
np.save(OUTPUT_MEANS, scaler.mean_)
