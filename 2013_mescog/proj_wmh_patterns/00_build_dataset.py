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
 - CAD-WMH-MNI-subjects.txt: ID of subjects
 - population.csv: clinical status of subjects
 - wmh_mask.nii: mask of WMH
 - CAD-WMH-MNI.npy: concatenation of masked images

"""
import os
import os.path
import glob
import nibabel as nib
import numpy as np
import pandas as pd

INPUT_DIR = "/neurospin/mescog/neuroimaging/original/munich/"
RESOURCES_DIR = "/neurospin/mescog/neuroimaging_ressources/"

INPUT_CSV = "/neurospin/mescog/proj_predict_cog_decline/data/dataset_clinic_niglob_20140205_nomissing_BPF-LLV_imputed.csv"
INPUT_MASK = os.path.join(RESOURCES_DIR,
                          "MNI152_T1_1mm_brain_mask.nii.gz")

OUTPUT_DIR = "/neurospin/mescog/proj_wmh_patterns"
if not(os.path.exists(OUTPUT_DIR)):
    os.makedirs(OUTPUT_DIR)
OUTPUT_X = os.path.join(OUTPUT_DIR, "CAD-WMH-MNI.npy")
OUTPUT_SUBJECTS = os.path.join(OUTPUT_DIR, "CAD-WMH-MNI-subjects.txt")
OUTPUT_CLINIC = os.path.join(OUTPUT_DIR, "population.csv")
OUTPUT_WMH_MASK = os.path.join(OUTPUT_DIR, "wmh_mask.nii")
OUTPUT_X = os.path.join(OUTPUT_DIR, "CAD-WMH-MNI.npy")

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

# Find images and subject list
images_path = glob.glob(os.path.join(INPUT_DIR,
                                     "CAD_norm_M0",
                                     "WMH_norm",
                                     "*M0-WMH_norm.nii.gz"))
print "Found %i images" % len(images_path)
#Found 343 subjects
wmh_subjects_id = [int(os.path.basename(p)[0:4]) for p in images_path]
wmh_subjects_id.sort()

# Intersection of both subjects lists
subjects_id = np.intersect1d(wmh_subjects_id, clinic_subjects_id)
subjects_id.sort()
#lines_to_keep = np.where(np.in1d(wmh_subjects_id, subjects_id))[0]
print "Found", len(subjects_id), "correct subjects"

# Save subjects
with open(OUTPUT_SUBJECTS, "w") as fo:
    subject_list_newline = [str(subject) + "\n" for subject in subjects_id]
    fo.writelines(subject_list_newline)

# Save population file
pop = clinic_data.loc[subjects_id]
pop.to_csv(OUTPUT_CLINIC)

###################################
# Read images and create WMH mask #
###################################

# Open MNI mask
babel_mni_mask = nib.load(INPUT_MASK)
mni_mask = babel_mni_mask.get_data() != 0
n_voxels_in_mni_mask = np.count_nonzero(mni_mask)
print "MNI mask: {n} voxels".format(n=n_voxels_in_mni_mask)

# Read images and concatenate them
n = len(subjects_id)
p = n_voxels_in_mni_mask
X = np.zeros((n, p))
trm = None
for i, ID in enumerate(subjects_id):
    file_path = os.path.join(INPUT_DIR,
                             "CAD_norm_M0",
                             "WMH_norm",
                             "{id}-M0-WMH_norm.nii.gz".format(id=ID))
    im = nib.load(file_path)
    if trm is None:
        trm = im.get_affine()
    if not np.all(trm == im.get_affine()):
        raise ValueError("Volume has wrong transformation")
    if im.get_data().shape != IM_SHAPE:
        raise ValueError("Volume has wrong dimension")
    X[i] = im.get_data()[mni_mask].ravel()

# Compute the WMH mask
features_mask = np.any(X != 0, axis=0)
mask_index = np.where(features_mask)[0]
n_features = mask_index.shape[0]
print "Found {n} features".format(n=n_features)
wmh_mask = np.zeros(mni_mask.shape, dtype=bool)
wmh_mask[mni_mask] = features_mask
wmh_mask_babel = nib.Nifti1Image(wmh_mask.astype(np.uint8),
                                 babel_mni_mask.get_affine(),
                                 header=babel_mni_mask.get_header())
nib.save(wmh_mask_babel, OUTPUT_WMH_MASK)

# Extract images
X = X[:, mask_index]
np.save(OUTPUT_X, X)

# QC on first image
im_idx = 0
im_id = subjects_id[im_idx]
im_data = np.zeros(IM_SHAPE)
im_data[wmh_mask] = X[im_idx, :]
out_im = nib.Nifti1Image(im_data,
                         affine=babel_mni_mask.get_affine())
out_im.to_filename(OUTPUT_DIR+"/%d-QC.nii.gz" % im_id)
