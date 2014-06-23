# -*- coding: utf-8 -*-
"""
Create WMH dataset:
 - images are normalized into the MNI
 - due to the large size, we extract only voxels in the
   MNI brain mask

INPUT_DIR = "/neurospin/mescog/neuroimaging/processed"
OUTPUT_DIR = "/neurospin/mescog/datasets"
CAD-WMH-MNI.npy

"""
import os
import os.path
import glob
import nibabel as nib
import numpy as np

INPUT_DIR = "/neurospin/mescog/neuroimaging/original/munich/"
RESOURCES_DIR = "/neurospin/mescog/neuroimaging_ressources/"

INPUT_MASK = os.path.join(RESOURCES_DIR,
                          "MNI152_T1_1mm_brain_mask.nii.gz")

OUTPUT_DIR = "/neurospin/mescog/datasets"
OUTPUT_X = os.path.join(OUTPUT_DIR, "CAD-WMH-MNI.npy")
OUTPUT_subjects = os.path.join(OUTPUT_DIR, "CAD-WMH-MNI-subjects.txt")

##############
# Parameters #
##############

IM_SHAPE = (182, 218, 182)

#################
# Actual script #
#################

# Open MNI mask
babel_mni_mask = nib.load(INPUT_MASK)
mni_mask = babel_mni_mask.get_data() != 0
n_voxels_in_mask = np.count_nonzero(mni_mask)
print "MNI mask: {n} voxels".format(n=n_voxels_in_mask)

# Get path and sorted ID
subject_paths = glob.glob(os.path.join(INPUT_DIR,
                                       "CAD_norm_M0",
                                       "WMH_norm",
                                       "*M0-WMH_norm.nii.gz"))
print "Found %i subjects" % len(subject_paths)
#Found 343 subjects
subject_list = [int(os.path.basename(p)[0:4]) for p in subject_paths]
subject_list.sort()

# Read images & save them
n = len(subject_paths)
p = n_voxels_in_mask
X = np.zeros((n, p))
trm = None
for i, file_path in enumerate(subject_paths):
    im = nib.load(file_path)
    if trm is None:
        trm = im.get_affine()
    if not np.all(trm == im.get_affine()):
        raise ValueError("Volume has wrong transformation")
    if im.get_data().shape != IM_SHAPE:
        raise ValueError("Volume has wrong dimension")
    X[i] = im.get_data()[mni_mask].ravel()
np.save(OUTPUT_X, X)

# Save subjects
fo = open(OUTPUT_subjects, "w")
subject_list_newline = [str(subject) + "\n" for subject in subject_list]
fo.writelines(subject_list_newline)
fo.close()

### QC on first image
#image = nib.load(OUTPUT_DIR+"/MNI152_T1_1mm_brain_mask.nii.gz")
#out_im = nib.Nifti1Image(np.reshape(X[0,:], image.get_data().shape), affine=image.get_affine())
#out_im.to_filename(OUTPUT_DIR+"/QC_CAD-WMH-MNI_%s-M0-WMH-MNI.nii.gz" % subject_list_newline[0].strip())
