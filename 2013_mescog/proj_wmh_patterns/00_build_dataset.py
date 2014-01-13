# -*- coding: utf-8 -*-
"""
Create WMH dataset images are normalized into the MNI

INPUT_DIR = "/neurospin/mescog/neuroimaging/processed"
OUTPUT_DIR = "/neurospin/mescog/datasets"
CAD-WMH-MNI.npy

"""
import os
import os.path
import glob
import nibabel as nib
import numpy as np


INPUT_DIR = "/neurospin/mescog/neuroimaging/processed"
OUTPUT_DIR = "/neurospin/mescog/datasets"
OUTPUT_X = os.path.join(OUTPUT_DIR, "CAD-WMH-MNI.npy")
OUTPUT_subjects = os.path.join(OUTPUT_DIR, "CAD-WMH-MNI-subjects.txt")

subject_paths = glob.glob(os.path.join(INPUT_DIR,
                                       "CAD_bioclinica_nifti",
                                       "*",
                                       "*M0-WMH-MNI.nii.gz"))
print "Found %i subjects" % len(subject_paths)
#Found 342 subjects
arr_list = list()
trm = None
subject_list = list()
for file_path in subject_paths:
    im = nib.load(file_path)
    if trm is None:
        trm = im.get_affine()
    if not np.all(trm == im.get_affine()):
        raise ValueError("Volume has wrong transformation")
    if im.get_data().shape != (91, 109, 91):
        raise ValueError("Volume has wrong dimension")
    arr_list.append(im.get_data().ravel())
    subject_list.append(os.path.basename(os.path.dirname(file_path)))

X = np.vstack(arr_list)

# binnarize everybody
X[X != 0] = 1.
np.save(OUTPUT_X, X)
fo = open(OUTPUT_subjects, "w")
subject_list_newline = [subject + "\n" for subject in subject_list]
fo.writelines(subject_list_newline)
fo.close()

## QC on first image
image = nib.load(OUTPUT_DIR+"/MNI152_T1_2mm_brain_mask.nii.gz")
out_im = nib.Nifti1Image(np.reshape(X[0,:], image.get_data().shape), affine=image.get_affine())
out_im.to_filename(OUTPUT_DIR+"/QC_CAD-WMH-MNI_%s-M0-WMH-MNI.nii.gz" % subject_list_newline[0].strip())
