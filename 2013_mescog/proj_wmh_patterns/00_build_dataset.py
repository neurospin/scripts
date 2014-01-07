# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 17:08:06 2013

@author: ed203246
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
np.save(OUTPUT_X, X)
fo = open(OUTPUT_subjects, "w")
subject_list_newline = [subject + "\n" for subject in subject_list]
fo.writelines(subject_list_newline)
fo.close()
