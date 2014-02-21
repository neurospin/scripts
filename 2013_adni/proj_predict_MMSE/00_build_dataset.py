# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 16:02:32 2014

@author: md238665

Read the images, mask them and dump them.
Similarly read MMSE and dump it.

Data are then centered.

"""

# TODO: Images?
# TODO: Mask?

import os
import glob

import numpy as np
import pandas as pd

import sklearn.preprocessing as sklp

import nibabel

BASE_PATH = "/neurospin/brainomics/2013_adni"

CLINIC_PATH = os.path.join(BASE_PATH, "clinic")
INPUT_CLINIC_FILE = os.path.join(CLINIC_PATH, "adni510_m18_nonnull_groups.csv")
INPUT_GROUPS = ['MCIc', 'AD']

INPUT_ADNI510_CATI_PATH = "/neurospin/cati/ADNI/ADNI_510"
INPUT_QC_PATH = os.path.join(INPUT_ADNI510_CATI_PATH,
                             "qualityControlSPM", "QC")
INPUT_QC_GRADE = os.path.join(INPUT_QC_PATH, "final_grade.csv")
INPUT_QUALITY = ['A', 'B']

INPUT_TEMPLATE_PATH = os.path.join(BASE_PATH, 
                                   "templates",
                                   "template_FinalQC_MCIc-AD")
INPUT_IMAGE_PATH = os.path.join(INPUT_TEMPLATE_PATH,
                                "registered_images")
#INPUT_IMAGEFILE_FORMAT = os.path.join(INPUT_IMAGE_PATH,
#                                      "{group}",
#                                      "{PTID}_*", "t1mri", 
#                                      "default_acquisition",
#                                      "spm_new_s"
#                                      )
INPUT_IMAGEFILE_FORMAT = os.path.join(INPUT_IMAGE_PATH,
                                      "mw{PTID}*_Nat_dartel_greyProba.nii")

INPUT_MASK_PATH = os.path.join(BASE_PATH, 
                               "masks",
                               "template_FinalQC_MCIc-AD")
INPUT_MASK = os.path.join(INPUT_MASK_PATH,
                          "mask.img")

OUTPUT_PATH = "/neurospin/brainomics/2013_adni/proj_predict_MMSE"
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)
OUTPUT_X_FILE = os.path.join(OUTPUT_PATH, "X.npy")
OUTPUT_X_CENTER_FILE = os.path.join(OUTPUT_PATH, "X.center.npy")
OUTPUT_X_MEAN_FILE = os.path.join(OUTPUT_PATH, "X.mean.npy")
OUTPUT_Y_FILE = os.path.join(OUTPUT_PATH, "y.npy")
OUTPUT_Y_CENTER_FILE = os.path.join(OUTPUT_PATH, "y.center.npy")
OUTPUT_Y_MEAN_FILE = os.path.join(OUTPUT_PATH, "y.mean.npy")

# Read clinic data
m18_clinic = pd.read_csv(INPUT_CLINIC_FILE,
                         index_col=0)

# Read QC
qc = pd.read_csv(INPUT_QC_GRADE,
                 index_col=0)

# Merge
m18_clinic_qc = pd.merge(m18_clinic, qc, left_index=True, right_index=True)

# Subsample for grade A and B images
is_ab = m18_clinic_qc['Grade'].isin(INPUT_QUALITY)

# Subsample for interest groups
is_MCIcAD = m18_clinic_qc['Group.ADNI'].isin(INPUT_GROUPS)

# Subsampling
is_cool = is_ab & is_MCIcAD
pop = m18_clinic_qc[is_cool]
n = len(pop)
print "Found", n, " subjects"

# Open mask
babel_mask = nibabel.load(INPUT_MASK)
mask = babel_mask.get_data() != 0
p = np.count_nonzero(mask)
print "Mask: {n} voxels".format(n=p)

# Load images
X = np.zeros((n, p))
for i, PTID in enumerate(pop.index):
    #bv_group = m18_clinic_qc['Group.BV'].loc[PTID]
    #adni_group = m18_clinic_qc['Group.ADNI'].loc[PTID]
    #print "Subject", PTID, bv_group, adni_group
    imagefile_pattern = INPUT_IMAGEFILE_FORMAT.format(PTID=PTID)
    #print imagefile_pattern
    imagefile_name = glob.glob(imagefile_pattern)[0]
    babel_image = nibabel.load(imagefile_name)
    image = babel_image.get_data()
    # Apply mask (returns a flat image)
    X[i, :] = image[mask]

# Store X
np.save(OUTPUT_X_FILE, X)

# Store y
y = np.array(pop['MMSE'], dtype='float64')
np.save(OUTPUT_Y_FILE, y)

# Center X
x_scaler = sklp.StandardScaler(with_std=False)
X_center = x_scaler.fit_transform(X)
np.save(OUTPUT_X_CENTER_FILE, X_center)
np.save(OUTPUT_X_MEAN_FILE, x_scaler.mean_)

# Center y
y_scaler = sklp.StandardScaler(with_std=False)
y_center = y_scaler.fit_transform(y)
np.save(OUTPUT_Y_CENTER_FILE, y_center)
np.save(OUTPUT_Y_MEAN_FILE, y_scaler.mean_)
