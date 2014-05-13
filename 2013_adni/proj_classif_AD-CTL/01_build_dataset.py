# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 16:02:32 2014

@author: md238665

Read the non-smoothed images for all the subjects, mask them and dump them.
Similarly read group and dump it.

Data are then centered. The mean is computed on the whole dataset.

"""
#Mask stem from
# proj_classif_AD-CTL_v2014-04/SPM/template_FinalQC_CTL_AD/mask.nii
import os
import numpy as np
import glob
import pandas as pd
import nibabel
#import proj_classif_config
GROUP_MAP = {'CTL': 0, 'AD': 1}
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

INPUT_CSV = os.path.join(BASE_PATH, "proj_classif_AD-CTL", "population.csv")
INPUT_MASK = os.path.join(BASE_PATH, "proj_classif_AD-CTL", "mask.nii")
OUTPUT_X = os.path.join(BASE_PATH, "proj_classif_AD-CTL", "X.npy")
OUTPUT_y = os.path.join(BASE_PATH, "proj_classif_AD-CTL", "y.npy")

# Read input subjects
input_subjects = pd.read_table(INPUT_SUBJECTS_LIST_FILENAME, sep=" ",
                               header=None)
input_subjects = [x[:10] for x in input_subjects[1]]

# Read pop csv
pop = pd.read_csv(INPUT_CSV)
pop['PTGENDER.num'] = pop["PTGENDER"].map(GENDER_MAP)

n = len(pop)
# Open mask
babel_mask = nibabel.load(INPUT_MASK)
mask = babel_mask.get_data() != 0
p = np.count_nonzero(mask)
print "Mask: {n} voxels".format(n=p)
#Mask: 311341 voxels

X = np.zeros((n, 3 + p)) # Intercept + Age + Gender + 311341 voxels
X[:, 0] = 1 # Intercept
y = np.zeros((n, 1)) # DX

for i, PTID in enumerate(pop['PTID']):
    #i, PTID = 0, '011_S_0002'
    #bv_group = m18_clinic_qc['Group.BV'].loc[PTID]
    #adni_group = m18_clinic_qc['Group.ADNI'].loc[PTID]
    cur = pop[pop.PTID == PTID]
    print cur
    imagefile_pattern = INPUT_IMAGEFILE_FORMAT.format(PTID=PTID)
    #print imagefile_pattern
    imagefile_name = glob.glob(imagefile_pattern)
    if len(imagefile_name) != 1:
        raise ValueError("Found %i files" % len(imagefile_name))
    imagefile_name = imagefile_name[0]
    babel_image = nibabel.load(imagefile_name)
    image_data = babel_image.get_data()
    # Apply mask (returns a flat image)
    X[i, 1:3] = np.asarray(cur[["AGE", "PTGENDER.num"]]).ravel()
    X[i, 3:]  = image_data[mask]
    y[i, 0] = cur["DX.bl.num"]



np.save(OUTPUT_X, X)
fh = open(OUTPUT_X.replace("npy", "txt"), "w")
fh.write("shape = (270, 311344): Intercept + Age + Gender + 311341 voxels")
fh.close()

np.save(OUTPUT_y, y)
