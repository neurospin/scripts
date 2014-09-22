# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 17:01:26 2014

@author: md238665 and hl237680

Compute the mean of the images obtained after new_segment segmentation (wmc)
for the six tissue classes to build the new TPM.

This TPM is then normalized and saved under:
/neurospin/tmp/hl/TPM.nii

"""

import os

import numpy as np
import nibabel as nib

from glob import glob

# Directory of segmented images
INPUT_DIR = '/neurospin/tmp/hl/'
if not os.path.exists(INPUT_DIR):
    os.makedirs(INPUT_DIR)

# We need to have a * because the
INPUT_FORMAT = os.path.join(INPUT_DIR,
                            'mwc{tissue_class}{subject:012}*.nii')

INPUT_SUBJECTS_ID = [5352545,
                     31548817,
                     14900642,
                     17221560,
                     1298929,
                     18801650,
                     10974785,
                     36232609,
                     737577,
                     20401625,
                     57728231,
                     98958194,
                     34241299,
                     65159486,
                     13818625,
                     26470206,
                     18931943,
                     18952353,
                     6006399,
                     12945962,
                     1885390,
                     7347207,
                     3712195,
                     13646768,
                     14743165,
                     43102413,
                     7779851,
                     28827850,
                     45668095,
                     67659985,
                     19912904,
                     21496213,
                     7127936,
                     10111807,
                     215284,
                     297685,
                     4796887,
                     14141334,
                     2042617,
                     4254555,
                     6756801,
                     7824030,
                     9329324,
                     10951966,
                     17994216,
                     23079648,
                     6021063,
                     10813059,
                     3629479,
                     4622874,
                     3970752,
                     20451157,
                     4908925,
                     22858673,
                     9715842,
                     23896510,
                     240546,
                     1689886,
                     1038801,
                     6507225,
                     2758817,
                     8349556,
                     2996147,
                     4702312,
                     1380042,
                     2270310,
                     4941007,
                     7167925,
                     1383133,
                     3410870,
                     2109942,
                     2437251,
                     1123104,
                     4631192,
                     469693,
                     1441076,
                     106871,
                     2757585,
                     642263,
                     1338177,
                     3287674,
                     3328367,
                     75717,
                     1647364,
                     4285802,
                     4527509,
                     112288,
                     1617607,
                     540905,
                     829055,
                     4068298,
                     5019076,
                     613223,
                     1934375,
                     1023924,
                     3991592]

OUTPUT_DIR = INPUT_DIR
OUTPUT_TPM = os.path.join(OUTPUT_DIR,
                          'TPM.nii')

##############
# Parameters #
##############
TISSUE_CLASSES = range(1, 7)

IM_SHAPE = (121, 145, 121)

##########
# Script #
##########

# For TPM, SPM uses a 4D image (the last dimension is the class of tissue)
# For instance, see:
# '/i2bm/local/spm8-standalone-5236/spm8_mcr/spm8/toolbox/Seg/TPM.nii'
IMAGES = np.zeros((IM_SHAPE[0],
                   IM_SHAPE[1],
                   IM_SHAPE[2],
                   len(TISSUE_CLASSES),
                   len(INPUT_SUBJECTS_ID)))

# Load images
for i, tissue_class in enumerate(TISSUE_CLASSES):
    for j, subject in enumerate(INPUT_SUBJECTS_ID):
        im_format = INPUT_FORMAT.format(tissue_class=tissue_class,
                                        subject=subject)
        im_list = glob(im_format)
        assert(len(im_list) == 1)
        im = im_list[0]
        print "Loading", im
        im = nib.load(im)
        IMAGES[:, :, :, i, j] = im.get_data()

# Average per class
TPM = IMAGES.mean(axis=-1)
NORM = np.reshape(TPM.sum(axis=-1), (IM_SHAPE[0], IM_SHAPE[1], IM_SHAPE[2], 1))
NORM_TPM = TPM / NORM
TPM_IM = nib.Nifti1Image(NORM_TPM,
                         im.get_affine(),
                         header=im.get_header())
nib.save(TPM_IM, OUTPUT_TPM)