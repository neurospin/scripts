# -*- coding: utf-8 -*-
"""
Created on Wed May 21 14:24:55 2014

@author: hl237680
"""

import os
import numpy as np
import pandas as pd
import nibabel as ni
import tables
import bmi_utils
from sklearn.linear_model import ElasticNet

##############
# Parameters #
##############
# Input data
BASE_PATH = '/neurospin/brainomics/2013_imagen_bmi/'
DATA_PATH = os.path.join(BASE_PATH, 'data')
IMAGES_FILE = os.path.join(DATA_PATH, 'smoothed_images.hdf5')
SNPS_FILE = os.path.join(DATA_PATH, 'SNPs.csv')
BMI_FILE = os.path.join(DATA_PATH, 'BMI.csv')

# Shared data
BASE_SHARED_DIR = "/neurospin/tmp/brainomics/md238665"
SHARED_DIR = os.path.join(BASE_SHARED_DIR, 'multiblock_analysis')
if not os.path.exists(SHARED_DIR):
    os.makedirs(SHARED_DIR)


#############
# Read data #
#############
# SNPs and BMI
SNPs = pd.io.parsers.read_csv(os.path.join(DATA_PATH, "SNPs.csv"), dtype='float64', index_col=0).as_matrix()
BMI = pd.io.parsers.read_csv(os.path.join(DATA_PATH, "BMI.csv"), index_col=0).as_matrix()

# Images
h5file = tables.openFile(IMAGES_FILE)
mask = bmi_utils.read_array(h5file,'/standard_mask/mask')   #get the mask applied to the images
masked_images = bmi_utils.read_array(h5file, "/standard_mask/residualized_images_gender_center_TIV_pds")    #images already masked
print "Data loaded"

X = masked_images
Y = SNPs
z = BMI

np.save(os.path.join(SHARED_DIR, "X.npy"), X)
np.save(os.path.join(SHARED_DIR, "Y.npy"), Y)
np.save(os.path.join(SHARED_DIR, "z.npy"), z)

print "Data saved"


#####################
# Elastic Net Model #
#####################        

alpha = 0.5
l1_ratio = 0.9

beta_map = np.zeros(X.shape[1])

debut = range(0, X.shape[1], 10000)
fin = debut + [X.shape[1]]
fin = fin[1:]

#Build a beta-map for a set of values (alpha, l1_ratio)
for d, f in zip(debut, fin):
    print d,f
    enet = ElasticNet(alpha, l1_ratio)
    enet.fit(X[:, d:f], z)
    beta = enet.coef_
    beta_map[d:f] = beta[:]

template_for_size = os.path.join(BASE_PATH, 'data', 'VBM', 'gaser_vbm8/', 'smwp1000074104786s401a1004.nii')
template_for_size_img = ni.load(template_for_size)

image = np.zeros(template_for_size_img.get_data().shape)    #initialize a 3D volume of shape the initial images' shape
image[mask != 0] = beta_map     #mask != 0 lists all indices of non-zero values in order to project the beta_coeff in a 3D volume (np.sum(mask !=0) = beta_map.shape)
pn = os.path.join(BASE_PATH, 'results', 'BMI_beta_map.nii.gz')
ni.save(ni.Nifti1Image(image, template_for_size_img.get_affine()), pn)
print "The estimators' map has been saved."
