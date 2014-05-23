# -*- coding: utf-8 -*-
"""
Created on Wed May 21 16:10:28 2014

@author: hl237680
"""

import os
import math
import numpy as np
import pandas as pd
#import nibabel as ni
import tables
import bmi_utils
#import parsimony.estimators as estimators
#import pylab as pl
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score

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
masked_images = bmi_utils.read_array(h5file, "/standard_mask/residualized_images_gender_center_TIV_pds")
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

#Maximization of the R2_mean for a set of values (alpha, l1_ratio)
#alpha = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
#alpha = [0.25, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
#l1_ratio = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
alpha = 0.5
l1_ratio = [0.1, 0.25, 0.5, 0.6, 0.75, 0.9]
#l1_ratio = 0.75
#alpha = [0.25, 0.5, 1, 2, 5]

n_samples = X.shape[0]
R2_moy = []

debut = range(0, n_samples, n_samples/10)
fin = debut + [n_samples]
fin = fin[1:]
#debut = debut[:-1]

#for i in alpha:
#    print i
for j in l1_ratio:
    print j
    R2 = []
    for d, f in zip(debut, fin):
        print d,f
        X_test = X[d:f]
        X_train = np.delete(X, np.s_[d:f], axis=0)
        z_test = z[d:f]
        z_train = np.delete(z, np.s_[d:f], axis=0)
        print "Draw beta map"
        enet = ElasticNet(alpha, j)
#            enet.fit(X[:, d:f], z)
#            beta = enet.coef_
        print "Compute R2 value for one set (alpha, l1_ratio)"
        z_pred_enet = enet.fit(X_train, z_train).predict(X_test)
        r2_score_enet = r2_score(z_test, z_pred_enet)
        R2.append(r2_score_enet)
    r2_moy = sum(R2) / float(len(R2))   #R2_moy for that set of model's parameters (alpha,l1_ratio)
    print r2_moy
    R2_moy.append(r2_moy)

R2_max = max(R2_moy)


#Return index of the list in order to get optima alpha and l1_ratio
#m = len(alpha)
n = len(l1_ratio)

#alpha_ind = np.cast[np.int](math.floor(max(R2_moy))/m)
#l1_ratio_ind = math.floor((R2_moy.index(R2_max))/m)
l1_ratio_ind = np.cast[np.int](math.floor((max(R2_moy))/n))

#alpha_opt = alpha[alpha_ind]
#print alpha_opt
l1_ratio_opt = l1_ratio[l1_ratio_ind]
print l1_ratio_opt