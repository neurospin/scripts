# -*- coding: utf-8 -*-
"""
Created on Mon May 26 11:07:33 2014

@author: hl237680

Multivariate study between smoothed gaser images and BMI using ElasticNet
and "manual" cross validation.

INPUT:
    - images:
IMAGES_FILE: /neurospin/brainomics/2013_imagen_bmi/data/smoothed_images.hdf5

    - BMI:
BMI_FILE: /neurospin/brainomics/2013_imagen_bmi/data/bmi.csv

OUTPUT: list of R2-values

"""

import os
import numpy as np
import pandas as pd
import tables
import bmi_utils
#import parsimony.estimators as estimators
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
BASE_SHARED_DIR = '/volatile/lajous/'
SHARED_DIR = os.path.join(BASE_SHARED_DIR, 'multivariate_analysis')
if not os.path.exists(SHARED_DIR):
    os.makedirs(SHARED_DIR)


#############
# Read data #
#############

# SNPs and BMI
def load_data(cache):
    if not(cache):
        # SNPs
        SNPs = pd.io.parsers.read_csv(os.path.join(DATA_PATH, 'SNPs.csv'),
                                      dtype='float64',
                                      index_col=0).as_matrix()
        # BMI
        BMI = pd.io.parsers.read_csv(os.path.join(DATA_PATH, 'BMI.csv'),
                                     index_col=0).as_matrix()

        # Load images that have already been masked
        h5file = tables.openFile(IMAGES_FILE)
        masked_images = bmi_utils.read_array(h5file,
                    '/standard_mask/residualized_images_gender_center_TIV_pds')
        print "Data loaded"

        X = masked_images
        Y = SNPs
        z = BMI
        np.save(os.path.join(SHARED_DIR, 'X.npy'), X)
        np.save(os.path.join(SHARED_DIR, 'Y.npy'), Y)
        np.save(os.path.join(SHARED_DIR, 'z.npy'), z)
        h5file.close()
        print "Data saved"
    else:
        X = np.load(os.path.join(SHARED_DIR, 'X.npy'))
        Y = np.load(os.path.join(SHARED_DIR, 'Y.npy'))
        z = np.load(os.path.join(SHARED_DIR, 'z.npy'))
        print "Data read from cache"
    return X, Y, z


#####################
# Elastic Net Model #
#####################

# Get data
X, Y, z = load_data(True)

# Maximization of the R2_mean for a set of values (alpha, l1_ratio)
#alpha = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01]
#l1_ratio = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
alpha = 0.007
l1_ratio = [0.9, 0.8, 0.7, 0.6, 0.5]

n_samples = X.shape[0]
R2_moy = []

debut = range(0, n_samples, n_samples / 10)
fin = debut + [n_samples]
fin = fin[1:]

for j in l1_ratio:
    print "l1_ratio %.1f" % j
    R2 = []
    for d, f in zip(debut, fin):
        print d, f
        X_test = X[d:f]
        X_train = np.delete(X, np.s_[d:f], axis=0)
        z_test = z[d:f]
        z_train = np.delete(z, np.s_[d:f], axis=0)
        print "Computation of the model"
        enet = ElasticNet(alpha, j, fit_intercept=True)
        print "Cross validation (test sample vs. train sample)"
        z_pred_enet = enet.fit(X_train, z_train).predict(X_test)
        print "Compute R2 value for one set (%.3f, %.1f)" % (alpha, j)
        r2_score_enet = r2_score(z_test, z_pred_enet)
        R2.append(r2_score_enet)

    # R2_moy for that set of model's parameters (alpha,l1_ratio)
    r2_moy = sum(R2) / float(len(R2))
    print "R2 moyen sur un set (%.3f, %.1f) : %.10f" % (alpha, j, r2_moy)
    R2_moy.append(r2_moy)