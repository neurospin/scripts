# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 10:53:37 2014

Copyrignt : CEA NeuroSpin - 2014
"""
import os, sys
import numpy as np
import pandas as pd
import tables
import time 

from sklearn.cross_validation import KFold
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score
import parsimony.estimators as estimators

sys.path.append(os.path.join(os.getenv('HOME'), 'gits', 'scripts', '2013_imagen_bmi', 'scripts'))
import bmi_utils

# Pathnames
BASE_PATH = '/neurospin/brainomics/2013_imagen_bmi/'
DATA_PATH = os.path.join(BASE_PATH, 'data')
CLINIC_DATA_PATH = os.path.join(DATA_PATH, 'clinic')
IMAGES_FILE = os.path.join(DATA_PATH, 'smoothed_images.hdf5')
BMI_FILE = os.path.join(DATA_PATH, 'BMI.csv')
# Shared data
BASE_SHARED_DIR = "/neurospin/tmp/brainomics/"
SHARED_DIR = os.path.join(BASE_SHARED_DIR, 'residualized_bmi_cache')


def load_residualized_bmi_data(cache):
    if not(cache):
        #SNPs = pd.io.parsers.read_csv(os.path.join(DATA_PATH, "SNPs.csv"), dtype='float64', index_col=0).as_matrix()
        BMI = pd.io.parsers.read_csv(os.path.join(DATA_PATH, "BMI.csv"), index_col=0).as_matrix()
        # Images
        h5file = tables.openFile(IMAGES_FILE)
        masked_images = bmi_utils.read_array(h5file, "/standard_mask/residualized_images_gender_center_TIV_pds")    #images already masked
        print "Data loaded, processing"
        
        X = masked_images[0:50,:]
        z = BMI[0:50]
        
        np.save(os.path.join(SHARED_DIR, "X.npy"), X)
        np.save(os.path.join(SHARED_DIR, "z.npy"), z)
        h5file.close()
        print "Data saved"
    else:
        X = np.load(os.path.join(SHARED_DIR, "X.npy"))    
        z = np.load(os.path.join(SHARED_DIR, "z.npy"))        
        print "Data read from cache"
    
    return X, z



if __name__ == "__main__":
    #load
    X, z = load_residualized_bmi_data(cache=True)
    n, p = X.shape
    #create  rsampled data set
    NFOLDS = 5
    cv = [[tr.tolist(), te.tolist()] for tr,te in KFold(n, n_folds=NFOLDS, shuffle=True, random_state=2505)]    
    FOLD = 0; TRAIN = 0; TEST = 1
    Xtrain = X[cv[FOLD][TRAIN],...]
    Xtest = X[cv[FOLD][TEST],...]
    ztrain = z[cv[FOLD][TRAIN],...]
    ztest = z[cv[FOLD][TEST],...]
    
    # alpha l1_ratio
    alpha = 1.0
    l1_ratio = 0.9    
    
    
    #parsimony
    XtrainPP = np.hstack((np.ones((ztrain.shape[0],1)),Xtrain))
    XtestPP = np.hstack((np.ones((ztest.shape[0],1)),Xtest))
    mod_PP = estimators.ElasticNet(l1_ratio, alpha=alpha, penalty_start=1, mean=True)
    
    time_curr = time.time()
    z_pred_PP = mod_PP.fit(XtrainPP,ztrain).predict(XtestPP)
    print "Parsimony elapsed time: ", time.time() - time_curr
    print "Parsimony r2:", r2_score(ztest, z_pred_PP)    
    
    
    time_curr = time.time()
    mod_SL = ElasticNet(alpha, l1_ratio, fit_intercept=True)
    z_pred_SL = mod_SL.fit(Xtrain,ztrain).predict(Xtest)
    print "Scikit elapsed time: ", time.time() - time_curr
    print "Scikit r2:", r2_score(ztest, z_pred_SL)    

