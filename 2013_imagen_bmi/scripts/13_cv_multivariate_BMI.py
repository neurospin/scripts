# -*- coding: utf-8 -*-
"""
Created on Mon May 26 13:33:13 2014

@author: hl237680
"""

# -*- coding: utf-8 -*-
import os
import json
import numpy as np
import pandas as pd
import tables
import sys
sys.path.append('/home/hl237680/gits/pylearn-parsimony')     #add pathway en dur
import parsimony.estimators as estimators
#from sklearn.linear_model import ElasticNet
from sklearn.cross_validation import KFold
from sklearn.metrics import r2_score
sys.path.append('/home/hl237680/gits/scripts/2013_imagen_bmi/scripts/')     #add pathway en dur
import bmi_utils



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
BASE_SHARED_DIR = "/volatile/lajous/"
SHARED_DIR = os.path.join(BASE_SHARED_DIR, 'multivariate_analysis')
if not os.path.exists(SHARED_DIR):
    os.makedirs(SHARED_DIR)




##############################################################################
## User map/reduce functions
def mapper(key, output_collector):
    import mapreduce  as GLOBAL # access to global variables (GLOBAL.DATA)
    # GLOBAL.DATA ::= {"X":[Xtrain, ytrain], "y":[Xtest, ytest]}
#    mod = ElasticNet(alpha=key[0], l1_ratio=key[1])
    alpha, l1_ratio = key[0], key[1]
    mod = estimators.ElasticNet(alpha*l1_ratio, penalty_start = 1, mean = True)
    z_pred = mod.fit(GLOBAL.DATA["X"][0], GLOBAL.DATA["z"][0]).predict(GLOBAL.DATA["X"][1])
    output_collector.collect(key, dict(z_pred=z_pred, z_true=GLOBAL.DATA["z"][1]))


def reducer(key, values):
    # values are OutputCollerctors containing a path to the results.
    # load return dict correspondning to mapper ouput. they need to be loaded.
    values = [item.load() for item in values]
    y_true = np.concatenate([item["z_true"].ravel() for item in values])
    y_pred = np.concatenate([item["z_pred"].ravel() for item in values])
    return dict(param=key, r2=r2_score(y_true, y_pred))


# SNPs and BMI
def load_data(cache):
    if not(cache):
        SNPs = pd.io.parsers.read_csv(os.path.join(DATA_PATH, "SNPs.csv"), dtype='float64', index_col=0).as_matrix()
        BMI = pd.io.parsers.read_csv(os.path.join(DATA_PATH, "BMI.csv"), index_col=0).as_matrix()
        #images
        h5file = tables.openFile(IMAGES_FILE)
        masked_images = bmi_utils.read_array(h5file, "/standard_mask/residualized_images_gender_center_TIV_pds")    #images already masked
        print "Data loaded"
        X = masked_images
        Y = SNPs
        z = BMI
        np.save(os.path.join(SHARED_DIR, "X.npy"), X)
        np.save(os.path.join(SHARED_DIR, "Y.npy"), Y)
        np.save(os.path.join(SHARED_DIR, "z.npy"), z)
        h5file.close()
        print "Data saved"
    else:
        X = np.load(os.path.join(SHARED_DIR, "X.npy"))        
        Y = np.load(os.path.join(SHARED_DIR, "Y.npy"))
        z = np.load(os.path.join(SHARED_DIR, "z.npy"))        
        print "Data read from cache"
    return X, Y, z




if __name__ == "__main__":
    WD = "/neurospin/tmp/brainomics/bmi_imagesPL"
    if not os.path.exists(WD): os.makedirs(WD)

    #############################################################################
    ## Get data
    X, Y, z = load_data(True)
    n, p = X.shape
#    X = np.random.rand(n, p)
#    beta = np.random.rand(p, 1)
#    y = np.dot(X, beta)
    np.save(os.path.join(WD, 'X.npy'), np.hstack((np.ones((z.shape[0],1)),X)))
    np.save(os.path.join(WD, 'z.npy'), z)
    
    #############################################################################
    ## Create config file
    cv = [[tr.tolist(), te.tolist()] for tr,te in KFold(n, n_folds=2)]
    params = [[alpha, l1_ratio] for alpha in [0.003, 0.007, 0.010] for l1_ratio in np.arange(0.5, 1., .2)]
    user_func_filename = os.path.abspath(__file__)

    config = dict(data=dict(X=os.path.join(WD, "X.npy"),
                            z=os.path.join(WD, "z.npy")),
                  params=params, resample=cv,
                  map_output=os.path.join(WD, "results"),
                  user_func=user_func_filename,
                  ncore=1,
                  reduce_input=os.path.join(WD, "results/*/*"),
                  reduce_group_by=os.path.join(WD, "results/.*/(.*)"))
    json.dump(config, open(os.path.join(WD, "config.json"), "w"))

    #############################################################################
    print "# Run Locally:"
    print "mapreduce.py --mode map --config %s/config.json" % WD
    #os.system("mapreduce.py --mode map --config %s/config.json" % WD)
    
    #############################################################################
    print "# Run on the cluster with 4 PBS Jobs"
    print "mapreduce.py --pbs_njob 4 --config %s/config.json" % WD
    
    #############################################################################
    print "# Reduce"
    print "mapreduce.py --mode reduce --config %s/config.json" % WD
