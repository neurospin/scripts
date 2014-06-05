# -*- coding: utf-8 -*-
"""
Created on Mon May 26 13:33:13 2014

@author: hl237680
"""

import os, sys
import json
import numpy as np
import pandas as pd
import tables

from sklearn.cross_validation import KFold
from sklearn.metrics import r2_score
import parsimony.estimators as estimators

sys.path.append(os.path.join(os.getenv('HOME'), 'gits', 'scripts', '2013_imagen_bmi', 'scripts'))
import bmi_utils
    
sys.path.append(os.path.join(os.environ["HOME"], "gits", "scripts", "2013_imagen_subdepression", "lib"))
import utils
        


##############################################################################
def load_globals(config):
    import mapreduce as GLOBAL  # access to global variables
    print "start load_globals"
    GLOBAL.DATA = GLOBAL.load_data(config["data"])
    print "finished load_globals"

def resample(config, resample_nb):
    import mapreduce as GLOBAL  # access to global variables
    resample = config["resample"][resample_nb]    
    print "reslicing %d" %resample_nb
    GLOBAL.DATA_RESAMPLED = {k: [GLOBAL.DATA[k][idx, ...] for idx in resample]
                            for k in GLOBAL.DATA}
    print "done reslicing %d" %resample_nb                        

## User map/reduce functions
def mapper(key, output_collector):
    import mapreduce as GLOBAL # access to global variables (GLOBAL.DATA)
    alpha, l1_ratio = key[0], key[1]
    # mod = ElasticNet(alpha=key[0], l1_ratio=key[1])
    print "i am a work that works"
    #mod = estimators.ElasticNet(alpha*l1_ratio, penalty_start = 1, mean = True)
    mod = estimators.ElasticNet(alpha*l1_ratio, penalty_start = 11, mean = True)     #since we residualize BMI with 2 categorical covariables (8 columns) and 2 ordinal variables
    z_pred = mod.fit(GLOBAL.DATA_RESAMPLED["X"][0], GLOBAL.DATA_RESAMPLED["z"][0]).predict(GLOBAL.DATA_RESAMPLED["X"][1])
    output_collector.collect(key, dict(z_pred=z_pred, z_true=GLOBAL.DATA_RESAMPLED["z"][1]), beta=mod.beta)

def reducer(key, values):
    # values are OutputCollerctors containing a path to the results.
    # load return dict correspondning to mapper ouput. they need to be loaded.
    values = [item.load() for item in values]
    z_true = np.concatenate([item["z_true"].ravel() for item in values])
    z_pred = np.concatenate([item["z_pred"].ravel() for item in values])
    return dict(param=key, r2=r2_score(z_true, z_pred))


# SNPs and BMI
def load_data(cache):
    if not(cache):
        BMI = pd.io.parsers.read_csv(os.path.join(DATA_PATH, "BMI.csv"), index_col=0).as_matrix()

        # Dataframe      
        COFOUND = ["Subject", "Gender de Feuil2", "ImagingCentreCity", "tiv_gaser", "mean_pds"]
        df = pd.io.parsers.read_csv(os.path.join(CLINIC_DATA_PATH, "1534bmi-vincent2.csv"), index_col=0)
        df = df[COFOUND]

        # Conversion dummy coding
        design_mat = utils.make_design_matrix(df, regressors=COFOUND).as_matrix()

        # Keep only subjects for which we have all data and remove the 1. column containing subject_id from the numpy array design_mat
        subjects_id = np.genfromtxt(os.path.join(DATA_PATH, "subjects_id.csv"), dtype=None, delimiter=',', skip_header=1)
        design_mat = np.delete(np.delete(design_mat, np.where(np.in1d(design_mat[:,0], np.delete(design_mat, np.where(np.in1d(design_mat[:,0], subjects_id)), 0))), 0),0,1)
               
        # Images
        h5file = tables.openFile(IMAGES_FILE)
        masked_images = bmi_utils.read_array(h5file, "/standard_mask/residualized_images_gender_center_TIV_pds")    #images already masked
        print "Data loaded"
        
        # Concatenate images with covariates gender, imaging city centrr, tiv_gaser and mean pds status in order to do as though BMI had been residualized
        X = np.concatenate((design_mat, masked_images), axis=1)
        z = BMI
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
     
    WD = "/neurospin/tmp/brainomics/residual_bmi_images_cluster"
    if not os.path.exists(WD):
        os.makedirs(WD)
        
    print "#################"
    print "# Build dataset #"
    print "#################"
    if True:
        # Pathnames
        BASE_PATH = '/neurospin/brainomics/2013_imagen_bmi/'
        DATA_PATH = os.path.join(BASE_PATH, 'data')
        CLINIC_DATA_PATH = os.path.join(DATA_PATH, 'clinic')
        IMAGES_FILE = os.path.join(DATA_PATH, 'smoothed_images.hdf5')
        BMI_FILE = os.path.join(DATA_PATH, 'BMI.csv')
        
        # Shared data
        BASE_SHARED_DIR = "/neurospin/tmp/brainomics/"
        SHARED_DIR = os.path.join(BASE_SHARED_DIR, 'residualized_bmi_cache')
        if not os.path.exists(SHARED_DIR):
            os.makedirs(SHARED_DIR)

    #############################################################################
    ## Get data
    X, z = load_data(True)
    n, p = X.shape
    np.save(os.path.join(WD, 'X.npy'), np.hstack((np.ones((z.shape[0],1)),X)))
    np.save(os.path.join(WD, 'z.npy'), z)
    
    #############################################################################
    
    print "#####################"
    print "# Build config file #"
    print "#####################"    
    ## Create config file
    cv = [[tr.tolist(), te.tolist()] for tr,te in KFold(n, n_folds=5)]
    params = [[alpha, l1_ratio] for alpha in [0.0009, 0.001] for l1_ratio in np.arange(0.1, 0.4, .1)]
    user_func_filename = os.path.abspath(__file__)

    config = dict(data=dict(X=os.path.join(WD, "X.npy"),
                            z=os.path.join(WD, "z.npy")),
                  params=params,
                  map_output=os.path.join(WD, "results"),
                  user_func=user_func_filename,
                  resample=cv,
                  ncore=2,
                  reduce_input=os.path.join(WD, "results/*/*"),
                  reduce_group_by=os.path.join(WD, "results/.*/(.*)"))
    json.dump(config, open(os.path.join(WD, "config.json"), "w"))

    #############################################################################
    print "# Run Locally:"
    print "# Map"   
    print "mapreduce.py -m %s/config.json --ncore 2" % WD
    
    #############################################################################
    print "# Reduce"
    print "mapreduce.py -r %s/config.json" % WD
