# -*- coding: utf-8 -*-
"""
Created on Tue May 27 08:26:28 2014

@author: 
Copyrignt : CEA NeuroSpin - 2014
"""


import os, sys
import json
import numpy as np

import pandas as pd
import tables

from sklearn.cross_validation import KFold
from sklearn.metrics import r2_score
import parsimony.estimators as estimators


#
#import os
#import json
#import numpy as np
#from sklearn.cross_validation import StratifiedKFold
#import nibabel
#from sklearn.metrics import precision_recall_fscore_support
#from sklearn.feature_selection import SelectKBest
#from parsimony.estimators import LogisticRegressionL1L2TV
#import parsimony.functions.nesterov.tv as tv_helper


def load_globals(config):
    import mapreduce as GLOBAL  # access to global variables
    GLOBAL.DATA = GLOBAL.load_data(config["data"])
#    STRUCTURE = nibabel.load(config["structure"])
#    A, _ = tv_helper.A_from_mask(STRUCTURE.get_data())
#    GLOBAL.A, GLOBAL.STRUCTURE = A, STRUCTURE


def resample(config, resample_nb):
    import mapreduce as GLOBAL  # access to global variables
    GLOBAL.DATA = GLOBAL.load_data(config["data"])
    resample = config["resample"][resample_nb]
    GLOBAL.DATA_RESAMPLED = {k: [GLOBAL.DATA[k][idx, ...] for idx in resample]
                            for k in GLOBAL.DATA}

def mapper(key, output_collector):
    import mapreduce as GLOBAL # access to global variables:
        #raise ImportError("could not import ")
    # GLOBAL.DATA, GLOBAL.STRUCTURE, GLOBAL.A
    # GLOBAL.DATA ::= {"X":[Xtrain, ytrain], "y":[Xtest, ytest]}
    # key: list of parameters
    alpha, l1_ratio = key[0], key[1]
    #
    Xtr = GLOBAL.DATA_RESAMPLED["X"][0]
    Xte = GLOBAL.DATA_RESAMPLED["X"][1]
    ztr = GLOBAL.DATA_RESAMPLED["z"][0]
    zte = GLOBAL.DATA_RESAMPLED["z"][1]
    print key, "Data shape:", Xtr.shape, Xte.shape, ztr.shape, zte.shape
    #
    mod = estimators.ElasticNet(alpha*l1_ratio, penalty_start = 1, mean = True)
    z_pred = mod.fit(Xtr,ztr).predict(Xte)
    ret = dict(z_pred=z_pred, z_true=zte)
    output_collector.collect(key, ret)
    

def reducer(key, values):
    # key : string of intermediary key
    # load return dict correspondning to mapper ouput. they need to be loaded.
    # DEBUG
    #import glob, mapreduce
    #values = [mapreduce.OutputCollector(p) for p in glob.glob("/neurospin/brainomics/2014_mlc/GM_UNIV/results/*/0.05_0.45_0.45_0.1_-1.0/")]
    # Compute sd; ie.: compute results on each folds
    values = [item.load() for item in values]
    y_true = np.concatenate([item["z_true"].ravel() for item in values])
    y_pred = np.concatenate([item["z_pred"].ravel() for item in values])
    scores =  dict(param=key, r2=r2_score(y_true, y_pred))
    return scores

#    values = [item.load("*.npy") for item in values]
#    recall_mean_std = np.std([np.mean(precision_recall_fscore_support(
#        item["y_true"].ravel(), item["y_pred"])[1]) for item in values]) / np.sqrt(len(values))
#    y_true = [item["y_true"].ravel() for item in values]
#    y_pred = [item["y_pred"].ravel() for item in values]
#    y_true = np.concatenate(y_true)
#    y_pred = np.concatenate(y_pred)
#    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average=None)
#    n_ite = None
#    scores = dict(key=key,
#               recall_0=r[0], recall_1=r[1], recall_mean=r.mean(), recall_mean_std=recall_mean_std,
#               precision_0=p[0], precision_1=p[1], precision_mean=p.mean(),
#               f1_0=f[0], f1_1=f[1], f1_mean=f.mean(),
#               support_0=s[0] , support_1=s[1], n_ite=n_ite)
#    return scores


##############################################################################
## Run all
def run_all():
    import mapreduce
#    WD = "/neurospin/brainomics/2013_adni/proj_classif_MCIc-MCInc_gtvenet"
#    key = '0.01_0.01_0.98_0.01_10000'
#    #class GLOBAL: DATA = dict()
#    mapreduce.A, mapreduce.STRUCTURE = A_from_structure(os.path.join(WD,  "mask_atlas.nii.gz"))
#    OUTPUT = os.path.join(os.path.dirname(WD), 'logistictvenet_univ_all', key)
#    # run /home/ed203246/bin/mapreduce.py
#    oc = mapreduce.OutputCollector(OUTPUT)
#    #if not os.path.exists(OUTPUT): os.makedirs(OUTPUT)
#    X = np.load(os.path.join(WD,  'X_atlas.npy'))
#    y = np.load(os.path.join(WD,  'y.npy'))
#    mapreduce.DATA["X"] = [X, X]
#    mapreduce.DATA["y"] = [y, y]
#    params = np.array([float(p) for p in key.split("_")])
#    mapper(params, oc)
#    #oc.collect(key=key, value=ret)

sys.path.append(os.path.join(os.getenv('HOME'),
                                'gits','scripts','2013_imagen_bmi', 'scripts'))
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
BASE_SHARED_DIR = "/neurospin/tmp/brainomics/"
SHARED_DIR = os.path.join(BASE_SHARED_DIR, 'bmi_cache')
if not os.path.exists(SHARED_DIR):
    os.makedirs(SHARED_DIR)


#############
# Read data #
#############
# SNPs and BMI
def load_bmi_data(cache=False):
    if not(cache):
        #SNPs = pd.io.parsers.read_csv(os.path.join(DATA_PATH, "SNPs.csv"), dtype='float64', index_col=0).as_matrix()
        BMI = pd.io.parsers.read_csv(os.path.join(DATA_PATH, "BMI.csv"), index_col=0).as_matrix()
        
        # Images
        h5file = tables.openFile(IMAGES_FILE)
        masked_images = bmi_utils.read_array(h5file, "/standard_mask/residualized_images_gender_center_TIV_pds")    #images already masked
        print "Data loaded"
        
        X = masked_images
        #Y = SNPs
        z = BMI
        
        np.save(os.path.join(SHARED_DIR, "X.npy"), X)
        #np.save(os.path.join(SHARED_DIR, "Y.npy"), Y)
        np.save(os.path.join(SHARED_DIR, "z.npy"), z)
        h5file.close()
        
        print "Data saved"
    else:
        X = np.load(os.path.join(SHARED_DIR, "X.npy"))        
        #Y = np.load(os.path.join(SHARED_DIR, "Y.npy"))
        z = np.load(os.path.join(SHARED_DIR, "z.npy"))        
        print "Data read from cache"
    
    return X, z #X, Y, z





if __name__ == "__main__":
    ## Set pathes
    WD = "/neurospin/tmp/brainomics/bmi_images_cluster"
    if not os.path.exists(WD): os.makedirs(WD)
    
    ## get update and save data in WD for the mapreduce jobs
    X, z = load_bmi_data(cache=True)
    n, p = X.shape
    np.save(os.path.join(WD, 'X.npy'), np.hstack((np.ones((z.shape[0],1)),X)))
    np.save(os.path.join(WD, "z.npy"), z)

    
    ## Parameterize the mapreduce 
    ##   1) pathes
    INPUT_DATA_X = os.path.join('X.npy')
    INPUT_DATA_z = os.path.join('z.npy')
    NFOLDS = 5
    ## 2) cv idex and parameters to test
    cv = [[tr.tolist(), te.tolist()] for tr,te in KFold(n, n_folds=5)]    
    ## 2) cv idex and parameters to test
    params = [[alpha, l1_ratio] for alpha in [0.003, 0.007, 0.010] for l1_ratio in np.arange(0.5, 1., .2)]
    # User map/reduce function file:
    try:
        user_func_filename = os.path.abspath(__file__)
    except:
        user_func_filename = os.path.join(os.environ["HOME"],
        "git", "scripts", "2013_imagen_bmi", "scripts","proj_BMI_mltvar" 
        "15_cv_mltva_BMI.py")
    #print __file__, os.path.abspath(__file__)
    print "user_func", user_func_filename
    #import sys
    #sys.exit(0)
    # Use relative path from config.json
    config = dict(data=dict(X=INPUT_DATA_X, z=INPUT_DATA_z),
                  params=params, resample=cv,
                  structure="",
                  map_output="results",
                  user_func=user_func_filename,
                  reduce_input="results/*/*", 
                  reduce_group_by="results/.*/(.*)",
                  reduce_output="results.csv")
    json.dump(config, open(os.path.join(WD, "config.json"), "w"))

    #############################################################################
    # Build utils files: sync (push/pull) and PBS
    sys.path.append(os.path.join(os.getenv('HOME'),
                                'gits','scripts'))
    import brainomics.cluster_gabriel as clust_utils
    sync_push_filename, sync_pull_filename, WD_CLUSTER = \
        clust_utils.gabriel_make_sync_data_files(WD)
    cmd = "mapreduce.py --map --config %s/config.json  --ncore 4" % WD_CLUSTER
    clust_utils.gabriel_make_qsub_job_files(WD, cmd)
    #############################################################################
    # Sync to cluster
    print "Sync data to gabriel.intra.cea.fr: "
    os.system(sync_push_filename)
    STOP
    #############################################################################
    print "# Start by running Locally with 2 cores, to check that everything os OK)"
    print "Interrupt after a while CTL-C"
    print "mapreduce.py --map --config %s/config.json --ncore 2" % WD
    #os.system("mapreduce.py --mode map --config %s/config.json" % WD)
    print "# 1) Log on gabriel:"
    print 'ssh -t gabriel.intra.cea.fr'
    print "# 2) Run one Job to test"
    print "qsub -I"
    print "cd %s" % WD_CLUSTER
    print "./job_Global_long.pbs"
    print "# 3) Run on cluster"
    print "qsub job_Global_long.pbs"
    print "# 4) Log out and pull Pull"
    print "exit"
    print sync_pull_filename
    #############################################################################
    print "# Reduce"
    print "mapreduce.py --mode reduce --config %s/config.json" % WD
