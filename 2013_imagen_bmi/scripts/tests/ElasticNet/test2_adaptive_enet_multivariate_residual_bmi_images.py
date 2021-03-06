# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 09:20:23 2014

@author: hl237680

Implementation of adaptive Enet algorithm based on Zou's article in order
to maximize the prediction score between images' voxels and BMI.
"""

import os, sys
import numpy as np
import pandas as pd
import tables
import math
import time

import matplotlib.pyplot as plt

import parsimony.estimators as estimators
import parsimony.functions.nesterov.gl as gl
import parsimony.algorithms.proximal as proximal

sys.path.append(os.path.join(os.getenv('HOME'), 'gits', 'scripts', '2013_imagen_bmi', 'scripts'))
import bmi_utils
    
sys.path.append(os.path.join(os.environ["HOME"], "gits", "scripts", "2013_imagen_subdepression", "lib"))
import utils




#############
# Read data #
#############
def load_residualized_bmi_data(cache):
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
        design_mat = design_mat[0:50,:]
        # Images
        h5file = tables.openFile(IMAGES_FILE)
        masked_images = bmi_utils.read_array(h5file, "/standard_mask/residualized_images_gender_center_TIV_pds")    #images already masked
        print "Data loaded - Processing"
        # Concatenate images with covariates gender, imaging city centrr, tiv_gaser and mean pds status in order to do as though BMI had been residualized
        X_init = masked_images[0:50,:]
        z = BMI[0:50,:]
        X_res = np.concatenate((design_mat, X_init), axis=1)
        np.save(os.path.join(SHARED_DIR, "X_init.npy"), X_init)
        np.save(os.path.join(SHARED_DIR, "X_res.npy"), X_res)
        np.save(os.path.join(SHARED_DIR, "z.npy"), z)
        h5file.close()        
        print "Data saved"
    else:
        X_init = np.load(os.path.join(SHARED_DIR, "X_init.npy"))
        X_res = np.load(os.path.join(SHARED_DIR, "X_res.npy"))
        z = np.load(os.path.join(SHARED_DIR, "z.npy"))        
        print "Data read from cache"    
    return X_init, X_res, z




if __name__ == "__main__":
    # Set pathes
    WD = "/neurospin/tmp/brainomics/adaptive_enet"
    if not os.path.exists(WD):
        os.makedirs(WD)
    
    # Build dataset
    print "#################"
    print "# Build dataset #"
    print "#################"

    # Pathnames
    BASE_PATH = '/neurospin/brainomics/2013_imagen_bmi/'
    DATA_PATH = os.path.join(BASE_PATH, 'data')
    CLINIC_DATA_PATH = os.path.join(DATA_PATH, 'clinic')
    IMAGES_FILE = os.path.join(DATA_PATH, 'smoothed_images.hdf5')
    BMI_FILE = os.path.join(DATA_PATH, 'BMI.csv')
    
    # Shared data
    BASE_SHARED_DIR = "/neurospin/tmp/brainomics/"
    SHARED_DIR = os.path.join(BASE_SHARED_DIR, 'test_bmi_cache_50indiv')
    if not os.path.exists(SHARED_DIR):
        os.makedirs(SHARED_DIR)

    # Load data
    print "Load data"
    X_init, X_res, z = load_residualized_bmi_data(cache=True)
    np.save(os.path.join(WD, 'X_res.npy'), X_res)
    np.save(os.path.join(WD, "z.npy"), z)
   
    # Adaptive Elastic Net algo    
    (n, p) = X_init.shape
    gamma = 1
    groups = [[j] for j in range(0, p)]
    print "Compute ElasticNet algorithm"
    enet_PP = estimators.ElasticNet(l=0.8, alpha=0.006, penalty_start=11, mean=True)
    enet_PP.fit(X_res,z)
    print "Compute beta values"
    beta = enet_PP.beta
    beta = beta[11:]  #do not consider covariates
    print "Compute the weights using Parsimony's ElasticNet algorithm."
    weights = [math.pow(abs(beta[j[0]])+1/float(n), -gamma) for j in groups]
    # Adaptive Elasticnet algorithm
    A=gl.A_from_groups(p, groups, weights=weights)
    adaptive_enet = estimators.LinearRegressionL1L2GL(
                    l1=0, l2=0.8, gl=0.006,
                    A=A,
                    algorithm=proximal.FISTA(),
                    algorithm_params=dict(max_iter=10000),
                    penalty_start=11,
                    mean=True)

    stime = time.time()
    print "================================================================="
    print "Now fitting the model"
    adaptive_enet.fit(X_res, z)
    print "Fit duration : ", time.time() - stime
    print "================================================================="
    
    # Interpretation
    beta_w = adaptive_enet.beta[11:]
    plt.plot(beta_w)
    plt.show()





#def load_globals(config):
#    import mapreduce as GLOBAL  # access to global variables
#    GLOBAL.DATA = GLOBAL.load_data(config["data"])
#
#
#def resample(config, resample_nb):
#    import mapreduce as GLOBAL  # access to global variables
#    #GLOBAL.DATA = GLOBAL.load_data(config["data"])
#    resample = config["resample"][resample_nb]
#    print "reslicing %d" %resample_nb
#    GLOBAL.DATA_RESAMPLED = {k: [GLOBAL.DATA[k][idx, ...] for idx in resample]
#                            for k in GLOBAL.DATA}                            
#    print "done reslicing %d" %resample_nb
#
#
#def mapper(key, output_collector):
#    import mapreduce as GLOBAL # access to global variables:
#    # key: list of parameters
#    alpha, l1_ratio = key[0], key[1]
#    Xtr = GLOBAL.DATA_RESAMPLED["X"][0]
#    Xte = GLOBAL.DATA_RESAMPLED["X"][1]
#    ztr = GLOBAL.DATA_RESAMPLED["z"][0]
#    zte = GLOBAL.DATA_RESAMPLED["z"][1]
#    print key, "Data shape:", Xtr.shape, Xte.shape, ztr.shape, zte.shape
#    #
#    mod = estimators.ElasticNet(l1_ratio, alpha, penalty_start=11, mean=True)     #since we residualized BMI with 2 categorical covariables (Gender and ImagingCentreCity - 8 columns) and 2 ordinal variables (tiv_gaser and mean_pds - 2 columns)
#    stime = time.time()
#    print "====================="
#    print "Now fitting the model"
#    z_pred = mod.fit(Xtr,ztr).predict(Xte)
#    print "Fit duration : ", time.time() - stime
#    print "====================="
##    beta = mod.beta[1:] 
##    mask = (mod.beta[1:] != 0.).ravel()
##    mask = (beta*beta>1e-8)
##    from bgutils.pway_interpret import pw_status, pw_beta_thresh
##    pw_status(pw, snpList, mask.ravel())
#    ret = dict(z_pred=z_pred, z_true=zte, beta=mod.beta)
#    output_collector.collect(key, ret)
#    
#
#def reducer(key, values):
#    # key : string of intermediary key
#    # load return dict correspondning to mapper ouput. they need to be loaded.
#    values = [item.load() for item in values]
#    z_true = np.concatenate([item["z_true"].ravel() for item in values])
#    z_pred = np.concatenate([item["z_pred"].ravel() for item in values])
#    scores =  dict(param=key, r2=r2_score(z_true, z_pred))
#    return scores
#
#
#
##"""
##run /home/hl237680/gits/scripts/2013_imagen_bmi/scripts/15_cv_multivariate_residualized_BMI.py
##"""
#if __name__ == "__main__":
#
#    ## Set pathes
#    WD = "/neurospin/tmp/brainomics/adaptive_enet"
#    if not os.path.exists(WD): os.makedirs(WD)
#
#    print "#################"
#    print "# Build dataset #"
#    print "#################"
#
#    # Pathnames
#    BASE_PATH = '/neurospin/brainomics/2013_imagen_bmi/'
#    DATA_PATH = os.path.join(BASE_PATH, 'data')
#    CLINIC_DATA_PATH = os.path.join(DATA_PATH, 'clinic')
#    IMAGES_FILE = os.path.join(DATA_PATH, 'smoothed_images.hdf5')
#    BMI_FILE = os.path.join(DATA_PATH, 'BMI.csv')
#    
#    # Shared data
#    BASE_SHARED_DIR = "/neurospin/tmp/brainomics/"
#    SHARED_DIR = os.path.join(BASE_SHARED_DIR, 'residualized_bmi_cache')
#    if not os.path.exists(SHARED_DIR):
#        os.makedirs(SHARED_DIR)
#        
#    X, z = load_residualized_bmi_data(cache=True)
#    np.save(os.path.join(WD, 'X.npy'), X)
#    np.save(os.path.join(WD, "z.npy"), z)
#
#    print "#####################"
#    print "# Build config file #"
#    print "#####################"
#    ## Parameterize the mapreduce 
#    ##   1) pathes
#    NFOLDS = 5
#    ## 2) cv index and parameters to test
#    cv = [[tr.tolist(), te.tolist()] for tr,te in KFold(X.shape[0], n_folds=NFOLDS)]
#    params = [[alpha, l1_ratio] for alpha in np.arange(0.002, 0.01, 0.001) for l1_ratio in np.arange(0.1, 1., .1)]
#    # User map/reduce function file:
#    #try:
#    #    user_func_filename = os.path.abspath(__file__)
#    #except:
#    user_func_filename = os.path.join("/home/hl237680",
#        "gits", "scripts", "2013_imagen_bmi", "scripts", 
#        "17_adaptive_enet_cv_multivariate_residual_bmi_images.py")
#    #print __file__, os.path.abspath(__file__)
#    print "user_func", user_func_filename
#    # Use relative path from config.json
#    config = dict(data=dict(X='X.npy', z='z.npy'),
#                  params=params, resample=cv,
#                  structure="",
#                  map_output="results",
#                  user_func=user_func_filename,
#                  reduce_input="results/*/*", 
#                  reduce_group_by="results/.*/(.*)",
#                  reduce_output="results.csv")
#    json.dump(config, open(os.path.join(WD, "config.json"), "w"))
#
#    #########################################################################
#    # Build utils files: sync (push/pull) and PBS
#    sys.path.append(os.path.join(os.getenv('HOME'),
#                                'gits','scripts'))
#    import brainomics.cluster_gabriel as clust_utils
#    sync_push_filename, sync_pull_filename, WD_CLUSTER = \
#        clust_utils.gabriel_make_sync_data_files(WD, user="hl237680")
#    cmd = "mapreduce.py -m %s/config.json  --ncore 12" % WD_CLUSTER
#    clust_utils.gabriel_make_qsub_job_files(WD, cmd)
#    #########################################################################
#    # Sync to cluster
#    print "Sync data to gabriel.intra.cea.fr: "
#    os.system(sync_push_filename)
#
#    #############################################################################
#    print "# Start by running Locally with 12 cores, to check that everything is OK)"
#    print "Interrupt after a while CTL-C"
#    print "mapreduce.py -m %s/config.json --ncore 12" % WD
#    #os.system("mapreduce.py --mode map --config %s/config.json" % WD)
#    print "# 1) Log on gabriel:"
#    print 'ssh -t gabriel.intra.cea.fr'
#    print "# 2) Run one Job to test"
#    print "qsub -I"
#    print "cd %s" % WD_CLUSTER
#    print "./job_Global_long.pbs"
#    print "# 3) Run on cluster"
#    print "qsub job_Global_long.pbs"
#    print "# 4) Log out and pull Pull"
#    print "exit"
#    print sync_pull_filename
#    #############################################################################
#    print "# Reduce"
#    print "mapreduce.py -r %s/config.json" % WD_CLUSTER
#    #ATTENTION ! Si envoi sur le cluster, modifier le path de config-2.json : /neurospin/tmp/hl237680/residual_bmi_images_cluster-2/config-2.json