# -*- coding: utf-8 -*-
"""
Created on Thu Jun  5 10:42:35 2014

@author: hl237680

CV using mapreduce and ElasticNet between images and BMI.

INPUT:
- /neurospin/brainomics/2013_imagen_bmi/data/standard_mask/residualized_images_gender_center_TIV_pds/smoothed_images.hdf5:
    masked images (hdf5 format)
- /neurospin/brainomics/2013_imagen_bmi/data/BMI.csv:
    BMI of the 1265 subjects for which we also have neuroimaging data

METHOD: Search for the optimal set of hyperparameters (alpha, l1_ratio) that
        maximizes the correlation between the Linear Model predicted BMI
        values and true ones using Elastic Net algorithm, mapreduce and
        cross validation.
        Computation involves to send jobs to Gabriel (here, connexion
        directly specified with hl237680's account).

OUTPUT:
- the Mapper returns predicted and true values of BMI, model estimators.
- the Reducer returns R2 scores between prediction and truth.

"""

import os
import sys
import json
import numpy as np
import pandas as pd
import tables

from sklearn.cross_validation import KFold
from sklearn.metrics import r2_score
import parsimony.estimators as estimators

sys.path.append(os.path.join(os.getenv('HOME'), 'gits', 'scripts',
                             '2013_imagen_bmi', 'scripts'))
import bmi_utils

sys.path.append(os.path.join(os.environ['HOME'], 'gits', 'scripts',
                             '2013_imagen_subdepression', 'lib'))
import utils


#######################
# Function definition #
#######################

def load_globals(config):
    import mapreduce as GLOBAL
    GLOBAL.DATA = GLOBAL.load_data(config['data'])


def resample(config, resample_nb):
    import mapreduce as GLOBAL
    #GLOBAL.DATA = GLOBAL.load_data(config["data"])
    resample = config['resample'][resample_nb]
    print "reslicing %d" % resample_nb
    GLOBAL.DATA_RESAMPLED = {k: [GLOBAL.DATA[k][idx, ...] for idx in resample]
                            for k in GLOBAL.DATA}
    print "done reslicing %d" % resample_nb


def mapper(key, output_collector):
    import mapreduce as GLOBAL
    # key: list of parameters
    alpha, l1_ratio = key[0], key[1]
    Xtr = GLOBAL.DATA_RESAMPLED['X'][0]
    Xte = GLOBAL.DATA_RESAMPLED['X'][1]
    ztr = GLOBAL.DATA_RESAMPLED['z'][0]
    zte = GLOBAL.DATA_RESAMPLED['z'][1]
    print key, "Data shape:", Xtr.shape, Xte.shape, ztr.shape, zte.shape
    # penalty_start since we residualized BMI with 2 categorical covariables
    # (Gender and ImagingCentreCity - 8 columns) and 2 ordinal variables
    # (tiv_gaser and mean_pds - 2 columns)
    penalty_start = 11
    mod = estimators.ElasticNet(l1_ratio,
                                alpha,
                                penalty_start=penalty_start,
                                mean=True)
    z_pred = mod.fit(Xtr, ztr).predict(Xte)
    ret = dict(z_pred=z_pred, z_true=zte, beta=mod.beta)
    output_collector.collect(key, ret)


def reducer(key, values):
    # key: string of intermediary keys
    values = [item.load() for item in values]
    z_true = np.concatenate([item['z_true'].ravel() for item in values])
    z_pred = np.concatenate([item['z_pred'].ravel() for item in values])
    scores = dict(param=key, r2=r2_score(z_true, z_pred))
    return scores


#############
# Read data #
#############
# Masked images and BMI
def load_residualized_bmi_data(cache):
    if not(cache):
        # BMI
        BMI = pd.io.parsers.read_csv(os.path.join(DATA_PATH, 'BMI.csv'),
                                     index_col=0).as_matrix()

        # Dataframe
        COFOUND = ['Gender de Feuil2', 'ImagingCentreCity', 'tiv_gaser',
                   'mean_pds']
        df = pd.io.parsers.read_csv(os.path.join(CLINIC_DATA_PATH,
                                                 'population.csv'),
                                                 index_col=0)
        df = df[COFOUND]

        # Keep only subjects for which we have all data
        subjects_id = np.genfromtxt(os.path.join(DATA_PATH, 'subjects_id.csv'),
                                    dtype=None,
                                    delimiter=',',
                                    skip_header=1)

        clinic_data = df.loc[subjects_id]

        # Conversion dummy coding
        covar = utils.make_design_matrix(clinic_data,
                                         regressors=COFOUND).as_matrix()

        # Images -that have already been masked-
        h5file = tables.openFile(IMAGES_FILE)
        masked_images = bmi_utils.read_array(h5file,
                    '/standard_mask/residualized_images_gender_center_TIV_pds')
        print "Data loaded"

        # Concatenate images and covariates
        # (gender, imaging city centre, tiv_gaser and mean pds status)
        # in order to do as though BMI had been residualized
        X = np.hstack((covar, masked_images))
        z = BMI

        np.save(os.path.join(SHARED_DIR, 'X.npy'), X)
        np.save(os.path.join(SHARED_DIR, 'z.npy'), z)

        h5file.close()

        print "Data saved"
    else:
        X = np.load(os.path.join(SHARED_DIR, 'X.npy'))
        z = np.load(os.path.join(SHARED_DIR, 'z.npy'))
        print "Data read from cache"

    return X, z


#"""
#run
#"""
if __name__ == "__main__":

    ## Set pathes
    WD = "/neurospin/tmp/brainomics/residual_bmi_images_opt_hyperparameter_validation"
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

        X, z = load_residualized_bmi_data(cache=False)
        #assert X.shape == (1265, 336188)
        n, p = X.shape
        np.save(os.path.join(WD, 'X.npy'), X)
        np.save(os.path.join(WD, 'z.npy'), z)

    print "#####################"
    print "# Build config file #"
    print "#####################"
    ## Parameterize the mapreduce
    ##   1) pathes
    NFOLDS = 5
    ## 2) cv index and parameters to test
    cv = [[tr.tolist(), te.tolist()] for tr, te in KFold(n, n_folds=NFOLDS)]
    params = [[alpha, l1_ratio] for alpha in [0.007, 0.008, 0.009, 0.01]
                                for l1_ratio in np.arange(0.4, 1., .1)]
    # User map/reduce function file:
    #try:
    #    user_func_filename = os.path.abspath(__file__)
    #except:
    user_func_filename = os.path.join("/home/hl237680",
        "gits", "scripts", "2013_imagen_bmi", "scripts",
        "15_cv_multivariate_residual_bmi_images_opt_hyperparameter.py")
    #print __file__, os.path.abspath(__file__)
    print "user_func", user_func_filename
    # Use relative path from config.json
    config = dict(data=dict(X='X.npy', z='z.npy'),
                  params=params, resample=cv,
                  structure="",
                  map_output="results",
                  user_func=user_func_filename,
                  reduce_input="results/*/*",
                  reduce_group_by="results/.*/(.*)",
                  reduce_output="results.csv")
    json.dump(config, open(os.path.join(WD, "config.json"), "w"))

    #########################################################################
    # Build utils files: sync (push/pull) and PBS
    sys.path.append(os.path.join(os.getenv('HOME'), 'gits', 'scripts'))
    import brainomics.cluster_gabriel as clust_utils
    sync_push_filename, sync_pull_filename, WD_CLUSTER = \
        clust_utils.gabriel_make_sync_data_files(WD, user='hl237680')
    cmd = "mapreduce.py -m %s/config.json  --ncore 12" % WD_CLUSTER
    clust_utils.gabriel_make_qsub_job_files(WD, cmd)
    #########################################################################
    # Sync to cluster
    print "Sync data to gabriel.intra.cea.fr: "
    os.system(sync_push_filename)

    #########################################################################
    print "# Map"
    print "mapreduce.py -m %s/config.json --ncore 12" % WD
    #os.system("mapreduce.py --mode map --config %s/config.json" % WD)
    print "# 1) Log on gabriel:"
    print 'ssh -t gabriel.intra.cea.fr'
    print "# 3) Run on cluster"
    print "qsub job_Global_long.pbs"
    print sync_pull_filename
    #########################################################################
    print "# Reduce"
    print "mapreduce.py -r %s/config.json" % WD_CLUSTER