# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 17:48:18 2014

@author: hl237680

Multivariate correlation between residualized BMI and one subcortical feature
of interest:
   CV using mapreduce and ElasticNet between the feature of interest and BMI.

The resort to Freesurfer should prevent us from the artifacts that may be
induced by the normalization step of the SPM segmentation process.

INPUT:
- /neurospin/brainomics/2013_imagen_bmi/data/Freesurfer:
   .csv file containing volume of subcortical structures obtained by Freesurfer
- /neurospin/brainomics/2013_imagen_bmi/data/BMI.csv:
   BMI of the 1265 subjects for which we also have neuroimaging data

METHOD: Search for the optimal set of hyperparameters (alpha, l1_ratio) that
        maximizes the correlation between predicted BMI values via the Linear
        Model  and true ones using Elastic Net algorithm, mapreduce and
        cross validation.
--NB: Computation involves to send jobs to Gabriel.--

OUTPUT:
- the Mapper returns predicted and true values of BMI, model estimators.
- the Reducer returns R2 scores between prediction and true values.

"""

import os
import sys
import numpy as np
import pandas as pd
import json
from glob import glob

from sklearn.preprocessing import StandardScaler

from sklearn.cross_validation import KFold
from sklearn.metrics import r2_score
import parsimony.estimators as estimators

sys.path.append(os.path.join(os.environ['HOME'], 'gits', 'scripts',
                             '2013_imagen_subdepression', 'lib'))
import utils


###############################
# Mapreduce related functions #
###############################

def load_globals(config):
    import mapreduce as GLOBAL
    GLOBAL.DATA = GLOBAL.load_data(config["data"])


def resample(config, resample_nb):
    import mapreduce as GLOBAL
    #GLOBAL.DATA = GLOBAL.load_data(config["data"])
    resample = config["resample"][resample_nb]
    print "reslicing %d" % resample_nb
    GLOBAL.DATA_RESAMPLED = {k: [GLOBAL.DATA[k][idx, ...] for idx in resample]
                            for k in GLOBAL.DATA}
    print "done reslicing %d" % resample_nb


def mapper(key, output_collector):
    import mapreduce as GLOBAL
    # key: list of parameters
    alpha, l1_ratio = key[0], key[1]
    Xtr = GLOBAL.DATA_RESAMPLED["X"][0]
    Xte = GLOBAL.DATA_RESAMPLED["X"][1]
    ztr = GLOBAL.DATA_RESAMPLED["z"][0]
    zte = GLOBAL.DATA_RESAMPLED["z"][1]
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
    z_true = np.concatenate([item["z_true"].ravel() for item in values])
    z_pred = np.concatenate([item["z_pred"].ravel() for item in values])
    scores = dict(param=key, r2=r2_score(z_true, z_pred))
    return scores


#############
# Read data #
#############
# Load data on BMI
# Load dataframe containing features of subcortical structures extracted by
# Freesurfer
def load_residualized_bmi_data(cache):
    if not(cache):
        # BMI
        BMI_df = pd.io.parsers.read_csv(os.path.join(DATA_PATH, 'BMI.csv'),
                                        sep=',',
                                        index_col=0)

        # Freesurfer
        labels = np.genfromtxt(os.path.join(FREESURFER_PATH,
                                    'IMAGEN_Freesurfer_data_29juil2014.csv'),
                                dtype=None,
                                delimiter=',',
                                skip_header=1,
                                usecols=1)

        subject_labels = []
        for i, s in enumerate(labels):
            subject_labels.append(int(s[25:]))

        freesurfer_index = pd.Index(subject_labels)

        # Freesurfer's spreadsheet from IMAGEN database
        freesurfer_df = pd.io.parsers.read_csv(os.path.join(FREESURFER_PATH,
                                    'IMAGEN_Freesurfer_data_29juil2014.csv'),
                                        sep=',',
                                        usecols=['ICV',
                                                 'lhCortexVol',
                                                 'rhCortexVol',
                                                 'CortexVol',
                                                 'SubCortGrayVol',
                                                 'TotalGrayVol',
                                                 'SupraTentorialVol',
                                                 'lhCorticalWhiteMatterVol',
                                                 'rhCorticalWhiteMatterVol',
                                                 'CorticalWhiteMatterVol'])

        # Set the new dataframe index: subjects ID in the right format
        freesurfer_df = freesurfer_df.set_index(freesurfer_index)

        # Dataframe for picking out only clinical cofounds of non interest
        clinical_df = pd.io.parsers.read_csv(os.path.join(CLINIC_DATA_PATH,
                                                          'population.csv'),
                                             index_col=0)

        # Cofounds
        clinical_cofounds = ['Gender de Feuil2', 'ImagingCentreCity',
                             'tiv_gaser', 'mean_pds']

        clinical_df = clinical_df[clinical_cofounds]

        # Consider subjects for which we have neuroimaging and genetic data
        subjects_id = np.genfromtxt(os.path.join(DATA_PATH,
                                                 'subjects_id.csv'),
                                    dtype=None,
                                    delimiter=',',
                                    skip_header=1)

        freesurfer_data = freesurfer_df.loc[subjects_id]

        # Drop rows that have any NaN values
        freesurfer_data = freesurfer_data.dropna()

        # Get indices of subjects for which we have both neuroimaging and
        # genetic data, but also Freesurfer subcortical features
        index = freesurfer_data.index

        # Keep only subjects for which we have ALL data (neuroimaging,
        # genetic data, subcortical features)
        clinical_data = clinical_df.loc[index]
        BMI = BMI_df.loc[index]

        # Conversion dummy coding
        covar = utils.make_design_matrix(clinical_data,
                                    regressors=clinical_cofounds).as_matrix()

        # Concatenate BMI and covariates
        design_mat = np.hstack((covar, freesurfer_data))

        X = design_mat
        # Center & scale X
        skl = StandardScaler()
        X = skl.fit_transform(X)

        z = BMI

        np.save(os.path.join(SHARED_DIR, 'X.npy'), X)
        np.save(os.path.join(SHARED_DIR, 'z.npy'), z)

        print "Data saved"
    else:
        X = np.load(os.path.join(SHARED_DIR, 'X.npy'))
        z = np.load(os.path.join(SHARED_DIR, 'z.npy'))
        print "Data read from cache"
    return X, z


#"""#
#run#
#"""#
if __name__ == "__main__":

    ## Set pathes
    WD = "/neurospin/tmp/brainomics/multivariate_bmi_freesurfer_IMAGEN"
    if not os.path.exists(WD):
        os.makedirs(WD)

    print "#################"
    print "# Build dataset #"
    print "#################"

    # Pathnames
    BASE_PATH = '/neurospin/brainomics/2013_imagen_bmi/'
    DATA_PATH = os.path.join(BASE_PATH, 'data')
    CLINIC_DATA_PATH = os.path.join(DATA_PATH, 'clinic')
    BMI_FILE = os.path.join(DATA_PATH, 'BMI.csv')
    FREESURFER_PATH = os.path.join(DATA_PATH, 'Freesurfer')

    # Shared data
    BASE_SHARED_DIR = "/neurospin/tmp/brainomics/"
    SHARED_DIR = os.path.join(BASE_SHARED_DIR,
                              'bmi_Freesurfer_cache_IMAGEN')
    if not os.path.exists(SHARED_DIR):
        os.makedirs(SHARED_DIR)

    X, z = load_residualized_bmi_data(cache=False)
    n, p = X.shape
    np.save(os.path.join(WD, 'X.npy'), X)
    np.save(os.path.join(WD, 'z.npy'), z)

    print "#####################"
    print "# Build config file #"
    print "#####################"
    # Parameterize the mapreduce
    NFOLDS = 5
    # CV index and parameters to test
    cv = [[tr.tolist(), te.tolist()] for tr, te in KFold(n, n_folds=NFOLDS)]
    params = ([[alpha, l1_ratio] for alpha in [0.0001, 0.0005, 0.001,
               0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100]
               for l1_ratio in np.arange(0.1, 1., .1)])

    user_func_filename = os.path.join('/home/hl237680', 'gits', 'scripts',
                                      '2013_imagen_bmi', 'scripts',
                                      'Freesurfer',
                                      '02_multivariate_bmi_freesurfer.py')

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
        clust_utils.gabriel_make_sync_data_files(WD, user="hl237680")
    cmd = "mapreduce.py -m %s/config.json  --ncore 12" % WD_CLUSTER
    clust_utils.gabriel_make_qsub_job_files(WD, cmd)
    #########################################################################
    # Synchronize to cluster
    print "Sync data to gabriel.intra.cea.fr: "
    os.system(sync_push_filename)
    #########################################################################
    print "# Map"
    print "mapreduce.py -m %s/config.json --ncore 12" % WD
    #os.system("mapreduce.py --mode map --config %s/config.json" % WD)
    print "# Run on cluster Gabriel"
    print "qsub job_Global_long.pbs"
    #########################################################################
    print "# Reduce"
    print "mapreduce.py -r %s/config.json" % WD_CLUSTER