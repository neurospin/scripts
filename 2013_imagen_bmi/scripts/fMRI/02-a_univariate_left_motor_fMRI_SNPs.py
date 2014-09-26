# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 17:36:00 2014

@author: hl237680

INPUT:
- '/neurospin/brainomics/2013_imagen_bmi/data/clinic/population.csv'
    useful clinical data

- '/neurospin/brainomics/2013_imagen_bmi/data/fMRI_GCA_motor_left/
    subjects_id_left_motor_fMRI.csv'
    List of subjects ID for whom we have both motor left fMRI tasks and
    sulci data

- '/neurospin/brainomics/2013_imagen_bmi/data/BMI_associated_SNPs_measures.csv'
    Genetic measures on SNPs of interest, that is SNPs at the intersection
    between BMI-associated SNPs referenced in the literature and SNPs read
    by the Illumina platform

- '/neurospin/brainomics/2013_imagen_bmi/data/fMRI_GCA_motor_left/
    GCA_motor_left_images.npy'
    unfolded fMRI images for left motor tasks saved in an array-like format

METHOD: MUOLS

NB: Subcortical features and covariates are centered-scaled.

OUTPUT:
- '/neurospin/brainomics/2013_imagen_bmi/data/fMRI_GCA_motor_left/Results/
    MULM_left_motor_fMRI_SNPs_after_Bonferroni_correction.txt':
    Since we focus here on 22 SNPs, we keep the probability-values
    p < (0.05 / 22) that meet a significance threshold of 0.05 after
    Bonferroni correction.

- "/neurospin/brainomics/2013_imagen_bmi/data/fMRI_GCA_motor_left/Results/
    MUOLS_left_motor_fMRI_SNPs_beta_values_df.csv":
    Beta values from the General Linear Model run on SNPs for fMRI left motor
    tasks.

"""


import os
import sys
import numpy as np
import pandas as pd
import json

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
    Ytr = GLOBAL.DATA_RESAMPLED['Y'][0]
    Yte = GLOBAL.DATA_RESAMPLED['Y'][1]
    print key, "Data shape:", Xtr.shape, Xte.shape, Ytr.shape, Yte.shape
    # penalty_start since we residualized BMI with 2 categorical covariables
    # (Gender and ImagingCentreCity - 8 columns) and 2 ordinal variables
    # (tiv_gaser and mean_pds - 2 columns)
    penalty_start = 11
    mod = estimators.ElasticNet(l1_ratio,
                                alpha,
                                penalty_start=penalty_start,
                                mean=True)
    Y_pred = mod.fit(Xtr, Ytr).predict(Xte)
    ret = dict(Y_pred=Y_pred, z_true=Yte, beta=mod.beta)
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
# Load data on BMI, SNPs and sulci features
def load_sulci_SNPs_bmi_data(cache):
    if not(cache):
        # Dataframe for picking out only clinical cofounds of non interest
        clinical_df = pd.io.parsers.read_csv(os.path.join(CLINIC_DATA_PATH,
                                                          'population.csv'),
                                             index_col=0)

        # Cofounds of non interest
        clinical_cofounds = ['Gender de Feuil2',
                             'ImagingCentreCity',
                             'tiv_gaser',
                             'mean_pds']

        clinical_df = clinical_df[clinical_cofounds]

        # SNPs
        SNPs_df = pd.io.parsers.read_csv(os.path.join(DATA_PATH,
                                          'BMI_associated_SNPs_measures.csv'),
                                         index_col=0)

        # fMRI left motor tasks
        masked_images = np.load(os.path.join(GCA_motor_left_PATH,
                                             'GCA_motor_left_images.npy'))

        # List of all subjects who had an fMRI examination
        fMRI_subjects = pd.io.parsers.read_csv(
                                os.path.join(GCA_motor_left_PATH,
                                            'subjects_id_left_motor_fMRI.csv'),
                                index_col=0)

        # Get the intersept of indices of subjects for whom we have both
        # genetic data and fMRI examination
        subjects_intercept = np.intersect1d(SNPs_df.index.values,
                                            fMRI_subjects.index.values)
        subjects_id = np.intersect1d(subjects_intercept,
                                     clinical_df.index.values).tolist()

        # Keep only subjects for whom we have both genetic data and fMRI
        # examination
        clinical_data = clinical_df.loc[subjects_id]
        SNPs = SNPs_df.loc[subjects_id]

        # Conversion dummy coding
        covar = utils.make_design_matrix(clinical_data,
                                    regressors=clinical_cofounds).as_matrix()

        # Center and scale covariates, but not constant regressor's column
        cov = covar[:, 0:-1]
        skl = StandardScaler()
        cov = skl.fit_transform(cov)

        # Constant regressor to mimick the fit intercept
        constant_regressor = np.ones((masked_images.shape[0], 1))

        # Concatenate sulci data, constant regressor and covariates
        design_mat = np.hstack((cov, constant_regressor, masked_images))

        X = design_mat
        Y = SNPs

        np.save(os.path.join(SHARED_DIR, 'X.npy'), X)
        np.save(os.path.join(SHARED_DIR, 'Y.npy'), Y)

        print 'Data saved.'
    else:
        X = np.load(os.path.join(SHARED_DIR, 'X.npy'))
        Y = np.load(os.path.join(SHARED_DIR, 'Y.npy'))
        print 'Data read from cache.'
    return X, Y


#"""#
#run#
#"""#
if __name__ == "__main__":

    # Set pathes
    WD = '/neurospin/tmp/brainomics/multivariate_bmi_assoc_sulci_SNPs'
    if not os.path.exists(WD):
        os.makedirs(WD)

    print "#################"
    print "# Build dataset #"
    print "#################"

    # Pathnames
    BASE_PATH = '/neurospin/brainomics/2013_imagen_bmi/'
    DATA_PATH = os.path.join(BASE_PATH, 'data')
    CLINIC_DATA_PATH = os.path.join(DATA_PATH, 'clinic')
    GCA_motor_left_PATH = os.path.join(DATA_PATH, 'GCA_motor_left')

    # Shared data
    BASE_SHARED_DIR = '/neurospin/tmp/brainomics/'
    SHARED_DIR = os.path.join(BASE_SHARED_DIR,
                              'left_motor_fMRI_SNPs_assoc_cache')
    if not os.path.exists(SHARED_DIR):
        os.makedirs(SHARED_DIR)

    X, Y = load_sulci_SNPs_bmi_data(cache=False)
    n, p = X.shape
    np.save(os.path.join(WD, 'X.npy'), X)
    np.save(os.path.join(WD, 'Y.npy'), Y)

    print "#####################"
    print "# Build config file #"
    print "#####################"
    # Parameterize the mapreduce
    NFOLDS = 5
    # CV index and parameters to test
    cv = [[tr.tolist(), te.tolist()] for tr, te in KFold(n, n_folds=NFOLDS)]
    params = ([[alpha, l1_ratio] for alpha in [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 10]
               for l1_ratio in np.arange(0.1, 1., .1)])

    user_func_filename = os.path.join('/home/hl237680', 'gits', 'scripts',
                                     '2013_imagen_bmi', 'scripts', 'Sulci',
                                     '02-a_univariate_left_motor_fMRI_SNPs.py')

    # Use relative path from config.json
    config = dict(data=dict(X='X.npy', Y='Y.npy'),
                  params=params, resample=cv,
                  structure='',
                  map_output='results',
                  user_func=user_func_filename,
                  reduce_input='results/*/*',
                  reduce_group_by='results/.*/(.*)',
                  reduce_output='results_left_motor_fMRI_SNPs_assoc.csv')
    json.dump(config, open(os.path.join(WD, 'config.json'), 'w'))

    #########################################################################
    # Build utils files: sync (push/pull) and PBS
    sys.path.append(os.path.join(os.getenv('HOME'), 'gits', 'scripts'))
    import brainomics.cluster_gabriel as clust_utils
    sync_push_filename, sync_pull_filename, WD_CLUSTER = \
        clust_utils.gabriel_make_sync_data_files(WD, user='hl237680')
    cmd = 'mapreduce.py -m %s/config.json  --ncore 12' % WD_CLUSTER
    clust_utils.gabriel_make_qsub_job_files(WD, cmd)
    #########################################################################
    # Synchronize to cluster
    print 'Sync data to gabriel.intra.cea.fr: '
    os.system(sync_push_filename)
    #########################################################################
    print '# Map'
    print 'mapreduce.py -m %s/config.json --ncore 12' % WD
    print '# Run on cluster Gabriel'
    print 'qsub job_Global_long.pbs'
    #########################################################################
    print '# Reduce'
    print 'mapreduce.py -r %s/config.json' % WD_CLUSTER