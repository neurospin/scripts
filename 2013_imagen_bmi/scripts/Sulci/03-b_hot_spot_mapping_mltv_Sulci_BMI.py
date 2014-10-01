# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 11:21:20 2014

@author: hl237680

Mapping of hot spots which have been revealed by high correlation ratio
after optimization of (alpha, l1) hyperparameters using Enet algorithm.
This time, we have determined the optimum hyperparameters
(02-b_multivariate_sulci_BMI.py), so that we can run the model on the
whole dataset.
"""

import os
import sys
import numpy as np
import pandas as pd
import csv

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import r2_score
import parsimony.estimators as estimators

sys.path.append(os.path.join(os.environ['HOME'],
                             'gits', 'scripts', '2013_imagen_subdepression',
                             'lib'))
import utils


# Pathnames
BASE_PATH = '/neurospin/brainomics/2013_imagen_bmi/'
DATA_PATH = os.path.join(BASE_PATH, 'data')
CLINIC_DATA_PATH = os.path.join(DATA_PATH, 'clinic')
BMI_FILE = os.path.join(DATA_PATH, 'BMI.csv')
SULCI_PATH = os.path.join(DATA_PATH, 'Imagen_mainSulcalMorphometry')
FULL_SULCI_PATH = os.path.join(SULCI_PATH, 'full_sulci')
QC_PATH = os.path.join(FULL_SULCI_PATH, 'Quality_control')

# Shared data
BASE_SHARED_DIR = "/neurospin/tmp/brainomics/"
SHARED_DIR = os.path.join(BASE_SHARED_DIR,
                          'BMI_asso_Sulci_SNPs_cache_true')
if not os.path.exists(SHARED_DIR):
    os.makedirs(SHARED_DIR)


#############
# Read data #
#############
# Load data on BMI and sulci
def load_sulci_bmi_data(cache):
    if not(cache):
        # BMI
        BMI_df = pd.io.parsers.read_csv(os.path.join(DATA_PATH, 'BMI.csv'),
                                     sep=',',
                                     index_col=0)

        # Sulci maximal depth
        sulci_depthMax_df = pd.io.parsers.read_csv(os.path.join(QC_PATH,
                                                    'sulci_depthMax_df.csv'),
                                                   sep=',',
                                                   index_col=0)

        # SNPs
        SNPs_df = pd.io.parsers.read_csv(os.path.join(DATA_PATH,
                                          'BMI_associated_SNPs_measures.csv'),
                                         index_col=0)

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

        # Get the intersept of indices of subjects for whom we have
        # neuroimaging and genetic data, but also robustly segmented sulci
        subjects_intercept = np.intersect1d(SNPs_df.index.values,
                                            BMI_df.index.values)
        subjects_id = np.intersect1d(subjects_intercept,
                                     sulci_depthMax_df.index.values)

        # Keep only subjects for which we have ALL data (neuroimaging,
        # genetic data and sulci features)
        clinical_data = clinical_df.loc[subjects_id]
        BMI = BMI_df.loc[subjects_id]
        sulci_data = sulci_depthMax_df.loc[subjects_id]

        # Conversion dummy coding
        covar = utils.make_design_matrix(clinical_data,
                                    regressors=clinical_cofounds).as_matrix()

        # Center and scale covariates, but not constant regressor's column
        cov = covar[:, 0:-1]
        skl = StandardScaler()
        cov = skl.fit_transform(cov)

        # Center & scale BMI
        BMI = skl.fit_transform(BMI)

        # Constant regressor to mimick the fit intercept
        constant_regressor = np.ones((sulci_data.shape[0], 1))

        # Concatenate sulci data, constant regressor and covariates
        design_mat = np.hstack((cov, constant_regressor, sulci_data))

        X = design_mat
        z = BMI

        np.save(os.path.join(SHARED_DIR, 'X.npy'), X)
        np.save(os.path.join(SHARED_DIR, 'z.npy'), z)

        print 'Data saved.'
    else:
        X = np.load(os.path.join(SHARED_DIR, 'X.npy'))
        z = np.load(os.path.join(SHARED_DIR, 'z.npy'))
        print 'Data read from cache.'
    return X, z, sulci_data


#######
# Run #
#######
if __name__ == "__main__":

    # Load data
    # BMI has been residualized when looking for the optimum set of
    # hyperparameters
    X, z, sulci_data = load_sulci_bmi_data(cache=False)

    # Get variables' names
    variables = np.hstack((sulci_data.columns))

    # Initialize beta_map
    beta_map = np.zeros(X.shape[1])

    # Elasticnet algorithm via Pylearn-Parsimony
    print "Elasticnet algorithm"
    alpha = 0.5
    l1_ratio = 0.2

    # Since we residualized BMI with 2 categorical covariables (Gender and
    # ImagingCentreCity - 8 columns) and 2 ordinal variables (tiv_gaser and
    # mean_pds - 2 columns)
    penalty_start = 11
    mod = estimators.ElasticNet(l1_ratio,
                                alpha,
                                penalty_start=penalty_start,
                                mean=True)
    mod.fit(X, z)
    print "Compute beta values"
    beta_map = mod.beta
    print "Compute R2"
    r2 = r2_score(z, mod.predict(X))
    print r2

    # Store beta values in a .csv file
    beta_Sulci_BMI_assoc_path = os.path.join(BASE_PATH, 'results',
                                            'beta_Sulci_BMI_assoc.txt')
    with open(beta_Sulci_BMI_assoc_path, 'wb') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=' ', quotechar=' ')

        for i in np.arange(0, beta_map.shape[0] - penalty_start):
            variable = variables[i]
            spamwriter.writerow([
        'The beta value of']
        + [variable]
        + ['to be associated to the BMI while taking into account data on maximal depth sulci is ']
        + [beta_map[i + penalty_start]])
        spamwriter.writerow(['\n R2-score equals'] + [r2])