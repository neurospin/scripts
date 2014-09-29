# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 17:36:00 2014

@author: hl237680

Univariate association study between left motor tasks in fMRI and SNPs at
the intersection between BMI-associated SNPs referenced in the literature
and SNPs read by the Illumina platform.

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

NB: Covariates are centered-scaled.

OUTPUT:
- '/neurospin/brainomics/2013_imagen_bmi/data/fMRI_GCA_motor_left/Results/
    MULM_left_motor_fMRI_SNPs_after_Bonferroni_correction.txt'
    Since we focus here on 22 SNPs, we keep the probability-values
    p < (0.05 / 22) that meet a significance threshold of 0.05 after
    Bonferroni correction.

- '/neurospin/brainomics/2013_imagen_bmi/data/fMRI_GCA_motor_left/Results/
    MUOLS_left_motor_fMRI_SNPs_beta_values_df.csv'
    Beta values from the General Linear Model run on SNPs for fMRI left motor
    tasks.

"""

import os
import sys
import numpy as np
import pandas as pd
import csv

from mulm.models import MUOLS

from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.join(os.environ['HOME'], 'gits', 'scripts',
                             '2013_imagen_subdepression', 'lib'))
import utils


# Pathnames
BASE_PATH = '/neurospin/brainomics/2013_imagen_bmi/'
DATA_PATH = os.path.join(BASE_PATH, 'data')
CLINIC_DATA_PATH = os.path.join(DATA_PATH, 'clinic')
GCA_motor_left_PATH = os.path.join(DATA_PATH, 'GCA_motor_left')

# Output results
OUTPUT_DIR = os.path.join(GCA_motor_left_PATH, 'Results')
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Shared data
BASE_SHARED_DIR = '/neurospin/tmp/brainomics/'
SHARED_DIR = os.path.join(BASE_SHARED_DIR,
                          'left_motor_fMRI_SNPs_assoc_cache')
if not os.path.exists(SHARED_DIR):
    os.makedirs(SHARED_DIR)


#############
# Read data #
#############
# Load left motor fMRI activation maps data, clinical data and SNPs
def load_left_motor_fMRI_SNPs_data(cache):
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

        # Concatenate data on left motor tasks fMRI, constant regressor and
        # covariates
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
    return X, Y, SNPs


#"""#
#run#
#"""#
if __name__ == "__main__":

    # Set pathes
    WD = '/neurospin/tmp/brainomics/multivariate_left_motor_fMRI_assoc_SNPs'
    if not os.path.exists(WD):
        os.makedirs(WD)

    print "#################"
    print "# Build dataset #"
    print "#################"

    X, Y, SNPs = load_left_motor_fMRI_SNPs_data(cache=False)
    n, p = X.shape
    np.save(os.path.join(WD, 'X.npy'), X)
    np.save(os.path.join(WD, 'Y.npy'), Y)

    colnames = SNPs.columns
    penalty_start = 11

    # Initialize beta_map
    beta_map = np.zeros((X.shape[1], Y.shape[1]))

    print "##############################################################"
    print ("# Perform Mass-Univariate Linear Modeling "
           "based Ordinary Least Squares #")
    print "##############################################################"

    #MUOLS
    bigols = MUOLS()
    bigols.fit(X, Y)
    t, p, df = bigols.stats_t_coefficients(X, Y,
                                          contrast=[0.] * penalty_start +
                                          [1.] * (X.shape[1] - penalty_start),
                                          pval=True)

    proba = []
    for i in np.arange(0, p.shape[0]):
        if (p[i] > 0.95):
            p[i] = 1 - p[i]
        proba.append('%.15f' % p[i])

    # Beta values: coefficients of the fit
    beta_map = bigols.coef_

    beta_df = pd.DataFrame(beta_map[penalty_start:, :].transpose(),
                           index=colnames,
                           columns=['betas'])

    # Save beta values from the GLM on sulci features as a dataframe
    beta_df.to_csv(os.path.join(OUTPUT_DIR, 'MUOLS_beta_values_df.csv'))
    print "Dataframe containing beta values for each SNP has been saved."

    # Since we focus here on 22 SNPs, we only keep the probability values
    # p < (0.05 / 22) that meet a significance threshold of 0.05 after
    # Bonferroni correction.
    # Write results of MULM computation for each feature of interest in a
    # csv file
    bonferroni_correction = 0.05 / (Y.shape[1])

    MULM_after_Bonferroni_correction_file_path = os.path.join(OUTPUT_DIR,
                                    'MULM_after_Bonferroni_correction.txt')

    with open(MULM_after_Bonferroni_correction_file_path, 'wb') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=' ', quotechar=' ')

        for i in np.arange(0, len(proba)):

            if float(proba[i]) < bonferroni_correction:
                SNP_name = colnames[i][11:]
                spamwriter.writerow(
                    ['The MULM probability of the SNP:']
                    + [SNP_name]
                    + ['to be associated to left motor tasks is']
                    + [float(proba[i]) * Y.shape[1]]
                    )