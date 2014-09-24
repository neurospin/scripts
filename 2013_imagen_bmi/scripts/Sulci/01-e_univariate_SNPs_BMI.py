# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 18:07:26 2014

@author: hl237680

Univariate correlation between BMI and the intercept of SNPs referenced in
the literature as associated to the BMI and SNPs read by the Illumina
platform on IMAGEN subjects.

INPUT:
- "/neurospin/brainomics/2013_imagen_bmi/data/clinic/population.csv"
    useful clinical data

- "/neurospin/brainomics/2013_imagen_bmi/data/BMI.csv"
    BMI of the 1.265 subjects for which we also have neuroimaging data

- "/neurospin/brainomics/2013_imagen_bmi/data/BMI_associated_SNPs_measures.csv"
    Genetic measures on SNPs of interest, that is SNPs at the intersection
    between BMI-associated SNPs referenced in the literature and SNPs read
    by the Illumina platform

METHOD: MUOLS

NB: Subcortical features and covariates are centered-scaled.

OUTPUT:
- "/neurospin/brainomics/2013_imagen_bmi/data/genetics/Results/
    MULM_BMI_SNPs_after_Bonferroni_correction.txt":
    Since we focus here on 22 SNPs, we keep the probability-values
    p < (0.05 / 22) that meet a significance threshold of 0.05 after
    Bonferroni correction.

- "/neurospin/brainomics/2013_imagen_bmi/data/genetics/Results/
    MUOLS_BMI_SNPs_beta_values_df.csv":
    Beta values from the General Linear Model run on SNPs for BMI.

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
BMI_FILE = os.path.join(DATA_PATH, 'BMI.csv')
SULCI_PATH = os.path.join(DATA_PATH, 'Imagen_mainSulcalMorphometry')
FULL_SULCI_PATH = os.path.join(SULCI_PATH, 'full_sulci')
QC_PATH = os.path.join(FULL_SULCI_PATH, 'Quality_control')
GENETICS_PATH = os.path.join(DATA_PATH, 'genetics')
PLINK_PATH = os.path.join(GENETICS_PATH, 'Plink')

# Output results
OUTPUT_DIR = os.path.join(GENETICS_PATH, 'Results')
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Shared data
BASE_SHARED_DIR = "/neurospin/tmp/brainomics/"
SHARED_DIR = os.path.join(BASE_SHARED_DIR,
                          'SNPs_BMI_cache_IMAGEN')
if not os.path.exists(SHARED_DIR):
    os.makedirs(SHARED_DIR)


#############
# Read data #
#############
# Sulci and SNPs
def load_sulci_SNPs_data(cache):
    if not(cache):

        # BMI
        BMI_df = pd.io.parsers.read_csv(os.path.join(DATA_PATH, 'BMI.csv'),
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

        # Cofounds
        clinical_cofounds = ['Gender de Feuil2',
                             'ImagingCentreCity',
                             'tiv_gaser',
                             'mean_pds']

        clinical_df = clinical_df[clinical_cofounds]

        # Get the intersept of indices of subjects for whom we have
        # neuroimaging and genetic data
        subjects_id = np.intersect1d(SNPs_df.index.values,
                                        BMI_df.index.values)

        # Check whether all these subjects are actually stored into both
        # dataframes
        SNPs = SNPs_df.loc[subjects_id]
        BMI = BMI_df.loc[subjects_id]
        clinical_data = clinical_df.loc[subjects_id]

        # Conversion dummy coding
        covar = utils.make_design_matrix(clinical_data,
                                     regressors=clinical_cofounds).as_matrix()

        # Center and scale covariates, but not constant regressor's column
        cov = covar[:, 0:-1]
        skl = StandardScaler()
        cov = skl.fit_transform(cov)

        # Center & scale BMI
        BMI = skl.fit_transform(BMI)
        print 'BMI loaded.'

        # Constant regressor to mimick the fit intercept
        constant_regressor = np.ones((BMI.shape[0], 1))

        # Concatenate BMI, constant regressor and covariates
        design_mat = np.hstack((cov, constant_regressor, BMI))

        X = design_mat
        Y = SNPs

        np.save(os.path.join(SHARED_DIR, 'X.npy'), X)
        np.save(os.path.join(SHARED_DIR, 'Y.npy'), Y)
        print 'Data saved.'
    else:
        X = np.load(os.path.join(SHARED_DIR, 'X.npy'))
        Y = np.load(os.path.join(SHARED_DIR, 'Y.npy'))
        print 'Data read from cache.'
    return X, Y, SNPs_df


#"""#
#run#
#"""#
if __name__ == "__main__":

    # Set pathes
    WD = '/neurospin/tmp/brainomics/univariate_bmi_full_sulci_IMAGEN'
    if not os.path.exists(WD):
        os.makedirs(WD)

    print "#################"
    print "# Build dataset #"
    print "#################"

    # Load data
    X, Y, SNPs_df = load_sulci_SNPs_data(cache=False)
    colnames = SNPs_df.columns
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
    beta_df.to_csv(os.path.join(OUTPUT_DIR,
                                'MUOLS_SNPs_BMI_beta_values.csv'))
    print 'Dataframe containing beta values for each SNP has been saved.'

    # Since we focus here on the 22 SNPs at the intersection between
    # BMI-associated SNPs referenced in the literature and SNPs read by the
    # Illumina platform we only keep the probability values p < (0.05 / 22)
    # that meet a significance threshold of 0.05 after Bonferroni correction.
    # Write results of MULM computation for each SNP in a .csv file.
    bonferroni_correction = 0.05 / (Y.shape[1])

    MULM_after_Bonferroni_correction_file_path = os.path.join(OUTPUT_DIR,
                            'MULM_SNPs_BMI_after_Bonferroni_correction.txt')

    with open(MULM_after_Bonferroni_correction_file_path, 'wb') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=' ', quotechar=' ')

        for i in np.arange(0, len(proba)):

            if float(proba[i]) < bonferroni_correction:
                SNP_name = colnames[i]
                spamwriter.writerow(
                   ['The MULM probability for the SNP']
                   + [SNP_name]
                   + ['to be associated to the BMI is']
                   + [float(proba[i]) * Y.shape[1]])