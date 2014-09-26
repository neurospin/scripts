# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 13:43:46 2014

@author: hl237680

Univariate correlation between BMI and volume of subcortical structures
(Freesurfer) on IMAGEN subjects.

The resort to Freesurfer should prevent us from the artifacts that may be
induced by the normalization step of the SPM segmentation process.

INPUT:
- /neurospin/brainomics/2013_imagen_bmi/data/Freesurfer:
   .csv file containing volume of subcortical structures obtained by Freesurfer

- /neurospin/brainomics/2013_imagen_bmi/data/BMI.csv:
   BMI of the 1265 subjects for whom we also have neuroimaging and genetic
   data

METHOD: MUOLS

NB: Features extracted by Freesurfer, BMI and covariates are centered-scaled.

OUTPUT:
- /neurospin/brainomics/2013_imagen_bmi/data/Freesurfer/Results/
  MULM_bmi_freesurfer.txt:
    Results of MULM computation, i.e. p-value for each feature of interest,
    that this feature extracted by Freesurfer algorithm is significantly
    associated to BMI

- /neurospin/brainomics/2013_imagen_bmi/data/Freesurfer/Results/
  MULM_after_Bonferroni_correction.txt:
    Since we focus here on 9 features extracted by Freesurfer, we only keep
    the probability-values p < (0.05 / 9) that meet a significance threshold
    of 0.05 after Bonferroni correction.

- /neurospin/brainomics/2013_imagen_bmi/data/Imagen_mainSulcalMorphometry/
  full_sulci/Results/MUOLS_beta_values_df.csv:
    Beta values from the General Linear Model run on subcortical structures
    extracted by Freesurfer.

"""

import os
import sys
import csv
import numpy as np
import pandas as pd

from mulm.models import MUOLS

from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.join(os.environ["HOME"], "gits", "scripts",
                             "2013_imagen_subdepression", "lib"))
import utils


# Pathnames
BASE_PATH = '/neurospin/brainomics/2013_imagen_bmi/'
DATA_PATH = os.path.join(BASE_PATH, 'data')
CLINIC_DATA_PATH = os.path.join(DATA_PATH, 'clinic')
BMI_FILE = os.path.join(DATA_PATH, 'BMI.csv')
FREESURFER_PATH = os.path.join(DATA_PATH, 'Freesurfer')

# Output results
OUTPUT_DIR = os.path.join(FREESURFER_PATH, 'Results')
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Shared data
BASE_SHARED_DIR = "/neurospin/tmp/brainomics/"
SHARED_DIR = os.path.join(BASE_SHARED_DIR,
                          'bmi_Freesurfer_cache_IMAGEN')
if not os.path.exists(SHARED_DIR):
    os.makedirs(SHARED_DIR)


#############
# Read data #
#############
# Subcortical features and BMI
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
                                        usecols=['lhCortexVol',
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

        # Get indices of subjects fot which we have both neuroimaging and
        # genetic data, but also sulci features
        index = freesurfer_data.index

        # Keep only subjects for which we have ALL data (neuroimaging,
        # genetic data, sulci features)
        clinical_data = clinical_df.loc[index]
        BMI = BMI_df.loc[index]

        # Conversion dummy coding
        covar = utils.make_design_matrix(clinical_data,
                                    regressors=clinical_cofounds).as_matrix()

        # Center and scale covariates, but not constant regressor's column
        cov = covar[:, 0:-1]
        skl = StandardScaler()
        cov = skl.fit_transform(cov)

        # Center & scale BMI
        BMI = skl.fit_transform(BMI)

        # Center & scale freesurfer_data
        freesurfer_data = skl.fit_transform(freesurfer_data)

        # Constant regressor to mimick the fit intercept
        constant_regressor = np.ones((freesurfer_data.shape[0], 1))

        # Concatenate BMI, constant regressor and covariates
        design_mat = np.hstack((cov, constant_regressor, BMI))

        X = design_mat
        Y = freesurfer_data

        np.save(os.path.join(SHARED_DIR, 'X.npy'), X)
        np.save(os.path.join(SHARED_DIR, 'Y.npy'), Y)

        print "Data saved"
    else:
        X = np.load(os.path.join(SHARED_DIR, 'X.npy'))
        Y = np.load(os.path.join(SHARED_DIR, 'Y.npy'))
        print "Data read from cache"
    return X, Y


#"""#
#run#
#"""#
if __name__ == "__main__":

    # Set pathes
    WD = "/neurospin/tmp/brainomics/univariate_bmi_Freesurfer_IMAGEN"
    if not os.path.exists(WD):
        os.makedirs(WD)

    print "#################"
    print "# Build dataset #"
    print "#################"

    # Load data
    X, Y = load_residualized_bmi_data(cache=False)
    penalty_start = 11

    # Parameters of subcortical structure of interest
    freesurfer_features = ['lhCortexVol',
                           'rhCortexVol',
                           'CortexVol',
                           'SubCortGrayVol',
                           'TotalGrayVol',
                           'SupraTentorialVol',
                           'lhCorticalWhiteMatterVol',
                           'rhCorticalWhiteMatterVol',
                           'CorticalWhiteMatterVol']

    print "##############################################################"
    print ("# Perform Mass-Univariate Linear Modeling "
           "based Ordinary Least Squares #")
    print "##############################################################"

    # MUOLS
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

        proba.append('%f' % p[i])

    # Beta values: coefficients of the fit
    betas = bigols.coef_

    beta_df = pd.DataFrame(betas[penalty_start:, :].transpose(),
                           index=freesurfer_features,
                           columns=['betas'])

    # Save beta values from the GLM on Freesurfer features as a dataframe
    beta_df.to_csv(os.path.join(OUTPUT_DIR, 'MUOLS_beta_values_df.csv'))
    print "Dataframe containing beta values for each Freesurfer feature has been saved."

    # Write results of MULM computation, i.e. p-value for each feature of
    # interest, in a csv file
    MULM_file_path = os.path.join(OUTPUT_DIR, 'MULM_bmi_freesurfer.txt')

    with open(MULM_file_path, 'wb') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=' ', quotechar=' ')

        for i in np.arange(0, len(proba)):
            spamwriter.writerow(['The probability of MULM computation is:']
                                + [proba[i]] + ['for']
                                + [freesurfer_features[i]]
                                + ['extracted by Freesurfer.'])

    # Bonferroni correction: since we focus here on 9 features extracted by
    # Freesurfer algorithm, we only keep the probability values p < (0.05 / 9)
    # that meet a significance threshold of 0.05 after Bonferroni correction.
    # Write results of MULM computation for each feature of interest in a
    # csv file
    bonferroni_correction = 0.05 / (Y.shape[1])

    MULM_after_Bonferroni_correction_file_path = os.path.join(OUTPUT_DIR,
                                    'MULM_after_Bonferroni_correction.txt')

    with open(MULM_after_Bonferroni_correction_file_path, 'wb') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=' ', quotechar=' ')

        for i in np.arange(0, len(proba)):

            if float(proba[i]) < bonferroni_correction:

                spamwriter.writerow(['The probability of MULM computation is:']
                                    + [float(proba[i]) * Y.shape[1]]
                                    + ['for']
                                    + [freesurfer_features[i]]
                                    + ['extracted by Freesurfer.'])