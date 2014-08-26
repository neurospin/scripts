# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 17:29:01 2014

@author: hl237680

Univariate correlation between residualized BMI and some sulci of interest
on IMAGEN subjects.
The selected sulci are particularly studied because of their robustness to
the segmentation process. These sulci are respectively split into various
subsamples by the segmentation process. As a results, they have previously
been gathered again.
Here, we select the central, precentral, collateral sulci and the calloso-
marginal fissure.
NB: Their features have previously been filtered by the quality control step.
(cf 00_quality_control.py)

The resort to sulci -instead of considering images of anatomical structures-
should prevent us from the artifacts that may be induced by the normalization
step of the segmentation process.

INPUT:
- /neurospin/brainomics/2013_imagen_bmi/data/Imagen_mainSulcalMorphometry/
  full_sulci/Quality_control/sulci_df_qc.csv:
    sulci features after quality control

- /neurospin/brainomics/2013_imagen_bmi/data/BMI.csv:
    BMI of the 1265 subjects for which we also have neuroimaging data

METHOD: MUOLS

NB: Subcortical features, BMI and covariates are centered-scaled.

OUTPUT: returns a probability that the feature of interest of the selected
        sulci is significantly associated to BMI.

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


#############
# Read data #
#############
# Sulci and BMI
def load_residualized_bmi_data(cache):
    if not(cache):

        # BMI
        BMI_df = pd.io.parsers.read_csv(os.path.join(DATA_PATH, 'BMI.csv'),
                                     sep=',',
                                     index_col=0)

        # Sulci features
        sulci_df_qc = pd.io.parsers.read_csv(os.path.join(QC_PATH,
                                                          'sulci_df_qc.csv'),
                                              sep=',',
                                              index_col=0)
        colnames = sulci_df_qc.columns

        # Dataframe for picking out only clinical cofounds of non interest
        clinical_df = pd.io.parsers.read_csv(os.path.join(CLINIC_DATA_PATH,
                                                          'population.csv'),
                                             index_col=0)

        # Add one cofound since sulci follows a power law
        clinical_df['tiv2'] = pow(clinical_df['tiv_gaser'], 2)

        clinical_cofounds = ['Gender de Feuil2',
                             'ImagingCentreCity',
                             'tiv_gaser',
                             'tiv2',
                             'mean_pds']

        clinical_df = clinical_df[clinical_cofounds]

        # Consider subjects for whom we have neuroimaging and genetic data
        subjects_id = np.genfromtxt(os.path.join(DATA_PATH,
                                                 'subjects_id.csv'),
                                    dtype=None,
                                    delimiter=',',
                                    skip_header=1)

        # Get the intersept of indices of subjects for whom we have
        # neuroimaging and genetic data, but also sulci features
        subjects_index = np.intersect1d(subjects_id, sulci_df_qc.index.values)

        # Check whether all these subjects are actually stored into the qc
        # dataframe
        sulci_data = sulci_df_qc.loc[subjects_index]

        # Keep only subjects for which we have ALL data (neuroimaging,
        # genetic data and sulci features)
        clinical_data = clinical_df.loc[subjects_index]
        BMI = BMI_df.loc[subjects_index]

        # Conversion dummy coding
        covar = utils.make_design_matrix(clinical_data,
                                    regressors=clinical_cofounds).as_matrix()

        # Center and scale covariates, but not constant regressor's column
        cov = covar[:, 0:-1]
        skl = StandardScaler()
        cov = skl.fit_transform(cov)

        # Center & scale BMI
        BMI = skl.fit_transform(BMI)

        # Center & scale sulci_data
        sulci_data = skl.fit_transform(sulci_data)
        print "sulci_data loaded"
        # Constant regressor to mimick the fit intercept
        constant_regressor = np.ones((sulci_data.shape[0], 1))

        # Concatenate BMI, constant regressor and covariates
        design_mat = np.hstack((cov, constant_regressor, BMI))

        X = design_mat
        Y = sulci_data

        np.save(os.path.join(SHARED_DIR, 'X.npy'), X)
        np.save(os.path.join(SHARED_DIR, 'Y.npy'), Y)

        print "Data saved"
    else:
        X = np.load(os.path.join(SHARED_DIR, 'X.npy'))
        Y = np.load(os.path.join(SHARED_DIR, 'Y.npy'))
        print "Data read from cache"
    return X, Y, colnames


#"""#
#run#
#"""#
if __name__ == "__main__":

    ## Set pathes
    WD = "/neurospin/tmp/brainomics/univariate_bmi_full_sulci_IMAGEN"
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
    SULCI_PATH = os.path.join(DATA_PATH, 'Imagen_mainSulcalMorphometry')
    FULL_SULCI_PATH = os.path.join(SULCI_PATH, 'full_sulci')
    QC_PATH = os.path.join(FULL_SULCI_PATH, 'Quality_control')

    # Output results
    OUTPUT_DIR = os.path.join(FULL_SULCI_PATH, 'Results')
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Shared data
    BASE_SHARED_DIR = "/neurospin/tmp/brainomics/"
    SHARED_DIR = os.path.join(BASE_SHARED_DIR,
                              'bmi_full_sulci_cache_IMAGEN')
    if not os.path.exists(SHARED_DIR):
        os.makedirs(SHARED_DIR)

    X, Y, colnames = load_residualized_bmi_data(cache=False)

    print "##############################################################"
    print ("# Perform Mass-Univariate Linear Modeling "
           "based Ordinary Least Squares #")
    print "##############################################################"

    #MUOLS
    bigols = MUOLS()
    bigols.fit(X, Y)
    s, p = bigols.stats_t_coefficients(X, Y,
#                contrast=[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
            # if add one more cofound
            contrast=[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
            pval=True)

    proba = []
    for i in np.arange(0, p.shape[0]):
        if (p[i] > 0.95):
            p[i] = 1 - p[i]
        proba.append('%.15f' % p[i])

    # Write results of MULM computation for each feature of interest in a
    # csv file
    MULM_file_path = os.path.join(OUTPUT_DIR, 'MULM_bmi_full_sulci.txt')
    with open(MULM_file_path, 'wb') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=' ', quotechar=' ')
        for i in np.arange(0, len(proba)):
            sulcus_name = colnames[proba.index(proba[i])][11:]
            spamwriter.writerow(['The MULM probability for the feature:']
                                + [sulcus_name]
                                + ['of the sulcus']
                                + [sulcus_name]
                                + ['is']
                                + [proba[i]]
                                #+ ['\n']
                                )