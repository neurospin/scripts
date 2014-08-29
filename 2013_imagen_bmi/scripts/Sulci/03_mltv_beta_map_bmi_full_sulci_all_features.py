# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 12:09:39 2014

@author: hl237680

Mapping of hot spots which have been revealed by high correlation ratio
after optimization of (alpha, l1) hyperparameters using Enet algorithm.
This time, we have determined the optimum hyperparameters, so that we can
run the model on the whole dataset.

The selected sulci are robust to the segmentation process. These sulci are
respectively split into various subsamples by the segmentation process. As
a results, they have previously been gathered again.
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

METHOD: multivariate GLM

NB: Subcortical features, BMI and covariates are centered-scaled.

OUTPUT:
- /neurospin/brainomics/2013_imagen_bmi/data/Imagen_mainSulcalMorphometry/
  full_sulci/Results/mltv_beta_values_df.csv:
    returns for each sulcus the beta value associated to the GLM
    determined by the selected optimized set of hyperparameters.

"""


import os
import sys
import numpy as np
import pandas as pd
import parsimony.estimators as estimators
from sklearn.metrics import r2_score

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
RESULTS_PATH = os.path.join(FULL_SULCI_PATH, 'Results')

# Shared data
BASE_SHARED_DIR = "/neurospin/tmp/brainomics/"
SHARED_DIR = os.path.join(BASE_SHARED_DIR,
                          'bmi_full_sulci_all_features_cache')
if not os.path.exists(SHARED_DIR):
    os.makedirs(SHARED_DIR)


#############
# Read data #
#############
def load_full_sulci_all_features_bmi_data(cache):
    if not(cache):
        # BMI
        BMI_df = pd.io.parsers.read_csv(os.path.join(DATA_PATH, 'BMI.csv'),
                                     sep=',',
                                     index_col=0)

        # Sulci features
        labels = np.genfromtxt(os.path.join(QC_PATH, 'sulci_df_qc.csv'),
                                dtype=None,
                                delimiter=',',
                                skip_header=1,
                                usecols=0).tolist()

        sulci_index = pd.Index(labels)

        # Sulci features
        sulci_df_qc = pd.io.parsers.read_csv(os.path.join(QC_PATH,
                                                          'sulci_df_qc.csv'),
                              sep=',',
                              index_col=0)

        # Set the new dataframe index: subjects ID in the right format
        sulci_df_qc = sulci_df_qc.set_index(sulci_index)

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
        print "Sulci_data loaded"

        # Constant regressor to mimick the fit intercept
        constant_regressor = np.ones((sulci_data.shape[0], 1))

        # Concatenate sulci data, constant regressor and covariates
        design_mat = np.hstack((cov, constant_regressor, sulci_data))

        X = design_mat
        z = BMI

        sulci_df = pd.DataFrame(sulci_df_qc, index=subjects_index)

        np.save(os.path.join(SHARED_DIR, 'X.npy'), X)
        np.save(os.path.join(SHARED_DIR, 'z.npy'), z)

        print "Data saved"
    else:
        X = np.load(os.path.join(SHARED_DIR, 'X.npy'))
        z = np.load(os.path.join(SHARED_DIR, 'z.npy'))
        print "Data read from cache"
    return X, z, sulci_df


#######
# Run #
#######
if __name__ == "__main__":
    # Load data
    X, z, sulci_df = load_full_sulci_all_features_bmi_data(cache=False)

    # Initialize beta_map
    beta_map = np.zeros(X.shape[1])

    # Elasticnet algorithm via Pylearn-Parsimony
    print "Elasticnet algorithm"
    alpha = 0.1
    l1_ratio = 0.3
    # BMI has been residualized with 2 categorical covariables
    # (Gender and ImagingCentreCity - 8 columns) and 3 ordinal variables
    # (tiv_gaser, tiv² and mean_pds - 3 columns)
    penalty_start = 12
    mod = estimators.ElasticNet(l1_ratio,
                                alpha,
                                penalty_start=penalty_start,
                                mean=True)

    print "Compute beta values"
    mod.fit(X, z)
    beta_map = mod.beta

    print "Compute R2"
    r2 = r2_score(z, mod.predict(X))
    print r2

    beta_df = pd.DataFrame(beta_map[penalty_start:],
                           index=sulci_df.columns,
                           columns=['betas'])

    # Save beta values from the GLM on sulci features as a dataframe
    beta_df.to_csv(os.path.join(RESULTS_PATH, 'mltv_beta_values_df.csv'))
    print "Dataframe containing beta values for each sulcus has been saved."

#    # To get the max of positive beta values:
#    beta_df.max(axis=0)
#    # and the index corresponding to this maximum:
#    beta_df.idxmax(axis=0)
#
#    # To get the max of negative beta values:
#    beta_df.min(axis=0)
#    # and the index corresponding to this minimum:
#    beta_df.idxmin(axis=0)