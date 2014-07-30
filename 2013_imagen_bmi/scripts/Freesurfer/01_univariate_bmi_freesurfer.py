# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 13:43:46 2014

@author: hl237680

Univariate correlation between residualized BMI and volume of subcortical
structures (Freesurfer) on IMAGEN subjects.

The resort to Freesurfer should prevent us from the artifacts that may be
induced by the normalization step of the SPM segmentation process.

INPUT:
- /neurospin/brainomics/2013_imagen_bmi/data/Freesurfer:
   .csv file containing volume of subcortical structures obtained by Freesurfer
- /neurospin/brainomics/2013_imagen_bmi/data/BMI.csv:
   BMI of the 1265 subjects for which we also have neuroimaging data

METHOD: MUOLS

OUTPUT: returns a probability for each subcortical structure to be
        significantly associated to BMI.

"""

import os
import sys
import numpy as np
import pandas as pd

from mulm.models import MUOLS

from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.join(os.environ["HOME"], "gits", "scripts",
                             "2013_imagen_subdepression", "lib"))
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

        # Parameters of subcortical structure of interest
        freesurfer_feature = ['SubCortGrayVol']
        # Other parameters to be considered
        #'ICV', 'lhCortexVol', 'rhCortexVol', 'CortexVol', 'SubCortGrayVol',
        #'TotalGrayVol', 'SupraTentorialVol', 'lhCorticalWhiteMatterVol',
        #'rhCorticalWhiteMatterVol', 'CorticalWhiteMatterVol'

        freesurfer_df = freesurfer_df[freesurfer_feature]

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

        # Concatenate BMI and covariates
        design_mat = np.hstack((covar, BMI))

        X = design_mat
        # Center & scale X
        skl = StandardScaler()
        X = skl.fit_transform(X)

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

    ## Set pathes
    WD = "/neurospin/tmp/brainomics/univariate_bmi_Freesurfer_IMAGEN"
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

    X, Y = load_residualized_bmi_data(cache=False)
    np.save(os.path.join(WD, 'X.npy'), X)
    np.save(os.path.join(WD, 'Y.npy'), Y)

    stat = []
    proba = []

    print "##############################################################"
    print ("# Perform Mass-Univariate Linear Modeling "
           "based Ordinary Least Squares #")
    print "##############################################################"

    #MUOLS
    bigols = MUOLS()
    bigols.fit(X, Y)
    s, p = bigols.stats_t_coefficients(X, Y,
                    contrast=[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
                    pval=True)
    #stat.append(s)
    proba.append(p)

    print "The minimum probability is:", min(min(proba))