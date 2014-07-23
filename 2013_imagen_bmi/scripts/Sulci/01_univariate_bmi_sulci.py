# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 09:49:41 2014

@author: hl237680

Univariate correlation between residualized BMI and sulci features (mean
geodesic depth, gray matter thickness, fold opening) on IMAGEN subjects.
The resort to sulci -instead of considering images of anatomical structures-
should prevent us from the artifacts that may be induced by the normalization
step of the segmentation process.
"""

import os, sys
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
# SNPs and BMI
def load_residualized_bmi_data(sulci_filename, cache):
    if not(cache):

        # BMI
        BMI_df = pd.io.parsers.read_csv(os.path.join(DATA_PATH, "BMI.csv"),
                                     sep=',',
                                     index_col=0)

        # Sulcus: dataframe for study of sulci
        sulci_df = pd.io.parsers.read_csv(os.path.join(SULCI_PATH,
                                                       sulci_filename),
                                          sep=';',
                                          index_col=0)

        # Sulci feature of interest among geodesic depth mean ('depthMean'),
        # gray matter thickness ('GM_thickness') and opening ('opening')
        sulci_feature = ['depthMean']
        #sulci_feature = ['GM_thickness']
        #sulci_feature = ['opening']

        sulci_df = sulci_df[sulci_feature]

        # Dataframe for picking out only clinical cofounds of non interest
        clinical_df = pd.io.parsers.read_csv(os.path.join(CLINIC_DATA_PATH,
                                                          'population.csv'),
                                             index_col=0)

        clinical_cofounds = ['Gender de Feuil2', 'ImagingCentreCity',
                             'tiv_gaser', 'mean_pds']

        clinical_df = clinical_df[clinical_cofounds]

        # Consider subjects for which we have neuroimaging and genetic data
        subjects_id = np.genfromtxt(os.path.join(DATA_PATH,
                                                 "subjects_id.csv"),
                                    dtype=None,
                                    delimiter=',',
                                    skip_header=1)

        sulci_data = sulci_df.loc[subjects_id]

        # Drop rows that have any NaN values
        sulci_data = sulci_data.dropna()

        # Get indices of subjects fot which we have both neuroimaging and
        # genetic data, but also sulci features
        index = sulci_data.index

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

        Y = sulci_data

        np.save(os.path.join(SHARED_DIR, "X.npy"), X)
        np.save(os.path.join(SHARED_DIR, "Y.npy"), Y)

        print "Data saved"
    else:
        X = np.load(os.path.join(SHARED_DIR, "X.npy"))
        Y = np.load(os.path.join(SHARED_DIR, "Y.npy"))
        print "Data read from cache"
    return X, Y



#"""#
#run#
#"""#
if __name__ == "__main__":

    ## Set pathes
    WD = "/neurospin/tmp/brainomics/univariate_bmi_sulci_IMAGEN"
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
    SULCI_FILENAMES = os.listdir(SULCI_PATH)

    # Shared data
    BASE_SHARED_DIR = "/neurospin/tmp/brainomics/"
    SHARED_DIR = os.path.join(BASE_SHARED_DIR,
                              'bmi_sulci_cache_IMAGEN')
    if not os.path.exists(SHARED_DIR):
        os.makedirs(SHARED_DIR)

    stat = []
    proba = []

    for i in SULCI_FILENAMES:
        print "Sulcus considered:", i
        X, Y = load_residualized_bmi_data(i, cache=False)

#    n, p = Y.shape
#    np.save(os.path.join(WD, 'X.npy'), X)
#    np.save(os.path.join(WD, "Y.npy"), Y)

        print "########################################################################"
        print "# Perform Mass-Univariate Linear Modeling based Ordinary Least Squares #"
        print "########################################################################"
    
        #MUOLS
        bigols = MUOLS()
        bigols.fit(X, Y)
        s, p = bigols.stats_t_coefficients(X, Y,
                            contrast=[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.],
                            pval=True)
        #stat.append(s)
        proba.append(p)
    
    sulcus = SULCI_FILENAMES[proba.index(max(proba))]
    print "Sulcus whose mean geodesic depth is the most associated to BMI:", sulcus