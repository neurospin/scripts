# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 09:27:49 2014

@author: hl237680

Work on normal group.
Mapping of hot spots which have been revealed by high correlation ratio
after optimization of (alpha, l1) hyperparameters using Enet algorithm.
This time, we have determined the optimum hyperparameters, so that we can
run the model on the whole dataset.
"""

import os
import sys
import numpy as np
import pandas as pd
import tables
import nibabel as ni
import parsimony.estimators as estimators
from sklearn.metrics import r2_score

sys.path.append(os.path.join(os.getenv('HOME'), 'gits', 'scripts',
                             '2013_imagen_bmi', 'scripts'))
import bmi_utils

sys.path.append(os.path.join(os.environ['HOME'], 'gits', 'scripts',
                             '2013_imagen_subdepression', 'lib'))
import utils

# Pathnames
BASE_PATH = '/neurospin/brainomics/2013_imagen_bmi/'
DATA_PATH = os.path.join(BASE_PATH, 'data')
CLINIC_DATA_PATH = os.path.join(DATA_PATH, 'clinic')
MASK_PATH = os.path.join(DATA_PATH, 'mask', 'mask.nii')
IMAGES_FILE = os.path.join(DATA_PATH, 'smoothed_images.hdf5')
BMI_FILE = os.path.join(DATA_PATH, 'BMI.csv')

ORIGIN_IMG_DIR = os.path.join(BASE_PATH, '2013_imagen_bmi', 'data',
                                      'VBM', 'new_segment_spm8')
# Shared data
BASE_SHARED_DIR = "/neurospin/tmp/brainomics/"
SHARED_DIR = os.path.join(BASE_SHARED_DIR,
                          'residualized_bmi_cache_normal_group')


#############
# Read data #
#############
def load_residualized_bmi_data(cache):
    if not(cache):
#        SNPs = pd.io.parsers.read_csv(os.path.join(DATA_PATH, 'SNPs.csv'),
#                                      dtype='float64',
#                                      index_col=0).as_matrix()
        BMI = pd.io.parsers.read_csv(os.path.join(DATA_PATH, 'BMI.csv'),
                                     index_col=0).as_matrix()

        # Dataframe
        df = pd.io.parsers.read_csv(os.path.join(CLINIC_DATA_PATH,
                                                 'normal_group.csv'),
                                    index_col=0)

        COFOUND = ['Gender de Feuil2',
                   'ImagingCentreCity',
                   'tiv_gaser',
                   'mean_pds']

        df = df[COFOUND]

        # Conversion dummy coding
        covar = utils.make_design_matrix(df, regressors=COFOUND).as_matrix()

        # Images
        h5file = tables.openFile(IMAGES_FILE)
        images_file = bmi_utils.read_array(h5file,
                "/standard_mask/residualized_images_gender_center_TIV_pds")
                 #images already masked

        masked_images = images_file[???????????, :]

        print "Data loaded - Processing"

        z = BMI
        # Concatenate images and covariates
        # (gender, imaging city centre, tiv_gaser and mean pds status)
        # in order to do as though BMI had been residualized.
        X_res = np.hstack((covar, masked_images))

        np.save(os.path.join(SHARED_DIR, "X_res.npy"), X_res)
        np.save(os.path.join(SHARED_DIR, "z.npy"), z)

        h5file.close()
        print "Data saved"
    else:
        X_res = np.load(os.path.join(SHARED_DIR, "X_res.npy"))
        z = np.load(os.path.join(SHARED_DIR, "z.npy"))
        print "Data read from cache"
    return X_res, z


#######
# Run #
#######
if __name__ == "__main__":

    # Load data
    # BMI has been residualized when looking for the optimum set of
    # hyperparameters
    X_res, z = load_residualized_bmi_data(cache=False)
    # Since we residualized BMI with 2 categorical covariables (Gender and
    # ImagingCentreCity - 8 columns) and 2 ordinal variables (tiv_gaser and
    # mean_pds - 2 columns), penalty_start = 11
    penalty_start = 11

    # Initialize beta_map
    beta_map = np.zeros(X_res.shape[1] - penalty_start)

    # Elasticnet algorithm via Pylearn-Parsimony
    print "Elasticnet algorithm"
    alpha = 0.006
    l1_ratio = 0.8
    #l1_ratio = 0
    mod = estimators.ElasticNet(l1_ratio, alpha, penalty_start=penalty_start,
                                mean=True)
    mod.fit(X_res, z)
    print "Compute beta values"
    beta_map = mod.beta
    print "Compute R2"
    r2 = r2_score(z, mod.predict(X_res))
    print r2

    # Use mask
    template_for_size_img = ni.load(MASK_PATH)
    mask_data = template_for_size_img.get_data()
    masked_data_index = (mask_data != 0.0)

    # Draw beta map
    print "Draw beta map"
    image = np.zeros(template_for_size_img.get_data().shape)
    image[masked_data_index] = beta_map[11:, 0]
    BMI_beta_map = os.path.join(BASE_PATH, 'results',
                                'BMI_beta_map_0.006_0.8_normal_group.nii.gz')
    ni.save(ni.Nifti1Image(image, template_for_size_img.get_affine()),
            BMI_beta_map)
    print "Beta map for normal group has been saved."