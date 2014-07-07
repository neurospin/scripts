# -*- coding: utf-8 -*-
"""
Created on Fri Jul  4 16:29:44 2014

@author: hl237680

Univariate correlation between residualized BMI and images.
This script aims at checking out whether the crowns that appear on beta maps
obtained by multivariate analysis should be considered as an artifact due to
the segmentation process or are to be taken into account.
"""

import os, sys
import numpy as np
import nibabel as ni
import pandas as pd
import tables

from mulm.models import MUOLS

from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.join(os.getenv('HOME'), 'gits', 'scripts',
                             '2013_imagen_bmi', 'scripts'))
import bmi_utils
    
sys.path.append(os.path.join(os.environ["HOME"], "gits", "scripts",
                             "2013_imagen_subdepression", "lib"))
import utils


#############
# Read data #
#############
# SNPs and BMI
def load_residualized_bmi_data(cache):
    if not(cache):
        # BMI
        BMI = pd.io.parsers.read_csv(os.path.join(DATA_PATH, "BMI.csv"),
                                     index_col=0).as_matrix()

        # Dataframe
        COFOUND = ["Gender de Feuil2", "ImagingCentreCity", "tiv_gaser",
                   "mean_pds"]
        df = pd.io.parsers.read_csv(os.path.join(CLINIC_DATA_PATH,
                                                 "1534bmi-vincent2.csv"),
                                                 index_col=0)
        df = df[COFOUND]

        # Keep only subjects for which we have all data
        subjects_id = np.genfromtxt(os.path.join(DATA_PATH, "subjects_id.csv"),
                                    dtype=None, delimiter=',', skip_header=1)

        clinic_data = df.loc[subjects_id]

        # Conversion dummy coding
        covar = utils.make_design_matrix(clinic_data,
                                         regressors=COFOUND).as_matrix()

        # Concatenate BMI and covariates
        # (gender, imaging city centre, tiv_gaser and mean pds status)
        design_mat = np.hstack((covar, BMI))

        # Images
        h5file = tables.openFile(IMAGES_FILE)
        masked_images = bmi_utils.read_array(h5file,
                "/standard_mask/residualized_images_gender_center_TIV_pds")
        print "Images loaded"
        
        X = design_mat
        # Center & scale X
        skl = StandardScaler()
        X = skl.fit_transform(X)
        Y = masked_images
        
        np.save(os.path.join(SHARED_DIR, "X.npy"), X)
        np.save(os.path.join(SHARED_DIR, "Y.npy"), Y)
        h5file.close()
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
    WD = "/neurospin/tmp/brainomics/univariate_residual_bmi_images"
    if not os.path.exists(WD):
        os.makedirs(WD)
    
    print "#################"
    print "# Build dataset #"
    print "#################"

    # Pathnames
    BASE_PATH = '/neurospin/brainomics/2013_imagen_bmi/'
    DATA_PATH = os.path.join(BASE_PATH, 'data')
    CLINIC_DATA_PATH = os.path.join(DATA_PATH, 'clinic')
    IMAGES_FILE = os.path.join(DATA_PATH, 'smoothed_images.hdf5')
    BMI_FILE = os.path.join(DATA_PATH, 'BMI.csv')

    # Shared data
    BASE_SHARED_DIR = "/neurospin/tmp/brainomics/"
    SHARED_DIR = os.path.join(BASE_SHARED_DIR, 'residualized_bmi_cache')
    if not os.path.exists(SHARED_DIR):
        os.makedirs(SHARED_DIR)

    X, Y = load_residualized_bmi_data(cache=False)
    n, p = Y.shape
    np.save(os.path.join(WD, 'X.npy'), X)
    np.save(os.path.join(WD, "Y.npy"), Y)
   
    print "########################################################################"
    print "# Perform Mass-Univariate Linear Modeling based Ordinary Least Squares #"
    print "########################################################################"

    #MUOLS
    beta_map = np.zeros(p) 
    bigols = MUOLS()
    bigols.fit(X, Y)
    s, p = bigols.stats_t_coefficients(X, Y, contrast=[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.], pval=True)
    beta_map[:] = p[:]

    template_for_size = os.path.join(DATA_PATH, 'mask', 'mask.nii')
    template_for_size_img = ni.load(template_for_size)
    mask_data = template_for_size_img.get_data()
    masked_data_index = (mask_data == 1.0)
    
    image = np.zeros(template_for_size_img.get_data().shape)
    image[masked_data_index] = beta_map
    filename = os.path.join(BASE_PATH, 'results', 'beta_map_MULM.nii.gz')
    ni.save(ni.Nifti1Image(image, template_for_size_img.get_affine()), filename)
    print "Beta map obtained with MULM has been saved."