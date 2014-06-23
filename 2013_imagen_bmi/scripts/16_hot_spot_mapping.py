# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 09:27:49 2014

@author: hl237680

Mapping of hot spots which have been revealed by high correlation ratio
after optimization of (alpha, l1) hyperparameters using Enet algorithm.
This time, we have determined the optimum hyperparameters, so that we can
run the model on the whole data.
"""

import os, sys
import numpy as np
import pandas as pd
import tables
import nibabel as ni
import parsimony.estimators as estimators

sys.path.append(os.path.join(os.getenv('HOME'), 'gits', 'scripts', '2013_imagen_bmi', 'scripts'))
import bmi_utils
sys.path.append(os.path.join(os.environ["HOME"], "gits", "scripts", "2013_imagen_subdepression", "lib"))
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
SHARED_DIR = os.path.join(BASE_SHARED_DIR, 'residualized_bmi_cache')


#############
# Read data #
#############
def load_residualized_bmi_data(cache):
    if not(cache):
        #SNPs = pd.io.parsers.read_csv(os.path.join(DATA_PATH, "SNPs.csv"), dtype='float64', index_col=0).as_matrix()
        BMI = pd.io.parsers.read_csv(os.path.join(DATA_PATH, "BMI.csv"), index_col=0).as_matrix()
        # Dataframe      
        COFOUND = ["Subject", "Gender de Feuil2", "ImagingCentreCity", "tiv_gaser", "mean_pds"]
        df = pd.io.parsers.read_csv(os.path.join(CLINIC_DATA_PATH, "1534bmi-vincent2.csv"), index_col=0)
        df = df[COFOUND]
        # Conversion dummy coding
        design_mat = utils.make_design_matrix(df, regressors=COFOUND).as_matrix()
        # Keep only subjects for which we have all data and remove the 1. column containing subject_id from the numpy array design_mat
        subjects_id = np.genfromtxt(os.path.join(DATA_PATH, "subjects_id.csv"), dtype=None, delimiter=',', skip_header=1)
        design_mat = np.delete(np.delete(design_mat, np.where(np.in1d(design_mat[:,0], np.delete(design_mat, np.where(np.in1d(design_mat[:,0], subjects_id)), 0))), 0),0,1)            
        # Images
        h5file = tables.openFile(IMAGES_FILE)
        masked_images = bmi_utils.read_array(h5file, "/standard_mask/residualized_images_gender_center_TIV_pds")    #images already masked
        print "Data loaded - Processing"
        # Concatenate images with covariates gender, imaging city centrr, tiv_gaser and mean pds status in order to do as though BMI had been residualized
        X_init = masked_images
        z = BMI
        X_cov = np.concatenate((design_mat, masked_images), axis=1) #added non interest cov in order to residualize BMI "by hand"
        X_res = np.hstack((np.ones((z.shape[0],1)),X_cov))  #added ones' column for later penalties using parsimony
        #Y = SNPs
        np.save(os.path.join(SHARED_DIR, "X_init.npy"), X_init)
        np.save(os.path.join(SHARED_DIR, "X_res.npy"), X_res)
        #np.save(os.path.join(SHARED_DIR, "Y.npy"), Y)
        np.save(os.path.join(SHARED_DIR, "z.npy"), z)
        h5file.close()        
        print "Data saved"
    else:
        X = np.load(os.path.join(SHARED_DIR, "X.npy"))        
        #Y = np.load(os.path.join(SHARED_DIR, "Y.npy"))
        z = np.load(os.path.join(SHARED_DIR, "z.npy"))        
        print "Data read from cache"    
    return X, z


#######
# Run #
#######
if __name__ == "__main__":
    
    WD = "/neurospin/tmp/brainomics/hot_spots"
    if not os.path.exists(WD): os.makedirs(WD)
    
    # Load data
    X_init, X_res, z = load_residualized_bmi_data(cache=False)   #BMI has been residualized when looking for the optimum set of hyperparameters

    # Initialize beta_map
    beta_map = np.zeros(X_init.shape[1])

    # Elasticnet algorithm via Pylearn-Parsimony
    alpha = 0.009
    l1_ratio = 0.5
    mod = estimators.ElasticNet(l1_ratio, alpha, penalty_start = 11, mean = True)     #since we residualized BMI with 2 categorical covariables (Gender and ImagingCentreCity - 8 columns) and 2 ordinal variables (tiv_gaser and mean_pds - 2 columns)
    beta_map = mod.beta

    # Use mask
    template_for_size_img = ni.load(MASK_PATH)
    mask_data = template_for_size_img.get_data()
    masked_data_index = (mask_data != 0.0)

    # Draw beta map
    image = np.zeros(template_for_size_img.get_data().shape)
    image[masked_data_index] = beta_map
    BMI_beta_map = os.path.join(BASE_PATH, 'results', 'BMI_beta_map.nii.gz')
    ni.save(ni.Nifti1Image(image, template_for_size_img.get_affine()), BMI_beta_map)