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
import json
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


def load_globals(config):
    import mapreduce as GLOBAL  # access to global variables
    GLOBAL.DATA = GLOBAL.load_data(config["data"])


def resample(config, resample_nb):
    import mapreduce as GLOBAL  # access to global variables
    #GLOBAL.DATA = GLOBAL.load_data(config["data"])
    resample = config["resample"][resample_nb]
    print "reslicing %d" %resample_nb
    GLOBAL.DATA_RESAMPLED = {k: [GLOBAL.DATA[k][idx, ...] for idx in resample]
                            for k in GLOBAL.DATA}                            
    print "done reslicing %d" %resample_nb


def mapper(key, output_collector):
    import mapreduce as GLOBAL # access to global variables:
    # key: list of parameters
    alpha, l1_ratio = key[0], key[1]
    Xtr = GLOBAL.DATA_RESAMPLED["X"][0]
    Xte = GLOBAL.DATA_RESAMPLED["X"][1]
    ztr = GLOBAL.DATA_RESAMPLED["z"][0]
    zte = GLOBAL.DATA_RESAMPLED["z"][1]
    print key, "Data shape:", Xtr.shape, Xte.shape, ztr.shape, zte.shape
    #
    mod = estimators.ElasticNet(l1_ratio, alpha, penalty_start = 11, mean = True)     #since we residualized BMI with 2 categorical covariables (Gender and ImagingCentreCity - 8 columns) and 2 ordinal variables (tiv_gaser and mean_pds - 2 columns)
    z_pred = mod.fit(Xtr,ztr).predict(Xte)
    ret = dict(z_pred=z_pred, z_true=zte, beta=mod.beta)
    output_collector.collect(key, ret)


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

    print "#################"
    print "# Build dataset #"
    print "#################"
    X_init, X_res, z = load_residualized_bmi_data(cache=False)   #BMI has been residualized when looking for the optimum set of hyperparameters

    print "#####################"
    print "# Build config file #"
    print "#####################"
    ## Parameterize the mapreduce
    alpha = 0.009
    l1_ratio = 0.5
    user_func_filename = os.path.join("/home/hl237680",
        "gits", "scripts", "2013_imagen_bmi", "scripts", 
        "16_hot_spot_mapping.py")
    print "user_func", user_func_filename
    # Use relative path from config.json
    config = dict(data=dict(X_res='X_res.npy', z='z.npy'),
                  params=[[alpha, l1_ratio]],
                  structure="",
                  map_output="results",
                  user_func=user_func_filename,
                  reduce_input="results/*/*", 
                  reduce_group_by="results/.*/(.*)",
                  reduce_output="results.csv")
    json.dump(config, open(os.path.join(WD, "config.json"), "w"))

    #########################################################################
    # Build utils files: sync (push/pull) and PBS
    sys.path.append(os.path.join(os.getenv('HOME'),
                                'gits','scripts'))
    import brainomics.cluster_gabriel as clust_utils
    sync_push_filename, sync_pull_filename, WD_CLUSTER = \
        clust_utils.gabriel_make_sync_data_files(WD, user="hl237680")
    cmd = "mapreduce.py -m %s/config.json  --ncore 12" % WD_CLUSTER
    clust_utils.gabriel_make_qsub_job_files(WD, cmd)
    #########################################################################
    # Sync to cluster
    print "Sync data to gabriel.intra.cea.fr: "
    os.system(sync_push_filename)
    
    #########################################################################
    print "mapreduce.py -m %s/config.json --ncore 12" % WD
    print sync_pull_filename
    
    #########################################################################
#    print "# Reduce"
#    print "mapreduce.py -r %s/config.json" % WD
    #########################################################################

    beta_map = np.zeros(X_init.shape[1])
    template_for_size_img = ni.load(MASK_PATH)
    
    mask_data = template_for_size_img.get_data()
    masked_data_index = (mask_data != 0.0)

    image = np.zeros(template_for_size_img.get_data().shape)
    image[masked_data_index] = beta_map
    BMI_beta_map = os.path.join(BASE_PATH, 'results', 'BMI_beta_map.nii.gz')
    ni.save(ni.Nifti1Image(image, template_for_size_img.get_affine()), BMI_beta_map)
