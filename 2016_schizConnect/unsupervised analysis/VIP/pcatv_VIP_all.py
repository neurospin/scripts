#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 18:11:55 2017

@author: ad247405
"""

import os
import json
import time
import shutil
from itertools import product
from collections import OrderedDict
import numpy as np
import scipy
import pandas as pd
import sklearn.decomposition
from sklearn.cross_validation import StratifiedKFold
import nibabel
import parsimony.functions.nesterov.tv
import pca_tv
import metrics
from brainomics import array_utils
import brainomics.cluster_gabriel as clust_utils
import nibabel as nib
import parsimony.functions.nesterov.tv as tv_helper
################
# Input & output #
##################

INPUT_BASE_DIR ='/neurospin/brainomics/2016_schizConnect/analysis/VIP/Freesurfer'
INPUT_MASK = '/neurospin/brainomics/2016_schizConnect/analysis/VIP/Freesurfer/data/mask.npy'   
INPUT_DATA_X = '/neurospin/brainomics/2016_schizConnect/analysis/VIP/Freesurfer/data/X.npy'                     
OUTPUT_DIR = '/neurospin/brainomics/2016_schizConnect/analysis/VIP/Freesurfer/results/pcatv_all'


#############################################################################

##############
# Parameters #
##############

# Parameters for the function create_config
# Note that value at index 1 will be the name of the task on the cluster
CONFIGS = [[5,"pcattv_FS_VIP_all", "config.json", True]]
N_COMP = 10


PARAMS= [('struct_pca', 0.1, 0.5, 0.5),('struct_pca', 0.1, 0.5, 0.8),\
('struct_pca', 0.1, 0.5, 0.1),('struct_pca', 0.1, 0.1, 0.8),\
('struct_pca', 0.1, 0.1, 0.1),('struct_pca', 0.1, 0.1, 0.5),\
('struct_pca', 0.1, 0.8, 0.1),('struct_pca', 0.1, 0.8, 0.5),('struct_pca', 0.1, 0.8, 0.8),("pca", 0.0, 0.0, 0.0)]


#            

#############
# Functions #
#############

def load_globals(config):
    import mapreduce as GLOBAL  # access to global variables
    GLOBAL.DATA = GLOBAL.load_data(config["data"])
    STRUCTURE = np.load(config["structure"])
    A = tv_helper.A_from_mask(STRUCTURE)
    GLOBAL.Atv = A
    GLOBAL.FULL_RESAMPLE = config['full_resample']



def resample(config, resample_nb):
    import mapreduce as GLOBAL  # access to global variables
    GLOBAL.DATA = GLOBAL.load_data(config["data"])
    resample = config["resample"][resample_nb]
    if resample is not None:
        GLOBAL.DATA_RESAMPLED = {k: [GLOBAL.DATA[k][idx, ...] for idx in resample]
                            for k in GLOBAL.DATA}
    else:  # resample is None train == test
        GLOBAL.DATA_RESAMPLED = {k: [GLOBAL.DATA[k] for idx in [0, 1]]
                            for k in GLOBAL.DATA}


def mapper(key, output_collector):
    import mapreduce as GLOBAL  # access to global variables:
    model_name, global_pen, tv_ratio, l1_ratio = key
    if model_name == 'pca':
        global_pen = tv_ratio = l1_ratio = 0

    if model_name == 'sparse_pca':   
        
        global_pen = tv_ratio = 0
        ll1=l1_ratio 

    if model_name == 'struct_pca':
        ltv = global_pen * tv_ratio
        ll1 = l1_ratio * global_pen * (1 - tv_ratio)
        ll2 = (1 - l1_ratio) * global_pen * (1 - tv_ratio)
        assert(np.allclose(ll1 + ll2 + ltv, global_pen))
        
    penalty_start = 3   
    X_train = GLOBAL.DATA_RESAMPLED["X"][0][:,penalty_start:]
    print (X_train.shape)
    n, p = X_train.shape
    X_test = GLOBAL.DATA_RESAMPLED["X"][1][:,penalty_start:]
    print (X_test.shape)
    Atv = GLOBAL.Atv


    # Fit model
    if model_name == 'pca':
        model = sklearn.decomposition.PCA(n_components=N_COMP)

    if model_name == 'sparse_pca':
        model = sklearn.decomposition.SparsePCA(n_components=N_COMP,alpha = ll1)                                  
                                    
                                    
    if model_name == 'struct_pca':
        model = pca_tv.PCA_L1_L2_TV(n_components=N_COMP,
                                    l1=ll1, l2=ll2, ltv=ltv,
                                    Atv=Atv,
                                    criterion="frobenius",
                                    eps=1e-6,
                                    max_iter=100,
                                    inner_max_iter=int(1e4),
                                    output=False)

    model.fit(X_train)



    # Save the projectors
    if (model_name == 'pca') or (model_name == 'sparse_pca'):
        V = model.components_.T
    if model_name == 'struct_pca' :
        V = model.V

    # Project train & test data
    if (model_name == 'pca')or (model_name == 'sparse_pca'):
        X_train_transform = model.transform(X_train)
        X_test_transform = model.transform(X_test)
        
    if (model_name == 'struct_pca'):
        X_train_transform, _ = model.transform(X_train)
        X_test_transform, _ = model.transform(X_test)

    # Reconstruct train & test data
    # For SparsePCA or PCA, the formula is: UV^t (U is given by transform)
    # For StructPCA this is implemented in the predict method (which uses
    # transform)
    if (model_name == 'pca') or (model_name == 'sparse_pca'):
        X_train_predict = np.dot(X_train_transform, V.T)
        X_test_predict = np.dot(X_test_transform, V.T)
        
    if (model_name == 'struct_pca') :
        X_train_predict = model.predict(X_train)
        X_test_predict = model.predict(X_test)

    # Compute Frobenius norm between original and recontructed datasets
    frobenius_train = np.linalg.norm(X_train - X_train_predict, 'fro')
    frobenius_test = np.linalg.norm(X_test - X_test_predict, 'fro')
    print(frobenius_test) 


    # Compute explained variance ratio
    evr_train = metrics.adjusted_explained_variance(X_train_transform)
    evr_train /= np.var(X_train, axis=0).sum()
    evr_test = metrics.adjusted_explained_variance(X_test_transform)
    evr_test /= np.var(X_test, axis=0).sum()

    # Remove predicted values (they are huge)
    del X_train_predict, X_test_predict

    ret = dict(frobenius_train=frobenius_train,
               frobenius_test=frobenius_test,
               components=V,
               X_train_transform=X_train_transform,
               X_test_transform=X_test_transform,
               evr_train=evr_train,
               evr_test=evr_test)

    output_collector.collect(key, ret)
    
    
def reducer(key, values):
    output_collectors = values
    global N_COMP
    import mapreduce as GLOBAL
    N_FOLDS = GLOBAL.N_FOLDS
    components=np.zeros((67665,N_COMP,5))
    frobenius_train = np.zeros((N_COMP,5))
    frobenius_test = np.zeros((1,5))
    evr_train = np.zeros((N_COMP,5))
    evr_test = np.zeros((N_COMP,5))
    l0 = np.zeros((N_COMP,5))
    l1 = np.zeros((N_COMP,5))
    l2 =np.zeros((N_COMP,5))
    tv = np.zeros((N_COMP,5))
    # N_FOLDS is the number of true folds (not the number of resamplings)
    # key : string of intermediary key
    # load return dict corresponding to mapper ouput. they need to be loaded.]
    # Avoid taking into account the fold 0
    for item in output_collectors:
        if item !=N_FOLDS:
            values = output_collectors[item+1].load()
            components[:,:,item] = values["components"]
            frobenius_train[:,item] = values["frobenius_train"]
            frobenius_test[:,item] = values["frobenius_test"]
#            l0[:,item] = values["l0"]
#            l1[:,item] = values["l1"]
#            l2[:,item] = values["l2"]
#            tv[:,item] = values["tv"]
#            evr_train[:,item] = values["evr_train"]
#            evr_test[:,item] = values["evr_test"]
#            times[:,item] = values["time"]
       
    #Solve non-identifiability problem  (baseline = first fold)
    for i in range(1,5):
        if np.abs(np.corrcoef(components[:,0,0],components[:,0,i])[0,1]) <  np.abs(np.corrcoef(components[:,0,0],components[:,1,i])[0,1]):
            print("components inverted") 
            print(i)
            temp_comp1 = np.copy(components[:,1,i])
            components[:,1,i] = components[:,0,i]
            components[:,0,i] = temp_comp1
            
        if np.abs(np.corrcoef(components[:,1,0],components[:,1,i])[0,1]) <  np.abs(np.corrcoef(components[:,1,0],components[:,2,i])[0,1]):
            print("components inverted") 
            print(i)
            temp_comp2 = np.copy(components[:,2,i])
            components[:,2,i] = components[:,1,i]
            components[:,1,i] = temp_comp2    
            
    # Thesholded components (list of tuples (comp, threshold))
    thresh_components = np.empty(components.shape)
    thresholds = np.empty((N_COMP, N_FOLDS))
    for l in range(N_FOLDS):
        for k in range(N_COMP):
            thresh_comp, t = array_utils.arr_threshold_from_norm2_ratio(
                                components[:, k, l],
                                .99)
            thresh_components[:, k, l] = thresh_comp
            thresholds[k, l] = t
    # Average precision/recall across folds for each component
    av_frobenius_train = frobenius_train.mean(axis=1)
    av_frobenius_test = frobenius_test.mean(axis=1)

    # Align sign of loading vectors to the first fold for each component
    aligned_thresh_comp = np.copy(thresh_components)
    REF_FOLD_NUMBER = 0
    for k in range(N_COMP):
        for i in range(N_FOLDS):
            ref = thresh_components[:, k, REF_FOLD_NUMBER].T
            if i != REF_FOLD_NUMBER:
                r = np.corrcoef(thresh_components[:, k, i].T,
                                ref)
                if r[0, 1] < 0:
                    #print "Reverting comp {k} of fold {i} for model {key}".format(i=i+1, k=k, key=key)
                    aligned_thresh_comp[:, k, i] *= -1
    # Compute fleiss_kappa and DICE on thresholded components
    fleiss_kappas = np.empty(N_COMP)
    dice_bars = np.empty(N_COMP)
    dices = np.zeros((10,N_COMP))
    
    for k in range(N_COMP):
        # One component accross folds
        thresh_comp = aligned_thresh_comp[:, k, :]
        fleiss_kappas[k] = metrics.fleiss_kappa(thresh_comp)
        dice_bars[k],dices[:,k] = metrics.dice_bar(thresh_comp) 
       
#    print dices.mean(axis=1) 
#    dices_mean_path = os.path.join(OUTPUT_DIR,'fmri_5folds/results','dices_mean_%s.npy' %key[0])
#    if key[0] == 'struct_pca' and key[2]==1e-6:
#        dices_mean_path = os.path.join(OUTPUT_DIR,'fmri_5folds/results','dices_mean_%s.npy' %'enet_pca')
        
#    print dices_mean_path    
#    np.save(dices_mean_path,dices.mean(axis=1) )

    print(key)
    scores = OrderedDict((
        ('model', key[0]),
        ('global_pen', key[1]),
        ('tv_ratio', key[2]),
        ('l1_ratio', key[3]),
        ('frobenius_train', av_frobenius_train[0]),
        ('frobenius_test', av_frobenius_test[0]),
        ('kappa_0', fleiss_kappas[0]),
        ('kappa_1', fleiss_kappas[1]),
        ('kappa_2', fleiss_kappas[2]),
        ('kappa_mean', np.mean(fleiss_kappas)),
        ('dice_bar_0', dice_bars[0]),
        ('dice_bar_1', dice_bars[1]),
        ('dice_bar_2', dice_bars[2]),
        ('dices_mean',dice_bars.mean()),
        ('evr_test_0',evr_test.mean(axis=1)[0]),
        ('evr_test_1',evr_test.mean(axis=1)[1]),
        ('evr_test_2',evr_test.mean(axis=1)[2]),
        ('evr_test_sum',evr_test.mean(axis=1)[0]+evr_test.mean(axis=1)[1]+evr_test.mean(axis=1)[2]),
        ('frob_test_fold1',frobenius_test[0][0]),
        ('frob_test_fold2',frobenius_test[0][1]),
        ('frob_test_fold3',frobenius_test[0][2]),
        ('frob_test_fold4',frobenius_test[0][3]),
        ('frob_test_fold5',frobenius_test[0][4])))

    return scores


def run_test(wd, config):
    print("In run_test")
    import mapreduce
    os.chdir(wd)
    params = config['params'][-1]
    key = '_'.join([str(p) for p in params])
    load_globals(config)
    OUTPUT = os.path.join('test', key)
    oc = mapreduce.OutputCollector(OUTPUT)
    X = np.load(config['data']['X'])
    mapreduce.DATA_RESAMPLED = {}
    mapreduce.DATA_RESAMPLED["X"] = [X, X]
    mapper(params, oc)


def create_config(y, n_folds, output_dir, filename,
                  include_full_resample=True):

    full_output_dir = os.path.join(OUTPUT_DIR, output_dir)
    if not os.path.exists(full_output_dir):
        os.makedirs(full_output_dir)

#    skf = StratifiedKFold(y=y,
#                          n_folds=n_folds)
#    resample_index = [[tr.tolist(), te.tolist()] for tr, te in skf]
    resample_index = []
    if include_full_resample:
        resample_index.insert(0, None)  # first fold is None
    #
    #resample_index.insert(0,None)

    # Create config file
    user_func_filename = os.path.abspath(__file__)
    config = dict(data=dict(X="X.npy"),
                   structure="mask.npy",
                  params=PARAMS,
                  resample=resample_index,
                  map_output="results",
                  n_folds=n_folds,
                  full_resample=include_full_resample,
                  user_func=user_func_filename,
                  reduce_group_by="params",
                  reduce_output="results.csv")
    config_full_filename = os.path.join(full_output_dir, filename)
    json.dump(config, open(config_full_filename, "w"))
    return config

#################
# Actual script #
#################

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    #Retreive variables
    X= np.load(INPUT_DATA_X)    
    y = np.ones(X.shape[0])
    shutil.copy(INPUT_DATA_X, os.path.join(OUTPUT_DIR,"pcattv_FS_VIP_all"))
    shutil.copy(INPUT_MASK, os.path.join(OUTPUT_DIR,"pcattv_FS_VIP_all"))

    #############################################################################
       # Create config files
    config_5folds = create_config(y, *(CONFIGS[0]))

    DEBUG = False
    if DEBUG:
        run_test(OUTPUT_DIR, config_5folds)
        
     # Build utils files: sync (push/pull) and PBS

    sync_push_filename, sync_pull_filename, WD_CLUSTER = \
        clust_utils.gabriel_make_sync_data_files(os.path.join(OUTPUT_DIR,"pcattv_FS_VIP_all"))
    cmd = "mapreduce.py --map  %s/config.json" % WD_CLUSTER
    clust_utils.gabriel_make_qsub_job_files(os.path.join(OUTPUT_DIR,"pcattv_FS_VIP_all"), cmd,walltime = "300:00:00")    