# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 12:56:34 2016

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
import brainomics.mesh_processing as mesh_utils
import brainomics.cluster_gabriel as clust_utils
################
# Input & output #
##################

INPUT_BASE_DIR = "/neurospin/brainomics/2014_pca_struct/adni/fs_5"    
TEMPLATE_PATH = os.path.join(INPUT_BASE_DIR, "freesurfer_template")                      
OUTPUT_DIR = "/neurospin/brainomics/2014_pca_struct/adni/fs_5"


#############################################################################

##############
# Parameters #
##############

# Parameters for the function create_config
# Note that value at index 1 will be the name of the task on the cluster
CONFIGS = [[5, "adni_5folds", "config_5folds.json", True]]
N_COMP = 3
# Global penalty
GLOBAL_PENALTIES = np.array([0.1])
# Relative penalties
# 0.33 ensures that there is a case with TV = L1 = L2
TVRATIO = np.array([0.5,0.1])
L1RATIO = np.array([0.5,1e-1])

PCA_PARAMS = [('pca', 0.0, 0.0, 0.0)]

STRUCT_PCA_PARAMS = list(product(['struct_pca'],
                                 GLOBAL_PENALTIES,
                                 TVRATIO,
                                 L1RATIO))                            
                                 
                                 
SPARSE_PCA = [('sparse_pca', 0.1, 1e-6, 0.01),('sparse_pca', 0.1,  1e-6, 0.5)]
PARAMS = PCA_PARAMS + STRUCT_PCA_PARAMS + SPARSE_PCA

#############
# Functions #
#############



def load_globals(config):
    import mapreduce as GLOBAL
    mesh_coord, mesh_triangles = mesh_utils.mesh_arrays(os.path.join(TEMPLATE_PATH, "lrh.pial.gii"))
    mask = np.load(os.path.join(INPUT_BASE_DIR, "mask.npy"))
    import parsimony.functions.nesterov.tv as tv_helper
    Atv = tv_helper.linear_operator_from_mesh(mesh_coord, mesh_triangles, mask=mask)
    GLOBAL.Atv = Atv
    GLOBAL.N_FOLDS = config['n_folds']
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
        # Force the key
        global_pen = tv_ratio = l1_ratio = 0
    if model_name == 'sparse_pca':
        # Force the key
        ltv = 1e-6
        ll1 = l1_ratio * global_pen
        ll2 = (1 - l1_ratio) * global_pen * (1 - tv_ratio)
        
    if model_name == 'struct_pca':
        ltv = global_pen * tv_ratio
        ll1 = l1_ratio * global_pen * (1 - tv_ratio)
        ll2 = (1 - l1_ratio) * global_pen * (1 - tv_ratio)
        assert(np.allclose(ll1 + ll2 + ltv, global_pen))

    X_train = GLOBAL.DATA_RESAMPLED["X"][0]
    n, p = X_train.shape
    X_test = GLOBAL.DATA_RESAMPLED["X"][1]

    # A matrices
    Atv = GLOBAL.Atv


    # Fit model
    if model_name == 'pca':
        model = sklearn.decomposition.PCA(n_components=N_COMP)
    if model_name == 'sparse_pca':
       model = pca_tv.PCA_L1_L2_TV(n_components=N_COMP,
                                    l1=ll1, l2=ll2, ltv=ltv,
                                    Atv=Atv,
                                    criterion="frobenius",
                                    eps=1e-6,
                                    max_iter=100,
                                    inner_max_iter=int(1e4),
                                    output=False)
    if model_name == 'struct_pca':
        model = pca_tv.PCA_L1_L2_TV(n_components=N_COMP,
                                    l1=ll1, l2=ll2, ltv=ltv,
                                    Atv=Atv,
                                    criterion="frobenius",
                                    eps=1e-6,
                                    max_iter=100,
                                    inner_max_iter=int(1e4),
                                    output=False)
    t0 = time.clock()
    t0 = time.clock()
    model.fit(X_train)
    t1 = time.clock()
    _time = t1 - t0
    #print "X_test", GLOBAL.DATA["X"][1].shape

    # Save the projectors
    if (model_name == 'pca') :
        V = model.components_.T
    if model_name == 'struct_pca' or (model_name == 'sparse_pca'):
        V = model.V

    # Project train & test data
    if (model_name == 'pca'):
        X_train_transform = model.transform(X_train)
        X_test_transform = model.transform(X_test)
        
    if (model_name == 'struct_pca')  or (model_name == 'sparse_pca'):
        X_train_transform, _ = model.transform(X_train)
        X_test_transform, _ = model.transform(X_test)

    # Reconstruct train & test data
    # For SparsePCA or PCA, the formula is: UV^t (U is given by transform)
    # For StructPCA this is implemented in the predict method (which uses
    # transform)
    if (model_name == 'pca') :
        X_train_predict = np.dot(X_train_transform, V.T)
        X_test_predict = np.dot(X_test_transform, V.T)
        
    if (model_name == 'struct_pca') or (model_name == 'sparse_pca'):
        X_train_predict = model.predict(X_train)
        X_test_predict = model.predict(X_test)

    # Compute Frobenius norm between original and recontructed datasets
    frobenius_train = np.linalg.norm(X_train - X_train_predict, 'fro')
    frobenius_test = np.linalg.norm(X_test - X_test_predict, 'fro')
    print frobenius_test 


    # Compute explained variance ratio
    evr_train = metrics.adjusted_explained_variance(X_train_transform)
    evr_train /= np.var(X_train, axis=0).sum()
    evr_test = metrics.adjusted_explained_variance(X_test_transform)
    evr_test /= np.var(X_test, axis=0).sum()

    # Remove predicted values (they are huge)
    del X_train_predict, X_test_predict

    # Compute geometric metrics and norms of components
    TV = parsimony.functions.nesterov.tv.TotalVariation(1, A=Atv)
    l0 = np.zeros((N_COMP,))
    l1 = np.zeros((N_COMP,))
    l2 = np.zeros((N_COMP,))
    tv = np.zeros((N_COMP,))
    for i in range(N_COMP):
        # Norms
        l0[i] = np.linalg.norm(V[:, i], 0)
        l1[i] = np.linalg.norm(V[:, i], 1)
        l2[i] = np.linalg.norm(V[:, i], 2)
        tv[i] = TV.f(V[:, i])

    ret = dict(frobenius_train=frobenius_train,
               frobenius_test=frobenius_test,
               components=V,
               X_train_transform=X_train_transform,
               X_test_transform=X_test_transform,
               evr_train=evr_train,
               evr_test=evr_test,
               l0=l0,
               l1=l1,
               l2=l2,
               tv=tv,
               time=_time)

    output_collector.collect(key, ret)
    
    
def reducer(key, values):
    output_collectors = values
    print output_collectors
    global N_COMP
    import mapreduce as GLOBAL
    N_FOLDS = GLOBAL.N_FOLDS
    components=np.zeros((317091,3,5))
    frobenius_train = np.zeros((3,5))
    frobenius_test = np.zeros((1,5))
    evr_train = np.zeros((3,5))
    evr_test = np.zeros((3,5))
    l0 = np.zeros((3,5))
    l1 = np.zeros((3,5))
    l2 =np.zeros((3,5))
    tv = np.zeros((3,5))
    times = np.zeros((1,5))
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
            l0[:,item] = values["l0"]
            l1[:,item] = values["l1"]
            l2[:,item] = values["l2"]
            tv[:,item] = values["tv"]
            evr_train[:,item] = values["evr_train"]
            evr_test[:,item] = values["evr_test"]
            times[:,item] = values["time"]
       

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
    av_frobenius_train = frobenius_train.mean(axis=0)
    av_frobenius_test = frobenius_test.mean(axis=0)

    # Compute correlations of components between all folds
    n_corr = N_FOLDS * (N_FOLDS - 1) / 2
    correlations = np.zeros((N_COMP, n_corr))
    for k in range(N_COMP):
        R = np.corrcoef(np.abs(components[:, k, :].T))
        # Extract interesting coefficients (upper-triangle)
        correlations[k] = R[np.triu_indices_from(R, 1)]

    # Transform to z-score
    Z = 1. / 2. * np.log((1 + correlations) / (1 - correlations))
    # Average for each component
    z_bar = np.mean(Z, axis=1)
    # Transform back to average correlation for each component
    r_bar = (np.exp(2 * z_bar) - 1) / (np.exp(2 * z_bar) + 1)

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
    # Save aligned comp
#    for l, oc in zip(range(N_FOLDS), output_collectors[1:]):
#        filename = os.path.join(oc.output_dir, "aligned_thresh_comp.npz")
#        np.savez(filename, aligned_thresh_comp[:, :, l])

    # Compute fleiss_kappa and DICE on thresholded components
    fleiss_kappas = np.empty(N_COMP)
    dice_bars = np.empty(N_COMP)
    for k in range(N_COMP):
        # One component accross folds
        thresh_comp = aligned_thresh_comp[:, k, :]
        fleiss_kappas[k] = metrics.fleiss_kappa(thresh_comp)
        dice_bars[k] = metrics.dice_bar(thresh_comp)
        
    print key
    scores = OrderedDict((
        ('model', key[0]),
        ('global_pen', key[1]),
        ('tv_ratio', key[2]),
        ('l1_ratio', key[3]),
        ('frobenius_train', av_frobenius_train[0]),
        ('frobenius_test', av_frobenius_test[0]),
        ('correlation_0', r_bar[0]),
        ('correlation_1', r_bar[1]),
        ('correlation_2', r_bar[2]),
        ('correlation_mean', np.mean(r_bar)),
        ('kappa_0', fleiss_kappas[0]),
        ('kappa_1', fleiss_kappas[1]),
        ('kappa_2', fleiss_kappas[2]),
        ('kappa_mean', np.mean(fleiss_kappas)),
        ('dice_bar_0', dice_bars[0]),
        ('dice_bar_1', dice_bars[1]),
        ('dice_bar_2', dice_bars[2]),
        ('dice_bar_mean', np.mean(dice_bars)),
        ('time', np.mean(times))))

    return scores


def run_test(wd, config):
    print "In run_test"
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

    skf = StratifiedKFold(y=y.reshape(y.shape[0]),
                          n_folds=n_folds)
    resample_index = [[tr.tolist(), te.tolist()] for tr, te in skf]
    if include_full_resample:
        resample_index.insert(0, None)  # first fold is None

    # Copy the learning data & mask
    INPUT_DATASET = os.path.join(INPUT_BASE_DIR,'X.npy')
    shutil.copy2(INPUT_DATASET, full_output_dir)

    # Create config file
    user_func_filename = os.path.abspath(__file__)

    config = dict(data=dict(X=os.path.basename(INPUT_DATASET)),
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

    # Create files to synchronize with the cluster
    sync_push_filename, sync_pull_filename, CLUSTER_WD = \
    clust_utils.gabriel_make_sync_data_files(full_output_dir)

    # Create job files
    # As the dataset is big we don't use standard files
    limits = OrderedDict()
    limits['host'] = OrderedDict()
    limits['host']['nodes'] = 10
    limits['host']['ppn'] = 6
    limits['mem'] = "25gb"
    limits['walltime'] = "96:00:00"
    queue = "Cati_long"
    cluster_cmd = "mapreduce.py -m {dir}/{file} --ncore {ppn}".format(
                            dir=CLUSTER_WD,
                            file=filename,
                            ppn=limits['host']['ppn'])
    job_file_name = os.path.join(full_output_dir, "job_" + queue +".pbs")
    clust_utils.write_job_file(job_file_name,
                               job_name=output_dir,
                               cmd=cluster_cmd,
                               queue=queue,
                               job_limits=limits)
    return config

#################
# Actual script #
#################

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    #Retreive variables
    #############################################################################
    y=np.load(os.path.join(INPUT_BASE_DIR,'y.npy'))
   
    # Create config files
    config_5folds = create_config(y, *(CONFIGS[0]))

    DEBUG = False
    if DEBUG:
        run_test(OUTPUT_DIR, config_5folds)