# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 16:00:53 2016

@author: ad247405

Run standard PCA, sparse PCA , Enet PCA and PCA-TV on fmri dataset. 
"""

import os
import json
from collections import OrderedDict
import numpy as np
import sklearn.decomposition
import pca_tv
import metrics 
import sklearn
import array_utils
import parsimony.functions.nesterov.tv as tv_helper
from sklearn.cross_validation import StratifiedKFold
import shutil




WD = '/neurospin/brainomics/2016_AUSZ/results/Freesurfer/FS_pca_tv_patients_only'
def config_filename(): return os.path.join(WD,"config_dCV.json")
def results_filename(): return os.path.join(WD,"results_dCV_5folds.xlsx")
#############################################################################
#############
# Functions #
#############


def load_globals(config):
    import mapreduce as GLOBAL  # access to global variables
    GLOBAL.DATA = GLOBAL.load_data(config["data"])
    STRUCTURE = np.load(config["structure"])
    A = tv_helper.A_from_mask(STRUCTURE)
    N_COMP = config["N_COMP"]
    GLOBAL.A, GLOBAL.STRUCTURE,GLOBAL.N_COMP = A, STRUCTURE,N_COMP



def resample(config, resample_nb):
    import mapreduce as GLOBAL  # access to global variables
    GLOBAL.DATA = GLOBAL.load_data(config["data"])
    resample = config["resample"][resample_nb]
    GLOBAL.DATA_RESAMPLED = {k: [GLOBAL.DATA[k][idx, ...] for idx in resample]
                            for k in GLOBAL.DATA}


def mapper(key, output_collector):
    import mapreduce as GLOBAL  # access to global variables:
    model_name, global_pen, tv_ratio, l1_ratio = key
    if model_name == 'pca':
        # Force the key
        global_pen = tv_ratio = l1_ratio = 0
     
    if model_name == 'sparse_pca':   
        
        global_pen = tv_ratio = 0
        ll1=l1_ratio 

    if model_name == 'struct_pca':
        ltv = global_pen * tv_ratio
        ll1 = l1_ratio * global_pen * (1 - tv_ratio)
        ll2 = (1 - l1_ratio) * global_pen * (1 - tv_ratio)
        assert(np.allclose(ll1 + ll2 + ltv, global_pen))

        
    
    X_train = GLOBAL.DATA_RESAMPLED["X"][0]
    n, p = X_train.shape
    X_test = GLOBAL.DATA_RESAMPLED["X"][1]
    # A matrices
    Atv = GLOBAL.A
    N_COMP = GLOBAL.N_COMP

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
    
    
def scores(key, paths, config):
    import mapreduce
    values = [mapreduce.OutputCollector(p) for p in paths]
    values = [item.load() for item in values] 
    if key == "nestedcv":
        model = "nestedcv"

    else:      
        if key.split("_")[0] == "struct" and float(key.split("_")[3]) < 0.001:
            model = "enet"
        if key.split("_")[0] == "struct" and float(key.split("_")[3]) > 0.001:
            model = "enettv"              
        if key.split("_")[0] == "pca":
            model = "pca"    
        if key.split("_")[0] == "sparse":
            model = "sparse"    


    frobenius_train = np.vstack([item["frobenius_train"] for item in values])
    frobenius_test = np.vstack([item["frobenius_test"] for item in values])


    #save mean pariwise across folds for further analysis   
#    dices_mean_path = os.path.join(WD,'dices_mean_%s.npy' %key.split("_")[0])
#    if key.split("_")[0] == 'struct_pca' and key.split("_")[2]<1e-3:
#        dices_mean_path = os.path.join(WD,'dices_mean_%s.npy' %"enet_pca")
#
#    np.save(dices_mean_path,dice_pairwise_values.mean(axis=0))

    
    #Mean frobenius norm across folds  
    mean_frobenius_train = frobenius_train.mean()
    mean_frobenius_test = frobenius_test.mean()

    print(key)
    scores = OrderedDict((
        ('param_key',key),
        ('model', model),
        ('frobenius_train', mean_frobenius_train),
        ('frobenius_test', mean_frobenius_test)))             
    return scores
    
    
def reducer(key, values):
    import os, glob, pandas as pd
    dir_path =os.getcwd()
    config_path = os.path.join(dir_path,"config_dCV.json")
    results_file_path = os.path.join(dir_path,"results_dCV_5folds.xlsx")
    config = json.load(open(config_path))
    paths = glob.glob(os.path.join(config['map_output'], "*", "*", "*"))
    paths = [p for p in paths if not p.count("struct_pca_0.1_1e-06_0.8")]

   
    def close(vec, val, tol=1e-4):
        return np.abs(vec - val) < tol

    def groupby_paths(paths, pos):
        groups = {g:[] for g in set([p.split("/")[pos] for p in paths])}
        for p in paths:
            groups[p.split("/")[pos]].append(p)
        return groups

    def argmaxscore_bygroup(data, groupby='fold', param_key="param_key", score="frobenius_test"):
        arg_min_byfold = list()
        for fold, data_fold in data.groupby(groupby):
            assert len(data_fold) == len(set(data_fold[param_key]))  # ensure all  param are diff
            arg_min_byfold.append([fold, data_fold.ix[data_fold[score].argmin()][param_key], data_fold[score].min()])
        return pd.DataFrame(arg_min_byfold, columns=[groupby, param_key, score])

    print('## Refit scores')
    print('## ------------')
    byparams = groupby_paths([p for p in paths if p.count("all") and not p.count("all/all")],3) 
    byparams_scores = {k:scores(k, v, config) for k, v in list(byparams.items())}

    data = [list(byparams_scores[k].values()) for k in byparams_scores]

    columns = list(byparams_scores[list(byparams_scores.keys())[0]].keys())
    scores_refit = pd.DataFrame(data, columns=columns)
    

    with pd.ExcelWriter(results_file_path) as writer:
        scores_refit.to_excel(writer, sheet_name='scores_all', index=False)
        

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


#################
# Actual script #
#################

if __name__ == "__main__":
    WD = '/neurospin/brainomics/2016_AUSZ/results/Freesurfer/FS_pca_tv_patients_only'    
    INPUT_DATA_X = '/neurospin/brainomics/2016_AUSZ/results/Freesurfer/data/X_patients_only.npy'
    INPUT_MASK_PATH = '/neurospin/brainomics/2016_AUSZ/results/Freesurfer/data/mask.npy'

    NFOLDS_OUTER = 5
    NFOLDS_INNER = 5
    N_COMP = 5

    #############################################################################
    ## Create config file
    ###########################
    #create resampling pipeline
    #create DOUbleCV resampling pipeline
    y = np.ones(np.load(INPUT_DATA_X).shape[0])
    cv_outer = [[tr, te] for tr,te in StratifiedKFold(y.ravel(), n_folds=NFOLDS_OUTER, random_state=42)]
    if cv_outer[0] is not None: # Make sure first fold is None
        cv_outer.insert(0, None)   
        null_resampling = list(); null_resampling.append(np.arange(0,len(y))),null_resampling.append(np.arange(0,len(y)))
        cv_outer[0] = null_resampling
                
    import collections
    cv = collections.OrderedDict()
    for cv_outer_i, (tr_val, te) in enumerate(cv_outer):
        if cv_outer_i == 0:
            cv["all/all"] = [tr_val, te]
        else:    
            cv["cv%02d/all" % (cv_outer_i -1)] = [tr_val, te]
            
    for k in cv:
        cv[k] = [cv[k][0].tolist(), cv[k][1].tolist()]
      
    print((list(cv.keys())))  

    ###########################
    #Grid of parameters to explore
    ###########################
    params = [("pca", 0.0, 0.0, 0.0),("sparse_pca", 0.0, 0.0, 0.1),("sparse_pca", 0.0, 0.0, 1.0),\
             ("sparse_pca", 0.0, 0.0, 10.0), ("struct_pca", 0.1, 1e-06, 0.01),\
             ("struct_pca", 0.1, 1e-06, 0.1), ("struct_pca", 0.1, 1e-06, 0.5),("struct_pca", 0.1, 1e-06, 0.8),\
             ("struct_pca", 0.01, 1e-04, 0.01),("struct_pca", 0.01, 1e-04, 0.1), ("struct_pca", 0.01, 1e-04, 0.5),\
             ("struct_pca", 0.01, 1e-04, 0.8),('struct_pca', 0.1, 0.1, 0.1),('struct_pca', 0.1, 0.1, 0.5),\
             ('struct_pca', 0.1, 0.5, 0.1),('struct_pca', 0.1, 0.5, 0.5),('struct_pca', 0.01, 0.1, 0.1),\
             ('struct_pca', 0.01, 0.1, 0.5),('struct_pca', 0.01, 0.5, 0.1),('struct_pca', 0.01, 0.5, 0.5),\
             ('struct_pca', 0.1, 0.5, 0.8),('struct_pca', 0.1, 0.1, 0.8),('struct_pca', 0.1, 0.8, 0.1),\
             ('struct_pca', 0.1, 0.8, 0.5),('struct_pca', 0.1, 0.8, 0.8)]         
    shutil.copy(INPUT_DATA_X, WD)
    shutil.copy(INPUT_MASK_PATH, WD)       
    
    user_func_filename = "/home/ad247405/git/scripts/2016_AUSZ/Freesurfer/FS_pcatv_patients.py"
    
    config = dict(data=dict(X="X_patients_only.npy"),
                  params=params, resample=cv,
                  structure="mask.npy",
                  N_COMP = N_COMP,
                  map_output="model_selectionCV", 
                  user_func=user_func_filename,
                  reduce_input="results/*/*",
                  reduce_group_by="params",
                  reduce_output="model_selectionCV.csv")
    json.dump(config, open(os.path.join(WD, "config_dCV.json"), "w"))
    
    
    
        #################################################################
    # Build utils files: sync (push/pull) and PBS
    import brainomics.cluster_gabriel as clust_utils
    sync_push_filename, sync_pull_filename, WD_CLUSTER = \
        clust_utils.gabriel_make_sync_data_files(WD)
    cmd = "mapreduce.py --map  %s/config_dCV.json" % WD_CLUSTER
    clust_utils.gabriel_make_qsub_job_files(WD, cmd,walltime="1000:00:00")
#        ################################################################
#        # Sync to cluster
    #print ("Sync data to gabriel.intra.cea.fr: ")
    #os.system(sync_push_filename)

    """######################################################################
    print "# Start by running Locally with 2 cores, to check that everything is OK)"
    print ("mapreduce.py --map %s/config.json --ncore 2" % WD)
    #os.system("mapreduce.py --mode map --config %s/config_dCV.json" % WD)
    print ("# 1) Log on gabriel:")
    print ('ssh -t gabriel.intra.cea.fr')
    print ("# 2) Run one Job to test")
    print ("qsub -I")
    print ("cd %s" % WD_CLUSTER)
    print ("./job_Global_long.pbs")
    print ("# 3) Run on cluster")
    print ("qsub job_Global_long.pbs")
    print ("# 4) Log out and pull Pull")
    print ("exit")
    print (sync_pull_filename)
    #########################################################################
    print "# Reduce"
    print "mapreduce.py --reduce %s/config.json" % WD"""

