# -*- coding: utf-8 -*-
"""



Process dice5 datasets with standard PCA and our structured PCA.
We use several values for global penalization, TV ratio and L1 ratio.

We generate a map_reduce configuration file for each dataset and the files
needed to run on the cluster.
Due to the cluster setup and the way mapreduce work we need to copy the datsets
and the masks on the cluster. Therefore we copy them on the output directory
which is synchronised on the cluster.

The output directory is results/data_{shape}_{snr}.
"""

import os
import json
import shutil
from collections import OrderedDict
import numpy as np
import sklearn.decomposition
from sklearn import metrics 
import parsimony.functions.nesterov.tv
import metrics
from brainomics import array_utils
import brainomics.cluster_gabriel as clust_utils
from parsimony.datasets.regression import dice5
import dice5_data
from sklearn.cross_validation import StratifiedKFold
import scipy.sparse as sparse
import parsimony.functions.nesterov.tv as nesterov_tv
import pca_struct
################
# Input/Output #
################
INPUT_BASE_DIR = "/neurospin/brainomics/2016_pca_struct/dice"
INPUT_MASK_DIR = os.path.join(INPUT_BASE_DIR, "masks")
INPUT_DATA_DIR_FORMAT = os.path.join(INPUT_BASE_DIR,"data_0.1","data_{s[0]}_{s[1]}_{set}")
INPUT_STD_DATASET_FILE = "data.std.npy"
INPUT_OBJECT_MASK_FILE_FORMAT = "mask_{o}.npy"
INPUT_SNR_FILE = os.path.join(INPUT_BASE_DIR,"SNR.npy")
OUTPUT_BASE_DIR = os.path.join(INPUT_BASE_DIR, "2017","results")
OUTPUT_DIR_FORMAT = os.path.join(OUTPUT_BASE_DIR,"data_{s[0]}_{s[1]}_{set}")

##############
# Parameters #
##############

N_COMP = 10
NFOLDS_OUTER = 2
NFOLDS_INNER = 5


#############
# Functions #
#############


def load_globals(config):
    import mapreduce as GLOBAL  # access to global variables
    GLOBAL.DATA = GLOBAL.load_data(config["data"])
    IM_SHAPE = config["im_shape"]
    A= sparse.vstack(nesterov_tv.linear_operator_from_shape(IM_SHAPE))
    N_COMP = config["n_comp"]
    GLOBAL.A,GLOBAL.N_COMP = A,N_COMP



def resample(config, resample_nb):
    import mapreduce as GLOBAL  # access to global variables
    GLOBAL.DATA = GLOBAL.load_data(config["data"])
    resample = config["resample"][resample_nb]
    GLOBAL.DATA_RESAMPLED = {k: [GLOBAL.DATA[k][idx, ...] for idx in resample]
                            for k in GLOBAL.DATA}



def mapper(key, output_collector):
    import mapreduce as GLOBAL  # access to global variables:
    model_name, global_pen, gn_ratio, l1_ratio = key
    lgn = global_pen * gn_ratio
    ll1 = l1_ratio * global_pen * (1 - gn_ratio)
    ll2 = (1 - l1_ratio) * global_pen * (1 - gn_ratio)
    assert(np.allclose(ll1 + ll2 + lgn, global_pen))

    X_train = GLOBAL.DATA_RESAMPLED["X"][0]
    X_test = GLOBAL.DATA_RESAMPLED["X"][1]
    Agn = GLOBAL.A
    N_COMP = GLOBAL.N_COMP

                            
    model = pca_struct.PCAGraphNet(n_components=N_COMP,
                                    l1=ll1, l2=ll2, lgn=lgn,
                                    Agn=Agn,
                                    criterion="frobenius",
                                    eps=1e-6,
                                    max_iter=500,
                                    output=False)

    model.fit(X_train)
    

    V = model.V
    X_train_transform, _ = model.transform(X_train)
    X_test_transform, _ = model.transform(X_test)
    X_train_predict = model.predict(X_train)
    X_test_predict = model.predict(X_test)

    # Compute Frobenius norm between original and recontructed datasets
    frobenius_train = np.linalg.norm(X_train - X_train_predict, 'fro')
    frobenius_test = np.linalg.norm(X_test - X_test_predict, 'fro')
    print(frobenius_test) 


    # Remove predicted values (they are huge)
    del X_train_predict, X_test_predict

    ret = dict(frobenius_train=frobenius_train,
               frobenius_test=frobenius_test,
               components=V,
               X_train_transform=X_train_transform,
               X_test_transform=X_test_transform)


    output_collector.collect(key, ret)


def scores(key, paths, config):
    import mapreduce
    values = [mapreduce.OutputCollector(p) for p in paths]
    values = [item.load() for item in values] 
    model = "graphNet"


    frobenius_train = np.vstack([item["frobenius_train"] for item in values])
    frobenius_test = np.vstack([item["frobenius_test"] for item in values])
    components = np.dstack([item["components"] for item in values])


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
#    paths = [p for p in paths if not p.count("graphNet_pca_0.1_1e-06_0.1")]
#    paths = [p for p in paths if not p.count("graphNet_pca_0.1_1e-06_0.5")]
#    paths = [p for p in paths if not p.count("graphNet_pca_0.1_1e-06_0.8")]
#    paths = [p for p in paths if not p.count("graphNet_pca_0.1_0.1_0.5")]
#    paths = [p for p in paths if not p.count("graphNet_pca_0.1_0.5_0.5")]
#    paths = [p for p in paths if not p.count("graphNet_pca_0.0_0.0_10.0")]
#    paths = [p for p in paths if not p.count("graphNet_pca_0.1_0.5_0.1")]


   
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
    byparams = groupby_paths([p for p in paths if p.count("cv00/all") and not p.count("all/all")],3) 
    byparams_scores = {k:scores(k, v, config) for k, v in list(byparams.items())}

    data = [list(byparams_scores[k].values()) for k in byparams_scores]

    columns = list(byparams_scores[list(byparams_scores.keys())[0]].keys())
    scores_refit = pd.DataFrame(data, columns=columns)
    
    print('## doublecv scores by outer-cv and by params')
    print('## -----------------------------------------')
    data = list()
    bycv = groupby_paths([p for p in paths if p.count("cvnested")],1)
    for fold, paths_fold in list(bycv.items()):
        print(fold)
        if fold == "cv01":
            byparams = groupby_paths([p for p in paths_fold], 3)
            byparams_scores = {k:scores(k, v, config) for k, v in list(byparams.items())}
            data += [[fold] + list(byparams_scores[k].values()) for k in byparams_scores]
            scores_dcv_byparams = pd.DataFrame(data, columns=["fold"] + columns)
        if fold == "cv00":
            print ("do not take into account second fold")

        
    print('## Model selection')
    print('## ---------------')
    folds_gn= argmaxscore_bygroup(scores_dcv_byparams[scores_dcv_byparams["model"] == "graphNet"])
    scores_argmax_byfold_gn = folds_gn
    

    print('## Apply best model on refited')
    print('## ---------------------------')
    scores_gn = scores("nestedcv", [os.path.join(config['map_output'], row["fold"], "all", row["param_key"]) for index, row in folds_gn.iterrows()], config)
    scores_cv_gn = pd.DataFrame([["gn"] + list(scores_gn.values())], columns=["method"] + list(scores_gn.keys()))
     
         
    with pd.ExcelWriter(results_file_path) as writer:
        scores_refit.to_excel(writer, sheet_name='scores_all', index=False)
        scores_dcv_byparams.to_excel(writer, sheet_name='scores_dcv_byparams', index=False)
        scores_argmax_byfold_gn.to_excel(writer, sheet_name='scores_argmax_byfold_gn', index=False)
        scores_cv_gn.to_excel(writer, sheet_name = 'scores_cv_gn', index=False)

#################
# Actual script #
#################

if __name__ == '__main__':
    # Read SNRs
    #input_snrs = np.load(INPUT_SNR_FILE)
    input_snrs=[0.1]
    # Resample 

    y = np.ones((500))
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
            cv_inner = StratifiedKFold(y[tr_val].ravel(), n_folds=NFOLDS_INNER, random_state=42)
            for cv_inner_i, (tr, val) in enumerate(cv_inner):
                cv["cv%02d/cvnested%02d" % ((cv_outer_i-1), cv_inner_i)] = [tr_val[tr], tr_val[val]]
    for k in cv:
        cv[k] = [cv[k][0].tolist(), cv[k][1].tolist()]
      
    print((list(cv.keys())))  
    


########
    params = [('graphNet_pca', 0.1, 0.1, 0.1),('graphNet_pca', 0.1, 0.1, 0.5),\
             ('graphNet_pca', 0.1, 0.5, 0.1),('graphNet_pca', 0.1, 0.5, 0.5),('graphNet_pca', 0.01, 0.1, 0.1),\
             ('graphNet_pca', 0.01, 0.1, 0.5),('graphNet_pca', 0.01, 0.5, 0.1),('graphNet_pca', 0.01, 0.5, 0.5)]

    # Create a mapreduce config file for each dataset
    for set in range(50):
        #set=0.1
        input_dir = INPUT_DATA_DIR_FORMAT.format(s=dice5_data.SHAPE,
                                                 set=set)

        
        # Local output directory for this dataset
        output_dir = os.path.join(OUTPUT_BASE_DIR,
                                  OUTPUT_DIR_FORMAT.format(s=dice5_data.SHAPE,
                                                           set=set))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Copy the learning data
        src_datafile = os.path.join(input_dir, INPUT_STD_DATASET_FILE)
        shutil.copy(src_datafile, output_dir)

        # Copy the objects masks
        for i in range(3):
            filename = INPUT_OBJECT_MASK_FILE_FORMAT.format(o=i)
            src_filename = os.path.join(INPUT_MASK_DIR, filename)
            dst_filename = os.path.join(output_dir, filename)
            shutil.copy(src_filename, dst_filename)



        # Create config file
        user_func_filename = os.path.abspath(__file__)

        config = OrderedDict([
            ('data', dict(X=INPUT_STD_DATASET_FILE)),
            ('resample', cv),
            ('im_shape', (100,100)),
            ('params', params),
            ('n_comp', N_COMP),
            ('map_output', "results"),
            ('user_func', user_func_filename),
            ('reduce_group_by', "params"),
            ('reduce_output', "results.csv")])
        json.dump(config, open(os.path.join(output_dir, "config_dCV.json"), "w"))
        
        
        
            # Build utils files: sync (push/pull) and PBS
        import brainomics.cluster_gabriel as clust_utils
        sync_push_filename, sync_pull_filename, WD_CLUSTER = \
            clust_utils.gabriel_make_sync_data_files(output_dir)
        cmd = "mapreduce.py --map  %s/config_dCV.json" % WD_CLUSTER
        clust_utils.gabriel_make_qsub_job_files(output_dir, cmd,walltime="200:00:00")
    #        ################################################################
        os.system(sync_push_filename)

            


        
       
