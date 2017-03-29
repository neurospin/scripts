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
import time
import shutil
from collections import OrderedDict
import numpy as np
import sklearn.decomposition
from sklearn import metrics 
import parsimony.functions.nesterov.tv
import pca_tv
from brainomics import array_utils
import brainomics.cluster_gabriel as clust_utils
from parsimony.datasets.regression import dice5
import dice5_data
import brainomics.cluster_gabriel as clust_utils
#cd home/ad247405/git/scripts/2014_pca_struct/
#import dice5_data

################
# Input/Output #
################
INPUT_BASE_DIR = "/neurospin/brainomics/2016_pca_struct/dice"
INPUT_BASE_DATA_DIR = os.path.join(INPUT_BASE_DIR, "data_0.1")
INPUT_MASK_DIR = os.path.join(INPUT_BASE_DIR, "masks")
INPUT_DATA_DIR = os.path.join(INPUT_BASE_DATA_DIR,"data_100_100_0")
INPUT_STD_DATASET_FILE = "data.std.npy"
INPUT_OBJECT_MASK_FILE_FORMAT =  os.path.join(INPUT_MASK_DIR,"mask_{o}.npy")
INPUT_SNR_FILE = os.path.join(INPUT_BASE_DIR,"SNR.npy")
OUTPUT_BASE_DIR = os.path.join(INPUT_BASE_DIR, "local_minima_experiment")
OUTPUT_DIR_FORMAT = os.path.join(OUTPUT_BASE_DIR,
                                 "data_100_100_{set}")

##############
# Parameters #
##############

N_COMP = 3

##
TRAIN_RANGE = range(int(dice5_data.N_SAMPLES/2))
TEST_RANGE = range(int(dice5_data.N_SAMPLES/2), dice5_data.N_SAMPLES)



params = [('struct_pca', 0.01,0.5,0.5)]

JSON_DUMP_OPT = {'indent': 4,
                 'separators': (',', ': ')}

#############
# Functions #
#############


def compute_coefs_from_ratios(global_pen, tv_ratio, l1_ratio):
    ltv = global_pen * tv_ratio
    ll1 = l1_ratio * global_pen * (1 - tv_ratio)
    ll2 = (1 - l1_ratio) * global_pen * (1 - tv_ratio)
    assert(np.allclose(ll1 + ll2 + ltv, global_pen))
    return ll1, ll2, ltv
    



def load_globals(config):
    import mapreduce as GLOBAL  # access to global variables
    GLOBAL.DATA = GLOBAL.load_data(config["data"])
    IM_SHAPE = config["im_shape"]
    A= parsimony.functions.nesterov.tv.A_from_shape(IM_SHAPE)
    N_COMP = config["n_comp"]
    GLOBAL.A,GLOBAL.N_COMP = A,N_COMP



def resample(config, resample_nb):
    import mapreduce as GLOBAL  # access to global variables
    GLOBAL.DATA = GLOBAL.load_data(config["data"])
    resample = config["resample"][resample_nb]
    if resample is not None:
        GLOBAL.DATA_RESAMPLED = {k: [GLOBAL.DATA[k][idx, ...]
                                 for idx in resample]
                                 for k in GLOBAL.DATA}
    else:  # resample is None train == test
        GLOBAL.DATA_RESAMPLED = {k: [GLOBAL.DATA[k]
                                 for idx in [0, 1]]
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

    X_train = GLOBAL.DATA_RESAMPLED["X"][0]
    X_test = GLOBAL.DATA_RESAMPLED["X"][1]
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
                                    output=True)
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
#    evr_train = metrics.adjusted_explained_variance(X_train_transform)
#    evr_train /= np.var(X_train, axis=0).sum()
#    evr_test = metrics.adjusted_explained_variance(X_test_transform)
#    evr_test /= np.var(X_test, axis=0).sum()

    # Remove predicted values (they are huge)
    del X_train_predict, X_test_predict

    ret = dict(frobenius_train=frobenius_train,
               frobenius_test=frobenius_test,
               components=V,
               X_train_transform=X_train_transform,
               X_test_transform=X_test_transform)
               #evr_train=evr_train,
               #evr_test=evr_test)

    output_collector.collect(key, ret)


def reducer(key, values):
    output_collectors = values
    print (output_collectors[0])
    global N_COMP
    import mapreduce as GLOBAL
    components=np.zeros((10000,3))
    frobenius_train = np.zeros((1))
    frobenius_test = np.zeros((1,))
    mse=np.zeros((1,))
    l0 = np.zeros((3))
    l1 = np.zeros((3))
    l2 =np.zeros((3))
    tv = np.zeros((3))
    times = np.zeros((1))

    values = output_collectors[0].load() 
    components = values["components"]
    frobenius_train = values["frobenius_train"]
    frobenius_test = values["frobenius_test"]
    l0= values["l0"]
    l1 = values["l1"]
    l2 = values["l2"]
    tv = values["tv"]
    times = values["time"]
         

    scores = OrderedDict((
        ('model', key[0]),
        ('global_pen', key[1]),
        ('tv_ratio', key[2]),
        ('l1_ratio', key[3]),
        ('frobenius_train', frobenius_train),
        ('frobenius_test', frobenius_test),
        ('time', np.mean(times))))


    return scores


def mse(imageA, imageB):
    err = np.sum(((imageA) - (imageB)) ** 2)
    err /= (imageA.shape[0] * imageA.shape[1])
    return err
    



#################
# Actual script #
#################

if __name__ == '__main__':
    # Read SNRs
    #input_snrs = np.load(INPUT_SNR_FILE)
    input_snrs=[0.1]
    # Resample
    resamplings = [[TRAIN_RANGE, TEST_RANGE]]
    
    include_full_resample=True
    from sklearn.cross_validation import StratifiedKFold
    skf = StratifiedKFold(y=np.ones((500)),n_folds=2)
    resample_index = [[tr.tolist(), te.tolist()] for tr, te in skf]
    if include_full_resample:
        resample_index.insert(0, None)  # first fold is None    
    
    
    # Create a mapreduce config file for each dataset
    for set in range(50):
        #set=0.1
        input_dir = INPUT_DATA_DIR
              
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
#        for i in range(3):
#            filename = INPUT_OBJECT_MASK_FILE_FORMAT.format(o=i)
#            src_filename = os.path.join(INPUT_MASK_DIR, filename)
#            dst_filename = os.path.join(output_dir, filename)
#            shutil.copy(src_filename, dst_filename)

                   #################################################################
        # Build utils files: sync (push/pull) and PBS

        sync_push_filename, sync_pull_filename, WD_CLUSTER = \
            clust_utils.gabriel_make_sync_data_files(output_dir)
        cmd = "mapreduce.py --map  %s/config.json" % WD_CLUSTER
        clust_utils.gabriel_make_qsub_job_files(output_dir, cmd,walltime="1000:00:00")
    #        ################################################################
    #        # Sync to cluster
        print ("Sync data to gabriel.intra.cea.fr: ")
        os.system(sync_push_filename)    # Create config file
        
        user_func_filename = "/home/ad247405/git/scripts/2016_pca_struct/dice/04_local_minima_experiment.py"

        config = OrderedDict([
            ('data', dict(X=INPUT_STD_DATASET_FILE)),
            ('resample', resample_index),
            ('im_shape', (100,100)),
            ('params', params),
            ('n_comp', N_COMP),
            ('map_output', "results"),
            ('user_func', user_func_filename),
            ('reduce_group_by', "params"),
            ('reduce_output', "results.csv")])
        config_full_filename = os.path.join(output_dir, "config.json")
        
        json.dump(config,
                  open(config_full_filename, "w"))

        # Create job files
#        cluster_cmd = "mapreduce.py -m %s/config.json  --ncore 20" % CLUSTER_WD
#        clust_utils.gabriel_make_qsub_job_files(output_dir, cluster_cmd)

