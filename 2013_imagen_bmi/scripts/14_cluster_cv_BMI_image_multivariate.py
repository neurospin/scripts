# -*- coding: utf-8 -*-
"""
Created on Tue May 27 08:26:28 2014

@author: 
Copyrignt : CEA NeuroSpin - 2014
"""

import os, sys
import json
import numpy as np

import pandas as pd
import tables

from sklearn.cross_validation import KFold
from sklearn.metrics import r2_score
import parsimony.estimators as estimators
#from sklearn.linear_model import ElasticNet


##############################################################################
## User map/reduce functions 

## A_from_structure is not necessary for ElasticNet
#def A_from_structure(structure_filepath):
#    # Input: structure_filepath. Output: A, structure
#    STRUCTURE = nibabel.load(structure_filepath)
#    A, _ = tv_helper.A_from_mask(STRUCTURE.get_data())
#    return A, STRUCTURE

## mapper
def mapper(key, output_collector):
    import mapreduce  as GLOBAL # access to global variables (GLOBAL.DATA)
    # GLOBAL.DATA ::= {"X":[Xtrain, ytrain], "y":[Xtest, ytest]}
    #mod = ElasticNet(alpha=key[0], l1_ratio=key[1])
    alpha, l1_ratio = key[0], key[1]
    mod = estimators.ElasticNet(alpha*l1_ratio, penalty_start = 1, mean = True)
    z_pred = mod.fit(GLOBAL.DATA["X"][0], GLOBAL.DATA["z"][0]).predict(GLOBAL.DATA["X"][1])
    ret = dict(z_pred=z_pred, z_true=GLOBAL.DATA["z"][1])
    output_collector.collect(key, ret)    

#r# educer
def reducer(key, values):
    # key : string of intermediary key
    # load return dict correspondning to mapper ouput. they need to be loaded.
    # values are OutputCollerctors containing a path to the results.
    # load return dict correspondning to mapper ouput. they need to be loaded.
    values = [item.load() for item in values]
    y_true = np.concatenate([item["z_true"].ravel() for item in values])
    y_pred = np.concatenate([item["z_pred"].ravel() for item in values])
    return dict(param=key, r2=r2_score(y_true, y_pred))


##############################################################################
##  utils for BMI data
sys.path.append(os.path.join(os.getenv('HOME'),
                                'gits','scripts','2013_imagen_bmi', 'scripts'))
import bmi_utils
##############
# Parameters #
##############
# Input data
BASE_PATH = '/neurospin/brainomics/2013_imagen_bmi/'
DATA_PATH = os.path.join(BASE_PATH, 'data')
IMAGES_FILE = os.path.join(DATA_PATH, 'smoothed_images.hdf5')
SNPS_FILE = os.path.join(DATA_PATH, 'SNPs.csv')
BMI_FILE = os.path.join(DATA_PATH, 'BMI.csv')

# Shared data
BASE_SHARED_DIR = "/neurospin/tmp/brainomics/md238665"
SHARED_DIR = os.path.join(BASE_SHARED_DIR, 'multiblock_analysis')
if not os.path.exists(SHARED_DIR):
    os.makedirs(SHARED_DIR)


#############
# Read data #
#############
# SNPs and BMI
def load_data(cache=False):
    if not(cache):
        #SNPs = pd.io.parsers.read_csv(os.path.join(DATA_PATH, "SNPs.csv"), dtype='float64', index_col=0).as_matrix()
        BMI = pd.io.parsers.read_csv(os.path.join(DATA_PATH, "BMI.csv"), index_col=0).as_matrix()
        
        # Images
        h5file = tables.openFile(IMAGES_FILE)
        masked_images = bmi_utils.read_array(h5file, "/standard_mask/residualized_images_gender_center_TIV_pds")    #images already masked
        print "Data loaded"
        
        X = masked_images
        #Y = SNPs
        z = BMI
        
        np.save(os.path.join(SHARED_DIR, "X.npy"), X)
        #np.save(os.path.join(SHARED_DIR, "Y.npy"), Y)
        np.save(os.path.join(SHARED_DIR, "z.npy"), z)
        h5file.close()
        
        print "Data saved"
    else:
        X = np.load(os.path.join(SHARED_DIR, "X.npy"))        
        #Y = np.load(os.path.join(SHARED_DIR, "Y.npy"))
        z = np.load(os.path.join(SHARED_DIR, "z.npy"))        
        print "Data read from cache"
    
    return X, z #X, Y, z


##############################################################################
## Cluster utils
job_template_pbs =\
"""#!/bin/bash
#PBS -S /bin/bash
#PBS -N %(job_name)s
#PBS -l nodes=1:ppn=%(ppn)s
#PBS -l walltime=48:00:00
#PBS -q %(queue)s

%(script)s
"""
#PBS -d %(job_dir)s

def utils_sync_jobs(WD, WD_CLUSTER, config_basename="config.json", 
                    cmd_path="mapreduce.py", jobname="map"):
    # Build Sync pull/push utils files
    push_str = 'rsync -azvu %s gabriel.intra.cea.fr:%s/' % (
        os.path.dirname(WD),
        os.path.dirname(os.path.dirname(WD_CLUSTER)))
    sync_push_filename = os.path.join(WD, "sync_push.sh")
    with open(sync_push_filename, 'wb') as f:
        f.write(push_str)
    os.chmod(sync_push_filename, 0777)
    pull_str = 'rsync -azvu gabriel.intra.cea.fr:%s %s/' % (
        os.path.dirname(WD_CLUSTER),
        os.path.dirname(os.path.dirname(WD)))
    sync_pull_filename = os.path.join(WD, "sync_pull.sh")
    with open(sync_pull_filename, 'wb') as f:
        f.write(pull_str)
    os.chmod(sync_pull_filename, 0777)
    project_name = jobname
    # Build PBS files
    config_filename = os.path.join(WD_CLUSTER, config_basename)
    #job_dir = os.path.dirname(config_filename)
    #for nb in xrange(options.pbs_njob):
    params = dict()
    params['job_name'] = '%s' % project_name
    params['ppn'] = 12
    #params['job_dir'] = job_dir
    params['script'] = '%s --mode map --config %s --ncore %i' % (cmd_path, config_filename, params['ppn'])
    params['queue'] = "Cati_LowPrio"
    qsub = job_template_pbs % params
    job_filename = os.path.join(WD, 'job_Cati_LowPrio.pbs')
    with open(job_filename, 'wb') as f:
        f.write(qsub)
    os.chmod(job_filename, 0777)
    params['ppn'] = 8
    params['queue'] = "Global_long"
    params['script'] = '%s --mode map --config %s --ncore %i' % (cmd_path, config_filename, params['ppn'])
    qsub = job_template_pbs % params
    job_filename = os.path.join(WD, 'job_Global_long.pbs')
    with open(job_filename, 'wb') as f:
        f.write(qsub)
    os.chmod(job_filename, 0777)
    return sync_push_filename, sync_pull_filename


##############################################################################
## Run all
#def run_all():
#    WD = "/neurospin/brainomics/2014_mlc/GM"
#    key = '0.01_0.01_0.98_0.01'
#    OUTPUT = os.path.join(os.path.dirname(WD), 'logistictvenet_all', key)
#    if not os.path.exists(OUTPUT): os.makedirs(OUTPUT)
#    X = np.load(os.path.join(WD,  'GMtrain.npy'))
#    y = np.load(os.path.join(WD,  'ytrain.npy'))
#    A, STRUCTURE = A_from_structure(os.path.join(WD,  "mask.nii"))
#    params = np.array([float(p) for p in key.split("_")])
#    l1, l2, tv = params[0] * params[1:]
#    mod = LogisticRegressionL1L2TV(l1, l2, tv, A, penalty_start=1, 
#                                   class_weight="auto")
#    mod.fit(X, y)
#    #CPU times: user 1936.73 s, sys: 0.66 s, total: 1937.39 s
#    # Wall time: 1937.13 s / 2042.58 s
#    y_pred = mod.predict(X)
#    p, r, f, s = precision_recall_fscore_support(y, y_pred, average=None)
#    n_ite = mod.algorithm.num_iter
#    scores = dict(
#               recall_0=r[0], recall_1=r[1], recall_mean=r.mean(),
#               precision_0=p[0], precision_1=p[1], precision_mean=p.mean(),
#               f1_0=f[0], f1_1=f[1], f1_mean=f.mean(),
#               support_0=s[0] , support_1=s[1], n_ite=n_ite, intercept=mod.beta[0, 0])
#    beta3d = np.zeros(STRUCTURE.get_data().shape)
#    beta3d[STRUCTURE.get_data() != 0 ] = mod.beta[1:].ravel()
#    out_im = nibabel.Nifti1Image(beta3d, affine=STRUCTURE.get_affine())
#    ret = dict(y_pred=y_pred, y_true=y, beta=mod.beta, beta3d=out_im, scores=scores)
#    # run /home/ed203246/bin/mapreduce.py
#    oc = OutputCollector(OUTPUT)
#    oc.collect(key=key, value=ret)

if __name__ == "__main__":
    ## Set pathes
    WD = "/neurospin/tmp/brainomics/bmi_images_cluster"
    WD_CLUSTER = WD.replace("/neurospin/tmp/brainomics", "/neurospin/tmp/brainomics")
    #print "Sync data to %s/ " % os.path.dirname(WD)
    #os.system('rsync -azvu %s %s/' % (BASE, os.path.dirname(WD)))
    if not os.path.exists(WD): os.makedirs(WD)
    
    
    ## get update and save data in WD for the mapreduce jobs
    X, z = load_data(cache=True)
    n, p = X.shape
    np.save(os.path.join(WD, 'X.npy'), np.hstack((np.ones((z.shape[0],1)),X)))
    np.save(os.path.join(WD, "z.npy"), z)

    
    ## Parameterize the mapreduce 
    ##   1) pathes
    INPUT_DATA_X = os.path.join('X.npy')
    INPUT_DATA_y = os.path.join('z.npy')
    NFOLDS = 5
    ## 2) cv idex and parameters to test
    cv = [[tr.tolist(), te.tolist()] for tr,te in KFold(n, n_folds=5)]    
    ## 2) cv idex and parameters to test
    params = [[alpha, l1_ratio] for alpha in [0.003, 0.007, 0.010] for l1_ratio in np.arange(0.5, 1., .2)]
    # User map/reduce function file:
    try:
        user_func_filename = os.path.abspath(__file__)
    except:
        user_func_filename = os.path.join(os.environ["HOME"],
        "git", "scripts", "2013_imagen_bmi", "scripts", 
        "14_cluster_cv_BMI_image_multivariate.py")
        print "USE", user_func_filename

    # Use relative path from config.json    
    config = dict(data=dict(X=INPUT_DATA_X, y=INPUT_DATA_y),
                  params=params, resample=cv,
                  structure="",#No structure information
                  map_output="results",#os.path.join(OUTPUT, "results"),
                  user_func=user_func_filename,
                  ncore=4,
                  reduce_input="results/*/*", #os.path.join(OUTPUT, "results/*/*"),
                  reduce_group_by="results/.*/(.*)",#os.path.join(OUTPUT, "results/.*/(.*)"),
                  reduce_output="results.csv")#os.path.join(OUTPUT, "results.csv"))
    json.dump(config, open(os.path.join(WD, "config.json"), "w"))

    
    #############################################################################
    # Build utils files: sync (push/pull) and PBS
    jobname = os.path.basename(os.path.dirname(WD))
    sync_push_filename, sync_pull_filename = utils_sync_jobs(WD, WD_CLUSTER,
                                                             jobname=jobname)
    #############################################################################
    # Sync to cluster
    print "Sync data to gabriel.intra.cea.fr: "
    os.system(sync_push_filename)
    #############################################################################
    print "# Start by running Locally with 2 cores, to check that everything os OK)"
    print "Interrupt after a while CTL-C"
    print "mapreduce.py --mode map --config %s/config.json --ncore 2" % WD
    #os.system("mapreduce.py --mode map --config %s/config.json" % WD)
    print "# 1) Log on gabriel:"
    print 'ssh -t gabriel.intra.cea.fr'
    print "# 2) Run one Job to test"
    print "qsub -I"
    print "cd %s" % WD_CLUSTER
    print "./job_Global_long.pbs"
    print "# 3) Run on cluster"
    print "qsub job_Global_long.pbs"
    print "# 4) Log out and pull Pull"
    print "exit"
    print sync_pull_filename
    #############################################################################
    print "# Reduce"
    print "mapreduce.py --mode reduce --config %s/config.json" % WD
    