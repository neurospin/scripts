"""
Created on Fri Feb 21 19:15:48 2014

@author:  edouard.duchesnay@cea.fr
@license: BSD-3-Clause
"""
import os
import json
import numpy as np
from sklearn.cross_validation import StratifiedKFold
import nibabel
from sklearn.metrics import precision_recall_fscore_support
from parsimony.estimators import LogisticRegressionL1L2TV
import parsimony.functions.nesterov.tv as tv_helper

##############################################################################
## User map/reduce functions
def A_from_structure(structure_filepath):
    # Input: structure_filepath. Output: A, structure
    STRUCTURE = nibabel.load(structure_filepath)
    A, _ = tv_helper.A_from_mask(STRUCTURE.get_data())
    return A, STRUCTURE

def mapper(key, output_collector):
    import mapreduce  as GLOBAL # access to global variables:
    # GLOBAL.DATA, GLOBAL.STRUCTURE, GLOBAL.A
    # GLOBAL.DATA ::= {"X":[Xtrain, ytrain], "y":[Xtest, ytest]}
    # key: list of parameters
    alpha, ratio_l1, ratio_l2, ratio_tv = key
    class_weight="auto" # unbiased
    l1, l2, tv = alpha *  np.array((ratio_l1, ratio_l2, ratio_tv))
    mod = LogisticRegressionL1L2TV(l1, l2, tv, GLOBAL.A, penalty_start=3, 
                                        class_weight=class_weight)
    mod.fit(GLOBAL.DATA["GMtrain"][0], GLOBAL.DATA["ytrain"][0])
    y_pred = mod.predict(GLOBAL.DATA["GMtrain"][1])
    ret = dict(y_pred=y_pred, y_true=GLOBAL.DATA["ytrain"][1], beta=mod.beta)
    output_collector.collect(key, ret)

def reducer(key, values):
    # key : string of intermediary key
    # load return dict correspondning to mapper ouput. they need to be loaded.
    values = [item.load("*.npy") for item in values]
    y_true = [item["y_true"].ravel() for item in values]
    y_pred = [item["y_pred"].ravel() for item in values]
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average=None)
    n_ite = None
    scores = dict(key=key,
               recall_0=r[0], recall_1=r[1], recall_mean=r.mean(),
               precision_0=p[0], precision_1=p[1], precision_mean=p.mean(),
               f1_0=f[0], f1_1=f[1], f1_mean=f.mean(),
               support_0=s[0] , support_1=s[1], n_ite=n_ite)
    return scores

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

if __name__ == "__main__":
    WD = "/neurospin/brainomics/2014_mlc/GM"
    #BASE = "/neurospin/tmp/brainomics/testenettv"
    WD_CLUSTER = WD.replace("/neurospin/brainomics", "/neurospin/tmp/brainomics")
    #print "Sync data to %s/ " % os.path.dirname(WD)
    #os.system('rsync -azvu %s %s/' % (BASE, os.path.dirname(WD)))
    INPUT_DATA_X = os.path.join('GMtrain.npy')
    INPUT_DATA_y = os.path.join('ytrain.npy')
    INPUT_MASK_PATH = os.path.join("mask.nii")
    NFOLDS = 5
    #WD = os.path.join(WD, 'logistictvenet_5cv')
    if not os.path.exists(WD): os.makedirs(WD)
    os.chdir(WD)

    #############################################################################
    ## Create config file
    y = np.load(INPUT_DATA_y)
    cv = [[tr.tolist(), te.tolist()] for tr,te in StratifiedKFold(y.ravel(), n_folds=5)]
    # parameters grid
    # Re-run with 
    tv_range = np.hstack([np.arange(0, 1., .1), [0.05, 0.01, 0.005, 0.001]])
    ratios = np.array([[1., 0., 1], [0., 1., 1], [.5, .5, 1], [.9, .1, 1],
                       [.1, .9, 1], [.01, .99, 1], [.001, .999, 1]])    
    alphas = [.01, .05, .1 , .5, 1.]
    l1l2tv =[np.array([[float(1-tv), float(1-tv), tv]]) * ratios for tv in tv_range]
    l1l2tv.append(np.array([[0., 0., 1.]]))
    l1l2tv = np.concatenate(l1l2tv)
    alphal1l2tv = np.concatenate([np.c_[np.array([[alpha]]*l1l2tv.shape[0]), l1l2tv] for alpha in alphas])
    params = [params.tolist() for params in alphal1l2tv]
    # User map/reduce function file:
    try:
        user_func_filename = os.path.abspath(__file__)
    except:
        user_func_filename = os.path.join(os.environ["HOME"],
        "git", "scripts", "2014_mlc", 
        "02_logistictvenet_gm.py")
        print "USE", user_func_filename
    # Use relative path from config.json    
    config = dict(data=dict(X=INPUT_DATA_X, y=INPUT_DATA_y),
                  params=params, resample=cv,
                  structure=INPUT_MASK_PATH,
                  map_output="results",#os.path.join(OUTPUT, "results"),
                  user_func=user_func_filename,
                  ncore=12,
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
    