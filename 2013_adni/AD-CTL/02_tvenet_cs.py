# -*- coding: utf-8 -*-
"""
Created on Fri May 30 20:03:12 2014

@author: edouard.duchesnay@cea.fr
"""

import os
import json
import numpy as np
from collections import OrderedDict
from sklearn.cross_validation import StratifiedKFold
import nibabel
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from sklearn.feature_selection import SelectKBest
from parsimony.estimators import LogisticRegressionL1L2TV
import parsimony.functions.nesterov.tv as tv_helper


def load_globals(config):
    import mapreduce as GLOBAL  # access to global variables
    GLOBAL.DATA = GLOBAL.load_data(config["data"])
    STRUCTURE = nibabel.load(config["mask_filename"])
    A, _ = tv_helper.A_from_mask(STRUCTURE.get_data())
    GLOBAL.A, GLOBAL.STRUCTURE, GLOBAL.CONFIG = A, STRUCTURE, config


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

def mapper_fix(key, output_collector):
    """This mapper do not fit, re-use the precomputed stored beta and compute
    proba of test samples. Call it using mapreduce.py -m -f config.json
    """
    import mapreduce as GLOBAL # access to global variables:
        #raise ImportError("could not import ")
    # GLOBAL.DATA, GLOBAL.STRUCTURE, GLOBAL.A
    # GLOBAL.DATA ::= {"X":[Xtrain, ytrain], "y":[Xtest, ytest]}
    # key: list of parameters
    Xte = GLOBAL.DATA_RESAMPLED["X"][1]
    #print output_collector,
    STRUCTURE = GLOBAL.STRUCTURE
    penalty_start = GLOBAL.CONFIG["penalty_start"]
    class_weight="auto" # unbiased
    alpha = float(key[0])
    l1, l2, tv, k = alpha * float(key[1]), alpha * float(key[2]), alpha * float(key[3]), key[4]
    values = output_collector.load()
    if k != -1:
        k = int(k)
        mask3d_to_1d = STRUCTURE.get_data() != 0
        mask3d_to_1dr = values["mask"]
        mask_1d_to_1dr = mask3d_to_1dr[mask3d_to_1d]
        A, _ = tv_helper.A_from_shape((3, 3, 3))  # dummy A
        Xte_r = np.hstack([Xte[:, :penalty_start], Xte[:, penalty_start:][:, mask_1d_to_1dr]])
    else:
        Xte_r = Xte
        A = GLOBAL.A
    mod = LogisticRegressionL1L2TV(l1, l2, tv, A, penalty_start=penalty_start,
                                   class_weight=class_weight)
    mod.beta = values["beta"]
    # check prevously predicted equals new predictions
    assert np.all(mod.predict(Xte_r) == values["y_pred"])
    proba_pred = mod.predict_probability(Xte_r)
    ret = dict(proba_pred=proba_pred)
    if output_collector:
        output_collector.collect(key, ret)
    else:
        return ret


def mapper(key, output_collector):
    import mapreduce as GLOBAL # access to global variables:
        #raise ImportError("could not import ")
    # GLOBAL.DATA, GLOBAL.STRUCTURE, GLOBAL.A
    # GLOBAL.DATA ::= {"X":[Xtrain, ytrain], "y":[Xtest, ytest]}
    # key: list of parameters
    Xtr = GLOBAL.DATA_RESAMPLED["X"][0]
    Xte = GLOBAL.DATA_RESAMPLED["X"][1]
    ytr = GLOBAL.DATA_RESAMPLED["y"][0]
    yte = GLOBAL.DATA_RESAMPLED["y"][1]
    print key, "Data shape:", Xtr.shape, Xte.shape, ytr.shape, yte.shape
    STRUCTURE = GLOBAL.STRUCTURE
    #alpha, ratio_l1, ratio_l2, ratio_tv, k = key
    #key = np.array(key)
    penalty_start = GLOBAL.CONFIG["penalty_start"]
    class_weight="auto" # unbiased
    alpha = float(key[0])
    l1, l2, tv, k = alpha * float(key[1]), alpha * float(key[2]), alpha * float(key[3]), key[4]
    print "l1:%f, l2:%f, tv:%f, k:%i" % (l1, l2, tv, k)
    if k != -1:
        k = int(k)
        aov = SelectKBest(k=k)
        aov.fit(Xtr[..., penalty_start:], ytr.ravel())
        mask = STRUCTURE.get_data() != 0
        mask[mask] = aov.get_support()
        #print mask.sum()
        A, _ = tv_helper.A_from_mask(mask)
        Xtr_r = np.hstack([Xtr[:, :penalty_start], Xtr[:, penalty_start:][:, aov.get_support()]])
        Xte_r = np.hstack([Xte[:, :penalty_start], Xte[:, penalty_start:][:, aov.get_support()]])
    else:
        mask = np.ones(Xtr.shape[0], dtype=bool)
        Xtr_r = Xtr
        Xte_r = Xte
        A = GLOBAL.A
    mod = LogisticRegressionL1L2TV(l1, l2, tv, A, penalty_start=penalty_start,
                                   class_weight=class_weight)
    mod.fit(Xtr_r, ytr)
    y_pred = mod.predict(Xte_r)
    proba_pred = mod.predict_probability(Xte_r)
    ret = dict(y_pred=y_pred, proba_pred=proba_pred, y_true=yte, beta=mod.beta,  mask=mask)
    if output_collector:
        output_collector.collect(key, ret)
    else:
        return ret

def reducer(key, values):
    # key : string of intermediary key
    # load return dict correspondning to mapper ouput. they need to be loaded.
    # DEBUG
    #import glob, mapreduce
    #values = [mapreduce.OutputCollector(p) for p in glob.glob("/neurospin/brainomics/2013_adni/AD-CTL/results/*/0.1_0.0_0.0_1.0_-1.0/")]
    #values = [mapreduce.OutputCollector(p) for p in glob.glob("/home/ed203246/tmp/MCIc-MCInc_cs/results/*/0.1_0.0_0.0_1.0_-1.0/")]
    # Compute sd; ie.: compute results on each folds
    values = [item.load() for item in values[1:]]
    recall_mean_std = np.std([np.mean(precision_recall_fscore_support(
        item["y_true"].ravel(), item["y_pred"])[1]) for item in values]) / np.sqrt(len(values))
    y_true = [item["y_true"].ravel() for item in values]
    y_pred = [item["y_pred"].ravel() for item in values]
    prob_pred = [item["proba_pred"].ravel() for item in values]
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    prob_pred = np.concatenate(prob_pred)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average=None)
    auc = roc_auc_score(y_true, prob_pred) #area under curve score.
    n_ite = None
    betas = np.hstack([item["beta"] for item in values]).T
    R = np.corrcoef(betas)
    beta_cor_mean = np.mean(R[np.triu_indices_from(R, 1)])
    a, l1, l2 , tv , k = [float(par) for par in key.split("_")]
    print a, l1, l2, tv, k, beta_cor_mean
    a, l1, l2 , tv , k = [float(par) for par in key.split("_")]
    scores = OrderedDict()
    scores['a'] = a
    scores['l1'] = l1
    scores['l2'] = l2
    scores['tv'] = tv
    scores['l1l2_ratio'] = l1 / float(1-tv)
    scores['recall_0'] = r[0]
    scores['recall_1'] = r[1]
    scores['recall_mean'] = r.mean()
    scores['recall_mean_std'] = recall_mean_std
    scores['auc'] = auc
    scores['beta_cor_mean'] = beta_cor_mean
    scores['precision_0'] = p[0]
    scores['precision_1'] = p[1]
    scores['precision_mean'] = p.mean()
    scores['f1_0'] = f[0]
    scores['f1_1'] = f[1]
    scores['f1_mean'] = f.mean()
    scores['support_0'] = s[0]
    scores['support_1'] = s[1]
    scores['n_ite'] = n_ite
    scores['k'] = k
    scores['key'] = key
    return scores



##############################################################################
## Run all
def run_all(config):
    import mapreduce
    WD = "/neurospin/brainomics/2013_adni/AD-CTL_cs"
    key = '0.01_0.01_0.98_0.01_10000'
    #class GLOBAL: DATA = dict()
    load_globals(config)
    OUTPUT = os.path.join(os.path.dirname(WD), 'logistictvenet_all', key)
    # run /home/ed203246/bin/mapreduce.py
    oc = mapreduce.OutputCollector(OUTPUT)
    #if not os.path.exists(OUTPUT): os.makedirs(OUTPUT)
    X = np.load(os.path.join(WD,  'X.npy'))
    y = np.load(os.path.join(WD,  'y.npy'))
    mapreduce.DATA["X"] = [X, X]
    mapreduce.DATA["y"] = [y, y]
    params = np.array([float(p) for p in key.split("_")])
    mapper(params, oc)
    #oc.collect(key=key, value=ret)

if __name__ == "__main__":
    WD = "/neurospin/brainomics/2013_adni/AD-CTL_cs"
    #BASE = "/neurospin/tmp/brainomics/testenettv"
    #WD_CLUSTER = WD.replace("/neurospin/brainomics", "/neurospin/tmp/brainomics")
    #print "Sync data to %s/ " % os.path.dirname(WD)
    #os.system('rsync -azvu %s %s/' % (BASE, os.path.dirname(WD)))
    INPUT_DATA_X = os.path.join('X.npy')
    INPUT_DATA_y = os.path.join('y.npy')
    INPUT_MASK_PATH = os.path.join("mask.nii.gz")
    NFOLDS = 5
    #WD = os.path.join(WD, 'logistictvenet_5cv')
    if not os.path.exists(WD): os.makedirs(WD)
    os.chdir(WD)

    #############################################################################
    ## Create config file
    y = np.load(INPUT_DATA_y)
    if os.path.exists("config.json"):
        inf = open("config.json", "r")
        old_conf = json.load(inf)
        cv = old_conf["resample"]
        inf.close()
    else:
        cv = [[tr.tolist(), te.tolist()] for tr,te in StratifiedKFold(y.ravel(), n_folds=5)]
    if cv[0] is not None: # Make sure first fold is None
        cv.insert(0, None)
    assert len(cv) == 6
    # parameters grid
    # Re-run with
    tv_range = np.hstack([np.arange(0, 1., .1), [0.05, 0.01, 0.005, 0.001]])
    ratios = np.array([[1., 0., 1], [0., 1., 1], [.5, .5, 1], [.9, .1, 1],
                       [.1, .9, 1], [.01, .99, 1], [.001, .999, 1]])
    alphas = [.01, .05, .1 , .5, 1.]
    k_range = [100, 1000, 10000, 100000, -1]
    l1l2tv =[np.array([[float(1-tv), float(1-tv), tv]]) * ratios for tv in tv_range]
    l1l2tv.append(np.array([[0., 0., 1.]]))
    l1l2tv = np.concatenate(l1l2tv)
    alphal1l2tv = np.concatenate([np.c_[np.array([[alpha]]*l1l2tv.shape[0]), l1l2tv] for alpha in alphas])
    alphal1l2tvk = np.concatenate([np.c_[alphal1l2tv, np.array([[k]]*alphal1l2tv.shape[0])] for k in k_range])
    params = [params.tolist() for params in alphal1l2tvk]
    # User map/reduce function file:
#    try:
#        user_func_filename = os.path.abspath(__file__)
#    except:
    user_func_filename = os.path.join(os.environ["HOME"],
        "git", "scripts", "2013_adni", "AD-CTL",
        "02_tvenet_cs.py")
    #print __file__, os.path.abspath(__file__)
    print "user_func", user_func_filename
    #import sys
    #sys.exit(0)
    # Use relative path from config.json
    config = dict(data=dict(X=INPUT_DATA_X, y=INPUT_DATA_y),
                  params=params, resample=cv,
                  mask_filename=INPUT_MASK_PATH,
                  penalty_start = 2,
                  map_output="5cv",
                  user_func=user_func_filename,
                  reduce_input="5cv/*/*",
                  reduce_group_by="5cv/.*/(.*)",
                  reduce_output="AD-CTL_cs.csv")
    json.dump(config, open(os.path.join(WD, "config.json"), "w"))

    #############################################################################
    # Build utils files: sync (push/pull) and PBS
    import brainomics.cluster_gabriel as clust_utils
    sync_push_filename, sync_pull_filename, WD_CLUSTER = \
        clust_utils.gabriel_make_sync_data_files(WD)
    cmd = "mapreduce.py --map  %s/config.json" % WD_CLUSTER
    clust_utils.gabriel_make_qsub_job_files(WD, cmd)
    #############################################################################
    # Sync to cluster
    print "Sync data to gabriel.intra.cea.fr: "
    os.system(sync_push_filename)
    #############################################################################
    print "# Start by running Locally with 2 cores, to check that everything os OK)"
    print "Interrupt after a while CTL-C"
    print "mapreduce.py --map %s/config.json --ncore 2" % WD
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
    print "mapreduce.py --reduce %s/config.json" % WD
