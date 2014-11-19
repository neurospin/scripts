# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 10:23:04 2014

@author: cp243490

Use Logistic Regression Model with demographic variables (Age, Gender)
"""

import os
import json
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from parsimony.estimators import LogisticRegression
from parsimony.estimators import LogisticRegressionL1L2TV
from sklearn.cross_validation import StratifiedKFold
import shutil
import statsmodels.discrete.discrete_model as sm
from scipy import sparse

from collections import OrderedDict


NFOLDS = 5


def load_globals(config):
    import mapreduce as GLOBAL  # access to global variables
    GLOBAL.DATA = GLOBAL.load_data(config["data"])
    GLOBAL.DATA["X"] = GLOBAL.DATA["X"][:, 0:3]
    GLOBAL.MODALITY = config["modality"]


def resample(config, resample_nb):
    import mapreduce as GLOBAL  # access to global variables
    resample = config["resample"][resample_nb]
    if resample is not None:
        GLOBAL.DATA_RESAMPLED = {k: [GLOBAL.DATA[k][idx, ...]
                        for idx in resample]
                            for k in GLOBAL.DATA}
    else:  # resample is None train == test
        GLOBAL.DATA_RESAMPLED = {k: [GLOBAL.DATA[k] for idx in [0, 1]]
                            for k in GLOBAL.DATA}


def mapper(key, output_collector):
    import mapreduce as GLOBAL  # access to global variables:
        #raise ImportError("could not import ")
    # GLOBAL.DATA, GLOBAL.STRUCTURE, GLOBAL.A
    # GLOBAL.DATA ::= {"X":[Xtrain, Xtest], "y":[ytrain, ytest]}
    # key: list of parameters
    print "key: ", key
    Xtr = GLOBAL.DATA_RESAMPLED["X"][0]
    Xte = GLOBAL.DATA_RESAMPLED["X"][1]
    ytr = GLOBAL.DATA_RESAMPLED["y"][0]
    yte = GLOBAL.DATA_RESAMPLED["y"][1]
    print key, "Data shape:", Xtr.shape, Xte.shape, ytr.shape, yte.shape
    method = key[0]
    if method == "statsmodels":
        # Logistic Regression with statsmodels tool, Logit
        logit_mod = sm.Logit(ytr, Xtr)
        logit_res = logit_mod.fit(disp=0)
        prob_pred = logit_res.predict(Xte)
        y_pred = np.zeros((Xte.shape[0]))
        y_pred[prob_pred >= 0.5] = 1
        beta = logit_res.params.reshape(-1, 1)
    elif method == "log_parsimony":
        # Logistic Regression with parsimnoy tool, LogisticRegression
        mod = LogisticRegression()
        mod.fit(Xtr, ytr)
        y_pred = mod.predict(Xte)
        prob_pred = mod.predict_probability(Xte)  # a posteriori probability
        beta = mod.beta
    elif method == "enettv_parsimony":
        # enettv with l1, l2, tv null
        l1, l2, tv = 0, 0, 0
        class_weight = "auto"
        penalty_start = 1
        A = [sparse.csr_matrix((2, 2)) for i in xrange(3)]
        mod = LogisticRegressionL1L2TV(l1, l2, tv, A,
                                       penalty_start=penalty_start,
                                       class_weight=class_weight)
        mod.fit(Xtr, ytr)
        y_pred = mod.predict(Xte)
        prob_pred = mod.predict_probability(Xte)  # a posteriori probability
        beta = mod.beta
    elif method == 'enettv_parsimony_early_stopping':
        # enettv with l1, l2, tv null
        l1, l2, tv = 0, 0, 0
        class_weight = "auto"
        penalty_start = 1
        A = [sparse.csr_matrix((2, 2)) for i in xrange(3)]
        mod = LogisticRegressionL1L2TV(l1, l2, tv, A,
                                       penalty_start=penalty_start,
                                       class_weight=class_weight,
                                       algorithm_params={'max_iter': 100})
        mod.fit(Xtr, ytr)
        y_pred = mod.predict(Xte)
        prob_pred = mod.predict_probability(Xte)  # a posteriori probability
        beta = mod.beta
    ret = dict(y_pred=y_pred, prob_pred=prob_pred,
               y_true=yte,
               beta=beta)
    if output_collector:
        output_collector.collect(key, ret)
    else:
        return ret


def reducer(key, values):
    # key : string of intermediary key
    # load return dict correspondning to mapper ouput. they need to be loaded.
    # DEBUG
    # import glob, mapreduce
    # values = [mapreduce.OutputCollector(p)
    #        for p in glob.glob("/neurospin/brainomics/2014_deptms/MRI/results/*/0.05_0.45_0.45_0.1_-1.0/")]
    # Compute sd; ie.: compute results on each folds
    values = [item.load() for item in values[1:]]
    recall_mean_std = np.std([np.mean(precision_recall_fscore_support(
            item["y_true"].ravel(), item["y_pred"])[1]) for item in values]) \
            / np.sqrt(len(values))
    y_true = [item["y_true"].ravel() for item in values]
    y_pred = [item["y_pred"].ravel() for item in values]
    prob_pred = [item["prob_pred"].ravel() for item in values]
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    prob_pred = np.concatenate(prob_pred)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average=None)
    auc = roc_auc_score(y_true, prob_pred)  # area under curve score.
    n_ite = None
    betas = np.hstack([item["beta"] for item in values]).T
    R = np.corrcoef(betas)
    beta_cor_mean = np.mean(R[np.triu_indices_from(R, 1)])
    scores = OrderedDict()
    scores['recall_0'] = r[0]
    scores['recall_1'] = r[1]
    scores['recall_mean'] = r.mean()
    scores['recall_mean_std'] = recall_mean_std
    scores['precision_0'] = p[0]
    scores['precision_1'] = p[1]
    scores['precision_mean'] = p.mean()
    scores['f1_0'] = f[0]
    scores['f1_1'] = f[1]
    scores['f1_mean'] = f.mean()
    scores['support_0'] = s[0]
    scores['support_1'] = s[1]
    scores['n_ite'] = n_ite
    scores['auc'] = auc
    scores['beta_cor_mean'] = beta_cor_mean
    return scores

if __name__ == "__main__":

   #########################################################################
    ## load data
    BASE_PATH = "/neurospin/brainomics/2014_deptms"

    DATASET_PATH = os.path.join(BASE_PATH,    "datasets")

    modality = "MRI"
    INPUT_MOD_DIR = os.path.join(DATASET_PATH, modality)
    
    #OUTPUT_PATH = "/volatile/2014_deptms"
    #########################################################################
    ## Build config file

    WD = os.path.join(BASE_PATH,   "results_Demographic_only")

    if not os.path.exists(WD):
        os.makedirs(WD)

    INPUT_DATA_X = os.path.join(INPUT_MOD_DIR,
                                'X_' + modality + '_wb.npy')
    INPUT_DATA_y = os.path.join(INPUT_MOD_DIR,
                                'y.npy')
    # copy X, y, mask file names in the current directory
    shutil.copy2(INPUT_DATA_X, WD)
    shutil.copy2(INPUT_DATA_y, WD)

    #################################################################
    ## Create config file
    y = np.load(INPUT_DATA_y)
    SEED = 23071991
    cv = [[tr.tolist(), te.tolist()]
                    for tr, te in StratifiedKFold(y.ravel(), n_folds=NFOLDS,
                      shuffle=True, random_state=SEED)]
    cv.insert(0, None)  # first fold is None

    INPUT_DATA_X = os.path.basename(INPUT_DATA_X)
    INPUT_DATA_y = os.path.basename(INPUT_DATA_y)
    # parameters grid
    # Re-run with
    params = [["statsmodels"], ["log_parsimony"],
              ['enettv_parsimony'], ['enettv_parsimony_early_stopping']]
    user_func_filename = os.path.abspath(__file__)
    print "user_func", user_func_filename
    config = dict(data=dict(X=INPUT_DATA_X, y=INPUT_DATA_y),
                  params=params, resample=cv,
                  map_output="results",
                  user_func=user_func_filename,
                  reduce_group_by="params",
                  reduce_output="results.csv",
                  modality=modality)
    json.dump(config, open(os.path.join(WD, "config.json"), "w"))

    #################################################################
    # Build utils files: sync (push/pull) and PBS
    import brainomics.cluster_gabriel as clust_utils
    sync_push_filename, sync_pull_filename, WD_CLUSTER = \
        clust_utils.gabriel_make_sync_data_files(WD)
    cmd = "mapreduce.py --map  %s/config.json" % WD_CLUSTER
    clust_utils.gabriel_make_qsub_job_files(WD, cmd)
    ################################################################
    # Sync to cluster
    print "Sync data to gabriel.intra.cea.fr: "
    os.system(sync_push_filename)

    """######################################################################
    print "# Start by running Locally with 2 cores, to check that everything is OK)"
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
    #########################################################################
    print "# Reduce"
    print "mapreduce.py --reduce %s/config.json" % WD"""