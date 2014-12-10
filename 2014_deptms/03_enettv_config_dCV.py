# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 15:44:50 2014

@author: cp243490

Create the config file for the multiple analysis enettv for the maskdep
for several datasets and contains the map and reduce
functions.
Double Cross validation:
    the first to select the model
    the second one to validate the model
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.cross_validation import StratifiedKFold
import nibabel
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from parsimony.estimators import LogisticRegressionL1L2TV
from parsimony.algorithms.utils import Info
import parsimony.functions.nesterov.tv as tv_helper
import shutil
from scipy.stats import binom_test

from collections import OrderedDict


NFOLDS = 5


def load_globals(config):
    import mapreduce as GLOBAL  # access to global variables
    GLOBAL.DATA = GLOBAL.load_data(config["data"])
    GLOBAL.PENALTY_START = config["penalty_start"]
    STRUCTURE = nibabel.load(config["structure"])
    GLOBAL.A, _ = tv_helper.A_from_mask(STRUCTURE.get_data())
    GLOBAL.STRUCTURE = STRUCTURE


def resample(config, resample_nb):
    import mapreduce as GLOBAL  # access to global variables
    resample = config["resample"][resample_nb]
    if resample is not None:
        if resample[3] is not None:
            GLOBAL.DATA_RESAMPLED_VALIDMODEL = {k: [GLOBAL.DATA[k][idx, ...]
                            for idx in resample[1:3]]
                                for k in GLOBAL.DATA}
            GLOBAL.DATA_RESAMPLED_SELECTMODEL = {k: [
                        GLOBAL.DATA_RESAMPLED_VALIDMODEL[k][0][idx, ...]
                            for idx in resample[3:]]
                                for k in GLOBAL.DATA}
            GLOBAL.N_FOLD = resample[0]
        elif resample[3] is None:
            # test = train for model selection and as before for model validation
            GLOBAL.DATA_RESAMPLED_VALIDMODEL = {k: [GLOBAL.DATA[k][idx, ...]
                            for idx in resample[1:3]]
                                for k in GLOBAL.DATA}
            GLOBAL.DATA_RESAMPLED_SELECTMODEL = {k: [
                        GLOBAL.DATA_RESAMPLED_VALIDMODEL[k][0]
                            for idx in [0, 1]]
                                for k in GLOBAL.DATA}
            GLOBAL.N_FOLD = resample[0]
    else:  # resample is None train == test
        GLOBAL.DATA_RESAMPLED_VALIDMODEL = {k: [GLOBAL.DATA[k]
                        for idx in [0, 1]] for k in GLOBAL.DATA}
        GLOBAL.DATA_RESAMPLED_SELECTMODEL = {k: [GLOBAL.DATA[k]
                        for idx in [0, 1]] for k in GLOBAL.DATA}
        GLOBAL.N_FOLD = 0


def mapper(key, output_collector):
    import mapreduce as GLOBAL  # access to global variables:
        #raise ImportError("could not import ")
    # GLOBAL.DATA, GLOBAL.STRUCTURE, GLOBAL.A
    # GLOBAL.DATA ::= {"X":[Xtrain, Xtest], "y":[ytrain, ytest]}
    # key: list of parameters
    n_fold = GLOBAL.N_FOLD
    Xcalib = GLOBAL.DATA_RESAMPLED_VALIDMODEL["X"][0]
    Xte = GLOBAL.DATA_RESAMPLED_VALIDMODEL["X"][1]
    Xtr = GLOBAL.DATA_RESAMPLED_SELECTMODEL["X"][0]
    Xval = GLOBAL.DATA_RESAMPLED_SELECTMODEL["X"][1]
    ycalib = GLOBAL.DATA_RESAMPLED_VALIDMODEL["y"][0]
    yte = GLOBAL.DATA_RESAMPLED_VALIDMODEL["y"][1]
    ytr = GLOBAL.DATA_RESAMPLED_SELECTMODEL["y"][0]
    yval = GLOBAL.DATA_RESAMPLED_SELECTMODEL["y"][1]
    print key, "Data shape:", Xcalib.shape, Xte.shape, Xtr.shape, Xval.shape,
    print ycalib.shape, yte.shape, ytr.shape, yval.shape
    STRUCTURE = GLOBAL.STRUCTURE
    #alpha, ratio_l1, ratio_l2, ratio_tv, k = key
    #key = np.array(key)
    penalty_start = GLOBAL.PENALTY_START
    class_weight = "auto"  # unbiased
    alpha = float(key[0])
    l1, l2 = alpha * float(key[1]), alpha * float(key[2])
    tv, k_ratio = alpha * float(key[3]), key[4]
    print "l1:%f, l2:%f, tv:%f, k_ratio:%f" % (l1, l2, tv, k_ratio)
    mask = STRUCTURE.get_data() != 0
    Xtr_r = Xtr
    Xval_r = Xval
    A = GLOBAL.A
    info = [Info.num_iter]
    mod = LogisticRegressionL1L2TV(l1, l2, tv, A, penalty_start=penalty_start,
                                   class_weight=class_weight,
                                   algorithm_params={'info': info})
    mod.fit(Xtr_r, ytr)
    y_pred = mod.predict(Xval_r)
    proba_pred = mod.predict_probability(Xval_r)  # a posteriori probability
    beta = mod.beta
    ret = dict(y_pred=y_pred, proba_pred=proba_pred, y_true=yval,
               X_calib=Xcalib, y_calib=ycalib, X_test=Xte, y_test=yte,
               n_fold=n_fold, beta=beta,  mask=mask,
               n_iter=mod.get_info()['num_iter'])
    if output_collector:
        output_collector.collect(key, ret)
    else:
        return ret


def reducer(key, values):
    import mapreduce as GLOBAL
    # key : string of intermediary key
    # load return dict correspondning to mapper ouput. they need to be loaded.
    # DEBUG
    # import glob, mapreduce
    # values = [mapreduce.OutputCollector(p)
    #        for p in glob.glob("/neurospin/brainomics/2014_deptms/MRI/results/*/0.05_0.45_0.45_0.1_-1.0/")]
    # Compute sd; ie.: compute results on each folds
    penalty_start = GLOBAL.PENALTY_START
    keys = ["0.05_1.0_0.0_0.0_-1.0",  "0.05_0.0_0.0_1.0_-1.0",
            "0.05_0.3_0.0_0.7_-1.0",  "0.05_0.7_0.0_0.3_-1.0"]
    scores_CV = OrderedDict()
    for key in keys:
        prob_class1 = GLOBAL.PROB_CLASS1
        values = [item.load() for item in values[1:]]
        n_fold = [item["n_fold"] for item in values]
#        X_calib = [item["X_calib"].ravel() for item in values]
#        y_calib = [item["y_calib"].ravel() for item in values]
#        X_test = [item["X_test"].ravel() for item in values]
#        y_test = [item["y_test"].ravel() for item in values]
        recall_mean_std = np.std([np.mean(precision_recall_fscore_support(
             item["y_true"].ravel(), item["y_pred"])[1]) for item in values]) \
             / np.sqrt(len(values))
        y_true = [item["y_true"].ravel() for item in values]
        y_pred = [item["y_pred"].ravel() for item in values]
        prob_pred = [item["proba_pred"].ravel() for item in values]
        betas = [item["beta"][penalty_start:]  for item in values]
        n_iter = [item["n_iter"] for item in values]
        dict_dcv = OrderedDict()
        dict_dcv["n_fold"] = n_fold
#        dict_dcv["X_calib"] = X_calib, dict_dcv["y_calib"] = y_calib
#        dict_dcv["X_test"] = X_test, dict_dcv["y_test"] = y_test
        dict_dcv["y_true"] = y_true, dict_dcv["y_pred"] = y_pred
        dict_dcv["prob_pred"] = prob_pred, dict_dcv["betas"] = betas
        dict_dcv["recall_mean_std"] = recall_mean_std,
        dict_dcv["n_iter"] = n_iter
        pd.DataFrame.from_dict(dict_dcv)
        n_fold_groups = dict_dcv.groupby("n_fold")
        for n_fold_val, n_fold_group in n_fold_groups:
            y_true = np.concatenate(n_fold_group.y_true)
            y_pred = np.concatenate(n_fold_group.y_pred)
            prob_pred = np.concatenate(n_fold_group.prob_pred)
            p, r, f, s = precision_recall_fscore_support(y_true, y_pred,
                                                         average=None)
            auc = roc_auc_score(y_true, prob_pred)  # area under curve score.
            n_ite = np.mean(np.array(n_fold_group.n_iter))
            betas = np.hstack(n_fold_group.betas).T
            R = np.corrcoef(betas)
            beta_cor_mean = np.mean(R[np.triu_indices_from(R, 1)])
            success = r * s
            success = success.astype('int')
            pvalue_class0 = binom_test(success[0], s[0], 1 - prob_class1)
            pvalue_class1 = binom_test(success[1], s[1], prob_class1)
            a, l1, tv = float(key[0]), float(key[1]), float(key[2])
            scores_CV['n_fold'] = n_fold_val
            scores_CV['a'] = a
            scores_CV['l1'] = l1
            scores_CV['tv'] = tv
            scores_CV['recall_0'] = r[0]
            scores_CV['pvalue_recall_0'] = pvalue_class0
            scores_CV['recall_1'] = r[1]
            scores_CV['pvalue_recall_1'] = pvalue_class1
            scores_CV['min_recall'] = np.minimum(r[0], r[1])
            scores_CV['max_pvalue_recall'] = np.maximum(pvalue_class0,
                                                        pvalue_class1)
            scores_CV['recall_mean'] = r.mean()
            scores_CV['recall_mean_std'] = recall_mean_std
            scores_CV['precision_0'] = p[0]
            scores_CV['precision_1'] = p[1]
            scores_CV['precision_mean'] = p.mean()
            scores_CV['f1_0'] = f[0]
            scores_CV['f1_1'] = f[1]
            scores_CV['f1_mean'] = f.mean()
            scores_CV['support_0'] = s[0]
            scores_CV['support_1'] = s[1]
            scores_CV['n_ite_mean'] = n_ite
            scores_CV['auc'] = auc
            scores_CV['beta_cor_mean'] = beta_cor_mean
            # proportion of non zeros elements in betas matrix over all folds
            scores_CV['prop_non_zeros_mean'] = float(np.count_nonzero(betas)) \
                                            / float(np.prod(betas.shape))
    pd.DataFrame.from_dict(scores_CV)
    fold_groups = scores_CV.groupby('fold')
    scores_dCV = OrderedDict()
    criteria = ['recall_mean', 'min_recall']
    for fold_val, fold_group in fold_groups:
        scores_dCV['n_fold'] = fold_val
        for i, crit in enumerate(criteria):
            scores_dCV['criteria_' + i] = criteria
            loc_opt = np.argmax(fold_group[crit])
            value_opt = np.max(fold_group['crit'])
            scores_dCV['value_criteria_' + i] = value_opt
            a_opt = fold_group.a[loc_opt], l1_opt = fold_group.l1[loc_opt]
            tv_opt = fold_group.tv[loc_opt]
            scores_dCV['a_opt_' + i] = a_opt
            scores_dCV['l1_opt_' + i] = l1_opt
            scores_dCV['tv_opt_' + i] = tv_opt
    return scores_dCV


##############################################################################
## Run all
def run_all(config):
    import mapreduce
    BASE_PATH = "/neurospin/brainomics/2014_deptms"
    OUTPUT_ENETTV = os.path.join(BASE_PATH,   "results_enettv")
    params = config["params"][0]
    key = '_'.join([str(p) for p in params])
    modality = config["modality"]
    roi = config["roi"]
    WD = os.path.join(OUTPUT_ENETTV, modality + '_' + roi)
    #class GLOBAL: DATA = dict()
    load_globals(config)
    OUTPUT = os.path.join(WD, 'test', key)
    if not os.path.exists(OUTPUT):
        os.makedirs(OUTPUT)
    # run /home/ed203246/bin/mapreduce.py
    oc = mapreduce.OutputCollector(OUTPUT)
    X = np.load(os.path.join(WD,
                             'X_' + modality + '_' + roi + '.npy'))
    y = np.load(os.path.join(WD,  'y.npy'))
    mapreduce.DATA_RESAMPLED = {}
    mapreduce.DATA_RESAMPLED["X"] = [X, X]
    mapreduce.DATA_RESAMPLED["y"] = [y, y]
    params = np.array([float(p) for p in key.split("_")])
    mapper(params, oc)
    #oc.collect(key=key, value=ret)


if __name__ == "__main__":

   #########################################################################
    ## load data
    BASE_PATH = "/neurospin/brainomics/2014_deptms"

    DATASET_PATH = os.path.join(BASE_PATH,    "datasets")

    MASKDEP_PATH = os.path.join(BASE_PATH,   "maskdep")
    if not os.path.exists(MASKDEP_PATH):
        os.makedirs(MASKDEP_PATH)

    penalty_start = 3

    #########################################################################
    ## Build config file for all couple (Modality, roi)
    modality = "MRI"
    roi = "Mask_dep"

    WD = os.path.join(MASKDEP_PATH, 'results_enettv')

    if not os.path.exists(WD):
        os.makedirs(WD)
    INPUT_DATA_X = os.path.join(DATASET_PATH, "MRI",
                                'X_MRI_Maskdep.npy')
    INPUT_DATA_y = os.path.join(DATASET_PATH, "MRI",
                                'y.npy')
    INPUT_MASK = os.path.join(DATASET_PATH, "MRI",
                              'mask_MRI_Maskdep.nii')
    # copy X, y, mask file names in the current directory
    shutil.copy2(INPUT_DATA_X, os.path.join(WD, 'X.npy'))
    shutil.copy2(INPUT_DATA_y, WD)
    shutil.copy2(INPUT_MASK, os.path.join(WD, 'mask.nii'))

    #################################################################
    ## Create config file
    y = np.load(INPUT_DATA_y)
    prob_class1 = np.count_nonzero(y) / float(len(y))

    # resampling indexes for the double cross validation
    SEED_CV1 = 23071991
    SEED_CV2 = 5061931
    dcv = []
    n_fold = 1
    for calib, te in StratifiedKFold(y.ravel(), n_folds=NFOLDS,
                                     shuffle=True, random_state=SEED_CV1):
        assert((len(calib) + len(te)) == len(y))
        y_calib = np.array([y[i] for i in calib.tolist()])
        for val, tr in StratifiedKFold(y_calib.ravel(), n_folds=NFOLDS,
                                       shuffle=True, random_state=SEED_CV2):
            assert((len(val) + len(tr)) == len(calib))
            dcv.append([n_fold, calib.tolist(), te.tolist(),
                        val.tolist(), tr.tolist()])
            dcv.append([n_fold, calib.tolist(), te.tolist(), None])
        n_fold += 1
    dcv.insert(0, None)  # first fold is None

    INPUT_DATA_X = 'X.npy'
    INPUT_DATA_y = 'y.npy'
    INPUT_MASK = 'mask.nii'
    # parameters grid
    # Re-run with
    l1 = np.array([[0], [0.3], [0.7], [1]])
    l1l2tv = np.hstack([l1, l1 - l1, 1 - l1])
    alphas = [.01, .05, .1, .5, 1.]
    k_ratio = 1
    alphal1l2tv = np.concatenate([np.c_[np.array([[alpha]] * \
                                                   l1l2tv.shape[0]),
                                        l1l2tv] for alpha in alphas])
    alphal1l2tvk = np.concatenate([np.c_[alphal1l2tv,
                                         np.array([[k_ratio]] * \
                                                 alphal1l2tv.shape[0])]])
    params = [params.tolist() for params in alphal1l2tvk]
    user_func_filename = os.path.join(os.environ["HOME"], "git",
        "scripts", "2014_deptms", "03_enettv_config_dCV.py")
    print "user_func", user_func_filename
    config = dict(data=dict(X=INPUT_DATA_X, y=INPUT_DATA_y),
                  params=params, resample=dcv,
                  structure=INPUT_MASK,
                  map_output="results_dCV",
                  user_func=user_func_filename,
                  reduce_group_by="params",
                  reduce_output="results_dCV.csv",
                  penalty_start=penalty_start,
                  prob_class1=prob_class1,
                  modality=modality,
                  roi=roi)
    json.dump(config, open(os.path.join(WD, "config_dCV.json"), "w"))

    #################################################################
    # Build utils files: sync (push/pull) and PBS
    import brainomics.cluster_gabriel as clust_utils
    sync_push_filename, sync_pull_filename, WD_CLUSTER = \
        clust_utils.gabriel_make_sync_data_files(WD)
    cmd = "mapreduce.py --map  %s/config_dCV.json" % WD_CLUSTER
    clust_utils.gabriel_make_qsub_job_files(WD, cmd)
    ################################################################
    # Sync to cluster
    print "Sync data to gabriel.intra.cea.fr: "
    os.system(sync_push_filename)

    """######################################################################
    print "# Start by running Locally with 2 cores, to check that everything is OK)"
    print "mapreduce.py --map %s/config.json --ncore 2" % WD
    #os.system("mapreduce.py --mode map --config %s/config_dCV.json" % WD)
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