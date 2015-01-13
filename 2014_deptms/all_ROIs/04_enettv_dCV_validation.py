# -*- coding: utf-8 -*-
"""
Created on Mon Jan 12 16:08:55 2015

@author: cp243490
"""
import os
import json
import numpy as np
import pandas as pd
import nibabel
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from parsimony.estimators import LogisticRegressionL1L2TV
from parsimony.algorithms.utils import Info
import parsimony.functions.nesterov.tv as tv_helper
import shutil
from scipy.stats import binom_test

from collections import OrderedDict


NFOLDS = 10


def load_globals(config):
    import mapreduce as GLOBAL  # access to global variables
    GLOBAL.DATA = GLOBAL.load_data(config["data"])
    GLOBAL.CRITERIA = config["params"]
    GLOBAL.MAP_OUTPUT = config['map_output']
    GLOBAL.OUTPUT_VALIDATION = config['output_validation']
    GLOBAL.PROB_CLASS1 = config["prob_class1"]
    GLOBAL.PENALTY_START = config["penalty_start"]
    STRUCTURE = nibabel.load(config["structure"])
    GLOBAL.A, _ = tv_helper.A_from_mask(STRUCTURE.get_data())
    GLOBAL.STRUCTURE = STRUCTURE
    GLOBAL.ROI = config["roi"]
    GLOBAL.MODEL = config["model"]


def resample(config, resample_nb):
    import mapreduce as GLOBAL  # access to global variables
    resample = config["resample"][resample_nb]
    if resample is not None:
        GLOBAL.DATA_RESAMPLED = {k: [GLOBAL.DATA[k][idx, ...]
                        for idx in resample[1:]]
                            for k in GLOBAL.DATA}
        GLOBAL.FOLD = resample[0]
    else:  # resample is None train == test
        GLOBAL.DATA_RESAMPLED = {k: [GLOBAL.DATA[k] for idx in [0, 1]]
                            for k in GLOBAL.DATA}
        GLOBAL.FOLD = 0


def mapper(key, output_collector):
    import mapreduce as GLOBAL  # access to global variables:
        #raise ImportError("could not import ")
    # GLOBAL.DATA, GLOBAL.STRUCTURE, GLOBAL.A
    # GLOBAL.DATA ::= {"X":[Xtrain, Xtest], "y":[ytrain, ytest]}
    # key: list of parameters
    Xtr = GLOBAL.DATA_RESAMPLED["X"][0]
    Xte = GLOBAL.DATA_RESAMPLED["X"][1]
    ytr = GLOBAL.DATA_RESAMPLED["y"][0]
    yte = GLOBAL.DATA_RESAMPLED["y"][1]
    print key, "Data shape:", Xtr.shape, Xte.shape, ytr.shape, yte.shape
    STRUCTURE = GLOBAL.STRUCTURE
    penalty_start = GLOBAL.PENALTY_START
    class_weight = "auto"  # unbiased
    n_fold = GLOBAL.FOLD
    model = GLOBAL.MODEL[key][n_fold]
    model_params = model.split('_')
    alpha = float(model_params[0])
    l1, l2 = alpha * float(model_params[1]), alpha * float(model_params[2])
    tv, k_ratio = alpha * float(model_params[3]), model_params[4]
    print "l1:%f, l2:%f, tv:%f, k_ratio:%f" % (l1, l2, tv, k_ratio)
    mask = STRUCTURE.get_data() != 0
    Xtr_r = Xtr
    Xte_r = Xte
    A = GLOBAL.A

    info = [Info.num_iter]
    mod = LogisticRegressionL1L2TV(l1, l2, tv, A, penalty_start=penalty_start,
                                   class_weight=class_weight,
                                   algorithm_params={'info': info})
    mod.fit(Xtr_r, ytr)
    y_pred = mod.predict(Xte_r)
    proba_pred = mod.predict_probability(Xte_r)  # a posteriori probability
    beta = mod.beta
    ret = dict(y_pred=y_pred, proba_pred=proba_pred, y_true=yte,
                   beta=beta,  mask=mask, model=model,
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
    penalty_start = GLOBAL.PENALTY_START
    prob_class1 = GLOBAL.PROB_CLASS1
    values = [item.load() for item in values[1:]]
    recall_mean_std = np.std([np.mean(precision_recall_fscore_support(
            item["y_true"].ravel(), item["y_pred"])[1]) for item in values]) \
            / np.sqrt(len(values))
    y_true = [item["y_true"].ravel() for item in values]
    y_pred = [item["y_pred"].ravel() for item in values]
    prob_pred = [item["proba_pred"].ravel() for item in values]
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    prob_pred = np.concatenate(prob_pred)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average=None)
    auc = roc_auc_score(y_true, prob_pred)  # area under curve score.
    n_ite = np.mean(np.array([item["n_iter"] for item in values]))
    betas = np.hstack([item["beta"][penalty_start:]  for item in values]).T
    R = np.corrcoef(betas)
    beta_cor_mean = np.mean(R[np.triu_indices_from(R, 1)])
    success = r * s
    success = success.astype('int')
    accuracy = (r[0] * s[0] + r[1] * s[1])
    accuracy = accuracy.astype('int')
    pvalue_recall0 = binom_test(success[0], s[0], 1 - prob_class1)
    pvalue_recall1 = binom_test(success[1], s[1], prob_class1)
    pvalue_accuracy = binom_test(accuracy, s[0] + s[1], p=0.5)
    scores = OrderedDict()
    scores['recall_0'] = r[0]
    scores['pvalue_recall_0'] = pvalue_recall0
    scores['recall_1'] = r[1]
    scores['pvalue_recall_1'] = pvalue_recall1
    scores['max_pvalue_recall'] = np.maximum(pvalue_recall0, pvalue_recall1)
    scores['recall_mean'] = r.mean()
    scores['recall_mean_std'] = recall_mean_std
    scores['accuracy'] = accuracy / float(s[0] + s[1])
    scores['pvalue_accuracy'] = pvalue_accuracy
    scores['precision_0'] = p[0]
    scores['precision_1'] = p[1]
    scores['precision_mean'] = p.mean()
    scores['f1_0'] = f[0]
    scores['f1_1'] = f[1]
    scores['f1_mean'] = f.mean()
    scores['support_0'] = s[0]
    scores['support_1'] = s[1]
    scores['n_ite_mean'] = n_ite
    scores['auc'] = auc
    scores['beta_cor_mean'] = beta_cor_mean
    # proportion of non zeros elements in betas matrix over all folds
    scores['prop_non_zeros_mean'] = float(np.count_nonzero(betas)) / \
                                    float(np.prod(betas.shape))
    return scores

if __name__ == "__main__":

   #########################################################################
    ## load data
    BASE_PATH = "/neurospin/brainomics/2014_deptms"

    DATASET_PATH = os.path.join(BASE_PATH,    "datasets")
    BASE_DATA_PATH = os.path.join(BASE_PATH,    "base_data")

    INPUT_ROIS_CSV = os.path.join(BASE_DATA_PATH,  "ROI_labels.csv")

    OUTPUT_ENETTV = os.path.join(BASE_PATH,   "results_enettv")
    if not os.path.exists(OUTPUT_ENETTV):
        os.makedirs(OUTPUT_ENETTV)

    penalty_start = 3

#
#########################################################################
#    ## Read ROIs csv
#    atlas = []
#    dict_rois = {}
#    df_rois = pd.read_csv(INPUT_ROIS_CSV)
#    for i, ROI_name_aal in enumerate(df_rois["ROI_name_aal"]):
#        cur = df_rois[df_rois.ROI_name_aal == ROI_name_aal]
#        label_ho = cur["label_ho"].values[0]
#        atlas_ho = cur["atlas_ho"].values[0]
#        roi_name = cur["ROI_name_deptms"].values[0]
#        if ((not cur.isnull()["atlas_ho"].values[0])
#            and (not cur.isnull()["ROI_name_deptms"].values[0])):
#            if not roi_name in dict_rois:
#                labels = np.asarray(label_ho.split(), dtype="int")
#                dict_rois[roi_name] = [labels]
#                dict_rois[roi_name].append(atlas_ho)
#
#    rois = list(set(df_rois["ROI_name_deptms"].values.tolist()))
#    rois = [x for x in rois if str(x) != 'nan']
#    rois.append("brain")  # add whole brain to rois
    #########################################################################
    ## Build config file for all couple (Modality, roi)

    modality = "MRI"
    print "Modality: ", modality
    DATA_MODALITY_PATH = os.path.join(DATASET_PATH, modality)

#    for roi in rois:
    roi = "Roiho-cingulumAnt"
    print "ROI", roi

    WD = os.path.join(OUTPUT_ENETTV, modality + '_' + roi)

    if not os.path.exists(WD):
        os.makedirs(WD)
    # copy X, y, mask file names in the current directory
    if not os.path.isfile(os.path.join(WD, 'X.npy')):
        INPUT_DATA_X = os.path.join(DATA_MODALITY_PATH,
                                    'X_' + modality + '_' + roi + '.npy')
        shutil.copy2(INPUT_DATA_X, os.path.join(WD, 'X.npy'))
    if not os.path.isfile(os.path.join(WD, 'y.npy')):
        INPUT_DATA_y = os.path.join(DATA_MODALITY_PATH,
                                    'y.npy')
        shutil.copy2(INPUT_DATA_y, os.path.join(WD, 'y.npy'))
    if not os.path.isfile(os.path.join(WD, 'mask.nii')):
        INPUT_MASK = os.path.join(DATA_MODALITY_PATH,
                                 'mask_' + modality + '_' + roi + '.nii')
        shutil.copy2(INPUT_MASK, os.path.join(WD, 'mask.nii'))

    #################################################################
    ## Read config file selection

    config_selection_file = os.path.join(WD, "config_dCV_selection.json")

    # open config file
    config_selection = json.load(open(config_selection_file))
    # get resample index for each fold
    resample = config_selection["resample"]
    resample = np.asarray(resample)
    # cv=[[tr, te] for each fold]
    cv = [[i + 1, resample[1 + (NFOLDS + 1) * i][1],
           resample[1 + (NFOLDS + 1) * i][2]] \
                        for i in xrange(NFOLDS)]
    cv.insert(0, None)  # first fold is None

    # Summary_file
    ##############
    summary_file = os.path.join(WD, config_selection["output_summary"])
    summary = pd.read_csv(summary_file)

    criteria = ['recall_mean', 'min_recall', 'max_pvalue_recall',
        'accuracy', 'pvalue_accuracy']
    model = {}
    for i, criterium in enumerate(criteria):
        model[criterium] = summary['param_opt_' + str(i + 1)].values.tolist()

    user_func_filename = os.path.abspath(__file__)
    print "user_func", user_func_filename
    config = dict(data=config_selection['data'],
                  params=criteria, model=model,
                  resample=cv,
                  structure=config_selection['structure'],
                  map_output=config_selection['map_output_validation'],
                  user_func=user_func_filename,
                  reduce_group_by="params",
                  output_validation="results_dCV_validation.csv",
                  penalty_start=config_selection['penalty_start'],
                  prob_class1=config_selection['prob_class1'],
                  roi=roi)
    json.dump(config, open(os.path.join(WD, "config_dCV_validation.json"),
                           "w"))

    #################################################################
    # Build utils files: sync (push/pull) and PBS
    import brainomics.cluster_gabriel as clust_utils
    sync_push_filename, sync_pull_filename, WD_CLUSTER = \
        clust_utils.gabriel_make_sync_data_files(WD)
    cmd = "mapreduce.py --map  %s/config_dCV_validation.json" % WD_CLUSTER
    clust_utils.gabriel_make_qsub_job_files(WD, cmd)
        ################################################################
#        # Sync to cluster
#        print "Sync data to gabriel.intra.cea.fr: "
#        os.system(sync_push_filename)

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