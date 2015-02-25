# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 14:49:40 2015

@author: cp243490

Multivariate analysis (Machine Learning).
5-folds Double Cross-validation (10 x 10 folds) + logistic regression Enettv
on randomly permutated y vector (to tets the significance of prediction rates).

Create config file for the outer cross-validation (10folds): Model Validation
of NRNDPERMS = 1000 permuted y.
Outer Cross-validation on (X_valid, y_valid) and (X_calib, y_calib)

Validation of the selected model for each criterion:
criteria = {recall_mean, min_recall, accuracy}

config_file:
/neurospin/brainomics/2014_deptms/results_enettv/MRI_Roiho-hippo/config_rndperm_dCV_validation.json

OUTPUT_DIRECTORY:
 /neurospin/brainomics/2014_deptms/results_enettv/MRI_Roiho-hippo/rndperm_results/*

OUTPUT FILE wich gives the corrected p-values:
 /neurospin/brainomics/2014_deptms/results_enettv/MRI_Roiho-hippo/rndperm_results/pvals_stats_permutations.csv
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


NFOLDS = 10
NRNDPERMS = 1000


def load_globals(config):
    import mapreduce as GLOBAL  # access to global variables
    GLOBAL.DATA = GLOBAL.load_data(config["data"])
    GLOBAL.CRITERIA = config["params"]
    GLOBAL.MAP_OUTPUT = config["map_output"]
    GLOBAL.OUTPUT_PATH = config["output_path"]
    GLOBAL.OUTPUT_PERMUTATIONS = config["output_permutations"]
    GLOBAL.PENALTY_START = config["penalty_start"]
    STRUCTURE = nibabel.load(config["structure"])
    GLOBAL.A, _ = tv_helper.A_from_mask(STRUCTURE.get_data())
    GLOBAL.STRUCTURE = STRUCTURE
    GLOBAL.ROI = config["roi"]
    GLOBAL.SELECTION = pd.read_csv(os.path.join(config["output_path"],
                                                config["output_summary"]))


def resample(config, resample_nb):
    import mapreduce as GLOBAL  # access to global variables
    #GLOBAL.DATA = GLOBAL.load_data(config["data"])
    resample = config["resample"][resample_nb]
    GLOBAL.NRNDPERM = resample[0]
    GLOBAL.N_FOLD = resample[1]
    y_perm = resample[2]
    y_perm = np.asarray(y_perm)
    if resample is not None:
        GLOBAL.DATA_RESAMPLED = dict(
            X=[GLOBAL.DATA['X'][idx, ...] for idx in resample[3:]],
            y=[y_perm[idx, ...] for idx in resample[3:]])
    else:  # resample is None train == test
        GLOBAL.DATA_RESAMPLED = {k: [GLOBAL.DATA[k] for idx in [0, 1]]
                            for k in GLOBAL.DATA}
        GLOBAL.N_FOLD = 0


def mapper(key, output_collector):
    import mapreduce as GLOBAL  # access to global variables:
        #raise ImportError("could not import ")
    # GLOBAL.DATA, GLOBAL.STRUCTURE, GLOBAL.A
    # GLOBAL.DATA ::= {"X":[Xtrain, ytrain], "y":[Xtest, ytest]}
    # key: list of parameters
    nfold = GLOBAL.N_FOLD
    nrndperm = GLOBAL.NRNDPERM
    # data for model validation (2nd cross validation, outer loop)
    Xvalid = GLOBAL.DATA_RESAMPLED["X"][0]
    Xcalib = GLOBAL.DATA_RESAMPLED["X"][1]
    yvalid = GLOBAL.DATA_RESAMPLED["y"][0]
    ycalib = GLOBAL.DATA_RESAMPLED["y"][1]

    criterion = ''
    for c in key: criterion += c
    print criterion, "Data shape:", Xcalib.shape, Xvalid.shape, \
                                    ycalib.shape, yvalid.shape
    penalty_start = GLOBAL.PENALTY_START
    class_weight = "auto"  # unbiased
    selection = GLOBAL.SELECTION
    #set of parameters (alpha, l1, l2, tv) selected
    model = selection[(selection.n_fold == nfold) & \
                      (selection.permutation == nrndperm)] \
                     ['param_opt_' + criterion].values[0]
    model_params = model.split('_')
    alpha = float(model_params[0])
    l1, l2 = alpha * float(model_params[1]), alpha * float(model_params[2])
    tv, k_ratio = alpha * float(model_params[3]), float(model_params[4])
    print "l1:%f, l2:%f, tv:%f, k_ratio:%f" % (l1, l2, tv, k_ratio)
    A = GLOBAL.A
    info = [Info.num_iter]
    mod = LogisticRegressionL1L2TV(l1, l2, tv, A, penalty_start=penalty_start,
                                   class_weight=class_weight,
                                   algorithm_params={'info': info})
    mod.fit(Xcalib, ycalib)
    y_pred = mod.predict(Xvalid)
    proba_pred = mod.predict_probability(Xvalid)  # a posteriori probability
    ret = dict(y_pred=y_pred, proba_pred=proba_pred, y_true=yvalid)
    if output_collector:
        output_collector.collect(key, ret)
    else:
        return ret


def reducer(key, values):
    # key : string of intermediary key
    # load return dict correspondning to mapper ouput. they need to be loaded.
    # DEBUG
    import mapreduce as GLOBAL
    output_permutations = GLOBAL.OUTPUT_PERMUTATIONS
    map_output = GLOBAL.MAP_OUTPUT
    output_path = GLOBAL.OUTPUT_PATH
    roi = GLOBAL.ROI
    BASE = os.path.join("/neurospin/brainomics/2014_deptms/results_enettv/",
                        "MRI_" + roi,
                        map_output)
    INPUT = BASE + "/%i/%s"
    OUTPUT = BASE + "/../" + output_path
    if not os.path.exists(OUTPUT):
        os.makedirs(OUTPUT)
    criteria = GLOBAL.CRITERIA
    keys = ['_'.join(str(e) for e in a) for a in criteria]
    OK = 0
    # params = criteria = ['recall_mean', 'min_recall', 'max_pvalue_recall',
    #                     'accuracy', 'pvalue_accuracy']
    if not OK:
        for key in keys:
            print "key: ", key
            paths_CV_all = [INPUT % (perm, key) \
                    for perm in xrange(NFOLDS * NRNDPERMS)]
            idx_CV_blocks = range(0, (NFOLDS * NRNDPERMS) + NFOLDS, NFOLDS)
            recall_0_perms = np.zeros(NRNDPERMS)
            recall_1_perms = np.zeros(NRNDPERMS)
            recall_mean_perms = np.zeros(NRNDPERMS)
            accuracy_perms = np.zeros(NRNDPERMS)
            auc_perms = np.zeros(NRNDPERMS)
            crit = key[0:len(key):2]
            if not os.path.isfile(OUTPUT + \
                                  "/perms_validation_" + crit + ".npz"):
                for perm in xrange(NRNDPERMS):
                    print "perm: ", perm
                    paths_CV_blocks = paths_CV_all[idx_CV_blocks[perm]:\
                                                    idx_CV_blocks[perm + 1]]
                    values = [GLOBAL.OutputCollector(p) \
                                for p in paths_CV_blocks]
                    values = [item.load() for item in values]
                    y_true = [item["y_true"].ravel() for item in values]
                    y_pred = [item["y_pred"].ravel() for item in values]
                    prob_pred = [item["proba_pred"].ravel() for item in values]
                    y_true = np.concatenate(y_true)
                    y_pred = np.concatenate(y_pred)
                    prob_pred = np.concatenate(prob_pred)
                    p, r, f, s = precision_recall_fscore_support(y_true,
                                                                 y_pred,
                                                                 average=None)
                    auc = roc_auc_score(y_true, prob_pred)
                    success = r * s
                    success = success.astype('int')
                    accuracy = (r[0] * s[0] + r[1] * s[1])
                    accuracy = accuracy.astype('int')
                    recall_0_perms[perm] = r[0]
                    recall_1_perms[perm] = r[1]
                    recall_mean_perms[perm] = r.mean()
                    accuracy_perms[perm] = accuracy / float(s[0] + s[1])
                    auc_perms[perm] = auc
                # END PERMS
                print "save", crit
                np.savez_compressed(OUTPUT + \
                                    "/perms_validation_" + crit + ".npz",
                                recall_0=recall_0_perms,
                                recall_1=recall_1_perms,
                                recall_mean=recall_mean_perms,
                                accuracy=accuracy_perms,
                                auc=auc_perms)
        OK = 1
    #pvals
    if  not os.path.isfile(os.path.join(OUTPUT, output_permutations)):
        print "Derive p-values"
        perms = dict()
        for i, key in enumerate(keys):
            print "crit: ", crit
            crit = key[0:len(key):2]
            perms[crit] = np.load(OUTPUT + \
                                    "/perms_validation_" + crit + ".npz")
        print keys
        [recall_mean, min_recall, accuracy] = [keys[0][0:len(keys[0]):2],
                                               keys[1][0:len(keys[1]):2],
                                               keys[2][0:len(keys[2]):2]]
        print [recall_mean, min_recall, accuracy]
        # Read true scores
        true = pd.read_csv(os.path.join(BASE, "..",
                                        "results_dCV_validation.csv"))
        true_recall_mean = true[true.params == recall_mean].iloc[0]
        true_min_recall = true[true.params == min_recall].iloc[0]
        true_accuracy = true[true.params == accuracy].iloc[0]
        # pvals corrected for multiple comparisons
        nperms = float(len(perms[recall_mean]['recall_0']))
        from collections import OrderedDict
        pvals = OrderedDict()
        #cond: criterion used to select the model
        pvals["cond"] = ['recall_mean'] * 5 + ['min_recall'] * 5 + \
                        ['accuracy'] * 5
        #stat: statitics associated to the p-value
        pvals["stat"] = ['recall_0', 'recall_1', 'recall_mean',
                         'accuracy', 'auc'] * 3
        pvals["pval"] = [
        np.sum(perms[recall_mean]['recall_0'] > true_recall_mean["recall_0"]),
        np.sum(perms[recall_mean]['recall_1'] > true_recall_mean["recall_1"]),
        np.sum(perms[recall_mean]['recall_mean'] > true_recall_mean["recall_mean"]),
        np.sum(perms[recall_mean]['accuracy'] > true_recall_mean["accuracy"]),
        np.sum(perms[recall_mean]['auc'] > true_recall_mean["auc"]),
    
        np.sum(perms[min_recall]['recall_0'] > true_min_recall["recall_0"]),
        np.sum(perms[min_recall]['recall_1'] > true_min_recall["recall_1"]),
        np.sum(perms[min_recall]['recall_mean'] > true_min_recall["recall_mean"]),
        np.sum(perms[min_recall]['accuracy'] > true_min_recall["accuracy"]),
        np.sum(perms[min_recall]['auc'] > true_min_recall["auc"]),
    
        np.sum(perms[accuracy]['recall_0'] > true_accuracy["recall_0"]),
        np.sum(perms[accuracy]['recall_1'] > true_accuracy["recall_1"]),
        np.sum(perms[accuracy]['recall_mean'] > true_accuracy["recall_mean"]),
        np.sum(perms[accuracy]['accuracy'] > true_accuracy["accuracy"]),
        np.sum(perms[accuracy]['auc'] > true_accuracy["auc"])]
    
        pvals = pd.DataFrame(pvals)
        pvals["pval"] /= float(nperms)
        pvals.to_csv(os.path.join(OUTPUT, output_permutations),
                     index=False)
    return {}


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

#    #########################################################################
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
#           if ((not roi_name in dict_rois)
#              and (roi_name != "Maskdep-sub")
#              and (roi_name != "Maskdep-cort")):
#                labels = np.asarray(label_ho.split(), dtype="int")
#                dict_rois[roi_name] = [labels]
#                dict_rois[roi_name].append(atlas_ho)
#
#    rois = list(set(df_rois["ROI_name_deptms"].values.tolist()))
#    rois = [x for x in rois if str(x) != 'nan']
#    rois.append("brain")  # add whole brain to rois
    #########################################################################
    ## Build config file
    modality = "MRI"
    print "Modality: ", modality
    DATA_MODALITY_PATH = os.path.join(DATASET_PATH, modality)

#    for roi in rois:
    roi = "Roiho-caudate"
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
                                 'mask_' + roi + '.nii')
        shutil.copy2(INPUT_MASK, os.path.join(WD, 'mask.nii'))
    INPUT_DATA_X = 'X.npy'
    INPUT_DATA_y = 'y.npy'
    INPUT_MASK = 'mask.nii'
    #################################################################
    ## Create config file
    # config permutations file
    config_permutation_file = os.path.join(WD, "config_rndperm_dCV_selection.json")
    # open config file selection
    config_permutation = json.load(open(config_permutation_file))
    # get resample index for each fold
    resample = config_permutation["resample"]

    # rndperm=[n_perm, n_fold, y_perm, valid, calib]]
    rndperm = [resample[i * NFOLDS][:5] for i in xrange(NFOLDS * NRNDPERMS)]

    criteria = ['recall_mean', 'min_recall', 'accuracy']
    user_func_filename = os.path.abspath(__file__)
    print "user_func", user_func_filename
    # Use relative path from config.json
    config = dict(data=dict(X=INPUT_DATA_X, y=INPUT_DATA_y),
                  params=criteria, resample=rndperm,
                  structure=INPUT_MASK,
                  penalty_start=3,
                  map_output="rndperm_validation",
                  output_path=config_permutation["output_path"],
                  output_summary=config_permutation["output_summary"],
                  output_permutations=config_permutation["output_permutations"],
                  user_func=user_func_filename,
                  reduce_group_by="params",
                  roi=roi)
    json.dump(config, open(os.path.join(WD, "config_rndperm_dCV_validation.json"),
                           "w"))

    #####################################################################
    # Build utils files: sync (push/pull) and PBS
    import brainomics.cluster_gabriel as clust_utils
    sync_push_filename, sync_pull_filename, WD_CLUSTER = \
        clust_utils.gabriel_make_sync_data_files(WD)
    cmd = "mapreduce.py --map  %s/config_rndperm_dCV_validation.json" % WD_CLUSTER
    clust_utils.gabriel_make_qsub_job_files(WD, cmd, suffix="_rndperm")
#        ####################################################################
#        # Sync to cluster
#        print "Sync data to gabriel.intra.cea.fr: "
#        os.system(sync_push_filename)
#    ########################################################################
#    print "# Start by running Locally with 2 cores, to check that everything os OK)"
#    print "Interrupt after a while CTL-C"
#    print "mapreduce.py --map %s/config_rndperm.json --ncore 2" % WD
#    #os.system("mapreduce.py --mode map --config %s/config.json" % WD)
#    print "# 1) Log on gabriel:"
#    print 'ssh -t gabriel.intra.cea.fr'
#    print "# 2) Run one Job to test"
#    print "qsub -I"
#    print "cd %s" % WD_CLUSTER
#    print "./job_Global_long.pbs"
#    print "# 3) Run on cluster"
#    print "qsub job_Global_long.pbs"
#    print "# 4) Log out and pull Pull"
#    print "exit"
#    print sync_pull_filename
#    ########################################################################