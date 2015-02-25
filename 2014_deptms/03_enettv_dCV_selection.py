# -*- coding: utf-8 -*-
"""
Created on Wed Jan  7 10:32:30 2015

@author: cp243490

Multivariate analysis (Machine Learning).
5-folds Double Cross-validation (10 x 10 folds) + logistic regression Enettv.
X =  [X_valid, X calib] = [X_valid, X_test, X_train]
y =  [y_valid, y calib] = [y_valid, y_test, y_train]
(X_calib = [X_test, X_train] and y_calib = [y_test, y_train])

Create config file for the inner cross-validation (10folds): Model Selection.
Inner Cross-validation on (X_test, y_test) and (X_train, y_train)

Select the model among a range of parameters (alpha, l1, l2, tv):
alphal1l2tv = [[0.05, 0.07, 0.63, 0.3], [0.05, 0.63, 0.07, 0.3],
              [0.5, 0.07, 0.63, 0.3], [0.5, 0.63, 0.07, 0.3] ]
Select the model that maximize a criterion:
    criteria = {recall_mean, min_recall, accuracy}


config_file:
/neurospin/brainomics/2014_deptms/results_enettv/MRI_Roiho-hippo/config_dCV_selection.json

OUTPUT_DIRECTORY:
 /neurospin/brainomics/2014_deptms/results_enettv/MRI_Roiho-hippo/results_dCV_selection/*
OUTPUT_FILE:
 /neurospin/brainomics/2014_deptms/results_enettv/MRI_Roiho-hippo/results_dCV_selection.csv
 /neurospin/brainomics/2014_deptms/results_enettv/MRI_Roiho-hippo/summary_selection.csv
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


NFOLDS = 10


def load_globals(config):
    import mapreduce as GLOBAL  # access to global variables
    GLOBAL.DATA = GLOBAL.load_data(config["data"])
    GLOBAL.PARAMS = config["params"]
    GLOBAL.MAP_OUTPUT = config['map_output']
    GLOBAL.OUTPUT_SELECTION = config['output_selection']
    GLOBAL.OUTPUT_SUMMARY = config['output_summary']
    GLOBAL.PROB_CLASS1 = config["prob_class1"]
    GLOBAL.PENALTY_START = config["penalty_start"]
    STRUCTURE = nibabel.load(config["structure"])
    GLOBAL.A, _ = tv_helper.A_from_mask(STRUCTURE.get_data())
    GLOBAL.STRUCTURE = STRUCTURE
    GLOBAL.ROI = config["roi"]


def resample(config, resample_nb):
    import mapreduce as GLOBAL  # access to global variables
    resample = config["resample"][resample_nb]
    # resample = [n_fold_outer, valid, calib, test, train] where
    # test + train = calib
    # for model selection: cross validation on (test, train)
    if resample is not None:
        # general case
        if resample[3] is not None:
            GLOBAL.DATA_RESAMPLED_VALIDMODEL = {k: [GLOBAL.DATA[k][idx, ...]
                            for idx in resample[1:3]]
                                for k in GLOBAL.DATA}
            GLOBAL.DATA_RESAMPLED_SELECTMODEL = {k: [
                        GLOBAL.DATA_RESAMPLED_VALIDMODEL[k][1][idx, ...]
                            for idx in resample[3:]]
                                for k in GLOBAL.DATA}
            GLOBAL.N_FOLD = resample[0]
        # test = calib and train = calib (inner fold 0)
        elif resample[3] is None:
            # test = train for model selection
            # and as before for model validation
            GLOBAL.DATA_RESAMPLED_VALIDMODEL = {k: [GLOBAL.DATA[k][idx, ...]
                            for idx in resample[1:3]]
                                for k in GLOBAL.DATA}
            GLOBAL.DATA_RESAMPLED_SELECTMODEL = {k: [
                        GLOBAL.DATA_RESAMPLED_VALIDMODEL[k][1]
                            for idx in [0, 1]]
                                for k in GLOBAL.DATA}
            GLOBAL.N_FOLD = resample[0]
    # resample is None train == test
    else:
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

    # data for model validation (2nd cross validation, outer loop)
    Xvalid = GLOBAL.DATA_RESAMPLED_VALIDMODEL["X"][0]
    Xcalib = GLOBAL.DATA_RESAMPLED_VALIDMODEL["X"][1]
    yvalid = GLOBAL.DATA_RESAMPLED_VALIDMODEL["y"][0]
    ycalib = GLOBAL.DATA_RESAMPLED_VALIDMODEL["y"][1]

    # data for model selection (1rst cross validation, outer loop)
    Xtest = GLOBAL.DATA_RESAMPLED_SELECTMODEL["X"][0]
    Xtrain = GLOBAL.DATA_RESAMPLED_SELECTMODEL["X"][1]
    ytest = GLOBAL.DATA_RESAMPLED_VALIDMODEL["y"][0]
    ytrain = GLOBAL.DATA_RESAMPLED_VALIDMODEL["y"][1]

    print key, "Data shape:", Xvalid.shape, Xcalib.shape, Xtest.shape,
    Xtrain.shape
    STRUCTURE = GLOBAL.STRUCTURE
    #(alpha, ratio_l1, ratio_l2, ratio_tv, ratio_k) = key
    #key = np.array(key)
    penalty_start = GLOBAL.PENALTY_START
    class_weight = "auto"  # unbiased
    alpha = float(key[0])
    l1, l2 = alpha * float(key[1]), alpha * float(key[2])
    tv, k_ratio = alpha * float(key[3]), key[4]
    print "l1:%f, l2:%f, tv:%f, k_ratio:%f" % (l1, l2, tv, k_ratio)
    mask = STRUCTURE.get_data() != 0
    A = GLOBAL.A
    info = [Info.num_iter]
    mod = LogisticRegressionL1L2TV(l1, l2, tv, A, penalty_start=penalty_start,
                                   class_weight=class_weight,
                                   algorithm_params={'info': info})
    mod.fit(Xtrain, ytrain)
    y_pred = mod.predict(Xtest)
    proba_pred = mod.predict_probability(Xtest)  # a posteriori probability
    beta = mod.beta
    ret = dict(y_pred=y_pred, proba_pred=proba_pred, y_true=ytest,
               X_calib=Xcalib, y_calib=ycalib, X_valid=Xvalid, y_test=yvalid,
               n_fold=n_fold, beta=beta,  mask=mask,
               n_iter=mod.get_info()['num_iter'])
    if output_collector:
        output_collector.collect(key, ret)
    else:
        return ret


def reducer(key, values):
    import mapreduce as GLOBAL
    # key : string of intermediary key
    # load return dict corresponding to mapper ouput. they need to be loaded.
    # Compute sd; ie.: compute results on each folds
    roi = GLOBAL.ROI
    criteria = {'recall_mean': [np.argmax, np.max],
                'min_recall': [np.argmax, np.max],
                'accuracy': [np.argmax, np.max]}
    output_selection = GLOBAL.OUTPUT_SELECTION
    output_summary = GLOBAL.OUTPUT_SUMMARY
    map_output = GLOBAL.MAP_OUTPUT
    BASE = os.path.join("/neurospin/brainomics/2014_deptms/results_enettv/",
                        "MRI_" + roi,
                        map_output)
    INPUT = BASE + "/%i/%s"
    penalty_start = GLOBAL.PENALTY_START
    prob_class1 = GLOBAL.PROB_CLASS1
    params = GLOBAL.PARAMS
    # load all keys (sets of parameters)
    keys = ['_'.join(str(e) for e in a) for a in params]
    compt = 0
    if not os.path.isfile(output_selection):
        print "Model Construction, first cross-validation"
        # loop for the selection of the model
        for fold in xrange(0, NFOLDS + 1):  # outer folds
            # inner folds (NFOLDS) associated to the outer fold
            idx_block = range(fold * (NFOLDS + 1),
                              (fold + 1) * (NFOLDS + 1) - 1)
            for key in keys:
                # paths of the map results of all inner folds associated to
                # a key and an outer fold
                paths_dCV = [INPUT % (idx, key) for idx in idx_block]
                scores_CV = OrderedDict()
                # get values
                values = [GLOBAL.OutputCollector(p) for p in paths_dCV]
                values = [item.load() for item in values]
                n_fold = [item["n_fold"] for item in values]
                assert n_fold == ([fold for i in xrange(NFOLDS)])
                recall_mean_std = np.std([np.mean(
                    precision_recall_fscore_support(
                    item["y_true"].ravel(), item["y_pred"])[1]) \
                    for item in values]) \
                    / np.sqrt(len(values))
                recall = [precision_recall_fscore_support(
                                item["y_true"].ravel(), item["y_pred"].ravel(),
                                average=None)[1] for item in values]
                support = [precision_recall_fscore_support(
                                item["y_true"].ravel(), item["y_pred"].ravel(),
                                average=None)[3] for item in values]
                accuracy_std = np.std([((recall[i][0] * support[i][0] + \
                         recall[i][1] * support[i][1]) \
                                 / (float(support[i][0] + support[i][1]))) \
                                 for i in xrange(len(values))]) \
                         / np.sqrt(len(values))
                y_true = [item["y_true"].ravel() for item in values]
                y_true = np.hstack(y_true)
                y_pred = [item["y_pred"].ravel() for item in values]
                y_pred = np.hstack(y_pred)
                prob_pred = [item["proba_pred"].ravel() for item in values]
                prob_pred = np.hstack(prob_pred)
                p, r, f, s = precision_recall_fscore_support(y_true, y_pred,
                                                         average=None)
                auc = roc_auc_score(y_true, prob_pred)
                betas = [item["beta"][penalty_start:]  for item in values]
                betas = np.hstack(betas).T
                n_ite = np.mean(np.array([item["n_iter"] for item in values]))
                R = np.corrcoef(betas)
                beta_cor_mean = np.mean(R[np.triu_indices_from(R, 1)])
                success = r * s
                success = success.astype('int')
                accuracy = (r[0] * s[0] + r[1] * s[1])
                accuracy = accuracy.astype('int')
                pvalue_class0 = binom_test(success[0], s[0], 1 - prob_class1)
                pvalue_class1 = binom_test(success[1], s[1], prob_class1)
                pvalue_accuracy = binom_test(accuracy, s[0] + s[1], p=0.5)
                k = key.split('_')
                a, l1 = float(k[0]), float(k[1])
                l2, tv = float(k[2]), float(k[3])
                left = float(1 - tv)
                if left == 0:
                    left = 1.
                scores_CV['n_fold'] = n_fold[0]
                scores_CV['parameters'] = key
                scores_CV['a'] = a
                scores_CV['l1'] = l1
                scores_CV['l2'] = l2
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
                scores_CV['accuracy'] = accuracy / float(s[0] + s[1])
                scores_CV['pvalue_accuracy'] = pvalue_accuracy
                scores_CV['accuracy_std'] = accuracy_std
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
                scores_CV['prop_non_zeros_mean'] = float(np.count_nonzero(betas)) \
                                                / float(np.prod(betas.shape))
                # stock results in dataframe scores_tab
                if compt == 0:
                    scores_tab = pd.DataFrame(columns=scores_CV.keys())
                scores_tab.loc[compt, ] = scores_CV.values()
                compt += 1
        print "save results of the inner cross-validation : ", output_selection
        scores_tab.to_csv(output_selection, index=False)

    if not os.path.isfile(output_summary):
        print "Model Selection"
        scores_tab = pd.read_csv(output_selection)
        fold_groups = scores_tab.groupby('n_fold')
        compt = 0
        for fold_val, fold_group in fold_groups:
            scores_dCV = OrderedDict()
            scores_dCV['n_fold'] = fold_val
            # for each outer fold and ecah criterion, select the set of
            # parameters that optimizes the criterion
            for item, val in criteria.items():
                scores_dCV['criteria_' + item] = item
                loc_opt = val[0](fold_group[item])
                value_opt = val[1](fold_group[item])
                scores_dCV['value_opt_' + item] = value_opt
                param_opt = fold_group.parameters[loc_opt]
                a_opt = fold_group.a[loc_opt]
                l1_opt = fold_group.l1[loc_opt]
                tv_opt = fold_group.tv[loc_opt]
                scores_dCV['param_opt_' + item] = param_opt
                scores_dCV['a_opt_' + item] = a_opt
                scores_dCV['l1_opt_' + item] = l1_opt
                scores_dCV['tv_opt_' + item] = tv_opt
            # stock results in dataframe scores_select_model
            if compt == 0:
                scores_select_model = pd.DataFrame(columns=scores_dCV.keys())
            scores_select_model.loc[compt, ] = scores_dCV.values()
            compt += 1
        print "save results of the model selection : ", output_summary
        scores_select_model.to_csv(output_summary, index=False)
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

    #########################################################################
    ## Read ROIs csv
    atlas = []
    dict_rois = {}
    df_rois = pd.read_csv(INPUT_ROIS_CSV)
    for i, ROI_name_aal in enumerate(df_rois["ROI_name_aal"]):
        cur = df_rois[df_rois.ROI_name_aal == ROI_name_aal]
        label_ho = cur["label_ho"].values[0]
        atlas_ho = cur["atlas_ho"].values[0]
        roi_name = cur["ROI_name_deptms"].values[0]
        if ((not cur.isnull()["atlas_ho"].values[0])
            and (not cur.isnull()["ROI_name_deptms"].values[0])):
            if ((not roi_name in dict_rois)
              and (roi_name != "Maskdep-sub")
              and (roi_name != "Maskdep-cort")):
                labels = np.asarray(label_ho.split(), dtype="int")
                dict_rois[roi_name] = [labels]
                dict_rois[roi_name].append(atlas_ho)

    rois = list(set(df_rois["ROI_name_deptms"].values.tolist()))
    rois = [x for x in rois if str(x) != 'nan']
    rois.append("brain")  # add whole brain to rois

    #########################################################################
    ## Build config file for each ROI
    modality = "MRI"
    print "Modality: ", modality
    DATA_MODALITY_PATH = os.path.join(DATASET_PATH, modality)

    for roi in rois:
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

        #################################################################
        ## Create config file
        y = np.load(os.path.join(WD, 'y.npy'))
        prob_class1 = np.count_nonzero(y) / float(len(y))

        # resampling indexes for the double cross validation
        SEED_CV1 = 23071991
        SEED_CV2 = 5061931
        dcv = []
        n_fold_outer = 0

        '''
            Double cross-validation:
            X = [X_valid, X_calib] = [X_valid, X_test, X_train]
            y = [y_valid, y_calib] = [y_valid, y_test, y_train]

            dcv = [n_fold_outer, valid_index, calib_index,
                   test_index, train_index]
            # outer loop (2nd cross validation): model validation
            X = X_valid + X_calib
            y = y_valid + y_calib
            # inner loop (1st cross validation): model selection
            X_calib = X_test + X_train
            y_calib = y_test + y_train
        '''
        for i in xrange(2):
            # if n_fold_outer == 0: y_calib = y and y_test = y
            # splitted into NFOLDS
            if n_fold_outer == 0:
                y_calib = y
                y_valid = y
                calib = np.array([i for i in xrange(len(y))])
                valid = np.array([i for i in xrange(len(y))])
                for train, test in StratifiedKFold(y_calib.ravel(),
                                                   n_folds=NFOLDS,
                                                   shuffle=True,
                                                   random_state=SEED_CV2):
                    assert((len(train) + len(test)) == len(calib))
                    dcv.append([n_fold_outer, valid.tolist(), calib.tolist(),
                            test.tolist(), train.tolist()])
                # y_val and y_tr = y_calib
                dcv.append([n_fold_outer, valid.tolist(), calib.tolist(),
                            None])
                n_fold_outer += 1
            else:
                for calib, valid in StratifiedKFold(y.ravel(), n_folds=NFOLDS,
                                         shuffle=True, random_state=SEED_CV1):
                    assert((len(calib) + len(valid)) == len(y))
                    y_calib = np.array([y[i] for i in calib.tolist()])
                    for train, test in StratifiedKFold(y_calib.ravel(),
                                                   n_folds=NFOLDS,
                                                   shuffle=True,
                                                   random_state=SEED_CV2):
                        assert((len(train) + len(test)) == len(calib))
                        dcv.append([n_fold_outer, valid.tolist(),
                                    calib.tolist(),
                                    test.tolist(), train.tolist()])
                    # if n_fold_inner == 0: y_val and y_tr = y_calib
                    dcv.append([n_fold_outer, valid.tolist(), calib.tolist(),
                                None])
                    n_fold_outer += 1

        INPUT_DATA_X = 'X.npy'
        INPUT_DATA_y = 'y.npy'
        INPUT_MASK = 'mask.nii'

        # parameters grid
        # Re-run with
        # alpha = [0.5, 0.05]
        # tv = 0.3
        # l1l2_ratio = [0.1, 0.9]
        # alphal1l2tv = [[0.05, 0.07, 0.63, 0.3], [0.05, 0.63, 0.07, 0.3],
        #                [0.5, 0.07, 0.63, 0.3], [0.5, 0.63, 0.07, 0.3] ]
        tv = 0.3
        ratios = np.array([[0.1, 0.9, 1], [0.9, 0.1, 1]])
        alphas = [.05, .5]
        k_ratio = 1
        l1l2tv = [np.array([[float(1 - tv),
                             float(1 - tv),
                             tv]]) * ratios]
        l1l2tv = np.concatenate(l1l2tv)
        alphal1l2tv = np.concatenate([np.c_[np.array([[alpha]] * \
                                                       l1l2tv.shape[0]),
                                            l1l2tv] for alpha in alphas])
        alphal1l2tvk = np.concatenate([np.c_[alphal1l2tv,
                                             np.array([[k_ratio]] * \
                                                     alphal1l2tv.shape[0])]])
        params = [params.tolist() for params in alphal1l2tvk]
        user_func_filename = os.path.abspath(__file__)
        print "user_func", user_func_filename
        config = dict(data=dict(X=INPUT_DATA_X, y=INPUT_DATA_y),
                      params=params, resample=dcv,
                      structure=INPUT_MASK,
                      map_output="results_dCV_selection",
                      map_output_validation="results_dCV_validation",
                      user_func=user_func_filename,
                      reduce_group_by="resample_index",
                      output_selection="results_dCV_selection.csv",
                      output_summary="summary_selection.csv",
                      output_validation="results_dCV_validation.csv",
                      penalty_start=penalty_start,
                      prob_class1=prob_class1,
                      roi=roi)
                      #reduce_output="results_dCV.csv")
        json.dump(config, open(os.path.join(WD, "config_dCV_selection.json"),
                               "w"))

        #################################################################
        # Build utils files: sync (push/pull) and PBS
        import brainomics.cluster_gabriel as clust_utils
        sync_push_filename, sync_pull_filename, WD_CLUSTER = \
            clust_utils.gabriel_make_sync_data_files(WD)
        cmd = "mapreduce.py --map  %s/config_dCV_selection.json" % WD_CLUSTER
        clust_utils.gabriel_make_qsub_job_files(WD, cmd)
#        ################################################################
#        # Sync to cluster
#        print "Sync data to gabriel.intra.cea.fr: "
#        os.system(sync_push_filename)

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