# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 17:40:05 2015

@author: cp243490
"""

import os
import json
from collections import OrderedDict
import numpy as np
import pandas as pd
import nibabel
from sklearn.metrics import precision_recall_fscore_support
from parsimony.estimators import LogisticRegressionL1L2TV
from parsimony.algorithms.utils import Info
import parsimony.functions.nesterov.tv as tv_helper
from scipy.stats import binom_test
import shutil
from sklearn.cross_validation import StratifiedKFold


NFOLDS = 10
NRNDPERMS = 1000


def load_globals(config):
    import mapreduce as GLOBAL  # access to global variables
    GLOBAL.DATA = GLOBAL.load_data(config["data"])
    GLOBAL.PARAMS = config["params"]
    GLOBAL.MAP_OUTPUT = config['map_output']
    GLOBAL.OUTPUT_SELECTION = config['output_selection']
    GLOBAL.OUTPUT_SUMMARY = config['output_summary']
    GLOBAL.PENALTY_START = config["penalty_start"]
    STRUCTURE = nibabel.load(config["structure"])
    GLOBAL.A, _ = tv_helper.A_from_mask(STRUCTURE.get_data())
    GLOBAL.STRUCTURE = STRUCTURE
    GLOBAL.ROI = config["roi"]
    GLOBAL.OUTPUT_PATH = config["output_path"]


def resample(config, resample_nb):
    import mapreduce as GLOBAL  # access to global variables
    #GLOBAL.DATA = GLOBAL.load_data(config["data"])
    resample = config["resample"][resample_nb]
    # resample = [n_perm, n_fold, y_perm,
    #             valid_idx, calib_idx,
    #             test_idx, train_idx]
    # where test + train = calib
    GLOBAL.NRNDPERM = resample[0]
    GLOBAL.N_FOLD = resample[1]
    y_perm = resample[2]
    y_perm = np.asarray(y_perm)
    if resample is not None:
        # general case
        GLOBAL.DATA_RESAMPLED_VALIDMODEL = dict(
            X=[GLOBAL.DATA['X'][idx, ...] for idx in resample[3:5]],
            y=[y_perm[idx, ...] for idx in resample[3:5]])
        GLOBAL.DATA_RESAMPLED_SELECTMODEL = {k: [
                        GLOBAL.DATA_RESAMPLED_VALIDMODEL[k][1][idx, ...]
                            for idx in resample[5:]]
                                for k in GLOBAL.DATA_RESAMPLED_VALIDMODEL}
    # resample is None train == test
    else:
        GLOBAL.DATA_RESAMPLED_VALIDMODEL = dict(
            X=[GLOBAL.DATA['X'] for idx in [0, 1]],
            y=[y_perm for idx in [0, 1]])
        GLOBAL.DATA_RESAMPLED_SELECTMODEL = dict(
            X=[GLOBAL.DATA['X'] for idx in [0, 1]],
            y=[y_perm for idx in [0, 1]])
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
    #Xvalid = GLOBAL.DATA_RESAMPLED_VALIDMODEL["X"][0]
    #Xcalib = GLOBAL.DATA_RESAMPLED_VALIDMODEL["X"][1]
    #yvalid = GLOBAL.DATA_RESAMPLED_VALIDMODEL["y"][0]
    #ycalib = GLOBAL.DATA_RESAMPLED_VALIDMODEL["y"][1]

    # data for model selection (1st cross validation, inner loop)
    Xtest = GLOBAL.DATA_RESAMPLED_SELECTMODEL["X"][0]
    Xtrain = GLOBAL.DATA_RESAMPLED_SELECTMODEL["X"][1]
    ytest = GLOBAL.DATA_RESAMPLED_SELECTMODEL["y"][0]
    ytrain = GLOBAL.DATA_RESAMPLED_SELECTMODEL["y"][1]

    print key, "Data shape:", Xtest.shape, Xtrain.shape
    #alpha, ratio_l1, ratio_l2, ratio_tv, k = key
    #key = np.array(key)
    penalty_start = GLOBAL.PENALTY_START
    class_weight = "auto"  # unbiased
    alpha = float(key[0])
    l1, l2 = alpha * float(key[1]), alpha * float(key[2])
    tv, k_ratio = alpha * float(key[3]), float(key[4])
    print "l1:%f, l2:%f, tv:%f, k_ratio:%i" % (l1, l2, tv, k_ratio)
    A = GLOBAL.A
    info = [Info.num_iter]
    mod = LogisticRegressionL1L2TV(l1, l2, tv, A, penalty_start=penalty_start,
                                   class_weight=class_weight,
                                   algorithm_params={'info': info})
    mod.fit(Xtrain, ytrain)
    y_pred = mod.predict(Xtest)
    proba_pred = mod.predict_probability(Xtest)  # a posteriori probability
    ret = dict(y_pred=y_pred, proba_pred=proba_pred, y_true=ytest,
               nfold=nfold, nrndperm=nrndperm,
               n_iter=mod.get_info()['num_iter'])
    if output_collector:
        output_collector.collect(key, ret)
    else:
        return ret


def reducer(key, values):
    # key : string of intermediary key
    # load return dict correspondning to mapper ouput. they need to be loaded.
    # DEBUG
    import mapreduce as GLOBAL
    criteria = {'recall_mean': [np.argmax, np.max],
                'min_recall': [np.argmax, np.max],
                'accuracy': [np.argmax, np.max]}
    output_selection = GLOBAL.OUTPUT_SELECTION
    output_summary = GLOBAL.OUTPUT_SUMMARY
    output_path = GLOBAL.OUTPUT_PATH
    map_output = GLOBAL.MAP_OUTPUT
    roi = GLOBAL.ROI
    BASE = os.path.join("/neurospin/brainomics/2014_deptms/results_enettv/",
                        "MRI_" + roi,
                        map_output)
    INPUT = BASE + "/%i/%s"
    OUTPUT = BASE + "/../" + output_path
    if not os.path.exists(OUTPUT):
        os.makedirs(OUTPUT)
    params = GLOBAL.PARAMS
    keys = ['_'.join(str(e) for e in a) for a in params]
    compt = 0
    if not os.path.isfile(os.path.join(OUTPUT, output_selection)):
        print "Model Construction, first cross-validation"
        for key in keys:
            print "key: ", key
            paths_dCV_all = [INPUT % (perm, key) \
                    for perm in xrange(NFOLDS * NFOLDS * NRNDPERMS)]
            idx_dCV_blocks = range(0,
                             (NFOLDS * NFOLDS * NRNDPERMS) + NFOLDS * NFOLDS,
                             NFOLDS * NFOLDS)
            for perm in xrange(NRNDPERMS):
                print "perm: ", perm
                paths_dCV_blocks = paths_dCV_all[idx_dCV_blocks[perm]:\
                                                idx_dCV_blocks[perm + 1]]
                idx_fold_blocks = range(0, NFOLDS * NFOLDS + NFOLDS, NFOLDS)
                # for each outer fold
                for fold in xrange(0, NFOLDS):
                    path_fold_blocks = paths_dCV_blocks[idx_fold_blocks[fold]:\
                                                    idx_fold_blocks[fold + 1]]
                    scores_CV = OrderedDict()
                    values = [GLOBAL.OutputCollector(p) \
                                    for p in path_fold_blocks]
                    values = [item.load() for item in values]
                    n_fold = [item["nfold"] for item in values]
                    assert n_fold == ([fold for i in xrange(NFOLDS)])
                    y_true = [item["y_true"].ravel() for item in values]
                    y_true = np.hstack(y_true)
                    y_pred = [item["y_pred"].ravel() for item in values]
                    y_pred = np.hstack(y_pred)
                    prob_pred = [item["proba_pred"].ravel() for item in values]
                    prob_pred = np.hstack(prob_pred)
                    p, r, f, s = precision_recall_fscore_support(y_true,
                                                                 y_pred,
                                                                 average=None)
                    accuracy = (r[0] * s[0] + r[1] * s[1])
                    accuracy = accuracy.astype('int')
                    k = key.split('_')
                    a, l1 = float(k[0]), float(k[1])
                    l2, tv = float(k[2]), float(k[3])
                    left = float(1 - tv)
                    if left == 0:
                        left = 1.
                    scores_CV['permutation'] = perm
                    scores_CV['n_fold'] = n_fold[0]
                    scores_CV['parameters'] = key
                    scores_CV['recall_0'] = r[0]
                    scores_CV['recall_1'] = r[1]
                    scores_CV['min_recall'] = np.minimum(r[0], r[1])
                    scores_CV['recall_mean'] = r.mean()
                    scores_CV['accuracy'] = accuracy / float(s[0] + s[1])
                    if compt == 0:
                        scores_tab = pd.DataFrame(columns=scores_CV.keys())
                    scores_tab.loc[compt, ] = scores_CV.values()
                    compt += 1
        scores_tab.to_csv(os.path.join(OUTPUT, output_selection), index=False)

    if not os.path.isfile(os.path.join(OUTPUT, output_summary)):
        print "Model Selection"
        compt = 0
        scores_tab = pd.read_csv(os.path.join(OUTPUT, output_selection))
        perm_groups = scores_tab.groupby('permutation')
        for perm_val, perm_group in perm_groups:
            fold_groups = perm_group.groupby('n_fold')
            for fold_val, fold_group in fold_groups:
                scores_dCV = OrderedDict()
                scores_dCV['permutation'] = perm_val
                scores_dCV['n_fold'] = fold_val
                n_crit = 0
                for item, val in criteria.items():
                    n_crit += 1
                    scores_dCV['criteria_' + item] = item
                    loc_opt = val[0](fold_group[item])
                    value_opt = val[1](fold_group[item])
                    scores_dCV['value_opt_' + item] = value_opt
                    param_opt = fold_group.parameters[loc_opt]
                    scores_dCV['param_opt_' + item] = param_opt
                if compt == 0:
                    scores_select_model = pd.DataFrame(
                                                   columns=scores_dCV.keys())
                scores_select_model.loc[compt, ] = scores_dCV.values()
                compt += 1
        scores_select_model.to_csv(os.path.join(OUTPUT, output_summary),
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
#            if not roi_name in dict_rois:
#                labels = np.asarray(label_ho.split(), dtype="int")
#                dict_rois[roi_name] = [labels]
#                dict_rois[roi_name].append(atlas_ho)
#
#    rois = list(set(df_rois["ROI_name_deptms"].values.tolist()))
#    rois = [x for x in rois if str(x) != 'nan']
#    rois.append("brain")  # add whole brain to rois
#    #########################################################################
    ## Build config file
    modality = "MRI"
    print "Modality: ", modality
    DATA_MODALITY_PATH = os.path.join(DATASET_PATH, modality)

#    for roi in rois:
    roi = "Roiho-frontalPole"
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
    ## Create config file
    config_selection_file = os.path.join(WD, "config_dCV_selection.json")
    # open config file selection
    config_selection = json.load(open(config_selection_file))
    # get resample index for each fold
    params = config_selection["params"]

    if os.path.exists("config_rndperm_dCV_selection.json"):
        inf = open("config_rndperm_dCV_selection.json", "r")
        old_conf = json.load(inf)
        rndperm = old_conf["resample"]
        inf.close()
    else:
        y = np.load(os.path.join(WD, 'y.npy'))
        # permute y and resample indexes for the double cross validation
        # with random permutations
        SEED_CV1 = 23071991
        SEED_CV2 = 5061931
        rndperm = []
        ''' dcv = [n_perm, n_fold_outer, y_perm,
                   valid_index, calib_index,
                   test_index, train_index]
            # n_perm = 0:NRNDPERMS ; n_fold_outer =  0:NFOLDS
            # y_perm: permuted y
            # outer loop (2nd cross validation): model validation
            X = X_valid + X_calib
            y = y_valid + y_calib
            # inner loop (1st cross validation): model selection
            X_calib = X_test + X_train
            y_calib = y_test + y_train
        '''
        for perm in xrange(NRNDPERMS):
            n_fold_outer = 0
            rnd_state = np.random.get_state()
            np.random.seed(perm)
            y_perm = np.random.permutation(y)
            np.random.set_state(rnd_state)
            for calib, valid in StratifiedKFold(y_perm.ravel(),
                        n_folds=NFOLDS, shuffle=True,
                        random_state=SEED_CV1):
                assert((len(calib) + len(valid)) == len(y))
                y_perm_calib = np.array([y_perm[i] \
                                    for i in calib.tolist()])
                for train, test in StratifiedKFold(y_perm_calib.ravel(),
                        n_folds=NFOLDS, shuffle=True,
                        random_state=SEED_CV2):
                    assert((len(train) + len(test)) == len(calib))
                    rndperm.append([perm, n_fold_outer, y_perm.tolist(),
                                valid.tolist(), calib.tolist(),
                                test.tolist(), train.tolist()])
                n_fold_outer += 1
    INPUT_DATA_X = 'X.npy'
    INPUT_DATA_y = 'y.npy'
    INPUT_MASK = 'mask.nii'
    user_func_filename = os.path.abspath(__file__)
    print "user_func", user_func_filename
    # Use relative path from config.json
    config = dict(data=dict(X=INPUT_DATA_X, y=INPUT_DATA_y),
                  params=params, resample=rndperm,
                  structure=INPUT_MASK,
                  penalty_start=3,
                  map_output="rndperm_selection",
                  output_selection="results_rndperm_dCV_selection.csv",
                  output_path="rndperm_results",
                  output_summary="summary_rndperm_selection.csv",
                  output_validation="results_rndperm_dCV_validation.csv",
                  output_permutations="pvals_stats_permutations.csv",
                  user_func=user_func_filename,
                  reduce_group_by="params",
                  roi=roi)
    json.dump(config,
              open(os.path.join(WD, "config_rndperm_dCV_selection.json"),
                   "w"))

    #####################################################################
    # Build utils files: sync (push/pull) and PBS
    import brainomics.cluster_gabriel as clust_utils
    sync_push_filename, sync_pull_filename, WD_CLUSTER = \
        clust_utils.gabriel_make_sync_data_files(WD)
    cmd = "mapreduce.py --map  %s/config_rndperm_dCV_selection.json" % WD_CLUSTER
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
#    #############################################################################
#    print "# Reduce"
#    print "mapreduce.py --reduce %s/config_rndperm.json" % WD