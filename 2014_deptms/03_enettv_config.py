# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 15:00:01 2014

@author: cp243490
Create the config file for several datasets and contains the map and reduce
functions.
"""


import os
import json
import numpy as np
import pandas as pd
from sklearn.cross_validation import StratifiedKFold
import nibabel
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectKBest
from parsimony.estimators import LogisticRegressionL1L2TV
import parsimony.functions.nesterov.tv as tv_helper
import shutil
from scipy import sparse

from collections import OrderedDict


NFOLDS = 5


def load_globals(config):
    import mapreduce as GLOBAL  # access to global variables
    GLOBAL.DATA = GLOBAL.load_data(config["data"])
    MODALITY = config["modality"]
    penalty_start = config["penalty_start"]
    if np.logical_or(MODALITY == "MRI", MODALITY == "PET"):
        STRUCTURE = nibabel.load(config["structure"])
        A, _ = tv_helper.A_from_mask(STRUCTURE.get_data())

    elif MODALITY == "MRI+PET":
        STRUCTURE = nibabel.load(config["structure"])
        A1, _ = tv_helper.A_from_mask(STRUCTURE.get_data())
        # construct matrix A
        # Ax, Ay, Az are block diagonale matrices and diagonal elements
        # are elements of A1
        # eg: Ax = diagblock(A1x, A1x)
        A = []
        for i in range(3):
            a = sparse.bmat([[A1[i], None], [None, A1[i]]])
            A.append(a)

    GLOBAL.A, GLOBAL.STRUCTURE, GLOBAL.MODALITY = A, STRUCTURE, MODALITY
    GLOBAL.PENALTY_START = penalty_start


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
    MODALITY = GLOBAL.MODALITY
    Xtr = GLOBAL.DATA_RESAMPLED["X"][0]
    Xte = GLOBAL.DATA_RESAMPLED["X"][1]
    ytr = GLOBAL.DATA_RESAMPLED["y"][0]
    yte = GLOBAL.DATA_RESAMPLED["y"][1]
    print key, "Data shape:", Xtr.shape, Xte.shape, ytr.shape, yte.shape
    STRUCTURE = GLOBAL.STRUCTURE
    n_voxels = np.count_nonzero(STRUCTURE.get_data())
    #alpha, ratio_l1, ratio_l2, ratio_tv, k = key
    #key = np.array(key)
    penalty_start = GLOBAL.PENALTY_START
    class_weight = "auto"  # unbiased
    alpha = float(key[0])
    l1, l2 = alpha * float(key[1]), alpha * float(key[2])
    tv, k_ratio = alpha * float(key[3]), key[4]
    print "l1:%f, l2:%f, tv:%f, k_ratio:%f" % (l1, l2, tv, k_ratio)
    if np.logical_or(MODALITY == "MRI", MODALITY == "PET"):
        if k_ratio != -1:
            k = n_voxels * k_ratio
            k = int(k)
            aov = SelectKBest(k=k)
            aov.fit(Xtr[..., penalty_start:], ytr.ravel())
            mask = STRUCTURE.get_data() != 0
            mask[mask] = aov.get_support()
            #print mask.sum()
            A, _ = tv_helper.A_from_mask(mask)
            Xtr_r = np.hstack([Xtr[:, :penalty_start],
                               Xtr[:, penalty_start:][:, aov.get_support()]])
            Xte_r = np.hstack([Xte[:, :penalty_start],
                               Xte[:, penalty_start:][:, aov.get_support()]])
        else:
            mask = np.ones(Xtr.shape[0], dtype=bool)
            Xtr_r = Xtr
            Xte_r = Xte
            A = GLOBAL.A

    elif MODALITY == "MRI+PET":
        if k_ratio != -1:
            k = n_voxels * k_ratio
            k = int(k)
            aov_MRI = SelectKBest(k=k)
            aov_MRI.fit(Xtr[..., penalty_start:(penalty_start + n_voxels)],
                            ytr.ravel())
            mask = STRUCTURE.get_data() != 0
            mask[mask] = aov_MRI.get_support()
            A1, _ = tv_helper.A_from_mask(mask)

            aov_PET = SelectKBest(k=k)
            aov_PET.fit(Xtr[..., (penalty_start + n_voxels):],
                            ytr.ravel())
            mask = STRUCTURE.get_data() != 0
            mask[mask] = aov_PET.get_support()
            A2, _ = tv_helper.A_from_mask(mask)
            # construct matrix A
            # Ax, Ay, Az are block diagonale matrices and diagonal elements
            # are elements of A1
            # eg: Ax = diagblock(A1x, A1x)
            A = []
            for i in range(3):
                a = sparse.bmat([[A1[i], None], [None, A2[i]]])
                A.append(a)

            Xtr_r = np.hstack([Xtr[:, :penalty_start],
                               Xtr[:, penalty_start:(penalty_start + n_voxels)][:, aov_MRI.get_support()],
                               Xtr[:, (penalty_start + n_voxels):][:, aov_PET.get_support()]])
            Xte_r = np.hstack([Xte[:, :penalty_start],
                               Xte[:, penalty_start:(penalty_start + n_voxels)][:, aov_MRI.get_support()],
                               Xte[:, (penalty_start + n_voxels):][:, aov_PET.get_support()]])

        else:
            mask = np.ones(Xtr.shape[0], dtype=bool)
            Xtr_r = Xtr
            Xte_r = Xte
            A = GLOBAL.A

    mod = LogisticRegressionL1L2TV(l1, l2, tv, A, penalty_start=penalty_start,
                                   class_weight=class_weight)
    mod.fit(Xtr_r, ytr)
    y_pred = mod.predict(Xte_r)
    proba_pred = mod.predict_probability(Xte_r)  # a posteriori probability
    ret = dict(y_pred=y_pred, proba_pred=proba_pred, y_true=yte,
               beta=mod.beta,  mask=mask)
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
    prob_pred = [item["proba_pred"].ravel() for item in values]
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
    scores['a'] = key[0]
    scores['l1'] = key[1]
    scores['l2'] = key[2]
    scores['tv'] = key[3]
    scores['k_ratio'] = key[4]
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

    MODALITIES = ["MRI", "PET", "MRI+PET"]

    DATASET_PATH = os.path.join(BASE_PATH,    "datasets")
    BASE_DATA_PATH = os.path.join(BASE_PATH,    "base_data")

    INPUT_ROIS_CSV = os.path.join(BASE_DATA_PATH,  "ROI_labels.csv")

    OUTPUT_ENETTV = os.path.join(BASE_PATH,   "results_enettv")

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
        if ((not cur.isnull()["label_ho"].values[0])
            and (not cur.isnull()["ROI_name_deptms"].values[0])):
            if not roi_name in dict_rois:
                labels = np.asarray(label_ho.split(), dtype="int")
                dict_rois[roi_name] = [labels]
                dict_rois[roi_name].append(atlas_ho)

    rois = list(set(df_rois["ROI_name_deptms"].values.tolist()))
    rois = [x for x in rois if str(x) != 'nan']
    rois.append("wb")  # add whole brain to rois

    #########################################################################
    ## Build config file for all couple (Modality, roi)

    for modality in MODALITIES:
        print "Modality: ", modality
        DATA_MODALITY_PATH = os.path.join(DATASET_PATH, modality)

        for roi in rois:
            print "ROI", roi

            WD = os.path.join(OUTPUT_ENETTV, modality + '_' + roi)

            if not os.path.exists(WD):
                os.makedirs(WD)

            INPUT_DATA_X = os.path.join(DATA_MODALITY_PATH,
                                        'X_' + modality + '_' + roi + '.npy')
            INPUT_DATA_y = os.path.join(DATA_MODALITY_PATH,
                                        'y.npy')
            INPUT_MASK = os.path.join(DATA_MODALITY_PATH,
                                      'mask_' + modality + '_' + roi + '.nii')
            # copy X, y, mask file names in the current directory
            shutil.copy2(INPUT_DATA_X, WD)
            shutil.copy2(INPUT_DATA_y, WD)
            if np.logical_or(modality == "MRI", modality == "PET"):
                shutil.copy2(INPUT_MASK, WD)
            elif modality == "MRI+PET":
                shutil.copy2(os.path.join(DATASET_PATH, "MRI",
                                      'mask_MRI_' + roi + '.nii'),
                             WD)
                INPUT_MASK = os.path.join(WD, 'mask_MRI_' + roi + '.nii')
            #################################################################
            ## Create config file
            y = np.load(INPUT_DATA_y)
            cv = [[tr.tolist(), te.tolist()]
                    for tr, te in StratifiedKFold(y.ravel(), n_folds=NFOLDS)]
            cv.insert(0, None)  # first fold is None

            INPUT_DATA_X = os.path.basename(INPUT_DATA_X)
            INPUT_DATA_y = os.path.basename(INPUT_DATA_y)
            INPUT_MASK = os.path.basename(INPUT_MASK)
            # parameters grid
            # Re-run with
            tv_range = np.hstack([np.arange(0, 1., .1),
                                  [0.05, 0.01, 0.005, 0.001]])
            ratios = np.array([[1., 0., 1], [0., 1., 1], [.5, .5, 1],
                               [.9, .1, 1], [.1, .9, 1], [.01, .99, 1],
                               [.001, .999, 1]])
            alphas = [.01, .05, .1, .5, 1.]
            k_range_ratio = [0.1 / 100., 1 / 100., 10 / 100., 50 / 100., -1]
            l1l2tv = [np.array([[float(1 - tv),
                                 float(1 - tv),
                                 tv]]) * ratios for tv in tv_range]
            l1l2tv.append(np.array([[0., 0., 1.]]))
            l1l2tv = np.concatenate(l1l2tv)
            alphal1l2tv = np.concatenate([np.c_[np.array([[alpha]] * l1l2tv.shape[0]), l1l2tv] for alpha in alphas])
            alphal1l2tvk = np.concatenate([np.c_[alphal1l2tv, np.array([[k_ratio]] * alphal1l2tv.shape[0])] for k_ratio in k_range_ratio])
            params = [params.tolist() for params in alphal1l2tvk]
            user_func_filename = os.path.join(os.environ["HOME"], "git",
                "scripts", "2014_deptms", "03_enettv_config.py")
            print "user_func", user_func_filename
            config = dict(data=dict(X=INPUT_DATA_X, y=INPUT_DATA_y),
                          params=params, resample=cv,
                          structure=INPUT_MASK,
                          map_output="results",
                          user_func=user_func_filename,
                          reduce_group_by="params",
                          reduce_output="results.csv",
                          penalty_start=penalty_start,
                          modality=modality,
                          roi=roi)
            json.dump(config, open(os.path.join(WD, "config.json"), "w"))

            #################################################################
#            # Build utils files: sync (lasso regressionpush/pull) and PBS
#            import brainomics.cluster_gabriel as clust_utils
#            sync_push_filename, sync_pull_filename, WD_CLUSTER = \
#                clust_utils.gabriel_make_sync_data_files(WD)
#            cmd = "mapreduce.py --map  %s/config.json" % WD_CLUSTER
#            clust_utils.gabriel_make_qsub_job_files(WD, cmd)
            #################################################################
            # Sync to cluster
#            print "Sync data to gabriel.intra.cea.fr: "
#            os.system(sync_push_filename)

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