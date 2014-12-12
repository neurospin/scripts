# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 14:30:20 2014

@author: cp243490
"""

import os
import json
import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import KFold
import nibabel
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectKBest
from parsimony.estimators import LogisticRegressionL1L2TV
from parsimony.algorithms.utils import Info
import parsimony.functions.nesterov.tv as tv_helper
import shutil
import matplotlib.pyplot as plt


from collections import OrderedDict


NFOLDS = 5
NRNDPERMS = 1000


def load_globals(config):
    import mapreduce as GLOBAL  # access to global variables
    GLOBAL.DATA = GLOBAL.load_data(config["data"])
    GLOBAL.PROB_CLASS1 = config["prob_class1"]
    GLOBAL.PENALTY_START = config["penalty_start"]
    STRUCTURE = nibabel.load(config["structure"])
    GLOBAL.A, _ = tv_helper.A_from_mask(STRUCTURE.get_data())
    GLOBAL.STRUCTURE = STRUCTURE


def resample(config, resample_nb):
    import mapreduce as GLOBAL  # access to global variables
    resample = config["resample"][resample_nb]
    if resample is not None:
        #GLOBAL.DATA_RESAMPLED = {k: [GLOBAL.DATA[k][idx, ...] for idx in resample]
        #                    for k in GLOBAL.DATA}
        # TODO permute first y then apply CV
        rnd_state = np.random.get_state()
        #yp = np.random.permutation(GLOBAL.DATA['y'])
        np.random.seed(resample_nb)
        GLOBAL.DATA_RESAMPLED = dict(
            X=[GLOBAL.DATA['X'][idx, ...]
                for idx in resample],
            y=[np.random.permutation(GLOBAL.DATA['y'][idx, ...])
                for idx in resample])
        np.random.set_state(rnd_state)
    else:  # resample is None train == test
        GLOBAL.DATA_RESAMPLED = {k: [GLOBAL.DATA[k] for idx in [0, 1]]
                            for k in GLOBAL.DATA}


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
    n_voxels = np.count_nonzero(STRUCTURE.get_data())
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
               beta=beta,  mask=mask,
               n_iter=mod.get_info()['num_iter'])
    if output_collector:
        output_collector.collect(key, ret)
    else:
        return ret


def reducer(key, values):
    import glob
    import mapreduce
    # key : string of intermediary key
    # load return dict correspondning to mapper ouput. they need to be loaded.
    # DEBUG
    BASE = "/neurospin/brainomics/2014_deptms/maskdep/results_enettv/results_rndperm"
    INPUT = BASE + "/%i/%s"
    OUTPUT = BASE + "/../results/rndperm"
    penalty_start = GLOBAL.PENALTY_START
    keys = ["0.05_1.0_0.0_0.0_-1.0",  "0.05_0.0_0.0_1.0_-1.0",
            "0.05_0.3_0.0_0.7_-1.0",  "0.05_0.7_0.0_0.3_-1.0"]
    for key in keys:
        paths_5cv_all = [INPUT % (perm, key)for perm in xrange(NFOLDS * NRNDPERMS)]
        idx_5cv_blocks = range(0, (NFOLDS * NRNDPERMS) + NFOLDS, NFOLDS)
        cpt = 0
        qc = dict()
        recall_mean_std_perms = np.zeros(NRNDPERMS)
        r_perms = np.zeros(NRNDPERMS)
        for perm_i in xrange(len(idx_5cv_blocks) - 1):
            paths_5cv = paths_5cv_all[idx_5cv_blocks[perm_i]:idx_5cv_blocks[perm_i+1]]
            for p in paths_5cv:
                if os.path.exists(p) and not(p in qc):
                    if p in qc:
                        qc[p] += 1
                    else:
                        qc[p] = 1
                    cpt += 1
            #
            values = [mapreduce.OutputCollector(p) for p in paths_5cv]
            values = [item.load() for item in values]
            recall_mean_std = np.std([np.mean(precision_recall_fscore_support(
                item["y_true"].ravel(), item["y_pred"])[1]) \
                for item in values]) / np.sqrt(len(values))
            y_true = [item["y_true"].ravel() for item in values]
            y_pred = [item["y_pred"].ravel() for item in values]
            y_true = np.concatenate(y_true)
            y_pred = np.concatenate(y_pred)
            p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average=None)
            betas = np.hstack([item["beta"][penalty_start:]  for item in values]).T
            R = np.corrcoef(betas)
            beta_cor_mean = np.mean(R[np.triu_indices_from(R, 1)])
            #
            r_perms[perm_i] = r
        # END PERMS
        print "save", key
        np.savez_compressed(OUTPUT+"/perms_"+key+".npz",
                            recall_0=r_perms[0], recall_1=r_perms[1])
        #
        perms = dict()
        fig, axis = plt.subplots(len(keys), 4)  #, sharex='col')
        for i, key in enumerate(keys):
            perms[key] = np.load(OUTPUT + "/perms_" + key + ".npz")
            n, bins, patches = axis[i, 0].hist(perms[key]['recall_0'], 50, normed=1, histtype='stepfilled')
            axis[i, 0].set_title(key + "_recall_0")
            n, bins, patches = axis[i, 1].hist(perms[key]['recall_1'], 50, normed=1, histtype='stepfilled')
            axis[i, 1].set_title(key + "_recall_1")
        plt.show()
        l1l2tv, l1tv, l1l2, l1 = ["0.001_0.3335_0.3335_0.333_-1",  "0.001_0.5_0_0.5_-1",  
                             "0.001_0.5_0.5_0_-1",  "0.001_1_0_0_-1"]
        #l1, tv, l12tv, 2l1tv = ["0.05_1.0_0.0_0.0_-1.0",  "0.05_0.0_0.0_1.0_-1.0", "0.05_0.3_0.0_0.7_-1.0",  "0.05_0.7_0.0_0.3_-1.0"]
        l1, tv, l1tvtv, l1l1tv  = ["0.05_1.0_0.0_0.0_-1.0",  "0.05_0.0_0.0_1.0_-1.0", "0.05_0.3_0.0_0.7_-1.0",  "0.05_0.7_0.0_0.3_-1.0"]

        # Read true scores
        import pandas as pd
        true = pd.read_csv(os.path.join(BASE, "..", "results.csv"))
        true = true[true.a == 0.05]
        true_2l1tv = true[(true.l1 == 0.7) & (true.tv == 0.3)].iloc[0]
        true_l12tv = true[(true.l1 == 0.3) & (true.l2 == 0.7)].iloc[0]
        true_l1 = true[(true.l1 == 1)].iloc[0]
        true_tv = true[(true.tv == 1.)].iloc[0]
    
        # pvals
        nperms = float(len(perms[l1]['r2']))
        from collections import OrderedDict
        pvals = OrderedDict()
        pvals["cond"] = ['l1', 'tv', 'l12tv', '2l1tv'] * 2 + \
                ['l1 vs l1l1tv'] * 2  + ['tv vs l1tvtv'] * 2
        pvals["stat"] = ['recall_0'] * 4 + ['recall_1'] * 4 + \
                ['recall_0', 'recall_1'] * 2
        pvals["pval"] = [
            np.sum(perms[l1]['recall_0'] > true_l1["recall_0"]),
            np.sum(perms[tv]['recall_0'] > true_tv["recall_0"]),
            np.sum(perms[l1l1tv]['recall_0'] > true_2l1tv["recall_0"]),
            np.sum(perms[l1tvtv]['recall_0'] > true_l12tv["recall_0"]),

            np.sum(perms[l1]['recall_1'] > true_l1["recall_1"]),
            np.sum(perms[tv]['recall_1'] > true_tv["recall_1"]),
            np.sum(perms[l1l1tv]['recall_1'] > true_2l1tv["recall_1"]),
            np.sum(perms[l1tvtv]['recall_1'] > true_l12tv["recall_1"]),

            # l1 vs 2l1tv
            np.sum((perms[l1l1tv]['recall_0'] - perms[l1]['recall_0']) > (true_2l1tv["recall_0"] - true_l1["recall_0"])),
            np.sum((perms[l1l1tv]['recall_1'] - perms[l1]['recall_1']) > (true_2l1tv["recall_1"] - true_l1["recall_1"])),

            # tv vs l12tv
            np.sum((perms[l1tvtv]['recall_0'] - perms[tv]['recall_0']) > (true_l12tv["recall_0"] - true_tv["recall_0"])),
            np.sum((perms[l1tvtv]['recall_1'] - perms[tv]['recall_1']) > (true_l12tv["recall_1"] - true_tv["recall_1"]))
                        ]

        pvals = pd.DataFrame(pvals)
        pvals["pval"] /= nperms
        pvals.to_csv(os.path.join(OUTPUT, "pvals_stats_permutations.csv"),
                     index=False)

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
    ## Build config file for maskdep, MRI images
    images = "MRI"
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

    if os.path.exists("config_rndperm.json"):
        inf = open("config_rndperm.json", "r")
        old_conf = json.load(inf)
        rndperm = old_conf["resample"]
        inf.close()
    else:
        rndperm = [[tr.tolist(), te.tolist()] for perm in xrange(NRNDPERMS)
            for tr, te in KFold(n=len(y), n_folds=NFOLDS, random_state=0)]

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
    user_func_filename = os.path.abspath(__file__)
    print "user_func", user_func_filename
    config = dict(data=dict(X=INPUT_DATA_X, y=INPUT_DATA_y),
                  params=params, resample=rndperm,
                  structure=INPUT_MASK,
                  map_output="results_rndperm",
                  user_func=user_func_filename,
                  reduce_group_by="params",
                  reduce_output="mask_deprndperm.csv",
                  penalty_start=penalty_start)
    json.dump(config, open(os.path.join(WD, "config_rndperm.json"), "w"))

    #################################################################
    # Build utils files: sync (push/pull) and PBS
    import brainomics.cluster_gabriel as clust_utils
    sync_push_filename, sync_pull_filename, WD_CLUSTER = \
        clust_utils.gabriel_make_sync_data_files(WD)
    cmd = "mapreduce.py --map  %s/config_rndperm.json" % WD_CLUSTER
    clust_utils.gabriel_make_qsub_job_files(WD, cmd)
    ################################################################
#            # Sync to cluster
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