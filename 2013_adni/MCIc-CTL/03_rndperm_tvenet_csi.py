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
from brainomics import array_utils
from statsmodels.stats.inter_rater import fleiss_kappa

NFOLDS = 5
NRNDPERMS = 1000


def load_globals(config):
    import mapreduce as GLOBAL  # access to global variables
    GLOBAL.DATA = GLOBAL.load_data(config["data"])
    STRUCTURE = nibabel.load(config["mask_filename"])
    A = tv_helper.linear_operator_from_mask(STRUCTURE.get_data())
    GLOBAL.A, GLOBAL.STRUCTURE, GLOBAL.CONFIG = A, STRUCTURE, config


def resample(config, resample_nb):
    import mapreduce as GLOBAL  # access to global variables
    #GLOBAL.DATA = GLOBAL.load_data(config["data"])
    resample = config["resample"][resample_nb]
    if resample is not None:
        rnd_state = np.random.get_state()
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
        A  = tv_helper.linear_operator_from_mask(mask)
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


def reducer_manually():#key, values):
    """ Run this script manually in ipython
    """
    config_filenane = "/neurospin/brainomics/2013_adni/MCIc-CTL_csi/config.json"        
    # key : string of intermediary key
    # load return dict correspondning to mapper ouput. they need to be loaded.
    # DEBUG
    import glob, mapreduce
    BASE = os.path.join(os.path.dirname(config_filenane), "rndperm")
    INPUT = BASE + "/%i/%s"
    OUTPUT = os.path.join(os.path.dirname(config_filenane), "rndperm_results")
    keys = ["0.01_0.0_0.0_1.0_-1.0", 
            "0.01_0.0_0.5_0.5_-1.0", 
            "0.01_0.0_1.0_0.0_-1.0", 
            "0.01_0.35_0.35_0.3_-1.0",  
            "0.01_0.5_0.0_0.5_-1.0",  
            "0.01_0.5_0.5_0.0_-1.0",  
            "0.01_1.0_0.0_0.0_-1.0"]
    tv_k, l2tv_k, l2_k, l1l2tv_k, l1tv_k, l1l2_k, l1_k = keys
    tv_v, l2tv_v, l2_v, l1l2tv_v, l1tv_v, l1l2_v, l1_v = [[float(p) for p in key.split("_")] for key in keys]
    for key in keys:
        #key = keys[0]
        paths_5cv_all = [INPUT % (perm, key) for perm in xrange(NFOLDS * NRNDPERMS)]
        idx_5cv_blocks = range(0, (NFOLDS * NRNDPERMS) + NFOLDS, NFOLDS)
        cpt = 0
        qc = dict()
        auc_perms = np.full(NRNDPERMS, np.nan)
        recall_0_perms = np.full(NRNDPERMS, np.nan)
        recall_1_perms = np.full(NRNDPERMS, np.nan)
        recall_mean_perms = np.full(NRNDPERMS, np.nan)
        r_bar_perms = np.full(NRNDPERMS, np.nan)
        fleiss_kappa_stat_perms = np.full(NRNDPERMS, np.nan)
        dice_bar_perms = np.full(NRNDPERMS, np.nan)
        for perm_i in xrange(len(idx_5cv_blocks)-1):
            print perm_i
            #perm_i = 0
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
            try:
                values = [item.load() for item in values]
            except:
                print "Failed loading perm: %i" % perm_i
                continue
            y_true = [item["y_true"].ravel() for item in values]
            y_pred = [item["y_pred"].ravel() for item in values]
            prob_pred = [item["proba_pred"].ravel() for item in values]
            y_true = np.concatenate(y_true)
            y_pred = np.concatenate(y_pred)
            prob_pred = np.concatenate(prob_pred)
            p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average=None)
            auc = roc_auc_score(y_true, prob_pred) #area under curve score.
            #r2 = r2_score(y_true, y_pred)
            #corr = np.corrcoef(y_true.ravel(), y_pred.ravel())[0, 1]
            betas = np.hstack([item["beta"] for item in values]).T
            #
            ## Compute beta similarity measures
            #
            # Correlation
            R = np.corrcoef(betas)
            R = R[np.triu_indices_from(R, 1)]
            # Fisher z-transformation / average
            z_bar = np.mean(1. / 2. * np.log((1 + R) / (1 - R)))
            # bracktransform
            r_bar = (np.exp(2 * z_bar) - 1) /  (np.exp(2 * z_bar) + 1)
            #
            # threshold betas to compute fleiss_kappa and DICE
            try:
                betas_t = np.vstack([array_utils.arr_threshold_from_norm2_ratio(betas[i, :], .99)[0] for i in xrange(betas.shape[0])])
                #print "--", np.sqrt(np.sum(betas_t ** 2, 1)) / np.sqrt(np.sum(betas ** 2, 1))
                #print np.allclose(np.sqrt(np.sum(betas_t ** 2, 1)) / np.sqrt(np.sum(betas ** 2, 1)), [0.99]*5,
                #                   rtol=0, atol=1e-02)
                #
                # Compute fleiss kappa statistics
                beta_signed = np.sign(betas_t)
                table = np.zeros((beta_signed.shape[1], 3))
                table[:, 0] = np.sum(beta_signed == 0, 0)
                table[:, 1] = np.sum(beta_signed == 1, 0)
                table[:, 2] = np.sum(beta_signed == -1, 0)
                fleiss_kappa_stat = fleiss_kappa(table)
                #
                # Paire-wise Dice coeficient
                beta_n0 = betas_t != 0
                ij = [[i, j] for i in xrange(5) for j in xrange(i+1, 5)]
                #print [[idx[0], idx[1]] for idx in ij]
                dice_bar = np.mean([float(np.sum(beta_signed[idx[0], :] == beta_signed[idx[1], :])) /\
                     (np.sum(beta_n0[idx[0], :]) + np.sum(beta_n0[idx[1], :]))
                     for idx in ij])
            except:
                dice_bar = fleiss_kappa_stat = 0.
            #
            auc_perms[perm_i] = auc
            recall_0_perms[perm_i] = r[0]
            recall_1_perms[perm_i] = r[1]
            recall_mean_perms[perm_i] = r.mean()
            r_bar_perms[perm_i] = r_bar
            fleiss_kappa_stat_perms[perm_i] = fleiss_kappa_stat
            dice_bar_perms[perm_i] = dice_bar
        # END PERMS
        print "save", key
        np.savez_compressed(OUTPUT+"/perms_"+key+".npz",
                            auc=auc_perms, recall_0=recall_0_perms, recall_1=recall_1_perms, recall_mean=recall_mean_perms,
                            r_bar=r_bar_perms, fleiss_kappa=fleiss_kappa_stat_perms,
                            dice_bar=dice_bar_perms)
    # %pylab qt
    not_missing = np.ones(NRNDPERMS, bool)
    for i, key in enumerate(keys):
        not_missing &= np.logical_not(np.isnan(np.load(OUTPUT+"/perms_"+key+".npz")['recall_mean']))

    perms = dict()
    fig, axis = plt.subplots(len(keys), 4)#, sharex='col')
    for i, key in enumerate(keys):
        perms[key] = np.load(OUTPUT+"/perms_"+key+".npz")
        # RM Nan
        perms[key] = {k:perms[key][k] for k in perms[key].keys()}
        perms[key]['recall_0'] = perms[key]['recall_0'][not_missing]
        perms[key]['recall_1'] = perms[key]['recall_1'][not_missing]
        perms[key]['recall_mean'] = perms[key]['recall_mean'][not_missing]
        perms[key]['beta_r_bar'] = perms[key]['r_bar'][not_missing]
        perms[key]['beta_fleiss_kappa'] = perms[key]['fleiss_kappa'][not_missing]
        perms[key]['beta_dice_bar'] = perms[key]['dice_bar'][not_missing]
        #'auc', 'dice_bar', 'fleiss_kappa', 'r_bar']
        #perms[key]['recalls'].mean(axis=1)
        n, bins, patches = axis[i, 0].hist(perms[key]['recall_mean'], 50, normed=1, histtype='stepfilled')
        axis[i, 0].set_title(key + "_recall_mean")
        n, bins, patches = axis[i, 1].hist(perms[key]['beta_r_bar'], 50, normed=1, histtype='stepfilled')
        axis[i, 1].set_title(key + "_r_bar")
        n, bins, patches = axis[i, 2].hist(perms[key]['beta_fleiss_kappa'], 50, histtype='stepfilled')
        axis[i, 2].set_title(key + "_fleiss_kappa")
        n, bins, patches = axis[i, 3].hist(perms[key]['beta_dice_bar'], 50)#, 50, normed=1, histtype='stepfilled')
        axis[i, 3].set_title(key + "_dice_bar")
    plt.show()

    def close(vec, val, tol=1e-4):
        return np.abs(vec - val) < tol
    # Read true scores
    import pandas as pd
    true_d = pd.read_csv(os.path.join(BASE, "..", "MCIc-CTL_csi.csv"))
    #tv_v, l2tv_v, l2_v, l1l2tv_v, l1tv_v, l1l2_v, l1_v
    true_d = true_d[close(true_d.a, tv_v[0]) & close(true_d.k, tv_v[4])]
    true = {
        tv_k : true_d[close(true_d.l1, tv_v[1]) & close(true_d.l2, tv_v[2]) & close(true_d.tv, tv_v[3])],
        l2tv_k : true_d[close(true_d.l1, l2tv_v[1]) & close(true_d.l2, l2tv_v[2]) & close(true_d.tv, l2tv_v[3])] ,
        l1l2tv_k : true_d[close(true_d.l1, l1l2tv_v[1]) & close(true_d.l2, l1l2tv_v[2]) & close(true_d.tv, l1l2tv_v[3])],
        l1tv_k : true_d[close(true_d.l1, l1tv_v[1]) & close(true_d.l2, l1tv_v[2]) & close(true_d.tv, l1tv_v[3])],
        l1l2_k : true_d[close(true_d.l1, l1l2_v[1]) & close(true_d.l2, l1l2_v[2]) & close(true_d.tv, l1l2_v[3])],
        l1_k : true_d[close(true_d.l1, l1_v[1]) & close(true_d.l2, l1_v[2]) & close(true_d.tv, l1_v[3])],
        l2_k : true_d[close(true_d.l1, l2_v[1]) & close(true_d.l2, l2_v[2]) & close(true_d.tv, l2_v[3])]
        }

    # Compute P values
    scores =  ['recall_0', 'recall_1', 'recall_mean', 'auc', 'beta_fleiss_kappa', 'beta_r_bar']
    counts = list()
    #k = l1tv_k
    for k in keys:
        nperms = float(perms[k]['recall_mean'].shape[0])
        counts.append([k] + 
                     [np.sum(perms[k][score] > true[k][score].values[0], axis=0) for score in scores])
    # l1 vs l1tv
    counts.append(["l1 vs l1tv"] +
            [np.sum((perms[l1tv_k][score] - perms[l1_k][score]) > (true[l1tv_k][score].values[0] - true[l1_k][score].values[0]))
                    for score in scores])

    # l1 vs l1tv
    counts.append(["l2 vs l2tv"] +
            [np.sum((perms[l2tv_k][score] - perms[l2_k][score]) > (true[l2tv_k][score].values[0] - true[l2_k][score].values[0]))
                    for score in scores])
    """
    score = 'recall_mean'
    score = 'beta_r_bar'
    np.sum((perms[l2tv_k][score] - perms[l2_k][score]) > (true[l2tv_k][score].values[0] - true[l2_k][score].values[0]))
    np.mean(perms[l2tv_k][score] - perms[l2_k][score])
    np.mean(perms[l2tv_k][score])
    np.mean(perms[l2_k][score])
    """
    # l1 vs l1tv
    counts.append(["l1l2 vs l1l2tv"] +
            [np.sum((perms[l1l2tv_k][score] - perms[l1l2_k][score]) > (true[l1l2tv_k][score].values[0] - true[l1l2_k][score].values[0]))
                    for score in scores])
    #
    counts = pd.DataFrame(counts, columns=['condition'] + scores)
    counts.to_csv(os.path.join(OUTPUT, "count_stats_permutations.csv"), index=False)


if __name__ == "__main__":
    WD = "/neurospin/brainomics/2013_adni/MCIc-CTL_csi"
    INPUT_DATA_X = os.path.join('X.npy')
    INPUT_DATA_y = os.path.join('y.npy')
    INPUT_MASK_PATH = os.path.join("mask.nii")
    NFOLDS = 5
    #WD = os.path.join(WD, 'logistictvenet_5cv')
    if not os.path.exists(WD):
        os.makedirs(WD)

    os.chdir(WD)

    #############################################################################
    ## Create config file
    y = np.load(INPUT_DATA_y)
    if os.path.exists("config_rndperm.json"):
        inf = open("config_rndperm.json", "r")
        old_conf = json.load(inf)
        rndperm = old_conf["resample"]
        inf.close()
    else:
        rndperm = [[tr.tolist(), te.tolist()] for perm in xrange(NRNDPERMS)
            for tr, te in StratifiedKFold(y.ravel(), n_folds=NFOLDS)]
    params = \
        [(0.01, 0.0, 1.0, 0.0, -1.0), # l2
        (0.01, 0.0, 0.5, 0.5, -1.0),  # l2tv
        (0.01, 1.0, 0.0, 0.0, -1.0),  # l1
        (0.01, 0.5, 0.0, 0.5, -1.0),  # l1tv
        (0.01, 0.0, 0.0, 1.0, -1.0),  # tv
        (0.01, 0.5, 0.5, 0.0, -1.0),  # l1l2
        (0.01, 0.35, 0.35, 0.3, -1.0)] #l1l2tv

    user_func_filename = os.path.join(os.environ["HOME"],
        "git", "scripts", "2013_adni", "MCIc-CTL",
        "03_rndperm_tvenet_csi.py")
    #print __file__, os.path.abspath(__file__)
    print "user_func", user_func_filename
    # Use relative path from config.json
    config = dict(data=dict(X=INPUT_DATA_X, y=INPUT_DATA_y),
                  params=params, resample=rndperm,
                  mask_filename=INPUT_MASK_PATH,
                  penalty_start = 3,
                  map_output="rndperm",
                  user_func=user_func_filename,
                  reduce_group_by="params",
                  reduce_output="MCIc-CTL_csi_rndperm.csv")
    json.dump(config, open(os.path.join(WD, "config_rndperm.json"), "w"))

    #############################################################################
    # Build utils files: sync (push/pull) and PBS
    import brainomics.cluster_gabriel as clust_utils
    sync_push_filename, sync_pull_filename, WD_CLUSTER = \
        clust_utils.gabriel_make_sync_data_files(WD)
    cmd = "mapreduce.py --map  %s/config_rndperm.json" % WD_CLUSTER
    clust_utils.gabriel_make_qsub_job_files(WD, cmd)
    #############################################################################
    # Sync to cluster
    print "Sync data to gabriel.intra.cea.fr: "
    os.system(sync_push_filename)
    #############################################################################
    print "# Start by running Locally with 2 cores, to check that everything os OK)"
    print "Interrupt after a while CTL-C"
    print "mapreduce.py --map %s/config_rndperm.json --ncore 2" % WD
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
    print "mapreduce.py --reduce %s/config_rndperm.json" % WD
