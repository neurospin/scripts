# -*- coding: utf-8 -*-
"""
Created on Fri May 30 20:03:12 2014

@author: edouard.duchesnay@cea.fr
"""

import os
import json
from collections import OrderedDict
import numpy as np
from sklearn.cross_validation import KFold
import nibabel
from sklearn.metrics import r2_score
from sklearn.feature_selection import SelectKBest
from parsimony.estimators import LinearRegressionL1L2TV
import parsimony.functions.nesterov.tv as tv_helper
from brainomics import array_utils
from statsmodels.stats.inter_rater import fleiss_kappa
import matplotlib.pyplot as plt

NFOLDS = 5
NRNDPERMS = 1000


def load_globals(config):
    import mapreduce as GLOBAL  # access to global variables
    GLOBAL.DATA = GLOBAL.load_data(config["data"])
    STRUCTURE = nibabel.load(config["mask_filename"])
    A, _ = tv_helper.A_from_mask(STRUCTURE.get_data())
    GLOBAL.A, GLOBAL.STRUCTURE, GLOBAL.CONFIG = A, STRUCTURE, config


def resample(config, resample_nb):
    import mapreduce as GLOBAL  # access to global variables
    #GLOBAL.DATA = GLOBAL.load_data(config["data"])
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
    #class_weight="auto" # unbiased
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
    mod = LinearRegressionL1L2TV(l1, l2, tv, A, penalty_start=penalty_start)
    mod.fit(Xtr_r, ytr)
    y_pred = mod.predict(Xte_r)
    ret = dict(y_pred=y_pred, y_true=yte, beta=mod.beta,  mask=mask)
    if output_collector:
        output_collector.collect(key, ret)
    else:
        return ret

def reducer_(key, values):
    # key : string of intermediary key
    # load return dict correspondning to mapper ouput. they need to be loaded.
    # DEBUG
    import glob, mapreduce
    BASE = "/neurospin/brainomics/2013_adni/ADAS11-MCIc-CTL/rndperm"
    INPUT = BASE + "/%i/%s"
    OUTPUT = BASE + "/../results/rndperm"
    keys = ["0.001_0.3335_0.3335_0.333_-1",  "0.001_0.5_0_0.5_-1",  "0.001_0.5_0.5_0_-1",  "0.001_1_0_0_-1"]
    for key in keys:
        #key = keys[0]
        paths_5cv_all = [INPUT % (perm, key) for perm in xrange(NFOLDS * NRNDPERMS)]
        idx_5cv_blocks = range(0, (NFOLDS * NRNDPERMS) + NFOLDS, NFOLDS)
        cpt = 0
        qc = dict()
        r2_perms = np.zeros(NRNDPERMS)
        corr_perms = np.zeros(NRNDPERMS)
        r_bar_perms = np.zeros(NRNDPERMS)
        fleiss_kappa_stat_perms = np.zeros(NRNDPERMS)
        dice_bar_perms = np.zeros(NRNDPERMS)
        for perm_i in xrange(len(idx_5cv_blocks)-1):
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
            y_true = [item["y_true"].ravel() for item in values]
            y_pred = [item["y_pred"].ravel() for item in values]
            y_true = np.concatenate(y_true)
            y_pred = np.concatenate(y_pred)
            r2 = r2_score(y_true, y_pred)
            corr = np.corrcoef(y_true.ravel(), y_pred.ravel())[0, 1]
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
                print "--", np.sqrt(np.sum(betas_t ** 2, 1)) / np.sqrt(np.sum(betas ** 2, 1))
                print np.allclose(np.sqrt(np.sum(betas_t ** 2, 1)) / np.sqrt(np.sum(betas ** 2, 1)), [0.99]*5,
                                   rtol=0, atol=1e-02)
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
            r2_perms[perm_i] = r2
            corr_perms[perm_i] = corr
            r_bar_perms[perm_i] = r_bar
            fleiss_kappa_stat_perms[perm_i] = fleiss_kappa_stat
            dice_bar_perms[perm_i] = dice_bar
        # END PERMS
        print "save", key
        np.savez_compressed(OUTPUT+"/perms_"+key+".npz",
                            r2=r2_perms, corr=corr_perms,
                            r_bar=r_bar_perms, fleiss_kappa=fleiss_kappa_stat_perms,
                            dice_bar=dice_bar_perms)
        #
        perms = dict()
        fig, axis = plt.subplots(len(keys), 4)#, sharex='col')
        for i, key in enumerate(keys):
            perms[key] = np.load(OUTPUT+"/perms_"+key+".npz")
            n, bins, patches = axis[i, 0].hist(perms[key]['r2'], 50, normed=1, histtype='stepfilled')
            axis[i, 0].set_title(key + "_r2")
            n, bins, patches = axis[i, 1].hist(perms[key]['r_bar'], 50, normed=1, histtype='stepfilled')
            axis[i, 1].set_title(key + "_r_bar")
            n, bins, patches = axis[i, 2].hist(perms[key]['fleiss_kappa'], 50, histtype='stepfilled')
            axis[i, 2].set_title(key + "_fleiss_kappa")
            n, bins, patches = axis[i, 3].hist(perms[key]['dice_bar'], 50)#, 50, normed=1, histtype='stepfilled')
            axis[i, 3].set_title(key + "_dice_bar")
        plt.show()

        l1l2tv, l1tv, l1l2, l1 = ["0.001_0.3335_0.3335_0.333_-1",  "0.001_0.5_0_0.5_-1",  
                             "0.001_0.5_0.5_0_-1",  "0.001_1_0_0_-1"]

        # Read true scores
        import pandas as pd
        true = pd.read_csv(os.path.join(BASE, "..", "ADAS11-MCIc-CTL.csv"))
        true = true[true.a == 0.001]
        true_l1l2tv = true[true.l1 == 0.3335].iloc[0]
        true_l1l2 = true[(true.l1 == 0.5) & (true.l2 == 0.5)].iloc[0]
        true_l1tv = true[(true.l1 == 0.5) & (true.tv == 0.5)].iloc[0]
        true_l1 = true[(true.l1 == 1.)].iloc[0]

        # pvals
        nperms = float(len(perms[l1]['r2']))
        from collections import OrderedDict
        pvals = OrderedDict()
        pvals["cond"] = ['l1', 'l1tv', 'l1l2', 'l1l2tv'] * 4 + \
                ['l1 vs l1tv'] * 4  + ['l1l2 vs l1l2tv'] * 4
        pvals["stat"] = ['r2'] * 4 + ['r_bar'] * 4 + ['fleiss_kappa'] * 4 + ['dice_bar'] * 4 +\
                ['r2', 'r_bar', 'fleiss_kappa', 'dice_bar'] * 2
        pvals["pval"] = [
            np.sum(perms[l1]['r2'] > true_l1["r2"]),
            np.sum(perms[l1tv]['r2'] > true_l1tv["r2"]),
            np.sum(perms[l1l2]['r2'] > true_l1l2["r2"]),
            np.sum(perms[l1l2tv]['r2'] > true_l1l2tv["r2"]),
    
            np.sum(perms[l1]['r_bar'] > true_l1["beta_r_bar"]),
            np.sum(perms[l1tv]['r_bar'] > true_l1tv["beta_r_bar"]),
            np.sum(perms[l1l2]['r_bar'] > true_l1l2["beta_r_bar"]),
            np.sum(perms[l1l2tv]['r_bar'] > true_l1l2tv["beta_r_bar"]),
    
            np.sum(perms[l1]['fleiss_kappa'] > true_l1["beta_fleiss_kappa"]),
            np.sum(perms[l1tv]['fleiss_kappa'] > true_l1tv["beta_fleiss_kappa"]),
            np.sum(perms[l1l2]['fleiss_kappa'] > true_l1l2["beta_fleiss_kappa"]),
            np.sum(perms[l1l2tv]['fleiss_kappa'] > true_l1l2tv["beta_fleiss_kappa"]),
    
            np.sum(perms[l1]['dice_bar'] > true_l1["beta_dice_bar"]),
            np.sum(perms[l1tv]['dice_bar'] > true_l1tv["beta_dice_bar"]),
            np.sum(perms[l1l2]['dice_bar'] > true_l1l2["beta_dice_bar"]),
            np.sum(perms[l1l2tv]['dice_bar'] > true_l1l2tv["beta_dice_bar"]),
    
            # l1 vs l1tv
            np.sum((perms[l1tv]['r2'] - perms[l1]['r2']) > (true_l1tv["r2"] - true_l1["r2"])),
            np.sum((perms[l1tv]['r_bar'] - perms[l1]['r_bar']) > (true_l1tv["beta_r_bar"] - true_l1["beta_r_bar"])),
            np.sum((perms[l1tv]['fleiss_kappa'] - perms[l1]['fleiss_kappa']) > (true_l1tv["beta_fleiss_kappa"] - true_l1["beta_fleiss_kappa"])),
            np.sum((perms[l1tv]['dice_bar'] - perms[l1]['dice_bar']) > (true_l1tv["beta_dice_bar"] - true_l1["beta_dice_bar"])),
    
            # l1l2 vs l1l2tv
            np.sum((perms[l1l2]['r2'] - perms[l1l2tv]['r2']) > (true_l1l2["r2"] - true_l1l2tv["r2"])),
            np.sum((perms[l1l2tv]['r_bar'] - perms[l1l2]['r_bar']) > (true_l1l2tv["beta_r_bar"] - true_l1l2["beta_r_bar"])),
            np.sum((perms[l1l2tv]['fleiss_kappa'] - perms[l1l2]['fleiss_kappa']) > (true_l1l2tv["beta_fleiss_kappa"] - true_l1l2["beta_fleiss_kappa"])),
            np.sum((perms[l1l2tv]['dice_bar'] - perms[l1l2]['dice_bar']) > (true_l1l2tv["beta_dice_bar"] - true_l1l2["beta_dice_bar"]))]

        pvals = pd.DataFrame(pvals)
        pvals["pval"] /= nperms
        pvals.to_csv(os.path.join(OUTPUT, "pvals_stats_permutations.csv"), index=False)


##############################################################################
## Run all
def run_all(config):
    import mapreduce
    WD = "/neurospin/brainomics/2013_adni/ADAS11-MCIc-CTL"
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
    WD = "/neurospin/brainomics/2013_adni/ADAS11-MCIc-CTL"
    INPUT_DATA_X = os.path.join('X.npy')
    INPUT_DATA_y = os.path.join('y.npy')
    INPUT_MASK_PATH = os.path.join("mask.nii.gz")

    #WD = os.path.join(WD, 'logistictvenet_5cv')
    if not os.path.exists(WD): os.makedirs(WD)
    os.chdir(WD)

    #############################################################################
    ## Create config file
    y = np.load(INPUT_DATA_y)
    #NRNDPERMS = 3
    #y = np.arange(4)
    if os.path.exists("config_rndperm.json"):
        inf = open("config_rndperm.json", "r")
        old_conf = json.load(inf)
        rndperm = old_conf["resample"]
        inf.close()
    else:
        rndperm = [[tr.tolist(), te.tolist()] for perm in xrange(NRNDPERMS)
            for tr, te in KFold(n=len(y), n_folds=NFOLDS, random_state=0)]
    #[[i, j] for i in range(2) for j in range(3)]
    #[[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]]
    # parameters grid
    # Re-run with
    params = \
        [[0.001,	0.5,	0.5,	0,     -1], # l1l2
         [0.001,	0.3335,	0.3335,	0.333, -1], # l1l2tv
         [0.001,	1,	0,	0,     -1], # l1
         [0.001,	0.5,	0,	0.5,   -1]] # l1tv
    # User map/reduce function file:
    user_func_filename = os.path.join(os.environ["HOME"],
        "git", "scripts", "2013_adni", "ADAS11-MCIc-CTL",
        "03_rndperm_tvenet_cs.py")
    #print __file__, os.path.abspath(__file__)
    print "user_func", user_func_filename
    # Use relative path from config.json
    config = dict(data=dict(X=INPUT_DATA_X, y=INPUT_DATA_y),
                  params=params, resample=rndperm,
                  mask_filename=INPUT_MASK_PATH,
                  penalty_start = 3,
                  map_output="rndperm",
                  user_func=user_func_filename,
                  #reduce_input="rndperm/*/*",
                  reduce_group_by="params",
                  reduce_output="ADAS11-MCIc-CTL_rndperm.csv")
    json.dump(config, open(os.path.join(WD, "config_rndperm.json"), "w"))

    #############################################################################
    # Build utils files: sync (push/pull) and PBS
    import brainomics.cluster_gabriel as clust_utils
    sync_push_filename, sync_pull_filename, WD_CLUSTER = \
        clust_utils.gabriel_make_sync_data_files(WD)
    cmd = "mapreduce.py --map  %s/config_rndperm.json" % WD_CLUSTER
    clust_utils.gabriel_make_qsub_job_files(WD, cmd, suffix="_rndperm")
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
