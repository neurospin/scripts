# -*- coding: utf-8 -*-
"""
Created on Fri May 30 20:03:12 2014

@author: edouard.duchesnay@cea.fr

cd /neurospin/brainomics/2013_adni/MCIc-CTL_cs_s_modselectcv
cp ../MCIc-CTL_cs_s/y.* .
cp ../MCIc-CTL_cs_s/X.* .
cp ../MCIc-CTL_cs_s/mask.* .

%run -i ~/git/scripts/2013_adni/MCIc-CTL/02_tvenet_modselectcv_cs_s.py

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

WD = "/neurospin/brainomics/2013_adni/MCIc-CTL_cs_s_modselectcv"
def config_filenane(): return os.path.join(WD, "config_modselectcv.json")
def results_filenane(): return os.path.join(WD, "MCIc-CTL_cs_s_modselectcv.xlsx")

def init():
    INPUT_DATA_X = os.path.join('X.npy')
    INPUT_DATA_y = os.path.join('y.npy')
    INPUT_MASK_PATH = os.path.join("mask.nii")
    NFOLDS_INNER, NFOLDS_OUTER  = 5, 5
    #WD = os.path.join(WD, 'logistictvenet_5cv')
    if not os.path.exists(WD):
        os.makedirs(WD)

    os.chdir(WD)

    #############################################################################
    ## Create config file
    y = np.load(INPUT_DATA_y)
    X = np.load(INPUT_DATA_X)
    from parsimony.utils.penalties import l1_max_logistic_loss
    assert l1_max_logistic_loss(X[:, 2:], y) == 0.18046445850741652
    if os.path.exists(config_filenane()):
        old_conf = json.load(open(config_filenane()))
        cv = old_conf["resample"]
    else:
        cv_outer = [[tr, te] for tr,te in StratifiedKFold(y.ravel(), n_folds=NFOLDS_OUTER, random_state=42)]
        """
        cv_outer = [[np.array(tr), np.array(te)] for tr,te in json.load(open("/neurospin/brainomics/2013_adni/MCIc-CTL_cs_s/config.json", "r"))["resample"][1:]]
        """
        import collections
        cv = collections.OrderedDict()
        for cv_outer_i, (tr_val, te) in enumerate(cv_outer):
            cv["cv%02d/refit" % cv_outer_i] = [tr_val, te]
            cv_inner = StratifiedKFold(y[tr_val].ravel(), n_folds=NFOLDS_INNER, random_state=42)
            for cv_inner_i, (tr, val) in enumerate(cv_inner):
                cv["cv%02d/cvnested%02d" % (cv_outer_i, cv_inner_i)] = [tr_val[tr], tr_val[val]]
        for k in cv:
            cv[k] = [cv[k][0].tolist(), cv[k][1].tolist()]

    print cv.keys()
    # Some QC
    N = float(len(y)); p0 = np.sum(y==0) / N; p1 = np.sum(y==1) / N;
    for k in cv:
        tr, val = cv[k]
        tr, val = np.array(tr), np.array(val)
        print k, "\t: tr+val=", len(tr) + len(val)
        assert not set(tr).intersection(val)
        assert abs(np.sum(y[tr]==0)/float(len(y[tr])) - p0) < 0.01
        assert abs(np.sum(y[tr]==1)/float(len(y[tr])) - p1) < 0.01
        if k.count("refit"):
            te = val
            assert len(tr) + len(te) == len(y)
            assert abs(len(y[tr])/N - (1 - 1./NFOLDS_OUTER)) < 0.01
        else:
            te = np.array(cv[k.split("/")[0] + "/refit"])[1]
            assert abs(len(y[tr])/N - (1 - 1./NFOLDS_OUTER) * (1 - 1./NFOLDS_INNER)) < 0.01
            assert not set(tr).intersection(te)
            assert not set(val).intersection(te)
            len(tr) + len(val) + len(te) == len(y)

    tv_ratios = [0., .2, .8]
    l1_ratios = [np.array([1., .1, .9, 1]), np.array([1., .9, .1, 1])]  # [alpha, l1 l2 tv]
    alphas_l1l2tv = [.01, .1]
    alphas_l2tv = [round(alpha, 10) for alpha in 10. ** np.arange(-2, 4)]
    k_range = [-1]
    l1l2tv =[np.array([alpha, float(1-tv), float(1-tv), tv]) * l1_ratio
        for alpha in alphas_l1l2tv for tv in tv_ratios for l1_ratio in l1_ratios]
    # specific case for without l1 since it supports larger penalties
    l2tv =[np.array([alpha, 0., float(1-tv), tv])
        for alpha in alphas_l2tv for tv in tv_ratios]
    params = l1l2tv + l2tv
    params = [param.tolist() + [k] for k in k_range for param in params]
    params = {"_".join([str(p) for p in param]):param for param in params}
    #assert len(params) == 30
    user_func_filename = os.path.join(os.environ["HOME"],
        "git", "scripts", "2013_adni", "MCIc-CTL",
        "02_tvenet_modselectcv_cs_s.py")
    #print __file__, os.path.abspath(__file__)
    print "user_func", user_func_filename
    #import sys
    #sys.exit(0)
    # Use relative path from config.json
    config = dict(data=dict(X=INPUT_DATA_X, y=INPUT_DATA_y),
                  params=params, resample=cv,
                  mask_filename=INPUT_MASK_PATH,
                  penalty_start = 2,
                  map_output="modselectcv",
                  user_func=user_func_filename,
                  #reduce_input="rndperm/*/*",
                  reduce_group_by="user_defined",
                  reduce_output="MCIc-CTL_cs_s_modselectcv.csv")
    json.dump(config, open(os.path.join(WD, "config_modselectcv.json"), "w"))

    #############################################################################
    # Build utils files: sync (push/pull) and PBS
    import brainomics.cluster_gabriel as clust_utils
    sync_push_filename, sync_pull_filename, WD_CLUSTER = \
        clust_utils.gabriel_make_sync_data_files(WD)
    cmd = "mapreduce.py --map  %s/config_modselectcv.json" % WD_CLUSTER
    clust_utils.gabriel_make_qsub_job_files(WD, cmd)
    #############################################################################
    # Sync to cluster
    print "Sync data to gabriel.intra.cea.fr: "
    os.system(sync_push_filename)
    #############################################################################
    print "# Start by running Locally with 2 cores, to check that everything os OK)"
    print "Interrupt after a while CTL-C"
    print "mapreduce.py --map %s/config_modselectcv.json --ncore 2" % WD
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
    print "mapreduce.py --reduce %s/config_modselectcv.json" % WD

def load_globals(config):
    import mapreduce as GLOBAL  # access to global variables
    GLOBAL.DATA = GLOBAL.load_data(config["data"])
    STRUCTURE = nibabel.load(config["mask_filename"])
    try:
        A = tv_helper.linear_operator_from_mask(STRUCTURE.get_data())
    except:
        A, _ = tv_helper.A_from_mask(STRUCTURE.get_data())
    GLOBAL.A, GLOBAL.STRUCTURE, GLOBAL.CONFIG = A, STRUCTURE, config


def resample(config, resample_nb):
    import mapreduce as GLOBAL  # access to global variables
    GLOBAL.DATA = GLOBAL.load_data(config["data"])
    resample = config["resample"][resample_nb]
    if resample is not None:
        GLOBAL.DATA_RESAMPLED = {k: [GLOBAL.DATA[k][idx, ...] for idx in resample]
                            for k in GLOBAL.DATA}
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
        A = tv_helper.linear_operator_from_mask(mask)
        Xtr_r = np.hstack([Xtr[:, :penalty_start], Xtr[:, penalty_start:][:, aov.get_support()]])
        Xte_r = np.hstack([Xte[:, :penalty_start], Xte[:, penalty_start:][:, aov.get_support()]])
    else:
        mask = STRUCTURE.get_data() != 0
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

def scores(key, paths, config, ret_y=False):
    import glob, mapreduce
    print key
    values = [mapreduce.OutputCollector(p) for p in paths]
    values = [item.load() for item in values]
    recall_mean_std = np.std([np.mean(precision_recall_fscore_support(
        item["y_true"].ravel(), item["y_pred"])[1]) for item in values]) / np.sqrt(len(values))
    y_true = [item["y_true"].ravel() for item in values]
    y_pred = [item["y_pred"].ravel() for item in values]
    prob_pred = [item["proba_pred"].ravel() for item in values]
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    prob_pred = np.concatenate(prob_pred)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average=None)
    auc = roc_auc_score(y_true, prob_pred) #area under curve score.
    n_ite = None
    betas = np.hstack([item["beta"][config['penalty_start']:, :] for item in values]).T
    ## Compute beta similarity measures
    # Correlation
    R = np.corrcoef(betas)
    #print R
    R = R[np.triu_indices_from(R, 1)]
    print R
    # Fisher z-transformation / average
    z_bar = np.mean(1. / 2. * np.log((1 + R) / (1 - R)))
    # bracktransform
    r_bar = (np.exp(2 * z_bar) - 1) /  (np.exp(2 * z_bar) + 1)

    # threshold betas to compute fleiss_kappa and DICE
    try:
        betas_t = np.vstack([array_utils.arr_threshold_from_norm2_ratio(betas[i, :], .99)[0] for i in xrange(betas.shape[0])])
        #print "--", np.sqrt(np.sum(betas_t ** 2, 1)) / np.sqrt(np.sum(betas ** 2, 1))
        print np.allclose(np.sqrt(np.sum(betas_t ** 2, 1)) / np.sqrt(np.sum(betas ** 2, 1)), [0.99]*5,
                           rtol=0, atol=1e-02)

        # Compute fleiss kappa statistics
        beta_signed = np.sign(betas_t)
        table = np.zeros((beta_signed.shape[1], 3))
        table[:, 0] = np.sum(beta_signed == 0, 0)
        table[:, 1] = np.sum(beta_signed == 1, 0)
        table[:, 2] = np.sum(beta_signed == -1, 0)
        fleiss_kappa_stat = fleiss_kappa(table)

        # Paire-wise Dice coeficient
        ij = [[i, j] for i in xrange(5) for j in xrange(i+1, 5)]
        dices = list()
        for idx in ij:
            A, B = beta_signed[idx[0], :], beta_signed[idx[1], :]
            dices.append(float(np.sum((A == B)[(A != 0) & (B != 0)])) / (np.sum(A != 0) + np.sum(B != 0)))
        dice_bar = np.mean(dices)
    except:
        dice_bar = fleiss_kappa_stat = 0.

    scores = OrderedDict()
    try:
        a, l1, l2 , tv , k = [float(par) for par in key.split("_")]
        scores['a'] = a
        scores['l1'] = l1
        scores['l2'] = l2
        scores['tv'] = tv
        left = float(1 - tv)
        if left == 0: left = 1.
        scores['l1_ratio'] = float(l1) / left
        scores['k'] = k
    except:
        pass
    scores['recall_0'] = r[0]
    scores['recall_1'] = r[1]
    scores['recall_mean'] = r.mean()
    scores['recall_mean_std'] = recall_mean_std
    scores['auc'] = auc
#    scores['beta_cor_mean'] = beta_cor_mean
    scores['precision_0'] = p[0]
    scores['precision_1'] = p[1]
    scores['precision_mean'] = p.mean()
    scores['f1_0'] = f[0]
    scores['f1_1'] = f[1]
    scores['f1_mean'] = f.mean()
    scores['support_0'] = s[0]
    scores['support_1'] = s[1]
#    scores['corr']= corr
    scores['beta_r'] = str(R)
    scores['beta_r_bar'] = r_bar
    scores['beta_fleiss_kappa'] = fleiss_kappa_stat
    scores['beta_dice'] = str(dices)
    scores['beta_dice_bar'] = dice_bar
    scores['n_ite'] = n_ite
    scores['param_key'] = key
    if ret_y:
        scores["y_true"], scores["y_pred"], scores["prob_pred"] = y_true, y_pred, prob_pred
    return scores

def reducer():
    import os, glob, pandas as pd
    os.chdir(os.path.dirname(config_filenane()))
    config = json.load(open(config_filenane()))
    paths = glob.glob(os.path.join(config['map_output'], "*", "*", "*"))
    #paths = [p for p in paths if not p.count("0.8_-1")]

    def close(vec, val, tol=1e-4):
        return np.abs(vec - val) < tol

    def groupby_paths(paths, pos):
        groups = {g:[] for g in set([p.split("/")[pos] for p in paths])}
        for p in paths:
            groups[p.split("/")[pos]].append(p)
        return groups

    def argmaxscore_bygroup(data, groupby='fold', param_key="param_key", score="recall_mean"):
        arg_max_byfold = list()
        for fold, data_fold in data.groupby(groupby):
            assert len(data_fold) == len(set(data_fold[param_key]))  # ensure all  param are diff
            arg_max_byfold.append([fold, data_fold.ix[data_fold[score].argmax()][param_key], data_fold[score].max()])
        return pd.DataFrame(arg_max_byfold, columns=[groupby, param_key, score])

    print '## Refit scores'
    print '## ------------'
    byparams = groupby_paths([p for p in paths if p.count("refit")], 3)
    byparams_scores = {k:scores(k, v, config) for k, v in byparams.iteritems()}
    data = [byparams_scores[k].values() for k in byparams_scores]

    columns = byparams_scores[byparams_scores.keys()[0]].keys()
    scores_refit = pd.DataFrame(data, columns=columns)
    
    print '## doublecv scores by outer-cv and by params'
    print '## -----------------------------------------'
    data = list()
    bycv = groupby_paths([p for p in paths if p.count("cvnested")], 1)
    for fold, paths_fold in bycv.iteritems():
        print fold
        byparams = groupby_paths([p for p in paths_fold], 3)
        byparams_scores = {k:scores(k, v, config) for k, v in byparams.iteritems()}
        data += [[fold] + byparams_scores[k].values() for k in byparams_scores]
    scores_dcv_byparams = pd.DataFrame(data, columns=["fold"] + columns)

    # rm small l1 with large tv & large l1 with small tv
    rm = \
        (close(scores_dcv_byparams.l1_ratio, 0.1) & close(scores_dcv_byparams.tv, 0.8)) |\
        (close(scores_dcv_byparams.l1_ratio, 0.9) & close(scores_dcv_byparams.tv, 0.2))
    np.sum(rm)
    scores_dcv_byparams = scores_dcv_byparams[np.logical_not(rm)]

    # model selection on nested cv for 8 cases
    l2 = scores_dcv_byparams[(scores_dcv_byparams.l1 == 0) & (scores_dcv_byparams.tv == 0)]
    l2tv = scores_dcv_byparams[(scores_dcv_byparams.l1 == 0) & (scores_dcv_byparams.tv != 0)]
    l1l2 = scores_dcv_byparams[(scores_dcv_byparams.l1 != 0) & (scores_dcv_byparams.tv == 0)]
    l1l2tv = scores_dcv_byparams[(scores_dcv_byparams.l1 != 0) & (scores_dcv_byparams.tv != 0)]
    # large ans small l1
    l1l2_ll1 = scores_dcv_byparams[close(scores_dcv_byparams.l1_ratio, 0.9) & (scores_dcv_byparams.tv == 0)]
    l1l2tv_ll1 = scores_dcv_byparams[close(scores_dcv_byparams.l1_ratio, 0.9) & (scores_dcv_byparams.tv != 0)]
    l1l2_sl1 = scores_dcv_byparams[close(scores_dcv_byparams.l1_ratio, 0.1) & (scores_dcv_byparams.tv == 0)]
    l1l2tv_sl1 = scores_dcv_byparams[close(scores_dcv_byparams.l1_ratio, 0.1) & (scores_dcv_byparams.tv != 0)]

    print '## Model selection'
    print '## ---------------'
    l2 = argmaxscore_bygroup(l2); l2["method"] = "l2"
    l2tv = argmaxscore_bygroup(l2tv); l2tv["method"] = "l2tv"
    l1l2 = argmaxscore_bygroup(l1l2); l1l2["method"] = "l1l2"
    l1l2tv = argmaxscore_bygroup(l1l2tv); l1l2tv["method"] = "l1l2tv"
 
    l1l2_ll1 = argmaxscore_bygroup(l1l2_ll1); l1l2_ll1["method"] = "l1l2_ll1"
    l1l2tv_ll1 = argmaxscore_bygroup(l1l2tv_ll1); l1l2tv_ll1["method"] = "l1l2tv_ll1"
    l1l2_sl1 = argmaxscore_bygroup(l1l2_sl1); l1l2_sl1["method"] = "l1l2_sl1"
    l1l2tv_sl1 = argmaxscore_bygroup(l1l2tv_sl1); l1l2tv_sl1["method"] = "l1l2tv_sl1"

    scores_argmax_byfold = pd.concat([l2, l2tv, l1l2, l1l2tv, l1l2_ll1, l1l2tv_ll1, l1l2_sl1, l1l2tv_sl1])

    print '## Apply best model on refited'
    print '## ---------------------------'
    scores_l2 = scores("nestedcv", [os.path.join(config['map_output'], row["fold"], "refit", row["param_key"]) for index, row in l2.iterrows()], config)
    scores_l2tv = scores("nestedcv", [os.path.join(config['map_output'], row["fold"], "refit", row["param_key"]) for index, row in l2tv.iterrows()], config)
    scores_l1l2 = scores("nestedcv", [os.path.join(config['map_output'], row["fold"], "refit", row["param_key"]) for index, row in l1l2.iterrows()], config)
    scores_l1l2tv = scores("nestedcv", [os.path.join(config['map_output'], row["fold"], "refit", row["param_key"]) for index, row in l1l2tv.iterrows()], config)

    scores_l1l2_ll1 = scores("nestedcv", [os.path.join(config['map_output'], row["fold"], "refit", row["param_key"]) for index, row in l1l2_ll1.iterrows()], config)
    scores_l1l2tv_ll1 = scores("nestedcv", [os.path.join(config['map_output'], row["fold"], "refit", row["param_key"]) for index, row in l1l2tv_ll1.iterrows()], config)

    scores_l1l2_sl1 = scores("nestedcv", [os.path.join(config['map_output'], row["fold"], "refit", row["param_key"]) for index, row in l1l2_sl1.iterrows()], config)
    scores_l1l2tv_sl1 = scores("nestedcv", [os.path.join(config['map_output'], row["fold"], "refit", row["param_key"]) for index, row in l1l2tv_sl1.iterrows()], config)

    scores_cv = pd.DataFrame([["l2"] + scores_l2.values(),
                  ["l2tv"] + scores_l2tv.values(),
                  ["l1l2"] + scores_l1l2.values(),
                  ["l1l2tv"] + scores_l1l2tv.values(),

                  ["l1l2_ll1"] + scores_l1l2_ll1.values(),
                  ["l1l2tv_ll1"] + scores_l1l2tv_ll1.values(),

                  ["l1l2_sl1"] + scores_l1l2_sl1.values(),
                  ["l1l2tv_sl1"] + scores_l1l2tv_sl1.values()], columns=["method"] + scores_l2.keys())
    
    with pd.ExcelWriter(results_filenane()) as writer:
        scores_refit.to_excel(writer, sheet_name='scores_refit', index=False)
        scores_dcv_byparams.to_excel(writer, sheet_name='scores_dcv_byparams', index=False)
        scores_argmax_byfold.to_excel(writer, sheet_name='scores_argmax_byfold', index=False)
        scores_cv.to_excel(writer, sheet_name='scores_cv', index=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--init', action='store_true', default=False,
                        help="Init config file & sync to cluster")
 
    parser.add_argument('-r', '--reduce', action='store_true', default=False,
                        help="Reduce, ie.: compute scores")

    options = parser.parse_args()

    if options.init:
        init()

    elif options.reduce:
        reducer()
