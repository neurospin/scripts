# -*- coding: utf-8 -*-
"""
Created on Fri May 30 20:03:12 2014

@author: edouard.duchesnay@cea.fr

cd /neurospin/brainomics/2013_adni/MCIc-CTL_csi_modselectcv
mapreduce.py
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

def scores(key, paths):
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
    betas = np.hstack([item["beta"] for item in values]).T
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
        print "--", np.sqrt(np.sum(betas_t ** 2, 1)) / np.sqrt(np.sum(betas ** 2, 1))
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
        beta_n0 = betas_t != 0
        ij = [[i, j] for i in xrange(5) for j in xrange(i+1, 5)]
        #print [[idx[0], idx[1]] for idx in ij]
        dice_bar = np.mean([float(np.sum(beta_signed[idx[0], :] == beta_signed[idx[1], :])) /\
             (np.sum(beta_n0[idx[0], :]) + np.sum(beta_n0[idx[1], :]))
             for idx in ij])
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
    scores['beta_dice_bar'] = dice_bar
    scores['n_ite'] = n_ite
    scores['param_key'] = key
    return scores

def reducer():
    import os, glob, pandas as pd
    config_filenane = "/neurospin/brainomics/2013_adni/MCIc-CTL_csi_modselectcv/config_modselectcv.json"
    os.chdir(os.path.dirname(config_filenane))
    config = json.load(open(config_filenane))
    paths = glob.glob(os.path.join(config['map_output'], "*", "*", "*"))
    #paths = [p for p in paths if not p.count("0.8_-1")]

    def close(vec, val, tol=1e-4):
        return np.abs(vec - val) < tol

    def groupby_paths(paths, pos):
        groups = {g:[] for g in set([p.split("/")[pos] for p in paths])}
        for p in paths:
            groups[p.split("/")[pos]].append(p)
        return groups

    def model_selection(data, groupby='fold', param_key="param_key", score="recall_mean"):
        arg_max_byfold = list()
        for fold, data_fold in data.groupby(groupby):
            assert len(data_fold) == len(set(data_fold[param_key]))  # ensure all  param are diff
            arg_max_byfold.append([fold, data_fold.ix[data_fold[score].argmax()][param_key], data_fold[score].max()])
        return pd.DataFrame(arg_max_byfold, columns=[groupby, param_key, score])

    # Refit scores
    byparams = groupby_paths([p for p in paths if p.count("refit")], 3)
    byparams_scores = {k:scores(k, v) for k, v in byparams.iteritems()}
    data = [byparams_scores[k].values() for k in byparams_scores]

    columns = byparams_scores[byparams_scores.keys()[0]].keys()
    scores_tab_refit = pd.DataFrame(data, columns=columns)
    
    ## doublecv scores by outer-cv and by params
    data = list()
    bycv = groupby_paths([p for p in paths if p.count("cvnested")], 1)
    for fold, paths_fold in bycv.iteritems():
        print fold
        byparams = groupby_paths([p for p in paths_fold], 3)
        byparams_scores = {k:scores(k, v) for k, v in byparams.iteritems()}
        data += [[fold] + byparams_scores[k].values() for k in byparams_scores]
    scores_tab_doublecv_byparams = pd.DataFrame(data, columns=["fold"] + columns)

    # rm small l1 with large tv & large l1 with small tv
    rm = \
        (close(scores_tab_doublecv_byparams.l1_ratio, 0.1) & close(scores_tab_doublecv_byparams.tv, 0.8)) |\
        (close(scores_tab_doublecv_byparams.l1_ratio, 0.9) & close(scores_tab_doublecv_byparams.tv, 0.2))
    np.sum(rm)
    scores_tab_doublecv_byparams = scores_tab_doublecv_byparams[np.logical_not(rm)]

    # model selection on nested cv for 8 cases
    l2 = scores_tab_doublecv_byparams[(scores_tab_doublecv_byparams.l1 == 0) & (scores_tab_doublecv_byparams.tv == 0)]
    l2tv = scores_tab_doublecv_byparams[(scores_tab_doublecv_byparams.l1 == 0) & (scores_tab_doublecv_byparams.tv != 0)]
    l1l2 = scores_tab_doublecv_byparams[(scores_tab_doublecv_byparams.l1 != 0) & (scores_tab_doublecv_byparams.tv == 0)]
    l1l2tv = scores_tab_doublecv_byparams[(scores_tab_doublecv_byparams.l1 != 0) & (scores_tab_doublecv_byparams.tv != 0)]
    # large ans small l1
    l1l2_ll1 = scores_tab_doublecv_byparams[close(scores_tab_doublecv_byparams.l1_ratio, 0.9) & (scores_tab_doublecv_byparams.tv == 0)]
    l1l2tv_ll1 = scores_tab_doublecv_byparams[close(scores_tab_doublecv_byparams.l1_ratio, 0.9) & (scores_tab_doublecv_byparams.tv != 0)]
    l1l2_sl1 = scores_tab_doublecv_byparams[close(scores_tab_doublecv_byparams.l1_ratio, 0.1) & (scores_tab_doublecv_byparams.tv == 0)]
    l1l2tv_sl1 = scores_tab_doublecv_byparams[close(scores_tab_doublecv_byparams.l1_ratio, 0.1) & (scores_tab_doublecv_byparams.tv != 0)]

    l2 = model_selection(l2); l2["method"] = "l2"
    l2tv = model_selection(l2tv); l2tv["method"] = "l2tv"
    l1l2 = model_selection(l1l2); l1l2["method"] = "l1l2"
    l1l2tv = model_selection(l1l2tv); l1l2tv["method"] = "l1l2tv"
 
    l1l2_ll1 = model_selection(l1l2_ll1); l1l2_ll1["method"] = "l1l2_ll1"
    l1l2tv_ll1 = model_selection(l1l2tv_ll1); l1l2tv_ll1["method"] = "l1l2tv_ll1"
    l1l2_sl1 = model_selection(l1l2_sl1); l1l2_sl1["method"] = "l1l2_sl1"
    l1l2tv_sl1 = model_selection(l1l2tv_sl1); l1l2tv_sl1["method"] = "l1l2tv_sl1"

   scores_tab_argmax_byfold = pd.concat([l2, l2tv, l1l2, l1l2tv, l1l2_ll1, l1l2tv_ll1, l1l2_sl1, l1l2tv_sl1])

    # Apply best model on refited
    scores_l2 = scores("nestedcv", [os.path.join(config['map_output'], row["fold"], "refit", row["param_key"]) for index, row in l2.iterrows()])
    scores_l2tv = scores("nestedcv", [os.path.join(config['map_output'], row["fold"], "refit", row["param_key"]) for index, row in l2tv.iterrows()])
    scores_l1l2 = scores("nestedcv", [os.path.join(config['map_output'], row["fold"], "refit", row["param_key"]) for index, row in l1l2.iterrows()])
    scores_l1l2tv = scores("nestedcv", [os.path.join(config['map_output'], row["fold"], "refit", row["param_key"]) for index, row in l1l2tv.iterrows()])

    scores_l1l2_ll1 = scores("nestedcv", [os.path.join(config['map_output'], row["fold"], "refit", row["param_key"]) for index, row in l1l2_ll1.iterrows()])
    scores_l1l2tv_ll1 = scores("nestedcv", [os.path.join(config['map_output'], row["fold"], "refit", row["param_key"]) for index, row in l1l2tv_ll1.iterrows()])

    scores_l1l2_sl1 = scores("nestedcv", [os.path.join(config['map_output'], row["fold"], "refit", row["param_key"]) for index, row in l1l2_sl1.iterrows()])
    scores_l1l2tv_sl1 = scores("nestedcv", [os.path.join(config['map_output'], row["fold"], "refit", row["param_key"]) for index, row in l1l2tv_sl1.iterrows()])

    scores_tab_cv = pd.DataFrame([["l2"] + scores_l2.values(),
                  ["l2tv"] + scores_l2tv.values(),
                  ["l1l2"] + scores_l1l2.values(),
                  ["l1l2tv"] + scores_l1l2tv.values(),

                  ["l1l2_ll1"] + scores_l1l2_ll1.values(),
                  ["l1l2tv_ll1"] + scores_l1l2tv_ll1.values(),

                  ["l1l2_sl1"] + scores_l1l2_sl1.values(),
                  ["l1l2tv_sl1"] + scores_l1l2tv_sl1.values()], columns=["method"] + scores_l2.keys())
    
    with pd.ExcelWriter("scores_doublecv.xls") as writer:
        scores_tab_refit.to_excel(writer, sheet_name='refit', index=False)
        scores_tab_doublecv_byparams.to_excel(writer, sheet_name='doublecv_byparams', index=False)
        scores_tab_argmax_byfold.to_excel(writer, sheet_name='argmax_byfold', index=False)
        scores_tab_cv.to_excel(writer, sheet_name='scores_cv', index=False)

if __name__ == "__main__":
    WD = "/neurospin/brainomics/2013_adni/MCIc-CTL_csi_modselectcv"
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
    assert l1_max_logistic_loss(X[:, 3:], y) == 0.20434911093262279
    if os.path.exists("config_modselectcv.json"):
        old_conf = json.load(open("config_modselectcv.json"))
        cv = old_conf["resample"]
    else:
        cv_outer = [[tr, te] for tr,te in StratifiedKFold(y.ravel(), n_folds=NFOLDS_OUTER, random_state=42)]
        """
        cv_outer = [[np.array(tr), np.array(te)] for tr,te in json.load(open("/neurospin/brainomics/2013_adni/MCIc-CTL_csi/config.json", "r"))["resample"][1:]]
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
    assert len(params) == 30
    user_func_filename = os.path.join(os.environ["HOME"],
        "git", "scripts", "2013_adni", "MCIc-CTL",
        "02_tvenet_modselectcv_csi.py")
    #print __file__, os.path.abspath(__file__)
    print "user_func", user_func_filename
    #import sys
    #sys.exit(0)
    # Use relative path from config.json
    config = dict(data=dict(X=INPUT_DATA_X, y=INPUT_DATA_y),
                  params=params, resample=cv,
                  mask_filename=INPUT_MASK_PATH,
                  penalty_start = 3,
                  map_output="modselectcv",
                  user_func=user_func_filename,
                  #reduce_input="rndperm/*/*",
                  reduce_group_by="user_defined",
                  reduce_output="MCIc-CTL_csi_modselectcv.csv")
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

###############################################################################
def plot_perf():
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    # SOME ERROR WERE HERE CORRECTED 27/04/2014 think its good
    #INPUT_vbm = "/home/ed203246/mega/data/2015_logistic_nestv/adni/MCIc-CTL/MCIc-CTL_cs.csv"
    INPUT = "/neurospin/brainomics/2013_adni/MCIc-CTL_csi_modselectcv/MCIc-CTL_csi_modselectcv.csv"
    y_col = 'recall_mean'
    x_col = 'tv'
    y_col = 'auc'
    a = 0.01
    #color_map = {0.:'#D40000', 0.01: 'black', 0.1:'#F0a513',  0.5:'#2CA02C',  0.9:'#87AADE',  1.:'#214478'}
    color_map = {0.:'#D40000', 0.01:'#F0a513',  0.1:'#2CA02C',  0.5:'#87AADE',  .9:'#214478', 1.: 'black'}
    #                reds dark => brigth,      green         blues: brigth => dark
    input_filename = INPUT
    #input_filename = INPUTS[data_type]["filename"]
    outut_filename = input_filename.replace(".csv", "_%s.pdf" % y_col)
    #print outut_filename
    # Filter data
    data = pd.read_csv(input_filename)
    #data.l1l2_ratio = data.l1l2_ratio.round(5)
    # avoid poor rounding
    data.l1l2_ratio = np.asarray(data.l1l2_ratio).round(3)
    data.tv = np.asarray(data.tv).round(5)
    data.a = np.asarray(data.a).round(5)
    data = data[data.k == -1]
    data = data[data.l1l2_ratio.isin([0, 0.01, 0.1, 0.5, 0.9, 1.])]
    data = data[(data.tv >= 0.1) | (data.tv == 0)]
    def close(vec, val, tol=1e-4):
        return np.abs(vec - val) < tol
    assert np.sum(data.l1l2_ratio == 0.01) == np.sum(close(data.l1l2_ratio, 0.01))
    #data = data[data.a <= 1]
    # for each a, l1l2_ratio, append the last point tv==1
    last = list()
    for a_ in np.unique(data.a):
        full_tv = data[(data.a == a_) & (data.tv == 1)]
        for l1l2_ratio in np.unique(data.l1l2_ratio):
            new = full_tv.copy()
            new.l1l2_ratio = l1l2_ratio
            last.append(new)
    #
    last = pd.concat(last)
    data = pd.concat([data, last])
    data.drop_duplicates(inplace=True)
    #
    from brainomics.plot_utilities import plot_lines
    figures = plot_lines(df=data,
    x_col=x_col, y_col=y_col, colorby_col='l1l2_ratio',
                       splitby_col='a', color_map=color_map)
    pdf = PdfPages(outut_filename)
    for fig in figures:
        print fig, figures[fig]
        pdf.savefig(figures[fig]); plt.clf()
    pdf.close()

def build_summary():
    import pandas as pd
    config_filenane = "/neurospin/brainomics/2013_adni/MCIc-CTL_csi_modselectcv/config_modselectcv.json"
    os.chdir(os.path.dirname(config_filenane))
    config = json.load(open(config_filenane))
    from collections import OrderedDict
    models = OrderedDict()
    models["l2"]     = (0.010,	0.000, 1.000, 0.000)
    models["l2tv"]    = (0.010,	0.000, 0.500, 0.500)
    models["l1"]      = (0.010,	1.000, 0.000, 0.000)
    models["l1tv"]    = (0.010,	0.500, 0.000, 0.500)
    models["tv"]      = (0.010,	0.000, 0.000, 1.000)
    models["l1l2"]    = (0.010,	0.500, 0.500, 0.000)
    models["l1l2tv"]  = (0.010,	0.350, 0.350, 0.300)
    models["l1sl2"]    = (0.010,	0.1, 0.9, 0.000)
    models["l1sl2tv"]  = (0.010,	0.1 * (1-.3), 0.9*(1-.3), 0.300)


    def close(vec, val, tol=1e-4):
        return np.abs(vec - val) < tol

    orig_cv = pd.read_csv(config['reduce_output'])
    cv = orig_cv[["k", "a", "l1", "l2", "tv", 'recall_0', u'recall_1', u'recall_mean',
              'auc', "beta_r_bar", 'beta_fleiss_kappa']]
    summary = list()
    for k in models:
        #k = "l2" k="l1sl2"
        a, l1, l2, tv = models[k]
        l = cv[(cv.k == -1) & close(cv.a, a) & close(cv.l1, l1) & close(cv.l2, l2) & close(cv.tv, tv)]
        assert l.shape[0] == 1
        l["algo"] = k
        summary.append(l)
    summary = pd.concat(summary)
    summary.drop("k", 1, inplace=True)
    cols_diff_in = ["recall_mean", "auc", "beta_r_bar", "beta_fleiss_kappa"]
    cols_diff = ["delta_"+ c for c in cols_diff_in]
    for c in cols_diff:
        summary[c] = None
    delta = summary.ix[summary.algo == "l2tv", cols_diff_in].as_matrix() - summary.ix[summary.algo == "l2", cols_diff_in].as_matrix()
    summary.ix[summary.algo == "l2tv", cols_diff] = delta
    delta = summary.ix[summary.algo == "l1tv", cols_diff_in].as_matrix() - summary.ix[summary.algo == "l1", cols_diff_in].as_matrix()
    summary.ix[summary.algo == "l1tv", cols_diff] = delta
    delta = summary.ix[summary.algo == "l1l2tv", cols_diff_in].as_matrix() - summary.ix[summary.algo == "l1l2", cols_diff_in].as_matrix()
    summary.ix[summary.algo == "l1l2tv", cols_diff] = delta
    delta = summary.ix[summary.algo == "tv", cols_diff_in].as_matrix() - summary.ix[summary.algo == "l2", cols_diff_in].as_matrix()
    summary.ix[summary.algo == "tv", cols_diff] = delta
    delta = summary.ix[summary.algo == "l1sl2tv", cols_diff_in].as_matrix() - summary.ix[summary.algo == "l1sl2", cols_diff_in].as_matrix()
    summary.ix[summary.algo == "l1sl2tv", cols_diff] = delta
    xlsx = pd.ExcelWriter(config['reduce_output'].replace("csv" , "xlsx"))
    orig_cv.to_excel(xlsx, 'All')
    summary.to_excel(xlsx, 'Summary')
    xlsx.save()
