# -*- coding: utf-8 -*-
"""
Created on Fri May 30 20:03:12 2014

@author: edouard.duchesnay@cea.fr
"""

import os
import json
from collections import OrderedDict
import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from sklearn.feature_selection import SelectKBest
from parsimony.estimators import LogisticRegressionL1L2TV
import parsimony.functions.nesterov.tv as tv_helper
from brainomics import array_utils
from statsmodels.stats.inter_rater import fleiss_kappa


def load_globals(config):
    import mapreduce as GLOBAL  # access to global variables
    GLOBAL.DATA = GLOBAL.load_data(config["data"])
    import brainomics.mesh_processing as mesh_utils
    mesh_coord, mesh_triangles = mesh_utils.mesh_arrays(config["structure"]["mesh"])
    mask = np.load(config["structure"]["mask"])
    GLOBAL.mesh_coord, GLOBAL.mesh_triangles, GLOBAL.mask = mesh_coord, mesh_triangles, mask
    try:
        A = tv_helper.linear_operator_from_mesh(GLOBAL.mesh_coord, GLOBAL.mesh_triangles, GLOBAL.mask)
    except:
        A = tv_helper.nesterov_linear_operator_from_mesh(GLOBAL.mesh_coord, GLOBAL.mesh_triangles, GLOBAL.mask)
    GLOBAL.A = A
    GLOBAL.CONFIG = config

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
    # STRUCTURE = GLOBAL.STRUCTURE
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
        mask = GLOBAL.mask != 0
        mask[mask] = aov.get_support()
        #print mask.sum()
        A, _ = tv_helper.nesterov_linear_operator_from_mesh(GLOBAL.mesh_coord, GLOBAL.mesh_triangles, mask)
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
    ret = dict(y_pred=y_pred, y_true=yte, beta=mod.beta,  mask=mask, proba_pred=proba_pred)
    if output_collector:
        output_collector.collect(key, ret)
    else:
        return ret

def reducer(key, values):
    # key : string of intermediary key
    # load return dict correspondning to mapper ouput. they need to be loaded.
    # DEBUG
    #import glob, mapreduce
    #values = [mapreduce.OutputCollector(p) for p in glob.glob("/neurospin/brainomics/2013_adni/AD-CTL/results/*/0.1_0.0_0.0_1.0_-1.0/")]
    #values = [mapreduce.OutputCollector(p) for p in glob.glob("/home/ed203246/tmp/MCIc-MCInc_cs/results/*/0.1_0.0_0.0_1.0_-1.0/")]
    # values = [mapreduce.OutputCollector(p) for p in glob.glob("/home/ed203246/tmp/MCIc-CTL_cs/results/*/0.1_0.0_1.0_0.0_-1.0/")]
    # values = [mapreduce.OutputCollector(p) for p in glob.glob("/home/ed203246/tmp/MCIc-CTL_cs/results/*/0.1_0.0_0.5_0.5_-1.0/")]
    # Compute sd; ie.: compute results on each folds
    #values = [item.load() for item in values]
    print "key", key
    values = [item.load() for item in values[1:]]
    recall_mean_std = np.std([np.mean(precision_recall_fscore_support(
        item["y_true"].ravel(), item["y_pred"])[1]) for item in values]) / np.sqrt(len(values))
    y_true = np.concatenate([item["y_true"].ravel() for item in values])
    y_pred = np.concatenate([item["y_pred"].ravel() for item in values])
    y_post = np.concatenate([item["proba_pred"].ravel() for item in values])
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average=None)
    n_ite = None
    auc = roc_auc_score(y_true, y_post) #area under curve score.
    betas = np.hstack([item["beta"] for item in values]).T

    ## Compute beta similarity measures

    # Correlation
    R = np.corrcoef(betas)
    R = R[np.triu_indices_from(R, 1)]
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

#    R = np.corrcoef(betas)
#    beta_cor_mean = np.mean(R[np.triu_indices_from(R, 1)])
    a, l1, l2 , tv , k = key#[float(par) for par in key.split("_")]
    scores = OrderedDict()
    scores['a'] = a
    scores['l1'] = l1
    scores['l2'] = l2
    scores['tv'] = tv
    left = float(1-tv)
    if left == 0: left = 1.
    scores['l1l2_ratio'] = l1 / left
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
    scores['k'] = k
    scores['key'] = key
    return scores


if __name__ == "__main__":
    WD = "/neurospin/brainomics/2013_adni/MCIc-CTL-FS_s"
    #BASE = "/neurospin/tmp/brainomics/testenettv"
    #WD_CLUSTER = WD.replace("/neurospin/brainomics", "/neurospin/tmp/brainomics")
    #print "Sync data to %s/ " % os.path.dirname(WD)
    #os.system('rsync -azvu %s %s/' % (BASE, os.path.dirname(WD)))
    INPUT_DATA_X = 'X.npy'
    INPUT_DATA_y = 'y.npy'
    STRUCTURE = dict(mesh="lrh.pial.gii", mask="mask.npy")
    NFOLDS = 5
    #WD = os.path.join(WD, 'logistictvenet_5cv')
    if not os.path.exists(WD): os.makedirs(WD)
    os.chdir(WD)

    #############################################################################
    ## Create config file
    y = np.load(INPUT_DATA_y)
    if os.path.exists("config_5cv.json"):
        inf = open("config_5cv.json", "r")
        old_conf = json.load(inf)
        cv = old_conf["resample"]
        inf.close()
    else:
        cv = [[tr.tolist(), te.tolist()] for tr,te in StratifiedKFold(y.ravel(), n_folds=5)]
    if cv[0] is not None: # Make sure first fold is None
        cv.insert(0, None)
    #cv = [[tr.tolist(), te.tolist()] for tr,te in StratifiedKFold(y.ravel(), n_folds=5)]
    #cv.insert(0, None)  # first fold is None
    # parameters grid
    # Re-run with
    tv_range = np.array([0., 0.01, 0.05, 0.1, .2, .3, .4, .5 , .6, .7, .8, .9])
    l1l2_ratios = np.array([[1., 0., 1], [0., 1., 1], [.5, .5, 1], [.9, .1, 1],
                       [.1, .9, 1], [.01, .99, 1]])
    alphas = [.01, .05, .1]
    k_range = [-1]
    l1l2tv =[np.array([[float(1-tv), float(1-tv), tv]]) * l1l2_ratios for tv in tv_range]
    l1l2tv.append(np.array([[0., 0., 1.]]))
    l1l2tv = np.concatenate(l1l2tv)
    alphal1l2tv = np.concatenate([np.c_[np.array([[alpha]]*l1l2tv.shape[0]), l1l2tv] for alpha in alphas])
    alphal1l2tvk = np.concatenate([np.c_[alphal1l2tv, np.array([[k]]*alphal1l2tv.shape[0])] for k in k_range])
    params = [params.tolist() for params in alphal1l2tvk]
    # User map/reduce function file:
#    try:
#        user_func_filename = os.path.abspath(__file__)
#    except:
    user_func_filename = os.path.join(os.environ["HOME"],
        "git", "scripts", "2013_adni", "MCIc-CTL-FS",
        "03_tvenet_cs.py")
    #print __file__, os.path.abspath(__file__)
    print "user_func", user_func_filename
    #import sys
    #sys.exit(0)
    # Use relative path from config.json
    config = dict(data=dict(X=INPUT_DATA_X, y=INPUT_DATA_y),
                  params=params, resample=cv,
                  structure=STRUCTURE,
                  map_output="5cv",
                  penalty_start = 2,
                  user_func=user_func_filename,
                  #reduce_input="5cv/*/*",
                  reduce_group_by='params',
                  reduce_output="MCIc-CTL-FS_s.csv")
    json.dump(config, open(os.path.join(WD, "config_5cv.json"), "w"))

    #############################################################################
    # Build utils files: sync (push/pull) and PBS
    import brainomics.cluster_gabriel as clust_utils
    sync_push_filename, sync_pull_filename, WD_CLUSTER = \
        clust_utils.gabriel_make_sync_data_files(WD)
    cmd = "mapreduce.py --map  %s/config_5cv.json" % WD_CLUSTER
    clust_utils.gabriel_make_qsub_job_files(WD, cmd)
    #############################################################################
    # Sync to cluster
    print "Sync data to gabriel.intra.cea.fr: "
    os.system(sync_push_filename)
    #############################################################################
    print "# Start by running Locally with 2 cores, to check that everything os OK)"
    print "Interrupt after a while CTL-C"
    print "mapreduce.py --map %s/config_5cv.json --ncore 2" % WD
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
    print "mapreduce.py --reduce %s/config_5cv.json" % WD

###############################################################################
def plot_perf():
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    
    # SOME ERROR WERE HERE CORRECTED 27/04/2014 think its good
    #INPUT_vbm = "/home/ed203246/mega/data/2015_logistic_nestv/adni/MCIc-CTL/MCIc-CTL_cs.csv"
    INPUT = "/neurospin/brainomics/2013_adni/MCIc-CTL-FS_s/MCIc-CTL-FS_s.csv"
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
    config_filenane = "/neurospin/brainomics/2013_adni/MCIc-CTL-FS_s/config_5cv.json"
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
        #k = "l2"
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
    delta = summary.ix[summary.algo == "l1sl2tv", cols_diff_in].as_matrix() - summary.ix[summary.algo == "l1sl2", cols_diff_in].as_matrix()
    summary.ix[summary.algo == "l1sl2tv", cols_diff] = delta
    xlsx = pd.ExcelWriter(config['reduce_output'].replace("csv" , "xlsx"))
    orig_cv.to_excel(xlsx, 'All')
    summary.to_excel(xlsx, 'Summary')
    xlsx.save()