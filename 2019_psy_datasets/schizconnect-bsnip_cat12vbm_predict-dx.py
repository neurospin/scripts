"""
# Copy data

cd /home/ed203246/data/psy_sbox/analyses/201906_schizconnect-vip-prague-bsnip-biodb-icaar-start_assemble-all/data/cat12vbm
rsync -azvu triscotte.intra.cea.fr:/neurospin/psy_sbox/analyses/201906_schizconnect-vip-prague-bsnip-biodb-icaar-start_assemble-all/data/cat12vbm/*participants*.csv ./
rsync -azvu triscotte.intra.cea.fr:/neurospin/psy_sbox/analyses/201906_schizconnect-vip-prague-bsnip-biodb-icaar-start_assemble-all/data/cat12vbm/*t1mri_mwp1_mask.nii.gz ./
rsync -azvu triscotte.intra.cea.fr:/neurospin/psy_sbox/analyses/201906_schizconnect-vip-prague-bsnip-biodb-icaar-start_assemble-all/data/cat12vbm/*mwp1_gs-raw_data64.npy ./

# NS => Laptop
rsync -azvun triscotte.intra.cea.fr:/neurospin/psy_sbox/analyses/202004_schizconnect-bsnip_cat12vbm_predict-dx/* /home/ed203246/data/psy_sbox/analyses/202004_schizconnect-bsnip_cat12vbm_predict-dx/
"""
# %load_ext autoreload
# %autoreload 2

import os, sys
import numpy as np
import glob
import pandas as pd
import nibabel
import brainomics.image_preprocessing as preproc
from brainomics.image_statistics import univ_stats, plot_univ_stats, residualize, ml_predictions
import shutil
# import mulm
# import sklearn
# import re
# from nilearn import plotting
import nilearn.image
import matplotlib
if not hasattr(sys, 'ps1'): # if not interactive use pdf backend
    matplotlib.use('pdf')
import matplotlib.pyplot as plt
import re
# import glob
import seaborn as sns
import copy
import pickle
import time

# ML
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import sklearn.linear_model as lm
import sklearn.ensemble as ensemble
from sklearn.model_selection import StratifiedKFold, KFold, LeaveOneOut
from nitk.stats import Residualizer
from nitk.utils import dict_product, parallel, reduce_cv_classif
from nitk.utils import maps_similarity, arr_threshold_from_norm2_ratio

INPUT_PATH = '/neurospin/psy_sbox/analyses/201906_schizconnect-vip-prague-bsnip-biodb-icaar-start_assemble-all/data/cat12vbm/'
OUTPUT_PATH = '/neurospin/psy_sbox/analyses/202004_schizconnect-bsnip_cat12vbm_predict-dx'
NJOBS = 8

# On laptop
if not os.path.exists(INPUT_PATH):
    INPUT_PATH = INPUT_PATH.replace('/neurospin', '/home/ed203246/data')
    OUTPUT_PATH = OUTPUT_PATH.replace('/neurospin', '/home/ed203246/data' )
    NJOBS = 2

os.makedirs(OUTPUT_PATH, exist_ok=True)

scaling, harmo = 'gs', 'raw'
DATASET_FULL = 'schizconnect-vip-bsnip'
DATASET_TRAIN = 'schizconnect-vip'
target, target_num = "diagnosis", "diagnosis_num"
NSPLITS = 5

################################################################################
#
# Utils
#
################################################################################

def PATH(dataset, modality='t1mri', mri_preproc='mwp1', scaling=None, harmo=None,
    type=None, ext=None, basepath=""):
    # scaling: global scaling? in "raw", "gs"
    # harmo (harmonization): in [raw, ctrsite, ressite, adjsite]
    # type data64, or data32

    return os.path.join(basepath, dataset + "_" + modality+ "_" + mri_preproc +
                 ("" if scaling is None else "_" + scaling) +
                 ("" if harmo is None else "-" + harmo) +
                 ("" if type is None else "_" + type) + "." + ext)

def INPUT(*args, **kwargs):
    return PATH(*args, **kwargs, basepath=INPUT_PATH)

def OUTPUT(*args, **kwargs):
    return PATH(*args, **kwargs, basepath=OUTPUT_PATH)

def fit_predict(key, estimator_img, residualize, split):
    print(key)
    start_time = time.time()
    train, test = split
    Xim_train, Xim_test, Xdemoclin_train, Xdemoclin_test, Z_train, Z_test, y_train =\
    Xim[train, :], Xim[test, :], Xdemoclin[train, :], Xdemoclin[test, :], Z[train, :], Z[test, :], y[train]

    # Images based predictor

    # Residualization
    if residualize:
        # residualizer.fit(Xim_, Z_) biased residualization
        residualizer.fit(Xim_train, Z_train)
        Xim_train = residualizer.transform(Xim_train, Z_train)
        Xim_test = residualizer.transform(Xim_test, Z_test)

    scaler = StandardScaler()
    Xim_train = scaler.fit_transform(Xim_train)
    Xim_test = scaler.transform(Xim_test)
    try: # if coeficient can be retrieved given the key
        estimator_img.coef_ = KEY_VALS[key]['coef_img']
    except: # if not fit
        estimator_img.fit(Xim_train, y_train)

    y_test_img = estimator_img.predict(Xim_test)
    score_test_img = estimator_img.decision_function(Xim_test)
    score_train_img = estimator_img.decision_function(Xim_train)
    try:
        coef_img = estimator_img.coef_
    except:
        coef_img = None
    time_elapsed = round(time.time() - start_time, 2)

    return dict(y_test_img=y_test_img, score_test_img=score_test_img, time=time_elapsed,
                coef_img=coef_img)

"""
# Wrap user define CV to new sklearn CV (Leave out study CV)
from sklearn.model_selection import BaseCrossValidator
class CVIterableWrapper(BaseCrossValidator):
    "Wrapper class for old style cv objects and iterables."
    def __init__(self, cv):
        self.cv = list(cv)

    def get_n_splits(self, X=None, y=None, groups=None):
        return len(self.cv)

    def split(self, X=None, y=None, groups=None):
        for train, test in self.cv:
            yield train, test

# [(fold, train, test) for fold, (train, test) in enumerate(cv.split(X, y))]
# for fold, (train, test) in enumerate(cv.split(X, y)): print(fold, (train, test))
def scores_train_test(estimator, X_tr, X_te, y_tr, y_te):
    from sklearn import metrics
    y_pred_tr, y_pred_te = estimator.predict(X_tr), estimator.predict(X_te)
    return [metrics.accuracy_score(y_tr, y_pred_tr), metrics.accuracy_score(y_te, y_pred_te)]
"""


if not os.path.exists(OUTPUT(dataset=DATASET_FULL, scaling=scaling, harmo=harmo, type="data64", ext="npy")):

    print("""
    ################################################################################
    #
    # Dataset: concatenate [schizconnect-vip  bsnip]
    #
    ################################################################################
    """)
    # SCZ (schizconnect-vip <=> bsnip)

    datasets = ['schizconnect-vip', 'bsnip']

    # Read clinical data
    pop = pd.concat([pd.read_csv(INPUT(dataset=DATASET_FULL, scaling=None, harmo=None, type="participants", ext="csv")) for dataset in datasets], axis=0)
    mask_row = pop[target].isin(['schizophrenia', 'FEP', 'control'])
    pop = pop[mask_row]
    # FEP of PRAGUE becomes 1
    pop[target_num] = pop[target].map({'schizophrenia': 1, 'FEP':1, 'control': 0}).values

    pop["GM_frac"] = pop.gm / pop.tiv
    pop["sex_c"] = pop["sex"].map({0: "M", 1: "F"})

    # Leave study out CV
    studies = np.sort(pop["study"].unique())
    # array(['BIOBD', 'BSNIP', 'PRAGUE', 'SCHIZCONNECT-VIP'], dtype=object)

    # Check all sites have both labels
    # print([[studies[i], np.unique(pop[target].values[te]), np.unique(pop[target_num].values[te])] for i, (tr, te) in
    #    enumerate(cv_lstudieout.split(None, pop[target_num].values))])

    # Load arrays
    imgs_arr = np.concatenate([np.load(INPUT(dataset=DATASET_FULL, scaling=scaling, harmo=harmo, type="data64", ext="npy"), mmap_mode='r') for dataset in datasets])[mask_row]
    # Save and relaod in mm
    np.save(OUTPUT(DATASET_FULL, scaling=scaling, harmo=harmo, type="data64", ext="npy"), imgs_arr)
    del imgs_arr
    import gc
    gc.collect()

    # reload and do QC
    imgs_arr = np.load(OUTPUT(DATASET_FULL, scaling=scaling, harmo=harmo, type="data64", ext="npy"), mmap_mode='r')
    print(imgs_arr.shape)

    # Recompute mask
    from nitk.image import compute_brain_mask
    ref_img = nibabel.load(INPUT(datasets[0], scaling=None, harmo=None, type="mask", ext="nii.gz"))
    mask_img = compute_brain_mask(imgs_arr, target_img=ref_img)
    mask_img.to_filename(OUTPUT(DATASET_FULL, scaling=None, harmo=None, type="mask", ext="nii.gz"))
    pop.to_csv(OUTPUT(DATASET_FULL, scaling=None, harmo=None, type="participants", ext="csv"), index=False)

    mask_arr = mask_img.get_data() != 0
    assert np.sum(mask_arr != 0) == 367689
    print(mask_arr.shape, imgs_arr.squeeze().shape)
    print("Sizes. mask_arr:%.2fGb" % (imgs_arr.nbytes / 1e9))


if not os.path.exists(OUTPUT(DATASET_FULL, scaling=scaling, harmo=harmo, type="residualization-l2", ext="xlsx")):
    print("""
    ###############################################################################
    #
    # Residualization study on schizconnect-vip-bsnip
    #
    ###############################################################################
    """)

    pop = pd.read_csv(OUTPUT(DATASET_FULL, scaling=None, harmo=None, type="participants", ext="csv"))

    # Working population df with no NAs
    pop_w = pop.copy()
    assert np.all(pop_w[["sex", "age", "site"]].isnull().sum()  == 0)

    imgs_arr = np.load(OUTPUT(DATASET_FULL, scaling=scaling, harmo=harmo, type="data64", ext="npy"), mmap_mode='r')
    mask_img = nibabel.load(OUTPUT(DATASET_FULL, scaling=None, harmo=None, type="mask", ext="nii.gz"))
    mask_arr = mask_img.get_data() != 0
    assert mask_arr.sum() == 367689

    print("""
    #==============================================================================
    # Select SCHIZCONNECT-VIP+BSNIP : 5CV(SCHIZCONNECT-VIP) + LSO(SCHIZCONNECT-VIP+BSNIP)
    """)


    msk = pop.study.isin(['SCHIZCONNECT-VIP', 'BSNIP'])
    assert msk.sum() == 999
    Xim = imgs_arr.squeeze()[:, mask_arr][msk]
    del imgs_arr
    y = pop[target + "_num"][msk].values
    print("Sizes. mask_arr:%.2fGb" % (Xim.nbytes / 1e9))

    vars_clinic = []
    vars_demo = ['age', 'sex']
    Xdemoclin = pop.loc[msk, vars_demo + vars_clinic].values

    # -----------------------------------------------------------------------------
    # Residualization bloc: Sex + Sites + age with some descriptives stats

    pop_ = pop_w[msk]
    print([[s, np.sum(pop_.sex == s)] for s in pop_.sex.unique()])
    # [[0.0, 592], [1.0, 407]]
    desc_stats = pd.DataFrame([pop_[pop_.site == s].study.unique().tolist() +
                        [s, np.sum(pop_.site == s),
                         round(pop_[pop_.site == s][target_num].mean(), 2),
                         round(pop_[pop_.site == s]["age"].mean(), 2),
                         round(pop_[pop_.site == s]["sex"].mean(), 2)] for s in pop_.site.unique()],
    columns=['study', "site", 'count', 'DX%', 'age_mean', "sex%F"])
    print(desc_stats)
    """
                  study       site  count   DX%  age_mean  sex%F
    0  SCHIZCONNECT-VIP        MRN    164  0.47     37.84   0.23
    1  SCHIZCONNECT-VIP      WUSTL    269  0.43     30.61   0.45
    2  SCHIZCONNECT-VIP        vip     92  0.42     34.38   0.45
    3  SCHIZCONNECT-VIP         NU     80  0.53     32.05   0.42
    4             BSNIP     Boston     51  0.49     32.65   0.43
    5             BSNIP     Dallas     66  0.33     39.98   0.52
    6             BSNIP   Hartford    109  0.52     33.14   0.43
    7             BSNIP  Baltimore    141  0.60     39.89   0.41
    8             BSNIP    Detroit     27  0.22     28.44   0.52
    """

    formula_res, formula_full = "site + age + sex", "site + age + sex + " + target_num
    residualizer = Residualizer(data=pop_w[msk], formula_res=formula_res, formula_full=formula_full)
    Z = residualizer.get_design_mat()

    assert Xim.shape[0] == Z.shape[0] == y.shape[0]

    # -----------------------------------------------------------------------------
    # CV: 5CV(SCHIZCONNECT-VIP) + LSO(SCHIZCONNECT-VIP+BSNIP)

    pop_ = pop_w[msk]
    pop_ = pop_.reset_index(drop=True)

    # ~~~~~~~
    # CV LSO

    cv_lso_dict = {s:[np.where(pop_.site != s)[0], np.where(pop_.site == s)[0]] for s in pop_.site.unique()}

    # QC all test stem from single left-out site
    for k, fold in cv_lso_dict.items():
        sites_ =  pop_.site[fold[1]].unique()
        assert len(sites_) == 1 and sites_[0] == k

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 5cv on SCHIZCONNECT-VIP that will be applied on SCHIZCONNECT-VIP+bsnip

    # store idx of the large dataset, cv in the smaller, map back using stored idx
    df_ = pop_[["participant_id", target_num]]
    df_["idx"] = np.arange(len(df_)) # store idx of large dataset
    df_ = df_[pop_.study.isin(["SCHIZCONNECT-VIP"])] # select smaller
    cv_ = StratifiedKFold(n_splits=5, shuffle=True, random_state=3)

    # do CV on the smaller but use index of the larger
    cv5_schizconnect = {"CV%i" % fold : [df_.idx[train].values, df_.idx[test].values] for fold, (train, test) in enumerate(cv_.split(df_[target_num].values, df_[target_num].values))}

    # Check all split cover all SCHIZCONNECT-VIP sample
    assert np.all(np.array([len(np.unique(train.tolist() + test.tolist()))  for fold, (train, test) in cv5_schizconnect.items()]) == pop_.study.isin(["SCHIZCONNECT-VIP"]).sum())
    # Check all test cover all SCHIZCONNECT-VIP sample
    assert np.all(len(np.unique([test.tolist()  for fold, (train, test) in cv5_schizconnect.items()])) == pop_.study.isin(["SCHIZCONNECT-VIP"]).sum())
    del df_, cv_

    print("""
    #==============================================================================
    # Run l2 5CV(SCHIZCONNECT-VIP) + LSO(SCHIZCONNECT-VIP+BSNIP)
    """)

    estimators_dict = {"l2_C:%f" % 1: lm.LogisticRegression(C=1, class_weight='balanced', fit_intercept=False)}

    # LSO
    args_collection = dict_product(estimators_dict, dict(noresidualize=False, residualize=True), cv_lso_dict)
    key_vals_lso = parallel(fit_predict, args_collection, n_jobs=NJOBS, pass_key=True, verbose=20)
    cv_scores_lso = reduce_cv_classif(key_vals_lso, cv_lso_dict, y_true=y)
    cv_scores_lso["CV"] = 'LSO(SCHIZCONNECT-VIP+BSNIP)'

    # 5CV

    args_collection = dict_product(estimators_dict, dict(noresidualize=False, residualize=True), cv5_schizconnect)
    key_vals_cv5_schizconnect = parallel(fit_predict, args_collection, n_jobs=NJOBS, pass_key=True, verbose=20)
    cv_scores_cv5_schizconnect = reduce_cv_classif(key_vals_cv5_schizconnect, cv5_schizconnect, y_true=y)
    cv_scores_cv5_schizconnect["CV"] = '5CV(SCHIZCONNECT-VIP)'

    cv_scores = cv_scores_lso.append(cv_scores_cv5_schizconnect)

    # =>
    #
    mean = cv_scores.groupby(["CV", "param_1"]).mean()
    sd = cv_scores.groupby(["CV", "param_1"]).std()
    sd = sd[["auc", "bacc"]].rename(columns={'auc':'auc_std', 'bacc':'bacc_std'})
    stat = pd.concat([mean, sd], axis=1)

    xls_filename = OUTPUT(DATASET_FULL, scaling=scaling, harmo=harmo, type="residualization-l2", ext="xlsx")
    with pd.ExcelWriter(xls_filename) as writer:
        cv_scores.to_excel(writer, sheet_name='folds', index=False)
        stat.to_excel(writer, sheet_name='mean')
        desc_stats.to_excel(writer, sheet_name='desc_stats', index=False)

    del cv_scores, cv_scores_lso

if not os.path.exists(OUTPUT(DATASET_TRAIN, scaling=scaling, harmo=harmo, type="models-5cv-l1-l2-enet-filter-rfe", ext="xlsx")) or\
   not os.path.exists(OUTPUT(DATASET_TRAIN, scaling=scaling, harmo=harmo, type="models-5cv-enettv", ext="xlsx")):

    print("""
    ###############################################################################
    #
    # Comparison analysis and Sensitivity study on schizconnect-vip 5CV
    #
    ###############################################################################
    """)

    pop = pd.read_csv(OUTPUT(DATASET_FULL, scaling=None, harmo=None, type="participants", ext="csv"))
    imgs_arr = np.load(OUTPUT(DATASET_FULL, scaling=scaling, harmo=harmo, type="data64", ext="npy"), mmap_mode='r')
    mask_img = nibabel.load(OUTPUT(DATASET_FULL, scaling=None, harmo=None, type="mask", ext="nii.gz"))
    mask_arr = mask_img.get_data() != 0
    assert mask_arr.sum() == 367689

    print("""
    #==============================================================================
    # Select dataset 5CV on SCHIZCONNECT-VIP
    """)

    msk = pop.study.isin(['SCHIZCONNECT-VIP'])
    assert msk.sum() == 605
    Xim = imgs_arr.squeeze()[:, mask_arr][msk]
    del imgs_arr
    y = pop[target + "_num"][msk].values
    print("Sizes. mask_arr:%.2fGb" % (Xim.nbytes / 1e9))
    #Sizes. mask_arr:1.78Gb

    Xdemoclin = Z = np.zeros((Xim.shape[0], 1))

    cv = StratifiedKFold(n_splits=NSPLITS, shuffle=True, random_state=3)
    cv_dict = {"CV%i" % fold:split for fold, split in enumerate(cv.split(Xim, y))}
    cv_dict["ALL"] = [np.arange(Xim.shape[0]), np.arange(Xim.shape[0])]

    print([[lab, np.sum(y == lab)] for lab in np.unique(y)])
    #  [[0, 330], [1, 275]]


if not os.path.exists(OUTPUT(DATASET_TRAIN, scaling=scaling, harmo=harmo, type="models-5cv-l1-l2-enet-filter-rfe", ext="xlsx")):
    print("""
    #==============================================================================
    # l1, l2, enet, filter, rfe
    """)

    # parameters range:
    # from sklearn.svm import l1_min_c
    # Cmin = l1_min_c(StandardScaler().fit_transform(Xim), y, loss='log')
    # Cs = Cmin * np.logspace(0, -5, 10)

    Cs = np.logspace(14, -14, 20)
    l2 = {"l2_C:%.16f" % C: lm.LogisticRegression(C=C, class_weight='balanced', fit_intercept=False) for C in Cs}
    l1 = {"l1_C:%.16f" % C: lm.LogisticRegression(C=C, penalty='l1', class_weight='balanced', fit_intercept=False) for C in Cs}
    enet = {"enet_C:%.16f" % C: lm.LogisticRegression(C=C, penalty='elasticnet', class_weight='balanced', l1_ratio=.1, fit_intercept=False, solver='saga') for C in Cs}
    assert len(Cs) == len(l2) == len(l1) == len(enet)

    from sklearn.pipeline import make_pipeline, Pipeline
    from sklearn.feature_selection import SelectKBest, f_regression, RFE
    Ks = np.logspace(1, np.log10(Xim.shape[1]), 20).astype(int)
    fl2 = {"fl2_k:%i" % k:make_pipeline(SelectKBest(k=k), lm.LogisticRegression(C=1, class_weight='balanced', fit_intercept=False))
            for k in Ks}

    rfe = {"rfel2_k:%i" % k:RFE(estimator=lm.LogisticRegression(C=1, class_weight='balanced', fit_intercept=False), n_features_to_select=k, step=.05)
            for k in Ks}

    estimators_dict = dict()
    estimators_dict.update(l1)
    estimators_dict.update(l2)
    estimators_dict.update(enet)
    estimators_dict.update(fl2)
    estimators_dict.update(rfe)

    args_collection = dict_product(estimators_dict, dict(noresdualize=False), cv_dict)
    print("Nb Tasks=%i" % len(args_collection))

    key_vals = parallel(fit_predict, args_collection, n_jobs=NJOBS, pass_key=True, verbose=20)

    models_filename = OUTPUT(DATASET_TRAIN, scaling=scaling, harmo=harmo, type="models-5cv-l1-l2-enet-filter-rfe", ext="pkl")
    with open(models_filename, 'wb') as fd:
        pickle.dump(key_vals, fd)

    cv_scores = reduce_cv_classif(key_vals, cv_dict, y_true=y)

    xls_filename = OUTPUT(DATASET_TRAIN, scaling=scaling, harmo=harmo, type="models-5cv-l1-l2-enet-filter-rfe", ext="xlsx")
    with pd.ExcelWriter(xls_filename) as writer:
        cv_scores.to_excel(writer, sheet_name='folds', index=False)
        cv_scores.groupby(["param_0"]).mean().to_excel(writer, sheet_name='mean')

    print("""
    #------------------------------------------------------------------------------
    # plot
    """)

    xls_filename = OUTPUT(DATASET_TRAIN, scaling=scaling, harmo=harmo, type="models-5cv-l1-l2-enet-filter-rfe", ext="xlsx")
    cv_scores = pd.read_excel(xls_filename)
    cv_scores["param"] = [float(s.split(":")[1]) for s in cv_scores["param_0"]]
    cv_scores["algo"] = [s.split("_")[0] for s in cv_scores["param_0"]]

    sns.set_style("whitegrid")
    import matplotlib.pylab as pl
    from matplotlib import rc
    rc('text', usetex=True)
    rc('font', size=14)
    rc('legend', fontsize=14)
    rc('text.latex', preamble=r'\usepackage{lmodern}')
    plt.rcParams["font.family"] = ["Latin Modern Roman"]
    # -------------

    # seach l2 C closre to 1
    x_ = cv_scores[cv_scores.algo == 'l2']["param"].unique()
    C1_almost = x_[np.argmin(np.abs(x_ - 1))]
    baseline_l2C1_auc = cv_scores[(cv_scores.algo == 'l2') & (cv_scores.param == C1_almost)]["auc"]
    baseline_l2C1_bacc = cv_scores[(cv_scores.algo == 'l2') & (cv_scores.param == C1_almost)]["bacc"]

    cv_scores["Model"] = cv_scores.algo.map({'l1':r'$\ell_1$', 'l2':r'$\ell_2$', "enet":r'$\ell_1\ell_2$', 'rfel2':'RFE+$\ell_2$',  'fl2':'Filter+$\ell_2$'})

    # filter and RFE + l2

    # AUC
    df_ = cv_scores[cv_scores["algo"].isin(["fl2", "rfel2"])]
    fig = plt.figure(figsize=(7.25, 5), dpi=300)
    g = sns.lineplot(x="param", y="auc", hue="Model", data=df_)#, palette=palette)
    g.set(xscale="log")
    plt.xlabel(xlabel=r'k', fontsize=20)
    plt.ylabel(ylabel=r'AUC', fontsize=16)
    g.axes.axhline(baseline_l2C1_auc.mean(), ls='--', color='black', linewidth=1, alpha=0.5)  # no feature selection performance
    g.set(ylim=(.45, .9))
    plt.tight_layout()
    plt.savefig(OUTPUT(DATASET_TRAIN, scaling=None, harmo=None, type="sensibility-filter-rfe_auc", ext="pdf"))

    # bACC
    df_ = cv_scores[cv_scores["algo"].isin(["fl2", "rfel2"])]
    fig = plt.figure(figsize=(7.25, 5), dpi=300)
    g = sns.lineplot(x="param", y="bacc", hue="Model", data=df_)#, palette=palette)
    g.set(xscale="log")
    plt.xlabel(xlabel=r'k', fontsize=20)
    plt.ylabel(ylabel=r'bACC', fontsize=16)
    g.axes.axhline(baseline_l2C1_bacc.mean(), ls='--', color='black', linewidth=1, alpha=0.5)  # no feature selection performance
    g.set(ylim=(.45, .9))
    plt.tight_layout()
    plt.savefig(OUTPUT(DATASET_TRAIN, scaling=None, harmo=None, type="sensibility-filter-rfe_bacc", ext="pdf"))

    # l1, enet

    # AUC
    df_ = cv_scores[cv_scores["algo"].isin(["l2", "l1", "enet"])]
    fig = plt.figure(figsize=(7.25, 5), dpi=300)
    g = sns.lineplot(x="param", y="auc", hue="Model", data=df_)#, palette=palette)
    g.set(xscale="log")
    plt.xlabel(xlabel=r'C', fontsize=20)
    plt.ylabel(ylabel=r'AUC', fontsize=16)
    g.axes.axhline(baseline_l2C1_auc.mean(), ls='--', color='black', linewidth=1, alpha=0.5)  # no feature selection performance
    g.set(ylim=(.45, .9))
    g.set(xlim=(1e-4, 1e4))
    plt.tight_layout()
    plt.savefig(OUTPUT(DATASET_TRAIN, scaling=None, harmo=None, type="sensibility-l2-l1-enet_auc", ext="pdf"))

    # bACC
    df_ = cv_scores[cv_scores["algo"].isin(["l2", "l1", "enet"])]
    fig = plt.figure(figsize=(7.25, 5), dpi=300)
    g = sns.lineplot(x="param", y="bacc", hue="Model", data=df_)#, palette=palette)
    g.set(xscale="log")
    plt.xlabel(xlabel=r'C', fontsize=20)
    plt.ylabel(ylabel=r'bACC', fontsize=16)
    g.axes.axhline(baseline_l2C1_bacc.mean(), ls='--', color='black', linewidth=1, alpha=0.5)  # no feature selection performance
    g.set(ylim=(.45, .9))
    g.set(xlim=(1e-4, 1e4))
    plt.tight_layout()
    plt.savefig(OUTPUT(DATASET_TRAIN, scaling=None, harmo=None, type="sensibility-l2-l1-enet_bacc", ext="pdf"))


if not os.path.exists(OUTPUT(DATASET_TRAIN, scaling=scaling, harmo=harmo, type="models-5cv-enettv", ext="xlsx")):

    print("""
    #==============================================================================
    # Enet-TV
    """)

    print("# Enet-TV")
    # estimators
    import parsimony.algorithms as algorithms
    import parsimony.estimators as estimators
    import parsimony.functions.nesterov.tv as nesterov_tv
    from parsimony.utils.linalgs import LinearOperatorNesterov

    if not os.path.exists(OUTPUT(DATASET_FULL, scaling=None, harmo=None, type="Atv", ext="npz")):
        Atv = nesterov_tv.linear_operator_from_mask(mask_img.get_fdata(), calc_lambda_max=True)
        Atv.save(OUTPUT(DATASET_FULL, scaling=None, harmo=None, type="Atv", ext="npz"))

    Atv = LinearOperatorNesterov(filename=OUTPUT(DATASET_FULL, scaling=None, harmo=None, type="Atv", ext="npz"))
    assert np.allclose(Atv.get_singular_values(0), 11.94026967367116)

    def ratios_to_param(alpha, l1l2ratio, tvcoef):
        tv = alpha * tvcoef
        l1 = alpha * l1l2ratio
        l2 = alpha * 1
        return l1, l2, tv

    # Large range
    alphas = [.01, .1, 1.]
    l1l2ratios = [0, 0.01, 0.1]
    tvcoefs = [0, 0.0001, 0.001, 0.01, 0.1, 1]

    # Smaller range
    # alphas = [.1]
    # l1l2ratios = [.1]
    # tvcoefs = [0.001, 0.01, 0.1, 1]

    import itertools
    estimators_dict = dict()
    for alpha, l1l2ratio, tvcoef in itertools.product(alphas, l1l2ratios, tvcoefs):
        # print(alpha, l1l2ratio, tvcoef)
        l1, l2, tv = ratios_to_param(alpha, l1l2ratio, tvcoef)
        key = "enettv_%.3f:%.6f:%.6f" % (alpha, l1l2ratio, tvcoef)

        conesta = algorithms.proximal.CONESTA(max_iter=1)#10000)
        estimator = estimators.LogisticRegressionL1L2TV(l1, l2, tv, Atv, algorithm=conesta,
                                                class_weight="auto", penalty_start=0)
        estimators_dict[key] = estimator

    args_collection = dict_product(estimators_dict, dict(noresidualize=False), cv_dict)

    models_filename = OUTPUT(DATASET_TRAIN, scaling=scaling, harmo=harmo, type="models-5cv-enettv", ext="pkl")
    if os.path.exists(models_filename):
        with open(models_filename, 'rb') as fd:
            KEY_VALS = pickle.load(fd)
        #key = list(args_collection)[10]
        #list(KEY_VALS)
    key_vals = parallel(fit_predict, args_collection, n_jobs=NJOBS, pass_key=True, verbose=20)

    with open(models_filename, 'wb') as fd:
        pickle.dump(key_vals, fd)

    cv_scores = reduce_cv_classif(key_vals, cv_dict, y_true=y)

    xls_filename = OUTPUT(DATASET_TRAIN, scaling=scaling, harmo=harmo, type="models-5cv-enettv", ext="xlsx")
    with pd.ExcelWriter(xls_filename) as writer:
        cv_scores.to_excel(writer, sheet_name='folds', index=False)
        cv_scores.groupby(["param_0"]).mean().to_excel(writer, sheet_name='mean')

    # [Parallel(n_jobs=8)]: Done 180 out of 180 | elapsed: 4900.6min finished

    print("""
    #------------------------------------------------------------------------------
    # maps' similarity measures accross CV-folds
    """)
    models_filename = OUTPUT(DATASET_TRAIN, scaling=scaling, harmo=harmo, type="models-5cv-enettv", ext="pkl")
    with open(models_filename, 'rb') as fd:
        key_vals = pickle.load(fd)

    # Agregate maps by key[0]
    by_param = {k:[] for k in set([k[0] for k in key_vals])}
    for k, v in key_vals.items():
        by_param[k[0]].append(v['coef_img'].ravel())

    maps_similarity_l = list()
    # Compute similarity measures
    for k, v in by_param.items():
        maps = np.array(v)
        maps_t = np.vstack([arr_threshold_from_norm2_ratio(maps[i, :], .99)[0] for i in range(maps.shape[0])])
        r_bar, dice_bar, fleiss_kappa_stat = maps_similarity(maps_t)
        prop_non_zeros_mean = np.count_nonzero(maps_t) / np.prod(maps_t.shape)
        maps_similarity_l.append([k, prop_non_zeros_mean, r_bar, dice_bar, fleiss_kappa_stat])

    map_sim = pd.DataFrame(maps_similarity_l, columns=['param_0', 'prop_non_zeros_mean', 'r_bar', 'dice_bar', 'fleiss_kappa_stat'])

    # Update excel file with similariy measures
    xls_filename = OUTPUT(DATASET_TRAIN, scaling=scaling, harmo=harmo, type="models-5cv-enettv", ext="xlsx")
    cv_scores = pd.read_excel(xls_filename, sheet_name='folds')
    pred_score_mean_ = cv_scores.groupby(["param_0"]).mean()
    pred_score_mean_ = pred_score_mean_.reset_index()
    pred_score_mean = pd.merge(pred_score_mean_, map_sim)
    assert pred_score_mean_.shape[0] == map_sim.shape[0] == pred_score_mean.shape[0]

    with pd.ExcelWriter(xls_filename) as writer:
        cv_scores.to_excel(writer, sheet_name='folds', index=False)
        pred_score_mean.to_excel(writer, sheet_name='mean')
    del pred_score_mean_

    print("""
    #------------------------------------------------------------------------------
    # plot
    """)

    xls_filename = OUTPUT(DATASET_TRAIN, scaling=scaling, harmo=harmo, type="models-5cv-enettv", ext="xlsx")
    cv_scores = pd.read_excel(xls_filename, sheet_name='folds')
    cv_scores = cv_scores.reset_index(drop=True)
    cv_scores_mean = pd.read_excel(xls_filename, sheet_name='mean')
    cv_scores_mean = cv_scores_mean.reset_index(drop=True)

    keys_ = pd.DataFrame([[s.split("_")[0]] + [float(v) for v in s.split("_")[1].split(":")] for s in cv_scores["param_0"]],
                 columns=["model", "alpha", "l1l2", "tv"])
    cv_scores = pd.concat([keys_, cv_scores], axis=1)
    keys_ = pd.DataFrame([[s.split("_")[0]] + [float(v) for v in s.split("_")[1].split(":")] for s in cv_scores_mean["param_0"]],
                 columns=["model", "alpha", "l1l2", "tv"])
    cv_scores_mean = pd.concat([keys_, cv_scores_mean], axis=1)
    del keys_

    from matplotlib.backends.backend_pdf import PdfPages
    sns.set_style("whitegrid")
    import matplotlib.pylab as pl
    from matplotlib import rc
    rc('text', usetex=True)
    rc('font', size=14)
    rc('legend', fontsize=14)
    rc('text.latex', preamble=r'\usepackage{lmodern}')
    plt.rcParams["font.family"] = ["Latin Modern Roman"]

    # cv_scores["alpha"].unique(): array([0.01, 0.1 , 1.  ])
    # cv_scores["l1l2"].unique(): array([0.  , 0.01, 0.1 ])
    # cv_scores["tv"].unique() array([0.001, 0.01 , 0.1  , 1.   ])

    pdf_filename = OUTPUT(DATASET_TRAIN, scaling=scaling, harmo=harmo, type="models-5cv-enettv", ext="pdf")
    with PdfPages(pdf_filename) as pdf:
        for l1l2 in cv_scores["l1l2"].unique():
            print(l1l2)
            df_ = cv_scores[cv_scores["l1l2"].isin([l1l2])]
            dfm_ = cv_scores_mean[cv_scores_mean["l1l2"].isin([l1l2])]
            df_["alpha"] = df_["alpha"].map({0.01:"1e-2'", 0.1:"1e-1'" , 1.:"1'"})
            dfm_["alpha"] = dfm_["alpha"].map({0.01:"1e-2'", 0.1:"1e-1'" , 1.:"1'"})
            df_.rename(columns={"alpha":"alpha", 'auc':'AUC', 'bacc':'bAcc'}, inplace=True)
            dfm_.rename(columns={'r_bar':'$r_w$', 'prop_non_zeros_mean':'non-null', 'dice_bar':'dice', 'fleiss_kappa_stat':'Fleiss-Kappa'}, inplace=True)

            fig, axs = plt.subplots(3, 2, figsize=(2 * 7.25, 3 * 5), dpi=300)
            g = sns.lineplot(x="tv", y='AUC', hue="alpha", data=df_, ax=axs[0, 0], palette="Blues"); g.set(xscale="log")
            g = sns.lineplot(x="tv", y='bAcc', hue="alpha", data=df_, ax=axs[0, 1], palette="Blues"); g.set(xscale="log")
            g = sns.lineplot(x="tv", y='$r_w$', hue="alpha", data=dfm_, ax=axs[1, 0], palette="Blues"); g.set(xscale="log")
            g = sns.lineplot(x="tv", y='non-null', hue="alpha", data=dfm_, ax=axs[1, 1], palette="Blues"); g.set(xscale="log")
            g = sns.lineplot(x="tv", y='dice', hue="alpha", data=dfm_, ax=axs[2, 0], palette="Blues"); g.set(xscale="log")
            g = sns.lineplot(x="tv", y='Fleiss-Kappa', hue="alpha", data=dfm_, ax=axs[2, 1], palette="Blues"); g.set(xscale="log")
            #plt.tight_layout()
            fig.suptitle('$\ell_1/\ell_2=%.3f$' % l1l2)
            #plt.savefig(OUTPUT(DATASET_TRAIN, scaling=None, harmo=None, type="sensibility-l2-l1-enet_auc", ext="pdf"))
            pdf.savefig()  # saves the current figure into a pdf page
            fig.clf()
            plt.close()
