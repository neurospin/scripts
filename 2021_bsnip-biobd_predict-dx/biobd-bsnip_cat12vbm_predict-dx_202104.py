# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

# Copy NS => Laptop
rsync -azvun --delete ed203246@is234606.intra.cea.fr:/neurospin/tmp/psy_sbox/analysis/202104_biobd-bsnip_cata12vbm_predict-dx /home/ed203246/data/psy_sbox/analyses/

"""


import os
import os.path
import numpy as np
import pandas as pd
import glob
import click
import copy
import time
import gc
import pickle
from shutil import copyfile, make_archive, unpack_archive, move
import subprocess

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
# Neuroimaging
import nibabel
#from nitk.image import img_to_array, global_scaling, compute_brain_mask, rm_small_clusters, img_plot_glass_brain
#from nitk.bids import get_keys
#from nitk.data import fetch_data

from nitk.utils import maps_similarity, arr_threshold_from_norm2_ratio, arr_clusters
from nitk.image import img_to_array, global_scaling, compute_brain_mask, rm_small_clusters, plot_glass_brains
#from nitk.stats import Residualizer
from nitk.mapreduce import dict_product, MapReduce, reduce_cv_classif
from mulm.residualizer import Residualizer

# sklearn for QC
import sklearn.linear_model as lm
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold, KFold, LeaveOneOut
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

# estimators
import parsimony.algorithms as algorithms
import parsimony.estimators as estimators
import parsimony.functions.nesterov.tv as nesterov_tv
from parsimony.utils.linalgs import LinearOperatorNesterov


###############################################################################
#
#%% Config: Input/Output
#
###############################################################################

STUDIES = ["biobd", "bsnip1"]
INPUT_DIR = "/neurospin/tmp/psy_sbox/all_studies/derivatives/arrays"

OUTPUT_DIR = "/neurospin/tmp/psy_sbox/analysis/202104_biobd-bsnip_cata12vbm_predict-dx"
OUTPUT = OUTPUT_DIR + "/{data}_{model}_{experience}_{type}.{ext}"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.chdir(OUTPUT_DIR)

DATASET, TARGET, TARGET_NUM = "-".join(STUDIES), "dx", "dx"
#DATASET_TRAIN = dataset
VAR_CLINIC  = []
VAR_DEMO = ['age', 'sex']
NSPLITS = 5
NBOOTS = 500

mask_filename = os.path.join(INPUT_DIR, "mni_cerebrum-gm-mask_1.5mm.nii.gz")
mask_img = nibabel.load(mask_filename)
mask_arr = mask_img.get_fdata() != 0
assert np.sum(mask_arr != 0) == 331695

###############################################################################
#
#%% 1) Utils
#
###############################################################################

###############################################################################
#%% 1.1) Mapper

def fit_predict(key, estimator_img, residualize, split, Xim, y, Zres, Xdemoclin, residualizer):
    estimator_img = copy.deepcopy(estimator_img)
    train, test = split
    print("fit_predict", Xim.shape, Xdemoclin.shape, Zres.shape, y.shape, np.max(train), np.max(test))
    Xim_train, Xim_test, Xdemoclin_train, Xdemoclin_test, Zres_train, Zres_test, y_train =\
        Xim[train, :], Xim[test, :], Xdemoclin[train, :], Xdemoclin[test, :], Zres[train, :], Zres[test, :], y[train]

    # Images based predictor
    # Residualization
    if residualize == 'yes':
        residualizer.fit(Xim_train, Zres_train)
        Xim_train = residualizer.transform(Xim_train, Zres_train)
        Xim_test = residualizer.transform(Xim_test, Zres_test)

    elif residualize == 'biased':
        residualizer.fit(Xim, Zres)
        Xim_train = residualizer.transform(Xim_train, Zres_train)
        Xim_test = residualizer.transform(Xim_test, Zres_test)

    elif residualize == 'no':
        pass

    scaler = StandardScaler()
    Xim_train = scaler.fit_transform(Xim_train)
    Xim_test = scaler.transform(Xim_test)

    # try: # if coeficient can be retrieved given the key
    #     estimator_img.coef_ = KEY_VALS[key]['coef_img']
    # except: # if not fit
    estimator_img.fit(Xim_train, y_train)

    y_test_img = estimator_img.predict(Xim_test)
    score_test_img = estimator_img.decision_function(Xim_test)
    score_train_img = estimator_img.decision_function(Xim_train)

    # Demographic/clinic based predictor
    estimator_democlin = lm.LogisticRegression(C=1, class_weight='balanced', fit_intercept=False)
    scaler = StandardScaler()
    Xdemoclin_train = scaler.fit_transform(Xdemoclin_train)
    Xdemoclin_test = scaler.transform(Xdemoclin_test)
    estimator_democlin.fit(Xdemoclin_train, y_train)
    y_test_democlin = estimator_democlin.predict(Xdemoclin_test)
    score_test_democlin = estimator_democlin.decision_function(Xdemoclin_test)
    score_train_democlin = estimator_democlin.decision_function(Xdemoclin_train)

    # STACK DEMO + IMG
    estimator_stck = lm.LogisticRegression(C=1, class_weight='balanced', fit_intercept=True)
    # SVC
    # from sklearn.svm import SVC
    # estimator_stck = SVC(kernel='rbf', probability=True, gamma=1 / 100)
    # GB
    # from sklearn.ensemble import GradientBoostingClassifier
    # estimator_stck = GradientBoostingClassifier()

    Xstck_train = np.c_[score_train_democlin, score_train_img]
    Xstck_test = np.c_[score_test_democlin, score_test_img]
    scaler = StandardScaler()
    Xstck_train = scaler.fit_transform(Xstck_train)
    Xstck_test = scaler.transform(Xstck_test)
    estimator_stck.fit(Xstck_train, y_train)

    y_test_stck = estimator_stck.predict(Xstck_test)
    score_test_stck = estimator_stck.predict_log_proba(Xstck_test)[:, 1]
    score_train_stck = estimator_stck.predict_log_proba(Xstck_train)[:, 1]

    return dict(y_test_img=y_test_img, score_test_img=score_test_img,
                y_test_democlin=y_test_democlin, score_test_democlin=score_test_democlin,
                y_test_stck=y_test_stck, score_test_stck=score_test_stck,
                coef_img=estimator_img.coef_)

###############################################################################
#%% 1.2) l1, l2, tv parametrisation function

def ratios_to_param(alpha, l1l2ratio, tvl2ratio):
    tv = alpha * tvl2ratio
    l1 = alpha * l1l2ratio
    l2 = alpha * 1
    return l1, l2, tv

###############################################################################
#%% 1.3) Dataset loader

def load_dataset(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR):
    # Mask
    mask_filename = os.path.join(input_dir, "mni_cerebrum-gm-mask_1.5mm.nii.gz")
    mask_img = nibabel.load(mask_filename)
    mask_arr = mask_img.get_fdata() != 0
    assert np.sum(mask_arr != 0) == 331695

    # TV Linear operator
    linoperatortv_filename = os.path.join(input_dir, "mni_cerebrum-gm-mask_1.5mm_Atv.npz")
    if not os.path.exists(linoperatortv_filename):
        import parsimony.functions.nesterov.tv as nesterov_tv
        from parsimony.utils.linalgs import LinearOperatorNesterov
        Atv = nesterov_tv.linear_operator_from_mask(mask_img.get_fdata(), calc_lambda_max=True)
        Atv.save(linoperatortv_filename)
        Atv = LinearOperatorNesterov(filename=linoperatortv_filename)
        #assert np.allclose(Atv.get_singular_values(0), 11.940682881834617) # whole brain
        assert np.allclose(Atv.get_singular_values(0), 11.928817868042772) # rm brainStem+cerrebelum

    linoperatortv_filename = os.path.join(output_dir, "mni_cerebrum-gm-mask_1.5mm_Atv.npz")
    if not os.path.exists(linoperatortv_filename):
        copyfile(os.path.join(input_dir, "mni_cerebrum-gm-mask_1.5mm_Atv.npz"),
                 linoperatortv_filename)

    def load_study(input_dir, study, mask_arr):
        participants_filename = os.path.join(input_dir, "{study}_cat12vbm_participants.csv".format(study=study))
        imgs_filename = os.path.join(input_dir, "{study}_cat12vbm_mwp1-gs.npy".format(study=study))
        rois_filename = os.path.join(input_dir, "{study}_cat12vbm_rois-gs.csv".format(study=study))

        # Load Participants selection mask
        participants = pd.read_csv(participants_filename)
        select = participants.diagnosis.isin(['control', 'bipolar disorder', 'psychotic bipolar disorder'])
        participants["dx"] = participants.diagnosis.map({'control':0, 'bipolar disorder':1, 'psychotic bipolar disorder':1})
        participants = participants[select].reset_index(drop=True)

        # Load ROIs
        rois = pd.read_csv(rois_filename)
        rois = rois[select].reset_index(drop=True)

        # Load images
        #assert np.all(participants.diagnosis.isin(['control', 'bipolar disorder']))
        Xim = np.load(imgs_filename, mmap_mode='r').squeeze()[:, mask_arr]
        Xim = Xim[select, :]
        assert Xim.shape == (participants.shape[0], np.sum(mask_arr != 0))

        return participants, Xim, rois

    study = 'biobd-bsnip'
    participants_filename = os.path.join(output_dir, "{study}_cat12vbm_participants.csv".format(study=study))
    imgs_filename = os.path.join(output_dir, "{study}_cat12vbm_mwp1-gs-flat.npy".format(study=study))
    rois_filename = os.path.join(output_dir, "{study}_cat12vbm_rois-gs.csv".format(study=study))

    if not os.path.exists(participants_filename) or \
       not os.path.exists(imgs_filename) or \
       not os.path.exists(rois_filename):

        participants_biobd, Xim_biobd, rois_biobd = load_study(input_dir, study="biobd", mask_arr=mask_arr)
        participants_bsnip1, Xim_bsnip1, rois_bsnip1 = load_study(input_dir, study="bsnip1", mask_arr=mask_arr)
        assert np.sum(participants_biobd.dx == 0) ==  356
        assert np.sum(participants_biobd.dx == 1) ==  307
        assert np.sum(participants_bsnip1.dx == 0) ==  199
        assert np.sum(participants_bsnip1.dx == 1) ==  116

        # Concat studies
        participants = pd.concat([participants_biobd,
                                  participants_bsnip1], axis=0).reset_index(drop=True)
        rois = pd.concat([rois_biobd,
                          rois_bsnip1], axis=0).reset_index(drop=True)
        Xim = np.concatenate([Xim_biobd, Xim_bsnip1], axis=0)

        del participants_biobd, Xim_biobd, participants_bsnip1, Xim_bsnip1, rois_biobd, rois_bsnip1

        participants.to_csv(participants_filename, index=False)
        rois.to_csv(rois_filename, index=False)
        np.save(imgs_filename, Xim)

    else:

        participants = pd.read_csv(participants_filename)
        rois = pd.read_csv(rois_filename)
        Xim = np.load(imgs_filename, mmap_mode='r')

    assert Xim.shape == (participants.shape[0], np.sum(mask_arr != 0))
    assert rois.shape[0] == participants.shape[0]

    # Zres (residualization design matrix, Xdemo+site), Xdemoclin and target y
    msk = np.ones(participants.shape[0]).astype(bool)
    y = participants[TARGET_NUM][msk].values

    Xclin = participants.loc[msk, VAR_CLINIC].values
    Xdemo = participants.loc[msk, VAR_DEMO].values
    Xsite = pd.get_dummies(participants.site[msk]).values
    Xdemoclin = np.concatenate([Xdemo, Xclin], axis=1)
    formula_res, formula_full = "site + age + sex", "site + age + sex + " + TARGET_NUM
    residualizer = Residualizer(data=participants[msk], formula_res=formula_res, formula_full=formula_full)
    Zres = residualizer.get_design_mat(participants[msk])

    # Leave-site-out CV
    cv_lso_dict = {s:[np.where(participants.site != s)[0],
                      np.where(participants.site == s)[0]]
                   for s in participants.site.unique()}

    assert Xclin.shape[0] == Xdemoclin.shape[0] == Zres.shape[0] \
        == y.shape[0] == Xim.shape[0] == participants.shape[0]

    dataset = dict(Xim=Xim, y=y, Xdemoclin=Xdemoclin,
                  msk=msk, residualizer=residualizer, Zres=Zres,
                  mask_img=mask_img, mask_arr=mask_arr, participants=participants,
                  rois=rois,
                  cv_lso_dict=cv_lso_dict)
    gc.collect()

    return dataset


###############################################################################
#%% 1.4) Plot images

def plot_coefmap_stats(coef_vec, coef_vecs, mask_img, thresh_norm_ratio=0.99):
    """computes statistics and plot images from coef_vec and coef_vecs

    Parameters
    ----------
    coef_vec : array
        Coefficient vector.
    coef_vecs : [array]
        CV or bootstrappped coefficient vectors.
    mask_img : nii
        mask image.
    thresh_norm_ratio : float
        Threshold to apply to compute selection rate and to plot coef maps.

    Returns
    -------
    fig, axes, maps : dict
        dict containing all statistics images.
    """

    from nitk.utils import arr_threshold_from_norm2_ratio
    import nilearn.image
    from nilearn import plotting
    from nitk.image import vec_to_img, plot_glass_brains
    # arr_threshold_from_norm2_ratio(coef_vec, thresh_norm_ratio)[0]
    coef_vecs_t = np.vstack([arr_threshold_from_norm2_ratio(coef_vecs[i, :],
                                                            thresh_norm_ratio)[0]
                             for i in range(coef_vecs.shape[0])])

    w_selectrate = np.sum(coef_vecs_t != 0, axis=0) / coef_vecs_t.shape[0]
    w_zscore = np.nan_to_num(np.mean(coef_vecs, axis=0) / np.std(coef_vecs, axis=0))
    w_mean = np.mean(coef_vecs, axis=0)
    w_std = np.std(coef_vecs, axis=0)
    # 95% CI compute sign product of lower and hhigher 95%CI
    coef_vecs_ci = np.quantile(coef_vecs, [0.025, 0.975], axis=0)
    coef_vecs_ci_sign = np.sign(coef_vecs_ci.prod(axis=0))
    coef_vecs_ci_sign[coef_vecs_ci_sign == -1] = 0

    # Vectors to images
    coefmap_img = vec_to_img(coef_vec, mask_img)
    coefmap_cvmean_img = vec_to_img(w_mean, mask_img)
    w_mean[coef_vecs_ci_sign != 1] = 0
    coefmap_cvmean95ci_img = vec_to_img(w_mean, mask_img)
    coefmap_cvstd_img = vec_to_img(w_std, mask_img)
    coefmap_cvzscore_img = vec_to_img(w_zscore, mask_img)
    coefmap_cvselectrate_img = vec_to_img(w_selectrate, mask_img)

    threshold = arr_threshold_from_norm2_ratio(coef_vec, thresh_norm_ratio)[1]
    vmax = np.quantile(np.abs(coef_vec), 0.99)

    # Plot
    fig, axes = plot_glass_brains(
        imgs = [coefmap_img, coefmap_cvmean_img, coefmap_cvzscore_img, coefmap_cvselectrate_img],
        thresholds = [threshold, threshold, 3., None],
        vmax = [vmax, vmax, None, None],
        plot_abs = [False, False, False, True],
        colorbars = [True, True, True, True],
        cmaps = [plt.cm.bwr, plt.cm.bwr, None, None],
        titles = ['Coefs. Refit', 'Coefs. CV-Mean', 'Z-scores CV', 'Select. rate CV'])

    maps = {"coefmap": coefmap_img, "coefmap_mean": coefmap_cvmean_img,
            "coefmap_cvstd": coefmap_cvstd_img, "coefmap_cvzscore": coefmap_cvzscore_img,
            "coefmap_cvselectrate": coefmap_cvselectrate_img}

    return fig, axes, maps

###############################################################################
#%% 2) Descriptives stats

pdf_filename = OUTPUT.format(data='mwp1-gs', model="descriptives", experience="stats", type="scores", ext="pdf")
if False and not os.path.exists(pdf_filename):

    df = load_dataset()
    participants = df["participants"]
    assert participants.shape[0] ==  978

    stats_desc = participants[['study', 'site', 'age', 'dx']].groupby(['study', 'site']).agg(
        count=pd.NamedAgg(column="dx", aggfunc="count"),
        bp=pd.NamedAgg(column="dx", aggfunc="sum"),
        age=pd.NamedAgg(column="age", aggfunc="mean"))

    # TODO make some plots

###############################################################################
#%% 3) L2 LR mwp1-gs

xls_filename = OUTPUT.format(data='mwp1-gs', model="l2lr", experience="cvlso", type="scores", ext="xlsx")
models_filename = OUTPUT.format(data='mwp1-gs', model="l2lr", experience="cvlso", type="scores-coefs", ext="pkl")

if not os.path.exists(xls_filename):
    df = load_dataset()

    dataset = DATASET

    Cs = np.logspace(-2, 2, 5)
    # Cs = [10]

    estimators_dict = {"l2lr_C:%.6f" % C: lm.LogisticRegression(C=C, class_weight='balanced', fit_intercept=False) for C in Cs}
    estimators_dict.update({"l2lr-inter_C:%.6f" % C: lm.LogisticRegression(C=C, class_weight='balanced', fit_intercept=True) for C in Cs})

    cv_dict = df["cv_lso_dict"]
    #cv_dict = {"CV%02d" % fold:split for fold, split in enumerate(cv.split(df['Xim'], df['y']))}
    cv_dict["ALL"] = [np.arange(df['Xim'].shape[0]), np.arange(df['Xim'].shape[0])]

    # key_values_input = dict_product(estimators_dict, dict(resdualizeYes="yes"), cv_dict,
    key_values_input = dict_product(estimators_dict, dict(resdualizeNo="No", resdualizeYes="yes"), cv_dict,
        {'Xim_%s' % dataset :df['Xim']}, {'y_%s' % dataset :df['y']},
        {'Zres_%s' % dataset :df['Zres']}, {'Xdemoclin_%s' % dataset :df['Xdemoclin']},
        {'residualizer_%s' % dataset :df['residualizer']})
    print("Nb Tasks=%i" % len(key_values_input))

    start_time = time.time()
    key_vals_output = MapReduce(n_jobs=6, pass_key=True, verbose=20).map(fit_predict, key_values_input)
    print("# Centralized mapper completed in %.2f sec" % (time.time() - start_time))
    cv_scores_all = reduce_cv_classif(key_vals_output, cv_dict, y_true=df['y'], index_fold=2)

    cv_scores = cv_scores_all[cv_scores_all.fold != "ALL"]
    cv_scores_mean = cv_scores.groupby(["param_1", "param_0", "pred"]).mean().reset_index()
    cv_scores_mean.sort_values(["param_1", "param_0", "pred"], inplace=True, ignore_index=True)
    print(cv_scores_mean)

    with open(models_filename, 'wb') as fd:
        pickle.dump(key_vals_output, fd)

    with pd.ExcelWriter(xls_filename) as writer:
        cv_scores.to_excel(writer, sheet_name='folds', index=False)
        cv_scores_mean.to_excel(writer, sheet_name='mean', index=False)

###############################################################################
#%% 4) L2 LR rois-gs

xls_filename = OUTPUT.format(data='rois-gs', model="l2lr", experience="cv", type="scores", ext="xlsx")
models_filename = OUTPUT.format(data='rois-gs', model="l2lr", experience="cv", type="scores-coefs", ext="pkl")

if not os.path.exists(xls_filename):
    df = load_dataset()
    dataset = DATASET

    Cs = np.logspace(-4, 2, 7)
    # Cs = [10]

    estimators_dict = {"l2lr_C:%.6f" % C: lm.LogisticRegression(C=C, class_weight='balanced', fit_intercept=False) for C in Cs}
    estimators_dict.update({"l2lr-inter_C:%.6f" % C: lm.LogisticRegression(C=C, class_weight='balanced', fit_intercept=True) for C in Cs})

    # LSOCV
    cv_dict = df["cv_lso_dict"]
    #cv_dict = {"CV%02d" % fold:split for fold, split in enumerate(cv.split(df['Xim'], df['y']))}
    cv_dict["ALL"] = [np.arange(df['Xim'].shape[0]), np.arange(df['Xim'].shape[0])]

    # 5CV
    cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=3)
    cv5_dict = {"CV%i" % fold:split for fold, split in enumerate(cv5.split(df['Xim'], df['participants'].site))}
    cv_dict.update(cv5_dict)

    # key_values_input = dict_product(estimators_dict, dict(resdualizeYes="yes"), cv_dict,
    key_values_input = dict_product(estimators_dict, dict(resdualizeNo="No", resdualizeYes="yes"), cv_dict,
        {'Xim_%s' % dataset :df['rois'].loc[:, 'l3thVen_GM_Vol':].values}, {'y_%s' % dataset :df['y']},
        {'Zres_%s' % dataset :df['Zres']}, {'Xdemoclin_%s' % dataset :df['Xdemoclin']},
        {'residualizer_%s' % dataset :df['residualizer']})
    print("Nb Tasks=%i" % len(key_values_input))

    start_time = time.time()
    key_vals_output = MapReduce(n_jobs=16, pass_key=True, verbose=20).map(fit_predict, key_values_input)
    print("# Centralized mapper completed in %.2f sec" % (time.time() - start_time))
    cv_scores_all = reduce_cv_classif(key_vals_output, cv_dict, y_true=df['y'], index_fold=2)

    # 5CV
    cv5_scores = cv_scores_all[cv_scores_all.fold.isin(cv5_dict.keys())]
    cv5_scores_mean = cv5_scores.groupby(["param_1", "param_0", "pred"]).mean().reset_index()
    cv5_scores_mean.sort_values(["param_1", "param_0", "pred"], inplace=True, ignore_index=True)

    # LSOCV
    cvlso_scores = cv_scores_all[cv_scores_all.fold.isin(df["cv_lso_dict"].keys())]
    cvlso_scores_mean = cvlso_scores.groupby(["param_1", "param_0", "pred"]).mean().reset_index()
    cvlso_scores_mean.sort_values(["param_1", "param_0", "pred"], inplace=True, ignore_index=True)

    with open(models_filename, 'wb') as fd:
        pickle.dump(key_vals_output, fd)

    with pd.ExcelWriter(xls_filename) as writer:
        cv_scores.to_excel(writer, sheet_name='folds', index=False)
        cvlso_scores_mean.to_excel(writer, sheet_name='lsocv_mean', index=False)
        cv5_scores_mean.to_excel(writer, sheet_name='5cv_mean', index=False)

###############################################################################
#%% 5) EnetTV LR mwp1-gs

xls_filename = OUTPUT.format(data='mwp1-gs', model="enettv", experience="cvlso", type="scores", ext="xlsx")
models_filename = OUTPUT.format(data='mwp1-gs', model="enettv", experience="cvlso", type="scores-coefs", ext="pkl")
mapreduce_sharedir =  OUTPUT.format(data='mwp1-gs', model="enettv", experience="cvlso", type="models", ext="mapreduce")

if not os.path.exists(xls_filename):
    print(" %% 4.4) ENETTV 5CV grid search")
    df = load_dataset()
    dataset = DATASET

    mask_arr = df['mask_arr']
    mask_img = df['mask_img']

    # TV Linear operator
    linoperatortv_filename = os.path.join(INPUT_DIR, "mni_cerebrum-gm-mask_1.5mm_Atv.npz")
    Atv = LinearOperatorNesterov(filename=linoperatortv_filename)
    #assert np.allclose(Atv.get_singular_values(0), 11.940682881834617) # whole brain
    assert np.allclose(Atv.get_singular_values(0), 11.928817868042772) # rm brainStem+cerrebelum

    # Small range
    alphas = [0.1]
    l1l2ratios = [0.01]
    tvl2ratios = [0, 0.0001, 0.001, 0.01, 0.1, 1]

    # Smaller range
    # tv, l1, l2 from 202004 => tv, l1, l2 = 0.1, 0.001, 0.1
    # <=> alpha = 0.1, l1l2ratios=0.01, tvl2ratios = 1
    # alphas = [0.1]
    # l1l2ratios = [0.01]
    # tvl2ratios = [1]

    import itertools
    estimators_dict = dict()
    for alpha, l1l2ratio, tvl2ratio in itertools.product(alphas, l1l2ratios, tvl2ratios):
        print(alpha, l1l2ratio, tvl2ratio)
        l1, l2, tv = ratios_to_param(alpha, l1l2ratio, tvl2ratio)
        key = "enettv_%.3f:%.6f:%.6f" % (alpha, l1l2ratio, tvl2ratio)

        conesta = algorithms.proximal.CONESTA(max_iter=10000)
        estimator = estimators.LogisticRegressionL1L2TV(l1, l2, tv, Atv, algorithm=conesta,
                                                class_weight="auto", penalty_start=0)
        estimators_dict[key] = estimator

    # LSOCV
    cv_dict = df["cv_lso_dict"]
    #cv_dict = {"CV%02d" % fold:split for fold, split in enumerate(cv.split(df['Xim'], df['y']))}
    cv_dict["ALL"] = [np.arange(df['Xim'].shape[0]), np.arange(df['Xim'].shape[0])]

    #key_values_input = dict_product(estimators_dict, dict(resdualizeYes="yes"), cv_dict)
    key_values_input = dict_product(estimators_dict, dict(resdualizeYes="yes"), cv_dict,
        {'Xim_%s' % dataset :df['Xim']}, {'y_%s' % dataset :df['y']},
        {'Zres_%s' % dataset :df['Zres']}, {'Xdemoclin_%s' % dataset :df['Xdemoclin']},
        {'residualizer_%s' % dataset :df['residualizer']})

    print("Nb Tasks=%i" % len(key_values_input))


    ###########################################################################
    # 3) Distributed Mapper

    if os.path.exists(mapreduce_sharedir):
        print("# Existing shared dir, delete for fresh restart: ")
        print("rm -rf %s" % mapreduce_sharedir)

    os.makedirs(mapreduce_sharedir, exist_ok=True)


    start_time = time.time()
    mp = MapReduce(n_jobs=6, shared_dir=mapreduce_sharedir, pass_key=True, verbose=20)
    mp.map(fit_predict, key_values_input)
    key_vals_output = mp.reduce_collect_outputs()
    # key_vals_output = mp.reduce_collect_outputs(force=True)
    # [Parallel(n_jobs=6)]: Done  84 out of  84 | elapsed: 2691.0min finished 44h
    # 32 min / job

    ###########################################################################
    # 3) Centralized Mapper
    # start_time = time.time()
    # key_vals_output = MapReduce(n_jobs=NJOBS, pass_key=True, verbose=20).map(fit_predict, key_values_input)
    # print("#  Centralized mapper completed in %.2f sec" % (time.time() - start_time))

    ###############################################################################
    # 4) Reducer: output key/value pairs => CV scores""")

    if key_vals_output is not None:
        mp.make_archive()
        # make_archive(mapreduce_sharedir, "zip", root_dir=os.path.dirname(mapreduce_sharedir), base_dir=os.path.basename(mapreduce_sharedir))

        print("# Distributed mapper completed in %.2f sec" % (time.time() - start_time))
        cv_scores_all = reduce_cv_classif(key_vals_output, cv_dict, y_true=df['y'], index_fold=2)
        cv_scores = cv_scores_all[cv_scores_all.fold != "ALL"]
        cv_scores_mean = cv_scores.groupby(["param_1", "param_0", "pred"]).mean().reset_index()
        cv_scores_std = cv_scores.groupby(["param_1", "param_0", "pred"]).std().reset_index()
        cv_scores_mean.sort_values(["param_1", "param_0", "pred"], inplace=True, ignore_index=True)
        cv_scores_std.sort_values(["param_1", "param_0", "pred"], inplace=True, ignore_index=True)
        print(cv_scores_mean)

        # with open(models_filename, 'wb') as fd:
        #   pickle.dump(key_vals_output, fd)

        with pd.ExcelWriter(xls_filename) as writer:
            cv_scores.to_excel(writer, sheet_name='folds', index=False)
            cv_scores_mean.to_excel(writer, sheet_name='mean', index=False)
            cv_scores_std.to_excel(writer, sheet_name='std', index=False)


###############################################################################
#%% 6) Plot coef maps

mod_str = 'enettv_0.100:0.010000:0.100000'
pdf_filename =  OUTPUT.format(data='mwp1-gs', model=mod_str, experience="cvlso", type="coefmap", ext="pdf")

if not os.path.exists(pdf_filename):
    mapreduce_sharedir =  OUTPUT.format(data='mwp1-gs', model="enettv", experience="cvlso", type="models", ext="mapreduce")
    mp = MapReduce(n_jobs=6, shared_dir=mapreduce_sharedir, pass_key=True, verbose=20)
    key_vals_output = mp.reduce_collect_outputs(force=True)

    # Refit all coef map
    coef_vec = np.vstack([key_vals_output[k]['coef_img'].ravel() for k in
         [k for k in key_vals_output.keys() if
          (k[0] == mod_str and k[1] == "resdualizeYes" and k[2] == "ALL")]])[0]

    # CV
    coef_vecs = np.vstack([key_vals_output[k]['coef_img'].ravel() for k in
         [k for k in key_vals_output.keys() if
          (k[0] == mod_str and k[1] == "resdualizeYes" and k[2] != "ALL")]])

    # arr_threshold_from_norm2_ratio(coef_vec, .999)
    # threshold= 7.559217591801115e-05)
    #         arr_threshold_from_norm2_ratio(coef_vec, .99)
    # Out[94]: (array([0., 0., 0., ..., 0., 0., 0.]), 0.0001944127542099329)
    # arr_threshold_from_norm2_ratio(coef_vec, .9)

    pdf = PdfPages(pdf_filename)
    fig, axes, maps =  plot_coefmap_stats(coef_vec, coef_vecs, mask_img, thresh_norm_ratio=0.99)
    pdf.savefig(); plt.close(fig); pdf.close()

    #'coefmap_cvzscore' 'coefmap' 'coefmap_mean'
    nii_filename =  OUTPUT.format(data='mwp1-gs', model=mod_str, experience="refit", type="coefmap", ext="nii.gz")
    maps['coefmap'].to_filename(nii_filename)
    nii_filename =  OUTPUT.format(data='mwp1-gs', model=mod_str, experience="cvlso", type="coefmap-zcore", ext="nii.gz")
    maps['coefmap_cvzscore'].to_filename(nii_filename)

    # Cluster analysis
    cmd = "/home/ed203246/git/nitk/nitk/image/image_clusters_analysis.py %s --thresh_neg_high -3 --thresh_pos_low 3 --thresh_size 100  --save_atlas" % nii_filename
    p = subprocess.run(cmd.split())


mod_str = 'l2lr_C:10.000000'
pdf_filename =  OUTPUT.format(data='mwp1-gs', model=mod_str, experience="cvlso", type="coefmap", ext="pdf")
models_filename = OUTPUT.format(data='mwp1-gs', model="l2lr", experience="cvlso", type="scores-coefs", ext="pkl")

if not os.path.exists(pdf_filename):
    with open(models_filename, 'rb') as fd:
        key_vals_output = pickle.load(fd)

    # Refit all coef map
    coef_vec = np.vstack([key_vals_output[k]['coef_img'].ravel() for k in
         [k for k in key_vals_output.keys() if
          (k[0] == mod_str and k[1] == "resdualizeYes" and k[2] == "ALL")]])[0]

    # CV
    coef_vecs = np.vstack([key_vals_output[k]['coef_img'].ravel() for k in
         [k for k in key_vals_output.keys() if
          (k[0] == mod_str and k[1] == "resdualizeYes" and k[2] != "ALL")]])

    pdf = PdfPages(pdf_filename)
    fig, axes, maps =  plot_coefmap_stats(coef_vec, coef_vecs, mask_img, thresh_norm_ratio=0.99)
    pdf.savefig(); plt.close(fig); pdf.close()

    #'coefmap_cvzscore' 'coefmap' 'coefmap_mean'
    nii_filename =  OUTPUT.format(data='mwp1-gs', model=mod_str, experience="refit", type="coefmap", ext="nii.gz")
    maps['coefmap'].to_filename(nii_filename)
    nii_filename =  OUTPUT.format(data='mwp1-gs', model=mod_str, experience="cvlso", type="coefmap-zcore", ext="nii.gz")
    maps['coefmap_cvzscore'].to_filename(nii_filename)

###############################################################################
#%% 6) Clusters of z-scores > 3

mod_str = 'enettv_0.100:0.010000:0.100000'
#pdf_filename =  OUTPUT.format(data='mwp1-gs', model=mod_str, experience="cvlso", type="coefmap", ext="pdf")

datasets = load_dataset()


coef_img_filename =  OUTPUT.format(data='mwp1-gs', model=mod_str, experience="refit", type="coefmap", ext="nii.gz")
clust_info_filename =  OUTPUT.format(data='mwp1-gs', model=mod_str, experience="cvlso", type="coefmap-zcore_clust_info", ext="csv")
clust_img_filename =  OUTPUT.format(data='mwp1-gs', model=mod_str, experience="cvlso", type="coefmap-zcore_clust_labels", ext="nii.gz")

coef_vec = nibabel.load(coef_img_filename).get_fdata()[mask_arr]
clust_vec = nibabel.load(clust_img_filename).get_fdata()[mask_arr]
clust_info = pd.read_csv(clust_info_filename)

clust_rois = datasets['participants'].copy()
for idx, row in clust_info.iterrows():
    if row["prop_norm2_weight"] > 0.01:
        vec_mask = row["label"] == clust_vec
        assert np.sum(vec_mask) == row["size"]
        roi_name = "_".join([n.strip().replace(', ',':').replace(' ','-') for n in row[['ROI_HO-cort_peak_pos', 'ROI_HO-cort_peak_neg',
                         'ROI_HO-sub_peak_pos', 'ROI_HO-sub_peak_neg']] if pd.notnull(n)])
        roi_name = "%3i__%s" % (row["label"], roi_name)
        print(roi_name)
        coef_vec_masked = coef_vec.copy()
        coef_vec_masked[np.logical_not(vec_mask)] = 0
        clust_rois[roi_name] = np.dot(datasets['Xim'], coef_vec_masked)

clust_roi_names = [n for n in clust_rois.columns if n[3:5] == '__']
assert len(clust_roi_names) == 21

# groups clusters
df = clust_rois[clust_roi_names]
# Compute the correlation matrix
corr = df.corr()


d = 2 * (1 - np.abs(corr))

from sklearn.cluster import AgglomerativeClustering
clustering = AgglomerativeClustering(n_clusters=10, linkage='single', affinity="precomputed").fit(d)
lab=0

clusters = [list(corr.columns[clustering.labels_==lab]) for lab in set(clustering.labels_)]
print(clusters)

reordered = np.concatenate(clusters)

R = corr.loc[reordered, reordered]

cmap = sns.color_palette("RdBu_r", 11)

f, ax = plt.subplots(figsize=(5.5, 4.5))
# Draw the heatmap with the mask and correct aspect ratio
_ = sns.heatmap(R, mask=None, cmap=cmap, vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

"""
for c in clusters:
    print(c)
[' 48__Precentral-Gyrus_Left-Cerebral-Cortex', '107__Insular-Cortex_Left-Cerebral-Cortex', '115__Angular-Gyrus_Left-Cerebral-Cortex', '113__Postcentral-Gyrus_Left-Cerebral-Cortex', '  1__Middle-Temporal-Gyrus:temporooccipital-part_Right-Cerebral-White-Matter', ' 27__Background_Right-Pallidum', ' 95__Superior-Frontal-Gyrus_Left-Cerebral-Cortex', '  3__Planum-Polare_Right-Cerebral-Cortex', ' 41__Postcentral-Gyrus_Right-Cerebral-Cortex', ' 34__Frontal-Orbital-Cortex_Right-Cerebral-Cortex', ' 46__Frontal-Pole_Right-Cerebral-Cortex', ' 44__Superior-Frontal-Gyrus_Right-Cerebral-Cortex']
[' 59__Lingual-Gyrus_Right-Cerebral-Cortex']
[' 87__Temporal-Pole_Left-Cerebral-Cortex']
['110__Inferior-Temporal-Gyrus:anterior-division_Left-Cerebral-Cortex']
[' 26__Cingulate-Gyrus:posterior-division_Right-Hippocampus']
[' 81__Frontal-Orbital-Cortex_Left-Cerebral-Cortex']
[' 55__Subcallosal-Cortex_Left-Cerebral-Cortex']
[' 83__Occipital-Pole_Left-Cerebral-Cortex']
['126__Central-Opercular-Cortex_Left-Cerebral-White-Matter']
['117__Precentral-Gyrus_Left-Cerebral-Cortex']
"""