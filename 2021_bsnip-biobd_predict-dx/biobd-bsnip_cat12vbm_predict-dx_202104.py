# -*- coding: utf-8 -*-
"""
# Copy NS => Laptop
rsync -azvu is234606.intra.cea.fr:/neurospin/tmp/psy_sbox/all_studies/derivatives/arrays/mni_cerebrum-gm-mask_1.5mm_* ./

NS => Laptop
rsync -azvun --delete ed203246@is234606.intra.cea.fr:/neurospin/tmp/psy_sbox/analyses/202104_biobd-bsnip_cata12vbm_predict-dx /home/ed203246/data/psy_sbox/analyses/

Laptop => NS
rsync -azvun --delete /home/ed203246/data/psy_sbox/analyses/202104_biobd-bsnip_cata12vbm_predict-dx ed203246@is234606.intra.cea.fr:/neurospin/tmp/psy_sbox/analyses/

# BIOBD NS => Laptop
rsync -azvun is234606.intra.cea.fr:/neurospin/tmp/psy_sbox/all_studies/derivatives/arrays/biobd* /home/ed203246/data/psy_sbox/all_studies/derivatives/arrays/
"""


import os
import os.path
import numpy as np
import scipy
import pandas as pd
import glob
import copy
import time
import gc
import pickle
from collections import OrderedDict
from shutil import copyfile, make_archive, unpack_archive, move
import subprocess
import json

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
# Neuroimaging
import scipy.ndimage
import nibabel
from nilearn.image import new_img_like
from nilearn.plotting import plot_glass_brain

#from nitk.image import niimgs_bids_to_array, global_scaling, compute_brain_mask, rm_small_clusters, img_plot_glass_brain
#from nitk.bids import get_keys
#from nitk.data import fetch_data

#from nitk.utils import maps_similarity, arr_threshold_from_norm2_ratio, arr_clusters
from nitk.image import flat_to_array
from nitk.image import vec_to_niimg

# from nitk.image import niimgs_bids_to_array

#from nitk.image import niimgs_bids_to_array, global_scaling, compute_brain_mask, rm_small_clusters, plot_glass_brains, flat_to_array
#from nitk.stats import Residualizer
from nitk.mapreduce import dict_product, MapReduce, reduce_cv_classif
from mulm.residualizer import Residualizer

# sklearn
import sklearn.metrics as metrics
import sklearn.linear_model as lm
import sklearn.svm as svm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold, KFold, LeaveOneOut
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


# estimators
import parsimony.algorithms as algorithms
import parsimony.estimators as estimators
import parsimony.functions.nesterov.tv as nesterov_tv
from parsimony.utils.linalgs import LinearOperatorNesterov

from xgboost import XGBClassifier # conda install -c conda-forge xgboost

###############################################################################
#
#%% Config: Input/Output
#
###############################################################################

STUDIES = ["biobd", "bsnip1"]
INPUT_DIR = "/neurospin/psy_sbox/all_studies/derivatives/arrays"
#INPUT_DIR = "/neurospin/tmp/psy_sbox/all_studies/derivatives/arrays"

OUTPUT_DIR = "/neurospin/psy_sbox/analyses/202104_biobd-bsnip_cat12vbm_predict-dx"
#OUTPUT_DIR = "/neurospin/psy_sbox/analyses/202104_biobd-bsnip_cat12vbm_predict-dx"

# On laptop
if not os.path.exists(OUTPUT_DIR):
    OUTPUT_DIR = OUTPUT_DIR.replace('/neurospin/tmp', '/home/ed203246/data')
    INPUT_DIR = INPUT_DIR.replace('/neurospin/tmp', '/home/ed203246/data')


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

    if hasattr(estimator_img, 'decision_function'):
        score_test_img = estimator_img.decision_function(Xim_test)
        score_train_img = estimator_img.decision_function(Xim_train)
    elif hasattr(estimator_img, 'predict_log_proba'):
        score_test_img = estimator_img.predict_log_proba(Xim_test)[:, 1]
        score_train_img = estimator_img.predict_log_proba(Xim_train)[:, 1]
    elif hasattr(estimator_img, 'predict_proba'):
        score_test_img = estimator_img.predict_proba(Xim_test)[:, 1]
        score_train_img = estimator_img.predict_proba(Xim_train)[:, 1]
    else:
        raise AttributeError("Missing: decision_function or predict_log_proba or predict_proba")

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

    # Retrieve coeficient if possible
    coef_img = None

    if hasattr(estimator_img, 'best_estimator_'):  # GridSearch case
        estimator_img = estimator_img.best_estimator_

    if hasattr(estimator_img, 'coef_'):
        coef_img = estimator_img.coef_

    return dict(y_test_true=y[test], y_test_img=y_test_img, score_test_img=score_test_img,
                y_test_democlin=y_test_democlin, score_test_democlin=score_test_democlin,
                y_test_stck=y_test_stck, score_test_stck=score_test_stck,
                coef_img=coef_img)


###############################################################################
#%% 1.2) Regularization path of L1- Logistic Regression

def l1_lr_path(X_train, y_train, X_test, y_test, verbose=True):
    from time import time
    from sklearn.svm import l1_min_c
    cs_ = l1_min_c(X_train, y_train, loss='log') * np.logspace(0, 16, 2000)
    print("Computing regularization path ...")
    start = time()
    lr = lm.LogisticRegression(penalty='l1', solver='liblinear',
                                          tol=1e-6, max_iter=int(1e6),
                                          warm_start=True,
                                          class_weight='balanced', fit_intercept=False)
    n_nonnull = 0

    coefs = [np.zeros(X_train.shape[1])]
    active = list()
    aucs = list()
    baccs = list()
    cs = list()
    for c in cs_:
        lr.set_params(C=c)
        lr.fit(X_train, y_train)
        coefs_ = lr.coef_.ravel()
        n_nonnull_ = np.sum(coefs_ != 0)
        if n_nonnull_ > n_nonnull:
            # which one(s) just get int ?
            new_idx = np.where((coefs_ != 0) != (coefs[-1] != 0))[0]
            if verbose:
                print(n_nonnull_, new_idx)
            active.append(new_idx)
            n_nonnull = n_nonnull_

            score_pred_test = lr.decision_function(X_test)
            y_pred_test = lr.predict(X_test)
            bacc_ = metrics.balanced_accuracy_score(y_test, y_pred_test)
            auc_ = metrics.roc_auc_score(y_test, score_pred_test)
            aucs.append(auc_)
            baccs.append(bacc_)
            cs.append(c)
            coefs.append(coefs_.copy())
        if np.all(coefs_ != 0):
            break

    if verbose:
        print("This took %0.3fs" % (time() - start))

    active = np.concatenate(active)
    assert len(active) == X_train.shape[1]
    coefs = np.concatenate([c[:, np.newaxis] for c in coefs], axis=1)
    return cs, active, coefs, aucs, baccs


###############################################################################
#%% 1.3) l1, l2, tv parametrisation function

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

        # DEBUG
        # arr = np.load(imgs_filename)#, mmap_mode='r')
        # arr_ = flat_to_array(data_flat=Xim, mask_arr=mask_arr)
        # arr[:, 0, ~mask_arr] = 0
        # np.all(arr == arr_)
        # np.all(flat_to_array(Xim, mask_arr).squeeze()[:, mask_arr] == Xim)
        # DEBUG

        # Load images
        #assert np.all(participants.diagnosis.isin(['control', 'bipolar disorder']))
        Xim = np.load(imgs_filename, mmap_mode='r').squeeze()[:, mask_arr]
        Xim = Xim[select, :]
        assert Xim.shape == (participants.shape[0], np.sum(mask_arr != 0))

        return participants, Xim, rois

    study = 'biobd-bsnip'
    participants_filename = os.path.join(output_dir, "{study}_cat12vbm_participants.csv".format(study=study))
    imgs_flat_filename = os.path.join(output_dir, "{study}_cat12vbm_mwp1-gs-flat.npy".format(study=study))
    imgs_filename = os.path.join(output_dir, "{study}_cat12vbm_mwp1-gs.npy".format(study=study))
    rois_filename = os.path.join(output_dir, "{study}_cat12vbm_rois-gs.csv".format(study=study))

    if not os.path.exists(participants_filename) or \
       not os.path.exists(imgs_flat_filename) or \
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
        np.save(imgs_flat_filename, Xim)

        img_arr = flat_to_array(Xim, mask_arr)
        assert np.all(img_arr.squeeze()[:, mask_arr] == Xim)
        np.save(imgs_filename, img_arr)

    else:

        participants = pd.read_csv(participants_filename)
        rois = pd.read_csv(rois_filename)
        Xim = np.load(imgs_flat_filename, mmap_mode='r')
        # imgs_arr = np.load(imgs_filename, mmap_mode='r')
        # assert np.all(img_arr.squeeze()[:, mask_arr] == Xim)

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
#%% 1.4) Images utils

def vec_to_arr(vec, mask_arr, fill=0):
    arr = np.zeros(mask_arr.shape)
    if fill != 0:
        arr[::] = fill
    arr[mask_arr] = vec
    return arr

def clean_mask(mask, clust_size_thres=10):
    # Remove small branches
    mask = scipy.ndimage.binary_opening(mask)
    # Avoid isolated clusters: remove all cluster smaller that clust_size_thres
    mask = rm_small_clusters(mask, clust_size_thres=clust_size_thres)
    return mask

###############################################################################
#%% 1.4) Plot images

def plot_coefmap_stats(coef_vec, coefs_vec, mask_img, thresh_norm_ratio=0.99):
    """computes statistics and plot images from coef_vec and coefs_vec

    Parameters
    ----------
    coef_vec : array
        Coefficient vector.
    coefs_vec : [array]
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
    from nitk.image import vec_to_niimg, plot_glass_brains
    # arr_threshold_from_norm2_ratio(coef_vec, thresh_norm_ratio)[0]
    coefs_vec_t = np.vstack([arr_threshold_from_norm2_ratio(coefs_vec[i, :],
                                                            thresh_norm_ratio)[0]
                             for i in range(coefs_vec.shape[0])])

    w_selectrate = np.sum(coefs_vec_t != 0, axis=0) / coefs_vec_t.shape[0]
    w_zscore = np.nan_to_num(np.mean(coefs_vec, axis=0) / np.std(coefs_vec, axis=0))
    w_mean = np.mean(coefs_vec, axis=0)
    w_std = np.std(coefs_vec, axis=0)
    # 95% CI compute sign product of lower and hhigher 95%CI
    coefs_vec_ci = np.quantile(coefs_vec, [0.025, 0.975], axis=0)
    coefs_vec_ci_sign = np.sign(coefs_vec_ci.prod(axis=0))
    coefs_vec_ci_sign[coefs_vec_ci_sign == -1] = 0

    # Vectors to images
    coefmap_img = vec_to_niimg(coef_vec, mask_img)
    coefmap_cvmean_img = vec_to_niimg(w_mean, mask_img)
    w_mean[coefs_vec_ci_sign != 1] = 0
    coefmap_cvmean95ci_img = vec_to_niimg(w_mean, mask_img)
    coefmap_cvstd_img = vec_to_niimg(w_std, mask_img)
    coefmap_cvzscore_img = vec_to_niimg(w_zscore, mask_img)
    coefmap_cvselectrate_img = vec_to_niimg(w_selectrate, mask_img)

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
#%% 1.5) Stratified resmpler

def sample_stratified(groups, size, shuffle=False, random_state=None):
    """Sample size observations stratified by groups.

    Parameters
    ----------
    groups : DataFrame (n_samples x grouping variables). Index of the array
        will be used to determine the returned indices.
        .
    size : int
        Desired sample size.

    shuffle : bool, default=False
            Whether to shuffle the data before splitting into batches.

    random_state int, RandomState instance or None, default=None
        When shuffle is True, random_state affects the ordering of the indices,
        which controls the randomness of each fold.

    Returns
    -------
        array of indices ma
    """
    groups = groups.copy()
    n_subjects = groups.shape[0]

    cols = groups.columns.to_list()
    groups.insert(0, '_dummy_', 1)
    # nb_of selected subject for each sample size (stratified for site and site)
    count_tot = groups.groupby(cols).count()
    count = (count_tot[['_dummy_']] *  size / n_subjects).round(0).astype(int)

    if random_state is not None:
        np.random.set_state(random_state)

    indices = list()
    for k, d in groups.groupby(cols):
        indices_ = d.index.values.copy()
        if shuffle:
            np.random.shuffle(indices_)

        indices.append(indices_[:count.loc[k, '_dummy_']])

    indices = np.concatenate(indices)
    assert np.all(groups.loc[indices, :].groupby(cols).count() == count)

    return indices

###############################################################################
#%% 1.6) mean_se_z_t_ci_pval

def mean_se_z_t_ci_pval(X, mu=0, two_tailed=True, ddof=1):
    """Compute Mean, Standart Errors, z-scores, t-score, confidence interval, and pvalue.

    Parameters
    ----------
    vectors : array (n, p)
        vectors of p-dimensional arrays.
    mu : TYPE, optional, The default is 0.
        Null hypothesis.

    Returns
    -------
    mean : array (p, )
        Mean (accross axis 0).
    se :  array (p, )
        SEs.
    z :  array (p, )
        Z-scores.
    tstat :  array (p, )
        T-values.
    pval :  array (p, )
        P-values.

    Example
    -------
    import numpy as np
    #from mulm import mean_se_z_t_ci_pval
    X = [.1, .2, .3, -.1, .1, .2, .3]
    mean, se, z, tstat, pval, ci = mean_se_z_t_ci_pval(X)
    assert np.allclose(mean, 0.15714286)
    assert np.allclose(tstat, 2.9755)
    assert np.allclose(pval, 0.02478)
    assert np.allclose(ci[0], 0.02791636)
    assert np.allclose(ci[1], 0.28636936)
    """
    from mulm import ttest_ci, ttest_pval
    X = np.asarray(X)
    X = np.expand_dims(X, axis=1) if X.ndim == 1 else X

    n = X.shape[0]
    df = n - 1
    mean = X.mean(axis=0)
    sd = X.std(axis=0, ddof=ddof)
    z = (mean - mu) / sd
    se = sd / np.sqrt(n)
    _, _, tstat, ci = \
        ttest_ci(df=df, estimate=mean, se=se, mu=mu, two_tailed=two_tailed)
    pval = ttest_pval(df=df, tstat=tstat, two_tailed=two_tailed)

    return mean, se, z, tstat, pval, ci

###############################################################################
#%% 2) Descriptives stats

descriptive_filename = OUTPUT.format(data='mwp1-gs', model="descriptives", experience="stats", type="scores", ext="xlsx")
if False and not os.path.exists(descriptive_filename):

    datasets = load_dataset()
    participants = datasets["participants"]
    assert participants.shape[0] ==  978

    stats_desc = participants[['study', 'site', 'age', 'dx', 'sex']].groupby(['study', 'site']).agg(
        N=pd.NamedAgg(column="dx", aggfunc="count"),
        Age=pd.NamedAgg(column="age", aggfunc="mean"),
        Age_sd=pd.NamedAgg(column="age", aggfunc="std"),
        Sex_f=pd.NamedAgg(column="sex", aggfunc="sum"),
        Sex_fprop=pd.NamedAgg(column="sex", aggfunc="mean"),
        BD=pd.NamedAgg(column="dx", aggfunc="sum"),
        BD_prop=pd.NamedAgg(column="dx", aggfunc="mean")
        )

    stats_desc.Sex_fprop = stats_desc.Sex_fprop * 100
    stats_desc.BD_prop = stats_desc.BD_prop * 100

    stats_desc.astype(int).to_excel(descriptive_filename)

###############################################################################
#%% 3) Fit models

###############################################################################
#%% 3.1) L2 LR mwp1-gs

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

#%% 3.1.bis) L2 LR mwp1-gs from scratch

if False:

    from sklearn.metrics import balanced_accuracy_score, roc_auc_score

    df = load_dataset()
    C = 1.0
    key = "l2lr_C:%.6f" % C
    estimator_img = lm.LogisticRegression(C=C, class_weight='balanced', fit_intercept=False)
    residualize = 'yes'
    Xim = df['Xim']
    y = df['y']
    Zres = df['Zres']
    Xdemoclin = df['Xdemoclin']
    residualizer = df['residualizer']
    res = dict()
    for site, split in df["cv_lso_dict"].items():
        print(site)
        res_ = fit_predict(key, estimator_img, residualize, split, Xim, y, Zres, Xdemoclin, residualizer)
        res[site] = res_

    aucs = [roc_auc_score(res_['y_test_true'], res_['score_test_img']) for site, res_ in res.items()]
    baccs = [balanced_accuracy_score(res_['y_test_true'], res_['y_test_img']) for site, res_ in res.items()]

    assert (np.mean(aucs), np.std(aucs)) == (0.6841169748622756, 0.06769846496721038)
    assert (np.mean(baccs), np.std(baccs)) == (0.6274203590707195, 0.07945021508966936)

###############################################################################
#%% 3.2) L2 LR mwp1-gs Random permutations

csv_filename = OUTPUT.format(data='mwp1-gs', model="l2lr", experience="cvlso-randperm", type="scores", ext="csv")

if False and not os.path.exists(csv_filename):
    print(" %% 3.1) L2 LR mwp1-gs Random permutations")

    df = load_dataset()
    dataset = DATASET
    estimators_dict = {"l2lr_C:%.6f" % C: lm.LogisticRegression(C=10, class_weight='balanced', fit_intercept=False) for C in [10]}
    cv_dict = df["cv_lso_dict"]

    scores_perm = pd.DataFrame()
    perm_i = -1

    nperm = 20
    #for perm_i in range(nperm):
    for perm_i in range(perm_i+1, perm_i+1+nperm):
        print("# perm %i" % perm_i)
        rand_idx = np.random.permutation(len(df['y']))
        y_rand = df['y'][rand_idx]
        Z_rand = df['Zres'].copy()
        Z_rand[:, -1] = Z_rand[rand_idx, -1] # permute only the target (last col)
        # key_values_input = dict_product(estimators_dict, dict(resdualizeYes="yes"), cv_dict,
        key_values_input = dict_product(estimators_dict, dict(resdualizeYes="yes"), cv_dict,
            {'Xim_%s' % dataset :df['Xim']}, {'y_%s' % dataset :y_rand},
            {'Zres_%s' % dataset :Z_rand}, {'Xdemoclin_%s' % dataset :df['Xdemoclin']},
            {'residualizer_%s' % dataset :df['residualizer']})
        #print("Nb Tasks=%i" % len(key_values_input))

        key_vals_output = MapReduce(n_jobs=7, pass_key=True, verbose=20).map(fit_predict, key_values_input)
        #print("# Centralized mapper completed in %.2f sec" % (time.time() - start_time))
        cv_scores_all = reduce_cv_classif(key_vals_output, cv_dict, y_true=df['y'], index_fold=2)
        cv_scores = cv_scores_all[(cv_scores_all.fold != "ALL") & (cv_scores_all.pred == "test_img")]
        cv_scores['perm'] = perm_i

        cv_scores_mean = cv_scores.groupby(["param_1", "param_0", "pred"]).mean().reset_index()
        print(cv_scores_mean[["auc", "bacc"]])

        scores_perm = scores_perm.append(cv_scores)

    scores_perm.to_csv(csv_filename, index=False)

###############################################################################
#%% 3.3) L2 LR rois-gs

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
#%% 3.4) EnetTV LR mwp1-gs

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
#%% 4.1) Discriminative coef maps: zscores


mod_str = 'enettv_0.100:0.010000:0.100000'
pdf_filename =  OUTPUT.format(data='mwp1-gs', model=mod_str, experience="cvlso", type="coefmap", ext="pdf")
coefmap_cv_npz_filename =  OUTPUT.format(data='mwp1-gs', model=mod_str, experience="cvlso", type="coefmap", ext="npz")
coefmap_refit_nii_filename =  OUTPUT.format(data='mwp1-gs', model=mod_str, experience="refit", type="coefmap", ext="nii.gz")
coefmapzscore_refit_nii_filename =  OUTPUT.format(data='mwp1-gs', model=mod_str, experience="cvlso", type="coefmap-zcore", ext="nii.gz")

if not os.path.exists(coefmap_cv_npz_filename):

    # Read coeficient maps
    mapreduce_sharedir =  OUTPUT.format(data='mwp1-gs', model="enettv", experience="cvlso", type="models", ext="mapreduce")
    mp = MapReduce(n_jobs=6, shared_dir=mapreduce_sharedir, pass_key=True, verbose=20)
    key_vals_output = mp.reduce_collect_outputs(force=True)
    coefs_cv_vec = {k[2]: key_vals_output[k]['coef_img'].ravel() for k in
             [k for k in key_vals_output.keys() if (k[0] == mod_str and k[1] == "resdualizeYes")]}

    coefs_cv_vec = {k:coefs_cv_vec[k] for k in sorted(coefs_cv_vec)}

    np.savez_compressed(coefmap_cv_npz_filename, **coefs_cv_vec)
    coefs_cv_vec_ = np.load(coefmap_cv_npz_filename)
    assert np.all([np.all(coefs_cv_vec_[k] == coefs_cv_vec[k]) for k in coefs_cv_vec])


    coefs_cv_vec = np.load(coefmap_cv_npz_filename)
    coef_vec = coefs_cv_vec["ALL"]#[None, :]
    coefs_vec = np.vstack([v for k, v in coefs_cv_vec.items() if k != "ALL"])


    pdf = PdfPages(pdf_filename)
    fig, axes, maps =  plot_coefmap_stats(coef_vec, coefs_vec, mask_img, thresh_norm_ratio=0.99)
    pdf.savefig(); plt.close(fig); pdf.close()

    fig, axes = plot_glass_brains(
        imgs = [maps['coefmap_cvzscore']],
        thresholds = [3],
        vmax = [None],
        plot_abs = [False],
        colorbars = [True],
        cmaps = [None],
        titles = ["Enet-TV: Leave-Site-Out z-scores of model's coefficient map"])

    #'coefmap_cvzscore' 'coefmap' 'coefmap_mean'
    maps['coefmap'].to_filename(coefmap_refit_nii_filename)
    maps['coefmap_cvzscore'].to_filename(coefmapzscore_refit_nii_filename)

    # Cluster analysis
    cmd = "/home/ed203246/git/nitk/nitk/image/image_clusters_analysis.py %s --thresh_neg_high -3 --thresh_pos_low 3 --thresh_size 100  --save_atlas" % coefmapzscore_refit_nii_filename
    p = subprocess.run(cmd.split())


mod_str = 'l2lr_C:10.000000'
pdf_filename =  OUTPUT.format(data='mwp1-gs', model=mod_str, experience="cvlso", type="coefmap", ext="pdf")
models_filename = OUTPUT.format(data='mwp1-gs', model="l2lr", experience="cvlso", type="scores-coefs", ext="pkl")
coefmap_refit_nii_filename =  OUTPUT.format(data='mwp1-gs', model=mod_str, experience="refit", type="coefmap", ext="nii.gz")
coefmapzscore_refit_nii_filename =  OUTPUT.format(data='mwp1-gs', model=mod_str, experience="cvlso", type="coefmap-zcore", ext="nii.gz")

if not os.path.exists(pdf_filename):
    with open(models_filename, 'rb') as fd:
        key_vals_output = pickle.load(fd)

    # Refit all coef map
    coef_vec = np.vstack([key_vals_output[k]['coef_img'].ravel() for k in
         [k for k in key_vals_output.keys() if
          (k[0] == mod_str and k[1] == "resdualizeYes" and k[2] == "ALL")]])[0]

    # CV
    coefs_vec = np.vstack([key_vals_output[k]['coef_img'].ravel() for k in
         [k for k in key_vals_output.keys() if
          (k[0] == mod_str and k[1] == "resdualizeYes" and k[2] != "ALL")]])

    pdf = PdfPages(pdf_filename)
    fig, axes, maps =  plot_coefmap_stats(coef_vec, coefs_vec, mask_img, thresh_norm_ratio=0.99)
    pdf.savefig(); plt.close(fig); pdf.close()

    fig, axes = plot_glass_brains(
        imgs = [maps['coefmap_cvzscore']],
        thresholds = [3],
        vmax = [None],
        plot_abs = [False],
        colorbars = [True],
        cmaps = [None],
        titles = ["L2: Leave-Site-Out z-scores of model's coefficient map"])

    #'coefmap_cvzscore' 'coefmap' 'coefmap_mean'
    maps['coefmap'].to_filename(coefmap_refit_nii_filename)
    maps['coefmap_cvzscore'].to_filename(coefmapzscore_refit_nii_filename)


###############################################################################
#%% 4.2) Forward models Coef maps: zscores + clusters_analysis
# Haufe 2014 On the interpretation of weight vectors of linear models in multivariate neuroimaging


mod_str = 'enettv_0.100:0.010000:0.100000'
#pdf_filename =  OUTPUT.format(data='mwp1-gs', model=mod_str, experience="cvlso", type="coefmap", ext="pdf")
#coefmap_refit_nii_filename =  OUTPUT.format(data='mwp1-gs', model=mod_str, experience="refit", type="coefmap", ext="nii.gz")
#coefmapzscore_refit_nii_filename =  OUTPUT.format(data='mwp1-gs', model=mod_str, experience="cvlso", type="coefmap-zcore", ext="nii.gz")
coefmap_forward_cv_npz_filename =  OUTPUT.format(data='mwp1-gs', model=mod_str, experience="cvlso", type="coefmap-forward", ext="npz")
zmap_forward_filename =  OUTPUT.format(data='mwp1-gs', model=mod_str, experience="cvlso", type="coefmap-zscore-forward", ext="svg")

if not os.path.exists(coefmap_forward_cv_npz_filename):

    datasets = load_dataset()
    cv_dict = datasets["cv_lso_dict"]

    coefs_cv_vec = np.load(coefmap_cv_npz_filename)
    #coef_vec = coefs_cv_vec["ALL"]#[None, :]
    #coefs_vec = np.vstack([v for k, v in coefs_cv_vec.items() if k != "ALL"])


    fwds_cv_vec = dict()
    for i, (fold_name, (train, test)) in enumerate(cv_dict.items()):
         print("## %s (%i/%i)" % (fold_name, i+1, len(cv_dict)))
         residualizer = datasets['residualizer']
         X_train = residualizer.fit_transform(datasets['Xim'][train], datasets['Zres'][train])
         # X_test = residualizer.transform(datasets['Xim'][test], datasets['Zres'][test])
         scaler = preprocessing.StandardScaler()
         X_train = scaler.fit_transform(X_train)
         # X_test = scaler.transform(X_test)

         # Haufe 2014 Eq.5
         score_train = np.dot(X_train, coefs_cv_vec[fold_name])
         # score_test = np.dot(X_test, coefs_vec[fold_name])

         # Haufe 2014 Eq.7: get Forward map
         # Cov(X, score)
         fwd_vec = np.dot(X_train.T, score_train) / (score_train.shape[0] - 1)
         fwds_cv_vec[fold_name] = fwd_vec

    # Add Refit ALL
    residualizer = datasets['residualizer']
    X_train = residualizer.fit_transform(datasets['Xim'], datasets['Zres'])
    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)

    # Haufe 2014 Eq.5
    score_train = np.dot(X_train, coefs_cv_vec['ALL'])

    # Haufe 2014 Eq.7: get Forward map
    # Cov(X, score)
    fwd_vec = np.dot(X_train.T, score_train) / (score_train.shape[0] - 1)
    fwds_cv_vec['ALL'] = fwd_vec

    fwds_cv_vec = {k:fwds_cv_vec[k] for k in sorted(fwds_cv_vec)}

    np.savez_compressed(coefmap_forward_cv_npz_filename, **fwds_cv_vec)
    fwds_cv_vec_ = np.load(coefmap_forward_cv_npz_filename)
    assert np.all([np.all(fwds_cv_vec_[k] == fwds_cv_vec[k]) for k in fwds_cv_vec])

    from nitk.image import vec_to_niimg
    W_cv = np.asarray([v for k, v in coefs_cv_vec.items() if k != 'ALL'])
    w_mean, w_se, w_z, w_tstat, w_pval, w_ci = mean_se_z_t_ci_pval(X=W_cv, mu=0)

    Fwd_cv = np.asarray([v for k, v in fwds_cv_vec.items() if k != 'ALL'])
    fwd_mean, fwd_se, fwd_z, fwd_tstat, fwd_pval, fwd_ci = mean_se_z_t_ci_pval(X=Fwd_cv, mu=0)

    from nilearn.plotting import plot_glass_brain

    # plot_glass_brain(vec_to_niimg(w_z.ravel(), mask_img),
    #              threshold=3, vmax=12,
    #              plot_abs=False, colorbar=True, cmap=None)

    # plot_glass_brain(vec_to_niimg(fwd_z.ravel(), mask_img),
    #              threshold=3,# vmax=12,
    #              symmetric_cbar=False, plot_abs=False, colorbar=True, cmap=None)

    map_ = fwd_z.copy()
    map_[np.abs(w_z) < 3] = 0
    plot_glass_brain(vec_to_niimg(map_.ravel(), mask_img),
                 threshold=3,# vmax=12,
                 plot_abs=False, colorbar=True, cmap=None)

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(11.69, 1 * 11.69 * .4))
    plot_glass_brain(vec_to_niimg(map_.ravel(), mask_img), plot_abs=False, threshold=3,
                     symmetric_cbar=False, colorbar=True, figure=fig, axes=axes,
                     #title="Z-map of forward transformation of Enet-TV coefﬁcients"
                     )
    plt.savefig(zmap_forward_filename.replace(".svg", ".png"), bbox_inches='tight', pad_inches=0)
    plt.savefig(zmap_forward_filename, bbox_inches='tight', pad_inches=0)



###############################################################################
#%% 4.3) Classification scores significance


if False:
    permutations_csv_filename = OUTPUT.format(data='mwp1-gs', model="l2lr", experience="cvlso-randperm", type="scores", ext="csv")
    enettv_xls_filename = OUTPUT.format(data='mwp1-gs', model="enettv", experience="cvlso", type="scores", ext="xlsx")
    l2lr_xls_filename = OUTPUT.format(data='mwp1-gs', model="l2lr", experience="cvlso", type="scores", ext="xlsx")
    rois_xls_filename = OUTPUT.format(data='rois-gs', model="l2lr", experience="cv", type="scores", ext="xlsx")

    def _macro_scores_stat_plot(x, rnd, var='ROC-AUC'):
        from scipy.stats import ttest_1samp
        mean = x.mean()
        se = x.std() / np.sqrt(len(x))
        pval = ttest_1samp(x, 0.5)[1]
        #def ttest_1samp_(x):
        #    return ttest_1samp(x, 0.5)[1]
        #pval = df[['auc', 'bacc']].apply(ttest_1samp_)

        import seaborn as sns
        sns.set_style("darkgrid")
        print(plt.rcParams['figure.figsize']) # [6.0, 4.0]

        #from scipy import stats
        # stats.t(df=len(x)).ppf((0.025, 0.975))
        df = pd.concat([
            pd.DataFrame({var:x, 'Condition':'True'}),
            pd.DataFrame({var:rnd, 'Condition':'Random'})])

        def _barplot_true_rnd(df, y, x, mean, se, ylim=(0.4, 1)):
            fig = plt.figure(figsize=(2.0, 11.69 * .4))
            #ax = sns.boxplot(y=x)
            ax = sns.barplot(y=y, x=x, data=df)#, ci='sd')
            #ax = sns.stripplot(y=var, x="cond", jitter=True, color="grey", data=df[df["cond"]=='True'])
            ax.set_ylim(*ylim)
            ax.set_ylabel('%s: %.0f\u00B1%.0f%% (p<%.1e)' % (var, mean*100, se*100, pval*10), fontsize=18)
            ax.set_yticklabels(ax.get_yticks().round(2), size = 16)
            ax.set_xticklabels(ax.get_xticklabels(), size = 16)
            ax.set_xlabel(None)
            #ax.axhline(0.5, ls='--', color='grey')
            return fig, ax

        fig, ax = _barplot_true_rnd(df=df, y=var, x="Condition", mean=mean, se=se)

        return mean, se, pval, fig, ax


    perm = pd.read_csv(permutations_csv_filename)
    #rnd_perm = perm.groupby("perm")["auc"].mean()
    rnd = perm["auc"]


    xls_filename, mod_str = enettv_xls_filename, 'enettv_0.100:0.010000:0.100000'
    macro_scores = pd.read_excel(xls_filename, sheet_name='folds')
    macro_scores = macro_scores.loc[
        (macro_scores['param_0'] == mod_str) &\
        (macro_scores['param_1'] == 'resdualizeYes') &\
        (macro_scores['pred'] == 'test_img'), ['fold', 'auc', 'bacc']]

    mean, se, pval, fig, ax = _macro_scores_stat_plot(x=macro_scores['auc'], rnd=perm["auc"])
    print(mean, se, pval)
    # 0.6886015615937928 0.020621777038463084 9.315119837238758e-07
    mean, se, pval, fig, ax = _macro_scores_stat_plot(x=macro_scores['bacc'], rnd=perm["bacc"], var='BACC')
    print(mean, se, pval)
    # 0.616117126765515 0.018737082576036783 4.6060548372081294e-05

    xls_filename, mod_str = l2lr_xls_filename, 'l2lr_C:10.000000'
    macro_scores = pd.read_excel(xls_filename, sheet_name='folds')
    macro_scores = macro_scores.loc[
        (macro_scores['param_0'] == mod_str) &\
        (macro_scores['param_1'] == 'resdualizeYes') &\
        (macro_scores['pred'] == 'test_img'), ['fold', 'auc', 'bacc']]

    mean, se, pval, fig, ax = _macro_scores_stat_plot(x=macro_scores['auc'], rnd=perm["auc"])
    print(mean, se, pval)
    # 0.6825792725169888 0.01975691632183294 8.342402696336089e-07
    mean, se, pval, fig, ax = _macro_scores_stat_plot(x=macro_scores['bacc'], rnd=perm["bacc"], var='BACC')
    print(mean, se, pval)
    # 0.616117126765515 0.018737082576036783 4.6060548372081294e-05


    #lso_macro_mean, lso_macro_se, pval, fig, ax = _macro_scores_stat(df=macro_scores)


# %% 5) Search light

# https://nilearn.github.io/modules/generated/nilearn.decoding.SearchLight.html
# https://nilearn.github.io/decoding/searchlight.html
# https://github.com/nilearn/nilearn/blob/master/nilearn/decoding/searchlight.py

#searchlight_scores_filename = OUTPUT.format(data='mwp1-gs', model="searchlight-l2lr", experience="cvlso", type="auc-bacc", ext="pkl")
searchlight_cv_scores_arr_filename = OUTPUT.format(data='mwp1-gs', model="searchlight-l2lr", experience="cvlso", type="auc-bacc", ext="npz")
searchlight_cv_scores_arr_plot_filename = OUTPUT.format(data='mwp1-gs', model="searchlight-l2lr", experience="cvlso", type="auc-bacc", ext="svg")


if not os.path.exists(searchlight_cv_scores_arr_filename):

    #%% 5.1) Compute Manualy iterate over folds to residualized on training
    from nitk.image import flat_to_array, array_to_niimgs
    from nitk.image import search_light

    datasets = load_dataset()
    mask_arr = datasets['mask_arr']
    mask_img = datasets['mask_img']
    cv_dict = datasets['cv_lso_dict']
    y = datasets['y']
    residualizer = datasets['residualizer']

    results = dict()
    print('# Search light choose folds in (separated by space):', cv_dict.keys())
    folds = [f.strip() for f in input('Fold:').split(' ')]
    cv_dict = {k:v for k, v in cv_dict.items() if k in folds}

    for i, (fold_name, (train, test)) in enumerate(cv_dict.items()):
        print("## %s (%i/%i)" % (fold_name, i+1, len(cv_dict)))
        searchlight_cv_scores_arr_filename_ = OUTPUT.format(data='mwp1-gs', model="searchlight-l2lr", experience="cvlso-%s" % fold_name, type="auc-bacc", ext="pkl")
        print(searchlight_cv_scores_arr_filename_)

        y_train = datasets['y'][train]
        y_test = datasets['y'][test]

        residualizer = datasets['residualizer']
        X_train = residualizer.fit_transform(datasets['Xim'][train], datasets['Zres'][train])
        X_test = residualizer.transform(datasets['Xim'][test], datasets['Zres'][test])
        scaler = preprocessing.StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        arr_train = flat_to_array(data_flat=X_train, mask_arr=mask_arr, fill=0).squeeze()
        arr_test = flat_to_array(data_flat=X_test, mask_arr=mask_arr, fill=0).squeeze()

        # 4 mms of radius
        radius = 4 / mask_img.header.get_zooms()[0]
        estimator = lm.LogisticRegression(C=10, class_weight='balanced', fit_intercept=False, max_iter=1000)
        results_ = search_light(mask_arr, arr_train, arr_test, y_train, y_test, estimator, radius, verbose=1000)

        results[(fold_name, 'auc')] = results_['auc']
        results[(fold_name, 'bacc')] = results_['bacc']

        with open(searchlight_cv_scores_arr_filename_, 'wb') as fd:
           pickle.dump(results_, fd)


    # %% 5.2) Reduce
    import re
    regex = re.compile("cvlso-(.+)_")
    searchlight_cv_scores_arr_filename_ = OUTPUT.format(data='mwp1-gs', model="searchlight-l2lr", experience="cvlso-%s" % '*', type="auc-bacc", ext="pkl")
    assert len(set([regex.findall(filename)[0] for filename in glob.glob(searchlight_cv_scores_arr_filename_)])) == 13

    results = dict()

    for filename in glob.glob(searchlight_cv_scores_arr_filename_):
        fold = regex.findall(filename)[0]
        print(filename, fold)

        with open(filename, 'rb') as fd:
          results_ = pickle.load(fd)

        results.update(results_)

    searchlight_cv_scores_arr = {"_".join(k):v for k, v in results.items()}
    searchlight_cv_scores_arr = {k:searchlight_cv_scores_arr[k] for k in sorted(searchlight_cv_scores_arr)}
    np.savez_compressed(searchlight_cv_scores_arr_filename, **searchlight_cv_scores_arr)
    del results

    # %% 5.3) Reload
    searchlight_cv_scores_arr = np.load(searchlight_cv_scores_arr_filename)
    searchlight_cv_scores_arr = {k.split('_')[0]:searchlight_cv_scores_arr[k] for k in sorted(searchlight_cv_scores_arr) if k.split('_')[1] == 'auc'}


    from statsmodels.stats.multitest import multipletests
    auc_arr = np.stack([v for k, v in searchlight_cv_scores_arr.items()])
    auc_mean_vec, auc_se_vec, auc_z_vec, auc_tstat_vec, auc_pval_vec, auc_ci_vec= \
        mean_se_z_t_ci_pval(auc_arr[:, mask_arr], mu=0.5,  two_tailed=False)

    sns.histplot(auc_pval_vec)

    # Correction for multpiple comparison
    auc_pval_fwer_vec = multipletests(auc_pval_vec, alpha=0.05, method='bonferroni')[1]
    auc_pval_fdr_vec = multipletests(auc_pval_vec, alpha=0.05, method='fdr_bh')[1]
    assert np.sum(auc_pval_vec < 0.05) == 99355
    assert np.sum(auc_pval_fwer_vec < 0.05) == 28
    assert np.sum(auc_pval_fdr_vec < 0.05) == 37608

    # !! 1 - pval image
    auc_pval_arr = vec_to_arr(1 - auc_pval_vec, mask_arr, fill=0)
    auc_pval_fwer_arr = vec_to_arr(1 - auc_pval_fwer_vec, mask_arr, fill=0)
    auc_pval_fdr_arr = vec_to_arr(1 - auc_pval_fdr_vec, mask_arr, fill=0)
    assert np.sum(auc_pval_fwer_arr >= 0.95) == 28 and np.sum(auc_pval_fdr_arr >= 0.95) == 37608

    # Plot
    pval_arr = auc_pval_fwer_arr
    pval_arr = auc_pval_fdr_arr

    #auc_plot_arr = auc_mean_arr.copy()
    #auc_plot_arr = auc_t_arr.copy()
    #auc_plot_arr =  vec_to_arr(auc_z_vec, mask_arr, fill=0)
    auc_plot_arr =  vec_to_arr(auc_mean_vec - 0.5, mask_arr, fill=0)
    auc_plot_arr[auc_plot_arr<0] = 0

    auc_signif_vals = auc_plot_arr[pval_arr >= 0.95]
    print(len(auc_signif_vals), np.min(auc_signif_vals), np.max(auc_signif_vals), np.sum(pval_arr >= 0.95))
    auc_plot_arr[pval_arr < 0.95] = 0

    auc_plot_niimg = new_img_like(mask_img, auc_plot_arr)
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(11.69, 1 * 11.69 * .4))
    plot_glass_brain(auc_plot_niimg, plot_abs=True, threshold=np.min(auc_signif_vals),
                     symmetric_cbar=False, colorbar=True, figure=fig, axes=axes,
                     #title="T-statistics map tresholded for fwer(maxT) <= 0.05"
                     )
    plt.savefig(searchlight_cv_scores_arr_plot_filename.replace(".svg", ".png"), bbox_inches='tight', pad_inches=0)
    plt.savefig(searchlight_cv_scores_arr_plot_filename, bbox_inches='tight', pad_inches=0)


 # %% 6) VBM: Univariate statistics

vbm_dirname = OUTPUT.format(data='mwp1-gs', model="vbm", experience="dx", type="ttest-tfce", ext="mulm-fsl")
vbm_tstat_filename =  OUTPUT.format(data='mwp1-gs', model="vbm", experience="dx", type="tstats-fwermaxT", ext="svg")

if not os.path.exists(vbm_dirname):

    os.makedirs(vbm_dirname, exist_ok=True)

    from nitk.image import flat_to_array, array_to_niimgs, vec_to_niimg, niimgs_to_array
    from nilearn.image import smooth_img
    import mulm


    datasets = load_dataset()
    mask_arr = datasets['mask_arr']
    mask_img = datasets['mask_img']

    from collections import OrderedDict
    Design, t_contrasts, f_contrasts = mulm.design_matrix(formula="dx + age + sex + site", data=datasets['participants'])

    # FSL randomize: Design and contrast matrix for fsl5.0-randomise

    if False:
        prefix = vbm_dirname + '/fsl'
        if not os.path.exists(prefix +'_design.mat'):
            pd.DataFrame(Design).to_csv(prefix +'_design.txt', header=None, index=None, sep=' ', mode='a')
            subprocess.run(["fsl5.0-Text2Vest", prefix +'_design.txt', prefix +'_design.mat'], stdout=subprocess.PIPE)
            os.remove(prefix +'_design.txt')
            contrasts = np.vstack([t_contrasts['dx'], -1 * t_contrasts['dx']])
            np.savetxt(prefix +'_contrast.txt', contrasts, fmt='%i')
            subprocess.run(["fsl5.0-Text2Vest", prefix +'_contrast.txt', prefix +'_contrast.mat'], stdout=subprocess.PIPE)
            os.remove(prefix +'_contrast.txt')

        arr = flat_to_array(data_flat=datasets['Xim'], mask_arr=mask_arr, fill=0)

        # Smooth Data for fsl5.0-randomise
        niimgs = array_to_niimgs(ref_niimg=mask_img, arr=flat_to_array(data_flat=datasets['Xim'], mask_arr=mask_arr, fill=0).squeeze())
        assert np.all(niimgs_to_array(niimgs).squeeze()[:, mask_arr] == datasets['Xim'])
        niimgs = smooth_img(niimgs, fwhm=8)
        if not os.path.exists(prefix +"ALL.nii.gz"):
            niimgs.to_filename(prefix + "ALL.nii.gz")

        # Run randomise
        # https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/Randomise
        # -T : TFCE
        # -C <thresh> for t contrasts, Cluster-based thresholding

        cmd = ["fsl5.0-randomise", '-i', prefix + "ALL.nii.gz", "-m",  os.path.join(INPUT_DIR, "mni_cerebrum-gm-mask_1.5mm.nii.gz"),
         "-o", vbm_dirname,
         '-d', prefix +'_design.mat',
         '-t', prefix +'_contrast.mat', '-T', '-n', '500', "-C", "3"]

        print(" ".join(cmd))

    # MUOLS

    if not os.path.exists(vbm_dirname + '/mulm_pvals-maxT.nii.gz'):
        Y = niimgs_to_array(niimgs).squeeze()[:, mask_arr]
        mod_mulm = mulm.MUOLS(Y, Design).fit()
        tstat, pval, df = mod_mulm.t_test(t_contrasts['dx'], pval=True)
        pval_fwer =  pval * len(pval.ravel())
        pval_fwer[pval_fwer > 1] = 1
        print(np.quantile(pval_fwer, (0.0001, 0.001, 0.01)), np.sum(pval_fwer < 0.05))
        # [1.13558031e-04 1.16418598e-02 1.00000000e+00] 711

        tstat_, pval_maxt, df2 = mod_mulm.t_test_maxT(t_contrasts['dx'], two_tailed=True, nperms=1000)
        assert np.all(tstat_ == tstat)
        print(np.quantile(pval_maxt, (0.0001, 0.001, 0.01, 0.05)), np.sum(pval_maxt <= 0.05))
        # [0.   0.   0.06 0.56] 3150

        tstat_signif = tstat.copy()
        tstat_signif[pval_maxt > 0.05] = 0
        print(np.sum(tstat_signif != 0))

        # Save images
        tstat_niimg = vec_to_niimg(tstat.ravel(), mask_img)
        #tstat_signif_niimg = vec_to_niimg(tstat_signif.ravel(), mask_img)
        pval_niimg = vec_to_niimg(1 - pval.ravel(), mask_img)
        pval_fwer_niimg = vec_to_niimg(1 - pval_fwer.ravel(), mask_img)
        pval_maxt_niimg = vec_to_niimg(1 - pval_maxt.ravel(), mask_img)

        tstat_niimg = vec_to_niimg(tstat.ravel(), mask_img)

        tstat_niimg.to_filename(vbm_dirname + '/mulm_tstat.nii.gz')
        pval_niimg.to_filename(vbm_dirname + '/mulm_pvals.nii.gz')
        pval_fwer_niimg.to_filename(vbm_dirname + '/mulm_pvals-fwer.nii.gz')
        pval_maxt_niimg.to_filename(vbm_dirname + '/mulm_pvals-maxT.nii.gz')


    tstat_niimg = nibabel.load(vbm_dirname + '/mulm_tstat.nii.gz')
    pval_niimg = nibabel.load(vbm_dirname + '/mulm_pvals.nii.gz')
    pval_fwer_niimg = nibabel.load(vbm_dirname + '/mulm_pvals-fwer.nii.gz')
    pval_maxt_niimg = nibabel.load(vbm_dirname + '/mulm_pvals-maxT.nii.gz')

    assert np.sum(pval_maxt_niimg.get_fdata() > 0.95) == 3817

    tstat_signif_arr = tstat_niimg.get_fdata().copy()
    tstat_signif_arr[pval_maxt_niimg.get_fdata() < 0.95] = 0
    tstat_signif_niimg = new_img_like(mask_img, tstat_signif_arr)
    tstat_signif_niimg.to_filename(vbm_dirname + '/mulm_tstat_signif.nii.gz')


    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(11.69, 1 * 11.69 * .4))
    plot_glass_brain(tstat_signif_niimg, plot_abs=False,# threshold=3,
                     symmetric_cbar=False, colorbar=True, figure=fig, axes=axes,
                     #title="T-statistics map tresholded for fwer(maxT) <= 0.05"
                     )
    plt.savefig(vbm_tstat_filename.replace(".svg", ".png"), bbox_inches='tight', pad_inches=0)
    plt.savefig(vbm_tstat_filename, bbox_inches='tight', pad_inches=0)



###############################################################################
# %% 7) Patterns of regions
# Regions of z-scores > 3: Clusters' correlation matrix and cluster's importance
mod_str = 'enettv_0.100:0.010000:0.100000'
REGIONS_SCORE_FILENAME =  OUTPUT.format(data='mwp1-gs', model=mod_str, experience="refit", type="regions-score", ext="csv")
REGIONS_INFO_FILENAME =  OUTPUT.format(data='mwp1-gs', model=mod_str, experience="regions", type="info", ext="csv")
pattern_fwdzmap_filename =  OUTPUT.format(data='mwp1-gs', model=mod_str, experience="cvlso", type="coefmap-forward-zcore-{pattern}", ext="svg")

zscore_img_filename =  OUTPUT.format(data='mwp1-gs', model=mod_str, experience="cvlso", type="coefmap-zcore", ext="nii.gz")
regions_corrmat_filename =  OUTPUT.format(data='mwp1-gs', model=mod_str, experience="refit", type="regions-corrmat", ext="svg")
#feature_importance_filename =  OUTPUT.format(data='mwp1-gs', model="l2lr", experience="cvlso", type="feature-importance", ext="xlsx")

feature_importance_filename =  OUTPUT.format(data='mwp1-gs', model=mod_str, experience="cvlso", type="feature-importance", ext="xlsx")
feature_importance_plot_filename =  OUTPUT.format(data='mwp1-gs', model=mod_str, experience="cvlso", type="feature-importance", ext="svg")
feature_importance_summary_filename =  OUTPUT.format(data='mwp1-gs', model=mod_str, experience="cvlso", type="feature-importance_summary", ext="xlsx")

predictions_filename =  OUTPUT.format(data='mwp1-gs', model=mod_str, experience="cvlso", type="predictions", ext="pkl")



# %% 7.1) Load data
# from nitk.image import vec_to_niimg

datasets = load_dataset()
cv_dict = datasets["cv_lso_dict"]

# Coefs Backward and forward

coefs_cv_vec = np.load(coefmap_cv_npz_filename)
coefs_cv_arr = {k:vec_to_arr(vec, mask_arr) for k, vec in coefs_cv_vec.items()}
fwds_cv_vec = np.load(coefmap_forward_cv_npz_filename)

W_cv = np.asarray([v for k, v in coefs_cv_vec.items() if k != 'ALL'])
w_mean, w_se, w_z, w_tstat, w_pval, w_ci = mean_se_z_t_ci_pval(X=W_cv, mu=0, ddof=0) # Was computed with ddof = 0, keep it
w_mean_arr, w_z_arr, w_se_arr = vec_to_arr(w_mean, mask_arr), vec_to_arr(w_z, mask_arr), vec_to_arr(w_se, mask_arr)
# zscore_arr = nibabel.load(zscore_img_filename).get_fdata()
assert np.allclose(w_z_arr, nibabel.load(zscore_img_filename).get_fdata())


Fwd_cv = np.asarray([v for k, v in fwds_cv_vec.items() if k != 'ALL'])
fwd_mean, fwd_se, fwd_z, fwd_tstat, fwd_pval, fwd_ci = mean_se_z_t_ci_pval(X=Fwd_cv, mu=0)
fwd_mean_arr, fwd_z_arr, fwd_se_arr = vec_to_arr(fwd_mean, mask_arr), vec_to_arr(fwd_z, mask_arr), vec_to_arr(fwd_se, mask_arr)

# VBM

vbm_tstat_arr = nibabel.load(vbm_dirname + '/mulm_tstat.nii.gz').get_fdata()
vbm_pval_maxt_arr = nibabel.load(vbm_dirname + '/mulm_pvals-maxT.nii.gz').get_fdata()

# Searchlight

searchlight_cv_scores_arr = np.load(searchlight_cv_scores_arr_filename)
searchlight_cv_auc_arr = {k.split('_')[0]:searchlight_cv_scores_arr[k] for k in sorted(searchlight_cv_scores_arr) if k.split('_')[1] == 'auc'}

searchlight_auc_arr = np.stack([v for k, v in searchlight_cv_auc_arr.items()])
searchlight_auc_mean_vec, searchlight_auc_se_vec, searchlight_auc_z_vec, searchlight_auc_tstat_vec, searchlight_auc_pval_vec, searchlight_auc_ci_vec= \
    mean_se_z_t_ci_pval(searchlight_auc_arr[:, mask_arr], mu=0.5,  two_tailed=False)
searchlight_auc_mean_arr, searchlight_auc_se_arr = vec_to_arr(searchlight_auc_mean_vec, mask_arr), vec_to_arr(searchlight_auc_se_vec, mask_arr)


assert set(coefs_cv_vec.keys()) - set(cv_dict.keys()) == {"ALL"}

# Region region defined from threshold on z-scores

#coef_img_filename =  OUTPUT.format(data='mwp1-gs', model=mod_str, experience="refit", type="coefmap", ext="nii.gz")
regions_info_filename =  OUTPUT.format(data='mwp1-gs', model=mod_str, experience="cvlso", type="coefmap-zcore_clust_info", ext="csv")
regions_img_filename =  OUTPUT.format(data='mwp1-gs', model=mod_str, experience="cvlso", type="coefmap-zcore_clust_labels", ext="nii.gz")


def _region_name_from_peak_names(peak_names, label):
    peak_names = [n if not n in ['Background'] else None for n in peak_names]
    region_name = "_".join([n.strip().replace(', ',':').replace(' ','-').replace('-Cerebral-Cortex', '').replace('-Cerebral-Cortex', '').replace('-Cerebral-White-Matter', '')
              for n in peak_names if pd.notnull(n)])
    return "%3i__%s" % (label, region_name)

def _lab_from_region_name(name):
    try:
        lab, name = name.split('__')
        return int(lab.strip()), name
    except:
        return None, name


def map_region_names(names=None, mapback=False, default='id'):
    """
    >>> map_region_names([' 48__Precentral-Gyrus_Left',
                          '107__Insular-Cortex_Left', 'missing'])
     ['Cingulate G. L', 'Frontal Operculum L', 'missing']

    >>> map_region_names([' 48__Precentral-Gyrus_Left',
                          '107__Insular-Cortex_Left', 'missing'], default=None)
    ['Cingulate G. L', 'Frontal Operculum L', None]
    """
    orig = \
    [' 48__Precentral-Gyrus_Left',
     '107__Insular-Cortex_Left',
     '115__Angular-Gyrus_Left',
     '113__Postcentral-Gyrus_Left',
     '  1__Middle-Temporal-Gyrus:temporooccipital-part_Right',
     ' 27__Right-Pallidum',
     ' 55__Subcallosal-Cortex_Left',
     ' 95__Superior-Frontal-Gyrus_Left',
     '  3__Planum-Polare_Right',
     '126__Central-Opercular-Cortex_Left',
     ' 41__Postcentral-Gyrus_Right',
     ' 34__Frontal-Orbital-Cortex_Right',
     ' 83__Occipital-Pole_Left',
     ' 46__Frontal-Pole_Right',
     ' 81__Frontal-Orbital-Cortex_Left',
     ' 87__Temporal-Pole_Left',
     ' 44__Superior-Frontal-Gyrus_Right',
     '117__Precentral-Gyrus_Left',
     ' 26__Cingulate-Gyrus:posterior-division_Right-Hippocampus',
     '110__Inferior-Temporal-Gyrus:anterior-division_Left',
     ' 59__Lingual-Gyrus_Right']

    new = \
    ['Cingulate G. L',
     'Frontal Operculum L',
     'Mid. Temporal G. L',
     'Postcentral G. L',
     'Mid. Temporal G. R',
     'Putamen R',
     'Subcallosal Cort. L',
     'Sup. Frontal G. L',
     'Temporal Pole R',
     'Central Opercular L', # Frontal
     'Precentral G. R', # Frontal
     'Frontal Orbital R',
     'Occipital Pole L', # ?
     'Frontal Pole R',
     'Frontal Pole L',
     'Parahippocampal G. L',
     'Sup. Frontal G. R',
     'Precentral G. L',
     'Hippocampus R',
     'Temporal Fusiform G. L',
     'Lingual G. R']

    if names is None:
        return pd.DataFrame(dict(orig=orig, new=new))

    if mapback:
        mapping = {o:n for o, n in zip(new, orig)}
    else:
        mapping = {o:n for o, n in zip(orig, new)}

    if  default == 'id':
        return [mapping.get(n, n) for n in names]
    else:
        return [mapping.get(n, default) for n in names]


# %% 7.2) Regions' scores + maps of coef, coef's cv zscore

if not os.path.exists(REGIONS_SCORE_FILENAME) or not os.path.exists(REGIONS_INFO_FILENAME):

    regions_label_arr = nibabel.load(regions_img_filename).get_fdata()
    regions_label_vec = regions_label_arr[mask_arr]
    regions_info = pd.read_csv(regions_info_filename)

    # Select clusters where "prop_norm2_weight" >1%
    regions_info = regions_info[regions_info["prop_norm2_weight"] > 0.01]
    assert regions_info.shape[0] == 21

    from sklearn.preprocessing import scale
    residualizer = datasets['residualizer']
    residualizer.fit(datasets['Xim'], datasets['Zres'])
    Xim_res = residualizer.transform(datasets['Xim'], datasets['Zres'])
    Xim_s = scale(Xim_res)
    del Xim_res

    regions_scores = datasets['participants'].copy()
    regions_info_supp = list()
    # regions_w_z_arr = dict()
    # regions_coef_arr = dict()
    # regions_corr_arr = dict() # regions extended with correlation value
    #corr_threshold = .2
    # pval_threshold = 0.05 / mask_arr.sum() # FWER
    # print(-np.log10(pval_threshold))

    for idx, row in regions_info.iterrows():
        region_mask_arr = regions_label_arr ==  row["label"]
        region_mask_vec = regions_label_vec ==  row["label"]
        assert np.all(region_mask_vec == region_mask_arr[mask_arr])
        assert np.sum(region_mask_arr[mask_arr]) == row["size"]

        region_name = _region_name_from_peak_names(
            row[['ROI_HO-cort_peak_pos', 'ROI_HO-cort_peak_neg',
                 'ROI_HO-sub_peak_pos', 'ROI_HO-sub_peak_neg']].tolist(),
            row["label"])
        #region_names.append(region_name)
        # Clusters individuals values using refit ALL coef maps
        print("## %s" % region_name)
        if False and region_name == " 46__Frontal-Pole_Right":
        #if True and region_name == " 48__Precentral-Gyrus_Left":
            break

        # regions' maps (arrays) of zscores
        w_z_arr_masked = w_z_arr.copy()
        w_z_arr_masked[np.logical_not(region_mask_arr)] = 0
        assert np.sum(w_z_arr_masked != 0) == row["size"]
        # # - DEBUG START: keep the max
        # mask_arr_ = np.abs(w_z_arr_masked) == np.abs(w_z_arr_masked).max()
        # mask_arr_.sum()
        # w_z_arr_masked[np.logical_not(mask_arr_)] = 0
        # region_mask_arr = region_mask_arr & mask_arr_
        # assert region_mask_arr.sum() == 1
        # # - DEBUG END
        # regions_w_z_arr[region_name] = w_z_arr_masked

        # regions' maps (arrays) of coefs
        assert np.all(coefs_cv_arr["ALL"][mask_arr] == coefs_cv_vec["ALL"])
        #coef_vec = coefs_cv_vec["ALL"]
        coefs_region_arr = coefs_cv_arr["ALL"].copy()
        coefs_region_arr[np.logical_not(region_mask_arr)] = 0

        # regions' score (refit all)
        region_score = np.dot(Xim_s, coefs_region_arr[mask_arr])
        regions_scores[region_name] = region_score


        # Add Regions informations
        def _peak_mean(x, other):
            peak_idx = np.abs(x).argmax()
            return x[peak_idx], np.mean(x), other[peak_idx], np.mean(other)

        # W Backward
        # w_mean_arr, w_z_arr, w_se_arr
        w_ =  list(_peak_mean(w_mean_arr[region_mask_arr], w_se_arr[region_mask_arr]))

        # W Forward
        # fwd_mean_arr, fwd_se_arr
        fwd_ =  list(_peak_mean(fwd_mean_arr[region_mask_arr], fwd_se_arr[region_mask_arr]))

        # VBM
        # vbm_tstat_arr, vbm_pval_maxt_arr
        vbm_ = list(_peak_mean(vbm_tstat_arr[region_mask_arr], 1 - vbm_pval_maxt_arr[region_mask_arr]))

        # searchlight
        searchlight_ =  list(_peak_mean(searchlight_auc_mean_arr[region_mask_arr], searchlight_auc_se_arr[region_mask_arr]))


        regions_info_supp.append([region_name, map_region_names([region_name])[0]] + w_ + fwd_ + vbm_ + searchlight_)


    regions_info_supp = pd.DataFrame(regions_info_supp,
                 columns=['region_name', 'region_name_map'] +
                 ['w_peak', 'w_mean', 'w_se_peak', 'w_se_mean'] +
                 ['fwd_peak', 'fwd_mean', 'fwd_se_peak', 'fwd_se_mean'] +
                 ['vbmtstat_peak', 'vbmtstat_mean', 'vbm-pval-fwer-maxt_peak', 'vbm-pval-fwer-maxt_mean'] +
                 ['searchlight_peak', 'searchlight_mean', 'searchlight_se_peak', 'searchlight_se_mean']
                 )

    regions_info = pd.concat([regions_info, regions_info_supp], axis=1)
    assert len(regions_info) == 21
    regions_info.to_csv(REGIONS_INFO_FILENAME, index=False)
    regions_scores.to_csv(REGIONS_SCORE_FILENAME, index=False)


    # %% 7.3) Clustering of regions into patterns

    regions_scores = pd.read_csv(REGIONS_SCORE_FILENAME)
    regions_info = pd.read_csv(REGIONS_INFO_FILENAME)
    assert len(regions_info) == 21
    region_names = regions_info['region_name']

    # if not 'regions_scores' in globals():
    #    regions_scores = pd.read_csv(REGIONS_SCORE_FILENAME)
    #    region_names = [n for n in regions_scores.columns if n[3:5] == '__']

    # groups clusters
    df = regions_scores[region_names]
    # Compute the correlation matrix
    # https://stats.stackexchange.com/questions/165194/using-correlation-as-distance-metric-for-hierarchical-clustering
    corr = df.corr()
    # d = 2 * (1 - np.abs(corr))
    d = 2 * (1 - corr)

    from sklearn.cluster import AgglomerativeClustering
    clustering = AgglomerativeClustering(n_clusters=10, linkage='single', affinity="precomputed").fit(d)
    clusters = [list(corr.columns[clustering.labels_==lab]) for lab in set(clustering.labels_)]

    # map_region_names(clusters[0])
    # map_region_names(clusters[1])
    # map_region_names(clusters[2])

    # Map regions to patterns
    patterns = OrderedDict()
    patterns['FrontoTemporoParietal'] = clusters[1]
    patterns['FrontalOrbitalSubcallosal'] = clusters[2]
    # https://www.sciencedirect.com/topics/medicine-and-dentistry/subcallosal-gyrus
    # Subcallosal Cortex:100
    # Frontal Pole:56 Frontal Orbital Cortex:44
    patterns['Temporal'] = clusters[0]
    patterns['Independant'] = np.concatenate(clusters[3:]).tolist()

    assert patterns['FrontoTemporoParietal'] == [' 48__Precentral-Gyrus_Left',
     '107__Insular-Cortex_Left',
     '115__Angular-Gyrus_Left',
     '113__Postcentral-Gyrus_Left',
     ' 95__Superior-Frontal-Gyrus_Left',
     ' 34__Frontal-Orbital-Cortex_Right',
     ' 46__Frontal-Pole_Right',
     ' 87__Temporal-Pole_Left',
     ' 44__Superior-Frontal-Gyrus_Right']

    assert patterns['FrontalOrbitalSubcallosal'] == [' 55__Subcallosal-Cortex_Left', ' 81__Frontal-Orbital-Cortex_Left']

    assert patterns['Temporal'] == ['  3__Planum-Polare_Right',
     ' 26__Cingulate-Gyrus:posterior-division_Right-Hippocampus',
     '110__Inferior-Temporal-Gyrus:anterior-division_Left']

    assert patterns['Independant'] == [' 83__Occipital-Pole_Left',
     '117__Precentral-Gyrus_Left',
     ' 41__Postcentral-Gyrus_Right',
     ' 59__Lingual-Gyrus_Right',
     '  1__Middle-Temporal-Gyrus:temporooccipital-part_Right',
     '126__Central-Opercular-Cortex_Left',
     ' 27__Right-Pallidum']

    # The clustering is save here into: REGIONS_INFO_FILENAME
    if not 'pattern_name' in regions_info.columns:
        patterns_df = pd.DataFrame([[r, p] for p in patterns for r in patterns[p]], columns=['region_name', 'pattern_name'])
        m_ = pd.merge(regions_info, patterns_df, on='region_name')
        assert(np.all(m_.region_name == regions_info.region_name))
        regions_info.insert(3, 'pattern_name', m_['pattern_name'])
        regions_info.to_csv(REGIONS_INFO_FILENAME, index=False)

    # regions_info = pd.read_csv(REGIONS_INFO_FILENAME)
    reordered = np.concatenate([region_name for region_name in patterns.values()]).tolist()


    #reordered = np.concatenate(clusters)
    R = corr.loc[reordered, reordered]
    cmap = sns.color_palette("RdBu_r", 11)
    f, ax = plt.subplots(figsize=(5.5, 4.5))
    # Draw the heatmap with the mask and correct aspect ratio
    _ = sns.heatmap(R, mask=None, cmap=cmap, vmax=1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})


    # %% 7.4) Plot Patterns (Use forward z-map))


    regions_label_arr = nibabel.load(regions_img_filename).get_fdata()
    #regions_label_vec = regions_label_arr[mask_arr]
    fwds_cv_vec = np.load(coefmap_forward_cv_npz_filename)

    #W_cv = np.asarray([v for k, v in coefs_cv_vec.items() if k != 'ALL'])
    #w_mean, w_se, w_z, w_tstat, w_pval, w_ci = mean_se_z_t_ci_pval(X=W_cv, mu=0)

    Fwd_cv = np.asarray([v for k, v in fwds_cv_vec.items() if k != 'ALL'])
    fwd_mean, fwd_se, fwd_z, fwd_tstat, fwd_pval, fwd_ci = mean_se_z_t_ci_pval(X=Fwd_cv, mu=0)
    fwd_mean_arr, fwd_z_arr, fwd_se_arr = vec_to_arr(fwd_mean, mask_arr), vec_to_arr(fwd_z, mask_arr), vec_to_arr(fwd_se, mask_arr)


    from nilearn.plotting import plot_glass_brain
    w_z_arr = nibabel.load(zscore_img_filename).get_fdata()

    others = list(set(region_names) - set(clusters[0] + clusters[1]))

    def _plot_net(region_names, map_arr, label_arr):
        net_arr = np.zeros(map_arr.shape)
        for lab in [int(c.split('__')[0].strip()) for c in region_names]:
            values = map_arr[label_arr == lab]
            print("%i\t%.3f\t%.3f\t%.3f\t%i\t%i" % (lab, np.min(values), np.mean(values), np.max(values), np.sum(values<0), np.sum(values>=0)))
            net_arr[label_arr == lab] = values
        net_niimg = new_img_like(mask_img, net_arr)
        values = net_arr[net_arr != 0]
        print("%s\t%.3f\t%.3f\t%.3f\t%i\t%i" % ("net", np.min(values), np.mean(values), np.max(values), np.sum(values<0), np.sum(values>=0)))

        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(11.69, 1 * 11.69 * .4))
        plot_glass_brain(net_niimg, plot_abs=False, #threshold=3,
                     symmetric_cbar=False, colorbar=True, figure=fig, axes=axes,
                     #title="Z-map of forward transformation of Enet-TV coefﬁcients"
                     )

    _plot_net(region_names=patterns['FrontoTemporoParietal'], map_arr=fwd_z_arr, label_arr=regions_label_arr)
    pattern_fwdzmap_filename_ = pattern_fwdzmap_filename.format(pattern='FrontoTemporoParietal')
    plt.savefig(pattern_fwdzmap_filename_.replace(".svg", ".png"), bbox_inches='tight', pad_inches=0)
    plt.savefig(pattern_fwdzmap_filename_, bbox_inches='tight', pad_inches=0)
    # lab       min   mean    max   nneg    npos
    # 48	-32.722	-6.396	8.550	2439	172
    # 107	-35.812	-8.887	4.580	1850	32
    # 115	-19.260	-5.128	7.066	1922	119
    # 113	-26.854	-8.120	5.918	1071	37
    # 95	-28.261	-9.699	-0.164	425	    0
    # 34	-26.081	-6.760	2.261	489	    7
    # 46	-18.464	-8.719	0.394	232	    1
    # 87	-13.008	-5.534	3.564	312	    18
    # 44	-28.269	-9.902	1.611	202	    3
    # net	-35.812	-7.100	8.550	8942	389

    _plot_net(region_names=patterns['FrontalOrbitalSubcallosal'], map_arr=fwd_z_arr, label_arr=regions_label_arr)
    pattern_fwdzmap_filename_ = pattern_fwdzmap_filename.format(pattern='FrontalOrbitalSubcallosal')
    plt.savefig(pattern_fwdzmap_filename_.replace(".svg", ".png"), bbox_inches='tight', pad_inches=0)
    plt.savefig(pattern_fwdzmap_filename_, bbox_inches='tight', pad_inches=0)
    # 55	-7.493	3.218	13.406	34	196
    # 81	-6.581	1.484	12.948	118	216
    # net	-7.493	2.191	13.406	152	412


    _plot_net(region_names=patterns['Temporal'], map_arr=fwd_z_arr, label_arr=regions_label_arr)
    pattern_fwdzmap_filename_ = pattern_fwdzmap_filename.format(pattern='Temporal')
    plt.savefig(pattern_fwdzmap_filename_.replace(".svg", ".png"), bbox_inches='tight', pad_inches=0)
    plt.savefig(pattern_fwdzmap_filename_, bbox_inches='tight', pad_inches=0)
    # 3	    -16.746	-0.682	8.695	266	228
    # 26	-9.787	 2.887	11.982	20	131
    # 110	-10.041	-2.789	5.651	203	35
    # net	-16.746	-0.640	11.982	489	394

    _plot_net(region_names=patterns['Independant'], map_arr=fwd_z_arr, label_arr=regions_label_arr)
    pattern_fwdzmap_filename_ = pattern_fwdzmap_filename.format(pattern='Independant')
    plt.savefig(pattern_fwdzmap_filename_.replace(".svg", ".png"), bbox_inches='tight', pad_inches=0)
    plt.savefig(pattern_fwdzmap_filename_, bbox_inches='tight', pad_inches=0)
    # 83	-21.113	-6.290	0.670	428	4
    # 117	-12.325	-2.098	6.771	240	82
    # 41	-3.610	 2.465	13.112	41	310
    # 59	-8.070	-2.254	8.062	146	47
    # 1	    -10.825	 0.871	13.858	324	626
    # 126	-8.321	-0.447	9.641	297	237
    # net	-21.113	0.300	13.858	1524	2048



# %% 8) Regions' importance


if not os.path.exists(feature_importance_filename) or not os.path.exists(predictions_filename):

    coefs_cv_vec = np.load(coefmap_cv_npz_filename)
    coefs_cv_arr = {k:vec_to_arr(vec, mask_arr) for k, vec in coefs_cv_vec.items()}
    regions_label_arr = nibabel.load(regions_img_filename).get_fdata()

    #region_names = regions_info["region_name"]

    # %% 8.1) Precompute regions' cv(train/test) scores

    regions_info = pd.read_csv(REGIONS_INFO_FILENAME)
    regions_scores_cv = dict()
    for i, (fold_name, (train, test)) in enumerate(cv_dict.items()):
         print("## %s (%i/%i)" % (fold_name, i+1, len(cv_dict)))
         residualizer = datasets['residualizer']
         X_train = residualizer.fit_transform(datasets['Xim'][train], datasets['Zres'][train])
         X_test = residualizer.transform(datasets['Xim'][test], datasets['Zres'][test])
         scaler = preprocessing.StandardScaler()
         X_train = scaler.fit_transform(X_train)
         X_test = scaler.transform(X_test)

         for idx, row in regions_info.iterrows():
             region_name = row["region_name"]
             region_mask_arr = regions_label_arr ==  row["label"]

             coef_arr_masked_fold = coefs_cv_arr[fold_name].copy()
             coef_arr_masked_fold[np.logical_not(region_mask_arr)] = 0
             region_score_train = np.dot(X_train, coef_arr_masked_fold[mask_arr])
             region_score_test = np.dot(X_test, coef_arr_masked_fold[mask_arr])
             regions_scores_cv[(region_name, fold_name, 'train')] = region_score_train
             regions_scores_cv[(region_name, fold_name, 'test')] = region_score_test


   # %% 8.2) Compute regions'importance

    region_names = regions_info["region_name"]
    assert len(region_names) == 21

    rm_region_names =  {c:[c] for c in regions_info["region_name"]}
    rm_region_names["all"] = []

    patterns = {p:regions_info.loc[regions_info.pattern_name == p, "region_name"].to_list() for p in regions_info.pattern_name.unique()}
    rm_region_names.update(patterns)

    # patterns_names = ["Atrophy", "Atrophy_r", "Increase", "Independant", "Independant_r"]
    # patterns_names = ["Atrophy", "Increase", "Independant"]

    scores = list() # Macro scores
    predictions = dict() # For micro measure

    for i, (fold_name, (train, test)) in enumerate(cv_dict.items()):
        print("## %s (%i/%i)" % (fold_name, i+1, len(cv_dict)))

        y_train = datasets['y'][train]
        y_test = datasets['y'][test]

        residualizer = datasets['residualizer']
        X_train = residualizer.fit_transform(datasets['Xim'][train], datasets['Zres'][train])
        X_test = residualizer.transform(datasets['Xim'][test], datasets['Zres'][test])
        scaler = preprocessing.StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # 1) Refit origial models

        # Check with original coeficients: enettv_full
        lr = lm.LogisticRegression(C=10, class_weight='balanced', fit_intercept=False)
        lr.coef_ = coefs_cv_vec[fold_name][None, :]
        lr.intercept_ = 0
        lr.classes_ = np.array([0., 1.])
        score_pred_test = lr.decision_function(X_test)
        y_pred_test = lr.predict(X_test)
        bacc = metrics.balanced_accuracy_score(y_test, y_pred_test)
        auc = metrics.roc_auc_score(y_test, score_pred_test)
        scores.append(["enettv_full", fold_name, auc, bacc])
        predictions[("enettv_full", fold_name, 'score_pred_test')] = score_pred_test
        predictions[("enettv_full", fold_name, 'label_pred_test')] = y_pred_test

        # L2LR
        lr = lm.LogisticRegression(C=10, class_weight='balanced', fit_intercept=False)
        lr.fit(X_train, y_train)
        score_pred_test = lr.decision_function(X_test)
        y_pred_test = lr.predict(X_test)
        bacc = metrics.balanced_accuracy_score(y_test, y_pred_test)
        auc = metrics.roc_auc_score(y_test, score_pred_test)
        scores.append(["l2lr_full", fold_name, auc, bacc])
        predictions[("l2lr_full", fold_name, 'score_pred_test')] = score_pred_test
        predictions[("l2lr_full", fold_name, 'label_pred_test')] = y_pred_test

        # # Refit L2LR on full image
        # for rm_pattern_name, rm_names in rm_region_names.items():
        #     # rm_pattern_name, rm_names = 'Atrophy', rm_region_names['Atrophy']
        #     feat_mask = np.ones(regions_label_vec.shape, dtype=bool)
        #     for lab in [int(c.split('__')[0].strip()) for c in rm_names]:
        #         print(lab)
        #         feat_mask[regions_label_vec == lab] = False

        #     test_arr = np.zeros(mask_arr.shape)
        #     test_arr[mask_arr] = np.logical_not(feat_mask).astype(int)
        #     new_img_like(mask_img, test_arr).to_filename("/tmp/test.nii.gz")
        #     plot_glass_brain(new_img_like(mask_img, test_arr),
        #                      threshold=0, vmax=1,
        #                      plot_abs=False, colorbar=False, cmap=None)
        #     # L2LR
        #     lr = lm.LogisticRegression(C=10, class_weight='balanced', fit_intercept=False)
        #     lr.fit(X_train[:, feat_mask], y_train)
        #     score_pred_test = lr.decision_function(X_test[:, feat_mask])
        #     y_pred_test = lr.predict(X_test[:, feat_mask])
        #     bacc = metrics.balanced_accuracy_score(y_test, y_pred_test)
        #     auc = metrics.roc_auc_score(y_test, score_pred_test)
        #     scores.append(["l2lr_rm-%s" % rm_pattern_name, fold_name, auc, bacc])


        # 2) Precompute Feature importance Lasso path using LARS
        X_train = np.vstack([regions_scores_cv[(region_name, fold_name, 'train')] for region_name in region_names]).T
        X_test = np.vstack([regions_scores_cv[(region_name, fold_name, 'test')] for region_name in region_names]).T
        cs, active, coefs, aucs, baccs = l1_lr_path(X_train, y_train, X_test, y_test)
        regions_lars_rank = {region_names[idx]:  [rank, auc] for idx, rank, auc in zip(active, range(1, len(active + 1)),  aucs)}
        # regions_lars_rank.get(' 27__Right-Pallidum', None)
        # regions_lars_rank.get('toto', None)

        # 3) Feature importance by removing pattern
        for rm_pattern_name, rm_names in rm_region_names.items():

            # Remove regions
            keep  = list(set(region_names) - set(rm_names))
            print(rm_pattern_name, len(keep))

            X_train = np.vstack([regions_scores_cv[(region_name, fold_name, 'train')] for region_name in keep]).T
            X_test = np.vstack([regions_scores_cv[(region_name, fold_name, 'test')] for region_name in keep]).T

            # lr = make_pipeline(preprocessing.StandardScaler(),
            #                   lm.LogisticRegression(C=10, class_weight='balanced', fit_intercept=False))
            lr = lm.LogisticRegression(C=1, class_weight='balanced', fit_intercept=False)
            lr.fit(X_train, y_train)
            # lr.coef_ = np.ones(lr.coef_.shape)
            score_pred_test = lr.decision_function(X_test)
            y_pred_test = lr.predict(X_test)

            bacc_rm = metrics.balanced_accuracy_score(y_test, y_pred_test)
            auc_rm = metrics.roc_auc_score(y_test, score_pred_test)

            # 3) Feature importance by AUC of pattern
            auc_reg, bacc_reg = None, None
            if len(rm_names) > 0:
                keep  = list(set(rm_names))
                X_train = np.vstack([regions_scores_cv[(region_name, fold_name, 'train')] for region_name in keep]).T
                X_test = np.vstack([regions_scores_cv[(region_name, fold_name, 'test')] for region_name in keep]).T

                # lr = make_pipeline(preprocessing.StandardScaler(),
                #                   lm.LogisticRegression(C=10, class_weight='balanced', fit_intercept=False))
                lr = lm.LogisticRegression(C=1, class_weight='balanced', fit_intercept=False)
                lr.fit(X_train, y_train)
                # lr.coef_ = np.ones(lr.coef_.shape)
                score_pred_test = lr.decision_function(X_test)
                y_pred_test = lr.predict(X_test)

                bacc_reg = metrics.balanced_accuracy_score(y_test, y_pred_test)
                auc_reg = metrics.roc_auc_score(y_test, score_pred_test)

            scores.append([rm_pattern_name, fold_name, auc_rm, bacc_rm, auc_reg, bacc_reg] +\
                           regions_lars_rank.get(rm_pattern_name, [None, None]))


    _scores_ = scores.copy()

    with open(predictions_filename, 'wb') as fd:
        pickle.dump(predictions, fd)

    scores = pd.DataFrame(scores, columns=['region_name', 'fold', 'auc_rm', 'bacc_rm',
                                           'auc_reg', 'bacc_reg', 'lars_rank', 'lars_auc']).set_index(['fold'])

    # Compute feature importance, ie, delta with 'all'
    deltas = pd.DataFrame()
    for region_name, df in scores.groupby("region_name"):
        deltas_ = df[['auc_rm', 'bacc_rm']] - scores.loc[scores.region_name == 'all', ['auc_rm', 'bacc_rm']]
        deltas_.insert(0, 'region_name', region_name)
        deltas = deltas.append(deltas_)

    deltas = deltas.rename(columns={'auc_rm':'auc_loss', 'bacc_rm':'bacc_loss'})
    scores_ = pd.merge(scores.reset_index(), deltas.reset_index())
    assert scores_.shape[0] == scores.shape[0]
    scores = scores_

    # Add Pattern name
    scores = pd.merge(scores, regions_info[["region_name", "pattern_name"]], on="region_name", how='left')
    # And region simplified name
    scores.insert(2, 'region_name_mapped', map_region_names(scores["region_name"]))

    scores_mean = scores.groupby("region_name").mean().reset_index()
    scores_mean.insert(1, 'region_name_map', map_region_names(scores_mean["region_name"]))

    scores_std = scores.groupby("region_name").std().reset_index()
    scores_std.insert(1, 'region_name_map', map_region_names(scores_std["region_name"]))

    with pd.ExcelWriter(feature_importance_filename) as writer:
        scores.to_excel(writer, sheet_name='folds', index=False)
        scores_mean.to_excel(writer, sheet_name='mean', index=False)
        scores_std.to_excel(writer, sheet_name='std', index=False)


# %% 8.3) Add features inportance in REGION_INFO + Regions summary

def read_feature_importance_with_order(scores):

    # Sort region by Pattern and fetures importance
    pattern_name_sorted_ = dict()

    for pattern_name, df in scores.groupby('pattern_name'):
        print(pattern_name)
        df_sorted = df.groupby('region_name').mean().sort_values(by='auc_loss', ascending=True)
        pattern_name_sorted_[pattern_name] = df_sorted.index.to_list()

    # Manually re-order pattern for figures
    pattern_name_sorted = OrderedDict()
    pattern_name_sorted['FrontoTemporoParietal'] = pattern_name_sorted_['FrontoTemporoParietal']
    pattern_name_sorted['FrontalOrbitalSubcallosal'] = pattern_name_sorted_['FrontalOrbitalSubcallosal']
    pattern_name_sorted['Temporal'] = pattern_name_sorted_['Temporal']
    pattern_name_sorted['Independant'] = pattern_name_sorted_['Independant']
    region_name_sorted = np.concatenate([v for v in pattern_name_sorted.values()]).tolist()

    # Tune plot order
    region_pattern_name_sorted = \
        ['FrontoTemporoParietal'] + pattern_name_sorted['FrontoTemporoParietal'] + \
        ['FrontalOrbitalSubcallosal'] + pattern_name_sorted['FrontalOrbitalSubcallosal'] + \
        ['Temporal'] + pattern_name_sorted['Temporal'] + \
        ['Independant'] + pattern_name_sorted['Independant']

    return region_name_sorted, region_pattern_name_sorted


if not os.path.exists(REGIONS_INFO_FILENAME.replace(".csv", "_summary.xlsx")):

    regions_info = pd.read_csv(REGIONS_INFO_FILENAME)
    regions_info = regions_info.loc[:, :'searchlight_se_mean']

    feature_importance_mean = pd.read_excel(feature_importance_filename, sheet_name='mean')
    feature_importance_std = pd.read_excel(feature_importance_filename, sheet_name='std')


    scores_ = pd.merge(regions_info[['region_name', 'region_name_map', 'pattern_name']], feature_importance_mean)
    region_name_sorted, region_pattern_name_sorted = read_feature_importance_with_order(scores_)
    del scores_


    cols_to_keep = ['region_name', 'region_name_map', 'lars_rank', 'lars_auc', 'auc_loss', 'bacc_loss']
    feature_importance_mean_ = feature_importance_mean.loc[feature_importance_mean['region_name'].isin(regions_info['region_name']), cols_to_keep]
    feature_importance_std_ = feature_importance_std.loc[feature_importance_std['region_name'].isin(regions_info['region_name']), cols_to_keep]
    feature_importance_se_ = feature_importance_std_.copy()
    feature_importance_se_.iloc[:, 2:] = feature_importance_std_.iloc[:, 2:] / np.sqrt(len(cv_dict))

    regions_info_mllosses = pd.merge(feature_importance_mean_, feature_importance_se_, on=['region_name', 'region_name_map'], suffixes=('_mean', '_se'))


    regions_info_ = pd.merge(regions_info, regions_info_mllosses, on=['region_name', 'region_name_map'])
    assert regions_info_.shape == (regions_info.shape[0], regions_info.shape[1] + regions_info_mllosses.shape[1] - 2)
    assert np.all(regions_info_.region_name == regions_info.region_name)

    regions_info = regions_info_

    assert sorted(regions_info['region_name']) == sorted(region_name_sorted)
    regions_info = regions_info.set_index('region_name').loc[region_name_sorted, :].reset_index()
    regions_info.to_csv(REGIONS_INFO_FILENAME, index=False)
    # map_region_names(region_name_sorted)

    # Make a summary
    regions_info = pd.read_csv(REGIONS_INFO_FILENAME)

    cols_to_keep = ['region_name', 'region_name_map',
                    'size','x_center_mni', 'y_center_mni', 'z_center_mni',
                    'vbmtstat_peak', 'vbm-pval-fwer-maxt_peak',
                    'searchlight_peak', 'searchlight_se_peak',
                    'fwd_peak', 'fwd_se_peak',
                    'auc_loss_mean', 'auc_loss_se',
                    'lars_rank_mean', 'lars_rank_se',
                    'ROIs_HO-cort_prop (%)', 'ROIs_HO-sub_prop (%)']
    regions_info = regions_info[cols_to_keep]

    regions_info.to_excel(REGIONS_INFO_FILENAME.replace(".csv", "_summary.xlsx"), index=False)



# %% 8.3) Plot Regions and patterns importance




if not os.path.exists(feature_importance_plot_filename):

    # scores_s, region_name_sorted, region_pattern_name_sorted, pattern_name_sorted = \
    #     read_feature_importance_with_order(feature_importance_plot_filename)

    scores_s["1 - AUC"] = 1 - scores_s['auc_reg']
    #regions_loss['region_name'] = map_region_names(regions_loss["region_name"])
    scores_s['Loss of ROC-AUC'] = scores_s['auc_loss']
    scores_s['Rank in Lasso LARS'] = scores_s['lars_rank']

    # Palette
    sns.set_style("darkgrid")
    pal_ = sns.color_palette()
    palette_ = {'FrontalOrbitalSubcallosal': pal_[0], 'FrontalOrbitalSubcallosal': pal_[3],
                'Temporal': pal_[1], 'Independant': pal_[2]}
    lcolor ="black"

    #fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(9, 5))
    fig = plt.figure(figsize=(9, 5))
    from matplotlib import gridspec
    gs = gridspec.GridSpec(1, 3, width_ratios=[2, 3, 1])
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1], sharey=ax1)
    ax3 = plt.subplot(gs[2], sharey=ax1)

    #ax1, ax2, ax3 = plt.subplot(gs[0]), plt.subplot(gs[1]), plt.subplot(gs[2])

    # Default figure size
    # fig.get_size_inches()
    # array([6., 4.]) # width, height
    # figsize=(11.69, K * 11.69 * .4)
    # A4: 8.3 x 11.7
    # https://css.paperplaza.net/conferences/support/page.php
    # Text width 7.0in

    # 1 - AUC
    ax = sns.pointplot(y="region_name_mapped", x='1 - AUC', linestyles='',
                hue='pattern_name', palette=palette_,
                order=map_region_names(region_pattern_name_sorted, default='id'),
                data=scores_s, ax=ax1, legend=False)
    ax.axvline(0.5, ls='-', color="grey", lw=3)

    def add_axvlines(ax, xloc):
        ax.plot([xloc, xloc], [-0.5, 9.5], ls='-', color=palette_['FrontalOrbitalSubcallosal'], lw=4)
        ax.plot([xloc, xloc], [9.5, 12.5], ls='-', color=palette_['FrontalOrbitalSubcallosal'], lw=4)
        ax.plot([xloc, xloc], [12.5, 16.5], ls='-', color=palette_['Temporal'], lw=4)
        ax.plot([xloc, xloc], [16.5, 24.5], ls='-', color=palette_['Independant'], lw=4)

    xmin, xmax = ax.get_xlim()
    add_axvlines(ax, xmin)

    def add_axhlines(ax):
        ax.yaxis.label.set_visible(False)
        ax.get_legend().remove()
        ax.axhline(-0.5, ls='-', color=lcolor, lw=2) # FrontalOrbitalSubcallosal
        ax.axhline(0.5, ls='-', color=lcolor, lw=0.5)
        ax.axhline(9.5, ls='-', color=lcolor, lw=2)  # 'FrontalOrbitalSubcallosal'
        ax.axhline(10.5, ls='-', color=lcolor, lw=0.5)
        ax.axhline(12.5, ls='-', color=lcolor, lw=2) # 'Temporal'
        ax.axhline(13.5, ls='-', color=lcolor, lw=0.5)
        ax.axhline(16.5, ls='-', color=lcolor, lw=2) # 'Independant'
        ax.axhline(17.5, ls='-', color=lcolor, lw=0.5)
        ax.axhline(24.5, ls='-', color=lcolor, lw=2) # 'Independant'
    add_axhlines(ax)


    # Loss of ROC-AUC
    ax = sns.pointplot(y="region_name_mapped", x='Loss of ROC-AUC', linestyles='',
                hue='pattern_name', palette=palette_,
                order=map_region_names(region_pattern_name_sorted), default='id',
                data=scores_s, ax=ax2, legend=False)
    #ax.set_xlim(-.22, 0.01)
    ax.axvline(0, ls='-', color="grey", lw=3)
    ax.get_yaxis().set_visible(False)
    add_axhlines(ax)
    # Plot sum of regions losses
    ax.scatter([-0.063873, -0.027768, -0.008939, -0.048966],
               [0, 10, 13, 17],
               c="black",
               marker='x', s=50)


    # Rank in Lasso LARS
    ax = sns.pointplot(y="region_name_mapped", x='Rank in Lasso LARS', linestyles='',
            hue='pattern_name', palette=palette_,
            order=map_region_names(region_pattern_name_sorted, default='id'),
            data=scores_s, ax=ax3, legend=False)
    ax.get_yaxis().set_visible(False)
    ax.axvline(scores_s['Rank in Lasso LARS'].mean(), ls='-', color="grey", lw=3)

    add_axhlines(ax)
    xmin, xmax = ax.get_xlim()
    add_axvlines(ax, xmax)
    # Final tunning
    ax.set_ylim(25, -1)
    fig.tight_layout()
    #plt.show()

    # plt.savefig(feature_importance_plot_filename)


# # %% 8.4) Table of Regions and patterns importance

# if not os.path.exists(feature_importance_summary_filename):

#     scores, region_name_sorted, region_pattern_name_sorted, pattern_name_sorted = \
#         read_feature_importance_with_order(feature_importance_plot_filename)

#     # Compute stats

#     scores_means = scores.groupby('region_name').mean().reset_index()
#     scores_se = (scores.groupby('region_name').std() / len(scores.fold.unique())).reset_index()
#     scores_stat = pd.merge(scores_means, scores_se, on='region_name', suffixes=('_mean', '_se'))

#     # Merge with regions informations

#     regions_info = pd.read_csv(REGIONS_INFO_FILENAME)
#     regions_info = regions_info[['region_name', 'size', 'x_center_mni', 'y_center_mni', 'z_center_mni']]
#     scores_stat = pd.merge(scores_stat, regions_info, on='region_name', how='left')

#     # Reorder rows
#     assert set(scores_stat['region_name']) == set(region_pattern_name_sorted)
#     scores_stat = scores_stat.set_index('region_name').loc[region_pattern_name_sorted, :].reset_index()

#     # Select/reorder columns, rename regions
#     scores_stat["region_name_mapped"] = map_region_names(scores_stat['region_name'], default='id')

#     scores_stat = scores_stat[
#         ["region_name_mapped",
#          'x_center_mni', 'y_center_mni', 'z_center_mni', 'size',
#          'auc_reg_mean', 'auc_reg_se',
#          'auc_loss_mean', 'auc_loss_se',
#          'lars_rank_mean', 'lars_rank_se']]

#     scores_stat.to_excel(feature_importance_summary_filename)

#     #%% Build Final table
#     """

#     Sum of regions
#                         auc_delta  bacc_delta
#     region_name
#     FrontalOrbitalSubcallosal     -0.063873   -0.033568
#     FrontalOrbitalSubcallosal    -0.027768   -0.017199
#     Temporal   -0.008939   -0.008546
#     Independant         -0.048966   -0.031595

#     # Patterns losses
#                             auc      bacc  auc_delta  bacc_delta
#     region_name
#     FrontalOrbitalSubcallosal    0.625218  0.564342  -0.159615   -0.140471
#     FrontalOrbitalSubcallosal   0.751646  0.675008  -0.033187   -0.029806
#     Independant        0.733176  0.653362  -0.051657   -0.051452
#     Temporal  0.777694  0.695929  -0.007139   -0.008885
#     """


# %% 8.5) Re-plot Correlation matrix organized as clusters

if not os.path.exists(regions_corrmat_filename):
    regions_scores = pd.read_csv(REGIONS_SCORE_FILENAME)
    region_names = [n for n in regions_scores.columns if n[3:5] == '__']
    assert set(region_name_sorted) == set(region_names)

    assert len(region_names) == 21

    # groups clusters
    df = regions_scores[region_name_sorted]
    df.columns = map_region_names(df.columns)
    # df.columns = [_clean_name(name) for name in df.columns]
    # Compute the correlation matrix
    # https://stats.stackexchange.com/questions/165194/using-correlation-as-distance-metric-for-hierarchical-clustering
    corr = df.corr()
    # d = 2 * (1 - np.abs(corr))
    d = 2 * (1 - corr)

    reordered = map_region_names(region_name_sorted)
    # reordered = [_clean_name(name) for name in reordered]
    R = corr.loc[reordered, reordered]
    cmap = sns.color_palette("RdBu_r", 11)
    f, ax = plt.subplots(figsize=(5.5, 4.5))
    # Draw the heatmap with the mask and correct aspect ratio
    ax = sns.heatmap(R, mask=None, cmap=cmap, vmax=1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    # plt.plot([0, 9, 9, 0, 0], [0, 0, 9, 9, 0], ls='-', color='black', lw=2)
    # plt.plot([9, 11, 11, 9, 9], [9, 9, 11, 11, 9], ls='-', color='black', lw=2)
    # plt.plot([11, 14, 14, 11, 11], [11, 11, 14, 14, 11], ls='-', color='black', lw=2)
    # plt.plot([14, 21, 21, 14, 14], [14, 14, 21, 21, 14], ls='-', color='black', lw=2)

    # plt.savefig(regions_corrmat_filename)




# %% 9) Learning curves

xls_filename = OUTPUT.format(data='mwp1-gs', model="all", experience="cvlso-learningcurves", type="scores", ext="xlsx")
plot_filename = OUTPUT.format(data='mwp1-gs', model="all", experience="cvlso-learningcurves", type="scores", ext="svg")
xls_densenet121_filename = OUTPUT.format(data='mwp1-gs', model="denseNet121", experience="cvlso-learningcurves", type="scores", ext="xlsx")
#models_filename = OUTPUT.format(data='mwp1-gs', model="enettv", experience="cvlso", type="scores-coefs", ext="pkl")
mapreduce_sharedir =  OUTPUT.format(data='mwp1-gs', model="all", experience="cvlso-learningcurves", type="models", ext="mapreduce")
cv_filename =  OUTPUT.format(data='mwp1-gs', model="all", experience="cvlso-learningcurves", type="train-test-folds-by-size", ext="json")

if not os.path.exists(xls_filename):

    #%% 9.1) Fit models

    print("# %% 9) Learning curves")
    datasets = load_dataset()
    dataset = DATASET

    mask_arr = datasets['mask_arr']
    mask_img = datasets['mask_img']

    # Estimators

    estimators_dict = dict()


    # Enet-TV

    linoperatortv_filename = os.path.join(INPUT_DIR, "mni_cerebrum-gm-mask_1.5mm_Atv.npz")
    Atv = LinearOperatorNesterov(filename=linoperatortv_filename)
    # assert np.allclose(Atv.get_singular_values(0), 11.940682881834617) # whole brain
    # assert np.allclose(Atv.get_singular_values(0), 11.928817868042772) # rm brainStem+cerrebelum

    # Small range
    alphas = [0.1]
    l1l2ratios = [0.01]
    tvl2ratios = [0.1]

    import itertools
    estimators_enettv = dict()
    for alpha, l1l2ratio, tvl2ratio in itertools.product(alphas, l1l2ratios, tvl2ratios):
        print(alpha, l1l2ratio, tvl2ratio)
        l1, l2, tv = ratios_to_param(alpha, l1l2ratio, tvl2ratio)
        key = "enettv_%.3f:%.6f:%.6f" % (alpha, l1l2ratio, tvl2ratio)
        conesta = algorithms.proximal.CONESTA(max_iter=10000)
        estimator = estimators.LogisticRegressionL1L2TV(l1, l2, tv, Atv, algorithm=conesta,
                                                class_weight="auto", penalty_start=0)
        estimators_enettv[key] = estimator

    estimators_dict.update(estimators_enettv)


    # L2 LR

    Cs = [10]
    estimators_l2 = {"l2lr_C:%.6f" % C: lm.LogisticRegression(C=C, class_weight='balanced', fit_intercept=False) for C in Cs}
    estimators_dict.update(estimators_l2)


    # ElasticNet(CV)

    lrenet_cv = GridSearchCV(estimator=lm.SGDClassifier(loss='log', penalty='elasticnet'),
                 param_grid={'alpha': 10. ** np.arange(-1, 2),
                             'l1_ratio': [.1, .5, .9]},
                             cv=3, n_jobs=1)
    estimators_dict.update({'enetlr_cv':lrenet_cv})


    # SVM RBF

    svmrbf_cv= GridSearchCV(svm.SVC(),
                 # {'kernel': ['poly', 'rbf'], 'C': 10. ** np.arange(-3, 3)},
                 {'kernel': ['rbf'], 'C': 10. ** np.arange(-1, 2)},
                 cv=3, n_jobs=1)
    estimators_dict.update({'svmrbf_cv':svmrbf_cv})


    # MLP

    mlp_param_grid = {"hidden_layer_sizes":
                  [(100, ), (50, ), (25, ), (10, ), (5, ),          # 1 hidden layer
                   (100, 50, ), (50, 25, ), (25, 10, ), (10, 5, ),  # 2 hidden layers
                   (100, 50, 25, ), (50, 25, 10, ), (25, 10, 5, )], # 3 hidden layers
                  "activation": ["relu"], "solver": ["sgd"], 'alpha': [0.0001]}
    mlp_cv = GridSearchCV(estimator=MLPClassifier(random_state=1),
                 param_grid=mlp_param_grid,
                 cv=3, n_jobs=1)
    estimators_dict.update({'mlp_cv':mlp_cv})


    # Gradient Boosting xgboost
    from xgboost import XGBClassifier

    xgboost_cv = GridSearchCV(estimator=XGBClassifier(random_state=0, n_jobs=1),
                              param_grid={"n_estimators": [10, 50, 100],
                                          "learning_rate":[0.05, 0.1],
                                          "max_depth":[3, 6],
                                          "subsample":[0.8]}, cv=3, n_jobs=1)

    estimators_dict.update({'xgboost_cv':xgboost_cv})


    # LSOCV

    cv_dict = datasets["cv_lso_dict"]
    # cv_dict = {"CV%02d" % fold:split for fold, split in enumerate(cv.split(df['Xim'], df['y']))}
    # cv_dict["ALL"] = [np.arange(df['Xim'].shape[0]), np.arange(df['Xim'].shape[0])]

    # Learning-curve LSOCV: stratified for site and site
    n_subjects = datasets['y'].shape[0]
    participants = datasets['participants'][['site', 'dx', 'participant_id']]

    # LSO CV with various sizes stratified for dx and site
    if not os.path.exists(cv_filename):

        train_sizes_ = [len(train) for fold, (train, test) in cv_dict.items()]
        size_max_shared_ = np.min(train_sizes_) - (np.min(train_sizes_) % 100)
        size_mean = int(np.mean(train_sizes_).round(0))
        sizes = np.arange(100, size_max_shared_ + 100, 100)
        del train_sizes_, size_max_shared_

        cv_lso_lrncurv = dict()
        cpt_ = 0
        for fold, (train, test) in cv_dict.items():
            participants_train = participants.iloc[train]

            for size in sizes:
                train_sub = sample_stratified(participants_train[['site', 'dx']], size, shuffle=False, random_state=None)

                # Check 1: make sure that max diff of proportion between original
                # dataset and resampled is lower than 1%
                prop_diff_max = \
                    np.max(np.abs(participants_train.loc[:, ['site', 'dx', 'participant_id']].groupby(['site', 'dx']).count() / participants_train.shape[0] -\
                                  participants_train.loc[train_sub, ['site', 'dx', 'participant_id']].groupby(['site', 'dx']).count() / len(train_sub)))
                assert prop_diff_max.values[0] < 0.01

                # Check 2: participants_train[indices] == participants[indices]
                assert participants.loc[train_sub, ['site', 'dx', 'participant_id']].equals(
                    participants_train.loc[train_sub, ['site', 'dx', 'participant_id']])

                # Store train_sub for each size for ech fold
                cv_lso_lrncurv["%s-%s" % (fold, size)] = [train_sub, test]
                cpt_ += 1

        # Check
        assert cpt_ == len(sizes) * len(cv_dict) == len(cv_lso_lrncurv)
        del cpt_

        # Add original cvlso with size = mean size
        cv_lso_lrncurv.update({"%s-%s" % (fold, size_mean):v for fold, v in cv_dict.items()})
        assert len(cv_lso_lrncurv) == len(sizes) * len(cv_dict) + len(cv_dict)
        cv_lso_lrncurv_bak = cv_lso_lrncurv

        cv_lso_lrncurv_ = {k:[x.tolist() for x in v] for k, v in cv_lso_lrncurv.items()}
        with open(cv_filename, 'w') as outfile:
            json.dump(cv_lso_lrncurv_, outfile)

    else:
        with open(cv_filename) as json_file:
            cv_lso_lrncurv = json.load(json_file)
        cv_lso_lrncurv = {k:[np.array(x) for x in v] for k, v in cv_lso_lrncurv.items()}

    # Check when created
    if 'cv_lso_lrncurv_bak' in locals():
        assert np.all([np.all([np.all(cv_lso_lrncurv_bak[k][i] == cv_lso_lrncurv[k][i]) for i in range(len(cv_lso_lrncurv[k]))])
                       for k in cv_lso_lrncurv])

    #key_values_input = dict_product(estimators_dict, dict(resdualizeYes="yes"), cv_dict)
    key_values_input = dict_product(estimators_dict, dict(resdualizeYes="yes"), cv_lso_lrncurv,
        {'Xim_%s' % dataset :datasets['Xim']}, {'y_%s' % dataset :datasets['y']},
        {'Zres_%s' % dataset :datasets['Zres']}, {'Xdemoclin_%s' % dataset :datasets['Xdemoclin']},
        {'residualizer_%s' % dataset :datasets['residualizer']})

    assert (8 * len(cv_dict) + len(cv_dict)) * len(estimators_dict) == len(key_values_input)
    print("Nb Tasks=%i" % len(key_values_input)) # Nb Tasks=702


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
    # key_states = mp.get_states(keys=key_values_input.keys())
    # If some tasks failed, reset them
    # rested_keys = mp.reset(keys=key_values_input.keys())


    ###########################################################################
    # 3) Centralized Mapper
    # start_time = time.time()
    # key_vals_output = MapReduce(n_jobs=NJOBS, pass_key=True, verbose=20).map(fit_predict, key_values_input)
    # print("#  Centralized mapper completed in %.2f sec" % (time.time() - start_time))

    ###############################################################################
    # 4) Reducer: output key/value pairs => CV scores""")

    if key_vals_output is not None:
        # mp.make_archive()
        # make_archive(mapreduce_sharedir, "zip", root_dir=os.path.dirname(mapreduce_sharedir), base_dir=os.path.basename(mapreduce_sharedir))

        print("# Distributed mapper completed in %.2f sec" % (time.time() - start_time))
        cv_scores_all = reduce_cv_classif(key_vals_output, cv_lso_lrncurv, y_true=datasets['y'], index_fold=2)

        # Split fold into fold x size
        fold, size = zip(*[fold.split('-') for fold in cv_scores_all.fold])
        cv_scores_all['fold'] = fold
        cv_scores_all.insert(1, 'size', [int(s) for s in size])

        cv_scores = cv_scores_all[cv_scores_all.fold != "ALL"]
        cv_scores_mean = cv_scores.groupby(["param_1", "param_0", "size", "pred"]).mean().reset_index()
        cv_scores_std = cv_scores.groupby(["param_1", "param_0", "size", "pred"]).std().reset_index()
        cv_scores_mean.sort_values(["param_1", "param_0", "size", "pred"], inplace=True, ignore_index=True)
        cv_scores_std.sort_values(["param_1", "param_0", "size","pred"], inplace=True, ignore_index=True)
        print(cv_scores_mean)

        # with open(models_filename, 'wb') as fd:
        #   pickle.dump(key_vals_output, fd)

        with pd.ExcelWriter(xls_filename) as writer:
            cv_scores.to_excel(writer, sheet_name='folds', index=False)
            cv_scores_mean.to_excel(writer, sheet_name='mean', index=False)
            cv_scores_std.to_excel(writer, sheet_name='std', index=False)

    #%% 9.2) Plot curves

    stats = pd.read_excel(xls_filename)
    stats_densenet = pd.read_excel(xls_densenet121_filename)
    stats = stats.append(stats_densenet, ignore_index=True)


    stats = stats[stats["param_1"].isin(["resdualizeYes"]) & stats["pred"].isin(["test_img"])]
    stats = stats.rename(columns={"param_0":"Model", 'auc':'AUC', 'size':'Size'})
    stats = stats[['Model', 'fold', 'Size', 'AUC', 'bacc']]
    stats.Model =\
    stats.Model.map({'enetlr_cv': 'Enet',
                     'enettv_0.100:0.010000:0.100000':'Enet-TV',
                     'mlp_cv':'MLP',
                     'svmrbf_cv':'SVM-RBF',
                     'l2lr_C:10.000000':'L2',
                     'denseNet121':'DenseNet121',
                     'xgboost_cv':'GradientBoosting'})

    sns.set_style("darkgrid")
    plt.rc('xtick', labelsize=16)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=16)    # fontsize of the tick labels
    plt.rc('legend', fontsize=16)    # legend fontsize
    plt.rc('axes', labelsize=16)    # fontsize of the x and y labels
    hue_order = ['Enet-TV', 'Enet', 'L2', 'GradientBoosting', 'SVM-RBF', 'MLP', 'DenseNet121']
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(11.69, 1 * 11.69 * .5))
    sns.lineplot(x='Size', y='AUC', hue='Model', data=stats, lw=3, hue_order=hue_order)
    axes.set_xlabel('Training sample size')

    plt.savefig(plot_filename.replace(".svg", ".png"), bbox_inches='tight', pad_inches=0)
    plt.savefig(plot_filename, bbox_inches='tight', pad_inches=0)


    stats_all_mean = stats[stats.Size.isin([903])].groupby('Model').mean()[['AUC', 'bacc']]
    stats_all_se = stats[stats.Size.isin([903])].groupby('Model').std(ddof=1)[['AUC', 'bacc']]/ np.sqrt(len(stats.fold.unique()))

    stats_all_se.columns = ["se_%s" % col for col in stats_all_se.columns]

    stats_all = pd.concat([stats_all_mean, stats_all_se], axis=1)
    stats_all = stats_all[['AUC', 'se_AUC', 'bacc', 'se_bacc']]
    stats_all.to_csv('/tmp/classif_perf.csv')

    print(stats_all.round(3))

"""
                    AUC  se_AUC   bacc  se_bacc
Model
DenseNet121       0.637   0.018  0.592    0.017
Enet              0.687   0.024  0.570    0.021
Enet-TV           0.689   0.021  0.617    0.018
GradientBoosting  0.675   0.023  0.601    0.017
L2                0.683   0.020  0.627    0.022
MLP               0.670   0.015  0.621    0.012
SVM-RBF           0.674   0.018  0.589    0.017

"""

"""
stats.groupby(['Model']).count()['fold']
Model
DenseNet121          65
Enet                117
Enet-TV             117
GradientBoosting    117
L2                  117
MLP                 117
SVM-RBF             117
"""


#%% Misc: Clean failed executions
# cd /home/ed203246/data/psy_sbox/analyses/202104_biobd-bsnip_cata12vbm_predict-dx/mwp1-gs_all_cvlso-learningcurves_models.mapreduce
# ls task_*| grep -v lock|grep -v pkl>/tmp/toto
# cat /tmp/toto|while read f; do grep STARTED "$f" ; done
# cat /tmp/toto|while read f; do grep -l STARTED "$f" ; done
# rm /tmp/to_delete
# cat /tmp/toto|while read f; do grep -l STARTED "$f" >> /tmp/to_delete; done
# cat /tmp/to_delete|while read f; do rm "$f" ; done
