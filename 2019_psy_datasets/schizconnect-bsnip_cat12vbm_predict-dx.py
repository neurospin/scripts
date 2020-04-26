"""
# Copy data

cd /home/ed203246/data/psy_sbox/analyses/201906_schizconnect-vip-prague-bsnip-biodb-icaar-start_assemble-all/data/cat12vbm
rsync -azvu triscotte.intra.cea.fr:/neurospin/psy_sbox/analyses/201906_schizconnect-vip-prague-bsnip-biodb-icaar-start_assemble-all/data/cat12vbm/*participants*.csv ./
rsync -azvu triscotte.intra.cea.fr:/neurospin/psy_sbox/analyses/201906_schizconnect-vip-prague-bsnip-biodb-icaar-start_assemble-all/data/cat12vbm/*t1mri_mwp1_mask.nii.gz ./
rsync -azvu triscotte.intra.cea.fr:/neurospin/psy_sbox/analyses/201906_schizconnect-vip-prague-bsnip-biodb-icaar-start_assemble-all/data/cat12vbm/*mwp1_gs-raw_data64.npy ./
"""
# %load_ext autoreload
# %autoreload 2

import os
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
# matplotlib.use('Qt5Cairo')
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

INPUT_PATH = '/neurospin/psy_sbox/analyses/201906_schizconnect-vip-prague-bsnip-biodb-icaar-start_assemble-all/data/cat12vbm/'
OUTPUT_PATH = '/neurospin/psy_sbox/analyses/202004_schizconnect-bsnip_cat12vbm_predict-dx'
NJOBS = 8

# On laptop
if not os.path.exists(INPUT_PATH):
    INPUT_PATH = INPUT_PATH.replace('/neurospin', '/home/ed203246/data')
    OUTPUT_PATH = OUTPUT_PATH.replace('/neurospin', '/home/ed203246/data' )
    NJOBS = 3

os.makedirs(OUTPUT_PATH, exist_ok=True)

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
        estimator_img.coef_ = coefs_cv_cache[key]
    except: # if not fit
        estimator_img.fit(Xim_train, y_train)

    y_test_img = estimator_img.predict(Xim_test)
    score_test_img = estimator_img.decision_function(Xim_test)
    score_train_img = estimator_img.decision_function(Xim_train)
    time_elapsed = round(time.time() - start_time, 2)

    return dict(y_test_img=y_test_img, score_test_img=score_test_img, time=time_elapsed,
                coef_img=estimator_img.coef_)

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

print("""
################################################################################
#
# Dataset: concatenate [schizconnect-vip  bsnip]
#
################################################################################
""")
# SCZ (schizconnect-vip <=> bsnip)

datasets = ['schizconnect-vip', 'bsnip']
dataset, target, target_num = 'schizconnect-vip-bsnip', "diagnosis", "diagnosis_num"
scaling, harmo = 'gs', 'raw'

if not os.path.exists(OUTPUT(dataset, scaling=scaling, harmo=harmo, type="data64", ext="npy")):

    # Read clinical data
    pop = pd.concat([pd.read_csv(INPUT(dataset=dataset, scaling=None, harmo=None, type="participants", ext="csv")) for dataset in datasets], axis=0)
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
    imgs_arr = np.concatenate([np.load(INPUT(dataset=dataset, scaling=scaling, harmo=harmo, type="data64", ext="npy"), mmap_mode='r') for dataset in datasets])[mask_row]
    # Save and relaod in mm
    np.save(OUTPUT(dataset, scaling=scaling, harmo=harmo, type="data64", ext="npy"), imgs_arr)
    del imgs_arr
    import gc
    gc.collect()
    imgs_arr = np.load(OUTPUT(dataset, scaling=scaling, harmo=harmo, type="data64", ext="npy"), mmap_mode='r')
    print(imgs_arr.shape)

    # Recompute mask
    from nitk.image import compute_brain_mask
    ref_img = nibabel.load(INPUT(datasets[0], scaling=None, harmo=None, type="mask", ext="nii.gz"))
    mask_img = compute_brain_mask(imgs_arr, target_img=ref_img)
    mask_img.to_filename(OUTPUT(dataset, scaling=None, harmo=None, type="mask", ext="nii.gz"))
    pop.to_csv(OUTPUT(dataset, scaling=None, harmo=None, type="participants", ext="csv"), index=False)

    mask_arr = mask_img.get_data() != 0
    assert np.sum(mask_arr != 0) == 367689
    print(mask_arr.shape, imgs_arr.squeeze().shape)
    print("Sizes. mask_arr:%.2fGb" % (imgs_arr.nbytes / 1e9))


print("""
###############################################################################
#
# Sensitivity study on schizconnect-vip
#
###############################################################################
""")

dataset, target, target_num = 'schizconnect-vip-bsnip', "diagnosis", "diagnosis_num"
scaling, harmo = 'gs', 'raw'

pop = pd.read_csv(OUTPUT(dataset, scaling=None, harmo=None, type="participants", ext="csv"))
imgs_arr = np.load(OUTPUT(dataset, scaling=scaling, harmo=harmo, type="data64", ext="npy"), mmap_mode='r')
mask_img = nibabel.load(OUTPUT(dataset, scaling=None, harmo=None, type="mask", ext="nii.gz"))
mask_arr = mask_img.get_data() != 0
assert mask_arr.sum() == 367689

print("""
#==============================================================================
# Select dataset 5CV on SCHIZCONNECT-VIP
""")

dataset = 'schizconnect-vip'
NSPLITS = 5

msk = pop.study.isin(['SCHIZCONNECT-VIP'])
assert msk.sum() == 605
Xim = imgs_arr.squeeze()[:, mask_arr][msk]
del imgs_arr
y = pop[target + "_num"][msk].values
print("Sizes. mask_arr:%.2fGb" % (Xim.nbytes / 1e9))
Xdemoclin = Z = np.zeros((Xim.shape[0], 1))

cv = StratifiedKFold(n_splits=NSPLITS, shuffle=True, random_state=3)
cv_dict = {"CV%i" % fold:split for fold, split in enumerate(cv.split(Xim, y))}

print([[lab, np.sum(y == lab)] for lab in np.unique(y)])
#  [[0, 330], [1, 275]]

print("""
#==============================================================================
# l1, l2, enet, filter, rfe
""")

# parameters range:
# from sklearn.svm import l1_min_c
# Cmin = l1_min_c(StandardScaler().fit_transform(Xim), y, loss='log')
# Cs = Cmin * np.logspace(0, -5, 10)

Cs = np.logspace(14, -14, 20)
l1 = {"l2_C:%f" % C: lm.LogisticRegression(C=C, class_weight='balanced', fit_intercept=False) for C in Cs}
l2 = {"l1_C:%f" % C:lm.LogisticRegression(C=C, penalty='l1', class_weight='balanced', fit_intercept=False) for C in Cs}
enet = {"enet_C:%f" % C:lm.LogisticRegression(C=C, penalty='elasticnet', class_weight='balanced', l1_ratio=.1, fit_intercept=False, solver='saga') for C in Cs}

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
key_vals = parallel(fit_predict, args_collection, n_jobs=NJOBS, pass_key=True, verbose=20)

models_filename = OUTPUT(dataset, scaling=scaling, harmo=harmo, type="models-5cv-l1-l2-enet-filter-rfe", ext="pkl")
with open(models_filename, 'wb') as fd:
    pickle.dump(key_vals, fd)

print("""
#------------------------------------------------------------------------------
# Statistics
""")

models_filename = OUTPUT(dataset, scaling=scaling, harmo=harmo, type="models-5cv-l1-l2-enet-filter-rfe", ext="pkl")
with open(models_filename, 'rb') as fd:
    key_vals = pickle.load(fd)

cv_scores = reduce_cv_classif(key_vals, cv_dict, y_true=y)

xls_filename = OUTPUT(dataset, scaling=scaling, harmo=harmo, type="models-5cv-l1-l2-enet-filter-rfe", ext="xlsx")
with pd.ExcelWriter(xls_filename) as writer:
    cv_scores.to_excel(writer, sheet_name='folds', index=False)
    cv_scores.groupby(["param_0"]).mean().to_excel(writer, sheet_name='mean')

cv_scores["param"] = [float(s.split(":")[1]) for s in cv_scores["param_0"]]
cv_scores["algo"] = [s.split("_")[0] for s in cv_scores["param_0"]]

print("""
#------------------------------------------------------------------------------
# plot
""")

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
baseline_l2C1 = cv_scores[(cv_scores.algo == 'l2') & (cv_scores.param == C1_almost)]["auc"]

cv_scores["Model"] = cv_scores.algo.map({'l1':r'$\ell_1$', 'l2':r'$\ell_2$', "enet":r'$\ell_1\ell_2$', 'rfel2':'RFE+$\ell_2$',  'fl2':'Filter+$\ell_2$'})

# filter and RFE + l2
df_ = cv_scores[cv_scores["algo"].isin(["fl2", "rfel2"])]
fig = plt.figure(figsize=(7.25, 5), dpi=300)
g = sns.lineplot(x="param", y="auc", hue="Model", data=df_)#, palette=palette)
g.set(xscale="log")
plt.xlabel(xlabel=r'k', fontsize=20)
plt.ylabel(ylabel=r'AUC', fontsize=16)
g.axes.axhline(baseline_l2C1.mean(), ls='--', color='black', linewidth=1, alpha=0.5)  # no feature selection performance
g.set(ylim=(.5, .9))
plt.tight_layout()
plt.savefig(OUTPUT(dataset, scaling=None, harmo=None, type="sensibility-filter-rfe", ext="pdf"))

# l1, enet
df_ = cv_scores[cv_scores["algo"].isin(["l2", "l1", "enet"])]
fig = plt.figure(figsize=(7.25, 5), dpi=300)
g = sns.lineplot(x="param", y="auc", hue="Model", data=df_)#, palette=palette)
g.set(xscale="log")
plt.xlabel(xlabel=r'C', fontsize=20)
plt.ylabel(ylabel=r'AUC', fontsize=16)
g.axes.axhline(baseline_l2C1.mean(), ls='--', color='black', linewidth=1, alpha=0.5)  # no feature selection performance
g.set(ylim=(.5, .9))
plt.tight_layout()
plt.savefig(OUTPUT(dataset, scaling=None, harmo=None, type="sensibility-l2-l1-enet", ext="pdf"))

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

if not os.path.exists(OUTPUT(dataset, scaling=None, harmo=None, type="Atv", ext="npz")):
    Atv = nesterov_tv.linear_operator_from_mask(mask_img.get_fdata(), calc_lambda_max=True)
    Atv.save(OUTPUT(dataset, scaling=None, harmo=None, type="Atv", ext="npz"))

Atv = LinearOperatorNesterov(filename=OUTPUT(dataset, scaling=None, harmo=None, type="Atv", ext="npz"))
assert np.allclose(Atv.get_singular_values(0), 11.940517804227724)


def ratios_to_param(alpha, l1l2ratio, tvratio):
    tv = alpha * tvratio
    l1 = alpha * float(1 - tv) * l1l2ratio
    l2 = alpha * float(1 - tv) * (1- l1l2ratio)
    return l1, l2, tv


# Large range
alphas = [.01, .1, 1.]
l1l2ratios = [.1]
tvratios = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.]

# smaller range
# alphas = [.1]
# l1l2ratios = [.1]
# tvratios = [0, .2, .4, .6, .8, 1.]

import itertools
estimators_dict = dict()
for alpha, l1l2ratio, tvratio in itertools.product(alphas, l1l2ratios, tvratios):
    print(alpha, l1l2ratio, tvratio)
    l1, l2, tv = ratios_to_param(alpha, l1l2ratio, tvratio)
    key = "enettv_%.3f:%.3f:%.3f" % (alpha, l1l2ratio, tvratio)

    conesta = algorithms.proximal.CONESTA(max_iter=10000)
    estimator = estimators.LogisticRegressionL1L2TV(l1, l2, tv, Atv, algorithm=conesta,
                                            class_weight="auto", penalty_start=0)
    estimators_dict[key] = estimator


args_collection_1 = dict_product(estimators_dict, dict(noresdualize=False), cv_dict)


key_vals = parallel(fit_predict, args_collection_1, n_jobs=NJOBS, pass_key=True, verbose=20)

models_filename = OUTPUT(dataset, scaling=scaling, harmo=harmo, type="models-5cv-enettv", ext="pkl")
with open(models_filename, 'wb') as fd:
    pickle.dump(key_vals, fd)
