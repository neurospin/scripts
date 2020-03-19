"""
DBSCAN
https://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html#sphx-glr-auto-examples-cluster-plot-dbscan-py
"""

import numpy as np
import pandas as pd
import os
import nibabel
import matplotlib.pylab as plt
import seaborn as sns

INPUT = '/neurospin/psy/canbind'
INPUT = '/home/ed203246/data/psy/canbind'

INPUT_PATH = '/home/ed203246/data/psy/analyses/2020_canbind_response/data'
OUTPUT_PATH = '/home/ed203246/data/psy/analyses/2020_canbind_response/202003_predict_response_wk8_clustering'

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

# Future filename formating:
# dataset_sel-<selection>_mod-<t1mri|...>_improc-<mwp1|cat12roi|...>_grpproc-<[raw|gs]:[residualization]>_

################################################################################
#
# Dataset
#
################################################################################

pop = pd.read_csv(INPUT(dataset="canbind", mri_preproc='cat12roi', scaling=None, harmo=None, type="participants", ext="csv"))
assert pop.shape[0] == 761

targets = ['respond_wk8', 'respond_wk16']
for target in targets:
    pop[target + "_num"] = pop[target].map({'Responder':1, 'NonResponder':0})

pop["GM_frac"] = pop.GM_Vol / pop.TIV


################################################################################
# Subject selector

# Subcohorts: select V1 with response info
# dataset, target, target_num = 'canbind-ses-v1-respwk16', "respond_wk16", "respond_wk16_num"
dataset, target, target_num = 'canbind-ses-v1-respwk8', "respond_wk8", "respond_wk8_num"

msk_ctl = pop["treatment"].isin(['Control'])
assert msk_ctl.sum() == 272

msk_trt = pop.session.isin(["V1"]) & pop["treatment"].isin(['Treatment'])
assert msk_trt.sum() == 157

msk_tgt = pop.session.isin(["V1"]) & pop[target].notnull()
if target == "respond_wk16":
    assert msk_tgt.sum() == 124
elif target == "respond_wk8":
    assert msk_tgt.sum() == 135

################################################################################
# Images

from  nitk.image import img_to_array, global_scaling
imgs_arr, df_, target_img = img_to_array(pop.path)
assert np.all(pop[['participant_id', 'session', 'path']] == df_[['participant_id', 'session', 'path']])
imgs_arr = global_scaling(imgs_arr, axis0_values=np.array(pop.TIV), target=1500)
mask_img = nibabel.load(INPUT('canbind', scaling=None, harmo=None, type="mask", ext="nii.gz"))
mask_arr = mask_img.get_fdata() != 0
assert mask_arr.sum() == 369824
Xim = imgs_arr.squeeze()[:, mask_arr]
del imgs_arr

###############################################################################
# Clinical + Demo

vars_clinic = ['mde_num', 'duration', 'madrs_Baseline', 'age_onset']
vars_demo = ['age', 'sex', 'educ']

# pop_w: working popolation DataFrame = pop with Imputed missing data for patients only

pop_w = pop.copy()  # imputed missing data for patients only

print(pop.loc[msk_trt, vars_clinic + vars_demo].isnull().sum(axis=0))
"""
mde_num           45
duration           7
madrs_Baseline     4
age_onset          7
age                0
sex                0
educ               1
"""

# mde_num ~ age + age_onset + educ
import statsmodels.formula.api as smfrmla

df_ = pop.loc[msk_trt, ["mde_num", "age", "age_onset", 'madrs_Baseline', 'educ', 'sex', 'duration']].dropna()
ols_ = smfrmla.ols("mde_num ~ age + age_onset + educ", data=df_).fit()
del df_
print(ols_.rsquared_adj)
"""
0.092
"""
# Four patients have both age_onset and mde_num. So first imput age_onset and educ
np.sum(pop.loc[msk_trt, 'age_onset'].isnull() & pop.loc[msk_trt, 'mde_num'].isnull())

# age_onset => median
pop_w.loc[msk_trt & pop_w.loc[msk_trt, "age_onset"].isnull(), "age_onset"]=\
    pop_w.loc[msk_trt, "age_onset"].median()

pop_w["duration"] = pop_w["age"] - pop_w["age_onset"]

# educ => median
pop_w.loc[msk_trt & pop_w.loc[msk_trt, "educ"].isnull(), "educ"] = \
    pop_w.loc[msk_trt, "educ"].median()

# mde_num ~ age + age_onset + educ
pop_w.loc[msk_trt & pop_w.loc[msk_trt, "mde_num"].isnull(), "mde_num"] = \
    ols_.predict(pop_w)[msk_trt & pop_w.loc[msk_trt, "mde_num"].isnull()]

# madrs_Baseline => median
pop_w.loc[msk_trt & pop_w.loc[msk_trt, "madrs_Baseline"].isnull(), "madrs_Baseline"] = \
    pop_w.loc[msk_trt, "madrs_Baseline"].median()

print(pop_w.loc[msk_trt, vars_clinic + vars_demo].isnull().sum(axis=0))

vars_clinic = ['mde_num', 'duration', 'madrs_Baseline', 'age_onset']
vars_demo = ['age', 'sex', 'educ']

# Finally, extract blocs
Xclin = pop_w[vars_clinic].values
Xdemo = pop_w[vars_demo].values
Xsite = pd.get_dummies(pop_w.site).values
Xdemoclin = np.concatenate([Xdemo, Xclin], axis=1)

################################################################################
#
# PCA on ctl then apply on patients
#
################################################################################

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

scaler_ctl = StandardScaler()
pca_ctl = PCA(n_components=2)
PC_ctl_ = pca_ctl.fit_transform(scaler_ctl.fit_transform(Xim[msk_ctl, :]))
print("EV", pca_ctl.explained_variance_ratio_)
PC_tgt_ = pca_ctl.transform(scaler_ctl.transform(Xim[msk_tgt, :]))

sns.scatterplot(pop["GM_frac"][msk_tgt], PC_tgt_[:, 0], hue=pop[target][msk_tgt])
print("PC1 capture global GM atrophy")

# sns.scatterplot(pop["GM_frac"][msk_tgt], PC_tgt_[:, 1], hue=pop[target][msk_tgt])

sns.scatterplot(PC_tgt_[:, 0], PC_tgt_[:, 1], hue=pop[target][msk_tgt])
print("PC1-PC2 no specific pattern")

del PC_ctl_, PC_tgt_

################################################################################
# PCA on patients on both image and clinic

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


Xim_tgt_s_ = StandardScaler().fit_transform(Xim[msk_tgt, :])
Xdemoclin_tgt_s_ = StandardScaler().fit_transform(Xdemoclin[msk_tgt, :])

pca_im, pca_democlin = PCA(n_components=7), PCA(n_components=7)
PCim = pca_im.fit_transform(Xim_tgt_s_)
PCdemoClin = pca_democlin.fit_transform(Xdemoclin_tgt_s_)

del Xim_tgt_s_, Xdemoclin_tgt_s_

print("Imaging EV", pca_im.explained_variance_ratio_)

fig, axs = plt.subplots(ncols=3)

sns.scatterplot(PCim[:, 0], pop["GM_frac"][msk_tgt], hue=pop["respond_wk8"][msk_tgt], ax=axs[0])
print("PC1 capture global GM atrophy")

sns.scatterplot(PCim[:, 0], PCim[:, 1], hue=pop["respond_wk8"][msk_tgt], ax=axs[1])
print("PC1-PC2 no specific pattern associated with respond_wk8")

sns.scatterplot(PCim[:, 0], PCim[:, 1], hue=pop["respond_wk16"][msk_tgt], ax=axs[2])
print("PC1-PC2 no specific pattern associated with respond_wk16")
#fig.clf()

# DemoClin(PC1 + PC2) and response
print("DemoClin EV", pca_democlin.explained_variance_ratio_)

plt.figure()
sns.scatterplot(PCdemoClin[:, 0], PCdemoClin[:, 1], hue=pop["respond_wk8"][msk_tgt])
print("DemoClin(PC1 + PC2) pattern associated with respond_wk8")

plt.figure()
sns.scatterplot(PCdemoClin[:, 1], PCdemoClin[:, 2], hue=pop["respond_wk8"][msk_tgt])
#sns.scatterplot(PCdemoClin[:, 0], PCdemoClin[:, 1], hue=pop["respond_wk16"][msk_tgt])
#sns.scatterplot(PCdemoClin[:, 0], PCdemoClin[:, 2], hue=pop["respond_wk16"][msk_tgt])

################################################################################
#
# PLS
#
################################################################################

import scipy.linalg
from sklearn.cross_decomposition import PLSCanonical

Xim_tgt_s_ = StandardScaler().fit_transform(Xim[msk_tgt, :])
Xdemoclin_tgt_s_ = StandardScaler().fit_transform(Xdemoclin[msk_tgt, :])
_, s_, _ = scipy.linalg.svd(Xdemoclin_tgt_s_, full_matrices=False)
rank_ = np.sum(s_ > 1e-6)

plsca = PLSCanonical(n_components=rank_)
%time PLSim_scores, PLSclin_scores = plsca.fit_transform(Xim_tgt_s_, Xdemoclin_tgt_s_)

# Imaging components
df_ = pd.DataFrame(PLSim_scores)
df_["respond_wk8"] = pop["respond_wk8"][msk_tgt].values
df_["respond_wk16"] = pop["respond_wk16"][msk_tgt].values
df_["GM_frac"] = pop["GM_frac"][msk_tgt].values
sns.pairplot(df_, hue="respond_wk8")
print("PC1 capture global GM atrophy")

# Demo/Clinic components
df_ = pd.DataFrame(PLSclin_scores)
for var in vars_demo + vars_clinic + ["respond_wk8", "respond_wk16"]:
    df_[var] = pop[var][msk_tgt].values

sns.pairplot(df_, hue="respond_wk8")
print("PLSscore1 capture global age")
del df_, Xim_tgt_s_, Xdemoclin_tgt_s_, rank_

################################################################################
#
# Clustering
#
################################################################################

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn import metrics

################################################################################
# Fit on Imaging of Ctl => predict on imaging of treatment

scaler_ctl = StandardScaler()
Xim_ctl_ = scaler_ctl.fit_transform(Xim[msk_ctl, :])
Xim_tgt_ = scaler_ctl.transform(Xim[msk_tgt, :])

#------------------------------------------------------------------------------
# DBSCAN

%time db = DBSCAN(eps=850, min_samples=5, n_jobs=8).fit(Xim_ctl_)
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
print('# => Too hard to tune')

#------------------------------------------------------------------------------
# KMeans

%time km = KMeans(n_clusters=2, n_jobs=8, random_state=60).fit(Xim_ctl_)
clust_labels_kmeans_img = km.predict(Xim_tgt_)

# Interpretation
y_ = pop[target + "_num"][msk_tgt].values

print("# Responder and nonResponder in each cluster", "\n",
    {"clust%i_resp%i_#" % (clust_, resp): np.sum(y_[clust_labels_kmeans_img == clust_] == resp)
        for clust_ in range(km.n_clusters) for resp in np.unique(y_)}
)
"""
# Responder and nonResponder in each cluster
{'clust0_resp0_#': 36, 'clust0_resp1_#': 30, 'clust1_resp0_#': 36, 'clust1_resp1_#': 33}
"""

# point toward cluster 0
centers_delta = km.cluster_centers_[0, :] - km.cluster_centers_[1, :]
PC_tgt_ = pca_ctl.transform(Xim_tgt_)

# plots
fig = plt.figure()
sns.scatterplot(PC_tgt_[:, 0], np.dot(Xim_tgt_, centers_delta),
    hue=pop[target][msk_tgt], style=km.predict(Xim_tgt_))
plt.xlabel("PC1"); plt.ylabel("C0-C1 (point toward cluster 0)");
fig.suptitle("Direction toward cluster 0 == PC1")

fig = plt.figure()
sns.scatterplot(pop["GM_frac"][msk_tgt], np.dot(Xim_tgt_, centers_delta),
    hue=pop[target][msk_tgt], style=km.predict(Xim_tgt_))
plt.xlabel("GM_frac"); plt.ylabel("C0-C1 (point toward cluster 0)");
fig.suptitle("Direction toward cluster 0 == GM atrophy")

fig = plt.figure()
sns.scatterplot(pop["age"][msk_tgt], np.dot(Xim_tgt_, centers_delta),
    hue=pop[target][msk_tgt], style=km.predict(Xim_tgt_))
plt.xlabel("age"); plt.ylabel("C0-C1 (point toward cluster 0)");
fig.suptitle("cluster 0 are older patients")

fig = plt.figure()
sns.scatterplot(PC_tgt_[:, 0], PC_tgt_[:, 1], hue=pop[target][msk_tgt],
    style=km.predict(Xim_tgt_))
plt.xlabel("PC1"); plt.ylabel("PC2");
fig.suptitle("Clustering could have been performed on PC1")

# stats
df_ = pop[msk_tgt][vars_clinic + vars_demo + ["GM_frac", target]]
df_["cluster"] = km.predict(Xim_tgt_)

stats_clust_ = df_.groupby("cluster").describe().T
stats_clust_target_ = df_.groupby(["cluster", target]).describe().T

xls_filename = OUTPUT(dataset="canbind", mri_preproc='mpw1', scaling="gs", harmo=None,
    type="clusters-participants-stats", ext="xlsx")

with pd.ExcelWriter(xls_filename) as writer:
    stats_clust_.to_excel(writer, sheet_name='by_cluster')
    stats_clust_target_.to_excel(writer, sheet_name='by_cluster_by_response')

del Xim_ctl_, Xim_tgt_, PC_tgt_

################################################################################
# Fit clustering on patients' imaging PLSscores

X_ = StandardScaler().fit_transform(PLSim_scores)

#------------------------------------------------------------------------------
# DBSCAN

%time db = DBSCAN(eps=1.5, min_samples=5, n_jobs=8).fit(X_)

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
n_noise_ = list(db.labels_).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
"""
Estimated number of clusters: 1
Estimated number of noise points: 51
"""

clust_labels_dbscan_pls = db.labels_

#------------------------------------------------------------------------------
# KMeans

X_ = StandardScaler().fit_transform(PLSim_scores)

%time km_pls = KMeans(n_clusters=2, n_jobs=8).fit(X_)
clust_labels_kmeans_pls = km_pls.predict(X_)

del X_

################################################################################
#
# Cross-Validation
#
################################################################################

import sklearn.ensemble
import sklearn.linear_model as lm
from sklearn.model_selection import StratifiedKFold, KFold, LeaveOneOut
from sklearn.preprocessing import StandardScaler
import copy
from nitk.utils import dict_product, parallel, aggregate_cv

#nsplits = 32
nsplits = 5

cv = StratifiedKFold(n_splits=nsplits)
#cv = LeaveOneOut()


################################################################################
# Utils

def fit_predict(estimator_img, split):
    #residualizer = copy.deepcopy(residualizer)
    estimator_img = copy.deepcopy(estimator_img)
    train, test = split
    Xim_train, Xim_test, Xdemoclin_train, Xdemoclin_test, Xsite_train, Xsite_test, y_train =\
        Xim_[train, :], Xim_[test, :], Xdemoclin_[train, :], Xdemoclin_[test, :], Xsite_[train, :], Xsite_[test, :], y_[train]
    
    # Images based predictor
    """
    if RES_MOD is not None:
        if RES_MOD == 'RES_ALL':
            residualizer.fit(Xim, Z)
        elif RES_MOD == 'RES_TRAIN':
            residualizer.fit(Xim_train, Z_train)
        Xim_train = residualizer.transform(Xim_train, Z_train)
        Xim_test = residualizer.transform(Xim_test, Z_test)
    """

    scaler = StandardScaler()
    Xim_train = scaler.fit_transform(Xim_train)
    Xim_test = scaler.transform(Xim_test)
    estimator_img.fit(Xim_train, y_train)

    y_test_img = estimator_img.predict(Xim_test)
    # score_test_img = estimator_img.predict_log_proba(Xim_test)[:, 1]
    # score_train_img = estimator_img.predict_log_proba(Xim_train)[:, 1]
    score_test_img = estimator_img.decision_function(Xim_test)
    score_train_img = estimator_img.decision_function(Xim_train)

    # Demographic/clinic based predictor
    estimator_democlin = lm.LogisticRegression(C=1, class_weight='balanced', fit_intercept=False)
    scaler = StandardScaler()
    Xdemoclin_train = scaler.fit_transform(Xdemoclin_train)
    Xdemoclin_test = scaler.transform(Xdemoclin_test)
    estimator_democlin.fit(Xdemoclin_train, y_train)

    y_test_democlin = estimator_democlin.predict(Xdemoclin_test)
    # score_test_democlin = estimator_democlin.predict_log_proba(Xdemoclin_test)[:, 1]
    # score_train_democlin = estimator_democlin.predict_log_proba(Xdemoclin_train)[:, 1]
    score_test_democlin = estimator_democlin.decision_function(Xdemoclin_test)
    score_train_democlin = estimator_democlin.decision_function(Xdemoclin_train)

    # STACK DEMO-CLIN + IMG
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
                y_test_stck=y_test_stck, score_test_stck=score_test_stck)

################################################################################
# SETTINGS

#-------------------------------------------------------------------------------
SETTING = "ALL"
Xim_ = Xim[msk_tgt, :]
Xdemoclin_ = Xdemoclin[msk_tgt, :]
Xsite_ = Xsite[msk_tgt, :]
y_ = pop[target + "_num"][msk_tgt].values

#-------------------------------------------------------------------------------
# SETTING = "CLUST-DBSCAN-PLS"
assert msk_tgt.sum() == len(clust_labels_dbscan_pls)
Xim_ = Xim[msk_tgt, :][clust_labels_dbscan_pls != -1, :]
Xdemoclin_ = Xdemoclin[msk_tgt, :][clust_labels_dbscan_pls != -1, :]
Xsite_ = Xsite[msk_tgt, :][clust_labels_dbscan_pls != -1, :]
y_ = pop[target + "_num"][msk_tgt][clust_labels_dbscan_pls != -1].values

#-------------------------------------------------------------------------------
SETTING = "CLUST0-KMEANS-PLS"
assert msk_tgt.sum() == len(clust_labels_kmeans_pls)
clust_ = 0
Xim_ = Xim[msk_tgt, :][clust_labels_kmeans_pls == clust_, :]
Xdemoclin_ = Xdemoclin[msk_tgt, :][clust_labels_kmeans_pls == clust_, :]
Xsite_ = Xsite[msk_tgt, :][clust_labels_kmeans_pls == clust_, :]
y_ = pop[target + "_num"][msk_tgt][clust_labels_kmeans_pls ==clust_].values

#-------------------------------------------------------------------------------
SETTING = "CLUST1-KMEANS-PLS"
assert msk_tgt.sum() == len(clust_labels_kmeans_pls)
clust_ = 1
Xim_ = Xim[msk_tgt, :][clust_labels_kmeans_pls == clust_, :]
Xdemoclin_ = Xdemoclin[msk_tgt, :][clust_labels_kmeans_pls == clust_, :]
Xsite_ = Xsite[msk_tgt, :][clust_labels_kmeans_pls == clust_, :]
y_ = pop[target + "_num"][msk_tgt][clust_labels_kmeans_pls ==clust_].values

#-------------------------------------------------------------------------------
SETTING = "CLUST0-KMEANS-IMG"

assert msk_tgt.sum() == len(clust_labels_kmeans_img)
clust_ = 0
Xim_ = Xim[msk_tgt, :][clust_labels_kmeans_img == clust_, :]
Xdemoclin_ = Xdemoclin[msk_tgt, :][clust_labels_kmeans_img == clust_, :]
Xsite_ = Xsite[msk_tgt, :][clust_labels_kmeans_img == clust_, :]
y_ = pop[target + "_num"][msk_tgt][clust_labels_kmeans_img ==clust_].values

#-------------------------------------------------------------------------------
SETTING = "CLUST1-KMEANS-IMG"
assert msk_tgt.sum() == len(clust_labels_kmeans_img)
clust_ = 1
Xim_ = Xim[msk_tgt, :][clust_labels_kmeans_img == clust_, :]
Xdemoclin_ = Xdemoclin[msk_tgt, :][clust_labels_kmeans_img == clust_, :]
Xsite_ = Xsite[msk_tgt, :][clust_labels_kmeans_img == clust_, :]
y_ = pop[target + "_num"][msk_tgt][clust_labels_kmeans_img ==clust_].values



################################################################################
# RUN FOR EACH SETTING

cv_dict = {fold:split for fold, split in enumerate(cv.split(Xim_, y_))}
estimators_dict = dict(lr=lm.LogisticRegression(C=1, class_weight='balanced', fit_intercept=False))
args_collection = dict_product(estimators_dict, cv_dict)
%time cv_res = parallel(fit_predict, args_collection, n_jobs=min(nsplits, 8))
aggregate = aggregate_cv(cv_res, args_collection, 1)
mod_keys = set(list(zip(*[k for k in aggregate.keys()]))[0])

scores_ = dict(seed=seed, clust=clust_, N=len(y_))
scores_.update({"count_%i"%lab:np.sum(y_ == lab) for lab in np.unique(y_)})
for mod_key, pred_key in aggregate.keys():
    #print(mod_key, pred_key)
    scores_[mod_key] = mod_key
    if "y_" in pred_key:
        scores_[pred_key.replace("y_", "bacc_")] = metrics.recall_score(y_, aggregate[(mod_key, pred_key)], average=None).mean(),
    elif "score_" in pred_key:
        scores_[pred_key.replace("score_", "auc_")] = metrics.roc_auc_score(y_, aggregate[(mod_key, pred_key)])

print(pd.DataFrame(scores_))


"""
ALL 5CV
lr IMG  {'baccs_test': 0.5287698412698413, 'aucs_test': 0.47641093474426804, 'recalls_test': array([0.48611111, 0.57142857])}
lr DEMO-CLIN {'baccs_test': 0.6051587301587302, 'aucs_test': 0.6241181657848325, 'recalls_test': array([0.52777778, 0.68253968])}
lr STCK {'baccs_test': 0.5426587301587301, 'aucs_test': 0.5291005291005291, 'recalls_test': array([0.625     , 0.46031746])}

CLUST-DBSCAN-PLS 5CV *
lr IMG  {'baccs_test': 0.5480549199084668, 'aucs_test': 0.602974828375286, 'recalls_test': array([0.54347826, 0.55263158])}
lr DEMO-CLIN {'baccs_test': 0.5703661327231121, 'aucs_test': 0.6029748283752862, 'recalls_test': array([0.45652174, 0.68421053])}
lr STCK {'baccs_test': 0.6435926773455378, 'aucs_test': 0.6521739130434783, 'recalls_test': array([0.76086957, 0.52631579])}

CLUST0-KMEANS-PLS 5CV
lr IMG  {'baccs_test': 0.546875, 'aucs_test': 0.5681818181818181, 'recalls_test': array([0.59375, 0.5    ])}
lr DEMO-CLIN {'baccs_test': 0.5767045454545454, 'aucs_test': 0.6107954545454546, 'recalls_test': array([0.5625    , 0.59090909])}
lr STCK {'baccs_test': 0.5497159090909091, 'aucs_test': 0.6136363636363636, 'recalls_test': array([0.78125   , 0.31818182])}

CLUST1-KMEANS-PLS 5CV
lr IMG  {'baccs_test': 0.45670731707317075, 'aucs_test': 0.4408536585365854, 'recalls_test': array([0.45      , 0.46341463])}
lr DEMO-CLIN {'baccs_test': 0.5295731707317073, 'aucs_test': 0.5591463414634147, 'recalls_test': array([0.425     , 0.63414634])}
lr STCK {'baccs_test': 0.49329268292682926, 'aucs_test': 0.4725609756097561, 'recalls_test': array([0.45      , 0.53658537])}

CLUST0-KMEANS-IMG 5CV ***
lr IMG  {'baccs_test': 0.6333333333333333, 'aucs_test': 0.6555555555555556, 'recalls_test': array([0.66666667, 0.6       ])}
lr DEMO-CLIN {'baccs_test': 0.6722222222222223, 'aucs_test': 0.700925925925926, 'recalls_test': array([0.61111111, 0.73333333])}
lr STCK {'baccs_test': 0.7222222222222222, 'aucs_test': 0.7796296296296297, 'recalls_test': array([0.77777778, 0.66666667])}

clust	N	count_0	count_1	lr	auc_test_democlin	auc_test_img	    auc_test_stck	    bacc_test_democlin	bacc_test_img	    bacc_test_stck
0   	66	36	    30	    lr  0.700925925925926	0.655555555555556	0.77962962962963	0.672222222222222	0.633333333333333	0.722222222222222

CLUST1-KMEANS-IMG 5CV
lr IMG  {'baccs_test': 0.4949494949494949, 'aucs_test': 0.47643097643097637, 'recalls_test': array([0.44444444, 0.54545455])}
lr DEMO-CLIN {'baccs_test': 0.5113636363636364, 'aucs_test': 0.4528619528619528, 'recalls_test': array([0.41666667, 0.60606061])}
lr STCK {'baccs_test': 0.46085858585858586, 'aucs_test': 0.45286195286195285, 'recalls_test': array([0.52777778, 0.39393939])}


"""