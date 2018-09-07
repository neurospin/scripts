#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 18:41:41 2018

@author: ed203246
"""

###############################################################################
# Fast SVM

#############################################################################
# Models of respond_wk16_num + psyhis_mdd_age + age + sex_num + site
import os
import numpy as np
import nibabel
import pandas as pd
# from sklearn import datasets
import sklearn.svm as svm
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import StratifiedKFold
#from sklearn.feature_selection import SelectKBest
#from sklearn.feature_selection import f_classif
#from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import sklearn.metrics as metrics
import sklearn.linear_model as lm
from matplotlib import pyplot as plt
import mulm

WD = '/neurospin/psy/canbind'
#BASE_PATH = '/neurospin/brainomics/2018_euaims_leap_predict_vbm/results/VBM/1.5mm'

# Voxel size
# vs = "1mm"
#vs = "1.5mm-s8mm"
vs = "1.5mm"


INPUT = os.path.join(WD, "models", "vbm_resp_%s" % vs)
OUTPUT = INPUT

# load data
#DATASET = "XTreatTivSitePca"
DATASET = "XTreatTivSite"
#X = np.load(os.path.join(INPUT, "Xres.npy"))
#Xim = np.load(os.path.join(INPUT, "Xrawsc.npy"))
Xim = np.load(os.path.join(INPUT, DATASET+".npy"))


y = np.load(os.path.join(INPUT, "y.npy"))
pop = pd.read_csv(os.path.join(INPUT, "population.csv"))
assert np.all(pop['respond_wk16_num'] == y)

Xclin = pop[['age', 'sex_num', 'psyhis_mdd_age']]
Xclin.loc[Xclin["psyhis_mdd_age"].isnull(), "psyhis_mdd_age"] = Xclin["psyhis_mdd_age"].mean()
print(Xclin.isnull().sum())
Xclin = np.asarray(Xclin)

#mask_img = nibabel.load(os.path.join(INPUT, "mask.nii.gz"))
#mask_arr = mask_img.get_data()
#mask_arr = mask_arr == 1

###############################################################################
# Model 1: [Xclin, Xim]

#Xim = scaler.fit(Xim).transform(Xim)
X = np.concatenate([Xclin, Xim], axis=1)

scaler = preprocessing.StandardScaler()
X = scaler.fit(X).transform(X)


"""
###############################################################################
# Model 2:  LR_ImResClin-scaled_rs85
# resid-lm: MRI residual = MRI ~ psyhis_mdd_age + age + sex_num
#                    MRI residual ~ respond_wk16_num

## OLS with MULM
contrasts = [1] + [0] *(Xclin.shape[1] - 1)

mod = mulm.MUOLS(Xim, Xclin)
mod.fit()
residuals = Xim - mod.predict(Xclin)
X = residuals
X = scaler.fit(X).transform(X)
"""

###############################################################################
#
if False:

    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_samples, silhouette_score

    range_n_clusters = [2, 3, 4, 5, 6]
    for n_clusters in range_n_clusters:
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(X)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)

    """
    # XTreatTivSitePca
    For n_clusters = 2 The average silhouette_score is : 0.00444448044335
    For n_clusters = 3 The average silhouette_score is : 0.00226941102829
    For n_clusters = 4 The average silhouette_score is : -0.00550641362767
    For n_clusters = 5 The average silhouette_score is : -0.00101922453369
    For n_clusters = 6 The average silhouette_score is : 0.000944706623202

    clusterer = KMeans(n_clusters=2, random_state=10)
    cluster_labels = clusterer.fit_predict(X)
    from sklearn.metrics import confusion_matrix
    np.corrcoef(cluster_labels, y)
    confusion_matrix(y, cluster_labels)

        cluster
    y  [ 6, 26]
       [31, 61]

    Focus on 87 samples of cluster 1

    subset = cluster_labels == 1

    X = X[subset, :]
    y = y[subset]
    """

    # XTreatTivSite
    """
    For n_clusters = 2 The average silhouette_score is : 0.0477996449088
    For n_clusters = 3 The average silhouette_score is : 0.0192035904974
    For n_clusters = 4 The average silhouette_score is : 0.0174075116674
    For n_clusters = 5 The average silhouette_score is : 0.00614318245755
    For n_clusters = 6 The average silhouette_score is : 0.0112740182051
    """

    clusterer = KMeans(n_clusters=2, random_state=10)
    cluster_labels = clusterer.fit_predict(X)
    from sklearn.metrics import confusion_matrix
    np.corrcoef(cluster_labels, y)
    confusion_matrix(y, cluster_labels)

    """
        cluster
    y  [17, 15]
       [45, 47]
    """

    pop = pd.read_csv(os.path.join(INPUT, "population.csv"))
    pop["cluster"] = cluster_labels
    pop.to_csv(os.path.join(INPUT, DATASET+"-clust.csv"), index=False)

    import seaborn as sns

    from matplotlib.backends.backend_pdf import PdfPages

    pdf = PdfPages(os.path.join(OUTPUT, DATASET+'-clust.pdf'))
    fig = plt.figure()
    #fig.suptitle('Cluster x GMratio')
    sns.violinplot(x="cluster", y="GMratio", hue="respond_wk16", data=pop, split=True)
    pdf.savefig(); plt.close()

    fig = plt.figure()
    sns.lmplot(x="psyhis_mdd_age", y="GMratio", hue="cluster" , data=pop, fit_reg=False)
    pdf.savefig(); plt.close()

    # pop_nona = pop.copy()
    # pop_nona.loc[pop_nona["psyhis_mdd_age"].isnull(), "psyhis_mdd_age"] = pop_nona["psyhis_mdd_age"].mean()
    # g = sns.PairGrid(pop_nona[["GMratio", "age", "psyhis_mdd_age", "cluster", "respond_wk16"]], hue="cluster")
    # g.map_diag(plt.hist)
    # g.map_offdiag(plt.scatter)


    fig = plt.figure()
    #fig.suptitle('Cluster x GMratio')
    sns.lmplot(x="age", y="GMratio", hue="cluster", data=pop)
    pdf.savefig(); plt.close()

    fig = plt.figure()
    sns.lmplot(x="age", y="GMratio", hue="respond_wk16", col="cluster", data=pop, fit_reg=False)
    pdf.savefig()
    pdf.close()

    clust = pd.read_csv(os.path.join(INPUT, DATASET+"-clust.csv"))
    cluster_labels = clust["cluster"]
    # X = XTot; y = yTot
    # Focus on 62 samples of cluster 0
    subset = cluster_labels == 0

    # Focus on 62 samples of cluster 1
    subset = cluster_labels == 1


    X = np.concatenate([Xclin, Xim], axis=1)
    scaler = preprocessing.StandardScaler()
    X = scaler.fit(X).transform(X)
    X = X[subset, :]
    y = np.load(os.path.join(INPUT, "y.npy"))
    y = y[subset]

''''
    NFOLDS = 5
    def balanced_acc(estimator, X, y):
        return metrics.recall_score(y, estimator.predict(X), average=None).mean()
    scorers = {'auc': 'roc_auc', 'bacc':balanced_acc}
    cv = StratifiedKFold(n_splits=NFOLDS)
    model = lm.LogisticRegression(class_weight='balanced', fit_intercept=False, C=10)
    estimator = model
    %time cv_results = cross_validate(estimator=model, X=X, y=y, cv=cv, scoring=scorers, n_jobs=-1)

    """
        %time scores_auc = cross_val_score(estimator=model, X=X, y=y, cv=cv, scoring='roc_auc', n_jobs=-1)
        print(model, "\n", scores_auc, scores_auc.mean())
        %time scores_bacc = cross_val_score(estimator=model, X=X, y=y, cv=cv, scoring=balanced_acc, n_jobs=-1)
        np.mean(scores_bacc)

        X_, y_, groups_ = indexable(X, y, None)
        cv_ = check_cv(cv, y, classifier=is_classifier(estimator))
        scorers, _ = _check_multimetric_scoring(estimator, scoring='roc_auc')
        scorer = check_scoring(estimator, scoring='roc_auc')
        scorer(estimator, X_test, y_test)
        metrics.roc_auc_score(y_test, estimator.predict(X_test))
    """
    y_test_pred = np.zeros(len(y))
    y_test_prob_pred = np.zeros(len(y))
    y_train_pred = np.zeros(len(y))
    coefs_cv = np.zeros((NFOLDS, X.shape[1]))
    test_auc = list()
    test_recalls = list()
    for cv_i, (train, test) in enumerate(cv.split(X, y)):
        #for train, test in cv.split(X, y, None):
        print(cv_i)
        X_train, X_test, y_train, y_test = X[train, :], X[test, :], y[train], y[test]
        #estimator = clone(model)
        estimator.fit(X_train, y_train)
        y_test_pred[test] = estimator.predict(X_test)
        y_test_prob_pred[test] = estimator.predict_proba(X_test)[:, 1]
        y_train_pred[train] = estimator.predict(X_train)
        #model.coef_
        test_auc.append(metrics.roc_auc_score(y_test, estimator.predict_proba(X_test)[:, 1]))
        test_recalls.append(metrics.recall_score(y_test, estimator.predict(X_test), average=None))

cv_results['test_auc'], cv_results['test_auc'].mean()
cv_results['test_bacc'], cv_results['test_bacc'].mean()


test_auc = np.array(test_auc)
test_auc, np.mean(aucs)
test_bacc = np.array([np.mean(r) for r in test_recalls])
test_bacc, np.mean(test_bacc)

import sklearn.model_selection
from sklearn.utils import indexable, check_random_state, safe_indexing
from sklearn.base import is_classifier, clone
from sklearn.model_selection import check_cv
from sklearn.metrics.scorer import check_scoring, _check_multimetric_scoring
estimator = model

recall_test = metrics.recall_score(y, y_test_pred, average=None)
recall_train = metrics.recall_score(y, y_train_pred, average=None)
acc_test = metrics.accuracy_score(y, y_test_pred)
bacc_test = recall_test.mean()
auc_test = metrics.roc_auc_score(y, y_test_prob_pred)

print(bacc_test, auc_test)
'''
###############################################################################
# ML

#cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=31) # 51, 26(59+9), 31(59+9), 85(58+11) best is 31
cv = StratifiedKFold(n_splits=5)

def balanced_acc(estimator, X, y):
    return metrics.recall_score(y, estimator.predict(X), average=None).mean()

scoring = {'AUC': 'roc_auc', 'bAcc':balanced_acc}

model = lm.LogisticRegression(class_weight='balanced', fit_intercept=False)
# model = lm.LogisticRegression(class_weight='balanced')
#model = svm.LinearSVC(class_weight='balanced', dual=False)

Cs = [0.00001, 0.0001, 0.001, .01, .1, 1, 10, 100, 1000, 10000]

param_grid = {'C': Cs}
gs = GridSearchCV(model, cv=cv, param_grid=param_grid, scoring=scoring, refit='AUC', n_jobs=16)
gs.fit(X, y)
print(gs.best_params_)
gs.cv_results_

results = gs.cv_results_
#results_51 = gs.cv_results_
# results_85 = results
# results = results_51

###############################################################################
# Plot
plt.figure(figsize=(13, 13))
plt.title("GridSearchCV LR",
          fontsize=16)

plt.xlabel("C")
plt.ylabel("Score")
plt.grid()

ax = plt.axes()
#ax.set_xlim(0, 402)
ax.set_ylim(0.4, 1)
ax.set_xscale("log")

# Get the regular numpy array from the MaskedArray
X_axis = np.array(results['param_C'].data, dtype=float)

for scorer, color in zip(sorted(scoring), ['g', 'k']):
    for sample, style in (('train', '--'), ('test', '-')):
        sample_score_mean = results['mean_%s_%s' % (sample, scorer)]
        sample_score_std = results['std_%s_%s' % (sample, scorer)]
        ax.fill_between(X_axis, sample_score_mean - sample_score_std,
                        sample_score_mean + sample_score_std,
                        alpha=0.1 if sample == 'test' else 0, color=color)
        ax.plot(X_axis, sample_score_mean, style, color=color,
                alpha=1 if sample == 'test' else 0.7,
                label="%s (%s)" % (scorer, sample))

    best_index = np.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]
    best_score = results['mean_test_%s' % scorer][best_index]

    # Plot a dotted vertical line at the best score for that scorer marked by x
    ax.plot([X_axis[best_index], ] * 2, [0, best_score],
            linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)

    # Annotate the best score for that scorer
    ax.annotate("%0.2f" % best_score,
                (X_axis[best_index], best_score + 0.005))

plt.legend(loc="best")
plt.grid('off')
#plt.show()
#plt.savefig(os.path.join(OUTPUT, DATASET+"_ml-lr-rs31_ClinIm.pdf"))
#plt.savefig(os.path.join(OUTPUT, DATASET+"_ml-lr-rs31_ClinImClust1.pdf"))

#plt.savefig(os.path.join(OUTPUT, DATASET+"_ml-lr_ClinIm.pdf"))
plt.savefig(os.path.join(OUTPUT, DATASET+"-clust_ml-lr_ClinImClust1.pdf"))
# plt.savefig(os.path.join(OUTPUT, DATASET+"-clust_ml-lr_ClinImClust0.pdf"))

#plt.savefig(os.path.join(OUTPUT, DATASET+"_ml-lr_ClinIm.pdf"))
#plt.savefig(os.path.join(OUTPUT, DATASET+"_ml-lr_ClinImClust1.pdf"))
# plt.savefig(os.path.join(OUTPUT, "resp-rmSiteTIV-resClin-scaled_ml-lr_rs85.pdf"))




#univstats-RespNoResp_vbm_1.5mm
res = list()
for rndst in range(100):
    print(rndst)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=rndst)
    model = lm.LogisticRegression(class_weight='balanced', C=1)# smmothed: 1e-4
    scores = cross_val_score(estimator=model, X=X, y=y, cv=cv, scoring='roc_auc', n_jobs=8)
    print(model, "\n", scores.mean(), scores.std(), scores)
    res.append([rndst, scores.mean(), scores.std()])


scores = pd.DataFrame(res, columns=["rnd", "mu", "std"])

scores["zscore"] =  scores["mu"] / scores["std"]




model = lm.LogisticRegression(class_weight='balanced', C=1e-4)
#model = lm.LogisticRegression(class_weight='balanced', C=1)

model = svm.LinearSVC(class_weight='balanced', dual=False, C=gs.best_params_['C'])

%time scores = cross_val_score(estimator=model, X=X, y=y, cv=cv, scoring='roc_auc', n_jobs=-1)
print(model, "\n", scores, scores.mean())
"""

univstats-RespNoResp_vbm_1.5mm-s8mm
-----------------------------------
LogisticRegression
C10-4
53%

"Xcentersite.npy"
----------------
print(model, "\n", scores, scores.mean())
LogisticRegression(C=100, class_weight='balanced', dual=False,
          fit_intercept=True, intercept_scaling=1, max_iter=100,
          multi_class='ovr', n_jobs=1, penalty='l2', random_state=None,
          solver='liblinear', tol=0.0001, verbose=0, warm_start=False)
 [0.48120301 0.38345865 0.75925926 0.30555556 0.77777778] 0.5414508493455863


LogisticRegression(C=1, class_weight='balanced', dual=False,
          fit_intercept=True, intercept_scaling=1, max_iter=100,
          multi_class='ovr', n_jobs=1, penalty='l2', random_state=None,
          solver='liblinear', tol=0.0001, verbose=0, warm_start=False)
 [0.4962406  0.39097744 0.75925926 0.30555556 0.7962963 ] 0.5496658312447786

 "Xres.npy"
 ----------

LogisticRegression(C=10, class_weight='balanced', dual=False,
          fit_intercept=True, intercept_scaling=1, max_iter=100,
          multi_class='ovr', n_jobs=1, penalty='l2', random_state=None,
          solver='liblinear', tol=0.0001, verbose=0, warm_start=False)
[0.61654135 0.62406015 0.60185185 0.26851852 0.7037037 ] 0.5629351155666945

CPU times: user 4.01 s, sys: 1.88 s, total: 5.88 s
Wall time: 1min 19s
LinearSVC(C=0.1, class_weight='balanced', dual=False, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
     verbose=0)
 [0.54887218 0.62406015 0.61111111 0.27777778 0.74074074] 0.5605123920913394


"""


#%time scores = cross_val_score(estimator=model, X=X, y=y, cv=cv, scoring=balanced_acc, n_jobs=-1)

# Out[32]: array([0.60902256, 0.65413534, 0.57407407, 0.26851852, 0.7037037 ])

scores.mean()
# Out[33]: 0.5618908382066278


# array([0.52631579, 0.66165414, 0.56481481, 0.27777778, 0.64814815])
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier()
%time scores = cross_val_score(estimator=model, X=X, y=y, cv=cv, scoring=balanced_acc, n_jobs=-1)
scores.mean()
# array([0.43984962, 0.58270677, 0.47222222, 0.41666667, 0.58333333])
# Out[91]: 0.49895572263993315