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
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
#from sklearn.feature_selection import SelectKBest
#from sklearn.feature_selection import f_classif
#from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import sklearn.metrics as metrics
import sklearn.linear_model as lm
from matplotlib import pyplot as plt

WD = '/neurospin/psy/canbind'
#BASE_PATH = '/neurospin/brainomics/2018_euaims_leap_predict_vbm/results/VBM/1.5mm'

# Voxel size
# vs = "1mm"
vs = "1.5mm-s8mm"
#vs = "1.5mm"

INPUT = os.path.join(WD, "models", "predict_resp_from_baseline_%s" % vs)

# load data
#X = np.load(os.path.join(INPUT, "Xres.npy"))
X = np.load(os.path.join(INPUT, "Xcentersite.npy"))

y = np.load(os.path.join(INPUT, "y.npy"))
pop = pd.read_csv(os.path.join(INPUT, "population.csv"))
assert np.all(pop['respond_wk16_num'] == y)

mask_img = nibabel.load(os.path.join(INPUT, "mask.nii.gz"))
mask_arr = mask_img.get_data()
mask_arr = mask_arr == 1

scaler = preprocessing.StandardScaler()
X = scaler.fit(X).transform(X)
cv = StratifiedKFold(n_splits=5, random_state=0)

###############################################################################
# ML
def balanced_acc(estimator, X, y):
    return metrics.recall_score(y, estimator.predict(X), average=None).mean()

scoring = {'AUC': 'roc_auc', 'bAcc':balanced_acc}

model = lm.LogisticRegression(class_weight='balanced')
#model = svm.LinearSVC(class_weight='balanced', dual=False)

Cs = [0.00001, 0.0001, 0.001, .01, .1, 1, 10]

param_grid = {'C': Cs}
gs = GridSearchCV(model, cv=cv, param_grid=param_grid, scoring=scoring, refit='AUC', n_jobs=16)
gs.fit(X, y)
print(gs.best_params_)
gs.cv_results_

results = gs.cv_results_

###############################################################################
# Plot
plt.figure(figsize=(13, 13))
plt.title("GridSearchCV evaluating using multiple scorers simultaneously",
          fontsize=16)

plt.xlabel("min_samples_split")
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
plt.show()








model = lm.LogisticRegression(class_weight='balanced', C=gs.best_params_['C'])
model = lm.LogisticRegression(class_weight='balanced', C=1)

model = svm.LinearSVC(class_weight='balanced', dual=False, C=gs.best_params_['C'])

%time scores = cross_val_score(estimator=model, X=X, y=y, cv=cv, scoring='roc_auc', n_jobs=-1)
print(model, "\n", scores, scores.mean())
"""
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