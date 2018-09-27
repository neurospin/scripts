#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 09:07:06 2018

@author: ed203246
"""

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
import seaborn as sns

WD = '/neurospin/psy/canbind'
#BASE_PATH = '/neurospin/brainomics/2018_euaims_leap_predict_vbm/results/VBM/1.5mm'

# Voxel size
# vs = "1mm"
#vs = "1.5mm-s8mm"
vs = "1.5mm"


INPUT = os.path.join(WD, "models", "clustering")

OUTPUT = INPUT

# load data
#DATASET = "XTreatTivSitePca"
#DATASET = "XTreatTivSite-ClinIm"
DATASET = "XTreatTivSite-Im"
#X = np.load(os.path.join(INPUT, "Xres.npy"))
#Xim = np.load(os.path.join(INPUT, "Xrawsc.npy"))
Xim = np.load(os.path.join(INPUT, DATASET.split("-")[0]+".npy"))


y = np.load(os.path.join(INPUT, "y.npy"))
yorig = y.copy()

pop = pd.read_csv(os.path.join(INPUT, "population.csv"))
assert np.all(pop['respond_wk16_num'] == y)

Xclin = pop[['age', 'sex_num', 'psyhis_mdd_age']]
Xclin.loc[Xclin["psyhis_mdd_age"].isnull(), "psyhis_mdd_age"] = Xclin["psyhis_mdd_age"].mean()
print(Xclin.isnull().sum())
Xclin = np.asarray(Xclin)

###############################################################################
# ML
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import StratifiedKFold
import copy

clustering = pd.read_csv(os.path.join(INPUT, DATASET+"-clust.csv"))
cluster_labels = clustering.cluster
C = 0.1
NFOLDS = 5
cv = StratifiedKFold(n_splits=NFOLDS)
model = lm.LogisticRegression(class_weight='balanced', fit_intercept=False, C=C)
scaler = preprocessing.StandardScaler()
def balanced_acc(estimator, X, y):
    return metrics.recall_score(y, estimator.predict(X), average=None).mean()
scorers = {'auc': 'roc_auc', 'bacc':balanced_acc, 'acc':'accuracy'}

###############################################################################
# No Clustering

###############################################################################
## No Clustering / Classify ImVbm

X = scaler.fit(Xim).transform(Xim)
%time cv_results = cross_validate(estimator=copy.copy(model), X=X, y=y, cv=cv, scoring=scorers, n_jobs=-1)
print(
      cv_results["test_auc"], cv_results["test_auc"].mean(), "\n",
      cv_results["test_bacc"], cv_results["test_bacc"].mean(), "\n",
      cv_results["test_acc"], cv_results["test_acc"].mean())

"""
[ 0.45112782  0.58646617  0.59259259  0.42592593  0.75925926] 0.563074352548
 [ 0.42857143  0.5         0.5         0.5         0.52777778] 0.49126984127
 [ 0.23076923  0.26923077  0.25        0.25        0.29166667] 0.258333333333
"""

###############################################################################
## No Clustering / Classify Clin

X = scaler.fit(Xclin).transform(Xclin)
%time cv_results = cross_validate(estimator=copy.copy(model), X=X, y=y, cv=cv, scoring=scorers, n_jobs=-1)
#%time cv_results = cross_validate(estimator=LogisticRegression(), X=X, y=y, cv=cv, scoring=scorers, n_jobs=-1)

print(
      cv_results["test_auc"], cv_results["test_auc"].mean(), "\n",
      cv_results["test_bacc"], cv_results["test_bacc"].mean(), "\n",
      cv_results["test_acc"], cv_results["test_acc"].mean())
"""
[ 0.39849624  0.51879699  0.66203704  0.65740741  0.49074074] 0.545495683654
 [ 0.45112782  0.47744361  0.66666667  0.69444444  0.41666667] 0.54126984127
 [ 0.46153846  0.5         0.66666667  0.625       0.54166667] 0.558974358974
"""

###############################################################################
## No Clustering / Classify Stacking Clin-ImVbm

"""
Use mlxtend
http://rasbt.github.io/mlxtend/
conda install -c conda-forge mlxtend

https://rasbt.github.io/mlxtend/user_guide/classifier/StackingClassifier/
"""
from mlxtend.classifier import StackingClassifier
from mlxtend.feature_selection import ColumnSelector
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

XClinIm = np.concatenate([Xclin, Xim], axis=1)
assert Xclin.shape == (124, 3)
X = scaler.fit(XClinIm).transform(XClinIm)

pipe1 = make_pipeline(ColumnSelector(cols=np.arange(0, Xclin.shape[1])),
                      copy.copy(model))
pipe2 = make_pipeline(ColumnSelector(cols=np.arange(Xclin.shape[1], X.shape[1])),
                      copy.copy(model))

sclf = StackingClassifier(classifiers=[pipe1, pipe2],
                          meta_classifier=LogisticRegression())


%time cv_results = cross_validate(estimator=sclf, X=X, y=y, cv=cv, scoring=scorers, n_jobs=-1)
print(
      cv_results["test_auc"], cv_results["test_auc"].mean(), "\n",
      cv_results["test_bacc"], cv_results["test_bacc"].mean(), "\n",
      cv_results["test_acc"], cv_results["test_acc"].mean())

"""
[ 0.41729323  0.47744361  0.66666667  0.69444444  0.43981481] 0.539132553606
[ 0.42857143  0.5         0.5         0.5         0.52777778] 0.49126984127
[ 0.23076923  0.26923077  0.25        0.25        0.29166667] 0.258333333333
"""
###############################################################################
## Clust Clinic / Classify clinic

X = scaler.fit(Xclin).transform(Xclin)

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score


range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9]
for n_clusters in range_n_clusters:
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

'''
For n_clusters = 2 The average silhouette_score is : 0.411891797017
For n_clusters = 3 The average silhouette_score is : 0.438089604311
For n_clusters = 4 The average silhouette_score is : 0.459294080272
For n_clusters = 5 The average silhouette_score is : 0.486715074704
For n_clusters = 6 The average silhouette_score is : 0.496793757829
For n_clusters = 7 The average silhouette_score is : 0.461003693962
For n_clusters = 8 The average silhouette_score is : 0.453539452228
For n_clusters = 9 The average silhouette_score is : 0.419120007433

array([[16, 16],
       [32, 60]])
'''

X = scaler.fit(Xclin).transform(Xclin)
clustering = dict()
clustering['participant_id'] = pop['participant_id']

range_n_clusters = [2, 3, 4]
res = list()
for n_clusters in range_n_clusters:
    print("######################################")
    print("# nclust", n_clusters)
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)
    clustering["nbclusts:%i" % n_clusters] = cluster_labels
    for clust in np.unique(cluster_labels):
        print("===================================")
        subset = cluster_labels == clust
        print(clust, subset.sum())
        Xg = Xclin[subset, :]
        yg = yorig[subset]
        Xg = scaler.fit(Xg).transform(Xg)

        cv = StratifiedKFold(n_splits=NFOLDS)
        model = lm.LogisticRegression(class_weight='balanced', fit_intercept=False, C=C)

        cv_results = cross_validate(estimator=copy.copy(model), X=Xg, y=yg, cv=cv, scoring=scorers, n_jobs=-1)
        res.append(['Clinic', 'test_auc', n_clusters, clust, subset.sum()] + cv_results["test_auc"].tolist() + [cv_results["test_auc"].mean()])
        res.append(['Clinic', 'test_bacc', n_clusters, clust, subset.sum()] + cv_results["test_bacc"].tolist() + [cv_results["test_bacc"].mean()])
        res.append(['Clinic', 'test_acc', n_clusters, clust, subset.sum()] + cv_results["test_acc"].tolist() + [cv_results["test_acc"].mean()])

clustering = pd.DataFrame(clustering)
clustering.to_csv(os.path.join(INPUT, "Xclin-clust.csv"), index=False)

res = pd.DataFrame(res, columns=['data', 'score', 'n_clusters', 'clust', 'size'] + ["fold%i" % i for i in range(NFOLDS)] + ['avg'])

print(res)
'''
      data      score  n_clusters  clust  size     fold0     fold1     fold2     fold3     fold4       avg
0   Clinic   test_auc           2      0    48  0.571429  0.666667  0.388889  0.722222  0.333333  0.536508
1   Clinic  test_bacc           2      0    48  0.392857  0.547619  0.500000  0.500000  0.250000  0.438095
2   Clinic   test_acc           2      0    48  0.363636  0.500000  0.555556  0.555556  0.333333  0.461616
3   Clinic   test_auc           2      1    76  0.395833  0.944444  0.222222  0.472222  0.611111  0.529167
4   Clinic  test_bacc           2      1    76  0.541667  0.916667  0.250000  0.416667  0.708333  0.566667
5   Clinic   test_acc           2      1    76  0.562500  0.866667  0.400000  0.466667  0.733333  0.605833
6   Clinic   test_auc           3      0    20  0.833333  0.000000  0.750000  0.750000  1.000000  0.666667
7   Clinic  test_bacc           3      0    20  0.666667  0.000000  0.500000  0.500000  0.500000  0.433333
8   Clinic   test_acc           3      0    20  0.600000  0.000000  0.500000  0.500000  0.666667  0.453333
9   Clinic   test_auc           3      1    68  0.666667  0.939394  0.348485  0.590909  0.727273  0.654545
10  Clinic  test_bacc           3      1    68  0.560606  0.742424  0.393939  0.477273  0.659091  0.566667
11  Clinic   test_acc           3      1    68  0.500000  0.785714  0.428571  0.461538  0.769231  0.589011
12  Clinic   test_auc           3      2    36  0.333333  0.000000  0.100000  0.500000  0.200000  0.226667
13  Clinic  test_bacc           3      2    36  0.333333  0.000000  0.200000  0.450000  0.200000  0.236667
14  Clinic   test_acc           3      2    36  0.250000  0.000000  0.285714  0.428571  0.285714  0.250000
15  Clinic   test_auc           4      0    36  0.333333  0.000000  0.100000  0.500000  0.200000  0.226667
16  Clinic  test_bacc           4      0    36  0.333333  0.000000  0.200000  0.450000  0.200000  0.236667
17  Clinic   test_acc           4      0    36  0.250000  0.000000  0.285714  0.428571  0.285714  0.250000
18  Clinic   test_auc           4      1    29  0.125000  0.500000  0.375000  0.875000  0.250000  0.425000
19  Clinic  test_bacc           4      1    29  0.250000  0.625000  0.375000  0.625000  0.250000  0.425000
20  Clinic   test_acc           4      1    29  0.166667  0.666667  0.333333  0.666667  0.400000  0.446667
21  Clinic   test_auc           4      2    14  1.000000  0.500000  1.000000  1.000000  1.000000  0.900000
22  Clinic  test_bacc           4      2    14  1.000000  0.500000  1.000000  0.500000  0.500000  0.700000
23  Clinic   test_acc           4      2    14  1.000000  0.500000  1.000000  0.500000  0.500000  0.700000
24  Clinic   test_auc           4      3    45  0.437500  0.375000  0.812500  0.250000  1.000000  0.575000
25  Clinic  test_bacc           4      3    45  0.687500  0.687500  0.562500  0.125000  0.785714  0.569643
26  Clinic   test_acc           4      3    45  0.500000  0.444444  0.222222  0.222222  0.625000  0.402778

6   Clinic   test_auc           3      0    20  0.833333  0.000000  0.750000  0.750000  1.000000  0.666667
7   Clinic  test_bacc           3      0    20  0.666667  0.000000  0.500000  0.500000  0.500000  0.433333
8   Clinic   test_acc           3      0    20  0.600000  0.000000  0.500000  0.500000  0.666667  0.453333
9   Clinic   test_auc           3      1    68  0.666667  0.939394  0.348485  0.590909  0.727273  0.654545
10  Clinic  test_bacc           3      1    68  0.560606  0.742424  0.393939  0.477273  0.659091  0.566667
11  Clinic   test_acc           3      1    68  0.500000  0.785714  0.428571  0.461538  0.769231  0.589011
12  Clinic   test_auc           3      2    36  0.333333  0.000000  0.100000  0.500000  0.200000  0.226667
13  Clinic  test_bacc           3      2    36  0.333333  0.000000  0.200000  0.450000  0.200000  0.236667
14  Clinic   test_acc           3      2    36  0.250000  0.000000  0.285714  0.428571  0.285714  0.250000

YEAH, 3 clusters:
Clust(0, N=20): 0.666667, 0.433333
Clust(1, N=68): 0.654545, 0.566667
Clust(2, N=36): 0.226667, 0.236667
'''

###############################################################################
# Clustering Im

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

scaler = preprocessing.StandardScaler()
X = scaler.fit(Xim).transform(Xim)
clustering = dict()
clustering['participant_id'] = pop['participant_id']

range_n_clusters = [2, 3, 4, 5, 6]
range_n_clusters = [2]

for n_clusters in range_n_clusters:
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)
    clustering["nbclusts:%i" % n_clusters] = cluster_labels
    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

## WIP HERE
from nilearn import datasets, plotting, image

import  nibabel
mask_img = nibabel.load(os.path.join(INPUT, "mask.nii.gz"))

coef_arr = np.zeros(mask_img.get_data().shape)
coef = clusterer.cluster_centers_[1, :]

coef = clusterer.cluster_centers_[1, :] - clusterer.cluster_centers_[0, :]

coef_arr[mask_img.get_data() != 0] = coef
pd.Series(np.abs(coef)).describe()


coef_img = nibabel.Nifti1Image(coef_arr, affine=mask_img.affine)

plotting.plot_glass_brain(coef_img, threshold=0.1)#, figure=fig, axes=ax)
## WIP HERE

"""
For n_clusters = 2 The average silhouette_score is : 0.0477990828217
For n_clusters = 3 The average silhouette_score is : 0.0192031806854
For n_clusters = 4 The average silhouette_score is : 0.017407720471
For n_clusters = 5 The average silhouette_score is : 0.00614266130751
For n_clusters = 6 The average silhouette_score is : 0.0112740545788
"""

clustering = pd.DataFrame(clustering)
clustering.to_csv(os.path.join(INPUT, DATASET+"-clust.csv"), index=False)

###############################################################################
# Clustering Im classifiy Clin

clustering = pd.read_csv(os.path.join(INPUT, DATASET+"-clust.csv"))
#pop = pd.read_csv(os.path.join(INPUT,"population.csv"))
assert np.all(pop.participant_id == clustering.participant_id)

confusion_matrix(yorig, clustering["nbclusts:2"])

"""
array([[17, 15],
       [45, 47]])
"""
confusion_matrix(yorig, clustering["nbclusts:3"])[:2, :]
"""
array([[ 6, 18,  8],
       [33, 32, 27]])
"""

range_n_clusters = [2, 3]
res = list()
for n_clusters in range_n_clusters:
    print("######################################")
    print("# nclust", n_clusters)
    cluster_labels = clustering["nbclusts:%i" % n_clusters]
    for clust in np.unique(cluster_labels):
        print("===================================")
        subset = cluster_labels == clust
        print(clust, subset.sum())
        Xg = Xclin[subset, :]
        yg = yorig[subset]
        Xg = scaler.fit(Xg).transform(Xg)

        cv = StratifiedKFold(n_splits=NFOLDS)
        model = lm.LogisticRegression(class_weight='balanced', fit_intercept=False, C=C)

        cv_results = cross_validate(estimator=copy.copy(model), X=Xg, y=yg, cv=cv, scoring=scorers, n_jobs=-1)
        res.append(['Clinic', 'test_auc', n_clusters, clust, subset.sum()] + cv_results["test_auc"].tolist() + [cv_results["test_auc"].mean()])
        res.append(['Clinic', 'test_bacc', n_clusters, clust, subset.sum()] + cv_results["test_bacc"].tolist() + [cv_results["test_bacc"].mean()])
        res.append(['Clinic', 'test_acc', n_clusters, clust, subset.sum()] + cv_results["test_acc"].tolist() + [cv_results["test_acc"].mean()])


res = pd.DataFrame(res, columns=['data', 'score', 'n_clusters', 'clust', 'size'] + ["fold%i" % i for i in range(NFOLDS)] + ['avg'])

"""
      data      score  n_clusters  clust  size     fold0     fold1     fold2     fold3     fold4       avg
0   Clinic   test_auc           2      0    62  0.555556  0.361111  0.703704  0.592593  0.518519  0.546296
1   Clinic  test_bacc           2      0    62  0.513889  0.361111  0.666667  0.555556  0.666667  0.552778
2   Clinic   test_acc           2      0    62  0.615385  0.307692  0.666667  0.666667  0.666667  0.584615
3   Clinic   test_auc           2      1    62  0.666667  0.566667  0.851852  1.000000  0.518519  0.720741
4   Clinic  test_bacc           2      1    62  0.583333  0.466667  0.777778  0.722222  0.611111  0.632222
5   Clinic   test_acc           2      1    62  0.538462  0.538462  0.833333  0.583333  0.750000  0.648718
6   Clinic   test_auc           3      0    39  0.071429  0.714286  1.000000  1.000000  0.833333  0.723810
7   Clinic  test_bacc           3      0    39  0.214286  0.857143  0.857143  0.666667  0.750000  0.669048
8   Clinic   test_acc           3      0    39  0.333333  0.750000  0.750000  0.428571  0.571429  0.566667
9   Clinic   test_auc           3      1    50  0.285714  0.107143  0.375000  0.833333  0.444444  0.409127
10  Clinic  test_bacc           3      1    50  0.339286  0.214286  0.416667  0.666667  0.333333  0.394048
11  Clinic   test_acc           3      1    50  0.363636  0.272727  0.400000  0.666667  0.444444  0.429495
12  Clinic   test_auc           3      2    35  0.500000  0.416667  0.900000  0.400000  0.200000  0.483333
13  Clinic  test_bacc           3      2    35  0.500000  0.500000  0.800000  0.400000  0.200000  0.480000
14  Clinic   test_acc           3      2    35  0.500000  0.500000  0.714286  0.666667  0.333333  0.542857

YEAH
2 clusters:
    - cluster 0(#62) AUC=0.54
    - cluster 1(#62) AUC=0.72, ACC=0.648718

0   Clinic   test_auc           2      0    62  0.555556  0.361111  0.703704  0.592593  0.518519  0.546296
1   Clinic  test_bacc           2      0    62  0.513889  0.361111  0.666667  0.555556  0.666667  0.552778
2   Clinic   test_acc           2      0    62  0.615385  0.307692  0.666667  0.666667  0.666667  0.584615
3   Clinic   test_auc           2      1    62  0.666667  0.566667  0.851852  1.000000  0.518519  0.720741
4   Clinic  test_bacc           2      1    62  0.583333  0.466667  0.777778  0.722222  0.611111  0.632222
5   Clinic   test_acc           2      1    62  0.538462  0.538462  0.833333  0.583333  0.750000  0.648718
"""


###############################################################################
# Clustering Im classifiy Im

clustering = pd.read_csv(os.path.join(INPUT, DATASET+"-clust.csv"))
#pop = pd.read_csv(os.path.join(INPUT,"population.csv"))
assert np.all(pop.participant_id == clustering.participant_id)

confusion_matrix(yorig, clustering["nbclusts:2"])

"""
array([[17, 15],
       [45, 47]])
"""
confusion_matrix(yorig, clustering["nbclusts:3"])[:2, :]
"""
array([[ 6, 18,  8],
       [33, 32, 27]])
"""

range_n_clusters = [2]
res = list()
for n_clusters in range_n_clusters:
    print("######################################")
    print("# nclust", n_clusters)
    cluster_labels = clustering["nbclusts:%i" % n_clusters]
    for clust in np.unique(cluster_labels):
        print("===================================")
        subset = cluster_labels == clust
        print(clust, subset.sum())
        Xg = Xim[subset, :]
        yg = yorig[subset]
        Xg = scaler.fit(Xg).transform(Xg)

        cv = StratifiedKFold(n_splits=NFOLDS)
        model = lm.LogisticRegression(class_weight='balanced', fit_intercept=False, C=C)

        cv_results = cross_validate(estimator=copy.copy(model), X=Xg, y=yg, cv=cv, scoring=scorers, n_jobs=-1)
        res.append(['Ima', 'test_auc', n_clusters, clust, subset.sum()] + cv_results["test_auc"].tolist() + [cv_results["test_auc"].mean()])
        res.append(['Ima', 'test_bacc', n_clusters, clust, subset.sum()] + cv_results["test_bacc"].tolist() + [cv_results["test_bacc"].mean()])
        res.append(['Ima', 'test_acc', n_clusters, clust, subset.sum()] + cv_results["test_acc"].tolist() + [cv_results["test_acc"].mean()])

res = pd.DataFrame(res, columns=['data', 'score', 'n_clusters', 'clust', 'size'] + ["fold%i" % i for i in range(NFOLDS)] + ['avg'])

"""
  data      score  n_clusters  clust  size     fold0     fold1     fold2     fold3     fold4       avg
0  Ima   test_auc           2      0    62  0.472222  0.694444  0.444444  0.111111  0.740741  0.492593
1  Ima  test_bacc           2      0    62  0.500000  0.500000  0.500000  0.500000  0.500000  0.500000
2  Ima   test_acc           2      0    62  0.307692  0.307692  0.250000  0.250000  0.250000  0.273077
3  Ima   test_auc           2      1    62  0.433333  0.866667  0.888889  0.814815  0.444444  0.689630
4  Ima  test_bacc           2      1    62  0.500000  0.500000  0.500000  0.500000  0.500000  0.500000
5  Ima   test_acc           2      1    62  0.230769  0.230769  0.250000  0.250000  0.250000  0.242308

YEAH:
3  Ima   test_auc           2      1    62  0.433333  0.866667  0.888889  0.814815  0.444444  0.689630

"""


###############################################################################
# Clustering Clin classifiy Im

clustering = pd.read_csv(os.path.join(INPUT, "Xclin-clust.csv"))
#pop = pd.read_csv(os.path.join(INPUT,"population.csv"))
assert np.all(pop.participant_id == clustering.participant_id)

confusion_matrix(yorig, clustering["nbclusts:2"])

"""
array([[16, 16],
       [32, 60]])
"""
confusion_matrix(yorig, clustering["nbclusts:3"])[:2, :]
"""
array([[ 9, 13, 10],
       [11, 55, 26]])
"""

range_n_clusters = [2, 3]
res = list()
for n_clusters in range_n_clusters:
    print("######################################")
    print("# nclust", n_clusters)
    cluster_labels = clustering["nbclusts:%i" % n_clusters]
    for clust in np.unique(cluster_labels):
        print("===================================")
        subset = cluster_labels == clust
        print(clust, subset.sum())
        Xg = Xim[subset, :]
        yg = yorig[subset]
        Xg = scaler.fit(Xg).transform(Xg)

        cv = StratifiedKFold(n_splits=NFOLDS)
        model = lm.LogisticRegression(class_weight='balanced', fit_intercept=False, C=C)

        cv_results = cross_validate(estimator=copy.copy(model), X=Xg, y=yg, cv=cv, scoring=scorers, n_jobs=-1)
        res.append(['Ima', 'test_auc', n_clusters, clust, subset.sum()] + cv_results["test_auc"].tolist() + [cv_results["test_auc"].mean()])
        res.append(['Ima', 'test_bacc', n_clusters, clust, subset.sum()] + cv_results["test_bacc"].tolist() + [cv_results["test_bacc"].mean()])
        res.append(['Ima', 'test_acc', n_clusters, clust, subset.sum()] + cv_results["test_acc"].tolist() + [cv_results["test_acc"].mean()])


res = pd.DataFrame(res, columns=['data', 'score', 'n_clusters', 'clust', 'size'] + ["fold%i" % i for i in range(NFOLDS)] + ['avg'])

"""
   data      score  n_clusters  clust  size     fold0     fold1     fold2     fold3     fold4       avg
0   Ima   test_auc           2      0    48  0.714286  0.380952  0.166667  0.333333  0.666667  0.452381
1   Ima  test_bacc           2      0    48  0.500000  0.500000  0.500000  0.500000  0.500000  0.500000
2   Ima   test_acc           2      0    48  0.363636  0.300000  0.333333  0.333333  0.333333  0.332727
3   Ima   test_auc           2      1    76  0.145833  0.027778  0.638889  0.333333  0.750000  0.379167
4   Ima  test_bacc           2      1    76  0.375000  0.500000  0.500000  0.500000  0.500000  0.475000
5   Ima   test_acc           2      1    76  0.187500  0.200000  0.200000  0.200000  0.200000  0.197500
6   Ima   test_auc           3      0    20  0.166667  0.500000  0.000000  0.000000  0.000000  0.133333
7   Ima  test_bacc           3      0    20  0.250000  0.500000  0.250000  0.500000  0.500000  0.400000
8   Ima   test_acc           3      0    20  0.200000  0.500000  0.250000  0.500000  0.333333  0.356667
9   Ima   test_auc           3      1    68  0.121212  0.363636  0.515152  0.772727  0.772727  0.509091
10  Ima  test_bacc           3      1    68  0.500000  0.500000  0.500000  0.500000  0.500000  0.500000
11  Ima   test_acc           3      1    68  0.214286  0.214286  0.214286  0.153846  0.153846  0.190110
12  Ima   test_auc           3      2    36  0.750000  0.400000  0.500000  0.500000  0.500000  0.530000
13  Ima  test_bacc           3      2    36  0.500000  0.500000  0.500000  0.500000  0.500000  0.500000
14  Ima   test_acc           3      2    36  0.250000  0.285714  0.285714  0.285714  0.285714  0.278571

NOTHING
"""

###############################################################################
# Clustering Im classifiy ClinIm

clustering = pd.read_csv(os.path.join(INPUT, DATASET+"-clust.csv"))
#pop = pd.read_csv(os.path.join(INPUT,"population.csv"))
assert np.all(pop.participant_id == clustering.participant_id)

confusion_matrix(yorig, clustering["nbclusts:2"])

"""
array([[17, 15],
       [45, 47]])
"""
confusion_matrix(yorig, clustering["nbclusts:3"])[:2, :]
"""
array([[ 6, 18,  8],
       [33, 32, 27]])
"""

"""
Use mlxtend
http://rasbt.github.io/mlxtend/
conda install -c conda-forge mlxtend

https://rasbt.github.io/mlxtend/user_guide/classifier/StackingClassifier/
"""
from mlxtend.classifier import StackingClassifier
from mlxtend.feature_selection import ColumnSelector
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

XClinIm = np.concatenate([Xclin, Xim], axis=1)

range_n_clusters = [2]
res = list()
for n_clusters in range_n_clusters:
    print("######################################")
    print("# nclust", n_clusters)
    cluster_labels = clustering["nbclusts:%i" % n_clusters]
    for clust in np.unique(cluster_labels):
        print("===================================")
        subset = cluster_labels == clust
        print(clust, subset.sum())
        Xg = XClinIm[subset, :]
        yg = yorig[subset]
        Xg = scaler.fit(Xg).transform(Xg)

        cv = StratifiedKFold(n_splits=NFOLDS)
        lr = lm.LogisticRegression(class_weight='balanced', fit_intercept=False, C=C)
        pipe1 = make_pipeline(ColumnSelector(cols=np.arange(0, Xclin.shape[1])),
                      copy.copy(lr))
        pipe2 = make_pipeline(ColumnSelector(cols=np.arange(Xclin.shape[1], XClinIm.shape[1])),
                              copy.copy(lr))
        model = StackingClassifier(classifiers=[pipe1, pipe2],
                                  meta_classifier=LogisticRegression())

        cv_results = cross_validate(estimator=copy.copy(model), X=Xg, y=yg, cv=cv, scoring=scorers, n_jobs=-1)
        res.append(['ClinIma', 'test_auc', n_clusters, clust, subset.sum()] + cv_results["test_auc"].tolist() + [cv_results["test_auc"].mean()])
        res.append(['ClinIma', 'test_bacc', n_clusters, clust, subset.sum()] + cv_results["test_bacc"].tolist() + [cv_results["test_bacc"].mean()])
        res.append(['ClinIma', 'test_acc', n_clusters, clust, subset.sum()] + cv_results["test_acc"].tolist() + [cv_results["test_acc"].mean()])


res = pd.DataFrame(res, columns=['data', 'score', 'n_clusters', 'clust', 'size'] + ["fold%i" % i for i in range(NFOLDS)] + ['avg'])

"""
      data      score  n_clusters  clust  size     fold0     fold1     fold2     fold3     fold4       avg
0  ClinIma   test_auc           2      0    62  0.513889  0.361111  0.666667  0.555556  0.666667  0.552778
1  ClinIma  test_bacc           2      0    62  0.500000  0.500000  0.500000  0.500000  0.500000  0.500000
2  ClinIma   test_acc           2      0    62  0.307692  0.307692  0.250000  0.250000  0.250000  0.273077
3  ClinIma   test_auc           2      1    62  0.583333  0.466667  0.777778  0.722222  0.611111  0.632222
4  ClinIma  test_bacc           2      1    62  0.500000  0.500000  0.500000  0.500000  0.500000  0.500000
5  ClinIma   test_acc           2      1    62  0.230769  0.230769  0.250000  0.250000  0.250000  0.242308

NOTHING
"""
###############################################################################
# No Clustering / classifiy ImEnettv

import nibabel
import parsimony.algorithms as algorithms
import parsimony.estimators as estimators
import parsimony.functions.nesterov.tv as nesterov_tv
from parsimony.utils.linalgs import LinearOperatorNesterov

# Data
Xim = np.load(os.path.join(OUTPUT, "XTreatTivSite.npy"))
yorig = np.load(os.path.join(OUTPUT, "y.npy"))
mask = nibabel.load(os.path.join(OUTPUT, "mask.nii.gz"))
# Atv = nesterov_tv.linear_operator_from_mask(mask.get_data(), calc_lambda_max=True)
# Atv.save(os.path.join(OUTPUT, "Atv.npz"))
Atv = LinearOperatorNesterov(filename=os.path.join(OUTPUT, "Atv.npz"))
#assert Atv.get_singular_values(0) == Atv_.get_singular_values(0)
assert np.allclose(Atv.get_singular_values(0), 11.956104408414376)

X = Xim.copy()
y = yorig.copy()

# parameters
key = 'enettv_0.1_0.1_0.8'.split("_")
algo, alpha, l1l2ratio, tvratio = key[0], float(key[1]), float(key[2]), float(key[3])
tv = alpha * tvratio
l1 = alpha * float(1 - tv) * l1l2ratio
l2 = alpha * float(1 - tv) * (1- l1l2ratio)

print(key, algo, alpha, l1, l2, tv)

scaler = preprocessing.StandardScaler()
X = scaler.fit(X).transform(X)

y_test_pred = np.zeros(len(y))
y_test_prob_pred = np.zeros(len(y))
y_test_decfunc_pred = np.zeros(len(y))
y_train_pred = np.zeros(len(y))
coefs_cv = np.zeros((NFOLDS, X.shape[1]))

auc_test = list()
recalls_test = list()

acc_test = list()

for cv_i, (train, test) in enumerate(cv.split(X, y)):
    #for train, test in cv.split(X, y, None):
    print(cv_i)
    X_train, X_test, y_train, y_test = X[train, :], X[test, :], y[train], y[test]
    #estimator = clone(model)
    conesta = algorithms.proximal.CONESTA(max_iter=10000)
    estimator = estimators.LogisticRegressionL1L2TV(l1, l2, tv, Atv, algorithm=conesta,
                                                    class_weight="auto", penalty_start=0)
    estimator.fit(X_train, y_train.ravel())
    # Store prediction for micro avg
    y_test_pred[test] = estimator.predict(X_test).ravel()
    y_test_prob_pred[test] = estimator.predict_probability(X_test).ravel()#[:, 1]
    #y_test_decfunc_pred[test] = estimator.decision_function(X_test)
    y_train_pred[train] = estimator.predict(X_train).ravel()
    # Compute score for macro avg
    auc_test.append(metrics.roc_auc_score(y_test, estimator.predict_probability(X_test).ravel()))
    recalls_test.append(metrics.recall_score(y_test, estimator.predict(X_test).ravel(), average=None))
    acc_test.append(metrics.accuracy_score(y_test, estimator.predict(X_test).ravel()))

    coefs_cv[cv_i, :] = estimator.beta.ravel()


np.savez_compressed(os.path.join(OUTPUT, DATASET+"_enettv_0.1_0.1_0.8_5cv.npz"),
                    coefs_cv=coefs_cv, y_pred=y_test_pred, y_true=y,
                    proba_pred=y_test_prob_pred, beta=coefs_cv)

# Micro Avg
recall_test_microavg = metrics.recall_score(y, y_test_pred, average=None)
recall_train_microavg = metrics.recall_score(y, y_train_pred, average=None)
bacc_test_microavg = recall_test_microavg.mean()
auc_test_microavg = metrics.roc_auc_score(y, y_test_prob_pred)
acc_test_microavg = metrics.accuracy_score(y, y_test_pred)

print(auc_test_microavg, bacc_test_microavg, acc_test_microavg)
# 0.438519021739 0.451766304348 0.443548387097

###############################################################################
# Clustering Im classifiy ImEnettv

import nibabel
import parsimony.algorithms as algorithms
import parsimony.estimators as estimators
import parsimony.functions.nesterov.tv as nesterov_tv
from parsimony.utils.linalgs import LinearOperatorNesterov

# Data
Xim = np.load(os.path.join(OUTPUT, "XTreatTivSite.npy"))
yorig = np.load(os.path.join(OUTPUT, "y.npy"))
mask = nibabel.load(os.path.join(OUTPUT, "mask.nii.gz"))
# Atv = nesterov_tv.linear_operator_from_mask(mask.get_data(), calc_lambda_max=True)
# Atv.save(os.path.join(OUTPUT, "Atv.npz"))
Atv = LinearOperatorNesterov(filename=os.path.join(OUTPUT, "Atv.npz"))
#assert Atv.get_singular_values(0) == Atv_.get_singular_values(0)
assert np.allclose(Atv.get_singular_values(0), 11.956104408414376)

# Cluster
clustering = pd.read_csv(os.path.join(INPUT, DATASET+"-clust.csv"))
#pop = pd.read_csv(os.path.join(INPUT,"population.csv"))
assert np.all(pop.participant_id == clustering.participant_id)
confusion_matrix(yorig, clustering["nbclusts:2"])
subset = clustering["nbclusts:2"] == 1

X = Xim[subset, :]
y = yorig[subset]

# parameters
key = 'enettv_0.1_0.1_0.8'.split("_")
algo, alpha, l1l2ratio, tvratio = key[0], float(key[1]), float(key[2]), float(key[3])
tv = alpha * tvratio
l1 = alpha * float(1 - tv) * l1l2ratio
l2 = alpha * float(1 - tv) * (1- l1l2ratio)

print(key, algo, alpha, l1, l2, tv)

scaler = preprocessing.StandardScaler()
X = scaler.fit(X).transform(X)

y_test_pred = np.zeros(len(y))
y_test_prob_pred = np.zeros(len(y))
y_test_decfunc_pred = np.zeros(len(y))
y_train_pred = np.zeros(len(y))
coefs_cv = np.zeros((NFOLDS, X.shape[1]))

auc_test = list()
recalls_test = list()

acc_test = list()

for cv_i, (train, test) in enumerate(cv.split(X, y)):
    #for train, test in cv.split(X, y, None):
    print(cv_i)
    X_train, X_test, y_train, y_test = X[train, :], X[test, :], y[train], y[test]
    #estimator = clone(model)
    conesta = algorithms.proximal.CONESTA(max_iter=10000)
    estimator = estimators.LogisticRegressionL1L2TV(l1, l2, tv, Atv, algorithm=conesta,
                                                    class_weight="auto", penalty_start=0)
    estimator.fit(X_train, y_train.ravel())
    # Store prediction for micro avg
    y_test_pred[test] = estimator.predict(X_test).ravel()
    y_test_prob_pred[test] = estimator.predict_probability(X_test).ravel()#[:, 1]
    #y_test_decfunc_pred[test] = estimator.decision_function(X_test)
    y_train_pred[train] = estimator.predict(X_train).ravel()
    # Compute score for macro avg
    auc_test.append(metrics.roc_auc_score(y_test, estimator.predict_probability(X_test).ravel()))
    recalls_test.append(metrics.recall_score(y_test, estimator.predict(X_test).ravel(), average=None))
    acc_test.append(metrics.accuracy_score(y_test, estimator.predict(X_test).ravel()))

    coefs_cv[cv_i, :] = estimator.beta.ravel()

np.savez_compressed(os.path.join(OUTPUT,  DATASET+"-clust"+"_enettv_0.1_0.1_0.8_5cv.npz"),
                    coefs_cv=coefs_cv, y_pred=y_test_pred, y_true=y,
                    proba_pred=y_test_prob_pred, beta=coefs_cv)

# Micro Avg
recall_test_microavg = metrics.recall_score(y, y_test_pred, average=None)
recall_train_microavg = metrics.recall_score(y, y_train_pred, average=None)
bacc_test_microavg = recall_test_microavg.mean()
auc_test_microavg = metrics.roc_auc_score(y, y_test_prob_pred)
acc_test_microavg = metrics.accuracy_score(y, y_test_pred)

print(auc_test_microavg, bacc_test_microavg, acc_test_microavg)
# YEAH !!

# 0.697872340426 0.678014184397 0.58064516129
###############################################################################
# Clustering Im classifiy ClinImEnettv


# Cluster
clustering = pd.read_csv(os.path.join(INPUT, DATASET+"-clust.csv"))
#pop = pd.read_csv(os.path.join(INPUT,"population.csv"))
assert np.all(pop.participant_id == clustering.participant_id)
confusion_matrix(yorig, clustering["nbclusts:2"])
subset = clustering["nbclusts:2"] == 1

# Data
Xim = np.load(os.path.join(OUTPUT, "XTreatTivSite.npy"))
yorig = np.load(os.path.join(OUTPUT, "y.npy"))
mask = nibabel.load(os.path.join(OUTPUT, "mask.nii.gz"))
# Atv = nesterov_tv.linear_operator_from_mask(mask.get_data(), calc_lambda_max=True)
# Atv.save(os.path.join(OUTPUT, "Atv.npz"))
Atv = LinearOperatorNesterov(filename=os.path.join(OUTPUT, "Atv.npz"))
#assert Atv.get_singular_values(0) == Atv_.get_singular_values(0)
assert np.allclose(Atv.get_singular_values(0), 11.956104408414376)

scaler = preprocessing.StandardScaler()
Ximg = Xim[subset, :]
Xcling = Xclin[subset, :]
y = yorig[subset]
Ximg = scaler.fit(Ximg).transform(Ximg)
Xcling = scaler.fit(Xcling).transform(Xcling)

# Load models
modelscv = np.load(os.path.join(OUTPUT,  DATASET+"-clust"+"_enettv_0.1_0.1_0.8_5cv.npz"))


# parameters
key = 'enettv_0.1_0.1_0.8'.split("_")
algo, alpha, l1l2ratio, tvratio = key[0], float(key[1]), float(key[2]), float(key[3])
tv = alpha * tvratio
l1 = alpha * float(1 - tv) * l1l2ratio
l2 = alpha * float(1 - tv) * (1- l1l2ratio)

print(key, algo, alpha, l1, l2, tv)

# CV loop

y_test_pred_img = np.zeros(len(y))
y_test_prob_pred_img = np.zeros(len(y))
y_test_decfunc_pred_img = np.zeros(len(y))
y_train_pred_img = np.zeros(len(y))
coefs_cv_img = np.zeros((NFOLDS, Ximg.shape[1]))
auc_test_img = list()
recalls_test_img = list()
acc_test_img = list()

y_test_pred_clin = np.zeros(len(y))
y_test_prob_pred_clin = np.zeros(len(y))
y_test_decfunc_pred_clin = np.zeros(len(y))
y_train_pred_clin = np.zeros(len(y))
coefs_cv_clin = np.zeros((NFOLDS, Xcling.shape[1]))
auc_test_clin = list()
recalls_test_clin = list()
acc_test_clin = list()

y_test_pred_stck = np.zeros(len(y))
y_test_prob_pred_stck = np.zeros(len(y))
y_test_decfunc_pred_stck = np.zeros(len(y))
y_train_pred_stck = np.zeros(len(y))
coefs_cv_stck = np.zeros((NFOLDS, 2))
auc_test_stck = list()
recalls_test_stck = list()
acc_test_stck = list()

for cv_i, (train, test) in enumerate(cv.split(Ximg, y)):
    #for train, test in cv.split(X, y, None):
    print(cv_i)
    X_train_img, X_test_img, y_train, y_test = Ximg[train, :], Ximg[test, :], y[train], y[test]
    X_train_clin, X_test_clin = Xcling[train, :], Xcling[test, :]

    # Im
    conesta = algorithms.proximal.CONESTA(max_iter=10000)
    estimator_img = estimators.LogisticRegressionL1L2TV(l1, l2, tv, Atv, algorithm=conesta,
                                                    class_weight="auto", penalty_start=0)
    estimator_img.beta = modelscv["coefs_cv"][cv_i][:, None]
    # Store prediction for micro avg
    y_test_pred_img[test] = estimator_img.predict(X_test_img).ravel()
    y_test_prob_pred_img[test] = estimator_img.predict_probability(X_test_img).ravel()#[:, 1]
    y_test_decfunc_pred_img[test] = np.dot(X_test_img, estimator_img.beta).ravel()
    y_train_pred_img[train] = estimator_img.predict(X_train_img).ravel()
    # Compute score for macro avg
    auc_test_img.append(metrics.roc_auc_score(y_test, estimator_img.predict_probability(X_test_img).ravel()))
    recalls_test_img.append(metrics.recall_score(y_test, estimator_img.predict(X_test_img).ravel(), average=None))
    acc_test_img.append(metrics.accuracy_score(y_test, estimator_img.predict(X_test_img).ravel()))
    coefs_cv_img[cv_i, :] = estimator_img.beta.ravel()

    # Clin
    estimator_clin = lm.LogisticRegression(class_weight='balanced', fit_intercept=False, C=C)
    estimator_clin.fit(X_train_clin, y_train)
    y_test_pred_clin[test] = estimator_clin.predict(X_test_clin).ravel()
    y_test_prob_pred_clin[test] =  estimator_clin.predict_proba(X_test_clin)[:, 1]
    y_test_decfunc_pred_clin[test] = estimator_clin.decision_function(X_test_clin)
    y_train_pred_clin[train] = estimator_clin.predict(X_train_clin).ravel()
    # Compute score for macro avg
    auc_test_clin.append(metrics.roc_auc_score(y_test, estimator_clin.predict_proba(X_test_clin)[:, 1]))
    recalls_test_clin.append(metrics.recall_score(y_test, estimator_clin.predict(X_test_clin).ravel(), average=None))
    acc_test_clin.append(metrics.accuracy_score(y_test, estimator_clin.predict(X_test_clin).ravel()))
    coefs_cv_clin[cv_i, :] = estimator_clin.coef_.ravel()

    # Stacking
    X_train_stck = np.c_[
            np.dot(X_train_img, estimator_img.beta).ravel(),
            estimator_clin.decision_function(X_train_clin).ravel()]
    X_test_stck = np.c_[
            np.dot(X_test_img, estimator_img.beta).ravel(),
            estimator_clin.decision_function(X_test_clin).ravel()]
    X_train_stck = scaler.fit(X_train_stck).transform(X_train_stck)
    X_test_stck = scaler.transform(X_test_stck)

    #
    estimator_stck = lm.LogisticRegression(class_weight='balanced', fit_intercept=False, C=10)
    estimator_stck.fit(X_train_stck, y_train)
    y_test_pred_stck[test] = estimator_stck.predict(X_test_stck).ravel()
    y_test_prob_pred_stck[test] =  estimator_stck.predict_proba(X_test_stck)[:, 1]
    y_test_decfunc_pred_stck[test] = estimator_stck.decision_function(X_test_stck)
    y_train_pred_stck[train] = estimator_stck.predict(X_train_stck).ravel()
    # Compute score for macro avg
    auc_test_stck.append(metrics.roc_auc_score(y_test, estimator_stck.predict_proba(X_test_stck)[:, 1]))
    recalls_test_stck.append(metrics.recall_score(y_test, estimator_stck.predict(X_test_stck).ravel(), average=None))
    acc_test_stck.append(metrics.accuracy_score(y_test, estimator_stck.predict(X_test_stck).ravel()))
    coefs_cv_stck[cv_i, :] = estimator_stck.coef_.ravel()


# Micro Avg Img
recall_test_img_microavg = metrics.recall_score(y, y_test_pred_img, average=None)
recall_train_img_microavg = metrics.recall_score(y, y_train_pred_img, average=None)
bacc_test_img_microavg = recall_test_img_microavg.mean()
auc_test_img_microavg = metrics.roc_auc_score(y, y_test_prob_pred_img)
acc_test_img_microavg = metrics.accuracy_score(y, y_test_pred_img)

print(auc_test_img_microavg, bacc_test_img_microavg, acc_test_img_microavg)
# 0.697872340426 0.678014184397 0.58064516129

# Micro Avg Clin
recall_test_clin_microavg = metrics.recall_score(y, y_test_pred_clin, average=None)
recall_train_clin_microavg = metrics.recall_score(y, y_train_pred_clin, average=None)
bacc_test_clin_microavg = recall_test_clin_microavg.mean()
auc_test_clin_microavg = metrics.roc_auc_score(y, y_test_prob_pred_clin)
acc_test_clin_microavg = metrics.accuracy_score(y, y_test_pred_clin)

print(auc_test_clin_microavg, bacc_test_clin_microavg, acc_test_clin_microavg)
# 0.663829787234 0.629787234043 0.645161290323

# Micro Avg Stacking
recall_test_stck_microavg = metrics.recall_score(y, y_test_pred_stck, average=None)
recall_train_stck_microavg = metrics.recall_score(y, y_train_pred_stck, average=None)
bacc_test_stck_microavg = recall_test_stck_microavg.mean()
auc_test_stck_microavg = metrics.roc_auc_score(y, y_test_prob_pred_stck)
acc_test_stck_microavg = metrics.accuracy_score(y, y_test_pred_stck)

print(auc_test_stck_microavg, bacc_test_stck_microavg, acc_test_stck_microavg)

# YEAH !!!
# lm.LogisticRegression(class_weight='balanced', fit_intercept=False, C=10)
# 0.70780141844 0.721985815603 0.612903225806

# lm.LogisticRegression(class_weight='balanced', fit_intercept=False, C=1)
# 0.710638297872 0.68865248227 0.596774193548

df= pop.copy()[subset]
df["decfunc_pred_img"] = y_test_decfunc_pred_img
df["decfunc_pred_clin"] = y_test_decfunc_pred_clin

sns.lmplot(x="decfunc_pred_clin", y="decfunc_pred_img", hue="respond_wk16" , data=df, fit_reg=False)

###############################################################################
import os
import seaborn as sns
import pandas as pd

from matplotlib.backends.backend_pdf import PdfPages

df = pd.read_csv(os.path.join(INPUT, DATASET+"-clust.csv"))
pop = pd.read_csv(os.path.join(INPUT, "population.csv"))
df = pd.merge(df, pop)
assert df.shape[0] == 124

df["cluster"] = df['nbclusts:2']
pdf = PdfPages(os.path.join(OUTPUT, DATASET+'-clust.pdf'))

fig = plt.figure()
#fig.suptitle('Cluster x GMratio')
sns.set(style="whitegrid")
sns.violinplot(x="cluster", y="GMratio", hue="respond_wk16", data=df, split=True, label=None, legend_out = True)
sns.swarmplot(x="cluster", y="GMratio", hue="respond_wk16", data=df,  dodge=True, linewidth=1, edgecolor='black')
plt.legend(loc='lower right')

pdf.savefig(); plt.close()

fig = plt.figure()
#fig.suptitle('Cluster x GMratio')
sns.lmplot(x="age", y="GMratio", hue="cluster", data=df)
pdf.savefig(); plt.close()

fig = plt.figure()
sns.lmplot(x="psyhis_mdd_age", y="GMratio", hue="cluster" , data=df, fit_reg=False)
pdf.savefig(); plt.close()

fig = plt.figure()
df_nona = df.copy()
df_nona.loc[df_nona["psyhis_mdd_age"].isnull(), "psyhis_mdd_age"] = df_nona["psyhis_mdd_age"].mean()
g = sns.PairGrid(df_nona[["GMratio", "age", "psyhis_mdd_age", "cluster", "respond_wk16"]], hue="cluster")
g.map_diag(plt.hist)
g.map_offdiag(plt.scatter)
pdf.savefig(); plt.close()


fig = plt.figure()
sns.lmplot(x="age", y="GMratio", hue="respond_wk16", col="cluster", data=df, fit_reg=False)
pdf.savefig(); plt.close()

"""
#sns.lmplot(x="test_decfunc", y="GMratio", hue="respond_wk16", col="cluster", data=df, fit_reg=False)
fig = plt.figure()
tmp = df[["test_decfunc-clust-1", "respond_wk16", "cluster", "GMratio", "psyhis_mdd_age", "age"]].dropna()
sns.distplot(tmp["test_decfunc-clust-1"][tmp["respond_wk16"] == "Responder"], rug=True, color="red")
sns.distplot(tmp["test_decfunc-clust-1"][tmp["respond_wk16"] == "NonResponder"], rug=True, color="blue")
pdf.savefig(); plt.close()

fig = plt.figure()
sns.lmplot(x="test_decfunc-clust-1", y="GMratio", hue="respond_wk16" , data=tmp, fit_reg=False)
pdf.savefig(); plt.close()

fig = plt.figure()
sns.lmplot(x="test_decfunc-clust-1", y="psyhis_mdd_age", hue="respond_wk16" , data=tmp, fit_reg=False)
pdf.savefig(); plt.close()

fig = plt.figure()
sns.lmplot(x="test_decfunc-clust-1", y="age", hue="respond_wk16" , data=tmp, fit_reg=False)
pdf.savefig(); plt.close()

fig = plt.figure()
tmp = df[["test_decfunc", "respond_wk16", "GMratio", "psyhis_mdd_age", "age"]]
sns.distplot(tmp["test_decfunc"][tmp["respond_wk16"] == "Responder"], rug=True, color="red")
sns.distplot(tmp["test_decfunc"][tmp["respond_wk16"] == "NonResponder"], rug=True, color="blue")
pdf.savefig(); plt.close()

fig = plt.figure()
sns.lmplot(x="test_decfunc", y="GMratio", hue="respond_wk16" , data=tmp, fit_reg=False)
pdf.savefig(); plt.close()

fig = plt.figure()
sns.lmplot(x="test_decfunc", y="psyhis_mdd_age", hue="respond_wk16" , data=tmp, fit_reg=False)
pdf.savefig(); plt.close()

fig = plt.figure()
sns.lmplot(x="test_decfunc", y="age", hue="respond_wk16" , data=tmp, fit_reg=False)
pdf.savefig(); plt.close()
"""
pdf.close()

    # Stack
###############################################################################
# OLDIES

CLUSTER = 1
subset = clustering.cluster == CLUSTER
if DATASET == "XTreatTivSite-ClinIm":
    X = np.concatenate([Xclin, Xim], axis=1)
if DATASET == "XTreatTivSite-Im":
    X = np.copy(Xim)
#X = np.concatenate([Xclin, Xim], axis=1)
X = X[subset, :]
y = yorig[subset]

X = scaler.fit(X).transform(X)

model.fit(X, y)
model.coef_

# clustering
NFOLDS = 5
C = 0.1 if X.shape[0] == 62 else 1

# All
def balanced_acc(estimator, X, y):
    return metrics.recall_score(y, estimator.predict(X), average=None).mean()
scorers = {'auc': 'roc_auc', 'bacc':balanced_acc, 'acc':'accuracy'}

cv = StratifiedKFold(n_splits=NFOLDS)
model = lm.LogisticRegression(class_weight='balanced', fit_intercept=False, C=C)
estimator = model
%time cv_results = cross_validate(estimator=model, X=X, y=y, cv=cv, scoring=scorers, n_jobs=-1)
print(
      cv_results["test_auc"], cv_results["test_auc"].mean(), "\n",
      cv_results["test_bacc"], cv_results["test_bacc"].mean(), "\n",
      cv_results["test_acc"], cv_results["test_acc"].mean())



XTreatTivSite-ClinIm
[ 0.48120301  0.57894737  0.66666667  0.43518519  0.77777778] 0.587956001114
[ 0.42857143  0.5         0.5         0.5         0.5       ] 0.485714285714
[ 0.23076923  0.26923077  0.25        0.25        0.25      ] 0.25

XTreatTivSite-ClinIm-clust-1
[ 0.43333333  0.86666667  0.88888889  0.81481481  0.44444444] 0.68962962963
[ 0.5  0.5  0.5  0.5  0.5] 0.5
[ 0.23076923  0.23076923  0.25        0.25        0.25      ] 0.242307692308

XTreatTivSite-Im-clust-1

[ 0.43333333  0.86666667  0.88888889  0.81481481  0.44444444] 0.68962962963
[ 0.5  0.5  0.5  0.5  0.5] 0.5
[ 0.23076923  0.23076923  0.25        0.25        0.25      ] 0.242307692308
"""

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

y_test_pred = np.zeros(len(y))
y_test_prob_pred = np.zeros(len(y))
y_test_decfunc_pred = np.zeros(len(y))
y_train_pred = np.zeros(len(y))
coefs_cv = np.zeros((NFOLDS, X.shape[1]))

auc_test = list()
recalls_test = list()
acc_test = list()

for cv_i, (train, test) in enumerate(cv.split(X, y)):
    #for train, test in cv.split(X, y, None):
    print(cv_i)
    X_train, X_test, y_train, y_test = X[train, :], X[test, :], y[train], y[test]
    #estimator = clone(model)
    estimator = lm.LogisticRegression(class_weight='balanced', fit_intercept=False, C=C)
    estimator.fit(X_train, y_train)
    # Store prediction for micro avg
    y_test_pred[test] = estimator.predict(X_test)
    y_test_prob_pred[test] = estimator.predict_proba(X_test)[:, 1]
    y_test_decfunc_pred[test] = estimator.decision_function(X_test)
    y_train_pred[train] = estimator.predict(X_train)
    # Compute score for macro avg
    auc_test.append(metrics.roc_auc_score(y_test, estimator.predict_proba(X_test)[:, 1]))
    recalls_test.append(metrics.recall_score(y_test, estimator.predict(X_test), average=None))
    acc_test.append(metrics.accuracy_score(y_test, estimator.predict(X_test)))

    coefs_cv[cv_i, :] = estimator.coef_

# Macro Avg
auc_test = np.array(auc_test)
recalls_test = np.array(recalls_test)
acc_test = np.array(acc_test)

# Micro Avg
recall_test_microavg = metrics.recall_score(y, y_test_pred, average=None)
recall_train_microavg = metrics.recall_score(y, y_train_pred, average=None)
bacc_test_microavg = recall_test_microavg.mean()
auc_test_microavg = metrics.roc_auc_score(y, y_test_prob_pred)
acc_test_microavg = metrics.accuracy_score(y, y_test_pred)

print("AUC (Macro-cross_validate/Macro-manual-loop/Micro-manual-loop)")
print(cv_results["test_auc"], cv_results["test_auc"].mean())
print(auc_test, auc_test.mean())
print(auc_test_microavg)
import scipy.stats as stats
print(stats.mannwhitneyu(y_test_decfunc_pred[y == 0], y_test_decfunc_pred[y == 1]))

print("bAcc (Macro-cross_validate/Macro-manual-loop/Micro-manual-loop)")
print(cv_results["test_bacc"], cv_results["test_bacc"].mean())
print(recalls_test.mean(axis=1), recalls_test.mean(axis=1).mean())
print(bacc_test_microavg)

print("Acc (Macro-cross_validate/Macro-manual-loop/Micro-manual-loop)")
print(cv_results["test_acc"], cv_results["test_acc"].mean())
print(acc_test, acc_test.mean())
print(acc_test_microavg)



df = pd.read_csv(os.path.join(INPUT, DATASET+"-clust.csv"))

if X.shape[0] == 62:
    df["test_decfunc-clust-1"] = np.NaN
    df.loc[df.cluster == CLUSTER, "test_decfunc-clust-1"] = y_test_decfunc_pred

if X.shape[0] == 124:
    df["test_decfunc"] = y_test_decfunc_pred

df.to_csv(os.path.join(INPUT, DATASET+"-clust.csv"), index=False)

"""
XTreatTivSite-ClinIm
AUC (Macro-cross_validate/Macro-manual-loop/Micro-manual-loop)
[ 0.48120301  0.57894737  0.66666667  0.43518519  0.77777778] 0.587956001114
[ 0.48120301  0.57894737  0.66666667  0.43518519  0.77777778] 0.587956001114
0.547214673913
MannwhitneyuResult(statistic=1333.0, pvalue=0.21450387817058197)
bAcc (Macro-cross_validate/Macro-manual-loop/Micro-manual-loop)
[ 0.42857143  0.5         0.5         0.5         0.5       ] 0.485714285714
[ 0.42857143  0.5         0.5         0.5         0.5       ] 0.485714285714
0.484375
Acc (Macro-cross_validate/Macro-manual-loop/Micro-manual-loop)
[ 0.23076923  0.26923077  0.25        0.25        0.25      ] 0.25
[ 0.23076923  0.26923077  0.25        0.25        0.25      ] 0.25
0.25

XTreatTivSite-ClinIm-clust-1
AUC (Macro-cross_validate/Macro-manual-loop/Micro-manual-loop)
[ 0.43333333  0.86666667  0.88888889  0.81481481  0.44444444] 0.68962962963
[ 0.43333333  0.86666667  0.88888889  0.81481481  0.44444444] 0.68962962963
0.651063829787
MannwhitneyuResult(statistic=246.0, pvalue=0.040724923246777296)
bAcc (Macro-cross_validate/Macro-manual-loop/Micro-manual-loop)
[ 0.5  0.5  0.5  0.5  0.5] 0.5
[ 0.5  0.5  0.5  0.5  0.5] 0.5
0.5
Acc (Macro-cross_validate/Macro-manual-loop/Micro-manual-loop)
[ 0.23076923  0.23076923  0.25        0.25        0.25      ] 0.242307692308
[ 0.23076923  0.23076923  0.25        0.25        0.25      ] 0.242307692308
0.241935483871

XTreatTivSite-Im-clust-1
AUC (Macro-cross_validate/Macro-manual-loop/Micro-manual-loop)
[ 0.43333333  0.86666667  0.88888889  0.81481481  0.44444444] 0.68962962963
[ 0.43333333  0.86666667  0.88888889  0.81481481  0.44444444] 0.68962962963
0.651063829787
MannwhitneyuResult(statistic=246.0, pvalue=0.040724923246777296)
bAcc (Macro-cross_validate/Macro-manual-loop/Micro-manual-loop)
[ 0.5  0.5  0.5  0.5  0.5] 0.5
[ 0.5  0.5  0.5  0.5  0.5] 0.5
0.5
Acc (Macro-cross_validate/Macro-manual-loop/Micro-manual-loop)
[ 0.23076923  0.23076923  0.25        0.25        0.25      ] 0.242307692308
[ 0.23076923  0.23076923  0.25        0.25        0.25      ] 0.242307692308
0.241935483871
"""
###############################################################################
# Stack demo with Ima

X = np.concatenate([Xclin[subset, :], y_test_decfunc_pred[:, None]], axis=1)
X = Xclin[subset, :]
#X = Xclin.copy()
#y = yorig.copy()
X = scaler.fit(X).transform(X)

cv = StratifiedKFold(n_splits=NFOLDS)
model = lm.LogisticRegression(class_weight='balanced', fit_intercept=False, C=C)

estimator = model
%time cv_results = cross_validate(estimator=model, X=X, y=y, cv=cv, scoring=scorers, n_jobs=-1)
print(
      cv_results["test_auc"], cv_results["test_auc"].mean(), "\n",
      cv_results["test_bacc"], cv_results["test_bacc"].mean(), "\n",
      cv_results["test_acc"], cv_results["test_acc"].mean())

'''
XTreatTivSite-Im-clust-1

[ 0.7         0.46666667  0.7037037   0.92592593  0.66666667] 0.692592592593
[ 0.58333333  0.51666667  0.72222222  0.77777778  0.61111111] 0.642222222222
[ 0.53846154  0.61538462  0.75        0.66666667  0.75      ] 0.664102564103

Xclin
[ 0.39849624  0.51879699  0.66203704  0.65740741  0.49074074] 0.545495683654
[ 0.45112782  0.47744361  0.66666667  0.69444444  0.41666667] 0.54126984127
[ 0.46153846  0.5         0.66666667  0.625       0.54166667] 0.558974358974
'''



###############################################################################
# Clinic only

X = Xclin.copy()
y = yorig.copy()
X = scaler.fit(X).transform(X)

cv = StratifiedKFold(n_splits=NFOLDS)
model = lm.LogisticRegression(class_weight='balanced', fit_intercept=False, C=C)

estimator = model
%time cv_results = cross_validate(estimator=model, X=X, y=y, cv=cv, scoring=scorers, n_jobs=-1)
print(
      cv_results["test_auc"], cv_results["test_auc"].mean(), "\n",
      cv_results["test_bacc"], cv_results["test_bacc"].mean(), "\n",
      cv_results["test_acc"], cv_results["test_acc"].mean())

'''
[ 0.39849624  0.51879699  0.66203704  0.65740741  0.49074074] 0.545495683654
 [ 0.45112782  0.47744361  0.66666667  0.69444444  0.41666667] 0.54126984127
 [ 0.46153846  0.5         0.66666667  0.625       0.54166667] 0.558974358974
'''


np.r_['-1',
  np.r_["test_auc", cv_results["test_auc"], cv_results["test_auc"].mean()],
  np.r_[cv_results["test_bacc"], cv_results["test_bacc"].mean()],
  np.r_[cv_results["test_acc"], cv_results["test_acc"].mean()]]
