#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 09:07:06 2018

@author: ed203246
"""

"""
Created on Thu Jul  5 18:41:41 2018

@author: ed203246

cp /neurospin/psy/canbind/models/vbm_resp_1.5mm/XTreatTivSite.npy /neurospin/psy/canbind/models/clustering_v02/
cp /neurospin/psy/canbind/models/vbm_resp_1.5mm/XTreatTivSitePca.npy /neurospin/psy/canbind/models/clustering_v02/
cp /neurospin/psy/canbind/models/vbm_resp_1.5mm/population.csv /neurospin/psy/canbind/models/clustering_v02/
cp /neurospin/psy/canbind/models/vbm_resp_1.5mm/mask.nii.gz /neurospin/psy/canbind/models/clustering_v02/
cp /neurospin/psy/canbind/models/vbm_resp_1.5mm/y.npy /neurospin/psy/canbind/models/clustering_v02/

laptop to desktop
rsync -azvun /home/edouard/data/psy/canbind/models/clustering_v02/* ed203246@is234606.intra.cea.fr:/neurospin/psy/canbind/models/clustering_v02/

desktop to laptop
rsync -azvu ed203246@is234606.intra.cea.fr:/neurospin/psy/canbind/models/clustering_v02/* /home/edouard/data/psy/canbind/models/clustering_v02/

WD = '/neurospin/psy/canbind'
WD = '/home/edouard/data/psy/canbind'
"""

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
import getpass

if getpass.getuser() == 'ed203246':
    BASEDIR = '/neurospin/psy/canbind'
elif getpass.getuser() == 'edouard':
    BASEDIR = '/home/edouard/data/psy/canbind'

# Voxel size
# vs = "1mm"
#vs = "1.5mm-s8mm"
vs = "1.5mm"

"""
Xa = np.load("/neurospin/psy/canbind/models/vbm_resp_1.5mm/XTreatTivSite.npy")
Xb = np.load("/neurospin/psy/canbind/models/vbm_resp_1.5mm-/XTreatTivSite.npy")
np.all(Xa == Xb)
ya = np.load("/neurospin/psy/canbind/models/vbm_resp_1.5mm/y.npy")
yb = np.load("/neurospin/psy/canbind/models/vbm_resp_1.5mm-/y.npy")
np.all(ya == yb)
"""

WD = os.path.join(BASEDIR, "models", "clustering_v02")
os.chdir(WD)

# load data
#DATASET = "XTreatTivSitePca"
#DATASET = "XTreatTivSite-ClinIm"
IMADATASET = "XTreatTivSite"
# IMADATASET = "XTreatTivSitePca"

#X = np.load(os.path.join(WD, "Xres.npy"))
#Xim = np.load(os.path.join(WD, "Xrawsc.npy"))
Xim = np.load(os.path.join(WD, IMADATASET+".npy"))


yorig = np.load(os.path.join(WD, "y.npy"))

pop = pd.read_csv(os.path.join(WD, "population.csv"))
assert np.all(pop['respond_wk16_num'] == yorig)


###############################################################################
# ML
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import StratifiedKFold
import copy

#clustering = pd.read_csv(os.path.join(WD, DATASET+"-clust.csv"))
#cluster_labels = clustering.cluster
C = 0.1
NFOLDS = 5
cv = StratifiedKFold(n_splits=NFOLDS)
model = lm.LogisticRegression(class_weight='balanced', fit_intercept=False, C=C)
scaler = preprocessing.StandardScaler()
def balanced_acc(estimator, X, y):
    return metrics.recall_score(y, estimator.predict(X), average=None).mean()
scorers = {'auc': 'roc_auc', 'bacc':balanced_acc, 'acc':'accuracy'}


#############################################################################
# Clinical data: imput missing
democlin = pop[['participant_id', 'age', 'sex_num', 'educ', 'age_onset',
                'respond_wk16',
                'mde_num', 'madrs_Baseline', 'madrs_Screening']]
democlin.describe()

"""
              age     sex_num        educ   age_onset    mde_num  madrs_Baseline  madrs_Screening
count  124.000000  124.000000  123.000000  118.000000  88.000000      120.000000       117.000000
mean    35.693548    0.620968   16.813008   20.983051   3.840909       29.975000        30.427350
std     12.560214    0.487114    2.255593    9.964881   2.495450        5.630742         5.234692
min     18.000000    0.000000    9.000000    5.000000   1.000000       21.000000        22.000000
25%     25.000000    0.000000   16.000000   14.250000   2.000000       25.750000        27.000000
50%     33.000000    1.000000   17.000000   18.000000   3.000000       29.000000        29.000000
75%     46.000000    1.000000   19.000000   25.750000   5.000000       34.000000        33.000000
max     61.000000    1.000000   21.000000   55.000000  10.000000       47.000000        46.000000
"""
democlin.isnull().sum()
"""
age                 0
sex_num             0
educ                1
age_onset           6
respond_wk16        0
mde_num            36
madrs_Baseline      4
madrs_Screening     7
"""

# Imput missing value with the median

democlin.loc[democlin["educ"].isnull(), "educ"] = democlin["educ"].median()
democlin.loc[democlin["age_onset"].isnull(), "age_onset"] = democlin["age_onset"].median()
democlin.loc[democlin["mde_num"].isnull(), "mde_num"] = democlin["mde_num"].median()


democlin.loc[democlin["madrs_Baseline"].isnull(), "madrs_Baseline"] = democlin.loc[democlin["madrs_Baseline"].isnull(), "madrs_Screening"]
assert democlin["madrs_Baseline"].isnull().sum() == 0

democlin.pop("madrs_Screening")
assert(np.all(democlin.isnull().sum() == 0))

# add duration
democlin["duration"] = democlin["age"] - democlin["age_onset"]

if os.path.exists("demo-clin-imputed.csv"):
    democlin_ = pd.read_csv("demo-clin-imputed.csv")
    assert np.all(democlin_ == democlin)
else:
    democlin.to_csv("demo-clin-imputed.csv", index=False)

# Rm participant_id & response
np.all(democlin.pop('participant_id') == pop['participant_id'])
resp_ = democlin.pop("respond_wk16")
assert np.all((resp_ == "Responder") == yorig)

democlin.columns
"""
['age', 'sex_num', 'educ', 'age_onset', 'mde_num', 'madrs_Baseline', 'duration']
"""
Xclin = np.asarray(democlin)


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
For n_clusters = 2 The average silhouette_score is : 0.236327338035
For n_clusters = 3 The average silhouette_score is : 0.248227481469
For n_clusters = 4 The average silhouette_score is : 0.257889640987
For n_clusters = 5 The average silhouette_score is : 0.249662837914
For n_clusters = 6 The average silhouette_score is : 0.216915960545
For n_clusters = 7 The average silhouette_score is : 0.213423981512
For n_clusters = 8 The average silhouette_score is : 0.218288231815
For n_clusters = 9 The average silhouette_score is : 0.219306207968

array([[16, 16],
       [32, 60]])
'''

X = scaler.fit(Xclin).transform(Xclin)
clustering = dict()
clustering['participant_id'] = pop['participant_id']

range_n_clusters = [2, 3]
res = list()
for n_clusters in range_n_clusters:
    print("######################################")
    print("# nclust", n_clusters)
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)
    clustering["nclust=%i" % n_clusters] = cluster_labels
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
clustering.to_csv(os.path.join(WD, "Xclin-clust.csv"), index=False)

clustering["nclust=2"]

res = pd.DataFrame(res, columns=['data', 'score', 'n_clusters', 'clust', 'size'] + ["fold%i" % i for i in range(NFOLDS)] + ['avg'])

print(res)
'''
      data      score  n_clusters  clust  size     fold0     fold1     fold2     fold3     fold4       avg
0   Clinic   test_auc           2      0    42  0.555556  0.388889  0.444444  0.666667  0.300000  0.471111
1   Clinic  test_bacc           2      0    42  0.583333  0.416667  0.333333  0.633333  0.350000  0.463333
2   Clinic   test_acc           2      0    42  0.555556  0.444444  0.333333  0.625000  0.285714  0.448810
3   Clinic   test_auc           2      1    82  0.326923  0.634615  0.576923  0.282051  0.222222  0.408547
4   Clinic  test_bacc           2      1    82  0.355769  0.605769  0.682692  0.346154  0.250000  0.448077
5   Clinic   test_acc           2      1    82  0.411765  0.529412  0.647059  0.562500  0.400000  0.510147
6   Clinic   test_auc           3      0    19  1.000000  0.500000  1.000000  1.000000  0.000000  0.700000
7   Clinic  test_bacc           3      0    19  0.833333  0.583333  0.750000  0.500000  0.250000  0.583333
8   Clinic   test_acc           3      0    19  0.800000  0.600000  0.666667  0.666667  0.333333  0.613333
9   Clinic   test_auc           3      1    67  0.666667  0.545455  0.545455  0.500000  0.400000  0.531515
10  Clinic  test_bacc           3      1    67  0.606061  0.439394  0.469697  0.566667  0.500000  0.516364
11  Clinic   test_acc           3      1    67  0.571429  0.500000  0.357143  0.692308  0.500000  0.524176
12  Clinic   test_auc           3      2    38  0.500000  0.250000  0.100000  0.400000  0.200000  0.290000
13  Clinic  test_bacc           3      2    38  0.500000  0.250000  0.250000  0.700000  0.100000  0.360000
14  Clinic   test_acc           3      2    38  0.444444  0.375000  0.142857  0.571429  0.142857  0.335317

Nothing
'''

###############################################################################
# Clustering Im

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

scaler = preprocessing.StandardScaler()
X = scaler.fit(Xim).transform(Xim)
clustering = dict()
clustering['participant_id'] = pop['participant_id']

range_n_clusters = [2, 3, 4]
#range_n_clusters = [2]

for n_clusters in range_n_clusters:
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)
    clustering["nclust=%i" % n_clusters] = cluster_labels
    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

"""
XTreatTivSite
For n_clusters = 2 The average silhouette_score is : 0.0477990828217
For n_clusters = 3 The average silhouette_score is : 0.0192031806854
For n_clusters = 4 The average silhouette_score is : 0.017407720471

XTreatTivSitePca
For n_clusters = 2 The average silhouette_score is : 0.00444455719081
For n_clusters = 3 The average silhouette_score is : 0.00226944083272
For n_clusters = 4 The average silhouette_score is : -0.00550654879244
"""
clustering = pd.DataFrame(clustering)
clustering.to_csv(os.path.join(WD, IMADATASET+"-clust.csv"), index=False)

# refit with 2 cluster
clusterer = KMeans(n_clusters=2, random_state=10)
cluster_labels_ = clusterer.fit_predict(X)
assert np.all(cluster_labels_ == clustering["nclust=2"])

np.savez_compressed(os.path.join(WD, IMADATASET+"-clust_centers.npz"),
                    cluster_labels=cluster_labels_,
                    cluster_centers=clusterer.cluster_centers_)

clustering = pd.DataFrame(clustering)
clustering.to_csv(os.path.join(WD, IMADATASET+"-clust.csv"), index=False)

###############################################################################
# Clustering Im classifiy Clin

clustering = pd.read_csv(os.path.join(WD, IMADATASET+"-clust.csv"))
#pop = pd.read_csv(os.path.join(WD,"population.csv"))
assert np.all(pop.participant_id == clustering.participant_id)

metrics.confusion_matrix(yorig, clustering["nclust=2"])

"""
array([[17, 15],
       [45, 47]])

XTreatTivSitePca
array([[ 6, 26],
       [31, 61]])
"""
metrics.confusion_matrix(yorig, clustering["nclust=3"])[:2, :]
"""
array([[ 6, 18,  8],
       [33, 32, 27]])
"""

range_n_clusters = [2]#, 3]
res = list()
for n_clusters in range_n_clusters:
    print("######################################")
    print("# nclust", n_clusters)
    cluster_labels = clustering["nclust=%i" % n_clusters]
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
print(res)

"""
XTreatTivSite
      data      score  n_clusters  clust  size     fold0     fold1     fold2     fold3     fold4       avg
0   Clinic   test_auc           2      0    62  0.694444  0.472222  0.592593  0.740741  0.222222  0.544444
1   Clinic  test_bacc           2      0    62  0.458333  0.486111  0.555556  0.611111  0.166667  0.455556
2   Clinic   test_acc           2      0    62  0.538462  0.384615  0.500000  0.750000  0.250000  0.484615
3   Clinic   test_auc           2      1    62  0.700000  0.566667  0.888889  0.925926  0.740741  0.764444
4   Clinic  test_bacc           2      1    62  0.533333  0.466667  0.833333  0.833333  0.555556  0.644444
5   Clinic   test_acc           2      1    62  0.461538  0.538462  0.750000  0.750000  0.666667  0.633333
6   Clinic   test_auc           3      0    39  0.142857  0.857143  1.000000  1.000000  1.000000  0.800000
7   Clinic  test_bacc           3      0    39  0.142857  0.857143  0.785714  0.750000  0.833333  0.673810
8   Clinic   test_acc           3      0    39  0.222222  0.750000  0.625000  0.571429  0.714286  0.576587
9   Clinic   test_auc           3      1    50  0.285714  0.392857  0.333333  0.611111  0.611111  0.446825
10  Clinic  test_bacc           3      1    50  0.464286  0.339286  0.416667  0.666667  0.500000  0.477381
11  Clinic   test_acc           3      1    50  0.454545  0.363636  0.400000  0.555556  0.555556  0.465859
12  Clinic   test_auc           3      2    35  0.416667  0.500000  1.000000  0.000000  0.200000  0.423333
13  Clinic  test_bacc           3      2    35  0.416667  0.500000  0.900000  0.400000  0.100000  0.463333
14  Clinic   test_acc           3      2    35  0.375000  0.500000  0.857143  0.666667  0.166667  0.513095

YEAH
2 clusters:
3   Clinic   test_auc           2      1    62  0.700000  0.566667  0.888889  0.925926  0.740741  0.764444
4   Clinic  test_bacc           2      1    62  0.533333  0.466667  0.833333  0.833333  0.555556  0.644444
5   Clinic   test_acc           2      1    62  0.461538  0.538462  0.750000  0.750000  0.666667  0.633333

XTreatTivSitePca
     data      score  n_clusters  clust  size     fold0     fold1     fold2     fold3     fold4       avg
0  Clinic   test_auc           2      0    37  0.357143  0.666667  0.333333  0.500000  0.333333  0.438095
1  Clinic  test_bacc           2      0    37  0.535714  0.750000  0.250000  0.250000  0.583333  0.473810
2  Clinic   test_acc           2      0    37  0.555556  0.571429  0.428571  0.428571  0.285714  0.453968
3  Clinic   test_auc           2      1    87  0.538462  0.366667  0.866667  0.550000  0.283333  0.521026
4  Clinic  test_bacc           2      1    87  0.512821  0.308333  0.733333  0.591667  0.350000  0.499231
5  Clinic   test_acc           2      1    87  0.578947  0.352941  0.705882  0.588235  0.411765  0.527554
"""


###############################################################################
# Clustering Im classifiy Im

clustering = pd.read_csv(os.path.join(WD, IMADATASET+"-clust.csv"))
#pop = pd.read_csv(os.path.join(WD,"population.csv"))
assert np.all(pop.participant_id == clustering.participant_id)

metrics.confusion_matrix(yorig, clustering["nclust=2"])

"""
array([[17, 15],
       [45, 47]])

    XTreatTivSitePca
array([[ 6, 26],
       [31, 61]])
"""
metrics.confusion_matrix(yorig, clustering["nclust=3"])[:2, :]
"""
array([[ 6, 18,  8],
       [33, 32, 27]])
"""

range_n_clusters = [2]
res = list()
for n_clusters in range_n_clusters:
    print("######################################")
    print("# nclust", n_clusters)
    cluster_labels = clustering["nclust=%i" % n_clusters]
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
print(res)

"""
XTreatTivSite
  data      score  n_clusters  clust  size     fold0     fold1     fold2     fold3     fold4       avg
0  Ima   test_auc           2      0    62  0.472222  0.694444  0.444444  0.111111  0.740741  0.492593
1  Ima  test_bacc           2      0    62  0.500000  0.500000  0.500000  0.500000  0.500000  0.500000
2  Ima   test_acc           2      0    62  0.307692  0.307692  0.250000  0.250000  0.250000  0.273077
3  Ima   test_auc           2      1    62  0.433333  0.866667  0.888889  0.814815  0.444444  0.689630
4  Ima  test_bacc           2      1    62  0.500000  0.500000  0.500000  0.500000  0.500000  0.500000
5  Ima   test_acc           2      1    62  0.230769  0.230769  0.250000  0.250000  0.250000  0.242308

YEAH:
3  Ima   test_auc           2      1    62  0.433333  0.866667  0.888889  0.814815  0.444444  0.689630

XTreatTivSitePca
  data      score  n_clusters  clust  size     fold0     fold1     fold2     fold3     fold4       avg
0  Ima   test_auc           2      0    37  0.500000  0.333333  0.333333  0.000000  0.000000  0.233333
1  Ima  test_bacc           2      0    37  0.500000  0.500000  0.500000  0.500000  0.500000  0.500000
2  Ima   test_acc           2      0    37  0.222222  0.142857  0.142857  0.142857  0.142857  0.158730
3  Ima   test_auc           2      1    87  0.500000  0.300000  0.600000  0.350000  0.916667  0.533333
4  Ima  test_bacc           2      1    87  0.500000  0.500000  0.500000  0.500000  0.500000  0.500000
5  Ima   test_acc           2      1    87  0.315789  0.294118  0.294118  0.294118  0.294118  0.298452
"""


###############################################################################
# Clustering Clin classifiy Im

clustering = pd.read_csv(os.path.join(WD, "Xclin-clust.csv"))
#pop = pd.read_csv(os.path.join(WD,"population.csv"))
assert np.all(pop.participant_id == clustering.participant_id)

metrics.confusion_matrix(yorig, clustering["nclust=2"])

"""
array([[16, 16],
       [32, 60]])
"""
metrics.confusion_matrix(yorig, clustering["nclust=3"])[:2, :]
"""
array([[ 9, 13, 10],
       [11, 55, 26]])
"""

range_n_clusters = [2, 3]
res = list()
for n_clusters in range_n_clusters:
    print("######################################")
    print("# nclust", n_clusters)
    cluster_labels = clustering["nclust=%i" % n_clusters]
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
print(res)

"""
XTreatTivSite
   data      score  n_clusters  clust  size     fold0     fold1     fold2     fold3     fold4       avg
0   Ima   test_auc           2      0    42  0.777778  0.666667  0.222222  0.733333  0.000000  0.480000
1   Ima  test_bacc           2      0    42  0.500000  0.500000  0.500000  0.500000  0.500000  0.500000
2   Ima   test_acc           2      0    42  0.333333  0.333333  0.333333  0.375000  0.285714  0.332143
3   Ima   test_auc           2      1    82  0.576923  0.442308  0.730769  0.487179  0.555556  0.558547
4   Ima  test_bacc           2      1    82  0.500000  0.500000  0.500000  0.500000  0.500000  0.500000
5   Ima   test_acc           2      1    82  0.235294  0.235294  0.235294  0.187500  0.200000  0.218676
6   Ima   test_auc           3      0    19  0.333333  0.000000  0.500000  0.000000  0.000000  0.166667
7   Ima  test_bacc           3      0    19  0.500000  0.500000  0.500000  0.500000  0.500000  0.500000
8   Ima   test_acc           3      0    19  0.400000  0.400000  0.333333  0.333333  0.333333  0.360000
9   Ima   test_auc           3      1    67  0.424242  0.333333  0.181818  0.233333  0.300000  0.294545
10  Ima  test_bacc           3      1    67  0.500000  0.333333  0.500000  0.500000  0.500000  0.466667
11  Ima   test_acc           3      1    67  0.214286  0.142857  0.214286  0.230769  0.166667  0.193773
12  Ima   test_auc           3      2    38  0.666667  0.416667  0.000000  0.900000  0.000000  0.396667
13  Ima  test_bacc           3      2    38  0.500000  0.500000  0.500000  0.500000  0.500000  0.500000
14  Ima   test_acc           3      2    38  0.333333  0.250000  0.285714  0.285714  0.285714  0.288095

NOTHING

XTreatTivSitePca
   data      score  n_clusters  clust  size     fold0     fold1     fold2     fold3     fold4       avg
0   Ima   test_auc           2      0    42  0.722222  0.555556  0.444444  0.733333  0.300000  0.551111
1   Ima  test_bacc           2      0    42  0.500000  0.500000  0.500000  0.500000  0.500000  0.500000
2   Ima   test_acc           2      0    42  0.333333  0.333333  0.333333  0.375000  0.285714  0.332143
3   Ima   test_auc           2      1    82  0.576923  0.346154  0.615385  0.461538  0.472222  0.494444
4   Ima  test_bacc           2      1    82  0.500000  0.500000  0.500000  0.500000  0.500000  0.500000
5   Ima   test_acc           2      1    82  0.235294  0.235294  0.235294  0.187500  0.200000  0.218676
6   Ima   test_auc           3      0    19  0.333333  0.000000  1.000000  0.500000  0.000000  0.366667
7   Ima  test_bacc           3      0    19  0.500000  0.500000  0.500000  0.500000  0.500000  0.500000
8   Ima   test_acc           3      0    19  0.400000  0.400000  0.333333  0.333333  0.333333  0.360000
9   Ima   test_auc           3      1    67  0.424242  0.181818  0.151515  0.266667  0.200000  0.244848
10  Ima  test_bacc           3      1    67  0.500000  0.333333  0.500000  0.500000  0.500000  0.466667
11  Ima   test_acc           3      1    67  0.214286  0.142857  0.214286  0.230769  0.166667  0.193773
12  Ima   test_auc           3      2    38  0.722222  0.500000  0.300000  0.800000  0.300000  0.524444
13  Ima  test_bacc           3      2    38  0.500000  0.500000  0.500000  0.500000  0.500000  0.500000
14  Ima   test_acc           3      2    38  0.333333  0.250000  0.285714  0.285714  0.285714  0.288095

"""



###############################################################################
# Clustering Im classifiy ClinIm

clustering = pd.read_csv(os.path.join(WD, IMADATASET+"-clust.csv"))
#pop = pd.read_csv(os.path.join(WD,"population.csv"))
assert np.all(pop.participant_id == clustering.participant_id)

metrics.confusion_matrix(yorig, clustering["nclust=2"])

"""
array([[17, 15],
       [45, 47]])
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
    cluster_labels = clustering["nclust=%i" % n_clusters]
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
print(res)

"""
      data      score  n_clusters  clust  size     fold0     fold1     fold2     fold3     fold4       avg
0  ClinIma   test_auc           2      0    62  0.458333  0.486111  0.555556  0.611111  0.166667  0.455556
1  ClinIma  test_bacc           2      0    62  0.500000  0.500000  0.500000  0.500000  0.500000  0.500000
2  ClinIma   test_acc           2      0    62  0.307692  0.307692  0.250000  0.250000  0.250000  0.273077
3  ClinIma   test_auc           2      1    62  0.533333  0.466667  0.833333  0.833333  0.555556  0.644444
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
Xim = np.load(os.path.join(WD, IMADATASET + ".npy"))
yorig = np.load(os.path.join(WD, "y.npy"))
mask = nibabel.load(os.path.join(WD, "mask.nii.gz"))
# Atv = nesterov_tv.linear_operator_from_mask(mask.get_data(), calc_lambda_max=True)
# Atv.save(os.path.join(WD, "Atv.npz"))
Atv = LinearOperatorNesterov(filename=os.path.join(WD, "Atv.npz"))
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


np.savez_compressed(os.path.join(WD, IMADATASET+"_enettv_0.1_0.1_0.8_5cv.npz"),
                    coefs_cv=coefs_cv, y_pred=y_test_pred, y_true=y,
                    proba_pred=y_test_prob_pred, beta=coefs_cv)

# Micro Avg
recall_test_microavg = metrics.recall_score(y, y_test_pred, average=None)
recall_train_microavg = metrics.recall_score(y, y_train_pred, average=None)
bacc_test_microavg = recall_test_microavg.mean()
auc_test_microavg = metrics.roc_auc_score(y, y_test_prob_pred)
acc_test_microavg = metrics.accuracy_score(y, y_test_pred)

print("#", IMADATASET, X.shape)
print("#", auc_test_microavg, bacc_test_microavg, acc_test_microavg)

#print(auc_test_microavg, bacc_test_microavg, acc_test_microavg)
"""
XTreatTivSite
# 0.438519021739 0.451766304348 0.443548387097

# XTreatTivSitePca (124, 397559)
# 0.461956521739 0.457201086957 0.451612903226

"""

###############################################################################
# Clustering Im classifiy ImEnettv

import nibabel
import parsimony.algorithms as algorithms
import parsimony.estimators as estimators
import parsimony.functions.nesterov.tv as nesterov_tv
from parsimony.utils.linalgs import LinearOperatorNesterov

# Data
Xim = np.load(os.path.join(WD, IMADATASET + ".npy"))
yorig = np.load(os.path.join(WD, "y.npy"))
mask = nibabel.load(os.path.join(WD, "mask.nii.gz"))
# Atv = nesterov_tv.linear_operator_from_mask(mask.get_data(), calc_lambda_max=True)
# Atv.save(os.path.join(WD, "Atv.npz"))
Atv = LinearOperatorNesterov(filename=os.path.join(WD, "Atv.npz"))
#assert Atv.get_singular_values(0) == Atv_.get_singular_values(0)
assert np.allclose(Atv.get_singular_values(0), 11.956104408414376)

# Cluster
clustering = pd.read_csv(os.path.join(WD, IMADATASET+"-clust.csv"))
#pop = pd.read_csv(os.path.join(WD,"population.csv"))
assert np.all(pop.participant_id == clustering.participant_id)
metrics.confusion_matrix(yorig, clustering["nclust=2"])
"""
array([[ 6, 26],
       [31, 61]])
"""
CLUST = 1
subset = clustering["nclust=2"] == CLUST

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

np.savez_compressed(os.path.join(WD,  IMADATASET+"-clust%i"%CLUST +"_enettv_0.1_0.1_0.8_5cv.npz"),
                    coefs_cv=coefs_cv, y_pred=y_test_pred, y_true=y,
                    proba_pred=y_test_prob_pred, beta=coefs_cv)

# Micro Avg
recall_test_microavg = metrics.recall_score(y, y_test_pred, average=None)
recall_train_microavg = metrics.recall_score(y, y_train_pred, average=None)
bacc_test_microavg = recall_test_microavg.mean()
auc_test_microavg = metrics.roc_auc_score(y, y_test_prob_pred)
acc_test_microavg = metrics.accuracy_score(y, y_test_pred)

print("#", IMADATASET+"-clust%i"%CLUST, X.shape)
print("#", auc_test_microavg, bacc_test_microavg, acc_test_microavg)

#
# YEAH !!

# XTreatTivSite-clust1
# 0.697872340426 0.678014184397 0.58064516129

# XTreatTivSitePca-clust1 (87, 397559)
# 0.465321563682 0.474148802018 0.448275862069

###############################################################################
# Clustering Im classifiy ClinImEnettv

import nibabel
import parsimony.algorithms as algorithms
import parsimony.estimators as estimators
import parsimony.functions.nesterov.tv as nesterov_tv
from parsimony.utils.linalgs import LinearOperatorNesterov

# Cluster
clustering = pd.read_csv(os.path.join(WD, IMADATASET+"-clust.csv"))
#pop = pd.read_csv(os.path.join(WD,"population.csv"))
assert np.all(pop.participant_id == clustering.participant_id)
assert np.all(metrics.confusion_matrix(yorig, clustering["nclust=2"]) == \
    np.array([[17, 15],
              [45, 47]]))

CLUST = 1
subset = clustering["nclust=2"] == CLUST

# Data
Xim = np.load(os.path.join(WD, IMADATASET + ".npy"))
yorig = np.load(os.path.join(WD, "y.npy"))
mask = nibabel.load(os.path.join(WD, "mask.nii.gz"))
# Atv = nesterov_tv.linear_operator_from_mask(mask.get_data(), calc_lambda_max=True)
# Atv.save(os.path.join(WD, "Atv.npz"))
Atv = LinearOperatorNesterov(filename=os.path.join(WD, "Atv.npz"))
#assert Atv.get_singular_values(0) == Atv_.get_singular_values(0)
assert np.allclose(Atv.get_singular_values(0), 11.956104408414376)

scaler = preprocessing.StandardScaler()
Ximg = Xim[subset, :]
Xcling = Xclin[subset, :]
y = yorig[subset]
Ximg = scaler.fit(Ximg).transform(Ximg)
Xcling = scaler.fit(Xcling).transform(Xcling)

# Load models Coeficients
modelscv = np.load(os.path.join(WD,  IMADATASET+"-clust%i"%CLUST +"_enettv_0.1_0.1_0.8_5cv.npz"))

# Parameters
key = 'enettv_0.1_0.1_0.8'.split("_")
algo, alpha, l1l2ratio, tvratio = key[0], float(key[1]), float(key[2]), float(key[3])
tv = alpha * tvratio
l1 = alpha * float(1 - tv) * l1l2ratio
l2 = alpha * float(1 - tv) * (1- l1l2ratio)
# print(key, algo, alpha, l1, l2, tv)

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
    estimator_stck = lm.LogisticRegression(class_weight='balanced', fit_intercept=False, C=100)
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


print("#", IMADATASET+"-clust%i"%CLUST, Ximg.shape)

# Micro Avg Img
recall_test_img_microavg = metrics.recall_score(y, y_test_pred_img, average=None)
recall_train_img_microavg = metrics.recall_score(y, y_train_pred_img, average=None)
bacc_test_img_microavg = recall_test_img_microavg.mean()
auc_test_img_microavg = metrics.roc_auc_score(y, y_test_prob_pred_img)
acc_test_img_microavg = metrics.accuracy_score(y, y_test_pred_img)

print("#", auc_test_img_microavg, bacc_test_img_microavg, acc_test_img_microavg)
# 0.697872340426 0.678014184397 0.58064516129

# Micro Avg Clin
recall_test_clin_microavg = metrics.recall_score(y, y_test_pred_clin, average=None)
recall_train_clin_microavg = metrics.recall_score(y, y_train_pred_clin, average=None)
bacc_test_clin_microavg = recall_test_clin_microavg.mean()
auc_test_clin_microavg = metrics.roc_auc_score(y, y_test_prob_pred_clin)
acc_test_clin_microavg = metrics.accuracy_score(y, y_test_pred_clin)

print("#", auc_test_clin_microavg, bacc_test_clin_microavg, acc_test_clin_microavg)
# 0.723404255319 0.641843971631 0.629032258065

# Micro Avg Stacking
recall_test_stck_microavg = metrics.recall_score(y, y_test_pred_stck, average=None)
recall_train_stck_microavg = metrics.recall_score(y, y_train_pred_stck, average=None)
bacc_test_stck_microavg = recall_test_stck_microavg.mean()
auc_test_stck_microavg = metrics.roc_auc_score(y, y_test_prob_pred_stck)
acc_test_stck_microavg = metrics.accuracy_score(y, y_test_pred_stck)

print("#", auc_test_stck_microavg, bacc_test_stck_microavg, acc_test_stck_microavg)


# Save
df = pop[["participant_id", "respond_wk16", 'GMvol_l', 'WMvol_l', 'CSFvol_l', 'TIV_l', 'GMratio', 'WMratio', 'CSFratio']]

# Cluster
clustering = pd.read_csv(os.path.join(WD, IMADATASET+"-clust.csv"))
assert(np.all(clustering.participant_id == df["participant_id"]))
df["cluster"] = clustering["nclust=2"]

df.loc[df["cluster"] == 1, "y_test_pred_img"] = y_test_pred_img
df.loc[df["cluster"] == 1, "y_test_prob_pred_img"] = y_test_prob_pred_img
df.loc[df["cluster"] == 1, "y_test_decfunc_pred_img"] = y_test_decfunc_pred_img

df.loc[df["cluster"] == 1, "y_test_pred_clin"] = y_test_pred_clin
df.loc[df["cluster"] == 1, "y_test_prob_pred_clin"] = y_test_prob_pred_clin
df.loc[df["cluster"] == 1, "y_test_decfunc_pred_clin"] = y_test_decfunc_pred_clin

df.loc[df["cluster"] == 1, "y_test_pred_stck"] = y_test_pred_stck
df.loc[df["cluster"] == 1, "y_test_prob_pred_stck"] = y_test_prob_pred_stck
df.loc[df["cluster"] == 1, "y_test_decfunc_pred_stck"] = y_test_decfunc_pred_stck

df.to_csv(os.path.join(WD,  IMADATASET+"-clust%i"%CLUST +"_img-scores.csv"), index=False)

# 0.739007092199 0.68865248227 0.596774193548

# YEAH !!!
# XTreatTivSite-clust1 (62, 397559)
# 0.697872340426 0.678014184397 0.58064516129
# 0.723404255319 0.641843971631 0.629032258065
# 0.739007092199 0.68865248227 0.596774193548

# Some test on C
# C=0.1
# 0.748936170213 0.656737588652 0.548387096774

# C=1
# 0.73475177305 0.656737588652 0.548387096774

# C=10
# 0.736170212766 0.678014184397 0.58064516129

# C=100
# 0.739007092199 0.68865248227 0.596774193548

# C=1000
# 0.737588652482 0.68865248227 0.596774193548

###############################################################################
# Computes usefull scores

###############################################################################
# Caracterize Cluster 1/2: scatterplot Clinic vs image
# Run first Clustering Im classifiy ClinImEnettv
CLUST=1
# df = pd.read_csv(os.path.join(WD,  IMADATASET+"-clust%i"%CLUST +"_demo-clin-imputed_img-scores.csv"))
df = pd.read_csv(os.path.join(WD,  IMADATASET+"-clust%i"%CLUST +"_img-scores.csv"))

"""
sns.lmplot(x="y_test_prob_pred_clin", y="y_test_prob_pred_img", hue="respond_wk16" , data=df, fit_reg=False)
sns.lmplot(x="y_test_decfunc_pred_clin", y="y_test_decfunc_pred_img", hue="respond_wk16" , data=df, fit_reg=False)
#sns.jointplot(x=df["decfunc_pred_clin_clust1"], y=df["decfunc_pred_img_clust1"], color="respond_wk16", kind='scatter')

sns.distplot(df["y_test_prob_pred_clin"][(df.cluster == 1) & (df["respond_wk16"] == "Responder")], rug=True, color="red")
sns.distplot(df["y_test_prob_pred_clin"][(df.cluster == 1) & (df["respond_wk16"] == "NonResponder")], rug=True, color="blue")
# Or
sns.kdeplot(df["y_test_prob_pred_clin"][(df.cluster == 1) & (df["respond_wk16"] == "Responder")],  color="red")
sns.kdeplot(df["y_test_prob_pred_clin"][(df.cluster == 1) & (df["respond_wk16"] == "NonResponder")],  color="blue")

sns.distplot(df["y_test_prob_pred_img"][(df.cluster == 1) & (df["respond_wk16"] == "Responder")], rug=True, color="red")
sns.distplot(df["y_test_prob_pred_img"][(df.cluster == 1) & (df["respond_wk16"] == "NonResponder")], rug=True, color="blue")
# Or
sns.kdeplot(df["y_test_prob_pred_img"][(df.cluster == 1) & (df["respond_wk16"] == "Responder")], color="red")
sns.kdeplot(df["y_test_prob_pred_img"][(df.cluster == 1) & (df["respond_wk16"] == "NonResponder")], color="blue")

sns.distplot(df["y_test_decfunc_pred_stck"][(df.cluster == 1) & (df["respond_wk16"] == "Responder")], rug=True, color="red")
sns.distplot(df["y_test_decfunc_pred_stck"][(df.cluster == 1) & (df["respond_wk16"] == "NonResponder")], rug=True, color="blue")
# or
sns.kdeplot(df["y_test_decfunc_pred_stck"][(df.cluster == 1) & (df["respond_wk16"] == "Responder")], color="red")
sns.kdeplot(df["y_test_decfunc_pred_stck"][(df.cluster == 1) & (df["respond_wk16"] == "NonResponder")], color="blue")
"""

assert np.all(yorig == df.respond_wk16.map({'NonResponder':0, 'Responder':1}))

dfclust1 = df.loc[(df.cluster == 1), ['y_test_prob_pred_clin', 'y_test_prob_pred_img', "respond_wk16"]]
X = np.array(dfclust1[['y_test_prob_pred_clin', 'y_test_prob_pred_img']])
y = np.array(dfclust1.respond_wk16.map({'NonResponder':0, 'Responder':1}))
#scaler = preprocessing.StandardScaler()
#X = scaler.fit(X).transform(X)
estimator_stck = lm.LogisticRegression(class_weight='balanced', fit_intercept=True, C=100)
estimator_stck.fit(X, y)

recall_post_stck_microavg = metrics.recall_score(y, estimator_stck.predict(X), average=None)
bacc_post_stck_microavg = recall_post_stck_microavg.mean()
auc_post_stck_microavg = metrics.roc_auc_score(y, estimator_stck.predict_proba(X)[:, 1])
acc_post_stck_microavg = metrics.accuracy_score(y,  estimator_stck.predict(X))

print("#", auc_post_stck_microavg, bacc_post_stck_microavg, acc_post_stck_microavg, recall_post_stck_microavg)
# 0.782978723404 0.717730496454 0.709677419355 [ 0.73333333  0.70212766]

estimator_stck.coef_
# array([[ 5.5799257 ,  2.60588734]])
estimator_stck.intercept_
# Out[227]: array([-3.70611507])

df.loc[df.cluster == 1, "y_post_decfunc_pred_stck"] = estimator_stck.decision_function(X)
df.loc[df.cluster == 1, "y_post_prob_pred_stck"] = estimator_stck.predict_proba(X)[:, 1]

from matplotlib.backends.backend_pdf import PdfPages

# contour
# https://matplotlib.org/1.3.0/examples/pylab_examples/contour_demo.html

nx = ny = 100
x = np.linspace(0.1, 0.7, num=nx)
y = np.linspace(0.0, 0.9, num=ny)
xx, yy = np.meshgrid(x, y)
z_proba = estimator_stck.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
z_proba = z_proba.reshape(xx.shape)

palette = {"NonResponder":sns.xkcd_rgb["denim blue"], "Responder":sns.xkcd_rgb["pale red"]}
palette = {"NonResponder":"blue", "Responder":"red"}
sns.color_palette()[0]

sns.set(style="whitegrid")
palette = {"NonResponder":sns.color_palette()[0],
           "Responder":sns.color_palette()[2]}

"""
g = sns.JointGrid(x="y_test_prob_pred_clin", y="y_test_prob_pred_img", data=df)
g.ax_joint.scatter(df["y_test_prob_pred_clin"], df["y_test_prob_pred_img"], c=[palette[res] for res in df.respond_wk16])
#sns.lmplot(x="y_test_prob_pred_clin", y="y_test_prob_pred_img", hue="respond_wk16" , data=df, fit_reg=False, palette=palette, axis=g.ax_joint)
CS = g.ax_joint.contour(xx, yy, z_proba, 6, levels=[0.5], colors='k', axis=g.ax_joint)
plt.clabel(CS, fontsize=9, inline=1)
"""


pdf = PdfPages(os.path.join(WD, IMADATASET+'-clust_img-clin_scatter_density.pdf'))

fig = plt.figure()

plt.scatter(df["y_test_prob_pred_clin"], df["y_test_prob_pred_img"], c=[palette[res] for res in df.respond_wk16])
#sns.lmplot(x="y_test_prob_pred_clin", y="y_test_prob_pred_img", hue="respond_wk16" , data=df, fit_reg=False, palette=palette, axis=g.ax_joint)
CS1 = plt.contour(xx, yy, z_proba, 6, levels=[0.5], colors='k')#, axis=g.ax_joint)
CS2 = plt.contour(xx, yy, z_proba, 6, levels=[0.25, 0.75], linestyles="dashed", colors='grey')#, axis=g.ax_joint)
plt.clabel(CS1, CS1.levels, fontsize=9, inline=1)
plt.clabel(CS2, CS2.levels, fontsize=9, inline=1)
plt.xlabel("Clinic proba.")
plt.ylabel("Imaging proba.")
pdf.savefig(); plt.close()

"""
#sns.distplot(df.loc["y_test_prob_pred_clin"], kde=True, hist=False, color="r", ax=g.ax_marg_x)
sns.kdeplot(df["y_test_prob_pred_clin"][(df.cluster == 1) & (df["respond_wk16"] == "NonResponder")], bw=.05,
             color=palette["NonResponder"], ax=g.ax_marg_x, label="NonResponder")
sns.kdeplot(df["y_test_prob_pred_clin"][(df.cluster == 1) & (df["respond_wk16"] == "Responder")], bw=.05,
             color=palette["Responder"], ax=g.ax_marg_x, label="Responder")
sns.kdeplot(df["y_test_prob_pred_img"][(df.cluster == 1) & (df["respond_wk16"] == "NonResponder")],  bw=.1,
             color=palette["NonResponder"], ax=g.ax_marg_y, vertical=True, label="NonResponder")
sns.kdeplot(df["y_test_prob_pred_img"][(df.cluster == 1) & (df["respond_wk16"] == "Responder")],  bw=.05,
             color=palette["Responder"], ax=g.ax_marg_y, vertical=True, label="Responder")
g.ax_joint.set_xlim(0.0, .8)
g.ax_joint.set_ylim(0.0, 1)
"""

# rotate figure
#from matplotlib.transforms import Affine2D
#import mpl_toolkits.axisartist.floating_axes as floating_axes

fig = plt.figure()
sns.kdeplot(df["y_post_prob_pred_stck"][(df.cluster == 1) & (df["respond_wk16"] == "Responder")],
            color=palette["Responder"], shade=True, label="Responder")
sns.kdeplot(df["y_post_prob_pred_stck"][(df.cluster == 1) & (df["respond_wk16"] == "NonResponder")],
            color=palette["NonResponder"], shade=True, label="NonResponder")
pdf.savefig(); plt.close()
pdf.close()

"""
plot_extents = 0, 10, 0, 10
transform = Affine2D().rotate_deg(45)
helper = floating_axes.GridHelperCurveLinear(transform, plot_extents)
ax = floating_axes.FloatingSubplot(fig, 111, grid_helper=helper)

sns.kdeplot(df["y_post_prob_pred_stck"][(df.cluster == 1) & (df["respond_wk16"] == "Responder")],
            color=palette["Responder"], shade=True, label="Responder", ax=ax)
sns.kdeplot(df["y_post_prob_pred_stck"][(df.cluster == 1) & (df["respond_wk16"] == "NonResponder")],
            color=palette["NonResponder"], shade=True, label="NonResponder", ax=ax)

fig.add_subplot(ax)
plt.show()
"""



"""
sns.distplot(df["y_test_prob_pred_clin"][(df.cluster == 1) & (df["respond_wk16"] == "NonResponder")], kde=True, hist=False,
             color=palette["NonResponder"], ax=g.ax_marg_x)
sns.distplot(df["y_test_prob_pred_clin"][(df.cluster == 1) & (df["respond_wk16"] == "Responder")], kde=True, hist=False,
             color=palette["Responder"], ax=g.ax_marg_x)
sns.distplot(df["y_test_prob_pred_img"][(df.cluster == 1) & (df["respond_wk16"] == "NonResponder")], kde=True, hist=False,
             color=palette["NonResponder"], ax=g.ax_marg_y, vertical=True)
sns.distplot(df["y_test_prob_pred_img"][(df.cluster == 1) & (df["respond_wk16"] == "Responder")], kde=True, hist=False,
             color=palette["Responder"], ax=g.ax_marg_y, vertical=True)
"""

#plt.clabel(CS, fontsize=9, inline=1)

#plt.clabel(CS)
#plt.clabel(CS, fontsize=9, inline=1)
# plt.plot([0, 4], [1.5, 0], linewidth=2)

# density left an top
# https://stackoverflow.com/questions/49671053/seaborn-changing-line-styling-in-kdeplot



###############################################################################
# Caracterize Cluster Stats

# A) Cluster association with clinical variables
import os
import seaborn as sns
import pandas as pd
import scipy.stats as stats

CLUST=1

xls_filename = os.path.join(WD,
                     IMADATASET+"-clust%i"%CLUST +"_demo-clin-vs-cluster.xlsx")

# add cluster information
pop = pd.read_csv(os.path.join(WD, "population.csv"))
clust = pd.read_csv(os.path.join(WD,  IMADATASET+"-clust%i"%CLUST +"_img-scores.csv"))
pop = pd.merge(pop, clust[["participant_id", 'cluster']], on='participant_id')

pop["duration"] = pop['age'] - pop['age_onset']
# cluster effect on dem/clinique
"""
['age', 'sex_num', 'educ', 'age_onset']
['mde_num', 'madrs_Baseline', 'madrs_Screening']
'respond_wk16'
"""
cols_ = ['age', 'educ', 'age_onset', "duration", 'mde_num', 'madrs_Baseline',
         'GMratio', 'WMratio', 'TIV_l',
         "cluster"]

means = pop[cols_].groupby(by="cluster").mean().T.reset_index()
means.columns = ['var', 'mean_0', 'mean_1']

stds = pop[cols_].groupby(by="cluster").std().T.reset_index()
stds.columns = ['var', 'std_0', 'std_1']
desc = pd.merge(means, stds)

stat = pd.DataFrame(
[[col] + list(stats.ttest_ind(
        pop.loc[pop.cluster==1, col],
        pop.loc[pop.cluster==0, col],
        equal_var=False, nan_policy='omit'))
    for col in cols_], columns=['var', 'stat', 'pval'])

stat_clust_vs_var = pd.merge(desc, stat)

print(stat_clust_vs_var)
"""
              var     mean_0     mean_1      std_0     std_1       stat  \
0             age  41.580645  29.806452  12.277593  9.844427  -5.891231
1            educ  17.000000  16.622951   2.522034  1.950760  -0.928299
2       age_onset  23.779661  18.186441  11.365507  7.431236  -3.163806
3        duration  17.762712  11.932203  14.392693  9.428249  -2.602892
4         mde_num   4.047619   3.652174   2.251854  2.709796  -0.746835
5  madrs_Baseline  30.083333  29.866667   5.790968  5.512595  -0.209911
6         GMratio   0.459701   0.519362   0.031557  0.032438  10.380524
7         WMratio   0.299968   0.296910   0.023441  0.022140  -0.746770
8           TIV_l   1.503500   1.392831   0.155763  0.134993  -4.227700

           pval
0  3.796567e-08
1  3.552030e-01
2  2.063409e-03
3  1.064894e-02
4  4.572176e-01
5  8.340998e-01
6  1.788653e-18
7  4.566439e-01
8  4.642436e-05

Grp 1 are younger (6 years), earlier onset (3 years) and shorter duration
"""


# B) Effect of stratification to disantangle Resp/NoResp (ROC)
df = pop.copy()
df.cluster = "All"
df = pop.copy().append(df)


variables = ['age', 'educ'] +\
    ['age_onset', 'mde_num', 'madrs_Baseline', 'madrs_Screening', "duration"]

res = list()
for var in variables:
    for lab in df.cluster.unique():
        resp = df.loc[df.cluster == lab, "respond_wk16_num"]
        val = df.loc[df.cluster == lab, var]
        mask = val.notnull()
        resp, val = resp[mask], val[mask]
        auc = metrics.roc_auc_score(resp, val)
        auc = max(auc, 1 - auc)
        wilcox = stats.mannwhitneyu(*[val[resp == r] for r in np.unique(resp)])
        res.append([var, lab, auc, wilcox.statistic, wilcox.pvalue])

roc_clust_on_var = pd.DataFrame(res, columns=['var', 'clust', 'auc', 'Mann–Whitney-U', 'pval'])

with pd.ExcelWriter(xls_filename) as writer:
    stat_clust_vs_var.to_excel(writer, sheet_name='stat_clust_vs_var', index=False)
    roc_clust_on_var.to_excel(writer, sheet_name='roc_clust_on_var', index=False)



###############################################################################
# Caracterize Cluster Plot
palette_resp = {"NonResponder":sns.color_palette()[0],
           "Responder":sns.color_palette()[2]}
alphas_clust = {1:1, 0:.5}

pdf = PdfPages(os.path.join(WD,
                     IMADATASET+"-clust%i"%CLUST +"_demo-clin-vs-cluster.pdf"))

sns.set(style="whitegrid")

# A) Cluster association with clinical variables
df = pop.copy()

xy_cols = [
        ["age_onset", "GMratio"],
        ["duration", "GMratio"],
        ["age", "GMratio"]]

for x_col, y_col in xy_cols:
    print(x_col, y_col)
    fig = plt.figure()
    fig.suptitle('%s modulated by %s' % (x_col, y_col))
    for lab in df.cluster.unique():
        resp = df.loc[df.cluster == lab, "respond_wk16"]
        x = df.loc[df.cluster == lab, x_col]
        y = df.loc[df.cluster == lab, y_col]
        for r in resp.unique():
            plt.plot(x[resp == r], y[resp == r], "o", color=palette_resp[r],
                     alpha=alphas_clust[lab], label="grp %i / %s" % (lab, r))
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.legend()
    pdf.savefig(); plt.close()

# B) Effect of stratification to disantangle Resp/NoResp
df = pop.copy()
df.cluster = "All"
df = pop.copy().append(df)

fig = plt.figure()
fig.suptitle('Duration, cluster and response ')
ax = sns.violinplot(x="cluster", y="duration", hue="respond_wk16", data=df,
               split=True, label="", legend_out = True, palette=palette_resp)
handles, labels = ax.get_legend_handles_labels()
ax = sns.swarmplot(x="cluster", y="duration", hue="respond_wk16", data=df,
              dodge=True, linewidth=1, edgecolor='black', palette=palette_resp)
ax.legend(handles, labels)
pdf.savefig(); plt.close()


fig = plt.figure()
fig.suptitle('Age, cluster and response ')
ax = sns.violinplot(x="cluster", y="age", hue="respond_wk16", data=df,
               split=True, palette=palette_resp)
handles, labels = ax.get_legend_handles_labels()
ax = sns.swarmplot(x="cluster", y="age", hue="respond_wk16", data=df,
              dodge=True, linewidth=1, edgecolor='black', palette=palette_resp)
ax.legend(handles, labels)
pdf.savefig(); plt.close()

fig = plt.figure()
fig.suptitle('Age onset, cluster and response ')
ax = sns.violinplot(x="cluster", y="age_onset", hue="respond_wk16", data=df,
               split=True, palette=palette_resp)
handles, labels = ax.get_legend_handles_labels()
ax = sns.swarmplot(x="cluster", y="age_onset", hue="respond_wk16", data=df,
              dodge=True, linewidth=1, edgecolor='black', palette=palette_resp)
ax.legend(handles, labels)
pdf.savefig(); plt.close()

fig = plt.figure()
fig.suptitle('Grey matter, cluster and response ')
ax = sns.violinplot(x="cluster", y="GMratio", hue="respond_wk16", data=df,
               split=True, label=None, legend_out = True, palette=palette_resp)
handles, labels = ax.get_legend_handles_labels()
ax = sns.swarmplot(x="cluster", y="GMratio", hue="respond_wk16", data=df,
              dodge=True, linewidth=1, edgecolor='black', palette=palette_resp)
ax.legend(handles, labels)
pdf.savefig(); plt.close()

pdf.close()


##############################################################################
# Intra-group heterogeneity Distances

# Caracterize cluster
clusters = np.load(os.path.join(WD, IMADATASET+"-clust_centers.npz"))
clusters["cluster_labels"]
cluster_centers = clusters["cluster_centers"]

scaler = preprocessing.StandardScaler()

#
X = scaler.fit(Xim).transform(Xim)

# Clinic Imaging
import scipy
X = scaler.fit(Xclin).transform(Xclin)
# X = scaler.fit(Xim).transform(Xim)

# Average pairwise Euclidian distance
pairwise_dist = scipy.spatial.distance.cdist(X, X)
res = list()
cluster_labels_ = clusters["cluster_labels"]

for lab in np.unique(cluster_labels_):
    subset = cluster_labels_ == lab
    dist_ = pairwise_dist[subset][:, subset]
    res.append([lab, subset.sum(), np.mean(dist_[np.triu_indices(dist_.shape[0], k=1)])])

res = pd.DataFrame(res, columns=["clust", "size", "pairwise_dist_avg"])

# assess diff by perms
nperm = 1000
dist_avg = np.zeros((nperm, 2))
cluster_labels_ = clusters["cluster_labels"]
for i in range(nperm):
    if i != 0:
        cluster_labels_ = np.random.permutation(cluster_labels_)
    for lab in np.unique(cluster_labels_):
        subset = cluster_labels_ == lab
        dist_ = pairwise_dist[subset][:, subset]
        dist_avg[i, lab] = np.mean(dist_[np.triu_indices(dist_.shape[0], k=1)])

dist_avg
diff = dist_avg[:, 1] - dist_avg[:, 0]
pval = np.sum(diff <= diff[0]) / nperm

print(res)
print(pval)
"""
# Clinic
   clust  size  pairwise_dist_avg
0      0    62           3.794570
1      1    62           3.080776

pval <= 0.001

# Img
   clust  size  pairwise_dist_avg
0      0    62         861.994669
1      1    62         880.108586
0.964
"""
# spectral norm

res = list()
for lab in np.unique(clusters["cluster_labels"]):
    subset = clusters["cluster_labels"] == lab
    s = np.linalg.svd(X[subset], full_matrices=False, compute_uv=False)
    res.append([lab, subset.sum(), np.max(s) ** 2.0 / subset.sum(), np.sum(s ** 2) / subset.sum()])

res = pd.DataFrame(res, columns=["clust", "size", "spectral_norm", "singular_vals_sum"])

# assess diff by perms
nperm = 100
cluster_labels_ = clusters["cluster_labels"]
norms = np.zeros((nperm, 2))
for i in range(nperm):
    if i != 0:
        cluster_labels_ = np.random.permutation(cluster_labels_)
    for lab in np.unique(cluster_labels_):
        subset = cluster_labels_ == lab
        s = np.linalg.svd(X[subset], full_matrices=False, compute_uv=False)
        norms[i, lab] = np.max(s) ** 2.0
norms
diff = norms[:, 1] - norms[:, 0]
pval = np.sum(diff <= diff[0]) / nperm

print(res)
print(pval)

"""
# Clinic
   clust  size  spectral_norm  singular_vals_sum
0      0    62       2.532531           8.217296
1      1    62       1.756518           5.782704

pval <= 0.01

# Img
   clust  size  spectral_norm  singular_vals_sum
0      0    62   35736.756117      389704.767617
1      1    62   34454.160310      405413.232383
0.42
"""


###############################################################################
#Clustering Im-clust1 learn ClinImEnettv Im-clust1 predict Im-clust0

import nibabel
import parsimony.algorithms as algorithms
import parsimony.estimators as estimators
import parsimony.functions.nesterov.tv as nesterov_tv
from parsimony.utils.linalgs import LinearOperatorNesterov


# Cluster
clustering = pd.read_csv(os.path.join(WD, IMADATASET+"-clust.csv"))
#pop = pd.read_csv(os.path.join(WD,"population.csv"))
assert np.all(pop.participant_id == clustering.participant_id)
metrics.confusion_matrix(yorig, clustering["nclust=2"])
"""
array([[17, 15],
       [45, 47]])
"""
CLUST = 1
subset1 = clustering["nclust=2"] == CLUST
subset0 = clustering["nclust=2"] == 0

# Data
Xim = np.load(os.path.join(WD, IMADATASET + ".npy"))
yorig = np.load(os.path.join(WD, "y.npy"))
mask = nibabel.load(os.path.join(WD, "mask.nii.gz"))
# Atv = nesterov_tv.linear_operator_from_mask(mask.get_data(), calc_lambda_max=True)
# Atv.save(os.path.join(WD, "Atv.npz"))
Atv = LinearOperatorNesterov(filename=os.path.join(WD, "Atv.npz"))
#assert Atv.get_singular_values(0) == Atv_.get_singular_values(0)
assert np.allclose(Atv.get_singular_values(0), 11.956104408414376)

scaler_img = preprocessing.StandardScaler()
scaler_clin = preprocessing.StandardScaler()

# clust1
Ximg1 = Xim[subset1, :]
Xcling1 = Xclin[subset1, :]
y1 = yorig[subset1]
Ximg1 = scaler_img.fit(Ximg1).transform(Ximg1)
Xcling1 = scaler_clin.fit(Xcling1).transform(Xcling1)
[np.sum(y1 == lev) for lev in np.unique(y1)]

# clust0
Ximg0 = Xim[subset0, :]
Xcling0 = Xclin[subset0, :]
y0 = yorig[subset0]
Ximg0 = scaler_img.fit(Ximg0).transform(Ximg0)
Xcling0 = scaler_clin.fit(Xcling0).transform(Xcling0)
[np.sum(y0 == lev) for lev in np.unique(y0)]

# Load models
modelscv = np.load(os.path.join(WD,  IMADATASET+"-clust%i"%CLUST +"_enettv_0.1_0.1_0.8_5cv.npz"))

# parameters
key = 'enettv_0.1_0.1_0.8'.split("_")
algo, alpha, l1l2ratio, tvratio = key[0], float(key[1]), float(key[2]), float(key[3])
tv = alpha * tvratio
l1 = alpha * float(1 - tv) * l1l2ratio
l2 = alpha * float(1 - tv) * (1- l1l2ratio)
print(key, algo, alpha, l1, l2, tv)


conesta = algorithms.proximal.CONESTA(max_iter=10000)
estimator_img = estimators.LogisticRegressionL1L2TV(l1, l2, tv, Atv, algorithm=conesta,
                                                class_weight="auto", penalty_start=0)
estimator_img.fit(Ximg0, y0)
coef_all_clust0 = estimator_img.beta

estimator_img.fit(Ximg1, y1)
coef_all_clust1 = estimator_img.beta

modelscv.keys()
['coefs_cv', 'y_pred', 'y_true', 'proba_pred', 'beta']

np.savez_compressed(os.path.join(WD,  IMADATASET+"-clust%i"%CLUST +"_enettv_0.1_0.1_0.8_5cv.npz"),
                    coefs_cv=modelscv["coefs_cv"], y_pred=modelscv["y_pred"], y_true=modelscv["y_true"],
                    proba_pred=modelscv["proba_pred"],
                    coef_refitall_clust0=coef_all_clust0.ravel(), coef_refitall_clust1=coef_all_clust1.ravel())

"""
modelscv = np.load(os.path.join(WD,  IMADATASET+"-clust%i"%CLUST +"_enettv_0.1_0.1_0.8_5cv.npz"))
modelscv_["coefs_cv"] == modelscv["coefs_cv"]
modelscv[
#estimator_img.beta = modelscv["coefs_cv"].mean(axis=0)[:, None]
modelscv.keys()
"""

# Store prediction for micro avg
y_clust0_pred_img = estimator_img.predict(Ximg0).ravel()
y_clust0_prob_pred_img = estimator_img.predict_probability(Ximg0).ravel()#[:, 1]
y_clust0_decfunc_pred_img = np.dot(Ximg0, estimator_img.beta).ravel()
y_clust1_pred_img = estimator_img.predict(Ximg1).ravel()

# Compute score for macro avg
print("#",
    metrics.roc_auc_score(y0, y_clust0_prob_pred_img),
    metrics.recall_score(y0, y_clust0_pred_img, average=None),
    metrics.accuracy_score(y0, y_clust0_pred_img))

# 0.481045751634 [ 0.52941176  0.55555556] 0.548387096774
# Boff


###############################################################################
# Caracterize Cluster centers
from nilearn import datasets, plotting, image
import  nibabel
from matplotlib.backends.backend_pdf import PdfPages
CLUST = 1

clusters = np.load(os.path.join(WD, IMADATASET+"-clust_centers.npz"))
clusters["cluster_labels"]
cluster_centers = clusters["cluster_centers"]

## WIP HERE

mask_img = nibabel.load(os.path.join(WD, "mask.nii.gz"))
coef_arr = np.zeros(mask_img.get_data().shape)

pd.Series(np.abs(cluster_centers[0, :])).describe()
"""
count    397559.000000
mean          0.207656
std           0.128221
min           0.000002
25%           0.104007
50%           0.195704
75%           0.297715
max           0.694904
"""

pd.Series(np.abs(cluster_centers[1, :])).describe()
"""
count    397559.000000
mean          0.207656
std           0.128221
min           0.000002
25%           0.104007
50%           0.195704
75%           0.297715
max           0.694904
"""
pd.Series(np.abs(cluster_centers[1, :]- cluster_centers[0, :])).describe()
"""
count    397559.000000
mean          0.415313
std           0.256443
min           0.000003
25%           0.208014
50%           0.391407
75%           0.595429
max           1.389808
"""

c0 = cluster_centers[0, :]
c1 = cluster_centers[1, :]

scaler = preprocessing.StandardScaler()
X = scaler.fit(Xim).transform(Xim)
mean = scaler.mean_

proj_c1c0  = np.dot(X, c1 - c0)
imgscores = pd.read_csv(os.path.join(WD,  IMADATASET+"-clust%i"%CLUST +"_img-scores.csv"))
imgscores["proj_c1c0"] = proj_c1c0
np.all(imgscores.participant_id == pop.participant_id)

imgscores.to_csv(os.path.join(WD,  IMADATASET+"-clust%i"%CLUST +"_img-scores.csv"), index=False)

"""
df = imgscores.copy()
df["respond_wk16"] = pop["respond_wk16"]
df["sex"] = pop["sex"]
df["age"] = pop["age"]

plt.plot(imgscores.GMratio, imgscores.proj_c1c0, 'o')

ICI
sns.lmplot(x="GMratio", y="proj_c1c0", hue="respond_wk16" , data=df, fit_reg=False)
sns.lmplot(x="GMratio", y="proj_c1c0", hue="cluster" , data=df, fit_reg=False)
sns.lmplot(x="GMratio", y="proj_c1c0", hue="age" , data=df, fit_reg=False)
sns.lmplot(x="GMratio", y="age", hue="cluster" , data=df, fit_reg=False)
sns.lmplot(x="age", y="GMratio", hue="cluster" , data=df, fit_reg=False)
sns.lmplot(x="age", y="proj_c1c0", hue="cluster" , data=df, fit_reg=False)
"""

n0 = np.sum(clusters["cluster_labels"] == 0)
n1 = np.sum(clusters["cluster_labels"] == 1)

X0 = X[clusters["cluster_labels"] == 0, ]
X1 = X[clusters["cluster_labels"] == 1, ]
X0c = X0 - c0
X1c = X1 - c1

s = np.sqrt((np.sum(X0c ** 2, axis=0) * (n0 - 1) + np.sum(X1c ** 2, axis=0) * (n1 - 1)) / (n0 + n1 -2))

tmap = (c1 - c0) / (s * np.sqrt(1 / n1 + 1 / n0))
zmap = (c1 - c0) / s

figure_filename = os.path.join(WD, IMADATASET+"-clust_centers.pdf")
pdf = PdfPages(figure_filename)

fig = plt.figure()
coef_arr[mask_img.get_data() != 0] = c1 + mean
coef_img = nibabel.Nifti1Image(coef_arr, affine=mask_img.affine)
plotting.plot_stat_map(coef_img, display_mode='z', cut_coords=7,
                       title='Group 1 center (centered and scaled)', colorbar=True)
pdf.savefig(); plt.close()

fig = plt.figure()
coef_arr[mask_img.get_data() != 0] = c0
coef_img = nibabel.Nifti1Image(coef_arr, affine=mask_img.affine)
plotting.plot_stat_map(coef_img, display_mode='z', cut_coords=7,
                       title='Group 0 center (centered and scaled)', colorbar=True)
pdf.savefig(); plt.close()


c0_ = scaler.inverse_transform(c0)
coef_arr[mask_img.get_data() != 0] = c0_ - c0_.min()
coef_img = nibabel.Nifti1Image(coef_arr, affine=mask_img.affine)
plotting.plot_anat(coef_img, display_mode='ortho', cut_coords=(5, -13, 1), black_bg=True,  draw_cross=False,
                       title='Group 0 center')
pdf.savefig(); plt.close()

c1_ = scaler.inverse_transform(c1)
coef_arr[mask_img.get_data() != 0] = c1_ - c1_.min()
coef_img = nibabel.Nifti1Image(coef_arr, affine=mask_img.affine)
plotting.plot_anat(coef_img, display_mode='ortho', cut_coords=(5, -13, 1), black_bg=True, draw_cross=False,
                       title='Group 1 center')
pdf.savefig(); plt.close()


fig = plt.figure()
coef_arr[mask_img.get_data() != 0] = c1 - c0
coef_img = nibabel.Nifti1Image(coef_arr, affine=mask_img.affine)
plotting.plot_stat_map(coef_img, display_mode='z', cut_coords=7,
                       title='Difference of the centers: center 1 - center 2', colorbar=True)
pdf.savefig(); plt.close()

fig = plt.figure()
coef_arr[mask_img.get_data() != 0] = zmap
coef_img = nibabel.Nifti1Image(coef_arr, affine=mask_img.affine)
plotting.plot_stat_map(coef_img, display_mode='ortho', draw_cross=False, cut_coords=(5, -13, 1),
                       title='Z map of the difference', colorbar=True)
pdf.savefig(); plt.close()


fig = plt.figure()
coef_arr[mask_img.get_data() != 0] = zmap
coef_img = nibabel.Nifti1Image(coef_arr, affine=mask_img.affine)
plotting.plot_stat_map(coef_img, display_mode='z', cut_coords=7,
                       title='Z map of the difference', colorbar=True)
pdf.savefig(); plt.close()


fig = plt.figure()
coef_arr[mask_img.get_data() != 0] = zmap
coef_img = nibabel.Nifti1Image(coef_arr, affine=mask_img.affine)
plotting.plot_stat_map(coef_img, display_mode='y', cut_coords=7,
                       title='Z map of the difference', colorbar=True)
pdf.savefig(); plt.close()


fig = plt.figure()
coef_arr[mask_img.get_data() != 0] = tmap
coef_img = nibabel.Nifti1Image(coef_arr, affine=mask_img.affine)
plotting.plot_stat_map(coef_img, display_mode='z', cut_coords=7,
                       title='T map of the difference', colorbar=True)
pdf.savefig(); plt.close()

pdf.close()

"""
cd /home/edouard/data/psy/canbind/models/clustering_v02
convert XTreatTivSite-clust_centers.pdf toto.png

convert XTreatTivSite-clust_centers.pdf[2] images/group0_center.png
convert XTreatTivSite-clust_centers.pdf[3] images/group1_center.png
convert XTreatTivSite-clust_centers.pdf[5] images/zmap-diff_centers.png
convert XTreatTivSite-clust1_enettv_0.1_0.1_0.8_5.pdf[0] images/signature_glassview.png

"""


###############################################################################
# Bootstrap Signature

import nibabel
import parsimony.algorithms as algorithms
import parsimony.estimators as estimators
import parsimony.functions.nesterov.tv as nesterov_tv
from parsimony.utils.linalgs import LinearOperatorNesterov

# Cluster
clustering = pd.read_csv(os.path.join(WD, IMADATASET+"-clust.csv"))
#pop = pd.read_csv(os.path.join(WD,"population.csv"))
assert np.all(pop.participant_id == clustering.participant_id)
assert np.all(metrics.confusion_matrix(yorig, clustering["nclust=2"]) == \
    np.array([[17, 15],
              [45, 47]]))

CLUST = 1
subset = clustering["nclust=2"] == CLUST

# Data
Xim = np.load(os.path.join(WD, IMADATASET + ".npy"))
yorig = np.load(os.path.join(WD, "y.npy"))
mask = nibabel.load(os.path.join(WD, "mask.nii.gz"))
# Atv = nesterov_tv.linear_operator_from_mask(mask.get_data(), calc_lambda_max=True)
# Atv.save(os.path.join(WD, "Atv.npz"))
Atv = LinearOperatorNesterov(filename=os.path.join(WD, "Atv.npz"))
#assert Atv.get_singular_values(0) == Atv_.get_singular_values(0)
assert np.allclose(Atv.get_singular_values(0), 11.956104408414376)

scaler = preprocessing.StandardScaler()
Ximg = Xim[subset, :]
Xcling = Xclin[subset, :]
y = yorig[subset]
Ximg = scaler.fit(Ximg).transform(Ximg)
Xcling = scaler.fit(Xcling).transform(Xcling)

# Load models Coeficients
# modelscv = np.load(os.path.join(WD,  IMADATASET+"-clust%i"%CLUST +"_enettv_0.1_0.1_0.8_5cv.npz"))

# Parameters
keystr = 'enettv_0.1_0.1_0.8'
key = keystr.split("_")
algo, alpha, l1l2ratio, tvratio = key[0], float(key[1]), float(key[2]), float(key[3])
tv = alpha * tvratio
l1 = alpha * float(1 - tv) * l1l2ratio
l2 = alpha * float(1 - tv) * (1- l1l2ratio)
# print(key, algo, alpha, l1, l2, tv)

# BOOT loop
NBOOT = 100

coefs_boot_img = np.zeros((NBOOT, Ximg.shape[1]))
auc_test_img = list()
recalls_test_img = list()
acc_test_img = list()


coefs_boot_clin = np.zeros((NBOOT, Xcling.shape[1]))
auc_test_clin = list()
recalls_test_clin = list()
acc_test_clin = list()


coefs_boot_stck = np.zeros((NBOOT, 2))
auc_test_stck = list()
recalls_test_stck = list()
acc_test_stck = list()

# Stratified Bootstraping
idx_all = np.arange(y.shape[0])

for boot_i in range(NBOOT):
    print(boot_i)
    np.random.seed(seed=boot_i)
    # resample with replacement within groups
    train = np.concatenate([np.random.choice(idx_all[y == lab], size=np.sum(y == lab), replace=True) for lab in np.unique(y)])
    test = np.setdiff1d(idx_all, train, assume_unique=False)
    X_train_img, X_test_img, y_train, y_test = Ximg[train, :], Ximg[test, :], y[train], y[test]
    X_train_clin, X_test_clin = Xcling[train, :], Xcling[test, :]
    # Im
    conesta = algorithms.proximal.CONESTA(max_iter=10000)
    estimator_img = estimators.LogisticRegressionL1L2TV(l1, l2, tv, Atv, algorithm=conesta,
                                                    class_weight="auto", penalty_start=0)
    estimator_img.fit(X_train_img, y_train)

    # Predictions
    y_test_pred_img = estimator_img.predict(X_test_img).ravel()
    y_test_prob_pred_img = estimator_img.predict_probability(X_test_img).ravel()#[:, 1]
    y_test_decfunc_pred_img = np.dot(X_test_img, estimator_img.beta).ravel()

    coefs_boot_img[boot_i, :] = estimator_img.beta.ravel()
    np.savez_compressed(os.path.join(WD,  IMADATASET+"-clust%i"%CLUST + "_" + keystr + "_boot/cooef_boot-%.4i.npz" % boot_i),
                    y_pred=y_test_pred_img, y_test=y_test, seed=boot_i, train_idx=train, test_idx=test,
                    proba_pred=y_test_prob_pred_img, beta=estimator_img.beta.ravel())

    # Compute score for macro avg
    auc_test_img.append(metrics.roc_auc_score(y_test, y_test_prob_pred_img))
    recalls_test_img.append(metrics.recall_score(y_test, y_test_pred_img, average=None))
    acc_test_img.append(metrics.accuracy_score(y_test, y_test_pred_img))

    # Clin
    estimator_clin = lm.LogisticRegression(class_weight='balanced', fit_intercept=False, C=C)
    estimator_clin.fit(X_train_clin, y_train)

    # Compute score for macro avg
    auc_test_clin.append(metrics.roc_auc_score(y_test, estimator_clin.predict_proba(X_test_clin)[:, 1]))
    recalls_test_clin.append(metrics.recall_score(y_test, estimator_clin.predict(X_test_clin).ravel(), average=None))
    acc_test_clin.append(metrics.accuracy_score(y_test, estimator_clin.predict(X_test_clin).ravel()))
    coefs_boot_clin[boot_i, :] = estimator_clin.coef_.ravel()

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
    estimator_stck = lm.LogisticRegression(class_weight='balanced', fit_intercept=False, C=100)
    estimator_stck.fit(X_train_stck, y_train)

    # Compute score for macro avg
    auc_test_stck.append(metrics.roc_auc_score(y_test, estimator_stck.predict_proba(X_test_stck)[:, 1]))
    recalls_test_stck.append(metrics.recall_score(y_test, estimator_stck.predict(X_test_stck).ravel(), average=None))
    acc_test_stck.append(metrics.accuracy_score(y_test, estimator_stck.predict(X_test_stck).ravel()))
    coefs_boot_stck[boot_i, :] = estimator_stck.coef_.ravel()


import glob
# Compute macro avg

coef_boot_img = list()
auc_boot_img = list()
recalls_boot_img = list()
acc_boot_img = list()

filenames = glob.glob(os.path.join(WD,  IMADATASET+"-clust%i"%CLUST + "_" + keystr + "_boot/cooef_boot-*.npz"))
#f = filenames[0]
for f in filenames:
    boot = np.load(f)
    y_test = y[boot["test_idx"]]
    y_test_prob_pred_img = boot["y_pred"]
    y_test_pred_img = boot["y_pred"]
    coef_boot_img.append(boot["beta"])
    # Compute score for macro avg
    auc_boot_img.append(metrics.roc_auc_score(y_test, y_test_prob_pred_img))
    recalls_boot_img.append(metrics.recall_score(y_test, y_test_pred_img, average=None))
    acc_boot_img.append(metrics.accuracy_score(y_test, y_test_pred_img))

TODO

###############################################################################
# Signature
from nilearn import plotting, image
import  nibabel
from matplotlib.backends.backend_pdf import PdfPages
CLUST = 1

# CV
# --

modelscv = np.load(os.path.join(WD,  IMADATASET+"-clust%i"%CLUST +"_enettv_0.1_0.1_0.8_5cv.npz"))
mask_img = nibabel.load(os.path.join(WD, "mask.nii.gz"))
coef_arr = np.zeros(mask_img.get_data().shape)

coef_refit = modelscv['coef_refitall_clust1']
coef_refit0 = modelscv['coef_refitall_clust0']

coef_cv = modelscv['coefs_cv']
#coef_avgcv = coef_cv.mean(axis=0)

pd.Series(coef_refit).describe(percentiles=[0.01, 0.05, 0.1, .25, .5, .75, 0.9])
"""
mean     1.964499e-06
std      2.665711e-04
min     -3.996681e-02
1%      -3.560218e-05
5%      -9.006354e-07
10%     -4.374313e-07
25%     -5.879315e-08
50%      1.644117e-07
75%      8.587507e-07
90%      2.337638e-06
max      4.493354e-02
"""
pd.Series(np.abs(coef_refit)).describe(percentiles=[0.01, 0.05, 0.1, .25, .5, .75, 0.9])

"""
count    3.975590e+05
mean     1.594814e-05
std      2.661009e-04
min      0.000000e+00
1%       0.000000e+00
5%       8.501144e-09
10%      2.956393e-08
25%      1.116976e-07
50%      3.801250e-07
75%      1.095108e-06
90%      2.842495e-06
max      4.493354e-02
"""

# Bootstrap
# ---------

import glob

coef_boot_img = np.array(
        [np.load(f)["beta"] for f in glob.glob(os.path.join(WD,  IMADATASET+"-clust%i"%CLUST + "_" + keystr + "_boot/cooef_boot-*.npz"))]
)

print(coef_boot_img.shape)

# Plot
# ----
print(
np.corrcoef(coef_refit, coef_cv.mean(axis=0))[0, 1],
np.corrcoef(coef_refit, coef_boot_img.mean(axis=0))[0, 1],
np.corrcoef(coef_cv.mean(axis=0), coef_boot_img.mean(axis=0))[0, 1],
np.corrcoef(coef_cv.std(axis=0), coef_boot_img.std(axis=0))[0, 1])

# 0.942233238632 0.745839768565 0.806542301258 0.73123706908


"""
cd /neurospin/psy/canbind/models/clustering_v02/
cp XTreatTivSite-clust1_enettv_0.1_0.1_0.8_5_refit.nii.gz ./coefs_map
cd ./coefs_map

image_clusters_analysis_nilearn.py XTreatTivSite-clust1_enettv_0.1_0.1_0.8_5_refit.nii.gz -o ./ --thresh_norm_ratio 0.99 --thresh_size 10
"""
figure_filename = os.path.join(WD,  IMADATASET+"-clust%i"%CLUST +"_enettv_0.1_0.1_0.8_5.pdf")
pdf = PdfPages(figure_filename)

mask_img = nibabel.load(os.path.join(WD, "mask.nii.gz"))
coef_arr = np.zeros(mask_img.get_data().shape)

# Boot
fig = plt.figure()
coef_boot_img_avg = coef_boot_img.mean(axis=0)
coef_boot_img_std = coef_boot_img.std(axis=0)
coef_boot_img_avg[np.abs(coef_boot_img_avg) < coef_boot_img_std] = 0
coef_arr[mask_img.get_data() != 0] = coef_boot_img_avg
coef_img = nibabel.Nifti1Image(coef_arr, affine=mask_img.affine)
plotting.plot_glass_brain(coef_img,  vmax=5e-4, cmap=plt.cm.bwr, colorbar=True, plot_abs=False, title='Signature mean boot where mean>sd')#, figure=fig, axes=ax)
pdf.savefig(); plt.close()

# Refit all
coef_arr[mask_img.get_data() != 0] = coef_refit
#coef_arr[np.abs(coef_arr)<=1e-9] = np.nan
coef_img = nibabel.Nifti1Image(coef_arr, affine=mask_img.affine)
plotting.plot_glass_brain(coef_img,  vmax=5e-4, cmap=plt.cm.bwr, colorbar=True, plot_abs=False)#, figure=fig, axes=ax)
coef_img.to_filename(os.path.join(WD,  IMADATASET+"-clust%i"%CLUST +"_enettv_0.1_0.1_0.8_5_refit.nii.gz"))


fig = plt.figure()
plotting.plot_glass_brain(coef_img,  vmax=5e-4, cmap=plt.cm.bwr, colorbar=True, plot_abs=False, title='Signature refit')#, figure=fig, axes=ax)
pdf.savefig(); plt.close()

fig = plt.figure()
plotting.plot_stat_map(coef_img, display_mode='z', cut_coords=7,
                       title='Signature',  vmax=1e-4, colorbar=True, cmap=plt.cm.bwr, threshold=1e-6)
pdf.savefig(); plt.close()

fig = plt.figure()
plotting.plot_stat_map(coef_img, display_mode='y', cut_coords=7,
                       title='Signature',  vmax=1e-4, colorbar=True, cmap=plt.cm.bwr, threshold=1e-6)
pdf.savefig(); plt.close()

fig = plt.figure()
plotting.plot_stat_map(coef_img, display_mode='x', cut_coords=7,
                       title='Signature',  vmax=1e-4, colorbar=True, cmap=plt.cm.bwr, threshold=1e-6)

pdf.savefig(); plt.close()

pdf.close()

"""
Manual inspection fsleye

# Positive clusters

R clusters
R
39.0% Postcentral Gyrus (Show/Hide)
16.0% Supramarginal Gyrus, anterior division
0.00052 (5e-4)

Regional increases of cortical thickness in untreated, first-episode major depressive disorder
ranslational Psychiatry 4(4):e378
"Areas with cortical thickness differences between healthy controls and patients with major depression (left) after FDR correction.
Scatterplots show the negative correlation between HDRS with right rostral middle frontal gyrus and right supramarginal gyrus (right).
Warmer colors (positive values) represent cortical thickening; cooler colors (negative values) represent signi
ficant cortical thinning in MDD patients.

R
52.0% Precuneous Cortex (Show/Hide)
+0.00013 (1e-4)

53.0% Middle Temporal Gyrus, temporooccipital part
+0.00017 (1e-4)

Right Amygdala (Show/Hide)
Right Parahippocampal Gyrus, anterior division
Rigth and Left Temporal Fusiform
+0.0003 (3e-4)

Left Cerebelum anterior parts of VIIIa and VIIIb
+0.00084

Voxel-based lesion symptom mapping analysis of depressive mood in patients with isolated cerebellar stroke: A pilot study
https://www.sciencedirect.com/science/article/pii/S2213158216302170
Voxel-wise subtraction and χ (Ayerbe et al., 2014) analyses indicated that damage to the left posterior cerebellar hemisphere was associated with depression. Significant correlations were also found between the severity of depressive symptoms and lesions in lobules VI, VIIb, VIII, Crus I, and Crus II of the left cerebellar hemisphere (Pcorrected = 0.045). Our results suggest that damage to the left posterior cerebellum is associated with increased depressive mood severity in patients with isolated cerebellar stroke.

# Negative

R (L)
100.0% Left Thalamus (Show/Hide)
R:-0.0009
L:-0.00003

R (L)
90.6% Brain-Stem (Show/Hide)
R:-0.0009
L:-0.0009


de Brouwer E.J.M., Kockelkoren R., Claus J.J., et al. "Hippocampal Calcifications: Risk Factors and Association with Cognitive Function.” Radiology, June 12, 2018. https://doi.org/10.1148/radiol.2018172588
"""
