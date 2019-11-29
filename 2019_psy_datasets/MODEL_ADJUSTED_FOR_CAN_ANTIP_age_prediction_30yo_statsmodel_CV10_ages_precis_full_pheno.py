#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 14:16:38 2019

@author: anton
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sklearn
import sklearn.linear_model as lm
from sklearn.svm import LinearSVC
import sklearn.metrics as metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.externals.joblib import Parallel, delayed
from sklearn.base import is_classifier, clone
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import  make_scorer,accuracy_score,recall_score,precision_score
from sklearn import datasets
from sklearn.metrics import recall_score
from sklearn import svm, metrics, linear_model
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import  make_scorer,accuracy_score,recall_score,precision_score
from sklearn import svm
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV
from sklearn import datasets, svm
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.svm import LinearSVR
from sklearn.model_selection import LeaveOneGroupOut
from scipy.stats.stats import pearsonr, spearmanr
from scipy.stats import wilcoxon
from statannot.statannot import add_stat_annotation
from matplotlib import pyplot
import statsmodels.formula.api as sm
from scipy.stats import mannwhitneyu, ttest_ind

PATH = '/home/anton/Dropbox/PhD/Article_Age_epigenetique_et_cerebral/2019_Brain_age/Analysis_pooled_ROIs_pooled_phenotypes/data'

FS_cort = pd.read_csv(os.path.join(PATH,'FS_volume_SCHIZCONNECT_VIP_PRAGUE_BSNIP_BIOBD_ICAAR_START.tsv'), sep = '\t')
FS_subcort = pd.read_csv(os.path.join(PATH,'FS_subcort_vol_SCHIZCONNECT_VIP_PRAGUE_BSNIP_BIOBD_ICAAR_START.tsv'), sep = '\t')
FS_thick = pd.read_csv(os.path.join(PATH,'FS_thickness_SCHIZCONNECT_VIP_PRAGUE_BSNIP_BIOBD_ICAAR_START.tsv'), sep = '\t')
FS_surface = pd.read_csv(os.path.join(PATH,'FS_surface_area_SCHIZCONNECT_VIP_PRAGUE_BSNIP_BIOBD_ICAAR_START.tsv'), sep = '\t')
pheno = pd.read_csv(os.path.join(PATH,'phenotypes_SCHIZCONNECT_VIP_PRAGUE_BSNIP_BIOBD_ICAAR_START_ages_precis.tsv'), sep='\t')
assert FS_cort.shape == (2317, 69)
assert FS_subcort.shape == (2317, 67)
assert FS_thick.shape == (2317, 71)
assert FS_surface.shape == (2317, 71)
assert pheno.shape == (3871, 48)

# selection of ROIs like in Chung, JAMA 2018

# 92 features

#Right and left bankssts volume
#Right and left caudal anterior cingulate volume
#Right and left caudal middle frontal volume
#Right and left cuneus volume
#Right and left entorhinal volume
#Right and left fusiform volume
#Right and left inferior parietal volume
#Right and left inferior temporal volume
#Right and left isthmus cingulate volume
#Right and left lateral occipital volume
#Right and left lateral orbitofrontal volume
#Right and left lingual volume
#Right and left medial orbitofrontal volume
#Right and left middle temporal volume
#Right and left parahippocampal volume
#Right and left paracentral volume
#Right and left parsopercularis volume
#Right and left parsorbitalis volume
#Right and left parstriangularis volume
#Right and left pericalcarine volume
#Right and left postcentral volume
#Right and left posterior cingulate volume
#Right and left precentral volume
#Right and left precuneus volume
#Right and left rostral anterior cingulate volume
#Right and left rostral middle frontal volume
#Right and left superior frontal volume
#Right and left superior parietal volume
#Right and left superior temporal volume
#Right and left supramarginal volume
#Right and left frontalpole volume
#Right and left temporalpole volume
#Right and left transverse temporal volume
#Right and left insula volume
#Right and left lateral ventricle
#Right and left cortex volume
#Right and left white surface area ### to find
#Right and left mean thickness
#Right and left caudate volume
#Right and left putamen volume
#Right and left pallidum volume
#Right and left hippocampus volume
#Right and left amygdala volume
#Right and left accumbens volume
#Right and left thalamus volume
#Third ventricle volume
#Estimated total intracranial volume

list(FS_cort) # keep all
list(FS_subcort)
subcort = FS_subcort[['participant_id',
 'Left-Lateral-Ventricle',
 'Right-Lateral-Ventricle',
 'lhCortexVol',
 'rhCortexVol',
 'Left-Caudate',
 'Right-Caudate',
 'Left-Putamen',
 'Right-Putamen',
 'Left-Pallidum',
 'Right-Pallidum',
 'Left-Hippocampus',
 'Right-Hippocampus',
 'Left-Amygdala',
 'Right-Amygdala',
 'Left-Accumbens-area',
 'Right-Accumbens-area',
 'Left-Thalamus-Proper',
 'Right-Thalamus-Proper',
 '3rd-Ventricle',
 'EstimatedTotalIntraCranialVol']]
 
list(FS_thick)
thick = FS_thick[['participant_id','lh_MeanThickness','rh_MeanThickness']]

list(FS_surface)
surf = FS_surface[['participant_id','lh_WhiteSurfArea','rh_WhiteSurfArea']]

assert list(FS_cort.participant_id) == list(thick.participant_id) == list(subcort.participant_id) == list(surf.participant_id)
dfs = [FS_cort, subcort, thick, surf]
all_feat = pd.concat([df.set_index('participant_id') for df in dfs], axis=1, join='outer').reset_index()
assert all_feat.shape == (2317, 93)



"""
SELECT PATIENTS WITH AGE <= 30
"""

pheno = pheno[pheno.age <= 30]


data = pd.merge(pheno, all_feat, on='participant_id', how='left')
data = data[data.rh_WhiteSurfArea.notna() & data.age.notna() & data.sex.notna() & data.site.notna()]

assert data.shape == (1061, 140)
assert data.duplicated().sum() == 0

# ET

assert list(data).index('lh_bankssts') == 48
assert list(data).index('rh_WhiteSurfArea') == 139
X = data.iloc[:,np.r_[48:140]]
duplicates = data[X.duplicated(keep=False)]
assert duplicates.shape == (0, 140)

######################################################
set(data.site[data.study == 'BIOBD'])
# {'creteil', 'grenoble', 'mannheim', 'pittsburgh', 'udine'}
set(data.site[data.study == 'BSNIP'])
# {'Baltimore', 'Boston', 'Dallas', 'Detroit', 'Hartford'}
set(data.site[data.study == 'SCHIZCONNECT-VIP'])
# {'MRN', 'NU', 'WUSTL', 'vip'}
set(data.site[data.study == 'PRAGUE'])
# {'PRAGUE'}
set(data.site[data.study == 'ICAAR_EUGEI_START'])
# {'ICM', 'Sainte-Anne'}


###############################################################################

assert len(data[data.site == 'Baltimore']) == 100
assert len(data[data.site == 'Boston']) == 48
assert len(data[data.site == 'Dallas']) == 63
assert len(data[data.site == 'Detroit']) == 29
assert len(data[data.site == 'Hartford']) == 142
assert len(data[data.site == 'ICM']) == 73
assert len(data[data.site == 'MRN']) == 59
assert len(data[data.site == 'NU']) == 35
assert len(data[data.site == 'PRAGUE']) == 85
assert len(data[data.site == 'Sainte-Anne']) == 94
assert len(data[data.site == 'WUSTL']) == 173
assert len(data[data.site == 'creteil']) == 36
assert len(data[data.site == 'grenoble']) == 5
assert len(data[data.site == 'mannheim']) == 17
assert len(data[data.site == 'pittsburgh']) == 31
assert len(data[data.site == 'udine']) == 33
assert len(data[data.site == 'vip']) == 38


import statsmodels.formula.api as smf

data.sex = data.sex.map({1.0:'F',0.0:'M'})
data_r = data.copy()
data_r = data.rename(columns = {
        'Left-Lateral-Ventricle':'Left_Lateral_Ventricle',
        'Right-Lateral-Ventricle':'Right_Lateral_Ventricle',
        'Left-Caudate':'Left_Caudate',
        'Right-Caudate':'Right_Caudate',
        'Left-Putamen':'Left_Putamen',
        'Right-Putamen':'Right_Putamen',
        'Left-Pallidum':'Left_Pallidum',
        'Right-Pallidum':'Right_Pallidum',
        'Left-Hippocampus':'Left_Hippocampus',
        'Right-Hippocampus':'Right_Hippocampus',
        'Left-Amygdala':'Left_Amygdala',
        'Right-Amygdala':'Right_Amygdala',
        'Left-Accumbens-area':'Left_Accumbens_area',
        'Right-Accumbens-area':'Right_Accumbens_area',
        'Left-Thalamus-Proper':'Left_Thalamus_Proper',
        'Right-Thalamus-Proper':'Right_Thalamus_Proper',
        '3rd-Ventricle':'third_Ventricle'})

# ordinary least squares model applied on controls from the training set

data_r_train_ctrl = data_r[(data_r.diagnosis == 'control') & (data_r.study.isin(['SCHIZCONNECT-VIP','PRAGUE','BIOBD']))]
data_r_test_ctrl = data_r[(data_r.diagnosis == 'control') & (data_r.study.isin(['BSNIP']))]
data_r_uhr_test = data_r[(data_r.diagnosis.isin(['UHR-C', 'UHR-NC']))]

data_r_uhr_test.Eq_Chlorpromazine.fillna(0, inplace=True)
data_r_uhr_test.MADRS.fillna(0, inplace=True)
data_r_uhr_test.cannabis_last_month.fillna(0, inplace=True)

data_r_test_ctrl.to_csv('/home/anton/Dropbox/PhD/Article_Age_epigenetique_et_cerebral/2019_Brain_age/Analysis_pooled_ROIs_pooled_phenotypes/data/data_r_test_ctrl.csv', index=False)
data_r_train_ctrl.to_csv('/home/anton/Dropbox/PhD/Article_Age_epigenetique_et_cerebral/2019_Brain_age/Analysis_pooled_ROIs_pooled_phenotypes/data/data_r_train_ctrl.csv', index=False)
data_r_uhr_test.to_csv('/home/anton/Dropbox/PhD/Article_Age_epigenetique_et_cerebral/2019_Brain_age/Analysis_pooled_ROIs_pooled_phenotypes/data/data_r_uhr_test.csv', index=False)

# linear mixed model fait sur R

X_train_ctrl = pd.read_csv("/home/anton/Dropbox/PhD/Article_Age_epigenetique_et_cerebral/2019_Brain_age/Analysis_pooled_ROIs_pooled_phenotypes/data/residuals_data_r_train_ctrl.csv")
X_train_ctrl = np.array(X_train_ctrl)
assert X_train_ctrl.shape == (316, 92)
age_train_ctrl = np.array(data_r_train_ctrl.age)
assert age_train_ctrl.shape == (316,)

X_test_ctrl = pd.read_csv("/home/anton/Dropbox/PhD/Article_Age_epigenetique_et_cerebral/2019_Brain_age/Analysis_pooled_ROIs_pooled_phenotypes/data/residuals_data_r_test_ctrl.csv")
X_test_ctrl = np.array(X_test_ctrl)
assert X_test_ctrl.shape == (70, 92)
age_test_ctrl = np.array(data_r_test_ctrl.age)
assert age_test_ctrl.shape == (70,)

###############################################################################
""" MODEL ADJUSTED FOR ANTIPSYCHOTICS AND CANNABIS BUT NOT DEPRESSION """
###############################################################################

X_test_uhr = pd.read_csv("/home/anton/Dropbox/PhD/Article_Age_epigenetique_et_cerebral/2019_Brain_age/Analysis_pooled_ROIs_pooled_phenotypes/data/residuals_data_r_uhr_test_not_adjusted_for_MADRS.csv")
X_test_uhr = np.array(X_test_uhr)
assert X_test_uhr.shape == (96, 92)
participant_id = np.array(data_r_uhr_test.participant_id)
age_test_uhr = np.array(data_r_uhr_test.true_age)
status_test_uhr = np.array(data_r_uhr_test.diagnosis)
irm_test_uhr = np.array(data_r_uhr_test.irm)
sex_test_uhr = np.array(data_r_uhr_test.sex)
assert age_test_uhr.shape == status_test_uhr.shape == (96,)



###############################################################################


X = X_train_ctrl
y = age_train_ctrl

param_grid = {'alpha': 10. ** np.arange(-5, 5)}
model = GridSearchCV(lm.Ridge(max_iter=10000, tol = 0.0001, random_state = 42), param_grid, cv=10)
scaler = StandardScaler()

r2_train, r2_test = list(), list()

cv = KFold(n_splits=10, random_state = 42)
cv = [[train, test] for train, test in cv.split(X, y)]
   
for train, test in cv:
    X_train, X_test, y_train, y_test = X[train, :], X[test, :], y[train], y[test]
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    y_train_s = scaler.fit_transform(y_train.reshape(-1, 1))
    y_test_s = scaler.fit_transform(y_test.reshape(-1, 1))
    model.fit(X_train_s, y_train_s)
    y_train_prediction = model.predict(X_train_s)
    y_test_prediction = model.predict(X_test_s)
    r2_train.append(metrics.r2_score(y_train_s, y_train_prediction))
    r2_test.append(metrics.r2_score(y_test_s, y_test_prediction))
    
    
results = {'train':r2_train, 'test':r2_test}
results = pd.DataFrame(results)
print(results)
#      train      test
#0  0.383946  0.099812
#1  0.379211  0.150032
#2  0.194978  0.142170
#3  0.374392  0.284507
#4  0.196718  0.249329
#5  0.211872  0.077886
#6  0.215103  0.109933
#7  0.218516  0.102509
#8  0.204140  0.179489
#9  0.393684  0.062842

print("Train r2:%.2f" % np.mean(r2_train))
print("Test r2:%.2f" % np.mean(r2_test))
print(model.best_params_)
#Train r2:0.28
#Test r2:0.15
#{'alpha': 100.0}


# VALIDATION

X_train_ctrl_s = scaler.fit_transform(X_train_ctrl)
X_test_ctrl_s = scaler.transform(X_test_ctrl)
X_test_uhr_s = scaler.transform(X_test_uhr)
age_train_ctrl_s = scaler.fit_transform(age_train_ctrl.reshape(-1, 1))
age_test_ctrl_s = scaler.fit_transform(age_test_ctrl.reshape(-1, 1))
age_test_uhr_s = scaler.fit_transform(age_test_uhr.reshape(-1, 1))

# fit the model on all of the control train images
# fit the gap predictor on all of the control train ages

model = lm.Ridge(alpha = 100, random_state = 42)
model.fit(X_train_ctrl_s, age_train_ctrl_s)
age_train_ctrl_predicted = model.predict(X_train_ctrl_s)
print("Train r2:%.2f" % metrics.r2_score(age_train_ctrl_s, age_train_ctrl_predicted))
#Train r2:0.37

COEF = pd.DataFrame(model.coef_.T, columns=['coefficients'])
ROI = pd.DataFrame(list(data_r_uhr_test)[48:140], columns=['regions'])
ROI_weights = pd.concat([ROI,COEF], axis=1)
ROI_weights['abs_value'] = abs(ROI_weights.coefficients)
ROI_weights = ROI_weights.sort_values('abs_value', ascending=False)

ROI_weights.to_csv('/home/anton/Dropbox/PhD/Article_Age_epigenetique_et_cerebral/results/ROI_weights')


# test the fitted model on the control test population

age_test_ctrl_predicted = model.predict(X_test_ctrl_s)
print("Test r2:%.2f" % metrics.r2_score(age_test_ctrl_s, age_test_ctrl_predicted))
#Test r2:0.10

# apply the model to the uhr test population
age_test_uhr_predicted = model.predict(X_test_uhr_s)
print("Test r2:%.2f" % metrics.r2_score(age_test_uhr_s, age_test_uhr_predicted))
# Test r2:0.02



# get the predicted ages de-normalized
age_train_ctrl_predicted = age_train_ctrl_predicted * age_train_ctrl.std() + age_train_ctrl.mean()
age_test_ctrl_predicted = age_test_ctrl_predicted * age_test_ctrl.std() + age_test_ctrl.mean()
age_test_uhr_predicted = age_test_uhr_predicted * age_test_uhr.std() + age_test_uhr.mean()

###############################################################################

# PLOT AGE = f(prediction)

# for control training set
a_list = age_train_ctrl_predicted.tolist()
flat_list = [item for sublist in a_list for item in sublist]
age_train_ctrl_predicted = np.array(flat_list)

metrics.r2_score(age_train_ctrl, age_train_ctrl_predicted) # 0.3651324028837888
pearsonr(age_train_ctrl, age_train_ctrl_predicted) # (0.6177782910012306, 1.2130849330854466e-34)
    
real_age = pd.DataFrame(age_train_ctrl, columns=['real age'])
predicted_age = pd.DataFrame(age_train_ctrl_predicted, columns=['brain age'])
data = pd.concat([predicted_age, real_age],axis=1)
dims = (6,6)
fig, ax = pyplot.subplots(figsize=dims)
sns.scatterplot(x='real age',y='brain age',data=data, ax=ax)
plt.title("Correlation between real and brain age in the training set \n Pearson's r = 62%, p = 1.21e-34")



# for control test set
a_list = age_test_ctrl_predicted.tolist()
flat_list = [item for sublist in a_list for item in sublist]
age_test_ctrl_predicted = np.array(flat_list)

metrics.r2_score(age_test_ctrl, age_test_ctrl_predicted) # 0.10477152163834269
pearsonr(age_test_ctrl, age_test_ctrl_predicted)  # (0.34112698107569966, 0.0038535371772717426)

real_age = pd.DataFrame(age_test_ctrl, columns=['real age'])
predicted_age = pd.DataFrame(age_test_ctrl_predicted, columns=['brain age'])
data = pd.concat([predicted_age, real_age],axis=1)
dims = (6,6)
fig, ax = pyplot.subplots(figsize=dims)
sns.scatterplot(x='real age',y='brain age',data=data, ax=ax)
plt.title("Correlation between real and brain age in the testing set \n Pearson's r = 34%, p = 3.85e-3")


# for uhr test set at baseline
a_list = age_test_uhr_predicted.tolist()
flat_list = [item for sublist in a_list for item in sublist]
age_test_uhr_predicted = np.array(flat_list)

metrics.r2_score(real_age, predicted_age) # 0.017827744894631903
pearsonr(age_test_uhr, age_test_uhr_predicted) # (0.26348822694057783, 0.009491173442177205)

real_age = pd.DataFrame(age_test_uhr[irm_test_uhr == 'M0'], columns=['real age'])
predicted_age = pd.DataFrame(age_test_uhr_predicted[irm_test_uhr == 'M0'], columns=['brain age'])
status = pd.DataFrame(status_test_uhr[irm_test_uhr == 'M0'], columns=['status'])
data = pd.concat([predicted_age, real_age, status],axis=1)
dims = (6,6)
fig, ax = pyplot.subplots(figsize=dims)
sns.scatterplot(x='real age',y='brain age', hue='status', hue_order=['UHR-NC','UHR-C'],data=data, ax=ax)
plt.title("Correlation between real and brain age in the uhr set \n Pearson's r = 26%, p = 9.49e-3")

###############################################################################

ageTr = pd.DataFrame(age_train_ctrl, columns=['Real Age'])
ageTr_pred = pd.DataFrame(age_train_ctrl_predicted, columns=['Brain Age'])
train = pd.concat([ageTr, ageTr_pred],axis=1)
train['status'] = 'control train, R = 62%, p = 1.21e-34'
ageTe = pd.DataFrame(age_test_ctrl, columns=['Real Age'])
ageTe_pred = pd.DataFrame(age_test_ctrl_predicted, columns=['Brain Age'])
test = pd.concat([ageTe, ageTe_pred],axis=1)
test['status'] = 'control test, R = 34%, p = 3.85e-3'
ageUHR = pd.DataFrame(age_test_uhr, columns=['Real Age'])
ageUHR_pred = pd.DataFrame(age_test_uhr_predicted, columns=['Brain Age'])
status_uhr = pd.DataFrame(status_test_uhr, columns=['status'])
uhr = pd.concat([ageUHR, ageUHR_pred, status_uhr],axis=1)


# Correlation between real and predicted age in control population

# version 1
sns.set(style="white")
dims = (6,6)
fig, ax = pyplot.subplots(figsize=dims)
data_TR_TE = pd.concat([train, test], axis=0)
sns.scatterplot(x='Real Age',y='Brain Age', hue='status',data=data_TR_TE, ax=ax)
plt.title("Correlation between real and brain ages in \n the train and test control populations")
plt.xlabel("Real Age")
plt.ylabel("Brain Age")

# version 2
g = sns.pairplot(data_TR_TE, hue="status")
g.fig.suptitle("Correlation between real and brain ages in the train and test control populations", y=1.08)


# Correlation between real and predicted age in control train and uhr population

ageTr = pd.DataFrame(age_train_ctrl, columns=['Real Age'])
ageTr_pred = pd.DataFrame(age_train_ctrl_predicted, columns=['Brain Age'])
train = pd.concat([ageTr, ageTr_pred],axis=1)
train['status'] = 'control train, R = 62%, p = 1.21e-34'
ageUHR = pd.DataFrame(age_test_uhr, columns=['Real Age'])
ageUHR_pred = pd.DataFrame(age_test_uhr_predicted, columns=['Brain Age'])
status_uhr = pd.DataFrame(status_test_uhr, columns=['status'])
uhr = pd.concat([ageUHR, ageUHR_pred, status_uhr],axis=1)

pearsonr(age_test_uhr[status_test_uhr == 'UHR-NC'], age_test_uhr_predicted[status_test_uhr == 'UHR-NC'])
# (0.2996105027044387, 0.01616455730688896)
pearsonr(age_test_uhr[status_test_uhr == 'UHR-C'], age_test_uhr_predicted[status_test_uhr == 'UHR-C'])
# (0.2986065404214639, 0.09689655062979347)

uhr.status = uhr.status.map({'UHR-NC':'UHR-NC, R = 30%, p = 0.016','UHR-C':'UHR-C, R = 30%, p = 0.097'})
data1 = pd.concat([train, uhr], axis=0)

# Version 1
sns.set(style="white")
dims = (6,6)
fig, ax = pyplot.subplots(figsize=dims)
sns.scatterplot(x='Real Age',y='Brain Age', hue='status',data=data1, ax=ax)
plt.title("Correlation between real and brain ages in \n the control train and uhr populations")
plt.xlabel("Real Age")
plt.ylabel("Brain Age")

# Version 2
g = sns.pairplot(data1, hue="status")
g.fig.suptitle("Correlation between real and brain ages in the control train and uhr populations", y=1.08)


# Correlation between real and predicted age in uhr population

ageUHR = pd.DataFrame(age_test_uhr, columns=['Real Age'])
ageUHR_pred = pd.DataFrame(age_test_uhr_predicted, columns=['Brain Age'])
status_uhr = pd.DataFrame(status_test_uhr, columns=['status'])
uhr = pd.concat([ageUHR, ageUHR_pred, status_uhr],axis=1)

pearsonr(age_test_uhr[status_test_uhr == 'UHR-NC'], age_test_uhr_predicted[status_test_uhr == 'UHR-NC'])
# (0.30045717108100056, 0.015851911029486927)
pearsonr(age_test_uhr[status_test_uhr == 'UHR-C'], age_test_uhr_predicted[status_test_uhr == 'UHR-C'])
# (0.28731614921020526, 0.1108311598880322)

uhr.status = uhr.status.map({'UHR-NC':'UHR-NC, R = 30%, p = 0.016','UHR-C':'UHR-C, R = 29%, p = 0.11'})

# Version 1
sns.set(style="white")
dims = (6,6)
fig, ax = pyplot.subplots(figsize=dims)
sns.scatterplot(x='Real Age',y='Brain Age', hue='status', hue_order=['UHR-NC, R = 30%, p = 0.016','UHR-C, R = 30%, p = 0.097'],data=uhr, ax=ax)
plt.title("Correlation between real and brain ages in \n the UHR population")
plt.xlabel("Real Age")
plt.ylabel("Brain Age")

# Version 2
g = sns.pairplot(uhr, hue="status",hue_order=['UHR-NC, R = 30%, p = 0.016','UHR-C, R = 29%, p = 0.11'])
g.fig.suptitle("Correlation between real and brain ages in the UHR population", y=1.08)



###############################################################################
# RESULTS #
###############################################################################

# Résultats à M0

# pour les contrôles
brain_age_test_ctrl = pd.DataFrame(age_test_ctrl_predicted)
real_age_test_ctrl = pd.DataFrame(age_test_ctrl)
data_ctrl = pd.concat([real_age_test_ctrl, brain_age_test_ctrl], axis=1)
data_ctrl.columns = ['real age ctrl', 'predicted age ctrl']
data_ctrl['age_diff_ctrl'] = data_ctrl['predicted age ctrl'] - data_ctrl['real age ctrl']
assert data_ctrl['real age ctrl'].mean() == 24.385714285714286
assert data_ctrl['real age ctrl'].median() == 24.5
assert data_ctrl['predicted age ctrl'].mean() == 24.385714285714286
assert data_ctrl['predicted age ctrl'].median() == 24.48975631885893
assert data_ctrl['age_diff_ctrl'].mean() == -1.8778629445088362e-15

# pour les UHR

data['age_diff'] = data['brain age'] - data['real age']


###############################################################################

# The gap is the difference between predicted and real age.
# Is there a difference in this gap at M0, between converters and non-converters ?

age_diff_ctrl = data_ctrl.age_diff_ctrl
age_diff_C = data[data.status == 'UHR-C'].age_diff
age_diff_NC = data[data.status == 'UHR-NC'].age_diff

age_diff_C.mean() # 1.5713491902917616
age_diff_NC.mean() # -0.47343717811824515
age_diff_ctrl.mean() # -1.8778629445088362e-15

len(age_diff_NC) # 53
len(age_diff_C) # 27
len(age_diff_ctrl) # 70

# rejecting outliers
    
def reject_outliers(x, mad):
    return x[abs(x - x.median()) < 3 * mad]

mad_age_diff_C = 1.4826 * np.median(np.abs(age_diff_C - age_diff_C.median()))
mad_age_diff_NC = 1.4826 * np.median(np.abs(age_diff_NC - age_diff_NC.median()))
mad_age_diff_ctrl = 1.4826 * np.median(np.abs(age_diff_ctrl - age_diff_ctrl.median()))

age_diff_C_no_out = reject_outliers(age_diff_C, mad_age_diff_C)
age_diff_NC_no_out = reject_outliers(age_diff_NC, mad_age_diff_NC)
age_diff_ctrl_no_out = reject_outliers(age_diff_ctrl, mad_age_diff_ctrl)

age_diff_C_no_out.mean() # 1.5713491902917616
age_diff_C_no_out.std() # 2.0054287330501523
age_diff_NC_no_out.mean() # -0.47343717811824515
age_diff_NC_no_out.std() # 2.9659912382255307

# ES (Hedge's g) = 0.75441

age_diff_ctrl_no_out.mean() # -1.8778629445088362e-15
age_diff_ctrl_no_out.std() # 3.414460981242554




############# BOOTSTRAPPING POUR VOIR LA VARIATION DE LA DIFFERENCE D'AGE (calcul de l'intervalle de confiance) #####


# pour les contrôles
#X_ctrl = age_diff_ctrl
X_ctrl = age_diff_ctrl_no_out
nboot = 1000
variation_median_ctrl = []
for boot in range(nboot):
    boot = np.random.choice(X_ctrl, size=len(X_ctrl), replace=True)
    variation_median_ctrl += [np.median(boot)]
    
CI_1 = np.percentile(variation_median_ctrl,2.5) # -1.3189696253494656
CI_2 = np.percentile(variation_median_ctrl,97.5) # 0.8292190721994857

# pour les non-converteurs
#X_NC = age_diff_NC    
X_NC = age_diff_NC_no_out
nboot = 1000
variation_median_NC = []
for boot in range(nboot):
    boot = np.random.choice(X_NC, size=len(X_NC), replace=True)
    variation_median_NC += [np.median(boot)]
    
CI_1 = np.percentile(variation_median_NC,2.5) # -0.9898636440500219
CI_2 = np.percentile(variation_median_NC,97.5) # 0.5877642492428699

# pour les converteurs
#X_C = age_diff_C    
X_C = age_diff_C_no_out
nboot = 1000
variation_median_C = []
for boot in range(nboot):
    boot = np.random.choice(X_C, size=len(X_C), replace=True)
    variation_median_C += [np.median(boot)]
    
CI_1 = np.percentile(variation_median_C,2.5) # 1.0242676029420075
CI_2 = np.percentile(variation_median_C,97.5) # 2.7655019532897036
    
variation_median_ctrl = pd.DataFrame(variation_median_ctrl)  
variation_median_NC = pd.DataFrame(variation_median_NC)
variation_median_C = pd.DataFrame(variation_median_C)

# graphical representation

data_median = pd.concat([variation_median_ctrl, variation_median_NC, variation_median_C], axis=1)
data_median.columns = ['Controls','Non-Converters', 'Converters'] 

dims = (8, 12)
fig, ax = plt.subplots(figsize=dims)
sns.set(style="whitegrid")
ax.set_title("Difference in brain age gap between \n Controls, Non-Converters and Converters at baseline")
plt.ylabel('Brain age gap = Brain age - Real age')
my_pal = {"Controls": "g", "Non-Converters": "b", "Converters":"darkorange"}
ax = sns.boxplot(data=data_median, palette=my_pal)
add_stat_annotation(ax, data=data_median,
                    box_pairs=[("Controls", "Non-Converters"),
                                 ("Non-Converters", "Converters"), 
                                 ("Controls", "Converters")],
                    test='Mann-Whitney', text_format='star', loc='inside', verbose=2)



data_median = pd.concat([variation_median_NC, variation_median_C], axis=1)
data_median.columns = ['Non-Converters', 'Converters'] 

dims = (8, 12)
fig, ax = plt.subplots(figsize=dims)
sns.set(style="whitegrid")
ax.set_title("Difference in brain age gap between \n Non-Converters and Converters at baseline")
plt.ylabel('Brain age gap = Brain age - Real age')
ax = sns.boxplot(data=data_median)
add_stat_annotation(ax, data=data_median,
                    box_pairs=[("Non-Converters", "Converters")],
                    test='Mann-Whitney', text_format='star', loc='inside', verbose=2)



############# PERMUTATION TEST (calcul de p non-paramétrique) POUR C vs NC

    
C = age_diff_C_no_out
NC = age_diff_NC_no_out
C_array = np.array(C)
NC_array = np.array(NC)
assert len(C_array) == 27
assert len(NC_array) == 53
C_array = C_array[:,np.newaxis]
NC_array = NC_array[:,np.newaxis]

T_array = np.ones(27)
NT_array = np.zeros(53)
T_array = T_array[:,np.newaxis]
NT_array = NT_array[:,np.newaxis]
array_T = np.c_[C_array,T_array]
array_NT = np.c_[NC_array,NT_array]
data_array = np.r_[array_T,array_NT]
columns = ['brain_ageing','conversion']
df_brain_ageing = pd.DataFrame(data_array, columns=columns)
df_brain_ageing.conversion = df_brain_ageing.conversion.map({1.0:'yes', 0.0:'no'})

# Permutation: simulate the null hypothesis

nperm = 10000
perms = np.zeros(nperm + 1)
p = np.zeros(nperm + 1)
perms[0], p[0] = mannwhitneyu(df_brain_ageing.brain_ageing[df_brain_ageing.conversion == 'yes'],df_brain_ageing.brain_ageing[df_brain_ageing.conversion == 'no'])

# je veux permuter aléatoirement l'assignation à chaque groupe puis faire le wilcoxon
    # je permute le statut converteur
    # je les mets dans un nouveau tableau
    # je fais le mann whitney
    # je stocke la statistique U et p
    
for i in range(1,nperm):
    brain_ageing = df_brain_ageing.brain_ageing.values
    brain_ageing = brain_ageing[:,np.newaxis]
    perm = np.random.permutation(df_brain_ageing.conversion)
    perm = perm[:,np.newaxis]
    perm_assign = np.c_[brain_ageing,perm]
    df_permuted = pd.DataFrame(perm_assign, columns=columns)
    perms[i], p[i] = mannwhitneyu(df_permuted.brain_ageing[df_permuted.conversion == 'yes'],df_permuted.brain_ageing[df_permuted.conversion == 'no'])

# Two-tailed empirical p-value
pval_perm = np.sum(p <= p[0])/p.shape[0]
pval_perm #  0.0025997400259974

# Plot
from matplotlib.pyplot import figure
# Re-weight to obtain distribution
figure(num=None, figsize=(8,6), dpi=80, facecolor='w', edgecolor='k')
weights = np.ones(perms.shape[0]) / perms.shape[0]
plt.hist([perms[perms >= perms[0]], perms], histtype='stepfilled', bins=100, label=["U > U obs", "U < U obs (p-value)"], weights = [weights[perms >= perms[0]], weights])
plt.xlabel("Statistic distribution under the null hypothesis that there is no difference in \n brain age gap at baseline between converters and non-converters")
plt.title("Random permutations analysis (10 000 permutations)")
plt.axvline(x=perms[0], color='red', linewidth=1, label='observed statistic: p = 0.0026')
_ = plt.legend(loc='upper left')

############# PERMUTATION TEST (calcul de p non-paramétrique) pour ctrl vs NC
    
C = age_diff_ctrl_no_out
NC = age_diff_NC_no_out
C_array = np.array(C)
NC_array = np.array(NC)
assert len(C_array) == 70
assert len(NC_array) == 53
C_array = C_array[:,np.newaxis]
NC_array = NC_array[:,np.newaxis]

T_array = np.ones(70)
NT_array = np.zeros(53)
T_array = T_array[:,np.newaxis]
NT_array = NT_array[:,np.newaxis]
array_T = np.c_[C_array,T_array]
array_NT = np.c_[NC_array,NT_array]
data_array = np.r_[array_T,array_NT]
columns = ['brain_ageing','conversion']
df_brain_ageing = pd.DataFrame(data_array, columns=columns)
df_brain_ageing.conversion = df_brain_ageing.conversion.map({1.0:'yes', 0.0:'no'})

# Permutation: simulate the null hypothesis

nperm = 10000
perms = np.zeros(nperm + 1)
p = np.zeros(nperm + 1)
perms[0], p[0] = mannwhitneyu(df_brain_ageing.brain_ageing[df_brain_ageing.conversion == 'yes'],df_brain_ageing.brain_ageing[df_brain_ageing.conversion == 'no'])

# je veux permuter aléatoirement l'assignation à chaque groupe puis faire le wilcoxon
    # je permute le statut converteur
    # je les mets dans un nouveau tableau
    # je fais le mann whitney
    # je stocke la statistique U et p
    
for i in range(1,nperm):
    brain_ageing = df_brain_ageing.brain_ageing.values
    brain_ageing = brain_ageing[:,np.newaxis]
    perm = np.random.permutation(df_brain_ageing.conversion)
    perm = perm[:,np.newaxis]
    perm_assign = np.c_[brain_ageing,perm]
    df_permuted = pd.DataFrame(perm_assign, columns=columns)
    perms[i], p[i] = mannwhitneyu(df_permuted.brain_ageing[df_permuted.conversion == 'yes'],df_permuted.brain_ageing[df_permuted.conversion == 'no'])

# Two-tailed empirical p-value
pval_perm = np.sum(p <= p[0])/p.shape[0]
pval_perm # 0.6418358164183582 ~ 0.64

# Plot
from matplotlib.pyplot import figure
# Re-weight to obtain distribution
figure(num=None, figsize=(8,6), dpi=80, facecolor='w', edgecolor='k')
weights = np.ones(perms.shape[0]) / perms.shape[0]
plt.hist([perms[perms >= perms[0]], perms], histtype='stepfilled', bins=100, label=["U > U obs", "U < U obs (p-value)"], weights = [weights[perms >= perms[0]], weights])
plt.xlabel("Statistic distribution under the null hypothesis that there is no difference in \n brain age gap at baseline between controls and non-converters")
plt.title("Random permutations analysis (10 000 permutations)")
plt.axvline(x=perms[0], color='red', linewidth=1, label='observed statistic: p = 0.64')
_ = plt.legend(loc='upper left')

############# PERMUTATION TEST (calcul de p non-paramétrique) pour ctrl vs C
    
C = age_diff_C_no_out
NC = age_diff_ctrl_no_out
C_array = np.array(C)
NC_array = np.array(NC)
assert len(C_array) == 27
assert len(NC_array) == 70
C_array = C_array[:,np.newaxis]
NC_array = NC_array[:,np.newaxis]

T_array = np.ones(27)
NT_array = np.zeros(70)
T_array = T_array[:,np.newaxis]
NT_array = NT_array[:,np.newaxis]
array_T = np.c_[C_array,T_array]
array_NT = np.c_[NC_array,NT_array]
data_array = np.r_[array_T,array_NT]
columns = ['brain_ageing','conversion']
df_brain_ageing = pd.DataFrame(data_array, columns=columns)
df_brain_ageing.conversion = df_brain_ageing.conversion.map({1.0:'yes', 0.0:'no'})

# Permutation: simulate the null hypothesis

nperm = 10000
perms = np.zeros(nperm + 1)
p = np.zeros(nperm + 1)
perms[0], p[0] = mannwhitneyu(df_brain_ageing.brain_ageing[df_brain_ageing.conversion == 'yes'],df_brain_ageing.brain_ageing[df_brain_ageing.conversion == 'no'])

# je veux permuter aléatoirement l'assignation à chaque groupe puis faire le wilcoxon
    # je permute le statut converteur
    # je les mets dans un nouveau tableau
    # je fais le mann whitney
    # je stocke la statistique U et p
    
for i in range(1,nperm):
    brain_ageing = df_brain_ageing.brain_ageing.values
    brain_ageing = brain_ageing[:,np.newaxis]
    perm = np.random.permutation(df_brain_ageing.conversion)
    perm = perm[:,np.newaxis]
    perm_assign = np.c_[brain_ageing,perm]
    df_permuted = pd.DataFrame(perm_assign, columns=columns)
    perms[i], p[i] = mannwhitneyu(df_permuted.brain_ageing[df_permuted.conversion == 'yes'],df_permuted.brain_ageing[df_permuted.conversion == 'no'])

# Two-tailed empirical p-value
pval_perm = np.sum(p <= p[0])/p.shape[0]
pval_perm # 0.012098790120987902 ~ 0.01

# Plot
from matplotlib.pyplot import figure
# Re-weight to obtain distribution
figure(num=None, figsize=(8,6), dpi=80, facecolor='w', edgecolor='k')
weights = np.ones(perms.shape[0]) / perms.shape[0]
plt.hist([perms[perms >= perms[0]], perms], histtype='stepfilled', bins=100, label=["U > U obs", "U < U obs (p-value)"], weights = [weights[perms >= perms[0]], weights])
plt.xlabel("Statistic distribution under the null hypothesis that there is no difference in \n brain age gap at baseline between controls and converters")
plt.title("Random permutations analysis (10 000 permutations)")
plt.axvline(x=perms[0], color='red', linewidth=1, label='observed statistic: p = 0.01')
_ = plt.legend(loc='upper left')


###############################################################################

# PLOT DISTRIBUTION OF BRAIN AGE AS A FUNCTION OF AGE IN CONVERTERS AND NON-CONVERTERS

real_age = pd.DataFrame(age_test_uhr[irm_test_uhr == 'M0'], columns=['real age'])
predicted_age = pd.DataFrame(age_test_uhr_predicted[irm_test_uhr == 'M0'], columns=['brain age'])
status = pd.DataFrame(status_test_uhr[irm_test_uhr == 'M0'], columns=['status'])
data = pd.concat([predicted_age, real_age, status],axis=1)
sns.lmplot(x='real age',y='brain age',hue='status',hue_order=['UHR-NC','UHR-C'], data=data, size=7)
plt.title("Cross-sectional brain age acceleration in Non-Converters and Converters")



fit = sm.ols("data[data.status == 'UHR-C']['brain age'] ~ data[data.status == 'UHR-C']['real age']", data=data[data.status == 'UHR-C']).fit()
fit.summary()
"""
                                        OLS Regression Results                                       
=====================================================================================================
Dep. Variable:     data[data.status == 'UHR-C']['brain age']   R-squared:                       0.089
Model:                                                   OLS   Adj. R-squared:                  0.059
Method:                                        Least Squares   F-statistic:                     2.937
Date:                                       Sun, 03 Nov 2019   Prob (F-statistic):             0.0969
Time:                                               13:13:47   Log-Likelihood:                -48.045
No. Observations:                                         32   AIC:                             100.1
Df Residuals:                                             30   BIC:                             103.0
Df Model:                                                  1                                         
Covariance Type:                                   nonrobust                                         
============================================================================================================
                                               coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------------------------------------
Intercept                                   18.6742      2.206      8.463      0.000      14.168      23.180
data[data.status == 'UHR-C']['real age']     0.1795      0.105      1.714      0.097      -0.034       0.394
==============================================================================
Omnibus:                        0.060   Durbin-Watson:                   1.495
Prob(Omnibus):                  0.970   Jarque-Bera (JB):                0.248
Skew:                           0.071   Prob(JB):                        0.884
Kurtosis:                       2.593   Cond. No.                         235.
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
"""

fit1 = sm.ols("data[data.status == 'UHR-NC']['brain age'] ~ data[data.status == 'UHR-NC']['real age']", data=data[data.status == 'UHR-NC']).fit()
fit1.summary()
"""
                                        OLS Regression Results                                        
======================================================================================================
Dep. Variable:     data[data.status == 'UHR-NC']['brain age']   R-squared:                       0.090
Model:                                                    OLS   Adj. R-squared:                  0.075
Method:                                         Least Squares   F-statistic:                     6.114
Date:                                        Sun, 03 Nov 2019   Prob (F-statistic):             0.0162
Time:                                                13:14:27   Log-Likelihood:                -112.54
No. Observations:                                          64   AIC:                             229.1
Df Residuals:                                              62   BIC:                             233.4
Df Model:                                                   1                                         
Covariance Type:                                    nonrobust                                         
=============================================================================================================
                                                coef    std err          t      P>|t|      [0.025      0.975]
-------------------------------------------------------------------------------------------------------------
Intercept                                    18.8212      1.420     13.256      0.000      15.983      21.659
data[data.status == 'UHR-NC']['real age']     0.1512      0.061      2.473      0.016       0.029       0.273
==============================================================================
Omnibus:                        7.868   Durbin-Watson:                   1.881
Prob(Omnibus):                  0.020   Jarque-Bera (JB):                7.141
Skew:                          -0.729   Prob(JB):                       0.0281
Kurtosis:                       3.744   Cond. No.                         185.
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
"""



###############################################

# PLOT DISTRIBUTION OF BRAIN VOLUME 
# WITH RESPECT TO AGE IN CONVERTERS AND NON-CONVERTERS

# data_r_uhr_test

list(data_r_uhr_test)
data1 = data_r_uhr_test[['participant_id','diagnosis','EstimatedTotalIntraCranialVol','true_age','irm']]
data1 = data1[data1.irm == 'M0']
data1 = data1.rename(columns={'diagnosis':'status'})

fit1 = sm.ols("EstimatedTotalIntraCranialVol ~ true_age", data=data1).fit()
fit1.summary()

TIV_NC = np.array(data1[data1.status == 'UHR-NC'].EstimatedTotalIntraCranialVol)
age_NC = np.array(data1[data1.status == 'UHR-NC'].true_age)
pearsonr(TIV_NC, age_NC) # (-0.014427956321488009, 0.9183298494662081)

TIV_C = np.array(data1[data1.status == 'UHR-C'].EstimatedTotalIntraCranialVol)
age_C = np.array(data1[data1.status == 'UHR-C'].true_age)
pearsonr(TIV_C, age_C) # (-0.08496156291996229, 0.6735020137500112)


fit2 = sm.ols("data1[data1.status == 'UHR-NC'].EstimatedTotalIntraCranialVol ~ data1[data1.status == 'UHR-NC'].true_age", data=data1[data1.status == 'UHR-NC']).fit()
fit2.summary()
"""
                                                  OLS Regression Results                                                 
=========================================================================================================================
Dep. Variable:     data1[data1.status == 'UHR-NC'].EstimatedTotalIntraCranialVol   R-squared:                       0.000
Model:                                                                       OLS   Adj. R-squared:                 -0.019
Method:                                                            Least Squares   F-statistic:                   0.01062
Date:                                                           Sun, 03 Nov 2019   Prob (F-statistic):              0.918
Time:                                                                   14:14:02   Log-Likelihood:                -708.29
No. Observations:                                                             53   AIC:                             1421.
Df Residuals:                                                                 51   BIC:                             1425.
Df Model:                                                                      1                                         
Covariance Type:                                                       nonrobust                                         
============================================================================================================
                                               coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------------------------------------
Intercept                                 1.614e+06   1.66e+05      9.738      0.000    1.28e+06    1.95e+06
data1[data1.status == 'UHR-NC'].true_age   .0166   7200.763     -0.103      0.918   -1.52e+04    1.37e+04
==============================================================================
Omnibus:                        0.964   Durbin-Watson:                   1.945
Prob(Omnibus):                  0.618   Jarque-Bera (JB):                0.966
Skew:                           0.300   Prob(JB):                        0.617
Kurtosis:                       2.723   Cond. No.                         177.
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
"""


fit3 = sm.ols("data1[data1.status == 'UHR-C'].EstimatedTotalIntraCranialVol ~ data1[data1.status == 'UHR-C'].true_age", data=data1[data1.status == 'UHR-C']).fit()
fit3.summary()

"""
                                                 OLS Regression Results                                                 
========================================================================================================================
Dep. Variable:     data1[data1.status == 'UHR-C'].EstimatedTotalIntraCranialVol   R-squared:                       0.007
Model:                                                                      OLS   Adj. R-squared:                 -0.032
Method:                                                           Least Squares   F-statistic:                    0.1818
Date:                                                          Sun, 03 Nov 2019   Prob (F-statistic):              0.674
Time:                                                                  16:27:26   Log-Likelihood:                -368.58
No. Observations:                                                            27   AIC:                             741.2
Df Residuals:                                                                25   BIC:                             743.8
Df Model:                                                                     1                                         
Covariance Type:                                                      nonrobust                                         
===========================================================================================================
                                              coef    std err          t      P>|t|      [0.025      0.975]
-----------------------------------------------------------------------------------------------------------
Intercept                                1.781e+06   4.56e+05      3.905      0.001    8.42e+05    2.72e+06
data1[data1.status == 'UHR-C'].true_age -9331.0413   2.19e+04     -0.426      0.674   -5.44e+04    3.57e+04
==============================================================================
Omnibus:                        1.218   Durbin-Watson:                   1.451
Prob(Omnibus):                  0.544   Jarque-Bera (JB):                0.950
Skew:                           0.179   Prob(JB):                        0.622
Kurtosis:                       2.154   Cond. No.                         232.
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
"""
              
              
data1 = data1.rename(columns = {'EstimatedTotalIntraCranialVol':'Total Intra-Cranial Volume (mm³)','true_age':'Real Age'})
sns.lmplot(x='Real Age',y='Total Intra-Cranial Volume (mm³)',hue='status',hue_order=['UHR-NC','UHR-C'], data=data1, size=7)
ax = plt.gca()
ax.set_title("Total Intra-cranial Volume variation as a function of \n age in converters and non-converters")

# Is TIV associated with conversion to psychosis ? => NO !

mannwhitneyu(data1[data1.status == 'UHR-C'].EstimatedTotalIntraCranialVol, data1[data1.status == 'UHR-NC'].EstimatedTotalIntraCranialVol)
# MannwhitneyuResult(statistic=684.0, pvalue=0.3762211813592916)


###############################################################################

# !!! à corriger en utilisant Brain Parenchymal Fraction

# PLOT DISTRIBUTION OF BRAIN VOLUME NOT EXPLAINED BY CANNABIS AND ANTIPSYCHOTICS !!!
# WITH RESPECT TO AGE IN CONVERTERS AND NON-CONVERTERS

resid = pd.DataFrame(X_test_uhr)
resid.columns = list(data_r_uhr_test)[48:140]
age = pd.DataFrame(age_test_uhr, columns=['age'])
participant_id = pd.DataFrame(participant_id, columns=['participant_id'])
status = pd.DataFrame(status_test_uhr, columns=['status'])
timepoint = pd.DataFrame(irm_test_uhr, columns=['timepoint'])
depression = np.array(data_r_uhr_test.MADRS)
depression = pd.DataFrame(depression, columns=['depression'])
function = np.array(data_r_uhr_test.SOFAS)
function = pd.DataFrame(function, columns=['functional_score'])
predicted_age = pd.DataFrame(age_test_uhr_predicted, columns=['predicted_age'])

RESID = pd.concat([participant_id, age, status, timepoint, depression, function, predicted_age, resid], axis=1)
RESID = RESID[RESID.timepoint == 'M0']


# correlation of brain predicted_age with depression (MADRS) 
spearmanr(RESID.depression, RESID.predicted_age)
# correlation of brain predicted_age gap with depression (MADRS) 
RESID['age_gap'] = RESID.predicted_age - RESID.age
spearmanr(RESID.depression, RESID.age_gap)
# SpearmanrResult(correlation=-0.12706854463398606, pvalue=0.2613490313595849)

# correlation of brain predicted_age with function (SOFAS)
func = RESID[['functional_score', 'predicted_age', 'age_gap']]
func = func.dropna()
spearmanr(func.functional_score, func.predicted_age)
#SpearmanrResult(correlation=0.12636399902312143, pvalue=0.26712791637022587)
spearmanr(func.functional_score, func.age_gap)
# SpearmanrResult(correlation=0.04782993573866454, pvalue=0.675519183855146)






#########

fit1 = sm.ols("EstimatedTotalIntraCranialVol ~ age", data=RESID[RESID.status == 'UHR-NC']).fit()
fit1.summary()

fit2 = sm.ols("EstimatedTotalIntraCranialVol ~ age", data=RESID[RESID.status == 'UHR-C']).fit()
fit2.summary()

fit3 = sm.ols("EstimatedTotalIntraCranialVol ~ age", data=RESID).fit()
fit3.summary() 

# no significant difference


sns.lmplot(x='age',y='EstimatedTotalIntraCranialVol',hue='status',hue_order=['UHR-NC','UHR-C'], data=RESID, size=7)
ax = plt.gca()
ax.set_title("Total Intra-cranial Volume variation as a function of \n age in converters and non-converters")


# WITH MEAN THICKNESS

RESID['TotMeanThick'] = (RESID['rh_MeanThickness'] + RESID['lh_MeanThickness']) / 2

fit1 = sm.ols("TotMeanThick ~ age", data=RESID[RESID.status == 'UHR-NC']).fit()
fit1.summary()

fit2 = sm.ols("TotMeanThick ~ age", data=RESID[RESID.status == 'UHR-C']).fit()
fit2.summary()

fit3 = sm.ols("TotMeanThick ~ age", data=RESID).fit()
fit3.summary()

###############################################################################




###############################################################################

# PLOT DISTRIBUTION OF MEAN THICKNESS ACCORDING TO AGE IN CONVERTERS AND NON-CONVERTERS

# data_r_uhr_test

data2 = data_r_uhr_test
data2['TotMeanThick'] = (data2['rh_MeanThickness'] + data2['lh_MeanThickness']) / 2
data2 = data2[['participant_id','diagnosis','TotMeanThick','true_age','irm']]
data2 = data2[data2.irm == 'M0']
data2 = data2.rename(columns={'diagnosis':'status'})

meanT_NC = np.array(data2[data2.status == 'UHR-NC'].TotMeanThick)
age_NC = np.array(data2[data2.status == 'UHR-NC'].true_age)
pearsonr(meanT_NC, age_NC) # (0.05904263056694328, 0.6745190766098611)

meanT_C = np.array(data2[data2.status == 'UHR-C'].TotMeanThick)
age_C = np.array(data2[data2.status == 'UHR-C'].true_age)
pearsonr(meanT_C, age_C) # (-0.3253074977545321, 0.09777249410154615)


fit2 = sm.ols("data2[data2.status == 'UHR-NC'].TotMeanThick ~ data2[data2.status == 'UHR-NC'].true_age", data=data2[data2.status == 'UHR-NC']).fit()
fit2.summary()
"""
                                         OLS Regression Results                                         
========================================================================================================
Dep. Variable:     data2[data2.status == 'UHR-NC'].TotMeanThick   R-squared:                       0.003
Model:                                                      OLS   Adj. R-squared:                 -0.016
Method:                                           Least Squares   F-statistic:                    0.1784
Date:                                          Fri, 15 Nov 2019   Prob (F-statistic):              0.675
Time:                                                  15:11:09   Log-Likelihood:                 78.889
No. Observations:                                            53   AIC:                            -153.8
Df Residuals:                                                51   BIC:                            -149.8
Df Model:                                                     1                                         
Covariance Type:                                      nonrobust                                         
============================================================================================================
                                               coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------------------------------------
Intercept                                    2.5588      0.059     43.530      0.000       2.441       2.677
data2[data2.status == 'UHR-NC'].true_age     0.0011      0.003      0.422      0.675      -0.004       0.006
==============================================================================
Omnibus:                        0.629   Durbin-Watson:                   2.230
Prob(Omnibus):                  0.730   Jarque-Bera (JB):                0.133
Skew:                          -0.008   Prob(JB):                        0.936
Kurtosis:                       3.245   Cond. No.                         177.
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
"""

fit3 = sm.ols("data2[data2.status == 'UHR-C'].TotMeanThick ~ data2[data2.status == 'UHR-C'].true_age", data=data2[data2.status == 'UHR-C']).fit()
fit3.summary()

"""
                                         OLS Regression Results                                        
=======================================================================================================
Dep. Variable:     data2[data2.status == 'UHR-C'].TotMeanThick   R-squared:                       0.106
Model:                                                     OLS   Adj. R-squared:                  0.070
Method:                                          Least Squares   F-statistic:                     2.959
Date:                                         Fri, 15 Nov 2019   Prob (F-statistic):             0.0978
Time:                                                 15:12:57   Log-Likelihood:                 28.684
No. Observations:                                           27   AIC:                            -53.37
Df Residuals:                                               25   BIC:                            -50.78
Df Model:                                                    1                                         
Covariance Type:                                     nonrobust                                         
===========================================================================================================
                                              coef    std err          t      P>|t|      [0.025      0.975]
-----------------------------------------------------------------------------------------------------------
Intercept                                   2.9280      0.186     15.760      0.000       2.545       3.311
data2[data2.status == 'UHR-C'].true_age    -0.0153      0.009     -1.720      0.098      -0.034       0.003
==============================================================================
Omnibus:                        0.369   Durbin-Watson:                   2.092
Prob(Omnibus):                  0.832   Jarque-Bera (JB):                0.075
Skew:                          -0.129   Prob(JB):                        0.963
Kurtosis:                       2.984   Cond. No.                         232.
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
"""

data2 = data2.rename(columns = {'TotMeanThick':'Total mean thickness (cm)','true_age':'Real Age'})
sns.lmplot(x='Real Age',y='Total mean thickness (cm)',hue='status',hue_order=['UHR-NC','UHR-C'], data=data2, size=7)
ax = plt.gca()
ax.set_title("Total mean thickness variation as a function of \n age in converters and non-converters")

# Is TIV associated with conversion to psychosis ? => NO !

mannwhitneyu(data2[data2.status == 'UHR-C']['Total mean thickness (cm)'], data2[data2.status == 'UHR-NC']['Total mean thickness (cm)'])
# MannwhitneyuResult(statistic=591.0, pvalue=0.10353157916534111)

###############################################################################

# PLOT DISTRIBUTION OF MEAN SURFACE AREA ACCORDING TO AGE IN CONVERTERS VS NON-CONVERTERS

list(FS_surface)
pheno = pheno[pheno.age <= 30]
surf = pd.merge(pheno, FS_surface, on='participant_id', how='left')
list(surf)

surf = surf[(surf.diagnosis.isin(['UHR-C', 'UHR-NC']))]
list(surf)
del surf['lh_WhiteSurfArea']
del surf['rh_WhiteSurfArea']
assert list(surf).index('lh_bankssts') == 48
assert list(surf).index('rh_insula') == 115
surf['TotSurfArea']= surf.iloc[:, 48:116].sum(axis=1)
surf = surf[surf.irm == 'M0']
surf = surf.rename(columns = {'true_age':'Real Age', 'diagnosis':'status', 'TotSurfArea':'Total surface area'})

sns.lmplot(x='Real Age',y='Total surface area',hue='status',hue_order=['UHR-NC','UHR-C'], data=surf, size=7)
ax = plt.gca()
ax.set_title("Total surface area as a function of \n age in converters and non-converters")

mannwhitneyu(surf[surf.status == 'UHR-C']['Total surface area'], surf[surf.status == 'UHR-NC']['Total surface area'])
# MannwhitneyuResult(statistic=630.0, pvalue=0.19355565487627813)


###############################################################################
###############################################################################


# Is there a correlation between BRAIN AGE GAP at M0 and EPIGENETIC AGE ACCELERATION ?

# import methylation data
meth_path='/home/anton/Dropbox/PhD/Article_Age_epigenetique_et_cerebral/2019_Methylation_age/analysis_clock_horvath'
methyl = pd.read_csv(os.path.join(meth_path, 'results_methylation.csv'), sep='\t')
methyl.columns = ['code_icaar_start','Transition','Age_M0','Age_MF','Sexe','DNAmAge_M0','DNAmAge_MF', 'delta_time', 'delta_DNAmTime','mDNA_ageing_speed']
corresp = pd.read_csv('/home/anton/Dropbox/PhD/Article_Age_epigenetique_et_cerebral/2019_Brain_age/Analysis_pooled_ROIs_pooled_phenotypes/phenotype/clinic_icaar_201907.tsv',sep='\t')
corresp = corresp[['participant_id','code_icaar_start']]
methyl1 = pd.merge(corresp, methyl, how='inner', on='code_icaar_start')

# imagery data
imag = data_final

# merged data
imag_methyl = pd.merge(methyl1, imag, how='inner', on='participant_id')
imag_methyl['mDNA_age_gap'] = methyl.DNAmAge_M0 - methyl.Age_M0


# correlation between methylation age acceleration and baseline brain age gap ? => NO
mannwhitneyu(imag_methyl.age_diff, imag_methyl.mDNA_ageing_speed) # MannwhitneyuResult(statistic=55.0, pvalue=0.3713329514598411)

# correlation between baseline methylation age gap and baseline brain age gap ? => YES
mannwhitneyu(imag_methyl.age_diff, imag_methyl.mDNA_age_gap) # MannwhitneyuResult(statistic=27.0, pvalue=0.015119510508074318)

dims = (6,6)
fig, ax = pyplot.subplots(figsize=dims)
sns.scatterplot(x='age_diff',y='mDNA_age_gap', hue='clinical status', hue_order=['UHR-NC','UHR-C'], data=imag_methyl, ax=ax)




###############################################################################
###############################################################################

# ANALYSE LONGITUDINALE

brain_age = pd.DataFrame(age_test_uhr_predicted)
real_age = pd.DataFrame(age_test_uhr)
status = pd.DataFrame(status_test_uhr)
participant_id = pd.DataFrame(participant_id)
timepoint = pd.DataFrame(irm_test_uhr)
data_final = pd.concat([participant_id,real_age, brain_age, status, timepoint], axis=1)
data_final.columns = ['participant_id','real_age', 'predicted_age', 'status', 'timepoint']
corresp = pd.read_csv('/home/anton/Dropbox/PhD/Article_Age_epigenetique_et_cerebral/corresp_ID_icaar_start')
data = pd.merge(corresp, data_final, on='participant_id', how='inner')

data = pd.merge(pheno, all_feat, on='participant_id', how='left')
data_M0 = data[data.timepoint == 'M0']
data_MF = data[data.timepoint == 'MF']
longitudinal = pd.merge(data_M0, data_MF, how='inner', on='code_icaar_start', suffixes=('_M0','_MF'))
list(longitudinal)
del longitudinal['Unnamed: 0_M0']
del longitudinal['Unnamed: 0_MF']

longitudinal['delta_time'] = longitudinal['real_age_MF'] - longitudinal['real_age_M0']
longitudinal['delta_brainTime'] = longitudinal['predicted_age_MF'] - longitudinal['predicted_age_M0']
longitudinal['ageing_speed'] = longitudinal.delta_brainTime / longitudinal.delta_time


longitudinal.to_csv('/home/anton/Dropbox/PhD/Article_Age_epigenetique_et_cerebral/longitudinal_new.csv', sep='\t', index=False)


age_acceleration_C = longitudinal[longitudinal.status_M0 == 'UHR-C'].ageing_speed
age_acceleration_NC = longitudinal[longitudinal.status_M0 == 'UHR-NC'].ageing_speed

age_acceleration_C.mean() # -0.24030091740079687
age_acceleration_NC.mean() # 0.26301035880544876


# rejecting outliers
    
def reject_outliers(x, mad):
    return x[abs(x - x.median()) < 3 * mad]

mad_age_acceleration_C = 1.4826 * np.median(np.abs(age_acceleration_C - age_acceleration_C.median()))
mad_age_acceleration_NC = 1.4826 * np.median(np.abs(age_acceleration_NC - age_acceleration_NC.median()))


age_acceleration_C_no_out = reject_outliers(age_acceleration_C, mad_age_acceleration_C)
age_acceleration_NC_no_out = reject_outliers(age_acceleration_NC, mad_age_acceleration_NC)

age_acceleration_C_no_out.mean() # -0.45385839456796234
age_acceleration_NC_no_out.mean() # 0.26301035880544876


############# BOOTSTRAPPING POUR VOIR LA VARIATION DE LA DIFFERENCE D'AGE (calcul de l'intervalle de confiance) #####
    
from scipy.stats import mannwhitneyu

# pour les non-converteurs
X_NC = age_acceleration_NC
nboot = 1000
variation_median_NC = []
for boot in range(nboot):
    boot = np.random.choice(X_NC, size=len(X_NC), replace=True)
    variation_median_NC += [np.median(boot)]

# pour les converteurs  
X_C = age_acceleration_C
nboot = 1000
variation_median_C = []
for boot in range(nboot):
    boot = np.random.choice(X_C, size=len(X_C), replace=True)
    variation_median_C += [np.median(boot)]
    
variation_median_NC = pd.DataFrame(variation_median_NC)
variation_median_C = pd.DataFrame(variation_median_C)

# graphical representation

data_median = pd.concat([variation_median_NC, variation_median_C], axis=1)
data_median.columns = ['Non-Converters', 'Converters'] 

dims = (8, 12)
fig, ax = plt.subplots(figsize=dims)
sns.set(style="whitegrid")
ax.set_title("Brain ageing of Non-Converters and Converters")
plt.ylabel('Age acceleration')
ax = sns.boxplot(data=data_median)
add_stat_annotation(ax, data=data_median,
                    box_pairs=[("Non-Converters", "Converters")],
                    test='Mann-Whitney', text_format='full', loc='inside', verbose=2)


############# PERMUTATION TEST (calcul de p non-paramétrique) POUR C vs NC

    
C = age_acceleration_C
NC = age_acceleration_NC
C_array = np.array(C)
NC_array = np.array(NC)
assert len(C_array) == 4
assert len(NC_array) == 10
C_array = C_array[:,np.newaxis]
NC_array = NC_array[:,np.newaxis]

T_array = np.ones(4)
NT_array = np.zeros(10)
T_array = T_array[:,np.newaxis]
NT_array = NT_array[:,np.newaxis]
array_T = np.c_[C_array,T_array]
array_NT = np.c_[NC_array,NT_array]
data_array = np.r_[array_T,array_NT]
columns = ['brain_ageing','conversion']
df_brain_ageing = pd.DataFrame(data_array, columns=columns)
df_brain_ageing.conversion = df_brain_ageing.conversion.map({1.0:'yes', 0.0:'no'})

# Permutation: simulate the null hypothesis

nperm = 10000
perms = np.zeros(nperm + 1)
p = np.zeros(nperm + 1)
perms[0], p[0] = mannwhitneyu(df_brain_ageing.brain_ageing[df_brain_ageing.conversion == 'yes'],df_brain_ageing.brain_ageing[df_brain_ageing.conversion == 'no'])

# je veux permuter aléatoirement l'assignation à chaque groupe puis faire le wilcoxon
    # je permute le statut converteur
    # je les mets dans un nouveau tableau
    # je fais le mann whitney
    # je stocke la statistique U et p
    
for i in range(1,nperm):
    brain_ageing = df_brain_ageing.brain_ageing.values
    brain_ageing = brain_ageing[:,np.newaxis]
    perm = np.random.permutation(df_brain_ageing.conversion)
    perm = perm[:,np.newaxis]
    perm_assign = np.c_[brain_ageing,perm]
    df_permuted = pd.DataFrame(perm_assign, columns=columns)
    perms[i], p[i] = mannwhitneyu(df_permuted.brain_ageing[df_permuted.conversion == 'yes'],df_permuted.brain_ageing[df_permuted.conversion == 'no'])

# Two-tailed empirical p-value
pval_perm = np.sum(p <= p[0])/p.shape[0]
pval_perm # 0.24337566243375663

# Plot
from matplotlib.pyplot import figure
# Re-weight to obtain distribution
figure(num=None, figsize=(8,6), dpi=80, facecolor='w', edgecolor='k')
weights = np.ones(perms.shape[0]) / perms.shape[0]
plt.hist([perms[perms >= perms[0]], perms], histtype='stepfilled', bins=100, label=["U > U obs", "U < U obs (p-value)"], weights = [weights[perms >= perms[0]], weights])
plt.xlabel("Statistic distribution under the null hypothesis that there is no \n difference in age acceleration between converters and non-converters")
plt.title("Random permutations analysis (10 000 permutations)")
plt.axvline(x=perms[0], color='red', linewidth=1, label='observed statistic: p = 0.24')
_ = plt.legend(loc='upper left')


###############################################################################
###############################################################################

# 3.5.2 Correlation between baseline brain age gap and accelerated epigenetic ageing

# Is there a correlation between BRAIN AGE GAP at M0 and EPIGENETIC AGE ACCELERATION ?

# import methylation data
meth_path='/home/anton/Dropbox/PhD/Article_Age_epigenetique_et_cerebral/2019_Methylation_age/analysis_clock_horvath'
methyl = pd.read_csv(os.path.join(meth_path, 'results_methylation.csv'), sep=',')
list(methyl)
methyl.columns = ['code_icaar_start','Transition','Age_M0','Age_MF','Sexe','DNAmAge_M0','DNAmAge_MF', 'delta_time', 'delta_DNAmTime','mDNA_ageing_speed','Eq_Chlorpromazine']
corresp = pd.read_csv('/home/anton/Dropbox/PhD/Article_Age_epigenetique_et_cerebral/2019_Brain_age/Analysis_pooled_ROIs_pooled_phenotypes/phenotype/clinic_icaar_201907.tsv',sep='\t')
corresp = corresp[['participant_id','code_icaar_start']]
methyl1 = pd.merge(corresp, methyl, how='inner', on='code_icaar_start')

# imagery data
imag = data

# merged data
imag_methyl = pd.merge(methyl1, imag, how='inner', on='participant_id')
imag_methyl['mDNA_age_gap'] = methyl.DNAmAge_M0 - methyl.Age_M0
list(imag_methyl)

imag_methyl.brain_age_gap = imag_methyl['predicted_age'] - imag_methyl['real_age']

# correlation between methylation age acceleration and baseline brain age gap ? => NO
spearmanr(imag_methyl.brain_age_gap, imag_methyl.mDNA_ageing_speed) # SpearmanrResult(correlation=0.30000000000000004, pvalue=0.37008312228206786)

# correlation between baseline methylation age gap and baseline brain age gap ? => NO
spearmanr(imag_methyl.brain_age_gap, imag_methyl.mDNA_age_gap) #  SpearmanrResult(correlation=-0.11818181818181818, pvalue=0.72928477951978)

dims = (6,6)
fig, ax = pyplot.subplots(figsize=dims)
sns.scatterplot(x='age_diff',y='mDNA_age_gap', hue='clinical status', hue_order=['UHR-NC','UHR-C'], data=imag_methyl, ax=ax)

###############################################################################













