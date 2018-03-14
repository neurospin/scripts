# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 01:03:34 2014

@author: edouard
"""

import os.path
import sys
import numpy as np
import pylab as plt
import pandas as pd
from sklearn.metrics import roc_auc_score #roc_curve, auc,
from parsimony.estimators import ElasticNetLogisticRegression
from parsimony.utils.penalties import l1_max_logistic_loss

from scipy.stats import binom_test

#from sklearn import cross_validation
from sklearn.metrics import precision_recall_fscore_support


WD = "/home/ed203246/data/2013_predict-transition-caarms"
SRC = "/home/ed203246/git/scripts/2013_predict-transition-caarms"
OUTPUT = os.path.join(WD, "201701")
if not os.path.exists(OUTPUT):
    os.makedirs(OUTPUT)

#sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../lib'))
sys.path.append(SRC)
import utils

############################################################################
## PARAMETERS
############################################################################

#MODE = "cv"
#MODE = "permutations"

DATASET = "full"
#DATASET = "reduced"

############################################################################
## Dataset: CARMS ONLY
############################################################################
Xd, yd, df = utils.read_Xy(WD=WD)
Xd = Xd.drop(['PAS2gr', 'CB_EXPO'], 1)


#
df.AGE.describe()
"""
count    27.000000
mean     21.629630
std       3.701851
min      16.000000
25%      19.000000
50%      21.000000
75%      23.500000
max      30.000000
"""

df.REEVAL

s = "M15 MK"
df.REEVAL = [int(s.replace(" MK", "").strip().replace("M", "")) for s in df.REEVAL]

df.REEVAL.describe()
"""
count    27.000000
mean     22.370370
std       6.546568
min      12.000000
25%      18.000000
50%      22.000000
75%      27.000000
max      33.000000
"""

# Add intercept
inter= pd.DataFrame([1.]*Xd.shape[0], columns=["inter"])
Xd = pd.concat((inter, Xd), axis=1)

if DATASET == "full":
    ALPHA, L1_PROP = 1 / np.log2(Xd.shape[1]), 0.8

if DATASET == "reduced":
    keep = ["inter",
            "@4.3", # Anhedonia
            "@5.4", # Aggression/dangerous behavior
            "@7.4", # Mood swings/lability
            "@7.6"] # Obsessive compulsive symptoms
    #keep = ["@4.3", "@5.4", "@7.4", "@7.6"]
    Xd = Xd[keep]
    ALPHA, L1_PROP = 1 / np.log2(Xd.shape[1]), 0.0

print(ALPHA, L1_PROP)

"""
Xd2 = Xd.copy()
Xd2["TRANSITION"] = yd
Xd2.to_csv(WD+"/data/transitionPREP_ARonly_CAARMSonly.csv", index=False)
"""

X = np.asarray(Xd)
y = np.asarray(yd)
y = y.astype(float)[:, np.newaxis]
#A, n_compacts = A_empty(X.shape[1]-1)
l1_max_logistic_loss(X, y)

assert X.shape == (27, 26)

############################################################################
## FIXED PARAMETERS
############################################################################

#N_FOLDS = 10
N_PERMS = 10001
NBOOT = 1000
ALPHAS = [.01, 0.05, .1, 1/ np.log2(X.shape[1]), .5, 1, 10, 100]
    #L1_PROP = np.arange(0, 1.1, 0.1)
L1_PROPS = np.arange(0, 1.1, 0.1)

def scores(y_true, y_pred, prob_pred):
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average=None)
    try:
        auc = roc_auc_score(y_true, prob_pred) #area under curve score.
    except :
        auc = np.NaN
    success = r * s
    success = success.astype('int')
    #accuracy = (r[0] * s[0] + r[1] * s[1])
    #accuracy = accuracy.astype('int')
    prob_class1 = np.count_nonzero(y_true) / float(len(y_true))
    pval_r0 = binom_test(success[0], s[0], 1 - prob_class1)
    pval_r1 = binom_test(success[1], s[1], prob_class1)
    acc = success.sum() / s.sum()
    pval_acc = binom_test(success.sum(), s.sum(), p=0.5)
    bacc = r.mean()
    pval_bacc = binom_test(np.round(bacc * s.sum()), s.sum(), p=0.5)
    return bacc, pval_bacc, auc, r[0], pval_r0, r[1], pval_r1, acc, pval_acc

SCORE_COLUMNS = ["bacc", "pval_bacc", "auc", "r0", "pval_r0", "r1", "pval_r1", "acc", "pval_acc"]

############################################################################
## Model selection on 10 CV
############################################################################

RES = {alpha:{l1_ratio:dict(y_pred=[], y_true=[], prob_pred=[]) for l1_ratio in L1_PROPS} for alpha in ALPHAS}
for fold, (train, test) in enumerate(utils.CV10):
    print("fold",fold)
    Xtr = X[train, :]
    Xte = X[test, :]
    ytr = y[train, :]
    yte = y[test, :]
    for alpha in ALPHAS:
        for l1_prop in L1_PROPS:
            k, l, g = alpha * np.array([1-l1_prop, l1_prop, 0])
            mod = ElasticNetLogisticRegression(l1_prop, alpha, penalty_start=1,
                                               class_weight="auto")
            mod.fit(Xtr, ytr)
            RES[alpha][l1_prop]["y_pred"].append(mod.predict(Xte).ravel())
            RES[alpha][l1_prop]["prob_pred"].append(mod.predict_probability(Xte).ravel())
            RES[alpha][l1_prop]["y_true"].append(yte.ravel())


scores_l = list()
for alpha in ALPHAS:
    for l1_prop in L1_PROPS:
        y_pred = np.concatenate(RES[alpha][l1_prop]["y_pred"])
        prob_pred = np.concatenate(RES[alpha][l1_prop]["prob_pred"])
        y_true = np.concatenate(RES[alpha][l1_prop]["y_true"])
        scores_l.append([alpha, l1_prop] + list(scores(y_true, y_pred, prob_pred)))


SCORES_CVGRID = pd.DataFrame(scores_l, columns=["alpha", "l1_prop"] + SCORE_COLUMNS)
#SCORES_CVGRID.to_csv(os.path.join(OUTPUT, "enet_param-selection_10cv.csv"), index=False)
SCORES_CVGRID.to_csv(os.path.join(OUTPUT, "enet_param-selection_10cv__20170926.csv"), index=False)


############################################################################
## 10 CV with parameters scores + AUC and ROC
############################################################################

y_pred = list()
y_true = list()
y_prob_pred = list()
Coefs = list()

for fold, (train, test) in enumerate(utils.CV10):
    Xtr = X[train, :]
    Xte = X[test, :]
    ytr = y[train, :]
    yte = y[test, :]
    print("fold",fold)
    mod = ElasticNetLogisticRegression(l=L1_PROP, alpha=ALPHA, penalty_start=1,
                                       class_weight="auto")
    mod.fit(Xtr, ytr)
    y_pred.append(mod.predict(Xte).ravel())
    y_prob_pred.append(mod.predict_probability(Xte).ravel())
    y_true.append(yte.ravel())
    Coefs.append(mod.beta.copy().ravel())
    #print mod.predict(Xte).ravel() == yte.ravel()
    #if fold == 4: break
y_pred = np.concatenate(y_pred)
y_true = np.concatenate(y_true)
prob_pred = np.concatenate(y_prob_pred)

SCORES_CV = pd.DataFrame([scores(y_true, y_pred, prob_pred)], columns=SCORE_COLUMNS)
SCORES_CV = SCORES_CV.append(np.round(SCORES_CV * 100, 2))
print(SCORES_CV)

CoefsCv = pd.DataFrame(Coefs, columns=Xd.columns)
COEFS_CV = CoefsCv.describe(percentiles=[.99, .95, .9, .5, .1, .05, 0.01])
COEFS_CV = COEFS_CV.T

COEFS_CV.ix[:, "count"] = np.sum(CoefsCv != 0, axis=0)
COEFS_CV_SELECTED = COEFS_CV.ix[COEFS_CV.ix[:, "count"]  != 0, :]

print(COEFS_CV_SELECTED.round(2))
"""
# FULL

       bacc  pval_bacc       auc      r0   pval_r0        r1   pval_r1  \
0  0.877841   0.000049  0.835227  0.9375  0.003928  0.818182  0.010009

        acc  pval_acc
0  0.888889  0.000049

        count  mean   std   min    5%   10%   50%   90%   95%   max
inter     10 -0.02  0.20 -0.25 -0.22 -0.18 -0.06  0.09  0.30  0.50
@1.2       4  0.01  0.01  0.00  0.00  0.00  0.00  0.02  0.03  0.03
@1.3       1  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.01
@4.3      10 -0.20  0.05 -0.30 -0.28 -0.26 -0.20 -0.15 -0.14 -0.12
@5.4      10  0.30  0.08  0.16  0.20  0.24  0.29  0.40  0.43  0.46
@7.4      10 -0.26  0.05 -0.35 -0.33 -0.31 -0.26 -0.21 -0.20 -0.20
@7.6      10  0.16  0.09  0.04  0.05  0.06  0.15  0.26  0.27  0.28
@7.7       8  0.06  0.05  0.00  0.00  0.00  0.06  0.13  0.14  0.14

"@4.3" # Anhedonia
"@5.4" # Aggression/dangerous behavior
"@7.4" # Mood swings/lability
"@7.6' # Obsessive compulsive symptoms
"@7.7" # symptomes dissociatifs

# Reduced
       bacc  pval_bacc     auc      r0   pval_r0        r1   pval_r1  \
0  0.877841   0.000049  0.9375  0.9375  0.003928  0.818182  0.010009

        acc  pval_acc
0  0.888889  0.000049
       count  mean   std   min    5%   10%   50%   90%   95%   max
inter     10  0.30  0.12  0.06  0.13  0.19  0.30  0.39  0.47  0.54
@4.3      10 -0.31  0.03 -0.35 -0.35 -0.35 -0.31 -0.28 -0.27 -0.27
@5.4      10  0.37  0.04  0.31  0.32  0.33  0.36  0.41  0.43  0.45
@7.4      10 -0.36  0.02 -0.41 -0.40 -0.39 -0.36 -0.34 -0.34 -0.33
@7.6      10  0.26  0.04  0.21  0.21  0.21  0.25  0.32  0.32  0.32

"""
############################################################################
## Bootstraping CV
############################################################################
np.random.seed(42)

mod = ElasticNetLogisticRegression(l=L1_PROP, alpha=ALPHA, penalty_start=1,
                               class_weight="auto")
scores_l = list()
Coefs = list()

smpl_all = np.arange(X.shape[0])
for boot_i in range(NBOOT):
    print("** Boot", boot_i)
    boot_tr = np.random.choice(smpl_all, size=len(smpl_all), replace=True)
    boot_te = np.setdiff1d(smpl_all, boot_tr, assume_unique=False)
    Xtr = X[boot_tr, :]
    Xte = X[boot_te, :]
    ytr = y[boot_tr, :]
    yte = y[boot_te, :]
    mod.fit(Xtr, ytr)
    y_pred=mod.predict(Xte).ravel()
    prob_pred=mod.predict_probability(Xte).ravel()
    p, r, f, s = precision_recall_fscore_support(yte, y_pred, average=None)
    success = r * s
    try:
        auc = roc_auc_score(yte, prob_pred) #area under curve score.
    except :
        auc = None
    scores_l.append([r.mean(), success.sum() / s.sum(), auc] + r.tolist())
    Coefs.append(mod.beta.copy().ravel())

CoefsBoot = pd.DataFrame(Coefs, columns=Xd.columns)
COEFS_BOOT = CoefsBoot.describe(percentiles=[.99, .95, .9, .5, .1, .05, 0.01])
COEFS_BOOT = COEFS_BOOT.T

COEFS_BOOT.ix[:, "count"] = np.sum(CoefsBoot != 0, axis=0)
COEFS_BOOT_SELECTED = COEFS_BOOT.ix[COEFS_BOOT.ix[:, "count"]  > 500, :]
#COEFS_BOOT[['@4.3', '@5.4', '@7.4', '@7.6', '@7.7']].T

############################################################################
## Permutations + 10 CV
############################################################################

N_PERMS = 1000
mod = ElasticNetLogisticRegression(l=L1_PROP, alpha=ALPHA, penalty_start=1,
                               class_weight="auto")

scores_p = list()
coefs_count = list()
coefs_mean = list()
coefs_std = list()
for perm_i in range(N_PERMS):
    print("** Perm", perm_i)
    if perm_i == 0:
        yp = y
    else:
        yp = y[np.random.permutation(y.shape[0])]
    #cv = cross_validation.StratifiedKFold(y=yp.ravel(), n_folds=N_FOLDS)
    cv = utils.CV10
    y_pred = list()
    y_true = list()
    Coefs = list()
    y_prob_pred = list()
    for fold, (train, test) in enumerate(cv):
        print("fold", fold, len(test))
        Xtr = X[train, :]
        Xte = X[test, :]
        ytr = yp[train, :]
        yte = yp[test, :]
        mod.fit(Xtr, ytr)
        y_pred.append(mod.predict(Xte).ravel())
        y_true.append(yte.ravel())
        y_prob_pred.append(mod.predict_probability(Xte).ravel())
        Coefs.append(mod.beta.copy().ravel())
    y_pred = np.concatenate(y_pred)
    y_true = np.concatenate(y_true)
    y_prob_pred = np.concatenate(y_prob_pred)
    Coefs = np.r_[Coefs]
    #p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average=None)
    #fpr, tpr, thresholds = roc_curve(y_true, y_prob_pred)
    #roc_auc = auc(fpr, tpr)
    #scores.append(np.r_[p, p.mean(), r, r.mean(), f, f.mean(), s, roc_auc])
    scores_p.append(scores(y_true, y_pred, prob_pred))
    coefs_count.append(np.sum(Coefs != 0, axis=0))
    coefs_mean.append(np.mean(Coefs, axis=0))
    coefs_std.append(np.std(Coefs, axis=0))


coefs_count = np.r_[coefs_count]
coefs_mean = np.r_[coefs_mean]
coefs_std = np.r_[coefs_std]
# avoid division by 0
coefs_std[coefs_mean == 0] = 1.
coefs_count_pval = np.sum(coefs_count[1:, :] >= coefs_count[0, :],  axis=0) / float(coefs_count.shape[0] - 1)
coefs_mean_pval = np.sum(np.abs(coefs_mean[1:, :]) >= np.abs(coefs_mean[0, :]),  axis=0) / float(coefs_mean.shape[0] - 1)
coefs_z_pval = np.sum(
    np.abs(coefs_mean[1:, :] / coefs_std[1:, :]) >= \
    np.abs(coefs_mean[0, :] / coefs_std[0, :]),  axis=0) \
    / float(coefs_mean.shape[0] - 1)

scores_p = np.array(scores_p)#, columns=SCORE_COLUMNS)
scores_pval = np.sum(scores_p >= scores_p[0, :],  axis=0) / float(scores_p.shape[0])
#
nzero = coefs_count[0, ] !=0
COEFS_PERM = pd.DataFrame(dict(var=Xd.columns[nzero],
        count=coefs_count[0, nzero], count_pval=coefs_count_pval[nzero],
        mean=coefs_mean[0, nzero],    mean_pval=coefs_mean_pval[nzero],
        z=coefs_mean[0, nzero] / coefs_std[0, nzero],
        z_pval=coefs_z_pval[nzero]))
print(COEFS_PERM.to_string())


score_col_oi = ["bacc", "auc", "r0", "r1", "acc"]
idx = [np.where(s == np.array(SCORE_COLUMNS))[0][0] for s in ["bacc", "auc", "r0", "r1", "acc"]]
SCORES_PERM = pd.DataFrame([scores_pval[idx]], columns=score_col_oi)

SCORES_PERM.to_csv(os.path.join(OUTPUT, "enet_permuation%i_10cv_%s-dataset_coefs.csv" %\
    (N_PERMS, DATASET)), index=False)

############################################################################
## Save results
############################################################################

xls_filename = os.path.join(OUTPUT, "enet10cv_%s.xlsx" % DATASET)

with pd.ExcelWriter(xls_filename) as writer:
    SCORES_CVGRID.to_excel(writer, sheet_name='Scores CVGRID', index=False)
    SCORES_CV.to_excel(writer, sheet_name='Scores CV-alpha%.3f, l1%.2f' % (ALPHA, L1_PROP) , index=False)
    COEFS_CV.to_excel(writer, sheet_name='Coef CV-alpha%.3f, l1%.2f' % (ALPHA, L1_PROP))
    COEFS_CV_SELECTED.to_excel(writer, sheet_name='CoefNoNu CV-alpha%.3f, l1%.2f' % (ALPHA, L1_PROP))
    SCORES_PERM.to_excel(writer, sheet_name='Scores PERM-alpha%.3f, l1%.2f' % (ALPHA, L1_PROP), index=False)
    COEFS_PERM.to_excel(writer, sheet_name='Coef PERM-alpha%.3f, l1%.2f' % (ALPHA, L1_PROP))
    #SCORES_BOOT.to_excel(writer, sheet_name='Scores BOOT-alpha%.3f, l1%.2f' % (ALPHA, L1_PROP), index=False)
    COEFS_BOOT.to_excel(writer, sheet_name='Coef BOOT-alpha%.3f, l1%.2f' % (ALPHA, L1_PROP))


################################
## Plot coefs
################################
{'1.1': 'Unusual thought content',
 '1.2': 'non-bizarre ideas',
 '1.3': 'perceptual abnormalities',
 '1.4': 'Disorganized speech',
 '2.1': 'Subjective cognitive change',
 '2.2': 'Observed cognitive change',
 '3.1': 'Subjective emotional disturbance',
 '3.2': 'Observed blunter affect',
 '3.3': 'Observed inappropriate affect',
 '4.1': 'Alogia',
 '4.2': 'Avolition/apathy',
 '4.3': 'Anhedonia',
 '5.1': 'Social isolation',
 '5.2': 'Impaired role function',
 '5.3': 'Disorganizing/odd/stigmatising behaviour',
 '5.4': 'Aggression/dangerous behaviour',
 '6.1': 'Subjective complaints of impaired motor functioning',
 '6.2': 'Informant reported or observed changes in motor functioning',
 '6.3': 'Subjective complaints of impaired bodily sensation',
 '6.4': 'Subjective complaints of impaired autonomic functioning',
 '7.1': 'Mania',
 '7.2': 'Depression',
 '7.3': 'Suicidality and self harm',
 '7.4': 'Mood swings/lability',
 '7.5': 'Anxiety',
 '7.6': 'Obsessive compulsive symptoms',
 '7.7': 'Dissociative symptoms',
 '7.8': 'Impaired tolerance to normal stress'}

items = pd.read_excel(os.path.join(WD, "caarm_items_mapping.xlsx"))#, 0)

mapping = {str(row[1][0]):row[1][2] for row in items.iterrows()}

data = pd.read_excel(xls_filename, sheetname='Coef BOOT-alpha%.3f, l1%.2f' % (ALPHA, L1_PROP))
print(len(data.index), data.index)

data.columns
#Index(['count', 'mean', 'std', 'min', '1%', '5%', '10%', '50%', '90%', '95%',
#       '99%', 'max'],
#      dtype='object')

data = data[1:]
predictors = [s.replace('@', '') for s in data.index]
predictors = [mapping[p] for p in predictors]
assert len(predictors) == data.shape[0] == len(coef) == 25

coef = data['mean']
plt.clf()
#fig = plt.figure(figsize=(10, 15))
x_ = np.arange(len(coef))
#fig = plt.figure(figsize=(int(coef.shape[0]/3), 10))

plt.barh(x_, coef, facecolor='#9999ff',
        fc = 'lightgrey', lw=1,
        edgecolor='black', align='center',
        xerr = data['std'], ecolor='black', color='gray',
        label='Coefs Mean(Boot)')

plt.axvline(0, ls='-', lw=1, color="black")
#plt.tick_params(axis='x', labelsize=8)
plt.yticks(x_, predictors)#, rotation=90)
#plt.yticks(x_, x_)#, rotation=90)
plt.ylim(0-0.5, np.max(x_)+0.5)

plt.grid()
plt.tight_layout()
#plt.legend(loc='best')
#plt.title("Coefficients")

"""
RESULTS


Full dataset
============


10CV
----
Precisions:  [ 0.88235294  0.9       ] 0.891176470588
Recalls:     [ 0.9375      0.81818182] 0.877840909091
F:           [ 0.90909091  0.85714286] 0.883116883117
AUC:         0.835227272727
Support:     [16 11]


NPERM = 10000
-------------

   coef_count    pval    var
0          10  1.0000  inter
1           4  0.2613   @1.2
2          10  0.0415   @4.3
3          10  0.0705   @5.4
4          10  0.0954   @7.4
5          10  0.0757   @7.6
6           7  0.1559   @7.7
          score        val   p-val
0        prec_0   0.882353  0.0069
1        prec_1   0.900000  0.0008
2     prec_mean   0.891176  0.0009
3      recall_0   0.937500  0.0024
4      recall_1   0.818182  0.0336
5   recall_mean   0.877841  0.0008
6           f_0   0.909091  0.0008
7           f_1   0.857143  0.0008
8        f_mean   0.883117  0.0008
9     support_0  16.000000  1.0000
10     suppor_1  11.000000  1.0000
11          auc   0.835227  0.0065


NPERM = 2000
------------


   coef_count    pval    var
0          10  0.0000  inter
1           4  0.2195   @1.2
2          10  0.0000   @4.3
3          10  0.0000   @5.4
4          10  0.0000   @7.4
5          10  0.0000   @7.6
6           7  0.1360   @7.7

          score        val   p-val
0        prec_0   0.882353  0.0045
1        prec_1   0.900000  0.0000
2     prec_mean   0.891176  0.0000
3      recall_0   0.937500  0.0000
4      recall_1   0.818182  0.0045
5   recall_mean   0.877841  0.0000
6           f_0   0.909091  0.0000
7           f_1   0.857143  0.0000
8        f_mean   0.883117  0.0000
9     support_0  16.000000  0.0000
10     suppor_1  11.000000  0.0000
11          auc   0.835227  0.0055


Reduced dataset
===============

10CV:
Precisions:  [ 0.88235294  0.9       ] 0.891176470588
Recalls:     [ 0.9375      0.81818182] 0.877840909091
F:           [ 0.90909091  0.85714286] 0.883116883117
AUC:         0.9375
Support:     [16 11]

NPERM = 10000
-------------
   coef_count  pval    var
0          10     1  inter
1          10     1   @4.3
2          10     1   @5.4
3          10     1   @7.4
4          10     1   @7.6
          score        val   p-val
0        prec_0   0.882353  0.0011
1        prec_1   0.900000  0.0000
2     prec_mean   0.891176  0.0000
3      recall_0   0.937500  0.0001
4      recall_1   0.818182  0.0138
5   recall_mean   0.877841  0.0000
6           f_0   0.909091  0.0000
7           f_1   0.857143  0.0000
8        f_mean   0.883117  0.0000
9     support_0  16.000000  1.0000
10     suppor_1  11.000000  1.0000
11          auc   0.937500  0.0001


"""

"""
J'ai refait tout un tour de littérature. Voici l'article en l'état du jour (+ corrections)
Avant de finaliser le tour sur la discussion, Edouard peux tu faire 2 petites analyses comme discuté:

Corrélation 5.4 et 7.3 ? Comportement dangereux agressif (5.4) et suicidalité (7.3) ?
0.114949
(0.11494888257209551, 0.56804924717286553)

Corrélation 7.6 (Obsessive compulsive symptoms) et 7.3 (suicidalité) ?
0.036473
(0.036473062759285919, 0.85667251072361938)

Et si tu as le temps une figure comme celle de l'article ci dessous qui reprend tous les items de la CAARMS et qui met en ev des histogrammes à gauche et à droite (en + et en -) correspond au potentiel prédictif? Voici le lien de la revue:
http://www.schres-journal.com/article/S0920-9964(16)30558-8/pdf

Merci beaucoup beaucoup
"""

###########################################################################
## Some correlation
###########################################################################

Xd[["@5.4", "@7.3", "@7.6"]]

          @5.4      @7.3      @7.6
@5.4  1.000000  0.114949  0.481341
@7.3  0.114949  1.000000  0.036473
@7.6  0.481341  0.036473  1.000000

import scipy.stats as stats
import seaborn as sns

stats.pearsonr(Xd["@5.4"], Xd["@7.3"])

stats.pearsonr(Xd["@7.6"], Xd["@7.3"])

d = Xd.copy()
d.pop("inter")

# Compute the correlation matrix
corr = d.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
