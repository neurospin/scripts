# -*- coding: utf-8 -*-
"""
Created on Tue May 26 21:45:00 2015

@author: edouard.duchesnay@cea.fr

Pour l’étape 1 :

discriminer SCZ vs témoins sur le MASC et prédire le statut clinique en
fonction du MASC (t test, box plot, aire sous la courbe) colonne CI
« MASCtom » qui correspond au nombre de bonnes réponses (sur 45) à l’épreuve
MASC

ceci dit, on peut aussi comparer les types de mauvaises réponses
« MASCexc » : hypermentalisation
« MASCless »  : hypomentalisation
« MASCno » : pas de Théorie de l’Esprit (no ToM)
"""

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.weightstats import ttest_ind
from sklearn.metrics import roc_auc_score
from sklearn.lda import LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from scipy.stats import binom_test
from collections import OrderedDict

WD = "/home/ed203246/data/2015_ausz"
input_filename = os.path.join(WD, "AUSZ_data_pour_analyse_29072014.xlsx")

data = pd.read_excel(input_filename, 0)

data.groupe3bras.replace({2:"SCZ", 3:"HC"}, inplace=True)

# t-test
results = smf.ols('MASCtom ~ groupe3bras', data=data).fit()
# Inspect the results
print results.summary()

"""
======================================================================================
                         coef    std err          t      P>|t|      [95.0% Conf. Int.]
--------------------------------------------------------------------------------------
Intercept             33.2632      0.848     39.231      0.000        31.552    34.974
groupe3bras[T.SCZ]    -4.4632      1.125     -3.968      0.000        -6.733    -2.193

t value = -3.9678104288973262
P value = 0.00027751794034740397
df = 42

Area Under Curve of ROC analysis =  0.81578947368421051
Max is 1, Random is 0.5

Score        Value     P value
Accuracy     0.704545  0.004780
Specificity  0.720000  0.021643
Sensitivity  0.684211  0.083534

"""

scz = data[data.groupe3bras=="SCZ"]
hc = data[data.groupe3bras=="HC"]
print(ttest_ind(scz.MASCtom, hc.MASCtom))
#(-3.9678104288973262, 0.00027751794034740397, 42.0)

# AUC
print roc_auc_score(data.groupe3bras.replace({"SCZ":0, "HC":1}), data.MASCtom)
# 0.81578947368421051

# Prediction
y =  np.array(data.groupe3bras.replace({"HC":0, "SCZ":1}))
X = np.array(data.MASCtom, dtype=float)[:, None]

mod = LDA(priors=[.5, .5])
#mod = RandomForestClassifier()

cv = cross_validation.StratifiedKFold(y, n_folds=10, random_state=42)
#cv = cross_validation.StratifiedShuffleSplit(y, n_iter=50, train_size=.9,
#                                             random_state=42)
y_true, y_pred = list(), list()
for i, (tr, te) in enumerate(cv):
    Xte, yte = X[te, :], y[te]
    Xtr, ytr = X[tr,:], y[tr]
    y_pred.append(mod.fit(Xtr, ytr).predict(Xte))
    y_true.append(yte)

_, r, _, s = precision_recall_fscore_support(
    np.concatenate(y_true),
    np.concatenate(y_pred))
acc = accuracy_score(np.concatenate(y_true), np.concatenate(y_pred))

r2 = \
np.array(
    [precision_recall_fscore_support(y_true[i], y_pred[i])[1]
        for i in xrange(len(y_true))]).mean(axis=0)

print r, r2
#[ 0.68421053  0.72      ] [ 0.7         0.71666667]

scores = [acc] + r.tolist()
pvalues = [
    binom_test(acc * len(y), len(y), p=0.5) / 2,
    binom_test(r[0] * s[0], s[0], p=0.5) / 2,
    binom_test(r[1] * s[1], s[1], p=0.5) / 2]

pred = pd.DataFrame(dict(scores=["Accuracy", "Specificity", "Sensitivity"], values=scores, pvalues=pvalues))

# box plot
#boxplot(column=None, by=None)
%pylab qt
data.boxplot(column="MASCtom", by="groupe3bras", widths=.7)

"""
with pd.ExcelWriter(OUTPUT_CREDITSAFE_RECODED_DESCRIPTIVE) as writer:
    mapping_tab.to_excel(writer, sheet_name='Mapping', index=False)
    basic_after_recode.to_excel(writer, sheet_name='Recoded desc. all', index=False)
    desc_num_after_recode.to_excel(writer, sheet_name='Recoded desc. num.', index=False)
    desc_cat_after_recode.to_excel(writer, sheet_name='Recoded desc. cat.', index=False)
    basic_before_recode.to_excel(writer, sheet_name='Original desc. all', index=False)
    desc_num_before_recode.to_excel(writer, sheet_name='Original desc. num.', index=False)
    desc_cat_before_recode.to_excel(writer, sheet_name='Original desc. cat.', index=False)
"""