# -*- coding: utf-8 -*-
"""
Created on Sat May 30 13:25:21 2015

@author: edouard.duchesnay@cea.fr

Pour l’étape 2 :

au sein du groupe SCZ, association scores cliniques / type d’erreurs (hyper, hypo, no)
colonne à partir de AD
- PANSSpos et suivante (> PANSSn4)
- SANS1a7 et suivantes (> SANStot)
puis colonnes à partir de CF
- TLC
- EQ
- AQ
puis étendre à analyse multivariée : quels sont les score qui prédisent bien

ceci dit, on pourrait aussi faire ça avec les âges de début de la maladie
colonnes J à P
agdebsub
ag1symp
ag1dysf
agprodrome
ag1contact
ag1episode
agdg
et avec la durée des prodromes si possibles, c’est à dire le délai ag1episode - agprodrome

"""
import os
import numpy as np
import pandas as pd

from sklearn import cross_validation


WD = "/home/ed203246/data/2015_ausz"
os.chdir(WD)
input_filename = os.path.join(WD, "AUSZ_data_pour_analyse_29072014.xlsx")
#_filename = os.path.join(WD, "szc_.xlsx")

data = pd.read_excel(input_filename, 0)

data.groupe3bras.replace({2:"SCZ", 3:"HC"}, inplace=True)


clinic_cols = \
data.columns[
    np.where(data.columns == 'PANSSpos')[0][0]:\
    (np.where(data.columns == 'PANSSn4')[0][0]+1)
    ].tolist() +\
data.columns[
    np.where(data.columns == 'SANS1a7')[0][0]:\
    (np.where(data.columns == 'SANStot')[0][0]+1)
    ].tolist() +\
["TLC", "EQ", "AQ"]

history_cols = ["agdebsubj","ag1sympt", "ag1dysf", "agprodrom","ag1contact", "ag1episode", "agdg"]

data["duration_prodrom"] = data.ag1episode - data.agprodrom
history_cols += ["duration_prodrom"]

masc_cols = ["MASCtom", "MASCexc", "MASCless", "MASCno"]#, "MASCemo", "MASCpens"]

scz = data[data.groupe3bras=="SCZ"]

#############################################
# Clinic vs masc assoc
#############################################

predictor_cols = clinic_cols
output_cols = masc_cols
from mulm.dataframe.mulm_dataframe import MULM
formulas = ["%s ~ % s" % (y, x) for x in predictor_cols for y in output_cols]
model = MULM(data=scz, formulas=formulas)

stats = model.t_test(contrasts=1, out_filemane=None)
stats.to_excel("within_scz_masc_vs_clinic.xlsx", "univ-stats-assoc", index=False)


############################
# Mas prediction from Clinic
############################

# Prediction
for target in masc_cols:
    #target = masc_cols[0]
    y = scz[target]
    X = scz[clinic_cols]
    X.isnull().sum()
    """
    PANSSpos      0
    ...
    SANStot       0
    TLC           3
    EQ            2
    AQ            2
    dtype: int64
    """
    col_with_NA = X.columns[X.isnull().sum() !=0]
    for col in col_with_NA:
        X[col].fillna(X[col].mean(), inplace=True)

    assert len(X.columns[X.isnull().sum() !=0]) == 0
    X = np.array(X, dtype=float)

    from sklearn.linear_model import ElasticNetCV
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.svm import SVR
    from sklearn import grid_search

    param_svr = {'kernel':('linear', 'rbf'), 'C':[1, 10]}

    models = dict(
        enet = ElasticNetCV( fit_intercept=True),
        rf = RandomForestRegressor(),
        svr = grid_search.GridSearchCV(SVR(), param_svr))

    results = dict()
    for mod in models.keys():
        scores = cross_validation.cross_val_score(models[mod], X, y, cv=5)
        results[mod] = scores.mean()

    print target, results

"""
Rien
MASCtom {'rf': -1.4004315632204798, 'svr': -0.45414150685117305, 'enet': -1.2284542957757831}
MASCexc {'rf': -0.32887110084726112, 'svr': -0.08297830284338481, 'enet': -0.18533394491988431}
MASCless {'rf': -0.42449838342458052, 'svr': -0.76873047338163614, 'enet': -0.59094744607912064}
MASCno {'rf': -2.6605775641025646, 'svr': -2.3992377312691184, 'enet': -2.3262548252006967}
"""


############################
# Mas prediction from history_cols
############################

# Prediction
for target in masc_cols:
    #target = masc_cols[0]
    y = scz[target]
    X = scz[history_cols]
    X.isnull().sum()
    """
    PANSSpos      0
    ...
    SANStot       0
    TLC           3
    EQ            2
    AQ            2
    dtype: int64
    """
    col_with_NA = X.columns[X.isnull().sum() !=0]
    for col in col_with_NA:
        X[col].fillna(X[col].mean(), inplace=True)

    assert len(X.columns[X.isnull().sum() !=0]) == 0
    X = np.array(X, dtype=float)

    from sklearn.linear_model import ElasticNetCV
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.svm import SVR
    from sklearn import grid_search

    param_svr = {'kernel':('linear', 'rbf'), 'C':[1, 10]}

    models = dict(
        enet = ElasticNetCV( fit_intercept=True),
        rf = RandomForestRegressor(),
        svr = grid_search.GridSearchCV(SVR(), param_svr))

    results = dict()
    for mod in models.keys():
        scores = cross_validation.cross_val_score(models[mod], X, y, cv=5)
        results[mod] = scores.mean()

    print target, results

"""
MASCtom {'rf': -1.1659913942861846, 'svr': -0.36364296779613159, 'enet': -0.91737194831262447}
MASCexc {'rf': -0.42349280336871942, 'svr': -0.16244820954351363, 'enet': -0.28588915789284636}
MASCless {'rf': -1.4727013605076293, 'svr': -0.74615277520044843, 'enet': -0.71545895803630244}
MASCno {'rf': -2.2029967948717952, 'svr': -2.60410399161352, 'enet': -2.2116762820512816}

"""
"""
Analyse d'association univariées
--------------------------------


Voir fichier: within_scz_masc_vs_clinic.xsls

Voilà ce qui passe, attention ce n'est pas corrigé
target	contrast	effect	sd	tvalue	pvalue	df
MASCless 	SANS1a7	-0.2889869439	0.1251770404	-2.3086257919	0.0302967098	23
MASCtom 	SANS1a7	0.2868109284	0.1558225243	1.8406256077	0.0786149373	23
MASCless 	PANSStot	-0.1018256217	0.0560237573	-1.8175436029	0.0821857033	23

Analyse multivariée
-------------------

1: Prediction de ["MASCtom", "MASCexc", "MASCless", "MASCno"] à partir de
[u'PANSSpos',
 u'PANSSneg',
 u'PANSSg',
 u'PANSStot',
 u'PANSSp4',
 u'PANSSn4',
 u'SANS1a7',
 u'SANS9a12',
 u'SANS14a16',
 u'SANS18a21',
 u'SANS23a24',
 u'SANSglob',
 u'SANSsscore',
 u'SANStot',
 'TLC',
 'EQ',
 'AQ']

=> Rien

2. Prediction de ["MASCtom", "MASCexc", "MASCless", "MASCno"] à partir de
['agdebsubj',
 'ag1sympt',
 'ag1dysf',
 'agprodrom',
 'ag1contact',
 'ag1episode',
 'agdg',
 'duration_prodrom']

Rien non plus
"""