#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 08:49:36 2021

@author: ed203246


Re-salut Edouard,

J'ai oublié de te dire que les calculs avaient fini de tourner pour la learning curve sur biobd+bsnip.

Les résultats sont dans: /neurospin/psy_sbox/bd261576/checkpoints/bipolar_prediction/BIOBD_BSNIP/DenseNet/N_(n)/Test_CV_DenseNet_BIOBD_BSNIP_N(n)_fold(i)_epoch99.pkl

C'est un .pickle où tu pourras trouver y_pred et y_true pour les différents folds.

D'autre part, j'ai fait 2 fichiers (run.py et residualizer_dataset.py) que j'ai mis dans mon repo Github: https://github.com/Duplums/pynet/tree/master/bipolar_disorder_classif-2021

Tu peux t'en inspirer si tu as besoin à l'avenir de faire tourner d'autres modèles de deep sur des datasets particuliers. J'utilise ma version de pynet pour faire tourner les modèles de deep en l'occurence.

Enfin, concernant les résultats en eux-même, ils suivent ceux attendus même si je suis surpris à N=100 et N=900 (peut être vais-je les relancer, dis moi ce que tu penses).

Bonne soirée,
Benoit
"""
import os
import numpy as np
import pandas as pd
import glob
import pickle
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, recall_score
import re

STUDIES = ["biobd", "bsnip1"]
INPUT_DIR = "/neurospin/tmp/psy_sbox/all_studies/derivatives/arrays"

OUTPUT_DIR = "/neurospin/tmp/psy_sbox/analyses/202104_biobd-bsnip_cata12vbm_predict-dx"

# On laptop
if not os.path.exists(OUTPUT_DIR):
    OUTPUT_DIR = OUTPUT_DIR.replace('/neurospin/tmp', '/home/ed203246/data')
    #INPUT_DIR = INPUT_DIR.replace('/neurospin/tmp', '/home/ed203246/data')


OUTPUT = OUTPUT_DIR + "/{data}_{model}_{experience}_{type}.{ext}"

regex = re.compile("N_([0-9]+)/Test_CV_DenseNet_BIOBD_BSNIP_N([0-9]+)_fold([0-9]+)_epoch99.pkl")

xls_filename = OUTPUT.format(data='mwp1-gs', model="denseNet121", experience="cvlso-learningcurves", type="scores", ext="xlsx")

# /neurospin/psy_sbox/bd261576/checkpoints/bipolar_prediction/BIOBD_BSNIP/DenseNet/N_(n)/Test_CV_DenseNet_BIOBD_BSNIP_N(n)_fold(i)_epoch99.pkl

foldname_mapping = \
    {11: 'Baltimore', #[[0.0, 58], [1.0, 31]],
     10: 'Hartford', #[[0.0, 51], [1.0, 27]],
     12: 'Detroit', #[[0.0, 21], [1.0, 6]],
     7:  'geneve', #[[0.0, 28], [1.0, 25]],
     0:  'sandiego', #[[0.0, 74], [1.0, 43]],
     3:  'udine', #[[0.0, 90], [1.0, 36]],
     2:  'creteil', #[[0.0, 39], [1.0, 34]],
     6:  'grenoble', #[[0.0, 9], [1.0, 23]],
     9:  'Dallas', #[[0.0, 44], [1.0, 24]],
     1:  'mannheim', #[[0.0, 38], [1.0, 41]],
     5:  'pittsburgh', #[[0.0, 37], [1.0, 77]],
     4:  'galway', #[[0.0, 41], [1.0, 28]],
     8:  'Boston'} #[[0.0, 25], [1.0, 28]]

# dict_ = dict()
scores = list()

for filename in glob.glob("/neurospin/psy_sbox/bd261576/checkpoints/bipolar_prediction/BIOBD_BSNIP/DenseNet/N_*/Test_CV_DenseNet_BIOBD_BSNIP_N*_fold*_epoch99.pkl"):
    N, N_, foldnum = [int(v) for v in regex.findall(filename)[0]]
    assert N == N_
    print(filename, N, foldnum)
    with open(filename, 'rb') as fd:
        res_ = pickle.load(fd)
    # 'y_pred', 'y_true', 'loss', 'metrics'

    res_['metrics']
    y_true, score_pred = np.array(res_['y_true']), np.array(res_['y_pred'])
    y_pred = (score_pred >= 0).astype(int)

    bacc = balanced_accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, score_pred)
    recalls = recall_score(y_true, y_pred, average=None)
    assert recalls.mean() == bacc
    assert (res_['metrics']['balanced_accuracy on validation set'], res_['metrics']['roc_auc on validation set']) == \
        (bacc, auc)

    # dict_[foldnum] = [[lab, np.sum(y_true == lab)] for lab in np.unique(y_true)]

    count = [np.sum(y_true == lab) for lab in np.unique(y_true)]

    fold = foldname_mapping[foldnum]

    scores_= \
        ["denseNet121", N, 'resdualizeYes', fold] + \
        [None, None, None, None, None, "test_img"]+\
        [auc, bacc]+ list(recalls) + count
    scores.append(scores_)


scores = pd.DataFrame(scores,
             columns = ["param_0", "size", "param_1", "fold", "param_3", "param_4", "param_5", "param_6", "param_7", "pred", "auc", "bacc", "recall_0", "recall_1", "count_0", "count_1"])

assert len(scores["fold"].unique()) == 13
assert len(scores["size"].unique()) == 5
assert len(scores["fold"].unique()) * len(scores["size"].unique()) == scores.shape[0]

scores.to_excel(xls_filename, index=False)
