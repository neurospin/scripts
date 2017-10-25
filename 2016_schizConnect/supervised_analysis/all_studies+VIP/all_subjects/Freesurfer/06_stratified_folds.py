#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 13:42:02 2017

@author: ad247405
"""

import os
import json
import numpy as np
import itertools
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import KFold
from sklearn.metrics import recall_score, roc_auc_score, precision_recall_fscore_support
import parsimony.estimators as estimators
import parsimony.algorithms as algorithms
import parsimony.utils as utils
from parsimony.utils.linalgs import LinearOperatorNesterov
from scipy.stats import binom_test
from collections import OrderedDict
from sklearn import preprocessing
import pandas as pd
import shutil
from brainomics import array_utils
import mapreduce
from statsmodels.stats.inter_rater import fleiss_kappa


site = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/Freesurfer/all_subjects/data/site.npy")
y = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/Freesurfer/all_subjects/data/y.npy")

site_1_num = sum(site==1)
site_1_num = sum(site==1)
site_2_num = sum(site==2)
site_3_num = sum(site==3)
site_4_num = sum(site==4)



site_1_scz = np.array(np.where(np.logical_and(site==1,y==1))).T.ravel()
site_1_scz_fold1,site_1_scz_fold2,site_1_scz_fold3,site_1_scz_fold4,site_1_scz_fold5 = np.array_split(site_1_scz,5)
site_1_con = np.array(np.where(np.logical_and(site==1,y==0))).T.ravel()
site_1_con_fold1,site_1_con_fold2,site_1_con_fold3,site_1_con_fold4,site_1_con_fold5 = np.array_split(site_1_con,5)

site_2_scz = np.array(np.where(np.logical_and(site==2,y==1))).T.ravel()
site_2_scz_fold1,site_2_scz_fold2,site_2_scz_fold3,site_2_scz_fold4,site_2_scz_fold5 = np.array_split(site_2_scz,5)
site_2_con = np.array(np.where(np.logical_and(site==2,y==0))).T.ravel()
site_2_con_fold1,site_2_con_fold2,site_2_con_fold3,site_2_con_fold4,site_2_con_fold5 = np.array_split(site_2_con,5)

site_3_scz = np.array(np.where(np.logical_and(site==3,y==1))).T.ravel()
site_3_scz_fold1,site_3_scz_fold2,site_3_scz_fold3,site_3_scz_fold4,site_3_scz_fold5 = np.array_split(site_3_scz,5)
site_3_con = np.array(np.where(np.logical_and(site==3,y==0))).T.ravel()
site_3_con_fold1,site_3_con_fold2,site_3_con_fold3,site_3_con_fold4,site_3_con_fold5 = np.array_split(site_3_con,5)

site_4_scz = np.array(np.where(np.logical_and(site==4,y==1))).T.ravel()
site_4_scz_fold1,site_4_scz_fold2,site_4_scz_fold3,site_4_scz_fold4,site_4_scz_fold5 = np.array_split(site_4_scz,5)
site_4_con = np.array(np.where(np.logical_and(site==4,y==0))).T.ravel()
site_4_con_fold1,site_4_con_fold2,site_4_con_fold3,site_4_con_fold4,site_4_con_fold5 = np.array_split(site_4_con,5)


fold1 = np.concatenate((site_1_scz_fold1,site_1_con_fold1,site_2_scz_fold1,site_2_con_fold1,\
                  site_3_scz_fold1,site_3_con_fold1,site_4_scz_fold1,site_4_con_fold1))


fold2 = np.concatenate((site_1_scz_fold2,site_1_con_fold2,site_2_scz_fold2,site_2_con_fold2,\
                  site_3_scz_fold2,site_3_con_fold2,site_4_scz_fold2,site_4_con_fold2))

fold3 = np.concatenate((site_1_scz_fold3,site_1_con_fold3,site_2_scz_fold3,site_2_con_fold3,\
                  site_3_scz_fold3,site_3_con_fold3,site_4_scz_fold3,site_4_con_fold3))

fold4 = np.concatenate((site_1_scz_fold4,site_1_con_fold4,site_2_scz_fold4,site_2_con_fold4,\
                  site_3_scz_fold4,site_3_con_fold4,site_4_scz_fold4,site_4_con_fold4))

fold5 = np.concatenate((site_1_scz_fold5,site_1_con_fold5,site_2_scz_fold5,site_2_con_fold5,\
                  site_3_scz_fold5,site_3_con_fold5,site_4_scz_fold5,site_4_con_fold5))



np.save("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/Freesurfer/all_subjects/data/fold_stratified/fold1.npy",fold1)
np.save("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/Freesurfer/all_subjects/data/fold_stratified/fold2.npy",fold2)
np.save("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/Freesurfer/all_subjects/data/fold_stratified/fold3.npy",fold3)
np.save("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/Freesurfer/all_subjects/data/fold_stratified/fold4.npy",fold4)
np.save("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/Freesurfer/all_subjects/data/fold_stratified/fold5.npy",fold5)


sum(site[fold1]==1)/len(fold1)
sum(site[fold1]==2)/len(fold1)
sum(site[fold1]==3)/len(fold1)
sum(site[fold1]==4)/len(fold1)

sum(y[fold1]==1)/len(fold1)

sum(site[fold2]==1)/len(fold2)
sum(site[fold2]==2)/len(fold2)
sum(site[fold2]==3)/len(fold2)
sum(site[fold2]==4)/len(fold2)
sum(y[fold2]==1)/len(fold2)

sum(site[fold3]==1)/len(fold3)
sum(site[fold3]==2)/len(fold3)
sum(site[fold3]==3)/len(fold3)
sum(site[fold3]==4)/len(fold3)
sum(y[fold3]==1)/len(fold3)


sum(site[fold4]==1)/len(fold4)
sum(site[fold4]==2)/len(fold4)
sum(site[fold4]==3)/len(fold4)
sum(site[fold4]==4)/len(fold4)
sum(y[fold4]==1)/len(fold4)
