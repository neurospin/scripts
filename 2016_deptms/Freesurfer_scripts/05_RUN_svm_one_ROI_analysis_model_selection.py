#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 17:03:53 2016

@author: ad247405
"""

import os
import json
import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn import svm
from scipy.stats import binom_test
from collections import OrderedDict
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score, recall_score
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
import shutil


BASE_PATH ="/neurospin/brainomics/2016_deptms"
INPUT_DATA_STATS = "/neurospin/brainomics/2016_deptms/analysis/Freesurfer/freesurfer_stats"
INPUT_DATA_y = '/neurospin/brainomics/2016_deptms/analysis/Freesurfer/data/y.npy'
INPUT_DATA_X = '/neurospin/brainomics/2016_deptms/analysis/Freesurfer/data/Xrois_thickness.npy'
OUTPUT = "/neurospin/brainomics/2016_deptms/analysis/Freesurfer/results/svm_rois/ROIs"
INPUT_CSV = "/neurospin/brainomics/2016_deptms/analysis/Freesurfer/population.csv"


df = pd.read_csv(os.path.join(INPUT_DATA_STATS,"aseg_volume_all.csv"),sep='\t')
  

   
for roi in df.keys()[40:]:
    print ("ROI", roi)
    if roi != "Measure:volume":
        WD = os.path.join(OUTPUT,roi)
        os.chdir(WD)
        map_cmd = "mapreduce.py config_dCV.json --map --ncore 8"
        os.system(map_cmd)

        
        
        
  
table = np.zeros((df.shape[1]-1,4))  
df_table = pd.DataFrame(table,index = df.keys()[1:],columns = ["auc","balanced_acc","specificity","sensitivity"])
for roi in df.keys()[1:]:
    print ("ROI", roi)
    if roi != "Measure:volume":
        WD = os.path.join(OUTPUT,roi)
        scores = pd.read_excel(os.path.join(WD,"results_dCV_5folds.xlsx"),sheetname="scores_cv")
        df_table.ix[roi].auc = float(scores.auc)
        df_table.ix[roi].balanced_acc = float(scores.recall_mean)
        df_table.ix[roi].specificity = float(scores.recall_0)
        df_table.ix[roi].sensitivity = float(scores.recall_1)
        

        
df_part1 = df_table.ix[:34,["auc","balanced_acc","specificity","sensitivity"]]
df_part2 = df_table.ix[34:,["auc","balanced_acc","specificity","sensitivity"]]
 
        
import matplotlib.pyplot as plt
import pandas as pd
from pandas.tools.plotting import table

ax = plt.subplot(111, frame_on=False) # no visible frame
ax.xaxis.set_visible(False)  # hide the x axis
ax.yaxis.set_visible(False)  # hide the y axis
table(ax, df_part1)  # where df is your data frame

ax = plt.subplot(111, frame_on=False) # no visible frame
ax.xaxis.set_visible(False)  # hide the x axis
ax.yaxis.set_visible(False)  # hide the y axis
table(ax, df_part2)  # where df is your data frame