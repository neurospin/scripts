#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 11:37:56 2018

@author: ad247405
"""
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt


d = {'auc': [0.78, 0.76,0.68,0.64,0.72], 'classifier': ["svm","Enet-TV","svm","Enet-TV","svm"],\
"features": ["VBM","VBM","vertex","vertex","ROIS"]}
df = pd.DataFrame(data=d)

sns.set_style("whitegrid")
ax = sns.barplot(x="features", y="auc", hue="classifier", data=df)



d = {'Area Under the Curve': [0.74,0.74,0.69,0.70,0.78,0.78, 0.76,0.68,0.64,0.72],\
     'Classifier': ["SVM","Enet-TV","SVM","Enet-TV","SVM","SVM","Enet-TV","SVM","Enet-TV","SVM"],\
"Features": ["Grey Matter\n VBM","Grey Matter\n VBM","Vertex-based\n cortical thickness",\
"Vertex-based\n cortical thickness","ROIs-based\n volume","Grey Matter\n VBM","Grey Matter\n VBM",\
"Vertex-based\n cortical thickness","Vertex-based\n cortical thickness","ROIs-based\n volume"],\
"Dataset":["HC vs SCZ","HC vs SCZ","HC vs SCZ","HC vs SCZ","HC vs SCZ",\
"HC vs FEP","HC vs FEP","HC vs FEP","HC vs FEP","HC vs FEP"]}
df = pd.DataFrame(data=d)
plt.figure(figsize=(50,20))
sns.set_style("whitegrid")
ax = sns.factorplot(x="Features", y="Area Under the Curve", hue="Classifier",\
                    data=df,col = "Dataset",kind="bar")
plt.savefig("/neurospin/brainomics/2016_schizConnect/analysis/z.submission/plot_auc_performances")
