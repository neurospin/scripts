# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 13:45:39 2017

@author: ed203246
"""
import os
import subprocess
import json
import numpy as np
from sklearn.cross_validation import StratifiedKFold
import nibabel
from sklearn import svm
from sklearn.metrics import precision_recall_fscore_support
from sklearn.feature_selection import SelectKBest
import brainomics.image_atlas
import brainomics.array_utils
from scipy.stats import binom_test
from collections import OrderedDict
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score, recall_score
import pandas as pd
from collections import OrderedDict
import nilearn  
from nilearn import plotting
from nilearn import image
import seaborn as sns
import matplotlib.pylab as plt

# Describe X, y # Key = subject = [S|E]IRM
POPULATION_CSV = "/neurospin/brainomics/2016_deptms/analysis/VBM/population.csv"
# All clinical data KEY=IRM
CLINIC_ALL_XLS = "/neurospin/brainomics/2016_deptms/clinic/HAMLOC_EDW.xlsx"

# extraction from HAMLOC_EDW.xlsx KEY=IRM
#CLINIC_JOIN_CSV =  '/neurospin/brainomics/2016_deptms/deptms_info.csv'

DATA_X = "/neurospin/brainomics/2016_deptms/analysis/VBM/data/X.npy"
DATA_y = "/neurospin/brainomics/2016_deptms/analysis/VBM/data/y.npy"


IMAGE_PATH = "/neurospin/brainomics/2016_deptms/analysis/VBM/results/enettv/model_selection_5folds/model_selectionCV/all/all/"
OUTPUT = "/neurospin/brainomics/2016_deptms/analysis/VBM/results/enettv/model_selection_5folds/pattern_interpretation"

###############################################################################
# Rebuild clinical data using CLINIC_USED_CSV to make the join

import re

pop = pd.read_csv(POPULATION_CSV)
clinic_all = pd.read_excel(CLINIC_ALL_XLS, sheetname="Sheet1")
clinic_all['Sex'] = clinic_all.pop('sexe')
#clinic_join = pd.read_csv(CLINIC_JOIN_CSV)[["Code", "subject"]]

pop['IRM'] = [int(re.sub('[^0-9]', '', s)) for s in pop['subject']]

popclin = pd.merge(pop, clinic_all, on=["IRM", "Age", "Sex"])

# check order respect population
np.all(pop['IRM'] == popclin['IRM'])

popclin['Durée maladie(ans)']

##################################################################################
babel_mask  = nibabel.load("/neurospin/brainomics/2016_deptms/analysis/VBM/data/mask.nii")
mask_bool = babel_mask.get_data()
mask_bool= np.array(mask_bool !=0)
number_features = mask_bool.sum()
##################################################################################

'''
# 2017/02/01
cd /neurospin/brainomics/2016_deptms/analysis/VBM/results/enettv/model_selection_5folds


Look at
results_dCV_5folds_reduced_grid

picke a reduced grid with l1 != 0

Something like
alpha = [0.1, 0.01]
l1_ratio = [0.1, 0.9]
tv[??_ratio??] = [0.2, 0.8]

The best with such reduced is 
param=0.1_0.02_0.18_0.8
'''

param = '0.1_0.02_0.18_0.8'

penalty_start = 3
#IMAGE_PATH = '/neurospin/brainomics/2016_deptms/analysis/VBM/results/enettv/model_selection_5folds/model_selectionCV/all/all'
beta_tot = np.load(os.path.join(IMAGE_PATH, param, "beta.npz"))['arr_0']
beta = beta_tot[penalty_start:]

output = os.path.join(OUTPUT, param)
try:
    os.makedirs(output)
except:
    pass

arr = np.zeros(mask_bool.shape);
arr[mask_bool] = beta.ravel()
out_im = nibabel.Nifti1Image(arr, affine=babel_mask.get_affine())
map_filename = os.path.join(output, "weight_map.nii.gz")
out_im.to_filename(map_filename)

#beta = nibabel.load(map_filename).get_data()
#beta_t, t = brainomics.array_utils.arr_threshold_from_norm2_ratio(beta, .99)

#play with parameter vmax and vmin
nilearn.plotting.plot_glass_brain(map_filename,colorbar=True,plot_abs=False,threshold = t, vmax =0.001,vmin = -0.001)


vmax = 0.001
thresh_norm_ratio = 0.99
thresh_size = 10
CMD = "image_clusters_analysis_nilearn.py %s -o %s --vmax %f --thresh_norm_ratio %f --thresh_size %i" % \
    (map_filename, output, vmax, thresh_norm_ratio, thresh_size)


a = subprocess.call(CMD.split())
###############################################################################
"""
weight_map_clust_info.csv
weight_map_clust_info.pdf
weight_map_clust_labels.nii.gz
weight_map_clust_values.nii.gz
weight_map.nii.gz
"""

labels_img  = nibabel.load(os.path.join(output, "weight_map_clust_labels.nii.gz"))
labels_arr = labels_img.get_data()
labels_flt = labels_arr[mask_bool]



Xtot = np.load(DATA_X)
X = Xtot[:, penalty_start:]
y = np.load(DATA_y)
assert np.all(pop['Response.num'] == y)
assert X.shape[1] == beta.shape[0] 
assert labels_flt.shape[0] == beta.shape[0]
N = X.shape[0]

# Extract a single score for each cluster
K = len(np.unique(labels_flt))  # nb cluster
Xp = np.zeros((N, K))

for k in np.unique(labels_flt):
    mask = labels_flt == k
    print("Cluster:",k, "size:", mask.sum())
    Xp[:, k] = np.dot(X[:, mask], beta[mask]).ravel()

plt.scatter(Xp[y==0, 1], Xp[y==0, 20], c='r', label='N')
plt.scatter(Xp[y==1, 1], Xp[y==1, 20], c='b', label='Y')

plt.legend()

clusts = dict(TemporalL=20, Occipital=1, FrontalL=15, CaudateL=17)

clust_oi = [clusts[k] for k in clusts]

Dp = pd.DataFrame(np.concatenate([Xp[:, clust_oi], y[:, np.newaxis]], axis=1), columns=list(clusts.keys())+["Response"])

Dp['Duree_maladie'] = popclin['Durée maladie(ans)']
Dp['Age'] = popclin['Age']

g = sns.PairGrid(Dp, hue="Response")
g.map_diag(plt.hist)
g.map_offdiag(plt.scatter)
g.add_legend()

Xp[y==1, :][:, clust_oi].mean(axis=0) - Xp[y==0, :][:, clust_oi].mean(axis=0)
# dict_keys(['FrontalL', 'CaudateL', 'Occipital', 'TemporalL'])
# array([-0.03157284, -0.00355261,  1.05848566, -0.0858497 ])

dela_grandmean = X[y==1, :].mean() -  X[y==0, :].mean()
# -0.046514042021454705

 dela_grandmean / X.mean()
# => - 8%
# -0.085906596756939804

plt.scatter(Dp.TemporalL + Dp.Occipital, Dp['Duree_maladie'], c=Dp.Response)
plt.scatter(Dp.TemporalL + Dp.Occipital, popclin['Age'], c=Dp.Response)
plt.scatter(Dp.Occipital, popclin['Age'], c=Dp.Response)

plt.scatter(Dp.Occipital, popclin['Age'], c=popclin['Sex'])
plt.legend()
