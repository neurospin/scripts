import sys
import os
import time

import numpy as np
import nibabel
import pandas as pd
import matplotlib.pylab as plt
import nilearn
from nilearn import plotting
import argparse
import glob

from matplotlib.backends.backend_pdf import PdfPages


#sys.path.append('/home/ed203246/git/scripts/2013_mescog/proj_wmh_patterns')
#import pca_tv
from parsimony.decomposition import PCAL1L2TV
import  parsimony.decomposition.pca_tv as pca_tv

import parsimony.functions.nesterov.tv
from brainomics import array_utils


#from brainomics import plot_utilities
#import parsimony.utils.check_arrays as check_arrays

########################################################################################################################
STUDY_PATH = '/neurospin/brainomics/2019_rundmc_wmh'
DATA_PATH = os.path.join(STUDY_PATH, 'sourcedata', 'wmhmask')

ANALYSIS_PATH = os.path.join(STUDY_PATH, 'analysis', '201905_rundmc_wmh_pca')
ANALYSIS_DATA_PATH = os.path.join(ANALYSIS_PATH, "data")
ANALYSIS_MODELS_PATH = os.path.join(ANALYSIS_PATH, "models")

OUTPUT_DIR = os.path.join(ANALYSIS_MODELS_PATH, '201905_rundmc_wmh_pca', '{key}')


########################################################################################################################
# Read Data
mask_img = nibabel.load(os.path.join(ANALYSIS_DATA_PATH, "mask.nii.gz"))
mask_arr = mask_img.get_data() == 1
shape = mask_arr.shape

assert mask_arr.sum() == 51637
WMH = nibabel.load(os.path.join(ANALYSIS_DATA_PATH, "WMH.nii.gz"))
WMH_flat = WMH.get_data()[mask_arr].T
X = WMH_flat - WMH_flat.mean(axis=0)

assert X.shape == (503, 51637)
assert mask_arr.sum() == X.shape[1]
assert np.allclose(X.mean(axis=0), 0)

# Fit model
N_COMP = 3

########################################################################################################################
# Read models
mods = {os.path.basename(dirname):np.load(os.path.join(dirname, "model.npz")) for dirname in glob.glob(os.path.join(ANALYSIS_MODELS_PATH, 'pca*'))}
info = pd.concat([pd.read_csv(os.path.join(dirname, "info.csv")) for dirname in glob.glob(os.path.join(ANALYSIS_MODELS_PATH, 'pca*'))])
info = info.sort_values(by=['comp', 'rsquared'], ascending=False)

#k = 'pca_enettv_0.0000_1.000_0.100'
Vs_arr = np.hstack((mods[k]['V'] - mods[k]['V'].mean(axis=0))[:, :2].T.ravel()[:, np.newaxis] for k in mods)
PC_arr = np.hstack((mods[k]['PC'] - mods[k]['PC'].mean(axis=0))[:, :2].T.ravel()[:, np.newaxis] for k in mods)

df = pd.DataFrame(Vs_arr, columns=mods.keys())
#df = pd.DataFrame(PC_arr, columns=mods.keys())

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Compute the correlation matrix
corr = df.corr()
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(5.5, 4.5))
#cmap = sns.color_palette("RdBu_r", 50)
cmap = sns.color_palette("hot_r", 50)

# Draw the heatmap with the mask and correct aspect ratio
_ = sns.heatmap(corr, mask=None, cmap=cmap, vmax=1, center=0.5,
square=True, linewidths=.5, cbar_kws={"shrink": .5})

d = 2 * (1 - np.abs(corr))
from sklearn.cluster import AgglomerativeClustering
clustering = AgglomerativeClustering(n_clusters=4, linkage='single', affinity="precomputed").fit(d)
lab=0
clusters = [list(corr.columns[clustering.labels_==lab]) for lab in set(clustering.labels_)]
print(clusters)
reordered = np.concatenate(clusters)
R = corr.loc[reordered, reordered]
f, ax = plt.subplots(figsize=(5.5, 4.5))
# Draw the heatmap with the mask and correct aspect ratio
_ = sns.heatmap(R, mask=None, cmap=cmap, vmax=1, center=0.5,
square=True, linewidths=.5, cbar_kws={"shrink": .5})

"""
Base on V : 4 clusters

TV >=0.5 : cluster 1
TV <= 0.01, l1 <= 0.01, cluster 2 like PCA
TV == 0.1 cluster 3
pca cluster 4

[['pca_enettv_0.0000_1.000_1.000',
  'pca_enettv_0.0456_1.000_0.500',
  'pca_enettv_0.0046_1.000_1.000',
  'pca_enettv_0.0456_1.000_1.000',
  'pca_enettv_0.0046_1.000_0.500',
  'pca_enettv_0.0005_1.000_1.000',
  'pca_enettv_0.0005_1.000_0.500'],
  
 ['pca_enettv_0.0000_1.000_0.010',
  'pca_enettv_0.0000_1.000_0.001',
  'pca_enettv_0.0000_1.000_0.000'],
  
 ['pca_enettv_0.0000_1.000_0.100',
  'pca_enettv_0.0005_1.000_0.100',
  'pca_enettv_0.0456_1.000_0.100',
  'pca_enettv_0.0046_1.000_0.100'],
 
 ['pca']]

Based on PCs 3 clusters
no L1 small/no TV

pca


[['pca_enettv_0.0000_1.000_0.001', 'pca_enettv_0.0000_1.000_0.000'],
 ['pca'],
 ['pca_enettv_0.0000_1.000_0.100',
  'pca_enettv_0.0000_1.000_0.010',
  'pca_enettv_0.0000_1.000_1.000',
  'pca_enettv_0.0456_1.000_0.500',
  'pca_enettv_0.0046_1.000_1.000',
  'pca_enettv_0.0005_1.000_0.100',
  'pca_enettv_0.0456_1.000_1.000',
  'pca_enettv_0.0456_1.000_0.100',
  'pca_enettv_0.0046_1.000_0.500',
  'pca_enettv_0.0005_1.000_1.000',
  'pca_enettv_0.0005_1.000_0.500',
  'pca_enettv_0.0046_1.000_0.100']]
"""

# Select best rsquared in each cluster
info_pc2 = info[info.comp == "PC2"]

best_r2 = list()

for k in range(len(clusters)):
    info_pc2_clust0 = info_pc2[info_pc2.key.isin(clusters[k])].reset_index(drop=True)
    best_r2.append(info_pc2_clust0.iloc[info_pc2_clust0.rsquared.idxmax()])

best_r2
"""
[key               pca_enettv_0.0005_1.000_0.500
 comp                                        PC2
 rsquared                               0.413727
 rsquared_ratio                       0.00511588
 v_max                                 0.0154899
 v_min                              -3.20613e-07
 v_abs_mean                          0.000675874
 v_prop_nonnull                         0.881248
 time                                    1690.73
 
 key               pca_enettv_0.0000_1.000_0.000
 comp                                        PC2
 rsquared                               0.986604
 rsquared_ratio                       0.00259148
 v_max                                0.00828734
 v_min                               -0.00202959
 v_abs_mean                          0.000925844
 v_prop_nonnull                         0.362628
 time                                    16.3457
 
 key               pca_enettv_0.0046_1.000_0.100
 comp                                        PC2
 rsquared                               0.836672
 rsquared_ratio                        0.0175312
 v_max                                 0.0256441
 v_min                              -0.000602655
 v_abs_mean                          0.000432196
 v_prop_nonnull                        0.0194047
 time                                    1579.91
 Name: 0, dtype: object,
 
 key                      pca
 comp                     PC2
 rsquared            0.986604
 rsquared_ratio    0.00259148
 v_max              0.0229645
 v_min            -0.00562382
 v_abs_mean        0.00256523
 v_prop_nonnull      0.361717
 time                 1.78688
"""

# Select best rsquared_ratio in each cluster
info_pc2 = info[info.comp == "PC2"]

best_r2_ratio = list()

for k in range(len(clusters)):
    info_pc2_clust0 = info_pc2[info_pc2.key.isin(clusters[k])].reset_index(drop=True)
    best_r2_ratio.append(info_pc2_clust0.iloc[info_pc2_clust0.rsquared_ratio.idxmax()])

best_r2_ratio

"""
[key               pca_enettv_0.0046_1.000_1.000
 comp                                        PC2
 rsquared                                0.35724
 rsquared_ratio                        0.0126917
 v_max                                 0.0223584
 v_min                              -4.01883e-06
 v_abs_mean                          6.92472e-05
 v_prop_nonnull                        0.0114453
 time                                    1666.15
 
 key               pca_enettv_0.0000_1.000_0.000
 comp                                        PC2
 rsquared                               0.986604
 rsquared_ratio                       0.00259148
 v_max                                0.00828734
 v_min                               -0.00202959
 v_abs_mean                          0.000925844
 v_prop_nonnull                         0.362628
 time                                    16.3457
 
 key               pca_enettv_0.0000_1.000_0.100
 comp                                        PC2
 rsquared                                0.83558
 rsquared_ratio                        0.0179982
 v_max                                  0.029446
 v_min                              -0.000374288
 v_abs_mean                           0.00118221
 v_prop_nonnull                          0.45448
 time                                     2482.3
 
 key                      pca
 comp                     PC2
 rsquared            0.986604
 rsquared_ratio    0.00259148
 v_max              0.0229645
 v_min            -0.00562382
 v_abs_mean        0.00256523
 v_prop_nonnull      0.361717
 time                 1.78688
 Name: 0, dtype: object]
"""
pop = pd.read_csv(os.path.join(ANALYSIS_PATH, "participants.csv"))

##############################################################################################################
# Save results

models =  ["pca_enettv_0.0046_1.000_1.000", "pca_enettv_0.0000_1.000_0.100", "pca"]

mod_str = ["pcatv_largeTV", "pca_enettv_0.0046_1.000_1.000"]
mod_str = ["pcatv_mediumTV", "pca_enettv_0.0000_1.000_0.100"]
mod_str = ["pca", "pca"]


prefix = OUTPUT_DIR.format(key=mod_str[0])
mod = mods[mod_str[1]]

##############################################################################################################
# Save components
xls_filename = prefix+"_components.xlsx"

PC_df = pd.DataFrame(mod["PC"], columns=["PC%i" % (i+1) for i in range(mod["PC"].shape[1])])
PCcor = PC_df.corr()
assert pop.shape[0] == PC_df.shape[0]
PC_df = pd.concat([pop, PC_df], axis=1)

V_df = pd.DataFrame(mod["V"], columns=["v%i" % (i+1) for i in range(mod["V"].shape[1])])
V_df.corr()


info[info.key == mod_str[1]]

with pd.ExcelWriter(xls_filename) as writer:
    PC_df.to_excel(writer, sheet_name='Components', index=False)
    PCcor.to_excel(writer, sheet_name='Corr components', index=False)
    info[info.key == mod_str[1]].to_excel(writer, sheet_name='Components info', index=False)

##############################################################################################################
# Plot
import nilearn.plotting.cm as cmnl

U, d, V, PC, explained_variance = mod["U"], mod["d"], mod["V"], mod["PC"], mod["explained_variance"]
explained_variance_ratio = np.concatenate([[explained_variance[0]], np.ediff1d(explained_variance)])

from brainomics.array_utils import arr_get_threshold_from_norm2_ratio
thresholds = np.array([arr_get_threshold_from_norm2_ratio(V[: ,k], ratio=.99) for k in range(V.shape[1])])
V[V < thresholds] = 0



mask_img = nibabel.load(os.path.join(ANALYSIS_DATA_PATH, "mask.nii.gz"))
#V = pca.components_.T

pdf = PdfPages(prefix+"_components-brain-maps.pdf")

fig = plt.figure(figsize=(13.33, 10 * U.shape[1]))
fig.suptitle(mod_str[0])
axis = fig.subplots(nrows=U.shape[1] * 2, ncols=1)

for k in range(U.shape[1]):
    #k = 0
    idx = 2 * k
    map_arr = np.zeros(shape)
    map_arr[mask_arr] = V[:, k]# * 100
    map_img = nibabel.Nifti1Image(map_arr, mask_img.affine)

    #ax = fig.add_subplot(111)
    #ax.set_title("T-stats T>%.2f" %  tstats_thres)

    vmax = np.abs(map_arr).max()

    axis[idx].set_title("PC%i (EV:%.3f%%)" %  (k+1, explained_variance_ratio[k] * 100))
    plotting.plot_glass_brain(map_img, colorbar=True, vmax=vmax, figure=fig, axes=axis[idx])
    #pdf.savefig()
    display = plotting.plot_stat_map(map_img, colorbar=True, draw_cross=True, cmap=cmnl.cold_white_hot, figure=fig, axes=axis[idx+1])#, symmetric_cbar=False)#, cmap=plt.cm.hot_r)#,  cut_coords=[16, -4, 0], symmetric_cbar=False, cmap=cold_blue)#, threshold=3,)#, figure=fig, axes=ax)
    plt.show()

pdf.savefig()
plt.savefig(prefix+"_components-brain-maps.png")
plt.close(fig)
pdf.close()
