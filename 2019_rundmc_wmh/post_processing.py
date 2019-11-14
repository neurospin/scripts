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

#ANALYSIS_PATH = os.path.join(STUDY_PATH, 'analysis', '201905_rundmc_wmh_pca')
ANALYSIS_PATH = os.path.join(STUDY_PATH, 'analyses', '201909_rundmc_wmh_pca')
ANALYSIS_DATA_PATH = os.path.join(ANALYSIS_PATH, "data")
ANALYSIS_MODELS_PATH = os.path.join(ANALYSIS_PATH, "models")

OUTPUT_DIR = os.path.join(ANALYSIS_MODELS_PATH, '{key}')


########################################################################################################################
# Read Data
mask_img = nibabel.load(os.path.join(ANALYSIS_DATA_PATH, "mask.nii.gz"))
mask_arr = mask_img.get_data() == 1
shape = mask_arr.shape

#assert mask_arr.sum() == 51637
assert mask_arr.sum() == 371278

WMH = nibabel.load(os.path.join(ANALYSIS_DATA_PATH, "WMH_2006.nii.gz"))
WMH_flat = WMH.get_data()[mask_arr].T
X = WMH_flat - WMH_flat.mean(axis=0)

assert X.shape == (267, 371278)
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

########################################################################################################################
# Compare models group them in 4 groups by computing correlation between contactenate projectors
# Useless here since we only have 4 models

# Global correlation between concateneted projectors
Vs_arr = np.hstack((mods[k]['V'] - mods[k]['V'].mean(axis=0))[:, :2].T.ravel()[:, np.newaxis] for k in mods)
PC_arr = np.hstack((mods[k]['PC'] - mods[k]['PC'].mean(axis=0))[:, :2].T.ravel()[:, np.newaxis] for k in mods)

#k = 'pca_enettv_0.000035_1.000_0.005'

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



# Select best rsquared in each cluster
info_pc2 = info[info.comp == "PC2"]

best_r2 = list()

for k in range(len(clusters)):
    info_pc2_clust0 = info_pc2[info_pc2.key.isin(clusters[k])].reset_index(drop=True)
    best_r2.append(info_pc2_clust0.iloc[info_pc2_clust0.rsquared.idxmax()])

best_r2
"""
[key                       pca
 comp                      PC2
 rsquared             0.237925
 rsquared_ratio      0.0424806
 v_max               0.0127184
 v_min              -0.0114102
 v_abs_mean        0.000795718
 v_prop_nonnull       0.252056
 time                  6.95859
 Name: 0, dtype: object, key               pca_enettv_0.000350_1.000_0.001
 comp                                          PC2
 rsquared                                 0.225811
 rsquared_ratio                          0.0352804
 v_max                                  0.00534634
 v_min                                  -0.0045806
 v_abs_mean                            0.000243292
 v_prop_nonnull                           0.143951
 time                                      15492.6
 Name: 0, dtype: object, key               pca_enettv_0.000035_1.000_0.005
 comp                                          PC2
 rsquared                                 0.200193
 rsquared_ratio                          0.0224507
 v_max                                  0.00200296
 v_min                                 -0.00234016
 v_abs_mean                            0.000264132
 v_prop_nonnull                           0.526121
 time                                      6511.46
 Name: 0, dtype: object, key               pca_enettv_0.000001_1.000_0.000
 comp                                          PC2
 rsquared                                 0.237922
 rsquared_ratio                          0.0424779
 v_max                                  0.00715963
 v_min                                 -0.00640329
 v_abs_mean                            0.000446674
 v_prop_nonnull                           0.250481
 time                                      317.351
 Name: 0, dtype: object]
"""

# Select best rsquared_ratio in each cluster
info_pc2 = info[info.comp == "PC2"]

best_r2_ratio = list()

for k in range(len(clusters)):
    info_pc2_clust0 = info_pc2[info_pc2.key.isin(clusters[k])].reset_index(drop=True)
    best_r2_ratio.append(info_pc2_clust0.iloc[info_pc2_clust0.rsquared_ratio.idxmax()])

best_r2_ratio

"""
[key                       pca
 comp                      PC2
 rsquared             0.237925
 rsquared_ratio      0.0424806
 v_max               0.0127184
 v_min              -0.0114102
 v_abs_mean        0.000795718
 v_prop_nonnull       0.252056
 time                  6.95859
 Name: 0, dtype: object, key               pca_enettv_0.000350_1.000_0.001
 comp                                          PC2
 rsquared                                 0.225811
 rsquared_ratio                          0.0352804
 v_max                                  0.00534634
 v_min                                  -0.0045806
 v_abs_mean                            0.000243292
 v_prop_nonnull                           0.143951
 time                                      15492.6
 Name: 0, dtype: object, key               pca_enettv_0.000035_1.000_0.005
 comp                                          PC2
 rsquared                                 0.200193
 rsquared_ratio                          0.0224507
 v_max                                  0.00200296
 v_min                                 -0.00234016
 v_abs_mean                            0.000264132
 v_prop_nonnull                           0.526121
 time                                      6511.46
 Name: 0, dtype: object, key               pca_enettv_0.000001_1.000_0.000
 comp                                          PC2
 rsquared                                 0.237922
 rsquared_ratio                          0.0424779
 v_max                                  0.00715963
 v_min                                 -0.00640329
 v_abs_mean                            0.000446674
 v_prop_nonnull                           0.250481
 time                                      317.351
 Name: 0, dtype: object]
"""
pop = pd.read_csv(os.path.join(ANALYSIS_DATA_PATH, "WMH_2006_participants.csv"))

##############################################################################################################
# Save results

mask_img = nibabel.load(os.path.join(ANALYSIS_DATA_PATH, "mask.nii.gz"))
from nilearn.datasets import load_mni152_template
mni152_resamp = nilearn.image.resample_to_img(source_img=load_mni152_template(), target_img=mask_img, interpolation='continuous')
mni152_resamp.to_filename(OUTPUT_DIR.format(key="mni152.nii.gz"))


models =  ['pca_enettv_0.000035_1.000_0.005', 'pca_enettv_0.000001_1.000_0.000', 'pca_enettv_0.000350_1.000_0.001', 'pca']
#mod_str = ["pcatv_small-tv", 'pca_enettv_0.000350_1.000_0.001']
mod_str = ["pcatv_medium-tv", 'pca_enettv_0.000035_1.000_0.005']
#mod_str = ["pca", "pca"]

prefix = OUTPUT_DIR.format(key=mod_str[1]) + "/postproc_"
mod = mods[mod_str[1]]

##############################################################################################################
# Save components

xls_filename = prefix+"components.xlsx"

PC_df = pd.DataFrame(mod["PC"], columns=["PC%i" % (i+1) for i in range(mod["PC"].shape[1])])
PCcor = PC_df.corr()
assert pop.shape[0] == PC_df.shape[0]
PC_df = pd.concat([pop, PC_df], axis=1)

V_df = pd.DataFrame(mod["V"], columns=["v%i" % (i+1) for i in range(mod["V"].shape[1])])
V_df.corr()


with pd.ExcelWriter(xls_filename) as writer:
    PC_df.to_excel(writer, sheet_name='Components', index=False)
    PCcor.to_excel(writer, sheet_name='Corr components', index=False)
    info[info.key == mod_str[1]].to_excel(writer, sheet_name='Components info', index=False)

##############################################################################################################
# Plot
import nilearn.plotting.cm as cmnl

U, d, V, PC, explained_variance = mod["U"], mod["d"], mod["V"], mod["PC"], mod["explained_variance"]
explained_variance_ratio = np.concatenate([[explained_variance[0]], np.ediff1d(explained_variance)])

# Some small thresholding
from brainomics.array_utils import arr_get_threshold_from_norm2_ratio
thresholds = np.array([arr_get_threshold_from_norm2_ratio(V[: ,k], ratio=.99) for k in range(V.shape[1])])
V[V < thresholds] = 0

# Save loadings as 4D image
map_arr = np.zeros(list(shape) + [V.shape[1]])
map_arr[mask_arr] = V
map_img = nibabel.Nifti1Image(map_arr, mask_img.affine)
map_img.to_filename(prefix+"components-brain-maps.nii.gz")
map_img_l = nibabel.four_to_three(map_img)

#V = pca.components_.T

pdf = PdfPages(prefix+"components-brain-maps.pdf")

fig = plt.figure(figsize=(13.33, 10 * U.shape[1]))
fig.suptitle(mod_str[0])
axis = fig.subplots(nrows=U.shape[1] * 2, ncols=1)

for k in range(U.shape[1]):
    #k = 0
    idx = 2 * k
    map_img = map_img_l[k]

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

########################################################################################################################
cd /neurospin/brainomics/2019_rundmc_wmh/analyses/201909_rundmc_wmh_pca/models/pca_enettv_0.000350_1.000_0.001
fsl5.0-fslsplit components-brain-maps.nii.gz ./components-brain-maps_PC -t
~/git/scripts/brainomics/image_clusters_analysis_nilearn.py /neurospin/brainomics/2019_rundmc_wmh/analyses/201909_rundmc_wmh_pca/models/pca_enettv_0.000350_1.000_0.001/components-brain-maps_PC0000.nii.gz --atlas JHU --thresh_size 10
~/git/scripts/brainomics/image_clusters_analysis_nilearn.py /neurospin/brainomics/2019_rundmc_wmh/analyses/201909_rundmc_wmh_pca/models/pca_enettv_0.000350_1.000_0.001/components-brain-maps_PC0001.nii.gz --atlas JHU --thresh_size 10
~/git/scripts/brainomics/image_clusters_analysis_nilearn.py /neurospin/brainomics/2019_rundmc_wmh/analyses/201909_rundmc_wmh_pca/models/pca_enettv_0.000350_1.000_0.001/components-brain-maps_PC0002.nii.gz --atlas JHU --thresh_size 10

#
cd /neurospin/brainomics/2019_rundmc_wmh/analyses/201909_rundmc_wmh_pca/models/pca_enettv_0.000035_1.000_0.005
fsl5.0-fslsplit components-brain-maps.nii.gz ./components-brain-maps_PC -t

~/git/scripts/brainomics/image_clusters_analysis_nilearn.py components-brain-maps_PC0000.nii.gz --atlas JHU --thresh_size 10
~/git/scripts/brainomics/image_clusters_analysis_nilearn.py components-brain-maps_PC0001.nii.gz --atlas JHU --thresh_size 10
~/git/scripts/brainomics/image_clusters_analysis_nilearn.py components-brain-maps_PC0002.nii.gz --atlas JHU --thresh_size 10
