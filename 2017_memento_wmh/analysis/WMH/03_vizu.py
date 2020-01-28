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
import time
from matplotlib.backends.backend_pdf import PdfPages


#sys.path.append('/home/ed203246/git/scripts/2013_mescog/proj_wmh_patterns')
#import pca_tv
#from parsimony.decomposition import PCAL1L2TV
#import  parsimony.decomposition.pca_tv as pca_tv

#import parsimony.functions.nesterov.tv
#from brainomics import array_utils


#from brainomics import plot_utilities
#import parsimony.utils.check_arrays as check_arrays

from nilearn.image import resample_to_img

########################################################################################################################
WD = "/neurospin/brainomics/2017_memento/analysis/WMH/models/wmh_memento_pcatv"
ANALYSIS_PATH = "/neurospin/brainomics/2017_memento/analysis/WMH"
ANALYSIS_DATA_PATH = os.path.join(ANALYSIS_PATH, "data")
PARTICIPANTS_CSV = os.path.join(ANALYSIS_PATH , "population.csv")


CONF = dict(clust_size_thres=20, NI="WMH", vs=1.5, shape=(121, 145, 121))


import nilearn.datasets
import brainomics.image_resample
mni152_t1_1mm_img = nibabel.load('/usr/share/data/fsl-mni152-templates/MNI152_T1_1mm.nii.gz')

coefs_filename = os.path.join(WD, "components-brain-maps.nii.gz")
coefs_img = nibabel.load(coefs_filename)
mni152_t1_15mm_img = resample_to_img(mni152_t1_1mm_img, coefs_img)
mni152_t1_15mm_img.to_filename(os.path.join(WD, "MNI152_T1_1.5mm.nii.gz"))

########################################################################################################################
mod = np.load(os.path.join(WD, "model.npz"))

U, d, V, PC, explained_variance = mod['U'], mod['d'], mod['V'], mod['PC'], mod['explained_variance']

NI_arr_msk = np.load(os.path.join(ANALYSIS_DATA_PATH, "%s_arr_msk.npy" % CONF["NI"]))
assert NI_arr_msk.shape == (1755, 116037)
wmhvol = NI_arr_msk.sum(axis=1)

pop = pd.read_csv(PARTICIPANTS_CSV)

assert pop.shape[0] == PC.shape[0]

pd.Series(wmhvol).describe()

thres = -np.inf
df = pop[wmhvol > thres]
df.shape

df["PC1"] = PC[wmhvol > thres, 0]
df["PC2"] = PC[wmhvol > thres, 1]
df["PC3"] = PC[wmhvol > thres, 2]
df["wmh_tot"] = wmhvol[wmhvol > thres]

#df[['participant_id', "PC1", "PC2", "PC3"]].to_csv(os.path.join(WD, "components_participants_id.csv"), index=False)

import seaborn as sns

# PCs vs wmh_tot
sns.scatterplot(x="PC1", y="wmh_tot", data=df)
sns.scatterplot(x="PC2", y="wmh_tot", data=df)
sns.scatterplot(x="PC3", y="wmh_tot", data=df)
smfrmla.ols("wmh_tot ~ PC1 + PC2 + PC3", data=df).fit().summary()

# PCs vs age
sns.scatterplot(x="PC1", y="age_cons", data=df)
sns.scatterplot(x="PC2", y="age_cons", data=df)
sns.scatterplot(x="PC3", y="age_cons", data=df)
smfrmla.ols("age_cons ~ PC1 + PC2 + PC3", data=df).fit().summary()

# PCs
sns.scatterplot(x="PC1", y="PC2", data=df)
sns.scatterplot(x="PC1", y="PC2", hue="mmssctot", data=df)

sns.lmplot(x="wmh_tot", y="mmssctot", data=df)
sns.lmplot(x="PC1", y="mmssctot", data=df)
sns.lmplot(x="PC2", y="mmssctot", data=df)

sns.lmplot(x="wmh_tot", y="tmta_taux", data=df)
sns.lmplot(x="PC1", y="tmta_taux", data=df)
sns.lmplot(x="PC2", y="tmta_taux", data=df)

sns.scatterplot(x="wmh_tot", y="flu_p", data=df)
sns.scatterplot(x="PC1", y="flu_p", data=df)
sns.scatterplot(x="PC2", y="flu_p", data=df)

sns.scatterplot(x="wmh_tot", y="flu_anim", data=df)
sns.scatterplot(x="PC1", y="flu_anim", data=df)
sns.scatterplot(x="PC2", y="flu_anim", data=df)

import statsmodels.formula.api as smfrmla
import statsmodels.api as sm


df["sex"] = df.sex.astype('object')

smfrmla.ols("mmssctot ~ PC1 + PC2 + PC3 + age_cons + sex", data=df).fit().summary()
smfrmla.ols("mmssctot ~ PC1 + PC2 + PC3 + sex", data=df).fit().summary()

smfrmla.ols("mmssctot ~ PC1 + sex + age_cons", data=df).fit().summary()
smfrmla.ols("mmssctot ~ PC2 + sex + age_cons", data=df).fit().summary()
smfrmla.ols("mmssctot ~ PC3 + sex + age_cons", data=df).fit().summary()

smfrmla.ols("tmta_taux ~ PC1 + PC2 + PC3 + age_cons + sex", data=df).fit().summary()
smfrmla.ols("tmta_taux ~ PC1 + PC2 + PC3 + sex", data=df).fit().summary()

smfrmla.ols("tmta_taux ~ PC1 + sex + age_cons", data=df).fit().summary()
smfrmla.ols("tmta_taux ~ PC2 + sex + age_cons", data=df).fit().summary()
smfrmla.ols("tmta_taux ~ PC3 + sex + age_cons", data=df).fit().summary()

smfrmla.ols("flu_p ~ PC1 + PC2 + PC3 + age_cons + sex", data=df).fit().summary()
smfrmla.ols("flu_p ~ PC1 + PC2 + PC3 + sex", data=df).fit().summary()

smfrmla.ols("flu_p ~ PC1 + sex + age_cons", data=df).fit().summary()
smfrmla.ols("flu_p ~ PC2 + sex + age_cons", data=df).fit().summary()
smfrmla.ols("flu_p ~ PC3 + sex + age_cons", data=df).fit().summary()

smfrmla.ols("flu_anim ~ PC1 + PC2 + PC3 + age_cons + sex", data=df).fit().summary()
smfrmla.ols("flu_anim ~ PC1 + PC2 + PC3 + sex", data=df).fit().summary()

smfrmla.ols("flu_anim ~ PC1 + sex + age_cons", data=df).fit().summary()
smfrmla.ols("flu_anim ~ PC2 + sex + age_cons", data=df).fit().summary()
smfrmla.ols("flu_anim ~ PC3 + sex + age_cons", data=df).fit().summary()

