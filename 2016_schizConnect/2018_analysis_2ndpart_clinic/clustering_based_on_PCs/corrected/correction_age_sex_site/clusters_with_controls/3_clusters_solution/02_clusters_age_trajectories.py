
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib
import pandas as pd
import nibabel as nib
import json
from nilearn import plotting
from nilearn import image
from scipy.stats.stats import pearsonr
import shutil
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
import parsimony.utils.check_arrays as check_arrays
from sklearn import preprocessing
import statsmodels.api as sm
from statsmodels.formula.api import ols
import seaborn as sns

DATA_PATH = "/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/data"
y_all = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/y.npy")


pop = pd.read_csv("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/population.csv")
age = pop["age"].values
sex = pop["sex_num"].values

y = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/y.npy")
site = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/site.npy")
labels_cluster = np.load("/neurospin/brainomics/2016_schizConnect/\
2018_analysis_2ndpart_clinic/results/clustering/corrected_results/\
correction_age_sex_site/clusters_with_controls/2_clusters_solution/labels_cluster.npy")


U_all = np.load("/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/results/clustering/U_scores_corrected/U_all.npy")
U_all = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/results/pcatv_scz/data/U_all.npy")
U0 = U_all[:,0]
################################################################################


#Plot
df_con = pd.DataFrame()
df_con["Age"] = age[y_all==0]
df_con["Age2"] = (age[y_all==0])*(age[y_all==0])
df_con["sex"] = sex[y_all==0]
df_con["site"] = site[y_all==0]
df_con["U0"] = U0[y_all==0]

df_scz = pd.DataFrame()
df_scz["Age"] = age[y_all==1]
df_scz["Age2"] = (age[y_all==1])*(age[y_all==1])
df_scz["sex"] = sex[y_all==1]
df_scz["site"] = site[y_all==1]
df_scz["U0"] = U0[y_all==1]


df_clust1 = pd.DataFrame()
df_clust1["Age"] = age[y_all==1][labels_cluster==0]
df_clust1["Age2"] = (age[y_all==1][labels_cluster==0])*(age[y_all==1][labels_cluster==0])
df_clust1["sex"] = sex[y_all==1][labels_cluster==0]
df_clust1["site"] = site[y_all==1][labels_cluster==0]
df_clust1["U0"] = U0[y_all==1][labels_cluster==0]
for i in range(1,11):
    df_clust1["Score on comp U%s"%i] = U_all[y_all==1][labels_cluster==0][:,i-1]

df_clust2 = pd.DataFrame()
df_clust2["Age"] = age[y_all==1][labels_cluster==1]
df_clust2["Age2"] = (age[y_all==1][labels_cluster==1])*(age[y_all==1][labels_cluster==1])
df_clust2["sex"] = sex[y_all==1][labels_cluster==1]
df_clust2["site"] = site[y_all==1][labels_cluster==1]
df_clust2["U0"] = U0[y_all==1][labels_cluster==1]
for i in range(1,11):
    df_clust2["Score on comp U%s"%i] = U_all[y_all==1][labels_cluster==1][:,i-1]



output = "/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/\
results/clustering/corrected_results/correction_age_sex_site/\
clusters_with_controls/2_clusters_solution/age/corrected_U"
sns.set(color_codes=True)
plt.figure()
sns.regplot(x="Age", y="U0", data=df_clust1,label= "SCZ Cluster 1",marker='o')
sns.regplot(x="Age", y="U0", data=df_clust2,label="SCZ Cluster 2",marker='d')
plt.legend()
plt.savefig(os.path.join(output,"U0_vs_age.png"))

################################################################################
output = "/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/\
results/clustering/corrected_results/correction_age_sex_site/\
clusters_with_controls/2_clusters_solution/age/non_corrected_U"
sns.set(color_codes=True)
plt.figure()
sns.regplot(x="Age", y="U0", data=df_clust1,label= "SCZ Cluster 1",marker='o')
sns.regplot(x="Age", y="U0", data=df_clust2,label="SCZ Cluster 2",marker='d')
plt.legend()
plt.savefig(os.path.join(output,"U0_vs_age.png"))


for i in range(1,11):
    plt.figure()
    sns.regplot(x="Age", y="Score on comp U%s"%i, data=df_clust1,label= "SCZ Cluster 1",marker='o')
    sns.regplot(x="Age", y="Score on comp U%s"%i, data=df_clust2,label="SCZ Cluster 2",marker='d')
    plt.legend()
    plt.savefig(os.path.join(output,"comp%s"%i))