
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
INPUT_CLINIC_FILENAME = "/neurospin/abide/schizConnect/data/december_2017_clinical_score/schizconnect_NUSDAST_assessmentData_4495.csv"
U_all = np.load("/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/results/clustering/U_scores_corrected/U_all.npy")
y_all = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/y.npy")


pop = pd.read_csv(os.path.join(DATA_PATH,"pop_nudast_scz.csv"))
clinic = pd.read_csv(INPUT_CLINIC_FILENAME)
age = pop["age"].values
sex = pop["sex_num"].values

y = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/y.npy")
site = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/site.npy")
site = site[y==1]
labels_cluster = np.load("/neurospin/brainomics/2016_schizConnect/\
2018_analysis_2ndpart_clinic/results/clustering/corrected_results/\
correction_age_sex_site/clusters_with_controls/2_clusters_solution/labels_cluster.npy")
labels_cluster = labels_cluster[site==3]

U0 =U_all[:,0][y_all==1][site==3]


df_scores = pd.DataFrame()
df_scores["subjectid"] = pop.subjectid
for score in clinic.question_id.unique():
    df_scores[score] = np.nan

for s in pop.subjectid:
    curr = clinic[clinic.subjectid ==s]
    for key in clinic.question_id.unique():
        if curr[curr.question_id == key].empty == False:
            df_scores.loc[df_scores["subjectid"]== s,key] = curr[curr.question_id == key].question_value.values[0]


################################################################################

df_stats = pd.DataFrame(columns=["r","p"])
df_stats.insert(0,"clinical_scores",clinic.question_id.unique())
################################################################################
output = "/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/\
results/clustering/corrected_results/correction_age_sex_site/clusters_with_controls/2_clusters_solution/nudast\
correlation_clinics_p_values.csv"
for key in clinic.question_id.unique():
    try:
        neurospycho = df_scores[key].astype(np.float).values

        df = pd.DataFrame()
        df[key] = neurospycho[np.array(np.isnan(neurospycho)==False)]
        df["age"] = age[np.array(np.isnan(neurospycho)==False)]
        df["sex"] = sex[np.array(np.isnan(neurospycho)==False)]
        df["labels"]=labels_cluster[np.array(np.isnan(neurospycho)==False)]
        df["U0"]=U0[np.array(np.isnan(neurospycho)==False)]
        r,p = scipy.stats.pearsonr(df["U0"],df[key])
        print(p)
        df_stats.loc[df_stats.clinical_scores==key,"r"] = r
        df_stats.loc[df_stats.clinical_scores==key,"p"] = p

    except:
        print("issue")
        df_stats.loc[df_stats.clinical_scores==key,"T"] = np.nan
        df_stats.loc[df_stats.clinical_scores==key,"p"] = np.nan
df_stats.to_csv(output)


plt.plot(df["U0"],df[key],'o')

plt.plot(df[df["labels"]==0]["U0"],df[df["labels"]==0][key],'o')
plt.plot(df[df["labels"]==1]["U0"],df[df["labels"]==1][key],'o')
plt.xlabel("U0")
plt.ylabel(key)


df_scores["sansSUbtot"] = df_scores["sans8"].astype(np.float).values+df_scores["sans13"].astype(np.float).values+\
df_scores["sans17"].astype(np.float).values+df_scores["sans22"].astype(np.float).values+\
df_scores["sans25"].astype(np.float).values

df_scores["sapsSUbtot"] = df_scores["saps7"].astype(np.float).values+df_scores["saps20"].astype(np.float).values+\
df_scores["saps25"].astype(np.float).values+df_scores["saps34"].astype(np.float).values


################################################################################
output = "/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/\
results/clustering/corrected_results/correction_age_sex_site/clusters_with_controls/\
2_clusters_solution/nudast/correlations"

key = "vocabsca"
key = "matrxsca"
key = "cvlfps"
key = "yrschool"
key = "sstscalc"
key = "lnsscalc"
key = "dstscalc"
key = "lmb1rimr"


key = "sansSUbtot"
key = "sapsSUbtot"

df = pd.DataFrame()
neurospycho = df_scores[key].astype(np.float).values
df[key] = neurospycho[np.array(np.isnan(neurospycho)==False)]
df["age"] = age[np.array(np.isnan(neurospycho)==False)]
df["sex"] = sex[np.array(np.isnan(neurospycho)==False)]
df["labels"]=labels_cluster[np.array(np.isnan(neurospycho)==False)]
df["U0"]=U0[np.array(np.isnan(neurospycho)==False)]
r,p = scipy.stats.pearsonr(df["U0"],df[key])
plt.plot(df[df["labels"]==0]["U0"],df[df["labels"]==0][key],'o',label="SCZ Cluster 1")
plt.plot(df[df["labels"]==1]["U0"],df[df["labels"]==1][key],'o',label="SCZ Cluster 2")
plt.xlabel("U0")
plt.ylabel(key)
plt.title("Pearson corr: R = %s, and  p= %s"%(r,p))
plt.legend()
plt.savefig(os.path.join(output,"%s.png"%key))

df['color']= np.where( df['labels']==True , "r", "g")
ax = sns.regplot(data = df, x ="U0",y=key,scatter_kws={'facecolors':df['color'],'s':50})
ind = 0
for i, grp in df.groupby(['color']):
    grp.plot(kind = 'scatter', x = 'x', y = 'y', c = i, ax = ax, label = labels[ind], zorder = 0)
    ind += 1
ax.legend(loc=2)


#from scipy import stats
#slope, intercept, r_value, p_value, std_err = stats.linregress(df["U0"],df[key])
#print(slope, intercept, r_value, p_value, std_err)
