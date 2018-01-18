#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 15:42:03 2017

@author: ad247405
"""


import os
import numpy as np
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

INPUT_CLINIC_FILENAME = "/neurospin/abide/schizConnect/data/december_2017_clinical_score/schizconnect_NUSDAST_assessmentData_4495.csv"
INPUT_POPULATION = "/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST/VBM/population.csv"
WD = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/results/pcatv_scz/data"

U_nudast = np.load(os.path.join(WD,"U_nudast.npy"))
U_nudast_scz = np.load(os.path.join(WD,"U_nudast_scz.npy"))
U_nudast_con = np.load(os.path.join(WD,"U_nudast_con.npy"))

y_nudast = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/data_by_site/NUSDAST/y.npy")
X_nudast = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/data_by_site/NUSDAST/X.npy")


clinic = pd.read_csv(INPUT_CLINIC_FILENAME)
pop = pd.read_csv(INPUT_POPULATION)
age = pop["age"].values
sex = pop["sex_num"].values



df_scores = pd.DataFrame()
df_scores["subjectid"] = pop.subjectid
for score in clinic.question_id.unique():
    df_scores[score] = np.nan

for s in pop.subjectid:
    curr = clinic[clinic.subjectid ==s]
    for key in clinic.question_id.unique():
        if curr[curr.question_id == key].empty == False:
            df_scores.loc[df_scores["subjectid"]== s,key] = curr[curr.question_id == key].question_value.values[0]




# Turn interactive plotting off
plt.plot(age[y_nudast==0],df_scores["matrxraw"].values[y_nudast==0],'o')
plt.plot(age[y_nudast==1],df_scores["matrxraw"].values[y_nudast==1],'o')

################################################################################
for key in clinic.question_id.unique():
    print("%s" %(key))
    output = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/results/pcatv_scz/results/projection_nudast/scores/%s" %key
    if os.path.isdir(output) == False:
        os.makedirs(output)
        neurospycho = df_scores[key].astype(np.float).values[y_nudast==1]
        for i in range(10):
            print(i+1)
            df = pd.DataFrame()
            df["neurospycho"] = neurospycho[np.array(np.isnan(neurospycho)==False)]
            df["age"] = age[y_nudast==1][np.array(np.isnan(neurospycho)==False)]
            df["sex"] = sex[y_nudast==1][np.array(np.isnan(neurospycho)==False)]
            df["U"] = U_nudast_scz[:,i][np.array(np.isnan(neurospycho)==False)]
            mod = ols("U ~ neurospycho +age+sex",data = df).fit()
            print(mod.pvalues["neurospycho"])
            fig = plt.figure(figsize=(10,6))
            fig = sm.graphics.plot_partregress_grid(mod, fig=fig)
            plt.figtext(0.1,-0.1,"neurospycho effect on U:\n Tvalue = %s \n pvalue = %s"
                  %(mod.tvalues["neurospycho"],mod.pvalues["neurospycho"]))

            plt.figtext(0.7, -0.1,"age effect on U:\n Tvalue = %s \n pvalue = %s"
                  %(mod.tvalues["age"],mod.pvalues["age"]))
            plt.figtext(0.4,-0.1,"sex effect on U:\n Tvalue = %s \n pvalue = %s"
                  %(mod.tvalues["sex"],mod.pvalues["sex"]))
            plt.tight_layout()
            plt.savefig(os.path.join(output,"comp%s"%(i+1)),bbox_inches = 'tight')
            plt.close(fig)

################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
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


INPUT_CLINIC_FILENAME = "/neurospin/abide/schizConnect/data/december_2017_clinical_score/schizconnect_NUSDAST_assessmentData_4495.csv"
INPUT_POPULATION = "/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST/VBM/population.csv"
WD = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/results/pcatv_scz/data"

U_nudast = np.load(os.path.join(WD,"U_nudast.npy"))
U_nudast_scz = np.load(os.path.join(WD,"U_nudast_scz.npy"))
U_nudast_con = np.load(os.path.join(WD,"U_nudast_con.npy"))

y_nudast = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/data_by_site/NUSDAST/y.npy")
X_nudast = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/data_by_site/NUSDAST/X.npy")


clinic = pd.read_csv(INPUT_CLINIC_FILENAME)
pop = pd.read_csv(INPUT_POPULATION)
pop = pop[pop["dx_num"]==1]
age = pop["age"].values
sex = pop["sex_num"].values

pop["PC1_whole_brain"] = U_nudast_scz[:, 0]
pop["PC2_caudate_putamen"] = U_nudast_scz[:,1]
pop["PC3_hippocampus_amygdala"] = U_nudast_scz[:, 2]
pop["PC4_temporal_gyrus"] = U_nudast_scz[:, 3]
pop["PC5_frontal_orbital_cortex"] = U_nudast_scz[:, 4]
pop["PC6_thalamus"] = U_nudast_scz[:, 5]
pop["PC7_left_thalamus"] = U_nudast_scz[:, 6]
pop["PC8_occipital"] = U_nudast_scz[:, 7]
pop["PC9_superior_frontal_gyrus"] = U_nudast_scz[:, 8]
pop["PC10_heschl_gyrus"] = U_nudast_scz[:, 9]


df_scores = pd.DataFrame()
df_scores["subjectid"] = pop.subjectid
for score in clinic.question_id.unique():
    df_scores[score] = np.nan

for s in pop.subjectid:
    curr = clinic[clinic.subjectid ==s]
    for key in clinic.question_id.unique():
        if curr[curr.question_id == key].empty == False:
            df_scores.loc[df_scores["subjectid"]== s,key] = curr[curr.question_id == key].question_value.values[0]
    print(s)


clusters = 'PC1_whole_brain', 'PC2_caudate_putamen',\
       'PC3_hippocampus_amygdala', 'PC4_temporal_gyrus',\
       'PC5_frontal_orbital_cortex', 'PC6_thalamus',\
       'PC7_left_thalamus', 'PC8_occipital',\
       'PC9_superior_frontal_gyrus', 'PC10_heschl_gyrus'


df_stats = pd.DataFrame(columns=clusters)
df_stats.insert(0,"clinical_scores",clinic.question_id.unique())
################################################################################
output = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/results/pcatv_scz/results/projection_nudast/PC_clinics_p_values.csv"
for key in clinic.question_id.unique():
    try:
        neurospycho = df_scores[key].astype(np.float).values
        for clust in clusters:
            print(clust)
            df = pd.DataFrame()
            df[key] = neurospycho[np.array(np.isnan(neurospycho)==False)]
            df["age"] = age[np.array(np.isnan(neurospycho)==False)]
            df["sex"] = sex[np.array(np.isnan(neurospycho)==False)]
            df[clust] = pop[clust][np.array(np.isnan(neurospycho)==False)].values
            mod = ols("%s ~ %s +age+sex"%(clust,key),data = df).fit()
            print(mod.pvalues[key])
            df_stats.loc[df_stats.clinical_scores==key,clust] = mod.pvalues[key]

    except:
            print("issue")
            df_stats.loc[df_stats.clinical_scores==key,clust] = np.nan


df_stats.to_csv(output)
