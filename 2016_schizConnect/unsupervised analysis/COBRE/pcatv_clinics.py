
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
import matplotlib.pyplot as plt


INPUT_CLINIC_FILENAME = "/neurospin/abide/schizConnect/data/schizconnect_COBRE_assessmentData_1829.csv"
INPUT_POPULATION = "/neurospin/brainomics/2016_schizConnect/analysis/COBRE/VBM/population.csv"



#PROBLEME multiple scores for each subject!!
clinic = pd.read_csv(INPUT_CLINIC_FILENAME)
PANSS_MAP = {'Absent': 0, 'Minimal':1 ,'Mild': 2,'Moderate': 3,'Severe': 4,'Moderate severe': 4,'Extreme': 5,}



pop = pd.read_csv(INPUT_POPULATION)
pop["PANSS"] =  "NaN"

for s in pop.subjectid:
    print (s)
    curr = clinic[clinic.subjectid ==s]
    most_recent_visit = curr.visit.unique()[-1]
    curr_visit = curr[curr.visit == most_recent_visit]
    
    current_PANSS = curr_visit[curr_visit.assessment_description == "Positive and Negative Symptom Scale"].question_value
    current_PANSS = current_PANSS.map(PANSS_MAP).values

    if len(current_PANSS) != 0:
        pop.loc[pop.subjectid ==s,"PANSS"] = np.nansum(current_PANSS)
        print (current_PANSS.sum())

        
#investigate distribution of SAPS and SANS scores across SCZ population     
PANSS_scores =  pop[pop.dx_num ==1].PANSS.astype(np.float).values


scores_PCA_path = "/neurospin/brainomics/2016_schizConnect/analysis/COBRE/VBM/results/pcatv/5_folds_COBRE/results/0/struct_pca_0.1_0.1_0.5/X_train_transform.npz"
scores_comp = np.load(scores_PCA_path)['arr_0']

#Pearson correlation
pearsonr(scores_comp[:,0],PANSS_scores)


pearsonr(scores_comp[:,1],PANSS_scores)

pearsonr(scores_comp[:,2],PANSS_scores)

pearsonr(scores_comp[:,3],PANSS_scores)

pearsonr(scores_comp[:,4],PANSS_scores)


#COMPONENT 1
plt.plot(scores_comp[:,0],SAPS_scores,'o')
plt.xlabel('Score on component 1')
plt.ylabel('SAPS score')
plt.text(-0.05,74,"Pearson's correlation = -0.09",fontsize=12)

plt.plot(scores_comp[:,0],SANS_scores,'o')
plt.xlabel('Score on component 1')
plt.ylabel('SANS score')
plt.text(-0.05,74,"Pearson's correlation = 0.01",fontsize=12)

#COMPONENT 1
plt.plot(scores_comp[:,1],SAPS_scores,'o')
plt.xlabel('Score on component 2')
plt.ylabel('SAPS score')
plt.text(-0.05,74,"Pearson's correlation = -0.09",fontsize=12)

plt.plot(scores_comp[:,1],SANS_scores,'o')
plt.xlabel('Score on component 2')
plt.ylabel('SANS score')
plt.text(-0.05,74,"Pearson's correlation = 0.03",fontsize=12)


#COMPONENT 3
plt.plot(scores_comp[:,2],SAPS_scores,'o')
plt.xlabel('Score on component 3')
plt.ylabel('SAPS score')
plt.text(-0.05,74,"Pearson's correlation = -0.067",fontsize=12)

plt.plot(scores_comp[:,2],SANS_scores,'o')
plt.xlabel('Score on component 3')
plt.ylabel('SANS score')
plt.text(-0.05,74,"Pearson's correlation = -0.11",fontsize=12)


#COMPONENT 4
plt.plot(scores_comp[:,3],SAPS_scores,'o')
plt.xlabel('Score on component 4')
plt.ylabel('SAPS score')
plt.text(-0.05,74,"Pearson's correlation = -0.017",fontsize=12)

plt.plot(scores_comp[:,3],SANS_scores,'o')
plt.xlabel('Score on component 4')
plt.ylabel('SANS score')
plt.text(-0.05,74,"Pearson's correlation = 0.04",fontsize=12)



#COMPONENT 4
plt.plot(scores_comp[:,4],SAPS_scores,'o')
plt.xlabel('Score on component 5')
plt.ylabel('SAPS score')
plt.text(-0.15,74,"Pearson's correlation = 0.18, p = 0.04",fontsize=12)

plt.plot(scores_comp[:,4],SANS_scores,'o')
plt.xlabel('Score on component 5')
plt.ylabel('SANS score')
plt.text(-0.05,74,"Pearson's correlation = -0.02",fontsize=12)


