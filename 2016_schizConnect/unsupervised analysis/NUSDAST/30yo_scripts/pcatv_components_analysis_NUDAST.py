
import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import pandas as pd
import nibabel as nib
import json
from nilearn import plotting
from nilearn import image
from scipy.stats import pearsonr
################
# Input/Output #
################

INPUT_BASE_DIR = '/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST/VBM/results_30yo/pcatv/pcatv_NUDAST_30yo'
INPUT_DIR = os.path.join(INPUT_BASE_DIR,"results")
INPUT_MASK = '/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST/VBM/data/data_30yo/mask.nii'              


INPUT_CLINIC_FILENAME = "/neurospin/abide/schizConnect/data/schizconnect_NUSDAST_assessmentData_1829.csv"
INPUT_POPULATION = "/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST/VBM/population_30yo.csv"



#PROBLEME multiple scores for each subject!!
clinic = pd.read_csv(INPUT_CLINIC_FILENAME)
pop = pd.read_csv(INPUT_POPULATION)
pop["SAPS"] =  "NaN"
pop["SANS"] =  "NaN"
for s in pop.subjectid:
    print (s)
    curr = clinic[clinic.subjectid ==s]
    most_recent_visit = curr.visit.unique()[-1]
    curr = curr[curr.visit == most_recent_visit]
    current_SAPS = curr[curr.assessment_description == "Scale for the Assessment of Positive Symptoms"].question_value.astype(np.int64).values
    current_SANS = curr[curr.assessment_description == "Scale for the Assessment of Negative Symptoms"].question_value.astype(np.int64).values           
    if len(current_SANS) != 0:
        pop.loc[pop.subjectid ==s,"SAPS"] = current_SAPS.sum()
        print (current_SAPS.sum())
    if len(current_SAPS) != 0:    
        pop.loc[pop.subjectid ==s,"SANS"] = current_SANS.sum()
    

        
#investigate distribution of SAPS and SANS scores across SCZ population     
SAPS_scores =  pop[pop.dx_num ==1].SAPS.astype(np.float).values
SANS_scores =  pop[pop.dx_num ==1].SANS.astype(np.float).values  


scores_PCA_path = "/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST/VBM/results_30yo/pcatv/pcatv_NUDAST_30yo/results/0/struct_pca_0.1_0.5_0.1/X_train_transform.npz"
scores_comp = np.load(scores_PCA_path)['arr_0']

#Pearson correlation
pearsonr(scores_comp[:,0],SAPS_scores)
pearsonr(scores_comp[:,0],SANS_scores)

pearsonr(scores_comp[:,1],SAPS_scores)
pearsonr(scores_comp[:,1],SANS_scores)

pearsonr(scores_comp[:,2],SAPS_scores)
pearsonr(scores_comp[:,2],SANS_scores)

pearsonr(scores_comp[:,3],SAPS_scores)
pearsonr(scores_comp[:,3],SANS_scores)

pearsonr(scores_comp[:,4],SAPS_scores)
pearsonr(scores_comp[:,4],SANS_scores)


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


