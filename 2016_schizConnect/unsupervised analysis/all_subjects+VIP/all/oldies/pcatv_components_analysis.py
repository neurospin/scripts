
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
################
# Input/Output #
################

INPUT_BASE_DIR = '/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST/VBM/results/pcatv'
INPUT_DIR = os.path.join(INPUT_BASE_DIR,"5_folds_NUDAST","results")
INPUT_MASK = '/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST/VBM/data/mask.nii'              


INPUT_CLINIC_FILENAME = "/neurospin/abide/schizConnect/data/schizconnect_NUSDAST_assessmentData_1829.csv"
INPUT_POPULATION = "/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST/VBM/population.csv"

X = np.load("/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST/VBM/data/X.npy")
X_scz = np.load("/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST/VBM/data/X_scz_only.npy")
y = np.load("/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST/VBM/data/y.npy")


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
age =  pop[pop.dx_num ==1].age.values


pred=linear_model.LinearRegression().fit(age.reshape(-1, 1),SAPS_scores.reshape(-1, 1)).predict(age.reshape(-1, 1))
res_SAPS= SAPS_scores.reshape(-1, 1)-pred
res_SAPS = res_SAPS.reshape(118)
pred=linear_model.LinearRegression().fit(age.reshape(-1, 1),SANS_scores.reshape(-1, 1)).predict(age.reshape(-1, 1))

res_SANS= SANS_scores.reshape(-1, 1)-pred
res_SANS = res_SANS.reshape(118)


pearsonr(scores_comp[:,0],res_SAPS)
pearsonr(scores_comp[:,0],res_SANS)
pearsonr(scores_comp[:,0],SAPS_scores)
pearsonr(scores_comp[:,0],SANS_scores)

pearsonr(scores_comp[:,1],res_SAPS)
pearsonr(scores_comp[:,1],res_SANS)
pearsonr(scores_comp[:,1],SAPS_scores)
pearsonr(scores_comp[:,1],SANS_scores)

pearsonr(scores_comp[:,2],res_SAPS)
pearsonr(scores_comp[:,2],res_SANS)
pearsonr(scores_comp[:,2],SAPS_scores)
pearsonr(scores_comp[:,2],SANS_scores)

pearsonr(scores_comp[:,3],res_SAPS)
pearsonr(scores_comp[:,3],res_SANS)
pearsonr(scores_comp[:,3],SAPS_scores)
pearsonr(scores_comp[:,3],SANS_scores)


pearsonr(scores_comp[:,4],res_SAPS)
pearsonr(scores_comp[:,4],res_SANS)
pearsonr(scores_comp[:,4],SAPS_scores)
pearsonr(scores_comp[:,4],SANS_scores)



scores_PCA_path = "/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST/\
VBM/results/pcatv_10comp/5_folds_NUDAST_10comp/results/0/struct_pca_0.1_0.1_0.1/X_train_transform.npz"
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

pearsonr(scores_comp[:,5],SAPS_scores)
pearsonr(scores_comp[:,5],SANS_scores)


pearsonr(scores_comp[:,6],SAPS_scores)
pearsonr(scores_comp[:,6],SANS_scores)

pearsonr(scores_comp[:,7],SAPS_scores)
pearsonr(scores_comp[:,7],SANS_scores)

pearsonr(scores_comp[:,8],SAPS_scores)
pearsonr(scores_comp[:,8],SANS_scores)

pearsonr(scores_comp[:,9],SAPS_scores)
pearsonr(scores_comp[:,9],SANS_scores)





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


