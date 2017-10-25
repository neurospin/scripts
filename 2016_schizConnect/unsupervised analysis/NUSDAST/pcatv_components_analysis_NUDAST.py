
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


###############################################################################
# SCZ ONLY
###############################################################################

INPUT_CLINIC_FILENAME = "/neurospin/abide/schizConnect/data/schizconnect_NUSDAST_assessmentData_1829.csv"
INPUT_POPULATION = "/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST/VBM/population.csv"
BASE_PATH = "/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST/VBM/results/pcatv_scz/5_folds_NUDAST_10comp"
INPUT_DATA_X = '/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST/VBM/data/X_scz_only.npy'
INPUT_RESULTS = os.path.join(BASE_PATH,"results","0")

X = np.load(INPUT_DATA_X)

# Compute clinical Scores

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


SAPS_scores =  pop[pop.dx_num ==1].SAPS.astype(np.float).values
SANS_scores =  pop[pop.dx_num ==1].SANS.astype(np.float).values
age =  pop[pop.dx_num ==1].age.values
sex =  pop[pop.dx_num ==1].sex_num.values


plt.plot(age,SAPS_scores,'o')
corr,p = scipy.stats.pearsonr(age,SAPS_scores)
plt.title("Pearson' s correlation = %.02f \n p = %.01e" % (corr,p),fontsize=12)
plt.xlabel("age")
plt.ylabel("SAPS score")

plt.plot(age,SANS_scores,'o')
corr,p = scipy.stats.pearsonr(age,SANS_scores)
plt.title("Pearson' s correlation = %.02f \n p = %.01e" % (corr,p),fontsize=12)
plt.xlabel("age")
plt.ylabel("SANS score")

scores = np.load(os.path.join(INPUT_RESULTS,"struct_pca_0.1_0.1_0.1","X_test_transform.npz"))['arr_0']


for i in range(scores.shape[1]):
#     corr,p = pearsonr(scores[:,i],SAPS_scores)
#     print(corr)
#     corr,p=pearsonr(scores[age<40,i],SAPS_scores[age<40])
#     print(corr)
#     if p < 0.05:
    print ("Significant correlation between SAPS score and score on component %s" % (i))
    plt.figure()
    plt.plot(scores[:,i],SAPS_scores,'o')
    plt.xlabel('Score on component %s' %(i))
    plt.ylabel('SAPS score')
    plt.title("Pearson's correlation = %.02f, p = %.01e" % (corr,p),fontsize=12)

    corr, p = pearsonr(scores[:,i],SANS_scores)
    if p < 0.05:
        print ("Significant correlation between SANS score and score on component %s" % (i))
        plt.figure()
        plt.plot(scores[:,i],SANS_scores,'o')
        plt.xlabel('Score on component %s' %(i))
        plt.ylabel('SANS score')
        plt.title("Pearson's correlation = %.02f, p = %.01e" % (corr,p),fontsize=12)

    corr,p = pearsonr(scores[:,i],age)
    if p < 0.05:
        print ("Significant correlation between age and score on component %s" % (i))
        plt.figure()
        plt.plot(scores[:,i],age,'o')
        plt.xlabel('Score on component %s' %(i))
        plt.ylabel('age')
        plt.title("Pearson's correlation = %.02f, p = %.01e" % (corr,p),fontsize=12)
###############################################################################
###############################################################################
# PLOT ALL CORRELATIONW WITH AGE
fig, axs = plt.subplots(2,5, figsize=(15, 6), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = 1.0, wspace=.3)
axs = axs.ravel()
for i in range(10):
    corr,p = pearsonr(scores[:,i],age)
    axs[i].plot(scores[sex==0,i],age[sex==0],'o', markersize = 4,label = "male")
    axs[i].plot(scores[sex==1,i],age[sex==1],'o', markersize = 4,label = "female")
    axs[i].set_title("Pearson' s correlation = %.02f \n p = %.01e" % (corr,p),fontsize=12)
    axs[i].xaxis.set_ticks(np.arange(-0.3,0.4,0.2))
    axs[i].set_xlabel('Score on component %s' %(i+1))
    axs[i].set_ylabel('age')
    axs[i].yaxis.set_ticks(np.arange(10,80,10))
    axs[i].legend()
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')
plt.tight_layout()
plt.savefig("/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST/VBM/results/pcatv_scz/5_folds_NUDAST_10comp/correlation_Age.pdf")
plt.savefig("/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST/VBM/results/pcatv_scz/5_folds_NUDAST_10comp/correlation_Age.png")
###############################################################################

###############################################################################
 # PLOT ALL CORRELATIONW WITH SAPS
fig, axs = plt.subplots(2,5, figsize=(15, 6), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = 1.0, wspace=.3)
axs = axs.ravel()
for i in range(10):
    corr,p = pearsonr(scores[:,i],SAPS_scores)
    axs[i].plot(scores[sex==0,i],SAPS_scores[sex==0],'o', markersize = 4,label = "male")
    axs[i].plot(scores[sex==1,i],SAPS_scores[sex==1],'o', markersize = 4,label = "female")
    axs[i].set_title("Pearson' s correlation = %.02f \n p = %.01e" % (corr,p),fontsize=12)
    axs[i].xaxis.set_ticks(np.arange(-0.3,0.4,0.2))
    axs[i].set_xlabel('Score on component %s' %(i+1))
    axs[i].set_ylabel('SAPS score')
    axs[i].yaxis.set_ticks(np.arange(10,80,10))
plt.tight_layout()
plt.savefig("/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST/VBM/results/pcatv_scz/5_folds_NUDAST_10comp/correlation_SAPSS.pdf")
plt.savefig("/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST/VBM/results/pcatv_scz/5_folds_NUDAST_10comp/correlation_SAPSS.png")

#SAPS, MEN ONLY
fig, axs = plt.subplots(2,5, figsize=(15, 6), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = 1.0, wspace=.3)
axs = axs.ravel()
for i in range(10):
    corr,p = pearsonr(scores[sex==0,i],SAPS_scores[sex==0])
    axs[i].plot(scores[sex==0,i],SAPS_scores[sex==0],'o', markersize = 4,label = "male")
    axs[i].set_title("Pearson corr MALE ONLY = %.02f \n p = %.01e" % (corr,p),fontsize=12)
    axs[i].xaxis.set_ticks(np.arange(-0.3,0.4,0.2))
    axs[i].set_xlabel('Score on component %s' %(i+1))
    axs[i].set_ylabel('SAPS score')
    axs[i].yaxis.set_ticks(np.arange(10,80,10))
plt.tight_layout()

############################
corr,p = pearsonr(scores[sex==0,4],SAPS_scores[sex==0])
plt.title("Pearson corr MALE ONLY = %.02f \n p = %.01e" % (corr,p),fontsize=12)
plt.plot(scores[sex==0,4],SAPS_scores[sex==0],'o', markersize = 4,label = "male")
plt.xlabel("score component 4")
plt.ylabel("SAPS_score")

corr,p = pearsonr(scores[sex==0,4],SANS_scores[sex==0])
plt.title("Pearson corr MALE ONLY = %.02f \n p = %.01e" % (corr,p),fontsize=12)
plt.plot(scores[sex==0,4],SANS_scores[sex==0],'o', markersize = 4,label = "male")
plt.xlabel("score component 4")
plt.ylabel("SANS_score")
#########################
#3D plot
fig = pyplot.figure()
ax = Axes3D(fig)

sequence_containing_x_vals = scores[sex==0,4]
sequence_containing_y_vals = SANS_scores[sex==0]
sequence_containing_z_vals = SAPS_scores[sex==0]


ax.scatter(sequence_containing_x_vals, sequence_containing_y_vals, sequence_containing_z_vals)
ax.set_xlabel('scores on component 4')
ax.set_ylabel('SANS scores')
ax.set_zlabel('SAPS scores')
pyplot.show()

############################
#SAPS, FEMALE ONLY
fig, axs = plt.subplots(2,5, figsize=(15, 6), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = 1.0, wspace=.3)
axs = axs.ravel()
for i in range(10):
    corr,p = pearsonr(scores[sex==1,i],SAPS_scores[sex==1])
    axs[i].plot(scores[sex==1,i],SAPS_scores[sex==1],'o', markersize = 4,label = "male")
    axs[i].set_title("Pearson corr FEMALE ONLY = %.02f \n p = %.01e" % (corr,p),fontsize=12)
    axs[i].xaxis.set_ticks(np.arange(-0.3,0.4,0.2))
    axs[i].set_xlabel('Score on component %s' %(i+1))
    axs[i].set_ylabel('SAPS score')
    axs[i].yaxis.set_ticks(np.arange(10,80,10))
plt.tight_layout()


 # PLOT ALL CORRELATIONW WITH SANS
fig, axs = plt.subplots(2,5, figsize=(15, 6), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = 1.0, wspace=.3)
axs = axs.ravel()
for i in range(10):
    corr,p = pearsonr(scores[:,i],SANS_scores)
    axs[i].plot(scores[sex==0,i],SANS_scores[sex==0],'o', markersize = 4,label = "male")
    axs[i].plot(scores[sex==1,i],SANS_scores[sex==1],'o', markersize = 4,label = "female")
    axs[i].set_title("Pearson' s correlation = %.02f \n p = %.01e" % (corr,p),fontsize=12)
    axs[i].xaxis.set_ticks(np.arange(-0.3,0.4,0.2))
    axs[i].set_xlabel('Score on component %s' %(i+1))
    axs[i].set_ylabel('SANS score')
    axs[i].yaxis.set_ticks(np.arange(10,80,10))
plt.tight_layout()
plt.savefig("/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST/VBM/results/pcatv_scz/5_folds_NUDAST_10comp/correlation_SANSS.pdf")
plt.savefig("/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST/VBM/results/pcatv_scz/5_folds_NUDAST_10comp/correlation_SANSS.png")


#SAPS, MEN ONLY
fig, axs = plt.subplots(2,5, figsize=(15, 6), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = 1.0, wspace=.3)
axs = axs.ravel()
for i in range(10):
    corr,p = pearsonr(scores[sex==0,i],SANS_scores[sex==0])
    axs[i].plot(scores[sex==0,i],SANS_scores[sex==0],'o', markersize = 4,label = "male")
    axs[i].set_title("Pearson corr MALE ONLY = %.02f \n p = %.01e" % (corr,p),fontsize=12)
    axs[i].xaxis.set_ticks(np.arange(-0.3,0.4,0.2))
    axs[i].set_xlabel('Score on component %s' %(i+1))
    axs[i].set_ylabel('SANS score')
    axs[i].yaxis.set_ticks(np.arange(10,80,10))
plt.tight_layout()

#SAPS, FEMALE ONLY
fig, axs = plt.subplots(2,5, figsize=(15, 6), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = 1.0, wspace=.3)
axs = axs.ravel()
for i in range(10):
    corr,p = pearsonr(scores[sex==1,i],SANS_scores[sex==1])
    axs[i].plot(scores[sex==1,i],SANS_scores[sex==1],'o', markersize = 4,label = "male")
    axs[i].set_title("Pearson corr FEMALE ONLY = %.02f \n p = %.01e" % (corr,p),fontsize=12)
    axs[i].xaxis.set_ticks(np.arange(-0.3,0.4,0.2))
    axs[i].set_xlabel('Score on component %s' %(i+1))
    axs[i].set_ylabel('SANS score')
    axs[i].yaxis.set_ticks(np.arange(10,80,10))
plt.tight_layout()
























###############################################################################
# CONTROLS ONLY
###############################################################################

INPUT_CLINIC_FILENAME = "/neurospin/abide/schizConnect/data/schizconnect_NUSDAST_assessmentData_1829.csv"
INPUT_POPULATION = "/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST/VBM/population.csv"
BASE_PATH = "/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST/VBM/results/pcatv_controls/5_folds_NUDAST_controls"
INPUT_DATA_X = '/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST/VBM/data/X_controls_only.npy'
INPUT_RESULTS = os.path.join(BASE_PATH,"results","0")

X = np.load(INPUT_DATA_X)

# Compute clinical Scores

clinic = pd.read_csv(INPUT_CLINIC_FILENAME)
pop = pd.read_csv(INPUT_POPULATION)

age_controls =  pop[pop.dx_num ==0].age.values


scores_controls = np.load(os.path.join(INPUT_RESULTS,"struct_pca_0.1_0.1_0.1","X_test_transform.npz"))['arr_0']


###############################################################################




#we corrected the DTI indices for age before building the statistical atlas.
#The correction was performed with a Generalized Linear Model fitting procedure
# to estimate the linear regression between age and DTI indices in normal
# subjects only (n = 47) [29], We found 47/145 regions showed statistically
#significant correlations (p<0.05). The extracted regression coefficients were
# used to correct the DTI indices for age in all subjects (including mTBI participants).

 #############################################"
# Try to apply a Correction of score for age and then to correlate with PANSS














