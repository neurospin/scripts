import os
import json
import numpy as np
import pandas as pd
from brainomics import array_utils
import matplotlib.pyplot as plt
import scipy.stats
import seaborn as sns, matplotlib.pyplot as plt
import scipy.stats
import nibabel
import scipy.stats

INPUT_DATA_X = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/\
VBM/all_subjects/data/mean_centered_by_site_all/X.npy"

INPUT_DATA_y = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/\
VBM/all_subjects/data/mean_centered_by_site_all/y.npy"

POPULATION_CSV = "/neurospin/brainomics/2016_schizConnect/analysis/VIP/VBM/population_and_scores.csv"

pop = pd.read_csv(POPULATION_CSV)
age = pop["age"].values


X = np.load(INPUT_DATA_X)
y = np.load(INPUT_DATA_y).ravel()
site = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/site.npy")

X_vip = X[site==4,:]
y_vip = y[site==4]
X_vip_scz = X_vip[y_vip==1,2:]
assert X_vip_scz.shape == (39, 125959)

MASK_PATH = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/mask.nii"
babel_mask  = nibabel.load(MASK_PATH)
mask_bool = babel_mask.get_data()
mask_bool= np.array(mask_bool !=0)
number_features = mask_bool.sum()


WD = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/\
results/enetall_all+VIP_all/5cv/refit/refit/enettv_0.1_0.1_0.8"
beta = np.load(os.path.join(WD,"beta.npz"))['arr_0'][3:]


CLUSTER_LABELS = "/neurospin/brainomics/2016_schizConnect/analysis/\
all_studies+VIP/VBM/all_subjects/results/enetall_all+VIP_all/5cv/refit/refit/\
enettv_0.1_0.1_0.8/weight_map_clust_labels.nii.gz"


labels_img  = nibabel.load(CLUSTER_LABELS)
labels_arr = labels_img.get_data()
labels_flt = labels_arr[mask_bool]


N = X_vip_scz.shape[0]
N_all = X_vip.shape[0]

# Extract a single score for each cluster
K = len(np.unique(labels_flt))  # nb cluster
scores_vip_scz = np.zeros((N, K))
scores_vip_all = np.zeros((N_all, K))

K_interest = [18,14,33,20,4,25,23,22,15,41]

for k in range (K):
    mask = labels_flt == k
    print("Cluster:",k, "size:", mask.sum())
    scores_vip_scz[:, k] = np.dot(X_vip_scz[:, mask], beta[mask]).ravel()
    scores_vip_all[:, k] = np.dot(X_vip[:, mask], beta[mask]).ravel()


decision_function_vip = np.dot(X_vip_scz,beta).ravel()

#Antipsychotic - Age of onset
##############################################################################
age_onset_antipsychotic = pop["TTT_VIE_NEURO_AAO"].values[y_vip==1]
age_onset_atypical_antipsychotic = pop["TTT_VIE_ANTIPSY_AAO"].values[y_vip==1]

df = pd.DataFrame()
df["age_onset_antipsychotic"] = age_onset_antipsychotic
df["age_onset_atypical_antipsychotic"] = age_onset_atypical_antipsychotic
df = df.fillna(999)
df = df.replace(to_replace = "NC",value = 999)
df = df.astype(float)
df["age_onset"] = df[["age_onset_antipsychotic","age_onset_atypical_antipsychotic"]].min(axis=1)

age_onset = df["age_onset"].values
np.save("/neurospin/brainomics/2016_schizConnect/analysis/VIP/VBM/data/treatment/age_onset_treatment.npy",age_onset)
scores_vip_scz = scores_vip_scz[age_onset!= 999,:]
age_onset = age_onset[age_onset!= 999]


#Plot age correlations
for i in K_interest:
    corr,p = scipy.stats.pearsonr(scores_vip_scz[:,i],age_onset)
    plt.figure()
    plt.plot(scores_vip_scz[:,i] ,age_onset,'o')
    plt.title("Pearson' s correlation = %.02f \n p = %.01e" % (corr,p),fontsize=12)
    plt.xlabel('Score on cluster %s' %(i))
    plt.ylabel('Age of treatment onset')
    plt.tight_layout()
    plt.savefig("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/results/clinic_correlations/age_of_onset/cluster%r.png"%(i))


##############################################################################


#Antipsychotic - dose received per day
##############################################################################
pop = pop[pop["diagnostic"]==3].reset_index()
pop["dose_1"] = 0
pop["dose_2"] = 0
pop["dose_3"] = 0
pop["dose_4"] = 0
pop["dose_5"] = 0
pop["dose_6"] = 0
pop["dose_7"] = 0
pop["dose_8"] = 0

#antipsychotic = ['CLOZAPINE','ARIPIPRAZOLE','RISPERIDONE',"RISPERDALCONSTA",'OLANZAPINE','LEVOMEPROMAZINE',\
#                 "HALOPERIDOL","LOXAPINE","AMISULPRIDE", "CYAMEMAZINE"]
#
#antipsychotic_dose = [100/50,100/7.5,100/2,(100/25),100/5,1,100/2,100/10,100/10,100/10]

antipsychotic = ['CLOZAPINE','ARIPIPRAZOLE','RISPERIDONE',"RISPERDALCONSTA",'OLANZAPINE','LEVOMEPROMAZINE',\
                 "HALOPERIDOL","LOXAPINE","AMISULPRIDE"]

antipsychotic_dose = [100/50,100/7.5,100/2,(100/25),100/5,1,100/2,100/10,100/10]

#LEVOMEPROMAZINE to Chlorpromazine conversion from:
#https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1488906/
for i in range(1,9):
    print("MED :"+ str(i))
    print(pop[pop["MEDLIATC%r"%(i)] == "RISPERIDONE"]["MEDNOM%r"%(i)])


pop.loc[pop['code_patient']=='C0828-001-161-001',["MEDLIATC1"]] = "RISPERDALCONSTA"
pop.loc[pop['code_patient']=='C0828-001-205-001',["MEDLIATC5"]] = "RISPERDALCONSTA"
pop.loc[pop['code_patient']=='C0828-001-228-001',["MEDLIATC1"]] = "RISPERDALCONSTA"



pop['MEDPOSO1'] = pop['MEDPOSO1'].str.replace(pat = "MG/J",repl = "")
pop['MEDPOSO1'] = pop['MEDPOSO1'].str.replace(pat = "MG",repl = "")
pop.loc[pop['MEDPOSO1'] == "50µ/14J",['MEDPOSO1']] = 0.05
pop.loc[pop['MEDPOSO1'] == "75/4SEM",['MEDPOSO1']] = 75/28



pop['MEDPOSO2'] = pop['MEDPOSO2'].str.replace(pat = "MG/J",repl = "")
pop['MEDPOSO2'] = pop['MEDPOSO2'].str.replace(pat = "MG",repl = "")
pop.loc[pop['MEDPOSO2'] == "NC",['MEDPOSO2']] = "NaN"
pop.loc[pop['MEDPOSO2'] == "137.5µG",['MEDPOSO2']] = 0.1375
pop.loc[pop['MEDPOSO2'] == "75  JUSQU'A 3 FOIS PAR JOUR",['MEDPOSO2']] = "NaN"
pop.loc[pop['MEDPOSO2'] == "200GOUTTES/J",['MEDPOSO2']] = 200


pop['MEDPOSO3'] = pop['MEDPOSO3'].str.replace(pat = "MG/J",repl = "")
pop['MEDPOSO3'] = pop['MEDPOSO3'].str.replace(pat = "MG",repl = "")
pop['MEDPOSO3'] = pop['MEDPOSO3'].str.replace(pat = "/J",repl = "")
pop.loc[pop['MEDPOSO3'] == "1 JUSQU'A 3 SI BESOIN",['MEDPOSO3']] = "NaN"
pop.loc[pop['MEDPOSO3'] == "1/72H CUTANE",['MEDPOSO3']] = "NaN"

pop['MEDPOSO4'] = pop['MEDPOSO4'].str.replace(pat = "MG/J",repl = "")
pop['MEDPOSO4'] = pop['MEDPOSO4'].str.replace(pat = "MG",repl = "")
pop['MEDPOSO4'] = pop['MEDPOSO4'].str.replace(pat = "/J",repl = "")
pop.loc[pop['MEDPOSO4'] == "LP 8",['MEDPOSO4']] = "NaN"
pop.loc[pop['MEDPOSO4'] == "2X2",['MEDPOSO4']] = "NaN"

pop['MEDPOSO5'] = pop['MEDPOSO5'].str.replace(pat = "MG/J",repl = "")
pop['MEDPOSO5'] = pop['MEDPOSO5'].str.replace(pat = "MG",repl = "")

pop['MEDPOSO6'] = pop['MEDPOSO6'].str.replace(pat = "MG/J",repl = "")
pop['MEDPOSO6'] = pop['MEDPOSO6'].str.replace(pat = "MG",repl = "")

pop['MEDPOSO7'] = pop['MEDPOSO7'].str.replace(pat = "MG/J",repl = "")
pop['MEDPOSO7'] = pop['MEDPOSO7'].str.replace(pat = "MG",repl = "")


for i in range (len(pop)):
    print("sujet " + str(i))
    curr= pop[pop.index == i]
    for j in range(1,9):
        if curr['MEDLIATC%r'%(j)].isnull()[i] == False:
            print(curr['MEDLIATC%r'%(j)].values)
            for idx,  med in enumerate(antipsychotic):
                if curr['MEDLIATC%r'%(j)][i] == med:
                    print ("dose of " + med)
                    pop["dose_%r"%(j)][i] = (curr['MEDPOSO%r'%(j)].astype(float) * antipsychotic_dose[idx]).values



pop = pop.fillna(value=0)
pop["dose_total"] = pop["dose_1"].astype(float)+pop["dose_2"].astype(float)+pop["dose_3"].astype(float)\
+pop["dose_4"].astype(float)+pop["dose_5"].astype(float)+pop["dose_6"].astype(float)\
+pop["dose_7"].astype(float)+pop["dose_8"].astype(float)




dose_total = pop["dose_total"] .values

np.save("/neurospin/brainomics/2016_schizConnect/analysis/VIP/VBM/data/treatment/\
dose_ongoing_treatment.npy",dose_total)


#np.save("/neurospin/brainomics/2016_schizConnect/analysis/VIP/VBM/data/treatment/\
#dose_ongoing_treatment_with_cyamemazine.npy",dose_total)
#

#Plot correlations
for i in K_interest:
    corr,p = scipy.stats.pearsonr(scores_vip_scz[:,i],dose_total)
    plt.figure()
    plt.plot(scores_vip_scz[:,i] ,dose_total,'o')
    plt.title("Pearson' s correlation = %.02f \n p = %.01e" % (corr,p),fontsize=12)
    plt.xlabel('Score on cluster %s' %(i))
    plt.ylabel('Dose of anti-psychotic treatment (CPZeq in mg/day)')
    plt.tight_layout()
    plt.savefig("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/results/clinic_correlations/dose_of_treatment/cluster%r.png"%(i))




corr,p = scipy.stats.pearsonr(decision_function_vip,dose_total)
plt.figure()
plt.plot(decision_function_vip ,dose_total,'o')
plt.title("Pearson' s correlation = %.02f \n p = %.01e" % (corr,p),fontsize=12)
plt.xlabel('Decision function %s' %(i))
plt.ylabel('Dose of anti-psychotic treatment (CPZeq in mg/day)')
plt.tight_layout()
plt.savefig("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/results/clinic_correlations/dose_of_treatment.png")




#Clozapine*100/50
#        curr[curr['MEDLIATC1']=='ARIPIPRAZOLE']['MEDPOSO1'] *100/7.5
#        curr[curr['MEDLIATC1']=='RISPERIDONE']['MEDPOSO1'] *100/2
#        curr[curr['MEDLIATC1']=='OLANZAPINE']['MEDPOSO1'] *100/5i=1
#        curr[curr['MEDLIATC1']=='LEVOMEPROMAZINE']['MEDPOSO1'] ???
#        curr[curr['MEDLIATC1']=='CYAMEMAZINE']['MEDPOSO1']  ??
#        curr[curr['MEDLIATC1']=='LOXAPINE']['MEDPOSO1'] *100/10
#        curr[curr['MEDLIATC1']=='AMISULPRIDE']['MEDPOSO1'] *100/10



#LEVOMEPROMAZINE: antipsychotic eqOlan??
#TROPATEPINE ??
#CLONAZEPAM,LORAZEPAM PAROXETINE = anxiolytique
#ZOPICLONE,ALIMEMAZINE sommeil
#VALPROIC ACID thymorégulateurs
#CARBAMAZEPINE
#ESCITALOPRAM depression
#BUSPIRONE anxiolytique
#TROPATEPINE anxiolytiaue
#ESCITALOPRAM
#'OXAZEPAM'
pop['MEDLIATC1'].unique()
pop['MEDLIATC2'].unique()
pop['MEDLIATC3'].unique()
pop['MEDLIATC4'].unique()
pop['MEDLIATC5'].unique()
pop['MEDLIATC6'].unique()
pop['MEDLIATC7'].unique()
pop['MEDLIATC8'].unique()