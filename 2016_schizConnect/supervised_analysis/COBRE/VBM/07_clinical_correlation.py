"""
Created on Tue Feb 28 12:21:57 2017

@author: ad247405
"""

import os
import numpy as np
import pandas as pd
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


plt.hist(PANSS_scores,bins=25)
plt.ylabel("PANSS scores")
plt.xlabel("patients")

