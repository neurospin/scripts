"""
Created on Tue Feb 28 12:21:57 2017

@author: ad247405
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


INPUT_CLINIC_FILENAME = "/neurospin/abide/schizConnect/data/schizconnect_NMorphCH_assessmentData_1829.csv"
INPUT_POPULATION = "/neurospin/brainomics/2016_schizConnect/analysis/NMorphCH/VBM/population.csv"



#PROBLEME multiple scores for each subject!!
clinic = pd.read_csv(INPUT_CLINIC_FILENAME)
pop = pd.read_csv(INPUT_POPULATION)
pop["SAPS"] =  "NaN"
pop["SANS"] =  "NaN"
for s in pop.subjectid:
    print (s)
    curr = clinic[clinic.subjectid ==s]
    most_recent_visit = curr.visit.unique()[-1]
    curr_visit = curr[curr.visit == most_recent_visit]
    if len(curr_visit[curr_visit.assessment_description == "Scale for the Assessment of Positive Symptoms"]) <1: 
        curr_visit = curr[curr.visit == curr.visit.unique()[-2]]

    current_SAPS = curr_visit[curr_visit.assessment_description == "Scale for the Assessment of Positive Symptoms"].question_value.astype(np.float).values
    current_SANS = curr_visit[curr_visit.assessment_description == "Scale for the Assessment of Negative Symptoms"].question_value.astype(np.float).values           
    if len(current_SANS) != 0:
        pop.loc[pop.subjectid ==s,"SAPS"] = np.nansum(current_SAPS)
        print (current_SAPS.sum())
    if len(current_SAPS) != 0:    
        pop.loc[pop.subjectid ==s,"SANS"] = np.nansum(current_SANS)
    

        
#investigate distribution of SAPS and SANS scores across SCZ population     
SAPS_scores =  pop[pop.dx_num ==1].SAPS.astype(np.float).values
SANS_scores =  pop[pop.dx_num ==1].SANS.astype(np.float).values  


plt.hist(SAPS_scores,bins=20)
plt.ylabel("SAPS scores")
plt.xlabel("patients")


plt.hist(SANS_scores,bins=20)
plt.ylabel("SANS scores")
plt.xlabel("patients")

plt.plot(SANS_scores,SAPS_scores,'o')
plt.ylabel("SANS scores")
plt.xlabel("SAPS scores")
