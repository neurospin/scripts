#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 16:56:52 2017

@author: Pauline
"""
from __future__ import division
import os.path
from fn_get_clinical_data import get_clinical_data
from fn_get_dti_data import get_dti_data
  
def get_combined_data(BASE_PATH):
   
    
    ##############################################################################
    ########################### Data Definition ##################################
    ##############################################################################
        
    #Import dti data
    pop, _, _, _, = get_dti_data(BASE_PATH)
    df_dti = pop
    df_dti = df_dti.sort_values(by=['siteID','subjectID'])

    
    ##############################################################################
    #Import clinical data
    df_clinic = get_clinical_data(BASE_PATH)
    df_clinic = df_clinic.sort_values(by=['siteID','subjectID'])
    
    
    ##############################################################################   
    #Merge the dataframes
    df_full = df_dti.merge(df_clinic,how='outer',on=['siteID','subjectID']) # Only subject with dti AND clinical data 
    
    #Remove X_y var
    df_full['Age'] = df_full['Age_x']
    df_full['Sex'] = df_full['Sex_x']
    df_full['DX'] = df_full['DX_x']
    df_full = df_full.drop(['DX_x','DX_y','Age_x','Age_y','Sex_x','Sex_y'],1)
    col = ['subjectID', 'siteID', 'DX', 'Age', 'Sex', 'Age of Onset', 'Onset Time',
           'Illness Duration', 'BD Type', 'Mood Phase', 'Depression Scale', 'Depression Score',
           'Number of Depressive Episodes', 'Mania Scale', 'Mania Score',
           'Number of Manic Episodes', 'Total Episodes', 'Density of Episodes', 'Severity', 'Psychotic', 
           'On Meds', 'Antipsychotics','Antidepressants', 'Anticonvulsants', 'Lithium', 'Alcohol', 
           'ACR-L','ACR-R', 'ALIC-L', 'ALIC-R', 'AverageFA', 'BCC', 'CC',
           'CGC-L', 'CGC-R', 'CGH-L', 'CGH-R', 'CR-L', 'CR-R',
           'CST-L', 'CST-R', 'EC-L', 'EC-R', 'FX', 'FX/ST-L', 'FX/ST-R',
           'GCC', 'IC-L', 'IC-R', 'IFO-L', 'IFO-R', 
           'PCR-L', 'PCR-R', 'PLIC-L', 'PLIC-R', 'PTR-L', 'PTR-R',
           'RLIC-L', 'RLIC-R', 'SCC', 'SCR-L', 'SCR-R', 
           'SFO-L', 'SFO-R', 'SLF-L', 'SLF-R', 'SS-L', 'SS-R',
           'UNC-L', 'UNC-R'] 
    df_full = df_full[col]
    
    #Keep only subjects with DTI data
    df_full = df_full.drop(df_full[len(df_dti):].index)
    
    #Remove BD NOS
    df_full = df_full.drop(df_full[df_full['BD Type'] == 'BP NOS'].index)
    
    #Remove controls with history of depressive episods and/or on meds
    df_full = df_full.drop(df_full[(df_full['DX'] == 'HC') & (df_full['Number of Depressive Episodes'] > 0)].index)
    df_full = df_full.drop(df_full[(df_full['DX'] == 'HC') & (df_full['On Meds'] > 0)].index)
    
    ## Find duplicates
#    n_dupl = df_full.duplicated(['subjectID']).sum() # specify columns for finding duplicates
#    n_dupl = df_full[df_full.duplicated(['subjectID'])]
    #pb: 257 duplicates IDs !! (USCD, Grenoble, UNSW, KI)
    #sol: new subj ID with subj and site ID
#    df_full.loc[df_full.siteID == '2_UNSW','subjectID'] = df_full.loc[df_full.siteID == '2_UNSW','subjectID'] + '_UNSW'
#    df_full.loc[df_full.siteID == '5_KI','subjectID'] = df_full.loc[df_full.siteID == '5_KI','subjectID'] + '_KI'
#    df_full.loc[df_full.siteID == '16_UCSD','subjectID'] = df_full.loc[df_full.siteID == '16_UCSD','subjectID'] + '_UCSD'
#    df_full.loc[df_full.siteID == '17_Grenoble','subjectID'] = df_full.loc[df_full.siteID == '17_Grenoble','subjectID'] + '_Gre'
    
    df_full = df_full.sort(['siteID','subjectID']) 
    assert df_full.shape == (len(df_dti["subjectID"]), 26+44)
    
    #Keep only the patients
    df_full_BD = df_full.drop(df_full[df_full.DX =='HC'].index)
    #df_full_BD.to_csv(os.path.join('~/Dropbox/Post-Doc2/ENIGMA/Data/FA_data', 'FA-and-Clincal_data_enigma.csv')) 

    
    return(df_full, df_full_BD) 
    