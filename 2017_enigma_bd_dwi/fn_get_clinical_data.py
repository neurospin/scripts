#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 15:45:15 2017

@author: Pauline
"""
from __future__ import division
import os.path
import pandas as pd
import numpy as np
import os
import math
def roundup(x):
    return int(math.ceil(x))

def get_clinical_data(BASE_PATH):
    
    ##############################################################################     
    ##################### Import csv file and reshape df #########################
    ##############################################################################
    
    columns = ['siteID','subjectID', 'Age at Time of Scan', 'Age of Onset', 'Sex',
           'Diagnosis (BP1, BP2, Control)', 'Mood Phase at Time of Scan',
           'Depression Scale', 'Depression Score at Time of Scan',
           'Number of Depressive Episodes', 'Mania Scale',
           'Mania Score at time of Scan', 'Number of Manic Episodes',
           'Psychotic Features (Y/N – throughout Lifetime)',
           'Suicide Attempt  (Y/N – throughout Lifetime)',
           'On Medication at Time of Scan (Y/N)', 'Meds, Antipsychotics',
           'Length of Time on Antipsychotics', 'Meds, Antidepressants',
           'Length of Time on Antidepressants', 'Meds, Anticonvulsants',
           'Length of Time on Anticonvulsants', 'Lithium (y/n)',
           'Length of Time on Lithium', 'History of Alcohol Dependence (Y/N)',
           'Rapid Cycling (Y/N)']
    
    
    #####----- MUNSTER -----#####
    Munster_1 = os.path.join(BASE_PATH,'Clinical_data','1_Munster_Clinical_data.csv')
    Munster_1 = pd.read_csv(Munster_1)
    Munster_1 = Munster_1.rename(columns={'Subject ID':'subjectID',
                                          'Age at Time of Scan ':'Age at Time of Scan',
                                          'Age of Onset ':'Age of Onset', 
                                          'Depression Scale ':'Depression Scale',
                                          'Number of Depressive Episodes ':'Number of Depressive Episodes', 
                                          'Number of Manic Episodes ':'Number of Manic Episodes',
                                          'Meds, Antipsychotics ':'Meds, Antipsychotics', 
                                          'Meds, Antidepressants ':'Meds, Antidepressants', 
                                          'Meds, Anticonvulsants ':'Meds, Anticonvulsants',
                                          'Mania Score at time of Scan':'Mania Scale_temp',
                                          'Mania Scale':'Mania Score at time of Scan'})
    Munster_1 = Munster_1.rename(columns={'Mania Scale_temp':'Mania Scale'}) #Because scale and score reversed in original file
    Munster_1['siteID'] = '1_Munster'
    Munster_1 = Munster_1[columns]
    Munster_1['Mood Phase at Time of Scan'] = Munster_1['Mood Phase at Time of Scan'].replace({'dep':'Depressed'}) 
    Munster_1['Depression Scale'] = 'HDRS-21'
    Munster_1['Mania Scale'] = 'YMRS'
    Munster_1.ix[41:479,'Diagnosis (BP1, BP2, Control)'] = 'Control'
    # Rename subjects ID (remove second and third part of the original IDs to match FA df)
    subj_ID = Munster_1['subjectID'].str.split('_',expand=True)
    subj_ID = subj_ID[0].str.split('-',expand=True)
    Munster_1['subjectID'] = subj_ID[0]
    #Get age and sex from ROI table
    Munster_1 = Munster_1.drop(['Age at Time of Scan','Sex'],1)
    Munster_1_info = pd.read_csv('/Users/Pauline/Documents/POST-DOC2/ENIGMA/csvFiles/1_Münster/1_Muenster_OriginalData/Muenster Original Data Unzipped/muenster_Covariates.txt') #!!!PATH!!!
    Munster_1_info = Munster_1_info.rename(columns={'AffectionStatus':'Diagnosis (BP1, BP2, Control)',
                                                    'Age':'Age at Time of Scan',
                                                    'Sex ':'Sex'})
    Munster_1 = pd.merge(Munster_1,Munster_1_info[['subjectID','Age at Time of Scan','Sex']],on='subjectID',how='outer')
    Munster_1 = Munster_1[columns]
    Munster_1 = Munster_1.drop(Munster_1[Munster_1['Age at Time of Scan'] < 18].index)
    Munster_1['Sex'] = Munster_1['Sex'].replace({1:'M', 2:'F'}) 
    del(Munster_1_info)
    # Too many controls to keep 1:3 ratio --> randomely delete some subjects (see Melissa's logbook)
    list_ID = pd.read_csv('/Users/Pauline/Documents/POST-DOC2/ENIGMA/csvFiles/1_Münster/1_Muenster_Logbook.txt', 
                          sep=" ", header=None, error_bad_lines=False)    
    list_ID = list_ID[0].str.split('\t',expand=True)
    list_ID = list_ID.ix[12:320,0]
    list_ID = list_ID.str.split('_',expand=True)
    list_ID = list_ID[0].tolist()
    Munster_1 = Munster_1.drop(Munster_1[Munster_1.subjectID.isin(list_ID)].index)
    Munster_1['Diagnosis (BP1, BP2, Control)'] = Munster_1['Diagnosis (BP1, BP2, Control)'].replace({np.nan:'BP'})
    
    
    #####----- UNSW -----##### 
    Unsw_2 = os.path.join(BASE_PATH,'Clinical_data','2_UNSW_Clinical_data.csv')
    Unsw_2 = pd.read_csv(Unsw_2)
    Unsw_2 = Unsw_2.rename(columns={'Subject ID':'subjectID',
                                    'Age at the Time of Scan ':'Age at Time of Scan',
                                    'Mood Phase at Time of Scan (Depressive/Manic/Neither)':'Mood Phase at Time of Scan',
                                    'Depression Score at Time of Scan (MASDRS)':'Depression Score at Time of Scan',
                                    'Mania Score at time of Scan (YMRS)':'Mania Score at time of Scan',
                                    'Psychotic Features':'Psychotic Features (Y/N – throughout Lifetime)',
                                    'Suicide Attempt':'Suicide Attempt  (Y/N – throughout Lifetime)',
                                    'On Medication at Time of Scan':'On Medication at Time of Scan (Y/N)',
                                    'Lithium ':'Lithium (y/n)',
                                    'Alcohol Dependence':'History of Alcohol Dependence (Y/N)',
                                    'Rapid Cycling':'Rapid Cycling (Y/N)'})
    Unsw_2['Depression Scale'] = 'MADRS'
    Unsw_2['Mania Scale'] = 'YMRS'
    Unsw_2['siteID'] = '2_UNSW'
    Unsw_2['subjectID'] = Unsw_2['subjectID'].astype(int).astype(str)
    Unsw_2 = Unsw_2.replace({-99:np.nan,-98:np.nan,-96:np.nan,'-99':np.nan,'-98':np.nan})
    Unsw_2['Number of Manic Episodes'] = Unsw_2['Number of Manic Episodes'] + Unsw_2['Number of Hypomanic Episodes'] 
    Unsw_2 = Unsw_2.drop(['Number of Hypomanic Episodes'],1)
    miss_col = pd.DataFrame(columns=list(set(columns)-set(Unsw_2.columns.tolist())))
    Unsw_2 = Unsw_2.append(miss_col)
    Unsw_2 = Unsw_2[columns]
    Unsw_2['Sex'] = Unsw_2['Sex'].replace({'Male':'M', 'Female':'F'}) 
    Unsw_2['Diagnosis (BP1, BP2, Control)'] = Unsw_2['Diagnosis (BP1, BP2, Control)'].replace({'Bipolar I':'BP1', 'Bipolar II':'BP2'}) 
    Unsw_2 = Unsw_2.drop(Unsw_2[Unsw_2['Age at Time of Scan'] < 18].index) 
    Unsw_2['Mood Phase at Time of Scan'] = Unsw_2['Mood Phase at Time of Scan'].replace({'Neither':np.nan, 'Depressive':'Depressed'})
    Unsw_2 = Unsw_2.replace({'Yes':'Y','No':'N'})
    
        
    #####----- UNC -----##### # !!!! N final doesn't match logbook !!!!
    Unc_3 = os.path.join(BASE_PATH,'Clinical_data','3a_UNC_Clinical_data.csv') ## See new variable file !!!
    Unc_3 = pd.read_csv(Unc_3)
    Unc_3['Depression Scale']='MADRS'
    Unc_3['Mania Scale'] = 'YMRS'
    Unc_3['Meds, Antipsychotics'] = Unc_3['SCID_current_typical_Antipsychotics'] + Unc_3['SCID_current_Atypical_Antipsychotics']
    Unc_3['Meds, Antipsychotics'] = Unc_3['Meds, Antipsychotics'].replace(2,1)
    # __> loop diagnostic
    Unc_3['Diagnosis (BP1, BP2, Control)']=''
    for i, row in Unc_3.iterrows():
        if row['Group'] == 'BD I':
            Unc_3.ix[i,'Diagnosis (BP1, BP2, Control)']="BP1"
        elif row['Group'] == 'BD II':
            Unc_3.ix[i,'Diagnosis (BP1, BP2, Control)']="BP2"
        elif row['Group'] == 'BD NOS':
            Unc_3.ix[i,'Diagnosis (BP1, BP2, Control)']="BP NOS"
        elif row['Group'] == 'Control':
            Unc_3.ix[i,'Diagnosis (BP1, BP2, Control)']="Control"
    Unc_3 = Unc_3.rename(columns={'Label':'subjectID','M/F':'Sex',
                                  'Age':'Age at Time of Scan',
                                  'SCID_current_mood':'Mood Phase at Time of Scan','SCID_number_Of_depressive_Disorders':'Number of Depressive Episodes',
                                  'MADRAS_Total':'Depression Score at Time of Scan','SCID_number_Of_manic_Episodes':'Number of Manic Episodes',
                                  'SCID_ever_psychotic':'Psychotic Features (Y/N – throughout Lifetime)','YMRS_Total':'Mania Score at time of Scan',
                                  'SCID_current_medication':'On Medication at Time of Scan (Y/N)',
                                  'SCID_current_antidepressants':'Meds, Antidepressants',
                                  'SCID_current_lithium':'Lithium (y/n)', 'SCID_curr_lithium_duration':'Length of Time on Lithium',
                                  'SCID_current_anticonv':'Meds, Anticonvulsants','SCID_current_BZD':'Meds, Benzodiazepine',
                                  'SCID.rapid_cycling':'Rapid Cycling (Y/N)','Alcohol_BP':'History of Alcohol Dependence (Y/N)'})
    Unc_3 = Unc_3.drop(['Group','Scanner','SCID_Dep_current_suicidal','SCID_Dep_type_suicidal','SCID_curr_lithium_dose',
                        'SCID_curr_stimulants','SCID_current_typical_Antipsychotics','SCID_current_Atypical_Antipsychotics',
                        'SCID.age_first_alcohol_symptoms','Meds, Benzodiazepine'],1)
    Unc_3 = Unc_3.drop(Unc_3.columns.to_series()['SCID_age_anxiety_sym':"SCID.age_first_psycho_episode"], axis=1)
    # Add missing variables and sort columns
    miss_col = pd.DataFrame(columns=list(set(columns)-set(Unc_3.columns.tolist())))
    Unc_3 = Unc_3.append(miss_col)
    Unc_3['siteID'] = '3.1_UNC' #Clinical data for UNC_1 only (Siemens) 
    Unc_3 = Unc_3[columns]
    # Recode variables
    for i, row in Unc_3.iterrows():
        if row['Number of Depressive Episodes'] >= 50: # too many episod coded 99
            Unc_3.ix[i,'Number of Depressive Episodes'] = 50
        if row['Number of Manic Episodes'] >= 50: # too many episod coded 99
            Unc_3.ix[i,'Number of Manic Episodes'] = 50
        if row['Psychotic Features (Y/N – throughout Lifetime)'] >= 1: # too many episod coded 99
            Unc_3.ix[i,'Psychotic Features (Y/N – throughout Lifetime)'] = 'Y'
        else: Unc_3.ix[i,'Psychotic Features (Y/N – throughout Lifetime)'] = 'N'
    Unc_3 = Unc_3.drop(Unc_3[Unc_3['Age at Time of Scan'] < 18].index) 
    Unc_3 = Unc_3.drop(Unc_3[Unc_3['Age at Time of Scan'] > 65].index) 
    Unc_3['subjectID'] = Unc_3['subjectID'].astype(int).astype(str)
    Unc_3['Mood Phase at Time of Scan'] = Unc_3['Mood Phase at Time of Scan'].replace({1:'Euthymic', 
                 2:'Depressed',3:'Manic',4:'Hypomanic',5:'Mixed',6:np.nan}) #See email
    Unc_3.Sex[Unc_3.subjectID == '30168'] = 'M'
    Unc_3['Diagnosis (BP1, BP2, Control)'] = Unc_3['Diagnosis (BP1, BP2, Control)'].replace({'':'BP'})
    
    
    #####----- UCT -----######
    Uct_4 = os.path.join(BASE_PATH,'Clinical_data','4_UCT_Clinical_data.csv')
    Uct_4 = pd.read_csv(Uct_4, decimal=',')
    Uct_4 = Uct_4.drop(['Site','History Drug Dependence'],1)
    Uct_4 = Uct_4.rename(columns={'ID':'subjectID'})
    Uct_4['subjectID'] = Uct_4['subjectID'].astype(str)
    Uct_4['siteID'] = '4_UCT'
    Uct_4 = Uct_4[columns]
    Uct_4['Sex'] = Uct_4['Sex'].replace({1:'F', 0:'M'})
    Uct_4['Diagnosis (BP1, BP2, Control)'] = Uct_4['Diagnosis (BP1, BP2, Control)'].replace({'Bipolar I HX psychosis':'BP1',np.nan:'Control'})
    Uct_4['Psychotic Features (Y/N – throughout Lifetime)']=Uct_4['Psychotic Features (Y/N – throughout Lifetime)'].replace({'significant history':'Y'})
    Uct_4['Psychotic Features (Y/N – throughout Lifetime)']=Uct_4['Psychotic Features (Y/N – throughout Lifetime)'].replace({'N':np.nan})
    Uct_4.ix[0:20,'Age of Onset'] = Uct_4.ix[0:20,'Age of Onset'].apply(roundup)
    Uct_4.ix[21:44,'Mania Scale':'Rapid Cycling (Y/N)'] = np.nan
    Uct_4.ix[(1,18),'Meds, Antipsychotics'] = 'Y'
    Uct_4.ix[0,'Meds, Antipsychotics'] = 'N'
    Uct_4.ix[15,'Meds, Antidepressants'] = 'Y'
    Uct_4['Meds, Antidepressants'] = Uct_4['Meds, Antidepressants'].replace('N ','N')
    Uct_4['Depression Scale'] = 'HDRS-21'
    
    
    #####----- KAROLINSKA INSTITUTE (KI) -----##### !!! Error in logbook : UNC instead of KI !!!
    Ki_5 = os.path.join(BASE_PATH,'Clinical_data','5_KI_Clinical_data.csv')
    Ki_5 = pd.read_csv(Ki_5)
    Ki_5 = Ki_5.rename(columns={'Subject ID ENIGMA':'subjectID'})
    Ki_5['siteID'] = '5_KI'
    Ki_5 = Ki_5[columns]
    Ki_5['Sex'] = Ki_5['Sex'].replace({'Male':'M', 'Female':'F'}) 
    Ki_5['Diagnosis (BP1, BP2, Control)'] = Ki_5['Diagnosis (BP1, BP2, Control)'].replace({'other':'BP NOS'}) 
    Ki_5 = Ki_5.drop(Ki_5[Ki_5['Age at Time of Scan'] > 65].index) 
    Ki_5['Mood Phase at Time of Scan'] = Ki_5['Mood Phase at Time of Scan'].replace({'euthymia':'Euthymic'})
    # Too many controls to keep 1:3 ratio --> randomely delete some subjects (see Melissa's logbook)
    list_ID = pd.read_csv('/Users/Pauline/Documents/POST-DOC2/ENIGMA/csvFiles/5_KI/5_KI_Logbook.txt', sep=" ", header=None, error_bad_lines=False)    
    list_ID = list_ID[0].str.split('\t',expand=True)
    list_ID = list_ID.ix[18:56,0].astype(int)
    list_ID = list_ID.tolist()
    Ki_5 = Ki_5.drop(Ki_5[Ki_5.subjectID.isin(list_ID)].index)
    
    
    #####----- CARDIFF -----#####
    Cardiff_6 = os.path.join(BASE_PATH,'Clinical_data','6_Cardiff_Clinical_data.csv')
    Cardiff_6 = pd.read_csv(Cardiff_6)
    Cardiff_6 = Cardiff_6.rename(columns={'Subject ID':'subjectID'})
    Cardiff_6['siteID'] = '6_Cardiff'
    Cardiff_6 = Cardiff_6[columns]
    Cardiff_6['Diagnosis (BP1, BP2, Control)'] = Cardiff_6['Diagnosis (BP1, BP2, Control)'].replace({1:'HC', 2:'BP'})
    Cardiff_6['Sex'] = Cardiff_6['Sex'].replace({2:'F', 1:'M'})
    #Convert variable type
    Cardiff_6['subjectID'] = Cardiff_6['subjectID'].astype(str)
    Cardiff_6['Depression Scale'] = 'HDRS-21'
    
    
    #####----- EDIMBURGH -----#####
    Edimburgh_7 = os.path.join(BASE_PATH,'Clinical_data','7ab_Edimburgh_Clinical_data.csv')
    Edimburgh_7 = pd.read_csv(Edimburgh_7)
    Edimburgh_7 = Edimburgh_7.rename(columns={'Subject ID':'subjectID'})
    Edimburgh_7['subjectID'] = Edimburgh_7['subjectID'].astype(str)
    #Define siteID
    Edimburgh_7['siteID'] = ''
    for i, row in Edimburgh_7.iterrows():
        if row['subjectID'].startswith('SF'):
            Edimburgh_7.ix[i,'siteID']='7.1_Edimburgh'
        else: Edimburgh_7.ix[i, 'siteID']='7.2_Edimburgh' 
    #Rearrange columns and coding
    Edimburgh_7 = Edimburgh_7[columns]
    Edimburgh_7['Diagnosis (BP1, BP2, Control)'] = Edimburgh_7['Diagnosis (BP1, BP2, Control)'].replace({0:'Control', 1:'BP1'})
    Edimburgh_7['Sex'] = Edimburgh_7['Sex'].replace({0:'M', 1:'F'}) # Error in logbook because Sex is coded as 0/1
    Edimburgh_7['Mania Scale'] = Edimburgh_7['Mania Scale'].replace('Young','YMRS') 
    #Get age from ROI table
    Edimburgh_7 = Edimburgh_7.drop('Age at Time of Scan',1)
    Edimburgh_71_ROI = pd.read_csv('/Users/Pauline/Documents/POST-DOC2/ENIGMA/csvFiles/7ab_Edinburgh/7ab_OriginalData/combinedROItablew_sexIncluded.csv')
    Edimburgh_71_info = Edimburgh_71_ROI.ix[:,0:4]
    Edimburgh_71_info = Edimburgh_71_info.rename(columns={'subjectID':'subjectID',
                                                  'Diagnosis':'Diagnosis (BP1, BP2, Control)',
                                                  'Age':'Age at Time of Scan',
                                                  'Gender':'Sex'})
    Edimburgh_72_ROI = pd.read_csv('/Users/Pauline/Documents/POST-DOC2/ENIGMA/csvFiles/7ab_Edinburgh/7ab_OriginalData/emotion_results_Clara_Alloza.csv')
    Edimburgh_72_info = Edimburgh_72_ROI.ix[:,0:4]
    Edimburgh_72_info = Edimburgh_72_info.rename(columns={'subjectID':'subjectID',
                                                  'diagnosis':'Diagnosis (BP1, BP2, Control)',
                                                  'age':'Age at Time of Scan',
                                                  'sex':'Sex'})
    Edimburgh_info = pd.concat([Edimburgh_71_info,Edimburgh_72_info])  
    Edimburgh_7 = pd.merge(Edimburgh_7,Edimburgh_info[['subjectID','Age at Time of Scan']],on='subjectID',how='outer')
    Edimburgh_7['Age at Time of Scan'] = Edimburgh_7['Age at Time of Scan'].apply(roundup)
    Edimburgh_7 = Edimburgh_7[columns]
    Edimburgh_7 = Edimburgh_7.drop(Edimburgh_7[Edimburgh_7['Age at Time of Scan'] < 18].index) 
    Edimburgh_7 = Edimburgh_7.drop(Edimburgh_7[Edimburgh_7['Age at Time of Scan'] > 65].index) 
    del(Edimburgh_71_info,Edimburgh_71_ROI,Edimburgh_72_info,Edimburgh_72_ROI,Edimburgh_info)
    # Recoding meds
    Edimburgh_7.ix[(40,43,44,45,54,55,56,58,63,65,69,75,79,86,87,98),'Meds, Antipsychotics'] = 1
    Edimburgh_7.ix[(57,59,60,61,62,66,72,77,78,88,91,95,96),'Meds, Antipsychotics'] = 0
    Edimburgh_7.ix[(43,47,54,55,56,58,59,69,75,77,79,96),'Meds, Antidepressants'] = 1
    Edimburgh_7.ix[(57,60,61,62,63,65,66,72,78,86,87,88,91,95,98),'Meds, Antidepressants'] = 0
    Edimburgh_7.ix[(54,58,59,60,61,65,69,77,78,95,98),'Meds, Anticonvulsants'] = 1
    Edimburgh_7.ix[(55,56,57,62,63,66,72,75,79,86,87,88,91,96),'Meds, Anticonvulsants'] = 0
    Edimburgh_7.ix[(55,56,57,58,60,61,62,63,66,69,72,75,79,86,87,91),'Lithium (y/n)'] = 'Y'
    Edimburgh_7.ix[(54,59,65,77,78,88,95,96,98),'Lithium (y/n)'] = 'N'
    Edimburgh_7['Depression Scale'] = 'HDRS-21'
       
    #####----- OSLO -----##### 
    Oslo_9 = os.path.join(BASE_PATH,'Clinical_data','9b_Oslo_Clinical_data.csv')
    Oslo_9 = pd.read_csv(Oslo_9, decimal=',')
    Oslo_9 = Oslo_9.rename(columns={'SubjectId':'subjectID',
                                    'AGE AT THE TIME OF SCAN':'Age at Time of Scan',
                                    'AGE OF ONSET (NA=HC)':'Age of Onset',
                                    'GENDER (0=F) ':'Sex', 
                                    'DIAGNOSIS (0=HC; 1=BDII)':'Diagnosis (BP1, BP2, Control)',                                
                                    'MED FREE PAT=0, MEDICATED PAT=1, HC=2':'On Medication at Time of Scan (Y/N)',
                                    'ANTIDEPRESSANTS FREE PAT=0, USING PAT=1, HC=2':'Meds, Antidepressants',
                                    'ANTIEPILEPTICS FREE PAT=0, USING PAT=1, HC=2':'Meds, Anticonvulsants',
                                    'MED FREE PAT=0, MEDICATED PAT=1, HC=2.1':'Duplicate',
                                    'ANTIPSYCHOTICS FREE PAT=0, USING PAT=1, HC=2':'Meds, Antipsychotics',
                                    'LITHIUM FREE PAT=0, USING PAT=1, HC=2':'Lithium (y/n)', 
                                    'MADRS SCORE':'Depression Score at Time of Scan',
                                    'YMRS SCORE ':'Mania Score at time of Scan'})
    Oslo_9 = Oslo_9.drop(['FAMILY HISTORY OF BD (YES=1)', 'Duplicate', 
                          'DEPRESSIVE EPISODE; YES=1; NO=0; HC=2',
                          'HYPOMANIC EPISODE; YES=1; NO=0; HC=2'],1)
    miss_col = pd.DataFrame(columns=list(set(columns)-set(Oslo_9.columns.tolist())))
    Oslo_9 = Oslo_9.append(miss_col)
    Oslo_9 = Oslo_9[columns]
    Oslo_9['siteID'] = '9.2_Oslo_Malt'
    Oslo_9['Depression Scale'] = 'MADRS'
    Oslo_9['Mania Scale'] = 'YMRS'
    Oslo_9['Sex'] = Oslo_9['Sex'].replace({1:'M', 0:'F'}) 
    Oslo_9['Diagnosis (BP1, BP2, Control)'] = Oslo_9['Diagnosis (BP1, BP2, Control)'].replace({1:'BP2',0:'HC'}) 
    Oslo_9['Age at Time of Scan'] = Oslo_9['Age at Time of Scan'].apply(roundup)
    Oslo_9['Age of Onset'] = Oslo_9['Age of Onset'].replace({'.':',','Missing data':np.nan}).astype(float)
    Oslo_9 = Oslo_9.replace({'Missing data':np.nan})
    Oslo_9['On Medication at Time of Scan (Y/N)'] = Oslo_9['On Medication at Time of Scan (Y/N)'].astype(float).replace({2:np.nan})
    Oslo_9['Meds, Antidepressants'] = Oslo_9['Meds, Antidepressants'].astype(float).replace({2:np.nan})
    Oslo_9['Meds, Antipsychotics'] = Oslo_9['Meds, Antipsychotics'].astype(float).replace({2:np.nan})
    Oslo_9['Meds, Anticonvulsants'] = Oslo_9['Meds, Anticonvulsants'].astype(float).replace({2:np.nan})
    Oslo_9['Lithium (y/n)'] = Oslo_9['Lithium (y/n)'].astype(float).replace({2:np.nan})
    
    
    #####----- VITA SALUTE -----#####
    VitaSalute_10 = os.path.join(BASE_PATH,'Clinical_data','10_VitaSalute_Clinical_data.csv')
    VitaSalute_10 = pd.read_csv(VitaSalute_10)
    VitaSalute_10 = VitaSalute_10.drop(['Ill_dur','Epi_dur','Tot_epi'],1)
    VitaSalute_10 = VitaSalute_10.rename(columns={'SubjID':'subjectID','Age':'Age at Time of Scan', 
                                                  'Diagnosis':'Diagnosis (BP1, BP2, Controls)',
                                                  'Onset':'Age of Onset','N_dep':'Number of Depressive Episodes',
                                                  'N_man':'Number of Manic Episodes', 'Phase':'Mood Phase at Time of Scan',
                                                  'Delus':'Psychotic Features (Y/N – throughout Lifetime)',
                                                  'SA':'Suicide Attempt  (Y/N – throughout Lifetime)'})
    # Add missing variables and sort columns
    miss_col = pd.DataFrame(columns=list(set(columns)-set(VitaSalute_10.columns.tolist())))
    VitaSalute_10 = VitaSalute_10.append(miss_col)
    VitaSalute_10['siteID'] = '10_VitaSalute'
    VitaSalute_10 = VitaSalute_10[columns]
    VitaSalute_10['Sex'] = VitaSalute_10['Sex'].replace({0:'F', 1:'M'}) #See email 
    #Convert variable type
    VitaSalute_10['subjectID'] = VitaSalute_10['subjectID'].astype(int).astype(str)
    VitaSalute_10.ix[114,'Number of Depressive Episodes'] = 3
    VitaSalute_10['Mood Phase at Time of Scan'] = VitaSalute_10['Mood Phase at Time of Scan'].replace({0:'Euthymic', 
                 1:'Depressed',2:'Manic',3:'Mixed'}) #See email
    for i, row in VitaSalute_10.iterrows():
        if pd.isnull(row['Mood Phase at Time of Scan']):
            VitaSalute_10.ix[i,'Diagnosis (BP1, BP2, Control)']='Control'
        else: VitaSalute_10.ix[i, 'Diagnosis (BP1, BP2, Control)']='BP' 
    VitaSalute_10.ix[VitaSalute_10['Suicide Attempt  (Y/N – throughout Lifetime)'] >= 1, 'Suicide Attempt  (Y/N – throughout Lifetime)'] = 1
    VitaSalute_10 = VitaSalute_10.drop(VitaSalute_10[VitaSalute_10['Age at Time of Scan'] > 65].index) 
    
    
    #####----- FIDMAG -----######
    Fidmag_11 = os.path.join(BASE_PATH,'Clinical_data','11_FIDMAG_Clinical_data.csv')
    Fidmag_11 = pd.read_csv(Fidmag_11,decimal=',')
    Fidmag_11 = Fidmag_11.rename(columns={'Diagnosis':'Diagnosis (BP1, BP2, Control)'})
    Fidmag_11['siteID'] = '11_FIDMAG'
    Fidmag_11 = Fidmag_11[columns]
    Fidmag_11['Sex'] = Fidmag_11['Sex'].replace({2:'F', 1:'M'})
    #Convert variable type
    Fidmag_11['subjectID'] = Fidmag_11['subjectID'].astype(str)
    Fidmag_11 = Fidmag_11.replace('missing','')
    Fidmag_11 = Fidmag_11.replace('no info','')
    Fidmag_11.ix[56,'Number of Manic Episodes'] = ''
    Fidmag_11.ix[70,'Number of Depressive Episodes'] = ''
    Fidmag_11.ix[71,'Number of Depressive Episodes'] = ''
    Fidmag_11.ix[90,'Number of Depressive Episodes'] = ''
    Fidmag_11.ix[90,'Number of Manic Episodes'] = ''
    Fidmag_11.ix[92,'Number of Depressive Episodes'] = ''
    Fidmag_11.info()
    
        
    #####----- CRETEIL -----#####
    Creteil_12 = os.path.join(BASE_PATH,'Clinical_data','12_Creteil_Clinical_data.csv')
    Creteil_12 = pd.read_csv(Creteil_12)
    Creteil_12 = Creteil_12.drop(['ID', 'Rang protocole', 'HANDEDNESS', 'BRAIN_NORMALIZED',
                                  'GM_NORMALIZED', 'WM_NORMALIZED', 'R_HIPP_mm3', 'L_HIPP_mm3',
                                  'ONSET_TYPE', 'SCANNER', 'HANDEDNESS.1', 'OH_POS_LT_SUBGROUP',
                                  'BMRMS_UDINE', 'HRS17TOT', 'HRS25TOT', 'TOT#PSYCHOTROP',
                                  'LITHIUM_LIFETIME', 'LITHIUM_DAO_vip', 'LITHIUM_DURATION_YEARS', 'DDN',
                                  'DATE_INC'],1)
    Creteil_12 = Creteil_12.rename(columns={'NIP': 'subjectID',
                                            'BD_HC': 'Diagnosis (BP1, BP2, Control)', 
                                            'AGEATMRI': 'Age at Time of Scan', 
                                            'PF': 'Psychotic Features (Y/N – throughout Lifetime)', 
                                            'SEX':'Sex', 
                                            'MADRS': 'Depression Score at Time of Scan',
                                            'YMRS': 'Mania Score at time of Scan', 
                                            'SYMPTOMATIQUE': 'Mood Phase at Time of Scan', 
                                            'AGE_AT_ONSET': 'Age of Onset', 
                                            'ANTIPSYCH': 'Meds, Antipsychotics', 
                                            'MOODSTAB': 'Meds, Anticonvulsants', 
                                            'ANTIDEP': 'Meds, Antidepressants', 
                                            'LITHIUM': 'Lithium (y/n)', 
                                            'TS': 'Suicide Attempt  (Y/N – throughout Lifetime)'})
    Creteil_12['Age of Onset'] = Creteil_12['Age at Time of Scan'] - Creteil_12['Illness_Duration']
    Creteil_12 = Creteil_12.drop(['Illness_Duration'],1)
    miss_col = pd.DataFrame(columns=list(set(columns)-set(Creteil_12.columns.tolist())))
    Creteil_12 = Creteil_12.append(miss_col)
    Creteil_12 = Creteil_12[columns]
    Creteil_12['siteID'] = '12_Creteil'
    Creteil_12['Depression Scale'] = 'MADRS'
    Creteil_12['Mania Scale'] = 'YMRS'
    Creteil_12['Sex'] = Creteil_12['Sex'].replace({1:'M', 2:'F'}) 
    Creteil_12['Diagnosis (BP1, BP2, Control)'] = Creteil_12['Diagnosis (BP1, BP2, Control)'].replace({1:'BP',0:'HC'}) 
    Creteil_12['Mood Phase at Time of Scan'] = Creteil_12['Mood Phase at Time of Scan'].replace({0:'Euthymic', 1:'Depressed'}) # !!! To be verified !!!
    Creteil_12.ix[64,'Suicide Attempt  (Y/N – throughout Lifetime)'] = 1 # !!! Control subject !!!
    Creteil_12.ix[88,'Diagnosis (BP1, BP2, Control)'] = 'BP' # !!!To be checked !!!
    
    for i, row in Creteil_12.iterrows():
        if (row['Meds, Antipsychotics']==1) or (row['Meds, Anticonvulsants']==1) or (row['Meds, Antidepressants']==1) or (row['Lithium (y/n)']==1):
            Creteil_12.ix[i,'On Medication at Time of Scan (Y/N)']=1
        elif row['Meds, Antipsychotics' and 'Meds, Anticonvulsants' and 'Meds, Antidepressants' and 'Lithium (y/n)'] == 0:
            Creteil_12.ix[i,'On Medication at Time of Scan (Y/N)']=0
        else: Creteil_12.ix[i,'On Medication at Time of Scan (Y/N)']=np.nan
    Creteil_12['On Medication at Time of Scan (Y/N)'].astype(float)
    # Drop subjects ip100526 and ip100558 ?? see logbook 
    for i, row in Creteil_12.iterrows(): 
        Creteil_12.ix[i,'subjectID'] = Creteil_12.ix[i,'subjectID'][:6]
        
        
    #####----- UCLA -----#####
    Ucla_13 = os.path.join(BASE_PATH,'Clinical_data','13_UCLA_Clinical_data.csv')
    Ucla_13 = pd.read_csv(Ucla_13)
    Ucla_13['Meds, Antipsychotics'] = Ucla_13['Gen1AntiPsych'] + Ucla_13['Gen2AntiPsych'] 
    Ucla_13['Meds, Antipsychotics'] = Ucla_13['Meds, Antipsychotics'].replace(2,1)
    Ucla_13 = Ucla_13.drop(['FullDx','Gen1AntiPsych','Gen2AntiPsych'],1)
    Ucla_13 = Ucla_13.rename(columns={'MRIcode':'subjectID', 'Dx':'Diagnosis (BP1, BP2, Control)','Age':'Age at Time of Scan',
                                      'Li':'Lithium (y/n)','AntiEpileptic':'Meds, Anticonvulsants',
                                      'AntiDep':'Meds, Antidepressants', 'MoodState':'Mood Phase at Time of Scan',
                                      'AgeofOnset':'Age of Onset', 'HistoryPsychosis':'Psychotic Features (Y/N – throughout Lifetime)'})
    # Add missing variables and sort columns
    miss_col = pd.DataFrame(columns=list(set(columns)-set(Ucla_13.columns.tolist())))
    Ucla_13 = Ucla_13.append(miss_col)
    Ucla_13['siteID'] = '13_UCLA'
    Ucla_13 = Ucla_13[columns]
    Ucla_13['Diagnosis (BP1, BP2, Control)'] = Ucla_13['Diagnosis (BP1, BP2, Control)'].replace({0:'HC', 1:'BP'}) # Error in logbook
    Ucla_13['Sex'] = Ucla_13['Sex'].replace({2:'F', 1:'M'})
    Ucla_13 = Ucla_13.drop(Ucla_13[Ucla_13['Age at Time of Scan'] > 65].index) # Rq Melissa found 31 subj with age > 65 ; here 29
    #Convert variable type
    Ucla_13['subjectID'] = Ucla_13['subjectID'].astype(str)
    
    
    #####----- MANNHEIM -----#####
    Mannheim_15 = os.path.join(BASE_PATH,'Clinical_data','15_Mannheim_Clinical_data.csv')
    Mannheim_15 = pd.read_csv(Mannheim_15)
    Mannheim_15['Depression Scale']='MADRS'
    Mannheim_15['Mania Scale'] = 'YMRS' 
    Mannheim_15 = Mannheim_15.drop(['ID_Original','HANDEDNESS','BMRMS','HRS17','HRS25','SYMPTOMATIQUE','ILLNESS_DURATION',
                                    'TOTAL_PSYCHOTROP','LITHIUM_LIFETIME'],1)
    Mannheim_15 = Mannheim_15.rename(columns={'Current ID':'subjectID','BD_HC':'Diagnosis (BP1, BP2, Control)',
                                              'AGE':'Age at Time of Scan','SEX':'Sex',
                                              'OH_DEPENDENCE':'History of Alcool Dependence (Y/N)',
                                              'MADRS':'Depression Score at Time of Scan',
                                              'YMRS':'Mania Score at Time of Scan','AAO':'Age of Onset',
                                              'ANTIPSICH':'Meds, Antipsychotics','MOODSTAB':'Meds, Anticonvulsants',
                                              'ANTIDEP':'Meds, Antidepressants','LITHIUM':'Lithium (y/n)',
                                              'LITHIUM_DURATION_YEARS':'Length of Time on Lithium'})
    # Add missing variables and sort columns
    miss_col = pd.DataFrame(columns=list(set(columns)-set(Mannheim_15.columns.tolist())))
    Mannheim_15 = Mannheim_15.append(miss_col)
    Mannheim_15['siteID'] = '15_Mannheim'
    Mannheim_15 = Mannheim_15[columns]
    Mannheim_15['Diagnosis (BP1, BP2, Control)'] = Mannheim_15['Diagnosis (BP1, BP2, Control)'].replace({'BD':'BP'})
    Mannheim_15['Sex'] = Mannheim_15['Sex'].replace({'female':'F', 'male':'M'})
    Mannheim_15 = Mannheim_15.drop(Mannheim_15[Mannheim_15['Age at Time of Scan'] > 65].index) 
    #Convert variable type
    Mannheim_15['subjectID'] = Mannheim_15['subjectID'].astype(str)
    # No demographic for C6_HB_003, C6_HB_025, C6_HB_027, C6_HK_009, C6_HK_017 and C6_HK_034
    Mannheim_15 = Mannheim_15.drop(Mannheim_15[Mannheim_15.subjectID.isin({'C6_HB_003',
                                            'C6_HB_025', 'C6_HB_027', 'C6_HK_009', 
                                            'C6_HK_017', 'C6_HK_034'})].index)
    Mannheim_15 = Mannheim_15.replace('#NULL!','')
    
                            
    #####----- UCSD -----#####
    Ucsd_16 = os.path.join(BASE_PATH,'Clinical_data','16_UCSD_Clinical_data.csv')
    Ucsd_16 = pd.read_csv(Ucsd_16, encoding = "ISO-8859-1", sep=';', decimal=',')
    Ucsd_16 = Ucsd_16.rename(columns={'Subject ID':'subjectID',
                                      'Psychotic Features (Y/N Ð throughout Lifetime)':'Psychotic Features (Y/N – throughout Lifetime)',
                                      'Suicide Attempt  (Y/N Ð throughout Lifetime)':'Suicide Attempt  (Y/N – throughout Lifetime)',
                                      'Length of Time on Antipsychotics (weeks)':'Length of Time on Antipsychotics',
                                      'Length of Time on Antidepressants (weeks)':'Length of Time on Antidepressants',
                                      '':''})
    Ucsd_16 = Ucsd_16.drop(['Meds, Anti-anxiety','Length of Time on Anti-anxiety', 
                            'Meds, Mood Stabilizer','Length of Time on Mood Stabilizer',
                            'Other', 'Length of time on Other','History of ECT','Notes'],1)
    Ucsd_16['siteID'] = '16_UCSD'
    Ucsd_16['subjectID'] = Ucsd_16['subjectID'].str.replace('BD','')
    Ucsd_16 = Ucsd_16[columns]
    Ucsd_16['Age at Time of Scan'] = Ucsd_16['Age at Time of Scan'].apply(roundup)
    Ucsd_16 = Ucsd_16.drop(Ucsd_16[Ucsd_16['Age at Time of Scan'] > 65].index) 
    Ucsd_16['Depression Scale'] = 'HDRS-17'
    for i, row in Ucsd_16.iterrows():
        if pd.isnull(row['Meds, Antipsychotics']):
            Ucsd_16.ix[i,'Meds, Antipsychotics']='N'
        else: Ucsd_16.ix[i, 'Meds, Antipsychotics']='Y' 
    for i, row in Ucsd_16.iterrows():
        if pd.isnull(row['Meds, Anticonvulsants']):
            Ucsd_16.ix[i,'Meds, Anticonvulsants']='N'
        else: Ucsd_16.ix[i, 'Meds, Anticonvulsants']='Y' 
    for i, row in Ucsd_16.iterrows():
        if pd.isnull(row['Meds, Antidepressants']):
            Ucsd_16.ix[i,'Meds, Antidepressants']='N'
        else: Ucsd_16.ix[i, 'Meds, Antidepressants']='Y'
    for i, row in Ucsd_16.iterrows():
        if pd.isnull(row['Lithium (y/n)']) and (row['Diagnosis (BP1, BP2, Control)']=='HC'):
            Ucsd_16.ix[i,'Lithium (y/n)']='N'
    Ucsd_16 = Ucsd_16.replace({'-9':np.nan, '-8':np.nan, '?':np.nan})
    Ucsd_16['Diagnosis (BP1, BP2, Control)'] = Ucsd_16['Diagnosis (BP1, BP2, Control)'].replace({'BD':'BP'})
    Ucsd_16['Rapid Cycling (Y/N)'] = Ucsd_16['Rapid Cycling (Y/N)'].replace({'y':'Y'})
        
    
    #####----- GRENOBLE -----##### # Why only 3 HC? Normally 12!! 
    Grenoble_17 = os.path.join(BASE_PATH,'Clinical_data','17_Grenoble_Clinical_data.csv')
    Grenoble_17 = pd.read_csv(Grenoble_17)
    Grenoble_17['Depression Scale']='MADRS'
    Grenoble_17['Mania Scale'] = 'YMRS' 
    Grenoble_17['Number of Manic Episodes'] = Grenoble_17['Nb Maniaques'] + Grenoble_17['Nb Hypomaniques'] 
    Grenoble_17 = Grenoble_17.drop(['Patient','Groupe','Machine','Comorbidités','Durée moy maladie',
                                    'Traitement','QIDS1','Altman1','Nb Maniaques','Nb Hypomaniques'],1)
    Grenoble_17 = Grenoble_17.rename(columns={'N°':'subjectID','Age':'Age at Time of Scan','Sexe':'Sex',
                                              'Type':'Diagnosis (BP1, BP2, Control)','Age 1er épisode':'Age of Onset',
                                              'Nb EDM':'Number of Depressive Episodes',
                                              'Symptômes psychotiques ':'Psychotic Features (Y/N – throughout Lifetime)',
                                              'MADRS1':'Depression Score at Time of Scan',
                                              'YMRS1':'Mania Score at Time of Scan'})
    # Add missing variables and sort columns
    miss_col = pd.DataFrame(columns=list(set(columns)-set(Grenoble_17.columns.tolist())))
    Grenoble_17 = Grenoble_17.append(miss_col)
    Grenoble_17['siteID'] = '17_Grenoble'
    Grenoble_17 = Grenoble_17[columns]
    Grenoble_17['Diagnosis (BP1, BP2, Control)'] = Grenoble_17['Diagnosis (BP1, BP2, Control)'].replace({'1':'BP1','2':'BP2','?':'BP NOS'})
    Grenoble_17['Sex'] = Grenoble_17['Sex'].replace({1:'F', 0:'M'})
    Grenoble_17['Mood Phase at Time of Scan'] = 'Euthymic' 
    Grenoble_17['Diagnosis (BP1, BP2, Control)'] = Grenoble_17['Diagnosis (BP1, BP2, Control)'].replace({np.nan:'BP'})
    Grenoble_17.ix[4,'subjectID'] = 6
    Grenoble_17.ix[21,'subjectID'] = 25
    Grenoble_17['subjectID'] = Grenoble_17['subjectID'].astype(int).astype(str)
    
    
    #####----- SAO PAULO -----#####
    SaoPaulo_19 = os.path.join(BASE_PATH,'Clinical_data','19_SaoPaulo_Clinical_data.csv')
    SaoPaulo_19 = pd.read_csv(SaoPaulo_19)
    SaoPaulo_19['Psychotic Features (Y/N – throughout Lifetime)'] = ''
    SaoPaulo_19 = SaoPaulo_19.rename(columns={'Variables ':'subjectID',' Age':'Age at Time of Scan','Age_of_Onset':'Age of Onset',
                                              'DX':'Diagnosis'})
    for i, row in SaoPaulo_19.iterrows():
        if row['Dx full w/ w/o psychosis'] == 1:
            SaoPaulo_19.ix[i,'Psychotic Features (Y/N – throughout Lifetime)']='N' 
            SaoPaulo_19.ix[i,'Diagnosis (BP1, BP2, Control)']="BP1"
        elif row['Dx full w/ w/o psychosis'] == 2:
            SaoPaulo_19.ix[i,'Psychotic Features (Y/N – throughout Lifetime)']='Y'
            SaoPaulo_19.ix[i,'Diagnosis (BP1, BP2, Control)']="BP1"
        elif row['Dx full w/ w/o psychosis'] == 3:
            SaoPaulo_19.ix[i,'Psychotic Features (Y/N – throughout Lifetime)']='N' 
            SaoPaulo_19.ix[i,'Diagnosis (BP1, BP2, Control)']="BP2"
        elif row['Dx full w/ w/o psychosis'] == 4:
            SaoPaulo_19.ix[i,'Psychotic Features (Y/N – throughout Lifetime)']=np.nan
            SaoPaulo_19.ix[i,'Diagnosis (BP1, BP2, Control)']="Control"
        elif row['Dx full w/ w/o psychosis'] == 5:
            SaoPaulo_19.ix[i,'Psychotic Features (Y/N – throughout Lifetime)']='N'
            SaoPaulo_19.ix[i,'Diagnosis (BP1, BP2, Control)']="BP1"
        elif row['Dx full w/ w/o psychosis'] == 6:
            SaoPaulo_19.ix[i,'Psychotic Features (Y/N – throughout Lifetime)']='Y'
            SaoPaulo_19.ix[i,'Diagnosis (BP1, BP2, Control)']="BP1"
        elif row['Dx full w/ w/o psychosis'] == 7:
            SaoPaulo_19.ix[i,'Psychotic Features (Y/N – throughout Lifetime)']='N'
            SaoPaulo_19.ix[i,'Diagnosis (BP1, BP2, Control)']="BP2"
        elif row['Dx full w/ w/o psychosis'] == 8:
            SaoPaulo_19.ix[i,'Psychotic Features (Y/N – throughout Lifetime)']=np.nan
            SaoPaulo_19.ix[i,'Diagnosis (BP1, BP2, Control)']="Control"    
    SaoPaulo_19 = SaoPaulo_19.drop(['Dx full w/ w/o psychosis'],1)
    # Add missing variables and sort columns
    miss_col = pd.DataFrame(columns=list(set(columns)-set(SaoPaulo_19.columns.tolist())))
    SaoPaulo_19 = SaoPaulo_19.append(miss_col)
    SaoPaulo_19['siteID'] = '19_SaoPaulo'
    SaoPaulo_19 = SaoPaulo_19[columns]
    SaoPaulo_19['Diagnosis (BP1, BP2, Control)'] = SaoPaulo_19['Diagnosis (BP1, BP2, Control)'].replace({0:'Control', 1:'BP'})
    SaoPaulo_19['Sex'] = SaoPaulo_19['Sex'].replace({0:'F', 1:'M'}) # Error in logbook because Sex is coded as 0/1 --> see email for correction
    SaoPaulo_19 = SaoPaulo_19.drop(SaoPaulo_19[SaoPaulo_19['Age at Time of Scan'] < 18].index) 
    SaoPaulo_19['subjectID'] = SaoPaulo_19['subjectID'].str.replace('Input ','')
    SaoPaulo_19['subjectID'] = SaoPaulo_19['subjectID'].str.replace('_T1_LPS_N4','').astype(str)
    SaoPaulo_19['Age of Onset'] = SaoPaulo_19['Age of Onset'].replace(' ',np.nan)
    SaoPaulo_19['Age of Onset'] = SaoPaulo_19['Age of Onset'].astype(float)
    
    
    #####----- PITTSBURGH -----######
    Pittsburgh_20 = os.path.join(BASE_PATH,'Clinical_data','20_Pittsburgh_Clinical_data.csv')
    Pittsburgh_20 = pd.read_csv(Pittsburgh_20, decimal=",")
    Pittsburgh_20['Diagnosis (BP1, BP2, Control)']=''
    Pittsburgh_20['Mood Phase at Time of Scan']=''
    for i, row in Pittsburgh_20.iterrows():
        if row['COHORT2'] == 1:
            Pittsburgh_20.ix[i,'Diagnosis (BP1, BP2, Control)']="Control"
            Pittsburgh_20.ix[i,'Mood Phase at Time of Scan']= "NA"
        elif row['COHORT2'] == 4:
            Pittsburgh_20.ix[i,'Diagnosis (BP1, BP2, Control)']="BP1"
            Pittsburgh_20.ix[i,'Mood Phase at Time of Scan']= "Depressed"
        elif row['COHORT2'] == 5:
            Pittsburgh_20.ix[i,'Diagnosis (BP1, BP2, Control)']="BP1"
            Pittsburgh_20.ix[i,'Mood Phase at Time of Scan']= "Euthymic"
    Pittsburgh_20['Depression Scale']='HRS17'
    Pittsburgh_20['Mania Scale'] = 'YMRS'
    Pittsburgh_20['Number of Manic Episodes'] = Pittsburgh_20['Mania_Episodes'] + Pittsburgh_20['Hypomania_Episodes'] 
    Pittsburgh_20 = Pittsburgh_20.rename(columns={'ID':'subjectID','AgeatMRI':'Age at Time of Scan',
                                                  'HRS17TOT':'Depression Score at Time of Scan',
                                                  'YOUNGTOT':'Mania Score at Time of Scan',
                                                  'Illness_AgeatOnset':'Age of Onset','Dep_Episodes':'Number of Depressive Episodes',
                                                  'AntiDep':'Meds, Antidepressants',
                                                  'AntiPsych':'Meds, Antipsychotics',
                                                  'MoodStab':'Meds, Mood Stabilizers',
                                                  'Benzo':'Meds, Benzodiazepine'})
    Pittsburgh_20 = Pittsburgh_20.drop(['COHORT','COHORT2','MRI_DATE','DOB','HRS25TOT', 'Mania_AgeatOnset',
                                        'Dep_AgeatOnset','DSMIV','Duration_Dep','Illness_Duration', 'Mania_Episodes',
                                        'Hypomania_Episodes','PMD_Episodes','Total#_Psychotropics','MEDLOAD_TOTAL'],1)
    Pittsburgh_20 = Pittsburgh_20.drop(Pittsburgh_20.columns.to_series()['@1_MEDCLASS':"@5_DOSETYPE"], axis=1)
    Pittsburgh_20 = Pittsburgh_20.drop(Pittsburgh_20.columns.to_series()['NART_FullScaleIQ':"VISIT2_COMMENT"], axis=1)
    # Get 'on meds' variable
    for i, row in Pittsburgh_20.iterrows():
        if (row['Meds, Antidepressants']==0) and (row['Meds, Antipsychotics']==0):
            Pittsburgh_20.ix[i,'On Medication at Time of Scan (Y/N)']=0
        elif (row['Meds, Antidepressants']==1) or (row['Meds, Antipsychotics']==1):
            Pittsburgh_20.ix[i,'On Medication at Time of Scan (Y/N)']=1
        else: Pittsburgh_20.ix[i,'On Medication at Time of Scan (Y/N)']=np.nan   
    # Add missing variables and sort columns
    miss_col = pd.DataFrame(columns=list(set(columns)-set(Pittsburgh_20.columns.tolist())))
    Pittsburgh_20 = Pittsburgh_20.append(miss_col)
    Pittsburgh_20['siteID'] = '20_Pittsburgh'
    Pittsburgh_20 = Pittsburgh_20[columns]
    Pittsburgh_20['Sex'] = Pittsburgh_20['Sex'].replace({1:'M', 2:'F'}) 
    #Convert variable type
    Pittsburgh_20['subjectID'] = Pittsburgh_20['subjectID'].astype(int).astype(str)
    Pittsburgh_20['Age at Time of Scan'] = Pittsburgh_20['Age at Time of Scan'].apply(roundup)
    Pittsburgh_20 = Pittsburgh_20.replace('#NULL!', np.nan)
    Pittsburgh_20['Number of Manic Episodes'] = np.nan #To be verified
    Pittsburgh_20['Age of Onset'] = Pittsburgh_20['Age of Onset'].astype(float)
    Pittsburgh_20.info()
    
    
    #####----- COLUMBIA -----######
    Columbia_21 = os.path.join(BASE_PATH,'Clinical_data','21_Columbia_Clinical_data.csv')
    Columbia_21 = pd.read_csv(Columbia_21, decimal=',')
    Columbia_21 = Columbia_21.rename(columns={'Subject ID':'subjectID','weeks':'Length of Time on Antipsychotics'})
    Columbia_21['siteID'] = '21_Columbia'
    Columbia_21 = Columbia_21[columns]
    Columbia_21 = Columbia_21.replace('nAn',np.nan)  
    Columbia_21 = Columbia_21.replace({'y':'Y','n':'N',' Y':'Y',' N':'N'})  
    Columbia_21['Mood Phase at Time of Scan'] = Columbia_21['Mood Phase at Time of Scan'].replace({'Euthymia ':'Euthymic',
                                                'Depression':'Depressed','Depression with mixed symptons':'Mixed'})
    for i, row in Columbia_21.iterrows():
        if row['Meds, Antipsychotics'] =='N':
            Columbia_21.ix[i,'Meds, Antipsychotics']='N'
        else: Columbia_21.ix[i, 'Meds, Antipsychotics']='Y' 
        if row['Meds, Anticonvulsants'] =='N':
            Columbia_21.ix[i,'Meds, Anticonvulsants']='N'
        else: Columbia_21.ix[i, 'Meds, Anticonvulsants']='Y'
        if row['Meds, Antidepressants'] =='N':
            Columbia_21.ix[i,'Meds, Antidepressants']='N'
        else: Columbia_21.ix[i, 'Meds, Antidepressants']='Y'

    Columbia_21['Depression Scale'] = 'HDRS-21'
    
    #####----- SINGAPOUR -----######
    Singapour_22 = os.path.join(BASE_PATH,'Clinical_data','22_Singapour_Clinical_data.csv')
    Singapour_22 = pd.read_csv(Singapour_22, decimal=',')
    Singapour_22['siteID'] = '22_Singapour'
    Singapour_22 = Singapour_22.rename(columns={'Psychotic Features (Y/N- throughout Life time)':'Psychotic Features (Y/N – throughout Lifetime)',
                                                'Suicide Attempt (Y/N- throughout lifetime)':'Suicide Attempt  (Y/N – throughout Lifetime)',
                                                'History of Alcohol Dependence (y/n)':'History of Alcohol Dependence (Y/N)',
                                                'Rapid Cycling (y/n)':'Rapid Cycling (Y/N)'})
    Singapour_22 = Singapour_22[columns]
    Singapour_22 = Singapour_22.replace({'y':'Y','n':'N',' Y':'Y', ' N':'N'})
    Singapour_22['Sex'] = Singapour_22['Sex'].replace({1:'M',2:'F'}) # (Confirmed by email)     

    
    
    
    ###############################################################################
    ############################# BIG DATAFRAME ###################################
    ###############################################################################
    
    # Merge data
    sites = [Unc_3, Uct_4, Cardiff_6, Edimburgh_7, VitaSalute_10, Fidmag_11, Ucla_13,
             Mannheim_15, Grenoble_17, SaoPaulo_19, Pittsburgh_20, Munster_1, Ki_5,
             Unsw_2, Ucsd_16, Oslo_9, Creteil_12, Columbia_21, Singapour_22]
    df_clinic = pd.concat(sites)
    df_clinic = df_clinic.replace(np.nan,'')
    df_clinic = df_clinic.replace('',np.nan)    
    df_clinic = df_clinic.sort_values(['siteID','subjectID'])
    df_clinic = df_clinic.reset_index(drop=True) #Reset index to avoid confusion between sites with same subjectID

    nbsub = len(df_clinic['subjectID'])


##    # Recode subjects ID 
#    df_clinic['subjectID'] = df_clinic['subjectID'].astype(str)
#    df_clinic['subjectID'] = df_clinic['subjectID'] + '_' + df_clinic['siteID']
                            
    ### Find and remove duplicates ###
#    dupl = pd.DataFrame(columns = df_clinic.columns)
#    i = 0
#    for s, site in df_clinic.groupby(['siteID']):
#        n = site.duplicated(['subjectID']).sum() # specify columns for finding duplicates
#        temp = site[site.duplicated(['subjectID'])]
#        dupl = pd.concat([dupl, temp])
#        i+=n
    # no duplicates                    
    
    assert df_clinic.shape == (nbsub, 26)

    
    ### Recode numeric variables ###
    var_num = ['Age at Time of Scan', 'Age of Onset', 
               'Depression Score at Time of Scan', 'Number of Depressive Episodes', 
               'Mania Score at time of Scan', 'Number of Manic Episodes']
    
#    #Find outliers
#    outliers = pd.DataFrame(columns=var_num) #find aberrant values
    for var in var_num:
        df_clinic[var] = df_clinic[var].astype(float)
#        outliers[var] = (df_clinic[var] - df_clinic[var].mean()).abs() > 3 * df_clinic[var].std()
#    outliers = outliers.astype(str).replace({'False':0,'True':1})
#    outliers.loc['Total'] = pd.Series(outliers.sum())
    
    
    var_int = ['Meds, Antipsychotics',  
               'Meds, Antidepressants', 
               'Meds, Anticonvulsants',
               'On Medication at Time of Scan (Y/N)',
               'Psychotic Features (Y/N – throughout Lifetime)',
               'Suicide Attempt  (Y/N – throughout Lifetime)',
               'Lithium (y/n)',
               'History of Alcohol Dependence (Y/N)',
               'Rapid Cycling (Y/N)']
    df_clinic[var_int] = df_clinic[var_int].replace({'y':1,'n':0,'Y':1,'N':0,'Yes':1,'No':0,
                                                     'yes':1,'no':0,'nan':np.nan}).astype(float)  
    
    # Redefine "on medication" variable 
    for i, row in df_clinic.iterrows():
        if (row['On Medication at Time of Scan (Y/N)'] ==1) or (row['Meds, Antipsychotics']==1) or (row['Meds, Anticonvulsants']==1) or (row['Meds, Antidepressants']==1) or (row['Lithium (y/n)']==1):
            df_clinic.ix[i,'On Medication at Time of Scan (Y/N)']=1
        elif (row['On Medication at Time of Scan (Y/N)']==0) or (row['Meds, Antipsychotics' and 'Meds, Anticonvulsants' and 'Meds, Antidepressants' and 'Lithium (y/n)'] == 0):
            df_clinic.ix[i,'On Medication at Time of Scan (Y/N)']=0
        elif (row['Meds, Antipsychotics' and 'Meds, Anticonvulsants' and 'Meds, Antidepressants' and 'Lithium (y/n)'] ==np.nan):
            df_clinic.ix[i,'On Medication at Time of Scan (Y/N)']=np.nan    
        else: df_clinic.ix[i,'On Medication at Time of Scan (Y/N)']=np.nan
    
   
    for var in var_int: #check unique values
        df_clinic[var] = df_clinic[var].astype(float)
        print(var)
        print(df_clinic[var].unique())
        print('\n')
    
    
    ### Recode categorical variables ###   
    var_cat = ['siteID','subjectID', 'Sex',
           'Diagnosis (BP1, BP2, Control)', 'Mood Phase at Time of Scan',
           'Depression Scale', 'Mania Scale']
    
    df_clinic['Diagnosis (BP1, BP2, Control)'] = df_clinic['Diagnosis (BP1, BP2, Control)'].replace({'HC':'Control'})
    df_clinic['Mood Phase at Time of Scan'] = df_clinic['Mood Phase at Time of Scan'].replace({'nan':np.nan,'NA':np.nan,
                                      'euthymic':'Euthymic','Euthymia':'Euthymic'})
    df_clinic['Depression Scale'] = df_clinic['Depression Scale'].replace({'HAMD':'HDRS','Hamilton D':'HDRS',
                              'Hamilton':'HDRS','HRS17':'HDRS-17'})    
    df_clinic['Mania Scale'] = df_clinic['Mania Scale'].replace('Young','YMRS')
    
    df_clinic[var_cat] = df_clinic[var_cat].replace({np.nan:'nan'}).astype(str)  
    
    for var in var_cat: #check unique values
        df_clinic[var] = df_clinic[var].astype(str)
        print(var)
        print(df_clinic[var].unique())
        print('\n')
    
    
    ### Rearrange df and variables ###
    df_clinic['DX'] = df_clinic['Diagnosis (BP1, BP2, Control)']
    df_clinic['DX'] = df_clinic['DX'].replace({'BP NOS':'BD','BP1':'BD','BP2':'BD','BP':'BD','Control':'HC'})
    df_clinic['Diagnosis (BP1, BP2, Control)'] = df_clinic['Diagnosis (BP1, BP2, Control)'].replace({'BP':'nan'})  
    
    # Create new variable: Early (E) vs Late (L) (Bellivier et al., 2003)
    for i, row in df_clinic.iterrows():
        if row['Age of Onset'] <= 18:
            df_clinic.ix[i,'Onset Time'] = 'Early'
        elif 18 < row['Age of Onset'] <= 35: 
            df_clinic.ix[i,'Onset Time'] = 'Intermediate'
        elif row['Age of Onset'] > 35: 
            df_clinic.ix[i,'Onset Time'] = 'Late'
        else: df_clinic.ix[i,'Onset Time'] = 'nan'
            
    #Recode abberant number of depressive / manic episods
    for i, row in df_clinic.iterrows():
        if row['Number of Depressive Episodes'] > 20:
            df_clinic.ix[i,'Number of Depressive Episodes'] = 20
        if row['Number of Manic Episodes'] > 20:
            df_clinic.ix[i,'Number of Manic Episodes'] = 20
    
    # Create new variables for severity
    df_clinic['Total Episodes'] = df_clinic['Number of Depressive Episodes'] + df_clinic['Number of Manic Episodes']
    df_clinic['Illness Duration'] = df_clinic['Age at Time of Scan'] - df_clinic['Age of Onset']
    df_clinic['Density of Episodes'] = df_clinic['Total Episodes'] / df_clinic['Illness Duration']
    
    for i, row in df_clinic.iterrows():
        if row['Illness Duration'] == 0:
            df_clinic.ix[i,'Density of Episodes'] = np.nan
        if row['Illness Duration'] < 0:
            df_clinic.ix[i,'Illness Duration'] = np.nan
            df_clinic.ix[i,'Density of Episodes'] = np.nan
    
    # Create binary variable for sevrity: Hight (H) vs Low (L)
    for i, row in df_clinic.iterrows():
        if row['Density of Episodes'] >= 1: #more than one episod per year
            df_clinic.ix[i,'Severity'] = 'High'
        elif row['Density of Episodes'] < 1: 
            df_clinic.ix[i,'Severity'] = 'Low'
        else: df_clinic.ix[i,'Severity'] = 'nan'
    
    #### Rearrange df_clinic ####
    new_columns = ['siteID','subjectID', 'DX', 'Age at Time of Scan', 'Sex', 'Age of Onset', 'Onset Time',
           'Illness Duration', 'Diagnosis (BP1, BP2, Control)', 'Mood Phase at Time of Scan',
           'Depression Scale', 'Depression Score at Time of Scan',
           'Number of Depressive Episodes', 'Mania Scale',
           'Mania Score at time of Scan', 'Number of Manic Episodes', 'Total Episodes', 'Density of Episodes',
           'Severity', 'Psychotic Features (Y/N – throughout Lifetime)',
           'On Medication at Time of Scan (Y/N)', 'Meds, Antipsychotics',
           'Meds, Antidepressants','Meds, Anticonvulsants','Lithium (y/n)',
           'History of Alcohol Dependence (Y/N)'] #Delete length on Meds + Rapid cycling +  suicide beacause not enough info
    df_clinic = df_clinic[new_columns]
    df_clinic = df_clinic.rename(columns={'Age at Time of Scan':'Age',
                       'Diagnosis (BP1, BP2, Control)':'BD Type', 
                       'Mood Phase at Time of Scan':'Mood Phase', 
                       'Depression Score at Time of Scan': 'Depression Score',
                       'Mania Score at time of Scan': 'Mania Score', 
                       'Psychotic Features (Y/N – throughout Lifetime)':'Psychotic',
                       'On Medication at Time of Scan (Y/N)':'On Meds', 
                       'Meds, Antipsychotics':'Antipsychotics',
                       'Meds, Antidepressants':'Antidepressants',
                       'Meds, Anticonvulsants':'Anticonvulsants',
                       'Lithium (y/n)':'Lithium',
                       'History of Alcohol Dependence (Y/N)':'Alcohol'})

    #Remove BD NOS 
    df_clinic = df_clinic.drop(df_clinic[df_clinic['BD Type'] == 'BP NOS'].index)
    #Remove controls with history of depressive episods and/or on meds
    df_clinic = df_clinic.drop(df_clinic[(df_clinic['DX'] == 'HC') & (df_clinic['Number of Depressive Episodes'] > 0)].index)
    df_clinic = df_clinic.drop(df_clinic[(df_clinic['DX'] == 'HC') & (df_clinic['On Meds'] > 0)].index)
  
    df_clinic = df_clinic.sort_values(['siteID','subjectID'])
    
    assert df_clinic.shape == (2359-(30+16+3), 26) # 30 NOS +  16 HC with DEP + 3 additional HC on Meds
    # Save clinical data
    #df_clinic.to_csv(os.path.join('~/Dropbox/Post-Doc2/ENIGMA/Data/Clinical_data', 'clinical_data_enigma.csv'))    
 
    
    return df_clinic
