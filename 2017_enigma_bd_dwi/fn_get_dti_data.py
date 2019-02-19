#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 16:11:57 2017

@author: Pauline
"""

from __future__ import division
import os
import os.path
import pandas as pd
import statsmodels.formula.api as smfrmla
import math
def roundup(x):
    return int(math.ceil(x))

def get_dti_data(BASE_PATH):

    ##############################################################################
    ########################### Data Definition ##################################
    ##############################################################################
    
    #Import csv file
    INPUT_CSV = os.path.join(BASE_PATH,"FA_data","FA_ENIGMA_BiPolar_Combined_final.csv")
    pop = pd.read_csv(INPUT_CSV)
    cols = pop.columns.tolist()

    #Add Data from Colombia
    Columbia_21 = os.path.join(BASE_PATH,'FA_data','21_Columbia_FA.csv')
    Columbia_21 = pd.read_csv(Columbia_21, decimal='.')
    Columbia_21['siteID'] = 21
    Columbia_21['DX'] = 1
    Columbia_21['Sex'] = Columbia_21['Sex'].replace({2:0}) # 1 = M / 2 = F (confirmed by clinical data)     
    Columbia_21 = Columbia_21[cols]
    
    #Add Data from Singapour
    Singapour_22 = os.path.join(BASE_PATH,'FA_data','22_Singapour_FA.csv')
    Singapour_22 = pd.read_csv(Singapour_22, decimal=',')
    Singapour_22.columns = Singapour_22.columns.str.replace('_','-')
    Singapour_22 = Singapour_22.rename(columns={'Average-FA':'AverageFA',
                                                'FX-ST-L':'FX/ST-L',
                                                'FX-ST-R':'FX/ST-R',
                                                'FX-ST':'FXST'})
    Singapour_22_cli = pd.read_csv(os.path.join(BASE_PATH,'Clinical_data/22_Singapour_Clinical_data.csv'), decimal = ',')
    Singapour_22_demo = pd.DataFrame()
    Singapour_22_demo['subjectID'] = Singapour_22_cli['subjectID'] 
    Singapour_22_demo['DX'] = Singapour_22_cli['Diagnosis (BP1, BP2, Control)'].replace({'BP1':1,'Controls':0}) 
    Singapour_22_demo['Age'] = Singapour_22_cli['Age at Time of Scan'] 
    Singapour_22_demo['Sex'] = Singapour_22_cli['Sex'] 
    Singapour_22 = Singapour_22_demo.merge(Singapour_22, on='subjectID', how =  'left')
    Singapour_22 = Singapour_22.drop(Singapour_22[Singapour_22['DX_y'].isnull()].index) #Remove subj with no DTI data
    Singapour_22 = Singapour_22.drop(['DX_y'],1)
    Singapour_22 = Singapour_22.rename(columns={'DX_x':'DX'}).replace({'BP1':1,'Control':0})
    Singapour_22['Sex'] = Singapour_22['Sex'].replace({2:0}) # 1 = M / 2 = F (confirmed by mail)     
    Singapour_22['siteID'] = 22
    Singapour_22 = Singapour_22[cols] 
    Singapour_22['IC'] = pop['IC'].mean()
    Singapour_22['IC-L'] = pop['IC-L'].mean()
    Singapour_22['IC-R'] = pop['IC-R'].mean()
    
    #Add Halifax
    Halifax_23 = os.path.join(BASE_PATH,'FA_data','23_Halifax_FA.csv')
    Halifax_23 = pd.read_csv(Halifax_23, decimal='.')
    Halifax_23 = Halifax_23.rename(columns={'BPD':'DX'})
    Halifax_23['siteID'] = 23
    Halifax_23['Sex'] = Halifax_23['Sex'].replace({2:0}) #In original file male=1 and female=2 (see Jason Newport's mail)
    #Remove NOS
    Halifax_23 = Halifax_23.drop(Halifax_23[Halifax_23['DX'] == 'NOS'].index)
    Halifax_23['DX'] = Halifax_23['DX'].replace({'1':1,'2':1,'3':0}) #In original file 1= BPI, 2=BP2, 3=CTL
    #Round-up age
    Halifax_23['Age'] = Halifax_23['Age'].apply(roundup)
    Halifax_23 = Halifax_23[cols]
    Halifax_23 = Halifax_23.drop(Halifax_23[Halifax_23['Age'] > 65].index)

    
    #Concatenate dataframes
    pop = pd.concat([pop,Columbia_21,Singapour_22,Halifax_23])
    nbsubjects = len(pop) # count the number of subjects      
    
    #Error control/patient
    pop.loc[(pop.subjectID == '178') & (pop.siteID == 10),'DX'] = 0
    
    #Recode categorical variables and sujects IDs
    pop['DX'] = pop['DX'].replace({0:'HC', 1:'BD'})
    pop['Sex'] = pop['Sex'].replace({0:'F', 1:'M'})
    pop['siteID'] = pop['siteID'].astype(str)
    pop['siteID'] = pop['siteID'].replace({'1.0':'1_Munster',\
            '2.0':'2_UNSW',  
            '3.1':'3.1_UNC',
            '3.2':'3.2_UNC',
            '4.0':'4_UCT',
            '5.0':'5_KI',
            '6.0':'6_Cardiff',
            '7.1':'7.1_Edimburgh',
            '7.2':'7.2_Edimburgh',
            '8.1':'8.1_IoL',
            '8.2':'8.2_IoL',
            '8.3':'8.3_IoL',
            '9.1':'9.1_Oslo_Malt',
            '9.2':'9.2_Oslo_Malt',
            '10.0':'10_VitaSalute',
            '11.0':'11_FIDMAG',
            '12.0':'12_Creteil',
            '13.0':'13_UCLA',
            '15.0':'15_Mannheim',
            '16.0':'16_UCSD',
            '17.0':'17_Grenoble',
            '19.0':'19_SaoPaulo',
            '20.0':'20_Pittsburgh',
            '21.0':'21_Columbia',
            '22.0':'22_Singapour',
            '23.0':'23_Halifax'}) 
    
    #Corrections of coding for Sex
    pop.loc[pop.siteID == '2_UNSW','Sex'] = pop.loc[pop.siteID == '2_UNSW','Sex'].replace({'F':'1','M':'0'})
    pop.loc[pop.siteID == '4_UCT','Sex'] = pop.loc[pop.siteID == '4_UCT','Sex'].replace({'F':'1','M':'0'}) #!!!! Not sure !!!!
    pop.loc[pop.siteID == '3.1_UNC','Sex'] = pop.loc[pop.siteID == '3.1_UNC','Sex'].replace({'F':'1','M':'0'})
    pop.loc[pop.siteID == '2_UNSW','Sex'] = pop.loc[pop.siteID == '2_UNSW','Sex'].replace({'0':'F','1':'M'})
    pop.loc[pop.siteID == '4_UCT','Sex'] = pop.loc[pop.siteID == '4_UCT','Sex'].replace({'0':'F','1':'M'}) #!!!! Not sure !!!!
    pop.loc[pop.siteID == '3.1_UNC','Sex'] = pop.loc[pop.siteID == '3.1_UNC','Sex'].replace({'0':'F','1':'M'})                
    
#    ## Find duplicates
#    n_dupl = pop.duplicated(['subjectID']).sum() # specify columns for finding duplicates
#    n_dupl_df = pop[pop.duplicated(['subjectID'])]
#    pop.loc[pop.siteID == '10_VitaSalute','subjectID'] = pop.loc[pop.siteID == '10_VitaSalute','subjectID'] + '_VS'
    pop.loc[pop.siteID == '5_KI','subjectID'] = pop.loc[pop.siteID == '5_KI','subjectID'] + '_KI'
#    pop.loc[pop.siteID == '16_UCSD','subjectID'] = pop.loc[pop.siteID == '16_UCSD','subjectID'] + '_UCSD'
#    pop.loc[pop.siteID == '17_Grenoble','subjectID'] = pop.loc[pop.siteID == '17_Grenoble','subjectID'] + '_Gre'
 
    assert pop.shape == (nbsubjects, 68) 
    
    #Drop NOS patients
    bd_nos = {'grenoble_Philipps__BipEd21','grenoble_Philipps__BipEd59','grenoble_Philipps__BipEd60', 
              'grenoble_Philipps__BipEd62','grenoble_Philipps__BipEd67', 'grenoble_Philipps__BipEd09',
              '30043','30124','30126','30142','30167','30168','30177','30200','30213','30258','30298','30366'}
    for i, row in pop.iterrows():
        if row['subjectID'] in bd_nos:
            pop.drop(i, inplace = True)
            
            
    #Drop HC with history of depression or on meds
    hc_dep = {'107_1_dti','136_1_dti','152_1_dti','230_1_dti','098_1_dti','15_KI',
              '104_1_dti','120_1_dti','121_1_dti','124_1_dti','144_1_dti','168_1_dti',
              '181_1_dti','192_1_dti','221_1_dti','226_1_dti','228_1_dti','232_1_dti','074_1_dti'}
    for i, row in pop.iterrows():
        if row['subjectID'] in hc_dep:
            pop.drop(i, inplace = True)
    
    assert pop.shape == ((nbsubjects-(len(bd_nos)+len(hc_dep))), 68) 
    
    #Modify sujbectsID to match clinical data
    pop.loc[pop.siteID == '1_Munster','subjectID'] = pop.loc[pop.siteID == '1_Munster','subjectID'].str.replace('_FA','')
    pop.loc[pop.siteID == '2_UNSW','subjectID'] = pop.loc[pop.siteID == '2_UNSW','subjectID'].str.replace('_1_dti','')
    pop.loc[pop.siteID == '2_UNSW','subjectID'] =pop.loc[pop.siteID == '2_UNSW','subjectID'].astype(int).astype(str)
    pop.loc[pop.siteID == '12_Creteil','subjectID'] = pop.loc[pop.siteID == '12_Creteil','subjectID'].str.replace('creteil__','')
    pop.loc[pop.siteID == '12_Creteil','subjectID'] = pop.loc[pop.siteID == '12_Creteil','subjectID'].str.replace('eM1005','em1005')
    pop.loc[pop.siteID == '13_UCLA','subjectID'] = pop.loc[pop.siteID == '13_UCLA','subjectID'].str.replace('_dti_FA_FA','')
    pop.loc[pop.siteID == '15_Mannheim','subjectID'] = pop.loc[pop.siteID == '15_Mannheim','subjectID'].str.replace('mannheim__','')
    pop.loc[pop.siteID == '16_UCSD','subjectID'] = pop.loc[pop.siteID == '16_UCSD','subjectID'].str.replace('uniCAsandiego__','')
    pop.loc[pop.siteID == '17_Grenoble','subjectID'] = pop.loc[pop.siteID == '17_Grenoble','subjectID'].str.replace('grenoble_Philipps__BipEd','')
    pop.loc[pop.siteID == '17_Grenoble','subjectID'] =pop.loc[pop.siteID == '17_Grenoble','subjectID'].astype(int).astype(str)
    pop.loc[pop.siteID == '20_Pittsburgh','subjectID'] = pop.loc[pop.siteID == '20_Pittsburgh','subjectID'].str.replace('pittsburgh__','')
    pop.loc[pop.siteID == '20_Pittsburgh','subjectID'] =pop.loc[pop.siteID == '20_Pittsburgh','subjectID'].astype(int).astype(str)


    pop = pop.sort_values(['siteID','subjectID'])
    pd.value_counts(pop['DX'].values)
    
    #Remove left and right tracts
    pop2 = pop.drop(['ACR-L','ACR-R','ALIC-L','ALIC-R','CGC-L','CGC-R','CGH-L','CGH-R',
                     'CR-L','CR-R','CST-L','CST-R','EC-L','EC-R','FX/ST-L','FX/ST-R',
                     'IC-L','IC-R','IFO-L','IFO-R','PCR-L','PCR-R','PLIC-L','PLIC-R',
                     'PTR-L','PTR-R','RLIC-L','RLIC-R','SCR-L','SCR-R','SFO-L','SFO-R',
                     'SLF-L','SLF-R','SS-L','SS-R','UNC-L','UNC-R'], axis=1)
    #Remove combined R&L tracts
    pop = pop.drop(['ACR', 'ALIC','CGC', 'CGH','CR', 'CST', 'EC', 
                'FXST', 'IC', 'IFO', 'PCR', 'PLIC','PTR', 'RLIC', 
                'SCR','SFO', 'SLF', 'SS', 'UNC'], axis = 1) 
    
    
    assert pop.shape == ((nbsubjects-(len(bd_nos)+len(hc_dep))), 49) 
    #Save dti data
    #pop.to_csv(os.path.join('~/Dropbox/Post-Doc2/ENIGMA/Data/FA_data', 'FA_data_enigma.csv'))    
    #pop2.to_csv(os.path.join('~/Dropbox/Post-Doc2/ENIGMA/Data/FA_data', 'FA_data_enigma_bilat.csv'))    
    
    ##############################################################################
    #### Normalise the data by sites ####
    zpop = pop.copy()
    #zpop['Age'] = (zpop['Age'] - zpop['Age'].mean(axis=0)) / zpop['Age'].std(axis=0)
    tracts = zpop.columns[5:].values
    d = pd.DataFrame()
    for s, site in zpop.groupby(['siteID']):
        temp = pd.DataFrame()
        #temp['Age'] = (site['Age'] - site['Age'].mean(axis=0)) / site['Age'].std(axis=0)
        #temp[:, tracts] = (site.ix[:, tracts] - site.ix[:, tracts].mean(axis=0)) / site.ix[:, tracts].std(axis=0)  
        for tra in tracts:
            temp[tra] = (site[tra] - site[tra].mean(axis=0)) / site[tra].std(axis=0)  
        temp.index = site.index
        temp = pd.merge(site.ix[:,0:5], temp, left_index = True, right_index = True)
        d = pd.concat([d, temp])
    zpop = d
#    zpop = zpop.rename(columns={'Age_y':'Age'})
#    zpop = zpop.drop(['Age_x'],1)
    #assert zpop.shape == (nbsubjects, 68) 
    
    ## Get dummies covariates and rearrange df
    #zpop['Sex'] = zpop['Sex'].replace({0:'F', 1:'M'})
    #zpop['DX'] = zpop['DX'].replace({0:'HC', 1:'BD'})
    #zpop_sex = pd.get_dummies(zpop['Sex'])
    #zpop_site = pd.get_dummies(zpop['siteID'])
    #zpop = pd.concat([zpop, zpop_sex], axis=1)
    #zpop = zpop.drop(['subjectID'], axis=1)
    #cols = zpop.columns.tolist()
    #cols = cols[0:3] + cols[-2:] + cols[3:47]
    #zpop = zpop[cols]

    assert zpop.shape == ((nbsubjects-(len(bd_nos)+len(hc_dep))),(49))
    
    #Save dti data
    #zpop.to_csv(os.path.join('~/Dropbox/Post-Doc2/ENIGMA/Data/FA_data', 'FA_data_enigma_zpop.csv')) 
    
    ##############################################################################
    #### Dataframe with the residues ####
    pop_res = pop.loc[:,('siteID','subjectID','DX','Age','Sex')]
    
    #get long format
    pop_long = pd.melt(pop, id_vars = ["siteID","subjectID","DX","Age","Sex"], var_name = "Tracts", value_name = "FA")
    pop_long['DX'] = pop_long['DX'].replace({0:'HC', 1:'BD'})
    pop_long['Sex'] = pop_long['Sex'].replace({0:'F', 1:'M'})
    
    #Get the residues
    for t in pop_long.Tracts.unique(): 
        lm = smfrmla.ols('FA ~ siteID + Age + Sex', pop_long[pop_long['Tracts'] == t]).fit()  
        res = lm.resid.values.tolist()
        pop_res[t] = res

    assert pop_res.shape == ((nbsubjects-(len(bd_nos)+len(hc_dep))), 49)
    #pop_res.to_csv(os.path.join('~/Dropbox/Post-Doc2/ENIGMA/Data/FA_data', 'FA_data_enigma_pop_res.csv')) 

    
    ##############################################################################
    #### Dataframe standardize by site with residues after age effect removed ####
    zpop_res = zpop.loc[:,("siteID","subjectID","DX","Age","Sex")]
    #zpop_res = zpop_res.rename(columns={'F':'Sex'}).replace({1:'F',0:'M'})

    #get long format
    zpop_long = pd.melt(zpop, id_vars = ["siteID","subjectID","DX","Age","Sex"], var_name = "Tracts", value_name = "FA")
    #zpop_long = zpop_long.rename(columns={'F':'Sex'}).replace({1:'F',0:'M'})
    #zpop_long = zpop_long.drop(['M'], axis = 1)
    
    #Get the residues
    for t in zpop_long.Tracts.unique(): 
        lm = smfrmla.ols('FA ~ Age + Sex', zpop_long[zpop_long['Tracts'] == t]).fit()  
        res = lm.resid.values.tolist()
        zpop_res[t] = res

    assert zpop_res.shape == ((nbsubjects-(len(bd_nos)+len(hc_dep))), 49)
    #zpop_res.to_csv(os.path.join('~/Dropbox/Post-Doc2/ENIGMA/Data/FA_data', 'FA_data_enigma_zpop_res.csv')) 

    
    return(pop, zpop, pop_res, zpop_res)
    
    
    
    
    