# -*- coding: utf-8 -*-
"""
@author: yl247234
Copyrignt : CEA NeuroSpin - 2016
"""
import pandas as pd
import numpy as np
import os, glob, re, json
import nibabel.gifti.giftiio as gio
import scipy.stats as stats

database_parcel = 'hcp'
path_parcels = '/media/yl247234/SAMSUNG/'+database_parcel+'/Freesurfer_mesh_database/'

feature_threshold = 'DPF'
OUTPUT = '/home/yl247234/Images/final_snap_sym/group_'+database_parcel+'_Freesurfer_new/'
INPUT = '/neurospin/brainomics/2016_HCP/new_pheno_threshold_'+feature_threshold+'/pheno_pits_sym_DPF_'+database_parcel+'_Freesurfer_new/'


sides = ['R', 'L']

dict_freq = {}
for side in sides:
    dict_freq[side] = {}
    filename = INPUT+"case_control/all_pits_side"+side+".csv"
    df = pd.read_csv(filename)
    nb_subj = df.shape[0]
    for col in df.columns:
        if col != 'IID':
            df[col] = df[col]-1
            nb_pits = np.count_nonzero(df[col])
            freq = float(nb_pits)/nb_subj
            if freq > 0.5:
                dict_freq[side][col]= [nb_pits, df.shape[0]-nb_pits]

labels = '/neurospin/brainomics/2016_HCP/LABELS/labelling_sym_template.csv'
df_labels = pd.read_csv(labels)
df_labels.index = df_labels['Parcel']

count = 0
array_L_trait = []
array_R_trait = []
array_L_pval = []
array_R_pval = []
for col in df.columns:
    if dict_freq['L'].has_key(col) and dict_freq['R'].has_key(col):
        count+=1
        contingency_table= [dict_freq['L'][col],dict_freq['R'][col]]
        oddsratio, pvalue = stats.fisher_exact(contingency_table)
        m = re.search('Parcel_(.+?)end', col+'end')
        if m:
            num = int(m.group(1))
        if pvalue < 1:
            if pvalue < 0.01:
                p = '%.1e'% pvalue
                p = p.replace('e-0', '·10-')
                p = p.replace('e-', '·10-')
            else:
                p = round(pvalue,2)
            if dict_freq['L'][col] > dict_freq['R'][col]:
                array_L_trait.append(df_labels.loc[num]['Name'].replace('_',' ').replace('.',''))
                array_L_pval.append(str(p))
                print "Left asymmetry of "+df_labels.loc[num]['Name'].replace('_',' ').replace('.','') +" pval fischer exact test "+ str(p)
            else:
                array_R_trait.append(df_labels.loc[num]['Name'].replace('_',' ').replace('.',''))
                array_R_pval.append(str(p))
                print "Right asymmetry of "+df_labels.loc[num]['Name'].replace('_',' ').replace('.','') +" pval fischer exact test "+ str(p)

df_L = pd.DataFrame()
output_L = '/neurospin/brainomics/2016_HCP/LABELS/fischer_test_L_all.csv'
df_R = pd.DataFrame()
output_R = '/neurospin/brainomics/2016_HCP/LABELS/fischer_test_R_all.csv'
df_L['Trait'] = array_L_trait
df_L['p Fischer exact test'] = array_L_pval
df_L.index = df_L['Trait']
df_L= df_L.sort_index()
df_L.to_csv(output_L, sep=',', header=True, index=False)
df_R['Trait'] = array_R_trait
df_R['p Fischer exact test'] = array_R_pval
df_R.index = df_R['Trait']
df_R= df_R.sort_index()
df_R.to_csv(output_R, sep=',', header=True, index=False)
